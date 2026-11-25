"""LTX-2.5 distilled pipeline with the DiT tensor-parallel on Neuron; every other stage on CPU (bring-up).
    torchrun --nproc_per_node=4 neuron/run_tp.py <ltx_pipelines.distilled CLI args>
Every rank runs the same pipeline (same seed, so the CPU stages agree); only the DiT is sharded over the ranks and
placed on neuron:<local rank>, each transformer block compiled once with torch.compile(backend="neuron"). Rank 0
writes the video. Prints per-stage times ([STAGE]) on rank 0. LTX_DIT_DEVICE=cpu keeps the DiT on CPU (reference).
"""
import collections
import functools
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol

rank, world, local = (int(os.environ.get(k, d)) for k, d in (("RANK", 0), ("WORLD_SIZE", 1), ("LOCAL_RANK", 0)))
DIT_DEV = os.environ.get("LTX_DIT_DEVICE", "neuron")
if DIT_DEV == "neuron":
    import torch_neuronx  # noqa: F401
    torch.neuron.set_device(local)
    dist.init_process_group(backend="neuron", rank=rank, world_size=world)
    dit_device = torch.device(f"neuron:{local}")
else:
    dit_device = torch.device("cpu")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tp  # noqa: E402

import ltx_core.devices as devs  # noqa: E402
import ltx_pipelines.distilled as D  # noqa: E402
from ltx_pipelines.utils.blocks import diffusion_stage as DS  # noqa: E402

# every stage but the DiT on CPU; the CPU decode needs a memory budget for its tiling planner
_BUDGET = 64 * 2**30
for _m in list(sys.modules.values()):
    if _m is None or not getattr(_m, "__name__", "").startswith(("ltx_core", "ltx_pipelines")):
        continue
    for _n in ("cuda_activation_budget_bytes", "cuda_allocatable_bytes", "activation_budget_bytes", "allocatable_bytes"):
        if hasattr(_m, _n):
            setattr(_m, _n, lambda device=None: _BUDGET)
    if hasattr(_m, "get_device"):
        # LTX_ALL_NEURON=1: the whole pipeline (latents, scheduler, every stage but the cached text encoder) on this
        # rank's core; otherwise the glue runs on CPU and only the heavy stages are on Neuron
        _pd = dit_device if os.environ.get("LTX_ALL_NEURON", "0") == "1" else torch.device("cpu")
        setattr(_m, "get_device", lambda _pd=_pd: _pd)

if os.environ.get("LTX_ALL_NEURON", "0") == "1":
    # Neuron has no seeded device generator: seeded generators live on CPU and seeded draws are made there and moved
    # (same noise as the CPU reference; a few MB once per denoising stage)
    _Gen, _randn, _randn_like = torch.Generator, torch.randn, torch.randn_like

    class _GM(type):
        def __instancecheck__(cls, o):
            return isinstance(o, _Gen)

    class _G(metaclass=_GM):  # torch.Generator(device=...) -> a CPU generator; isinstance still works
        def __new__(cls, device=None, *a, **k):
            return _Gen("cpu")
    torch.Generator = _G

    def _rn(*size, generator=None, device=None, **k):
        if generator is None:
            return _randn(*size, device=device, **k)
        return _randn(*size, generator=generator, device="cpu", **k).to(device if device is not None else "cpu")
    torch.randn = _rn

    def _rnl(x, generator=None, **k):
        if generator is None:
            return _randn_like(x, **k)
        return _randn(x.shape, generator=generator, dtype=k.get("dtype", x.dtype), device="cpu").to(k.get("device", x.device))
    torch.randn_like = _rnl

    # eager torch.lerp does not lower on Neuron: a + w * (b - a), the same formula
    _lerp = torch.lerp

    def _lp(start, end, weight, *a, **k):
        if start.device.type == "cpu":
            return _lerp(start, end, weight, *a, **k)
        return start + weight * (end - start)
    torch.lerp = _lp
    torch.Tensor.lerp = lambda self, end, weight: _lp(self, end, weight)

# --- serving shape: every model built once and kept resident; the prompt encoding cached by prompt ---
import contextlib
from ltx_core.loader.simple_builder import SimpleBuilder  # noqa: E402

_built = {}
_orig_sb_build = SimpleBuilder.build


def _sb_build(self, device=None, dtype=None, **kw):
    key = (id(self), str(device), str(dtype), tuple(sorted(kw.items())) if kw else ())
    if key not in _built:
        _built[key] = _orig_sb_build(self, device=device, dtype=dtype, **kw)
    return _built[key]


SimpleBuilder.build = _sb_build


@contextlib.contextmanager
def _keep(model, alloc_trim_strategy=None):
    yield model


for _m in list(sys.modules.values()):
    if _m is not None and getattr(_m, "__name__", "").startswith("ltx_pipelines") and hasattr(_m, "gpu_model"):
        setattr(_m, "gpu_model", _keep)

_prompts = {}


def _cache_prompt(cls):
    f = cls.__call__

    @functools.wraps(f)
    def w(self, prompts, *a, **k):
        key = (tuple(prompts), repr(sorted(k.items())))
        if key not in _prompts:
            dev0 = self._device
            if os.environ.get("LTX_TE_NEURON", "0") != "1":
                self._device = torch.device("cpu")
            dt0 = self._dtype
            if os.environ.get("LTX_TE_FP32", "0") == "1":   # gate reference: the text encoder in fp32 on CPU
                self._dtype = torch.float32
            try:
                _prompts[key] = f(self, prompts, *a, **k)
            finally:
                self._device, self._dtype = dev0, dt0
            if os.environ.get("LTX_TE_DUMP") and rank == 0:   # gate: the prompt encoding, after the timers
                torch.save(_move(_prompts[key], torch.device("cpu")), os.environ["LTX_TE_DUMP"])
        if os.environ.get("LTX_ALL_NEURON", "0") == "1":
            return _move(_prompts[key], dit_device)
        return _prompts[key]
    cls.__call__ = w

_cached = {}
# LTX_FULLGRAPH=1 (default): every compiled module must be one graph; a graph break is an error, never a silent split
_FG = os.environ.get("LTX_FULLGRAPH", "1") == "1"
_orig_build = DS.DiffusionStage._build_transformer

# LTX_STEP_GRAPH=1: one compiled graph per denoising step = denoiser (modality from the latent state, the DiT) +
# post-process + Euler update, with (sigma, sigma_next) as a device tensor so every step of a stage reuses it.
# The trace: ~20 ms of idle (enc_barrier + execute_schedule) before each of the 11 DiT steps, switching between the
# small eager step graphs and the DiT graph.
from ltx_pipelines.utils import samplers as _SM  # noqa: E402


def _euler_latent(state, denoised, s2):
    """post_process + EulerDiffusionStep with sigma kept a device tensor (the stock to_velocity calls sigma.item()).
    Returns only the new latent: the graph's outputs never alias its inputs. Same formulas."""
    if state is None or denoised is None:
        return None
    from ltx_pipelines.utils.helpers import post_process_latent
    d = post_process_latent(denoised, state.denoise_mask.tensor, state.clean_latent)
    x = state.latent
    sig, nxt = s2[0].float(), s2[1].float()
    v = ((x.float() - d.float()) / sig).to(x.dtype)
    return (x.float() + v.float() * (nxt - sig)).to(x.dtype)


def _step_graph_loop(sigmas, video_state, audio_state, transformer, denoiser, v_context=None, a_context=None):
    from dataclasses import replace as _rp
    dev = _SM._present_state(video_state, audio_state).latent.device
    sched = sigmas.to(dev)
    pairs = [sched[i:i + 2].clone() for i in range(len(sched) - 1)]

    def body(vs, as_, s2, vc, ac):
        vr, ar = denoiser(transformer, vs, as_, s2, 0, vc, ac)
        dv = vr.denoised if vr is not None else None
        da = ar.denoised if ar is not None else None
        return _euler_latent(vs, dv, s2), _euler_latent(as_, da, s2)
    key = ("step", id(transformer))
    if key not in _cached:
        _cached[key] = body if os.environ.get("LTX_STEP_EAGER") == "1" else \
            torch.compile(body, backend="neuron", dynamic=False, fullgraph=_FG)
    cb = _cached[key]
    for i, s2 in enumerate(pairs):
        vl, al = cb(video_state, audio_state, s2, v_context, a_context)
        if vl is not None:
            video_state = _rp(video_state, latent=vl)
        if al is not None:
            audio_state = _rp(audio_state, latent=al)
        if os.environ.get("LTX_STEP_DEBUG") == "1" and rank == 0 and vl is not None:
            f = vl.float()
            print(f"[STEP] {i} sigma {pairs[i].cpu().tolist()} mean {f.mean().item():.4f} std {f.std().item():.4f} "
                  f"nan {int(torch.isnan(f).sum())}", flush=True)
    return _SM.VideoAudio(video_state, audio_state)


def _anc_latent(state, denoised, s2, noise, terminal, eta, s_noise, model_dtype):
    """One ancestral-Euler step for one modality, as in _ancestral_euler_denoising_loop + EulerAncestralDiffusionStep,
    with sigma kept a device tensor. Returns only the new latent."""
    if state is None or denoised is None:
        return None
    from ltx_pipelines.utils.helpers import post_process_latent
    mask, clean = state.denoise_mask.tensor, state.clean_latent
    d = post_process_latent(denoised.float(), mask, clean)
    if terminal:
        return d.to(model_dtype)
    sig, nxt = s2[0].float(), s2[1].float()
    x = state.latent.float()
    sigma_down = nxt * (1.0 + (nxt / sig - 1.0) * eta)
    r = sigma_down / sig
    x_next = r * x + (1.0 - r) * d
    if eta > 0:
        a_next, a_down = 1.0 - nxt, 1.0 - sigma_down
        coeff = (nxt ** 2 - sigma_down ** 2 * a_next ** 2 / a_down ** 2).clamp(min=0) ** 0.5
        x_next = (a_next / a_down) * x_next + noise.float() * s_noise * coeff
        x_next = post_process_latent(x_next, mask, clean)
    return x_next.to(model_dtype)


_orig_anc = _SM.euler_ancestral_denoising_loop


def _anc_step_graph_loop(sigmas, video_state, audio_state, transformer, denoiser, v_context=None, a_context=None,
                         noise_seed=-1, new_noise_fn=_SM._get_plain_noise, model_dtype=torch.bfloat16, eta=1.0,
                         s_noise=1.0):
    """LTX_STEP_GRAPH=1 for the distilled pipeline's ancestral loop: the per-step noise is drawn up front from the
    same seeded generator in the stock order (video then audio, per non-terminal step), and each step (denoiser +
    ancestral update + mask) is one compiled graph."""
    if new_noise_fn is not _SM._get_plain_noise:
        return _orig_anc(sigmas, video_state, audio_state, transformer, denoiser, v_context, a_context,
                         noise_seed=noise_seed, new_noise_fn=new_noise_fn, model_dtype=model_dtype, eta=eta,
                         s_noise=s_noise)
    from dataclasses import replace as _rp
    dev = _SM._present_state(video_state, audio_state).latent.device
    host = sigmas.detach().float().cpu()
    sched = sigmas.to(dev)
    gen = torch.Generator("cpu").manual_seed(noise_seed)
    n = len(host) - 1
    steps = []
    for i in range(n):
        terminal = bool(host[i + 1] == 0)
        nv = na = None
        if not terminal and eta > 0:
            if video_state is not None:
                nv = torch.randn(video_state.latent.shape, generator=gen, dtype=video_state.latent.dtype).to(dev)
            if audio_state is not None:
                na = torch.randn(audio_state.latent.shape, generator=gen, dtype=audio_state.latent.dtype).to(dev)
        steps.append((sched[i:i + 2].clone(), nv, na, terminal))
        if terminal:
            break

    def body(vs, as_, s2, nv, na, vc, ac, terminal):
        vr, ar = denoiser(transformer, vs, as_, s2, 0, vc, ac)
        dv = vr.denoised if vr is not None else None
        da = ar.denoised if ar is not None else None
        return (_anc_latent(vs, dv, s2, nv, terminal, eta, s_noise, model_dtype),
                _anc_latent(as_, da, s2, na, terminal, eta, s_noise, model_dtype))
    key = ("anc", id(transformer))
    if key not in _cached:
        _cached[key] = torch.compile(body, backend="neuron", dynamic=False, fullgraph=_FG)
    cb = _cached[key]
    for i, (s2, nv, na, terminal) in enumerate(steps):
        vl, al = cb(video_state, audio_state, s2, nv, na, v_context, a_context, terminal)
        if vl is not None:
            video_state = _rp(video_state, latent=vl)
        if al is not None:
            audio_state = _rp(audio_state, latent=al)
        if os.environ.get("LTX_STEP_DEBUG") == "1" and rank == 0 and vl is not None:
            f = vl.float()
            print(f"[STEP] {i} mean {f.mean().item():.4f} std {f.std().item():.4f} nan {int(torch.isnan(f).sum())}",
                  flush=True)
    return _SM.VideoAudio(video_state, audio_state)


if os.environ.get("LTX_STEP_GRAPH", "0") == "1":
    _SM.euler_ancestral_denoising_loop = _anc_step_graph_loop
    D.euler_ancestral_denoising_loop = _anc_step_graph_loop
    _SM.euler_denoising_loop = _step_graph_loop
    DS.euler_denoising_loop = _step_graph_loop

# LTX_COMPILE=model: the perturbation checks read a host tensor (bool(...)): one graph break per block. Before each
# compiled call the host mask is checked eagerly: when nothing is perturbed (the distilled pipeline: no guidance
# passes), the checks return the constant False inside the graph; otherwise the call fails instead of running wrong.
from ltx_core.guidance.perturbations import BatchedPerturbationConfig as _BPC  # noqa: E402
_any0, _all0 = _BPC.any_in_batch, _BPC.all_in_batch
_NOPERT = [False]   # module level: batch_split copies the config, so a per-object flag would be lost
_BPC.any_in_batch = lambda self, t, b: False if _NOPERT[0] else _any0(self, t, b)
_BPC.all_in_batch = lambda self, t, b: False if _NOPERT[0] else _all0(self, t, b)


def _mark_unperturbed(perturbations):
    if perturbations is not None:
        assert bool((perturbations._block_masks.host_tensor != 0).all()), "perturbed batch under LTX_COMPILE=model"
    _NOPERT[0] = True


def _te_patch(cls):
    """LTX_TE_NEURON=1: the Gemma text encoder built on CPU once, tensor-parallel over the ranks, on this rank's core,
    every decoder layer compiled."""
    f = cls._build_text_encoder

    def b(self):
        if "te" in _cached:
            return _cached["te"]
        t0 = time.time()
        dev0 = self._device
        self._device = torch.device("cpu")
        try:
            m = f(self)
        finally:
            self._device = dev0
        tp.shard_gemma(m, rank, world, dist.group.WORLD if world > 1 else None)
        m = m.to(dit_device)
        for mod in m.modules():
            if type(mod).__name__ == "Gemma4TextDecoderLayer":
                mod.forward = torch.compile(mod.forward, backend="neuron", dynamic=False, fullgraph=_FG)
        if rank == 0:
            print(f"[te] gemma on {dit_device} in {time.time() - t0:.1f}s", flush=True)
        _cached["te"] = m
        return m
    cls._build_text_encoder = b


def _build_transformer(self, *, device=None, **kw):
    """Build on CPU once, shard over the TP group, move to this rank's core, compile every block; reuse after."""
    key = "dit"   # one DiT per process: reused across pipeline instances (LTX_RUNS > 1)
    if key in _cached:
        return _cached[key]
    t0 = time.time()
    m = _orig_build(self, device=torch.device("cpu"), **kw)
    if DIT_DEV == "neuron":
        tp.shard_transformer(m, rank, world, dist.group.WORLD if world > 1 else None)
        if os.environ.get("LTX_SP", "0") == "1" and world > 1:
            tp.enable_sequence_parallel(m, rank, world, dist.group.WORLD)
        m = m.to(dit_device)
        if os.environ.get("LTX_COMPILE", "model") == "model":
            # LTX_COMPILE=model: the whole DiT forward (embeddings, AdaLN, RoPE, every block, the head) as one graph
            _cf = torch.compile(m.forward, backend="neuron", dynamic=False, fullgraph=_FG)

            def _fw(video, audio, perturbations, _cf=_cf):
                _mark_unperturbed(perturbations)
                return _cf(video, audio, perturbations)
            m.forward = _fw
        else:
            for mod in m.modules():
                blocks = getattr(mod, "transformer_blocks", None)
                if blocks is not None:
                    for b in blocks:
                        b.forward = torch.compile(b.forward, backend="neuron", dynamic=False, fullgraph=_FG)
        m = _wrap_io(m)
    if rank == 0:
        print(f"[tp] DiT built, sharded and placed in {time.time() - t0:.1f} s on {dit_device}", flush=True)
    _cached[key] = m
    return m


DS.DiffusionStage._build_transformer = _build_transformer

# keep the DiT resident across calls (the pipeline frees models after each use)
_orig_lifetime = DS.DiffusionStage._transformer_lifetime


def _lifetime(self, transformer):
    import contextlib
    return contextlib.nullcontext(transformer)


DS.DiffusionStage._transformer_lifetime = _lifetime

# the DiT runs on neuron but receives CPU tensors from the CPU stages: move inputs in, outputs out
import dataclasses


def _move(x, dev):
    if isinstance(x, torch.Tensor):
        return x.to(dev)
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return dataclasses.replace(x, **{f.name: _move(getattr(x, f.name), dev) for f in dataclasses.fields(x) if f.init})
    if isinstance(x, tuple) and hasattr(x, "_fields"):
        return type(x)(*(_move(v, dev) for v in x))
    if isinstance(x, (list, tuple)):
        return type(x)(_move(v, dev) for v in x)
    if isinstance(x, dict):
        return {k: _move(v, dev) for k, v in x.items()}
    return x


def _wrap_io(m):
    f = m.forward

    @functools.wraps(f)
    def w(*a, **k):
        out = f(*_move(a, dit_device), **_move(k, dit_device))
        return out if os.environ.get("LTX_ALL_NEURON", "0") == "1" else _move(out, torch.device("cpu"))
    m.forward = w
    return m

T = collections.defaultdict(float)
N = collections.Counter()


def timed(cls, name):
    f = cls.__call__

    @functools.wraps(f)
    def w(self, *a, **k):
        t0 = time.perf_counter()
        r = f(self, *a, **k)
        T[name] += time.perf_counter() - t0
        N[name] += 1
        return r
    cls.__call__ = w


import ltx_pipelines.utils as U  # noqa: E402
_cache_prompt(U.PromptEncoder)
if os.environ.get("LTX_TE_NEURON", "0") == "1":
    _te_patch(U.PromptEncoder)
for cname, name in (("PromptEncoder", "text_encoder"), ("DiffusionStage", "dit"), ("VideoUpsampler", "upsampler"),
                    ("VideoDecoder", "video_decoder"), ("AudioDecoder", "audio_decoder")):
    cls = getattr(U, cname, None)
    if cls is not None:
        timed(cls, name)

if rank != 0:
    argv = sys.argv[1:]
    if "--output-path" in argv:
        argv[argv.index("--output-path") + 1] = f"/tmp/ltx_rank{rank}.mp4"
    sys.argv = [sys.argv[0]] + argv
# --- the conv video decoder on this rank's Neuron core, compiled once (LTX_VAE_NEURON=1, the default) ---
from ltx_pipelines.utils.blocks import video_decoder as VDm  # noqa: E402
_vdec = {}
_orig_vd_call = VDm.VideoDecoder.__call__

# the causal conv pads time into an empty(memory_format=...) buffer with three slice writes; aten.empty.memory_format
# does not lower on Neuron. One cat of the same frames: byte-identical values.
from ltx_core.model.video_vae import convolution as _CV  # noqa: E402


# LTX_VAE_SPLIT=1: the conv decoder runs on a gh x gw grid of spatial tiles, one per rank (the trace: the decode was
# 1123 ms replicated on all 16 ranks). The decoder's only cross-pixel ops are its 3x3x3 convs (pixel norm, no
# attention, no noise, zeros padding), so each conv gets a 1-pixel halo from its neighbours (one all-gather per axis)
# and the rank's tile is exact. The tiles are gathered into the full video at the end.
_VSPLIT = {}


def _halo(x, p, axis):
    """Pad x [B,C,T,H,W] by p on spatial axis (3=H, 4=W) with the neighbouring tiles' edges; zeros at the borders."""
    g = _VSPLIT
    lo, hi = x.narrow(axis, 0, p), x.narrow(axis, x.shape[axis] - p, p)
    allg = funcol.all_gather_tensor(torch.stack([lo, hi]).contiguous(), 0, g["group"])   # [2*world, ...]
    i, j, gh, gw = g["i"], g["j"], g["gh"], g["gw"]
    if axis == 3:
        prev = allg[2 * ((i - 1) * gw + j) + 1] if i > 0 else torch.zeros_like(hi)
        nxt = allg[2 * ((i + 1) * gw + j)] if i < gh - 1 else torch.zeros_like(lo)
    else:
        prev = allg[2 * (i * gw + j - 1) + 1] if j > 0 else torch.zeros_like(hi)
        nxt = allg[2 * (i * gw + j + 1)] if j < gw - 1 else torch.zeros_like(lo)
    return torch.cat([prev, x, nxt], dim=axis)


def _cc3d_forward(self, x, causal=True):
    front = self.time_kernel_size - 1 if causal else (self.time_kernel_size - 1) // 2
    back = 0 if causal else front
    if front or back:
        parts = [x[:, :, :1].expand(-1, -1, front, -1, -1), x]
        if back:
            parts.append(x[:, :, -1:].expand(-1, -1, back, -1, -1))
        x = torch.cat(parts, dim=2)
    c = self.conv
    if _VSPLIT and (c.padding[1] or c.padding[2]):
        assert c.stride[1:] == (1, 1) and c.padding_mode == "zeros", (c.stride, c.padding_mode)
        x = _halo(_halo(x, c.padding[1], 3), c.padding[2], 4)
        return torch.nn.functional.conv3d(x, c.weight, c.bias, c.stride, (0, 0, 0), c.dilation, c.groups)
    return c(x)


def _split_decode(dec, latent):
    """The decoder forward on this rank's latent tile, then the tiles gathered into the full video (one graph)."""
    g = _VSPLIT
    gh, gw, i, j = g["gh"], g["gw"], g["i"], g["j"]
    H, W = latent.shape[3], latent.shape[4]
    tile = latent[:, :, :, i * H // gh:(i + 1) * H // gh, j * W // gw:(j + 1) * W // gw]
    out = dec._orig_forward(tile)                                                           # [1,3,F,h,w]
    allt = funcol.all_gather_tensor(out.contiguous(), 0, g["group"])                        # [world,3,F,h,w]
    b, c, f, h, w = out.shape
    allt = allt.view(gh, gw, c, f, h, w).permute(2, 3, 0, 4, 1, 5).reshape(1, c, f, gh * h, gw * w)
    return allt


if DIT_DEV == "neuron":
    _CV.CausalConv3d.forward = _cc3d_forward

# --- vocoder (the trace: 587 ms, GpSimd 95% busy) -- LTX_VOC_FAST=1 ---
# UpSample1d's conv_transpose1d reverses its filter every call (134 ms of reverse DMA on Scalar) and runs on a
# zero-stuffed input; replicate padding lowers to an indirect gather. Polyphase form instead: y[s*q+p] =
# sum_j f[p+s*j] x[q-j], i.e. s depthwise conv1d with the reversed sub-filters, interleaved; replicate padding
# as one cat of edge copies. Same products and sums per output sample.
from ltx_core.model.audio_vae import vocoder as _VOC  # noqa: E402


def _rep_pad(x, left, right):
    parts = []
    if left:
        parts.append(x[..., :1].expand(*x.shape[:-1], left))
    parts.append(x)
    if right:
        parts.append(x[..., -1:].expand(*x.shape[:-1], right))
    return torch.cat(parts, dim=-1) if len(parts) > 1 else x


def _up_forward(self, x):
    b, c, _ = x.shape
    s, k = self.stride, self.kernel_size
    j = -(-k // s)                                                              # ceil(K / s)
    x = _rep_pad(x, self.pad, self.pad)
    lp = x.shape[-1]
    f = self.filter.to(dtype=x.dtype, device=x.device).reshape(-1)
    if j * s > k:                                                               # zero taps: same sums
        f = torch.cat([f, f.new_zeros(j * s - k)])
    w = torch.stack([f[p::s].flip(0) for p in range(s)])                         # [s, J]
    w = w.unsqueeze(0).expand(c, s, j).reshape(c * s, 1, j)
    h = torch.nn.functional.conv1d(torch.nn.functional.pad(x, (j - 1, j - 1)), w, groups=c)   # [B, C*s, Lp+J-1]
    y = h.view(b, c, s, -1).transpose(2, 3).reshape(b, c, -1)
    n = (lp - 1) * s + k                                                        # conv_transpose1d output length
    return self.ratio * y[..., self.pad_left: n - self.pad_right]


def _lp_forward(self, x):
    _, c, _ = x.shape
    if self.padding:
        if self.padding_mode == "replicate":
            x = _rep_pad(x, self.pad_left, self.pad_right)
        else:
            x = torch.nn.functional.pad(x, (self.pad_left, self.pad_right), mode=self.padding_mode)
    return torch.nn.functional.conv1d(x, self.filter.expand(c, -1, -1), stride=self.stride, groups=c)


# LTX_VOC_MM=1: the vocoder's depthwise filters (one filter shared by all channels) as banded matmuls on the Tensor
# engine (the trace after VOC_FAST: Vector 70% busy on run_tp.py:517/530, Tensor 36%). The time axis is cut into
# blocks of P outputs; each block's W <= 128 input samples (the block plus the filter's overlap) are gathered once
# into a contiguous [.., nb, W] tile and multiplied by a constant [W, P/s * R] Toeplitz matrix. Same products,
# summed in a different order (fp32).
def _fir_mats(ws, stride):
    """ws [R, K] (R interleaved output phases) -> (M [W, (P/stride)*R], P)."""
    r_, k = ws.shape
    p = ((128 - k + stride) // stride) * stride
    w = p - stride + k
    n = p // stride
    m = torch.zeros(w, n * r_, dtype=ws.dtype)
    for t in range(n):
        for r in range(r_):
            m[t * stride:t * stride + k, t * r_ + r] = ws[r]
    return m, p


def _fir_mm(x, mat, p, k, stride, r_):
    """Valid-mode y[t*R + r] = sum_j ws[r, j] x[t*stride + j] for every channel, via one matmul."""
    b, c, t_in = x.shape
    t_out = (t_in - k) // stride + 1
    nb = -(-(t_out * stride) // p)
    x = torch.nn.functional.pad(x, (0, (nb + 1) * p - t_in))
    main = x[..., :nb * p].reshape(b, c, nb, p)
    tail = x[..., p:(nb + 1) * p].reshape(b, c, nb, p)[..., :k - stride]
    y = torch.matmul(torch.cat([main, tail], dim=-1), mat.to(x.dtype))
    return y.reshape(b, c, -1)[..., :t_out * r_]


def _prep_voc_mm(model):
    for m in model.modules():
        if isinstance(m, _VOC.UpSample1d):
            s_, k = m.stride, m.kernel_size
            j = -(-k // s_)
            f = m.filter.detach().double().reshape(-1).cpu()
            if j * s_ > k:
                f = torch.cat([f, f.new_zeros(j * s_ - k)])
            ws = torch.stack([f[q::s_].flip(0) for q in range(s_)])
            mat, p = _fir_mats(ws, 1)
            m._mm = (mat.float().to(m.filter.device), p, j, 1, s_)
        elif isinstance(m, _VOC.LowPassFilter1d):
            ws = m.filter.detach().double().reshape(1, -1).cpu()
            mat, p = _fir_mats(ws, m.stride)
            m._mm = (mat.float().to(m.filter.device), p, m.kernel_size, m.stride, 1)


def _up_forward_mm(self, x):
    b, c, _ = x.shape
    mat, p, j, st, s_ = self._mm
    x = _rep_pad(x, self.pad, self.pad)
    lp = x.shape[-1]
    y = _fir_mm(torch.nn.functional.pad(x, (j - 1, j - 1)), mat, p, j, st, s_)
    n = (lp - 1) * s_ + self.kernel_size
    return self.ratio * y[..., self.pad_left: n - self.pad_right]


def _lp_forward_mm(self, x):
    mat, p, k, st, r_ = self._mm
    if self.padding:
        if self.padding_mode == "replicate":
            x = _rep_pad(x, self.pad_left, self.pad_right)
        else:
            x = torch.nn.functional.pad(x, (self.pad_left, self.pad_right), mode=self.padding_mode)
    return _fir_mm(x, mat, p, k, st, r_)


if DIT_DEV == "neuron" and os.environ.get("LTX_VOC_FAST", "0") == "1":
    _VOC.UpSample1d.forward = _up_forward
    _VOC.LowPassFilter1d.forward = _lp_forward
    if os.environ.get("LTX_VOC_MM", "0") == "1":
        _VOC.UpSample1d.forward = _up_forward_mm
        _VOC.LowPassFilter1d.forward = _lp_forward_mm


def _vd_call(self, latent, tiling_config=None, generator=None, **kw):
    if DIT_DEV != "neuron" or os.environ.get("LTX_VAE_NEURON", "1") != "1":
        return _orig_vd_call(self, latent, tiling_config, generator, **kw)
    if "dec" not in _vdec:
        from ltx_core.model.video_vae.memory_efficient_decode import disable_memory_efficient_decode
        dec = self._prepared_builder().build(device=dit_device, dtype=torch.bfloat16).eval()
        dec = disable_memory_efficient_decode(dec)
        if os.environ.get("LTX_VAE_SPLIT", "0") == "1" and world > 1:
            gs = int(os.environ.get("LTX_VAE_SPLIT_RANKS", world)); gh = int(os.environ.get("LTX_VAE_GRID_H", "4")); gw = gs // gh
            assert gh * gw == gs and world % gs == 0, (gh, gw, gs, world)   # gs < world: world/gs groups decode the same latent
            _VSPLIT.update(group=dist.group.WORLD if gs == world else _VGROUPS[rank // gs], gh=gh, gw=gw, i=rank % gs // gw, j=rank % gw)
            dec._orig_forward = dec.forward
            _sf = torch.compile(functools.partial(_split_decode, dec), backend="neuron", dynamic=False, fullgraph=_FG)
            dec.forward = lambda sample, timestep=None, generator=None, _sf=_sf: _sf(sample)
        else:
            dec.forward = torch.compile(dec.forward, backend="neuron", dynamic=False, fullgraph=_FG)
        _vdec["dec"] = dec
    dec = _vdec["dec"]
    if rank == 0:   # the latent the VAE gets: a NaN / zero latent is a broken stage upstream, not a VAE problem
        lf = latent.float()
        print(f"[LATENT] mean {lf.mean().item():.4f} std {lf.std().item():.4f} nan {int(torch.isnan(lf).sum())}", flush=True)
    with torch.no_grad():
        chunks = dec.decode_video(latent.to(dit_device, torch.bfloat16), None, generator)
        return iter([c.cpu() for c in chunks])


VDm.VideoDecoder.__call__ = _vd_call
# --- the latent upsampler and the audio decoder + vocoder on this rank's Neuron core, compiled once ---
from ltx_pipelines.utils.blocks import video_upsampler as VUm, audio_decoder as ADm  # noqa: E402
from ltx_core.model.upsampler.model import upsample_video  # noqa: E402
from ltx_core.model.audio_vae.audio_vae import decode_audio  # noqa: E402
_orig_up_call, _orig_ad_call = VUm.VideoUpsampler.__call__, ADm.AudioDecoder.__call__
_res = {}


def _on_neuron():
    return DIT_DEV == "neuron" and os.environ.get("LTX_STAGES_NEURON", "1") == "1"


def _up_call(self, latent):
    if not _on_neuron():
        return _orig_up_call(self, latent)
    if "up" not in _res:
        enc = self._encoder_builder.build(device=dit_device, dtype=self._dtype).eval()
        up = self._upsampler_builder.build(device=dit_device, dtype=self._dtype).eval()
        up.forward = torch.compile(up.forward, backend="neuron", dynamic=False, fullgraph=_FG)
        _res["up"] = (enc, up)
    enc, up = _res["up"]
    with torch.no_grad():
        out = upsample_video(latent=latent.to(dit_device, self._dtype), video_encoder=enc, upsampler=up)
        return out if os.environ.get("LTX_ALL_NEURON", "0") == "1" else out.cpu()


def _ad_call(self, latent):
    if not _on_neuron():
        return _orig_ad_call(self, latent)
    if "ad" not in _res:
        dec = self._decoder_builder.build(device=dit_device, dtype=self._dtype).eval()
        voc = self._vocoder_builder.build(device=dit_device, dtype=torch.float32).eval()
        dec.forward = torch.compile(dec.forward, backend="neuron", dynamic=False, fullgraph=_FG)
        if os.environ.get("LTX_VOC_MM", "0") == "1":
            _prep_voc_mm(voc)
        voc.forward = torch.compile(voc.forward, backend="neuron", dynamic=False, fullgraph=_FG)
        _res["ad"] = (dec, voc)
    dec, voc = _res["ad"]
    with torch.no_grad():
        a = decode_audio(latent.to(dit_device, self._dtype), dec, voc)
    return type(a)(waveform=a.waveform.cpu(), sampling_rate=a.sampling_rate)


VUm.VideoUpsampler.__call__ = _up_call
ADm.AudioDecoder.__call__ = _ad_call
_enc = D.encode_media_segments


def _timed_encode(*a, **k):
    t0 = time.perf_counter()
    r = _enc(*a, **k)
    T["decode+encode (video, audio, mp4)"] += time.perf_counter() - t0
    N["decode+encode (video, audio, mp4)"] += 1
    return r


D.encode_media_segments = _timed_encode
sys.argv = ["distilled"] + sys.argv[1:]
import extra  # noqa: E402  (new patches: see extra.py, keeps the graph source lines above unchanged)
extra.install(globals())
if os.environ.get("LTX_SERVE", "0") == "1":   # the demo server (ltx_serve.py): colo / dit / vae roles
    import ltx_serve  # noqa: E402
    ltx_serve.main(globals())
    sys.exit(0)
for run in range(int(os.environ.get("LTX_RUNS", "1"))):
    # LTX_RUNS=N: N generations in one process; the DiT is built and compiled once, so runs after the first are warm
    T.clear(); N.clear()
    if run > 0:
        # an idle gap between runs: the system trace's last run is everything after the last gap (attribution)
        time.sleep(float(os.environ.get("LTX_RUN_GAP_S", "0")))
    t0 = time.perf_counter()
    D.main()
    if rank == 0:
        for k in T:
            print(f"[STAGE] run {run} {k}: {T[k] * 1000:.0f} ms over {N[k]} calls", flush=True)
        print(f"[STAGE] run {run} total: {(time.perf_counter() - t0) * 1000:.0f} ms", flush=True)
    extra.report(rank)
