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
        setattr(_m, "get_device", lambda: torch.device("cpu"))

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
            _prompts[key] = f(self, prompts, *a, **k)
        return _prompts[key]
    cls.__call__ = w

_cached = {}
_orig_build = DS.DiffusionStage._build_transformer


def _build_transformer(self, *, device=None, **kw):
    """Build on CPU once, shard over the TP group, move to this rank's core, compile every block; reuse after."""
    key = "dit"   # one DiT per process: reused across pipeline instances (LTX_RUNS > 1)
    if key in _cached:
        return _cached[key]
    t0 = time.time()
    m = _orig_build(self, device=torch.device("cpu"), **kw)
    if DIT_DEV == "neuron":
        tp.shard_transformer(m, rank, world, dist.group.WORLD if world > 1 else None)
        m = _wrap_io(m.to(dit_device))
        for mod in m.modules():
            blocks = getattr(mod, "transformer_blocks", None)
            if blocks is not None:
                for b in blocks:
                    b.forward = torch.compile(b.forward, backend="neuron", dynamic=False)
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
        return _move(out, torch.device("cpu"))
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


def _vd_call(self, latent, tiling_config=None, generator=None, **kw):
    if DIT_DEV != "neuron" or os.environ.get("LTX_VAE_NEURON", "1") != "1":
        return _orig_vd_call(self, latent, tiling_config, generator, **kw)
    if "dec" not in _vdec:
        from ltx_core.model.video_vae.memory_efficient_decode import disable_memory_efficient_decode
        dec = self._prepared_builder().build(device=dit_device, dtype=torch.bfloat16).eval()
        dec = disable_memory_efficient_decode(dec)
        dec.forward = torch.compile(dec.forward, backend="neuron", dynamic=False)
        _vdec["dec"] = dec
    dec = _vdec["dec"]
    with torch.no_grad():
        chunks = dec.decode_video(latent.to(dit_device, torch.bfloat16), None, generator)
        return iter([c.cpu() for c in chunks])


VDm.VideoDecoder.__call__ = _vd_call
_enc = D.encode_media_segments


def _timed_encode(*a, **k):
    t0 = time.perf_counter()
    r = _enc(*a, **k)
    T["decode+encode (video, audio, mp4)"] += time.perf_counter() - t0
    N["decode+encode (video, audio, mp4)"] += 1
    return r


D.encode_media_segments = _timed_encode
sys.argv = ["distilled"] + sys.argv[1:]
for run in range(int(os.environ.get("LTX_RUNS", "1"))):
    # LTX_RUNS=N: N generations in one process; the DiT is built and compiled once, so runs after the first are warm
    T.clear(); N.clear()
    t0 = time.perf_counter()
    D.main()
    if rank == 0:
        for k in T:
            print(f"[STAGE] run {run} {k}: {T[k] * 1000:.0f} ms over {N[k]} calls", flush=True)
        print(f"[STAGE] run {run} total: {(time.perf_counter() - t0) * 1000:.0f} ms", flush=True)
