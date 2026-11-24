"""Patches applied after run_tp.py's definitions and before its run loop.

New patches live here, not in run_tp.py: the compiled graphs carry run_tp.py source lines (file:line metadata), so
editing run_tp.py above a compiled function shifts its lines, changes the graph cache key and recompiles the DiT.

LTX_TE_FREE=1: free the CPU Gemma text encoder once the prompt encoding is cached. Each rank held ~46 GB of host
memory (16 ranks ~740 GB per run), the resident 12B text encoder (~24 GB bf16) the largest part; the encoding itself
is cached per prompt, so the encoder is not needed again for the same prompt.
LTX_MEMDEBUG=1: after each run, rank 0 prints its RSS and the largest groups of CPU tensors.
"""
import gc
import os

import torch


def install(rt):
    """rt: the run_tp module namespace (globals())."""
    g = int(os.environ.get("LTX_BUILD_GROUP", "0"))
    if g > 0 and rt["world"] > 1:
        # LTX_BUILD_GROUP=G: the ranks build the DiT on CPU G at a time. All 16 at once peak at ~16 x 44 GB of host
        # memory (the l-lnc2 pod with a 900 GiB limit was OOM-killed); G=4 peaks at ~4 x 44 GB.
        DS, dist, cached, rank, world = rt["DS"], rt["dist"], rt["_cached"], rt["rank"], rt["world"]
        bt = DS.DiffusionStage._build_transformer

        def staged(self, *a, **k):
            if "dit" in cached:
                return bt(self, *a, **k)
            m = None
            for i in range(0, world, g):
                if i <= rank < i + g:
                    m = bt(self, *a, **k)
                    gc.collect()
                dist.barrier()
            return m
        DS.DiffusionStage._build_transformer = staged
    if int(os.environ.get("LTX_VOC_SPLIT", "0")) > 1 and rt["world"] > 1:
        _install_voc_split(rt, int(os.environ["LTX_VOC_SPLIT"]))
    if os.environ.get("LTX_TE_FREE", "0") == "1":
        U = rt["U"]
        built = rt["_built"]
        f = U.PromptEncoder.__call__

        def w(self, *a, **k):
            out = f(self, *a, **k)
            ids = {id(b) for b in (getattr(self, "_text_encoder_builder", None),
                                   getattr(self, "_enhancer_text_encoder_builder", None)) if b is not None}
            drop = [key for key in built if key[0] in ids]
            for key in drop:
                del built[key]
            if drop:
                gc.collect()
            return out
        U.PromptEncoder.__call__ = w


def report(rank):
    if os.environ.get("LTX_MEMDEBUG", "0") != "1" or rank != 0:
        return
    rss = int(open("/proc/self/status").read().split("VmRSS:")[1].split()[0]) / 2**20
    groups = {}
    for o in gc.get_objects():
        try:
            if isinstance(o, torch.Tensor) and o.device.type == "cpu" and not o.is_meta:
                n = o.untyped_storage().nbytes()
                k = (str(o.dtype), len(o.shape))
                c, b = groups.get(k, (0, 0))
                groups[k] = (c + 1, b + n)
        except Exception:
            pass
    top = sorted(groups.items(), key=lambda kv: -kv[1][1])[:6]
    print(f"[MEM] rss {rss:.1f} GB; cpu tensors: " +
          ", ".join(f"{k[0]}/{k[1]}d x{c} {b / 2**30:.1f}GB" for k, (c, b) in top), flush=True)


# ---------------- LTX_VOC_SPLIT=G: the vocoder split along time over G ranks ----------------
# The trace of the l-lnc2 baseline: the vocoder + BWE graph is 382 ms, run whole on every rank. Its only cross-time
# ops are the padded Conv1d / ConvTranspose1d, the anti-alias up/down filters (replicate padding), and the causal
# STFT left pad; everything else is pointwise. Split along time, each of those ops takes its halo from the
# neighbouring ranks (one all-gather per side) and keeps the stock padding at the true borders, so every rank computes
# exactly its slice of the stock output. The mel input is cut into G equal slices (ranks i, i+G, ... repeat the same
# slice); the slices are gathered at the end.
import torch.nn.functional as F  # noqa: E402
import torch.distributed._functional_collectives as funcol  # noqa: E402

_AS = {"on": False, "t": 1, "v": 1}


def _side(x, left, right, mode, kind=""):
    """[prev rank's last `left` samples, x, next rank's first `right` samples] (stock padding at the borders)."""
    g, i, grp = _AS["g"], _AS["i"], _AS["group"]
    if kind and kind in os.environ.get("LTX_VOC_NOHALO", "").split(","):   # compile bisect only: local padding, not exact
        pad = [x[..., :1].expand(*x.shape[:-1], left)] if (left and mode == "replicate") else \
            ([x.new_zeros(*x.shape[:-1], left)] if left else [])
        post = [x[..., -1:].expand(*x.shape[:-1], right)] if (right and mode == "replicate") else \
            ([x.new_zeros(*x.shape[:-1], right)] if right else [])
        return torch.cat(pad + [x] + post, dim=-1)
    L = x.shape[-1]
    valid = L * _AS["v"] // _AS["t"]
    if valid < L:
        # the last slice is partial: past the true end the stock op sees its border padding (zeros / the last sample)
        keep = torch.arange(L, device=x.device) < valid
        fill = x[..., valid - 1:valid] if mode == "replicate" else torch.zeros_like(x[..., :1])
        x = torch.where(keep, x, fill)
    # one all-gather per op for both edges (rule 1: fewer, bigger collectives; two per op put the vocoder graph at
    # ~1200 collectives and it failed at execution, ~800 ran)
    b = x.shape[0]
    a = funcol.all_gather_tensor(torch.cat([x[..., :right], x[..., L - left:]], dim=-1).contiguous(), 0, grp) \
        if (left or right) else None                                               # [g*B, C, right + left]
    parts = []
    if left:
        if i > 0:
            parts.append(a[(i - 1) * b:i * b, ..., right:])
        elif mode == "replicate":
            parts.append(x[..., :1].expand(*x.shape[:-1], left))
        else:
            parts.append(x.new_zeros(*x.shape[:-1], left))
    parts.append(x)
    if right:
        if i < g - 1:
            parts.append(a[(i + 1) * b:(i + 2) * b, ..., :right])
        elif mode == "replicate":
            parts.append(x[..., -1:].expand(*x.shape[:-1], right))
        else:
            parts.append(x.new_zeros(*x.shape[:-1], right))
    return torch.cat(parts, dim=-1)


def _install_voc_split(rt, g):
    dist, rank, world = rt["dist"], rt["rank"], rt["world"]
    from ltx_core.model.audio_vae import vocoder as V
    groups = [dist.new_group(list(range(j, min(j + g, world)))) for j in range(0, world, g)]
    _AS.update(g=g, i=rank % g, group=groups[rank // g])
    conv1d_fwd, convt_fwd = torch.nn.Conv1d.forward, torch.nn.ConvTranspose1d.forward

    def conv1d(self, x):
        if not (_AS["on"] and getattr(self, "_vs", False)) or self.padding[0] == 0:
            return conv1d_fwd(self, x)
        assert self.padding_mode == "zeros" and self.stride[0] == 1
        p = self.padding[0]
        return F.conv1d(_side(x, p, p, "zeros", "conv"), self.weight, self.bias, 1, 0, self.dilation, self.groups)

    def convt(self, x):
        # gathered form: the stock conv_transpose1d on the full (gathered) input on every rank, then this rank's
        # slice of the output. The halo-padded conv_transpose1d hit NCC_IIIT901; the overlap-add and polyphase forms
        # failed at execution ("Invalid NEFF"). The upsample layers are a small part of the vocoder's work.
        if not (_AS["on"] and getattr(self, "_vs", False)):
            return convt_fwd(self, x)
        L = x.shape[-1]
        valid = L * _AS["v"] // _AS["t"]
        if valid < L:
            x = torch.where(torch.arange(L, device=x.device) < valid, x, torch.zeros_like(x[..., :1]))
        b, c, _ = x.shape
        full = funcol.all_gather_tensor(x.reshape(b * c, L).contiguous(), 1, _AS["group"]).reshape(b, c, -1)
        y = convt_fwd(self, full)                                                    # [B, Cout, g*L*s]
        n = L * self.stride[0]
        return y[..., _AS["i"] * n:(_AS["i"] + 1) * n]

    torch.nn.Conv1d.forward, torch.nn.ConvTranspose1d.forward = conv1d, convt
    up_fwd, lp_fwd, stft_fwd, bwe_fwd = V.UpSample1d.forward, V.LowPassFilter1d.forward, V._STFTFn.forward, \
        V.VocoderWithBWE.forward
    rep_pad = rt["_rep_pad"]

    def with_halo(fwd):
        def f(self, x):
            if not _AS["on"]:
                return fwd(self, x)
            rt["_rep_pad"] = lambda t, l, r: _side(t, l, r, "replicate", "updown")   # the stock pad, from the neighbours
            try:
                return fwd(self, x)
            finally:
                rt["_rep_pad"] = rep_pad
        return f
    V.UpSample1d.forward, V.LowPassFilter1d.forward = with_halo(up_fwd), with_halo(lp_fwd)

    def stft(self, y):
        if not _AS["on"]:
            return stft_fwd(self, y)
        if y.dim() == 2:
            y = y.unsqueeze(1)
        left = max(0, self.win_length - self.hop_length)
        y = _side(y, left, 0, "zeros", "stft")
        spec = F.conv1d(y, self.forward_basis, stride=self.hop_length, padding=0)
        n = spec.shape[1] // 2
        real, imag = spec[:, :n], spec[:, n:]
        return torch.sqrt(real ** 2 + imag ** 2), torch.atan2(imag.float(), real.float()).to(real.dtype)
    V._STFTFn.forward = stft

    def bwe(self, mel_spec):
        T = mel_spec.shape[2]
        t = -(-T // g)                                   # frames per slice; the last slice may be partial
        v = max(0, min(t, T - _AS["i"] * t))             # this rank's real frames
        assert v > 0, f"{T} mel frames leave rank slot {_AS['i']} empty over {g} ranks"
        if t * g > T:
            mel_spec = F.pad(mel_spec, (0, 0, 0, t * g - T))   # values unused: masked before every cross-time op
        _AS.update(on=True, t=t, v=v)
        try:
            y = bwe_fwd(self, mel_spec[:, :, _AS["i"] * t:(_AS["i"] + 1) * t])         # [B, C, L]
        finally:
            _AS["on"] = False
        b, c, L = y.shape
        if os.environ.get("LTX_VOC_GATHER", "time") == "time":
            # gather along the time axis directly (no output permute: the permute form hit NCC_IIIT901
            # InsertIOTransposes "invalid split of dag" in the compiler)
            a = funcol.all_gather_tensor(y.reshape(b * c, L).contiguous(), 1, _AS["group"])   # [B*C, g*L]
            return a.reshape(b, c, g * L)[..., :T * (L // t)]
        a = funcol.all_gather_tensor(y.contiguous(), 0, _AS["group"])                # [g*B, C, L]
        return a.view(g, b, c, L).permute(1, 2, 0, 3).reshape(b, c, g * L)[..., :T * (L // t)]
    V.VocoderWithBWE.forward = bwe

    ADm = rt["ADm"]
    call = ADm.AudioDecoder.__call__

    def ad_call(self, latent):
        voc = self._vocoder_builder.build(device=rt["dit_device"], dtype=torch.float32)   # memoized: the one compiled
        for m in voc.modules():
            if isinstance(m, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
                m._vs = True
        return call(self, latent)
    ADm.AudioDecoder.__call__ = ad_call
