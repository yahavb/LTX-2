"""Serve-path host fixes (imported by ltx_serve.py only, so run_tp.py / extra.py line numbers and graph keys stay put).

LTX_STATE_CACHE=1: VideoLatentTools / AudioLatentTools.create_initial_state without an initial latent build the same
zeros / ones / patch-grid / pixel-position tensors every chunk, op by op on the Neuron device (each eager op a device
round trip). py-spy on the co-lo rank 0 (2026-11-24): 30.6% of the main thread's wall time in that function. The
state depends only on (tools, device, dtype), so it is built once and cloned per call. Exact.
LTX_TE_BCAST_CACHE=1: with LTX_TE_RANK0 every chunk broadcast the (cached) prompt encoding from rank 0 to all ranks as
a pickled object (9.0% of the main thread). Rank 0 now broadcasts only the prompt key; the other ranks keep the last
encoding and the full object is sent only when the prompt changes. Exact.
"""
import dataclasses
import os

import torch


def _clone(obj):
    if isinstance(obj, torch.Tensor):
        return obj.clone()
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.replace(obj, **{f.name: _clone(getattr(obj, f.name)) for f in dataclasses.fields(obj)
                                           if f.init})
    return obj


def install(rt):
    if os.environ.get("LTX_REUSE_PIPE", "0") == "1":
        # D.main built a new DistilledPipeline (prompt encoder, stage, upsampler, decoders: builders + checks) for every
        # chunk while the device idled at the chunk boundary. The pipeline keeps no per-run state (its __init__ is the
        # only place it assigns attributes), so one instance per constructor arguments is reused. Exact.
        import ltx_pipelines.distilled as DI
        made = {}

        def pipeline(*a, _cls=DI.DistilledPipeline, **k):   # bound now: a later loop in install() rebinds names
            key = repr((a, sorted(k.items())))
            if key not in made:
                made[key] = _cls(*a, **k)
                if rt["rank"] == 0:
                    print(f"[REUSE] built pipeline #{len(made)}", flush=True)
            return made[key]
        DI.DistilledPipeline = pipeline
    if os.environ.get("LTX_HOST_MEMO", "0") == "1":
        # Per-chunk host work while the device idles (DiT pod trace: ~250 ms of gaps per chunk; py-spy: checkpoint
        # header JSON parse + safetensors safe_open on the 22B transformer every chunk). Both depend only on the file:
        # memoized by (path, size, mtime). Exact.
        import functools as _ft
        from ltx_core.loader import checkpoint as CK
        from ltx_pipelines.utils import constants as CN
        import ltx_pipelines.distilled as DI
        rh = CK._read_header

        @_ft.lru_cache(maxsize=None)
        def _rh_key(path, size, mtime):
            return rh(path)

        def read_header(path):
            st = os.stat(path)
            return _rh_key(path, st.st_size, st.st_mtime_ns)
        CK._read_header = read_header
        dmv = CN.detect_model_version

        @_ft.lru_cache(maxsize=None)
        def _dmv_key(path, size, mtime):
            return dmv(path)

        def detect_model_version(path):
            st = os.stat(path)
            return _dmv_key(str(path), st.st_size, st.st_mtime_ns)
        CN.detect_model_version = detect_model_version
        DI.detect_model_version = detect_model_version
    if os.environ.get("LTX_VAE_RGB_GRAPH", "0") == "1":
        # the conv VAE's to_rgb (permute c f h w -> f h w c, then in-place add / mul / clamp) ran as eager device ops on
        # the 116 MB decoded video, and the host copy then read a permuted (strided) tensor. One compiled graph instead,
        # contiguous [f, h, w, c] output (rules: make irregular access regular, fewer DMAs). Same elementwise ops.
        import functools
        from ltx_core.model.video_vae import conv_video_decoder as CVD

        def _rgb(frames):
            return (frames[0].permute(1, 2, 3, 0) + 1.0).mul(0.5).clamp(0.0, 1.0).contiguous()
        rgb = torch.compile(_rgb, backend="neuron", dynamic=False, fullgraph=True)
        dv = CVD.ConvVideoDecoder.decode_video

        @functools.wraps(dv)
        def decode_video(self, latent, tiling_config=None, generator=None, **kw):
            if tiling_config is not None or rt["dit_device"].type != "neuron":
                yield from dv(self, latent, tiling_config, generator, **kw)
                return
            yield rgb(self(latent, generator=generator))
        CVD.ConvVideoDecoder.decode_video = decode_video
    gs = int(os.environ.get("LTX_VAE_SPLIT_RANKS", rt["world"]))
    if gs < rt["world"]:
        # LTX_VAE_SPLIT_RANKS=G (G < world, e.g. TP32 with the 16-rank 2x8 VAE grid, the only grid that fits SBUF at
        # 49 frames): world/G rank groups each decode the full latent (same result in every group)
        dist = rt["dist"]
        rt["_VGROUPS"] = [dist.new_group(list(range(k, k + gs))) for k in range(0, rt["world"], gs)]
    if os.environ.get("LTX_STATE_CACHE", "0") == "1":
        from ltx_core import tools as TL
        for cls in (TL.VideoLatentTools, TL.AudioLatentTools):
            f = cls.create_initial_state
            cache = {}

            def w(self, device, dtype, initial_latent=None, _f=f, _c=cache):
                if initial_latent is not None:
                    return _f(self, device, dtype, initial_latent)
                key = (type(self).__name__, type(self.patchifier).__name__, repr(getattr(self.patchifier, "patch_size", None)),
                       repr(self.target_shape), repr(getattr(self, "fps", None)), repr(getattr(self, "scale_factors", None)),
                       repr(getattr(self, "causal_fix", None)), str(device), dtype)
                if key not in _c:
                    _c[key] = _f(self, device, dtype, None)
                return _clone(_c[key])
            cls.create_initial_state = w
    if os.environ.get("LTX_TE_BCAST_CACHE", "0") == "1" and os.environ.get("LTX_TE_RANK0", "0") == "1" \
            and rt["world"] > 1:
        U, dist, rank = rt["U"], rt["dist"], rt["rank"]
        te = U.PromptEncoder.__call__          # extra.py's rank-0 encode + broadcast wrapper
        last = {}

        def te2(self, *a, **k):
            key = repr((a, sorted(k.items())))
            box = [key if rank == 0 else None]
            dist.broadcast_object_list(box, src=0)
            if last.get("key") == box[0]:
                return last["val"]
            val = te(self, *a, **k)
            last.update(key=box[0], val=val)
            return val
        U.PromptEncoder.__call__ = te2
