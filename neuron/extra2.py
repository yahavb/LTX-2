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
