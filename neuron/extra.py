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
