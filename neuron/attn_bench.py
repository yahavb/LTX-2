"""Microbench at LTX-2.5 TP16 shapes: NKI flash self-attention vs SDPA, both under torch.compile(fullgraph=True).
Accuracy: rel-RMS of each against an fp32 CPU reference (gate: kernel no worse than the bf16 SDPA baseline).
Run: python3 neuron/attn_bench.py   (one Neuron core)"""
import os, sys, time, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from kernels.ltx_flash_attention import LTXNeuronFlashAttention

dev = torch.device("neuron")
fa = LTXNeuronFlashAttention()


def sdpa(q, k, v, heads):
    return LTXNeuronFlashAttention._sdpa(q, k, v, heads, None)


def rel(a, r):
    return float((a.float().cpu() - r).pow(2).mean().sqrt() / r.pow(2).mean().sqrt())


f_k = torch.compile(lambda q, k, v: fa(q, k, v, H), backend="neuron", fullgraph=True, dynamic=False)
f_s = torch.compile(lambda q, k, v: sdpa(q, k, v, H), backend="neuron", fullgraph=True, dynamic=False)
for S, H, D in [(384, 2, 128), (1536, 2, 128), (int(os.environ.get("AUDIO_S", "128")), 2, 64)]:
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, S, H * D) for _ in range(3))
    ref = sdpa(q, k, v, H)
    qd, kd, vd = (t.to(torch.bfloat16).to(dev) for t in (q, k, v))
    row = [f"S={S} H={H} D={D}"]
    for name, f in (("sdpa", f_s), ("nki", f_k)):
        try:
            o = f(qd, kd, vd); torch.neuron.synchronize()
            for _ in range(5): o = f(qd, kd, vd)
            torch.neuron.synchronize(); t0 = time.perf_counter()
            for _ in range(50): o = f(qd, kd, vd)
            torch.neuron.synchronize(); ms = (time.perf_counter() - t0) / 50 * 1e3
            row.append(f"{name} {ms:.3f} ms rel {rel(o, ref):.2e}")
        except Exception as e:
            row.append(f"{name} FAILED {type(e).__name__}: {str(e)[:120]}")
    print(" | ".join(row), flush=True)
