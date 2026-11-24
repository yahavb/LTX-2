"""In-graph all-reduce latency at the DiT stage-1 trace sizes: TP16 over 4 chips vs TP4 inside one chip.
torchrun --nproc_per_node=16 neuron/cc_bench.py"""
import os, time
import torch, torch.distributed as dist
import torch.distributed._functional_collectives as funcol
dist.init_process_group(backend="neuron")
rank, world = dist.get_rank(), dist.get_world_size()
dev = torch.device(f"neuron:{int(os.environ['LOCAL_RANK'])}")
torch.neuron.set_device(dev) if hasattr(torch, "neuron") and hasattr(torch.neuron, "set_device") else None
g4 = [dist.new_group(list(range(i, i + 4))) for i in range(0, world, 4)][rank // 4]
N = 32   # all-reduces chained in one graph (each depends on the previous)


def chain(x, group):
    for _ in range(N):
        x = funcol.all_reduce(x, "sum", group) * 0.0625
    return x


for name, shape, dt in [("act 3MB bf16", (1, 384, 4096), torch.bfloat16), ("norm 1536 fp32", (1, 1536, 1), torch.float32),
                        ("norm 104 fp32", (1, 104, 1), torch.float32)]:
    x = torch.randn(shape, dtype=dt).to(dev)
    row = [name]
    for gname, grp in (("tp16", dist.group.WORLD), ("tp4-chip", g4)):
        f = torch.compile(lambda t, _g=grp: chain(t, _g), backend="neuron", fullgraph=True, dynamic=False)
        y = f(x); torch.neuron.synchronize() if hasattr(torch, "neuron") else None
        for _ in range(3): y = f(x)
        y.cpu(); t0 = time.perf_counter()
        for _ in range(10): y = f(x)
        y.cpu(); us = (time.perf_counter() - t0) / 10 / N * 1e6
        row.append(f"{gname} {us:.1f} us/AR")
    if rank == 0:
        print("[CC] " + " | ".join(row), flush=True)
