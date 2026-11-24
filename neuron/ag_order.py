"""In-graph funcol all-gather order on Neuron for subgroups of G ranks: each rank contributes its rank id; prints the
gathered order on a few ranks. torchrun --nproc_per_node=16 neuron/ag_order.py  env G (8)"""
import os
import torch, torch.distributed as dist
import torch.distributed._functional_collectives as funcol
dist.init_process_group(backend="neuron")
rank, world = dist.get_rank(), dist.get_world_size()
dev = torch.device(f"neuron:{int(os.environ['LOCAL_RANK'])}")
G = int(os.environ.get("G", "8"))
groups = [dist.new_group(list(range(j, j + G))) for j in range(0, world, G)]
grp = groups[rank // G]
x = torch.full((1, 2, 3), float(rank)).to(dev)
f0 = torch.compile(lambda t: funcol.all_gather_tensor(t, 0, grp), backend="neuron", fullgraph=True, dynamic=False)
f1 = torch.compile(lambda t: funcol.all_gather_tensor(t.reshape(2, 3), 1, grp), backend="neuron", fullgraph=True,
                   dynamic=False)
a0 = f0(x).cpu()[:, 0, 0].tolist()
a1 = f1(x).cpu()[0, ::3].tolist()
for r in (0, 3, 8, 13):
    if rank == r:
        print(f"[AG] rank {rank} dim0 {a0} dim-last {a1}", flush=True)
    dist.barrier()
