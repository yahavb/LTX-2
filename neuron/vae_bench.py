"""Microbench of the split conv-VAE decode at a given latent shape (no DiT): does it compile, how long per decode.
torchrun --nproc_per_node=16 neuron/vae_bench.py   env: FRAMES (latent frames, 7 = 49 video frames), GRID_H (4)
The split code is run_tp.py's own (_halo / _cc3d_forward / _split_decode), exec'd from the file, so the bench tests
exactly what the pipeline runs."""
import functools, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, torch.distributed as dist
import torch.distributed._functional_collectives as funcol
dist.init_process_group(backend="neuron")
rank, world = dist.get_rank(), dist.get_world_size()
dev = torch.device(f"neuron:{int(os.environ['LOCAL_RANK'])}")
from ltx_core.model.video_vae import convolution as _CV
from ltx_pipelines.utils.blocks.video_decoder import VideoDecoder
from ltx_core.model.video_vae.memory_efficient_decode import disable_memory_efficient_decode
src = open(os.path.join(os.path.dirname(__file__), "run_tp.py")).read()
ns = {"torch": torch, "funcol": funcol, "functools": functools, "os": os}
exec(src[src.index("_VSPLIT = {}"):src.index("if DIT_DEV == \"neuron\":\n    _CV.CausalConv3d.forward")], ns)
_CV.CausalConv3d.forward = ns["_cc3d_forward"]
gh = int(os.environ.get("GRID_H", "4")); gw = world // gh
ns["_VSPLIT"].update(group=dist.group.WORLD, gh=gh, gw=gw, i=rank // gw, j=rank % gw)
vd = VideoDecoder("/node-cache/ltx-2.5/vae/ltx-2.5-video-vae-conv-bf16.safetensors", torch.bfloat16, dev)
dec = disable_memory_efficient_decode(vd._prepared_builder().build(device=dev, dtype=torch.bfloat16).eval())
dec._orig_forward = dec.forward
f = torch.compile(functools.partial(ns["_split_decode"], dec), backend="neuron", dynamic=False, fullgraph=True)
F = int(os.environ.get("FRAMES", "7"))
torch.manual_seed(0)
lat = torch.randn(1, 128, F, int(os.environ.get("LAT_H", "16")), int(os.environ.get("LAT_W", "24")), dtype=torch.bfloat16).to(dev)
with torch.no_grad():
    t0 = time.time(); y = f(lat); y.cpu(); tc = time.time() - t0
    for _ in range(2):
        y = f(lat)
    y.cpu(); t0 = time.perf_counter()
    for _ in range(5):
        y = f(lat)
    y.cpu(); ms = (time.perf_counter() - t0) / 5 * 1e3
if rank == 0:
    print(f"[VAE] frames {F} grid {gh}x{gw} out {tuple(y.shape)} first {tc:.0f}s warm {ms:.1f} ms "
          f"mean {y.float().mean().item():.4f}", flush=True)
