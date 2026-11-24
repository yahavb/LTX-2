"""Microbench of the vocoder + BWE alone on Neuron (no DiT): stock vs LTX_VOC_SPLIT, compile check + warm time +
max |diff| split vs stock. torchrun --nproc_per_node=16 neuron/voc_bench.py   env: T (mel frames, 101), G (8)"""
import os, sys, time, types
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, torch.distributed as dist
dist.init_process_group(backend="neuron")
rank, world = dist.get_rank(), dist.get_world_size()
dev = torch.device(f"neuron:{int(os.environ['LOCAL_RANK'])}")
from ltx_core.model.audio_vae import vocoder as V
from ltx_pipelines.utils.blocks.audio_decoder import AudioDecoder
src = open(os.path.join(os.path.dirname(__file__), "run_tp.py")).read()
ns = {"torch": torch, "_VOC": V, "os": os}
exec(src[src.index("def _rep_pad"):src.index("# LTX_VOC_MM=1")], ns)
V.UpSample1d.forward, V.LowPassFilter1d.forward = ns["_up_forward"], ns["_lp_forward"]
ad = AudioDecoder("/node-cache/ltx-2.5/vae/ltx-2.5-audio-vae-bf16.safetensors", torch.bfloat16, dev)
voc = ad._vocoder_builder.build(device=dev, dtype=torch.float32).eval()
T, G = int(os.environ.get("T", "101")), int(os.environ.get("G", "8"))
torch.manual_seed(0)
mel = (torch.randn(1, 2, T, 64) * 2 - 4).to(dev)


def bench(fn):
    with torch.no_grad():
        t0 = time.time(); y = fn(mel); y.cpu(); tc = time.time() - t0
        for _ in range(2):
            y = fn(mel)
        y.cpu(); t0 = time.perf_counter()
        for _ in range(5):
            y = fn(mel)
        y = y.cpu(); return y, tc, (time.perf_counter() - t0) / 5 * 1e3


stock = torch.compile(voc.forward, backend="neuron", dynamic=False, fullgraph=True)
y0, c0, m0 = bench(stock)
import extra
rt = ns
rt.update({"dist": dist, "rank": rank, "world": world, "dit_device": dev,
           "ADm": types.SimpleNamespace(AudioDecoder=type("A", (), {"__call__": lambda s, l: None}))})
extra._install_voc_split(rt, G)
for m in voc.modules():
    if isinstance(m, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
        m._vs = True
split = torch.compile(V.VocoderWithBWE.forward.__get__(voc), backend="neuron", dynamic=False, fullgraph=True)
y1, c1, m1 = bench(split)
if rank == 0:
    print(f"[VOC] T {T} G {G} stock {m0:.1f} ms (first {c0:.0f}s) | split {m1:.1f} ms (first {c1:.0f}s) | "
          f"max|diff| {float((y1 - y0).abs().max()):.2e} of {float(y0.abs().max()):.3f}", flush=True)
