"""CPU check of LTX_VOC_SPLIT: torchrun --nproc_per_node=8 neuron/voc_split_test.py (gloo). Random-weight vocoder + BWE
at the LTX-2.5 config; the split output must equal the unsplit one (fp32)."""
import os, sys, types
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, torch.distributed as dist
dist.init_process_group("gloo")
rank, world = dist.get_rank(), dist.get_world_size()
from ltx_core.model.audio_vae import vocoder as V
src = open(os.path.join(os.path.dirname(__file__), "run_tp.py")).read()
ns = {"torch": torch, "_VOC": V, "os": os}
exec(src[src.index("def _rep_pad"):src.index("# LTX_VOC_MM=1")], ns)
V.UpSample1d.forward, V.LowPassFilter1d.forward = ns["_up_forward"], ns["_lp_forward"]
torch.manual_seed(0)
cv = dict(upsample_initial_channel=64, resblock="AMP1", upsample_rates=[5, 2, 2, 2, 2, 2],
          resblock_kernel_sizes=[3, 7, 11], upsample_kernel_sizes=[11, 4, 4, 4, 4, 4],
          resblock_dilation_sizes=[[1, 3, 5]] * 3, activation="snakebeta", use_tanh_at_final=False, use_bias_at_final=False)
cb = dict(cv, upsample_initial_channel=32, upsample_rates=[6, 5, 2, 2, 2], upsample_kernel_sizes=[12, 11, 4, 4, 4],
          apply_final_activation=False)
m = V.VocoderWithBWE(V.Vocoder(**cv), V.Vocoder(**cb), V.MelSTFT(512, 80, 512, 64), 16000, 48000, 80).float()
with torch.no_grad():
    for p in m.parameters():
        p.normal_(0, 0.05)
    m.mel_stft.mel_basis.uniform_(0, 0.1); m.mel_stft.stft_fn.forward_basis.normal_(0, 0.05)
mel = torch.randn(1, 2, int(os.environ.get("T", "101")), 64)
with torch.no_grad():
    ref = m(mel)
rt = ns   # the up/down forwards read _rep_pad from this namespace, as run_tp's do from run_tp's globals
rt.update({"dist": dist, "rank": rank, "world": world, "dit_device": torch.device("cpu"),
           "ADm": types.SimpleNamespace(AudioDecoder=type("A", (), {"__call__": lambda s, l: None}))})
import extra
extra._install_voc_split(rt, world)
for mod in m.modules():
    if isinstance(mod, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
        mod._vs = True
with torch.no_grad():
    out = m(mel)
if rank == 0:
    print("shapes", tuple(ref.shape), tuple(out.shape), "max abs diff", float((out - ref).abs().max()),
          "ref max", float(ref.abs().max()), flush=True)
