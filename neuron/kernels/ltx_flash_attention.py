"""LTX-2.3 flash self-attention on Neuron — adapter around the ti2v-5b NST kernel.

WHY: the profiler showed LTX t2v/v2v is MEMORY-BOUND — the dominant NEFF is
98.9% static DMA, 2.6M tiny (~1.5KB) transfers, MFU ~0.0004%: the attention
scores/layout are being shuffled through HBM. Flash attention keeps the QK^T
scores + running softmax in SBUF and never materializes the [seq,seq] matrix to
HBM, which is the single biggest HBM-consumption / MFU lever for this model.

This wraps `kernels.self_attention_nst.wan_flash_self_attn` (the parameterized,
adaptive-flash NST kernel from wan2-ti2v-5b) and exposes it as an LTX
`AttentionCallable` so it drops into LTX's pluggable `AttentionOps.attention_function`
— no fork of LTX internals.

LTX attention contract (see ltx_core PytorchAttention.__call__):
    call(q, k, v, heads) where q,k,v are [B, S, heads*dim_head]; returns [B, S, heads*dim_head].
Kernel contract (see self_attention_nst.wan_flash_self_attn):
    q(H, d, Sq), k(H, d, Sk), v(H, Sk, d); returns [Sq, H, d]. Treats batch=H
    (one flash attention per head). Under TP, `heads` here is the LOCAL head count
    (e.g. 4/rank), so each rank flash-attends its own heads — correct.

Only the SELF-attention path (q,k,v same S, no cross mask) is routed to the
kernel. Anything else (masked, cross-attn, B>1) falls back to eager SDPA so we
never silently mis-handle a shape.
"""
import math
import os

import torch

_KERNEL = None
_KERNEL_ERR = None


def _load_kernel():
    """Lazy-wrap the NKI kernel for the Neuron device (torch_neuronx.nki_hop)."""
    global _KERNEL, _KERNEL_ERR
    if _KERNEL is not None or _KERNEL_ERR is not None:
        return
    try:
        from torch_neuronx.nki_hop import wrap_nki
        from kernels.self_attention_nst import wan_flash_self_attn
        _KERNEL = wrap_nki(wan_flash_self_attn)
    except Exception as e:  # surface, never silently swallow
        _KERNEL_ERR = e


# seq_k is padded to a multiple of this (kernel tiles K in 512 chunks internally).
ATTN_SEQLEN_MULTIPLE = int(os.environ.get("ATTN_SEQLEN_MULTIPLE", "512"))
_P = 128


class LTXNeuronFlashAttention:
    """LTX AttentionCallable backed by the NST flash-attention NKI kernel.

    Signature matches LTX AttentionCallable: (q, k, v, heads, mask=None) -> out.
    Falls back to eager SDPA for the masked / cross-attn / unsupported paths.
    """

    label = "NeuronFlashNST"

    def __call__(self, q, k, v, heads, mask=None):
        _load_kernel()
        b, sq, inner = q.shape
        sk = k.shape[1]
        d = inner // heads

        # Route only the clean self-attention path to the kernel. Fall back
        # (loudly-safe) otherwise so we never mis-handle a shape.
        supported = (
            _KERNEL is not None and mask is None and b == 1
            and d <= 128 and q.shape == k.shape == v.shape
        )
        if not supported:
            return self._sdpa(q, k, v, heads, mask)

        scale = 1.0 / math.sqrt(d)
        # [B=1, S, H*d] -> per-head kernel layout
        qh = q.view(1, sq, heads, d)[0]   # [sq, H, d]
        kh = k.view(1, sk, heads, d)[0]
        vh = v.view(1, sk, heads, d)[0]
        q_kern = qh.permute(1, 2, 0).contiguous()   # [H, d, sq]
        k_kern = kh.permute(1, 2, 0).contiguous()   # [H, d, sk]
        v_kern = vh.permute(1, 0, 2).contiguous()   # [H, sk, d]

        pad_q = (_P - sq % _P) % _P
        pad_k = (ATTN_SEQLEN_MULTIPLE - sk % ATTN_SEQLEN_MULTIPLE) % ATTN_SEQLEN_MULTIPLE
        if pad_q:
            q_kern = torch.nn.functional.pad(q_kern, (0, pad_q))
        if pad_k:
            k_kern = torch.nn.functional.pad(k_kern, (0, pad_k))
            v_kern = torch.nn.functional.pad(v_kern, (0, 0, 0, pad_k))

        out = _KERNEL(q_kern, k_kern, v_kern,
                      softmax_scale=scale, actual_seqlen_k=sk, use_dynamic_loop=True)
        # out: [sq_padded, H, d] -> [1, sq, H*d]
        return out[:sq].unsqueeze(0).reshape(1, sq, heads * d)

    @staticmethod
    def _sdpa(q, k, v, heads, mask):
        b, _, inner = q.shape
        d = inner // heads
        qq, kk, vv = (t.view(b, -1, heads, d).transpose(1, 2) for t in (q, k, v))
        if mask is not None:
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
        out = torch.nn.functional.scaled_dot_product_attention(
            qq, kk, vv, attn_mask=mask, dropout_p=0.0, is_causal=False)
        return out.transpose(1, 2).reshape(b, -1, heads * d)
