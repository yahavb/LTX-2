"""Tensor parallelism for the LTX-2.5 AV transformer on Neuron (torch.distributed, one rank per logical core).

shard_transformer(model, rank, world, group) rewrites the model in place after it is loaded on CPU:
- every Attention: to_q / to_k / to_v / to_gate_logits keep this rank's heads (column parallel), q_norm / k_norm
  become RMS norms over the full inner dimension computed across ranks (sum of squares all-reduced), to_out keeps
  this rank's input columns (row parallel) and its output is all-reduced;
- every FeedForward: the GELU projection keeps this rank's columns, the output projection its rows, all-reduced.
Everything else (AdaLN tables, embeddings, the caption projection, the connector) stays replicated. The math is the
same as the unsharded model; only the order of the reductions changes (bf16 rounding).
"""
import functools
import os

import torch
import torch.distributed as dist

from ltx_core.model.transformer.attention import Attention
from ltx_core.model.transformer.feed_forward import FeedForward
from ltx_core.model.transformer.rope import apply_rotary_emb


import torch.distributed._functional_collectives as funcol


def all_reduce(x, group):
    """Sum over the TP group; the functional collective traces into a torch.compile graph (no graph break)."""
    if group is None:
        return x
    return funcol.all_reduce(x, "sum", group)


class TPRMSNorm(torch.nn.Module):
    """RMSNorm over the full (unsharded) last dimension, computed from this rank's slice."""

    def __init__(self, full: torch.nn.RMSNorm, lo: int, hi: int, full_dim: int, group):
        super().__init__()
        self.eps = full.eps if full.eps is not None else torch.finfo(torch.float32).eps
        self.weight = torch.nn.Parameter(full.weight.data[lo:hi].clone()) if full.weight is not None else None
        self.full_dim = full_dim
        self.group = group

    def forward(self, x):
        xf = x.float()
        ss = (xf * xf).sum(-1, keepdim=True)
        ss = all_reduce(ss, self.group)
        y = xf * torch.rsqrt(ss / self.full_dim + self.eps)
        if self.weight is not None:
            y = y * self.weight.float()
        return y.to(x.dtype)


class RowOut(torch.nn.Module):
    """y = all_reduce(x_local @ W_local^T) + b (the bias added once, after the reduction)."""

    def __init__(self, lin: torch.nn.Linear, lo: int, hi: int, group):
        super().__init__()
        self.weight = torch.nn.Parameter(lin.weight.data[:, lo:hi].clone())
        self.bias = torch.nn.Parameter(lin.bias.data.clone()) if lin.bias is not None else None
        self.group = group

    def forward(self, x):
        y = torch.nn.functional.linear(x, self.weight)
        y = all_reduce(y, self.group)
        return y + self.bias if self.bias is not None else y


def _col(lin: torch.nn.Linear, lo: int, hi: int) -> torch.nn.Linear:
    out = torch.nn.Linear(lin.in_features, hi - lo, bias=lin.bias is not None, dtype=lin.weight.dtype)
    out.weight.data.copy_(lin.weight.data[lo:hi])
    if lin.bias is not None:
        out.bias.data.copy_(lin.bias.data[lo:hi])
    return out


def _shard_attention(a: Attention, rank: int, world: int, group) -> None:
    h, d = a.heads, a.dim_head
    assert h % world == 0, f"{h} heads over {world} ranks"
    hl = h // world
    lo, hi = rank * hl * d, (rank + 1) * hl * d
    a.to_q, a.to_k, a.to_v = _col(a.to_q, lo, hi), _col(a.to_k, lo, hi), _col(a.to_v, lo, hi)
    a.q_norm = TPRMSNorm(a.q_norm, lo, hi, h * d, group)
    a.k_norm = TPRMSNorm(a.k_norm, lo, hi, h * d, group)
    if a.to_gate_logits is not None:
        a.to_gate_logits = _col(a.to_gate_logits, rank * hl, (rank + 1) * hl)
    a.to_out[0] = RowOut(a.to_out[0], lo, hi, group)
    a.heads = hl
    # split RoPE tables are per head ([B, H, T, D/2]): keep this rank's heads
    hlo, hhi = rank * hl, (rank + 1) * hl
    fwd = a.forward

    def _sl(pe):
        if pe is None:
            return None
        return tuple(t[:, hlo:hhi] if (t.ndim == 4 and t.shape[1] == h) else t for t in pe)

    def forward(x, context=None, mask=None, pe=None, k_pe=None, perturbation_mask=None, all_perturbed=False):
        return fwd(x, context=context, mask=mask, pe=_sl(pe), k_pe=_sl(k_pe), perturbation_mask=perturbation_mask,
                   all_perturbed=all_perturbed)
    a.forward = forward
    if os.environ.get("LTX_TP_FUSED_NORM", "0") == "1":
        a.forward = functools.partial(_fused_norm_forward, a, _sl)


def _fused_norm_forward(a, _sl, x, context=None, mask=None, pe=None, k_pe=None, perturbation_mask=None,
                        all_perturbed=False):
    """Attention.forward with the q and k RMS norms reduced in ONE all-reduce (LTX_TP_FUSED_NORM=1).

    The trace (DiT stage-1 pair) shows 576 tiny fp32 all-reduces per step, latency-bound (~10 us each): the q-norm
    and k-norm sums of squares are reduced separately. Here both are concatenated along the token axis and reduced
    once. Same sums, same formula; only the collective's buffer changes.
    """
    pe, k_pe = _sl(pe), _sl(k_pe)
    context = x if context is None else context
    v = a.to_v(context)
    if all_perturbed:
        out = v
    else:
        q, k = a.to_q(x), a.to_k(context)
        qn, kn = a.q_norm, a.k_norm
        qf, kf = q.float(), k.float()
        ss = torch.cat([(qf * qf).sum(-1, keepdim=True), (kf * kf).sum(-1, keepdim=True)], dim=1)
        ss = all_reduce(ss, qn.group)
        t = q.shape[1]
        q = (qf * torch.rsqrt(ss[:, :t] / qn.full_dim + qn.eps))
        k = (kf * torch.rsqrt(ss[:, t:] / kn.full_dim + kn.eps))
        if qn.weight is not None:
            q = q * qn.weight.float()
        if kn.weight is not None:
            k = k * kn.weight.float()
        q, k = q.to(x.dtype), k.to(x.dtype)
        if pe is not None:
            q = apply_rotary_emb(q, pe, a.rope_type)
            k = apply_rotary_emb(k, pe if k_pe is None else k_pe, a.rope_type)
        if mask is None:
            out = a.attention_function(q, k, v, a.heads)
        else:
            out = a.masked_attention_function(q, k, v, a.heads, mask)
    if a.to_gate_logits is not None and not all_perturbed:
        gates = 2.0 * torch.sigmoid(a.to_gate_logits(x))
        b, t, _ = out.shape
        out = (out.view(b, t, a.heads, a.dim_head) * gates.unsqueeze(-1)).view(b, t, a.heads * a.dim_head)
    if not all_perturbed and perturbation_mask is not None:
        out = out * perturbation_mask + v * (1 - perturbation_mask)
    return a.to_out(out)


def _shard_ff(f: FeedForward, rank: int, world: int, group) -> None:
    proj = f.net[0].proj
    n = proj.out_features
    assert n % world == 0
    lo, hi = rank * n // world, (rank + 1) * n // world
    f.net[0].proj = _col(proj, lo, hi)
    f.net[2] = RowOut(f.net[2], lo, hi, group)


def shard_transformer(model: torch.nn.Module, rank: int, world: int, group=None) -> torch.nn.Module:
    """Shard every transformer-block Attention and FeedForward of *model* in place (see the module docstring)."""
    if world == 1:
        return model
    n_att = n_ff = 0
    for name, m in list(model.named_modules()):
        if "transformer_blocks" not in name:
            continue
        if isinstance(m, Attention):
            _shard_attention(m, rank, world, group)
            n_att += 1
        elif isinstance(m, FeedForward):
            _shard_ff(m, rank, world, group)
            n_ff += 1
    if rank == 0:
        print(f"[tp] sharded {n_att} attentions and {n_ff} feed-forwards over {world} ranks", flush=True)
    return model


def shard_gemma(model: torch.nn.Module, rank: int, world: int, group=None) -> torch.nn.Module:
    """Tensor-parallel Gemma 4 text decoder: one query head per rank (16 heads over 16 ranks).

    q_proj keeps this rank's heads and o_proj their input columns (row parallel, all-reduced); a sliding layer's
    k/v keep the key-value head its query head uses (8 kv heads: rank // 2), a global layer's single shared k=v head
    stays whole; the MLP's gate/up keep this rank's columns and down_proj its rows (all-reduced). Per-head norms and
    RoPE are per head and need no change. Same math; only the reduction order changes.
    """
    if world == 1:
        return model
    n = 0
    for name, a in list(model.named_modules()):
        if type(a).__name__ == "Gemma4TextAttention":
            nq = a.q_proj.out_features // a.head_dim
            assert nq % world == 0, f"{nq} heads over {world} ranks"
            hl = nq // world
            lo, hi = rank * hl * a.head_dim, (rank + 1) * hl * a.head_dim
            a.q_proj = _col(a.q_proj, lo, hi)
            if getattr(a, "k_proj", None) is not None:
                nkv = a.k_proj.out_features // a.head_dim
                if nkv > 1:
                    q_per_kv = nq // nkv
                    kv0 = (rank * hl) // q_per_kv
                    kv1 = ((rank + 1) * hl - 1) // q_per_kv + 1
                    klo, khi = kv0 * a.head_dim, kv1 * a.head_dim
                    a.k_proj = _col(a.k_proj, klo, khi)
                    if a.v_proj is not None:
                        a.v_proj = _col(a.v_proj, klo, khi)
                    nkv_l = kv1 - kv0
                else:
                    nkv_l = 1
                a.num_key_value_groups = hl // nkv_l
            else:
                a.num_key_value_groups = hl   # kv-shared layer: uses the (sharded) states of an earlier layer
            a.o_proj = RowOut(a.o_proj, lo, hi, group)
            n += 1
        elif type(a).__name__ == "Gemma4TextMLP":
            m = a.gate_proj.out_features
            assert m % world == 0
            lo, hi = rank * m // world, (rank + 1) * m // world
            a.gate_proj, a.up_proj = _col(a.gate_proj, lo, hi), _col(a.up_proj, lo, hi)
            a.down_proj = RowOut(a.down_proj, lo, hi, group)
    if rank == 0:
        print(f"[tp] gemma: sharded {n} attentions over {world} ranks", flush=True)
    return model
