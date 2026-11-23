"""Tensor parallelism for the LTX-2.5 AV transformer on Neuron (torch.distributed, one rank per logical core).

shard_transformer(model, rank, world, group) rewrites the model in place after it is loaded on CPU:
- every Attention: to_q / to_k / to_v / to_gate_logits keep this rank's heads (column parallel), q_norm / k_norm
  become RMS norms over the full inner dimension computed across ranks (sum of squares all-reduced), to_out keeps
  this rank's input columns (row parallel) and its output is all-reduced;
- every FeedForward: the GELU projection keeps this rank's columns, the output projection its rows, all-reduced.
Everything else (AdaLN tables, embeddings, the caption projection, the connector) stays replicated. The math is the
same as the unsharded model; only the order of the reductions changes (bf16 rounding).
"""
import torch
import torch.distributed as dist

from ltx_core.model.transformer.attention import Attention
from ltx_core.model.transformer.feed_forward import FeedForward


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
