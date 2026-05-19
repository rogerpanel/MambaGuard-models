"""Temporal GATv2 layer with Bochner edge-time encoding and optional LipschitzNorm."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bochner_encoding import BochnerTimeEncoding
from .lipschitz_norm import LipschitzNorm

try:  # pragma: no cover - optional fast path
    from torch_geometric.nn import MessagePassing  # type: ignore
    from torch_geometric.utils import softmax as pyg_softmax  # type: ignore

    _HAS_PYG = True
except Exception:  # noqa: BLE001
    _HAS_PYG = False
    MessagePassing = object  # type: ignore[assignment]


class TemporalGATv2Layer(nn.Module):
    """GATv2 attention (Brody 2022) over a temporal multi-graph of agents.

    Implements
        e_{a a'} = q^T LeakyReLU(W_r [z^a ‖ z^{a'} ‖ e^feat ‖ Φ(Δt)])
    with H heads, averaged at the final layer, and an optional LipschitzNorm
    rescaling that bounds the layer's input-Lipschitz constant.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        edge_dim: int = 0,
        heads: int = 4,
        d_T: int = 64,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        lipschitz_norm: bool = True,
        concat: bool = False,
    ) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.heads = heads
        self.edge_dim = edge_dim
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.time_enc = BochnerTimeEncoding(d_T=d_T)
        in_score = 2 * d_in + edge_dim + d_T
        # GATv2: shared W_r feeds a single LeakyReLU before the per-head q.
        self.W_r = nn.Linear(in_score, heads * d_out, bias=True)
        self.q = nn.Parameter(torch.empty(heads, d_out))
        nn.init.xavier_uniform_(self.q)

        self.W_v = nn.Linear(d_in, heads * d_out, bias=False)
        self.lip_norm = LipschitzNorm() if lipschitz_norm else None
        self.eta = 1.0  # multiplier used in the analytic Lipschitz bound

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,                # (N_nodes, d_in)
        edge_index: torch.Tensor,       # (2, E)
        edge_attr: Optional[torch.Tensor] = None,   # (E, edge_dim)
        edge_time: Optional[torch.Tensor] = None,   # (E,)
    ) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        E = src.size(0)
        H, Do = self.heads, self.d_out

        z_src = x[src]
        z_dst = x[dst]

        feats = [z_src, z_dst]
        if edge_attr is not None and self.edge_dim > 0:
            feats.append(edge_attr)
        elif self.edge_dim > 0:
            feats.append(x.new_zeros(E, self.edge_dim))
        if edge_time is not None:
            feats.append(self.time_enc(edge_time))
        else:
            feats.append(x.new_zeros(E, self.time_enc.d_T))
        h = torch.cat(feats, dim=-1)                 # (E, in_score)
        h = F.leaky_relu(self.W_r(h), self.negative_slope)
        h = h.view(E, H, Do)
        logits = (h * self.q.unsqueeze(0)).sum(-1)   # (E, H)

        if self.lip_norm is not None:
            qn = self.q.norm(dim=-1).unsqueeze(0).expand(E, H)
            logits = self.lip_norm(logits, qn)

        # Softmax per destination node, per head.
        if _HAS_PYG:
            alpha = pyg_softmax(logits, dst, num_nodes=x.size(0))
        else:
            alpha = _segment_softmax(logits, dst, num_nodes=x.size(0))
        if self.dropout > 0 and self.training:
            alpha = F.dropout(alpha, p=self.dropout)

        v = self.W_v(z_src).view(E, H, Do)            # (E, H, Do)
        msg = v * alpha.unsqueeze(-1)
        out = x.new_zeros(x.size(0), H, Do)
        out.index_add_(0, dst, msg)

        if self.concat:
            out = out.reshape(x.size(0), H * Do)
        else:
            out = out.mean(dim=1)                     # average heads (final layer)
        return F.gelu(out)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def lipschitz_bound(self) -> float:
        """Bound ``‖W_v‖₂ · (1 + η·‖q‖₂·‖W_r‖₂)`` for the attention head."""
        w_v = float(torch.linalg.matrix_norm(self.W_v.weight, ord=2).item())
        w_r = float(torch.linalg.matrix_norm(self.W_r.weight, ord=2).item())
        q_n = float(self.q.norm(p=2).item())
        return w_v * (1.0 + self.eta * q_n * w_r)


def _segment_softmax(
    logits: torch.Tensor, index: torch.Tensor, num_nodes: int
) -> torch.Tensor:
    """Numerically-stable softmax over ``logits`` grouped by ``index``."""
    # Subtract per-group max for stability.
    max_per = logits.new_full((num_nodes, logits.size(-1)), float("-inf"))
    max_per = max_per.scatter_reduce(0, index.unsqueeze(-1).expand_as(logits), logits, reduce="amax", include_self=True)
    centred = logits - max_per[index]
    exp = centred.exp()
    denom = exp.new_zeros(num_nodes, logits.size(-1))
    denom.index_add_(0, index, exp)
    return exp / (denom[index] + 1e-16)
