"""MambaGuard top-level model: SSM temporal backbone + temporal GATv2 + detection head."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .detection_head import DetectionHead
from .gatv2_temporal import TemporalGATv2Layer
from .selective_ssm import SelectiveSSMBlock


class RMSNorm(nn.Module):
    """Root-mean-square layer norm; cheaper than LayerNorm, no centring (Zhang & Sennrich 2019)."""

    def __init__(self, d: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


@dataclass
class MambaGuardConfig:
    """Hyper-parameter container for :class:`MambaGuard`."""

    d_p: int = 384                  # canonicalised message embedding (MiniLM)
    d_mu: int = 32                  # metadata feature width
    d_model: int = 256
    d_state: int = 16
    expand: int = 2
    kernel_size: int = 4
    n_blocks: int = 4
    n_gat_layers: int = 2
    gat_heads: int = 4
    edge_dim: int = 16
    d_T: int = 64
    num_classes: int = 34
    dropout: float = 0.1
    dt_min: float = 1e-3
    dt_max: float = 1e-1


class MambaGuard(nn.Module):
    """Selective-SSM temporal encoder fused with a temporal GATv2 over the agent graph."""

    def __init__(self, cfg: Optional[MambaGuardConfig] = None) -> None:
        super().__init__()
        cfg = cfg or MambaGuardConfig()
        self.cfg = cfg

        self.input_proj = nn.Linear(cfg.d_p + cfg.d_mu, cfg.d_model)
        self.input_norm = RMSNorm(cfg.d_model)

        # Stack of 4 selective SSM blocks with residual + RMSNorm (pre-norm Mamba layout).
        self.blocks = nn.ModuleList(
            [
                SelectiveSSMBlock(
                    d_model=cfg.d_model,
                    d_state=cfg.d_state,
                    expand=cfg.expand,
                    dt_min=cfg.dt_min,
                    dt_max=cfg.dt_max,
                    kernel_size=cfg.kernel_size,
                )
                for _ in range(cfg.n_blocks)
            ]
        )
        self.norms = nn.ModuleList([RMSNorm(cfg.d_model) for _ in range(cfg.n_blocks)])
        self.final_norm = RMSNorm(cfg.d_model)

        # Temporal GATv2 over per-agent pooled features z^a_t.
        self.gat_layers = nn.ModuleList()
        for i in range(cfg.n_gat_layers):
            self.gat_layers.append(
                TemporalGATv2Layer(
                    d_in=cfg.d_model,
                    d_out=cfg.d_model,
                    edge_dim=cfg.edge_dim,
                    heads=cfg.gat_heads,
                    d_T=cfg.d_T,
                    dropout=cfg.dropout,
                    lipschitz_norm=True,
                    concat=False,
                )
            )

        self.head = DetectionHead(
            d_in=cfg.d_model, num_classes=cfg.num_classes, dropout=cfg.dropout
        )

    # ------------------------------------------------------------------
    def _encode_agent(self, p: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """Per-agent SSM stack. ``p``: (B, L, d_p); ``mu``: (B, L, d_mu)."""
        x = torch.cat([p, mu], dim=-1)
        x = self.input_norm(self.input_proj(x))
        for blk, norm in zip(self.blocks, self.norms):
            x = x + blk(norm(x))           # pre-norm residual; causality preserved by blk
        return self.final_norm(x)

    def forward(
        self,
        messages: Mapping[str, torch.Tensor],
        graph: Optional[Mapping[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        ``messages`` keys (required):
            * ``p``  : (A, L, d_p) per-agent canonicalised message embeddings.
            * ``mu`` : (A, L, d_mu) per-agent metadata features.
        ``messages`` keys (optional):
            * ``mask`` : (A, L) boolean, True where positions are valid.

        ``graph`` keys (optional):
            * ``edge_index`` : (2, E)
            * ``edge_attr``  : (E, edge_dim)
            * ``edge_time``  : (E,)

        Returns dict with ``logits`` (A, num_classes) and ``agent_features``.
        """
        p = messages["p"]
        mu = messages["mu"]
        if p.dim() != 3 or mu.dim() != 3:
            raise ValueError("messages['p'] and ['mu'] must be (A, L, *) tensors")

        h = self._encode_agent(p, mu)  # (A, L, d_model)

        # Per-agent temporal pooling: mean over valid positions (or last token).
        mask = messages.get("mask")
        if mask is not None:
            m = mask.to(h.dtype).unsqueeze(-1)
            z = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        else:
            z = h.mean(dim=1)            # (A, d_model)

        if graph is not None and "edge_index" in graph:
            edge_index = graph["edge_index"]
            edge_attr = graph.get("edge_attr")
            edge_time = graph.get("edge_time")
            for layer in self.gat_layers:
                z = z + layer(z, edge_index, edge_attr=edge_attr, edge_time=edge_time)

        logits = self.head(z)
        return {"logits": logits, "agent_features": z, "token_features": h}

    # ------------------------------------------------------------------
    @torch.no_grad()
    def lipschitz_bound(self) -> float:
        """Composed bound ``L_f ≤ M_h · L_gat · max_block (γβ(1+κ_g))/(1−ρ)``."""
        # Worst Mamba block bound (since they are residually stacked the product
        # multiplies, but residual gives (1 + L_blk) per block).
        block_bound = 1.0
        for blk in self.blocks:
            lb = blk.lipschitz_bound()
            if lb == float("inf"):
                return float("inf")
            block_bound *= 1.0 + lb

        gat_bound = 1.0
        for layer in self.gat_layers:
            gat_bound *= 1.0 + layer.lipschitz_bound()

        # Detection head: two linear layers with GELU (1-Lipschitz) and LayerNorm
        # (1-Lipschitz w.r.t. its scaled output) — bound by product of weight norms.
        w1 = float(torch.linalg.matrix_norm(self.head.fc1.weight, ord=2).item())
        w2 = float(torch.linalg.matrix_norm(self.head.fc2.weight, ord=2).item())
        head_bound = w1 * w2

        # Input projection contributes once.
        win = float(torch.linalg.matrix_norm(self.input_proj.weight, ord=2).item())

        return win * block_bound * gat_bound * head_bound

    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, cfg: Mapping[str, Any] | MambaGuardConfig) -> "MambaGuard":
        """Build from a dataclass instance or a plain mapping of hyper-parameters."""
        if isinstance(cfg, MambaGuardConfig):
            return cls(cfg)
        return cls(MambaGuardConfig(**dict(cfg)))
