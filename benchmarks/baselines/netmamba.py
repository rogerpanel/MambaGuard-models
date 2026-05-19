"""NetMamba baseline.

Reference:
    Wang, T. et al. "NetMamba: Efficient Network Traffic Classification via
    Pre-training Unidirectional Mamba." 2024.

Four stacked Mamba blocks operating directly on per-message embeddings; no
graph component. If ``mamba_ssm`` is unavailable, falls back to a GRU stack
of comparable depth so the baseline is still runnable.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba  # type: ignore

    _HAS_MAMBA = True
except Exception:  # pragma: no cover
    _HAS_MAMBA = False


class NetMambaBaseline(nn.Module):
    """4-block Mamba encoder with mean-pool classification head."""

    def __init__(
        self,
        in_channels: int = 64,
        d_model: int = 128,
        n_layers: int = 4,
        num_classes: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(in_channels, d_model)
        if _HAS_MAMBA:
            self.blocks = nn.ModuleList(
                [Mamba(d_model=d_model) for _ in range(n_layers)]
            )
        else:
            self.blocks = nn.ModuleList(
                [
                    nn.GRU(d_model, d_model, batch_first=True)
                    for _ in range(n_layers)
                ]
            )
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, batch: Any) -> torch.Tensor:
        x = batch["x"] if isinstance(batch, dict) else batch.x
        if x.dim() == 2:
            x = x.unsqueeze(0)
        h = self.input_proj(x)
        for blk, nrm in zip(self.blocks, self.norms):
            if _HAS_MAMBA:
                h = h + blk(nrm(h))
            else:
                h2, _ = blk(nrm(h))
                h = h + h2
        h = self.drop(h)
        return self.classifier(h.squeeze(0) if h.shape[0] == 1 else h.mean(dim=1))
