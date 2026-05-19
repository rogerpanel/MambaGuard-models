"""RTIDS baseline.

Reference:
    Wu, Z. et al. "RTIDS: A Robust Transformer-Based Approach for Intrusion
    Detection System." IEEE Access, 2022.

A compact transformer encoder operating on per-flow / per-message token
sequences, terminated with a CLS-style mean-pool classifier head.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class RTIDSBaseline(nn.Module):
    """Transformer encoder + mean-pool classifier."""

    def __init__(
        self,
        in_channels: int = 64,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        num_classes: int = 4,
        dropout: float = 0.1,
        ffn_mult: int = 4,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(in_channels, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_mult * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, batch: Any) -> torch.Tensor:
        x = batch["x"] if isinstance(batch, dict) else batch.x
        # Treat node features as a single sequence per batch element.
        if x.dim() == 2:
            x = x.unsqueeze(0)
        h = self.input_proj(x)
        h = self.encoder(h)
        h = self.norm(h)
        return self.classifier(h.squeeze(0) if h.shape[0] == 1 else h.mean(dim=1))
