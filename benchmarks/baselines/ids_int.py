"""IDS-INT baseline.

Reference:
    Ullah, F. et al. "IDS-INT: Intrusion Detection System using
    Transformer-based Transfer Learning for Imbalanced Network Traffic."
    Digital Communications and Networks, 2024.

Transformer encoder with class-balanced cross-entropy / focal loss; here we
expose just the architecture (loss is applied by the trainer).
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class IDSINTBaseline(nn.Module):
    """Transformer encoder with depth-wise conv stem for class-imbalanced IDS."""

    def __init__(
        self,
        in_channels: int = 64,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        num_classes: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(in_channels, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # Depth-wise conv1d for local context.
        self.dw_conv = nn.Conv1d(
            d_model, d_model, kernel_size=3, padding=1, groups=d_model
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, batch: Any) -> torch.Tensor:
        x = batch["x"] if isinstance(batch, dict) else batch.x
        if x.dim() == 2:
            x = x.unsqueeze(0)
        h = self.stem(x)
        # depth-wise conv along sequence dim
        h = self.dw_conv(h.transpose(1, 2)).transpose(1, 2)
        h = self.encoder(h)
        h = self.norm(h)
        out = h.squeeze(0) if h.shape[0] == 1 else h.mean(dim=1)
        return self.classifier(out)
