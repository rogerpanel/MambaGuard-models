"""Anomal-E baseline.

Reference:
    Caville, E. et al. "Anomal-E: A Self-Supervised Network Intrusion
    Detection System based on Graph Neural Networks." Knowledge-Based
    Systems, 2022.

E-GraphSAGE encoder + Deep Graph Infomax (DGI) self-supervised pre-training
target. Here we expose the encoder and a DGI-style projection head; the
classifier consumes the encoder output for supervised eval parity.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .e_graphsage import EGraphSAGEBaseline


class AnomalEBaseline(nn.Module):
    """E-GraphSAGE encoder with a DGI discriminator head."""

    def __init__(
        self,
        in_channels: int = 64,
        edge_dim: int = 8,
        hidden_channels: int = 128,
        num_classes: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.encoder = EGraphSAGEBaseline(
            in_channels=in_channels,
            edge_dim=edge_dim,
            hidden_channels=hidden_channels,
            num_classes=hidden_channels,  # use as embedding
            dropout=dropout,
        )
        self.dgi_proj = nn.Bilinear(hidden_channels, hidden_channels, 1)
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def encode(self, batch: Any) -> torch.Tensor:
        return self.encoder(batch)

    def discriminate(self, h: torch.Tensor, summary: torch.Tensor) -> torch.Tensor:
        s = summary.expand_as(h)
        return self.dgi_proj(h, s).squeeze(-1)

    def forward(self, batch: Any) -> torch.Tensor:
        h = self.encode(batch)
        return self.classifier(h)
