"""E-GraphSAGE baseline.

Reference:
    Lo, W. et al. "E-GraphSAGE: A Graph Neural Network based Intrusion
    Detection System for IoT." NOMS 2022.

Edge-aware GraphSAGE for flow-level intrusion detection.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class EGraphSAGEBaseline(nn.Module):
    """Two-layer SAGE with concatenated edge features as messages."""

    def __init__(
        self,
        in_channels: int = 64,
        edge_dim: int = 8,
        hidden_channels: int = 128,
        num_classes: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lin_msg1 = nn.Linear(in_channels + edge_dim, hidden_channels)
        self.lin_upd1 = nn.Linear(in_channels + hidden_channels, hidden_channels)
        self.lin_msg2 = nn.Linear(hidden_channels + edge_dim, hidden_channels)
        self.lin_upd2 = nn.Linear(hidden_channels + hidden_channels, hidden_channels)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_channels, num_classes)

    @staticmethod
    def _aggregate(
        h: torch.Tensor, edge_index: torch.Tensor, msgs: torch.Tensor
    ) -> torch.Tensor:
        n = h.shape[0]
        dst = edge_index[1]
        out = h.new_zeros(n, msgs.shape[-1])
        out.index_add_(0, dst, msgs)
        deg = h.new_zeros(n).index_add_(0, dst, torch.ones_like(dst, dtype=h.dtype))
        return out / deg.clamp_min(1.0).unsqueeze(-1)

    def _layer(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        lin_msg: nn.Linear,
        lin_upd: nn.Linear,
    ) -> torch.Tensor:
        src = edge_index[0]
        msg_in = torch.cat([h[src], edge_attr], dim=-1)
        msgs = self.act(lin_msg(msg_in))
        agg = self._aggregate(h, edge_index, msgs)
        return self.act(lin_upd(torch.cat([h, agg], dim=-1)))

    def forward(self, batch: Any) -> torch.Tensor:
        x = batch["x"] if isinstance(batch, dict) else batch.x
        ei = batch["edge_index"] if isinstance(batch, dict) else batch.edge_index
        ea = batch["edge_attr"] if isinstance(batch, dict) else batch.edge_attr
        h = self._layer(x, ei, ea, self.lin_msg1, self.lin_upd1)
        h = self.drop(h)
        h = self._layer(h, ei, ea, self.lin_msg2, self.lin_upd2)
        return self.classifier(h)
