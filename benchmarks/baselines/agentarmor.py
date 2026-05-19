"""AgentArmor baseline (proxy implementation).

Reference (cited in the paper):
    AgentArmor: defensive runtime for LLM agent toolchains. No public weights
    or open implementation are available at time of writing.

TODO: replace this proxy with the official implementation once released.
For now we use a bidirectional LSTM tagger that consumes per-message
embeddings and emits a per-message class logit, matching the input/output
contract of MambaGuard so it can be evaluated head-to-head.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class AgentArmorBaseline(nn.Module):
    """Proxy: BiLSTM tagger over per-message embeddings."""

    def __init__(
        self,
        in_channels: int = 64,
        hidden_channels: int = 128,
        n_layers: int = 2,
        num_classes: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.lstm = nn.LSTM(
            input_size=hidden_channels,
            hidden_size=hidden_channels,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(2 * hidden_channels)
        self.classifier = nn.Linear(2 * hidden_channels, num_classes)

    def forward(self, batch: Any) -> torch.Tensor:
        x = batch["x"] if isinstance(batch, dict) else batch.x
        if x.dim() == 2:
            x = x.unsqueeze(0)
        h = self.input_proj(x)
        h, _ = self.lstm(h)
        h = self.norm(h)
        out = h.squeeze(0) if h.shape[0] == 1 else h
        return self.classifier(out)
