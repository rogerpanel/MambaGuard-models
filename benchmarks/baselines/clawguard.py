"""ClawGuard baseline (proxy implementation).

Reference (cited in the paper):
    ClawGuard: prompt-injection / tool-misuse runtime guard for LLM agents.
    No public weights or implementation are available at time of writing.

TODO: replace this proxy with the official implementation once released.
The proxy mirrors AgentArmor's contract (BiLSTM tagger over message
embeddings) but uses a 1-layer GRU + linear head with a slightly different
inductive bias so the two baselines do not collapse to the same model.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class ClawGuardBaseline(nn.Module):
    """Proxy: 1-layer GRU + GELU MLP head."""

    def __init__(
        self,
        in_channels: int = 64,
        hidden_channels: int = 128,
        num_classes: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.gru = nn.GRU(
            input_size=hidden_channels,
            hidden_size=hidden_channels,
            batch_first=True,
            bidirectional=False,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_channels),
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_classes),
        )

    def forward(self, batch: Any) -> torch.Tensor:
        x = batch["x"] if isinstance(batch, dict) else batch.x
        if x.dim() == 2:
            x = x.unsqueeze(0)
        h = self.input_proj(x)
        h, _ = self.gru(h)
        out = h.squeeze(0) if h.shape[0] == 1 else h
        return self.head(out)
