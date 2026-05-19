"""SecurityBERT baseline.

Reference:
    Ferrag, M. A. et al. "SecurityBERT: A Lightweight BERT-Based Model for
    Cyber Threat Detection." IEEE Access, 2024.

A BERT-base wrapper that consumes tokenised log / payload text. If
``transformers`` is unavailable, falls back to a randomly initialised
TransformerEncoder of similar depth.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

try:
    from transformers import AutoModel  # type: ignore

    _HAS_HF = True
except Exception:  # pragma: no cover
    _HAS_HF = False


class SecurityBERTBaseline(nn.Module):
    """BERT-base encoder + linear classifier."""

    def __init__(
        self,
        pretrained_name: str = "bert-base-uncased",
        in_channels: int = 64,
        num_classes: int = 4,
        dropout: float = 0.1,
        d_model: int = 768,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        if _HAS_HF:
            try:
                self.bert = AutoModel.from_pretrained(pretrained_name)
                self.d_model = int(self.bert.config.hidden_size)
            except Exception:
                self.bert = None
        else:
            self.bert = None
        # Always provide a fallback embedding path consuming our numeric x.
        self.input_proj = nn.Linear(in_channels, self.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=8,
            dim_feedforward=4 * self.d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.fallback_encoder = nn.TransformerEncoder(layer, num_layers=4)
        self.classifier = nn.Linear(self.d_model, num_classes)

    def forward(self, batch: Any) -> torch.Tensor:
        if self.bert is not None and isinstance(batch, dict) and "input_ids" in batch:
            out = self.bert(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
            )
            pooled = out.last_hidden_state[:, 0]
            return self.classifier(pooled)
        x = batch["x"] if isinstance(batch, dict) else batch.x
        if x.dim() == 2:
            x = x.unsqueeze(0)
        h = self.input_proj(x)
        h = self.fallback_encoder(h)
        out = h.squeeze(0) if h.shape[0] == 1 else h.mean(dim=1)
        return self.classifier(out)
