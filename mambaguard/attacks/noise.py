"""Noise-based perturbation and label-flipping baselines."""
from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn


_EMBEDDING_KEY = "p"
_LABEL_KEY = "labels"


class GaussianNoise:
    """Additive isotropic Gaussian noise on the payload embedding."""

    def __init__(self, sigma: float = 0.05) -> None:
        self.sigma = float(sigma)

    def forward(
        self,
        model: nn.Module,
        batch: Mapping[str, Any],
        y: torch.Tensor | None = None,
        embedding_key: str = _EMBEDDING_KEY,
    ) -> dict[str, torch.Tensor]:
        x = batch[embedding_key].detach()
        noise = torch.randn_like(x) * self.sigma
        out_batch = dict(batch)
        out_batch[embedding_key] = (x + noise).detach()
        return out_batch

    def __call__(self, model: nn.Module, batch: Mapping[str, Any], y: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        return self.forward(model, batch, y)


class LabelMasking:
    """Randomly flip a fraction p of labels uniformly across the class set."""

    def __init__(self, p: float = 0.1, num_classes: int = 34, ignore_index: int = -100) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0,1]; got {p}")
        self.p = float(p)
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)

    def forward(
        self,
        model: nn.Module,
        batch: Mapping[str, Any],
        y: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        labels = batch[_LABEL_KEY] if y is None else y
        labels = labels.detach().clone()
        mask = torch.rand_like(labels, dtype=torch.float32) < self.p
        if mask.any():
            random_labels = torch.randint(0, self.num_classes, labels.shape, device=labels.device)
            labels = torch.where(mask, random_labels, labels)
        out_batch = dict(batch)
        out_batch[_LABEL_KEY] = labels
        return out_batch

    def __call__(self, model: nn.Module, batch: Mapping[str, Any], y: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        return self.forward(model, batch, y)
