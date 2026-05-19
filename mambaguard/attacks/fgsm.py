"""Fast Gradient Sign Method (Goodfellow et al. 2015) over payload embeddings."""
from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


_EMBEDDING_KEY = "p"
_LABEL_KEY = "labels"


def _extract_logits(out: Any) -> torch.Tensor:
    if isinstance(out, dict):
        return out["logits"]
    return out


class FGSM:
    """Single-step gradient sign attack on the payload embedding ``p``."""

    def __init__(self, epsilon: float = 0.03, norm: str = "inf", clip_min: float | None = None, clip_max: float | None = None) -> None:
        if norm not in {"inf", "l2"}:
            raise ValueError(f"unsupported norm: {norm}")
        self.epsilon = float(epsilon)
        self.norm = norm
        self.clip_min = clip_min
        self.clip_max = clip_max

    def forward(
        self,
        model: nn.Module,
        batch: Mapping[str, Any],
        y: torch.Tensor | None = None,
        embedding_key: str = _EMBEDDING_KEY,
    ) -> dict[str, torch.Tensor]:
        was_training = model.training
        model.eval()
        x = batch[embedding_key].detach().clone().requires_grad_(True)
        adv_batch = dict(batch)
        adv_batch[embedding_key] = x
        if y is None:
            y = batch[_LABEL_KEY]
        out = model(adv_batch)
        logits = _extract_logits(out)
        loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, x, retain_graph=False, create_graph=False)[0]
        if self.norm == "inf":
            delta = self.epsilon * grad.sign()
        else:
            flat = grad.reshape(grad.shape[0], -1)
            n = flat.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-12)
            delta = self.epsilon * (flat / n).reshape_as(grad)
        x_adv = (x.detach() + delta).detach()
        if self.clip_min is not None or self.clip_max is not None:
            x_adv = x_adv.clamp(min=self.clip_min, max=self.clip_max)
        if was_training:
            model.train()
        out_batch = dict(batch)
        out_batch[embedding_key] = x_adv
        return out_batch

    def __call__(self, model: nn.Module, batch: Mapping[str, Any], y: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        return self.forward(model, batch, y)
