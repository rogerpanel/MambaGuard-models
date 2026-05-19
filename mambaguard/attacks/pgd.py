"""Projected Gradient Descent (Madry et al. 2018) on payload embeddings."""
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


def _project(delta: torch.Tensor, epsilon: float, norm: str) -> torch.Tensor:
    if norm == "inf":
        return delta.clamp(-epsilon, epsilon)
    flat = delta.reshape(delta.shape[0], -1)
    n = flat.norm(p=2, dim=-1, keepdim=True)
    factor = torch.clamp(epsilon / n.clamp(min=1e-12), max=1.0)
    return (flat * factor).reshape_as(delta)


class PGD:
    """Iterative gradient sign attack with random start and per-step projection."""

    def __init__(
        self,
        epsilon: float = 0.03,
        alpha: float = 0.005,
        steps: int = 40,
        norm: str = "inf",
        random_start: bool = True,
        clip_min: float | None = None,
        clip_max: float | None = None,
    ) -> None:
        if norm not in {"inf", "l2"}:
            raise ValueError(f"unsupported norm: {norm}")
        if steps not in (10, 20, 40) and steps <= 0:
            raise ValueError("steps must be > 0")
        self.epsilon = float(epsilon)
        self.alpha = float(alpha)
        self.steps = int(steps)
        self.norm = norm
        self.random_start = bool(random_start)
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
        x0 = batch[embedding_key].detach()
        if y is None:
            y = batch[_LABEL_KEY]
        if self.random_start:
            if self.norm == "inf":
                delta = torch.empty_like(x0).uniform_(-self.epsilon, self.epsilon)
            else:
                delta = torch.randn_like(x0)
                delta = _project(delta, self.epsilon, "l2")
        else:
            delta = torch.zeros_like(x0)
        for _ in range(self.steps):
            delta = delta.detach().requires_grad_(True)
            x = x0 + delta
            adv_batch = dict(batch)
            adv_batch[embedding_key] = x
            out = model(adv_batch)
            logits = _extract_logits(out)
            loss = F.cross_entropy(logits, y)
            grad = torch.autograd.grad(loss, delta, retain_graph=False, create_graph=False)[0]
            if self.norm == "inf":
                delta = delta.detach() + self.alpha * grad.sign()
            else:
                flat = grad.reshape(grad.shape[0], -1)
                n = flat.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-12)
                step = self.alpha * (flat / n).reshape_as(grad)
                delta = delta.detach() + step
            delta = _project(delta, self.epsilon, self.norm)
        x_adv = (x0 + delta).detach()
        if self.clip_min is not None or self.clip_max is not None:
            x_adv = x_adv.clamp(min=self.clip_min, max=self.clip_max)
        if was_training:
            model.train()
        out_batch = dict(batch)
        out_batch[embedding_key] = x_adv
        return out_batch

    def __call__(self, model: nn.Module, batch: Mapping[str, Any], y: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        return self.forward(model, batch, y)
