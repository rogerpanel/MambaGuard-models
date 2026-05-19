"""Carlini & Wagner L2 attack (Carlini & Wagner 2017)."""
from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn


_EMBEDDING_KEY = "p"
_LABEL_KEY = "labels"


def _extract_logits(out: Any) -> torch.Tensor:
    if isinstance(out, dict):
        return out["logits"]
    return out


class CarliniWagnerL2:
    """L2 C&W attack minimising ||delta||^2 + c * f(x+delta) with margin loss."""

    def __init__(
        self,
        c: float = 1e-3,
        steps: int = 1000,
        lr: float = 5e-3,
        kappa: float = 0.0,
        targeted: bool = False,
    ) -> None:
        self.c = float(c)
        self.steps = int(steps)
        self.lr = float(lr)
        self.kappa = float(kappa)
        self.targeted = bool(targeted)

    def _f(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        n, k = logits.shape
        one_hot = torch.zeros_like(logits).scatter_(1, y.view(-1, 1), 1.0)
        real = (one_hot * logits).sum(dim=1)
        other = ((1.0 - one_hot) * logits - one_hot * 1e9).max(dim=1).values
        if self.targeted:
            return torch.clamp(other - real + self.kappa, min=0.0)
        return torch.clamp(real - other + self.kappa, min=0.0)

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
        delta = torch.zeros_like(x0, requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=self.lr)
        best_norm = torch.full((x0.shape[0],), float("inf"), device=x0.device)
        best_delta = torch.zeros_like(x0)
        for _ in range(self.steps):
            optimizer.zero_grad()
            x_adv = x0 + delta
            adv_batch = dict(batch)
            adv_batch[embedding_key] = x_adv
            out = model(adv_batch)
            logits = _extract_logits(out)
            l2 = delta.reshape(delta.shape[0], -1).pow(2).sum(dim=-1)
            f_term = self._f(logits, y)
            loss = (l2 + self.c * f_term).sum()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                success = (preds != y) if not self.targeted else (preds == y)
                cur_norm = l2.sqrt()
                improve = success & (cur_norm < best_norm)
                best_norm = torch.where(improve, cur_norm, best_norm)
                idx = improve.nonzero(as_tuple=False).flatten()
                if idx.numel() > 0:
                    best_delta[idx] = delta.detach()[idx]
        x_adv = (x0 + best_delta).detach()
        if was_training:
            model.train()
        out_batch = dict(batch)
        out_batch[embedding_key] = x_adv
        return out_batch

    def __call__(self, model: nn.Module, batch: Mapping[str, Any], y: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        return self.forward(model, batch, y)
