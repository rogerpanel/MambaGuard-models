"""DeepFool minimal-perturbation attack (Moosavi-Dezfooli et al. 2016)."""
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


class DeepFool:
    """Iterative linearised attack seeking the closest L2 decision boundary."""

    def __init__(
        self,
        max_iter: int = 50,
        overshoot: float = 0.02,
        num_classes: int = 10,
    ) -> None:
        self.max_iter = int(max_iter)
        self.overshoot = float(overshoot)
        self.num_classes = int(num_classes)

    def _attack_one(
        self,
        model: nn.Module,
        batch: Mapping[str, Any],
        embedding_key: str,
        idx: int,
    ) -> torch.Tensor:
        x0 = batch[embedding_key][idx : idx + 1].detach().clone()
        x = x0.clone().requires_grad_(True)
        single_batch = {k: (v[idx : idx + 1] if torch.is_tensor(v) and v.shape[:1] == batch[embedding_key].shape[:1] else v) for k, v in batch.items()}
        single_batch[embedding_key] = x
        out = model(single_batch)
        logits = _extract_logits(out).squeeze(0)
        k = logits.shape[-1]
        top_k = min(self.num_classes, k)
        orig_label = int(logits.argmax().item())
        r_tot = torch.zeros_like(x0)
        for _ in range(self.max_iter):
            single_batch[embedding_key] = (x0 + (1.0 + self.overshoot) * r_tot).detach().requires_grad_(True)
            out = model(single_batch)
            logits = _extract_logits(out).squeeze(0)
            cur_label = int(logits.argmax().item())
            if cur_label != orig_label:
                break
            top = torch.topk(logits, top_k).indices.tolist()
            grad_orig = torch.autograd.grad(logits[orig_label], single_batch[embedding_key], retain_graph=True)[0]
            best_dist = float("inf")
            best_w = None
            best_f = None
            for cls in top:
                if cls == orig_label:
                    continue
                grad_k = torch.autograd.grad(logits[cls], single_batch[embedding_key], retain_graph=True)[0]
                w_k = (grad_k - grad_orig).detach()
                f_k = float((logits[cls] - logits[orig_label]).item())
                w_norm = float(w_k.flatten().norm().item()) + 1e-12
                dist = abs(f_k) / w_norm
                if dist < best_dist:
                    best_dist = dist
                    best_w = w_k
                    best_f = f_k
            if best_w is None:
                break
            w_norm_sq = float(best_w.flatten().pow(2).sum().item()) + 1e-12
            r_i = (abs(best_f) / w_norm_sq) * best_w
            r_tot = r_tot + r_i
        return (x0 + (1.0 + self.overshoot) * r_tot).squeeze(0).detach()

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
        x_adv = x0.clone()
        for i in range(x0.shape[0]):
            x_adv[i] = self._attack_one(model, batch, embedding_key, i)
        if was_training:
            model.train()
        out_batch = dict(batch)
        out_batch[embedding_key] = x_adv.detach()
        return out_batch

    def __call__(self, model: nn.Module, batch: Mapping[str, Any], y: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        return self.forward(model, batch, y)
