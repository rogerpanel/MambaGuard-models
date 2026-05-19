"""Focal loss and Lipschitz regulariser for MambaGuard training."""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


_DEFAULT_LIP_KEYS: tuple[str, ...] = (
    "selective_ssm",
    "gatv2",
    "W_v",
    "W_r",
    "q_proj",
    "classifier",
    "in_proj",
    "out_proj",
)


class FocalLoss(nn.Module):
    """Multi-class focal loss (Lin et al. 2017): FL = -alpha (1 - p_t)^gamma log p_t."""

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        class_weights: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(f"invalid reduction: {reduction}")
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.ignore_index = int(ignore_index)
        self.reduction = reduction
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.dim() != 2:
            logits = logits.reshape(-1, logits.shape[-1])
            target = target.reshape(-1)
        valid = target != self.ignore_index
        if valid.sum() == 0:
            return logits.new_zeros(())
        logits_v = logits[valid]
        target_v = target[valid]
        log_probs = F.log_softmax(logits_v, dim=-1)
        log_pt = log_probs.gather(1, target_v.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        focal = -self.alpha * (1.0 - pt).pow(self.gamma) * log_pt
        if self.class_weights is not None:
            w = self.class_weights.to(focal.device).gather(0, target_v)
            focal = focal * w
        if self.reduction == "mean":
            return focal.mean()
        if self.reduction == "sum":
            return focal.sum()
        return focal


class LipschitzRegularizer(nn.Module):
    """Soft Lipschitz constraint: lambda_L * sum_l (||W^(l)||_2 - 1)_+^2."""

    def __init__(
        self,
        lambda_l: float = 1e-3,
        target_norm: float = 1.0,
        param_keys: Optional[Sequence[str]] = None,
        n_power_iter: int = 1,
    ) -> None:
        super().__init__()
        self.lambda_l = float(lambda_l)
        self.target_norm = float(target_norm)
        self.param_keys = tuple(param_keys) if param_keys is not None else _DEFAULT_LIP_KEYS
        self.n_power_iter = int(n_power_iter)
        self._u_cache: dict[str, torch.Tensor] = {}

    def _matches(self, name: str) -> bool:
        return any(k in name for k in self.param_keys)

    def _collect(self, model: nn.Module) -> Iterable[tuple[str, torch.Tensor]]:
        for name, p in model.named_parameters():
            if not p.requires_grad or p.dim() < 2:
                continue
            if not self._matches(name):
                continue
            yield name, p

    def _spectral(self, name: str, w: torch.Tensor) -> torch.Tensor:
        mat = w.reshape(w.shape[0], -1)
        u = self._u_cache.get(name)
        if u is None or u.shape[0] != mat.shape[0] or u.device != mat.device:
            u = torch.randn(mat.shape[0], device=mat.device, dtype=mat.dtype)
            u = u / (u.norm() + 1e-12)
        for _ in range(self.n_power_iter):
            v = torch.mv(mat.t(), u)
            v = v / (v.norm() + 1e-12)
            u = torch.mv(mat, v)
            u = u / (u.norm() + 1e-12)
        self._u_cache[name] = u.detach()
        sigma = torch.dot(u, torch.mv(mat, v))
        return sigma

    def forward(self, model: nn.Module) -> torch.Tensor:
        device = next(model.parameters()).device
        total = torch.zeros((), device=device)
        count = 0
        for name, p in self._collect(model):
            sigma = self._spectral(name, p)
            excess = F.relu(sigma - self.target_norm)
            total = total + excess.pow(2)
            count += 1
        if count == 0:
            return total
        return self.lambda_l * total


def combined_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    model: nn.Module,
    lipschitz_reg: Optional[LipschitzRegularizer],
    focal_loss: Optional[FocalLoss] = None,
) -> dict[str, torch.Tensor]:
    """Combine cross-entropy/focal classification loss with the Lipschitz penalty."""
    if focal_loss is None:
        ce = F.cross_entropy(logits, labels)
    else:
        ce = focal_loss(logits, labels)
    if lipschitz_reg is None:
        lip = logits.new_zeros(())
    else:
        lip = lipschitz_reg(model)
    total = ce + lip
    return {"loss": total, "ce": ce.detach(), "lip": lip.detach()}
