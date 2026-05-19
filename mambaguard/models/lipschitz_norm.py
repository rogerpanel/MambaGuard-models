"""Lipschitz-bounded attention rescaling (Dasoulas et al. 2021)."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class LipschitzNorm(nn.Module):
    """Rescales attention logits by 1/max(1, ‖[q;k]‖₂/τ) to bound Lipschitz constant.

    The rescaling preserves softmax direction while ensuring the attention
    map is 1-Lipschitz in its inputs up to the temperature τ.
    """

    def __init__(self, tau: float = 1.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.tau = tau
        self.eps = eps

    def forward(
        self,
        logits: torch.Tensor,
        query_norm: torch.Tensor,
        key_norm: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Rescale ``logits`` using the joint (q, k) norm bound."""
        if key_norm is None:
            joint = query_norm
        else:
            joint = torch.sqrt(query_norm.pow(2) + key_norm.pow(2) + self.eps)
        scale = torch.clamp(joint / self.tau, min=1.0)
        return logits / scale


def spectral_norm_constraint(weight: torch.Tensor, max_norm: float = 1.0) -> torch.Tensor:
    """Projects ``weight`` to have ℓ2 operator norm ≤ ``max_norm``.

    Used as a regulariser hook; returns the rescaled weight without mutating.
    """
    if weight.dim() < 2:
        return weight
    w = weight.reshape(weight.shape[0], -1)
    sigma = torch.linalg.matrix_norm(w, ord=2)
    factor = torch.clamp(max_norm / (sigma + 1e-12), max=1.0)
    return weight * factor
