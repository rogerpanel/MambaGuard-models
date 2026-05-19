"""Bochner-style learnable time encoding (TGAT, Xu et al. 2020)."""
from __future__ import annotations

import math
import torch
import torch.nn as nn


class BochnerTimeEncoding(nn.Module):
    """Maps Δt → [cos(ωΔt), sin(ωΔt)] · √(1/d_T) with learnable frequencies ω."""

    def __init__(self, d_T: int = 64) -> None:
        super().__init__()
        if d_T % 2 != 0:
            raise ValueError("d_T must be even (cos/sin pairs).")
        self.d_T = d_T
        half = d_T // 2
        # Log-spaced initial frequencies span ~1e-3 .. 1e+1 rad/unit-time so
        # the encoding has both slow drift and fast burst-scale sensitivity.
        omega = torch.logspace(-3.0, 1.0, half)
        self.omega = nn.Parameter(omega)
        self.phase = nn.Parameter(torch.zeros(half))
        self.register_buffer("_scale", torch.tensor(math.sqrt(1.0 / d_T)))

    def forward(self, delta_t: torch.Tensor) -> torch.Tensor:
        """``delta_t``: arbitrary shape ``(...)``. Returns ``(..., d_T)``."""
        x = delta_t.unsqueeze(-1) * self.omega + self.phase
        out = torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
        return out * self._scale
