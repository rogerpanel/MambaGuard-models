"""Causal depthwise 1-D convolution used as the short-range mixer in Mamba blocks."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseCausalConv1d(nn.Module):
    """Depthwise 1-D conv with left-padding to preserve causality (groups=d_model)."""

    def __init__(self, d_model: int, kernel_size: int = 4, bias: bool = True) -> None:
        super().__init__()
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.pad = kernel_size - 1
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            groups=d_model,
            bias=bias,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: (B, L, D). Returns (B, L, D)."""
        x = x.transpose(1, 2)  # (B, D, L)
        x = F.pad(x, (self.pad, 0))  # left-pad only → causal
        x = self.conv(x)
        return x.transpose(1, 2)

    @torch.no_grad()
    def lipschitz(self) -> float:
        """Per-channel ℓ2 bound: max over channels of Σ|w|.

        For depthwise conv each output channel is an independent FIR; the
        operator norm is upper-bounded by the ℓ1 norm of its kernel.
        """
        w = self.conv.weight.detach()  # (D, 1, K)
        per_channel = w.abs().sum(dim=(1, 2))
        return float(per_channel.max().item())
