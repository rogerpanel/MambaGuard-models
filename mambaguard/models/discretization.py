"""Zero-order-hold (ZOH) discretisation utilities for selective state-space models."""
from __future__ import annotations

from typing import Tuple

import math
import torch
import torch.nn.functional as F


def zoh_discretize(
    A: torch.Tensor,
    B: torch.Tensor,
    delta: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ZOH discretisation of (A, B) at step size ``delta``.

    Supports two regimes:
      * Dense ``A`` of shape ``(N, N)``: uses :func:`torch.linalg.matrix_exp`.
      * Diagonal ``A`` of shape ``(..., N)`` (S4D fast path): elementwise exp.

    Returns ``(A_bar, B_bar)`` with leading broadcast dims matching ``delta``.
    """
    if A.dim() == 1 or (A.dim() >= 2 and A.shape[-1] != A.shape[-2]):
        # Diagonal-A fast path (S4D). A: (..., N), B: (..., N), delta: (..., 1) or (..., N).
        dA = delta * A  # (..., N)
        A_bar = torch.exp(dA)
        # (e^{dA} - 1)/A  ·  B  — numerically stable via expm1.
        # Avoid divide-by-zero when A is tiny: use Taylor expansion fallback.
        small = dA.abs() < 1e-6
        coeff = torch.where(
            small,
            delta * (1.0 + 0.5 * dA + (dA * dA) / 6.0),
            torch.expm1(dA) / A,
        )
        B_bar = coeff * B
        return A_bar, B_bar

    # Dense matrix-A path.
    # delta is expected to broadcast to a scalar per batch element along last two dims.
    if delta.dim() < A.dim():
        delta = delta.view(*delta.shape, *([1] * (A.dim() - delta.dim())))
    dA = delta * A
    A_bar = torch.linalg.matrix_exp(dA)
    eye = torch.eye(A.shape[-1], dtype=A.dtype, device=A.device).expand_as(A_bar)
    # B_bar = A^{-1} (e^{ΔA} − I) B   (Δ absorbed because dA = ΔA already).
    rhs = (A_bar - eye) @ B
    B_bar = torch.linalg.solve(A, rhs)
    return A_bar, B_bar


def hippo_s4d_init(N: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """S4D-Lin Hurwitz diagonal initialisation: ``A_n = -1/2 + i π n``.

    Returns the real part (strictly negative) so that ``ρ(exp(ΔA)) < 1``;
    the imaginary structure of full S4D is discarded — the selective Mamba
    formulation uses a real diagonal A and learns frequency content via Δ.
    """
    n = torch.arange(N, dtype=dtype)
    # Real, strictly negative; magnitudes grow mildly with state index for
    # frequency-diverse modes (S4D-Lin parameterisation).
    A = -0.5 - 0.01 * n
    return A
