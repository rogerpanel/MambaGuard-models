"""Selective state-space (Mamba) block with ZOH discretisation and Lipschitz bound."""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .conv1d_depthwise import DepthwiseCausalConv1d
from .discretization import hippo_s4d_init, zoh_discretize

try:  # pragma: no cover - optional fast path
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn  # type: ignore

    _HAS_MAMBA = True
except Exception:  # noqa: BLE001
    _HAS_MAMBA = False


def _reference_selective_scan(
    u: torch.Tensor,       # (B, L, D)
    delta: torch.Tensor,   # (B, L, D)
    A: torch.Tensor,       # (D, N) — diagonal, real, negative
    B: torch.Tensor,       # (B, L, N)
    C: torch.Tensor,       # (B, L, N)
    D_skip: Optional[torch.Tensor] = None,  # (D,)
) -> torch.Tensor:
    """Pure-PyTorch reference scan — replace with selective_scan_fn for production.

    The scan must be causal: h_t depends only on (u_{≤t}, B_{≤t}). Iterating in
    Python is O(L) but exact and gradient-correct for unit tests / certification.
    """
    B_sz, L, D = u.shape
    N = A.shape[-1]
    h = u.new_zeros(B_sz, D, N)
    ys = []
    # Diagonal-A path: A_bar (B,L,D,N) and B_bar share that shape after broadcast.
    dA = delta.unsqueeze(-1) * A  # (B, L, D, N)
    A_bar = torch.exp(dA)
    small = dA.abs() < 1e-6
    coeff = torch.where(
        small,
        delta.unsqueeze(-1) * (1.0 + 0.5 * dA + (dA * dA) / 6.0),
        torch.expm1(dA) / A,
    )
    # u·B_t broadcast to per-channel input drive.
    uB = u.unsqueeze(-1) * B.unsqueeze(-2)  # (B, L, D, N)
    B_bar_u = coeff * uB
    for t in range(L):
        h = A_bar[:, t] * h + B_bar_u[:, t]      # (B, D, N)
        y_t = (h * C[:, t].unsqueeze(1)).sum(-1)  # (B, D)
        ys.append(y_t)
    y = torch.stack(ys, dim=1)  # (B, L, D)
    if D_skip is not None:
        y = y + u * D_skip
    return y


class SelectiveSSMBlock(nn.Module):
    """Mamba-style selective SSM block: input-dependent (Δ, B, C) over a diagonal A."""

    def __init__(
        self,
        d_model: int = 256,
        d_state: int = 16,
        expand: int = 2,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        kernel_size: int = 4,
        dt_rank: Optional[int] = None,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = expand * d_model
        self.dt_rank = dt_rank or max(1, math.ceil(d_model / 16))
        self.dt_min = dt_min
        self.dt_max = dt_max

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=bias)
        self.conv1d = DepthwiseCausalConv1d(self.d_inner, kernel_size=kernel_size)

        # Input-dependent selective projections: (Δ_t low-rank, B_t, C_t).
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Initialise dt bias so softplus(b) ∈ [dt_min, dt_max] (Mamba convention).
        dt_init = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        inv_dt = dt_init + torch.log(-torch.expm1(-dt_init))  # inverse-softplus
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # S4D-Lin log-parameterisation: store log(-A) so A = -exp(A_log) stays Hurwitz
        # under unconstrained optimisation.
        A = hippo_s4d_init(d_state).unsqueeze(0).expand(self.d_inner, d_state).contiguous()
        self.A_log = nn.Parameter(torch.log(-A))
        self.D_skip = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

    # ----- core forward -----
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: (B, L, d_model). Returns (B, L, d_model)."""
        B_sz, L, _ = x.shape
        xz = self.in_proj(x)  # (B, L, 2·d_inner)
        x_in, gate = xz.chunk(2, dim=-1)

        x_in = self.conv1d(x_in)
        x_in = F.silu(x_in)

        # Selective parameters.
        x_dbl = self.x_proj(x_in)  # (B, L, dt_rank + 2N)
        dt, B_t, C_t = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        delta = F.softplus(self.dt_proj(dt))  # (B, L, d_inner)

        A = -torch.exp(self.A_log)  # (d_inner, N), strictly negative

        if _HAS_MAMBA:  # pragma: no cover
            # mamba_ssm expects (B, D, L) ordering.
            y = selective_scan_fn(
                rearrange(x_in, "b l d -> b d l"),
                rearrange(delta, "b l d -> b d l"),
                A,
                rearrange(B_t, "b l n -> b n l"),
                rearrange(C_t, "b l n -> b n l"),
                self.D_skip,
                z=None,
                delta_bias=None,
                delta_softplus=False,
            )
            y = rearrange(y, "b d l -> b l d")
        else:
            y = _reference_selective_scan(x_in, delta, A, B_t, C_t, self.D_skip)

        y = y * F.silu(gate)
        return self.out_proj(y)

    # ----- certification helpers -----
    @torch.no_grad()
    def lipschitz_bound(self) -> float:
        """Composed bound ``(γ·β·(1+κ_g))/(1−ρ)`` over the worst-case discretisation.

        ρ = ‖Ā‖₂ at Δ = dt_max, β = ‖B̄‖₂ at the same Δ (data-independent worst
        case), γ = ‖W_C‖₂, κ_g = ‖W_gate‖₂. Returns +∞ if ρ ≥ 1.
        """
        A = -torch.exp(self.A_log)  # (d_inner, N)
        dt = torch.tensor(self.dt_max, dtype=A.dtype, device=A.device)
        A_bar = torch.exp(dt * A)
        rho = float(A_bar.abs().max().item())  # diagonal → ℓ2 = max |entry|
        if rho >= 1.0:
            return float("inf")

        # B̄ per-channel coefficient bound: max_n |(e^{ΔA_n}-1)/A_n|.
        coeff = torch.where(
            (dt * A).abs() < 1e-6,
            dt.expand_as(A),
            torch.expm1(dt * A) / A,
        )
        # The B input is x_proj's middle slice; its operator norm bounds the B drive.
        W_x = self.x_proj.weight  # (dt_rank+2N, d_inner)
        W_B = W_x[self.dt_rank : self.dt_rank + self.d_state]
        W_C = W_x[self.dt_rank + self.d_state :]
        beta = float(coeff.abs().max().item() * torch.linalg.matrix_norm(W_B, ord=2).item())
        gamma = float(torch.linalg.matrix_norm(W_C, ord=2).item())

        # Gate projection lives in the second half of in_proj.
        W_in = self.in_proj.weight  # (2·d_inner, d_model)
        W_gate = W_in[self.d_inner :]
        kappa_g = float(torch.linalg.matrix_norm(W_gate, ord=2).item())

        # Out and conv contributions fold into γ multiplicatively.
        W_out = float(torch.linalg.matrix_norm(self.out_proj.weight, ord=2).item())
        conv_lip = self.conv1d.lipschitz()
        gamma_eff = gamma * W_out * conv_lip

        return (gamma_eff * beta * (1.0 + kappa_g)) / (1.0 - rho)
