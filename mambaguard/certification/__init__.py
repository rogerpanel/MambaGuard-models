"""MambaGuard certification framework.

Three-layer composed certificate combining:
  1. Lipschitz bounds on the selective-SSM + GATv2 backbone (Section: Robustness).
  2. Strong Stackelberg equilibrium for the defender-attacker game (Section: Game).
  3. Hedge (multiplicative-weights) online defence with tight regret
     (Section: Online Defence, Algorithm 1).

The composed guarantee (Theorem):
    E[V_ach(π̂_T, ε)] ≥ V* − L_f · ε − B · sqrt(ln|A_D| / (2T)).
"""
from __future__ import annotations

from .lipschitz_bounds import (
    CertifiedRadius,
    composed_lipschitz,
    compute_lipschitz_bound,
    gatv2_lipschitz,
    spectral_norm,
    ssm_lipschitz,
)
from .stackelberg import StackelbergSolution, StackelbergSolver, default_utility_matrix
from .hedge import HedgeDefender, run_hedge_defence
from .certificate import CompositeCertificate, composed_certificate, report

__all__ = [
    # Lipschitz
    "spectral_norm",
    "ssm_lipschitz",
    "gatv2_lipschitz",
    "composed_lipschitz",
    "compute_lipschitz_bound",
    "CertifiedRadius",
    # Stackelberg
    "StackelbergSolver",
    "StackelbergSolution",
    "default_utility_matrix",
    # Hedge
    "HedgeDefender",
    "run_hedge_defence",
    # Composed certificate
    "CompositeCertificate",
    "composed_certificate",
    "report",
]
