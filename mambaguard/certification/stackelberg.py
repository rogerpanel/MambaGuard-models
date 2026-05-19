"""Strong Stackelberg equilibrium solver for the defender-attacker game.

Game (paper Section: Stackelberg Security Game):
    Γ = (A_D, A_A, u_D, u_A)
where the defender commits to a mixed strategy π_D ∈ Δ(A_D) and the
(rational) attacker best-responds. The defender's optimisation is

    π*_D = argmax_{π_D ∈ Δ(A_D)} min_{a ∈ BR(π_D)} u_D(π_D, a, G).

We solve it via the multiple-LP formulation of Conitzer & Sandholm (2006,
"Computing the Optimal Strategy to Commit to"): for each attacker pure
strategy a_j, solve the LP that *enforces* a_j to be a best response, and
keep the j that yields the largest defender value.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Canonical action labels --------------------------------------------------- #
DEFENDER_ACTIONS = [
    "deep_packet_inspection",
    "message_blocking",
    "session_termination",
    "agent_isolation",
    "capability_revocation",
]

ATTACKER_ACTIONS = [
    "tool_poisoning",
    "indirect_prompt_injection",
    "rug_pull_update",
    "capability_shadowing",
    "cross_server_confusion",
    "credential_exfil",
    "sandbox_bypass",
]


@dataclass
class StackelbergSolution:
    """Result of :meth:`StackelbergSolver.solve`."""

    pi_D: np.ndarray  # shape (|A_D|,) — defender mixed strategy
    attacker_best_response: int  # index into A_A
    value: float  # defender's expected utility at the SSE


class StackelbergSolver:
    """Strong Stackelberg equilibrium via the Conitzer-Sandholm multiple-LP.

    Parameters
    ----------
    defender_actions : list[str]
        Names of defender pure strategies (rows of ``utility_matrix``).
    attacker_actions : list[str]
        Names of attacker pure strategies (columns of ``utility_matrix``).
    utility_matrix : np.ndarray
        ``utility_matrix[i, j] = u_D(s_i, a_j)`` — defender utility when she
        plays s_i and the attacker plays a_j. We assume the attacker's utility
        is the zero-sum counterpart u_A = −u_D unless overridden via
        ``attacker_utility``.
    attacker_utility : np.ndarray | None
        Optional explicit attacker matrix of the same shape. Defaults to
        ``-utility_matrix`` (zero-sum).
    """

    def __init__(
        self,
        defender_actions: list[str],
        attacker_actions: list[str],
        utility_matrix: np.ndarray,
        attacker_utility: np.ndarray | None = None,
    ) -> None:
        U = np.asarray(utility_matrix, dtype=float)
        if U.shape != (len(defender_actions), len(attacker_actions)):
            raise AssertionError(
                f"utility_matrix shape {U.shape} ≠ "
                f"(|A_D|={len(defender_actions)}, |A_A|={len(attacker_actions)})"
            )
        if attacker_utility is None:
            UA = -U
        else:
            UA = np.asarray(attacker_utility, dtype=float)
            if UA.shape != U.shape:
                raise AssertionError("attacker_utility must match utility_matrix shape")
        self.defender_actions = list(defender_actions)
        self.attacker_actions = list(attacker_actions)
        self.U = U
        self.UA = UA

    # --------------------------------------------------------------- #
    def solve(self) -> StackelbergSolution:
        """Compute the strong Stackelberg equilibrium via |A_A| LPs."""
        try:
            import cvxpy as cp  # noqa: WPS433 - lazy import by design
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "StackelbergSolver.solve requires cvxpy. "
                "Install with: pip install cvxpy"
            ) from exc

        n_def, n_att = self.U.shape
        best_value = -np.inf
        best_pi: np.ndarray | None = None
        best_j: int = -1

        for j in range(n_att):
            pi = cp.Variable(n_def, nonneg=True)
            v = cp.Variable()
            attacker_payoff_j = self.UA[:, j] @ pi
            constraints = [cp.sum(pi) == 1, v == self.U[:, j] @ pi]
            # Force attacker action j to be a best response.
            for k in range(n_att):
                if k == j:
                    continue
                constraints.append(attacker_payoff_j >= self.UA[:, k] @ pi)
            prob = cp.Problem(cp.Maximize(v), constraints)
            try:
                prob.solve()
            except cp.error.SolverError:  # pragma: no cover
                continue
            if prob.status not in {"optimal", "optimal_inaccurate"}:
                continue
            if v.value is None or pi.value is None:
                continue
            if float(v.value) > best_value:
                best_value = float(v.value)
                best_pi = np.asarray(pi.value, dtype=float).clip(min=0.0)
                if best_pi.sum() > 0:
                    best_pi = best_pi / best_pi.sum()
                best_j = j

        if best_pi is None:
            raise RuntimeError("No feasible Stackelberg LP — check utility matrix.")
        return StackelbergSolution(
            pi_D=best_pi, attacker_best_response=best_j, value=best_value
        )


# ------------------------------------------------------------------ #
# Reference utility matrix                                           #
# ------------------------------------------------------------------ #
def default_utility_matrix(B: float = 1.0) -> np.ndarray:
    """Plausible 5×7 defender utility table u_D ∈ [-B, B].

    Rows = defender actions (DEFENDER_ACTIONS), columns = attacker families
    (ATTACKER_ACTIONS). Positive entries reflect successful detection /
    mitigation by the defender; negative entries reflect missed attacks or
    wasteful blocks. Magnitudes are illustrative — meant to populate
    unit-tests and configs, not encode real operational priors.
    """
    # columns:        TP   IPI   RP   CS   CSC  CEX  SBX
    M = np.array(
        [
            [0.7,  0.6,  0.3,  0.5,  0.6,  0.4,  0.2],  # DPI
            [0.8,  0.7,  0.2,  0.4,  0.5,  0.6,  0.3],  # Message blocking
            [0.5,  0.4,  0.8,  0.3,  0.4,  0.7,  0.6],  # Session termination
            [0.4,  0.5,  0.6,  0.7,  0.6,  0.5,  0.8],  # Agent isolation
            [0.3,  0.3,  0.7,  0.8,  0.5,  0.6,  0.7],  # Capability revocation
        ],
        dtype=float,
    )
    return B * M


# ------------------------------------------------------------------ #
# Sanity check                                                       #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    U = default_utility_matrix(B=1.0)
    print("Utility matrix shape:", U.shape)
    solver = StackelbergSolver(DEFENDER_ACTIONS, ATTACKER_ACTIONS, U)
    try:
        sol = solver.solve()
        print(f"SSE value V* = {sol.value:.4f}")
        print("π*_D =")
        for name, p in zip(DEFENDER_ACTIONS, sol.pi_D):
            print(f"   {name:24s}: {p:.4f}")
        print(
            "Attacker BR =",
            ATTACKER_ACTIONS[sol.attacker_best_response],
        )
    except ImportError as e:
        print(f"[skip] {e}")
