"""Hedge / multiplicative-weights online defence (Algorithm 1 in the paper).

Reference (paper Section: Online Defence):

    Algorithm 1 — Multiplicative-weights Hedge
    ------------------------------------------
    Initialise w_1(s) = 1 for all s ∈ A_D
    η = sqrt(2 · ln|A_D| / T)
    For t = 1..T:
        observe game state G(t)
        compute attacker scores via MambaGuard
        sample s_t ~ p_t(s) = w_t(s) / Σ_{s'} w_t(s')
        observe loss ℓ_t(s) ∈ [0, B]
        w_{t+1}(s) = w_t(s) · exp(−η · ℓ_t(s))

Tight regret bound (Cesa-Bianchi & Lugosi, *Prediction, Learning, and
Games*, 2006, Theorem 2.2 / Corollary 4.2):

    R(T) ≤ B · sqrt((T/2) · ln|A_D|)
    R(T)/T ≤ B · sqrt(ln|A_D| / (2T))   (average regret → 0).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


# ------------------------------------------------------------------ #
# Hedge / Multiplicative-Weights                                     #
# ------------------------------------------------------------------ #
class HedgeDefender:
    """Hedge (a.k.a. multiplicative-weights / exponential-weights) learner.

    Parameters
    ----------
    actions : list[str]
        Names of defender pure strategies (the experts).
    horizon : int
        Total number of rounds T (used to set the optimal learning rate).
    B : float
        Per-round loss range; losses must lie in [0, B].
    """

    def __init__(self, actions: list[str], horizon: int, B: float = 1.0) -> None:
        if horizon <= 0:
            raise ValueError(f"horizon must be positive, got {horizon}")
        if B <= 0:
            raise ValueError(f"B must be positive, got {B}")
        if len(actions) < 2:
            raise ValueError("HedgeDefender needs at least 2 actions")
        self.actions: list[str] = list(actions)
        self.horizon: int = int(horizon)
        self.B: float = float(B)
        # η chosen to minimise the standard regret bound; see CBL 2006 §2.
        self.eta: float = math.sqrt(2.0 * math.log(len(self.actions)) / self.horizon)
        self.weights: np.ndarray = np.ones(len(self.actions), dtype=float)
        self._t: int = 0  # round counter

    # --------------------------------------------------------------- #
    @property
    def num_actions(self) -> int:
        return len(self.actions)

    def distribution(self) -> np.ndarray:
        """Return the current sampling distribution p_t."""
        z = float(self.weights.sum())
        if z <= 0.0 or not math.isfinite(z):
            # Numerical fallback: re-normalise to uniform.
            return np.full(self.num_actions, 1.0 / self.num_actions)
        return self.weights / z

    def sample(self, rng: np.random.Generator | None = None) -> int:
        """Sample an action index from p_t."""
        if rng is None:
            rng = np.random.default_rng()
        p = self.distribution()
        return int(rng.choice(self.num_actions, p=p))

    def update(self, losses: np.ndarray) -> None:
        """Apply the multiplicative-weights update.

        Parameters
        ----------
        losses : np.ndarray
            Per-action loss vector ℓ_t of shape (|A_D|,), with 0 ≤ ℓ_t ≤ B.
        """
        losses = np.asarray(losses, dtype=float)
        if losses.shape != (self.num_actions,):
            raise ValueError(
                f"losses must have shape ({self.num_actions},), got {losses.shape}"
            )
        if not np.all((losses >= -1e-12) & (losses <= self.B + 1e-12)):
            raise AssertionError(
                f"Hedge requires losses ∈ [0, {self.B}]; got min={losses.min()}, "
                f"max={losses.max()}"
            )
        # Clip to legal range to absorb floating-point noise.
        losses = np.clip(losses, 0.0, self.B)
        self.weights *= np.exp(-self.eta * losses)
        # Numerical safety: re-scale if weights become tiny.
        m = float(self.weights.max())
        if m > 0 and m < 1e-30:
            self.weights /= m
        self._t += 1

    # --------------------------------------------------------------- #
    def regret_bound(self) -> float:
        """Cumulative-regret upper bound:  B · sqrt((T/2) · ln|A_D|)."""
        return self.B * math.sqrt(0.5 * self.horizon * math.log(self.num_actions))

    def average_regret_bound(self) -> float:
        """Average-regret upper bound:  B · sqrt(ln|A_D| / (2T))."""
        return self.B * math.sqrt(math.log(self.num_actions) / (2.0 * self.horizon))


# ------------------------------------------------------------------ #
# Reference Algorithm 1 driver                                       #
# ------------------------------------------------------------------ #
@dataclass
class HedgeTrajectory:
    """Per-round logs produced by :func:`run_hedge_defence`."""

    sampled_actions: list[int] = field(default_factory=list)
    per_round_losses: list[float] = field(default_factory=list)
    cumulative_loss: list[float] = field(default_factory=list)
    best_fixed_loss: float = 0.0
    realised_regret: float = 0.0
    regret_bound: float = 0.0

    def as_dict(self) -> dict:
        return {
            "sampled_actions": list(self.sampled_actions),
            "per_round_losses": list(self.per_round_losses),
            "cumulative_loss": list(self.cumulative_loss),
            "best_fixed_loss": self.best_fixed_loss,
            "realised_regret": self.realised_regret,
            "regret_bound": self.regret_bound,
        }


def run_hedge_defence(
    scorer: Callable[[int], np.ndarray],
    defender: HedgeDefender,
    T: int,
    rng: np.random.Generator | None = None,
) -> dict:
    """Reference implementation of Algorithm 1.

    Parameters
    ----------
    scorer : Callable[[int], np.ndarray]
        Function ``t → ℓ_t`` returning the per-action loss vector at round t.
        In practice this composes MambaGuard scoring with a chosen loss.
    defender : HedgeDefender
        Online learner (its horizon should be ≥ T).
    T : int
        Number of rounds to run.
    rng : np.random.Generator | None
        Source of randomness for action sampling.
    """
    if rng is None:
        rng = np.random.default_rng()
    if T > defender.horizon:
        raise ValueError(
            f"T={T} exceeds defender.horizon={defender.horizon}; "
            "regret bound would be invalid."
        )
    traj = HedgeTrajectory(regret_bound=defender.regret_bound())
    cumulative_per_action = np.zeros(defender.num_actions, dtype=float)
    cum_loss = 0.0
    for t in range(T):
        losses_t = np.asarray(scorer(t), dtype=float)
        a_t = defender.sample(rng=rng)
        loss_t = float(losses_t[a_t])
        traj.sampled_actions.append(a_t)
        traj.per_round_losses.append(loss_t)
        cum_loss += loss_t
        traj.cumulative_loss.append(cum_loss)
        cumulative_per_action += losses_t
        defender.update(losses_t)
    traj.best_fixed_loss = float(cumulative_per_action.min())
    traj.realised_regret = cum_loss - traj.best_fixed_loss
    return traj.as_dict()


# ------------------------------------------------------------------ #
# Sanity check                                                       #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    K, T = 10, 1000
    actions = [f"a{i}" for i in range(K)]
    defender = HedgeDefender(actions=actions, horizon=T, B=1.0)
    rng = np.random.default_rng(0)

    # Stationary stochastic loss: action 3 is best in expectation.
    true_means = rng.uniform(0.3, 0.8, size=K)
    true_means[3] = 0.1

    def scorer(_t: int) -> np.ndarray:
        # Bernoulli-style losses bounded in [0,1].
        return (rng.uniform(size=K) < true_means).astype(float)

    result = run_hedge_defence(scorer, defender, T=T, rng=rng)
    print(f"Hedge run: T={T}, K={K}, η={defender.eta:.4f}")
    print(f"  cumulative loss      = {result['cumulative_loss'][-1]:.2f}")
    print(f"  best-fixed-in-hindsight loss = {result['best_fixed_loss']:.2f}")
    print(f"  realised regret      = {result['realised_regret']:.2f}")
    print(f"  theoretical bound    = {result['regret_bound']:.2f}")
    print(f"  average regret       = {result['realised_regret']/T:.4f}")
    print(f"  average bound        = {defender.average_regret_bound():.4f}")
