"""Lipschitz bounds for the MambaGuard backbone.

References (paper):
  - Lemma (Selective-SSM Lipschitzness):
        L_ssm = (γ·β·(1+κ_g)) / (1 − ρ),   ρ = sup_t ‖Ā_t‖₂ < 1.
  - Lemma (GATv2 + LipschitzNorm):
        L_gat ≤ ‖W_v‖₂ · (1 + η · ‖q‖₂ · ‖W_r‖₂).
  - Proposition (Composition):
        L_f ≤ M_h · L_gat · (γβ(1+κ_g)) / (1 − ρ).

We rely on the SiLU activation having Lipschitz constant L_σ = 1
(see e.g. Latorre et al., 2020).
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor, nn


# ------------------------------------------------------------------ #
# Atomic spectral helpers                                            #
# ------------------------------------------------------------------ #
def spectral_norm(W: Tensor) -> float:
    """Return the operator 2-norm ‖W‖₂ of a weight tensor.

    For 4-D convolution kernels (out, in, kH, kW) we use the standard
    "matricised" upper bound by reshaping to (out, in·kH·kW); this is a
    valid Lipschitz upper bound (see Sedghi et al. 2019, §3 for a tighter
    circulant treatment if required).
    """
    if W.ndim == 4:
        W = W.reshape(W.shape[0], -1)
    elif W.ndim == 3:  # 1-D conv (out, in, k)
        W = W.reshape(W.shape[0], -1)
    elif W.ndim == 1:
        return float(W.abs().max().item())
    return float(torch.linalg.matrix_norm(W, ord=2).item())


def ssm_lipschitz(rho: float, beta: float, gamma: float, kappa_g: float) -> float:
    """Selective-SSM block Lipschitz bound (paper Lemma).

    L_ssm = γ · β · (1 + κ_g) / (1 − ρ).
    Requires ρ < 1 (contractive discretised state matrix).
    """
    if not (0.0 <= rho < 1.0):
        raise ValueError(
            f"ssm_lipschitz: ρ must lie in [0,1) for a contractive SSM, got ρ={rho!r}. "
            "Check that dt_max·A_min < 0 and |exp(dt·A)| < 1."
        )
    if min(beta, gamma, kappa_g) < 0:
        raise ValueError("β, γ, κ_g must be non-negative spectral norms.")
    return gamma * beta * (1.0 + kappa_g) / (1.0 - rho)


def gatv2_lipschitz(
    W_v_norm: float,
    q_norm: float,
    W_r_norm: float,
    eta_norm: float,
) -> float:
    """GATv2-with-LipschitzNorm bound (paper Lemma).

    L_gat ≤ ‖W_v‖₂ · (1 + η · ‖q‖₂ · ‖W_r‖₂).
    ``eta_norm`` is the LipschitzNorm temperature η.
    """
    for name, v in (
        ("W_v_norm", W_v_norm),
        ("q_norm", q_norm),
        ("W_r_norm", W_r_norm),
        ("eta_norm", eta_norm),
    ):
        if v < 0:
            raise ValueError(f"gatv2_lipschitz: {name}={v} must be non-negative")
    return W_v_norm * (1.0 + eta_norm * q_norm * W_r_norm)


def composed_lipschitz(layer_bounds: Iterable[float], head_factor: float = 1.0) -> float:
    """Compose per-layer Lipschitz constants by the chain rule.

    If f = f_L ∘ ... ∘ f_1 with each f_i being L_i-Lipschitz, then f is
    (∏_i L_i)-Lipschitz (Federer 1969, §2.2.6). For multi-head averaging the
    output is (1/H) Σ_h head_h, so the aggregated Lipschitz constant gains a
    factor M_h ≤ 1 (here ``head_factor``).
    """
    prod = 1.0
    for L in layer_bounds:
        if L < 0:
            raise ValueError(f"composed_lipschitz: per-layer bound must be ≥ 0 (got {L})")
        prod *= L
    return head_factor * prod


# ------------------------------------------------------------------ #
# Model introspection                                                #
# ------------------------------------------------------------------ #
def _spectral_fallback(module: nn.Module) -> float:
    """Product of spectral norms of all 2-D+ weights in *module*."""
    bound = 1.0
    found = False
    for name, p in module.named_parameters(recurse=True):
        if p.ndim >= 2 and "weight" in name:
            bound *= spectral_norm(p.detach())
            found = True
    return bound if found else 1.0


def compute_lipschitz_bound(model: nn.Module) -> dict:
    """Walk *model* and return per-component + global Lipschitz bounds.

    Recognised sub-modules (by class-name suffix, to avoid hard imports):
        SelectiveSSMBlock     → child.lipschitz_bound()
        TemporalGATv2Layer    → child.lipschitz_bound() if present
        DetectionHead         → child.lipschitz_bound() if present
    Otherwise we fall back to a product of spectral norms and emit a warning.
    """
    L_ssm_components: list[float] = []
    L_gat_components: list[float] = []
    L_head_components: list[float] = []

    for name, child in model.named_modules():
        cls = type(child).__name__
        if cls == "SelectiveSSMBlock":
            bound = _safe_call_lipschitz(child, name)
            L_ssm_components.append(bound)
        elif cls.endswith("GATv2Layer") or cls == "TemporalGATv2Layer":
            bound = _safe_call_lipschitz(child, name)
            L_gat_components.append(bound)
        elif cls == "DetectionHead":
            bound = _safe_call_lipschitz(child, name)
            L_head_components.append(bound)

    L_ssm = math.prod(L_ssm_components) if L_ssm_components else 1.0
    L_gat = math.prod(L_gat_components) if L_gat_components else 1.0
    L_head = math.prod(L_head_components) if L_head_components else 1.0
    L_f = L_ssm * L_gat * L_head

    return {
        "L_ssm": L_ssm,
        "L_gat": L_gat,
        "L_head": L_head,
        "L_f": L_f,
        "L_ssm_components": L_ssm_components,
        "L_gat_components": L_gat_components,
        "L_head_components": L_head_components,
    }


def _safe_call_lipschitz(child: nn.Module, name: str) -> float:
    fn = getattr(child, "lipschitz_bound", None)
    if callable(fn):
        try:
            return float(fn())
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"{name}.lipschitz_bound() failed ({exc!r}); falling back to spectral product."
            )
    else:
        warnings.warn(
            f"{name} ({type(child).__name__}) has no lipschitz_bound(); "
            "using spectral-norm product as a conservative surrogate."
        )
    return _spectral_fallback(child)


# ------------------------------------------------------------------ #
# Certified radius                                                   #
# ------------------------------------------------------------------ #
@dataclass
class CertifiedRadius:
    """Margin-based certified radius (Lipschitz-classifier certificate).

    For a Lipschitz-L_f classifier f : X → R^K, a point x is robust to any
    ℓ₂ perturbation of size at most
            ε = (f_{y*}(x) − max_{k≠y*} f_k(x)) / (√2 · L_f)
    (Tsuzuku et al. 2018; cf. Yang et al. 2020).
    """

    epsilon: float
    confidence_margin: float
    class_id: int

    @classmethod
    def radius_for_logits(
        cls,
        logits: Tensor,
        true_class: int,
        L_f: float,
    ) -> "CertifiedRadius":
        if logits.ndim != 1:
            logits = logits.view(-1)
        if L_f <= 0:
            raise ValueError("L_f must be strictly positive for a certified radius.")
        true_logit = float(logits[true_class].item())
        masked = logits.clone()
        masked[true_class] = float("-inf")
        runner_up = float(masked.max().item())
        margin = true_logit - runner_up
        epsilon = max(margin, 0.0) / (math.sqrt(2.0) * L_f)
        return cls(epsilon=epsilon, confidence_margin=margin, class_id=int(true_class))


# ------------------------------------------------------------------ #
# Sanity check                                                        #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    # Toy SSM-style numbers: ρ=0.4, β=0.7, γ=1.1, κ_g=0.9 → L_ssm
    L_ssm = ssm_lipschitz(rho=0.4, beta=0.7, gamma=1.1, kappa_g=0.9)
    L_gat = gatv2_lipschitz(W_v_norm=1.2, q_norm=0.5, W_r_norm=0.8, eta_norm=0.3)
    L_f = composed_lipschitz([L_ssm, L_gat, 1.0], head_factor=1.0 / 4)
    print(f"L_ssm = {L_ssm:.4f}")
    print(f"L_gat = {L_gat:.4f}")
    print(f"L_f   = {L_f:.4f}   (4 heads)")

    torch.manual_seed(0)
    logits = torch.tensor([2.5, 0.1, 1.7, -0.3])
    rad = CertifiedRadius.radius_for_logits(logits, true_class=0, L_f=L_f)
    print(f"Certified radius for class 0: ε = {rad.epsilon:.4f} (margin={rad.confidence_margin:.3f})")
