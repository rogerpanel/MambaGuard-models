"""Baseline model registry for the MambaGuard headline table.

Each baseline implements ``forward(batch) -> logits`` and is keyed by the
short name used in the paper's tables.
"""
from __future__ import annotations

from typing import Any, Callable

BASELINE_REGISTRY: dict[str, Callable[..., Any]] = {}


def _register(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def deco(cls: Callable[..., Any]) -> Callable[..., Any]:
        BASELINE_REGISTRY[name] = cls
        return cls

    return deco


def _safe_import() -> None:
    """Populate the registry, tolerating missing optional deps."""
    try:
        from .e_graphsage import EGraphSAGEBaseline

        BASELINE_REGISTRY.setdefault("e_graphsage", EGraphSAGEBaseline)
    except Exception:
        pass
    try:
        from .anomal_e import AnomalEBaseline

        BASELINE_REGISTRY.setdefault("anomal_e", AnomalEBaseline)
    except Exception:
        pass
    try:
        from .rtids import RTIDSBaseline

        BASELINE_REGISTRY.setdefault("rtids", RTIDSBaseline)
    except Exception:
        pass
    try:
        from .ids_int import IDSINTBaseline

        BASELINE_REGISTRY.setdefault("ids_int", IDSINTBaseline)
    except Exception:
        pass
    try:
        from .netmamba import NetMambaBaseline

        BASELINE_REGISTRY.setdefault("netmamba", NetMambaBaseline)
    except Exception:
        pass
    try:
        from .securitybert import SecurityBERTBaseline

        BASELINE_REGISTRY.setdefault("securitybert", SecurityBERTBaseline)
    except Exception:
        pass
    try:
        from .agentarmor import AgentArmorBaseline

        BASELINE_REGISTRY.setdefault("agentarmor", AgentArmorBaseline)
    except Exception:
        pass
    try:
        from .clawguard import ClawGuardBaseline

        BASELINE_REGISTRY.setdefault("clawguard", ClawGuardBaseline)
    except Exception:
        pass


_safe_import()

__all__ = ["BASELINE_REGISTRY"]
