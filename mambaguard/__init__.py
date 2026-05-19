"""MambaGuard: Certified Selective State-Space Detection for Multi-Protocol LLM Agent Security.

Top-level package. Heavy dependencies (mamba_ssm, torch_geometric, sentence_transformers)
are NOT imported here so the package remains importable in lightweight environments.
"""
from __future__ import annotations

__version__ = "1.0.0"

__all__ = ["__version__", "MambaGuard"]


def __getattr__(name: str):  # pragma: no cover - thin lazy import shim
    """Lazy attribute access for the top-level model name.

    Importing ``mambaguard.MambaGuard`` will trigger the (potentially heavy)
    ``mambaguard.models`` import only on demand.
    """
    if name == "MambaGuard":
        from .models import MambaGuard as _MambaGuard  # type: ignore[attr-defined]

        return _MambaGuard
    raise AttributeError(f"module 'mambaguard' has no attribute {name!r}")
