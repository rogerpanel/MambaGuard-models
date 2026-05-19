"""Shared pytest fixtures for MambaGuard tests."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture(scope="session")
def tiny_model_kwargs() -> dict[str, Any]:
    """Tiny MambaGuardConfig kwargs that run comfortably on CPU."""
    return {
        "d_p": 16,
        "d_mu": 8,
        "d_model": 32,
        "d_state": 4,
        "expand": 2,
        "kernel_size": 2,
        "n_blocks": 2,
        "n_gat_layers": 1,
        "gat_heads": 2,
        "edge_dim": 8,
        "d_T": 8,
        "num_classes": 4,
        "dropout": 0.0,
    }


@pytest.fixture(scope="session")
def tiny_config(tiny_model_kwargs) -> dict[str, Any]:
    """Nested training-style config (model + training + certification)."""
    return {
        "model": tiny_model_kwargs,
        "training": {
            "batch_size": 4,
            "epochs": 1,
            "lr": 1e-3,
            "weight_decay": 0.0,
            "optimizer": "adamw",
            "scheduler": "cosine",
            "focal_gamma": 0.0,
        },
        "certification": {"epsilon": 0.05, "B": 1.0},
    }


@pytest.fixture
def tiny_model(tiny_model_kwargs):
    """Build a tiny MambaGuard if importable; skip otherwise."""
    pytest.importorskip("torch")
    try:
        from mambaguard.models import MambaGuard
    except Exception as exc:
        pytest.skip(f"MambaGuard unavailable: {exc}")
    return MambaGuard.from_config(tiny_model_kwargs)


@pytest.fixture
def tiny_batch(tiny_model_kwargs):
    """Synthetic 8-message graph batch matching the ProtocolGraph contract."""
    torch = pytest.importorskip("torch")
    n_msgs = 8
    d_p = tiny_model_kwargs["d_p"]
    d_mu = tiny_model_kwargs["d_mu"]
    edge_dim = tiny_model_kwargs["edge_dim"]
    n_classes = tiny_model_kwargs["num_classes"]
    return {
        "p": torch.randn(n_msgs, d_p),
        "mu": torch.randn(n_msgs, d_mu),
        "x": torch.randn(n_msgs, d_p + d_mu),
        "edge_index": torch.tensor(
            [[0, 1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, 7]], dtype=torch.long
        ),
        "edge_attr": torch.randn(7, edge_dim),
        "edge_time": torch.arange(7, dtype=torch.float32),
        "edge_type": torch.zeros(7, dtype=torch.long),
        "labels": torch.randint(0, n_classes, (n_msgs,)),
        "batch": torch.zeros(n_msgs, dtype=torch.long),
    }


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    out = tmp_path / "out"
    out.mkdir(parents=True, exist_ok=True)
    return out


@pytest.fixture(autouse=True)
def _deterministic_env(monkeypatch):
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    yield
