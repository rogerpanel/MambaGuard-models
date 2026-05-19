"""Tests for the focal-loss implementation."""
from __future__ import annotations

import pytest


def test_focal_loss_matches_ce_when_gamma_zero():
    torch = pytest.importorskip("torch")
    try:
        from mambaguard.training import FocalLoss
    except Exception as exc:
        pytest.skip(f"FocalLoss unavailable: {exc}")

    torch.manual_seed(0)
    logits = torch.randn(8, 4)
    targets = torch.randint(0, 4, (8,))
    fl = FocalLoss(gamma=0.0, alpha=1.0)
    ce = torch.nn.functional.cross_entropy(logits, targets)
    out = fl(logits, targets)
    assert torch.allclose(out, ce, atol=1e-5)


def test_focal_loss_near_zero_on_perfect_preds():
    torch = pytest.importorskip("torch")
    try:
        from mambaguard.training import FocalLoss
    except Exception as exc:
        pytest.skip(f"FocalLoss unavailable: {exc}")

    targets = torch.tensor([0, 1, 2, 3])
    # Make logits massively favour the correct class.
    logits = torch.full((4, 4), -10.0)
    for i, t in enumerate(targets):
        logits[i, t] = 10.0
    fl = FocalLoss(gamma=2.0, alpha=1.0)
    out = float(fl(logits, targets))
    assert out < 1e-4
