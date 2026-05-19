"""Tests for evaluation metrics."""
from __future__ import annotations

import pytest


def test_compute_metrics_matches_sklearn():
    torch = pytest.importorskip("torch")
    pytest.importorskip("sklearn")
    try:
        from mambaguard.evaluation import compute_metrics
    except Exception as exc:
        pytest.skip(f"compute_metrics unavailable: {exc}")

    from sklearn.metrics import accuracy_score, f1_score

    torch.manual_seed(0)
    n, k = 200, 4
    logits = torch.randn(n, k)
    labels = torch.randint(0, k, (n,))
    preds = logits.argmax(dim=-1)

    out = compute_metrics(logits=logits, labels=labels, num_classes=k)
    assert out["accuracy"] == pytest.approx(
        accuracy_score(labels.tolist(), preds.tolist()), abs=1e-6
    )
    assert out["macro_f1"] == pytest.approx(
        f1_score(
            labels.tolist(),
            preds.tolist(),
            labels=list(range(k)),
            average="macro",
            zero_division=0,
        ),
        abs=1e-6,
    )


def test_compute_metrics_returns_per_class_lists():
    torch = pytest.importorskip("torch")
    try:
        from mambaguard.evaluation import compute_metrics
    except Exception as exc:
        pytest.skip(f"compute_metrics unavailable: {exc}")
    logits = torch.randn(50, 3)
    labels = torch.randint(0, 3, (50,))
    out = compute_metrics(logits=logits, labels=labels, num_classes=3)
    assert len(out["per_class_f1"]) == 3
    assert len(out["per_class_precision"]) == 3
    assert len(out["per_class_recall"]) == 3
