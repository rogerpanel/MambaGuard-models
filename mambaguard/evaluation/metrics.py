"""Classification metrics: macro/micro F1, per-class F1, ECE, confusion matrix."""
from __future__ import annotations

from typing import Any, Iterable

import torch


def _to_numpy(x: torch.Tensor):
    return x.detach().cpu().numpy()


def expected_calibration_error(
    probs: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 15,
) -> float:
    """ECE with equal-width bins over the predicted-class confidence."""
    conf, pred = probs.max(dim=-1)
    correct = (pred == labels).float()
    bin_edges = torch.linspace(0.0, 1.0, n_bins + 1, device=probs.device)
    n = labels.numel()
    if n == 0:
        return 0.0
    ece = 0.0
    for i in range(n_bins):
        lo, hi = float(bin_edges[i]), float(bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        m = float(mask.sum().item())
        if m == 0:
            continue
        acc_b = float(correct[mask].mean().item())
        conf_b = float(conf[mask].mean().item())
        ece += (m / n) * abs(acc_b - conf_b)
    return float(ece)


def compute_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int = 34,
) -> dict[str, Any]:
    """Return a dict with macro/micro F1, per-class F1/P/R, accuracy, ECE, confusion matrix."""
    from sklearn.metrics import (  # noqa: WPS433 - lazy import
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
    )

    probs = torch.softmax(logits.float(), dim=-1)
    preds = probs.argmax(dim=-1)
    y_true = _to_numpy(labels.long())
    y_pred = _to_numpy(preds.long())
    labels_arr = list(range(num_classes))
    macro_f1 = float(f1_score(y_true, y_pred, labels=labels_arr, average="macro", zero_division=0))
    micro_f1 = float(f1_score(y_true, y_pred, labels=labels_arr, average="micro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, labels=labels_arr, average="weighted", zero_division=0))
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_arr, average=None, zero_division=0
    )
    accuracy = float((preds == labels).float().mean().item())
    ece = expected_calibration_error(probs, labels)
    cm = confusion_matrix(y_true, y_pred, labels=labels_arr)
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "weighted_f1": weighted_f1,
        "ece": ece,
        "per_class_f1": [float(x) for x in f1],
        "per_class_precision": [float(x) for x in prec],
        "per_class_recall": [float(x) for x in rec],
        "per_class_support": [int(x) for x in support],
        "confusion_matrix": cm.tolist(),
    }


class MetricTracker:
    """Accumulate logits/labels across batches and produce a final aggregated report."""

    def __init__(self, num_classes: int = 34) -> None:
        self.num_classes = int(num_classes)
        self._logits: list[torch.Tensor] = []
        self._labels: list[torch.Tensor] = []

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        self._logits.append(logits.detach().cpu().float())
        self._labels.append(labels.detach().cpu().long())

    def reset(self) -> None:
        self._logits.clear()
        self._labels.clear()

    def __len__(self) -> int:
        return sum(t.shape[0] for t in self._labels)

    def report(self) -> dict[str, Any]:
        if not self._logits:
            return {}
        logits = torch.cat(self._logits, dim=0)
        labels = torch.cat(self._labels, dim=0)
        return compute_metrics(logits, labels, num_classes=self.num_classes)
