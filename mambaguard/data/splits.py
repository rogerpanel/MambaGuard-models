"""Temporal splitting utilities (Sarhan et al. 2022).

We strictly forbid random shuffling here: doing so leaks future information
into the training set for intrusion-detection benchmarks, which inflates
reported scores (see Sarhan et al., "Towards a Standard Feature Set of NIDS
Datasets", 2022 and Pendlebury et al. "TESSERACT", USENIX SEC 2019).
"""
from __future__ import annotations

from typing import Sequence

from .canonicalize import ProtocolMessage


def temporal_split(
    messages: Sequence[ProtocolMessage],
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> dict[str, list[int]]:
    """Return ``{"train": [...], "val": [...], "test": [...]}`` indices.

    Indices are sorted by ``ProtocolMessage.t_m`` ascending; the first
    ``train_frac`` fall in train, the next ``val_frac`` in val, the remainder
    in test. We do NOT shuffle: temporal order is the split criterion.
    """
    if not (0.0 < train_frac < 1.0 and 0.0 <= val_frac < 1.0):
        raise ValueError("train_frac in (0,1) and val_frac in [0,1) required")
    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be < 1.0 to leave a test split")

    n = len(messages)
    if n == 0:
        return {"train": [], "val": [], "test": []}

    order = sorted(range(n), key=lambda i: messages[i].t_m)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    return {
        "train": order[:n_train],
        "val": order[n_train : n_train + n_val],
        "test": order[n_train + n_val :],
    }


__all__ = ["temporal_split"]
