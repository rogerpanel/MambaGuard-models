"""Deterministic seeding utilities.

Cross-references the reproducibility appendix of the MambaGuard paper.
"""
from __future__ import annotations

import os
import random
from typing import Final

_DEFAULT_SEED: Final[int] = 42


def set_seed(seed: int = _DEFAULT_SEED) -> None:
    """Seed Python ``random``, NumPy, and PyTorch (CPU + CUDA).

    Also flips :pydata:`torch.backends.cudnn.deterministic` to ``True`` and
    disables benchmark mode for repeatable convolutional kernels.
    Safe to call when optional dependencies are missing.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np  # noqa: WPS433 (intentional lazy import)

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - numpy is a hard dep but be safe
        pass

    try:
        import torch  # noqa: WPS433

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
        torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
    except ImportError:  # pragma: no cover
        pass
