"""Logging helpers with an optional Weights & Biases integration."""
from __future__ import annotations

import logging
import sys
from typing import Any, Optional

_FORMAT: str = "[%(asctime)s] %(levelname)s %(name)s :: %(message)s"
_DATEFMT: str = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a logger with a consistent stream formatter.

    Repeat calls return the same logger without duplicating handlers.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(level)
    return logger


def init_wandb(
    project: str,
    config: Optional[dict[str, Any]] = None,
    run_name: Optional[str] = None,
    **kwargs: Any,
) -> Optional[Any]:
    """Initialise a Weights & Biases run if ``wandb`` is importable.

    Returns the run handle or ``None`` if W&B is unavailable / disabled.
    """
    try:
        import wandb  # type: ignore[import-not-found]
    except ImportError:
        get_logger(__name__).warning("wandb not installed; skipping init_wandb()")
        return None

    return wandb.init(project=project, name=run_name, config=config or {}, **kwargs)
