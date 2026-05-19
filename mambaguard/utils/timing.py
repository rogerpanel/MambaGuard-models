"""Wall-clock and CUDA-aware timing utilities for latency reporting."""
from __future__ import annotations

import functools
import time
from contextlib import AbstractContextManager
from typing import Any, Callable, Optional, TypeVar

from .logging import get_logger

_F = TypeVar("_F", bound=Callable[..., Any])
_LOGGER = get_logger(__name__)


def _cuda_sync() -> None:
    """Call ``torch.cuda.synchronize()`` iff CUDA is available."""
    try:
        import torch  # noqa: WPS433

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:  # pragma: no cover
        pass


class Timer(AbstractContextManager["Timer"]):
    """Context manager measuring elapsed seconds with optional CUDA sync.

    Example::

        with Timer("forward") as t:
            y = model(x)
        print(t.elapsed)
    """

    def __init__(self, name: str = "block", log: bool = False, sync_cuda: bool = True) -> None:
        self.name = name
        self.log = log
        self.sync_cuda = sync_cuda
        self.elapsed: float = 0.0
        self._start: Optional[float] = None

    def __enter__(self) -> "Timer":
        if self.sync_cuda:
            _cuda_sync()
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        if self.sync_cuda:
            _cuda_sync()
        assert self._start is not None
        self.elapsed = time.perf_counter() - self._start
        if self.log:
            _LOGGER.info("%s took %.4f s", self.name, self.elapsed)


def timed(fn: _F) -> _F:
    """Decorator: log function latency at INFO level (CUDA-synced)."""

    @functools.wraps(fn)
    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        with Timer(fn.__qualname__, log=True):
            return fn(*args, **kwargs)

    return _wrapper  # type: ignore[return-value]
