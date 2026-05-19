"""Train-time augmentations for protocol-message windows (Section 4.3).

All augmentations are pure functions over a *list* of
:class:`ProtocolMessage` objects; they are designed to be cheap so they can
run inside the DataLoader workers.
"""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Sequence

from .canonicalize import ProtocolMessage

if TYPE_CHECKING:  # pragma: no cover
    import torch


@dataclass
class MessageDropout:
    """Randomly drop messages from a window with iid probability ``p``."""

    p: float = 0.1
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.p < 1.0:
            raise ValueError("p must be in [0, 1)")
        self._rng = random.Random(self.seed)

    def __call__(self, window: Sequence[ProtocolMessage]) -> list[ProtocolMessage]:
        return [m for m in window if self._rng.random() >= self.p]


@dataclass
class LatencyJitter:
    """Add Gaussian noise (ms) to ``metadata['latency_ms']`` in-place on a copy."""

    sigma_ms: float = 5.0
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.sigma_ms < 0:
            raise ValueError("sigma_ms must be >= 0")
        self._rng = random.Random(self.seed)

    def __call__(self, window: Sequence[ProtocolMessage]) -> list[ProtocolMessage]:
        out: list[ProtocolMessage] = []
        for m in window:
            m2 = copy.copy(m)
            m2.metadata = dict(m.metadata)
            base = float(m2.metadata.get("latency_ms", 0.0))
            m2.metadata["latency_ms"] = max(0.0, base + self._rng.gauss(0.0, self.sigma_ms))
            out.append(m2)
        return out


@dataclass
class PayloadParaphrase:
    """Swap ``payload_emb`` for a pre-computed paraphrase embedding.

    ``paraphrase_bank`` maps ``msg_id`` to a tensor of shape ``(K, d_p)``
    holding K paraphrase candidates. With probability ``p`` per message we
    sample one and overwrite ``payload_emb``. Messages not present in the
    bank are passed through.
    """

    paraphrase_bank: Optional[dict[str, "torch.Tensor"]] = None
    p: float = 0.5
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.p <= 1.0:
            raise ValueError("p must be in [0, 1]")
        self._rng = random.Random(self.seed)

    def __call__(self, window: Sequence[ProtocolMessage]) -> list[ProtocolMessage]:
        if not self.paraphrase_bank:
            return list(window)
        out: list[ProtocolMessage] = []
        for m in window:
            if m.msg_id and m.msg_id in self.paraphrase_bank and self._rng.random() < self.p:
                bank = self.paraphrase_bank[m.msg_id]
                k = int(self._rng.randint(0, bank.shape[0] - 1))
                m2 = copy.copy(m)
                m2.payload_emb = bank[k]
                out.append(m2)
            else:
                out.append(m)
        return out


__all__ = ["MessageDropout", "LatencyJitter", "PayloadParaphrase"]
