"""Training callbacks: early stopping, Lipschitz logging, certified radius logging."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch

from ..utils.logging import get_logger


_LOGGER = get_logger(__name__)


class EarlyStopping:
    """Stop training when ``monitor`` fails to improve for ``patience`` epochs."""

    def __init__(
        self,
        patience: int = 5,
        monitor: str = "val_macro_f1",
        mode: str = "max",
        min_delta: float = 0.0,
    ) -> None:
        if mode not in {"max", "min"}:
            raise ValueError(f"mode must be 'max' or 'min'; got {mode}")
        self.patience = int(patience)
        self.monitor = monitor
        self.mode = mode
        self.min_delta = float(min_delta)
        self.best: Optional[float] = None
        self.counter: int = 0
        self.should_stop: bool = False

    def _is_better(self, value: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "max":
            return value > self.best + self.min_delta
        return value < self.best - self.min_delta

    def __call__(self, metrics: dict[str, Any]) -> bool:
        if self.monitor not in metrics:
            return self.should_stop
        value = float(metrics[self.monitor])
        if self._is_better(value):
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                _LOGGER.info("EarlyStopping: triggered after %d stagnant epochs", self.counter)
        return self.should_stop


class LipschitzMonitor:
    """Log the composed model Lipschitz bound L_f once per epoch."""

    def __init__(self, log_every: int = 1) -> None:
        self.log_every = int(log_every)
        self.history: list[float] = []

    def __call__(self, model: torch.nn.Module, metrics: dict[str, Any], epoch: int) -> None:
        if epoch % self.log_every != 0:
            return
        try:
            from ..certification import compute_lipschitz_bound

            bound = float(compute_lipschitz_bound(model))
        except Exception as exc:  # pragma: no cover
            _LOGGER.warning("LipschitzMonitor failed: %s", exc)
            return
        self.history.append(bound)
        metrics["lipschitz_bound"] = bound
        _LOGGER.info("[epoch %d] Lipschitz bound L_f = %.4f", epoch, bound)


@dataclass
class _RadiusRecord:
    epoch: int
    mean_radius: float
    epsilon: float


class CertifiedRadiusMonitor:
    """Track the certified radius rho = (logit_margin) / L_f on a fixed subset."""

    def __init__(
        self,
        epsilon: float = 0.05,
        subset_size: int = 256,
        device: str = "cuda",
    ) -> None:
        self.epsilon = float(epsilon)
        self.subset_size = int(subset_size)
        self.device = device
        self.history: list[_RadiusRecord] = []
        self._subset: Optional[list[Any]] = None

    def _prepare_subset(self, loader: Any) -> list[Any]:
        batches: list[Any] = []
        seen = 0
        for batch in loader:
            batches.append(batch)
            seen += _batch_size(batch)
            if seen >= self.subset_size:
                break
        return batches

    def __call__(
        self,
        model: torch.nn.Module,
        loader: Any,
        metrics: dict[str, Any],
        epoch: int,
    ) -> None:
        if loader is None:
            return
        if self._subset is None:
            self._subset = self._prepare_subset(loader)
        try:
            from ..certification import compute_lipschitz_bound

            lf = float(compute_lipschitz_bound(model))
        except Exception:  # pragma: no cover
            lf = 1.0
        lf = max(lf, 1e-6)
        model.eval()
        radii: list[float] = []
        with torch.no_grad():
            for batch in self._subset:
                batch = _to_device(batch, self.device)
                logits = _forward(model, batch)
                if logits is None:
                    continue
                top2 = torch.topk(logits, k=2, dim=-1).values
                margin = (top2[:, 0] - top2[:, 1]).clamp(min=0.0)
                rho = margin / lf
                radii.extend(rho.detach().cpu().tolist())
        if not radii:
            return
        mean_radius = float(sum(radii) / len(radii))
        certified_frac = float(sum(r >= self.epsilon for r in radii) / len(radii))
        self.history.append(_RadiusRecord(epoch, mean_radius, self.epsilon))
        metrics["certified_radius_mean"] = mean_radius
        metrics["certified_at_eps"] = certified_frac
        _LOGGER.info(
            "[epoch %d] mean certified radius=%.4f, frac>=eps(%.3f)=%.3f",
            epoch,
            mean_radius,
            self.epsilon,
            certified_frac,
        )


def _batch_size(batch: Any) -> int:
    if isinstance(batch, dict):
        for v in batch.values():
            if torch.is_tensor(v):
                return int(v.shape[0])
    if torch.is_tensor(batch):
        return int(batch.shape[0])
    return 1


def _to_device(batch: Any, device: str) -> Any:
    if isinstance(batch, dict):
        return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    if torch.is_tensor(batch):
        return batch.to(device)
    return batch


def _forward(model: torch.nn.Module, batch: Any) -> Optional[torch.Tensor]:
    try:
        out = model(batch) if isinstance(batch, dict) else model(batch)
    except Exception:  # pragma: no cover
        return None
    if isinstance(out, dict) and "logits" in out:
        return out["logits"]
    if torch.is_tensor(out):
        return out
    return None
