"""AblationRunner: toggle architectural components and tabulate the resulting metrics."""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Optional

import torch
import torch.nn as nn

from ..utils.logging import get_logger

_LOGGER = get_logger(__name__)


ABLATION_KEYS: tuple[str, ...] = (
    "no_lipschitz",
    "no_gatv2",
    "no_stackelberg",
    "softmax_instead_of_ssm",
)


@dataclass
class AblationResult:
    """One ablation row with its metric payload."""

    name: str
    metrics: dict[str, float] = field(default_factory=dict)


def _disable_gatv2(model: nn.Module) -> None:
    if hasattr(model, "gat_layers"):
        model.gat_layers = nn.ModuleList()


def _disable_lipschitz(trainer_cfg: dict[str, Any]) -> None:
    trainer_cfg["lipschitz_lambda"] = 0.0


def _disable_stackelberg(model: nn.Module) -> None:
    for attr in ("stackelberg", "stackelberg_controller", "controller"):
        if hasattr(model, attr):
            setattr(model, attr, None)


class _SoftmaxAttention(nn.Module):
    """Drop-in replacement for SelectiveSSMBlock: single-head softmax attention."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.scale = d_model ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.q(x), self.k(x), self.v(x)
        attn = torch.softmax(q @ k.transpose(-1, -2) * self.scale, dim=-1)
        return self.out(attn @ v)

    def lipschitz_bound(self) -> float:
        return float("inf")


def _replace_ssm_with_softmax(model: nn.Module) -> None:
    if not hasattr(model, "blocks"):
        return
    d_model = int(getattr(model.cfg, "d_model", 256)) if hasattr(model, "cfg") else 256
    model.blocks = nn.ModuleList([_SoftmaxAttention(d_model) for _ in model.blocks])


class AblationRunner:
    """Run the four paper ablations and aggregate their evaluation metrics into a CSV."""

    DEFAULT_ABLATIONS: tuple[str, ...] = ABLATION_KEYS

    def __init__(
        self,
        model_factory: Callable[[], nn.Module],
        train_fn: Callable[[nn.Module, dict[str, Any]], None],
        eval_fn: Callable[[nn.Module], dict[str, float]],
        out_csv: str = "outputs/ablations.csv",
        base_cfg: Optional[dict[str, Any]] = None,
    ) -> None:
        self.model_factory = model_factory
        self.train_fn = train_fn
        self.eval_fn = eval_fn
        self.out_csv = out_csv
        self.base_cfg = dict(base_cfg or {})
        self.results: list[AblationResult] = []

    # ------------------------------------------------------------------
    def _build_variant(self, name: str) -> tuple[nn.Module, dict[str, Any]]:
        cfg = dict(self.base_cfg)
        model = self.model_factory()
        if name == "no_lipschitz":
            _disable_lipschitz(cfg)
        elif name == "no_gatv2":
            _disable_gatv2(model)
        elif name == "no_stackelberg":
            _disable_stackelberg(model)
        elif name == "softmax_instead_of_ssm":
            _replace_ssm_with_softmax(model)
        elif name == "full":
            pass
        else:
            raise ValueError(f"unknown ablation: {name}")
        return model, cfg

    # ------------------------------------------------------------------
    def run(self, ablations: Optional[Iterable[str]] = None, include_full: bool = True) -> list[AblationResult]:
        names = list(ablations) if ablations is not None else list(self.DEFAULT_ABLATIONS)
        if include_full and "full" not in names:
            names = ["full"] + names
        self.results.clear()
        for name in names:
            _LOGGER.info("[ablation] running: %s", name)
            model, cfg = self._build_variant(name)
            try:
                self.train_fn(model, cfg)
                metrics = self.eval_fn(model)
            except Exception as exc:  # pragma: no cover
                _LOGGER.warning("[ablation %s] failed: %s", name, exc)
                metrics = {"error": float("nan")}
            self.results.append(AblationResult(name=name, metrics=dict(metrics)))
        self._write_csv()
        return self.results

    # ------------------------------------------------------------------
    def _write_csv(self) -> None:
        if not self.results:
            return
        os.makedirs(os.path.dirname(self.out_csv) or ".", exist_ok=True)
        keys: set[str] = set()
        for r in self.results:
            keys.update(r.metrics.keys())
        cols = ["ablation", *sorted(keys)]
        with open(self.out_csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(cols)
            for r in self.results:
                row = [r.name] + [r.metrics.get(k, "") for k in cols[1:]]
                writer.writerow(row)
        _LOGGER.info("Ablation CSV written to %s", self.out_csv)
