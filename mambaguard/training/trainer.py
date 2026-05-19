"""MambaGuardTrainer: epoch loop with AMP, grad clipping, checkpointing and callbacks."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..utils.logging import get_logger
from .losses import FocalLoss, LipschitzRegularizer, combined_loss
from .optimizer import build_optimizer, build_scheduler

_LOGGER = get_logger(__name__)


@dataclass
class TrainerState:
    """Mutable training state tracked across epochs and checkpoints."""

    epoch: int = 0
    global_step: int = 0
    best_metric: float = float("-inf")
    best_epoch: int = -1
    history: list[dict[str, float]] = field(default_factory=list)


@dataclass
class TrainerConfig:
    """Default config; the trainer also accepts duck-typed cfg objects."""

    epochs: int = 30
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 1e-2
    warmup_steps: int = 500
    grad_clip: float = 1.0
    amp_dtype: str = "bfloat16"
    monitor: str = "macro_f1"
    monitor_mode: str = "max"
    out_dir: str = "outputs/checkpoints"
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    lipschitz_lambda: float = 1e-3
    log_every: int = 50
    num_classes: int = 34


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _amp_dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}.get(
        str(name).lower(), torch.bfloat16
    )


def _to_device(batch: Any, device: str | torch.device) -> Any:
    if isinstance(batch, dict):
        return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    return batch


def _extract_logits(out: Any) -> torch.Tensor:
    if isinstance(out, dict):
        return out["logits"]
    return out


def _extract_labels(batch: Any) -> torch.Tensor:
    if isinstance(batch, dict) and "labels" in batch:
        return batch["labels"]
    raise KeyError("batch must contain a 'labels' tensor")


class MambaGuardTrainer:
    """Mixed-precision training loop with Lipschitz-regularised focal classification loss."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        cfg: Any = None,
        device: str | torch.device = "cuda",
        callbacks: Optional[Sequence[Callable[..., None]]] = None,
    ) -> None:
        self.cfg = cfg if cfg is not None else TrainerConfig()
        self.device = torch.device(device if torch.cuda.is_available() or str(device) == "cpu" else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.callbacks: list[Callable[..., None]] = list(callbacks or [])
        self.state = TrainerState()

        self.epochs = int(_cfg_get(self.cfg, "epochs", 30))
        self.grad_clip = float(_cfg_get(self.cfg, "grad_clip", 1.0))
        self.amp_dtype = _amp_dtype(str(_cfg_get(self.cfg, "amp_dtype", "bfloat16")))
        self.monitor = str(_cfg_get(self.cfg, "monitor", "macro_f1"))
        self.monitor_mode = str(_cfg_get(self.cfg, "monitor_mode", "max"))
        self.out_dir = str(_cfg_get(self.cfg, "out_dir", "outputs/checkpoints"))
        self.log_every = int(_cfg_get(self.cfg, "log_every", 50))
        self.num_classes = int(_cfg_get(self.cfg, "num_classes", 34))

        self.focal = FocalLoss(
            alpha=float(_cfg_get(self.cfg, "focal_alpha", 0.25)),
            gamma=float(_cfg_get(self.cfg, "focal_gamma", 2.0)),
        ).to(self.device)
        self.lip_reg = LipschitzRegularizer(
            lambda_l=float(_cfg_get(self.cfg, "lipschitz_lambda", 1e-3))
        ).to(self.device)

        self.optimizer = build_optimizer(
            self.model,
            lr=float(_cfg_get(self.cfg, "lr", 3e-4)),
            weight_decay=float(_cfg_get(self.cfg, "weight_decay", 1e-2)),
        )
        total_steps = max(1, self.epochs * max(1, len(self.train_loader)))
        self.scheduler = build_scheduler(
            self.optimizer,
            num_steps=total_steps,
            warmup_steps=int(_cfg_get(self.cfg, "warmup_steps", 500)),
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=(self.amp_dtype == torch.float16))
        os.makedirs(self.out_dir, exist_ok=True)
        self.state.best_metric = float("-inf") if self.monitor_mode == "max" else float("inf")

    # ------------------------------------------------------------------
    def add_callback(self, cb: Callable[..., None]) -> None:
        self.callbacks.append(cb)

    def _run_callbacks(self, metrics: dict[str, float]) -> None:
        for cb in self.callbacks:
            try:
                cb(model=self.model, metrics=metrics, epoch=self.state.epoch, trainer=self)
            except TypeError:
                try:
                    cb(self.model, metrics, self.state.epoch)
                except TypeError:
                    cb(metrics)

    # ------------------------------------------------------------------
    def _step_batch(self, batch: Mapping[str, Any]) -> dict[str, float]:
        batch = _to_device(batch, self.device)
        labels = _extract_labels(batch)
        self.optimizer.zero_grad(set_to_none=True)
        amp_enabled = self.device.type == "cuda"
        with torch.amp.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=amp_enabled):
            out = self.model(batch)
            logits = _extract_logits(out)
            losses = combined_loss(logits, labels, self.model, self.lip_reg, self.focal)
            loss = losses["loss"]
        if self.amp_dtype == torch.float16:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
        self.scheduler.step()
        return {
            "loss": float(loss.detach().item()),
            "ce": float(losses["ce"].item()),
            "lip": float(losses["lip"].item()),
        }

    # ------------------------------------------------------------------
    def train(self) -> TrainerState:
        for epoch in range(self.epochs):
            self.state.epoch = epoch
            self.model.train()
            t_epoch = time.perf_counter()
            running = {"loss": 0.0, "ce": 0.0, "lip": 0.0, "n": 0}
            for it, batch in enumerate(self.train_loader):
                stats = self._step_batch(batch)
                running["loss"] += stats["loss"]
                running["ce"] += stats["ce"]
                running["lip"] += stats["lip"]
                running["n"] += 1
                self.state.global_step += 1
                if self.log_every and self.state.global_step % self.log_every == 0:
                    _LOGGER.info(
                        "step=%d epoch=%d loss=%.4f ce=%.4f lip=%.4f lr=%.2e",
                        self.state.global_step, epoch, stats["loss"], stats["ce"], stats["lip"],
                        self.optimizer.param_groups[0]["lr"],
                    )
            n = max(1, running["n"])
            train_metrics = {
                "train_loss": running["loss"] / n,
                "train_ce": running["ce"] / n,
                "train_lip": running["lip"] / n,
                "epoch_time_s": time.perf_counter() - t_epoch,
            }
            val_metrics: dict[str, float] = {}
            if self.val_loader is not None:
                val_metrics = self.evaluate(self.val_loader)
                val_metrics = {f"val_{k}" if not k.startswith("val_") else k: v for k, v in val_metrics.items()}
            metrics = {**train_metrics, **val_metrics}
            self.state.history.append(metrics)
            self._run_callbacks(metrics)
            self._maybe_checkpoint(metrics)
            _LOGGER.info("[epoch %d] %s", epoch, _fmt_metrics(metrics))
            if any(getattr(cb, "should_stop", False) for cb in self.callbacks):
                _LOGGER.info("Early stop triggered at epoch %d", epoch)
                break
        return self.state

    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict[str, float]:
        from ..evaluation.metrics import compute_metrics

        self.model.eval()
        all_logits: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []
        latencies: list[float] = []
        n_msgs = 0
        t0 = time.perf_counter()
        for batch in loader:
            batch = _to_device(batch, self.device)
            labels = _extract_labels(batch)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            ti = time.perf_counter()
            with torch.amp.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.device.type == "cuda"):
                out = self.model(batch)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - ti
            logits = _extract_logits(out).float()
            bs = logits.shape[0]
            latencies.append(1000.0 * dt / max(1, bs))
            n_msgs += bs
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
        if not all_logits:
            return {}
        logits = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0)
        metrics = compute_metrics(logits, labels, num_classes=self.num_classes)
        elapsed = max(1e-9, time.perf_counter() - t0)
        flat = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        flat["mean_latency_ms"] = float(sum(latencies) / max(1, len(latencies)))
        flat["throughput_msg_s"] = float(n_msgs / elapsed)
        return flat

    # ------------------------------------------------------------------
    def _maybe_checkpoint(self, metrics: dict[str, float]) -> None:
        last_path = os.path.join(self.out_dir, "last.pt")
        self._save(last_path, metrics)
        key = self.monitor if self.monitor in metrics else f"val_{self.monitor}"
        if key not in metrics:
            return
        value = float(metrics[key])
        improved = value > self.state.best_metric if self.monitor_mode == "max" else value < self.state.best_metric
        if improved:
            self.state.best_metric = value
            self.state.best_epoch = self.state.epoch
            self._save(os.path.join(self.out_dir, "best.pt"), metrics)
            _LOGGER.info("Saved new best (%s=%.4f) at epoch %d", key, value, self.state.epoch)

    def _save(self, path: str, metrics: dict[str, float]) -> None:
        payload = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "state": self.state.__dict__,
            "metrics": metrics,
        }
        torch.save(payload, path)

    def load(self, path: str, strict: bool = True) -> None:
        payload = torch.load(path, map_location=self.device)
        self.model.load_state_dict(payload["model"], strict=strict)
        if "optimizer" in payload:
            self.optimizer.load_state_dict(payload["optimizer"])
        if "scheduler" in payload:
            self.scheduler.load_state_dict(payload["scheduler"])


def _fmt_metrics(metrics: Mapping[str, float]) -> str:
    parts = [f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()]
    return " ".join(parts)
