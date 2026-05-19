"""Latency / throughput / attack-success benchmarks for MambaGuard."""
from __future__ import annotations

import statistics
import time
from typing import Any, Iterable, Mapping

import torch
import torch.nn as nn


def _cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _to_device(batch: Any, device: torch.device) -> Any:
    if isinstance(batch, dict):
        return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    return batch


def _batch_size(batch: Any) -> int:
    if isinstance(batch, dict):
        for k in ("p", "labels", "messages"):
            if k in batch and torch.is_tensor(batch[k]):
                return int(batch[k].shape[0])
        for v in batch.values():
            if torch.is_tensor(v):
                return int(v.shape[0])
    if torch.is_tensor(batch):
        return int(batch.shape[0])
    return 1


def _extract_logits(out: Any) -> torch.Tensor:
    if isinstance(out, dict):
        return out["logits"]
    return out


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((q / 100.0) * (len(s) - 1)))))
    return s[k]


class LatencyBenchmark:
    """Per-message latency benchmark with warmup, percentile and throughput stats."""

    @staticmethod
    @torch.no_grad()
    def measure(
        model: nn.Module,
        sample_batch: Mapping[str, Any],
        warmup: int = 50,
        iters: int = 500,
        device: str | torch.device | None = None,
    ) -> dict[str, float]:
        if device is None:
            device = next(model.parameters()).device
        device = torch.device(device)
        model = model.to(device)
        model.eval()
        batch = _to_device(sample_batch, device)
        bs = max(1, _batch_size(batch))
        for _ in range(warmup):
            _ = model(batch)
        _cuda_sync(device)
        times: list[float] = []
        t_total0 = time.perf_counter()
        for _ in range(iters):
            _cuda_sync(device)
            t0 = time.perf_counter()
            _ = model(batch)
            _cuda_sync(device)
            times.append((time.perf_counter() - t0) * 1000.0)
        total = max(1e-9, time.perf_counter() - t_total0)
        per_msg_ms = [t / bs for t in times]
        mean_ms = sum(per_msg_ms) / len(per_msg_ms)
        throughput = (iters * bs) / total
        return {
            "mean_latency_ms": float(mean_ms),
            "p50_latency_ms": float(_percentile(per_msg_ms, 50)),
            "p95_latency_ms": float(_percentile(per_msg_ms, 95)),
            "p99_latency_ms": float(_percentile(per_msg_ms, 99)),
            "min_latency_ms": float(min(per_msg_ms)),
            "max_latency_ms": float(max(per_msg_ms)),
            "std_latency_ms": float(statistics.pstdev(per_msg_ms)) if len(per_msg_ms) > 1 else 0.0,
            "throughput_msg_s": float(throughput),
            "iters": int(iters),
            "batch_size": int(bs),
        }


class ThroughputBenchmark:
    """Sustained throughput in messages/second over a fixed wall-clock budget."""

    @staticmethod
    @torch.no_grad()
    def measure(
        model: nn.Module,
        loader: Iterable[Mapping[str, Any]],
        duration_s: float = 10.0,
        device: str | torch.device | None = None,
    ) -> dict[str, float]:
        if device is None:
            device = next(model.parameters()).device
        device = torch.device(device)
        model = model.to(device)
        model.eval()
        n_msgs = 0
        n_batches = 0
        loader_iter = iter(loader)
        _cuda_sync(device)
        t0 = time.perf_counter()
        while True:
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                continue
            batch = _to_device(batch, device)
            _ = model(batch)
            n_msgs += _batch_size(batch)
            n_batches += 1
            if (time.perf_counter() - t0) >= duration_s:
                break
        _cuda_sync(device)
        elapsed = max(1e-9, time.perf_counter() - t0)
        return {
            "duration_s": float(elapsed),
            "messages": int(n_msgs),
            "batches": int(n_batches),
            "throughput_msg_s": float(n_msgs / elapsed),
            "throughput_Mmsg_s": float(n_msgs / elapsed / 1e6),
        }


class AttackSuccessRate:
    """Compute the fraction of adversarially perturbed samples that are mis-classified."""

    @staticmethod
    def measure(
        model: nn.Module,
        attack: Any,
        loader: Iterable[Mapping[str, Any]],
        device: str | torch.device | None = None,
        max_batches: int | None = None,
    ) -> dict[str, float]:
        if device is None:
            device = next(model.parameters()).device
        device = torch.device(device)
        model = model.to(device).eval()
        n_total = 0
        n_succ = 0
        n_clean_correct = 0
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            batch = _to_device(batch, device)
            labels = batch["labels"]
            with torch.no_grad():
                clean_logits = _extract_logits(model(batch))
                clean_preds = clean_logits.argmax(dim=-1)
                n_clean_correct += int((clean_preds == labels).sum().item())
            adv_batch = attack(model, batch, labels) if hasattr(attack, "__call__") else attack.forward(model, batch, labels)
            with torch.no_grad():
                adv_logits = _extract_logits(model(adv_batch))
                adv_preds = adv_logits.argmax(dim=-1)
            n_succ += int((adv_preds != labels).sum().item())
            n_total += int(labels.numel())
        if n_total == 0:
            return {"attack_success_rate": 0.0, "clean_accuracy": 0.0, "n_samples": 0}
        return {
            "attack_success_rate": float(n_succ / n_total),
            "clean_accuracy": float(n_clean_correct / n_total),
            "robust_accuracy": float(1.0 - n_succ / n_total),
            "n_samples": int(n_total),
        }
