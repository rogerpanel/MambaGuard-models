"""AdamW optimiser builder with no-decay groups and cosine schedule with warmup."""
from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR


_NO_DECAY_SUFFIXES: tuple[str, ...] = ("bias",)
_NO_DECAY_MODULES: tuple[type, ...] = (
    nn.LayerNorm,
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.GroupNorm,
)


def _no_decay_param_ids(model: nn.Module) -> set[int]:
    ids: set[int] = set()
    for module in model.modules():
        if isinstance(module, _NO_DECAY_MODULES):
            for p in module.parameters(recurse=False):
                ids.add(id(p))
        if module.__class__.__name__ in {"RMSNorm", "LipschitzNorm"}:
            for p in module.parameters(recurse=False):
                ids.add(id(p))
    for name, p in model.named_parameters():
        if any(name.endswith(s) for s in _NO_DECAY_SUFFIXES):
            ids.add(id(p))
    return ids


def build_optimizer(
    model: nn.Module,
    lr: float = 3e-4,
    weight_decay: float = 1e-2,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
) -> Optimizer:
    """Build an AdamW optimiser with biases & norm parameters in a no-decay group."""
    no_decay_ids = _no_decay_param_ids(model)
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if id(p) in no_decay_ids or p.dim() < 2:
            no_decay.append(p)
        else:
            decay.append(p)
    groups: list[dict] = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    groups = [g for g in groups if len(g["params"]) > 0]
    return AdamW(groups, lr=lr, betas=betas, eps=eps)


def _cosine_lambda(num_steps: int, warmup_steps: int, min_ratio: float = 0.0):
    warmup_steps = max(1, int(warmup_steps))
    total = max(num_steps, warmup_steps + 1)

    def fn(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        cos = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cos

    return fn


def _linear_lambda(num_steps: int, warmup_steps: int):
    warmup_steps = max(1, int(warmup_steps))
    total = max(num_steps, warmup_steps + 1)

    def fn(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        remaining = max(0.0, float(total - step) / float(total - warmup_steps))
        return remaining

    return fn


def build_scheduler(
    optimizer: Optimizer,
    num_steps: int,
    warmup_steps: int = 500,
    kind: str = "cosine",
    min_ratio: float = 0.0,
) -> LambdaLR:
    """Return a LambdaLR with linear warmup and cosine/linear decay."""
    kind = kind.lower()
    if kind == "cosine":
        fn = _cosine_lambda(num_steps, warmup_steps, min_ratio=min_ratio)
    elif kind == "linear":
        fn = _linear_lambda(num_steps, warmup_steps)
    else:
        raise ValueError(f"unknown scheduler kind: {kind}")
    return LambdaLR(optimizer, lr_lambda=fn)
