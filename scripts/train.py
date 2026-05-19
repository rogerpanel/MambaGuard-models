"""Train MambaGuard end-to-end.

Usage:
    python -m scripts.train --config configs/default.yaml --dataset iis3d \
        --data-root data/raw/iis3d \
        --output-dir outputs/run_$(date +%Y%m%d_%H%M%S) --seed 42

Writes ``config.yaml``, ``metrics.json``, ``train.log`` and checkpoints under
``--output-dir``.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

try:
    from omegaconf import OmegaConf  # type: ignore

    _HAS_OMEGACONF = True
except Exception:  # pragma: no cover
    _HAS_OMEGACONF = False

import yaml


def _load_config(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    if _HAS_OMEGACONF:
        return OmegaConf.to_container(OmegaConf.load(path), resolve=True)  # type: ignore[return-value]
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _save_config(cfg: dict[str, Any], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)


def _setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("mambaguard.train")
    logger.setLevel(logging.INFO)
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def _model_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    """Flatten the model section of a Hydra-style config for MambaGuardConfig."""
    model_cfg = cfg.get("model", {}) or {}
    # Strip Hydra ``_target_`` / nested namespace keys if present.
    return {k: v for k, v in model_cfg.items() if not k.startswith("_")}


def _training_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    tcfg = dict(cfg.get("training", {}) or {})
    # Map keys to TrainerConfig fields used by MambaGuardTrainer._cfg_get.
    out: dict[str, Any] = {}
    if "epochs" in tcfg:
        out["epochs"] = int(tcfg["epochs"])
    if "grad_clip" in tcfg:
        out["grad_clip"] = float(tcfg["grad_clip"])
    if "batch_size" in tcfg:
        out["batch_size"] = int(tcfg["batch_size"])
    if "amp_dtype" in tcfg:
        out["amp_dtype"] = str(tcfg["amp_dtype"])
    fl = tcfg.get("focal_loss", {}) or {}
    if "alpha" in fl:
        out["focal_alpha"] = float(fl["alpha"])
    if "gamma" in fl:
        out["focal_gamma"] = float(fl["gamma"])
    lr_reg = tcfg.get("lipschitz_reg", {}) or {}
    if "lambda_L" in lr_reg:
        out["lipschitz_lambda"] = float(lr_reg["lambda_L"])
    opt = tcfg.get("optimizer", {}) or {}
    if "lr" in opt:
        out["lr"] = float(opt["lr"])
    if "weight_decay" in opt:
        out["weight_decay"] = float(opt["weight_decay"])
    out.setdefault("num_classes", int(cfg.get("model", {}).get("num_classes", 34)))
    return out


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train MambaGuard.")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--dataset", type=str, default="iis3d")
    p.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Filesystem root containing the extracted dataset.",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=f"outputs/run_{time.strftime('%Y%m%d_%H%M%S')}",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = _setup_logger(out_dir / "train.log")
    logger.info("Loading config from %s", args.config)
    cfg = _load_config(args.config)
    cfg["dataset"] = args.dataset
    cfg["seed"] = args.seed
    _save_config(cfg, out_dir / "config.yaml")

    from mambaguard.data import (  # noqa: WPS433 - lazy
        DATASET_REGISTRY,
        ProtocolMessageDataset,
        protocol_collate,
    )
    from mambaguard.data.splits import temporal_split
    from mambaguard.models import MambaGuard
    from mambaguard.training import MambaGuardTrainer
    from mambaguard.utils.seed import set_seed

    set_seed(args.seed)
    logger.info("Building model")
    model = MambaGuard.from_config(_model_kwargs(cfg))

    if args.dataset not in DATASET_REGISTRY:
        raise KeyError(
            f"Unknown dataset '{args.dataset}'. "
            f"Known: {sorted(DATASET_REGISTRY)}"
        )
    if args.data_root is None:
        raise SystemExit(
            "--data-root is required (path to the extracted dataset root)."
        )
    loader_fn = DATASET_REGISTRY[args.dataset]
    logger.info("Loading dataset %s from %s", args.dataset, args.data_root)
    all_msgs = list(loader_fn(args.data_root, split="train"))
    train_idx, val_idx, _ = temporal_split(all_msgs, train_frac=0.7, val_frac=0.15)
    train_msgs = [all_msgs[i] for i in train_idx]
    val_msgs = [all_msgs[i] for i in val_idx]
    train_ds = ProtocolMessageDataset(train_msgs)
    val_ds = ProtocolMessageDataset(val_msgs)

    import torch
    from torch.utils.data import DataLoader

    bs = int(cfg.get("training", {}).get("batch_size", 256))
    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        collate_fn=protocol_collate,
        num_workers=int(cfg.get("training", {}).get("num_workers", 0)),
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False, collate_fn=protocol_collate
    )

    trainer_cfg = _training_cfg(cfg)
    trainer_cfg["out_dir"] = str(out_dir / "checkpoints")
    device = args.device if torch.cuda.is_available() else "cpu"
    trainer = MambaGuardTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=trainer_cfg,
        device=device,
    )

    if args.resume:
        logger.info("Resuming from %s", args.resume)
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])

    logger.info("Starting training")
    state = trainer.train()
    metrics = {
        "epoch": state.epoch,
        "global_step": state.global_step,
        "best_metric": state.best_metric,
        "best_epoch": state.best_epoch,
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, default=str)
    logger.info("Training complete. State: %s", metrics)
    return 0


if __name__ == "__main__":
    sys.exit(main())
