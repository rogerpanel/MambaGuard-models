"""Evaluate a trained MambaGuard checkpoint on a dataset split.

Loads the checkpoint, runs inference over the requested split, and writes
``eval_metrics.json`` plus ``confusion_matrix.csv`` to ``--output-dir``.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate MambaGuard.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--output-dir", type=str, default="outputs/eval")
    return p


def _print_metrics_table(metrics: dict[str, Any]) -> None:
    rows = [(k, v) for k, v in metrics.items() if isinstance(v, (int, float))]
    width = max((len(k) for k, _ in rows), default=10)
    print(f"{'metric'.ljust(width)} | value")
    print("-" * (width + 10))
    for k, v in rows:
        val = f"{v:.4f}" if isinstance(v, float) else str(v)
        print(f"{k.ljust(width)} | {val}")


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from torch.utils.data import DataLoader

    from mambaguard.data import (  # noqa: WPS433 - lazy
        DATASET_REGISTRY,
        ProtocolMessageDataset,
        protocol_collate,
    )
    from mambaguard.evaluation import compute_metrics
    from mambaguard.models import MambaGuard
    from mambaguard.utils.logging import get_logger

    logger = get_logger("mambaguard.eval")
    logger.info("Loading checkpoint %s", args.checkpoint)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt.get("config", {})
    model = MambaGuard.from_config(cfg.get("model", cfg))
    model.load_state_dict(ckpt["model"])
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    num_classes = int(cfg.get("model", {}).get("num_classes", 34))

    if args.dataset not in DATASET_REGISTRY:
        raise KeyError(f"Unknown dataset '{args.dataset}'.")
    loader_fn = DATASET_REGISTRY[args.dataset]
    msgs = list(loader_fn(args.data_root, split=args.split))
    ds = ProtocolMessageDataset(msgs)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=protocol_collate,
    )

    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, dict):
                batch = {
                    k: (v.to(device) if hasattr(v, "to") else v)
                    for k, v in batch.items()
                }
                labels = batch["labels"]
            else:
                batch = batch.to(device)
                labels = batch.labels
            out = model(batch)
            logits = out["logits"] if isinstance(out, dict) else out
            all_logits.append(logits.detach().cpu().float())
            all_labels.append(labels.detach().cpu().long())

    if not all_logits:
        logger.warning("Empty dataset — nothing to evaluate.")
        return 0
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)

    metrics = compute_metrics(logits, labels, num_classes=num_classes)
    _print_metrics_table(metrics)
    with open(out_dir / "eval_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, default=str)

    cm = metrics.get("confusion_matrix", [])
    with open(out_dir / "confusion_matrix.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        n = len(cm)
        w.writerow(["true\\pred"] + [f"c{i}" for i in range(n)])
        for i, row in enumerate(cm):
            w.writerow([f"c{i}", *row])
    logger.info("Wrote eval_metrics.json and confusion_matrix.csv to %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
