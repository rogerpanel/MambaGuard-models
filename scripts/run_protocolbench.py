"""Run the MCP Adversary Bench (14 test cases) and report per-category rates.

Usage:
    python -m scripts.run_protocolbench --checkpoint outputs/run/best.ckpt
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run MCP Adversary Bench.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument(
        "--output-dir", type=str, default="benchmarks/results/protocolbench"
    )
    p.add_argument("--batch-size", type=int, default=64)
    return p


def _print_table(per_cat: dict[str, dict[str, float]]) -> None:
    cats = sorted(per_cat)
    width = max((len(c) for c in cats), default=12)
    print(f"{'category'.ljust(width)} | n  | detect_rate")
    print("-" * (width + 22))
    for c in cats:
        row = per_cat[c]
        print(
            f"{c.ljust(width)} | "
            f"{int(row['n']):>2} | {row['detect_rate']:.3f}"
        )


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import torch

        from mambaguard.attacks import MCPAdversaryBench
        from mambaguard.models import MambaGuard
        from mambaguard.utils.logging import get_logger
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"mambaguard import failed: {exc}") from exc

    logger = get_logger("mambaguard.protocolbench")
    logger.info("Loading checkpoint %s", args.checkpoint)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt.get("config", {})
    model = MambaGuard.from_config(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    bench = MCPAdversaryBench()
    cases = list(bench)
    logger.info("Loaded %d MCP test cases", len(cases))

    per_cat_hits: dict[str, list[int]] = defaultdict(list)
    rows: list[list[Any]] = []
    with torch.no_grad():
        for case in cases:
            batch = case.to_batch(device=device)
            logits = model(batch)
            pred = int(logits.argmax(dim=-1).flatten()[0].item())
            detected = int(pred == case.malicious_label)
            per_cat_hits[case.category].append(detected)
            rows.append([case.case_id, case.category, case.protocol, detected])

    per_cat: dict[str, dict[str, float]] = {}
    for cat, hits in per_cat_hits.items():
        per_cat[cat] = {
            "n": float(len(hits)),
            "detect_rate": sum(hits) / max(len(hits), 1),
        }

    _print_table(per_cat)

    with open(out_dir / "protocolbench_cases.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["case_id", "category", "protocol", "detected"])
        for row in rows:
            w.writerow(row)
    with open(out_dir / "protocolbench_summary.json", "w", encoding="utf-8") as fh:
        json.dump(per_cat, fh, indent=2)

    logger.info("Wrote results to %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
