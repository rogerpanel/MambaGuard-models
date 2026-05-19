"""Measure latency and throughput of a MambaGuard checkpoint."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Benchmark MambaGuard latency/throughput.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-warmup", type=int, default=50)
    p.add_argument("--n-iters", type=int, default=500)
    p.add_argument(
        "--output", type=str, default="benchmarks/results/latency.csv"
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)

    try:
        import torch

        from mambaguard.evaluation import LatencyBenchmark, ThroughputBenchmark
        from mambaguard.models import MambaGuard
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"mambaguard import failed: {exc}") from exc

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt.get("config", {})
    model = MambaGuard.from_config(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    lat = LatencyBenchmark(model=model, batch_size=args.batch_size, device=device)
    lat_stats = lat.run(n_warmup=args.n_warmup, n_iters=args.n_iters)

    thr = ThroughputBenchmark(model=model, batch_size=args.batch_size, device=device)
    thr_stats = thr.run(duration_s=10.0)

    print(f"p50 latency : {lat_stats['p50_ms']:.3f} ms")
    print(f"p95 latency : {lat_stats['p95_ms']:.3f} ms")
    print(f"p99 latency : {lat_stats['p99_ms']:.3f} ms")
    print(f"throughput  : {thr_stats['msgs_per_sec'] / 1e6:.3f} M msg/s")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["batch_size", "p50_ms", "p95_ms", "p99_ms", "throughput_mmsg_s"]
        )
        w.writerow(
            [
                args.batch_size,
                f"{lat_stats['p50_ms']:.4f}",
                f"{lat_stats['p95_ms']:.4f}",
                f"{lat_stats['p99_ms']:.4f}",
                f"{thr_stats['msgs_per_sec'] / 1e6:.4f}",
            ]
        )
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
