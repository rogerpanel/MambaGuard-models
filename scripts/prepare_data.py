"""Pre-process a raw dataset into canonicalised per-window ``.pt`` shards."""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prepare a MambaGuard dataset.")
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--root", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--window-width", type=int, default=128)
    p.add_argument("--window-stride", type=int, default=64)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import torch

        from mambaguard.data import DATASET_REGISTRY, ProtocolGraph
        from mambaguard.data.canonicalize import MessageCanonicaliser
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"mambaguard import failed: {exc}") from exc

    if args.dataset not in DATASET_REGISTRY:
        raise KeyError(f"Unknown dataset '{args.dataset}'.")
    ds = DATASET_REGISTRY[args.dataset](
        split="all", config={"root": args.root, "raw": True}
    )
    canon = MessageCanonicaliser()
    graph = ProtocolGraph()

    n_messages = 0
    label_counter: Counter[int] = Counter()
    agents: set[str] = set()
    tools: set[str] = set()
    for raw in ds:
        msg = canon(raw)
        graph.add_message(msg)
        n_messages += 1
        if getattr(msg, "label", None) is not None:
            label_counter[int(msg.label)] += 1
        if getattr(msg, "src_agent", None):
            agents.add(msg.src_agent)
        if getattr(msg, "dst_agent", None):
            agents.add(msg.dst_agent)
        if getattr(msg, "tool", None):
            tools.add(msg.tool)

    print(f"Processed {n_messages} messages")
    print(f"Agents: {len(agents)}")
    print(f"Tools : {len(tools)}")
    print(f"Label distribution: {dict(label_counter)}")

    for i, window in enumerate(
        graph.windows(width=args.window_width, stride=args.window_stride)
    ):
        shard_path = out_dir / f"window_{i:06d}.pt"
        torch.save(window.to_pyg_data(), shard_path)
    print(f"Wrote {i + 1 if n_messages else 0} shards to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
