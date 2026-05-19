"""Compute composed certified utility bound for a MambaGuard checkpoint.

Combines the three certificate components from the paper:

    E[V_ach(π̂_T, ε)] ≥ V* − L_f · ε − B · sqrt(ln|A_D| / (2T)).

Writes a ``certificate.json`` artefact summarising L_f, V*, π*_D, and the
composed lower bound.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Certify MambaGuard.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--epsilon", type=float, default=0.05)
    p.add_argument("--horizon", type=int, default=10_000)
    p.add_argument("--B", type=float, default=1.0, dest="B")
    p.add_argument("--output-dir", type=str, default="outputs/certify")
    p.add_argument(
        "--no-stackelberg",
        action="store_true",
        help="Skip the cvxpy-dependent Stackelberg step (use V*=1.0).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch

    from mambaguard.certification import (
        ATTACKER_ACTIONS,
        DEFENDER_ACTIONS,
        HedgeDefender,
        StackelbergSolver,
        composed_certificate,
        compute_lipschitz_bound,
        default_utility_matrix,
    )
    from mambaguard.models import MambaGuard

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt.get("config", {})
    model = MambaGuard.from_config(cfg.get("model", cfg))
    model.load_state_dict(ckpt["model"])
    model.eval()

    lip_report = compute_lipschitz_bound(model)
    L_f = float(lip_report.get("L_f", lip_report.get("composed", 1.0)))
    print(f"L_f = {L_f:.4f}")

    if args.no_stackelberg:
        v_star = 1.0
        pi_d = None
        attacker_br = None
    else:
        U = default_utility_matrix(B=args.B)
        solver = StackelbergSolver(
            defender_actions=list(DEFENDER_ACTIONS),
            attacker_actions=list(ATTACKER_ACTIONS),
            utility_matrix=U,
        )
        try:
            sol = solver.solve()
            v_star = float(sol.value)
            pi_d = [float(x) for x in sol.pi_D]
            attacker_br = ATTACKER_ACTIONS[sol.attacker_best_response]
            print(f"V* (Stackelberg) = {v_star:.4f}")
            print(f"π*_D            = {pi_d}")
            print(f"attacker BR     = {attacker_br}")
        except ImportError as exc:
            print(f"[skip Stackelberg: {exc}; falling back to V*=1.0]")
            v_star, pi_d, attacker_br = 1.0, None, None

    n_actions_d = len(DEFENDER_ACTIONS)
    hedge = HedgeDefender(
        actions=list(DEFENDER_ACTIONS), horizon=args.horizon, B=args.B
    )
    rng = np.random.default_rng(0)
    cum_loss = 0.0
    for _ in range(args.horizon):
        idx = hedge.sample(rng=rng)
        loss_vec = rng.uniform(0.0, args.B, size=n_actions_d)
        cum_loss += float(loss_vec[idx])
        hedge.update(loss_vec)
    avg_loss = cum_loss / max(args.horizon, 1)
    print(
        f"Hedge avg loss over {args.horizon} rounds = {avg_loss:.4f} "
        f"(theoretical avg regret bound = {hedge.average_regret_bound():.4f})"
    )

    lower = composed_certificate(
        V_star=v_star,
        L_f=L_f,
        epsilon=args.epsilon,
        B=args.B,
        T=args.horizon,
        num_actions=n_actions_d,
    )
    print(f"Composed certificate lower bound = {float(lower):.4f}")

    payload = {
        "L_f": L_f,
        "V_star": v_star,
        "pi_D": pi_d,
        "attacker_best_response": attacker_br,
        "epsilon": args.epsilon,
        "horizon": args.horizon,
        "num_actions": n_actions_d,
        "B": args.B,
        "hedge_avg_loss": avg_loss,
        "hedge_avg_regret_bound": float(hedge.average_regret_bound()),
        "certificate_lower_bound": float(lower),
    }
    with open(out_dir / "certificate.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
