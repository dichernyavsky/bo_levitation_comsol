"""CLI: smoke benchmark or placeholder for COMSOL-backed run."""

from __future__ import annotations

import argparse
from pathlib import Path

from magnet_bo.config import ProblemConfig
from magnet_bo.loop import run_bo
from magnet_bo.objectives import ComsolObjective, SmokeObjective
from magnet_bo.space import DesignSpace


def main() -> None:
    p = argparse.ArgumentParser(description="BoTorch magnet design optimization")
    p.add_argument(
        "--backend",
        choices=("smoke", "comsol"),
        default="smoke",
        help="smoke: synthetic objective; comsol: requires mph + model (not wired in CLI)",
    )
    p.add_argument("--n-init", type=int, default=16)
    p.add_argument("--n-bo", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--plot",
        type=str,
        default="convergence.png",
        help="PNG path; set empty to skip plotting",
    )
    args = p.parse_args()

    space = DesignSpace()
    plot_path = Path(args.plot) if args.plot.strip() else None
    cfg = ProblemConfig(
        n_init=args.n_init,
        n_bo=args.n_bo,
        seed=args.seed,
        plot_path=plot_path,
    )

    if args.backend == "smoke":
        objective = SmokeObjective()
        run_bo(space, objective, cfg)
        return

    raise SystemExit(
        "backend=comsol: start mph client, load .mph, then call run_bo(space, "
        "ComsolObjective(space, model, run_one), cfg) from your own script "
        "(see magnet_bo.objectives.ComsolObjective)."
    )


if __name__ == "__main__":
    main()
