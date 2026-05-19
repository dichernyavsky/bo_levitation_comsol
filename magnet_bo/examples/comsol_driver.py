"""
Example: wire COMSOL (mph) into `run_bo`.

Run from repository root `johannes/` so imports resolve:

    python magnet_bo/examples/comsol_driver.py

Adjust `MPH_FILE` and ensure COMSOL / mph client are available.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

COMSOL_DIR = ROOT / "COMSOL_Bayesian"
if str(COMSOL_DIR) not in sys.path:
    sys.path.insert(0, str(COMSOL_DIR))

# import comsol_interface  # noqa: E402
from comsol_interface import run_one  # noqa: E402

import mph  # noqa: E402

from magnet_bo.config import ProblemConfig  # noqa: E402
from magnet_bo.loop import run_bo  # noqa: E402
from magnet_bo.objectives import ComsolObjective  # noqa: E402
from magnet_bo.space import DesignSpace  # noqa: E402

MPH_FILE = ROOT / "COMSOL_Bayesian" / "3D_H-phi_PM-Pair_N38_ld1_z4_V3.mph"


def main() -> None:
    if not MPH_FILE.is_file():
        raise SystemExit(f"Model file not found: {MPH_FILE}")

    client = mph.start()
    try:
        model = client.load(str(MPH_FILE))
        space = DesignSpace()
        objective = ComsolObjective(space, model, run_one)
        cfg = ProblemConfig(plot_path=ROOT / "magnet_bo" / "convergence_comsol.png")
        run_bo(space, objective, cfg)
    finally:
        client.clear()


if __name__ == "__main__":
    main()
