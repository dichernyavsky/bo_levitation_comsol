"""Problem hyperparameters (mirrors COMSOL_Bayesian/config.py intent)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProblemConfig:
    """Budgets, seeds, optional plot path. Geometry lives on `DesignSpace`."""

    n_init: int = 16
    n_bo: int = 20
    seed: int = 42
    # Stop BO if this many consecutive EI proposals round to an already-evaluated design.
    max_duplicate_streak: int = 2
    plot_path: Path | None = None
