"""Convergence plot (non-interactive backend)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_convergence_plot(
    observed_y: list[float],
    best_trace: list[float],
    n_init: int,
    path: Path,
    *,
    y_label: str = "objective (maximize)",
    title: str = "BO: running best vs evaluations",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(observed_y)
    x = list(range(1, n + 1))
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(x, observed_y, "o-", ms=4, lw=1, alpha=0.75, label="observed y")
    ax.plot(x, best_trace, "-", lw=2.2, color="C1", label="best so far")
    if 0 < n_init < n:
        ax.axvline(n_init + 0.5, color="0.5", ls="--", lw=1, label="BO starts")
    ax.set_xlabel("evaluation #")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
