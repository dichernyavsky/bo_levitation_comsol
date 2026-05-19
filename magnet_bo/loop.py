"""Main BO loop: Sobol init, GP + LogEI, duplicate guard, optional convergence plot."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.quasirandom import SobolEngine

from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood

from magnet_bo.acquisition import next_acqf_candidate
from magnet_bo.config import ProblemConfig
from magnet_bo.objectives import ScalarObjective
from magnet_bo.plotting import save_convergence_plot
from magnet_bo.space import DesignSpace


@dataclass
class BOResult:
    train_x_phys: Tensor
    train_y: Tensor
    observed_y: list[float]
    best_trace: list[float]
    best_index: int
    stop_reason: str


def _is_in_train(x: Tensor, train_x_phys: Tensor) -> Tensor:
    return (train_x_phys == x.unsqueeze(0)).all(dim=1).any()


def run_bo(
    space: DesignSpace,
    objective: ScalarObjective,
    cfg: ProblemConfig,
    *,
    device: torch.device | None = None,
) -> BOResult:
    """
    Maximize `objective(x)` over feasible discrete designs.

    `train_x_phys` rows are mm in order PARAM_KEYS; `train_y` is column of scalars.
    """
    dev = device or torch.device("cpu")
    torch.manual_seed(cfg.seed)

    fallback_engine = SobolEngine(space.dim, scramble=True, seed=cfg.seed + 31_415)

    train_x_phys = space.sample_feasible_sobol(cfg.n_init, cfg.seed).to(device=dev)
    y_rows: list[Tensor] = []
    for i in range(train_x_phys.shape[0]):
        y_rows.append(objective(train_x_phys[i]).reshape(()))
    train_y = torch.stack(y_rows, dim=0).reshape(-1, 1)

    observed_y: list[float] = []
    best_trace: list[float] = []

    print("init (physical mm) -> y", flush=True)
    for i in range(train_x_phys.shape[0]):
        yi = float(train_y[i, 0])
        print(i, space.combo_from_vector(train_x_phys[i]), yi, flush=True)
        observed_y.append(yi)
        best_trace.append(max(observed_y))

    dup_streak = 0
    stop_reason = "completed"
    for it in range(cfg.n_bo):
        train_x = space.to_unit(train_x_phys)
        model = SingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        best_f = float(train_y.max())
        cand_u = next_acqf_candidate(
            space, model, best_f, fallback_engine,
        )
        x_new = space.round_to_step(
            space.from_unit(cand_u.reshape(-1, space.dim)[0])
        )
        if bool(_is_in_train(x_new, train_x_phys)):
            dup_streak += 1
            print(
                f"BO {it + 1}/{cfg.n_bo} duplicate in dataset (streak={dup_streak}), skip",
                flush=True,
            )
            if dup_streak > cfg.max_duplicate_streak:
                print(
                    "Stopping: max consecutive duplicate proposals exceeded.",
                    flush=True,
                )
                stop_reason = "duplicate_streak"
                break
            continue

        dup_streak = 0
        y_new = objective(x_new).reshape(1, 1)
        yn = float(y_new[0, 0])

        train_x_phys = torch.cat([train_x_phys, x_new.unsqueeze(0)], dim=0)
        train_y = torch.cat([train_y, y_new], dim=0)
        observed_y.append(yn)
        best_trace.append(max(observed_y))

        print(
            f"BO {it + 1}/{cfg.n_bo} best_f={best_f:.6f} new_y={yn:.6f} "
            f"combo={space.combo_from_vector(x_new)}",
            flush=True,
        )

    j = int(train_y.argmax())
    print("\nbest:", float(train_y[j, 0]), "combo:", space.combo_from_vector(train_x_phys[j]), flush=True)

    if cfg.plot_path is not None:
        save_convergence_plot(
            observed_y,
            best_trace,
            cfg.n_init,
            cfg.plot_path,
            y_label="objective (maximize)",
        )
        print(f"\nPlot saved: {cfg.plot_path.resolve()}", flush=True)

    return BOResult(
        train_x_phys=train_x_phys,
        train_y=train_y,
        observed_y=observed_y,
        best_trace=best_trace,
        best_index=j,
        stop_reason=stop_reason,
    )
