"""LogEI candidate in unit box with geometry check after rounding."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.quasirandom import SobolEngine

from botorch.acquisition import LogExpectedImprovement
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf

from magnet_bo.space import DesignSpace, fallback_feasible_unit


def next_acqf_candidate(
    space: DesignSpace,
    model: SingleTaskGP,
    best_f: float,
    fallback_engine: SobolEngine,
    *,
    acqf_retries: int = 30,
    num_restarts: int = 4,
    raw_samples: int = 128,
) -> Tensor:
    """Return candidate in unit coordinates, shape (1, d)."""
    d = space.dim
    bounds_u = torch.tensor([0.0] * d + [1.0] * d, dtype=torch.double).reshape(2, d)
    acq = LogExpectedImprovement(model, best_f=torch.tensor([best_f], dtype=torch.double))
    for _ in range(acqf_retries):
        cand_u, _ = optimize_acqf(
            acq_function=acq,
            bounds=bounds_u,
            q=1,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )
        cu = cand_u.reshape(-1, d)[0]
        x = space.round_to_step(space.from_unit(cu))
        if bool(space.is_feasible(x)):
            return space.to_unit(x).unsqueeze(0)
    return fallback_feasible_unit(space, fallback_engine)
