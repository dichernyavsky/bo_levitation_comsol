"""Scalar objectives: smoke benchmark and COMSOL wrapper (same vector interface)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Callable, Protocol

import torch
from torch import Tensor

if TYPE_CHECKING:
    from magnet_bo.space import DesignSpace


class ScalarObjective(Protocol):
    """Maps one physical design vector (5,) to a scalar reward to maximize."""

    def __call__(self, x_phys: Tensor) -> Tensor:
        ...


class SmokeObjective:
    """Smooth synthetic landscape near a known optimum (no COMSOL)."""

    def __call__(self, x_phys: Tensor) -> Tensor:
        target = torch.tensor([7.5, 7.0, 0.5, 9.0, 8.5], dtype=x_phys.dtype, device=x_phys.device)
        quad = -(x_phys - target).pow(2).sum(dim=-1)
        wiggle = 0.15 * torch.sin(0.8 * math.pi * x_phys[..., 2])
        return quad + wiggle


class ComsolObjective:
    """
    Maximize last sample of Fx (or your scalar) from COMSOL via `run_one`.

    `run_one` must match COMSOL_Bayesian/comsol_interface.run_one:
        (model, combo: dict) -> (time ndarray, fx ndarray)
    """

    def __init__(
        self,
        space: DesignSpace,
        model: Any,
        run_one: Callable[..., tuple[Any, Any]],
    ) -> None:
        self._space = space
        self._model = model
        self._run_one = run_one

    def __call__(self, x_phys: Tensor) -> Tensor:
        combo = self._space.combo_from_vector(x_phys)
        _t, fx = self._run_one(self._model, combo)
        y = float(fx[-1])
        return torch.tensor(y, dtype=x_phys.dtype, device=x_phys.device)
