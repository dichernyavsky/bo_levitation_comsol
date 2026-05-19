"""Design box, linear corridor feasibility, rounding — shared by BO and COMSOL parameterization."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.quasirandom import SobolEngine

# Same physical order as the old smoke script (vector row = one design).
PARAM_KEYS: tuple[str, ...] = ("M1_W", "M2_W", "offset", "M1_H", "M2_H")


@dataclass(frozen=True)
class DesignSpace:
    allowed_left: float = -7.5
    allowed_right: float = 7.5
    w_min: float = 3.0
    w_max: float = 12.0
    h_min: float = 3.0
    h_max: float = 15.0
    step_mm: float = 0.1

    def __post_init__(self) -> None:
        off_min = self.allowed_left + self.w_min
        off_max = -off_min
        lb = torch.tensor(
            [self.w_min, self.w_min, off_min, self.h_min, self.h_min],
            dtype=torch.double,
        )
        ub = torch.tensor(
            [self.w_max, self.w_max, off_max, self.h_max, self.h_max],
            dtype=torch.double,
        )
        object.__setattr__(self, "_lb", lb)
        object.__setattr__(self, "_ub", ub)

    @property
    def lb(self) -> Tensor:
        return self._lb

    @property
    def ub(self) -> Tensor:
        return self._ub

    @property
    def dim(self) -> int:
        return int(self.lb.shape[0])

    def is_feasible(self, x: Tensor) -> Tensor:
        """x: (..., 5) physical mm; corridor on widths + offset."""
        m1w, m2w, off = x[..., 0], x[..., 1], x[..., 2]
        al = off - m1w
        ar = off + m2w
        return (self.allowed_left <= al) & (ar <= self.allowed_right)

    def round_to_step(self, x: Tensor) -> Tensor:
        s = 1.0 / self.step_mm
        lb = self.lb.to(device=x.device, dtype=x.dtype)
        ub = self.ub.to(device=x.device, dtype=x.dtype)
        return (torch.round(x * s) / s).clamp(lb, ub)

    def to_unit(self, x: Tensor) -> Tensor:
        lb = self.lb.to(x)
        ub = self.ub.to(x)
        return (x - lb) / (ub - lb)

    def from_unit(self, u: Tensor) -> Tensor:
        lb = self.lb.to(u)
        ub = self.ub.to(u)
        return u * (ub - lb) + lb

    def combo_from_vector(self, x_phys: Tensor) -> dict[str, float]:
        """One design as COMSOL-style parameter dict (mm values, no unit suffix)."""
        v = x_phys.detach().flatten().tolist()
        return {k: float(v[i]) for i, k in enumerate(PARAM_KEYS)}

    def vector_from_combo(self, combo: dict[str, float]) -> Tensor:
        row = [float(combo[k]) for k in PARAM_KEYS]
        return torch.tensor(row, dtype=torch.double)

    def sample_feasible_sobol(self, n: int, seed: int, max_total: int = 50_000) -> Tensor:
        """Feasible initial designs in physical mm, shape (n, 5)."""
        d = self.dim
        eng = SobolEngine(d, scramble=True, seed=seed)
        out: list[Tensor] = []
        drawn = 0
        while len(out) < n and drawn < max_total:
            batch = min(256, max_total - drawn)
            u = eng.draw(batch).to(dtype=torch.double)
            x = self.from_unit(u)
            x = self.round_to_step(x)
            ok = self.is_feasible(x)
            for i in range(batch):
                if bool(ok[i]) and len(out) < n:
                    out.append(x[i].clone())
            drawn += batch
        if len(out) < n:
            raise RuntimeError(f"only {len(out)} feasible Sobol points in {max_total} tries")
        return torch.stack(out, dim=0)


def fallback_feasible_unit(space: DesignSpace, engine: SobolEngine) -> Tensor:
    """Next feasible design in *unit* space, shape (1, d); advances `engine`."""
    for _ in range(20_000):
        u = engine.draw(1).to(dtype=torch.double).squeeze(0)
        x = space.round_to_step(space.from_unit(u))
        if bool(space.is_feasible(x)):
            return space.to_unit(x).unsqueeze(0)
    raise RuntimeError("Sobol fallback: no feasible point")
