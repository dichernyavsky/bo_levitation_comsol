# COMSOL-Based Optimization of Superconducting Levitation Systems

A workflow for optimizing superconducting magnetic levitation configurations using COMSOL simulations and Bayesian optimization methods.

Currently, the project focuses on a two-magnet configuration interacting with a superconducting body. The current optimization target is a scalar force-related quantity evaluated from the COMSOL model:

```text
intop1(Jy*mfh.Bz-mfh.By*Jz)
```

which is currently treated as the objective value, with the last simulated value used as the scalar reward.

## Design Parameters

The current parameterization follows the older COMSOL script and uses five geometric design variables:

- `M1_W` — width of magnet 1
- `M2_W` — width of magnet 2
- `M1_H` — height of magnet 1
- `M2_H` — height of magnet 2
- `offset` — relative lateral shift / joint position between magnets

The basic search ranges are:

- magnet widths: `3–12 mm`
- magnet heights: `3–15 mm`
- allowed total corridor: approximately `[-7.5, 7.5] mm`
- offset range derived from the corridor constraints

Some physical/model parameters are fixed at this stage, including:

- `cooling_gap = 4 mm`
- `M_length = 30 mm`
- `Ec = 1e-4 V/m`

## Current Workflow

The repository contains two related parts:

1. `COMSOL_Bayesian/`  
   A COMSOL-facing prototype that starts an `mph` client, loads the COMSOL model, sets parameters, runs the simulation, evaluates the force expression, and saves results.

2. `magnet_bo/`  
   A cleaner Bayesian optimization module with:
   - Sobol initialization,
   - Gaussian Process surrogate model,
   - Log Expected Improvement acquisition function,
   - feasibility checks for geometric constraints,
   - rounding of parameters to discrete steps,
   - optional convergence plotting.

## Optimization Goal

At the current stage, the workflow is aimed at maximizing the final simulated force value for feasible magnet geometries. M

