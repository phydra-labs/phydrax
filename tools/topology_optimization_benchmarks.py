#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp

import phydrax as phx


class _DiagonalStateSolver(phx.optim.AbstractStateSolver):
    load: jax.Array
    modulus: Callable = eqx.field(static=True)

    def __init__(self, load, modulus):
        self.load = jnp.asarray(load)
        self.modulus = modulus

    @property
    def method_id(self) -> str:
        return "benchmark-diagonal-state"

    def solve(self, problem, design, initial_state, /, *, args):
        del initial_state
        state = self.load / self.modulus(design)
        return phx.optim.StateEquationResult(
            state,
            problem.residual(state, design, args),
            phx.optim.OptimizationStatus.SUCCESS,
            phx.optim.OptimizationDiagnostics(residual_evaluations=1),
        )


def run_case(cells: int, /) -> dict[str, object]:
    centers = jnp.stack((jnp.arange(cells, dtype=float), jnp.zeros((cells,))), axis=-1)
    measures = jnp.ones((cells,))
    density_filter = phx.applications.solid_mechanics.DensityFilterPlan(
        centers,
        measures,
        1.1,
    ).prepare()
    interpolation = phx.applications.solid_mechanics.SIMPInterpolation(
        1.0,
        minimum_modulus=0.05,
        penalty=1.0,
    )
    load = jnp.linspace(2.0, 0.25, cells)
    problem = phx.applications.solid_mechanics.ComplianceTopologyProblem(
        lambda state, modulus, _: modulus * state - load,
        load,
        density_filter,
        interpolation,
        0.5,
        _DiagonalStateSolver(
            load,
            lambda density: interpolation(density_filter.apply(density)),
        ),
        problem_id=f"diagonal-compliance-{cells}",
    )
    initial_density = jnp.full((cells,), 0.5)
    initial_state = load / interpolation(density_filter.apply(initial_density))
    initial_compliance = jnp.vdot(load, initial_state)
    started = time.perf_counter()
    result = phx.applications.solid_mechanics.solve_topology_optimization(
        problem,
        initial_state,
        initial_density,
        termination=phx.optim.OptimizationTermination(
            absolute_optimality=5.0e-5,
            relative_optimality=0.0,
            maximum_steps=120,
        ),
    )
    jax.block_until_ready(result.state_design.objective)
    wall_seconds = time.perf_counter() - started
    return {
        "cells": cells,
        "status": int(result.state_design.status),
        "initial_compliance": float(initial_compliance),
        "final_compliance": float(result.state_design.objective),
        "volume_ratio": float(result.volume_ratio),
        "minimum_density": float(jnp.min(result.physical_density)),
        "maximum_density": float(jnp.max(result.physical_density)),
        "filter_constant_residual": float(density_filter.constant_residual),
        "iterations": int(result.state_design.diagnostics.iterations),
        "state_and_adjoint_solves": int(result.state_design.diagnostics.linear_solves),
        "optimality": float(result.state_design.diagnostics.final_optimality_norm),
        "wall_seconds": wall_seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/topology_optimization.json"),
    )
    parser.add_argument("--cells", type=int, nargs="+", default=(8, 32))
    args = parser.parse_args()
    payload = {"cases": [run_case(int(cells)) for cells in args.cells]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
