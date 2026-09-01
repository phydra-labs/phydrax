#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp

import phydrax as phx


def _state_solver():
    def solve(problem, design, initial_state, args):
        del args
        zero = jax.tree.map(jnp.zeros_like, initial_state)
        one = jax.tree.map(jnp.ones_like, initial_state)
        offset = problem.residual(zero, design)
        slope = jax.tree.map(
            lambda at_one, at_zero: at_one - at_zero,
            problem.residual(one, design),
            offset,
        )
        state = jax.tree.map(
            lambda at_zero, diagonal: -at_zero / diagonal,
            offset,
            slope,
        )
        return phx.applications.solid_mechanics.MechanicsStateCandidate(
            state,
            diagnostics=phx.optim.OptimizationDiagnostics(residual_evaluations=2),
        )

    return phx.applications.solid_mechanics.FiniteElementStateSolver(
        solve,
        solver_id="benchmark-diagonal-fe",
    )


def _problem(cells: int):
    centers = jnp.stack((jnp.arange(cells, dtype=float), jnp.zeros((cells,))), axis=-1)
    measures = jnp.ones((cells,))
    prepared = phx.optim.DensityTransformPlan(
        phx.optim.ConicDensityFilterPlan(
            centers,
            1.1,
            jnp.ones((cells,), dtype=bool),
            None,
            measures,
        ),
        phx.optim.TanhDensityProjectionPlan(jnp.asarray(0.5)),
    ).prepare()
    transform = phx.applications.solid_mechanics.DensityTransform(prepared, beta=1.0)
    interpolation = phx.applications.solid_mechanics.MaterialInterpolation(
        1.0,
        minimum=0.05,
        penalty=1.0,
    )
    load = jnp.linspace(2.0, 0.25, cells)
    problem = phx.applications.solid_mechanics.TopologyMechanicsProblem(
        lambda state, modulus, case, args: modulus * state - case.load,
        (
            phx.applications.solid_mechanics.LoadCase(
                load,
                case_id="diagonal-compliance",
            ),
        ),
        transform,
        interpolation,
        0.5,
        _state_solver(),
        acceptance_policy=phx.optim.StateAcceptancePolicy(
            state_relative_tolerance=1.0e-9,
            state_absolute_tolerance=1.0e-10,
            adjoint_relative_tolerance=1.0e-7,
            adjoint_absolute_tolerance=1.0e-10,
        ),
        problem_id=f"diagonal-compliance-{cells}",
    )
    return problem, load


def run_case(cells: int, /) -> dict[str, object]:
    problem, load = _problem(cells)
    initial_density = jnp.full((cells,), 0.5)
    initial_material = problem.material_interpolation(
        problem.density_transform.apply(initial_density)
    )
    initial_state = load / initial_material
    initial_compliance = jnp.vdot(load, initial_state)
    started = time.perf_counter()
    result = phx.applications.solid_mechanics.solve_topology_optimization(
        problem,
        (jnp.zeros((cells,)),),
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
        "successful": bool(result.successful),
        "initial_compliance": float(initial_compliance),
        "final_compliance": float(result.state_design.objective),
        "volume_ratio": float(result.volume_ratio),
        "minimum_density": float(jnp.min(result.physical_density)),
        "maximum_density": float(jnp.max(result.physical_density)),
        "iterations": int(result.state_design.diagnostics.iterations),
        "state_and_adjoint_solves": int(result.state_design.diagnostics.linear_solves),
        "optimality": float(result.state_design.diagnostics.final_optimality_norm),
        "state_accepted": bool(result.state_design.state_acceptance.accepted),
        "adjoint_accepted": bool(result.state_design.adjoint_acceptance.accepted),
        "state_residual": float(result.state_design.state_acceptance.residual_norm),
        "state_threshold": float(result.state_design.state_acceptance.threshold),
        "adjoint_defect": float(
            result.state_design.adjoint_acceptance.transpose_defect_norm
        ),
        "adjoint_threshold": float(result.state_design.adjoint_acceptance.threshold),
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
