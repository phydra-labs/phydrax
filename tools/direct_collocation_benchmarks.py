#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _block(value: Any, /) -> None:
    for leaf in jax.tree.leaves(value):
        if isinstance(leaf, jax.Array):
            leaf.block_until_ready()


def _measure(operation, /, *, repeats: int):
    value = operation()
    _block(value)
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        value = operation()
        _block(value)
        samples.append(1.0e3 * (time.perf_counter() - started))
    return value, float(np.mean(samples)), float(np.std(samples))


def _problem():
    system = phx.dynamics.ContinuousSystem(
        lambda time, state, control, args: -0.2 * state + control,
        state_layout=phx.dynamics.StateLayout((2,)),
        input_layout=phx.dynamics.InputLayout((2,), roles="control"),
        system_id="benchmark-direct-collocation",
    )
    terminal = phx.control.BoundedTrajectoryConstraint(
        lambda trajectory, args: trajectory.final_state,
        lower=jnp.ones(2),
        upper=jnp.ones(2),
        constraint_id="benchmark-terminal",
    )
    return phx.control.TrajectoryOptimizationProblem(
        system,
        initial_state=jnp.zeros(2),
        running_cost=lambda time, state, control, args: jnp.vdot(control, control),
        trajectory_constraints=(terminal,),
        problem_id="benchmark-direct-collocation",
    )


def _record(intervals: int, repeats: int, /) -> dict[str, Any]:
    mesh = phx.discretization.TemporalMesh.uniform(
        0.0,
        1.0,
        intervals,
        role="collocation",
        mesh_id=f"benchmark-direct-mesh:{intervals}",
    )
    plan = phx.control.DirectCollocationPlan(
        mesh,
        method=phx.solver.ThetaMethod(0.5, endpoint=False),
        derivatives=phx.control.DirectCollocationDerivativePolicy(verify=False),
        plan_id=f"benchmark-direct-plan:{intervals}",
    )
    states = jnp.broadcast_to(mesh.nodes[:, None], (intervals + 1, 2))
    controls = jnp.ones((intervals, 2))
    started = time.perf_counter()
    compilation = phx.control.compile_direct_collocation(
        _problem(),
        plan,
        states,
        controls,
    )
    _block(compilation.initial_coordinates)
    compilation_ms = 1.0e3 * (time.perf_counter() - started)
    program = compilation.structured_program
    _, objective_ms, objective_std = _measure(
        lambda: program.objective(compilation.initial_coordinates, None),
        repeats=repeats,
    )
    _, constraint_ms, constraint_std = _measure(
        lambda: program.constraints(compilation.initial_coordinates, None),
        repeats=repeats,
    )
    coefficients, jacobian_ms, jacobian_std = _measure(
        lambda: program.jacobian_plan.coefficients(
            compilation.initial_coordinates,
            None,
        ),
        repeats=repeats,
    )
    direction = jnp.linspace(0.1, 1.0, program.num_variables)
    sparse_action = program.jacobian_plan.operator(
        compilation.initial_coordinates,
        None,
    ).mv(direction)
    direct_action = jax.jvp(
        lambda coordinates: program.constraints(coordinates, None),
        (compilation.initial_coordinates,),
        (direction,),
    )[1]
    maximum_action_error = float(jnp.max(jnp.abs(sparse_action - direct_action)))
    dense_entries = program.num_variables * program.num_constraints
    return {
        "intervals": intervals,
        "variables": program.num_variables,
        "constraints": program.num_constraints,
        "jacobian_nonzeros": program.jacobian_plan.nnz,
        "jacobian_density": program.jacobian_plan.nnz / dense_entries,
        "jacobian_coefficient_bytes": int(coefficients.nbytes),
        "compilation_ms": compilation_ms,
        "objective_mean_ms": objective_ms,
        "objective_std_ms": objective_std,
        "constraint_mean_ms": constraint_ms,
        "constraint_std_ms": constraint_std,
        "jacobian_mean_ms": jacobian_ms,
        "jacobian_std_ms": jacobian_std,
        "maximum_action_error": maximum_action_error,
        "passed": maximum_action_error <= 1.0e-10,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--intervals", nargs="+", type=int, default=(8, 32, 128))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/direct_collocation.json"),
    )
    arguments = parser.parse_args()
    if arguments.repeats < 1 or any(value < 1 for value in arguments.intervals):
        raise ValueError("repeats and every interval count must be positive")
    records = [
        _record(intervals, arguments.repeats) for intervals in arguments.intervals
    ]
    artifact = {
        "benchmark": "direct-collocation-sparse-scaling",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "jax": jax.__version__,
        "default_dtype": str(jnp.asarray(0.0).dtype),
        "records": records,
        "passed": all(record["passed"] for record in records),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    main()
