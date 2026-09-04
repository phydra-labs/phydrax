#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Benchmark bounded-grid HJB reference work and its nested refinement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp

import phydrax as phx
from benchmarks._io import write_json_atomic
from benchmarks._runtime import (
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_repeated,
)
from phydrax._fingerprint import array_tree_fingerprint


def _zero_drift(time, state, action, args):
    del time, state, action, args
    return 0.0


def _constant_diffusion(time, state, action, args):
    del time, state, action
    return args["sigma"]


def _quadratic_action_cost(time, state, action, args):
    del time, state, args
    return action * action


def _problem(
    spatial_points: int,
    time_points: int,
    action_count: int,
    /,
):
    if action_count < 1 or action_count % 2 == 0:
        raise ValueError("action_count must be a positive odd integer")
    terminal_time = 0.2
    sigma = 0.05
    spatial_grid = phx.control.stochastic.BoundedUniformGrid1D(-1.0, 1.0, spatial_points)
    time_grid = phx.dynamics.TimeGrid(
        jnp.linspace(0.0, terminal_time, time_points),
        time_id=(f"benchmark-hjb:x{spatial_points}:t{time_points}:a{action_count}"),
    )
    actions = (
        jnp.zeros((1,)) if action_count == 1 else jnp.linspace(-1.0, 1.0, action_count)
    )
    points = spatial_grid.points
    times = time_grid.times
    terminal = points * points
    boundary_trace = 1.0 + sigma * sigma * (terminal_time - times)
    boundary = jnp.stack((boundary_trace, boundary_trace), axis=-1)
    problem = phx.control.stochastic.DiscreteHJBProblem(
        spatial_grid,
        time_grid,
        actions,
        terminal,
        boundary,
        _zero_drift,
        _constant_diffusion,
        _quadratic_action_cost,
        args={"sigma": sigma},
        problem_id=time_grid.time_id,
    )
    return problem, terminal_time, sigma


def _certificates(result, terminal_time: float, sigma: float) -> dict[str, Any]:
    coarse = result.result
    coarse_expected = (
        coarse.spatial_grid.points[None, :] ** 2
        + sigma * sigma * (terminal_time - coarse.time_grid.times)[:, None]
    )
    refined_expected = (
        result.refined_spatial_grid.points[None, :] ** 2
        + sigma * sigma * (terminal_time - result.refined_time_grid.times)[:, None]
    )
    evidence = coarse.evidence
    return {
        "claim": "declared-bounded-grid-discrete-hjb-reference-only",
        "scope": evidence.scope,
        "method": evidence.method,
        "successful": bool(coarse.successful),
        "status": int(coarse.status),
        "status_label": coarse.status_label,
        "maximum_analytic_coarse_value_defect": float(
            jnp.max(jnp.abs(coarse.values - coarse_expected))
        ),
        "maximum_analytic_refined_value_defect": float(
            jnp.max(jnp.abs(result.refined_values - refined_expected))
        ),
        "maximum_boundary_residual": float(evidence.maximum_boundary_residual),
        "maximum_terminal_residual": float(evidence.maximum_terminal_residual),
        "maximum_operator_residual": float(evidence.maximum_operator_residual),
        "maximum_action_minimum_residual": float(
            evidence.maximum_action_minimum_residual
        ),
        "maximum_refinement_difference": float(evidence.maximum_refinement_difference),
        "refinement_threshold": float(evidence.refinement_threshold),
        "maximum_courant_number": float(evidence.maximum_courant_number),
        "minimum_monotonicity_margin": float(evidence.minimum_monotonicity_margin),
        "gates": {
            "finite": bool(evidence.finite),
            "boundary": bool(evidence.boundary_passed),
            "terminal": bool(evidence.terminal_passed),
            "operator": bool(evidence.operator_passed),
            "action_minimum": bool(evidence.action_minimum_passed),
            "refinement": bool(evidence.refinement_passed),
        },
        "output_fingerprint": array_tree_fingerprint(result),
        "not_claimed": [
            "continuum convergence",
            "viscosity-solution uniqueness",
            "global optimality outside the declared finite action and state grids",
        ],
    }


def _case(
    name: str,
    spatial_points: int,
    time_points: int,
    action_count: int,
    /,
    *,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    problem, terminal_time, sigma = _problem(spatial_points, time_points, action_count)
    fingerprint_inputs = (
        problem.spatial_grid.points,
        problem.time_grid.times,
        problem.actions,
        problem.terminal_values,
        problem.boundary_values,
        problem.args,
    )
    result, execution = measure_repeated(
        lambda: phx.control.stochastic.refine_discrete_hjb_reference(problem),
        warmup=warmup,
        repeats=repeats,
    )
    unavailable = compiler_evidence(
        None,
        None,
        source="host-numpy-reference-solver",
        unavailable_reason=(
            "the bounded-grid reference deliberately executes host NumPy loops "
            "and has no lowered device executable"
        ),
    )
    refined_spatial_points = result.refined_spatial_grid.num_points
    refined_time_points = result.refined_time_grid.num_times
    coarse_sites = (time_points - 1) * (spatial_points - 2) * action_count
    refined_sites = (
        (refined_time_points - 1) * (refined_spatial_points - 2) * action_count
    )
    return {
        "name": name,
        "dimensions": {
            "spatial_points": spatial_points,
            "time_points": time_points,
            "actions": action_count,
            "refined_spatial_points": refined_spatial_points,
            "refined_time_points": refined_time_points,
            "refinement_spatial_factor": 2,
            "refinement_time_step_factor": 4,
        },
        "dtype": str(problem.terminal_values.dtype),
        "input_fingerprint": array_tree_fingerprint(fingerprint_inputs),
        "lower": {
            "seconds": None,
            "scope": "not applicable to host NumPy reference solver",
        },
        "compile": {
            "seconds": None,
            "scope": "not applicable to host NumPy reference solver",
        },
        "run": {
            **execution.to_milliseconds_dict(),
            "scope": "coarse solve, nested refined solve, and common-grid evidence",
        },
        "memory": {
            "logical_input_bytes": logical_array_bytes(fingerprint_inputs),
            "logical_output_bytes": logical_array_bytes(result),
            "compiler_argument_bytes": unavailable.argument_bytes,
            "compiler_output_bytes": unavailable.output_bytes,
            "compiler_temporary_bytes": unavailable.temporary_bytes,
            "compiler_generated_code_bytes": unavailable.generated_code_bytes,
            "source": unavailable.source,
            "unavailable_reason": unavailable.unavailable_reason,
        },
        "work": {
            "compiler_flops": unavailable.flops,
            "compiler_bytes_accessed": unavailable.bytes_accessed,
            "coarse_coefficient_action_sites": coarse_sites,
            "refined_coefficient_action_sites": refined_sites,
            "coefficient_callbacks_per_action_site": 3,
            "common_grid_comparison_values": spatial_points * time_points,
        },
        "certificate": _certificates(result, terminal_time, sigma),
    }


def _specifications():
    return (
        ("baseline", 17, 17, 5),
        ("grid-9", 9, 17, 5),
        ("grid-65", 65, 17, 5),
        ("actions-1", 17, 17, 1),
        ("actions-17", 17, 17, 17),
        ("time-9", 17, 9, 5),
        ("time-65", 17, 65, 5),
        ("refinement-large", 33, 33, 5),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.warmup < 0 or arguments.repeats < 1:
        raise ValueError("warmup must be non-negative and repeats must be positive")
    cases = [
        _case(
            name,
            spatial_points,
            time_points,
            action_count,
            warmup=arguments.warmup,
            repeats=arguments.repeats,
        )
        for name, spatial_points, time_points, action_count in _specifications()
    ]
    payload = {
        "benchmark": "control-hjb-reference",
        "environment": capture_environment().to_dict(),
        "cases": cases,
        "all_valid": all(case["certificate"]["successful"] for case in cases),
    }
    if arguments.output is None:
        print(json.dumps(payload, allow_nan=False, indent=2, sort_keys=True))
    else:
        write_json_atomic(arguments.output, payload)


if __name__ == "__main__":
    main()
