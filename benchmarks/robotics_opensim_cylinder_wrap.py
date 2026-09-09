#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp

from benchmarks._runtime import capture_environment
from phydrax.applications.robotics import OpenSimCylinderRouteWrapPlan


def _case(batch_size: int, samples: int, repetitions: int) -> dict[str, object]:
    prepared = OpenSimCylinderRouteWrapPlan(samples).prepare(
        jnp.zeros(3), jnp.asarray((0.0, 0.0, 1.0)), 1.0, 8.0
    )
    offsets = jnp.linspace(-0.1, 0.1, batch_size)
    ones = jnp.ones_like(offsets)
    points = jnp.stack(
        (
            jnp.stack((-2.0 * ones, 0.35 + offsets, -0.8 * ones), axis=-1),
            jnp.stack((2.1 * ones, 0.65 - offsets, 1.2 * ones), axis=-1),
        ),
        axis=1,
    )
    source = prepared.initial_state()
    velocity = jnp.asarray(((0.12, -0.04, 0.17), (-0.03, 0.08, -0.11)))

    def advance(endpoints):
        candidate = prepared.propose(source, endpoints)
        state = prepared.commit(candidate, source)
        loads, power = prepared.tensile_force_pullback(
            state, endpoints, velocity, 120.0, force_owner="native-tension"
        )
        return candidate, loads, power

    action = eqx.filter_jit(jax.vmap(advance))
    begin = time.perf_counter()
    executable = action.lower(points).compile()
    compile_ms = 1000.0 * (time.perf_counter() - begin)
    begin = time.perf_counter()
    result = executable(points)
    jax.block_until_ready(result)
    first_execution_ms = 1000.0 * (time.perf_counter() - begin)
    begin = time.perf_counter()
    for _ in range(repetitions):
        result = executable(points)
    jax.block_until_ready(result)
    execution_ms = 1000.0 * (time.perf_counter() - begin) / repetitions
    candidate, _, power = result
    evidence = candidate.evaluation.evidence
    return {
        "route_count": batch_size,
        "obstacle_capacity_per_route": 1,
        "candidate_capacity_per_route": 3,
        "extra_windings": 0,
        "sample_count": samples,
        "nonlinear_iterations": 0,
        "numerical_realization": "exact-unrolled-common-axial-slope",
        "source_revision": evidence.source_revision,
        "source_sha256": evidence.source_sha256,
        "compile_ms": compile_ms,
        "first_execution_ms": first_execution_ms,
        "execution_ms": execution_ms,
        "routes_per_second": 1000.0 * batch_size / execution_ms,
        "input_array_bytes": points.nbytes,
        "retained_result_array_bytes": sum(
            leaf.nbytes for leaf in jax.tree_util.tree_leaves(result)
        ),
        "memory_scope": "input-and-retained-output-only-not-peak-device-allocation",
        "feasible_candidate_count": int(jnp.sum(evidence.candidate_feasible)),
        "infeasible_candidate_count": int(jnp.sum(~evidence.candidate_feasible)),
        "accepted_fraction": float(jnp.mean(candidate.successful)),
        "branch_change_fraction": float(jnp.mean(evidence.mode_changed)),
        "maximum_tangent_direction_residual": float(
            jnp.max(evidence.tangent_direction_residual)
        ),
        "maximum_surface_residual_m": float(jnp.max(evidence.surface_residual_m)),
        "minimum_shortest_lateral_gap_m": float(jnp.min(evidence.shortest_lateral_gap_m)),
        "maximum_power_residual_W": float(jnp.max(jnp.abs(power.power_residual_W))),
        "all_successful": bool(jnp.all(candidate.successful) & jnp.all(power.successful)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/robotics_opensim_cylinder_wrap.json"),
    )
    arguments = parser.parse_args()
    sizes = (1,) if arguments.smoke else (1, 128, 1024)
    samples = (8,) if arguments.smoke else (8, 32)
    repetitions = 2 if arguments.smoke else 20
    cases = [_case(size, count, repetitions) for size in sizes for count in samples]
    payload = {
        "environment": capture_environment().to_dict(),
        "scope": "single-static-cylinder-lateral-candidates-commit-endpoint-pullback",
        "shortest_path_scope": "two-zero-extra-winding-lateral-branches-not-finite-solid-caps",
        "cases": cases,
        "all_successful": all(case["all_successful"] for case in cases),
    }
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["all_successful"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
