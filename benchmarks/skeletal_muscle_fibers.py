#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax.numpy as jnp

from benchmarks._runtime import capture_environment
from phydrax.applications.skeletal_muscle.fibers import (
    PrescribedFiberStimulusSchedule,
    SkeletalFiberBundlePlan,
)


_SOLVER_STRUCTURE = {
    "integrator": "Diffrax Kvaerno5 implicit ESDIRK",
    "nonlinear_solver": (
        "Diffrax VeryChord with one full-state Jacobian linearization per solver step"
    ),
    "linear_solver": (
        "Lineax AutoLinearSolver(well_posed=None), which selects dense LU for the "
        "untagged square full-state Jacobian"
    ),
    "matrix_free": False,
}


def _case(fiber_count: int, node_count: int) -> dict[str, object]:
    mask = jnp.zeros((1, fiber_count, node_count), dtype=bool).at[0, :, 0].set(True)
    schedule = PrescribedFiberStimulusSchedule(
        jnp.asarray([0.0]),
        jnp.asarray([0.05]),
        jnp.asarray([150.0]),
        mask,
    )
    runtime = SkeletalFiberBundlePlan(
        tuple(f"fiber-{index}" for index in range(fiber_count)),
        node_count,
        jnp.full((fiber_count,), 10.0),
        jnp.full((fiber_count,), 0.1),
        schedule,
        maximum_step_ms=0.05,
    ).prepare()
    initial = runtime.initialize()
    start = time.perf_counter()
    candidate = runtime.candidate(initial, 0.05)
    candidate.candidate_state.values.block_until_ready()
    elapsed_ms = 1000.0 * (time.perf_counter() - start)
    return {
        "fiber_count": fiber_count,
        "node_count": node_count,
        "state_count": fiber_count * node_count * 56,
        "compile_and_solve_ms": elapsed_ms,
        "solver_steps": int(candidate.evidence.solver_steps),
        "successful": bool(candidate.evidence.successful),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/skeletal_muscle_fibers.json"),
    )
    arguments = parser.parse_args()
    cases = [_case(1, 3)] if arguments.smoke else [_case(1, 5), _case(4, 17)]
    largest = next(
        (case for case in reversed(cases) if case["successful"]),
        None,
    )
    payload = {
        "environment": capture_environment().to_dict(),
        "solver_structure": _SOLVER_STRUCTURE,
        "capacity": {
            "largest_qualified_workload": (
                None
                if largest is None
                else {
                    "fiber_count": largest["fiber_count"],
                    "node_count": largest["node_count"],
                    "state_count": largest["state_count"],
                }
            ),
            "qualification_boundary": (
                "Largest workload successfully completed in this recorded run; this "
                "is neither a measured failure threshold nor evidence of scalable "
                "asymptotics."
            ),
            "limitation": (
                "The recorded solver path materializes and factors a dense full-state "
                "Jacobian. Storage grows quadratically and dense LU work cubically "
                "with state count, so no scalability claim is made beyond the "
                "largest qualified workload."
            ),
        },
        "cases": cases,
        "all_successful": all(case["successful"] for case in cases),
    }
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["all_successful"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
