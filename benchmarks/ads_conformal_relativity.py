#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json

import equinox as eqx

from benchmarks._runtime import (
    capture_environment,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_repeated,
    measure_synchronized,
)
from phydrax.applications import numerical_relativity as nr


def benchmark_case(point_count: int, repeats: int):
    system = nr.ConformalEinsteinSystem(
        -3.0,
        scalar_curvature_gauge=-12.0,
        residual_tolerance=1e-12,
    )
    state, derivatives = nr.exact_ads_conformal_reference(system, (4, 4, 4))
    evaluate = eqx.filter_jit(nr.evaluate_conformal_einstein_zero_quantities)
    executable, compilation = measure_lower_and_compile(
        lambda: evaluate.lower(system, state, derivatives),
        lambda lowered: lowered.compile(),
    )
    zero, zero_warm_seconds = measure_synchronized(
        lambda: executable(system, state, derivatives)
    )
    _, zero_steady = measure_repeated(
        lambda: executable(system, state, derivatives),
        warmup=0,
        repeats=repeats,
    )

    scalar_plan = nr.ConformalAdSScalarPlan(
        point_count,
        time_step=0.25 * (0.5 * 3.141592653589793 / (point_count - 1)),
        maximum_steps=100,
    )
    scalar_initial, frequency = nr.conformal_scalar_normal_mode(scalar_plan, 1)
    scalar_run, scalar_warm_seconds = measure_synchronized(
        lambda: nr.run_conformal_ads_scalar(
            scalar_plan,
            scalar_initial,
            steps=20,
            energy_tolerance=5e-3,
        )
    )
    _, scalar_steady = measure_repeated(
        lambda: nr.run_conformal_ads_scalar(
            scalar_plan,
            scalar_initial,
            steps=20,
            energy_tolerance=5e-3,
        ),
        warmup=0,
        repeats=repeats,
    )
    return {
        "axes": {
            "zero_quantity_grid": [4, 4, 4],
            "scalar_points": point_count,
            "scalar_steps": 20,
            "scalar_frequency": frequency,
        },
        "ids": {
            "system": system.system_id,
            "state": state.state_id,
            "scalar_plan": scalar_plan.plan_id,
        },
        "zero_quantity": {
            "lowering_seconds": compilation.lowering_seconds,
            "compilation_seconds": compilation.compilation_seconds,
            "warm_seconds": zero_warm_seconds,
            "steady": zero_steady.to_seconds_dict(),
            "logical_bytes": logical_array_bytes((state, derivatives, zero)),
            "maximum_residual": float(zero.maximum_residual),
        },
        "scalar_runtime": {
            "warm_seconds": scalar_warm_seconds,
            "steady": scalar_steady.to_seconds_dict(),
            "logical_bytes": logical_array_bytes(scalar_run),
            "relative_energy_drift": float(scalar_run.evidence.relative_energy_drift),
            "boundary_residual": float(scalar_run.evidence.maximum_boundary_residual),
        },
        "successful": bool(zero.accepted and scalar_run.evidence.accepted),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", nargs="+", type=int, default=(33, 65))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=str)
    arguments = parser.parse_args()
    if any(value < 5 for value in arguments.points) or arguments.repeats < 1:
        raise ValueError("Point counts and repeats are invalid.")
    payload = {
        "environment": capture_environment().to_dict(),
        "cases": [benchmark_case(value, arguments.repeats) for value in arguments.points],
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    if arguments.output:
        with open(arguments.output, "w", encoding="utf-8") as stream:
            stream.write(encoded + "\n")
    else:
        print(encoded)


if __name__ == "__main__":
    main()
