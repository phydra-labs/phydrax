#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json

import jax.numpy as jnp

from benchmarks._runtime import (
    capture_environment,
    logical_array_bytes,
    measure_host,
    measure_synchronized,
)
from phydrax.interchange.hep import parse_slha, spectrum_observables_from_slha
from phydrax.particle_physics import ScaleBVPPlan, solve_scale_bvp


def benchmark_case(integration_steps: int):
    source = b"BLOCK MASS\n 25 1.251000000E+02\n 1000022 2.0E+02\n"
    document, parse_seconds = measure_host(lambda: parse_slha(source))
    observables, extraction_seconds = measure_host(
        lambda: spectrum_observables_from_slha(document)
    )
    plan = ScaleBVPPlan(
        ("g",),
        lambda log_scale, parameters: 0.2 * parameters,
        lambda parameters: jnp.zeros((0,), dtype=parameters.dtype),
        lambda parameters: jnp.asarray((parameters[0] - 3.0,)),
        low_residual_count=0,
        lower_scale=10.0,
        upper_scale=1000.0,
        integration_steps=integration_steps,
        maximum_newton_steps=8,
        residual_tolerance=1e-11,
        source_ids=("analytic-beta-control",),
    )
    result, solve_seconds = measure_synchronized(
        lambda: solve_scale_bvp(plan, jnp.asarray((1.0,)))
    )
    return {
        "axes": {
            "integration_steps": integration_steps,
            "parameter_count": plan.parameter_count,
            "slha_entries": document.diagnostics.entry_count,
        },
        "ids": {"slha": document.source_id, "bvp_plan": plan.plan_id},
        "host_seconds": {
            "slha_parse": parse_seconds,
            "observable_extract": extraction_seconds,
            "scale_bvp": solve_seconds,
        },
        "logical_bytes": {
            "observables": logical_array_bytes(observables),
            "bvp_result": logical_array_bytes(result),
        },
        "scientific_residuals": {
            "boundary_residual": float(jnp.linalg.norm(result.residual)),
            "terminal_parameter_error": float(jnp.abs(result.trajectory[-1, 0] - 3.0)),
        },
        "newton_steps": result.accepted_steps.shape[0],
        "successful": bool(result.converged and result.finite),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--integration-steps", nargs="+", type=int, default=(32, 128, 512)
    )
    parser.add_argument("--output", type=str)
    arguments = parser.parse_args()
    if any(value < 1 for value in arguments.integration_steps):
        raise ValueError("integration steps must be positive.")
    cases = [benchmark_case(value) for value in arguments.integration_steps]
    payload = {
        "environment": capture_environment().to_dict(),
        "cases": cases,
        "passed": all(case["successful"] for case in cases),
    }
    encoded = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True)
    if arguments.output:
        from benchmarks._io import write_json_atomic

        write_json_atomic(arguments.output, payload)
    else:
        print(encoded)
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
