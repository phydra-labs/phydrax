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
from phydrax.applications.phase_field import DoubleWellKinkPlan, solve_double_well_kink
from phydrax.discretization import FourierAxisSpec, TensorGridPlan
from phydrax.equations import DoubleWellFreeEnergy
from phydrax.geometry import RigidFrame
from phydrax.optics.wave import (
    EnvelopeNonlinearResponsePlan,
    EnvelopePropagationPlan,
    PlaneFieldSpace,
    prepare_envelope_nonlinear_response,
    prepare_envelope_propagation,
    propagate_envelope,
    PulseEnvelopeField,
    PulseTimeSpace,
)


def benchmark_case(kink_points: int, optical_steps: int):
    kink_plan = DoubleWellKinkPlan(
        jnp.linspace(-8.0, 8.0, kink_points),
        DoubleWellFreeEnergy(1.0),
        gradient_coefficient=1.0,
        residual_tolerance=1e-9,
    )
    kink, kink_seconds = measure_synchronized(lambda: solve_double_well_kink(kink_plan))

    plane_grid = TensorGridPlan(
        (FourierAxisSpec(2), FourierAxisSpec(2)), axis_names=("u", "v")
    ).prepare(jnp.asarray([[-1.0, -1.0], [1.0, 1.0]]))
    plane = PlaneFieldSpace(plane_grid, RigidFrame.identity(3), "periodic-cell")
    time_grid = TensorGridPlan((FourierAxisSpec(256),), axis_names=("time",)).prepare(
        jnp.asarray([[-20.0], [20.0]])
    )
    time = PulseTimeSpace(time_grid, topology="periodic-cell")
    values = (1.0 / jnp.cosh(time.coordinates)).astype(jnp.complex128)
    field = PulseEnvelopeField(
        plane,
        time,
        jnp.broadcast_to(values, plane.shape + time.shape),
        100.0,
        0.0,
        polarization="scalar",
    )
    response = prepare_envelope_nonlinear_response(
        EnvelopeNonlinearResponsePlan(1.0, source_id="benchmark-kerr"), time
    )
    prepared, prepare_seconds = measure_host(
        lambda: prepare_envelope_propagation(
            EnvelopePropagationPlan(
                time,
                {2: -1.0},
                step_count=optical_steps,
                maximum_spectral_edge_fraction=1e-7,
                maximum_refinement_error=1e-3,
            ),
            response,
        )
    )
    propagation, propagation_seconds = measure_synchronized(
        lambda: propagate_envelope(prepared, field, 0.2)
    )
    intensity_error = jnp.max(
        jnp.abs(jnp.abs(propagation.field.values) ** 2 - jnp.abs(field.values) ** 2)
    )
    return {
        "axes": {
            "kink_points": kink_points,
            "optical_time_points": time.size,
            "optical_steps": optical_steps,
        },
        "ids": {"kink": kink_plan.plan_id, "optics": prepared.prepared_id},
        "host_seconds": {
            "kink_solve": kink_seconds,
            "optics_prepare": prepare_seconds,
            "optics_propagate_with_refinement": propagation_seconds,
        },
        "logical_bytes": {
            "kink": logical_array_bytes(kink),
            "optics": logical_array_bytes(propagation),
        },
        "scientific_residuals": {
            "kink": float(kink.evidence.maximum_residual),
            "kink_energy": float(kink.evidence.energy),
            "optical_intensity": float(intensity_error),
            "optical_refinement": float(propagation.evidence.fixed_step_refinement_error),
            "optical_edge": float(propagation.evidence.spectral_edge_fraction),
        },
        "successful": bool(kink.evidence.accepted and propagation.evidence.accepted),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kink-points", nargs="+", type=int, default=(65, 129))
    parser.add_argument("--optical-steps", type=int, default=32)
    parser.add_argument("--output", type=str)
    arguments = parser.parse_args()
    if any(value < 5 for value in arguments.kink_points) or arguments.optical_steps < 1:
        raise ValueError("Benchmark resolutions must be positive and supported.")
    cases = [
        benchmark_case(points, arguments.optical_steps)
        for points in arguments.kink_points
    ]
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
