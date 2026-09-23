#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json

import jax
import jax.numpy as jnp

from benchmarks._runtime import (
    capture_environment,
    logical_array_bytes,
    measure_host,
    measure_repeated,
    measure_synchronized,
)
from phydrax.applications.supersymmetric_lattice import (
    assess_twisted_fermion_algebra,
    prepare_twisted_n2_rhmc,
    sample_twisted_n2_rhmc,
    TwistedN2SYMPlan,
    TwistedSYMCoordinateLayout,
)
from phydrax.sampling import RHMCResourcePolicy


def benchmark_case(extent: int, repeats: int):
    theory = TwistedN2SYMPlan(
        (extent, extent),
        matrix_rank=1,
        coupling=1.0,
        fermion_mass=1.0,
        coordinate_bound=0.25,
        temporal_axis=0,
        fermion_boundary_phase=-1.0,
        maximum_fermion_elements=1_000_000,
    )
    layout = TwistedSYMCoordinateLayout(theory.prepare_bosonic())
    coordinates = jnp.zeros(layout.coordinate_shape, dtype=jnp.float64)
    prepared, prepare_seconds = measure_host(
        lambda: prepare_twisted_n2_rhmc(
            theory,
            coordinates,
            step_size=1e-4,
            trajectory_steps=1,
            rational_poles=4,
            rational_verification_points=1025,
            bosonic_substeps=1,
            resources=RHMCResourcePolicy(
                maximum_terms=8,
                maximum_force_evaluations=128,
                maximum_retained_bytes=256 * 1024**2,
                maximum_output_bytes=256 * 1024**2,
                maximum_draws=max(1, repeats),
            ),
        )
    )
    maximum_dense = prepared.kahler_dirac.source.size**2
    algebra, algebra_seconds = measure_synchronized(
        lambda: assess_twisted_fermion_algebra(
            prepared.kahler_dirac,
            regulator_mass=theory.fermion_mass,
            structural_lower_bound=theory.normal_spectral_lower,
            structural_upper_bound=theory.normal_spectral_upper,
            maximum_dense_elements=maximum_dense,
            tolerance=1e-9,
        )
    )
    root_key = jax.random.key(101)
    keys = jax.random.split(root_key, repeats + 2)
    _, warm_seconds = measure_synchronized(
        lambda: sample_twisted_n2_rhmc(
            prepared,
            coordinates,
            keys[0],
            num_draws=1,
        )
    )
    run, _ = measure_synchronized(
        lambda: sample_twisted_n2_rhmc(
            prepared,
            coordinates,
            keys[1],
            num_draws=max(32, repeats),
        )
    )
    timing_keys = iter(keys[2:])
    _, steady = measure_repeated(
        lambda: sample_twisted_n2_rhmc(
            prepared,
            coordinates,
            next(timing_keys),
            num_draws=1,
        ),
        warmup=0,
        repeats=repeats,
    )
    return {
        "axes": {
            "lattice_shape": [extent, extent],
            "matrix_rank": 1,
            "fermion_dimension": prepared.kahler_dirac.source.size,
            "rational_poles": prepared.pseudofermion.action_approximation.function.num_poles,
            "trajectory_steps": prepared.kernel.plan.trajectory_steps,
        },
        "ids": {
            "theory": theory.plan_id,
            "prepared": prepared.prepared_id,
            "kernel": prepared.kernel.kernel_id,
        },
        "prepare_seconds": prepare_seconds,
        "algebra_seconds": algebra_seconds,
        "warm_transition_seconds": warm_seconds,
        "rng": {
            "root_seed": 101,
            "warmup_stream": 0,
            "acceptance_stream": 1,
            "timing_streams": list(range(2, repeats + 2)),
        },
        "steady": steady.to_seconds_dict(),
        "logical_bytes": {
            "prepared": logical_array_bytes(prepared),
            "sample": logical_array_bytes(run),
        },
        "scientific_residuals": {
            "antisymmetry": float(algebra.antisymmetry_residual),
            "adjoint": float(algebra.adjoint_residual),
            "action_rational": float(
                prepared.pseudofermion.action_approximation.maximum_relative_error
            ),
            "refresh_rational": float(
                prepared.pseudofermion.refresh_approximation.maximum_relative_error
            ),
            "maximum_energy_error": float(run.evidence.maximum_energy_error),
        },
        "acceptance_rate": float(run.evidence.acceptance_rate),
        "successful": bool(algebra.accepted and run.evidence.successful),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extents", nargs="+", type=int, default=(1,))
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", type=str)
    arguments = parser.parse_args()
    if any(value < 1 for value in arguments.extents) or arguments.repeats < 1:
        raise ValueError("Extents and repeats must be positive.")
    cases = [
        benchmark_case(extent, arguments.repeats) for extent in arguments.extents
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
