#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Decision benchmark for bounded soft-matter observables and path reduction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from _runtime import capture_environment, logical_array_bytes, measure_repeated

from phydrax.atomistic._observables import (
    lagged_msd_vacf,
    LaggedCorrelationPlan,
    static_structure_factor,
    StaticStructureFactorPlan,
)
from phydrax.stochastic.path_sampling import (
    DiscretePathThermodynamicsPlan,
    normalized_discrete_path_thermodynamics,
    PathBuffer,
)


def benchmark_case(
    particles: int,
    frames: int,
    wave_vectors: int,
    lags: int,
    paths: int,
    repeats: int,
) -> dict[str, object]:
    generator = np.random.default_rng(41)
    increments = generator.normal(size=(frames, particles, 3)) * 0.02
    positions = jnp.asarray(np.cumsum(increments, axis=0))
    velocities = jnp.asarray(increments / 0.01)
    times = jnp.arange(frames, dtype=positions.dtype) * 0.01
    k = np.zeros((wave_vectors, 3))
    k[:, 0] = np.linspace(0.0, 12.0, wave_vectors)
    structure_plan = StaticStructureFactorPlan(
        k,
        maximum_frames=frames,
        maximum_particles=particles,
    )
    lag_steps = np.unique(np.linspace(0, frames - 1, lags, dtype="int64"))
    correlation_plan = LaggedCorrelationPlan(
        lag_steps,
        maximum_frames=frames,
        maximum_particles=particles,
    )
    structure, structure_timing = measure_repeated(
        lambda: static_structure_factor(structure_plan, positions),
        warmup=1,
        repeats=repeats,
    )
    correlation, correlation_timing = measure_repeated(
        lambda: lagged_msd_vacf(
            correlation_plan,
            times,
            positions,
            velocities,
        ),
        warmup=1,
        repeats=repeats,
    )
    path_values = tuple(
        PathBuffer.from_trajectory(
            jnp.asarray([[0.0], [index / paths], [1.0]]),
            jnp.asarray([0.0, 0.5, 1.0]),
            capacity=4,
        )
        for index in range(paths)
    )
    probabilities = jnp.arange(1, paths + 1, dtype="float64")
    probabilities = probabilities / jnp.sum(probabilities)
    entropy = jnp.log(probabilities) - jnp.log(probabilities[::-1])
    heat = -entropy
    thermodynamic_plan = DiscretePathThermodynamicsPlan(
        1.0,
        maximum_paths=paths,
        maximum_steps=3,
        autocorrelation_lag=max(1, min(paths // 4, paths - 1)),
    )
    thermodynamics, path_timing = measure_repeated(
        lambda: normalized_discrete_path_thermodynamics(
            thermodynamic_plan,
            path_values,
            jnp.log(probabilities),
            jnp.log(probabilities[::-1]),
            jnp.zeros(paths),
            -heat,
            heat,
            jnp.zeros(paths),
        ),
        warmup=1,
        repeats=repeats,
    )
    return {
        "axes": {
            "particles": particles,
            "frames": frames,
            "wave_vectors": wave_vectors,
            "lags": lag_steps.size,
            "paths": paths,
        },
        "timing": {
            "static_structure_factor": structure_timing.to_milliseconds_dict(),
            "lagged_msd_vacf": correlation_timing.to_milliseconds_dict(),
            "path_thermodynamics": path_timing.to_milliseconds_dict(),
        },
        "logical_array_bytes": {
            "positions": logical_array_bytes(positions),
            "structure_result": logical_array_bytes(structure),
            "correlation_result": logical_array_bytes(correlation),
            "path_result": logical_array_bytes(thermodynamics),
        },
        "scientific_residuals": {
            "fluctuation_reversal": float(thermodynamics.reversal_residual),
            "first_law": float(jnp.max(jnp.abs(thermodynamics.first_law_residual))),
            "detailed_balance": float(
                jnp.max(jnp.abs(thermodynamics.detailed_balance_residual))
            ),
        },
        "successful": {
            "structure": bool(structure.successful),
            "correlation": bool(correlation.successful),
            "path_thermodynamics": bool(thermodynamics.successful),
        },
        "plan_ids": {
            "structure": structure_plan.plan_id,
            "correlation": correlation_plan.plan_id,
            "path_thermodynamics": thermodynamic_plan.plan_id,
        },
        "measured_peak_host_bytes": None,
        "measured_peak_device_bytes": None,
        "peak_measurement_note": (
            "Logical payload bytes are reported separately; allocator peak APIs are not used."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.repeats <= 0:
        raise ValueError("repeats must be positive.")
    records = [
        benchmark_case(*axes, arguments.repeats)
        for axes in (
            (16, 64, 16, 8, 8),
            (64, 256, 64, 32, 32),
            (128, 512, 128, 64, 64),
        )
    ]
    payload = {
        "environment": capture_environment().to_dict(),
        "records": records,
        "qualification_claim": False,
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    if arguments.output is None:
        print(encoded)
    else:
        arguments.output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
