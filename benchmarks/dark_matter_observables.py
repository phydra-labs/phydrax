from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp

import phydrax as phx
from benchmarks._runtime import capture_environment
from phydrax.applications.cosmology._dark_matter_observables import (
    MixedComponentSpectrumPlan,
    weighted_particle_statistics,
)


def _case(size: int, particle_count: int, repetitions: int):
    shape = (size, size, size)
    maximum_wavenumber = jnp.sqrt(3.0) * jnp.pi * size
    shell_edges = jnp.linspace(0.0, maximum_wavenumber, size + 1)
    shells = phx.discretization.PeriodicFourierShellPlan(
        shape,
        (1.0, 1.0, 1.0),
        shell_edges,
        source_id="dark-matter-observable-benchmark",
    )
    spectrum_plan = MixedComponentSpectrumPlan(
        shells,
        ("wave", "particles"),
        normalization="additive-field",
        closure_absolute_tolerance=1e-9,
        closure_relative_tolerance=1e-8,
    )
    coordinates = jnp.meshgrid(
        *(jnp.arange(size, dtype=float) / size for _ in range(3)),
        indexing="ij",
    )
    base = (
        jnp.sin(2.0 * jnp.pi * coordinates[0])
        + 0.5 * jnp.cos(4.0 * jnp.pi * coordinates[1])
        + 0.25 * jnp.sin(6.0 * jnp.pi * coordinates[2])
    )
    fields = jnp.stack((base, 0.75 * base + 0.1 * jnp.roll(base, 1, axis=0)))
    spectrum = eqx.filter_jit(
        lambda values: spectrum_plan.evaluate(
            values,
            jnp.asarray(1.0, dtype=values.dtype),
            source_product_ids=("wave-benchmark", "particle-benchmark"),
        )
    )
    first = spectrum(fields)
    jax.block_until_ready(first.direct_total_power)
    started = time.perf_counter()
    result = first
    for _ in range(repetitions):
        result = spectrum(fields)
    jax.block_until_ready(result.direct_total_power)
    spectrum_seconds = (time.perf_counter() - started) / repetitions

    index = jnp.arange(particle_count, dtype=float)
    values = jnp.stack(
        (
            jnp.sin(index * 0.31),
            jnp.cos(index * 0.17),
            jnp.sin(index * 0.11 + 0.2),
        ),
        axis=-1,
    )
    weights = 0.5 + jnp.mod(index, 7.0) / 7.0
    active = jnp.ones((particle_count,), dtype=bool)
    statistics = jax.jit(weighted_particle_statistics)
    first_statistics = statistics(values, weights, active)
    jax.block_until_ready(first_statistics.covariance)
    started = time.perf_counter()
    final_statistics = first_statistics
    for _ in range(repetitions):
        final_statistics = statistics(values, weights, active)
    jax.block_until_ready(final_statistics.covariance)
    statistics_seconds = (time.perf_counter() - started) / repetitions

    cell_count = size**3
    return {
        "grid_shape": shape,
        "particle_count": particle_count,
        "repetitions": repetitions,
        "spectrum_seconds": spectrum_seconds,
        "spectrum_cells_per_second": cell_count / spectrum_seconds,
        "weighted_statistics_seconds": statistics_seconds,
        "weighted_particles_per_second": particle_count / statistics_seconds,
        "mixed_spectrum_closure_residual": float(result.closure_residual),
        "maximum_total_power": float(jnp.max(result.direct_total_power)),
        "effective_sample_size": float(final_statistics.effective_sample_size),
        "spectrum_successful": bool(result.successful),
        "weighted_statistics_successful": bool(final_statistics.successful),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=[16, 32])
    parser.add_argument("--particle-count", type=int, default=100_000)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/dark_matter_observables.json"),
    )
    arguments = parser.parse_args()
    if (
        any(size < 4 for size in arguments.sizes)
        or arguments.particle_count < 2
        or arguments.repeats < 1
    ):
        raise ValueError(
            "sizes >= 4, particle-count >= 2, and repeats >= 1 are required."
        )
    cases = [
        _case(size, arguments.particle_count, arguments.repeats)
        for size in arguments.sizes
    ]
    payload = {
        "environment": capture_environment().to_dict(),
        "cases": cases,
        "all_successful": all(
            case["spectrum_successful"] and case["weighted_statistics_successful"]
            for case in cases
        ),
        "maximum_closure_residual": max(
            case["mixed_spectrum_closure_residual"] for case in cases
        ),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    arguments.output.write_text(encoded + "\n")
    print(encoded)
    if not payload["all_successful"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
