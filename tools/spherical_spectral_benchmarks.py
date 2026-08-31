#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp

import phydrax as phx
from benchmarks._runtime import measure_repeated, measure_synchronized


@dataclass(frozen=True)
class SphericalSpectralBenchmarkRecord:
    schema_version: int
    bandlimit: int
    sampling: str
    execution: str
    radius: float
    physical_points: int
    padded_coefficients: int
    logical_modes: int
    precompute_bytes: int
    prepare_ms: float
    project_first_jit_ms: float
    project_steady_ms: float
    laplacian_first_jit_ms: float
    laplacian_steady_ms: float
    roundtrip_error: float
    area_defect: float
    laplacian_error: float
    invalid_capacity_error: float
    finite: bool

    @property
    def passed(self) -> bool:
        return (
            self.finite
            and self.roundtrip_error <= 1e-9
            and self.area_defect <= 1e-10
            and self.laplacian_error <= 1e-9
            and self.invalid_capacity_error <= 1e-11
        )


def _measure(function, argument, repeats):
    compiled = eqx.filter_jit(function)
    value, first_seconds = measure_synchronized(lambda: compiled(argument))
    value, distribution = measure_repeated(
        lambda: compiled(argument),
        warmup=0,
        repeats=repeats,
    )
    return (
        value,
        1_000.0 * first_seconds,
        1_000.0 * float(distribution.mean_seconds),
    )


def run_spherical_spectral_benchmark(
    bandlimit: int = 16,
    /,
    *,
    sampling: str = "mw",
    execution: str = "recursive",
    radius: float = 1.0,
    repeats: int = 5,
) -> SphericalSpectralBenchmarkRecord:
    limit = int(bandlimit)
    repeat_count = int(repeats)
    if limit < 2 or repeat_count < 1:
        raise ValueError("bandlimit must exceed one and repeats must be positive.")
    started = time.perf_counter()
    space = phx.discretization.SphericalSpectralPlan(
        limit,
        sampling=sampling,
        execution=execution,
    ).prepare(radius=radius)
    jax.block_until_ready(space.transform.theta)
    prepare_ms = 1e3 * (time.perf_counter() - started)
    theta, phi = jnp.meshgrid(
        space.transform.theta,
        space.transform.phi,
        indexing="ij",
    )
    del phi
    values = jnp.cos(theta)
    coefficients, project_first, project_steady = _measure(
        space.project,
        values,
        repeat_count,
    )
    laplacian, laplacian_first, laplacian_steady = _measure(
        space.laplacian,
        values,
        repeat_count,
    )
    reconstructed = space.reconstruct(coefficients)
    contaminated = coefficients.at[0, 0].set(jnp.nan + 1j * jnp.inf)
    inert = space.reconstruct(contaminated)
    expected_area = 4.0 * jnp.pi * float(radius) ** 2
    expected_laplacian = -2.0 / float(radius) ** 2 * values
    roundtrip_error = jnp.max(jnp.abs(reconstructed - values))
    area_defect = jnp.abs(jnp.sum(space.quadrature_weights) - expected_area)
    laplacian_error = jnp.max(jnp.abs(laplacian - expected_laplacian))
    invalid_error = jnp.max(jnp.abs(inert - reconstructed))
    finite = bool(
        jnp.all(jnp.isfinite(coefficients))
        & jnp.all(jnp.isfinite(laplacian))
        & jnp.all(jnp.isfinite(inert))
    )
    resources = dict(space.preparation.resource_counts)
    return SphericalSpectralBenchmarkRecord(
        schema_version=1,
        bandlimit=limit,
        sampling=space.transform.sampling,
        execution=space.transform.execution,
        radius=float(radius),
        physical_points=resources["physical_points"],
        padded_coefficients=resources["padded_coefficients"],
        logical_modes=resources["logical_modes"],
        precompute_bytes=resources["precompute_bytes"],
        prepare_ms=float(prepare_ms),
        project_first_jit_ms=float(project_first),
        project_steady_ms=float(project_steady),
        laplacian_first_jit_ms=float(laplacian_first),
        laplacian_steady_ms=float(laplacian_steady),
        roundtrip_error=float(roundtrip_error),
        area_defect=float(area_defect),
        laplacian_error=float(laplacian_error),
        invalid_capacity_error=float(invalid_error),
        finite=finite,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark exact-sampling spherical spectral workflows."
    )
    parser.add_argument("--bandlimit", type=int, default=16)
    parser.add_argument("--sampling", choices=("mw", "mwss", "dh", "gl"), default="mw")
    parser.add_argument(
        "--execution", choices=("recursive", "precomputed"), default="recursive"
    )
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    record = run_spherical_spectral_benchmark(
        4 if arguments.smoke else arguments.bandlimit,
        sampling=arguments.sampling,
        execution=arguments.execution,
        radius=arguments.radius,
        repeats=1 if arguments.smoke else arguments.repeats,
    )
    payload = json.dumps({**asdict(record), "passed": record.passed}, indent=2)
    if arguments.output is not None:
        temporary = arguments.output.with_suffix(arguments.output.suffix + ".tmp")
        temporary.write_text(payload + "\n")
        os.replace(temporary, arguments.output)
    print(payload)
    if not record.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
