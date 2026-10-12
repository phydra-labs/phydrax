"""MAC shadowing least-squares workspace scaling; large dense runs compile only.

The baseline kernel is the untouched global design/SVD algebra from MAC shadowing.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from phydrax.ein import contract

from ._io import write_json_atomic
from ._memory import fitted_scaling_exponent, host_allocation_peak, process_peak_rss_bytes
from ._runtime import (
    capture_benchmark_identity,
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_repeated,
)


def baseline(
    bases: Array, tangents: Array, factors: Array, offsets: Array
) -> tuple[Array, Array, Array]:
    segments, samples, rank = bases.shape
    eye = jnp.eye(segments, dtype=bases.dtype)
    design = contract("snk,sr->snrk", bases, eye).reshape((-1, segments * rank))
    coupling = -contract("eij,er->eirj", factors[:-1], eye[:-1]) + contract(
        "ij,er->eirj", jnp.eye(rank, dtype=bases.dtype), eye[1:]
    )
    weighted = jnp.concatenate(
        (design, 100.0 * coupling.reshape(((segments - 1) * rank, -1))), axis=0
    )
    target = jnp.concatenate(
        (-tangents.reshape((-1,)), 100.0 * offsets[:-1].reshape((-1,)))
    )
    u, singular, vh = jnp.linalg.svd(weighted, full_matrices=False)
    cutoff = (
        jnp.finfo(bases.dtype).eps * max(weighted.shape) * jnp.maximum(singular[0], 1.0)
    )
    coefficients = vh.T @ (
        jnp.where(singular > cutoff, 1.0 / singular, 0.0) * (u.T @ target)
    )
    diagnostics = jnp.linalg.svd(weighted, compute_uv=False)
    return (
        coefficients,
        diagnostics[jnp.asarray([0, diagnostics.size - 1])],
        jnp.all(singular > cutoff) & jnp.all(jnp.isfinite(coefficients)),
    )


def candidate(
    bases: Array, tangents: Array, factors: Array, offsets: Array
) -> tuple[Array, Array, Array]:
    from phydrax.linalg._block_banded_least_squares import (
        solve_block_banded_least_squares,
    )

    result = solve_block_banded_least_squares(
        bases,
        -tangents,
        -100.0 * factors[:-1],
        jnp.broadcast_to(
            100.0 * jnp.eye(bases.shape[-1], dtype=bases.dtype), factors[:-1].shape
        ),
        100.0 * offsets[:-1],
    )
    return (
        result.value.reshape((-1,)),
        jnp.stack((result.maximum_singular_value, result.minimum_singular_value)),
        result.successful,
    )


def dense_route(
    bases: Array, tangents: Array, factors: Array, offsets: Array
) -> tuple[Array, Array, Array]:
    from phydrax.solver._mac_sensitivity import _dense_shadowing_least_squares

    value, full_rank, minimum, condition = _dense_shadowing_least_squares(
        bases, -tangents, factors, offsets, 100.0, 0.0, 1024 * 1024 * 1024
    )
    return (
        value.reshape((-1,)),
        jnp.stack((minimum * condition, minimum)),
        full_rank & jnp.all(jnp.isfinite(value)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--segments", nargs="+", type=int, default=[8, 16, 32])
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--rank", type=int, default=3)
    parser.add_argument(
        "--routes",
        nargs="+",
        choices=["baseline", "block-banded", "dense-svd"],
        default=["block-banded", "dense-svd"],
    )
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).with_suffix(".json")
    )
    options = parser.parse_args()
    cases: list[dict[str, Any]] = []
    for segments in options.segments:

        def inputs() -> tuple[Array, Array, Array, Array]:
            coordinates = jnp.arange(
                segments * options.samples * options.rank, dtype=jnp.float64
            ).reshape((segments, options.samples, options.rank))
            bases = jnp.sin(coordinates * 0.37) + jnp.broadcast_to(
                jnp.eye(options.samples, options.rank, dtype=jnp.float64),
                coordinates.shape,
            )
            return (
                bases,
                jnp.cos(coordinates[..., 0]),
                jnp.broadcast_to(
                    jnp.eye(options.rank, dtype=jnp.float64),
                    (segments, options.rank, options.rank),
                ),
                jnp.full((segments, options.rank), 0.01, dtype=jnp.float64),
            )

        values, host_peak = host_allocation_peak(inputs)
        for route in options.routes:
            entry = jax.jit(
                {
                    "baseline": baseline,
                    "block-banded": candidate,
                    "dense-svd": dense_route,
                }[route]
            )
            executable, timing = measure_lower_and_compile(
                lambda: entry.lower(*values), lambda lowered: lowered.compile()
            )
            evidence = compiler_evidence(
                executable.cost_analysis(),
                executable.memory_analysis(),
                source="mac-shadowing-least-squares",
            )
            warm = None
            retained = None
            extrema = None
            successful = None
            condition = None
            if not options.compile_only:
                result, timings = measure_repeated(
                    lambda: executable(*values), warmup=1, repeats=2
                )
                warm = timings.to_seconds_dict()
                retained = logical_array_bytes(result)
                extrema = [float(value) for value in result[1]]
                successful = bool(result[2])
                condition = float(result[1][0] / result[1][1])
            cases.append(
                {
                    "capacity": segments,
                    "samples": options.samples,
                    "rank": options.rank,
                    "route": route,
                    "compile_only": options.compile_only,
                    "lowering_seconds": timing.lowering_seconds,
                    "compilation_seconds": timing.compilation_seconds,
                    "warm": warm,
                    "compiler": asdict(evidence),
                    "host_peak_bytes": host_peak,
                    "retained_bytes": retained,
                    "singular_extrema": extrema,
                    "successful": successful,
                    "condition_number": condition,
                    "logical_design_bytes": (
                        segments * options.samples + (segments - 1) * options.rank
                    )
                    * segments
                    * options.rank
                    * 8
                    if route != "block-banded"
                    else segments * options.samples * options.rank * 8
                    + 2 * (segments - 1) * options.rank**2 * 8,
                }
            )
            jax.clear_caches()
    write_json_atomic(
        options.output,
        {
            "identity": capture_benchmark_identity(
                Path.cwd(), Path(__file__), ("cases", "environment")
            ).to_dict(),
            "environment": capture_environment().to_dict(),
            "cases": cases,
            "observed_peak_rss_bytes": process_peak_rss_bytes(),
            "compiler_scaling_exponents": {
                route: fitted_scaling_exponent(
                    options.segments,
                    [
                        max(case["compiler"]["temporary_bytes"], 1)
                        for case in cases
                        if case["route"] == route
                    ],
                )
                for route in options.routes
            },
        },
    )


if __name__ == "__main__":
    main()
