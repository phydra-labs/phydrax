#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp

from phydrax._interpolation import (
    apply_gather_stencil,
    cubic_hermite_interpolate,
    fourier_interpolate,
    fourier_resample,
    inverse_distance_stencil,
    local_cubic_slopes,
    rectilinear_stencil,
)


def _block(tree: Any) -> Any:
    return jax.tree.map(jax.block_until_ready, tree)


def _output_bytes(tree: Any) -> int:
    return sum(int(leaf.size * leaf.dtype.itemsize) for leaf in jax.tree.leaves(tree))


def _benchmark(
    function: Callable[[jax.Array], Any],
    argument: jax.Array,
    /,
    *,
    repeats: int,
) -> dict[str, float | int]:
    compiled = jax.jit(function)
    started = time.perf_counter()
    output = _block(compiled(argument))
    compile_and_first_ms = 1e3 * (time.perf_counter() - started)

    started = time.perf_counter()
    for _ in range(repeats):
        output = _block(compiled(argument))
    steady_ms = 1e3 * (time.perf_counter() - started) / repeats
    return {
        "compile_and_first_ms": compile_and_first_ms,
        "steady_ms": steady_ms,
        "output_bytes": _output_bytes(output),
    }


def run_benchmarks(*, repeats: int = 10) -> dict[str, Any]:
    if repeats < 1:
        raise ValueError("repeats must be at least one.")

    temporal_nodes = jnp.linspace(0.0, 1.0, 257)
    temporal_values = jnp.stack(
        tuple(jnp.sin((index + 1) * jnp.pi * temporal_nodes) for index in range(8)),
        axis=-1,
    )
    temporal_slopes = local_cubic_slopes(temporal_nodes, temporal_values)
    temporal_query = jnp.linspace(0.0, 1.0, 8192)

    source = jnp.linspace(-1.0, 1.0, 4096 * 8).reshape((4096, 8))
    candidate_indices = (
        jnp.arange(8192, dtype=jnp.int32)[:, None]
        + jnp.arange(16, dtype=jnp.int32)[None, :]
    ) % source.shape[0]
    candidate_distances = (0.01 + jnp.arange(16, dtype=float)[None, :]) ** 2
    candidate_distances = jnp.broadcast_to(
        candidate_distances,
        candidate_indices.shape,
    )

    x = jnp.linspace(-1.0, 1.0, 128)
    y = jnp.linspace(-1.0, 1.0, 96)
    xx, yy = jnp.meshgrid(x, y, indexing="ij")
    rectilinear_values = jnp.stack(
        (xx, yy, xx * yy, xx**2 + yy**2),
        axis=-1,
    )
    rectilinear_query = jnp.stack(
        jnp.meshgrid(
            jnp.linspace(-0.99, 0.99, 160),
            jnp.linspace(-0.99, 0.99, 120),
            indexing="ij",
        ),
        axis=-1,
    )

    fourier_values = jnp.stack(
        (
            jnp.sin(2.0 * jnp.pi * xx),
            jnp.cos(2.0 * jnp.pi * yy),
            jnp.sin(2.0 * jnp.pi * (xx + yy)),
            jnp.ones_like(xx),
        ),
        axis=-1,
    )

    point_x = jnp.arange(32, dtype=float) / 32.0
    point_y = jnp.arange(32, dtype=float) / 32.0
    point_xx, point_yy = jnp.meshgrid(point_x, point_y, indexing="ij")
    point_values = jnp.stack(
        (
            jnp.sin(2.0 * jnp.pi * point_xx),
            jnp.cos(4.0 * jnp.pi * point_yy),
        ),
        axis=-1,
    )
    point_query = jax.random.uniform(jax.random.key(21), (10_000, 2))

    point_3d_values = jax.random.normal(jax.random.key(22), (12, 12, 12, 2))
    point_3d_query = jax.random.uniform(jax.random.key(23), (4096, 3))

    records = {
        "temporal_cubic": _benchmark(
            lambda query: (
                cubic_hermite_interpolate(
                    temporal_nodes,
                    temporal_values,
                    query,
                    slopes=temporal_slopes,
                ).values
            ),
            temporal_query,
            repeats=repeats,
        ),
        "local_inverse_distance": _benchmark(
            lambda values: (
                apply_gather_stencil(
                    values,
                    inverse_distance_stencil(
                        candidate_indices,
                        candidate_distances,
                        source_size=int(source.shape[0]),
                        regularization=1e-12,
                    ),
                ).values
            ),
            source,
            repeats=repeats,
        ),
        "rectilinear_2d": _benchmark(
            lambda query: (
                apply_gather_stencil(
                    rectilinear_values.reshape((-1, 4)),
                    rectilinear_stencil(
                        (x, y),
                        query,
                        boundary=("clamp", "clamp"),
                    ),
                ).values
            ),
            rectilinear_query,
            repeats=repeats,
        ),
        "fourier_2d_odd_even": _benchmark(
            lambda values: fourier_resample(values, (160, 121)),
            fourier_values,
            repeats=repeats,
        ),
        "fourier_shifted_2d": _benchmark(
            lambda values: fourier_resample(
                values,
                (160, 121),
                phase_offsets=(0.125, -0.2),
            ),
            fourier_values,
            repeats=repeats,
        ),
        "fourier_points_direct_2d": _benchmark(
            lambda query: (
                fourier_interpolate(
                    point_values,
                    query,
                    spatial_ndim=2,
                ).values
            ),
            point_query,
            repeats=repeats,
        ),
        "fourier_points_nufft_2d": _benchmark(
            lambda query: (
                fourier_interpolate(
                    point_values,
                    query,
                    spatial_ndim=2,
                    method="nufft",
                    tolerance=1e-6,
                    query_chunk_size=2048,
                ).values
            ),
            point_query,
            repeats=repeats,
        ),
        "fourier_points_nufft_3d": _benchmark(
            lambda query: (
                fourier_interpolate(
                    point_3d_values,
                    query,
                    spatial_ndim=3,
                    method="nufft",
                    tolerance=1e-6,
                    query_chunk_size=512,
                ).values
            ),
            point_3d_query,
            repeats=repeats,
        ),
    }
    direct_reference = fourier_interpolate(
        point_values,
        point_query,
        spatial_ndim=2,
    ).values
    nufft_values = fourier_interpolate(
        point_values,
        point_query,
        spatial_ndim=2,
        method="nufft",
        tolerance=1e-6,
        query_chunk_size=2048,
    ).values
    direct_norm = jnp.linalg.norm(direct_reference)
    records["fourier_points_nufft_2d"]["relative_error"] = float(
        jnp.linalg.norm(nufft_values - direct_reference) / direct_norm
    )
    return {
        "jax_version": jax.__version__,
        "backend": jax.default_backend(),
        "repeats": repeats,
        "records": records,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark native interpolation and resampling kernels."
    )
    parser.add_argument("--repeats", type=int, default=10)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    print(json.dumps(run_benchmarks(repeats=arguments.repeats), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
