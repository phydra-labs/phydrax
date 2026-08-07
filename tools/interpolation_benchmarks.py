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
    bspline_evaluate,
    bspline_stencil,
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


def _stencil_bytes(stencil: Any, /) -> int:
    arrays = (
        stencil.indices,
        stencil.weights,
        stencil.valid,
        stencil.support,
    )
    return sum(int(array.size * array.dtype.itemsize) for array in arrays)


def _open_uniform_knots(control_count: int, degree: int, /) -> jax.Array:
    interior_count = control_count - degree - 1
    interior = jnp.linspace(0.0, 1.0, interior_count + 2)[1:-1]
    return jnp.concatenate(
        (
            jnp.zeros((degree + 1,)),
            interior,
            jnp.ones((degree + 1,)),
        )
    )


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
    bspline_query = jnp.linspace(0.0, 1.0, 8192)
    for degree in (1, 3, 5):
        for control_count in (32, 257, 4096):
            knots = _open_uniform_knots(control_count, degree)
            values = jnp.linspace(
                -1.0,
                1.0,
                control_count * 8,
            ).reshape((control_count, 8))
            key = f"bspline_dynamic_p{degree}_c{control_count}"
            records[key] = _benchmark(
                lambda query, knots=knots, values=values, degree=degree: (
                    apply_gather_stencil(
                        values,
                        bspline_stencil(
                            knots,
                            query,
                            degree=degree,
                        ),
                    ).values
                ),
                bspline_query,
                repeats=repeats,
            )
            plan = bspline_stencil(knots, bspline_query, degree=degree)
            records[key]["plan_bytes"] = _stencil_bytes(plan)
            records[key]["dense_plan_bytes"] = int(
                bspline_query.size * control_count * bspline_query.dtype.itemsize
            )

    representative_knots = _open_uniform_knots(257, 3)
    representative_values = jnp.linspace(-1.0, 1.0, 257 * 8).reshape((257, 8))
    representative_stencil = bspline_stencil(
        representative_knots,
        bspline_query,
        degree=3,
    )
    records["bspline_precomputed_p3_c257"] = _benchmark(
        lambda values: (
            apply_gather_stencil(
                values,
                representative_stencil,
            ).values
        ),
        representative_values,
        repeats=repeats,
    )
    records["bspline_precomputed_p3_c257"]["plan_bytes"] = _stencil_bytes(
        representative_stencil
    )

    rows = jnp.arange(bspline_query.size, dtype=jnp.int32)[:, None]
    dense_basis = (
        jnp.zeros((bspline_query.size, 257))
        .at[
            rows,
            representative_stencil.indices,
        ]
        .add(representative_stencil.weights)
    )
    records["bspline_dense_apply_p3_c257"] = _benchmark(
        lambda values: dense_basis @ values,
        representative_values,
        repeats=repeats,
    )
    records["bspline_dense_apply_p3_c257"]["plan_bytes"] = int(
        dense_basis.size * dense_basis.dtype.itemsize
    )
    local_reference = apply_gather_stencil(
        representative_values,
        representative_stencil,
    ).values
    dense_reference = dense_basis @ representative_values
    records["bspline_dense_apply_p3_c257"]["relative_error"] = float(
        jnp.linalg.norm(local_reference - dense_reference)
        / jnp.linalg.norm(dense_reference)
    )

    def representative_curve(query):
        return bspline_evaluate(
            representative_knots,
            representative_values,
            query,
            degree=3,
        ).values

    records["bspline_query_jvp_p3_c257"] = _benchmark(
        lambda query: jax.jvp(
            representative_curve,
            (query,),
            (jnp.ones_like(query),),
        )[1],
        bspline_query,
        repeats=repeats,
    )
    records["bspline_explicit_d1_p3_c257"] = _benchmark(
        lambda query: (
            bspline_evaluate(
                representative_knots,
                representative_values,
                query,
                degree=3,
                derivative_order=1,
            ).values
        ),
        bspline_query,
        repeats=repeats,
    )
    jvp_reference = jax.jvp(
        representative_curve,
        (bspline_query,),
        (jnp.ones_like(bspline_query),),
    )[1]
    explicit_reference = bspline_evaluate(
        representative_knots,
        representative_values,
        bspline_query,
        degree=3,
        derivative_order=1,
    ).values
    records["bspline_query_jvp_p3_c257"]["relative_error"] = float(
        jnp.linalg.norm(jvp_reference - explicit_reference)
        / jnp.linalg.norm(explicit_reference)
    )
    records["bspline_coefficient_gradient_p3_c257"] = _benchmark(
        jax.grad(
            lambda values: jnp.sum(
                bspline_evaluate(
                    representative_knots,
                    values,
                    bspline_query,
                    degree=3,
                ).values
                ** 2
            )
        ),
        representative_values,
        repeats=repeats,
    )

    scalar_curve = lambda query: (
        bspline_evaluate(
            representative_knots,
            representative_values[:, 0],
            query,
            degree=3,
        ).values
    )
    records["bspline_query_hessian_p3_c257"] = _benchmark(
        jax.vmap(jax.grad(jax.grad(scalar_curve))),
        bspline_query,
        repeats=repeats,
    )

    case_query = jnp.broadcast_to(
        jnp.linspace(0.0, 1.0, 2048),
        (4, 2048),
    )
    case_values = jnp.broadcast_to(
        representative_values,
        (4, *representative_values.shape),
    )
    case_stencil = bspline_stencil(
        representative_knots,
        case_query,
        degree=3,
        case_shape=(4,),
    )
    records["bspline_precomputed_cases_p3_c257"] = _benchmark(
        lambda values: apply_gather_stencil(values, case_stencil).values,
        case_values,
        repeats=repeats,
    )
    records["bspline_precomputed_cases_p3_c257"]["plan_bytes"] = _stencil_bytes(
        case_stencil
    )
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
