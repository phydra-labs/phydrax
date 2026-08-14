#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp

import phydrax as phx


def _metric(dimension: int) -> phx.metrix.RiemannianMetric:
    chart = phx.metrix.CoordinateChart(
        f"benchmark_{dimension}d",
        tuple(f"q{index}" for index in range(dimension)),
    )

    def matrix(coordinates):
        direction = jnp.sin(coordinates)
        diagonal = 1.5 + coordinates**2
        return jnp.diag(diagonal) + 0.05 * jnp.outer(direction, direction)

    return phx.metrix.RiemannianMetric(matrix, chart=chart)


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
) -> tuple[Any, dict[str, float | int]]:
    compiled = jax.jit(function)
    started = time.perf_counter()
    output = _block(compiled(argument))
    compile_and_first_ms = 1e3 * (time.perf_counter() - started)

    started = time.perf_counter()
    for _ in range(repeats):
        output = _block(compiled(argument))
    steady_ms = 1e3 * (time.perf_counter() - started) / repeats
    return output, {
        "compile_and_first_ms": compile_and_first_ms,
        "steady_ms": steady_ms,
        "output_bytes": _output_bytes(output),
    }


def _jet_outputs(metric: phx.metrix.RiemannianMetric, coordinates: jax.Array):
    jet = phx.metrix.metric_jet(metric, coordinates, order=2)
    assert jet.first_derivative is not None
    assert jet.second_derivative is not None
    return (
        jet.matrix,
        jet.inverse,
        jet.determinant,
        jet.volume_density,
        jet.log_volume_density,
        jet.first_derivative,
        jet.second_derivative,
    )


def _repeated_outputs(metric: phx.metrix.RiemannianMetric, coordinates: jax.Array):
    matrix_function = metric.matrix_function
    first_derivative = jax.jacfwd(matrix_function)
    return (
        metric(coordinates),
        metric.inverse(coordinates),
        jnp.linalg.det(metric(coordinates)),
        metric.volume_density(coordinates),
        metric.log_volume_density(coordinates),
        first_derivative(coordinates),
        jax.jacfwd(first_derivative)(coordinates),
    )


def _maximum_difference(left: Any, right: Any) -> float:
    differences = [
        jnp.max(jnp.abs(left_leaf - right_leaf))
        for left_leaf, right_leaf in zip(
            jax.tree.leaves(left),
            jax.tree.leaves(right),
            strict=True,
        )
    ]
    return float(jax.device_get(jnp.max(jnp.stack(differences))))


def run_state_geometry_benchmarks(
    dimensions: Sequence[int] = (2, 3, 8),
    /,
    *,
    repeats: int = 10,
) -> dict[str, Any]:
    """Benchmark on-manifold SO(n) and SPD(n) retraction kernels."""
    if repeats < 1:
        raise ValueError("repeats must be at least one.")
    records: list[dict[str, Any]] = []
    for dimension in dimensions:
        dimension_ = int(dimension)
        if dimension_ < 2:
            raise ValueError("state-geometry benchmark dimensions must be at least two.")
        raw = jnp.arange(dimension_**2, dtype=float).reshape((dimension_, dimension_))
        skew = 1e-2 * (raw - raw.T)
        symmetric = 5e-3 * (raw + raw.T)
        identity = jnp.eye(dimension_)
        so = phx.metrix.SpecialOrthogonalStateGeometry(dimension_)
        spd = phx.metrix.SymmetricPositiveDefiniteStateGeometry(dimension_)
        so_output, so_timing = _benchmark(
            lambda local: so.retract(identity, local),
            skew,
            repeats=repeats,
        )
        spd_output, spd_timing = _benchmark(
            lambda local: spd.retract(2.0 * identity, local),
            symmetric,
            repeats=repeats,
        )
        records.append(
            {
                "dimension": dimension_,
                "so_exponential": {
                    **so_timing,
                    "geometry_id": so.geometry_id,
                    "orthogonality_error": float(
                        jax.device_get(
                            jnp.max(jnp.abs(so_output.T @ so_output - identity))
                        )
                    ),
                    "determinant": float(jax.device_get(jnp.linalg.det(so_output))),
                },
                "spd_congruence_exponential": {
                    **spd_timing,
                    "geometry_id": spd.geometry_id,
                    "minimum_eigenvalue": float(
                        jax.device_get(jnp.min(jnp.linalg.eigvalsh(spd_output)))
                    ),
                },
            }
        )
    return {
        "jax_version": jax.__version__,
        "backend": jax.default_backend(),
        "repeats": repeats,
        "records": records,
    }


def run_benchmarks(
    dimensions: Sequence[int] = (2, 3, 8, 16),
    /,
    *,
    repeats: int = 10,
) -> dict[str, Any]:
    if repeats < 1:
        raise ValueError("repeats must be at least one.")
    records: list[dict[str, Any]] = []
    for dimension in dimensions:
        dimension_ = int(dimension)
        if dimension_ < 1:
            raise ValueError("benchmark dimensions must be positive.")
        metric = _metric(dimension_)
        coordinates = jnp.linspace(0.1, 0.8, dimension_)
        fused, fused_timing = _benchmark(
            lambda point: _jet_outputs(metric, point),
            coordinates,
            repeats=repeats,
        )
        repeated, repeated_timing = _benchmark(
            lambda point: _repeated_outputs(metric, point),
            coordinates,
            repeats=repeats,
        )
        records.append(
            {
                "dimension": dimension_,
                "fused_metric_jet": fused_timing,
                "repeated_metric_evaluation": repeated_timing,
                "maximum_absolute_difference": _maximum_difference(fused, repeated),
                "steady_speedup": (
                    repeated_timing["steady_ms"] / fused_timing["steady_ms"]
                ),
            }
        )
    return {
        "jax_version": jax.__version__,
        "backend": jax.default_backend(),
        "repeats": repeats,
        "records": records,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark Metrix metric jets or state-geometry retractions."
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        nargs="+",
        default=(2, 3, 8, 16),
    )
    parser.add_argument(
        "--state-geometry",
        action="store_true",
        help="Benchmark SO(n) and SPD(n) retractions instead of metric jets.",
    )
    parser.add_argument("--repeats", type=int, default=10)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    runner = run_state_geometry_benchmarks if arguments.state_geometry else run_benchmarks
    result = runner(arguments.dimensions, repeats=arguments.repeats)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
