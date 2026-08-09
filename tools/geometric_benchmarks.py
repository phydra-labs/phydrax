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

import phydrax as phx


def _block(tree: Any) -> Any:
    return jax.tree.map(jax.block_until_ready, tree)


def _output_bytes(tree: Any) -> int:
    return sum(
        int(leaf.size * leaf.dtype.itemsize)
        for leaf in jax.tree.leaves(tree)
        if isinstance(leaf, jax.Array)
    )


def _benchmark(
    name: str,
    function: Callable[[jax.Array], Any],
    argument: jax.Array,
    /,
    *,
    repeats: int,
) -> dict[str, Any]:
    compiled = jax.jit(function)
    started = time.perf_counter()
    output = _block(compiled(argument))
    first_seconds = time.perf_counter() - started
    started = time.perf_counter()
    for _ in range(repeats):
        output = _block(compiled(argument))
    steady_seconds = (time.perf_counter() - started) / repeats
    return {
        "name": name,
        "input_shape": tuple(argument.shape),
        "compile_and_first_seconds": first_seconds,
        "steady_seconds": steady_seconds,
        "output_bytes": _output_bytes(output),
    }


def _metric_jet_case(points: jax.Array):
    dimension = points.shape[-1]
    chart = phx.metrix.CoordinateChart(
        f"metric-{dimension}", tuple(f"q{index}" for index in range(dimension))
    )
    metric = phx.metrix.diagonal_metric(
        lambda q: 1.0 + q**2,
        chart=chart,
    )
    jet = phx.metrix.metric_jet(metric, points, order=2)
    return jet.matrix, jet.inverse, jet.first_derivative, jet.second_derivative


def _form_case(points: jax.Array):
    dimension = points.shape[-1]
    chart = phx.metrix.CoordinateChart(
        f"forms-{dimension}", tuple(f"q{index}" for index in range(dimension))
    )
    metric = phx.metrix.RiemannianMetric(lambda q: jnp.eye(dimension), chart=chart)
    form = phx.metrix.DifferentialForm(
        lambda q: q,
        chart=chart,
        degree=1,
    )
    exterior = phx.metrix.exterior_derivative(form)
    dual = phx.metrix.hodge_star(form, metric)
    return exterior(points), dual(points)


def _poisson_case(points: jax.Array):
    dimension = points.shape[-1]
    chart = phx.metrix.CoordinateChart(
        f"phase-{dimension}", tuple(f"z{index}" for index in range(dimension))
    )
    symplectic = phx.metrix.canonical_symplectic_form(chart)
    poisson = phx.metrix.symplectic_to_poisson(symplectic)

    def hamiltonian(point):
        return 0.5 * jnp.dot(point, point)

    return phx.metrix.hamiltonian_vector_field(hamiltonian, poisson, points)


def _horizontal_case(points: jax.Array):
    dimension = points.shape[-1]
    chart = phx.metrix.CoordinateChart(
        f"horizontal-{dimension}", tuple(f"q{index}" for index in range(dimension))
    )
    rank = dimension - 1
    cometric = phx.metrix.HorizontalCometric(
        lambda q: jnp.eye(dimension)[:, :rank],
        chart,
        rank,
    )

    def field(point):
        return jnp.dot(point, point)

    return phx.metrix.sub_laplacian(field, cometric, points)


def _lorentzian_case(points: jax.Array):
    dimension = points.shape[-1]
    chart = phx.metrix.CoordinateChart(
        f"spacetime-{dimension}", tuple(f"q{index}" for index in range(dimension))
    )
    metric = phx.metrix.minkowski_metric(chart)

    def field(point):
        return -(point[0] ** 2) + jnp.sum(point[1:] ** 2)

    return phx.metrix.dalembertian(field, metric, points)


def run_benchmarks(
    *,
    batch_size: int = 256,
    dimension: int = 4,
    repeats: int = 10,
) -> dict[str, Any]:
    """Benchmark representative geometric kernels without external comparisons."""
    if batch_size <= 0 or repeats <= 0:
        raise ValueError("batch_size and repeats must be positive.")
    if dimension < 2:
        raise ValueError("dimension must be at least two.")
    even_dimension = dimension if dimension % 2 == 0 else dimension + 1
    points = jnp.linspace(0.1, 0.8, batch_size * dimension).reshape(batch_size, dimension)
    phase_points = jnp.linspace(0.1, 0.8, batch_size * even_dimension).reshape(
        batch_size, even_dimension
    )
    records = [
        _benchmark("metric_jet", _metric_jet_case, points, repeats=repeats),
        _benchmark("differential_forms", _form_case, points, repeats=repeats),
        _benchmark("horizontal_sub_laplacian", _horizontal_case, points, repeats=repeats),
        _benchmark("lorentzian_dalembertian", _lorentzian_case, points, repeats=repeats),
        _benchmark("poisson_hamiltonian", _poisson_case, phase_points, repeats=repeats),
    ]
    return {
        "jax_version": jax.__version__,
        "backend": jax.default_backend(),
        "batch_size": batch_size,
        "dimension": dimension,
        "repeats": repeats,
        "records": records,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark Phydrax differentiable-geometric kernels."
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--dimension", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--smoke", action="store_true")
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    report = run_benchmarks(
        batch_size=8 if arguments.smoke else arguments.batch_size,
        dimension=4 if arguments.smoke else arguments.dimension,
        repeats=1 if arguments.smoke else arguments.repeats,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
