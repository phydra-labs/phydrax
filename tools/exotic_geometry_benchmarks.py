#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
from time import perf_counter

import equinox as eqx
import jax
import jax.numpy as jnp

import phydrax as phx


def _benchmark(function, argument, *, repeats):
    compiled = eqx.filter_jit(function)
    start = perf_counter()
    output = jax.block_until_ready(compiled(argument))
    first = perf_counter() - start
    start = perf_counter()
    for _ in range(repeats):
        output = jax.block_until_ready(compiled(argument))
    steady = (perf_counter() - start) / repeats
    return {
        "compile_and_first_seconds": first,
        "steady_seconds": steady,
        "output_bytes": int(output.size * output.dtype.itemsize),
    }


def run_benchmarks(*, dimension=2, repeats=10):
    chart = phx.metrix.CoordinateChart(
        "exotic-benchmark",
        tuple([f"x{i}" for i in range(dimension)] + [f"y{i}" for i in range(dimension)]),
    )
    convention = phx.metrix.ComplexCoordinateConvention(chart)
    measure = phx.metrix.WeightedRiemannianMeasure(
        phx.metrix.euclidean_metric(chart), lambda point: -0.5 * jnp.dot(point, point)
    )
    bigraded = phx.metrix.BigradedForm(
        lambda point: jnp.asarray([jnp.sum(convention.to_complex(point))]),
        convention=convention,
        bidegree=(0, 0),
    )
    information = phx.metrix.InformationMetricOperator(
        lambda vector: 2.0 * vector,
        jnp.zeros((2 * dimension,)),
        metric_id="benchmark-information",
    )
    point = jnp.linspace(0.1, 0.4, 2 * dimension)
    projective = phx.geometry.complex.ComplexProjectiveAtlas(dimension)
    records = {
        "weighted_laplacian": _benchmark(
            lambda value: measure.laplacian(lambda q: jnp.dot(q, q), value),
            point,
            repeats=repeats,
        ),
        "dolbeault": _benchmark(
            lambda value: phx.metrix.partial_bar(bigraded)(value),
            point,
            repeats=repeats,
        ),
        "information_action": _benchmark(
            information.mv,
            point,
            repeats=repeats,
        ),
        "fubini_study_metric": _benchmark(
            projective.metric(0),
            point,
            repeats=repeats,
        ),
    }
    return {
        "backend": jax.default_backend(),
        "jax_version": jax.__version__,
        "dimension": dimension,
        "repeats": repeats,
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimension", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    result = run_benchmarks(
        dimension=1 if arguments.smoke else arguments.dimension,
        repeats=1 if arguments.smoke else arguments.repeats,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
