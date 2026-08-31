#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

import phydrax as phx
from benchmarks._runtime import logical_array_bytes, measure_repeated


def _measure(operation: Callable[[], Any], /, *, repeats: int) -> tuple[Any, float]:
    result, distribution = measure_repeated(
        operation,
        warmup=1,
        repeats=repeats,
    )
    return result, 1_000.0 * float(distribution.mean_seconds)


def run_benchmarks(
    *,
    point_count: int,
    grid_size: int,
    time_count: int,
    repeats: int,
) -> dict[str, Any]:
    """Benchmark canonical point/grid materialization and field evaluation."""
    geometry = phx.domain.GeometryDomain(
        phx.geometry.Square(
            center=(0.0, 0.0),
            side=2.0,
            feature_id="domain-benchmark-square",
        ).compile()
    )
    domain = geometry @ phx.domain.TimeInterval(0.0, 1.0)
    component = domain.component()
    point_plan = phx.domain.PointSampling(
        point_count,
        layout=phx.domain.SampleLayout((("x", "t"),)),
        design=phx.sampling.LatinHypercubeDesign(),
    )
    grid_plan = phx.domain.GridSampling(
        {"x": (grid_size, grid_size)},
        dense=phx.domain.PointSampling(
            time_count,
            layout=phx.domain.SampleLayout((("t",),)),
            design=phx.sampling.SobolDesign(scrambled=True),
        ),
    )

    @domain.Function("x", "t")
    def field(x, t):
        return jnp.exp(-jnp.sum(x**2)) * jnp.cos(2.0 * jnp.pi * t)

    sample_points = eqx.filter_jit(lambda key: component.sample(point_plan, key=key))
    sample_grid = eqx.filter_jit(lambda key: component.sample(grid_plan, key=key))
    point_batch, point_sampling_ms = _measure(
        lambda: sample_points(jr.key(0)),
        repeats=repeats,
    )
    grid_batch, grid_sampling_ms = _measure(
        lambda: sample_grid(jr.key(1)),
        repeats=repeats,
    )

    evaluate_points = eqx.filter_jit(lambda batch: field(batch).data)
    evaluate_grid = eqx.filter_jit(lambda batch: field(batch).data)
    point_values, point_evaluation_ms = _measure(
        lambda: evaluate_points(point_batch),
        repeats=repeats,
    )
    grid_values, grid_evaluation_ms = _measure(
        lambda: evaluate_grid(grid_batch),
        repeats=repeats,
    )

    return {
        "configuration": {
            "point_count": point_count,
            "grid_size": grid_size,
            "time_count": time_count,
            "repeats": repeats,
        },
        "point": {
            "value_shape": list(point_values.shape),
            "working_set_bytes": logical_array_bytes(point_batch),
            "sampling_mean_ms": point_sampling_ms,
            "evaluation_mean_ms": point_evaluation_ms,
            "checksum": float(jnp.sum(point_values)),
        },
        "grid": {
            "value_shape": list(grid_values.shape),
            "working_set_bytes": logical_array_bytes(grid_batch),
            "sampling_mean_ms": grid_sampling_ms,
            "evaluation_mean_ms": grid_evaluation_ms,
            "checksum": float(jnp.sum(grid_values)),
        },
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the canonical Phydrax domain substrate."
    )
    parser.add_argument("--point-count", type=int, default=16_384)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--time-count", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args(argv)
    if min(args.point_count, args.grid_size, args.time_count, args.repeats) <= 0:
        parser.error("all benchmark sizes and repeats must be positive")
    print(
        json.dumps(
            run_benchmarks(
                point_count=args.point_count,
                grid_size=args.grid_size,
                time_count=args.time_count,
                repeats=args.repeats,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
