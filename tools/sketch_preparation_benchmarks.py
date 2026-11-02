#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Phase-separated sketch timings; run without competing benchmark jobs."""

from __future__ import annotations

import argparse
import json
import tracemalloc
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from phydrax.geometry.design._sketch import FixedPoint, Horizontal, PointDistance, Sketch


def scenario(count: int, family: str) -> dict[str, float | int | str]:
    x = np.arange(count + 1, dtype=np.float64)
    points = (
        np.column_stack((x, np.zeros(count + 1, dtype=np.float64)))
        if family == "axis"
        else np.column_stack((3.0 * x, 4.0 * (x % 2)))
    )
    if family == "axis":
        lines = np.column_stack(
            (np.arange(count, dtype=np.int32), np.arange(1, count + 1, dtype=np.int32))
        )
        constraints = (FixedPoint(0, jnp.asarray([0.0, 0.0])),) + tuple(
            Horizontal(index) for index in range(count)
        )
    else:
        lines = np.empty((0, 2), dtype=np.int32)
        edges = tuple((index, index + 1) for index in range(count)) + tuple(
            (index, index + 2) for index in range(count - 1)
        )
        constraints = (FixedPoint(0, jnp.asarray([0.0, 0.0])),) + tuple(
            PointDistance(
                first, second, float(np.linalg.norm(points[first] - points[second]))
            )
            for first, second in edges
        )
    sketch = Sketch(points, lines=lines, constraints=constraints, feature_id="benchmark")
    vector = jnp.asarray(points).reshape((-1,))
    tracemalloc.start()
    start = perf_counter()
    prepared = sketch._prepare_components(vector, sketch.dimensions, 1e-9)
    arrays = (prepared[1], prepared[2], *(item.offsets for item in prepared[5]))
    jax.block_until_ready(arrays)
    preparation_seconds = perf_counter() - start
    retained_host_bytes, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    residual = lambda value: sketch.residual(
        value.reshape(sketch.points.shape), sketch.circle_radii
    )
    start = perf_counter()
    derivative = jax.jacfwd(residual)(vector)
    jax.block_until_ready(derivative)
    derivative_seconds = perf_counter() - start
    start = perf_counter()
    solution = sketch.solve()
    jax.block_until_ready(solution.residual)
    solve_seconds = perf_counter() - start
    return {
        "family": family,
        "constraint_count": len(constraints),
        "preparation_seconds": preparation_seconds,
        "derivative_seconds": derivative_seconds,
        "solve_seconds": solve_seconds,
        "retained_host_bytes": retained_host_bytes,
        "retained_array_bytes": sum(
            value.size * value.dtype.itemsize for value in arrays
        ),
        "residual_norm": float(solution.residual_norm),
        "converged": int(solution.converged),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--counts", nargs="+", type=int, default=[4, 8, 16])
    args = parser.parse_args()
    if any(count < 1 for count in args.counts):
        parser.error("counts must be positive")
    print(
        json.dumps(
            [
                scenario(count, family)
                for family in ("axis", "rigid")
                for count in args.counts
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
