#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Paired sketch timings; run without competing benchmark jobs."""

from __future__ import annotations

import argparse
import json
import tracemalloc
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from phydrax.geometry.design._sketch import FixedPoint, Horizontal, PointDistance, Sketch
from phydrax.optim import (
    least_squares,
    LevenbergMarquardt,
    NonlinearLeastSquaresProblem,
    OptimizationTermination,
)


def scenario(count: int, family: str, *, perturbed: bool) -> dict[str, object]:
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
    initial_points = points.copy()
    if perturbed:
        initial_points += 0.01 * np.column_stack((np.sin(x + 1.0), np.cos(x + 1.0)))
    vector = jnp.asarray(initial_points).reshape((-1,))
    tracemalloc.start()
    start = perf_counter()
    prepared = sketch._prepare_components(vector, sketch.dimensions, 1e-9)
    arrays = (
        prepared[1],
        prepared[2],
        *(value for item in prepared[5] for value in (item.anchor_target, item.offsets)),
    )
    jax.block_until_ready(arrays)
    preparation_seconds = perf_counter() - start
    retained_host_bytes, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    reduced_preparation = {
        "seconds": preparation_seconds,
        "retained_host_bytes": retained_host_bytes,
        "retained_array_bytes": sum(
            value.size * value.dtype.itemsize for value in arrays
        ),
    }

    # The unreduced arm retains the same source sketch and initial vector, but
    # constructs only the native least-squares problem and solver controls.
    # Neither arm's preparation-memory figure includes those shared inputs.
    def prepare_unreduced() -> tuple[
        NonlinearLeastSquaresProblem, LevenbergMarquardt, OptimizationTermination
    ]:
        problem = NonlinearLeastSquaresProblem(
            lambda value, _args: sketch.residual(
                value.reshape(sketch.points.shape), sketch.circle_radii
            ),
            problem_id="benchmark-unreduced",
        )
        method = LevenbergMarquardt(initial_damping=1e-8)
        termination = OptimizationTermination(
            absolute_optimality=1e-9, relative_optimality=0.0, maximum_steps=32
        )
        return problem, method, termination

    tracemalloc.start()
    start = perf_counter()
    problem, method, termination = prepare_unreduced()
    unreduced_preparation_seconds = perf_counter() - start
    unreduced_host_bytes, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Both standalone derivative measurements are full-residual Jacobians at
    # the common initial point, not an isolated reduced-solver derivative.
    # The solves below are end-to-end and include their own preparation and
    # derivative work; these phase timings must not be added together.
    residual = lambda value: sketch.residual(
        value.reshape(sketch.points.shape), sketch.circle_radii
    )
    if perturbed and float(jnp.linalg.norm(residual(vector))) <= 1e-9:
        raise AssertionError("The perturbed input must require residual correction.")
    start = perf_counter()
    reduced_derivative = jax.jacfwd(residual)(vector)
    jax.block_until_ready(reduced_derivative)
    reduced_derivative_seconds = perf_counter() - start
    start = perf_counter()
    unreduced_derivative = jax.jacfwd(lambda value: problem.residual(value, None))(vector)
    jax.block_until_ready(unreduced_derivative)
    unreduced_derivative_seconds = perf_counter() - start

    start = perf_counter()
    solution = sketch.solve(
        initial_coordinates=vector, max_iterations=32, tolerance=1e-9, damping=1e-8
    )
    jax.block_until_ready(solution.residual)
    reduced_solve_seconds = perf_counter() - start
    start = perf_counter()
    problem, method, termination = prepare_unreduced()
    original = least_squares(problem, vector, method=method, termination=termination)
    jax.block_until_ready(original.residual)
    unreduced_solve_seconds = perf_counter() - start

    reduced_points = np.asarray(solution.points)
    unreduced_points = np.asarray(original.parameters).reshape(points.shape)
    for solved_points, reported_residual in (
        (reduced_points, solution.residual),
        (unreduced_points, original.residual),
    ):
        full_residual = np.asarray(residual(jnp.asarray(solved_points).reshape((-1,))))
        np.testing.assert_allclose(
            full_residual, np.asarray(reported_residual), atol=1e-9, rtol=0
        )
        if not np.all(np.isfinite(full_residual)) or np.linalg.norm(full_residual) > 1e-9:
            raise AssertionError("The full sketch residual did not converge to a root.")
        # Check the geometry independently of the residual implementation.
        np.testing.assert_allclose(solved_points[0], points[0], atol=1e-8, rtol=0)
        if family == "axis":
            np.testing.assert_allclose(
                solved_points[lines[:, 0], 1],
                solved_points[lines[:, 1], 1],
                atol=1e-8,
                rtol=0,
            )
        else:
            for first, second in edges:
                np.testing.assert_allclose(
                    np.linalg.norm(solved_points[first] - solved_points[second]),
                    np.linalg.norm(points[first] - points[second]),
                    atol=1e-8,
                    rtol=0,
                )
    if not bool(solution.converged) or not bool(solution.root_evidence.root_converged):
        raise AssertionError("The prepared sketch did not report a converged root.")
    if not bool(original.successful):
        raise AssertionError("The unreduced native solve did not report success.")

    return {
        "family": family,
        "count": count,
        "initial_state": "perturbed" if perturbed else "exact",
        "constraint_count": len(constraints),
        "timing_note": (
            "Standalone full-residual Jacobians and end-to-end solves overlap; "
            "Both solves include preparation and derivatives. Timings may include "
            "compilation and cache effects; compilation is not isolated. "
            "Reduced arm runs first. "
            "Retained array bytes count logical prepared arrays, not unique device "
            "allocations; shared sketch and initial vector are excluded."
        ),
        "reduced": {
            "preparation": reduced_preparation,
            "full_residual_derivative_seconds": reduced_derivative_seconds,
            "end_to_end_solve_seconds": reduced_solve_seconds,
            "residual_norm": float(solution.residual_norm),
            "converged": bool(solution.converged),
        },
        "unreduced": {
            "preparation": {
                "seconds": unreduced_preparation_seconds,
                "retained_host_bytes": unreduced_host_bytes,
                "retained_array_bytes": 0,
            },
            "full_residual_derivative_seconds": unreduced_derivative_seconds,
            "end_to_end_solve_seconds": unreduced_solve_seconds,
            "residual_norm": float(np.linalg.norm(np.asarray(original.residual))),
            "converged": bool(original.successful),
        },
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
                scenario(count, family, perturbed=perturbed)
                for family in ("axis", "rigid")
                for count in args.counts
                for perturbed in (False, True)
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
