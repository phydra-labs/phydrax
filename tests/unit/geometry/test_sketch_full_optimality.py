#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.geometry.design import FixedPoint, Sketch
from phydrax.geometry.design._solve import solve_prepared_sketch_residual
from phydrax.optim import (
    LevenbergMarquardt,
    NonlinearLeastSquaresProblem,
    OptimizationTermination,
    least_squares,
)


@pytest.mark.parametrize("partial", [False, True])
def test_reduced_candidate_requires_full_weighted_stationarity(partial: bool) -> None:
    # Two fixed targets differ by 5e-16, each with weight 1e12. At the
    # first target the residual norm is 5e-10 but the gradient is -5e-4.
    def residual(x: jax.Array) -> jax.Array:
        fixed = jnp.array(
            [1e6 * x[0], 1e6 * (x[0] - 5e-16), 1e6 * x[1]], dtype=jnp.float64
        )
        return jnp.concatenate((fixed, x[2:]))

    initial = jnp.zeros(3 if partial else 2, dtype=jnp.float64)
    pinned = (0, 1)
    tolerance = 1e-9
    candidate_residual = residual(initial)
    assert float(jnp.linalg.norm(candidate_residual)) < tolerance
    assert abs(float(1e6 * candidate_residual[1])) > tolerance

    actual, reduced = solve_prepared_sketch_residual(
        residual,
        initial,
        initial,
        pinned,
        initial[:2],
        (),
        (),
        max_iterations=100,
        tolerance=tolerance,
        damping=1e-8,
        problem_id="full-optimality",
        bounds=None,
    )
    oracle = least_squares(
        NonlinearLeastSquaresProblem(
            lambda x, _args: residual(x), problem_id="full-optimality-oracle"
        ),
        initial,
        method=LevenbergMarquardt(initial_damping=1e-8),
        termination=OptimizationTermination(
            absolute_optimality=tolerance, relative_optimality=0.0, maximum_steps=100
        ),
    )
    assert not reduced
    np.testing.assert_array_equal(actual.coordinates, oracle.parameters)
    np.testing.assert_array_equal(actual.residual, oracle.residual)
    assert int(actual.status) == int(oracle.status)


@pytest.mark.parametrize("partial", [False, True])
def test_sketch_fixed_targets_match_native_unreduced_solve(partial: bool) -> None:
    points = jnp.zeros((2 if partial else 1, 2), dtype=jnp.float64)
    sketch = Sketch(
        points,
        constraints=(
            FixedPoint(0, jnp.array([0.0, 0.0], dtype=jnp.float64), weight=1e12),
            FixedPoint(0, jnp.array([5e-16, 0.0], dtype=jnp.float64), weight=1e12),
        ),
    )
    initial = points.reshape((-1,))
    oracle = least_squares(
        NonlinearLeastSquaresProblem(
            lambda x, _args: sketch.residual(
                x.reshape(points.shape), sketch.circle_radii
            ),
            problem_id="fixed-target-oracle",
        ),
        initial,
        method=LevenbergMarquardt(initial_damping=1e-8),
        termination=OptimizationTermination(
            absolute_optimality=1e-9, relative_optimality=0.0, maximum_steps=100
        ),
    )
    actual = sketch.solve(max_iterations=100)
    assert not actual.preparation.reduced_root_accepted
    np.testing.assert_array_equal(actual.points.reshape((-1,)), oracle.parameters)
    np.testing.assert_array_equal(actual.residual, oracle.residual)
    assert int(actual.status) == int(oracle.status)


def test_bounded_reduced_candidate_uses_projected_stationarity() -> None:
    def residual(x: jax.Array) -> jax.Array:
        return jnp.array([1e6 * (x[0] + 5e-16)], dtype=jnp.float64)

    initial = jnp.array([0.0], dtype=jnp.float64)
    actual, reduced = solve_prepared_sketch_residual(
        residual,
        initial,
        initial,
        (0,),
        initial,
        (),
        (),
        max_iterations=100,
        tolerance=1e-9,
        damping=1e-8,
        problem_id="bounded-full-optimality",
        bounds=(
            jnp.array([0.0], dtype=jnp.float64),
            jnp.array([jnp.inf], dtype=jnp.float64),
        ),
    )
    assert reduced
    assert bool(actual.converged)
    np.testing.assert_array_equal(actual.coordinates, initial)
