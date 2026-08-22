#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def test_quadratic_program_native_bounds_preserve_public_constraint_axes():
    problem = phx.optim.QuadraticProgram(
        jnp.eye(2),
        jnp.asarray([-2.0, -4.0]),
        inequality_matrix=jnp.asarray([[1.0, 1.0]]),
        inequality_rhs=jnp.asarray([2.5]),
        bounds=phx.optim.Bounds(jnp.zeros(2), jnp.asarray([1.0, jnp.inf])),
    )
    result = phx.optim.solve_quadratic_program(problem)

    assert result.inequality_dual.shape == (1,)
    assert result.inequality_slack.shape == (1,)
    assert result.lower_bound_dual.shape == (2,)
    assert result.upper_bound_dual.shape == (2,)
    assert result.upper_bound_dual[0] > 0.0
    assert result.status == phx.optim.ConvexProgramStatus.OPTIMAL
    assert result.primal_residual_norm < 1e-7
    expected_gap = jnp.sum(result.inequality_slack * result.inequality_dual)
    expected_gap += jnp.sum(
        jnp.where(
            jnp.isfinite(problem.lower_bounds),
            (result.primal - problem.lower_bounds) * result.lower_bound_dual,
            0.0,
        )
    )
    expected_gap += jnp.sum(
        jnp.where(
            jnp.isfinite(problem.upper_bounds),
            (problem.upper_bounds - result.primal) * result.upper_bound_dual,
            0.0,
        )
    )
    np.testing.assert_allclose(result.complementarity_gap, expected_gap, atol=1e-12)


def test_fixed_bound_is_exposed_as_bound_dual_not_public_equality():
    problem = phx.optim.QuadraticProgram(
        jnp.eye(1),
        jnp.zeros(1),
        bounds=phx.optim.Bounds(2.0, 2.0),
    )
    result = phx.optim.solve_quadratic_program(problem)

    np.testing.assert_allclose(result.primal, [2.0], atol=1e-10)
    assert result.equality_dual.shape == (0,)
    assert result.lower_bound_dual.shape == (1,)
    assert result.upper_bound_dual.shape == (1,)
    np.testing.assert_allclose(
        result.upper_bound_dual - result.lower_bound_dual,
        [-2.0],
        atol=1e-10,
    )


def test_bounds_reject_inverted_or_batch_varying_roles():
    with pytest.raises(
        (RuntimeError, ValueError),
        match="Lower bounds must not exceed upper bounds",
    ):
        phx.optim.QuadraticProgram(
            jnp.eye(1),
            jnp.zeros(1),
            bounds=phx.optim.Bounds(2.0, 1.0),
        )

    with pytest.raises(ValueError, match="shared finite/fixed role pattern"):
        phx.optim.QuadraticProgram(
            jnp.broadcast_to(jnp.eye(1), (2, 1, 1)),
            jnp.zeros((2, 1)),
            bounds=phx.optim.Bounds(
                jnp.asarray([[0.0], [-jnp.inf]]),
                jnp.ones((2, 1)),
            ),
        )


def test_bounded_quadratic_solution_is_jittable_and_differentiable():
    bounds = phx.optim.Bounds(0.0, 1.0)

    def solution(target):
        problem = phx.optim.QuadraticProgram(
            jnp.eye(1),
            -jnp.asarray([target]),
            bounds=bounds,
        )
        return phx.optim.solve_quadratic_program_primal(problem)[0]

    compiled = jax.jit(solution)
    np.testing.assert_allclose(compiled(2.0), 1.0, atol=2e-5)
    np.testing.assert_allclose(jax.grad(solution)(2.0), 0.0, atol=2e-5)
