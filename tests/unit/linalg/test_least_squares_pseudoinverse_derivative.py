# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Mathematical derivatives of dense SVD least-squares solves ``x = A^+ b``.

Oracles are central finite differences of host NumPy pseudoinverses along
fixed-rank paths; they share no code with the native fixed-rank tangent.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax.linalg as la


_RNG = np.random.default_rng(1001)
_STEP = 1e-6


def _central_difference(function: Callable[[float], np.ndarray]) -> np.ndarray:
    return (function(_STEP) - function(-_STEP)) / (2.0 * _STEP)


def _solve_value(
    matrix: Array, rhs: Array, policy: la.LinearSolvePolicy
) -> la.LinearSolveResult:
    return la.solve(
        la.LeastSquaresProblem(la.DenseLinearOperator(matrix)), rhs, policy=policy
    )


# Batched underdetermined full-row-rank designs: x = A^T (A A^T)^-1 b, whose
# derivative has a nullspace term the row-space projection alone omits.
_UNDER = _RNG.standard_normal((2, 3, 5))
_UNDER_DIRECTION = _RNG.standard_normal((2, 3, 5))
_UNDER_RHS = _RNG.standard_normal((2, 3))
_UNDER_PROBE = _RNG.standard_normal((2, 5))


def _under_oracle(parameter: float) -> np.ndarray:
    matrix = _UNDER + parameter * _UNDER_DIRECTION
    return np.asarray(
        [
            _UNDER_PROBE[index] @ np.linalg.pinv(matrix[index]) @ _UNDER_RHS[index]
            for index in range(2)
        ]
    )


@pytest.mark.parametrize("mode", ("reverse", "forward"))
def test_underdetermined_full_row_rank_derivative_matches_finite_differences(
    mode: str,
) -> None:
    policy = la.LinearSolvePolicy(
        la.DenseSVD(),
        rank=la.RankPolicy(relative_cutoff=1e-12, require_full_rank=True),
    )

    def objective(parameter: Array) -> Array:
        matrix = jnp.asarray(_UNDER) + parameter * jnp.asarray(_UNDER_DIRECTION)
        value = _solve_value(matrix, jnp.asarray(_UNDER_RHS), policy).value
        return jnp.sum(jnp.asarray(_UNDER_PROBE) * value, axis=-1)

    parameter = jnp.asarray(0.0)
    derivative = (
        jax.jacrev(objective)(parameter)
        if mode == "reverse"
        else jax.jvp(objective, (parameter,), (jnp.asarray(1.0),))[1]
    )
    np.testing.assert_allclose(
        np.asarray(derivative), _central_difference(_under_oracle), rtol=1e-6, atol=1e-8
    )


# Rank-two 5x4 design on a fixed-rank path with an inconsistent right-hand
# side, so every term of d(A^+) b (range, cokernel, and nullspace) contributes.
_LEFT = _RNG.standard_normal((5, 2))
_RIGHT = _RNG.standard_normal((4, 2))
_LEFT_DIRECTION = _RNG.standard_normal((5, 2))
_RIGHT_DIRECTION = _RNG.standard_normal((4, 2))
_DEFICIENT_RHS = _RNG.standard_normal(5)
_RHS_DIRECTION = _RNG.standard_normal(5)
_DEFICIENT_PROBE = _RNG.standard_normal(4)


def _deficient_matrix(parameter: float) -> np.ndarray:
    return (_LEFT + parameter * _LEFT_DIRECTION) @ (
        _RIGHT + parameter * _RIGHT_DIRECTION
    ).T


def _deficient_oracle(parameter: float) -> np.ndarray:
    rhs = _DEFICIENT_RHS + parameter * _RHS_DIRECTION
    return np.asarray(
        _DEFICIENT_PROBE @ np.linalg.pinv(_deficient_matrix(parameter), rcond=1e-10) @ rhs
    )


def test_rank_deficient_derivative_with_certified_gap_matches_finite_differences() -> (
    None
):
    policy = la.LinearSolvePolicy(la.DenseSVD(), rank=la.RankPolicy(relative_cutoff=1e-8))

    def objective(parameter: Array) -> Array:
        left = jnp.asarray(_LEFT) + parameter * jnp.asarray(_LEFT_DIRECTION)
        right = jnp.asarray(_RIGHT) + parameter * jnp.asarray(_RIGHT_DIRECTION)
        rhs = jnp.asarray(_DEFICIENT_RHS) + parameter * jnp.asarray(_RHS_DIRECTION)
        return (
            jnp.asarray(_DEFICIENT_PROBE)
            @ _solve_value(left @ right.T, rhs, policy).value
        )

    result = _solve_value(
        jnp.asarray(_deficient_matrix(0.0)), jnp.asarray(_DEFICIENT_RHS), policy
    )
    assert int(result.diagnostics.rank) == 2
    assert result.derivative_regular is not None
    assert bool(result.derivative_regular)
    assert bool(result.derivative_valid)
    np.testing.assert_allclose(
        float(jax.grad(objective)(jnp.asarray(0.0))),
        float(_central_difference(_deficient_oracle)),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        float(jax.jvp(objective, (jnp.asarray(0.0),), (jnp.asarray(1.0),))[1]),
        float(_central_difference(_deficient_oracle)),
        rtol=1e-6,
    )


def test_operator_derivative_is_refused_at_a_rank_changing_point() -> None:
    # A(t) = [[1, 0, 0], [0, t, 0]] changes rank at t = 0; A^+ is not
    # differentiable there and the default cutoff cannot certify a gap.
    policy = la.LinearSolvePolicy(la.DenseSVD())
    rhs = jnp.asarray([1.0, 2.0])

    def matrix(parameter: Array) -> Array:
        return jnp.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]) + parameter * jnp.asarray(
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        )

    def objective(parameter: Array) -> Array:
        return jnp.sum(_solve_value(matrix(parameter), rhs, policy).value)

    result = _solve_value(matrix(jnp.asarray(0.0)), rhs, policy)
    assert result.derivative_regular is not None
    assert not bool(result.derivative_regular)
    assert not bool(result.derivative_valid)
    assert np.isnan(float(jax.grad(objective)(jnp.asarray(0.0))))
    # Away from the rank change the same map is certified and differentiable:
    # x = (1, 2 / t, 0) so d(sum x)/dt = -2 / t^2.
    np.testing.assert_allclose(
        float(jax.grad(objective)(jnp.asarray(0.5))), -8.0, rtol=1e-8
    )


def test_rhs_only_least_squares_derivative_needs_no_rank_gap() -> None:
    policy = la.LinearSolvePolicy(
        la.DenseSVD(), differentiation=la.DifferentiationPolicy("rhs-only")
    )
    matrix = jnp.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    gradient = jax.grad(lambda rhs: jnp.sum(_solve_value(matrix, rhs, policy).value))(
        jnp.asarray([1.0, 2.0])
    )
    np.testing.assert_allclose(
        np.asarray(gradient), np.linalg.pinv(np.asarray(matrix)).T @ np.ones(3)
    )
