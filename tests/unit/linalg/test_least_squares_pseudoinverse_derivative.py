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


@pytest.mark.parametrize("shape", ((5, 3), (3, 5)), ids=("tall", "wide"))
@pytest.mark.parametrize("complex_values", (False, True))
def test_weighted_regularized_multi_rhs_stationarity_derivative(
    shape: tuple[int, int], complex_values: bool
) -> None:
    rng = np.random.default_rng(109)
    rows, columns = shape
    matrix = rng.normal(size=(2, rows, columns))
    direction = rng.normal(size=matrix.shape)
    rhs = rng.normal(size=(2, rows, 3))
    rhs_direction = rng.normal(size=rhs.shape)
    regularizer = np.broadcast_to(0.7 * np.eye(columns), (2, columns, columns)).copy()
    regularizer_direction = 0.1 * rng.normal(size=regularizer.shape)
    if complex_values:
        matrix = matrix + 1j * rng.normal(size=matrix.shape)
        direction = direction + 1j * rng.normal(size=direction.shape)
        rhs = rhs + 1j * rng.normal(size=rhs.shape)
        regularizer = regularizer.astype(np.complex128)
        regularizer_direction = regularizer_direction + 0.1j * rng.normal(
            size=regularizer.shape
        )
    weights = rng.uniform(0.5, 1.5, size=(2, rows))
    weight_direction = rng.normal(size=weights.shape) * 0.1
    probe = rng.normal(size=(2, columns, 3))
    if complex_values:
        probe = probe + 1j * rng.normal(size=probe.shape)
    policy = la.LinearSolvePolicy(
        la.DenseSVD(damping=0.2),
        tolerance=la.TolerancePolicy(relative=1e-11, absolute=1e-13, max_steps=50),
    )

    def native(parameter: Array) -> Array:
        problem = la.LeastSquaresProblem(
            la.DenseLinearOperator(
                jnp.asarray(matrix) + parameter * jnp.asarray(direction)
            ),
            weights=jnp.asarray(weights) + parameter * jnp.asarray(weight_direction),
            regularizer=la.DenseLinearOperator(
                jnp.asarray(regularizer) + parameter * jnp.asarray(regularizer_direction)
            ),
        )
        return la.solve(
            problem,
            jnp.asarray(rhs) + parameter * jnp.asarray(rhs_direction),
            policy=policy,
            rhs_layout=la.RHSLayout((3,)),
        ).value

    def reference(parameter: float) -> np.ndarray:
        a = matrix + parameter * direction
        r = regularizer + parameter * regularizer_direction
        w = weights + parameter * weight_direction
        b = rhs + parameter * rhs_direction
        ah = a.conj().swapaxes(-1, -2)
        return np.linalg.solve(
            (ah * w[:, None, :]) @ a
            + r.conj().swapaxes(-1, -2) @ r
            + 0.04 * np.eye(columns),
            (ah * w[:, None, :]) @ b,
        )

    parameter = jnp.asarray(0.0)
    value, tangent = jax.jvp(native, (parameter,), (jnp.asarray(1.0),))
    expected_tangent = _central_difference(reference)
    np.testing.assert_allclose(value, reference(0.0), rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(tangent, expected_tangent, rtol=1e-6, atol=1e-8)
    derivative = jax.grad(lambda t: jnp.real(jnp.vdot(jnp.asarray(probe), native(t))))(
        parameter
    )
    np.testing.assert_allclose(
        derivative, np.real(np.vdot(probe, expected_tangent)), rtol=1e-6, atol=1e-8
    )


@pytest.mark.parametrize("shape", ((7, 3), (3, 7)))
@pytest.mark.parametrize("complex_values", (False, True))
def test_materialized_fixed_rank_pseudoinverse_derivative(
    shape: tuple[int, int], complex_values: bool
) -> None:
    rng = np.random.default_rng(817)
    left = rng.normal(size=(shape[0], 2))
    right = rng.normal(size=(shape[1], 2))
    dl = rng.normal(size=left.shape)
    dr = rng.normal(size=right.shape)
    probe = rng.normal(size=shape[::-1])
    if complex_values:
        left = left + 1j * rng.normal(size=left.shape)
        right = right + 1j * rng.normal(size=right.shape)
        dl = dl + 1j * rng.normal(size=dl.shape)
        dr = dr + 1j * rng.normal(size=dr.shape)
        probe = probe + 1j * rng.normal(size=probe.shape)

    def native(parameter: Array) -> Array:
        matrix = (jnp.asarray(left) + parameter * jnp.asarray(dl)) @ (
            jnp.asarray(right) + parameter * jnp.asarray(dr)
        ).conj().T
        factorization = la.factorize(
            la.DenseLinearOperator(matrix),
            la.FactorizationPolicy("svd", rank=la.RankPolicy(relative_cutoff=1e-10)),
        )
        result = factorization.materialize_pseudoinverse()
        return result.value

    def reference(parameter: float) -> np.ndarray:
        return np.linalg.pinv(
            (left + parameter * dl) @ (right + parameter * dr).conj().T, rcond=1e-10
        )

    value, tangent = jax.jvp(native, (jnp.asarray(0.0),), (jnp.asarray(1.0),))
    expected = _central_difference(reference)
    np.testing.assert_allclose(value, reference(0.0), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(tangent, expected, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(
        jax.grad(lambda t: jnp.real(jnp.vdot(jnp.asarray(probe), native(t))))(
            jnp.asarray(0.0)
        ),
        np.real(np.vdot(probe, expected)),
        rtol=1e-6,
        atol=1e-8,
    )
