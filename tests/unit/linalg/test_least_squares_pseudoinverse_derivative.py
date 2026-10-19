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
from phydrax.linalg._dense_pseudoinverse import (
    apply_connected_pseudoinverse,
    apply_pseudoinverse,
    DensePseudoinverseFactors,
    factor_pseudoinverse,
)


# Empirical `test_connected_gram_matches_dense_global_rank_cutoff_and_condition`: matched max |delta|=0; joint allowance fraction=0, margin=100%.
_CONNECTED_GRAM_PSEUDOINVERSE_ACTION_RTOL = 1e-12
# Empirical `test_connected_gram_matches_dense_global_rank_cutoff_and_condition`: matched max |delta|=0; joint allowance fraction=0, margin=100%.
_CONNECTED_GRAM_CONDITION_ESTIMATE_RTOL = 1e-12
# Empirical `test_connected_gram_preserves_ragged_blocks_and_exact_tiny_links`: matched max |delta|=8.88178e-15; joint allowance fraction=3.56568e-05, margin=99.9964%.
_RAGGED_GRAM_PSEUDOINVERSE_ACTION_RTOL = 1e-10
# Empirical `test_connected_gram_preserves_ragged_blocks_and_exact_tiny_links`: matched max |delta|=8.88178e-15; joint allowance fraction=3.56568e-05, margin=99.9964%.
_RAGGED_GRAM_PSEUDOINVERSE_ACTION_ATOL = 1e-12
# Empirical `test_connected_gram_nonfinite_and_true_null_modes_refuse_or_match_dense`: matched max |delta|=0; joint allowance fraction=0, margin=100%.
_NULL_MODE_PSEUDOINVERSE_ACTION_RTOL = 1e-12
# Empirical `test_connected_gram_nonfinite_and_true_null_modes_refuse_or_match_dense`: matched max |delta|=0; joint allowance fraction=0, margin=100%.
_NULL_MODE_PSEUDOINVERSE_ACTION_ATOL = 1e-14


_RNG = np.random.default_rng(1001)
_STEP = 1e-6


@pytest.mark.parametrize(
    ("largest", "floor", "cutoff", "retained", "solution", "condition"),
    (
        (0.5, None, 0.1875, (True, True, False), (2.0, 4.0, 0.0), 2.0),
        (0.5, 1.0, 0.3125, (True, False, False), (2.0, 0.0, 0.0), 1.0),
        (2.0, 1.0, 0.5625, (True, False, False), (0.5, 0.0, 0.0), 1.0),
    ),
    ids=("default-scale", "omitted-unit-scale", "computed-scale-larger-than-floor"),
)
@pytest.mark.parametrize("compiled", (False, True), ids=("eager", "jit"))
def test_pseudoinverse_rank_scale_floor_preserves_additive_policy_and_condition(
    largest: float,
    floor: float | None,
    cutoff: float,
    retained: tuple[bool, bool, bool],
    solution: tuple[float, float, float],
    condition: float,
    compiled: bool,
) -> None:
    # Binary diagonal entries make the original absolute + relative cutoff and
    # the retained reciprocal action independently exact for every case.
    matrix = jnp.diag(jnp.asarray((largest, 0.25, 0.125), dtype=jnp.float64))
    policy = la.RankPolicy(relative_cutoff=0.25, absolute_cutoff=0.0625)

    def factor(value: Array) -> DensePseudoinverseFactors:
        return factor_pseudoinverse(value, policy, hermitian=True, rank_scale_floor=floor)

    execute = jax.jit(factor) if compiled else factor
    factors = execute(matrix)
    np.testing.assert_array_equal(factors.rank_cutoff, cutoff)
    np.testing.assert_array_equal(factors.retained, retained)
    np.testing.assert_array_equal(factors.condition_estimate, condition)
    np.testing.assert_array_equal(
        apply_pseudoinverse(factors, jnp.ones((3,), dtype=jnp.float64)), solution
    )


@pytest.mark.parametrize("floor", (-1.0, np.inf, np.nan))
def test_pseudoinverse_rejects_invalid_rank_scale_floor(floor: float) -> None:
    with pytest.raises(ValueError, match="rank_scale_floor"):
        factor_pseudoinverse(
            jnp.eye(2, dtype=jnp.float64),
            la.RankPolicy(),
            rank_scale_floor=floor,
        )


def test_connected_gram_matches_dense_global_rank_cutoff_and_condition() -> None:
    # Separate diagonal components have radically unequal scales. A per-block
    # relative cutoff would incorrectly retain the small scalar component.
    matrix = jnp.diag(jnp.asarray((1e6, 2.0, 1e-4, 0.0)))
    rhs = jnp.asarray((1.0, 2.0, 3.0, 0.0))
    policy = la.RankPolicy(relative_cutoff=1e-6, absolute_cutoff=0.0)
    dense = factor_pseudoinverse(matrix, policy, hermitian=True)
    result = jax.jit(
        lambda a, b: apply_connected_pseudoinverse(
            a, b, policy, jnp.asarray((True, True, True, False)), jnp.asarray(1_000_000)
        )
    )(matrix, rhs)
    assert bool(result.finite)
    assert not bool(result.resource_refused)
    np.testing.assert_allclose(
        result.value,
        apply_pseudoinverse(dense, rhs),
        rtol=_CONNECTED_GRAM_PSEUDOINVERSE_ACTION_RTOL,
        atol=0.0,
    )
    np.testing.assert_array_equal(result.rank, dense.rank)
    np.testing.assert_allclose(result.rank_cutoff, dense.rank_cutoff, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        result.condition_estimate,
        dense.condition_estimate,
        rtol=_CONNECTED_GRAM_CONDITION_ESTIMATE_RTOL,
        atol=0.0,
    )
    assert float(result.value[2]) == 0.0


def test_connected_gram_preserves_ragged_blocks_and_exact_tiny_links() -> None:
    matrix = jnp.zeros((32, 32)).at[0, 0].set(2.0)
    block = jnp.eye(23) + 0.1 * jnp.ones((23, 23))
    matrix = matrix.at[1:24, 1:24].set(block)
    rhs = jnp.arange(32, dtype=jnp.float64)
    rhs = rhs.at[24:].set(0.0)
    active = jnp.arange(32) < 24
    policy = la.RankPolicy(relative_cutoff=1e-10, absolute_cutoff=1e-14)
    result = apply_connected_pseudoinverse(
        matrix, rhs, policy, active, jnp.asarray(10_000_000)
    )
    dense = factor_pseudoinverse(matrix, policy, hermitian=True)
    assert bool(result.finite)
    assert int(result.factor_blocks) == 2
    assert sorted(
        np.asarray(result.component_sizes)[
            np.asarray(result.component_sizes) > 0
        ].tolist()
    ) == [1, 23]
    np.testing.assert_allclose(
        result.value,
        apply_pseudoinverse(dense, rhs),
        rtol=_RAGGED_GRAM_PSEUDOINVERSE_ACTION_RTOL,
        atol=_RAGGED_GRAM_PSEUDOINVERSE_ACTION_ATOL,
    )
    tiny = jnp.asarray(((1.0, 1e-16), (1e-16, 1.0)))
    linked = apply_connected_pseudoinverse(
        tiny, jnp.ones(2), policy, jnp.ones(2, dtype=jnp.bool_), jnp.asarray(1_000_000)
    )
    assert int(linked.factor_blocks) == 1
    assert int(linked.component_sizes[0]) == 2


@pytest.mark.parametrize("resource", (0, 600))
def test_connected_gram_resource_refusal_records_only_executed_graph(
    resource: int,
) -> None:
    result = apply_connected_pseudoinverse(
        jnp.eye(4),
        jnp.ones(4),
        la.RankPolicy(),
        jnp.ones(4, dtype=jnp.bool_),
        jnp.asarray(resource),
    )
    assert bool(result.resource_refused)
    assert not bool(result.finite)
    assert int(result.work_units) <= resource
    np.testing.assert_array_equal(result.value, np.zeros(4))


def test_connected_gram_nonfinite_and_true_null_modes_refuse_or_match_dense() -> None:
    policy = la.RankPolicy(relative_cutoff=1e-10, absolute_cutoff=1e-14)
    matrix = jnp.asarray(
        (
            (1.0, 1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 3.0, 0.0),
            (0.0, 0.0, 0.0, 0.0),
        )
    )
    rhs = jnp.asarray((1.0, 1.0, 1.0, 0.0))
    active = jnp.asarray((True, True, True, False))
    result = apply_connected_pseudoinverse(
        matrix, rhs, policy, active, jnp.asarray(1_000_000)
    )
    dense = factor_pseudoinverse(matrix, policy, hermitian=True)
    assert bool(result.finite)
    assert int(result.rank) == 2
    np.testing.assert_allclose(
        result.value,
        apply_pseudoinverse(dense, rhs),
        rtol=_NULL_MODE_PSEUDOINVERSE_ACTION_RTOL,
        atol=_NULL_MODE_PSEUDOINVERSE_ACTION_ATOL,
    )
    failed = apply_connected_pseudoinverse(
        matrix.at[0, 0].set(jnp.nan), rhs, policy, active, jnp.asarray(1_000_000)
    )
    assert not bool(failed.finite)
    assert not bool(failed.resource_refused)


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
