#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import numpy as np
import pytest

from phydrax import linalg as la


def _nonnormal_matrix() -> jax.Array:
    return jnp.asarray([[-1.0, 8.0, -2.0], [0.0, -2.0, 3.0], [0.0, 0.0, -4.0]])


def test_taylor_exponential_matches_dense_reference_and_reuses_preparation():
    matrix = _nonnormal_matrix()
    operator = la.DenseLinearOperator(matrix, operator_id="taylor-nonnormal")
    policy = la.TaylorExponentialPolicy(error_tolerance=1e-10)
    first = jnp.asarray([1.0, -0.5, 0.25])
    second = jnp.asarray([-0.25, 0.75, 1.5])
    scale = jnp.asarray(0.4)

    prepared = la.prepare_taylor_exponential_action(operator, policy)
    first_result = la.matrix_exponential_action(prepared, first, scale)
    second_result = la.matrix_exponential_action(prepared, second, -scale)

    assert bool(first_result.successful)
    assert bool(second_result.successful)
    assert first_result.provenance.prepared_id == prepared.prepared_id
    assert first_result.provenance.norm_source == "exact-stored-coordinates"
    assert jnp.allclose(
        first_result.value,
        jsp.linalg.expm(scale * matrix) @ first,
        rtol=2e-9,
        atol=2e-10,
    )
    assert jnp.allclose(
        second_result.value,
        jsp.linalg.expm(-scale * matrix) @ second,
        rtol=2e-9,
        atol=2e-10,
    )


def test_taylor_zero_rhs_jvp_is_full_exponential_action():
    matrix = _nonnormal_matrix()
    operator = la.DenseLinearOperator(matrix, operator_id="taylor-zero-jvp")
    policy = la.TaylorExponentialPolicy(error_tolerance=1e-11)
    tangent = jnp.asarray([0.5, -1.0, 2.0])
    scale = jnp.asarray(0.3)

    def action(vector):
        return la.matrix_exponential_action(
            operator,
            vector,
            scale,
            policy=policy,
        ).value

    primal, derivative = jax.jvp(action, (jnp.zeros_like(tangent),), (tangent,))
    assert jnp.array_equal(primal, jnp.zeros_like(primal))
    assert jnp.allclose(
        derivative,
        jsp.linalg.expm(scale * matrix) @ tangent,
        rtol=2e-9,
        atol=2e-10,
    )


def test_taylor_estimated_planning_requires_key_and_is_replayable():
    matrix = _nonnormal_matrix()
    space = la.ArraySpace((3,), dtype=matrix.dtype)
    operator = la.FunctionLinearOperator(
        lambda vector: matrix @ vector,
        source=space,
        target=space,
        operator_id="taylor-matrix-free",
    )
    policy = la.TaylorExponentialPolicy(
        error_tolerance=1e-8,
        norm_mode="estimate",
    )
    vector = jnp.asarray([1.0, 2.0, -1.0])

    with pytest.raises(ValueError, match="key"):
        la.prepare_taylor_exponential_action(operator, policy)

    first = la.prepare_taylor_exponential_action(operator, policy, key=jr.key(7))
    replay = la.prepare_taylor_exponential_action(operator, policy, key=jr.key(7))
    assert jnp.array_equal(first.norm_one, replay.norm_one)
    result = la.matrix_exponential_action(first, vector, 0.2)
    assert bool(result.successful)
    assert not bool(result.diagnostics.error_bound_available)
    assert jnp.allclose(
        result.value,
        jsp.linalg.expm(0.2 * matrix) @ vector,
        rtol=2e-6,
        atol=2e-7,
    )


def test_taylor_resource_refusal_is_explicit():
    matrix = _nonnormal_matrix()
    operator = la.DenseLinearOperator(matrix, operator_id="taylor-resource-refusal")
    policy = la.TaylorExponentialPolicy(
        resources=la.TaylorExponentialResourcePolicy(
            max_degree=1,
            max_scaling_count=1,
            max_action_matvec_count=1,
        )
    )
    result = la.matrix_exponential_action(
        operator,
        jnp.ones((3,)),
        100.0,
        policy=policy,
    )
    assert result.status == int(la.MatrixFunctionStatus.RESOURCE_EXHAUSTED)
    assert not bool(result.successful)
    assert jnp.all(jnp.isnan(result.value))


def test_augmented_exponential_phi_combination_matches_dense_block_reference():
    matrix = _nonnormal_matrix()
    operator = la.DenseLinearOperator(matrix, operator_id="augmented-phi")
    state = jnp.asarray([1.0, -0.5, 0.25])
    first = jnp.asarray([0.25, 1.0, -0.75])
    second = jnp.asarray([-1.0, 0.5, 0.125])
    scale = jnp.asarray(0.35)

    result = la.matrix_exponential_phi_combination_action(
        operator,
        (state, first, second),
        scale,
        policy=la.MatrixFunctionPolicy("arnoldi", max_dimension=5, error_tolerance=1e-10),
    )

    augmented = jnp.zeros((5, 5), dtype=matrix.dtype)
    augmented = augmented.at[:3, :3].set(matrix)
    augmented = augmented.at[:3, 3].set(first)
    augmented = augmented.at[:3, 4].set(second)
    augmented = augmented.at[4, 3].set(1.0)
    initial = jnp.concatenate((state, jnp.asarray([1.0, 0.0])))
    expected = (jsp.linalg.expm(scale * augmented) @ initial)[:3]

    assert bool(result.successful)
    assert result.provenance.kind == "exp-phi-combination"
    assert jnp.allclose(result.value, expected, rtol=2e-9, atol=2e-10)


def test_augmented_combination_zero_scale_and_taylor_route():
    matrix = _nonnormal_matrix()
    operator = la.DenseLinearOperator(matrix, operator_id="augmented-phi-taylor")
    state = jnp.asarray([1.0, -0.5, 0.25])
    forcing = jnp.asarray([0.25, 1.0, -0.75])
    policy = la.TaylorExponentialPolicy(error_tolerance=1e-9)

    zero = la.matrix_exponential_phi_combination_action(
        operator,
        (state, forcing),
        0.0,
        policy=policy,
        key=jr.key(11),
    )
    positive = la.matrix_exponential_phi_combination_action(
        operator,
        (state, forcing),
        0.2,
        policy=policy,
        key=jr.key(12),
    )

    augmented = jnp.zeros((4, 4), dtype=matrix.dtype)
    augmented = augmented.at[:3, :3].set(matrix)
    augmented = augmented.at[:3, 3].set(forcing)
    initial = jnp.concatenate((state, jnp.asarray([1.0])))
    expected = (jsp.linalg.expm(0.2 * augmented) @ initial)[:3]

    assert jnp.array_equal(zero.value, state)
    assert bool(zero.successful)
    assert bool(positive.successful)
    assert jnp.allclose(positive.value, expected, rtol=2e-7, atol=2e-8)


def test_taylor_refresh_complex_scale_and_policy_boundaries():
    matrix = _nonnormal_matrix()
    operator = la.DenseLinearOperator(matrix, operator_id="taylor-refresh")
    policy = la.TaylorExponentialPolicy(error_tolerance=1e-10)
    prepared = la.prepare_taylor_exponential_action(operator, policy)
    changed = la.DenseLinearOperator(0.5 * matrix, operator_id="taylor-refresh")
    refreshed = la.refresh_taylor_exponential_action(prepared, changed)
    vector = jnp.asarray([1.0, -0.25, 0.5])

    result = la.matrix_exponential_action(refreshed, vector, 0.2j)
    expected = jsp.linalg.expm(0.1j * matrix) @ vector
    assert refreshed.numeric_version == 1
    assert bool(result.successful)
    assert result.provenance.trace_source == "exact-diagonal"
    assert jnp.allclose(result.value, expected, rtol=2e-9, atol=2e-10)

    with pytest.raises(ValueError, match="Frechet"):
        la.TaylorExponentialPolicy(
            differentiation=la.DifferentiationPolicy("mathematical")
        )
    with pytest.raises(TypeError, match="float32"):
        la.plan_taylor_exponential_action(
            la.DenseLinearOperator(jnp.eye(2, dtype=jnp.float16)),
        )


def test_taylor_rhs_only_differentiates_only_the_right_hand_side():
    vector = jnp.asarray([0.5, -1.0, 2.0])
    tangent = jnp.asarray([1.0, 0.25, -0.5])
    scale = jnp.asarray(0.2)
    policy = la.TaylorExponentialPolicy(
        error_tolerance=1e-10,
        differentiation=la.DifferentiationPolicy("rhs-only"),
    )

    def action(parameter, right_hand_side):
        operator = la.DenseLinearOperator(
            parameter * _nonnormal_matrix(),
            operator_id="taylor-rhs-only",
        )
        return la.matrix_exponential_action(
            operator,
            right_hand_side,
            scale,
            policy=policy,
        ).value

    parameter_gradient = jax.jacrev(lambda parameter: action(parameter, vector))(
        jnp.asarray(1.0)
    )
    _, rhs_derivative = jax.jvp(
        lambda rhs: action(jnp.asarray(1.0), rhs),
        (vector,),
        (tangent,),
    )
    assert jnp.array_equal(parameter_gradient, jnp.zeros_like(parameter_gradient))
    assert jnp.allclose(
        rhs_derivative,
        jsp.linalg.expm(scale * _nonnormal_matrix()) @ tangent,
        rtol=2e-9,
        atol=2e-10,
    )


def test_generated_taylor_thresholds_are_conservative_and_monotone():
    from phydrax.linalg._generated_exponential_taylor_thresholds import (
        TAYLOR_THRESHOLDS,
    )

    assert TAYLOR_THRESHOLDS.shape == (53, 55)
    assert TAYLOR_THRESHOLDS.dtype == np.dtype(np.float64)
    assert np.all(np.isfinite(TAYLOR_THRESHOLDS))
    assert np.all(TAYLOR_THRESHOLDS > 0.0)
    assert np.all(np.diff(TAYLOR_THRESHOLDS, axis=1) >= 0.0)
    assert np.all(np.diff(TAYLOR_THRESHOLDS, axis=0) <= 0.0)
