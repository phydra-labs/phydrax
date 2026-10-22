#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from phydrax.linalg import (
    condition_target_covariance,
    ConditionTargetBound,
    ConditionTargetPolicy,
)


@pytest.mark.parametrize("bound", ["gershgorin", "trace"])
def test_condition_target_bounds_normalized_not_raw_covariance(
    bound: ConditionTargetBound,
) -> None:
    scale = np.asarray([1e-3, 2.0, 1e3], dtype=np.float64)
    correlation = np.asarray(
        [[1.0, 0.7, 0.2], [0.7, 1.0, 0.3], [0.2, 0.3, 1.0]],
        dtype=np.float64,
    )
    covariance = scale[:, None] * correlation * scale[None, :]
    policy = ConditionTargetPolicy(12.0, bound=bound)
    result = condition_target_covariance(jnp.asarray(covariance), policy)
    actual = np.asarray(result.covariance)
    normalized = actual / scale[:, None] / scale[None, :]
    eigenvalues = np.linalg.eigvalsh(normalized)
    actual_condition = eigenvalues[-1] / eigenvalues[0]
    expected_upper = (
        np.max(np.sum(np.abs(correlation), axis=1))
        if bound == "gershgorin"
        else np.trace(correlation)
    )
    expected_eta = expected_upper / (policy.max_condition_number - 1.0)
    expected_diagonal = expected_eta * np.diag(covariance)

    np.testing.assert_allclose(result.normalized_upper_bound, expected_upper, rtol=1e-12)
    np.testing.assert_allclose(result.eta, expected_eta, rtol=1e-12)
    np.testing.assert_allclose(result.applied_diagonal, expected_diagonal, rtol=1e-12)
    np.testing.assert_allclose(
        actual, covariance + np.diag(expected_diagonal), rtol=1e-12
    )
    np.testing.assert_allclose(result.normalization_scale, scale, rtol=1e-12)
    assert actual_condition <= float(result.normalized_condition_bound) * (1.0 + 1e-12)
    assert float(result.normalized_condition_bound) <= 12.0 * (1.0 + 1e-12)
    assert np.linalg.cond(actual) > 1e8
    assert result.coordinate_system == "diagonal-normalized"
    assert bool(result.property_evidence.successful)
    assert result.property_evidence.properties.certifies("positive_semidefinite")
    np.testing.assert_allclose(result.property_evidence.matrix, correlation, rtol=1e-15)
    if bound == "gershgorin":
        assert result.normalized_row_sums is not None
        np.testing.assert_allclose(
            result.normalized_row_sums,
            np.sum(np.abs(correlation), axis=1),
            rtol=1e-15,
        )
        np.testing.assert_array_equal(
            result.normalized_upper_bound, jnp.max(result.normalized_row_sums)
        )
    else:
        assert result.normalized_row_sums is None


@pytest.mark.parametrize("bound", ["gershgorin", "trace"])
def test_condition_target_accepts_singular_psd_with_positive_diagonal(
    bound: ConditionTargetBound,
) -> None:
    covariance = jnp.asarray([[1.0, 1.0], [1.0, 1.0]], dtype=jnp.float64)
    result = condition_target_covariance(
        covariance, ConditionTargetPolicy(10.0, bound=bound)
    )
    eigenvalues = np.linalg.eigvalsh(np.asarray(result.covariance))
    assert eigenvalues[0] > 0.0
    np.testing.assert_allclose(eigenvalues[-1] / eigenvalues[0], 10.0, rtol=1e-12)
    np.testing.assert_allclose(result.covariance[0, 1], 1.0)
    assert float(result.normalized_condition_bound) > 10.0
    assert eigenvalues[-1] / eigenvalues[0] <= float(result.normalized_condition_bound)


@pytest.mark.parametrize("bound", ["gershgorin", "trace"])
def test_condition_target_singleton_is_truthful_noop(bound: ConditionTargetBound) -> None:
    covariance = jnp.asarray([[3.0]], dtype=jnp.float64)
    result = condition_target_covariance(
        covariance, ConditionTargetPolicy(10.0, bound=bound)
    )
    np.testing.assert_array_equal(result.covariance, covariance)
    np.testing.assert_array_equal(result.applied_diagonal, [0.0])
    assert float(result.eta) == 0.0
    assert float(result.normalized_condition_bound) == 1.0
    np.testing.assert_allclose(result.normalized_upper_bound, 1.0, rtol=1e-15)


@pytest.mark.parametrize("target", [0.0, 1.0, -2.0, np.inf, np.nan])
def test_condition_target_rejects_invalid_targets(target: float) -> None:
    with pytest.raises(ValueError, match="finite and greater than one"):
        ConditionTargetPolicy(target)


def test_condition_target_selector_is_parsed_and_policy_is_immutable() -> None:
    with pytest.raises(ValueError, match="bound"):
        ConditionTargetPolicy(10.0, bound="unknown")  # ty: ignore[invalid-argument-type]
    policy = ConditionTargetPolicy(10.0)
    with pytest.raises(AttributeError):
        policy.max_condition_number = 20.0


@pytest.mark.parametrize(
    "covariance",
    [
        [[1.0, 2.0], [2.0, 1.0]],
        [[-1.0, 0.0], [0.0, 1.0]],
        [[1.0, 0.3], [0.2, 1.0]],
        [[1.0, np.nan], [np.nan, 1.0]],
        [[np.inf, 0.0], [0.0, 1.0]],
        [[1e20, 0.0, 0.0], [0.0, 1.0, 2.0], [0.0, 2.0, 1.0]],
    ],
)
def test_condition_target_refuses_invalid_covariance(
    covariance: list[list[float]],
) -> None:
    with pytest.raises(
        eqx.EquinoxRuntimeError,
        match="finite symmetric positive-semidefinite|strictly positive diagonal",
    ):
        result = condition_target_covariance(
            jnp.asarray(covariance, dtype=jnp.float64), ConditionTargetPolicy(10.0)
        )
        jax.block_until_ready(result.covariance)


@pytest.mark.parametrize("covariance", [[[0.0, 0.0], [0.0, 1.0]], [[0.0]]])
def test_condition_target_distinguishes_zero_diagonal_psd_from_invalid_psd(
    covariance: list[list[float]],
) -> None:
    with pytest.raises(eqx.EquinoxRuntimeError, match="zero-diagonal PSD covariances"):
        result = condition_target_covariance(
            jnp.asarray(covariance, dtype=jnp.float64), ConditionTargetPolicy(10.0)
        )
        jax.block_until_ready(result.covariance)


def test_condition_target_dtype_precision_refusal() -> None:
    covariance = jnp.asarray([[2.0, 0.2], [0.2, 1.0]], dtype=jnp.float32)
    with pytest.raises(ValueError, match="meaningful precision"):
        condition_target_covariance(covariance, ConditionTargetPolicy(1e8))
    with pytest.raises(eqx.EquinoxRuntimeError, match="cannot certify"):
        result = condition_target_covariance(
            jnp.ones((2, 2), dtype=jnp.float32), ConditionTargetPolicy(1e6)
        )
        jax.block_until_ready(result.covariance)
    with pytest.raises(eqx.EquinoxRuntimeError, match="representably"):
        result = condition_target_covariance(
            covariance, ConditionTargetPolicy(1.0 + 1e-10)
        )
        jax.block_until_ready(result.covariance)
    with pytest.raises(TypeError, match="float32 and float64"):
        condition_target_covariance(
            covariance.astype(jnp.float16), ConditionTargetPolicy(10.0)
        )
    with pytest.raises(TypeError, match="real floating-point"):
        condition_target_covariance(
            covariance.astype(jnp.complex64), ConditionTargetPolicy(10.0)
        )
    with pytest.raises(TypeError, match="real floating-point"):
        condition_target_covariance(
            jnp.asarray([[2, 0], [0, 1]], dtype=jnp.int32), ConditionTargetPolicy(10.0)
        )


def test_condition_target_requires_square_nonempty_matrix_and_explicit_policy() -> None:
    policy = ConditionTargetPolicy(10.0)
    for covariance in (
        jnp.ones((2, 3), dtype=jnp.float64),
        jnp.ones((2,), dtype=jnp.float64),
        jnp.ones((0, 0), dtype=jnp.float64),
        jnp.ones((1, 2, 2), dtype=jnp.float64),
    ):
        with pytest.raises(ValueError, match="nonempty square"):
            condition_target_covariance(covariance, policy)
    with pytest.raises(TypeError):
        condition_target_covariance(jnp.eye(2))  # ty: ignore[missing-argument]


@pytest.mark.parametrize("bound", ["gershgorin", "trace"])
def test_condition_target_differentiates_eta_and_scaling_away_from_kinks(
    bound: ConditionTargetBound,
) -> None:
    base = jnp.asarray(
        [[4.0, 0.3, 0.7], [0.3, 1.0, 0.2], [0.7, 0.2, 2.0]], dtype=jnp.float64
    )
    direction = jnp.asarray(
        [[0.3, 0.1, -0.1], [0.1, -0.2, 0.05], [-0.1, 0.05, 0.4]],
        dtype=jnp.float64,
    )
    weights = jnp.arange(9, dtype=jnp.float64).reshape(3, 3) / 9.0
    policy = ConditionTargetPolicy(15.0, bound=bound)

    def objective(parameter: Array) -> Array:
        result = condition_target_covariance(base + parameter * direction, policy)
        return (
            jnp.sum(result.covariance * weights)
            + 0.4 * result.eta
            + 0.1 * jnp.sum(result.normalization_scale)
            + 0.2 * jnp.sum(result.applied_diagonal)
        )

    point = jnp.asarray(0.05, dtype=jnp.float64)
    step = jnp.asarray(1e-4, dtype=jnp.float64)
    finite_difference = (objective(point + step) - objective(point - step)) / (2.0 * step)
    derivative = jax.grad(objective)(point)
    _, tangent = jax.jvp(objective, (point,), (jnp.ones_like(point),))
    np.testing.assert_allclose(derivative, finite_difference, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(tangent, finite_difference, rtol=1e-7, atol=1e-9)

    def conditioned_covariance(covariance: Array) -> Array:
        return condition_target_covariance(covariance, policy).covariance

    compiled = jax.jit(conditioned_covariance)(base)
    np.testing.assert_allclose(
        compiled, condition_target_covariance(base, policy).covariance, rtol=1e-12
    )


def test_condition_target_preserves_off_diagonal_zeros_and_does_not_repair() -> None:
    covariance = jnp.asarray(
        [[2.0, 0.0, 0.3], [0.0, 1.0, 0.2], [0.3 + 1e-14, 0.2, 3.0]],
        dtype=jnp.float64,
    )
    result = condition_target_covariance(covariance, ConditionTargetPolicy(10.0))
    off_diagonal = ~np.eye(3, dtype=np.bool_)
    np.testing.assert_array_equal(
        np.asarray(result.covariance)[off_diagonal],
        np.asarray(covariance)[off_diagonal],
    )
    assert float(result.property_evidence.hermitian_defect) > 0.0


def test_condition_target_trace_evidence_accounts_for_numerical_psd_negativity() -> None:
    size = 8
    covariance = np.full((size, size), 1.0 + 5e-14, dtype=np.float64)
    np.fill_diagonal(covariance, 1.0)
    input_eigenvalues = np.linalg.eigvalsh(covariance)
    ideal_upper = np.trace(covariance)
    policy = ConditionTargetPolicy(20.0, bound="trace")
    result = condition_target_covariance(jnp.asarray(covariance), policy)

    assert input_eigenvalues[0] < 0.0
    assert input_eigenvalues[-1] > ideal_upper
    assert bool(result.property_evidence.successful)
    np.testing.assert_allclose(result.normalized_upper_bound, ideal_upper, rtol=1e-15)
    expected_eta = ideal_upper / (policy.max_condition_number - 1.0)
    np.testing.assert_allclose(result.eta, expected_eta, rtol=1e-15)

    lower = min(input_eigenvalues[0] - float(result.property_evidence.tolerance), 0.0)
    evidence_upper = ideal_upper - (size - 1) * lower
    expected_bound = (evidence_upper + expected_eta) / (expected_eta + lower)
    np.testing.assert_allclose(
        result.normalized_condition_bound, expected_bound, rtol=2e-14
    )
    output_eigenvalues = np.linalg.eigvalsh(np.asarray(result.covariance))
    actual_condition = output_eigenvalues[-1] / output_eigenvalues[0]
    assert actual_condition <= float(result.normalized_condition_bound)
    assert float(result.normalized_condition_bound) > policy.max_condition_number
