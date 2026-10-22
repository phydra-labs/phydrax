#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import prod
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jax import Array

import phydrax as phx
from phydrax.uq._gp_backend import (
    exact_gp_conditioner_from_covariances,
    exact_gp_predict_diagonal_from_covariances,
)
from phydrax.uq._gp_kernel_fit import _negative_log_marginal_likelihood


def _design() -> Array:
    coordinate = jnp.array([-1.0, -0.8, -0.3, 0.1, 0.35, 0.7, 1.1], dtype=jnp.float64)
    return jnp.stack((coordinate, coordinate**2 + 0.2 * coordinate), axis=1)


def _execution(points: Array, columns: int) -> phx.uq.ExactGaussianProcessExecutionPolicy:
    count = points.shape[0]
    entries = 8 * count + 16 + columns * (10 * count + 2 * prod(points.shape[1:]) + 8)
    return phx.uq.ExactGaussianProcessExecutionPolicy(
        max_workspace_bytes=entries * points.dtype.itemsize
    )


def _state(
    theta: Array,
    points: Array,
    columns: int,
    conditioning: phx.linalg.ConditionTargetPolicy | None = None,
) -> phx.uq.GaussianProcessLikelihoodState:
    return phx.uq.GaussianProcessLikelihoodState(
        kernel=phx.kernels.AmplitudeKernel(
            phx.kernels.Matern52Kernel(length_scale=jnp.exp(theta[1:3])),
            jnp.exp(theta[0]),
        ),
        noise_scale=jnp.exp(theta[3])
        + jnp.exp(theta[4]) * jnp.linspace(0.1, 1.0, points.shape[0]),
        jitter=jnp.exp(theta[5]),
        execution=_execution(points, columns),
        conditioning=conditioning,
    )


def _dense_log_probability(residual: Array, covariance: Array) -> Array:
    return -0.5 * (
        residual.size * jnp.log(2.0 * jnp.pi)
        + jnp.linalg.slogdet(covariance)[1]
        + residual @ jnp.linalg.solve(covariance, residual)
    )


def _covariance(points: Array, state: phx.uq.GaussianProcessLikelihoodState) -> Array:
    noise = jnp.broadcast_to(state.noise_scale, (points.shape[0],))
    return state.kernel.matrix(points, points) + jnp.diag(noise**2 + state.jitter)


def _theta() -> Array:
    return jnp.array([-0.7, -0.3, 0.1, -2.4, -3.5, -14.0, 0.4, -0.1], dtype=jnp.float64)


@pytest.mark.parametrize("columns", [1, 3, 7])
def test_exact_score_matches_dense_value_reverse_forward_and_higher_derivatives(
    columns: int,
) -> None:
    points = _design()
    observations = jnp.sin(1.7 * points[:, 0]) + 0.1 * points[:, 1]
    model = phx.uq.ExactGaussianProcessDiscrepancy(points, observations)

    def native(theta: Array) -> Array:
        return model.log_marginal_likelihood(
            theta[6] * points[:, 0] + theta[7],
            state=_state(theta, points, columns),
        )

    def reference(theta: Array) -> Array:
        residual = observations - theta[6] * points[:, 0] - theta[7]
        return _dense_log_probability(
            residual, _covariance(points, _state(theta, points, columns))
        )

    theta = _theta()
    tangent = jnp.linspace(-0.2, 0.3, theta.size)
    native_value, native_gradient = jax.jit(jax.value_and_grad(native))(theta)
    dense_value, dense_gradient = jax.value_and_grad(reference)(theta)
    assert _execution(points, columns).column_batch_size(points) == columns
    assert jnp.allclose(native_value, dense_value, rtol=2e-10, atol=2e-10)
    assert jnp.allclose(native_gradient, dense_gradient, rtol=2e-8, atol=2e-9)
    assert jnp.allclose(
        jax.jvp(native, (theta,), (tangent,))[1],
        jax.jvp(reference, (theta,), (tangent,))[1],
        rtol=2e-8,
        atol=2e-9,
    )
    assert jnp.allclose(
        jax.jvp(jax.grad(native), (theta,), (tangent,))[1],
        jax.jvp(jax.grad(reference), (theta,), (tangent,))[1],
        rtol=2e-7,
        atol=2e-8,
    )


def test_fixed_native_factor_residual_derivatives_and_covariance_helpers() -> None:
    points = _design()
    state = _state(_theta(), points, 2)
    factor = phx.uq.ExactGaussianProcessFactor(points, state=state)
    covariance = _covariance(points, state)
    residual = jnp.sin(points[:, 0])
    query = points[::2] + 0.04
    cross = state.kernel.matrix(query, points)
    prior = state.kernel.matrix(query, query)
    projection, conditional, variance = exact_gp_conditioner_from_covariances(
        factor.factorization, cross, prior
    )
    mean, diagonal = exact_gp_predict_diagonal_from_covariances(
        factor.factorization, cross, jnp.diag(prior), residual
    )
    reference_projection = jnp.linalg.solve(covariance, cross.T).T
    reference_covariance = prior - reference_projection @ cross.T
    assert factor.evidence.successful
    assert factor.evidence.derivative_valid
    assert factor.factor_storage_elements >= 2 * points.shape[0] ** 2
    assert factor.factor_storage_bytes <= state.execution.max_factor_storage_bytes
    assert jnp.allclose(
        factor.log_probability(residual), _dense_log_probability(residual, covariance)
    )
    assert jnp.allclose(
        jax.grad(factor.log_probability)(residual),
        -jnp.linalg.solve(covariance, residual),
    )
    assert jnp.allclose(
        jax.hessian(factor.log_probability)(residual),
        -jnp.linalg.solve(covariance, jnp.eye(residual.size)),
    )
    assert jnp.allclose(projection, reference_projection)
    assert jnp.allclose(conditional, reference_covariance)
    assert jnp.allclose(variance, jnp.diag(reference_covariance))
    assert jnp.allclose(mean, reference_projection @ residual)
    assert jnp.allclose(diagonal, variance)


def test_deep_kernel_parameter_mean_and_noise_score_matches_dense() -> None:
    points = _design()[:5]
    observations = jnp.cos(points[:, 0])
    model = phx.uq.ExactGaussianProcessDiscrepancy(points, observations)
    feature_map = eqx.nn.MLP(
        in_size=2,
        out_size=2,
        width_size=3,
        depth=1,
        activation=jnp.tanh,
        key=jax.random.key(23),
    )
    parameters = (feature_map, jnp.array([0.2, -2.0], dtype=jnp.float64))

    def state(
        candidate: tuple[eqx.nn.MLP, Array],
    ) -> phx.uq.GaussianProcessLikelihoodState:
        transform, theta = candidate
        return phx.uq.GaussianProcessLikelihoodState(
            kernel=phx.kernels.AmplitudeKernel(
                phx.kernels.InputTransformedKernel(
                    phx.kernels.SquaredExponentialKernel(
                        length_scale=jnp.array([0.7, 1.3])
                    ),
                    transform,
                    transform_id="exact-score-deep-kernel",
                    max_derivative_order=None,
                ),
                0.4,
            ),
            noise_scale=jnp.exp(theta[1]),
            execution=_execution(points, 2),
        )

    def native(candidate: tuple[eqx.nn.MLP, Array]) -> Array:
        return model.log_marginal_likelihood(
            candidate[1][0] * points[:, 0], state=state(candidate)
        )

    def reference(candidate: tuple[eqx.nn.MLP, Array]) -> Array:
        residual = observations - candidate[1][0] * points[:, 0]
        return _dense_log_probability(residual, _covariance(points, state(candidate)))

    value, gradient = eqx.filter_value_and_grad(native)(parameters)
    expected_value, expected_gradient = eqx.filter_value_and_grad(reference)(parameters)
    leaves = jax.tree_util.tree_leaves(eqx.filter(gradient, eqx.is_inexact_array))
    expected_leaves = jax.tree_util.tree_leaves(
        eqx.filter(expected_gradient, eqx.is_inexact_array)
    )
    assert jnp.allclose(value, expected_value, rtol=2e-10, atol=2e-10)
    assert leaves
    for actual, expected in zip(leaves, expected_leaves, strict=True):
        assert jnp.allclose(actual, expected, rtol=2e-7, atol=2e-8)


@pytest.mark.parametrize("bound", ["gershgorin", "trace"])
def test_condition_target_uses_complete_covariance_and_preserves_exact_score(
    bound: Literal["gershgorin", "trace"],
) -> None:
    points = _design()
    observations = jnp.sin(points[:, 0])
    model = phx.uq.ExactGaussianProcessDiscrepancy(points, observations)
    conditioning = phx.linalg.ConditionTargetPolicy(50.0, bound=bound)

    def native(theta: Array) -> Array:
        return model.log_marginal_likelihood(
            theta[6] * points[:, 0] + theta[7],
            state=_state(theta, points, 2, conditioning),
        )

    def reference(theta: Array) -> Array:
        covariance = _covariance(points, _state(theta, points, 2))
        scale = jnp.sqrt(jnp.diag(covariance))
        normalized = covariance / scale[:, None] / scale[None, :]
        upper = (
            jnp.trace(normalized)
            if bound == "trace"
            else jnp.max(jnp.sum(jnp.abs(normalized), axis=1))
        )
        eta = upper / (conditioning.max_condition_number - 1.0)
        transformed = covariance + jnp.diag(eta * jnp.diag(covariance))
        residual = observations - theta[6] * points[:, 0] - theta[7]
        return _dense_log_probability(residual, transformed)

    theta = _theta()
    value, gradient = jax.jit(jax.value_and_grad(native))(theta)
    expected_value, expected_gradient = jax.value_and_grad(reference)(theta)
    state = _state(theta, points, 2, conditioning)
    factor = phx.uq.ExactGaussianProcessFactor(points, state=state)
    plain = phx.uq.ExactGaussianProcessFactor(points, state=_state(theta, points, 2))
    evidence = factor.regularization
    assert evidence is not None
    assert plain.regularization is None
    assert evidence.property_evidence.successful
    assert jnp.allclose(
        evidence.normalization_scale**2, jnp.diag(_covariance(points, state))
    )
    assert jnp.allclose(
        evidence.applied_diagonal, evidence.eta * jnp.diag(_covariance(points, state))
    )
    assert jnp.allclose(value, expected_value, rtol=2e-10, atol=2e-10)
    assert jnp.allclose(gradient, expected_gradient, rtol=2e-7, atol=2e-8)
    assert factor.factor_storage_bytes > plain.factor_storage_bytes
    with pytest.raises(ValueError, match="retain"):
        phx.uq.ExactGaussianProcessFactor(
            points,
            state=phx.uq.GaussianProcessLikelihoodState(
                kernel=state.kernel,
                noise_scale=state.noise_scale,
                jitter=state.jitter,
                conditioning=conditioning,
                execution=phx.uq.ExactGaussianProcessExecutionPolicy(
                    max_factor_storage_bytes=factor.factor_storage_bytes - 1
                ),
            ),
        )


def test_exact_resource_refusals_and_native_failure_are_explicit() -> None:
    points = _design()
    minimum = _execution(points, 1).max_workspace_bytes
    with pytest.raises(ValueError, match="workspace"):
        phx.uq.ExactGaussianProcessFactor(
            points,
            state=phx.uq.GaussianProcessLikelihoodState(
                noise_scale=0.1,
                execution=phx.uq.ExactGaussianProcessExecutionPolicy(
                    max_workspace_bytes=minimum - 1
                ),
            ),
        )
    with pytest.raises(ValueError, match="retained"):
        phx.uq.ExactGaussianProcessFactor(
            points,
            state=phx.uq.GaussianProcessLikelihoodState(
                noise_scale=0.1,
                execution=phx.uq.ExactGaussianProcessExecutionPolicy(
                    max_factor_storage_bytes=1
                ),
            ),
        )
    repeated = jnp.zeros((3, 1), dtype=jnp.float64)
    failed_state = phx.uq.GaussianProcessLikelihoodState(
        kernel=phx.kernels.SquaredExponentialKernel(), noise_scale=0.0, jitter=1e-300
    )
    factor = phx.uq.ExactGaussianProcessFactor(repeated, state=failed_state)
    assert not factor.evidence.successful
    assert factor.evidence.status != int(phx.linalg.LinearSolveStatus.SUCCESS)
    with pytest.raises(eqx.EquinoxRuntimeError, match="solve failed"):
        factor.log_probability(jnp.zeros((3,), dtype=jnp.float64))
    failed_model = phx.uq.ExactGaussianProcessDiscrepancy(
        repeated, jnp.zeros((3,), dtype=jnp.float64)
    )
    with pytest.raises(eqx.EquinoxRuntimeError, match="solve failed"):
        failed_model.log_marginal_likelihood(
            jnp.zeros((3,), dtype=jnp.float64), state=failed_state
        )


def test_selected_native_inverse_columns_preserve_evidence_and_refuse_workspace() -> None:
    points = _design()
    factor = phx.uq.ExactGaussianProcessFactor(points, state=_state(_theta(), points, 2))
    indices = jnp.array([0, 3, 6], dtype=jnp.int32)
    estimate = factor.factorization.prepared_solve.plan.candidates[-1]
    required = indices.size * (
        points.shape[0] * points.dtype.itemsize
        + estimate.solve_workspace_bytes_per_rhs
        + estimate.preconditioner_apply_workspace_bytes_per_rhs
    )
    result = factor.factorization.inverse_columns(indices, max_workspace_bytes=required)
    expected = jnp.linalg.solve(
        _covariance(points, factor.state), jnp.eye(points.shape[0])[:, indices]
    )
    assert jnp.all(result.successful)
    assert jnp.all(result.derivative_valid)
    assert result.status.shape == (indices.size,)
    assert jnp.allclose(result.value, expected)
    with pytest.raises(ValueError, match="workspace"):
        factor.factorization.inverse_columns(indices, max_workspace_bytes=required - 1)
    with pytest.raises(TypeError, match="integer"):
        factor.factorization.inverse_columns(indices, max_workspace_bytes=True)
    with pytest.raises(TypeError, match="integer"):
        factor.factorization.inverse_columns(indices, max_workspace_bytes=1.5)  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="positive"):
        factor.factorization.inverse_columns(indices, max_workspace_bytes=0)
    with pytest.raises(TypeError, match="integer dtype"):
        factor.factorization.inverse_columns(
            jnp.array([1.0]), max_workspace_bytes=required
        )
    with pytest.raises(ValueError, match="nonempty"):
        factor.factorization.inverse_columns(
            jnp.array([], dtype=jnp.int32), max_workspace_bytes=required
        )
    with pytest.raises(eqx.EquinoxRuntimeError, match="outside"):
        factor.factorization.inverse_columns(
            jnp.array([-1], dtype=jnp.int32), max_workspace_bytes=required
        )
    with pytest.raises(eqx.EquinoxRuntimeError, match="outside"):
        factor.factorization.inverse_columns(
            jnp.array([points.shape[0]], dtype=jnp.int32), max_workspace_bytes=required
        )


def test_fitting_objective_uses_the_same_bounded_exact_density() -> None:
    points = _design()
    values = jnp.sin(points[:, 0])
    model = phx.uq.ExactGaussianProcessDiscrepancy(points, values)

    def fitting(theta: Array) -> Array:
        return _negative_log_marginal_likelihood(points, values, _state(theta, points, 1))

    def likelihood(theta: Array) -> Array:
        return -model.log_marginal_likelihood(
            jnp.zeros_like(values), state=_state(theta, points, 1)
        )

    theta = _theta()
    assert jnp.allclose(fitting(theta), likelihood(theta))
    assert jnp.allclose(jax.grad(fitting)(theta), jax.grad(likelihood)(theta))


def test_condition_target_is_refused_by_unsupported_scalar_routes() -> None:
    points = _design()
    values = jnp.sin(points[:, 0])
    conditioning = phx.linalg.ConditionTargetPolicy(50.0)
    state = phx.uq.GaussianProcessLikelihoodState(
        noise_scale=0.1, conditioning=conditioning
    )
    sparse = phx.uq.SparseGaussianProcessDiscrepancy(points, values, points[::3])
    with pytest.raises(ValueError, match="Condition-target"):
        sparse.log_marginal_likelihood(jnp.zeros_like(values), state=state)
    with pytest.raises(ValueError, match="Condition-target"):
        sparse.condition(jnp.zeros_like(values), points[:2], state=state)

    def features(point: Array) -> Array:
        return jnp.array([1.0, point[0]])

    feature_kernel = phx.kernels.FiniteFeatureKernel(
        features,
        jnp.eye(2),
        feature_map_id="condition-target-refusal",
    )
    feature_state = phx.uq.GaussianProcessLikelihoodState(
        kernel=feature_kernel, noise_scale=0.1, conditioning=conditioning
    )
    feature_model = phx.uq.ExactGaussianProcessDiscrepancy(points, values)
    feature_factor = feature_model.factor(state=feature_state)
    assert isinstance(feature_factor, phx.uq.ExactGaussianProcessFactor)
    assert feature_factor.regularization is not None
    assert jnp.allclose(
        feature_model.log_marginal_likelihood(
            jnp.zeros_like(values), state=feature_state
        ),
        feature_factor.log_probability(values),
    )
    with pytest.raises(ValueError, match="Condition-target"):
        phx.uq.FiniteFeatureGaussianProcessFactor(points, state=feature_state)
    with pytest.raises(ValueError, match="Condition-target"):
        phx.uq.condition_bernoulli_gaussian_process(points, values > 0, state=state)
    with pytest.raises(ValueError, match="Condition-target"):
        phx.uq.GaussianProcessBayesianOptimization(8, objective_surrogate=state)
    actions = phx.uq.FixedGaussianProcessActionPolicy(jnp.eye(points.shape[0])[:, :2])
    with pytest.raises(ValueError, match="Condition-target"):
        actions.resolve(points, state=state)
    with pytest.raises(ValueError, match="Condition-target"):
        phx.uq.ComputationAwareGaussianProcessFactor(points, state=state, actions=actions)


def test_residual_only_model_derivatives_match_fixed_native_factor() -> None:
    points = _design()
    observations = jnp.sin(points[:, 0])
    model = phx.uq.ExactGaussianProcessDiscrepancy(points, observations)
    state = _state(_theta(), points, 1)
    factor = model.factor(state=state)

    def native(mean_parameters: Array) -> Array:
        return model.log_marginal_likelihood(
            mean_parameters[0] * points[:, 0] + mean_parameters[1], state=state
        )

    def fixed(mean_parameters: Array) -> Array:
        residual = observations - mean_parameters[0] * points[:, 0] - mean_parameters[1]
        return factor.log_probability(residual)

    mean_parameters = jnp.array([0.4, -0.1], dtype=jnp.float64)
    assert jnp.allclose(
        jax.jit(jax.grad(native))(mean_parameters), jax.grad(fixed)(mean_parameters)
    )
    assert jnp.allclose(
        jax.hessian(native)(mean_parameters), jax.hessian(fixed)(mean_parameters)
    )


def test_single_observation_condition_target_remains_the_same_exact_model() -> None:
    points = _design()[3:4]
    observations = jnp.array([0.4], dtype=jnp.float64)
    model = phx.uq.ExactGaussianProcessDiscrepancy(points, observations)
    conditioning = phx.linalg.ConditionTargetPolicy(10.0)

    def conditioned(theta: Array) -> Array:
        return model.log_marginal_likelihood(
            theta[6] * points[:, 0] + theta[7],
            state=_state(theta, points, 1, conditioning),
        )

    def reference(theta: Array) -> Array:
        residual = observations - theta[6] * points[:, 0] - theta[7]
        return _dense_log_probability(
            residual, _covariance(points, _state(theta, points, 1))
        )

    theta = _theta()
    factor = model.factor(state=_state(theta, points, 1, conditioning))
    assert isinstance(factor, phx.uq.ExactGaussianProcessFactor)
    assert factor.regularization is not None
    assert factor.regularization.eta == 0.0
    assert jnp.allclose(conditioned(theta), reference(theta))
    assert jnp.allclose(jax.grad(conditioned)(theta), jax.grad(reference)(theta))


def test_inverse_columns_use_target_rhs_and_source_solution_event_shapes() -> None:
    matrix = jnp.array(
        [
            [3.0, 0.2, 0.0, 0.1],
            [0.1, 2.0, 0.3, 0.0],
            [0.0, 0.2, 1.5, 0.1],
            [0.2, 0.0, 0.1, 2.5],
        ],
        dtype=jnp.float64,
    )
    factorization = phx.linalg.factorize(
        phx.linalg.DenseLinearOperator(
            matrix,
            source=phx.linalg.ArraySpace((2, 2), dtype=matrix.dtype),
            target=phx.linalg.ArraySpace((4,), dtype=matrix.dtype),
        ),
        phx.linalg.FactorizationPolicy("lu"),
    )
    columns = jnp.array([0, 2], dtype=jnp.int32)
    result = factorization.inverse_columns(columns, max_workspace_bytes=64 * 1024)
    expected = jnp.linalg.solve(matrix, jnp.eye(4)[:, columns])
    assert result.value.shape == (2, 2, 2)
    assert jnp.all(result.successful)
    assert jnp.allclose(result.value.reshape((4, 2)), expected)


def test_streamed_condition_score_uses_the_owner_gershgorin_tie_convention() -> None:
    points = jnp.eye(4, dtype=jnp.float64)
    observations = jnp.array([0.1, 0.3, -0.2, 0.4], dtype=jnp.float64)
    model = phx.uq.ExactGaussianProcessDiscrepancy(points, observations)
    conditioning = phx.linalg.ConditionTargetPolicy(20.0)

    def state(noise_parameters: Array) -> phx.uq.GaussianProcessLikelihoodState:
        return phx.uq.GaussianProcessLikelihoodState(
            kernel=phx.kernels.SquaredExponentialKernel(length_scale=0.8),
            noise_scale=jnp.exp(noise_parameters),
            conditioning=conditioning,
            execution=_execution(points, 1),
        )

    def native(noise_parameters: Array) -> Array:
        return model.log_marginal_likelihood(
            jnp.zeros_like(observations), state=state(noise_parameters)
        )

    def reference(noise_parameters: Array) -> Array:
        covariance = _covariance(points, state(noise_parameters))
        scale = jnp.sqrt(jnp.diag(covariance))
        normalized = covariance / scale[:, None] / scale[None, :]
        upper = jnp.max(jnp.sum(jnp.abs(normalized), axis=1))
        eta = upper / (conditioning.max_condition_number - 1.0)
        transformed = covariance + jnp.diag(eta * jnp.diag(covariance))
        return _dense_log_probability(observations, transformed)

    noise_parameters = jnp.full((4,), -2.0, dtype=jnp.float64)
    factor = model.factor(state=state(noise_parameters))
    assert isinstance(factor, phx.uq.ExactGaussianProcessFactor)
    assert factor.regularization is not None
    rows = factor.regularization.normalized_row_sums
    assert rows is not None
    assert jnp.sum(rows == jnp.max(rows)) > 1
    direction = jnp.array([0.2, -0.1, 0.4, -0.3], dtype=jnp.float64)
    assert jnp.allclose(
        jax.grad(native)(noise_parameters),
        jax.grad(reference)(noise_parameters),
        rtol=2e-8,
        atol=2e-9,
    )
    assert jnp.allclose(
        jax.jvp(native, (noise_parameters,), (direction,))[1],
        jax.jvp(reference, (noise_parameters,), (direction,))[1],
        rtol=2e-8,
        atol=2e-9,
    )


def test_exact_native_rhs_derivatives_reuse_factors_with_masked_noise_scale() -> None:
    points = _design()[:5]
    inactive_noise = jnp.sqrt(jnp.sqrt(jnp.finfo(points.dtype).max))
    noise = jnp.array([0.1, 0.2, inactive_noise, 0.3, 0.4], dtype=points.dtype)
    state = phx.uq.GaussianProcessLikelihoodState(
        kernel=phx.kernels.SquaredExponentialKernel(length_scale=0.9),
        noise_scale=noise,
        jitter=1e-5,
        execution=_execution(points, 1),
    )
    factor = phx.uq.ExactGaussianProcessFactor(points, state=state)
    covariance = _covariance(points, state)
    right = jnp.stack((jnp.sin(points[:, 0]), jnp.cos(points[:, 0])), axis=1)
    direction = jnp.arange(right.size, dtype=points.dtype).reshape(right.shape) / 10.0

    def solve(value: Array) -> Array:
        return factor.factorization.solve(
            value, rhs_layout=phx.linalg.RHSLayout((value.shape[1],))
        ).value

    result = factor.factorization.solve(
        right, rhs_layout=phx.linalg.RHSLayout((right.shape[1],))
    )
    tangent = jax.jvp(solve, (right,), (direction,))[1]

    def objective(value: Array) -> Array:
        return jnp.sum(solve(value))

    gradient = jax.grad(objective)(right)
    assert jnp.all(result.successful)
    assert jnp.all(result.derivative_valid)
    assert jnp.all(jnp.isfinite(tangent))
    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.allclose(
        tangent, jnp.linalg.solve(covariance, direction), rtol=2e-9, atol=2e-10
    )
    assert jnp.allclose(
        gradient,
        jnp.linalg.solve(covariance.T, jnp.ones_like(right)),
        rtol=2e-9,
        atol=2e-10,
    )
