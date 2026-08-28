#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


class BlockLimitedSquaredExponentialKernel(phx.kernels.AbstractPositiveDefiniteKernel):
    max_left_count: int = eqx.field(static=True)

    def __init__(self, max_left_count: int):
        self.max_left_count = int(max_left_count)

    def pairwise(self, left, right, /):
        difference = jnp.ravel(left) - jnp.ravel(right)
        return jnp.exp(-0.5 * jnp.dot(difference, difference))

    def matrix(self, left, right, /):
        left_array = jnp.asarray(left)
        right_array = jnp.asarray(right)
        if int(left_array.shape[0]) > self.max_left_count:
            raise ValueError("kernel left block exceeded its declared test limit")
        return jax.vmap(
            lambda point: jax.vmap(lambda other: self.pairwise(point, other))(right_array)
        )(left_array)

    def diagonal(self, points, /):
        return jnp.ones((jnp.asarray(points).shape[0],), dtype=jnp.asarray(points).dtype)

    @property
    def max_derivative_order(self):
        return None

    @property
    def is_unit_diagonal(self):
        return True

    @property
    def kernel_id(self):
        return f"block-limited-se:{self.max_left_count}"


def _state(*, noise=0.08, length_scale=0.25):
    return phx.uq.GaussianProcessLikelihoodState(
        kernel=phx.kernels.Matern32Kernel(length_scale=length_scale),
        noise_scale=noise,
        jitter=1e-8,
    )


def _problem(count=10):
    points = jnp.linspace(0.0, 1.0, count)
    observations = 0.7 * points + 0.12 * jnp.sin(2.0 * jnp.pi * points)
    mean = 0.65 * points
    return points, observations, mean


def test_full_rank_actions_recover_exact_gp_and_exact_log_evidence():
    points, observations, mean = _problem(9)
    state = _state()
    exact = phx.uq.ExactGaussianProcessDiscrepancy(points, observations)
    computation_aware = phx.uq.ComputationAwareGaussianProcessDiscrepancy(
        points,
        observations,
    )
    actions = phx.uq.FixedGaussianProcessActionPolicy(jnp.eye(points.size))
    residual = observations - mean
    query = jnp.linspace(0.05, 0.95, 7)

    exact_factor = exact.factor(state=state)
    factor = computation_aware.factor(state=state, actions=actions)
    exact_condition = exact_factor.condition(residual, query)
    condition = factor.condition(residual, query)

    assert factor.diagnostics.valid
    assert jnp.allclose(condition.mean, exact_condition.mean, atol=1e-10, rtol=1e-10)
    assert jnp.allclose(
        condition.covariance,
        exact_condition.covariance,
        atol=1e-10,
        rtol=1e-10,
    )
    assert jnp.allclose(
        factor.elbo(residual),
        exact_factor.log_probability(residual),
        atol=1e-10,
        rtol=1e-10,
    )


def test_action_basis_changes_leave_posterior_and_elbo_invariant():
    points, observations, mean = _problem(11)
    state = _state()
    key_s, key_r = jr.split(jr.key(11))
    base = jr.normal(key_s, (points.size, 4))
    transform = jr.normal(key_r, (4, 4)) + 3.0 * jnp.eye(4)
    model = phx.uq.ComputationAwareGaussianProcessDiscrepancy(points, observations)
    residual = observations - mean
    query = jnp.linspace(0.0, 1.0, 8)

    first = model.factor(
        state=state,
        actions=phx.uq.FixedGaussianProcessActionPolicy(base),
    )
    second = model.factor(
        state=state,
        actions=phx.uq.FixedGaussianProcessActionPolicy(base @ transform),
    )
    first_condition = first.condition(residual, query)
    second_condition = second.condition(residual, query)

    assert jnp.allclose(first.elbo(residual), second.elbo(residual), atol=1e-8)
    assert jnp.allclose(first_condition.mean, second_condition.mean, atol=1e-8)
    assert jnp.allclose(
        first_condition.covariance,
        second_condition.covariance,
        atol=1e-8,
    )


def test_lower_rank_covariance_is_conservative_and_nested_actions_reduce_it():
    points, observations, mean = _problem(10)
    state = _state()
    model = phx.uq.ComputationAwareGaussianProcessDiscrepancy(points, observations)
    exact = phx.uq.ExactGaussianProcessDiscrepancy(points, observations)
    query = jnp.linspace(0.05, 0.95, 9)
    residual = observations - mean
    basis = jnp.eye(points.size)

    rank_two = model.condition(
        mean,
        query,
        state=state,
        actions=phx.uq.FixedGaussianProcessActionPolicy(basis[:, :2]),
    )
    rank_five = model.condition(
        mean,
        query,
        state=state,
        actions=phx.uq.FixedGaussianProcessActionPolicy(basis[:, :5]),
    )
    exact_condition = exact.condition(mean, query, state=state)

    conservative = rank_five.covariance - exact_condition.covariance
    nested_reduction = rank_two.covariance - rank_five.covariance
    assert jnp.linalg.eigvalsh(conservative).min() >= -1e-10
    assert jnp.linalg.eigvalsh(nested_reduction).min() >= -1e-10
    assert (
        model.elbo(
            mean,
            state=state,
            actions=phx.uq.FixedGaussianProcessActionPolicy(basis[:, :5]),
        )
        <= exact.log_marginal_likelihood(mean, state=state) + 1e-10
    )
    assert residual.shape == observations.shape


def test_diagonal_moments_match_full_condition_and_factor_reuses_residuals():
    points, observations, mean = _problem(12)
    state = _state(noise=jnp.linspace(0.04, 0.09, points.size))
    actions = phx.uq.BlockSparseGaussianProcessActionPolicy.from_random(
        jr.key(3), points.size, 4
    )
    model = phx.uq.ComputationAwareGaussianProcessDiscrepancy(points, observations)
    factor = model.factor(state=state, actions=actions)
    query = jnp.linspace(0.0, 1.0, 13)
    first_residual = observations - mean
    second_residual = observations - 0.6 * points

    diagonal_mean, diagonal_variance = factor.latent_moments(first_residual, query)
    conditioner = factor.conditioner(query, output_dim="query")
    first = conditioner.condition(first_residual)
    second = conditioner.condition(second_residual)
    direct_second = factor.condition(second_residual, query, output_dim="query")

    assert jnp.allclose(diagonal_mean, first.mean)
    assert jnp.allclose(diagonal_variance, first.variance)
    assert jnp.allclose(diagonal_variance, jnp.diag(first.covariance))
    assert jnp.allclose(second.mean, direct_second.mean)
    assert jnp.allclose(second.covariance, direct_second.covariance)
    assert conditioner.storage_elements < query.size * points.size + query.size**2
    assert first.output_dims == ("query",)


def test_chunked_kernel_action_never_requests_full_left_design():
    points, observations, mean = _problem(14)
    state = phx.uq.GaussianProcessLikelihoodState(
        kernel=BlockLimitedSquaredExponentialKernel(1),
        noise_scale=0.05,
    )
    model = phx.uq.ComputationAwareGaussianProcessDiscrepancy(points, observations)
    factor = model.factor(
        state=state,
        actions=phx.uq.BlockSparseGaussianProcessActionPolicy.from_random(
            jr.key(4), points.size, 4
        ),
        computation=phx.uq.GaussianProcessComputationPolicy(
            max_workspace_bytes=2048,
            checkpoint_kernel_blocks=False,
        ),
    )

    predicted_mean, predicted_variance = factor.latent_moments(
        observations - mean,
        points,
    )
    assert factor.diagnostics.kernel_entry_count == points.size**2
    assert factor.diagnostics.kernel_row_batch_size >= 1
    assert predicted_mean.shape == points.shape
    assert jnp.all(predicted_variance >= 0.0)


def test_full_covariance_resource_limit_preserves_diagonal_prediction():
    points, observations, mean = _problem(8)
    state = _state()
    factor = phx.uq.ComputationAwareGaussianProcessDiscrepancy(
        points,
        observations,
    ).factor(
        state=state,
        actions=phx.uq.BlockSparseGaussianProcessActionPolicy.from_random(
            jr.key(5), points.size, 3
        ),
        computation=phx.uq.GaussianProcessComputationPolicy(
            max_condition_covariance_bytes=64,
        ),
    )
    query = jnp.linspace(0.0, 1.0, 12)

    moments = factor.latent_moments(observations - mean, query)
    assert moments[0].shape == query.shape
    with pytest.raises(ValueError, match="latent_moments"):
        factor.condition(observations - mean, query)


def test_rank_deficient_actions_retain_failure_evidence_and_fail_on_use():
    points, observations, mean = _problem(7)
    duplicate = jnp.stack((jnp.ones(points.size), jnp.ones(points.size)), axis=1)
    factor = phx.uq.ComputationAwareGaussianProcessDiscrepancy(
        points,
        observations,
    ).factor(
        state=_state(),
        actions=phx.uq.FixedGaussianProcessActionPolicy(duplicate),
    )

    assert not factor.diagnostics.valid
    with pytest.raises(Exception, match="numerically invalid"):
        factor.elbo(observations - mean)


def test_kernel_noise_mean_and_action_gradients_are_finite():
    points, observations, mean = _problem(8)
    initial_actions = jnp.linspace(0.4, 1.4, points.size)

    def objective(parameters):
        state = phx.uq.GaussianProcessLikelihoodState(
            kernel=phx.kernels.AmplitudeKernel(
                phx.kernels.Matern32Kernel(
                    length_scale=jnp.exp(parameters[1]),
                ),
                jnp.exp(parameters[0]),
            ),
            noise_scale=jnp.exp(parameters[2]),
        )
        actions = phx.uq.BlockSparseGaussianProcessActionPolicy(
            parameters[4:],
            3,
        )
        model = phx.uq.ComputationAwareGaussianProcessDiscrepancy(points, observations)
        physical_mean = parameters[3] * points
        return model.elbo(physical_mean, state=state, actions=actions)

    parameters = jnp.concatenate(
        (
            jnp.asarray([jnp.log(0.3), jnp.log(0.25), jnp.log(0.08), 0.65]),
            initial_actions,
        )
    )
    value, gradient = jax.value_and_grad(objective)(parameters)
    assert jnp.isfinite(value)
    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.linalg.vector_norm(gradient) > 0.0
