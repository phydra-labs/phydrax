"""Independent dense references for bounded action-whitened GP geometry."""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


class QuadraticFactor(phx.uq.AbstractObservationFactor):
    def __init__(self) -> None:
        self.factor_id = "quadratic-test"
        self.semantics = "normalized_likelihood"

    def log_factors(self, physical: Any, batch: phx.uq.LikelihoodBatch, /) -> jax.Array:
        return -0.5 * ((physical - batch.data) ** 2 + jnp.log(2.0 * jnp.pi))


@pytest.mark.parametrize("budget_rows", [1, 3])
def test_bounded_variational_geometry_and_elbo_dense_oracle(budget_rows: int) -> None:
    n, m = 11, 3
    points = jnp.linspace(0.0, 1.0, n, dtype=jnp.float64)[:, None]
    actions = jnp.stack((jnp.ones(n), points[:, 0], points[:, 0] ** 2), axis=1)
    kernel = phx.kernels.SquaredExponentialKernel(length_scale=0.3)
    state = phx.uq.GaussianProcessLikelihoodState(kernel=kernel, noise_scale=0.1)
    factor = QuadraticFactor()
    geometry = phx.uq.ComputationAwareSparseVariationalGaussianProcessELBO(
        points,
        state,
        phx.uq.FixedGaussianProcessActionPolicy(actions),
        factor,
        computation=phx.uq.GaussianProcessComputationPolicy(
            max_workspace_bytes=budget_rows * (n + m) * 8
        ),
        likelihood_samples=4,
    )
    dense_action = np.asarray(kernel.matrix(points, points)) @ np.asarray(actions)
    prior = np.asarray(actions).T @ dense_action + 1e-8 * np.eye(m)
    prior = (prior + prior.T) / 2
    chol = np.linalg.cholesky(prior)
    features = np.linalg.solve(chol, dense_action.T).T
    np.testing.assert_allclose(geometry.action_prior_factor, chol, atol=2e-10, rtol=2e-10)
    np.testing.assert_allclose(
        geometry.observation_features, features, atol=2e-10, rtol=2e-10
    )
    variational = geometry.standard_normal_state()
    mean, variance = geometry.latent_moments(variational, points)
    np.testing.assert_allclose(mean, np.zeros(n), atol=1e-12)
    np.testing.assert_allclose(variance, np.ones(n), atol=1e-12)
    lower = np.asarray([[0.8, 0.0, 0.0], [0.1, 0.9, 0.0], [-0.05, 0.2, 1.2]])
    qmean = np.asarray([0.2, -0.1, 0.3])
    unconstrained = np.tril(lower, -1) + np.diag(np.log(np.diag(lower)))
    variational = phx.uq.SparseVariationalGaussianState(
        variational.inducing_points,
        jnp.asarray(qmean),
        jnp.asarray(unconstrained),
    )
    reference_mean = features @ qmean
    reference_variance = (
        1.0 - np.sum(features**2, axis=1) + np.sum((features @ lower) ** 2, axis=1)
    )
    mean, variance = geometry.latent_moments(variational, points)
    np.testing.assert_allclose(mean, reference_mean, atol=2e-10, rtol=2e-10)
    np.testing.assert_allclose(variance, reference_variance, atol=2e-10, rtol=2e-10)
    batch = phx.uq.LikelihoodBatch(
        jnp.linspace(-0.5, 0.5, n),
        jnp.ones(n, dtype=jnp.bool_),
        factor_ids=jnp.arange(n, dtype=jnp.int32),
        sampling_probabilities=jnp.ones(n),
        estimator_weights=jnp.ones(n),
    )
    key = jax.random.key(37)
    samples = jax.vmap(lambda k: jax.random.normal(k, (n,), dtype=jnp.float64))(
        jax.random.split(key, 4)
    )
    samples = (
        jnp.asarray(reference_mean) + jnp.sqrt(jnp.asarray(reference_variance)) * samples
    )
    reference = jnp.mean(
        jnp.sum(-0.5 * ((samples - batch.data) ** 2 + jnp.log(2.0 * jnp.pi)), axis=1)
    )
    reference -= 0.5 * (
        np.sum(lower**2) + np.sum(qmean**2) - m - 2 * np.sum(np.log(np.diag(lower)))
    )
    np.testing.assert_allclose(
        geometry(variational, batch, key=key), reference, atol=1e-11, rtol=1e-11
    )
    with pytest.raises(ValueError, match="one kernel row"):
        phx.uq.ComputationAwareSparseVariationalGaussianProcessELBO(
            points,
            state,
            phx.uq.FixedGaussianProcessActionPolicy(actions),
            factor,
            computation=phx.uq.GaussianProcessComputationPolicy(
                max_workspace_bytes=(n + m) * 8 - 1
            ),
        )
