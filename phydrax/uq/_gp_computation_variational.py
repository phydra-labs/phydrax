#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""UQI-lifecycle action-whitened computation-aware GP ELBO."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from .._strict import StrictModule
from ..linalg import MaterializationPolicy, materialize
from ._gp_actions import AbstractGaussianProcessActionPolicy
from ._gp_likelihood import GaussianProcessLikelihoodState
from ._minibatch_posterior import AbstractObservationFactor, LikelihoodBatch
from ._sparse_variational_gp import (
    SparseVariationalGaussianProcessELBO,
    SparseVariationalGaussianState,
)


class ComputationAwareSparseVariationalGaussianProcessELBO(StrictModule):
    """Action-whitened non-Gaussian ELBO using UQI state, batches, and fit loop."""

    kernel: Any
    observation_factor: AbstractObservationFactor
    regularization: Array
    likelihood_samples: int = eqx.field(static=True)

    observation_points: Array
    action_matrix: Array
    action_prior_factor: Array
    observation_features: Array
    prior_diagonal: Array
    active_mask: Array
    action_id: str = eqx.field(static=True)

    def __init__(
        self,
        observation_points: ArrayLike,
        state: GaussianProcessLikelihoodState,
        actions: AbstractGaussianProcessActionPolicy,
        observation_factor: AbstractObservationFactor,
        /,
        *,
        regularization: ArrayLike = 1e-8,
        likelihood_samples: int = 8,
    ):
        if not isinstance(state, GaussianProcessLikelihoodState):
            raise TypeError("state must be a GaussianProcessLikelihoodState.")
        points = jnp.asarray(observation_points)
        if points.ndim != state.kernel.input_ndim + 1 or points.shape[0] <= 0:
            raise ValueError("observation_points must match the kernel input rank.")
        if not isinstance(actions, AbstractGaussianProcessActionPolicy):
            raise TypeError("actions must implement AbstractGaussianProcessActionPolicy.")
        if actions.requires_residual:
            raise ValueError(
                "Residual-dependent actions are not reusable non-Gaussian inducing geometry."
            )
        base = SparseVariationalGaussianProcessELBO(
            state.kernel,
            observation_factor,
            regularization=regularization,
            likelihood_samples=likelihood_samples,
        )
        self.kernel = base.kernel
        self.observation_factor = base.observation_factor
        self.regularization = base.regularization
        self.likelihood_samples = base.likelihood_samples
        resolved = actions.resolve(points, state=state)
        policy = MaterializationPolicy(
            max_entries=resolved.num_observations * resolved.num_actions,
            max_bytes=(
                resolved.num_observations
                * resolved.num_actions
                * int(points.dtype.itemsize)
            ),
        )
        matrix = materialize(resolved.operator, policy)
        active = resolved.active_mask.astype(points.dtype)
        matrix = matrix * active[None, :]
        covariance = state.kernel.matrix(points, points)
        kernel_action = covariance @ matrix
        projected = matrix.T @ kernel_action + jnp.diag(1.0 - active)
        projected = 0.5 * (projected + projected.T)
        factor = jnp.linalg.cholesky(
            projected
            + jnp.asarray(regularization, dtype=projected.dtype) * jnp.diag(active)
        )
        factor = eqx.error_if(
            factor,
            jnp.any(~jnp.isfinite(factor)) | jnp.any(jnp.diag(factor) <= 0.0),
            "Action prior covariance is not positive definite.",
        )
        features = jsp.linalg.solve_triangular(factor, kernel_action.T, lower=True).T
        self.observation_points = points
        self.action_matrix = matrix
        self.action_prior_factor = factor
        self.observation_features = features
        self.prior_diagonal = state.kernel.diagonal(points)
        self.active_mask = resolved.active_mask
        self.action_id = resolved.action_id

    @property
    def action_count(self) -> int:
        return int(self.action_matrix.shape[1])

    def standard_normal_state(self, /) -> SparseVariationalGaussianState:
        """Construct UQI's canonical prior state for this action capacity."""
        coordinates = jnp.arange(self.action_count, dtype=self.observation_points.dtype)
        return SparseVariationalGaussianState.standard_normal(coordinates[:, None])

    def latent_moments(
        self,
        state: SparseVariationalGaussianState,
        points: ArrayLike,
        /,
    ) -> tuple[Array, Array]:
        inputs = jnp.asarray(points)
        if inputs.shape != self.observation_points.shape:
            raise ValueError(
                "Action-space latent_moments currently requires the prepared observation "
                "design; arbitrary queries require a separately prepared query epoch."
            )
        _validate_variational_capacity(state, self.action_count)
        features = self.observation_features
        mean = ein.contract("bi,i->b", features, state.mean)
        conditional = self.prior_diagonal - jnp.sum(features * features, axis=1)
        transformed = ein.contract("bi,ij->bj", features, state.scale_tril)
        variance = conditional + jnp.sum(transformed * transformed, axis=1)
        return mean, variance

    def predict(
        self,
        state: SparseVariationalGaussianState,
        query_points: ArrayLike,
        /,
    ) -> tuple[Array, Array]:
        return self.latent_moments(state, query_points)

    def __call__(
        self,
        state: SparseVariationalGaussianState,
        batch: LikelihoodBatch,
        /,
        *,
        key: Array,
    ) -> Array:
        _validate_variational_capacity(state, self.action_count)
        safe_ids = jnp.where(batch.factor_mask, batch.factor_ids, 0)
        features = self.observation_features[safe_ids]
        prior = self.prior_diagonal[safe_ids]
        mean = ein.contract("bi,i->b", features, state.mean)
        conditional = prior - jnp.sum(features * features, axis=1)
        transformed = ein.contract("bi,ij->bj", features, state.scale_tril)
        variance = conditional + jnp.sum(transformed * transformed, axis=1)
        variance = eqx.error_if(
            variance,
            jnp.any(batch.factor_mask & (~jnp.isfinite(variance) | (variance < 0.0))),
            "Action-space variational marginal variance is invalid.",
        )
        keys = jr.split(key, self.likelihood_samples)

        def one(sample_key: Array) -> Array:
            latent = mean + jnp.sqrt(jnp.maximum(variance, 0.0)) * jr.normal(
                sample_key, mean.shape, dtype=mean.dtype
            )
            factors = jnp.asarray(self.observation_factor.log_factors(latent, batch))
            if factors.shape != batch.factor_mask.shape:
                raise ValueError("Observation factors must align with batch capacity.")
            return jnp.sum(batch.estimator_weights * factors)

        expected = jnp.mean(jax.vmap(one)(keys))
        return expected - state.kl_standard_normal


def _validate_variational_capacity(
    state: SparseVariationalGaussianState,
    action_count: int,
    /,
) -> None:
    if not isinstance(state, SparseVariationalGaussianState):
        raise TypeError("state must be a SparseVariationalGaussianState.")
    if state.mean.shape != (action_count,):
        raise ValueError("UQI variational state must align with action capacity.")


__all__ = ["ComputationAwareSparseVariationalGaussianProcessELBO"]
