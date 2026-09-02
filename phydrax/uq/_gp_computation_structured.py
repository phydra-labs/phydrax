#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Shared action projection for functional and multi-output GP covariances."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from .._strict import StrictModule
from ..kernels import Matern32Kernel
from ..linalg import MaterializationPolicy, materialize
from ._gp_actions import (
    AbstractGaussianProcessActionPolicy,
    BlockSparseGaussianProcessActionPolicy,
    FixedGaussianProcessActionPolicy,
)
from ._gp_computation_backend import _factorize_positive, _solve_columns, _solve_vector
from ._gp_likelihood import GaussianProcessLikelihoodState


class StructuredComputationAwareGaussianProcessFactor(StrictModule):
    """Bounded projected geometry for a preassembled structured covariance."""

    action_matrix: Array
    active_mask: Array
    projected_covariance: Array
    covariance_factor: object
    observation_count: int = eqx.field(static=True)
    action_count: int = eqx.field(static=True)

    def __init__(
        self,
        covariance: ArrayLike,
        noise_scale: ArrayLike,
        jitter: ArrayLike,
        actions: AbstractGaussianProcessActionPolicy,
        /,
        *,
        residual: ArrayLike | None = None,
        max_factorization_bytes: int = 64 * 1024 * 1024,
    ):
        matrix = jnp.asarray(covariance)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] <= 0:
            raise ValueError("Structured GP covariance must be nonempty and square.")
        if not isinstance(
            actions,
            (FixedGaussianProcessActionPolicy, BlockSparseGaussianProcessActionPolicy),
        ):
            raise TypeError(
                "Structured computation-aware GP currently admits fixed or block-sparse "
                "actions; pseudo-input and scalar iterative policies lack a matching "
                "functional inducing design."
            )
        count = int(matrix.shape[0])
        noise = jnp.broadcast_to(jnp.asarray(noise_scale, dtype=matrix.dtype), (count,))
        jitter_array = jnp.asarray(jitter, dtype=matrix.dtype).reshape(())
        effective = noise * noise + jitter_array
        effective = eqx.error_if(
            effective,
            jnp.any(~jnp.isfinite(effective)) | jnp.any(effective <= 0.0),
            "Structured GP observation variance must be finite and positive.",
        )
        dummy_state = GaussianProcessLikelihoodState(
            kernel=Matern32Kernel(), noise_scale=0.0, jitter=1e-8
        )
        dummy_points = jnp.zeros((count, 1), dtype=matrix.dtype)
        resolved = actions.resolve(
            dummy_points,
            state=dummy_state,
            residual=None if residual is None else jnp.asarray(residual),
        )
        policy = MaterializationPolicy(
            max_entries=count * resolved.num_actions,
            max_bytes=count * resolved.num_actions * int(matrix.dtype.itemsize),
        )
        action_matrix = materialize(resolved.operator, policy)
        active = resolved.active_mask.astype(matrix.dtype)
        action_matrix = action_matrix * active[None, :]
        full_covariance = matrix + jnp.diag(effective)
        projected = action_matrix.T @ full_covariance @ action_matrix
        projected = projected + jnp.diag(1.0 - active)
        projected = 0.5 * (projected + projected.T)
        factor = _factorize_positive(
            projected,
            name="structured-projected-covariance",
            max_factorization_bytes=max_factorization_bytes,
        )
        self.action_matrix = action_matrix
        self.active_mask = resolved.active_mask
        self.projected_covariance = projected
        self.covariance_factor = factor
        self.observation_count = count
        self.action_count = resolved.num_actions

    def condition(
        self,
        residual: ArrayLike,
        cross_covariance: ArrayLike,
        query_covariance: ArrayLike,
        /,
    ) -> tuple[Array, Array, Array]:
        values = jnp.asarray(residual)
        cross = jnp.asarray(cross_covariance)
        prior = jnp.asarray(query_covariance)
        if values.shape != (self.observation_count,):
            raise ValueError("Residual must align with structured observations.")
        if cross.ndim != 2 or cross.shape[1] != self.observation_count:
            raise ValueError("Cross covariance must have shape (query, observation).")
        if prior.shape != (cross.shape[0], cross.shape[0]):
            raise ValueError("Query covariance must be square over query rows.")
        query_action = cross @ self.action_matrix
        projected_residual = self.action_matrix.T @ values
        alpha = _solve_vector(self.covariance_factor, projected_residual)
        solved, successful = _solve_columns(self.covariance_factor, query_action.T)
        solved = eqx.error_if(
            solved,
            ~successful,
            "Structured projected GP covariance solve failed.",
        )
        mean = ein.contract("qm,m->q", query_action, alpha)
        covariance = prior - query_action @ solved
        covariance = 0.5 * (covariance + covariance.T)
        variance = jnp.diag(covariance)
        return mean, covariance, variance


__all__ = ["StructuredComputationAwareGaussianProcessFactor"]
