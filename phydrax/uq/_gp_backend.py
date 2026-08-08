#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Array, ArrayLike

from phydrax.kernels import AbstractPositiveDefiniteKernel


def exact_gp_cholesky(
    observation_points: ArrayLike,
    /,
    *,
    kernel: AbstractPositiveDefiniteKernel,
    noise_scale: ArrayLike,
    jitter: ArrayLike,
) -> Array:
    """Factor an exact scalar GP observation covariance."""
    points = _as_points(observation_points)
    diagonal = _observation_diagonal(
        noise_scale,
        jitter,
        count=int(points.shape[0]),
    )
    covariance = kernel.matrix(points, points)
    return jnp.linalg.cholesky(covariance + jnp.diag(diagonal))


def exact_gp_log_probability(residual: ArrayLike, cholesky: ArrayLike, /) -> Array:
    """Evaluate a Gaussian log density from a reusable covariance factor."""
    value = jnp.asarray(residual)
    factor = jnp.asarray(cholesky)
    whitened = jsp.linalg.solve_triangular(factor, value, lower=True)
    return -0.5 * (
        value.size * jnp.log(2.0 * jnp.pi)
        + 2.0 * jnp.sum(jnp.log(jnp.diag(factor)))
        + whitened @ whitened
    )


def exact_gp_conditioner(
    observation_points: ArrayLike,
    query_points: ArrayLike,
    /,
    *,
    cholesky: ArrayLike,
    kernel: AbstractPositiveDefiniteKernel,
) -> tuple[Array, Array, Array]:
    """Precompute exact residual projection and latent query covariance."""
    points = _as_points(observation_points)
    query = _as_points(query_points)
    return exact_gp_conditioner_from_covariances(
        cholesky,
        kernel.matrix(query, points),
        kernel.matrix(query, query),
    )


def exact_gp_conditioner_from_covariances(
    cholesky: ArrayLike,
    cross_covariance: ArrayLike,
    query_covariance: ArrayLike,
    /,
) -> tuple[Array, Array, Array]:
    """Condition from preassembled covariance blocks."""
    cross = jnp.asarray(cross_covariance)
    query = jnp.asarray(query_covariance)
    projection = jsp.linalg.cho_solve(
        (jnp.asarray(cholesky), True),
        cross.T,
    ).T
    covariance = query - projection @ cross.T
    covariance = 0.5 * (covariance + covariance.T)
    return projection, covariance, jnp.maximum(jnp.diag(covariance), 0.0)


def fitc_factors(
    observation_points: ArrayLike,
    inducing_points: ArrayLike,
    /,
    *,
    kernel: AbstractPositiveDefiniteKernel,
    noise_scale: ArrayLike,
    jitter: ArrayLike,
) -> tuple[Array, Array, Array, Array]:
    """Build generic FITC factors from a positive-definite kernel."""
    points = _as_points(observation_points)
    inducing = _as_points(inducing_points)
    return fitc_factors_from_covariances(
        kernel.matrix(points, inducing),
        kernel.diagonal(points),
        kernel.matrix(inducing, inducing),
        noise_scale=noise_scale,
        jitter=jitter,
    )


def fitc_factors_from_covariances(
    observation_inducing_covariance: ArrayLike,
    observation_prior_diagonal: ArrayLike,
    inducing_covariance: ArrayLike,
    /,
    *,
    noise_scale: ArrayLike,
    jitter: ArrayLike,
) -> tuple[Array, Array, Array, Array]:
    """Build reusable FITC factors from preassembled covariance blocks."""
    point_inducing = jnp.asarray(observation_inducing_covariance)
    prior_diagonal = jnp.asarray(observation_prior_diagonal)
    inducing = jnp.asarray(inducing_covariance)
    jitter_array = jnp.asarray(jitter)
    inducing_cholesky = jnp.linalg.cholesky(
        inducing + jitter_array * jnp.eye(inducing.shape[0])
    )
    features = jsp.linalg.solve_triangular(
        inducing_cholesky,
        point_inducing.T,
        lower=True,
    ).T
    observation_diagonal = _observation_diagonal(
        noise_scale,
        jitter_array,
        count=int(point_inducing.shape[0]),
    )
    diagonal = jnp.maximum(
        prior_diagonal - jnp.sum(features * features, axis=1) + observation_diagonal,
        jitter_array,
    )
    correction_cholesky = low_rank_gp_correction_cholesky(features, diagonal)
    return features, diagonal, correction_cholesky, inducing_cholesky


def low_rank_gp_correction_cholesky(
    features: ArrayLike,
    diagonal: ArrayLike,
    /,
) -> Array:
    """Factor the Woodbury correction for a diagonal-plus-low-rank covariance."""
    feature_array = jnp.asarray(features)
    diagonal_array = jnp.asarray(diagonal)
    correction = jnp.eye(feature_array.shape[1]) + feature_array.T @ (
        feature_array / diagonal_array[:, None]
    )
    return jnp.linalg.cholesky(correction)


def sparse_gp_log_probability_from_factors(
    residual: ArrayLike,
    features: ArrayLike,
    diagonal: ArrayLike,
    correction_cholesky: ArrayLike,
    /,
) -> Array:
    """Evaluate a FITC log density from reusable factors."""
    values = jnp.asarray(residual)
    feature_array = jnp.asarray(features)
    diagonal_array = jnp.asarray(diagonal)
    correction_factor = jnp.asarray(correction_cholesky)
    scaled_values = values / diagonal_array
    right = feature_array.T @ scaled_values
    correction = jsp.linalg.cho_solve((correction_factor, True), right)
    quadratic = values @ scaled_values - right @ correction
    log_determinant = jnp.sum(jnp.log(diagonal_array)) + 2.0 * jnp.sum(
        jnp.log(jnp.diag(correction_factor))
    )
    return -0.5 * (values.size * jnp.log(2.0 * jnp.pi) + log_determinant + quadratic)


def sparse_gp_conditioner(
    observation_points: ArrayLike,
    inducing_points: ArrayLike,
    query_points: ArrayLike,
    /,
    *,
    features: ArrayLike,
    diagonal: ArrayLike,
    correction_cholesky: ArrayLike,
    inducing_cholesky: ArrayLike,
    kernel: AbstractPositiveDefiniteKernel,
) -> tuple[Array, Array, Array]:
    """Precompute FITC residual projection and latent query covariance."""
    _as_points(observation_points)
    inducing = _as_points(inducing_points)
    query = _as_points(query_points)
    return sparse_gp_conditioner_from_covariances(
        kernel.matrix(query, inducing),
        kernel.diagonal(query),
        features=features,
        diagonal=diagonal,
        correction_cholesky=correction_cholesky,
        inducing_cholesky=inducing_cholesky,
    )


def sparse_gp_conditioner_from_covariances(
    query_inducing_covariance: ArrayLike,
    query_prior_diagonal: ArrayLike,
    /,
    *,
    features: ArrayLike,
    diagonal: ArrayLike,
    correction_cholesky: ArrayLike,
    inducing_cholesky: ArrayLike,
) -> tuple[Array, Array, Array]:
    """Condition a FITC model from preassembled query covariance blocks."""
    query_inducing = jnp.asarray(query_inducing_covariance)
    query_diagonal = jnp.asarray(query_prior_diagonal)
    query_features = jsp.linalg.solve_triangular(
        jnp.asarray(inducing_cholesky),
        query_inducing.T,
        lower=True,
    ).T
    projected_diagonal = jnp.sum(query_features * query_features, axis=1)
    return low_rank_gp_conditioner(
        query_features,
        jnp.maximum(query_diagonal - projected_diagonal, 0.0),
        features=features,
        diagonal=diagonal,
        correction_cholesky=correction_cholesky,
    )


def low_rank_gp_conditioner(
    query_features: ArrayLike,
    query_residual_diagonal: ArrayLike,
    /,
    *,
    features: ArrayLike,
    diagonal: ArrayLike,
    correction_cholesky: ArrayLike,
) -> tuple[Array, Array, Array]:
    """Condition a diagonal-plus-low-rank Gaussian covariance."""
    feature_array = jnp.asarray(features)
    diagonal_array = jnp.asarray(diagonal)
    correction_factor = jnp.asarray(correction_cholesky)
    query_feature_array = jnp.asarray(query_features)
    residual_diagonal = jnp.asarray(query_residual_diagonal)
    cross_covariance = query_feature_array @ feature_array.T

    def solve_observation_covariance(right):
        scaled = (
            right / diagonal_array[:, None] if right.ndim == 2 else right / diagonal_array
        )
        projected = feature_array.T @ scaled
        correction = jsp.linalg.cho_solve((correction_factor, True), projected)
        adjusted = feature_array @ correction
        return scaled - (
            adjusted / diagonal_array[:, None]
            if right.ndim == 2
            else adjusted / diagonal_array
        )

    solved_cross = solve_observation_covariance(cross_covariance.T)
    projection = solved_cross.T
    prior_covariance = query_feature_array @ query_feature_array.T + jnp.diag(
        residual_diagonal
    )
    covariance = prior_covariance - cross_covariance @ solved_cross
    covariance = 0.5 * (covariance + covariance.T)
    return projection, covariance, jnp.maximum(jnp.diag(covariance), 0.0)


def _observation_diagonal(
    noise_scale: ArrayLike,
    jitter: ArrayLike,
    /,
    *,
    count: int,
) -> Array:
    noise = jnp.broadcast_to(jnp.asarray(noise_scale), (count,))
    return noise * noise + jnp.asarray(jitter)


def _as_points(value: ArrayLike) -> Array:
    array = jnp.asarray(value, dtype=float)
    if array.ndim == 1:
        return array[:, None]
    if array.ndim != 2:
        raise ValueError("GP points must have shape (point, coordinate).")
    return array


__all__ = [
    "exact_gp_cholesky",
    "exact_gp_conditioner",
    "exact_gp_conditioner_from_covariances",
    "exact_gp_log_probability",
    "fitc_factors",
    "fitc_factors_from_covariances",
    "low_rank_gp_conditioner",
    "low_rank_gp_correction_cholesky",
    "sparse_gp_conditioner",
    "sparse_gp_conditioner_from_covariances",
    "sparse_gp_log_probability_from_factors",
]
