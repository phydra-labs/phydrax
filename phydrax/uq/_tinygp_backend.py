#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Array, ArrayLike


KernelName = Literal["exp_squared", "matern32", "matern52"]


def gp_log_probability(
    observation_points: ArrayLike,
    residual: ArrayLike,
    /,
    *,
    amplitude: ArrayLike,
    length_scale: ArrayLike,
    noise_scale: ArrayLike,
    kernel: KernelName,
    jitter: ArrayLike,
) -> Array:
    """Evaluate an exact scalar-output GP marginal log likelihood."""
    cholesky = exact_gp_cholesky(
        observation_points,
        amplitude=amplitude,
        length_scale=length_scale,
        noise_scale=noise_scale,
        kernel=kernel,
        jitter=jitter,
    )
    return exact_gp_log_probability(residual, cholesky)


def gp_condition(
    observation_points: ArrayLike,
    residual: ArrayLike,
    query_points: ArrayLike,
    /,
    *,
    amplitude: ArrayLike,
    length_scale: ArrayLike,
    noise_scale: ArrayLike,
    kernel: KernelName,
    jitter: ArrayLike,
) -> tuple[Array, Array, Array]:
    """Return latent discrepancy conditional mean, covariance, and variance."""
    cholesky = exact_gp_cholesky(
        observation_points,
        amplitude=amplitude,
        length_scale=length_scale,
        noise_scale=noise_scale,
        kernel=kernel,
        jitter=jitter,
    )
    projection, covariance, variance = exact_gp_conditioner(
        observation_points,
        query_points,
        cholesky=cholesky,
        amplitude=amplitude,
        length_scale=length_scale,
        kernel=kernel,
        jitter=jitter,
    )
    return projection @ jnp.asarray(residual), covariance, variance


def exact_gp_cholesky(
    observation_points: ArrayLike,
    /,
    *,
    amplitude: ArrayLike,
    length_scale: ArrayLike,
    noise_scale: ArrayLike,
    kernel: KernelName,
    jitter: ArrayLike,
) -> Array:
    """Factor a fixed-hyperparameter scalar GP observation covariance."""
    points = _as_points(observation_points)
    scale_squared = jnp.asarray(amplitude) ** 2
    covariance = scale_squared * kernel_matrix(
        points,
        points,
        length_scale=length_scale,
        kernel=kernel,
    )
    diagonal = jnp.asarray(noise_scale) ** 2 + jnp.asarray(jitter)
    return jnp.linalg.cholesky(covariance + diagonal * jnp.eye(points.shape[0]))


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
    amplitude: ArrayLike,
    length_scale: ArrayLike,
    kernel: KernelName,
    jitter: ArrayLike,
) -> tuple[Array, Array, Array]:
    """Precompute the residual projection and covariance for one query design."""
    points = _as_points(observation_points)
    query = _as_points(query_points)
    scale_squared = jnp.asarray(amplitude) ** 2
    cross_covariance = scale_squared * kernel_matrix(
        query,
        points,
        length_scale=length_scale,
        kernel=kernel,
    )
    query_covariance = scale_squared * kernel_matrix(
        query,
        query,
        length_scale=length_scale,
        kernel=kernel,
    ) + jnp.asarray(jitter) * jnp.eye(query.shape[0])
    projection = jsp.linalg.cho_solve((jnp.asarray(cholesky), True), cross_covariance.T).T
    covariance = query_covariance - projection @ cross_covariance.T
    covariance = 0.5 * (covariance + covariance.T)
    return projection, covariance, jnp.maximum(jnp.diag(covariance), 0.0)


def multi_output_gp_log_probability(
    observation_points: ArrayLike,
    residual: ArrayLike,
    /,
    *,
    amplitude: ArrayLike,
    length_scale: ArrayLike,
    output_covariance: ArrayLike,
    noise_scale: ArrayLike,
    kernel: KernelName,
    jitter: ArrayLike,
) -> Array:
    """Evaluate a separable intrinsic-coregionalization GP likelihood."""
    points = _as_points(observation_points)
    values = jnp.asarray(residual)
    outputs = int(values.shape[1])
    spatial = jnp.asarray(amplitude) ** 2 * kernel_matrix(
        points, points, length_scale=length_scale, kernel=kernel
    )
    output = jnp.asarray(output_covariance)
    noise = jnp.broadcast_to(jnp.asarray(noise_scale), (outputs,))
    covariance = (
        jnp.kron(spatial, output)
        + jnp.kron(jnp.eye(points.shape[0]), jnp.diag(noise**2))
        + jnp.asarray(jitter) * jnp.eye(points.shape[0] * outputs)
    )
    return _normal_log_probability(values.reshape((-1,)), covariance)


def multi_output_gp_condition(
    observation_points: ArrayLike,
    residual: ArrayLike,
    query_points: ArrayLike,
    /,
    *,
    amplitude: ArrayLike,
    length_scale: ArrayLike,
    output_covariance: ArrayLike,
    noise_scale: ArrayLike,
    kernel: KernelName,
    jitter: ArrayLike,
) -> tuple[Array, Array, Array]:
    """Condition a separable multi-output GP at one query design."""
    points = _as_points(observation_points)
    query = _as_points(query_points)
    values = jnp.asarray(residual)
    outputs = int(values.shape[1])
    output = jnp.asarray(output_covariance)
    noise = jnp.broadcast_to(jnp.asarray(noise_scale), (outputs,))
    scale_squared = jnp.asarray(amplitude) ** 2
    k_oo = scale_squared * kernel_matrix(
        points, points, length_scale=length_scale, kernel=kernel
    )
    k_qo = scale_squared * kernel_matrix(
        query, points, length_scale=length_scale, kernel=kernel
    )
    k_qq = scale_squared * kernel_matrix(
        query, query, length_scale=length_scale, kernel=kernel
    )
    observation_covariance = (
        jnp.kron(k_oo, output)
        + jnp.kron(jnp.eye(points.shape[0]), jnp.diag(noise**2))
        + jnp.asarray(jitter) * jnp.eye(points.shape[0] * outputs)
    )
    cross_covariance = jnp.kron(k_qo, output)
    query_covariance = jnp.kron(k_qq, output) + jnp.asarray(jitter) * jnp.eye(
        query.shape[0] * outputs
    )
    cholesky = jnp.linalg.cholesky(observation_covariance)
    flat_residual = values.reshape((-1,))
    solved_residual = jsp.linalg.cho_solve((cholesky, True), flat_residual)
    solved_cross = jsp.linalg.cho_solve((cholesky, True), cross_covariance.T)
    mean = (cross_covariance @ solved_residual).reshape((query.shape[0], outputs))
    covariance = query_covariance - cross_covariance @ solved_cross
    covariance = 0.5 * (covariance + covariance.T)
    variance = jnp.maximum(jnp.diag(covariance), 0.0).reshape((query.shape[0], outputs))
    return mean, covariance, variance


def sparse_gp_log_probability(
    observation_points: ArrayLike,
    inducing_points: ArrayLike,
    residual: ArrayLike,
    /,
    *,
    amplitude: ArrayLike,
    length_scale: ArrayLike,
    noise_scale: ArrayLike,
    kernel: KernelName,
    jitter: ArrayLike,
) -> Array:
    """Evaluate a FITC likelihood in O(n m² + m³) time and O(n m) memory."""
    factors = fitc_factors(
        observation_points,
        inducing_points,
        amplitude=amplitude,
        length_scale=length_scale,
        noise_scale=noise_scale,
        kernel=kernel,
        jitter=jitter,
    )
    return sparse_gp_log_probability_from_factors(residual, *factors[:3])


def sparse_gp_condition(
    observation_points: ArrayLike,
    inducing_points: ArrayLike,
    residual: ArrayLike,
    query_points: ArrayLike,
    /,
    *,
    amplitude: ArrayLike,
    length_scale: ArrayLike,
    noise_scale: ArrayLike,
    kernel: KernelName,
    jitter: ArrayLike,
) -> tuple[Array, Array, Array]:
    """Condition the FITC latent process without forming an n×n matrix."""
    factors = fitc_factors(
        observation_points,
        inducing_points,
        amplitude=amplitude,
        length_scale=length_scale,
        noise_scale=noise_scale,
        kernel=kernel,
        jitter=jitter,
    )
    projection, covariance, variance = sparse_gp_conditioner(
        observation_points,
        inducing_points,
        query_points,
        features=factors[0],
        diagonal=factors[1],
        correction_cholesky=factors[2],
        inducing_cholesky=factors[3],
        amplitude=amplitude,
        length_scale=length_scale,
        kernel=kernel,
        jitter=jitter,
    )
    return projection @ jnp.asarray(residual), covariance, variance


def sparse_gp_log_probability_from_factors(
    residual: ArrayLike,
    features: ArrayLike,
    diagonal: ArrayLike,
    correction_cholesky: ArrayLike,
    /,
) -> Array:
    """Evaluate a FITC log density from reusable fixed-hyperparameter factors."""
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
    amplitude: ArrayLike,
    length_scale: ArrayLike,
    kernel: KernelName,
    jitter: ArrayLike,
) -> tuple[Array, Array, Array]:
    """Precompute a FITC residual projection and covariance for one query design."""
    points = _as_points(observation_points)
    inducing = _as_points(inducing_points)
    query = _as_points(query_points)
    feature_array = jnp.asarray(features)
    diagonal_array = jnp.asarray(diagonal)
    correction_factor = jnp.asarray(correction_cholesky)
    scale_squared = jnp.asarray(amplitude) ** 2
    query_inducing = scale_squared * kernel_matrix(
        query,
        inducing,
        length_scale=length_scale,
        kernel=kernel,
    )
    query_features = jsp.linalg.solve_triangular(
        jnp.asarray(inducing_cholesky),
        query_inducing.T,
        lower=True,
    ).T
    cross_covariance = query_features @ feature_array.T

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
    projected_query_covariance = query_features @ query_features.T
    residual_variance = jnp.maximum(
        scale_squared - jnp.diag(projected_query_covariance),
        0.0,
    )
    prior_covariance = projected_query_covariance + jnp.diag(residual_variance)
    covariance = (
        prior_covariance
        - cross_covariance @ solved_cross
        + jnp.asarray(jitter) * jnp.eye(query.shape[0])
    )
    covariance = 0.5 * (covariance + covariance.T)
    return projection, covariance, jnp.maximum(jnp.diag(covariance), 0.0)


def kernel_matrix(
    left: ArrayLike,
    right: ArrayLike,
    /,
    *,
    length_scale: ArrayLike,
    kernel: KernelName,
) -> Array:
    """Evaluate a supported stationary unit-amplitude kernel matrix."""
    return kernel_matrix_from_geometry(
        kernel_geometry(left, right),
        length_scale=length_scale,
        kernel=kernel,
    )


def kernel_geometry(left: ArrayLike, right: ArrayLike, /) -> Array:
    """Precompute per-coordinate squared differences for two fixed point designs."""
    left_points = _as_points(left)
    right_points = _as_points(right)
    return (left_points[:, None, :] - right_points[None, :, :]) ** 2


def kernel_matrix_from_geometry(
    squared_differences: ArrayLike,
    /,
    *,
    length_scale: ArrayLike,
    kernel: KernelName,
) -> Array:
    """Evaluate a stationary kernel from reusable squared-difference geometry."""
    geometry = jnp.asarray(squared_differences)
    distance_squared = jnp.sum(geometry / jnp.asarray(length_scale) ** 2, axis=-1)
    if kernel == "exp_squared":
        return jnp.exp(-0.5 * distance_squared)
    positive_distance = distance_squared > 0.0
    safe_distance_squared = jnp.where(
        positive_distance, distance_squared, jnp.ones_like(distance_squared)
    )
    distance = jnp.where(positive_distance, jnp.sqrt(safe_distance_squared), 0.0)
    if kernel == "matern32":
        scaled = jnp.sqrt(3.0) * distance
        return (1.0 + scaled) * jnp.exp(-scaled)
    if kernel == "matern52":
        scaled = jnp.sqrt(5.0) * distance
        return (1.0 + scaled + scaled**2 / 3.0) * jnp.exp(-scaled)
    raise ValueError(f"Unknown GP kernel {kernel!r}.")


def fitc_factors(
    observation_points: ArrayLike,
    inducing_points: ArrayLike,
    *,
    amplitude: ArrayLike,
    length_scale: ArrayLike,
    noise_scale: ArrayLike,
    kernel: KernelName,
    jitter: ArrayLike,
) -> tuple[Array, Array, Array, Array]:
    points = _as_points(observation_points)
    inducing = _as_points(inducing_points)
    scale_squared = jnp.asarray(amplitude) ** 2
    jitter_array = jnp.asarray(jitter)
    inducing_covariance = scale_squared * kernel_matrix(
        inducing,
        inducing,
        length_scale=length_scale,
        kernel=kernel,
    ) + jitter_array * jnp.eye(inducing.shape[0])
    inducing_cholesky = jnp.linalg.cholesky(inducing_covariance)
    point_inducing = scale_squared * kernel_matrix(
        points,
        inducing,
        length_scale=length_scale,
        kernel=kernel,
    )
    features = jsp.linalg.solve_triangular(
        inducing_cholesky,
        point_inducing.T,
        lower=True,
    ).T
    diagonal = jnp.maximum(
        scale_squared
        - jnp.sum(features**2, axis=1)
        + jnp.asarray(noise_scale) ** 2
        + jitter_array,
        jitter_array,
    )
    correction = jnp.eye(inducing.shape[0]) + features.T @ (features / diagonal[:, None])
    correction_cholesky = jnp.linalg.cholesky(correction)
    return features, diagonal, correction_cholesky, inducing_cholesky


def _normal_log_probability(value: Array, covariance: Array) -> Array:
    cholesky = jnp.linalg.cholesky(covariance)
    whitened = jsp.linalg.solve_triangular(cholesky, value, lower=True)
    return -0.5 * (
        value.size * jnp.log(2.0 * jnp.pi)
        + 2.0 * jnp.sum(jnp.log(jnp.diag(cholesky)))
        + whitened @ whitened
    )


def _as_points(value: ArrayLike) -> Array:
    array = jnp.asarray(value, dtype=float)
    if array.ndim == 1:
        return array[:, None]
    if array.ndim != 2:
        raise ValueError("GP points must have shape (point, coordinate).")
    return array


__all__ = [
    "KernelName",
    "exact_gp_cholesky",
    "exact_gp_conditioner",
    "exact_gp_log_probability",
    "fitc_factors",
    "gp_condition",
    "gp_log_probability",
    "kernel_geometry",
    "kernel_matrix",
    "kernel_matrix_from_geometry",
    "multi_output_gp_condition",
    "multi_output_gp_log_probability",
    "sparse_gp_condition",
    "sparse_gp_conditioner",
    "sparse_gp_log_probability",
    "sparse_gp_log_probability_from_factors",
]
