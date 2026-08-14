#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ._gaussian_factor import (
    _adjoint,
    _rank_aware_solve,
    add_independent_gaussian_factors,
    gaussian_cross_covariance,
    gaussian_factor_from_covariance,
    GaussianFactor,
)


ConditionalGaussianStatus = Literal[0, 1, 2, 3]
CONDITIONAL_GAUSSIAN_SUCCESS: ConditionalGaussianStatus = 0
CONDITIONAL_GAUSSIAN_NONFINITE: ConditionalGaussianStatus = 1
CONDITIONAL_GAUSSIAN_INVALID_FACTOR: ConditionalGaussianStatus = 2
CONDITIONAL_GAUSSIAN_INCONSISTENT_CONDITION: ConditionalGaussianStatus = 3


class ConditionalGaussianMoments(StrictModule):
    """Gaussian output moments and their covariance with an input event.

    ``cross_covariance`` has shape ``(..., input_size, output_size)`` and is
    oriented as ``Cov[input, output]``.  The represented Gaussian output has
    ``mean`` shape ``(..., output_size)`` and covariance ``factor.covariance``.
    """

    mean: Array
    factor: GaussianFactor
    cross_covariance: Array
    valid: Array
    status: Array
    moments_id: str = eqx.field(static=True)
    resolved_method: str = eqx.field(static=True)

    def __init__(
        self,
        mean: ArrayLike,
        factor: GaussianFactor,
        cross_covariance: ArrayLike,
        /,
        *,
        moments_id: str = "conditional-gaussian-moments",
        resolved_method: str = "provided-conditional-moments",
    ):
        mean_value = jnp.asarray(mean)
        cross_value = jnp.asarray(cross_covariance)
        if not jnp.issubdtype(mean_value.dtype, jnp.inexact):
            raise TypeError("mean must have an inexact dtype.")
        if not isinstance(factor, GaussianFactor):
            raise TypeError("factor must be a GaussianFactor.")
        if mean_value.ndim < 1 or mean_value.shape[-1] != factor.event_size:
            raise ValueError("mean must end in factor.event_size.")
        if cross_value.ndim < 2 or cross_value.shape[-1] != factor.event_size:
            raise ValueError(
                "cross_covariance must have shape (..., input_size, factor.event_size)."
            )
        if mean_value.shape[:-1] != factor.factor.shape[:-2]:
            raise ValueError("mean and factor must have identical batch dimensions.")
        if cross_value.shape[:-2] != mean_value.shape[:-1]:
            raise ValueError(
                "cross_covariance and mean must have identical batch dimensions."
            )
        if not isinstance(moments_id, str) or not moments_id:
            raise ValueError("moments_id must be a non-empty string.")
        if not isinstance(resolved_method, str) or not resolved_method:
            raise ValueError("resolved_method must be a non-empty string.")

        mean_finite = jnp.all(jnp.isfinite(mean_value), axis=-1)
        cross_finite = jnp.all(jnp.isfinite(cross_value), axis=(-2, -1))
        finite = mean_finite & cross_finite
        valid = finite & factor.valid
        status = jnp.where(
            ~finite,
            CONDITIONAL_GAUSSIAN_NONFINITE,
            jnp.where(
                ~factor.valid,
                CONDITIONAL_GAUSSIAN_INVALID_FACTOR,
                CONDITIONAL_GAUSSIAN_SUCCESS,
            ),
        ).astype(jnp.int32)

        self.mean = mean_value
        self.factor = factor
        self.cross_covariance = cross_value
        self.valid = valid
        self.status = status
        self.moments_id = moments_id
        self.resolved_method = resolved_method

    @property
    def covariance(self) -> Array:
        """Return the output covariance."""
        return self.factor.covariance

    @property
    def regularization(self) -> Array:
        """Return the explicit regularization recorded by the output factor."""
        return self.factor.regularization


class GaussianRegression(StrictModule):
    """An affine conditional Gaussian ``output = A input + b + noise``."""

    matrix: Array
    offset: Array
    noise_factor: GaussianFactor
    valid: Array
    status: Array
    regression_id: str = eqx.field(static=True)
    resolved_method: str = eqx.field(static=True)

    def __init__(
        self,
        matrix: ArrayLike,
        offset: ArrayLike,
        noise_factor: GaussianFactor,
        /,
        *,
        regression_id: str = "gaussian-regression",
        resolved_method: str = "provided-affine-regression",
    ):
        matrix_value = jnp.asarray(matrix)
        offset_value = jnp.asarray(offset)
        if not jnp.issubdtype(matrix_value.dtype, jnp.inexact) or not jnp.issubdtype(
            offset_value.dtype, jnp.inexact
        ):
            raise TypeError("matrix and offset must have inexact dtypes.")
        if not isinstance(noise_factor, GaussianFactor):
            raise TypeError("noise_factor must be a GaussianFactor.")
        if matrix_value.ndim < 2:
            raise ValueError("matrix must have shape (..., output_size, input_size).")
        if matrix_value.shape[-2] != noise_factor.event_size:
            raise ValueError("matrix output size must equal noise_factor.event_size.")
        if offset_value.shape != matrix_value.shape[:-1]:
            raise ValueError("offset must have shape (..., output_size).")
        if matrix_value.shape[:-2] != noise_factor.factor.shape[:-2]:
            raise ValueError("matrix and noise_factor must share batch dimensions.")
        if not isinstance(regression_id, str) or not regression_id:
            raise ValueError("regression_id must be a non-empty string.")
        if not isinstance(resolved_method, str) or not resolved_method:
            raise ValueError("resolved_method must be a non-empty string.")

        finite = jnp.all(jnp.isfinite(matrix_value), axis=(-2, -1)) & jnp.all(
            jnp.isfinite(offset_value), axis=-1
        )
        valid = finite & noise_factor.valid
        status = jnp.where(
            ~finite,
            CONDITIONAL_GAUSSIAN_NONFINITE,
            jnp.where(
                ~noise_factor.valid,
                CONDITIONAL_GAUSSIAN_INVALID_FACTOR,
                CONDITIONAL_GAUSSIAN_SUCCESS,
            ),
        ).astype(jnp.int32)

        self.matrix = matrix_value
        self.offset = offset_value
        self.noise_factor = noise_factor
        self.valid = valid
        self.status = status
        self.regression_id = regression_id
        self.resolved_method = resolved_method

    @classmethod
    def from_moments(
        cls,
        input_mean: ArrayLike,
        input_factor: GaussianFactor,
        output_moments: ConditionalGaussianMoments,
        /,
        *,
        rank_tolerance: ArrayLike = 0.0,
        regression_id: str = "moment-matched-regression",
    ) -> GaussianRegression:
        """Build ``input | output`` from joint first and second moments."""
        mean = jnp.asarray(input_mean)
        if mean.ndim < 1 or mean.shape[-1] != input_factor.event_size:
            raise ValueError("input_mean must end in input_factor.event_size.")
        if output_moments.cross_covariance.shape[-2] != input_factor.event_size:
            raise ValueError(
                "output_moments.cross_covariance must be Cov[input, output]."
            )
        tolerance = jnp.asarray(rank_tolerance)
        solved_cross = _rank_aware_solve(
            output_moments.covariance,
            _adjoint(output_moments.cross_covariance),
            tolerance,
        )
        matrix = _adjoint(solved_cross)
        offset = mean - jnp.einsum("...ij,...j->...i", matrix, output_moments.mean)
        conditional_covariance = input_factor.covariance - (
            matrix @ _adjoint(output_moments.cross_covariance)
        )
        noise = gaussian_factor_from_covariance(
            conditional_covariance,
            rank_tolerance=tolerance,
            hermitian_tolerance=tolerance,
            factor_id=f"{regression_id}-noise",
        )
        regression = cls(
            matrix,
            offset,
            noise,
            regression_id=regression_id,
            resolved_method="rank-aware-moment-conditioning",
        )
        source_factors_valid = input_factor.valid & output_moments.factor.valid
        regression = eqx.tree_at(
            lambda node: node.valid,
            regression,
            regression.valid & source_factors_valid,
        )
        status = jnp.where(
            ~source_factors_valid,
            CONDITIONAL_GAUSSIAN_INVALID_FACTOR,
            regression.status,
        ).astype(jnp.int32)
        return eqx.tree_at(lambda node: node.status, regression, status)

    def __call__(
        self,
        input_value: ArrayLike,
        /,
        *,
        moments_id: str = "gaussian-regression-output",
    ) -> ConditionalGaussianMoments:
        """Evaluate the conditional Gaussian at one input value."""
        value = jnp.asarray(input_value)
        if value.ndim < 1 or value.shape[-1] != self.matrix.shape[-1]:
            raise ValueError("input_value has an incompatible final dimension.")
        mean = jnp.einsum("...ij,...j->...i", self.matrix, value) + self.offset
        cross_covariance = jnp.zeros(
            (
                *mean.shape[:-1],
                self.matrix.shape[-1],
                self.matrix.shape[-2],
            ),
            dtype=jnp.result_type(value, mean),
        )
        evaluated = ConditionalGaussianMoments(
            mean,
            self.noise_factor,
            cross_covariance,
            moments_id=moments_id,
            resolved_method="affine-conditional-evaluation",
        )
        evaluated = eqx.tree_at(
            lambda node: node.valid,
            evaluated,
            evaluated.valid & self.valid,
        )
        status = jnp.where(~self.valid, self.status, evaluated.status).astype(jnp.int32)
        return eqx.tree_at(lambda node: node.status, evaluated, status)

    @property
    def regularization(self) -> Array:
        """Return the explicit regularization recorded by the noise factor."""
        return self.noise_factor.regularization


def predict_affine_gaussian(
    mean: ArrayLike,
    factor: GaussianFactor,
    matrix: ArrayLike,
    offset: ArrayLike,
    noise_factor: GaussianFactor | None = None,
    /,
    *,
    compress: bool = True,
    moments_id: str = "affine-gaussian-prediction",
) -> ConditionalGaussianMoments:
    """Propagate Gaussian moments through an affine map with independent noise."""
    mean_value = jnp.asarray(mean)
    matrix_value = jnp.asarray(matrix)
    offset_value = jnp.asarray(offset)
    if not isinstance(factor, GaussianFactor):
        raise TypeError("factor must be a GaussianFactor.")
    if mean_value.ndim < 1 or mean_value.shape[-1] != factor.event_size:
        raise ValueError("mean must end in factor.event_size.")
    if matrix_value.ndim < 2 or matrix_value.shape[-1] != factor.event_size:
        raise ValueError("matrix input size must equal factor.event_size.")
    if offset_value.shape != matrix_value.shape[:-1]:
        raise ValueError("offset must have shape (..., output_size).")
    if matrix_value.shape[:-2] != factor.factor.shape[:-2]:
        raise ValueError("matrix and factor must share batch dimensions.")
    if mean_value.shape[:-1] != factor.factor.shape[:-2]:
        raise ValueError("mean and factor must share batch dimensions.")

    predicted_mean = (
        jnp.einsum("...ij,...j->...i", matrix_value, mean_value) + offset_value
    )
    transformed = GaussianFactor(
        matrix_value @ factor.factor,
        regularization=factor.regularization,
        rank_tolerance=factor.rank_tolerance,
        factor_id=f"{moments_id}-transformed",
        resolved_method="affine-factor-pushforward",
    )
    cross_covariance = gaussian_cross_covariance(factor, transformed)
    if noise_factor is None:
        output_factor = transformed
    else:
        if noise_factor.event_size != matrix_value.shape[-2]:
            raise ValueError("noise_factor event size must equal matrix output size.")
        output_factor = add_independent_gaussian_factors(
            transformed,
            noise_factor,
            compress=compress,
            factor_id=f"{moments_id}-factor",
        )
    return ConditionalGaussianMoments(
        predicted_mean,
        output_factor,
        cross_covariance,
        moments_id=moments_id,
        resolved_method="affine-factor-moment-propagation",
    )


def condition_gaussian(
    input_mean: ArrayLike,
    input_factor: GaussianFactor,
    output_moments: ConditionalGaussianMoments,
    observed_value: ArrayLike,
    /,
    *,
    rank_tolerance: ArrayLike = 0.0,
    support_tolerance: ArrayLike = 0.0,
    moments_id: str = "conditioned-gaussian",
) -> ConditionalGaussianMoments:
    """Condition an input Gaussian on a possibly singular Gaussian output."""
    regression = GaussianRegression.from_moments(
        input_mean,
        input_factor,
        output_moments,
        rank_tolerance=rank_tolerance,
        regression_id=f"{moments_id}-regression",
    )
    conditioned = regression(observed_value, moments_id=moments_id)

    residual = jnp.asarray(observed_value) - output_moments.mean
    tolerance = jnp.asarray(rank_tolerance)
    support = jnp.asarray(support_tolerance)
    projected = output_moments.covariance @ _rank_aware_solve(
        output_moments.covariance, residual, tolerance
    )
    support_error = jnp.linalg.norm(residual - projected, axis=-1)
    supported = (support >= 0.0) & (support_error <= support)
    source_factors_valid = input_factor.valid & output_moments.factor.valid
    valid = conditioned.valid & output_moments.valid & source_factors_valid & supported
    status = jnp.where(
        ~source_factors_valid,
        CONDITIONAL_GAUSSIAN_INVALID_FACTOR,
        jnp.where(
            ~output_moments.valid,
            output_moments.status,
            jnp.where(
                ~supported,
                CONDITIONAL_GAUSSIAN_INCONSISTENT_CONDITION,
                conditioned.status,
            ),
        ),
    ).astype(jnp.int32)
    conditioned = eqx.tree_at(lambda node: node.valid, conditioned, valid)
    return eqx.tree_at(lambda node: node.status, conditioned, status)


def compose_gaussian_regressions(
    outer: GaussianRegression,
    inner: GaussianRegression,
    /,
    *,
    compress: bool = True,
    regression_id: str = "composed-gaussian-regression",
) -> GaussianRegression:
    """Compose independent affine regressions as ``outer(inner(input))``."""
    if not isinstance(outer, GaussianRegression) or not isinstance(
        inner, GaussianRegression
    ):
        raise TypeError("outer and inner must be GaussianRegression instances.")
    if outer.matrix.shape[-1] != inner.matrix.shape[-2]:
        raise ValueError("Regression input and output dimensions are incompatible.")
    matrix = outer.matrix @ inner.matrix
    offset = jnp.einsum("...ij,...j->...i", outer.matrix, inner.offset) + outer.offset
    propagated_inner_noise = GaussianFactor(
        outer.matrix @ inner.noise_factor.factor,
        regularization=inner.noise_factor.regularization,
        rank_tolerance=inner.noise_factor.rank_tolerance,
        factor_id=f"{regression_id}-propagated-inner-noise",
        resolved_method="affine-factor-pushforward",
    )
    noise = add_independent_gaussian_factors(
        propagated_inner_noise,
        outer.noise_factor,
        compress=compress,
        factor_id=f"{regression_id}-noise",
    )
    return GaussianRegression(
        matrix,
        offset,
        noise,
        regression_id=regression_id,
        resolved_method="independent-affine-regression-composition",
    )


def _condition_affine_gaussian_diagonal(
    prior_mean: ArrayLike,
    prior_covariance: ArrayLike,
    observation_matrix: ArrayLike,
    observation_offset: ArrayLike,
    observation_variance: ArrayLike,
    value: ArrayLike,
    mask: ArrayLike,
    /,
) -> tuple[Array, Array, Array, Array]:
    """Condition through diagonal observation noise in latent-state space."""
    mean = jnp.asarray(prior_mean)
    covariance = jnp.asarray(prior_covariance)
    matrix = jnp.asarray(observation_matrix)
    offset = jnp.asarray(observation_offset)
    variance = jnp.asarray(observation_variance)
    observed = jnp.asarray(mask, dtype=bool)
    state_size = mean.size
    observation_size = variance.size
    if mean.shape != (state_size,):
        raise ValueError("prior_mean must be a flat state vector.")
    if covariance.shape != (state_size, state_size):
        raise ValueError("prior_covariance has incompatible shape.")
    if matrix.shape != (observation_size, state_size):
        raise ValueError("observation_matrix has incompatible shape.")
    if offset.shape != (observation_size,):
        raise ValueError("observation_offset has incompatible shape.")
    if jnp.asarray(value).shape != (observation_size,):
        raise ValueError("value has incompatible shape.")
    if observed.shape != (observation_size,):
        raise ValueError("mask has incompatible shape.")

    variance_valid = jnp.isfinite(variance) & (variance > 0.0)
    safe_variance = jnp.where(variance_valid, variance, 1.0)
    effective_matrix = jnp.where(observed[:, None], matrix, 0.0)
    precision = jnp.where(observed, 1.0 / safe_variance, 0.0)
    residual = jnp.where(observed, jnp.asarray(value) - matrix @ mean - offset, 0.0)
    identity = jnp.eye(state_size, dtype=mean.dtype)
    prior_scale = jnp.linalg.cholesky(covariance)
    prior_information = jax.scipy.linalg.cho_solve((prior_scale, True), identity)
    posterior_information = prior_information + effective_matrix.T @ (
        precision[:, None] * effective_matrix
    )
    posterior_scale = jnp.linalg.cholesky(posterior_information)
    posterior_covariance = jax.scipy.linalg.cho_solve((posterior_scale, True), identity)
    projected = effective_matrix.T @ (precision * residual)
    posterior_mean = mean + posterior_covariance @ projected
    quadratic = residual @ (precision * residual) - projected @ (
        posterior_covariance @ projected
    )
    log_determinant = (
        jnp.sum(jnp.where(observed, jnp.log(safe_variance), 0.0))
        + 2.0 * jnp.sum(jnp.log(jnp.diag(prior_scale)))
        + 2.0 * jnp.sum(jnp.log(jnp.diag(posterior_scale)))
    )
    observed_count = jnp.sum(observed)
    log_likelihood = -0.5 * (
        quadratic + log_determinant + observed_count * jnp.log(2.0 * jnp.pi)
    )
    valid = (
        jnp.all(variance_valid | ~observed)
        & jnp.all(jnp.isfinite(prior_scale))
        & jnp.all(jnp.diag(prior_scale) > 0.0)
        & jnp.all(jnp.isfinite(posterior_scale))
        & jnp.all(jnp.diag(posterior_scale) > 0.0)
        & jnp.all(jnp.isfinite(posterior_mean))
        & jnp.all(jnp.isfinite(posterior_covariance))
        & jnp.isfinite(log_likelihood)
    )
    return (
        jnp.where(valid, posterior_mean, mean),
        jnp.where(
            valid,
            0.5 * (posterior_covariance + posterior_covariance.T),
            covariance,
        ),
        jnp.where(valid, log_likelihood, -jnp.inf),
        valid,
    )


__all__ = [
    "CONDITIONAL_GAUSSIAN_INCONSISTENT_CONDITION",
    "CONDITIONAL_GAUSSIAN_INVALID_FACTOR",
    "CONDITIONAL_GAUSSIAN_NONFINITE",
    "CONDITIONAL_GAUSSIAN_SUCCESS",
    "ConditionalGaussianMoments",
    "ConditionalGaussianStatus",
    "GaussianRegression",
    "compose_gaussian_regressions",
    "condition_gaussian",
    "predict_affine_gaussian",
]
