#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

import phydrax.linalg as la

from .._strict import StrictModule
from ._gaussian_factor import (
    gaussian_factor_from_covariance,
    gaussian_factor_log_determinant,
    gaussian_factor_quadratic_form,
)


class GaussianConditioningResult(StrictModule):
    """One affine Gaussian observation update with numerical evidence."""

    mean: Array
    covariance: Array
    innovation: Array
    normalized_innovation_squared: Array
    log_likelihood: Array
    observed_count: Array
    valid: Array
    finite: Array


def condition_gaussian_moments(
    predicted_mean: ArrayLike,
    predicted_covariance: ArrayLike,
    observation_mean: ArrayLike,
    observation_covariance: ArrayLike,
    cross_covariance: ArrayLike,
    observation: ArrayLike,
    /,
    *,
    mask: ArrayLike | None = None,
    covariance_regularization: float = 0.0,
    rank_tolerance: float = 0.0,
    moments_valid: Any = True,
) -> GaussianConditioningResult:
    """Condition Gaussian moments without reconstructing an observation Jacobian."""
    mean = jnp.asarray(predicted_mean)
    covariance = jnp.asarray(predicted_covariance, dtype=mean.dtype)
    observed_mean = jnp.asarray(observation_mean, dtype=mean.dtype).reshape((-1,))
    observed_covariance = jnp.asarray(observation_covariance, dtype=mean.dtype)
    cross = jnp.asarray(cross_covariance, dtype=mean.dtype)
    value = jnp.asarray(observation, dtype=mean.dtype).reshape((-1,))
    if mean.ndim != 1 or covariance.shape != (mean.size, mean.size):
        raise ValueError(
            "Predicted Gaussian moments must contain one vector and square covariance."
        )
    observation_size = int(observed_mean.size)
    if (
        value.shape != (observation_size,)
        or observed_covariance.shape != (observation_size, observation_size)
        or cross.shape != (mean.size, observation_size)
    ):
        raise ValueError("Observation and cross-covariance shapes are incompatible.")
    active_mask = (
        jnp.ones((observation_size,), dtype=bool)
        if mask is None
        else jnp.asarray(mask, dtype=bool).reshape((observation_size,))
    )
    regularization = float(covariance_regularization)
    tolerance = float(rank_tolerance)
    if regularization < 0.0 or tolerance < 0.0:
        raise ValueError("Gaussian conditioning tolerances must be non-negative.")

    active = active_mask.astype(mean.dtype)
    innovation = jnp.where(active_mask, value - observed_mean, 0.0)
    identity = jnp.eye(observation_size, dtype=mean.dtype)
    effective_covariance_raw = (
        observed_covariance * active[:, None] * active[None, :]
        + identity * (1.0 - active[:, None])
        + regularization * identity * active[:, None]
    )
    predicted_hermitian_defect = jnp.max(
        jnp.abs(covariance - jnp.conj(covariance.T)),
        initial=0.0,
    )
    observation_hermitian_defect = jnp.max(
        jnp.abs(observed_covariance - jnp.conj(observed_covariance.T)),
        initial=0.0,
    )
    operands_valid = (
        jnp.asarray(moments_valid, dtype=bool)
        & (predicted_hermitian_defect <= tolerance)
        & (observation_hermitian_defect <= tolerance)
        & jnp.all(jnp.isfinite(mean))
        & jnp.all(jnp.isfinite(covariance))
        & jnp.all(jnp.isfinite(observed_mean))
        & jnp.all(jnp.isfinite(observed_covariance))
        & jnp.all(jnp.isfinite(cross))
    )
    symmetric_effective = 0.5 * (
        effective_covariance_raw + jnp.conj(effective_covariance_raw.T)
    )
    effective_covariance = jnp.where(
        operands_valid,
        symmetric_effective,
        effective_covariance_raw,
    )
    effective_cross = cross * active[None, :]
    innovation_factor = gaussian_factor_from_covariance(
        effective_covariance,
        rank_tolerance=tolerance,
        hermitian_tolerance=tolerance,
        factor_id="gaussian-conditioning-innovation",
    )
    observed_count = jnp.sum(active_mask, dtype=jnp.int32)
    can_solve = (
        operands_valid
        & innovation_factor.valid
        & (innovation_factor.numerical_rank == observation_size)
        & jnp.all(jnp.isfinite(innovation))
    )

    def solve_update(_):
        solve_result = la.solve(
            la.LinearSystem(
                la.DenseLinearOperator(
                    effective_covariance,
                    properties=la.OperatorProperties(
                        self_adjoint=True,
                        positive_definite=True,
                        evidence={
                            "self_adjoint": "construction",
                            "positive_definite": "verified",
                        },
                    ),
                )
            ),
            jnp.conj(effective_cross.T),
            policy=la.LinearSolvePolicy(la.DenseCholesky()),
        )
        gain = jnp.conj(jnp.asarray(solve_result.value).T)
        filtered_mean = mean + gain @ innovation
        filtered_covariance_raw = covariance - gain @ jnp.conj(effective_cross.T)
        filtered_covariance = 0.5 * (
            filtered_covariance_raw + jnp.conj(filtered_covariance_raw.T)
        )
        filtered_factor = gaussian_factor_from_covariance(
            filtered_covariance,
            rank_tolerance=tolerance,
            hermitian_tolerance=tolerance,
            factor_id="gaussian-conditioning-filtered",
        )
        quadratic = gaussian_factor_quadratic_form(
            innovation_factor,
            innovation,
            rank_tolerance=tolerance,
            support_tolerance=tolerance,
        )
        logdet = gaussian_factor_log_determinant(
            innovation_factor,
            rank_tolerance=tolerance,
        )
        log_likelihood = -0.5 * (
            quadratic + logdet + observed_count * jnp.log(2.0 * jnp.pi)
        )
        solve_successful = jnp.all(solve_result.successful)
        finite = (
            solve_successful
            & jnp.all(jnp.isfinite(filtered_mean))
            & jnp.all(jnp.isfinite(filtered_covariance_raw))
            & jnp.isfinite(log_likelihood)
        )
        valid = solve_successful & filtered_factor.valid & finite
        return (
            filtered_mean,
            filtered_covariance,
            jnp.where(observed_count > 0, quadratic, 0.0),
            log_likelihood,
            valid,
            finite,
        )

    def skip_update(_):
        finite = (
            jnp.all(jnp.isfinite(mean))
            & jnp.all(jnp.isfinite(covariance))
            & jnp.all(jnp.isfinite(innovation))
            & jnp.all(jnp.isfinite(effective_covariance_raw))
        )
        return (
            mean,
            covariance,
            jnp.asarray(0.0, dtype=jnp.real(mean).dtype),
            jnp.asarray(0.0, dtype=jnp.real(mean).dtype),
            jnp.asarray(False),
            finite,
        )

    filtered_mean, filtered_covariance, nis, log_likelihood, valid, finite = jax.lax.cond(
        can_solve, solve_update, skip_update, None
    )
    return GaussianConditioningResult(
        filtered_mean,
        filtered_covariance,
        innovation,
        nis,
        log_likelihood,
        observed_count,
        valid,
        finite,
    )


__all__ = ["GaussianConditioningResult", "condition_gaussian_moments"]
