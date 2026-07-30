#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ._likelihoods import AbstractLikelihood


Reduction = Literal["mean", "sum", "none"]


def negative_log_likelihood(
    likelihood: AbstractLikelihood,
    location: ArrayLike,
    target: ArrayLike,
    /,
    *,
    reduction: Reduction = "mean",
    mask: ArrayLike | None = None,
    weights: ArrayLike | None = None,
    **parameters,
) -> Array:
    values = -likelihood.log_prob(location, target, **parameters)
    return _reduce(values, reduction=reduction, mask=mask, weights=weights)


def gaussian_crps(location: ArrayLike, scale: ArrayLike, target: ArrayLike, /) -> Array:
    """Closed-form CRPS for a Gaussian predictive distribution."""
    location_array = jnp.asarray(location, dtype=float)
    scale_array = jnp.asarray(scale, dtype=float)
    target_array = jnp.asarray(target, dtype=float)
    z = (target_array - location_array) / scale_array
    phi = jnp.exp(-0.5 * z**2) / jnp.sqrt(2.0 * jnp.pi)
    cdf = jsp.special.ndtr(z)
    score = scale_array * (z * (2.0 * cdf - 1.0) + 2.0 * phi - 1.0 / jnp.sqrt(jnp.pi))
    return jnp.where(scale_array > 0.0, score, jnp.nan)


def student_t_crps(
    location: ArrayLike,
    scale: ArrayLike,
    df: ArrayLike,
    target: ArrayLike,
    /,
) -> Array:
    """Closed-form CRPS for Student-t predictions with ``df > 1``."""
    location_array = jnp.asarray(location, dtype=float)
    scale_array = jnp.asarray(scale, dtype=float)
    df_array = jnp.asarray(df, dtype=float)
    target_array = jnp.asarray(target, dtype=float)
    z = (target_array - location_array) / scale_array
    x = df_array / (df_array + z**2)
    beta_tail = jsp.special.betainc(df_array / 2.0, 0.5, x)
    cdf = 0.5 + 0.5 * jnp.sign(z) * (1.0 - beta_tail)
    log_pdf = (
        jsp.special.gammaln((df_array + 1.0) / 2.0)
        - jsp.special.gammaln(df_array / 2.0)
        - 0.5 * jnp.log(df_array * jnp.pi)
        - 0.5 * (df_array + 1.0) * jnp.log1p(z**2 / df_array)
    )
    log_beta_num = jsp.special.betaln(0.5, df_array - 0.5)
    log_beta_den = jsp.special.betaln(0.5, df_array / 2.0)
    constant = (
        2.0
        * jnp.sqrt(df_array)
        / (df_array - 1.0)
        * jnp.exp(log_beta_num - 2.0 * log_beta_den)
    )
    standardized_score = (
        z * (2.0 * cdf - 1.0)
        + 2.0 * jnp.exp(log_pdf) * (df_array + z**2) / (df_array - 1.0)
        - constant
    )
    valid = (df_array > 1.0) & (scale_array > 0.0)
    return jnp.where(valid, scale_array * standardized_score, jnp.nan)


def ensemble_crps(
    samples: ArrayLike,
    target: ArrayLike,
    /,
    *,
    sample_axis: int = 0,
) -> Array:
    """Univariate empirical CRPS in ``O(M log M)`` time and ``O(M)`` memory."""
    sample_array = jnp.moveaxis(jnp.asarray(samples, dtype=float), sample_axis, 0)
    count = int(sample_array.shape[0])
    if count <= 0:
        raise ValueError("samples must contain at least one ensemble member.")
    target_array = jnp.asarray(target, dtype=float)
    first = jnp.mean(jnp.abs(sample_array - target_array), axis=0)
    ordered = jnp.sort(sample_array, axis=0)
    coefficients = (2 * jnp.arange(count) - count + 1).reshape(
        (count,) + (1,) * (ordered.ndim - 1)
    )
    pair_term = jnp.sum(coefficients * ordered, axis=0) / float(count**2)
    return first - pair_term


def _powered_euclidean_norm(difference: Array, exponent: float, /) -> Array:
    squared = jnp.sum(jnp.asarray(difference) ** 2, axis=-1)
    if exponent == 2.0:
        return squared
    positive = squared > 0.0
    safe_squared = jnp.where(positive, squared, jnp.ones_like(squared))
    return jnp.where(positive, safe_squared ** (0.5 * exponent), 0.0)


def energy_score(
    samples: ArrayLike,
    target: ArrayLike,
    /,
    *,
    sample_axis: int = 0,
    beta: float = 1.0,
    chunk_size: int | None = None,
) -> Array:
    """Multivariate energy score without materializing the full pairwise matrix."""
    exponent = float(beta)
    if not 0.0 < exponent <= 2.0:
        raise ValueError("beta must satisfy 0 < beta <= 2.")
    sample_array = jnp.moveaxis(jnp.asarray(samples, dtype=float), sample_axis, 0)
    count = int(sample_array.shape[0])
    if count <= 0:
        raise ValueError("samples must contain at least one ensemble member.")
    flat = sample_array.reshape((count, -1))
    target_flat = jnp.asarray(target, dtype=float).reshape((-1,))
    if target_flat.shape[0] != flat.shape[1]:
        raise ValueError("target shape must match one sample's event shape.")
    first = jnp.mean(_powered_euclidean_norm(flat - target_flat, exponent))
    block = count if chunk_size is None else int(chunk_size)
    if block <= 0:
        raise ValueError("chunk_size must be positive.")
    pair_sum = jnp.asarray(0.0, dtype=flat.dtype)
    for start_i in range(0, count, block):
        left = flat[start_i : start_i + block]
        for start_j in range(0, count, block):
            right = flat[start_j : start_j + block]
            distances = _powered_euclidean_norm(
                left[:, None, :] - right[None, :, :],
                exponent,
            )
            pair_sum = pair_sum + jnp.sum(distances)
    return first - 0.5 * pair_sum / float(count**2)


def energy_distance(
    left_samples: ArrayLike,
    right_samples: ArrayLike,
    /,
    *,
    sample_axis: int = 0,
    beta: float = 1.0,
    chunk_size: int | None = None,
) -> Array:
    """Empirical multivariate energy distance between two sample ensembles."""
    exponent = float(beta)
    if not 0.0 < exponent <= 2.0:
        raise ValueError("beta must satisfy 0 < beta <= 2.")
    left = jnp.moveaxis(jnp.asarray(left_samples, dtype=float), sample_axis, 0)
    right = jnp.moveaxis(jnp.asarray(right_samples, dtype=float), sample_axis, 0)
    if left.shape[1:] != right.shape[1:]:
        raise ValueError("Energy-distance ensembles must have equal event shapes.")
    left_count, right_count = int(left.shape[0]), int(right.shape[0])
    if left_count <= 0 or right_count <= 0:
        raise ValueError("Energy-distance ensembles must be non-empty.")
    left_flat = left.reshape((left_count, -1))
    right_flat = right.reshape((right_count, -1))
    block = max(left_count, right_count) if chunk_size is None else int(chunk_size)
    if block <= 0:
        raise ValueError("chunk_size must be positive.")

    def pair_mean(first: Array, second: Array) -> Array:
        total = jnp.asarray(0.0, dtype=first.dtype)
        for start_i in range(0, int(first.shape[0]), block):
            first_block = first[start_i : start_i + block]
            for start_j in range(0, int(second.shape[0]), block):
                second_block = second[start_j : start_j + block]
                distances = _powered_euclidean_norm(
                    first_block[:, None, :] - second_block[None, :, :],
                    exponent,
                )
                total = total + jnp.sum(distances)
        return total / float(int(first.shape[0]) * int(second.shape[0]))

    cross = pair_mean(left_flat, right_flat)
    within_left = pair_mean(left_flat, left_flat)
    within_right = pair_mean(right_flat, right_flat)
    return 2.0 * cross - within_left - within_right


def pinball_loss(prediction: ArrayLike, target: ArrayLike, quantile: float, /) -> Array:
    level = float(quantile)
    if not 0.0 < level < 1.0:
        raise ValueError("quantile must lie strictly between zero and one.")
    error = jnp.asarray(target, dtype=float) - jnp.asarray(prediction, dtype=float)
    return jnp.maximum(level * error, (level - 1.0) * error)


def interval_coverage(
    lower: ArrayLike,
    upper: ArrayLike,
    target: ArrayLike,
    /,
    *,
    reduction: Reduction = "mean",
    mask: ArrayLike | None = None,
    weights: ArrayLike | None = None,
) -> Array:
    target_array = jnp.asarray(target, dtype=float)
    covered = (target_array >= jnp.asarray(lower)) & (target_array <= jnp.asarray(upper))
    return _reduce(covered.astype(float), reduction=reduction, mask=mask, weights=weights)


def interval_width(
    lower: ArrayLike,
    upper: ArrayLike,
    /,
    *,
    reduction: Reduction = "mean",
    mask: ArrayLike | None = None,
    weights: ArrayLike | None = None,
) -> Array:
    width = jnp.asarray(upper, dtype=float) - jnp.asarray(lower, dtype=float)
    return _reduce(width, reduction=reduction, mask=mask, weights=weights)


def calibration_error(nominal: ArrayLike, empirical: ArrayLike, /) -> Array:
    nominal_array = jnp.asarray(nominal, dtype=float)
    empirical_array = jnp.asarray(empirical, dtype=float)
    return jnp.mean(jnp.abs(empirical_array - nominal_array))


class GaussianScaleCalibrator(StrictModule):
    """Closed-form held-out Gaussian scale-multiplier calibration."""

    scale_multiplier: Array

    def __init__(self, scale_multiplier: ArrayLike):
        multiplier = jnp.asarray(scale_multiplier, dtype=float).reshape(())
        if not bool(jnp.isfinite(multiplier)) or not bool(multiplier > 0.0):
            raise ValueError("scale_multiplier must be finite and positive.")
        self.scale_multiplier = multiplier

    @classmethod
    def fit(
        cls,
        location: ArrayLike,
        scale: ArrayLike,
        target: ArrayLike,
        /,
        *,
        mask: ArrayLike | None = None,
        weights: ArrayLike | None = None,
    ) -> "GaussianScaleCalibrator":
        scale_array = jnp.asarray(scale, dtype=float)
        if bool(jnp.any(~jnp.isfinite(scale_array))) or bool(jnp.any(scale_array <= 0.0)):
            raise ValueError("scale must be finite and strictly positive.")
        standardized_squared = (
            (jnp.asarray(target, dtype=float) - jnp.asarray(location, dtype=float))
            / scale_array
        ) ** 2
        mean_square = _reduce(
            standardized_squared,
            reduction="mean",
            mask=mask,
            weights=weights,
        )
        multiplier = jnp.sqrt(mean_square)
        if not bool(jnp.isfinite(multiplier)) or not bool(multiplier > 0.0):
            raise ValueError("Calibration data imply a non-positive or non-finite scale.")
        return cls(multiplier)

    def transform(self, scale: ArrayLike, /) -> Array:
        return self.scale_multiplier * jnp.asarray(scale, dtype=float)

    def __call__(self, scale: ArrayLike, /) -> Array:
        return self.transform(scale)


def _reduce(
    values: ArrayLike,
    /,
    *,
    reduction: Reduction,
    mask: ArrayLike | None,
    weights: ArrayLike | None,
) -> Array:
    value_array = jnp.asarray(values, dtype=float)
    if reduction == "none":
        if mask is not None or weights is not None:
            raise ValueError("mask and weights require a scalar reduction.")
        return value_array
    effective_weight = jnp.ones_like(value_array, dtype=float)
    if mask is not None:
        effective_weight = effective_weight * jnp.asarray(mask, dtype=bool)
    if weights is not None:
        weight_array = jnp.asarray(weights, dtype=float)
        if bool(jnp.any(~jnp.isfinite(weight_array))) or bool(
            jnp.any(weight_array < 0.0)
        ):
            raise ValueError("weights must be finite and non-negative.")
        effective_weight = effective_weight * weight_array
    weighted = jnp.where(effective_weight > 0.0, value_array * effective_weight, 0.0)
    if reduction == "sum":
        return jnp.sum(weighted)
    if reduction != "mean":
        raise ValueError("reduction must be 'mean', 'sum', or 'none'.")
    denominator = jnp.sum(effective_weight)
    return jnp.where(denominator > 0.0, jnp.sum(weighted) / denominator, jnp.nan)


__all__ = [
    "GaussianScaleCalibrator",
    "calibration_error",
    "energy_score",
    "energy_distance",
    "ensemble_crps",
    "gaussian_crps",
    "interval_coverage",
    "interval_width",
    "negative_log_likelihood",
    "pinball_loss",
    "student_t_crps",
]
