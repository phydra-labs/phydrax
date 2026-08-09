#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import prod

import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike

from ._base import (
    _broadcast_full,
    _normalize_axis,
    _prepare_pair,
    _real_dtype,
    _reduce_outputs,
    _reject_complex,
    _result,
    _weighted_mean,
    METRIC_ZERO_DENOMINATOR,
    MetricResult,
    OutputReduction,
)
from ._classification import _probability_inputs


def gaussian_negative_log_likelihood(
    y_true: ArrayLike,
    mean: ArrayLike,
    variance: ArrayLike,
    /,
    *,
    sample_weight: ArrayLike | None = None,
    mask: ArrayLike | None = None,
    sample_axis: int = -1,
    output_reduction: OutputReduction = "uniform_average",
) -> MetricResult:
    """Gaussian negative log likelihood including its normalization constant."""
    true, mean_, weights, active, invalid, axis = _prepare_pair(
        y_true,
        mean,
        sample_weight=sample_weight,
        mask=mask,
        sample_axis=sample_axis,
        metric="gaussian_negative_log_likelihood",
        allow_complex=False,
    )
    variance_ = jnp.broadcast_to(
        jnp.asarray(variance, dtype=_real_dtype(true, mean_)),
        true.shape,
    )
    variance_valid = jnp.isfinite(variance_) & (variance_ > 0.0)
    invalid = invalid | jnp.any(active & ~variance_valid, axis=axis)
    active = active & variance_valid
    loss = 0.5 * (
        jnp.log(2.0 * jnp.pi * jnp.where(variance_valid, variance_, 1.0))
        + (true - mean_) ** 2 / jnp.where(variance_valid, variance_, 1.0)
    )
    value, mass = _weighted_mean(loss, weights, active, axis)
    result = _result(value, invalid=invalid, effective_weight=mass)
    return _reduce_outputs(
        result,
        output_ndim=true.ndim - axis - 1,
        reduction=output_reduction,
    )


def dawid_sebastiani_score(
    y_true: ArrayLike,
    mean: ArrayLike,
    variance: ArrayLike,
    /,
    *,
    sample_weight: ArrayLike | None = None,
    mask: ArrayLike | None = None,
    sample_axis: int = -1,
    output_reduction: OutputReduction = "uniform_average",
) -> MetricResult:
    """Gaussian Dawid-Sebastiani score: log variance plus standardized error."""
    true, mean_, weights, active, invalid, axis = _prepare_pair(
        y_true,
        mean,
        sample_weight=sample_weight,
        mask=mask,
        sample_axis=sample_axis,
        metric="dawid_sebastiani_score",
        allow_complex=False,
    )
    variance_ = jnp.broadcast_to(
        jnp.asarray(variance, dtype=_real_dtype(true, mean_)),
        true.shape,
    )
    variance_valid = jnp.isfinite(variance_) & (variance_ > 0.0)
    invalid = invalid | jnp.any(active & ~variance_valid, axis=axis)
    active = active & variance_valid
    safe_variance = jnp.where(variance_valid, variance_, 1.0)
    loss = jnp.log(safe_variance) + (true - mean_) ** 2 / safe_variance
    value, mass = _weighted_mean(loss, weights, active, axis)
    result = _result(value, invalid=invalid, effective_weight=mass)
    return _reduce_outputs(
        result,
        output_ndim=true.ndim - axis - 1,
        reduction=output_reduction,
    )


def gaussian_crps(
    y_true: ArrayLike,
    mean: ArrayLike,
    scale: ArrayLike,
    /,
    *,
    sample_weight: ArrayLike | None = None,
    mask: ArrayLike | None = None,
    sample_axis: int = -1,
    output_reduction: OutputReduction = "uniform_average",
) -> MetricResult:
    """Closed-form continuous ranked probability score for a Gaussian forecast."""
    true, mean_, weights, active, invalid, axis = _prepare_pair(
        y_true,
        mean,
        sample_weight=sample_weight,
        mask=mask,
        sample_axis=sample_axis,
        metric="gaussian_crps",
        allow_complex=False,
    )
    scale_ = jnp.broadcast_to(
        jnp.asarray(scale, dtype=_real_dtype(true, mean_)),
        true.shape,
    )
    scale_valid = jnp.isfinite(scale_) & (scale_ > 0.0)
    invalid = invalid | jnp.any(active & ~scale_valid, axis=axis)
    active = active & scale_valid
    safe_scale = jnp.where(scale_valid, scale_, 1.0)
    standardized = (true - mean_) / safe_scale
    cdf = 0.5 * (1.0 + jax.lax.erf(standardized / jnp.sqrt(2.0)))
    density = jnp.exp(-0.5 * standardized**2) / jnp.sqrt(2.0 * jnp.pi)
    score = safe_scale * (
        standardized * (2.0 * cdf - 1.0) + 2.0 * density - 1.0 / jnp.sqrt(jnp.pi)
    )
    value, mass = _weighted_mean(score, weights, active, axis)
    result = _result(value, invalid=invalid, effective_weight=mass)
    return _reduce_outputs(
        result,
        output_ndim=true.ndim - axis - 1,
        reduction=output_reduction,
    )


def interval_score(
    y_true: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
    /,
    *,
    alpha: float = 0.05,
    sample_weight: ArrayLike | None = None,
    mask: ArrayLike | None = None,
    sample_axis: int = -1,
    output_reduction: OutputReduction = "uniform_average",
) -> MetricResult:
    """Central prediction-interval score at miscoverage level ``alpha``."""
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must lie strictly between zero and one.")
    true, lower_, weights, active, invalid, axis = _prepare_pair(
        y_true,
        lower,
        sample_weight=sample_weight,
        mask=mask,
        sample_axis=sample_axis,
        metric="interval_score",
        allow_complex=False,
    )
    upper_ = jnp.asarray(upper)
    _reject_complex(upper_, metric="interval_score")
    if upper_.shape != true.shape:
        raise ValueError("upper must have the same shape as y_true and lower.")
    upper_valid = jnp.isfinite(upper_)
    ordered = lower_ <= upper_
    invalid = invalid | jnp.any(active & ~(upper_valid & ordered), axis=axis)
    active = active & upper_valid & ordered
    width = upper_ - lower_
    score = (
        width
        + (2.0 / float(alpha)) * (lower_ - true) * (true < lower_)
        + (2.0 / float(alpha)) * (true - upper_) * (true > upper_)
    )
    value, mass = _weighted_mean(score, weights, active, axis)
    result = _result(value, invalid=invalid, effective_weight=mass)
    return _reduce_outputs(
        result,
        output_ndim=true.ndim - axis - 1,
        reduction=output_reduction,
    )


def ranked_probability_score(
    y_true: ArrayLike,
    probability: ArrayLike,
    /,
    *,
    sample_weight: ArrayLike | None = None,
    mask: ArrayLike | None = None,
    sample_axis: int = -1,
    from_logits: bool = False,
) -> MetricResult:
    """Ranked probability score for ordered categorical outcomes."""
    labels, probabilities, weights, active, invalid, classes = _probability_inputs(
        y_true,
        probability,
        sample_weight=sample_weight,
        mask=mask,
        sample_axis=sample_axis,
        metric="ranked_probability_score",
        from_logits=from_logits,
    )
    target = jax.nn.one_hot(labels, classes, dtype=probabilities.dtype)
    probability_cdf = jnp.cumsum(probabilities, axis=-1)[..., :-1]
    target_cdf = jnp.cumsum(target, axis=-1)[..., :-1]
    per_sample = jnp.sum((probability_cdf - target_cdf) ** 2, axis=-1)
    value, mass = _weighted_mean(per_sample, weights, active, labels.ndim - 1)
    return _result(value, invalid=invalid, effective_weight=mass)


def spherical_score(
    y_true: ArrayLike,
    probability: ArrayLike,
    /,
    *,
    sample_weight: ArrayLike | None = None,
    mask: ArrayLike | None = None,
    sample_axis: int = -1,
    from_logits: bool = False,
) -> MetricResult:
    """Positively oriented categorical spherical score (larger is better)."""
    labels, probabilities, weights, active, invalid, classes = _probability_inputs(
        y_true,
        probability,
        sample_weight=sample_weight,
        mask=mask,
        sample_axis=sample_axis,
        metric="spherical_score",
        from_logits=from_logits,
    )
    selected = jnp.sum(
        probabilities * jax.nn.one_hot(labels, classes, dtype=probabilities.dtype),
        axis=-1,
    )
    norm = jnp.linalg.norm(probabilities, axis=-1)
    per_sample = selected / jnp.where(norm > 0.0, norm, 1.0)
    value, mass = _weighted_mean(per_sample, weights, active, labels.ndim - 1)
    return _result(
        value,
        invalid=invalid,
        effective_weight=mass,
        undefined=jnp.any(active & (norm <= 0.0), axis=-1),
        undefined_status=METRIC_ZERO_DENOMINATOR,
    )


def _ensemble_crps(
    y_true: ArrayLike,
    ensemble: ArrayLike,
    /,
    *,
    sample_weight: ArrayLike | None,
    mask: ArrayLike | None,
    sample_axis: int,
    member_weight: ArrayLike | None,
    member_mask: ArrayLike | None,
    smoothing: float | None,
    metric: str,
    output_reduction: OutputReduction,
) -> MetricResult:
    true = jnp.asarray(y_true)
    forecast = jnp.asarray(ensemble)
    _reject_complex(true, forecast, metric=metric)
    if forecast.shape[:-1] != true.shape or forecast.shape[-1] <= 0:
        raise ValueError(f"{metric} requires ensemble.shape == y_true.shape + (member,).")
    true_, _, weights, active, invalid, axis = _prepare_pair(
        true,
        true,
        sample_weight=sample_weight,
        mask=mask,
        sample_axis=sample_axis,
        metric=metric,
        allow_complex=False,
    )
    member_shape = tuple(int(size) for size in forecast.shape)
    member_dtype = _real_dtype(true, forecast)
    if member_weight is None:
        member_weight_ = jnp.ones(member_shape, dtype=member_dtype)
    else:
        member_weight_ = jnp.broadcast_to(
            jnp.asarray(member_weight, dtype=member_dtype),
            member_shape,
        )
    member_included = _broadcast_full(
        member_mask,
        member_shape,
        dtype=bool,
        fill=True,
        name="member_mask",
    )
    valid_member = (
        member_included
        & jnp.isfinite(forecast)
        & jnp.isfinite(member_weight_)
        & (member_weight_ >= 0.0)
    )
    invalid_member = jnp.any(
        member_included
        & ~(
            jnp.isfinite(forecast)
            & jnp.isfinite(member_weight_)
            & (member_weight_ >= 0.0)
        ),
        axis=-1,
    )
    safe_member_weight = jnp.where(valid_member, member_weight_, 0.0)
    member_mass = jnp.sum(safe_member_weight, axis=-1)
    normalized = safe_member_weight / jnp.where(
        member_mass[..., None] > 0.0, member_mass[..., None], 1.0
    )
    point_difference = forecast - true_[..., None]
    pair_difference = forecast[..., :, None] - forecast[..., None, :]
    if smoothing is None:
        point_distance = jnp.abs(point_difference)
        pair_distance = jnp.abs(pair_difference)
    else:
        epsilon = float(smoothing)
        point_distance = jnp.sqrt(point_difference**2 + epsilon**2) - epsilon
        pair_distance = jnp.sqrt(pair_difference**2 + epsilon**2) - epsilon
    first = jnp.sum(normalized * point_distance, axis=-1)
    second = 0.5 * jnp.sum(
        normalized[..., :, None] * normalized[..., None, :] * pair_distance,
        axis=(-2, -1),
    )
    per_value = first - second
    invalid = invalid | jnp.any(active & invalid_member, axis=axis)
    member_empty = jnp.any(active & (member_mass <= 0.0), axis=axis)
    value, mass = _weighted_mean(per_value, weights, active, axis)
    result = _result(
        value,
        invalid=invalid,
        effective_weight=mass,
        undefined=member_empty,
        undefined_status=METRIC_ZERO_DENOMINATOR,
    )
    return _reduce_outputs(
        result,
        output_ndim=true.ndim - axis - 1,
        reduction=output_reduction,
    )


def crps_ensemble(
    y_true: ArrayLike,
    ensemble: ArrayLike,
    /,
    *,
    sample_weight: ArrayLike | None = None,
    mask: ArrayLike | None = None,
    sample_axis: int = -1,
    member_weight: ArrayLike | None = None,
    member_mask: ArrayLike | None = None,
    output_reduction: OutputReduction = "uniform_average",
) -> MetricResult:
    """Exact empirical CRPS, almost everywhere differentiable without sorting."""
    return _ensemble_crps(
        y_true,
        ensemble,
        sample_weight=sample_weight,
        mask=mask,
        sample_axis=sample_axis,
        member_weight=member_weight,
        member_mask=member_mask,
        smoothing=None,
        metric="crps_ensemble",
        output_reduction=output_reduction,
    )


def smooth_crps_ensemble(
    y_true: ArrayLike,
    ensemble: ArrayLike,
    /,
    *,
    smoothing: float = 1e-3,
    sample_weight: ArrayLike | None = None,
    mask: ArrayLike | None = None,
    sample_axis: int = -1,
    member_weight: ArrayLike | None = None,
    member_mask: ArrayLike | None = None,
    output_reduction: OutputReduction = "uniform_average",
) -> MetricResult:
    """Everywhere-smooth pseudo-Huber-distance CRPS surrogate."""
    if smoothing <= 0.0:
        raise ValueError("smoothing must be positive.")
    return _ensemble_crps(
        y_true,
        ensemble,
        sample_weight=sample_weight,
        mask=mask,
        sample_axis=sample_axis,
        member_weight=member_weight,
        member_mask=member_mask,
        smoothing=smoothing,
        metric="smooth_crps_ensemble",
        output_reduction=output_reduction,
    )


def _energy_score(
    y_true: ArrayLike,
    ensemble: ArrayLike,
    /,
    *,
    beta: float,
    sample_weight: ArrayLike | None,
    mask: ArrayLike | None,
    sample_axis: int,
    member_weight: ArrayLike | None,
    member_mask: ArrayLike | None,
    smoothing: float | None,
    metric: str,
) -> MetricResult:
    if not 0.0 < float(beta) < 2.0:
        raise ValueError("beta must lie strictly between zero and two.")
    true = jnp.asarray(y_true)
    forecast = jnp.asarray(ensemble)
    if true.ndim < 2:
        raise ValueError(f"{metric} requires an event axis after the sample axis.")
    axis = _normalize_axis(sample_axis, true.ndim)
    if axis >= true.ndim - 1:
        raise ValueError(f"{metric} requires one or more event axes after sample_axis.")
    if forecast.shape[:-1] != true.shape or forecast.shape[-1] <= 0:
        raise ValueError(f"{metric} requires ensemble.shape == y_true.shape + (member,).")
    prefix = true.shape[: axis + 1]
    event_size = prod(true.shape[axis + 1 :])
    member_count = forecast.shape[-1]
    true_flat = true.reshape(prefix + (event_size,))
    forecast_flat = jnp.swapaxes(
        forecast.reshape(prefix + (event_size, member_count)), -1, -2
    )
    feature_finite = jnp.all(jnp.isfinite(true_flat), axis=-1)
    forecast_finite = jnp.all(jnp.isfinite(forecast_flat), axis=-1)
    dummy = jnp.zeros(prefix, dtype=true.real.dtype)
    _, _, sample_weights, active, invalid, _ = _prepare_pair(
        dummy,
        dummy,
        sample_weight=sample_weight,
        mask=mask,
        sample_axis=-1,
        metric=metric,
        allow_complex=False,
    )
    invalid = invalid | jnp.any(active & ~feature_finite, axis=-1)
    member_shape = prefix + (member_count,)
    dtype = _real_dtype(true, forecast)
    if member_weight is None:
        member_weight_ = jnp.ones(member_shape, dtype=dtype)
    else:
        member_weight_ = jnp.broadcast_to(
            jnp.asarray(member_weight, dtype=dtype),
            member_shape,
        )
    included_member = _broadcast_full(
        member_mask,
        member_shape,
        dtype=bool,
        fill=True,
        name="member_mask",
    )
    member_valid = (
        included_member
        & forecast_finite
        & jnp.isfinite(member_weight_)
        & (member_weight_ >= 0.0)
    )
    invalid_member = jnp.any(
        included_member
        & ~(forecast_finite & jnp.isfinite(member_weight_) & (member_weight_ >= 0.0)),
        axis=-1,
    )
    safe_member_weight = jnp.where(member_valid, member_weight_, 0.0)
    member_mass = jnp.sum(safe_member_weight, axis=-1)
    normalized = safe_member_weight / jnp.where(
        member_mass[..., None] > 0.0, member_mass[..., None], 1.0
    )
    point_squared = jnp.sum(
        jnp.real(
            (forecast_flat - true_flat[..., None, :])
            * jnp.conj(forecast_flat - true_flat[..., None, :])
        ),
        axis=-1,
    )
    pair_delta = forecast_flat[..., :, None, :] - forecast_flat[..., None, :, :]
    pair_squared = jnp.sum(jnp.real(pair_delta * jnp.conj(pair_delta)), axis=-1)
    if smoothing is not None:
        point_squared = point_squared + float(smoothing) ** 2
        pair_squared = pair_squared + float(smoothing) ** 2
    exponent = 0.5 * float(beta)
    point_distance = point_squared**exponent
    pair_distance = pair_squared**exponent
    if smoothing is not None:
        offset = float(smoothing) ** float(beta)
        point_distance = point_distance - offset
        pair_distance = pair_distance - offset
    first = jnp.sum(normalized * point_distance, axis=-1)
    second = 0.5 * jnp.sum(
        normalized[..., :, None] * normalized[..., None, :] * pair_distance,
        axis=(-2, -1),
    )
    per_sample = first - second
    invalid = invalid | jnp.any(active & invalid_member, axis=-1)
    member_empty = jnp.any(active & (member_mass <= 0.0), axis=-1)
    active = active & feature_finite
    value, mass = _weighted_mean(per_sample, sample_weights, active, -1)
    return _result(
        value,
        invalid=invalid,
        effective_weight=mass,
        undefined=member_empty,
        undefined_status=METRIC_ZERO_DENOMINATOR,
    )


def energy_score(
    y_true: ArrayLike,
    ensemble: ArrayLike,
    /,
    *,
    beta: float = 1.0,
    sample_weight: ArrayLike | None = None,
    mask: ArrayLike | None = None,
    sample_axis: int = -2,
    member_weight: ArrayLike | None = None,
    member_mask: ArrayLike | None = None,
) -> MetricResult:
    """Multivariate empirical energy score; complex event vectors are supported."""
    return _energy_score(
        y_true,
        ensemble,
        beta=beta,
        sample_weight=sample_weight,
        mask=mask,
        sample_axis=sample_axis,
        member_weight=member_weight,
        member_mask=member_mask,
        smoothing=None,
        metric="energy_score",
    )


def smooth_energy_score(
    y_true: ArrayLike,
    ensemble: ArrayLike,
    /,
    *,
    beta: float = 1.0,
    smoothing: float = 1e-3,
    sample_weight: ArrayLike | None = None,
    mask: ArrayLike | None = None,
    sample_axis: int = -2,
    member_weight: ArrayLike | None = None,
    member_mask: ArrayLike | None = None,
) -> MetricResult:
    """Smoothed-norm empirical energy-score surrogate."""
    if smoothing <= 0.0:
        raise ValueError("smoothing must be positive.")
    return _energy_score(
        y_true,
        ensemble,
        beta=beta,
        sample_weight=sample_weight,
        mask=mask,
        sample_axis=sample_axis,
        member_weight=member_weight,
        member_mask=member_mask,
        smoothing=smoothing,
        metric="smooth_energy_score",
    )


__all__ = [
    "crps_ensemble",
    "dawid_sebastiani_score",
    "energy_score",
    "gaussian_crps",
    "gaussian_negative_log_likelihood",
    "interval_score",
    "ranked_probability_score",
    "smooth_crps_ensemble",
    "smooth_energy_score",
    "spherical_score",
]
