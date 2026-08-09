#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import ArrayLike

from ._base import (
    _prepare_pair,
    _reduce_outputs,
    _result,
    _weighted_mean,
    _weighted_sum,
    METRIC_ZERO_DENOMINATOR,
    MetricResult,
    OutputReduction,
)


def _squared_magnitude(value):
    return jnp.real(value * jnp.conj(value))


def mean_squared_error(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    /,
    *,
    sample_weight: ArrayLike | None = None,
    mask: ArrayLike | None = None,
    sample_axis: int = -1,
    output_reduction: OutputReduction = "uniform_average",
) -> MetricResult:
    """Weighted mean squared magnitude error.

    Axes before ``sample_axis`` are case axes and axes after it are output axes.
    Complex targets are supported through the squared magnitude of the residual.
    """
    true, pred, weights, active, invalid, axis = _prepare_pair(
        y_true,
        y_pred,
        sample_weight=sample_weight,
        mask=mask,
        sample_axis=sample_axis,
        metric="mean_squared_error",
        allow_complex=True,
    )
    dtype = jnp.result_type(true.dtype, pred.dtype, jnp.float32)
    error = true.astype(dtype) - pred.astype(dtype)
    value, mass = _weighted_mean(_squared_magnitude(error), weights, active, axis)
    result = _result(value, invalid=invalid, effective_weight=mass)
    return _reduce_outputs(
        result,
        output_ndim=true.ndim - axis - 1,
        reduction=output_reduction,
    )


def root_mean_squared_error(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    /,
    *,
    sample_weight: ArrayLike | None = None,
    mask: ArrayLike | None = None,
    sample_axis: int = -1,
    output_reduction: OutputReduction = "uniform_average",
) -> MetricResult:
    """Square root of each output's weighted MSE, then output reduction."""
    true, pred, weights, active, invalid, axis = _prepare_pair(
        y_true,
        y_pred,
        sample_weight=sample_weight,
        mask=mask,
        sample_axis=sample_axis,
        metric="root_mean_squared_error",
        allow_complex=True,
    )
    dtype = jnp.result_type(true.dtype, pred.dtype, jnp.float32)
    error = true.astype(dtype) - pred.astype(dtype)
    mse, mass = _weighted_mean(_squared_magnitude(error), weights, active, axis)
    result = _result(
        jnp.sqrt(jnp.maximum(mse, 0.0)), invalid=invalid, effective_weight=mass
    )
    return _reduce_outputs(
        result,
        output_ndim=true.ndim - axis - 1,
        reduction=output_reduction,
    )


def mean_absolute_error(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    /,
    *,
    sample_weight: ArrayLike | None = None,
    mask: ArrayLike | None = None,
    sample_axis: int = -1,
    output_reduction: OutputReduction = "uniform_average",
) -> MetricResult:
    """Weighted mean absolute error; complex errors use their modulus."""
    true, pred, weights, active, invalid, axis = _prepare_pair(
        y_true,
        y_pred,
        sample_weight=sample_weight,
        mask=mask,
        sample_axis=sample_axis,
        metric="mean_absolute_error",
        allow_complex=True,
    )
    dtype = jnp.result_type(true.dtype, pred.dtype, jnp.float32)
    value, mass = _weighted_mean(
        jnp.abs(true.astype(dtype) - pred.astype(dtype)), weights, active, axis
    )
    result = _result(value, invalid=invalid, effective_weight=mass)
    return _reduce_outputs(
        result,
        output_ndim=true.ndim - axis - 1,
        reduction=output_reduction,
    )


def r2_score(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    /,
    *,
    sample_weight: ArrayLike | None = None,
    mask: ArrayLike | None = None,
    sample_axis: int = -1,
    output_reduction: OutputReduction = "uniform_average",
) -> MetricResult:
    """Weighted coefficient of determination without forced-finite coercion."""
    true, pred, weights, active, invalid, axis = _prepare_pair(
        y_true,
        y_pred,
        sample_weight=sample_weight,
        mask=mask,
        sample_axis=sample_axis,
        metric="r2_score",
        allow_complex=True,
    )
    dtype = jnp.result_type(true.dtype, pred.dtype, jnp.float32)
    true = true.astype(dtype)
    pred = pred.astype(dtype)
    mean, mass = _weighted_mean(true, weights, active, axis)
    centered = true - jnp.expand_dims(mean, axis=axis)
    residual = true - pred
    ss_total = _weighted_sum(_squared_magnitude(centered), weights, active, axis)
    ss_residual = _weighted_sum(_squared_magnitude(residual), weights, active, axis)
    undefined = ss_total <= 0.0
    value = 1.0 - ss_residual / jnp.where(undefined, 1.0, ss_total)
    result = _result(
        value,
        invalid=invalid,
        effective_weight=mass,
        undefined=undefined,
        undefined_status=METRIC_ZERO_DENOMINATOR,
    )
    variance = ss_total / jnp.where(mass > 0.0, mass, 1.0)
    return _reduce_outputs(
        result,
        output_ndim=true.ndim - axis - 1,
        reduction=output_reduction,
        variance=variance,
    )


def explained_variance_score(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    /,
    *,
    sample_weight: ArrayLike | None = None,
    mask: ArrayLike | None = None,
    sample_axis: int = -1,
    output_reduction: OutputReduction = "uniform_average",
) -> MetricResult:
    """One minus weighted residual variance divided by target variance."""
    true, pred, weights, active, invalid, axis = _prepare_pair(
        y_true,
        y_pred,
        sample_weight=sample_weight,
        mask=mask,
        sample_axis=sample_axis,
        metric="explained_variance_score",
        allow_complex=True,
    )
    dtype = jnp.result_type(true.dtype, pred.dtype, jnp.float32)
    true = true.astype(dtype)
    error = true - pred.astype(dtype)
    true_mean, mass = _weighted_mean(true, weights, active, axis)
    error_mean, _ = _weighted_mean(error, weights, active, axis)
    true_variance, _ = _weighted_mean(
        _squared_magnitude(true - jnp.expand_dims(true_mean, axis=axis)),
        weights,
        active,
        axis,
    )
    error_variance, _ = _weighted_mean(
        _squared_magnitude(error - jnp.expand_dims(error_mean, axis=axis)),
        weights,
        active,
        axis,
    )
    undefined = true_variance <= 0.0
    value = 1.0 - error_variance / jnp.where(undefined, 1.0, true_variance)
    result = _result(
        value,
        invalid=invalid,
        effective_weight=mass,
        undefined=undefined,
        undefined_status=METRIC_ZERO_DENOMINATOR,
    )
    return _reduce_outputs(
        result,
        output_ndim=true.ndim - axis - 1,
        reduction=output_reduction,
        variance=true_variance,
    )


def pinball_loss(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    /,
    *,
    quantile: ArrayLike = 0.5,
    sample_weight: ArrayLike | None = None,
    mask: ArrayLike | None = None,
    sample_axis: int = -1,
    output_reduction: OutputReduction = "uniform_average",
) -> MetricResult:
    """Weighted quantile (pinball) loss for quantiles strictly between zero and one."""
    true, pred, weights, active, invalid, axis = _prepare_pair(
        y_true,
        y_pred,
        sample_weight=sample_weight,
        mask=mask,
        sample_axis=sample_axis,
        metric="pinball_loss",
        allow_complex=False,
    )
    dtype = jnp.result_type(true.dtype, pred.dtype, jnp.float32)
    quantile_ = jnp.asarray(quantile, dtype=dtype)
    output_shape = true.shape[axis + 1 :]
    quantile_ = jnp.broadcast_to(quantile_, output_shape)
    quantile_full = quantile_.reshape((1,) * (axis + 1) + output_shape)
    error = true.astype(dtype) - pred.astype(dtype)
    loss = jnp.maximum(quantile_full * error, (quantile_full - 1.0) * error)
    value, mass = _weighted_mean(loss, weights, active, axis)
    invalid_quantile = (
        (~jnp.isfinite(quantile_)) | (quantile_ <= 0.0) | (quantile_ >= 1.0)
    )
    invalid = invalid | jnp.broadcast_to(invalid_quantile, invalid.shape)
    result = _result(value, invalid=invalid, effective_weight=mass)
    return _reduce_outputs(
        result,
        output_ndim=true.ndim - axis - 1,
        reduction=output_reduction,
    )


__all__ = [
    "explained_variance_score",
    "mean_absolute_error",
    "mean_squared_error",
    "pinball_loss",
    "r2_score",
    "root_mean_squared_error",
]
