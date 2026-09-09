#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
from jax.scipy.special import ndtr
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule


ForecastAlternative: TypeAlias = Literal["two-sided", "less", "greater"]
ForecastComparisonStatus: TypeAlias = Literal[0, 1, 2]

FORECAST_COMPARISON_SUCCESS = 0
FORECAST_COMPARISON_INSUFFICIENT = 1
FORECAST_COMPARISON_NONFINITE = 2


class ForecastComparisonPlan(StrictModule):
    """Declared Diebold--Mariano loss-differential comparison."""

    hac_lags: int = eqx.field(static=True)
    horizon: int = eqx.field(static=True)
    alternative: ForecastAlternative = eqx.field(static=True)

    def __init__(
        self,
        *,
        hac_lags: int = 0,
        horizon: int = 1,
        alternative: ForecastAlternative = "two-sided",
    ):
        lags = int(hac_lags)
        forecast_horizon = int(horizon)
        if lags < 0:
            raise ValueError("hac_lags must be nonnegative.")
        if forecast_horizon < 1:
            raise ValueError("horizon must be positive.")
        if lags < forecast_horizon - 1:
            raise ValueError(
                "hac_lags must be at least horizon - 1 for overlapping forecasts."
            )
        if alternative not in ("two-sided", "less", "greater"):
            raise ValueError("alternative must be 'two-sided', 'less', or 'greater'.")
        self.hac_lags = lags
        self.horizon = forecast_horizon
        self.alternative = alternative


class ForecastComparisonResult(StrictModule):
    """Loss-differential test with dependence and degeneracy evidence."""

    loss_differential: Array
    valid_mask: Array
    mean_differential: Array
    long_run_variance: Array
    mean_variance: Array
    statistic: Array
    p_value: Array
    effective_sample_count: Array
    autocovariances: Array
    zero_variance: Array
    status: Array
    hac_lags: int = eqx.field(static=True)
    horizon: int = eqx.field(static=True)
    alternative: ForecastAlternative = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.status == FORECAST_COMPARISON_SUCCESS

    @property
    def first_has_lower_loss(self) -> Array:
        return self.mean_differential < 0.0


def _validated_losses(
    first: ArrayLike,
    second: ArrayLike,
    mask: ArrayLike | None,
) -> tuple[Array, Array]:
    first_ = jnp.asarray(first)
    second_ = jnp.asarray(second)
    if first_.shape != second_.shape or first_.ndim < 1:
        raise ValueError("loss arrays must have one identical, non-scalar shape.")
    if not jnp.issubdtype(first_.dtype, jnp.inexact):
        first_ = first_.astype(float)
    if not jnp.issubdtype(second_.dtype, jnp.inexact):
        second_ = second_.astype(float)
    dtype = jnp.result_type(first_, second_)
    first_ = first_.astype(dtype)
    second_ = second_.astype(dtype)
    valid = (
        jnp.ones(first_.shape, dtype=bool)
        if mask is None
        else jnp.asarray(mask, dtype=bool)
    )
    if valid.shape != first_.shape:
        raise ValueError("mask must have the same shape as the loss arrays.")
    invalid = valid & (~jnp.isfinite(first_) | ~jnp.isfinite(second_))
    differential = first_ - second_
    differential = eqx.error_if(
        differential,
        jnp.any(invalid),
        "active losses must be finite.",
    )
    return differential, valid


def compare_loss_differentials(
    loss_differential: ArrayLike,
    /,
    *,
    mask: ArrayLike | None = None,
    hac_lags: int = 0,
    horizon: int = 1,
    alternative: ForecastAlternative = "two-sided",
) -> ForecastComparisonResult:
    """Compare an ordered loss differential with a Bartlett HAC estimate.

    The final axis is time. Other axes are independent comparison cases. A
    zero-variance, zero-mean differential is an exact tie with p-value one;
    a nonzero constant differential has an infinite signed statistic and
    p-value zero in the compatible tail.
    """

    plan = ForecastComparisonPlan(
        hac_lags=hac_lags,
        horizon=horizon,
        alternative=alternative,
    )
    differential = jnp.asarray(loss_differential)
    if differential.ndim < 1:
        raise ValueError("loss_differential must have a final time axis.")
    zeros = jnp.zeros_like(differential)
    values, valid = _validated_losses(differential, zeros, mask)
    time_count = values.shape[-1]
    if plan.hac_lags >= time_count:
        raise ValueError("hac_lags must be smaller than the time-axis capacity.")

    count = jnp.sum(valid, axis=-1).astype(jnp.int32)
    denominator = jnp.maximum(count, 1).astype(values.dtype)
    safe_values = jnp.where(valid, values, 0.0)
    mean = jnp.sum(safe_values, axis=-1) / denominator
    centered = jnp.where(valid, values - mean[..., None], 0.0)

    autocovariances = []
    pair_counts = []
    for lag in range(plan.hac_lags + 1):
        left = centered[..., lag:]
        right = centered[..., : time_count - lag]
        pairs = valid[..., lag:] & valid[..., : time_count - lag]
        pair_count = jnp.sum(pairs, axis=-1).astype(jnp.int32)
        pair_denominator = jnp.maximum(pair_count, 1).astype(values.dtype)
        covariance = jnp.sum(jnp.where(pairs, left * right, 0.0), axis=-1)
        covariance = covariance / pair_denominator
        autocovariances.append(covariance)
        pair_counts.append(pair_count)
    autocovariance = jnp.stack(autocovariances, axis=-1)
    pair_count = jnp.stack(pair_counts, axis=-1)
    weights = 1.0 - jnp.arange(plan.hac_lags + 1, dtype=values.dtype) / (
        plan.hac_lags + 1.0
    )
    positive_lags = jnp.sum(weights[1:] * autocovariance[..., 1:], axis=-1, initial=0.0)
    long_run_variance = jnp.maximum(autocovariance[..., 0] + 2.0 * positive_lags, 0.0)
    mean_variance = long_run_variance / denominator
    scale = jnp.sqrt(mean_variance)
    zero_variance = mean_variance <= jnp.finfo(values.dtype).eps
    statistic = jnp.where(
        zero_variance,
        jnp.where(mean == 0.0, 0.0, jnp.copysign(jnp.inf, mean)),
        mean / scale,
    )
    if plan.alternative == "two-sided":
        p_value = 2.0 * ndtr(-jnp.abs(statistic))
    elif plan.alternative == "greater":
        p_value = ndtr(-statistic)
    else:
        p_value = ndtr(statistic)
    p_value = jnp.where(zero_variance & (mean == 0.0), 1.0, p_value)

    enough_pairs = jnp.all(pair_count > 0, axis=-1)
    finite = jnp.isfinite(mean) & jnp.isfinite(long_run_variance) & jnp.isfinite(p_value)
    required = max(2, plan.hac_lags + 2)
    enough = (count >= required) & enough_pairs
    status = jnp.where(
        ~finite,
        FORECAST_COMPARISON_NONFINITE,
        jnp.where(
            enough,
            FORECAST_COMPARISON_SUCCESS,
            FORECAST_COMPARISON_INSUFFICIENT,
        ),
    ).astype(jnp.int32)
    return ForecastComparisonResult(
        loss_differential=values,
        valid_mask=valid,
        mean_differential=mean,
        long_run_variance=long_run_variance,
        mean_variance=mean_variance,
        statistic=statistic,
        p_value=p_value,
        effective_sample_count=count,
        autocovariances=autocovariance,
        zero_variance=zero_variance,
        status=status,
        hac_lags=plan.hac_lags,
        horizon=plan.horizon,
        alternative=plan.alternative,
    )


def compare_forecasts(
    first_loss: ArrayLike,
    second_loss: ArrayLike,
    /,
    *,
    mask: ArrayLike | None = None,
    hac_lags: int = 0,
    horizon: int = 1,
    alternative: ForecastAlternative = "two-sided",
) -> ForecastComparisonResult:
    """Compare two chronological loss sequences as ``first - second``."""

    differential, valid = _validated_losses(first_loss, second_loss, mask)
    return compare_loss_differentials(
        differential,
        mask=valid,
        hac_lags=hac_lags,
        horizon=horizon,
        alternative=alternative,
    )


__all__ = [
    "FORECAST_COMPARISON_INSUFFICIENT",
    "FORECAST_COMPARISON_NONFINITE",
    "FORECAST_COMPARISON_SUCCESS",
    "ForecastAlternative",
    "ForecastComparisonPlan",
    "ForecastComparisonResult",
    "compare_forecasts",
    "compare_loss_differentials",
]
