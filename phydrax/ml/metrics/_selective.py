#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import prod

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ._base import (
    _nan_where_invalid,
    _prepare_pair,
    _result,
    _status,
    METRIC_ZERO_DENOMINATOR,
    MetricResult,
)


class SelectiveRiskCurveResult(StrictModule):
    """A weighted risk--coverage curve stored in fixed tie-block capacity."""

    aurc: Array
    score_threshold: Array
    coverage: Array
    retained_risk: Array
    retained_weight: Array
    point_mask: Array
    valid: Array
    status: Array
    effective_weight: Array

    def __init__(
        self,
        aurc: ArrayLike,
        /,
        *,
        score_threshold: ArrayLike,
        coverage: ArrayLike,
        retained_risk: ArrayLike,
        retained_weight: ArrayLike,
        point_mask: ArrayLike,
        valid: ArrayLike,
        status: ArrayLike,
        effective_weight: ArrayLike,
    ):
        self.aurc = jnp.asarray(aurc)
        self.score_threshold = jnp.asarray(score_threshold)
        self.coverage = jnp.asarray(coverage)
        self.retained_risk = jnp.asarray(retained_risk)
        self.retained_weight = jnp.asarray(retained_weight)
        self.point_mask = jnp.asarray(point_mask, dtype=bool)
        self.valid = jnp.asarray(valid, dtype=bool)
        self.status = jnp.asarray(status, dtype=jnp.int32)
        self.effective_weight = jnp.asarray(effective_weight)


def _tie_block_curve(
    loss: Array, rejection_score: Array, weight: Array, /
) -> tuple[Array, Array, Array, Array, Array, Array]:
    capacity = loss.shape[0]
    positive = weight > 0.0
    ordered_score = jnp.where(positive, rejection_score, jnp.inf)
    order = jnp.argsort(ordered_score, stable=True)
    sorted_loss = loss[order]
    sorted_score = ordered_score[order]
    sorted_weight = weight[order]
    sorted_positive = positive[order]

    block_start = jnp.concatenate(
        (
            jnp.ones((1,), dtype=bool),
            sorted_score[1:] != sorted_score[:-1],
        )
    )
    block_index = jnp.cumsum(block_start.astype(jnp.int32)) - 1
    block_weight = jax.ops.segment_sum(
        sorted_weight,
        block_index,
        num_segments=capacity,
        indices_are_sorted=True,
    )
    block_loss = jax.ops.segment_sum(
        sorted_weight * sorted_loss,
        block_index,
        num_segments=capacity,
        indices_are_sorted=True,
    )
    block_count = jax.ops.segment_sum(
        sorted_positive.astype(sorted_weight.dtype),
        block_index,
        num_segments=capacity,
        indices_are_sorted=True,
    )
    block_score = jax.ops.segment_sum(
        jnp.where(sorted_positive, sorted_score, 0.0),
        block_index,
        num_segments=capacity,
        indices_are_sorted=True,
    ) / jnp.where(block_count > 0.0, block_count, 1.0)

    retained_weight = jnp.cumsum(block_weight)
    retained_loss = jnp.cumsum(block_loss)
    total_weight = retained_weight[-1]
    point_mask = block_weight > 0.0
    retained_risk = retained_loss / jnp.where(retained_weight > 0.0, retained_weight, 1.0)
    coverage = retained_weight / jnp.where(total_weight > 0.0, total_weight, 1.0)
    aurc = jnp.sum(
        jnp.where(
            point_mask,
            (block_weight / jnp.where(total_weight > 0.0, total_weight, 1.0))
            * retained_risk,
            0.0,
        )
    )
    return (
        aurc,
        block_score,
        coverage,
        retained_risk,
        retained_weight,
        point_mask,
    )


def selective_risk_curve(
    loss: ArrayLike,
    rejection_score: ArrayLike,
    /,
    *,
    sample_weight: ArrayLike | None = None,
    mask: ArrayLike | None = None,
    sample_axis: int = -1,
) -> SelectiveRiskCurveResult:
    """Compute an exact weighted selective-risk curve.

    Larger rejection scores are rejected earlier. Each reported point retains all
    samples whose score is at most ``score_threshold``; equal-score samples are an
    indivisible block. Coverage is retained empirical weight mass, and ``aurc`` is
    the right-endpoint sum over the attainable coverage increments. An oracle is
    obtained explicitly with ``selective_risk_curve(loss, loss, ...)``.
    """
    loss_, score_, weights, active, invalid, axis = _prepare_pair(
        loss,
        rejection_score,
        sample_weight=sample_weight,
        mask=mask,
        sample_axis=sample_axis,
        metric="selective_risk_curve",
        allow_complex=False,
    )
    if axis != loss_.ndim - 1:
        raise ValueError("Selective-risk inputs must have case_shape + (sample,) axes.")
    invalid = invalid | jnp.any(active & (loss_ < 0.0), axis=-1)
    weights = jnp.where(active, weights, 0.0)
    mass = jnp.sum(weights, axis=-1)
    case_shape = loss_.shape[:-1]
    sample_count = loss_.shape[-1]
    if sample_count == 0:
        valid, status = _status(invalid=invalid, empty=mass <= 0.0)
        curve_shape = case_shape + (0,)
        empty = jnp.empty(curve_shape, dtype=weights.dtype)
        return SelectiveRiskCurveResult(
            _nan_where_invalid(jnp.zeros(case_shape, dtype=weights.dtype), valid),
            score_threshold=empty,
            coverage=empty,
            retained_risk=empty,
            retained_weight=empty,
            point_mask=jnp.empty(curve_shape, dtype=bool),
            valid=valid,
            status=status,
            effective_weight=mass,
        )
    case_count = prod(case_shape)
    curve = jax.vmap(_tie_block_curve)(
        loss_.reshape((case_count, sample_count)),
        score_.reshape((case_count, sample_count)),
        weights.reshape((case_count, sample_count)),
    )
    aurc = curve[0].reshape(case_shape)
    curve_shape = case_shape + (sample_count,)
    score_threshold = curve[1].reshape(curve_shape)
    coverage = curve[2].reshape(curve_shape)
    retained_risk = curve[3].reshape(curve_shape)
    retained_weight = curve[4].reshape(curve_shape)
    point_mask = curve[5].reshape(curve_shape)

    valid, status = _status(invalid=invalid, empty=mass <= 0.0)
    point_mask = point_mask & valid[..., None]
    score_threshold = jnp.where(point_mask, score_threshold, jnp.nan)
    coverage = jnp.where(point_mask, coverage, jnp.nan)
    retained_risk = jnp.where(point_mask, retained_risk, jnp.nan)
    retained_weight = jnp.where(point_mask, retained_weight, jnp.nan)
    return SelectiveRiskCurveResult(
        _nan_where_invalid(aurc, valid),
        score_threshold=score_threshold,
        coverage=coverage,
        retained_risk=retained_risk,
        retained_weight=retained_weight,
        point_mask=point_mask,
        valid=valid,
        status=status,
        effective_weight=mass,
    )


def _weighted_midranks(values: Array, weights: Array, /) -> Array:
    capacity = values.shape[0]
    positive = weights > 0.0
    ordered_value = jnp.where(positive, values, jnp.inf)
    order = jnp.argsort(ordered_value, stable=True)
    sorted_value = ordered_value[order]
    sorted_weight = weights[order]

    block_start = jnp.concatenate(
        (
            jnp.ones((1,), dtype=bool),
            sorted_value[1:] != sorted_value[:-1],
        )
    )
    block_index = jnp.cumsum(block_start.astype(jnp.int32)) - 1
    block_weight = jax.ops.segment_sum(
        sorted_weight,
        block_index,
        num_segments=capacity,
        indices_are_sorted=True,
    )
    preceding_weight = jnp.cumsum(block_weight) - block_weight
    total_weight = jnp.sum(block_weight)
    block_rank = (preceding_weight + 0.5 * block_weight) / jnp.where(
        total_weight > 0.0, total_weight, 1.0
    )
    sorted_rank = block_rank[block_index]
    return jnp.zeros_like(sorted_rank).at[order].set(sorted_rank)


def _spearman_case(
    first: Array, second: Array, weights: Array, /
) -> tuple[Array, Array, Array]:
    first_rank = _weighted_midranks(first, weights)
    second_rank = _weighted_midranks(second, weights)
    mass = jnp.sum(weights)
    denominator = jnp.where(mass > 0.0, mass, 1.0)
    first_mean = jnp.sum(weights * first_rank) / denominator
    second_mean = jnp.sum(weights * second_rank) / denominator
    first_centered = first_rank - first_mean
    second_centered = second_rank - second_mean
    first_variance = jnp.sum(weights * jnp.square(first_centered))
    second_variance = jnp.sum(weights * jnp.square(second_centered))
    covariance = jnp.sum(weights * first_centered * second_centered)
    correlation = covariance / jnp.sqrt(
        jnp.where(
            (first_variance > 0.0) & (second_variance > 0.0),
            first_variance * second_variance,
            1.0,
        )
    )
    return correlation, first_variance, second_variance


def spearman_rank_correlation(
    first: ArrayLike,
    second: ArrayLike,
    /,
    *,
    sample_weight: ArrayLike | None = None,
    mask: ArrayLike | None = None,
    sample_axis: int = -1,
) -> MetricResult:
    """Weighted Spearman correlation using empirical-CDF tie midranks.

    Integer weights agree with literal frequency replication. Sorting and tie
    membership are exact hard operations, so rank gradients are zero almost
    everywhere.
    """
    first_, second_, weights, active, invalid, axis = _prepare_pair(
        first,
        second,
        sample_weight=sample_weight,
        mask=mask,
        sample_axis=sample_axis,
        metric="spearman_rank_correlation",
        allow_complex=False,
    )
    if axis != first_.ndim - 1:
        raise ValueError("Spearman inputs must have case_shape + (sample,) axes.")
    weights = jnp.where(active, weights, 0.0)
    mass = jnp.sum(weights, axis=-1)
    case_shape = first_.shape[:-1]
    sample_count = first_.shape[-1]
    if sample_count == 0:
        return _result(
            jnp.zeros(case_shape, dtype=weights.dtype),
            invalid=invalid,
            effective_weight=mass,
            undefined=jnp.ones(case_shape, dtype=bool),
            undefined_status=METRIC_ZERO_DENOMINATOR,
        )
    case_count = prod(case_shape)
    correlation, first_variance, second_variance = jax.vmap(_spearman_case)(
        first_.reshape((case_count, sample_count)),
        second_.reshape((case_count, sample_count)),
        weights.reshape((case_count, sample_count)),
    )
    correlation = correlation.reshape(case_shape)
    undefined = ((first_variance <= 0.0) | (second_variance <= 0.0)).reshape(case_shape)
    return _result(
        correlation,
        invalid=invalid,
        effective_weight=mass,
        undefined=undefined,
        undefined_status=METRIC_ZERO_DENOMINATOR,
    )


__all__ = [
    "SelectiveRiskCurveResult",
    "selective_risk_curve",
    "spearman_rank_correlation",
]
