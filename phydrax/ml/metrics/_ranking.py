#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import prod
from typing import Literal

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ._base import (
    _prepare_pair,
    _result,
    METRIC_ZERO_DENOMINATOR,
    MetricResult,
)


Gain = Literal["identity", "exponential"]


def _ranking_inputs(
    relevance: ArrayLike,
    scores: ArrayLike,
    /,
    *,
    sample_weight: ArrayLike | None,
    mask: ArrayLike | None,
    sample_axis: int,
    metric: str,
) -> tuple[Array, Array, Array, Array, Array]:
    relevance_, scores_, weights, active, invalid, axis = _prepare_pair(
        relevance,
        scores,
        sample_weight=sample_weight,
        mask=mask,
        sample_axis=sample_axis,
        metric=metric,
        allow_complex=False,
    )
    if axis != relevance_.ndim - 1:
        raise ValueError("Ranking inputs must have case_shape + (item,) axes.")
    valid_relevance = relevance_ >= 0.0
    invalid = invalid | jnp.any(active & ~valid_relevance, axis=-1)
    active = active & valid_relevance
    weights = jnp.where(active, weights, 0.0)
    scores_ = jnp.where(weights > 0.0, scores_, -jnp.inf)
    mass = jnp.sum(weights, axis=-1)
    return relevance_, scores_, weights, invalid, mass


def _gain(relevance: Array, gain: Gain) -> Array:
    if gain == "identity":
        return relevance
    if gain == "exponential":
        return jnp.exp2(relevance) - 1.0
    raise ValueError("gain must be 'identity' or 'exponential'.")


def _top_k(k: int | None, item_count: int) -> int:
    if k is None:
        return item_count
    count = int(k)
    if count <= 0:
        raise ValueError("k must be positive.")
    return min(count, item_count)


def _map_cases(relevance: Array, scores: Array, weights: Array, function) -> Array:
    case_shape = relevance.shape[:-1]
    item_count = relevance.shape[-1]
    case_count = prod(case_shape)
    result = jax.vmap(function)(
        relevance.reshape((case_count, item_count)),
        scores.reshape((case_count, item_count)),
        weights.reshape((case_count, item_count)),
    )
    return result.reshape(case_shape)


def discounted_cumulative_gain(
    relevance: ArrayLike,
    scores: ArrayLike,
    /,
    *,
    k: int | None = None,
    gain: Gain = "exponential",
    sample_weight: ArrayLike | None = None,
    mask: ArrayLike | None = None,
    sample_axis: int = -1,
) -> MetricResult:
    """Exact DCG under a hard stable score sort."""
    relevance_, scores_, weights, invalid, mass = _ranking_inputs(
        relevance,
        scores,
        sample_weight=sample_weight,
        mask=mask,
        sample_axis=sample_axis,
        metric="discounted_cumulative_gain",
    )
    limit = _top_k(k, relevance_.shape[-1])

    def one_case(rel, score, weight):
        order = jnp.argsort(-score, stable=True)
        contribution = weight[order] * _gain(rel[order], gain)
        discount = 1.0 / jnp.log2(
            jnp.arange(contribution.size, dtype=contribution.dtype) + 2.0
        )
        return jnp.sum(contribution[:limit] * discount[:limit])

    value = _map_cases(relevance_, scores_, weights, one_case)
    return _result(value, invalid=invalid, effective_weight=mass)


def ndcg_score(
    relevance: ArrayLike,
    scores: ArrayLike,
    /,
    *,
    k: int | None = None,
    gain: Gain = "exponential",
    sample_weight: ArrayLike | None = None,
    mask: ArrayLike | None = None,
    sample_axis: int = -1,
) -> MetricResult:
    """Exact normalized DCG; both predicted and ideal rankings use hard sorts."""
    relevance_, scores_, weights, invalid, mass = _ranking_inputs(
        relevance,
        scores,
        sample_weight=sample_weight,
        mask=mask,
        sample_axis=sample_axis,
        metric="ndcg_score",
    )
    limit = _top_k(k, relevance_.shape[-1])

    def one_case(rel, score, weight):
        contribution = weight * _gain(rel, gain)
        predicted_order = jnp.argsort(-score, stable=True)
        ideal_order = jnp.argsort(-contribution, stable=True)
        discount = 1.0 / jnp.log2(jnp.arange(rel.size, dtype=contribution.dtype) + 2.0)
        dcg = jnp.sum(contribution[predicted_order][:limit] * discount[:limit])
        ideal = jnp.sum(contribution[ideal_order][:limit] * discount[:limit])
        return dcg, ideal

    case_shape = relevance_.shape[:-1]
    item_count = relevance_.shape[-1]
    case_count = prod(case_shape)
    value, ideal = jax.vmap(one_case)(
        relevance_.reshape((case_count, item_count)),
        scores_.reshape((case_count, item_count)),
        weights.reshape((case_count, item_count)),
    )
    value = value.reshape(case_shape)
    ideal = ideal.reshape(case_shape)
    value = value / jnp.where(ideal > 0.0, ideal, 1.0)
    return _result(
        value,
        invalid=invalid,
        effective_weight=mass,
        undefined=ideal <= 0.0,
        undefined_status=METRIC_ZERO_DENOMINATOR,
    )


def precision_at_k(
    relevance: ArrayLike,
    scores: ArrayLike,
    /,
    *,
    k: int,
    relevance_threshold: float = 0.0,
    sample_weight: ArrayLike | None = None,
    mask: ArrayLike | None = None,
    sample_axis: int = -1,
) -> MetricResult:
    """Exact weighted precision among the hard top-k items."""
    relevance_, scores_, weights, invalid, mass = _ranking_inputs(
        relevance,
        scores,
        sample_weight=sample_weight,
        mask=mask,
        sample_axis=sample_axis,
        metric="precision_at_k",
    )
    limit = _top_k(k, relevance_.shape[-1])

    def one_case(rel, score, weight):
        order = jnp.argsort(-score, stable=True)[:limit]
        selected_weight = weight[order]
        denominator = jnp.sum(selected_weight)
        numerator = jnp.sum(selected_weight * (rel[order] > relevance_threshold))
        return numerator / jnp.where(denominator > 0.0, denominator, 1.0), denominator

    case_shape = relevance_.shape[:-1]
    item_count = relevance_.shape[-1]
    case_count = prod(case_shape)
    value, denominator = jax.vmap(one_case)(
        relevance_.reshape((case_count, item_count)),
        scores_.reshape((case_count, item_count)),
        weights.reshape((case_count, item_count)),
    )
    value = value.reshape(case_shape)
    denominator = denominator.reshape(case_shape)
    return _result(
        value,
        invalid=invalid,
        effective_weight=mass,
        undefined=denominator <= 0.0,
        undefined_status=METRIC_ZERO_DENOMINATOR,
    )


def recall_at_k(
    relevance: ArrayLike,
    scores: ArrayLike,
    /,
    *,
    k: int,
    relevance_threshold: float = 0.0,
    sample_weight: ArrayLike | None = None,
    mask: ArrayLike | None = None,
    sample_axis: int = -1,
) -> MetricResult:
    """Exact weighted recall of relevant items in the hard top-k."""
    relevance_, scores_, weights, invalid, mass = _ranking_inputs(
        relevance,
        scores,
        sample_weight=sample_weight,
        mask=mask,
        sample_axis=sample_axis,
        metric="recall_at_k",
    )
    limit = _top_k(k, relevance_.shape[-1])

    def one_case(rel, score, weight):
        relevant_weight = weight * (rel > relevance_threshold)
        denominator = jnp.sum(relevant_weight)
        order = jnp.argsort(-score, stable=True)[:limit]
        numerator = jnp.sum(relevant_weight[order])
        return numerator / jnp.where(denominator > 0.0, denominator, 1.0), denominator

    case_shape = relevance_.shape[:-1]
    item_count = relevance_.shape[-1]
    case_count = prod(case_shape)
    value, denominator = jax.vmap(one_case)(
        relevance_.reshape((case_count, item_count)),
        scores_.reshape((case_count, item_count)),
        weights.reshape((case_count, item_count)),
    )
    value = value.reshape(case_shape)
    denominator = denominator.reshape(case_shape)
    return _result(
        value,
        invalid=invalid,
        effective_weight=mass,
        undefined=denominator <= 0.0,
        undefined_status=METRIC_ZERO_DENOMINATOR,
    )


def reciprocal_rank(
    relevance: ArrayLike,
    scores: ArrayLike,
    /,
    *,
    relevance_threshold: float = 0.0,
    sample_weight: ArrayLike | None = None,
    mask: ArrayLike | None = None,
    sample_axis: int = -1,
) -> MetricResult:
    """Exact reciprocal rank of the first relevant positive-weight item."""
    relevance_, scores_, weights, invalid, mass = _ranking_inputs(
        relevance,
        scores,
        sample_weight=sample_weight,
        mask=mask,
        sample_axis=sample_axis,
        metric="reciprocal_rank",
    )

    def one_case(rel, score, weight):
        active = weight > 0.0
        order = jnp.argsort(-score, stable=True)
        relevant = active[order] & (rel[order] > relevance_threshold)
        active_rank = jnp.cumsum(active[order])
        reciprocal = jnp.where(relevant, 1.0 / jnp.maximum(active_rank, 1), 0.0)
        return jnp.max(reciprocal), jnp.any(relevant)

    case_shape = relevance_.shape[:-1]
    item_count = relevance_.shape[-1]
    case_count = prod(case_shape)
    value, has_relevant = jax.vmap(one_case)(
        relevance_.reshape((case_count, item_count)),
        scores_.reshape((case_count, item_count)),
        weights.reshape((case_count, item_count)),
    )
    value = value.reshape(case_shape)
    has_relevant = has_relevant.reshape(case_shape)
    return _result(
        value,
        invalid=invalid,
        effective_weight=mass,
        undefined=~has_relevant,
        undefined_status=METRIC_ZERO_DENOMINATOR,
    )


def average_precision_score(
    relevance: ArrayLike,
    scores: ArrayLike,
    /,
    *,
    relevance_threshold: float = 0.0,
    sample_weight: ArrayLike | None = None,
    mask: ArrayLike | None = None,
    sample_axis: int = -1,
) -> MetricResult:
    """Exact weighted average precision under a hard stable ranking."""
    relevance_, scores_, weights, invalid, mass = _ranking_inputs(
        relevance,
        scores,
        sample_weight=sample_weight,
        mask=mask,
        sample_axis=sample_axis,
        metric="average_precision_score",
    )

    def one_case(rel, score, weight):
        order = jnp.argsort(-score, stable=True)
        weight = weight[order]
        relevant_weight = weight * (rel[order] > relevance_threshold)
        cumulative_relevant = jnp.cumsum(relevant_weight)
        cumulative_weight = jnp.cumsum(weight)
        precision = cumulative_relevant / jnp.where(
            cumulative_weight > 0.0, cumulative_weight, 1.0
        )
        denominator = jnp.sum(relevant_weight)
        value = jnp.sum(relevant_weight * precision) / jnp.where(
            denominator > 0.0, denominator, 1.0
        )
        return value, denominator

    case_shape = relevance_.shape[:-1]
    item_count = relevance_.shape[-1]
    case_count = prod(case_shape)
    value, denominator = jax.vmap(one_case)(
        relevance_.reshape((case_count, item_count)),
        scores_.reshape((case_count, item_count)),
        weights.reshape((case_count, item_count)),
    )
    value = value.reshape(case_shape)
    denominator = denominator.reshape(case_shape)
    return _result(
        value,
        invalid=invalid,
        effective_weight=mass,
        undefined=denominator <= 0.0,
        undefined_status=METRIC_ZERO_DENOMINATOR,
    )


def _soft_ranks(scores: Array, weights: Array, temperature: float) -> Array:
    active = weights > 0.0
    comparison = jax.nn.sigmoid(
        (scores[..., None, :] - scores[..., :, None]) / float(temperature)
    )
    identity = jnp.eye(scores.shape[-1], dtype=bool)
    comparison = jnp.where(identity | ~active[..., None, :], 0.0, comparison)
    return 1.0 + jnp.sum(comparison, axis=-1)


def smooth_ndcg_score(
    relevance: ArrayLike,
    scores: ArrayLike,
    /,
    *,
    temperature: float = 1.0,
    gain: Gain = "exponential",
    sample_weight: ArrayLike | None = None,
    mask: ArrayLike | None = None,
    sample_axis: int = -1,
) -> MetricResult:
    """Soft-rank NDCG surrogate, smooth in scores; the ideal relevance order is exact."""
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")
    relevance_, scores_, weights, invalid, mass = _ranking_inputs(
        relevance,
        scores,
        sample_weight=sample_weight,
        mask=mask,
        sample_axis=sample_axis,
        metric="smooth_ndcg_score",
    )
    contribution = weights * _gain(relevance_, gain)
    rank = _soft_ranks(scores_, weights, temperature)
    value = jnp.sum(contribution / jnp.log2(rank + 1.0), axis=-1)
    ideal_order = jnp.argsort(-contribution, axis=-1, stable=True)
    ideal_contribution = jnp.take_along_axis(contribution, ideal_order, axis=-1)
    discount = 1.0 / jnp.log2(
        jnp.arange(relevance_.shape[-1], dtype=contribution.dtype) + 2.0
    )
    ideal = jnp.sum(ideal_contribution * discount, axis=-1)
    value = value / jnp.where(ideal > 0.0, ideal, 1.0)
    return _result(
        value,
        invalid=invalid,
        effective_weight=mass,
        undefined=ideal <= 0.0,
        undefined_status=METRIC_ZERO_DENOMINATOR,
    )


def smooth_precision_at_k(
    relevance: ArrayLike,
    scores: ArrayLike,
    /,
    *,
    k: int,
    temperature: float = 1.0,
    relevance_temperature: float = 1.0,
    relevance_threshold: float = 0.0,
    sample_weight: ArrayLike | None = None,
    mask: ArrayLike | None = None,
    sample_axis: int = -1,
) -> MetricResult:
    """Soft-rank, soft-relevance precision-at-k surrogate."""
    if temperature <= 0.0 or relevance_temperature <= 0.0:
        raise ValueError("temperatures must be positive.")
    limit = int(k)
    if limit <= 0:
        raise ValueError("k must be positive.")
    relevance_, scores_, weights, invalid, mass = _ranking_inputs(
        relevance,
        scores,
        sample_weight=sample_weight,
        mask=mask,
        sample_axis=sample_axis,
        metric="smooth_precision_at_k",
    )
    rank = _soft_ranks(scores_, weights, temperature)
    selection = jax.nn.sigmoid((float(limit) + 0.5 - rank) / float(temperature))
    soft_relevant = jax.nn.sigmoid(
        (relevance_ - float(relevance_threshold)) / float(relevance_temperature)
    )
    selected_weight = weights * selection
    denominator = jnp.sum(selected_weight, axis=-1)
    value = jnp.sum(selected_weight * soft_relevant, axis=-1) / jnp.where(
        denominator > 0.0, denominator, 1.0
    )
    return _result(
        value,
        invalid=invalid,
        effective_weight=mass,
        undefined=denominator <= 0.0,
        undefined_status=METRIC_ZERO_DENOMINATOR,
    )


def smooth_recall_at_k(
    relevance: ArrayLike,
    scores: ArrayLike,
    /,
    *,
    k: int,
    temperature: float = 1.0,
    relevance_temperature: float = 1.0,
    relevance_threshold: float = 0.0,
    sample_weight: ArrayLike | None = None,
    mask: ArrayLike | None = None,
    sample_axis: int = -1,
) -> MetricResult:
    """Soft-rank, soft-relevance recall-at-k surrogate."""
    if temperature <= 0.0 or relevance_temperature <= 0.0:
        raise ValueError("temperatures must be positive.")
    limit = int(k)
    if limit <= 0:
        raise ValueError("k must be positive.")
    relevance_, scores_, weights, invalid, mass = _ranking_inputs(
        relevance,
        scores,
        sample_weight=sample_weight,
        mask=mask,
        sample_axis=sample_axis,
        metric="smooth_recall_at_k",
    )
    rank = _soft_ranks(scores_, weights, temperature)
    selection = jax.nn.sigmoid((float(limit) + 0.5 - rank) / float(temperature))
    soft_relevant = jax.nn.sigmoid(
        (relevance_ - float(relevance_threshold)) / float(relevance_temperature)
    )
    relevant_weight = weights * soft_relevant
    denominator = jnp.sum(relevant_weight, axis=-1)
    value = jnp.sum(relevant_weight * selection, axis=-1) / jnp.where(
        denominator > 0.0, denominator, 1.0
    )
    return _result(
        value,
        invalid=invalid,
        effective_weight=mass,
        undefined=denominator <= 0.0,
        undefined_status=METRIC_ZERO_DENOMINATOR,
    )


def smooth_reciprocal_rank(
    relevance: ArrayLike,
    scores: ArrayLike,
    /,
    *,
    temperature: float = 1.0,
    relevance_temperature: float = 1.0,
    relevance_threshold: float = 0.0,
    sample_weight: ArrayLike | None = None,
    mask: ArrayLike | None = None,
    sample_axis: int = -1,
) -> MetricResult:
    """Soft relevance-weighted reciprocal-rank surrogate."""
    if temperature <= 0.0 or relevance_temperature <= 0.0:
        raise ValueError("temperatures must be positive.")
    relevance_, scores_, weights, invalid, mass = _ranking_inputs(
        relevance,
        scores,
        sample_weight=sample_weight,
        mask=mask,
        sample_axis=sample_axis,
        metric="smooth_reciprocal_rank",
    )
    rank = _soft_ranks(scores_, weights, temperature)
    soft_relevant = jax.nn.sigmoid(
        (relevance_ - float(relevance_threshold)) / float(relevance_temperature)
    )
    relevance_weight = weights * soft_relevant
    denominator = jnp.sum(relevance_weight, axis=-1)
    value = jnp.sum(relevance_weight / rank, axis=-1) / jnp.where(
        denominator > 0.0, denominator, 1.0
    )
    return _result(
        value,
        invalid=invalid,
        effective_weight=mass,
        undefined=denominator <= 0.0,
        undefined_status=METRIC_ZERO_DENOMINATOR,
    )


def smooth_average_precision_score(
    relevance: ArrayLike,
    scores: ArrayLike,
    /,
    *,
    temperature: float = 1.0,
    relevance_temperature: float = 1.0,
    relevance_threshold: float = 0.0,
    sample_weight: ArrayLike | None = None,
    mask: ArrayLike | None = None,
    sample_axis: int = -1,
) -> MetricResult:
    """Pairwise soft-threshold average-precision surrogate."""
    if temperature <= 0.0 or relevance_temperature <= 0.0:
        raise ValueError("temperatures must be positive.")
    relevance_, scores_, weights, invalid, mass = _ranking_inputs(
        relevance,
        scores,
        sample_weight=sample_weight,
        mask=mask,
        sample_axis=sample_axis,
        metric="smooth_average_precision_score",
    )
    soft_relevant = jax.nn.sigmoid(
        (relevance_ - float(relevance_threshold)) / float(relevance_temperature)
    )
    soft_above = jax.nn.sigmoid(
        (scores_[..., None, :] - scores_[..., :, None]) / float(temperature)
    )
    soft_above = jnp.where(weights[..., None, :] > 0.0, soft_above, 0.0)
    total_above = jnp.sum(soft_above * weights[..., None, :], axis=-1)
    relevant_above = jnp.sum(
        soft_above * (weights * soft_relevant)[..., None, :], axis=-1
    )
    precision = relevant_above / jnp.where(total_above > 0.0, total_above, 1.0)
    relevance_weight = weights * soft_relevant
    denominator = jnp.sum(relevance_weight, axis=-1)
    value = jnp.sum(relevance_weight * precision, axis=-1) / jnp.where(
        denominator > 0.0, denominator, 1.0
    )
    return _result(
        value,
        invalid=invalid,
        effective_weight=mass,
        undefined=denominator <= 0.0,
        undefined_status=METRIC_ZERO_DENOMINATOR,
    )


__all__ = [
    "Gain",
    "average_precision_score",
    "discounted_cumulative_gain",
    "ndcg_score",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "smooth_average_precision_score",
    "smooth_ndcg_score",
    "smooth_precision_at_k",
    "smooth_recall_at_k",
    "smooth_reciprocal_rank",
]
