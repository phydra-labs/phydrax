#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from ..._differentiation import DerivativeRegularity
from .._numerics import pairwise_distances
from .._numerics._pairwise import blocked_logsumexp, blocked_top_k, pair_block_size


def size(shape: tuple[int, ...]) -> int:
    result = 1
    for value in shape:
        result *= int(value)
    return result


def validate_metric(metric: Any) -> Any:
    if not callable(metric) and metric not in (
        "euclidean",
        "squared-euclidean",
        "manhattan",
        "cosine",
    ):
        raise ValueError("metric must be callable or a supported native metric name.")
    return metric


def metric_regularity(metric: Any, /) -> DerivativeRegularity | None:
    """Value regularity of `x -> distance(x, y)` for a fixed point `y`.

    Euclidean distance has a cone at `y` and Manhattan distance kinks on the
    coordinate hyperplanes through `y`; cosine distance has no limit at the
    origin. Callable metrics carry no certificate and stay undeclared.
    """
    if metric == "squared-euclidean":
        return DerivativeRegularity.smooth(degree_bound=2)
    if metric == "euclidean":
        return DerivativeRegularity.piecewise_smooth(continuity=0)
    if metric == "manhattan":
        return DerivativeRegularity.piecewise_polynomial(continuity=0, degree_bound=1)
    if metric == "cosine":
        return DerivativeRegularity.piecewise_smooth(continuity=-1)
    return None


def distance_softmax_regularity(metric: Any, /) -> DerivativeRegularity | None:
    """Regularity of a map smooth in the distances to fixed points, e.g. a softmax."""
    regularity = metric_regularity(metric)
    return (
        None if regularity is None else regularity.compose(DerivativeRegularity.smooth())
    )


def masked_softmax(logits: Array, mask: Array) -> Array:
    """Normalize masked logits without NaNs for completely inactive rows."""
    active = jnp.asarray(mask, dtype=jnp.bool_)
    any_active = jnp.any(active, axis=-1, keepdims=True)
    safe_logits = jnp.where(active, jnp.asarray(logits), -jnp.inf)
    safe_logits = jnp.where(any_active, safe_logits, 0.0)
    probabilities = jax.nn.softmax(safe_logits, axis=-1)
    return jnp.where(active & any_active, probabilities, 0.0)


def case_distances(
    query: Array, support: Array, case_shape: tuple[int, ...], metric: Any
) -> tuple[Array, tuple[int, ...]]:
    x = jnp.asarray(query)
    minimum_rank = len(case_shape) + 1
    if x.ndim < minimum_rank or x.shape[-1] != support.shape[-1]:
        raise ValueError("Query must end with the fitted feature axis.")
    if case_shape:
        if x.shape[: len(case_shape)] != case_shape:
            raise ValueError(f"Query must begin with fitted case shape {case_shape}.")
        query_shape = tuple(x.shape[len(case_shape) : -1])
        q = size(query_shape) if query_shape else 1
        cases = size(case_shape)
        query_cases = x.reshape((cases, q, x.shape[-1]))
        support_cases = support.reshape((cases, support.shape[-2], support.shape[-1]))
        distance = jax.vmap(lambda a, b: pairwise_distances(a, b, metric=metric))(
            query_cases, support_cases
        )
        return distance.reshape(
            case_shape + query_shape + (support.shape[-2],)
        ), query_shape
    query_shape = tuple(x.shape[:-1])
    distance = pairwise_distances(x.reshape((-1, x.shape[-1])), support, metric=metric)
    return distance.reshape(query_shape + (support.shape[-2],)), query_shape


def blocked_case_top_k(
    query: Array,
    support: Array,
    mask: Array,
    case_shape: tuple[int, ...],
    metric: Any,
    count: int,
    *,
    workspace_bytes: int = 4 * 1024 * 1024,
) -> tuple[Array, Array]:
    x, query_shape = _case_queries(query, support, case_shape)
    indices, distances = blocked_top_k(
        x,
        support,
        mask,
        count,
        metric=metric,
        workspace_bytes=workspace_bytes // size(case_shape),
    )
    shape = case_shape + query_shape + (count,)
    return indices.reshape(shape), distances.reshape(shape)


def _case_queries(
    query: Array, support: Array, case_shape: tuple[int, ...]
) -> tuple[Array, tuple[int, ...]]:
    if query.ndim < len(case_shape) + 1 or query.shape[-1] != support.shape[-1]:
        raise ValueError("Query must end with the fitted feature axis.")
    if query.shape[: len(case_shape)] != case_shape:
        raise ValueError(f"Query must begin with fitted case shape {case_shape}.")
    query_shape = query.shape[len(case_shape) : -1]
    return query.reshape(case_shape + (size(query_shape), query.shape[-1])), query_shape


def blocked_case_logsumexp(
    query: Array,
    support: Array,
    log_weight: Array,
    scale: Array,
    case_shape: tuple[int, ...],
    *,
    workspace_bytes: int = 4 * 1024 * 1024,
) -> tuple[Array, tuple[int, ...]]:
    x, query_shape = _case_queries(query, support, case_shape)
    value = blocked_logsumexp(
        x, support, log_weight, scale, workspace_bytes=workspace_bytes // size(case_shape)
    )
    return value.reshape(case_shape + query_shape), query_shape


def blocked_radius_sum(
    query: Array,
    support: Array,
    mask: Array,
    weight: Array,
    values: Array,
    radius: Array,
    case_shape: tuple[int, ...],
    metric: Any,
    *,
    workspace_bytes: int = 4 * 1024 * 1024,
) -> tuple[Array, Array]:
    x, query_shape = _case_queries(query, support, case_shape)
    q, n = x.shape[-2], support.shape[-2]
    b = pair_block_size(
        q,
        n,
        x.shape[-1] + values.shape[-1],
        jnp.dtype(jnp.result_type(x, weight, values)).itemsize,
        workspace_bytes // size(case_shape),
    )
    if q == 0:
        return (
            jnp.zeros(
                case_shape + query_shape + (values.shape[-1],),
                dtype=jnp.result_type(values, weight),
            ),
            jnp.zeros(case_shape + query_shape, dtype=weight.dtype),
        )
    if b >= max(q, n):
        weights = jnp.where(
            mask[..., None, :]
            & (pairwise_distances(x, support, metric=metric) <= radius),
            weight[..., None, :],
            0.0,
        )
        numerator = jnp.sum(weights[..., None] * values[..., None, :, :], axis=-2)
        denominator = jnp.sum(weights, axis=-1)
        return numerator.reshape(
            case_shape + query_shape + (values.shape[-1],)
        ), denominator.reshape(case_shape + query_shape)
    qp, np_ = ((q + b - 1) // b) * b, ((n + b - 1) // b) * b
    pad = ((0, 0),) * len(case_shape)
    x = jnp.pad(x, pad + ((0, qp - q), (0, 0)))
    y = jnp.pad(support, pad + ((0, np_ - n), (0, 0)))
    w = jnp.pad(jnp.where(mask, weight, 0.0), pad + ((0, np_ - n),))
    v = jnp.pad(values, pad + ((0, np_ - n), (0, 0)))
    channels = values.shape[-1]
    dtype = jnp.result_type(values, weight)
    output = jnp.zeros(case_shape + (qp, channels + 1), dtype=dtype)

    def row_step(i: Array, output: Array) -> Array:
        xb = jax.lax.dynamic_slice_in_dim(x, i * b, b, axis=-2)

        def column_step(j: Array, total: Array) -> Array:
            yb = jax.lax.dynamic_slice_in_dim(y, j * b, b, axis=-2)
            wb = jax.lax.dynamic_slice_in_dim(w, j * b, b, axis=-1)
            vb = jax.lax.dynamic_slice_in_dim(v, j * b, b, axis=-2)
            weights = jnp.where(
                pairwise_distances(xb, yb, metric=metric) <= radius, wb[..., None, :], 0.0
            )
            augmented = jnp.concatenate(
                (vb, jnp.ones(vb.shape[:-1] + (1,), dtype=vb.dtype)), axis=-1
            )
            return total + weights @ augmented

        value = jax.lax.fori_loop(
            0,
            np_ // b,
            column_step,
            jnp.zeros(case_shape + (b, channels + 1), dtype=dtype),
        )
        return jax.lax.dynamic_update_slice_in_dim(output, value, i * b, axis=-2)

    result = jax.lax.fori_loop(0, qp // b, row_step, output)[..., :q, :]
    result = result.reshape(case_shape + query_shape + (channels + 1,))
    return result[..., :channels], jnp.real(result[..., -1])


def broadcast_support(
    value: Array, query_ndim: int, case_shape: tuple[int, ...]
) -> Array:
    return value.reshape(case_shape + (1,) * query_ndim + value.shape[len(case_shape) :])


def gather_support(values: Array, indices: Array, case_shape: tuple[int, ...]) -> Array:
    """Gather support-axis values for case/query/k indices."""
    cases = size(case_shape)
    query_shape = indices.shape[len(case_shape) : -1]
    q = size(tuple(query_shape)) if query_shape else 1
    k = indices.shape[-1]
    trailing = values.shape[len(case_shape) + 1 :]
    value_cases = values.reshape((cases, values.shape[len(case_shape)], -1))
    index_cases = indices.reshape((cases, q, k))
    gathered = jax.vmap(lambda v, i: v[i])(value_cases, index_cases)
    return gathered.reshape(case_shape + query_shape + (k,) + trailing)


def pad_support(array: Array, capacity: int, sample_axis: int, fill: Any) -> Array:
    count = array.shape[sample_axis]
    if capacity <= count:
        indices = jnp.arange(capacity)
        return jnp.take(array, indices, axis=sample_axis)
    width = [(0, 0)] * array.ndim
    width[sample_axis] = (0, capacity - count)
    return jnp.pad(array, tuple(width), constant_values=fill)


def chunked_call(model: Any, points: Array, chunk_size: int) -> Array:
    """Evaluate query blocks without a complete query-by-support allocation."""
    x = jnp.asarray(points)
    if x.ndim != 2:
        raise ValueError("chunked prediction currently requires (query, feature) input.")
    block = int(chunk_size)
    if block <= 0:
        raise ValueError("chunk_size must be positive.")
    if x.shape[0] == 0:
        return model(x)
    return jnp.concatenate(
        tuple(model(x[start : start + block]) for start in range(0, x.shape[0], block)),
        axis=0,
    )


def validated_weights(value: Array) -> Array:
    weights = jnp.asarray(value, dtype=jnp.float64)
    return eqx.error_if(
        weights,
        jnp.any(~jnp.isfinite(weights)) | jnp.any(weights < 0.0),
        "Sample and measure weights must be finite and nonnegative.",
    )


__all__ = [
    "broadcast_support",
    "case_distances",
    "chunked_call",
    "distance_softmax_regularity",
    "masked_softmax",
    "gather_support",
    "metric_regularity",
    "pad_support",
    "size",
    "validate_metric",
    "validated_weights",
]
