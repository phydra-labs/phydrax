#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from .._numerics import pairwise_distances


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


def masked_softmax(logits: Array, mask: Array) -> Array:
    """Normalize masked logits without NaNs for completely inactive rows."""
    active = jnp.asarray(mask, dtype=bool)
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
        query_shape = tuple(int(s) for s in x.shape[len(case_shape) : -1])
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
    query_shape = tuple(int(s) for s in x.shape[:-1])
    distance = pairwise_distances(x.reshape((-1, x.shape[-1])), support, metric=metric)
    return distance.reshape(query_shape + (support.shape[-2],)), query_shape


def broadcast_support(
    value: Array, query_ndim: int, case_shape: tuple[int, ...]
) -> Array:
    return value.reshape(case_shape + (1,) * query_ndim + value.shape[len(case_shape) :])


def gather_support(values: Array, indices: Array, case_shape: tuple[int, ...]) -> Array:
    """Gather support-axis values for case/query/k indices."""
    cases = size(case_shape)
    query_shape = indices.shape[len(case_shape) : -1]
    q = size(tuple(int(s) for s in query_shape)) if query_shape else 1
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
    weights = jnp.asarray(value, dtype=float)
    return eqx.error_if(
        weights,
        jnp.any(~jnp.isfinite(weights)) | jnp.any(weights < 0.0),
        "Sample and measure weights must be finite and nonnegative.",
    )


__all__ = [
    "broadcast_support",
    "case_distances",
    "chunked_call",
    "masked_softmax",
    "gather_support",
    "pad_support",
    "size",
    "validate_metric",
    "validated_weights",
]
