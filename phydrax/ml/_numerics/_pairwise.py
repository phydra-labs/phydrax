#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, TypeAlias

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike


MetricName: TypeAlias = Literal["euclidean", "squared-euclidean", "manhattan", "cosine"]


def squared_euclidean_distances(left: ArrayLike, right: ArrayLike, /) -> Array:
    x = jnp.asarray(left)
    y = jnp.asarray(right)
    if x.shape[-1] != y.shape[-1]:
        raise ValueError("Pairwise inputs must have equal feature dimensions.")
    x2 = jnp.sum(jnp.real(x * jnp.conj(x)), axis=-1, keepdims=True)
    y2 = jnp.sum(jnp.real(y * jnp.conj(y)), axis=-1, keepdims=True)
    cross = jnp.real(x @ jnp.conj(jnp.swapaxes(y, -1, -2)))
    return jnp.maximum(x2 + jnp.swapaxes(y2, -1, -2) - 2.0 * cross, 0.0)


def pairwise_distances(
    left: ArrayLike,
    right: ArrayLike | None = None,
    /,
    *,
    metric: MetricName | Callable[[Array, Array], Array] = "euclidean",
) -> Array:
    x = jnp.asarray(left)
    y = x if right is None else jnp.asarray(right)
    if x.ndim < 2 or y.ndim < 2 or x.shape[-1] != y.shape[-1]:
        raise ValueError("Pairwise inputs must end in (item, feature) axes.")
    if callable(metric):
        case_shape = jnp.broadcast_shapes(x.shape[:-2], y.shape[:-2])
        x_ = jnp.broadcast_to(x, case_shape + x.shape[-2:])
        y_ = jnp.broadcast_to(y, case_shape + y.shape[-2:])
        case_count = 1
        for size in case_shape:
            case_count *= int(size)
        x_cases = x_.reshape((case_count,) + x.shape[-2:])
        y_cases = y_.reshape((case_count,) + y.shape[-2:])
        distances = jax.vmap(
            lambda x_case, y_case: jax.vmap(
                lambda a: jax.vmap(lambda b: metric(a, b))(y_case)
            )(x_case)
        )(x_cases, y_cases)
        return distances.reshape(case_shape + (x.shape[-2], y.shape[-2]))
    if metric == "squared-euclidean":
        return squared_euclidean_distances(x, y)
    if metric == "euclidean":
        return jnp.sqrt(squared_euclidean_distances(x, y))
    if metric == "manhattan":
        return jnp.sum(jnp.abs(x[..., :, None, :] - y[..., None, :, :]), axis=-1)
    if metric == "cosine":
        numerator = jnp.real(x @ jnp.conj(jnp.swapaxes(y, -1, -2)))
        x_norm = jnp.linalg.norm(x, axis=-1)
        y_norm = jnp.linalg.norm(y, axis=-1)
        denominator = x_norm[..., :, None] * y_norm[..., None, :]
        similarity = jnp.where(denominator > 0.0, numerator / denominator, 0.0)
        return 1.0 - similarity
    raise ValueError(f"Unsupported metric {metric!r}.")


def chunked_pairwise_apply(
    left: ArrayLike,
    right: ArrayLike,
    reducer: Callable[[Array, int], Array],
    /,
    *,
    metric: MetricName = "squared-euclidean",
    chunk_size: int,
) -> Array:
    """Apply a row-block reducer without materializing a complete pairwise matrix."""
    x = jnp.asarray(left)
    y = jnp.asarray(right)
    block = int(chunk_size)
    if block <= 0:
        raise ValueError("chunk_size must be positive.")
    outputs = []
    for start in range(0, int(x.shape[-2]), block):
        stop = min(start + block, int(x.shape[-2]))
        outputs.append(
            reducer(pairwise_distances(x[..., start:stop, :], y, metric=metric), start)
        )
    return jnp.concatenate(outputs, axis=-1)


def soft_assignments(
    distances: ArrayLike,
    /,
    *,
    temperature: ArrayLike = 1.0,
    mask: ArrayLike | None = None,
) -> Array:
    temperature_ = jnp.asarray(temperature, dtype=float)
    logits = -jnp.asarray(distances) / jnp.maximum(temperature_, jnp.finfo(float).tiny)
    if mask is not None:
        logits = jnp.where(jnp.asarray(mask, dtype=bool), logits, -jnp.inf)
    probabilities = jax.nn.softmax(logits, axis=-1)
    return jnp.where(jnp.isfinite(probabilities), probabilities, 0.0)


def hard_assignments(distances: ArrayLike, /) -> Array:
    return jnp.argmin(jnp.asarray(distances), axis=-1).astype(jnp.int32)


__all__ = [
    "MetricName",
    "chunked_pairwise_apply",
    "hard_assignments",
    "pairwise_distances",
    "soft_assignments",
    "squared_euclidean_distances",
]
