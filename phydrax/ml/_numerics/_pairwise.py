#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, TypeAlias

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike


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
    for start in range(0, x.shape[-2], block):
        stop = min(start + block, x.shape[-2])
        outputs.append(
            reducer(pairwise_distances(x[..., start:stop, :], y, metric=metric), start)
        )
    return jnp.concatenate(outputs, axis=-1)


def pair_block_size(
    query_count: int,
    support_count: int,
    feature_count: int,
    itemsize: int,
    workspace_bytes: int,
    /,
    *,
    extra_row_bytes: int = 0,
) -> int:
    """Conservative two-axis tile admission, including pair-feature temporaries."""
    if workspace_bytes <= 0:
        raise ValueError("workspace_bytes must be positive.")
    extent = max(query_count, support_count, 1)
    per_pair = (feature_count + 3) * itemsize
    if (
        query_count * support_count * per_pair + query_count * extra_row_bytes
        <= workspace_bytes
    ):
        return extent
    return max(1, min(extent, workspace_bytes // (extent * per_pair + extra_row_bytes)))


def blocked_top_k(
    query: Array,
    support: Array,
    mask: Array,
    count: int,
    /,
    *,
    metric: MetricName | Callable[[Array, Array], Array] = "euclidean",
    block_size: int | None = None,
    workspace_bytes: int = 4 * 1024 * 1024,
    exclude_self: bool = False,
) -> tuple[Array, Array]:
    """Exact two-axis top-k with lexicographic (distance, support index) ties."""
    if (
        (block_size is not None and block_size <= 0)
        or count <= 0
        or count > support.shape[-2]
    ):
        raise ValueError("Invalid block size or neighbor count.")
    q, n = query.shape[-2], support.shape[-2]
    shape = jnp.broadcast_shapes(query.shape[:-2], support.shape[:-2])
    distance_dtype = jax.eval_shape(
        lambda a, b: pairwise_distances(a, b, metric=metric),
        jax.ShapeDtypeStruct(query.shape[:-2] + (1, query.shape[-1]), query.dtype),
        jax.ShapeDtypeStruct(support.shape[:-2] + (1, support.shape[-1]), support.dtype),
    ).dtype
    b = pair_block_size(
        q,
        n,
        query.shape[-1],
        max(
            jnp.dtype(distance_dtype).itemsize,
            query.dtype.itemsize,
            support.dtype.itemsize,
        ),
        workspace_bytes,
        extra_row_bytes=4
        * count
        * (jnp.dtype(distance_dtype).itemsize + jnp.dtype(jnp.int32).itemsize),
    )
    if q == 0:
        return (
            jnp.zeros(shape + (0, count), dtype=jnp.int32),
            jnp.zeros(shape + (0, count), dtype=distance_dtype),
        )
    if block_size is not None:
        b = min(b, block_size)
    if b >= max(q, n):
        distances = jnp.where(
            mask[..., None, :], pairwise_distances(query, support, metric=metric), jnp.inf
        )
        if exclude_self:
            distances = jnp.where(
                jnp.arange(q)[:, None] == jnp.arange(n)[None, :], jnp.inf, distances
            )
        negative, indices = jax.lax.top_k(-distances, count)
        return indices, -negative
    qp, np_ = ((q + b - 1) // b) * b, ((n + b - 1) // b) * b
    x = jnp.pad(query, ((0, 0),) * (query.ndim - 2) + ((0, qp - q), (0, 0)))
    y = jnp.pad(support, ((0, 0),) * (support.ndim - 2) + ((0, np_ - n), (0, 0)))
    m = jnp.pad(mask, ((0, 0),) * (mask.ndim - 1) + ((0, np_ - n),))
    distances = jnp.full(shape + (qp, count), jnp.inf, dtype=distance_dtype)
    indices = jnp.full(shape + (qp, count), n, dtype=jnp.int32)

    def query_step(i: Array, output: tuple[Array, Array]) -> tuple[Array, Array]:
        xb = jax.lax.dynamic_slice_in_dim(x, i * b, b, axis=-2)
        initial = (
            jnp.full(shape + (b, count), jnp.inf, dtype=distance_dtype),
            jnp.full(shape + (b, count), n, dtype=jnp.int32),
        )

        def support_step(j: Array, state: tuple[Array, Array]) -> tuple[Array, Array]:
            yb = jax.lax.dynamic_slice_in_dim(y, j * b, b, axis=-2)
            mb = jax.lax.dynamic_slice_in_dim(m, j * b, b, axis=-1)
            d = jnp.where(
                mb[..., None, :], pairwise_distances(xb, yb, metric=metric), jnp.inf
            )
            ids = jnp.broadcast_to(j * b + jnp.arange(b, dtype=jnp.int32), d.shape)
            if exclude_self:
                d = jnp.where((i * b + jnp.arange(b))[:, None] == ids, jnp.inf, d)
            ds = jnp.concatenate((state[0], d), axis=-1)
            ix = jnp.concatenate((state[1], ids), axis=-1)
            ds, ix = jax.lax.sort((ds, ix), dimension=-1, num_keys=2)
            return ds[..., :count], ix[..., :count]

        ds, ix = jax.lax.fori_loop(0, np_ // b, support_step, initial)
        return (
            jax.lax.dynamic_update_slice_in_dim(output[0], ds, i * b, axis=-2),
            jax.lax.dynamic_update_slice_in_dim(output[1], ix, i * b, axis=-2),
        )

    distances, indices = jax.lax.fori_loop(0, qp // b, query_step, (distances, indices))
    return indices[..., :q, :], distances[..., :q, :]


def blocked_logsumexp(
    query: Array,
    support: Array,
    log_weight: Array,
    scale: Array,
    /,
    *,
    block_size: int | None = None,
    workspace_bytes: int = 4 * 1024 * 1024,
) -> Array:
    """Stream Gaussian logits with a bounded pair tile and log-add-exp accumulator."""
    if block_size is not None and block_size <= 0:
        raise ValueError("block_size must be positive.")
    q, n = query.shape[-2], support.shape[-2]
    b = pair_block_size(
        q,
        n,
        query.shape[-1],
        jnp.dtype(jnp.result_type(query, support, log_weight)).itemsize,
        workspace_bytes,
    )
    if block_size is not None:
        b = min(b, block_size)
    if b >= max(q, n):
        logits = log_weight[..., None, :] - scale * squared_euclidean_distances(
            query, support
        )
        active = jnp.any(jnp.isfinite(log_weight), axis=-1)[..., None]
        value = jax.scipy.special.logsumexp(
            jnp.where(active[..., None], logits, 0.0), axis=-1
        )
        return jnp.where(active, value, -jnp.inf)
    qp, np_ = ((q + b - 1) // b) * b, ((n + b - 1) // b) * b
    x = jnp.pad(query, ((0, 0),) * (query.ndim - 2) + ((0, qp - q), (0, 0)))
    y = jnp.pad(support, ((0, 0),) * (support.ndim - 2) + ((0, np_ - n), (0, 0)))
    lw = jnp.pad(
        log_weight,
        ((0, 0),) * (log_weight.ndim - 1) + ((0, np_ - n),),
        constant_values=-jnp.inf,
    )
    shape = jnp.broadcast_shapes(query.shape[:-2], support.shape[:-2])
    output = jnp.full(shape + (qp,), -jnp.inf, dtype=jnp.float64)
    if q == 0:
        return output

    def query_step(i: Array, output: Array) -> Array:
        xb = jax.lax.dynamic_slice_in_dim(x, i * b, b, axis=-2)

        def support_step(j: Array, total: Array) -> Array:
            yb = jax.lax.dynamic_slice_in_dim(y, j * b, b, axis=-2)
            weights = jax.lax.dynamic_slice_in_dim(lw, j * b, b, axis=-1)
            logits = weights[..., None, :] - scale * squared_euclidean_distances(xb, yb)
            active = jnp.any(jnp.isfinite(weights), axis=-1)[..., None]
            value = jax.scipy.special.logsumexp(
                jnp.where(active[..., None], logits, 0.0), axis=-1
            )
            value = jnp.where(active, value, -jnp.inf)
            finite = jnp.isfinite(total) | jnp.isfinite(value)
            combined = jnp.logaddexp(
                jnp.where(finite, total, 0.0), jnp.where(finite, value, 0.0)
            )
            return jnp.where(finite, combined, -jnp.inf)

        total = jax.lax.fori_loop(
            0, np_ // b, support_step, jnp.full(shape + (b,), -jnp.inf, dtype=jnp.float64)
        )
        return jax.lax.dynamic_update_slice_in_dim(output, total, i * b, axis=-1)

    return jax.lax.fori_loop(0, qp // b, query_step, output)[..., :q]


def soft_assignments(
    distances: ArrayLike,
    /,
    *,
    temperature: ArrayLike = 1.0,
    mask: ArrayLike | None = None,
) -> Array:
    temperature_ = jnp.asarray(temperature, dtype=jnp.float64)
    logits = -jnp.asarray(distances) / jnp.maximum(
        temperature_, jnp.finfo(jnp.float64).tiny
    )
    if mask is not None:
        logits = jnp.where(jnp.asarray(mask, dtype=jnp.bool_), logits, -jnp.inf)
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
