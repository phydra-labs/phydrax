from __future__ import annotations

import jax
import jax.numpy as jnp


def segment_sum(
    data: jnp.ndarray,
    segment_ids: jnp.ndarray,
    num_segments: int | None = None,
) -> jnp.ndarray:
    return jax.ops.segment_sum(data, segment_ids, num_segments)


def segment_mean(
    data: jnp.ndarray,
    segment_ids: jnp.ndarray,
    num_segments: int | None = None,
) -> jnp.ndarray:
    numer = segment_sum(data, segment_ids, num_segments)
    ones = jnp.ones((data.shape[0],) + (1,) * (data.ndim - 1), dtype=data.dtype)
    denom = segment_sum(ones, segment_ids, num_segments)
    denom = jnp.maximum(denom, jnp.ones_like(denom))
    return numer / denom


def segment_variance(
    data: jnp.ndarray,
    segment_ids: jnp.ndarray,
    num_segments: int | None = None,
) -> jnp.ndarray:
    means = segment_mean(data, segment_ids, num_segments)[segment_ids]
    ones = jnp.ones((data.shape[0],) + (1,) * (data.ndim - 1), dtype=data.dtype)
    counts = segment_sum(ones, segment_ids, num_segments)
    counts = jnp.maximum(counts, jnp.ones_like(counts))
    variances = segment_sum(jnp.square(data - means), segment_ids, num_segments) / counts
    return variances


def segment_normalize(
    data: jnp.ndarray,
    segment_ids: jnp.ndarray,
    num_segments: int | None = None,
    *,
    eps: float = 1e-8,
) -> jnp.ndarray:
    means = segment_mean(data, segment_ids, num_segments)[segment_ids]
    variances = segment_variance(data, segment_ids, num_segments)[segment_ids]
    eps_arr = jnp.asarray(eps, dtype=variances.dtype)
    return (data - means) * jax.lax.rsqrt(jnp.maximum(variances, eps_arr))


def segment_max(
    data: jnp.ndarray,
    segment_ids: jnp.ndarray,
    num_segments: int | None = None,
) -> jnp.ndarray:
    return jax.ops.segment_max(data, segment_ids, num_segments)


def segment_min(
    data: jnp.ndarray,
    segment_ids: jnp.ndarray,
    num_segments: int | None = None,
) -> jnp.ndarray:
    return jax.ops.segment_min(data, segment_ids, num_segments)


def _replace_empty_segments_with_constant(
    aggregated_segments: jnp.ndarray,
    segment_ids: jnp.ndarray,
    num_segments: int | None,
    *,
    constant: float,
) -> jnp.ndarray:
    probe_shape = (segment_ids.shape[0],) + aggregated_segments.shape[1:]
    num_elements = segment_sum(
        jnp.ones(probe_shape, dtype=jnp.int32),
        segment_ids,
        num_segments,
    )
    constant_arr = jnp.asarray(constant, dtype=aggregated_segments.dtype)
    return jnp.where(num_elements > 0, aggregated_segments, constant_arr)


def segment_max_or_constant(
    data: jnp.ndarray,
    segment_ids: jnp.ndarray,
    num_segments: int | None = None,
    *,
    constant: float = 0.0,
) -> jnp.ndarray:
    maxs = segment_max(data, segment_ids, num_segments)
    return _replace_empty_segments_with_constant(
        maxs,
        segment_ids,
        num_segments,
        constant=constant,
    )


def segment_min_or_constant(
    data: jnp.ndarray,
    segment_ids: jnp.ndarray,
    num_segments: int | None = None,
    *,
    constant: float = 0.0,
) -> jnp.ndarray:
    mins = segment_min(data, segment_ids, num_segments)
    return _replace_empty_segments_with_constant(
        mins,
        segment_ids,
        num_segments,
        constant=constant,
    )


def segment_softmax(
    logits: jnp.ndarray,
    segment_ids: jnp.ndarray,
    num_segments: int,
) -> jnp.ndarray:
    max_per_segment = segment_max(logits, segment_ids, num_segments)
    max_per_segment = max_per_segment[segment_ids]
    logits = logits - max_per_segment
    exp_logits = jnp.exp(logits)
    denom = segment_sum(exp_logits, segment_ids, num_segments)
    denom = denom[segment_ids]
    denom = jnp.maximum(denom, jnp.asarray(1e-12, dtype=denom.dtype))
    return exp_logits / denom


def scatter_add(
    src: jnp.ndarray,
    index: jnp.ndarray,
    dim_size: int,
) -> jnp.ndarray:
    return segment_sum(src, index, dim_size)


def scatter_mean(
    src: jnp.ndarray,
    index: jnp.ndarray,
    dim_size: int,
) -> jnp.ndarray:
    return segment_mean(src, index, dim_size)


def scatter_max(
    src: jnp.ndarray,
    index: jnp.ndarray,
    dim_size: int,
) -> jnp.ndarray:
    out = segment_max(src, index, dim_size)
    return jnp.where(jnp.isfinite(out), out, jnp.zeros_like(out))


def scatter_min(
    src: jnp.ndarray,
    index: jnp.ndarray,
    dim_size: int,
) -> jnp.ndarray:
    out = segment_min(src, index, dim_size)
    return jnp.where(jnp.isfinite(out), out, jnp.zeros_like(out))


__all__ = [
    "segment_sum",
    "segment_mean",
    "segment_variance",
    "segment_normalize",
    "segment_max",
    "segment_min",
    "segment_max_or_constant",
    "segment_min_or_constant",
    "segment_softmax",
    "scatter_add",
    "scatter_mean",
    "scatter_max",
    "scatter_min",
]
