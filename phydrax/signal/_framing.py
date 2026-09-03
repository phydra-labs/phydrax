#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ._axis import _normalize_axis, _positive_int, _signal_to_last


def frame(
    values: ArrayLike,
    frame_length: int,
    hop_length: int,
    /,
    *,
    axis: int = -1,
) -> Array:
    """Extract complete overlapping frames along one signal axis.

    The signal axis is replaced by adjacent ``(frame, within_frame)`` axes.
    Incomplete trailing samples are not emitted.
    """
    length = _positive_int(frame_length, "frame_length")
    hop = _positive_int(hop_length, "hop_length")
    canonical, resolved_axis = _signal_to_last(values, axis)
    sample_count = int(canonical.shape[-1])
    if sample_count < length:
        raise ValueError(
            "The signal axis length must be at least frame_length; "
            f"got {sample_count} and {length}."
        )
    frame_count = 1 + (sample_count - length) // hop
    starts = hop * jnp.arange(frame_count, dtype=jnp.int64)
    indices = starts[:, None] + jnp.arange(length, dtype=jnp.int64)[None, :]
    framed = jnp.take(canonical, indices, axis=-1)
    return jnp.moveaxis(framed, (-2, -1), (resolved_axis, resolved_axis + 1))


def overlap_add(
    frames: ArrayLike,
    hop_length: int,
    /,
    *,
    frame_axis: int = -2,
    sample_axis: int = -1,
) -> Array:
    """Sum framed samples onto their natural overlap-add grid."""
    hop = _positive_int(hop_length, "hop_length")
    array = jnp.asarray(frames)
    if not jnp.issubdtype(array.dtype, jnp.inexact):
        array = array.astype(float)
    resolved_frame = _normalize_axis(frame_axis, array.ndim)
    resolved_sample = _normalize_axis(sample_axis, array.ndim)
    if resolved_frame == resolved_sample:
        raise ValueError("frame_axis and sample_axis must be distinct.")
    canonical = jnp.moveaxis(
        array,
        (resolved_frame, resolved_sample),
        (-2, -1),
    )
    frame_count = int(canonical.shape[-2])
    frame_length = int(canonical.shape[-1])
    if frame_count <= 0 or frame_length <= 0:
        raise ValueError("frames must contain positive frame and sample dimensions.")
    output_length = hop * (frame_count - 1) + frame_length
    indices = (
        hop * jnp.arange(frame_count, dtype=jnp.int64)[:, None]
        + jnp.arange(frame_length, dtype=jnp.int64)[None, :]
    )
    stream_shape = canonical.shape[:-2]
    flattened = canonical.reshape((-1, frame_count, frame_length))

    def _sum_stream(stream: Array) -> Array:
        return jnp.zeros((output_length,), dtype=stream.dtype).at[indices].add(stream)

    summed = jax.vmap(_sum_stream)(flattened).reshape((*stream_shape, output_length))
    output_axis = resolved_frame - int(resolved_sample < resolved_frame)
    return jnp.moveaxis(summed, -1, output_axis)


__all__ = ["frame", "overlap_add"]
