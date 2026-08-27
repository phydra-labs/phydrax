#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import prod

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike


Axis = int | tuple[int, ...] | None
_SUPPORTED_REAL_DTYPES = (jnp.dtype(jnp.float32), jnp.dtype(jnp.float64))
_BLOCK_SIZE = 32
_BLOCK_THRESHOLD = 2 * _BLOCK_SIZE


def two_sum(left: Array, right: Array, /) -> tuple[Array, Array]:
    """Return a rounded sum and its error-free residual."""
    total = left + right
    virtual_right = total - left
    error = (left - (total - virtual_right)) + (right - virtual_right)
    return total, error


def _real_dtype(dtype: jnp.dtype, /) -> jnp.dtype:
    if jnp.issubdtype(dtype, jnp.complexfloating):
        return jnp.empty((), dtype=dtype).real.dtype
    return dtype


def _flat_leading(value: Array, output_ndim: int, /) -> Array:
    output_shape = value.shape[value.ndim - output_ndim :] if output_ndim else ()
    reduction_shape = value.shape[: value.ndim - output_ndim]
    return value.reshape((prod(reduction_shape),) + output_shape)


def _accumulate(flat: Array, carry: tuple[Array, Array], /) -> tuple[Array, Array]:
    def step(current, term):
        high, correction = current
        next_high, error = two_sum(high, term)
        return (next_high, correction + error), None

    return jax.lax.scan(step, carry, flat, unroll=1)[0]


def _block_expansion(flat: Array, /) -> tuple[Array, Array]:
    block_count = (flat.shape[0] + _BLOCK_SIZE - 1) // _BLOCK_SIZE
    padding = block_count * _BLOCK_SIZE - flat.shape[0]
    padded = jnp.pad(
        flat,
        ((0, padding),) + ((0, 0),) * (flat.ndim - 1),
    )
    blocked = padded.reshape((block_count, _BLOCK_SIZE) + flat.shape[1:])
    block_shape = (block_count,) + flat.shape[1:]
    carry = (
        jnp.zeros(block_shape, dtype=flat.dtype),
        jnp.zeros(block_shape, dtype=flat.dtype),
    )
    return _accumulate(jnp.swapaxes(blocked, 0, 1), carry)


def compensated_sum_chunks(
    chunks: tuple[ArrayLike, ...],
    /,
    *,
    output_ndim: int = 0,
) -> Array:
    """Sum signed leading-axis chunks with a twofold compensated accumulator."""
    if not chunks:
        raise ValueError("compensated_sum_chunks requires at least one chunk.")
    output_ndim_ = int(output_ndim)
    if output_ndim_ < 0:
        raise ValueError("output_ndim must be non-negative.")
    arrays = tuple(jnp.asarray(chunk) for chunk in chunks)
    if any(array.ndim < output_ndim_ for array in arrays):
        raise ValueError("Every chunk must contain all declared output dimensions.")
    output_shape = (
        arrays[0].shape[arrays[0].ndim - output_ndim_ :] if output_ndim_ else ()
    )
    if any(
        (array.shape[array.ndim - output_ndim_ :] if output_ndim_ else ())
        != output_shape
        for array in arrays[1:]
    ):
        raise ValueError("Compensated chunks must have identical trailing output shapes.")
    dtype = jnp.result_type(*(array.dtype for array in arrays))
    arrays = tuple(array.astype(dtype) for array in arrays)
    flat_chunks = tuple(_flat_leading(array, output_ndim_) for array in arrays)
    if not jnp.issubdtype(dtype, jnp.inexact) or _real_dtype(dtype) not in (
        _SUPPORTED_REAL_DTYPES
    ):
        native = jnp.zeros(output_shape, dtype=dtype)
        for flat in flat_chunks:
            native = native + jnp.sum(flat, axis=0)
        return native
    reduced_chunks: list[Array] = []
    for flat in flat_chunks:
        if flat.shape[0] > _BLOCK_THRESHOLD:
            high, correction = _block_expansion(flat)
            reduced_chunks.extend((high, correction))
        else:
            reduced_chunks.append(flat)
    carry = (
        jnp.zeros(output_shape, dtype=dtype),
        jnp.zeros(output_shape, dtype=dtype),
    )
    for reduced in reduced_chunks:
        carry = _accumulate(reduced, carry)
    high, correction = carry
    return high + correction


def _resolve_axes(axis: Axis, ndim: int, /) -> tuple[int, ...]:
    if axis is None:
        return tuple(range(ndim))
    raw = (axis,) if isinstance(axis, int) else tuple(axis)
    resolved = tuple(int(value) + ndim if int(value) < 0 else int(value) for value in raw)
    if any(value < 0 or value >= ndim for value in resolved):
        raise ValueError("axis contains an out-of-range reduction dimension.")
    if len(set(resolved)) != len(resolved):
        raise ValueError("axis contains duplicate reduction dimensions.")
    return resolved


def compensated_sum(
    value: ArrayLike,
    /,
    *,
    axis: Axis = None,
    keepdims: bool = False,
) -> Array:
    """Sum explicit axes using the branch-free Sum2 error transformation."""
    array = jnp.asarray(value)
    axes = _resolve_axes(axis, array.ndim)
    if not axes:
        return array
    output_axes = tuple(index for index in range(array.ndim) if index not in axes)
    transposed = jnp.transpose(array, axes + output_axes)
    flattened = transposed.reshape(
        (prod(array.shape[index] for index in axes),)
        + tuple(array.shape[index] for index in output_axes)
    )
    result = compensated_sum_chunks(
        (flattened,),
        output_ndim=len(output_axes),
    )
    if not keepdims:
        return result
    shape = tuple(1 if index in axes else array.shape[index] for index in range(array.ndim))
    return result.reshape(shape)


__all__ = ["compensated_sum", "compensated_sum_chunks", "two_sum"]
