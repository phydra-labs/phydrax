#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import prod

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike


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


def _accumulate_reference(
    flat: Array, carry: tuple[Array, Array], /
) -> tuple[Array, Array]:
    def step(
        current: tuple[Array, Array], term: Array
    ) -> tuple[tuple[Array, Array], None]:
        high, correction = current
        next_high, error = two_sum(high, term)
        return (next_high, correction + error), None

    return jax.lax.scan(step, carry, flat, unroll=1)[0]


def _accumulate_pallas_real(
    flat: Array, carry: tuple[Array, Array], /
) -> tuple[Array, Array]:
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import mosaic_gpu as plgpu

    length = flat.shape[0]
    output_shape = carry[0].shape
    lane_count = carry[0].size
    full_tiles, remainder = divmod(lane_count, 128)

    def kernel(
        terms_ref: jax.Ref,
        high_ref: jax.Ref,
        correction_ref: jax.Ref,
        output_high_ref: jax.Ref,
        output_correction_ref: jax.Ref,
    ) -> None:
        program = jax.lax.axis_index("lanes")

        def run(start: Array, *, vector: bool) -> None:
            window = pl.ds(start, 128) if vector else start
            high = high_ref[window]
            correction = correction_ref[window]
            if vector:
                layout = plgpu.Layout.WG_STRIDED((128,), vec_size=1)
                high = plgpu.layout_cast(high, layout)
                correction = plgpu.layout_cast(correction, layout)

            def step(index: int, current: tuple[Array, Array]) -> tuple[Array, Array]:
                high, correction = current
                term = terms_ref[index, window]
                if vector:
                    term = plgpu.layout_cast(term, layout)
                next_high, error = two_sum(high, term)
                return next_high, correction + error

            # A device-local loop preserves every seeded Sum2 operation.
            high, correction = jax.lax.fori_loop(0, length, step, (high, correction))
            output_high_ref[window] = high
            output_correction_ref[window] = correction

        if full_tiles:

            @pl.when(program < full_tiles)
            def vector_tile() -> None:
                run(program * 128, vector=True)

        if remainder:

            @pl.when(program >= full_tiles)
            def scalar_lane() -> None:
                run(full_tiles * 128 + program - full_tiles, vector=False)

    shape = jax.ShapeDtypeStruct((lane_count,), flat.dtype)
    high, correction = plgpu.kernel(
        kernel,
        out_type=(shape, shape),
        grid=(full_tiles + remainder,),
        grid_names=("lanes",),
        compiler_params=plgpu.CompilerParams(
            lowering_semantics=plgpu.LoweringSemantics.Lane
        ),
        cost_estimate=pl.CostEstimate(
            flops=7 * length * lane_count,
            transcendentals=0,
            bytes_accessed=(length + 4) * lane_count * flat.dtype.itemsize,
        ),
    )(
        flat.reshape((length, lane_count)),
        carry[0].reshape((lane_count,)),
        carry[1].reshape((lane_count,)),
    )
    return high.reshape(output_shape), correction.reshape(output_shape)


def _accumulate_pallas(flat: Array, carry: tuple[Array, Array], /) -> tuple[Array, Array]:
    if jnp.issubdtype(flat.dtype, jnp.complexfloating):
        # Mosaic GPU has no complex scalar type. Components obey independent,
        # identically ordered add/subtract recurrences.
        def components(value: Array) -> Array:
            return jnp.stack((value.real, value.imag), axis=-1)

        high, correction = _accumulate_pallas_real(
            components(flat), (components(carry[0]), components(carry[1]))
        )
        return (
            jax.lax.complex(high[..., 0], high[..., 1]),
            jax.lax.complex(correction[..., 0], correction[..., 1]),
        )
    return _accumulate_pallas_real(flat, carry)


def _accumulate(flat: Array, carry: tuple[Array, Array], /) -> tuple[Array, Array]:
    if flat.shape[0] == 0 or carry[0].size == 0:
        return carry
    # Admit only shapes whose real-component flattened indices fit int32.
    # Larger arrays retain the genuine reference rather than narrowing offsets.
    components = 2 if jnp.issubdtype(flat.dtype, jnp.complexfloating) else 1
    if flat.size * components > (1 << 31) - 1:
        return _accumulate_reference(flat, carry)
    from ._compensated_native import accumulate_native

    # Platform-specific primitive lowerings choose execution at destination
    # lowering time, without exposing GPU implementation details to callers.
    return accumulate_native(flat, carry[0], carry[1])


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
        (array.shape[array.ndim - output_ndim_ :] if output_ndim_ else ()) != output_shape
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
    shape = tuple(
        1 if index in axes else array.shape[index] for index in range(array.ndim)
    )
    return result.reshape(shape)


__all__ = ["compensated_sum", "compensated_sum_chunks", "two_sum"]
