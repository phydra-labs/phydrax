#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fast unweighted ordering via L2 permutahedron projection.

The formulation follows Blondel, Teboul, Berthet, and Djolonga,
``Fast Differentiable Sorting and Ranking`` (ICML 2020).
"""

from __future__ import annotations

from typing import Any, overload

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike


Value = ArrayLike | cx.Field


def _pav_decreasing_with_blocks(values: Array, /) -> tuple[Array, Array]:
    """Project one vector onto the nonincreasing cone with a PAV stack."""
    count = values.shape[0]
    levels = jnp.zeros_like(values)
    block_sizes = jnp.zeros((count,), dtype=jnp.int32)

    def append(index, state):
        levels_, block_sizes_, num_blocks = state
        levels_ = levels_.at[num_blocks].set(values[index])
        block_sizes_ = block_sizes_.at[num_blocks].set(1)
        num_blocks = num_blocks + 1

        def violates(inner_state):
            levels__, _, num_blocks__ = inner_state
            previous = jnp.maximum(num_blocks__ - 2, 0)
            current = jnp.maximum(num_blocks__ - 1, 0)
            return (num_blocks__ > 1) & (levels__[previous] < levels__[current])

        def merge(inner_state):
            levels__, block_sizes__, num_blocks__ = inner_state
            previous = num_blocks__ - 2
            current = num_blocks__ - 1
            merged_size = block_sizes__[previous] + block_sizes__[current]
            merged_level = (
                block_sizes__[previous] * levels__[previous]
                + block_sizes__[current] * levels__[current]
            ) / merged_size
            levels__ = levels__.at[previous].set(merged_level)
            block_sizes__ = block_sizes__.at[previous].set(merged_size)
            block_sizes__ = block_sizes__.at[current].set(0)
            return levels__, block_sizes__, num_blocks__ - 1

        return jax.lax.while_loop(
            violates,
            merge,
            (levels_, block_sizes_, num_blocks),
        )

    levels, block_sizes, num_blocks = jax.lax.fori_loop(
        0,
        count,
        append,
        (levels, block_sizes, jnp.asarray(0, dtype=jnp.int32)),
    )
    block_ends = jnp.cumsum(block_sizes)
    block_index = jnp.arange(count, dtype=block_sizes.dtype)
    block_ends = jnp.where(block_index < num_blocks, block_ends, count + 1)
    assignments = jnp.searchsorted(
        block_ends,
        block_index,
        side="right",
    )
    return levels[assignments], assignments


@jax.custom_jvp
def _pav_decreasing(values: Array, /) -> Array:
    projected, _ = _pav_decreasing_with_blocks(values)
    return projected


@_pav_decreasing.defjvp
def _pav_decreasing_jvp(primals, tangents):
    (values,), (values_dot,) = primals, tangents
    projected, assignments = _pav_decreasing_with_blocks(values)
    count = values.shape[0]
    tangent_sums = jax.ops.segment_sum(
        values_dot,
        assignments,
        num_segments=count,
    )
    block_sizes = jax.ops.segment_sum(
        jnp.ones_like(values),
        assignments,
        num_segments=count,
    )
    projected_dot = tangent_sums[assignments] / block_sizes[assignments]
    return projected, projected_dot


def _standardize(values: Array, /) -> tuple[Array, Array, Array]:
    center = jnp.mean(values)
    centered = values - center
    variance = jnp.mean(centered**2)
    scale = jnp.where(variance > 0.0, jnp.sqrt(variance), 1.0)
    return centered / scale, center, scale


def _soft_sort_row(values: Array, temperature: Array, /) -> Array:
    count = values.shape[0]
    standardized, center, scale = _standardize(values)
    ordered = jnp.sort(standardized)
    denominator = max(count - 1, 1)
    anchors = jnp.arange(
        count - 1,
        -1,
        -1,
        dtype=values.dtype,
    ) / (denominator * temperature)
    relaxed = _pav_decreasing(anchors + ordered) - anchors
    return center + scale * relaxed


def _soft_rank_row(values: Array, temperature: Array, /) -> Array:
    count = values.shape[0]
    standardized, _, _ = _standardize(values)
    denominator = max(count - 1, 1)
    anchors = (
        jnp.arange(
            count - 1,
            -1,
            -1,
            dtype=values.dtype,
        )
        / denominator
    )
    scaled = standardized / temperature
    permutation = jnp.argsort(scaled, descending=True, stable=True)
    ordered = scaled[permutation]
    projected = ordered - _pav_decreasing(ordered - anchors)
    inverse = (
        jnp.zeros((count,), dtype=jnp.int32)
        .at[permutation]
        .set(jnp.arange(count, dtype=jnp.int32))
    )
    return (denominator * projected)[inverse]


def _data_axis(
    value: Value,
    /,
    *,
    axis: int | str,
) -> tuple[Array, int, tuple[Any, ...] | None]:
    if isinstance(value, cx.Field):
        if not isinstance(axis, str):
            raise TypeError("Named values fields require a named axis.")
        if axis not in value.named_dims:
            raise ValueError(f"values is missing named axis {axis!r}.")
        data = jnp.asarray(value.data)
        position = value.dims.index(axis)
        dims: tuple[Any, ...] | None = value.dims
    else:
        if not isinstance(axis, int):
            raise TypeError("Raw values arrays require an integer axis.")
        data = jnp.asarray(value)
        if data.ndim < 1:
            raise ValueError("values must have at least one dimension.")
        position = axis + data.ndim if axis < 0 else axis
        if position < 0 or position >= data.ndim:
            raise ValueError("values axis is out of range.")
        dims = None
    if jnp.issubdtype(data.dtype, jnp.complexfloating):
        raise TypeError("values must be real-valued.")
    if not jnp.issubdtype(data.dtype, jnp.floating):
        data = data.astype(jnp.result_type(float))
    if data.shape[position] < 1:
        raise ValueError("values ordering axis must be nonempty.")
    data = eqx.error_if(
        data,
        jnp.any(~jnp.isfinite(data)),
        "values must contain only finite values.",
    )
    return data, position, dims


def _temperature(value: ArrayLike, dtype: jnp.dtype, /) -> Array:
    temperature = jnp.asarray(value, dtype=dtype)
    if temperature.ndim != 0:
        raise ValueError("temperature must be a scalar.")
    return eqx.error_if(
        temperature,
        ~jnp.isfinite(temperature) | (temperature <= 0.0),
        "temperature must be finite and positive.",
    )


def _map_rows(
    data: Array,
    position: int,
    temperature: Array,
    operation,
    /,
) -> Array:
    moved = jnp.moveaxis(data, position, -1)
    leading_shape = moved.shape[:-1]
    count = moved.shape[-1]
    if count == 1:
        if operation is _soft_rank_row:
            output = jnp.zeros_like(moved)
        else:
            output = moved
    else:
        rows = moved.reshape((-1, count))
        output = jax.vmap(lambda row: operation(row, temperature))(rows)
        output = output.reshape(leading_shape + (count,))
    return jnp.moveaxis(output, -1, position)


def _restore(data: Array, dims: tuple[Any, ...] | None, /) -> Array | cx.Field:
    return data if dims is None else cx.Field(data, dims=dims)


@overload
def fast_soft_sort(
    values: ArrayLike,
    /,
    *,
    temperature: ArrayLike = 0.5,
    axis: int | str = -1,
    descending: bool = False,
) -> Array: ...


@overload
def fast_soft_sort(
    values: cx.Field,
    /,
    *,
    temperature: ArrayLike = 0.5,
    axis: int | str = -1,
    descending: bool = False,
) -> cx.Field: ...


def fast_soft_sort(
    values: Value,
    /,
    *,
    temperature: ArrayLike = 0.5,
    axis: int | str = -1,
    descending: bool = False,
) -> Array | cx.Field:
    """Return fast relaxed sorted values for an unweighted ordering axis.

    Rows are centered and variance-standardized before an L2 permutahedron
    projection solved by pool-adjacent violators. ``temperature`` is therefore
    dimensionless. Smaller values approach hard sorting; the map remains
    piecewise smooth rather than globally smooth. This operator returns values,
    not a transport plan, and does not support weights.
    """
    if not isinstance(descending, bool):
        raise TypeError("descending must be a bool.")
    data, position, dims = _data_axis(values, axis=axis)
    configured = _temperature(temperature, data.dtype)
    output = _map_rows(data, position, configured, _soft_sort_row)
    if descending:
        output = jnp.flip(output, axis=position)
    return _restore(output, dims)


@overload
def fast_soft_rank(
    values: ArrayLike,
    /,
    *,
    temperature: ArrayLike = 0.5,
    axis: int | str = -1,
    descending: bool = False,
) -> Array: ...


@overload
def fast_soft_rank(
    values: cx.Field,
    /,
    *,
    temperature: ArrayLike = 0.5,
    axis: int | str = -1,
    descending: bool = False,
) -> cx.Field: ...


def fast_soft_rank(
    values: Value,
    /,
    *,
    temperature: ArrayLike = 0.5,
    axis: int | str = -1,
    descending: bool = False,
) -> Array | cx.Field:
    """Return fast zero-based relaxed ranks for an unweighted ordering axis.

    The result is a membership-free rank surrogate on ``[0, n - 1]`` whose
    unweighted sum is ``n * (n - 1) / 2``. It is not interchangeable with the
    weighted barycentric ranks returned by :func:`soft_rank`.
    """
    if not isinstance(descending, bool):
        raise TypeError("descending must be a bool.")
    data, position, dims = _data_axis(values, axis=axis)
    configured = _temperature(temperature, data.dtype)
    output = _map_rows(data, position, configured, _soft_rank_row)
    if descending:
        output = data.shape[position] - 1 - output
    return _restore(output, dims)


__all__ = ["fast_soft_rank", "fast_soft_sort"]
