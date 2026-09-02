#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike


def _as_inexact_array(values: ArrayLike, /) -> Array:
    array = jnp.asarray(values)
    if not jnp.issubdtype(array.dtype, jnp.inexact):
        return array.astype(float)
    return array


def _normalize_axis(axis: int, ndim: int, /) -> int:
    if isinstance(axis, bool) or not isinstance(axis, int):
        raise TypeError("axis must be an integer.")
    if ndim <= 0:
        raise ValueError("Signal arrays must have positive rank.")
    if axis < -ndim or axis >= ndim:
        raise ValueError(f"axis {axis} is out of bounds for rank {ndim}.")
    return axis % ndim


def _positive_int(value: int, name: str, /) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


def _signal_to_last(values: ArrayLike, axis: int, /) -> tuple[Array, int]:
    array = _as_inexact_array(values)
    resolved = _normalize_axis(axis, array.ndim)
    return jnp.moveaxis(array, resolved, -1), resolved


def _signal_from_last(values: Array, axis: int, /) -> Array:
    return jnp.moveaxis(values, -1, axis)


def _shared_taps(taps: ArrayLike, /) -> Array:
    array = _as_inexact_array(taps)
    if array.ndim != 1:
        raise ValueError(f"taps must have shape (tap_count,); got {array.shape}.")
    if array.shape[0] == 0:
        raise ValueError("taps must contain at least one coefficient.")
    return array


def _promote_signal_and_taps(
    values: ArrayLike,
    taps: ArrayLike,
    /,
) -> tuple[Array, Array]:
    array = _as_inexact_array(values)
    coefficients = _shared_taps(taps)
    dtype = jnp.result_type(array.dtype, coefficients.dtype)
    return array.astype(dtype), coefficients.astype(dtype)


def _replace_axis_size(
    shape: Sequence[int],
    axis: int,
    size: int,
    /,
) -> tuple[int, ...]:
    dimensions = tuple(int(dimension) for dimension in shape)
    resolved = _normalize_axis(axis, len(dimensions))
    output = list(dimensions)
    output[resolved] = int(size)
    return tuple(output)


def _valid_prefix(
    capacity: int,
    valid_length: ArrayLike | None,
    /,
) -> tuple[Array, Array]:
    if valid_length is None:
        valid = jnp.asarray(capacity, dtype=jnp.int64)
    else:
        raw = jnp.asarray(valid_length)
        if raw.ndim != 0 or not jnp.issubdtype(raw.dtype, jnp.integer):
            raise TypeError("valid_length must be a scalar integer.")
        valid = raw.astype(jnp.int64)
    valid = eqx.error_if(
        valid,
        (valid < 0) | (valid > capacity),
        f"valid_length must lie in [0, {capacity}].",
    )
    active = jnp.arange(capacity, dtype=valid.dtype) < valid
    return valid, active


__all__: list[str] = []
