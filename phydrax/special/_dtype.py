#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Shared dtype promotion for named special functions."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from jax import Array, lax
from jax.typing import ArrayLike


def _float_layout(value: Array) -> tuple[Array, int, int, int, int]:
    if value.dtype == jnp.float32:
        return (
            lax.bitcast_convert_type(value, jnp.uint32),
            0x80000000,
            0x007FFFFF,
            0x00800000,
            149,
        )
    if value.dtype == jnp.float64:
        return (
            lax.bitcast_convert_type(value, jnp.uint64),
            0x8000000000000000,
            0x000FFFFFFFFFFFFF,
            0x0010000000000000,
            1074,
        )
    raise TypeError(f"bit-safe floating-point helpers do not support {value.dtype}")


def _exact_zero(value: Array) -> Array:
    bits, sign_mask, _, _, _ = _float_layout(value)
    magnitude_mask = jnp.asarray(sign_mask - 1, dtype=bits.dtype)
    return (bits & magnitude_mask) == 0


def _signbit(value: Array) -> Array:
    bits, sign_mask, _, _, _ = _float_layout(value)
    return (bits & jnp.asarray(sign_mask, dtype=bits.dtype)) != 0


def _positive_subnormal(value: Array) -> Array:
    bits, sign_mask, _, min_normal, _ = _float_layout(value)
    magnitude = bits & jnp.asarray(sign_mask - 1, dtype=bits.dtype)
    return (
        ((bits & jnp.asarray(sign_mask, dtype=bits.dtype)) == 0)
        & (magnitude != 0)
        & (magnitude < jnp.asarray(min_normal, dtype=bits.dtype))
    )


@jax.custom_jvp
def _positive_log(value: Array) -> Array:
    bits, _, mantissa_mask, _, subnormal_exponent = _float_layout(value)
    mantissa = bits & jnp.asarray(mantissa_mask, dtype=bits.dtype)
    subnormal_log = jnp.log(mantissa.astype(value.dtype)) - (
        jnp.asarray(subnormal_exponent, dtype=value.dtype) * math.log(2.0)
    )
    return jnp.where(_positive_subnormal(value), subnormal_log, jnp.log(value))


@_positive_log.defjvp
def _positive_log_jvp(
    primals: tuple[Array], tangents: tuple[Array]
) -> tuple[Array, Array]:
    (value,) = primals
    (tangent,) = tangents
    result = _positive_log(value)
    reciprocal = jnp.where(_positive_subnormal(value), jnp.exp(-result), 1.0 / value)
    return result, tangent * reciprocal


def promote_real(name: str, *values: ArrayLike) -> tuple[Array, ...]:
    """Promote real inputs to one inexact dtype, with low precision widened."""
    arrays = tuple(jnp.asarray(value) for value in values)
    if any(jnp.issubdtype(array.dtype, jnp.complexfloating) for array in arrays):
        raise TypeError(f"{name} does not support complex-valued inputs")

    dtype = jnp.result_type(*arrays)
    if not jnp.issubdtype(dtype, jnp.inexact):
        dtype = jnp.asarray(0.0).dtype
    elif dtype == jnp.float16 or dtype == jnp.bfloat16:
        dtype = jnp.float32
    return tuple(jnp.asarray(array, dtype=dtype) for array in arrays)


def complex_from_parts(name: str, real: ArrayLike, imag: ArrayLike) -> Array:
    """Promote and broadcast real parts before constructing a complex array."""
    promoted_real, promoted_imag = promote_real(name, real, imag)
    broadcast_real, broadcast_imag = jnp.broadcast_arrays(promoted_real, promoted_imag)
    return lax.complex(broadcast_real, broadcast_imag)


def promote_complex(value: ArrayLike, /) -> Array:
    """Promote an input to complex64 or complex128 without narrowing."""
    array = jnp.asarray(value)
    if jnp.issubdtype(array.dtype, jnp.complexfloating):
        return array

    (real,) = promote_real("wofz", array)
    dtype = jnp.complex128 if real.dtype == jnp.float64 else jnp.complex64
    return real.astype(dtype)
