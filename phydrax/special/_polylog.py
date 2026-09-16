#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Principal polylogarithm on an explicit, numerically supported envelope."""

from __future__ import annotations

import math
from functools import partial

import jax
import jax.numpy as jnp
from jax import Array, lax
from jax.custom_derivatives import SymbolicZero
from jax.typing import ArrayLike

from ._continuation import promote_principal


_TERM_AXIS_MAX_ELEMENTS = 1_048_576


def _term_count(dtype, /) -> int:
    return 224 if jnp.dtype(dtype) == jnp.float64 else 96


def _term_axis_is_bounded(z: Array, terms: int, /) -> bool:
    return z.ndim > 0 and math.prod(z.shape) * terms <= _TERM_AXIS_MAX_ELEMENTS


def _polylog_series_value_streaming(s: Array, z: Array, /) -> Array:
    """Evaluate the defining series with constant-size loop state."""
    terms = _term_count(jnp.real(s).dtype)
    real_dtype = jnp.real(s).dtype

    def body(index, state):
        power, value = state
        next_power = power * z
        logarithm = jnp.log(jnp.asarray(index, dtype=real_dtype)).astype(s.dtype)
        return next_power, value + next_power * jnp.exp(-s * logarithm)

    _, value = lax.fori_loop(2, terms + 1, body, (z, z))
    return value


def _polylog_series_value_term_axis(s: Array, z: Array, /) -> Array:
    """Evaluate the defining series by reducing one bounded trailing term axis."""
    terms = _term_count(jnp.real(s).dtype)
    indices = jnp.arange(1, terms + 1, dtype=jnp.int32)
    logarithms = jnp.log(indices.astype(jnp.real(s).dtype)).astype(s.dtype)
    summands = z[..., None] ** indices * jnp.exp(-s[..., None] * logarithms)
    return jnp.sum(summands, axis=-1)


def _polylog_series_value(s: Array, z: Array, /) -> Array:
    terms = _term_count(jnp.real(s).dtype)
    if _term_axis_is_bounded(z, terms):
        return _polylog_series_value_term_axis(s, z)
    return _polylog_series_value_streaming(s, z)


def _polylog_series_order_derivative_streaming(s: Array, z: Array, /) -> Array:
    terms = _term_count(jnp.real(s).dtype)
    real_dtype = jnp.real(s).dtype

    def body(index, state):
        power, derivative = state
        next_power = power * z
        logarithm = jnp.log(jnp.asarray(index, dtype=real_dtype)).astype(s.dtype)
        term = next_power * jnp.exp(-s * logarithm)
        return next_power, derivative - logarithm * term

    _, derivative = lax.fori_loop(2, terms + 1, body, (z, jnp.zeros_like(z)))
    return derivative


def _polylog_series_order_derivative_term_axis(s: Array, z: Array, /) -> Array:
    terms = _term_count(jnp.real(s).dtype)
    indices = jnp.arange(1, terms + 1, dtype=jnp.int32)
    logarithms = jnp.log(indices.astype(jnp.real(s).dtype)).astype(s.dtype)
    summands = z[..., None] ** indices * jnp.exp(-s[..., None] * logarithms)
    return -jnp.sum(logarithms * summands, axis=-1)


def _polylog_series_order_derivative(s: Array, z: Array, /) -> Array:
    terms = _term_count(jnp.real(s).dtype)
    if _term_axis_is_bounded(z, terms):
        return _polylog_series_order_derivative_term_axis(s, z)
    return _polylog_series_order_derivative_streaming(s, z)


def _polylog_series_argument_derivative_streaming(s: Array, z: Array, /) -> Array:
    terms = _term_count(jnp.real(s).dtype)
    real_dtype = jnp.real(s).dtype

    def body(index, state):
        power, derivative = state
        logarithm = jnp.log(jnp.asarray(index, dtype=real_dtype)).astype(s.dtype)
        weight = jnp.exp(-s * logarithm)
        next_derivative = derivative + index * power * weight
        return power * z, next_derivative

    _, derivative = lax.fori_loop(2, terms + 1, body, (z, jnp.ones_like(z)))
    return derivative


def _polylog_series_argument_derivative_term_axis(s: Array, z: Array, /) -> Array:
    terms = _term_count(jnp.real(s).dtype)
    indices = jnp.arange(1, terms + 1, dtype=jnp.int32)
    logarithms = jnp.log(indices.astype(jnp.real(s).dtype)).astype(s.dtype)
    weights = jnp.exp(-s[..., None] * logarithms)
    return jnp.sum(indices * z[..., None] ** (indices - 1) * weights, axis=-1)


def _polylog_series_argument_derivative(s: Array, z: Array, /) -> Array:
    terms = _term_count(jnp.real(s).dtype)
    if _term_axis_is_bounded(z, terms):
        return _polylog_series_argument_derivative_term_axis(s, z)
    return _polylog_series_argument_derivative_streaming(s, z)


def _domain_mask(s: Array, z: Array, /) -> Array:
    finite = (
        jnp.isfinite(jnp.real(s))
        & jnp.isfinite(jnp.imag(s))
        & jnp.isfinite(jnp.real(z))
        & jnp.isfinite(jnp.imag(z))
    )
    bounded_order = (jnp.abs(jnp.real(s)) <= 20.0) & (jnp.abs(jnp.imag(s)) <= 20.0)
    return finite & bounded_order & (jnp.abs(z) <= 0.75)


def _complex_nan(reference: Array, /) -> Array:
    real = jnp.full_like(jnp.real(reference), jnp.nan)
    return lax.complex(real, real)


def _polylog_value(s: Array, z: Array, /) -> Array:
    s, z = jnp.broadcast_arrays(s, z)
    supported = _domain_mask(s, z)
    safe_s = jnp.where(supported, s, jnp.ones_like(s) * 2.5)
    safe_z = jnp.where(supported, z, jnp.zeros_like(z))
    value = _polylog_series_value(safe_s, safe_z)
    return jnp.where(supported, value, _complex_nan(value))


def _polylog_order_derivative(s: Array, z: Array, /) -> Array:
    s, z = jnp.broadcast_arrays(s, z)
    supported = _domain_mask(s, z)
    safe_s = jnp.where(supported, s, jnp.ones_like(s) * 2.5)
    safe_z = jnp.where(supported, z, jnp.zeros_like(z))
    derivative = _polylog_series_order_derivative(safe_s, safe_z)
    invalid = (1.0 + s) * _complex_nan(derivative)
    return jnp.where(supported, derivative, invalid)


def _polylog_argument_derivative(s: Array, z: Array, /) -> Array:
    s, z = jnp.broadcast_arrays(s, z)
    supported = _domain_mask(s, z)
    safe_s = jnp.where(supported, s, jnp.ones_like(s) * 2.5)
    safe_z = jnp.where(supported, z, jnp.zeros_like(z))
    derivative = _polylog_series_argument_derivative(safe_s, safe_z)
    invalid = (1.0 + z) * _complex_nan(derivative)
    return jnp.where(supported, derivative, invalid)


@jax.custom_jvp
def _polylog_array(s: Array, z: Array, /) -> Array:
    return _polylog_value(s, z)


@partial(_polylog_array.defjvp, symbolic_zeros=True)
def _polylog_jvp(primals, tangents):
    s, z = primals
    s_tangent, z_tangent = tangents
    value = _polylog_value(s, z)
    tangent = jnp.zeros_like(value)
    if not isinstance(s_tangent, SymbolicZero):
        tangent = tangent + s_tangent * _polylog_order_derivative(s, z)
    if not isinstance(z_tangent, SymbolicZero):
        tangent = tangent + z_tangent * _polylog_argument_derivative(s, z)
    return value, tangent


def polylog(s: ArrayLike, z: ArrayLike, /) -> Array:
    """Evaluate principal ``Li_s(z)`` on a bounded, differentiated envelope.

    All results are complex. Orders satisfy ``|Re(s)|, |Im(s)| <= 20`` and
    arguments satisfy ``|z| <= 0.75``. Unsupported value or derivative lanes
    return complex NaN; use :func:`zeta` directly for the identity at ``z=1``.
    """
    order, argument = promote_principal(s, z)
    order, argument = jnp.broadcast_arrays(order, argument)
    return _polylog_array(order, argument)


__all__ = ["polylog"]
