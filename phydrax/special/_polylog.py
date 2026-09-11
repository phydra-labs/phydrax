#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Principal polylogarithm on an explicit, numerically supported envelope."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from jax import Array, lax
from jax.typing import ArrayLike

from ._continuation import principal_log, promote_principal
from ._dilog import _dilog_value_derivative
from ._zeta import _zeta_value_derivative


def _polylog_series_value_derivatives(
    s: Array,
    z: Array,
    /,
) -> tuple[Array, Array, Array]:
    power = z
    value = z
    derivative_s = jnp.zeros_like(z)
    derivative_z = jnp.ones_like(z)
    terms = 224 if jnp.real(s).dtype == jnp.float64 else 96
    for index in range(2, terms + 1):
        previous_power = power
        power = previous_power * z
        logarithm = jnp.asarray(math.log(index), dtype=s.dtype)
        weight = jnp.exp(-s * logarithm)
        term = power * weight
        value = value + term
        derivative_s = derivative_s - logarithm * term
        derivative_z = derivative_z + index * previous_power * weight
    return value, derivative_s, derivative_z


def _polylog_general(
    s: Array,
    z: Array,
    /,
) -> tuple[Array, Array, Array, Array]:
    finite = (
        jnp.isfinite(jnp.real(s))
        & jnp.isfinite(jnp.imag(s))
        & jnp.isfinite(jnp.real(z))
        & jnp.isfinite(jnp.imag(z))
    )
    bounded_order = (jnp.abs(jnp.real(s)) <= 20.0) & (jnp.abs(jnp.imag(s)) <= 20.0)
    supported = finite & bounded_order & (jnp.abs(z) <= 0.75)
    safe_s = jnp.where(supported, s, jnp.ones_like(s) * 2.5)
    safe_z = jnp.where(supported, z, jnp.zeros_like(z))
    value, derivative_s, derivative_z = _polylog_series_value_derivatives(safe_s, safe_z)
    return value, derivative_s, derivative_z, supported


def _polylog_value_derivatives(s: Array, z: Array, /) -> tuple[Array, Array, Array]:
    s, z = jnp.broadcast_arrays(s, z)
    value, derivative_s, derivative_z, supported = _polylog_general(s, z)

    zero_order = s == 0.0
    one_order = s == 1.0
    two_order = s == 2.0
    value = jnp.where(zero_order, z / (1.0 - z), value)
    value = jnp.where(one_order, -principal_log(1.0 - z), value)
    li_two, li_two_derivative = _dilog_value_derivative(z)
    value = jnp.where(two_order, li_two, value)
    derivative_z = jnp.where(zero_order, 1.0 / (1.0 - z) ** 2, derivative_z)
    derivative_z = jnp.where(one_order, 1.0 / (1.0 - z), derivative_z)
    derivative_z = jnp.where(two_order, li_two_derivative, derivative_z)

    at_one = z == 1.0
    one_supported = (
        jnp.isfinite(jnp.real(s))
        & jnp.isfinite(jnp.imag(s))
        & (jnp.real(s) > 1.0)
        & (jnp.abs(jnp.real(s)) <= 20.0)
        & (jnp.abs(jnp.imag(s)) <= 20.0)
    )
    zeta_value, zeta_derivative = _zeta_value_derivative(s)
    value = jnp.where(at_one & one_supported, zeta_value, value)
    derivative_s = jnp.where(at_one & one_supported, zeta_derivative, derivative_s)
    derivative_z = jnp.where(
        at_one & one_supported,
        _zeta_value_derivative(s - 1.0)[0],
        derivative_z,
    )
    supported = supported | (at_one & one_supported)

    invalid = lax.complex(
        jnp.full_like(jnp.real(value), jnp.nan),
        jnp.full_like(jnp.real(value), jnp.nan),
    )
    return (
        jnp.where(supported, value, invalid),
        jnp.where(supported, derivative_s, invalid),
        jnp.where(supported, derivative_z, invalid),
    )


@jax.custom_jvp
def _polylog_array(s: Array, z: Array, /) -> Array:
    return _polylog_value_derivatives(s, z)[0]


@_polylog_array.defjvp
def _polylog_jvp(primals, tangents):
    s, z = primals
    s_tangent, z_tangent = tangents
    value, derivative_s, derivative_z = _polylog_value_derivatives(s, z)
    return value, s_tangent * derivative_s + z_tangent * derivative_z


def polylog(s: ArrayLike, z: ArrayLike, /) -> Array:
    """Evaluate principal ``Li_s(z)`` on a bounded, differentiated envelope.

    All results are complex. Orders satisfy ``|Re(s)|, |Im(s)| <= 20``.
    General orders support ``|z| <= 0.75``; ``z=1`` is additionally supported
    when ``Re(s)>1``. Unsupported lanes return complex NaN.
    """
    order, argument = promote_principal(s, z)
    order, argument = jnp.broadcast_arrays(order, argument)
    return _polylog_array(order, argument)


__all__ = ["polylog"]
