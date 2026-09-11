#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Riemann and Hurwitz zeta functions with one Euler--Maclaurin substrate."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from ._bernoulli import even_bernoulli_coefficient
from ._continuation import _loggamma_lanczos, promote_principal
from ._dtype import promote_real


def _isfinite(value: Array, /) -> Array:
    if jnp.issubdtype(value.dtype, jnp.complexfloating):
        return jnp.isfinite(jnp.real(value)) & jnp.isfinite(jnp.imag(value))
    return jnp.isfinite(value)


def _nan_like(value: Array, /) -> Array:
    return jnp.full_like(value, jnp.nan)


def _hurwitz_em_value_derivatives(s: Array, a: Array, /) -> tuple[Array, Array, Array]:
    """Euler--Maclaurin value and its exact differentiated finite formula."""
    s, a = jnp.broadcast_arrays(s, a)
    real_dtype = jnp.real(s).dtype
    leading_terms = 32 if real_dtype == jnp.float64 else 18
    correction_terms = 12 if real_dtype == jnp.float64 else 7

    value = jnp.zeros_like(s)
    derivative_s = jnp.zeros_like(s)
    derivative_a = jnp.zeros_like(s)
    for index in range(leading_terms):
        base = a + index
        logarithm = jnp.log(base)
        term = jnp.exp(-s * logarithm)
        value = value + term
        derivative_s = derivative_s - logarithm * term
        derivative_a = derivative_a - s * term / base

    endpoint = a + leading_terms
    endpoint_log = jnp.log(endpoint)
    pole_distance = s - 1.0
    safe_pole_distance = jnp.where(pole_distance == 0.0, 1.0, pole_distance)
    endpoint_power = jnp.exp((1.0 - s) * endpoint_log)
    tail = endpoint_power / safe_pole_distance
    value = value + tail
    derivative_s = derivative_s + tail * (-endpoint_log - 1.0 / safe_pole_distance)
    derivative_a = derivative_a - jnp.exp(-s * endpoint_log)

    half_term = 0.5 * jnp.exp(-s * endpoint_log)
    value = value + half_term
    derivative_s = derivative_s - endpoint_log * half_term
    derivative_a = derivative_a - s * half_term / endpoint

    rising = jnp.ones_like(s)
    rising_derivative = jnp.zeros_like(s)
    for order in range(1, 2 * correction_terms):
        previous = rising
        rising = previous * (s + order - 1)
        rising_derivative = rising_derivative * (s + order - 1) + previous
        if order % 2 == 0:
            continue
        bernoulli_index = (order + 1) // 2
        coefficient = even_bernoulli_coefficient(bernoulli_index, s.dtype)
        power = jnp.exp(-(s + order) * endpoint_log)
        correction = coefficient * rising * power
        value = value + correction
        derivative_s = derivative_s + coefficient * power * (
            rising_derivative - rising * endpoint_log
        )
        derivative_a = derivative_a - (s + order) * correction / endpoint

    return value, derivative_s, derivative_a


def _hurwitz_value_derivatives(s: Array, a: Array, /) -> tuple[Array, Array, Array]:
    s, a = jnp.broadcast_arrays(s, a)
    valid = _isfinite(s) & _isfinite(a) & (jnp.real(a) > 0.0)
    safe_s = jnp.where(valid, s, jnp.ones_like(s) * 2.0)
    safe_a = jnp.where(valid, a, jnp.ones_like(a))
    value, derivative_s, derivative_a = _hurwitz_em_value_derivatives(safe_s, safe_a)

    pole = valid & (s == 1.0)
    infinity = jnp.asarray(jnp.inf, dtype=jnp.real(s).dtype).astype(s.dtype)
    value = jnp.where(pole, infinity, value)
    derivative_s = jnp.where(pole, -infinity, derivative_s)
    derivative_a = jnp.where(pole, -infinity, derivative_a)

    invalid_value = _nan_like(value)
    return (
        jnp.where(valid, value, invalid_value),
        jnp.where(valid, derivative_s, invalid_value),
        jnp.where(valid, derivative_a, invalid_value),
    )


def _zeta_value_derivative(s: Array, /) -> tuple[Array, Array]:
    one = jnp.ones_like(s)
    direct_value, direct_derivative, _ = _hurwitz_em_value_derivatives(s, one)

    reflected_argument = one - s
    reflected_base, reflected_base_derivative, _ = _hurwitz_em_value_derivatives(
        reflected_argument, one
    )

    def amplitude(argument: Array, /) -> Array:
        gamma_factor = jnp.exp(_loggamma_lanczos(1.0 - argument))
        scale = jnp.exp(argument * math.log(2.0) + (argument - 1.0) * math.log(math.pi))
        return scale * gamma_factor * jnp.sin(0.5 * math.pi * argument)

    reflected_amplitude, reflected_amplitude_derivative = jax.jvp(
        amplitude, (s,), (jnp.ones_like(s),)
    )
    reflected_value = reflected_amplitude * reflected_base
    reflected_derivative = (
        reflected_amplitude_derivative * reflected_base
        - reflected_amplitude * reflected_base_derivative
    )

    use_reflection = jnp.real(s) < 0.0
    value = jnp.where(use_reflection, reflected_value, direct_value)
    derivative = jnp.where(use_reflection, reflected_derivative, direct_derivative)

    real_axis = (
        jnp.imag(s) == 0.0 if jnp.iscomplexobj(s) else jnp.ones_like(s, dtype=bool)
    )
    nearest = jnp.round(jnp.real(s))
    trivial_zero = (
        use_reflection
        & real_axis
        & (jnp.real(s) == nearest)
        & (nearest < 0.0)
        & (jnp.remainder(nearest, 2.0) == 0.0)
    )
    value = jnp.where(trivial_zero, jnp.zeros_like(value), value)

    pole = (s == 1.0) & real_axis
    infinity = jnp.asarray(jnp.inf, dtype=jnp.real(s).dtype).astype(s.dtype)
    value = jnp.where(pole, infinity, value)
    derivative = jnp.where(pole, -infinity, derivative)

    if jnp.iscomplexobj(s):
        positive_infinity = jnp.zeros_like(s, dtype=bool)
    else:
        positive_infinity = jnp.isposinf(s)
    value = jnp.where(positive_infinity, jnp.ones_like(value), value)
    derivative = jnp.where(positive_infinity, jnp.zeros_like(derivative), derivative)

    valid = _isfinite(s) | positive_infinity
    invalid = _nan_like(value)
    return jnp.where(valid, value, invalid), jnp.where(valid, derivative, invalid)


@jax.custom_jvp
def _zeta_array(s: Array, /) -> Array:
    return _zeta_value_derivative(s)[0]


@_zeta_array.defjvp
def _zeta_jvp(primals, tangents):
    (s,) = primals
    (s_tangent,) = tangents
    value, derivative = _zeta_value_derivative(s)
    return value, s_tangent * derivative


@jax.custom_jvp
def _hurwitz_zeta_array(s: Array, a: Array, /) -> Array:
    return _hurwitz_value_derivatives(s, a)[0]


@_hurwitz_zeta_array.defjvp
def _hurwitz_zeta_jvp(primals, tangents):
    s, a = primals
    s_tangent, a_tangent = tangents
    value, derivative_s, derivative_a = _hurwitz_value_derivatives(s, a)
    return value, s_tangent * derivative_s + a_tangent * derivative_a


def zeta(s: ArrayLike, /) -> Array:
    """Evaluate the Riemann zeta function, including its principal continuation."""
    value = jnp.asarray(s)
    if jnp.issubdtype(value.dtype, jnp.complexfloating):
        (promoted,) = promote_principal(value)
    else:
        (promoted,) = promote_real("zeta", value)
    return _zeta_array(promoted)


def hurwitz_zeta(s: ArrayLike, a: ArrayLike, /) -> Array:
    """Evaluate Hurwitz zeta for positive-real-part ``a``."""
    order = jnp.asarray(s)
    parameter = jnp.asarray(a)
    if jnp.iscomplexobj(order) or jnp.iscomplexobj(parameter):
        order, parameter = promote_principal(order, parameter)
    else:
        order, parameter = promote_real("hurwitz_zeta", order, parameter)
    order, parameter = jnp.broadcast_arrays(order, parameter)
    return _hurwitz_zeta_array(order, parameter)


__all__ = ["hurwitz_zeta", "zeta"]
