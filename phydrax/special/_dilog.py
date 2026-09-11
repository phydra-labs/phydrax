#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Principal complex dilogarithm and Spence function."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from ._bernoulli import even_bernoulli_coefficient
from ._continuation import principal_log, promote_principal


def _dilog_series(z: Array, /) -> tuple[Array, Array]:
    power = z
    value = z
    derivative = jnp.ones_like(z)
    for index in range(2, 97):
        power = power * z
        denominator = jnp.asarray(index * index, dtype=z.dtype)
        value = value + power / denominator
        derivative = derivative + power * index / (z * denominator)
    derivative = jnp.where(z == 0.0, jnp.ones_like(z), derivative)
    return value, derivative


def _dilog_jonquiere(z: Array, /) -> tuple[Array, Array]:
    mu = principal_log(z)
    safe_mu = jnp.where(mu == 0.0, jnp.ones_like(mu), mu)
    log_minus_mu = principal_log(-safe_mu)
    value = (
        jnp.asarray(math.pi**2 / 6.0, dtype=z.dtype)
        + mu * (1.0 - log_minus_mu)
        - 0.25 * mu * mu
    )
    derivative_mu = -log_minus_mu - 0.5 * mu
    for index in range(1, 13):
        coefficient = even_bernoulli_coefficient(index, z.dtype)
        degree = 2 * index
        value = value - coefficient * mu ** (degree + 1) / (degree * (degree + 1))
        derivative_mu = derivative_mu - coefficient * mu**degree / degree
    derivative = derivative_mu / z
    at_one = z == 1.0
    infinity = jnp.asarray(jnp.inf, dtype=jnp.real(z).dtype).astype(z.dtype)
    return (
        jnp.where(at_one, jnp.asarray(math.pi**2 / 6.0, dtype=z.dtype), value),
        jnp.where(at_one, infinity, derivative),
    )


def _dilog_unit(z: Array, /) -> tuple[Array, Array]:
    direct_value, direct_derivative = _dilog_series(z)
    continued_value, continued_derivative = _dilog_jonquiere(z)
    direct = jnp.abs(z) <= 0.5
    return (
        jnp.where(direct, direct_value, continued_value),
        jnp.where(direct, direct_derivative, continued_derivative),
    )


def _dilog_value_derivative(z: Array, /) -> tuple[Array, Array]:
    safe_z = jnp.where(jnp.isfinite(z), z, jnp.ones_like(z))
    reciprocal = 1.0 / safe_z
    inside_value, inside_derivative = _dilog_unit(safe_z)
    reciprocal_value, reciprocal_derivative = _dilog_unit(reciprocal)
    logarithm = principal_log(-safe_z)
    outside_value = (
        -reciprocal_value
        - jnp.asarray(math.pi**2 / 6.0, dtype=z.dtype)
        - 0.5 * logarithm * logarithm
    )
    outside_derivative = reciprocal_derivative / (safe_z * safe_z) - logarithm / safe_z
    outside = jnp.abs(safe_z) > 1.0
    value = jnp.where(outside, outside_value, inside_value)
    derivative = jnp.where(outside, outside_derivative, inside_derivative)
    logarithm_two = jnp.asarray(math.log(2.0), dtype=z.dtype)
    pi_squared = jnp.asarray(math.pi**2, dtype=z.dtype)
    at_zero = safe_z == 0.0
    at_one = safe_z == 1.0
    at_minus_one = safe_z == -1.0
    at_half = safe_z == 0.5
    value = jnp.where(at_zero, jnp.zeros_like(value), value)
    value = jnp.where(at_one, pi_squared / 6.0, value)
    value = jnp.where(at_minus_one, -pi_squared / 12.0, value)
    value = jnp.where(at_half, pi_squared / 12.0 - 0.5 * logarithm_two**2, value)
    derivative = jnp.where(at_zero, jnp.ones_like(derivative), derivative)
    derivative = jnp.where(
        at_one,
        jnp.asarray(jnp.inf, dtype=jnp.real(z).dtype).astype(z.dtype),
        derivative,
    )
    derivative = jnp.where(at_minus_one, logarithm_two, derivative)
    derivative = jnp.where(at_half, 2.0 * logarithm_two, derivative)
    valid = jnp.isfinite(jnp.real(z)) & jnp.isfinite(jnp.imag(z))
    invalid = jnp.full_like(z, jnp.nan)
    return jnp.where(valid, value, invalid), jnp.where(valid, derivative, invalid)


@jax.custom_jvp
def _dilog_array(z: Array, /) -> Array:
    return _dilog_value_derivative(z)[0]


@_dilog_array.defjvp
def _dilog_jvp(primals, tangents):
    (z,) = primals
    (z_tangent,) = tangents
    value, derivative = _dilog_value_derivative(z)
    return value, z_tangent * derivative


def dilog(z: ArrayLike, /) -> Array:
    """Evaluate the principal complex dilogarithm ``Li_2(z)``."""
    (argument,) = promote_principal(z)
    return _dilog_array(argument)


def spence(z: ArrayLike, /) -> Array:
    """Evaluate Spence's function in the convention ``Li_2(1-z)``."""
    (argument,) = promote_principal(z)
    return _dilog_array(1.0 - argument)


__all__ = ["dilog", "spence"]
