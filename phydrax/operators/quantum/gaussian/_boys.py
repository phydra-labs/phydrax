#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Stable Boys-function ladders with exact first-order differentiation."""

from __future__ import annotations

from functools import partial
from math import factorial

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.scipy.special import gammainc, gammaln
from jaxtyping import Array, ArrayLike


def _boys_values_impl(maximum: int, value: Array, /) -> Array:
    real_dtype = value.dtype
    orders = jnp.arange(maximum + 1, dtype=real_dtype)
    expanded = value[..., None]
    series = jnp.zeros(value.shape + (maximum + 1,), dtype=real_dtype)
    power = jnp.ones_like(expanded)
    for term_index in range(18):
        denominator = factorial(term_index) * (2.0 * orders + 2.0 * term_index + 1.0)
        series = series + power / denominator
        power = power * (-expanded)
    tiny = jnp.finfo(real_dtype).tiny
    safe = jnp.maximum(expanded, tiny)
    shape = (1,) * value.ndim + (maximum + 1,)
    alpha = (orders + 0.5).reshape(shape)
    prefactor = 0.5 * jnp.exp(gammaln(alpha) - alpha * jnp.log(safe))
    regular = prefactor * gammainc(alpha, safe)
    asymptotic = prefactor
    return jnp.where(
        expanded < 1.0e-7,
        series,
        jnp.where(expanded > 40.0 + 2.0 * orders, asymptotic, regular),
    )


@partial(jax.custom_jvp, nondiff_argnums=(0,))
def boys_values(maximum_order: int, argument: ArrayLike, /) -> Array:
    """Return F₀ through Fₘ on the final axis for non-negative arguments."""

    maximum = int(maximum_order)
    if maximum < 0 or maximum > 64:
        raise ValueError("maximum_order must lie in [0, 64].")
    value = jnp.asarray(argument)
    value = eqx.error_if(
        value,
        jnp.any(value < 0.0),
        "Boys-function arguments must be non-negative.",
    )
    return _boys_values_impl(maximum, value)


@boys_values.defjvp
def _boys_values_jvp(maximum_order: int, primals, tangents):
    (argument,), (argument_tangent,) = primals, tangents
    extended = _boys_values_impl(maximum_order + 1, jnp.asarray(argument))
    values = extended[..., :-1]
    derivative = -extended[..., 1:]
    return values, derivative * jnp.asarray(argument_tangent)[..., None]


def boys0(argument: ArrayLike, /) -> Array:
    return boys_values(0, argument)[..., 0]


__all__ = ["boys0", "boys_values"]
