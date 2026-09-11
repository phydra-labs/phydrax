#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Integer-degree Gegenbauer polynomials and parameter derivatives."""

from __future__ import annotations

from numbers import Integral

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike


def _nonnegative_degree(value: int, name: str, /) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    degree = int(value)
    if degree < 0:
        raise ValueError(f"{name} must be nonnegative.")
    return degree


def _promote_arguments(alpha: ArrayLike, x: ArrayLike, /) -> tuple[Array, Array]:
    alpha_array = jnp.asarray(alpha)
    x_array = jnp.asarray(x)
    if jnp.issubdtype(alpha_array.dtype, jnp.complexfloating):
        raise TypeError("gegenbauer functions require real alpha")

    dtype = jnp.result_type(alpha_array, x_array)
    if not jnp.issubdtype(dtype, jnp.inexact):
        dtype = jnp.asarray(0.0).dtype
    elif dtype == jnp.float16 or dtype == jnp.bfloat16:
        dtype = jnp.float32
    real_dtype = jnp.asarray(0.0, dtype=dtype).real.dtype
    promoted_alpha = jnp.asarray(alpha_array, dtype=real_dtype)
    promoted_x = jnp.asarray(x_array, dtype=dtype)
    return jnp.broadcast_arrays(promoted_alpha, promoted_x)


def _valid_alpha(alpha: Array, /) -> Array:
    return jnp.isfinite(alpha) & (alpha > -0.5)


def _mask_domain(value: Array, alpha: Array, /) -> Array:
    return jnp.where(_valid_alpha(alpha), value, jnp.full_like(value, jnp.nan))


def _gegenbauer_value(degree: int, alpha: Array, x: Array, /) -> Array:
    previous = jnp.ones_like(x)
    if degree == 0:
        return previous

    current = 2.0 * alpha * x
    for index in range(1, degree):
        following = (
            2.0 * (index + alpha) * x * current - (index + 2.0 * alpha - 1.0) * previous
        ) / (index + 1.0)
        previous, current = current, following
    return current


def _gegenbauer_value_and_alpha_derivative(
    degree: int, alpha: Array, x: Array, /
) -> tuple[Array, Array]:
    previous = jnp.ones_like(x)
    previous_derivative = jnp.zeros_like(x)
    if degree == 0:
        return previous, previous_derivative

    current = 2.0 * alpha * x
    current_derivative = 2.0 * x
    for index in range(1, degree):
        denominator = index + 1.0
        following = (
            2.0 * (index + alpha) * x * current - (index + 2.0 * alpha - 1.0) * previous
        ) / denominator
        following_derivative = (
            2.0 * x * current
            + 2.0 * (index + alpha) * x * current_derivative
            - 2.0 * previous
            - (index + 2.0 * alpha - 1.0) * previous_derivative
        ) / denominator
        previous, current = current, following
        previous_derivative, current_derivative = (
            current_derivative,
            following_derivative,
        )
    return current, current_derivative


def gegenbauer_c(n: int, alpha: ArrayLike, x: ArrayLike, /) -> Array:
    """Evaluate the standard integer-degree Gegenbauer polynomial.

    ``n`` is static and nonnegative. ``alpha`` is a real numerical argument with
    domain ``alpha > -1/2`` and broadcasts with real or complex ``x``. At
    ``alpha == 0`` the standard generating-function convention is retained:
    ``C_0^0 = 1`` and ``C_n^0 = 0`` for every positive ``n``.
    """
    degree = _nonnegative_degree(n, "n")
    alpha_array, x_array = _promote_arguments(alpha, x)
    return _mask_domain(_gegenbauer_value(degree, alpha_array, x_array), alpha_array)


def gegenbauer_vander(alpha: ArrayLike, x: ArrayLike, degree: int, /) -> Array:
    """Evaluate all standard Gegenbauer modes through ``degree`` modes-last."""
    degree_ = _nonnegative_degree(degree, "degree")
    alpha_array, x_array = _promote_arguments(alpha, x)

    values = [jnp.ones_like(x_array)]
    if degree_ >= 1:
        values.append(2.0 * alpha_array * x_array)
    for index in range(1, degree_):
        values.append(
            (
                2.0 * (index + alpha_array) * x_array * values[-1]
                - (index + 2.0 * alpha_array - 1.0) * values[-2]
            )
            / (index + 1.0)
        )
    result = jnp.stack(values, axis=-1)
    return _mask_domain(result, alpha_array[..., None])


def gegenbauer_alpha_derivative(n: int, alpha: ArrayLike, x: ArrayLike, /) -> Array:
    """Differentiate ``C_n^alpha(x)`` with respect to ``alpha``.

    The value and its parameter derivative are advanced in one paired
    recurrence. In particular, the derivative at the exact standard-family
    collapse ``alpha == 0`` remains nonzero for positive degree.
    """
    degree = _nonnegative_degree(n, "n")
    alpha_array, x_array = _promote_arguments(alpha, x)
    _, derivative = _gegenbauer_value_and_alpha_derivative(degree, alpha_array, x_array)
    return _mask_domain(derivative, alpha_array)


__all__ = [
    "gegenbauer_alpha_derivative",
    "gegenbauer_c",
    "gegenbauer_vander",
]
