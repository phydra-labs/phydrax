#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ._axis import _positive_int


def _floating_dtype(dtype: jnp.dtype | None, /) -> jnp.dtype:
    resolved = jnp.asarray(0.0).dtype if dtype is None else jnp.dtype(dtype)
    if not jnp.issubdtype(resolved, jnp.floating):
        raise TypeError("Signal windows require a real floating dtype.")
    return resolved


def _window_coordinate(
    length: int,
    periodic: bool,
    dtype: jnp.dtype,
    /,
) -> Array:
    size = _positive_int(length, "length")
    if not isinstance(periodic, bool):
        raise TypeError("periodic must be a bool.")
    if size == 1:
        return jnp.zeros((1,), dtype=dtype)
    denominator = size if periodic else size - 1
    return jnp.arange(size, dtype=dtype) / jnp.asarray(denominator, dtype=dtype)


def hann_window(
    length: int,
    /,
    *,
    periodic: bool = True,
    dtype: jnp.dtype | None = None,
) -> Array:
    """Return a periodic or symmetric Hann window."""
    resolved_dtype = _floating_dtype(dtype)
    coordinate = _window_coordinate(length, periodic, resolved_dtype)
    if length == 1:
        return jnp.ones((1,), dtype=resolved_dtype)
    return 0.5 - 0.5 * jnp.cos(2.0 * jnp.pi * coordinate)


def hamming_window(
    length: int,
    /,
    *,
    periodic: bool = True,
    dtype: jnp.dtype | None = None,
) -> Array:
    """Return a periodic or symmetric Hamming window."""
    resolved_dtype = _floating_dtype(dtype)
    coordinate = _window_coordinate(length, periodic, resolved_dtype)
    if length == 1:
        return jnp.ones((1,), dtype=resolved_dtype)
    return 0.54 - 0.46 * jnp.cos(2.0 * jnp.pi * coordinate)


def blackman_window(
    length: int,
    /,
    *,
    periodic: bool = True,
    dtype: jnp.dtype | None = None,
) -> Array:
    """Return a periodic or symmetric Blackman window."""
    resolved_dtype = _floating_dtype(dtype)
    coordinate = _window_coordinate(length, periodic, resolved_dtype)
    if length == 1:
        return jnp.ones((1,), dtype=resolved_dtype)
    angle = 2.0 * jnp.pi * coordinate
    return 0.42 - 0.5 * jnp.cos(angle) + 0.08 * jnp.cos(2.0 * angle)


def kaiser_window(
    length: int,
    beta: ArrayLike,
    /,
    *,
    periodic: bool = True,
    dtype: jnp.dtype | None = None,
) -> Array:
    """Return a periodic or symmetric differentiable Kaiser window."""
    raw_beta = jnp.asarray(beta)
    if raw_beta.ndim != 0 or jnp.issubdtype(raw_beta.dtype, jnp.complexfloating):
        raise TypeError("beta must be a real scalar.")
    resolved_dtype = _floating_dtype(dtype)
    beta_value = jnp.asarray(raw_beta, dtype=resolved_dtype)
    beta_value = eqx.error_if(
        beta_value,
        ~jnp.isfinite(beta_value) | (beta_value < 0.0),
        "beta must be finite and nonnegative.",
    )
    coordinate = _window_coordinate(length, periodic, resolved_dtype)
    if length == 1:
        return jnp.ones((1,), dtype=resolved_dtype)
    radial = 2.0 * coordinate - 1.0
    argument = beta_value * jnp.sqrt(jnp.maximum(0.0, 1.0 - radial * radial))
    return jnp.i0(argument) / jnp.i0(beta_value)


def tukey_window(
    length: int,
    alpha: ArrayLike = 0.5,
    /,
    *,
    periodic: bool = True,
    dtype: jnp.dtype | None = None,
) -> Array:
    """Return a periodic or symmetric differentiable Tukey window."""
    raw_alpha = jnp.asarray(alpha)
    if raw_alpha.ndim != 0 or jnp.issubdtype(raw_alpha.dtype, jnp.complexfloating):
        raise TypeError("alpha must be a real scalar.")
    resolved_dtype = _floating_dtype(dtype)
    alpha_value = jnp.asarray(raw_alpha, dtype=resolved_dtype)
    alpha_value = eqx.error_if(
        alpha_value,
        ~jnp.isfinite(alpha_value),
        "alpha must be finite.",
    )
    coordinate = _window_coordinate(length, periodic, resolved_dtype)
    if length == 1:
        return jnp.ones((1,), dtype=resolved_dtype)
    safe_alpha = jnp.where(alpha_value == 0.0, 1.0, alpha_value)
    rising = 0.5 * (1.0 + jnp.cos(jnp.pi * (2.0 * coordinate / safe_alpha - 1.0)))
    falling = 0.5 * (
        1.0 + jnp.cos(jnp.pi * (2.0 * coordinate / safe_alpha - 2.0 / safe_alpha + 1.0))
    )
    tapered = jnp.where(
        coordinate < alpha_value / 2.0,
        rising,
        jnp.where(coordinate > 1.0 - alpha_value / 2.0, falling, 1.0),
    )
    return jnp.where(
        alpha_value <= 0.0,
        jnp.ones_like(tapered),
        jnp.where(
            alpha_value >= 1.0,
            hann_window(length, periodic=periodic, dtype=resolved_dtype),
            tapered,
        ),
    )


__all__ = [
    "blackman_window",
    "hamming_window",
    "hann_window",
    "kaiser_window",
    "tukey_window",
]
