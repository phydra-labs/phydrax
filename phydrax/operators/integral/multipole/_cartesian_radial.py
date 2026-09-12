#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from itertools import product
from math import factorial

import jax
import jax.numpy as jnp


def multi_index_factorial(exponent: tuple[int, int, int]) -> int:
    """Return the product of component factorials."""
    return factorial(exponent[0]) * factorial(exponent[1]) * factorial(exponent[2])


def multi_binomial(upper: tuple[int, int, int], lower: tuple[int, int, int]) -> int:
    """Return the Cartesian multi-index binomial coefficient."""
    coefficient = 1
    for axis in range(3):
        coefficient *= factorial(upper[axis]) // (
            factorial(lower[axis]) * factorial(upper[axis] - lower[axis])
        )
    return coefficient


def monomial(value: jax.Array, exponent: tuple[int, int, int]) -> jax.Array:
    """Evaluate one three-dimensional Cartesian monomial."""
    result = jnp.asarray(1.0, dtype=value.dtype)
    for axis in range(3):
        result = result * value[axis] ** exponent[axis]
    return result


def power_two_enclosing_scale(half_width: jax.Array) -> jax.Array:
    """Return a finite power-of-two radius enclosing one axis-aligned box."""
    width = jnp.asarray(half_width)
    radius = jnp.sqrt(jnp.sum(width * width))
    tiny = jnp.asarray(jnp.finfo(width.dtype).tiny, dtype=width.dtype)
    safe_radius = jnp.maximum(radius, tiny)
    exponent = jnp.ceil(jnp.log2(safe_radius))
    return jnp.exp2(exponent)


def plummer_scaled_cartesian_derivatives(
    exponents: tuple[tuple[int, int, int], ...],
    displacement: jax.Array,
    softening: float,
    gravitational_constant: float,
    common_scale: jax.Array,
) -> jax.Array:
    """Cartesian derivatives of ``-G/sqrt(r²+eps²)`` in scaled coordinates.

    The returned derivative for multi-index ``alpha`` is taken with respect to
    ``u = displacement/common_scale``. Physical derivatives therefore gain a
    factor ``common_scale**(-|alpha|-1)``.
    """
    relative = jnp.asarray(displacement)
    scale = jnp.asarray(common_scale, dtype=relative.dtype)
    scaled = relative / scale
    scaled_softening = jnp.asarray(softening, dtype=relative.dtype) / scale
    radius_squared = jnp.sum(scaled * scaled) + scaled_softening**2
    maximum_order = max(sum(exponent) for exponent in exponents)
    radial = [
        -jnp.asarray(gravitational_constant, dtype=relative.dtype)
        / jnp.sqrt(radius_squared)
    ]
    for order in range(1, maximum_order + 1):
        radial.append(radial[-1] * (-(2 * order - 1) / 2) / radius_squared)

    derivatives = []
    for exponent in exponents:
        value = jnp.asarray(0.0, dtype=relative.dtype)
        for paired in product(*(range(component // 2 + 1) for component in exponent)):
            remaining = tuple(exponent[axis] - 2 * paired[axis] for axis in range(3))
            radial_order = sum(exponent) - sum(paired)
            coefficient = 1
            for axis in range(3):
                coefficient *= factorial(exponent[axis]) // (
                    factorial(paired[axis]) * factorial(remaining[axis])
                )
            value = (
                value
                + coefficient * monomial(2 * scaled, remaining) * radial[radial_order]
            )
        derivatives.append(value)
    return jnp.stack(derivatives)


__all__ = [
    "monomial",
    "multi_binomial",
    "multi_index_factorial",
    "plummer_scaled_cartesian_derivatives",
    "power_two_enclosing_scale",
]
