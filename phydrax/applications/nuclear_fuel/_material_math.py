#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Pointwise operations shared by the fuel and cladding material laws.

Fluence, flux, burnup and stress start at exactly zero. Fractional powers and
the von Mises square root have an unbounded derivative there, and forward-mode
products ``inf * 0`` would turn every tangent into NaN, even along zero
directions. Each zero-safe helper substitutes a safe argument only where the
input is exactly zero, inside the branch that is not selected: the value at
zero is the exact limit and its derivative is zero. NaN and negative inputs
still propagate as NaN.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def cubic_polynomial(
    t: Array, constant: Array, linear: Array, quadratic: Array, cubic: Array, /
) -> Array:
    """Horner form of ``constant + linear t + quadratic t^2 + cubic t^3``."""
    return constant + t * (linear + t * (quadratic + t * cubic))


def zero_safe_power(base: Array, exponent: float | Array, /) -> Array:
    """``base**exponent`` for ``base >= 0``, with a finite derivative at zero."""
    zero = base == 0.0
    return jnp.where(zero, 0.0, jnp.where(zero, 1.0, base) ** exponent)


def zero_safe_log_power(base: Array, exponent: float | Array, /) -> Array:
    """``exp(exponent * log(base))`` for ``base >= 0``, with finite derivatives at zero.

    The UO2 correlations are evaluated in this exponential form, which can
    differ from ``base**exponent`` in the last bit.
    """
    zero = base == 0.0
    return jnp.where(zero, 0.0, jnp.exp(exponent * jnp.log(jnp.where(zero, 1.0, base))))


def von_mises_equivalent(deviatoric: Array, /) -> Array:
    """``sqrt(3/2 s:s)`` over the trailing ``(3, 3)`` axes of a deviatoric tensor."""
    squared = 1.5 * jnp.sum(deviatoric**2, axis=(-2, -1))
    zero = squared == 0.0
    return jnp.where(zero, 0.0, jnp.sqrt(jnp.where(zero, 1.0, squared)))


__all__ = [
    "cubic_polynomial",
    "von_mises_equivalent",
    "zero_safe_log_power",
    "zero_safe_power",
]
