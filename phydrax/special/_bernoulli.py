#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact host-side Bernoulli coefficients for analytic continuations."""

from __future__ import annotations

from fractions import Fraction

import jax.numpy as jnp
from jax import Array


# B_0, B_2, ..., B_40.  Keeping the source coefficients rational avoids
# baking binary64 rounding into float32 or extended-precision call sites.
EVEN_BERNOULLI_NUMBERS = (
    Fraction(1),
    Fraction(1, 6),
    Fraction(-1, 30),
    Fraction(1, 42),
    Fraction(-1, 30),
    Fraction(5, 66),
    Fraction(-691, 2730),
    Fraction(7, 6),
    Fraction(-3617, 510),
    Fraction(43867, 798),
    Fraction(-174611, 330),
    Fraction(854513, 138),
    Fraction(-236364091, 2730),
    Fraction(8553103, 6),
    Fraction(-23749461029, 870),
    Fraction(8615841276005, 14322),
    Fraction(-7709321041217, 510),
    Fraction(2577687858367, 6),
    Fraction(-26315271553053477373, 1919190),
    Fraction(2929993913841559, 6),
    Fraction(-261082718496449122051, 13530),
)


def even_bernoulli_coefficient(index: int, dtype, /) -> Array:
    """Return exact-host ``B_(2*index)/(2*index)!`` in ``dtype``."""
    value = EVEN_BERNOULLI_NUMBERS[index]
    factorial = 1
    for factor in range(2, 2 * index + 1):
        factorial *= factor
    return jnp.asarray(value.numerator / (value.denominator * factorial), dtype=dtype)
