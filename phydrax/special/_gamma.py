#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Stable real gamma-function combinations used by special functions."""

from __future__ import annotations

import jax.numpy as jnp
import jax.scipy.special as jsp
from jax import Array


_EULER_GAMMA = 0.577215664901532860606512090082402431


def log_gamma_one_plus_minus_difference(x: Array) -> Array:
    """Return ``log Γ(1+x) - log Γ(1-x)`` without cancellation near zero."""
    x_squared = x * x
    series = _EULER_GAMMA + x_squared * (
        1.2020569031595942854 / 3.0
        + x_squared
        * (
            1.0369277551433699263 / 5.0
            + x_squared
            * (
                1.0083492773819228268 / 7.0
                + x_squared
                * (
                    1.0020083928260822144 / 9.0
                    + x_squared
                    * (
                        1.0004941886041194646 / 11.0
                        + x_squared * 1.0001227133475784891 / 13.0
                    )
                )
            )
        )
    )
    stable = -2.0 * x * series
    direct = jsp.gammaln(1.0 + x) - jsp.gammaln(1.0 - x)
    return jnp.where(jnp.abs(x) < 0.1, stable, direct)


__all__ = ["log_gamma_one_plus_minus_difference"]
