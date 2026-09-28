#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact reference solutions used by bubble-dynamics qualification campaigns."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from jax.scipy.special import betainc, gammaln
from jax.typing import ArrayLike


def rayleigh_collapse_time(
    initial_radius: ArrayLike,
    pressure_difference: ArrayLike,
    density: ArrayLike,
    /,
    *,
    final_radius_ratio: ArrayLike = 0.0,
) -> Array:
    """Time for an empty cavity to shrink from `R₀` to `ε R₀` under a pressure step.

    Integrating `R³Ṙ² = (2Δp/3ρ)(R₀³ − R³)` gives
    `t(ε) = (R₀/U)(1/3) B(5/6, 1/2)[1 − I_{ε³}(5/6, 1/2)]` with
    `U = √(2Δp/(3ρ))`; `t(0) = 0.914681 R₀ √(ρ/Δp)` is Rayleigh's collapse time.
    """
    radius = jnp.asarray(initial_radius, dtype=jnp.float64)
    velocity = jnp.sqrt(2.0 * jnp.asarray(pressure_difference) / (3.0 * jnp.asarray(density)))
    a, b = 5.0 / 6.0, 0.5
    complete = jnp.exp(gammaln(a) + gammaln(b) - gammaln(a + b))
    ratio = jnp.asarray(final_radius_ratio, dtype=jnp.float64)
    remaining = 1.0 - betainc(a, b, ratio**3)
    return radius / velocity * complete * remaining / 3.0


def quasi_static_dissolution_radius(
    time: ArrayLike, initial_radius: ArrayLike, lifetime: ArrayLike, /
) -> Array:
    """Quasi-static Epstein–Plesset radius without surface tension, `R₀ √(1 − t/τ)`."""
    fraction = 1.0 - jnp.asarray(time, dtype=jnp.float64) / jnp.asarray(lifetime)
    return jnp.asarray(initial_radius) * jnp.sqrt(jnp.clip(fraction, 0.0, None))


__all__ = ["quasi_static_dissolution_radius", "rayleigh_collapse_time"]
