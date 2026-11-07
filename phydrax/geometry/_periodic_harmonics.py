#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exactly reduced periodic harmonics ``sin/cos(2 pi n u)`` and their jets."""

from __future__ import annotations

import functools
import math

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.custom_derivatives import SymbolicZero

from ._interval_enclosure import (
    Interval,
    interval_periodic_harmonic,
    IntervalExtensionError,
    register_custom_jvp_interval_extension,
)


def _reduced_turn(coordinate: Array, frequency: Array, quarter_turns: int, /) -> Array:
    """Return ``s in [0, 1)`` with ``s == n u + k / 4 (mod 1)``.

    ``u`` is reduced by its own floor before it is scaled, so integer-shifted
    coordinates ``u`` and ``u + m`` that are both binary64 values reduce to the
    same bits, and every later operation depends only on those bits. Periodic
    orbit consistency therefore never relies on ``sin(2 pi m)`` rounding to 0.
    """
    reduced = coordinate - jnp.floor(coordinate)
    phase = frequency * reduced
    phase = phase - jnp.floor(phase) + 0.25 * quarter_turns
    return phase - jnp.floor(phase)


@functools.cache
def _harmonic_jet(order: int, /) -> jax.custom_jvp[Array]:
    @jax.custom_jvp
    def jet(coordinate: Array, frequency: Array) -> Array:
        angle = (2.0 * math.pi) * _reduced_turn(coordinate, frequency, order)
        scale = (2.0 * math.pi * frequency) ** order
        return jnp.stack((scale * jnp.sin(angle), scale * jnp.cos(angle)), axis=-1)

    def rule(
        primals: tuple[Array, Array], tangents: tuple[Array, Array]
    ) -> tuple[Array, Array]:
        coordinate, frequency = primals
        coordinate_tangent, frequency_tangent = tangents
        if not isinstance(frequency_tangent, SymbolicZero):
            raise ValueError(
                "Periodic harmonic frequencies are integer table data, not differentiable."
            )
        values = jet(coordinate, frequency)
        if isinstance(coordinate_tangent, SymbolicZero):
            coordinate_tangent = jnp.zeros_like(coordinate)
        return values, _harmonic_jet(order + 1)(coordinate, frequency) * (
            coordinate_tangent[..., None]
        )

    def extension(operands: tuple[Interval, ...]) -> Interval:
        coordinate, frequency = operands
        if not np.array_equal(frequency[0], frequency[1]):
            raise IntervalExtensionError(
                "periodic harmonic frequencies must be exact table constants."
            )
        return interval_periodic_harmonic(coordinate, frequency[0], order)

    jet.defjvp(rule, symbolic_zeros=True)
    register_custom_jvp_interval_extension(rule, extension)
    return jet


def periodic_harmonic_jet(coordinate: Array, frequency: Array, /, *, order: int) -> Array:
    """Return ``d^k/du^k (sin(2 pi n u), cos(2 pi n u))`` stacked on a new last axis.

    ``frequency`` holds nonnegative integer values broadcast against
    ``coordinate``. Each derivative is the next jet, ``(2 pi n)^k`` times the
    harmonic advanced by ``k`` quarter turns. Every jet equation declares the
    interval extension of the real harmonic, so interval programs enclose
    ``sin/cos(2 pi n u)`` itself rather than its reduced binary64 primal.
    """
    if isinstance(order, bool) or not isinstance(order, int) or order < 0:
        raise ValueError("Periodic harmonic jet order must be a nonnegative integer.")
    return _harmonic_jet(order)(coordinate, frequency)


__all__ = ["periodic_harmonic_jet"]
