#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Synchrotron kernel functions on the nonnegative real axis."""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from . import _synchrotron_data as _data
from ._dtype import promote_real


class _Table(NamedTuple):
    """Generated coefficients for one function ``x**(p/3) * phi(x)``."""

    power_thirds: int
    series: tuple[tuple[float, ...], ...]
    panels: np.ndarray
    asymptotic: tuple[float, ...]


_F = _Table(
    _data.F_POWER_THIRDS,
    _data.F_SERIES,
    np.asarray(_data.F_PANELS, dtype=np.float64),
    _data.F_ASYMPTOTIC,
)
_G = _Table(
    _data.G_POWER_THIRDS,
    _data.G_SERIES,
    np.asarray(_data.G_PANELS, dtype=np.float64),
    _data.G_ASYMPTOTIC,
)
_H = _Table(
    _data.H_POWER_THIRDS,
    _data.H_SERIES,
    np.asarray(_data.H_PANELS, dtype=np.float64),
    _data.H_ASYMPTOTIC,
)


def _horner(coefficients: tuple[float, ...], argument: Array) -> Array:
    value = jnp.full_like(argument, coefficients[-1])
    for coefficient in reversed(coefficients[:-1]):
        value = value * argument + coefficient
    return value


def _clenshaw(coefficients: Array, argument: Array) -> Array:
    """Chebyshev series with per-lane coefficients on the trailing axis."""
    upper = jnp.zeros_like(argument)
    lower = jnp.zeros_like(argument)
    twice = 2.0 * argument
    for degree in range(coefficients.shape[-1] - 1, 0, -1):
        upper, lower = twice * upper - lower + coefficients[..., degree], upper
    return argument * upper - lower + coefficients[..., 0]


def _evaluate(x: Array, table: _Table) -> Array:
    """Select series, log-x panel, or asymptotic representation per lane."""
    interior = (x > 0.0) & jnp.isfinite(x)
    safe = jnp.where(interior, x, jnp.ones_like(x))
    root = jnp.cbrt(safe)
    scale = root**table.power_thirds
    root_square = root * root

    square = safe * safe
    series = jnp.zeros_like(safe)
    for polynomial in reversed(table.series):
        series = series * root_square + _horner(polynomial, square)
    series = scale * series

    position = (jnp.log(safe) - _data.PANEL_LOG_LOWER) / _data.PANEL_LOG_WIDTH
    index = jnp.clip(jnp.floor(position), 0.0, _data.PANEL_COUNT - 1.0)
    local = 2.0 * (position - index) - 1.0
    coefficients = jnp.asarray(table.panels)[index.astype(jnp.int32)]
    panel = scale * jnp.exp(-safe) * _clenshaw(coefficients, local)

    asymptotic = (
        jnp.sqrt(0.5 * math.pi * safe)
        * jnp.exp(-safe)
        * _horner(table.asymptotic, 1.0 / safe)
    )

    value = jnp.where(
        safe < _data.SERIES_UPPER,
        series,
        jnp.where(safe < _data.ASYMPTOTIC_LOWER, panel, asymptotic),
    )
    boundary = jnp.where(jnp.isnan(x) | (x < 0.0), jnp.nan, 0.0)
    return jnp.where(interior, value, boundary)


@jax.custom_jvp
def _synchrotron_f_array(x: Array) -> Array:
    return _evaluate(x, _F)


@jax.custom_jvp
def _synchrotron_g_array(x: Array) -> Array:
    return _evaluate(x, _G)


@jax.custom_jvp
def _synchrotron_h_array(x: Array) -> Array:
    return _evaluate(x, _H)


def _origin_slope(x: Array, slope: Array) -> Array:
    # Every function behaves as a positive multiple of x**(1/3) or x**(2/3).
    return jnp.where(x == 0.0, jnp.inf, slope)


# The closed system follows from K_{5/3} = K_{1/3} + 4 K_{2/3} / (3x) and
# K_nu' = -K_{nu-1} - nu K_nu / x with K_{-nu} = K_nu:
#   F' = F/x - x K_{5/3} = (F - 4G/3)/x - H,
#   G' = G/(3x) - H,
#   H' = 2H/(3x) - G.
@_synchrotron_f_array.defjvp
def _synchrotron_f_jvp(
    primals: tuple[Array], tangents: tuple[Array]
) -> tuple[Array, Array]:
    (x,) = primals
    (dx,) = tangents
    f = _synchrotron_f_array(x)
    g = _synchrotron_g_array(x)
    h = _synchrotron_h_array(x)
    slope = _origin_slope(x, (f - (4.0 / 3.0) * g) / x - h)
    return f, slope * dx


@_synchrotron_g_array.defjvp
def _synchrotron_g_jvp(
    primals: tuple[Array], tangents: tuple[Array]
) -> tuple[Array, Array]:
    (x,) = primals
    (dx,) = tangents
    g = _synchrotron_g_array(x)
    h = _synchrotron_h_array(x)
    slope = _origin_slope(x, g / (3.0 * x) - h)
    return g, slope * dx


@_synchrotron_h_array.defjvp
def _synchrotron_h_jvp(
    primals: tuple[Array], tangents: tuple[Array]
) -> tuple[Array, Array]:
    (x,) = primals
    (dx,) = tangents
    g = _synchrotron_g_array(x)
    h = _synchrotron_h_array(x)
    slope = _origin_slope(x, (2.0 / 3.0) * h / x - g)
    return h, slope * dx


def _prepare(name: str, x: ArrayLike) -> Array:
    (value,) = promote_real(name, x)
    if value.dtype != jnp.float64:
        raise TypeError(f"{name} requires float64 arguments; received {value.dtype}")
    return value


def synchrotron_f(x: ArrayLike) -> Array:
    """Synchrotron emissivity kernel ``F(x) = x * integral(K_{5/3}(t), t=x..inf)``.

    ``x`` is the photon frequency in units of the critical frequency. The
    float64 result has uniform relative error at most ``3.6e-15`` for positive
    normal results. ``F(0) = 0`` with an infinite right derivative,
    ``F(inf) = 0``, and negative or NaN arguments return NaN. The argument
    derivative ``F/x - x K_{5/3}(x)`` is analytic.
    """
    return _synchrotron_f_array(_prepare("synchrotron_f", x))


def synchrotron_g(x: ArrayLike) -> Array:
    """Polarized synchrotron kernel ``G(x) = x * K_{2/3}(x)``.

    The emissivity kernels polarized perpendicular and parallel to the
    projected magnetic field are ``F + G`` and ``F - G``. The float64 result
    has uniform relative error at most
    ``3.6e-15`` for positive normal results. ``G(0) = 0`` with an infinite
    right derivative, ``G(inf) = 0``, and negative or NaN arguments return NaN.
    The argument derivative ``G/(3x) - x K_{1/3}(x)`` is analytic.
    """
    return _synchrotron_g_array(_prepare("synchrotron_g", x))


def synchrotron_h(x: ArrayLike) -> Array:
    """Spin-dependent synchrotron kernel ``H(x) = x * K_{1/3}(x)``.

    The spin-odd part of the quantum synchrotron spectrum and the spin-flip
    channels carry ``K_{1/3}``. The float64 result has uniform relative error
    at most ``3.6e-15`` for positive normal results. ``H(0) = 0`` with an
    infinite right derivative, ``H(inf) = 0``, and negative or NaN arguments
    return NaN. The argument derivative ``2H/(3x) - x K_{2/3}(x)`` is analytic.
    """
    return _synchrotron_h_array(_prepare("synchrotron_h", x))


__all__ = ["synchrotron_f", "synchrotron_g", "synchrotron_h"]
