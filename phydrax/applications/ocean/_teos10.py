#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._hydrostatic import HydrostaticEOSResult


_GSW_SFAC = 0.0248826675584615
_GSW_OFFSET = 5.971840214030754e-1

# Official GSW 75-term specific-volume polynomial coefficients. GSW-C names
# v_ijk with temperature, salinity, and pressure orders respectively.
_GSW75_TERMS = (
    (0, 0, 0, 1.0769995862e-3),
    (0, 0, 1, -6.0799143809e-5),
    (0, 0, 2, 9.9856169219e-6),
    (0, 0, 3, -1.1309361437e-6),
    (0, 0, 4, 1.0531153080e-7),
    (0, 0, 5, -1.2647261286e-8),
    (0, 0, 6, 1.9613503930e-9),
    (0, 1, 0, -3.1038981976e-4),
    (0, 1, 1, 2.4262468747e-5),
    (0, 1, 2, -5.8484432984e-7),
    (0, 1, 3, 3.6310188515e-7),
    (0, 1, 4, -1.1147125423e-7),
    (0, 2, 0, 6.6928067038e-4),
    (0, 2, 1, -3.4792460974e-5),
    (0, 2, 2, -4.8122251597e-6),
    (0, 2, 3, 1.6746303780e-8),
    (0, 3, 0, -8.5047933937e-4),
    (0, 3, 1, 3.7470777305e-5),
    (0, 3, 2, 4.9263106998e-6),
    (0, 4, 0, 5.8086069943e-4),
    (0, 4, 1, -1.7322218612e-5),
    (0, 4, 2, -1.7811974727e-6),
    (0, 5, 0, -2.1092370507e-4),
    (0, 5, 1, 3.0927427253e-6),
    (0, 6, 0, 3.1932457305e-5),
    (1, 0, 0, -1.5649734675e-5),
    (1, 0, 1, 1.8505765429e-5),
    (1, 0, 2, -1.1736386731e-6),
    (1, 0, 3, -3.6527006553e-7),
    (1, 0, 4, 3.1454099902e-7),
    (1, 1, 0, 3.5009599764e-5),
    (1, 1, 1, -9.5677088156e-6),
    (1, 1, 2, -5.5699154557e-6),
    (1, 1, 3, -2.7295696237e-7),
    (1, 2, 0, -4.3592678561e-5),
    (1, 2, 1, 1.1100834765e-5),
    (1, 2, 2, 5.4620748834e-6),
    (1, 3, 0, 3.4532461828e-5),
    (1, 3, 1, -9.8447117844e-6),
    (1, 3, 2, -1.3544185627e-6),
    (1, 4, 0, -1.1959409788e-5),
    (1, 4, 1, 2.5909225260e-6),
    (1, 5, 0, 1.3864594581e-6),
    (2, 0, 0, 2.7762106484e-5),
    (2, 0, 1, -1.1716606853e-5),
    (2, 0, 2, 2.1305028740e-6),
    (2, 0, 3, 2.8695905159e-7),
    (2, 1, 0, -3.7435842344e-5),
    (2, 1, 1, -2.3678308361e-7),
    (2, 1, 2, 3.9137387080e-7),
    (2, 2, 0, 3.5907822760e-5),
    (2, 2, 1, 2.9283346295e-6),
    (2, 2, 2, -6.5731104067e-7),
    (2, 3, 0, -1.8698584187e-5),
    (2, 3, 1, -4.8826139200e-7),
    (2, 4, 0, 3.8595339244e-6),
    (3, 0, 0, -1.6521159259e-5),
    (3, 0, 1, 7.9279656173e-6),
    (3, 0, 2, -4.6132540037e-7),
    (3, 1, 0, 2.4141479483e-5),
    (3, 1, 1, -3.4558773655e-6),
    (3, 1, 2, 7.7618888092e-9),
    (3, 2, 0, -1.4353633048e-5),
    (3, 2, 1, 3.1655306078e-7),
    (3, 3, 0, 2.2863324556e-6),
    (4, 0, 0, 6.9111322702e-6),
    (4, 0, 1, -3.4102187482e-6),
    (4, 0, 2, -6.3352916514e-8),
    (4, 1, 0, -8.7595873154e-6),
    (4, 1, 1, 1.2956717783e-6),
    (4, 2, 0, 4.3703680598e-6),
    (5, 0, 0, -8.0539615540e-7),
    (5, 0, 1, 5.0736766814e-7),
    (5, 1, 0, -3.3052758900e-7),
    (6, 0, 0, 2.0543094268e-7),
)


_GSW_FREEZING_COEFFICIENTS = (
    0.017947064327968736,
    -6.076099099929818,
    4.883198653547851,
    -11.88081601230542,
    13.34658511480257,
    -8.722761043208607,
    2.082038908808201,
    -7.389420998107497,
    -2.110913185058476,
    0.2295491578006229,
    -0.9891538123307282,
    -0.08987150128406496,
    0.3831132432071728,
    1.054318231187074,
    1.065556599652796,
    -0.7997496801694032,
    0.3850133554097069,
    -2.078616693017569,
    0.8756340772729538,
    -2.079022768390933,
    1.596435439942262,
    0.1338002171109174,
    1.242891021876471,
)


def _specific_volume(sa: Array, ct: Array, pressure_dbar: Array, /) -> Array:
    x = jnp.sqrt(_GSW_SFAC * sa + _GSW_OFFSET)
    y = 0.025 * ct
    z = 1.0e-4 * pressure_dbar
    coefficients = {
        (salinity_order, temperature_order, pressure_order): jnp.asarray(
            value, dtype=sa.dtype
        )
        for temperature_order, salinity_order, pressure_order, value in _GSW75_TERMS
    }
    z_polynomial = jnp.zeros_like(sa)
    for k in range(6, -1, -1):
        y_polynomial = jnp.zeros_like(sa)
        for j in range(6, -1, -1):
            x_polynomial = jnp.zeros_like(sa)
            for i in range(6, -1, -1):
                x_polynomial = x * x_polynomial + coefficients.get(
                    (i, j, k), jnp.asarray(0.0, dtype=sa.dtype)
                )
            y_polynomial = y * y_polynomial + x_polynomial
        z_polynomial = z * z_polynomial + y_polynomial
    return z_polynomial


def _conservative_temperature_freezing(sa: Array, pressure_dbar: Array, /) -> Array:
    coefficients = tuple(
        jnp.asarray(value, dtype=sa.dtype) for value in _GSW_FREEZING_COEFFICIENTS
    )
    (
        c0,
        c1,
        c2,
        c3,
        c4,
        c5,
        c6,
        c7,
        c8,
        c9,
        c10,
        c11,
        c12,
        c13,
        c14,
        c15,
        c16,
        c17,
        c18,
        c19,
        c20,
        c21,
        c22,
    ) = coefficients
    salinity_fraction = 1.0e-2 * sa
    root_salinity = jnp.sqrt(jnp.maximum(salinity_fraction, 0.0))
    pressure_fraction = 1.0e-4 * pressure_dbar
    return (
        c0
        + salinity_fraction
        * (
            c1
            + root_salinity
            * (
                c2
                + root_salinity
                * (c3 + root_salinity * (c4 + root_salinity * (c5 + c6 * root_salinity)))
            )
        )
        + pressure_fraction * (c7 + pressure_fraction * (c8 + c9 * pressure_fraction))
        + salinity_fraction
        * pressure_fraction
        * (
            c10
            + pressure_fraction
            * (c12 + pressure_fraction * (c15 + c21 * salinity_fraction))
            + salinity_fraction
            * (c13 + c17 * pressure_fraction + c19 * salinity_fraction)
            + root_salinity
            * (
                c11
                + pressure_fraction * (c14 + c18 * pressure_fraction)
                + salinity_fraction
                * (c16 + c20 * pressure_fraction + c22 * salinity_fraction)
            )
        )
    )


def _in_oceanographic_funnel(sa: Array, ct: Array, pressure_dbar: Array, /) -> Array:
    below_deep = pressure_dbar < 6500.0
    at_depth = pressure_dbar >= 500.0
    freezing_pressure = jnp.where(at_depth, 500.0, pressure_dbar)
    freezing_temperature = _conservative_temperature_freezing(sa, freezing_pressure)
    return (
        (pressure_dbar <= 8000.0)
        & (sa >= 0.0)
        & (sa <= 42.0)
        & (ct >= freezing_temperature)
        & (~(at_depth & below_deep) | (sa >= 5.0e-3 * pressure_dbar - 2.5))
        & (~(at_depth & below_deep) | (ct <= 31.66666666666667 - pressure_dbar / 300.0))
        & (~(pressure_dbar >= 6500.0) | ((sa >= 30.0) & (ct <= 10.0)))
    )


class TEOS10GSW75EOS(StrictModule, NonTrainableState):
    """Official GSW 75-term SA/CT/sea-pressure specific-volume subset."""

    minimum_salinity: float = eqx.field(static=True)
    maximum_salinity: float = eqx.field(static=True)
    minimum_temperature: float | None = eqx.field(static=True)
    maximum_temperature: float | None = eqx.field(static=True)
    maximum_pressure_dbar: float = eqx.field(static=True)
    eos_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        minimum_salinity: float = 0.0,
        maximum_salinity: float = 42.0,
        minimum_temperature: float | None = None,
        maximum_temperature: float | None = None,
        maximum_pressure_dbar: float = 8_000.0,
    ):
        salinity_bounds = (float(minimum_salinity), float(maximum_salinity))
        temperature_bounds = (
            None if minimum_temperature is None else float(minimum_temperature),
            None if maximum_temperature is None else float(maximum_temperature),
        )
        maximum_pressure = float(maximum_pressure_dbar)
        finite_bounds = (
            *salinity_bounds,
            *(value for value in temperature_bounds if value is not None),
            maximum_pressure,
        )
        if (
            any(not math.isfinite(value) for value in finite_bounds)
            or salinity_bounds[0] < 0.0
            or salinity_bounds[1] <= salinity_bounds[0]
            or (
                temperature_bounds[0] is not None
                and temperature_bounds[1] is not None
                and temperature_bounds[1] <= temperature_bounds[0]
            )
            or maximum_pressure <= 0.0
        ):
            raise ValueError("GSW75 oceanographic funnel bounds are invalid.")
        self.minimum_salinity, self.maximum_salinity = salinity_bounds
        self.minimum_temperature, self.maximum_temperature = temperature_bounds
        self.maximum_pressure_dbar = maximum_pressure
        bounds = (
            *salinity_bounds,
            *temperature_bounds,
            maximum_pressure,
        )
        self.eos_id = canonical_fingerprint(
            {
                "kind": "teos10-gsw75-specific-volume",
                "variables": [
                    "absolute-salinity-g/kg",
                    "conservative-temperature-degC",
                    "sea-pressure-dbar",
                ],
                "funnel": list(bounds),
                "coefficient_count": len(_GSW75_TERMS),
                "funnel_algorithm": "gsw-infunnel-poly-freezing-zero-air-saturation",
            }
        )

    def evaluate(
        self,
        absolute_salinity: ArrayLike,
        conservative_temperature: ArrayLike,
        pressure_dbar: ArrayLike,
        /,
    ) -> HydrostaticEOSResult:
        salinity = jnp.asarray(absolute_salinity)
        salinity = jnp.asarray(salinity, dtype=jnp.result_type(salinity, jnp.float32))
        temperature = jnp.asarray(conservative_temperature, dtype=salinity.dtype)
        pressure = jnp.asarray(pressure_dbar, dtype=salinity.dtype)
        if salinity.shape != temperature.shape or pressure.shape != salinity.shape:
            raise ValueError("GSW75 SA, CT, and sea pressure must share one shape.")
        specific_volume = _specific_volume(salinity, temperature, pressure)
        ones = jnp.ones_like(salinity)
        zeros = jnp.zeros_like(salinity)
        derivative_salinity = jax.jvp(
            _specific_volume,
            (salinity, temperature, pressure),
            (ones, zeros, zeros),
        )[1]
        derivative_temperature = jax.jvp(
            _specific_volume,
            (salinity, temperature, pressure),
            (zeros, ones, zeros),
        )[1]
        derivative_pressure_dbar = jax.jvp(
            _specific_volume,
            (salinity, temperature, pressure),
            (zeros, zeros, ones),
        )[1]
        density = 1.0 / specific_volume
        alpha = derivative_temperature / specific_volume
        beta = -derivative_salinity / specific_volume
        density_pressure_derivative = (
            -derivative_pressure_dbar / specific_volume**2 / 1.0e4
        )
        funnel = (
            _in_oceanographic_funnel(salinity, temperature, pressure)
            & (salinity >= self.minimum_salinity)
            & (salinity <= self.maximum_salinity)
            & (pressure >= 0.0)
            & (pressure <= self.maximum_pressure_dbar)
        )
        if self.minimum_temperature is not None:
            funnel = funnel & (temperature >= self.minimum_temperature)
        if self.maximum_temperature is not None:
            funnel = funnel & (temperature <= self.maximum_temperature)
        finite = jnp.all(
            jnp.isfinite(
                jnp.stack(
                    (
                        density,
                        alpha,
                        beta,
                        density_pressure_derivative,
                    )
                )
            )
        )
        valid = jnp.all(funnel) & jnp.all(specific_volume > 0.0)
        return HydrostaticEOSResult(
            density=density,
            alpha=alpha,
            beta=beta,
            density_pressure_derivative=density_pressure_derivative,
            valid=valid,
            finite=finite,
            successful=finite & valid,
            eos_id=self.eos_id,
        )


__all__ = ["TEOS10GSW75EOS"]
