#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


class SaturationPressureEvaluation(StrictModule):
    value: Array
    domain_margin: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class AntoineSaturationPressurePlan(StrictModule, NonTrainableState):
    coefficient_a: float = eqx.field(static=True)
    coefficient_b: float = eqx.field(static=True)
    coefficient_c: float = eqx.field(static=True)
    pressure_scale: float = eqx.field(static=True)
    minimum_temperature: float = eqx.field(static=True)
    maximum_temperature: float = eqx.field(static=True)
    minimum_pressure: float = eqx.field(static=True)
    maximum_pressure: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        coefficient_a,
        coefficient_b,
        coefficient_c,
        /,
        *,
        temperature_interval,
        pressure_scale=133.322,
    ):
        a, b, c, scale = (
            float(coefficient_a),
            float(coefficient_b),
            float(coefficient_c),
            float(pressure_scale),
        )
        interval = np.asarray(temperature_interval, dtype=np.float64)
        if (
            interval.shape != (2,)
            or np.any(~np.isfinite(interval))
            or interval[0] >= interval[1]
            or any(not np.isfinite(value) for value in (a, b, c, scale))
            or b <= 0.0
            or scale <= 0.0
        ):
            raise ValueError("Antoine saturation-pressure parameters are invalid.")
        denominators = c + interval - 273.15
        if np.any(denominators <= 0.0):
            raise ValueError(
                "The Antoine temperature interval must remain above its pole."
            )
        pressure = scale * 10.0 ** (a - b / denominators)
        if np.any(~np.isfinite(pressure)) or np.any(pressure <= 0.0):
            raise ValueError("The Antoine pressure interval is invalid.")
        self.coefficient_a = a
        self.coefficient_b = b
        self.coefficient_c = c
        self.pressure_scale = scale
        self.minimum_temperature = float(interval[0])
        self.maximum_temperature = float(interval[1])
        self.minimum_pressure = float(pressure[0])
        self.maximum_pressure = float(pressure[1])
        self.plan_id = canonical_fingerprint(
            {
                "kind": "antoine-saturation-pressure",
                "coefficients": (a, b, c),
                "pressure_scale": scale,
                "temperature_interval": tuple(float(value) for value in interval),
            }
        )

    def evaluate_pressure(
        self, temperature: ArrayLike, /
    ) -> SaturationPressureEvaluation:
        value = jnp.asarray(temperature)
        if not jnp.issubdtype(value.dtype, jnp.inexact):
            value = value.astype(jnp.float32)
        celsius = value - 273.15
        denominator = self.coefficient_c + celsius
        pressure = self.pressure_scale * 10.0 ** (
            self.coefficient_a - self.coefficient_b / denominator
        )
        margin = jnp.minimum(
            value - self.minimum_temperature, self.maximum_temperature - value
        )
        finite = jnp.isfinite(value) & jnp.isfinite(pressure)
        tolerance = (
            64.0
            * jnp.finfo(value.dtype).eps
            * jnp.maximum(
                jnp.maximum(
                    jnp.abs(value),
                    jnp.maximum(
                        abs(self.minimum_temperature), abs(self.maximum_temperature)
                    ),
                ),
                1.0,
            )
        )
        return SaturationPressureEvaluation(
            pressure,
            margin,
            finite,
            finite & (margin >= -tolerance),
            self.plan_id,
        )

    def pressure(self, temperature: ArrayLike, /) -> Array:
        return self.evaluate_pressure(temperature).value

    def evaluate_temperature(
        self, pressure: ArrayLike, /
    ) -> SaturationPressureEvaluation:
        value = jnp.asarray(pressure)
        if not jnp.issubdtype(value.dtype, jnp.inexact):
            value = value.astype(jnp.float32)
        logarithm = jnp.log10(value / self.pressure_scale)
        denominator = self.coefficient_a - logarithm
        temperature = self.coefficient_b / denominator - self.coefficient_c + 273.15
        margin = jnp.minimum(value - self.minimum_pressure, self.maximum_pressure - value)
        finite = (
            jnp.isfinite(value)
            & (value > 0.0)
            & jnp.isfinite(temperature)
            & (denominator > 0.0)
        )
        tolerance = (
            64.0
            * jnp.finfo(value.dtype).eps
            * jnp.maximum(
                jnp.maximum(
                    jnp.abs(value),
                    jnp.maximum(abs(self.minimum_pressure), abs(self.maximum_pressure)),
                ),
                1.0,
            )
        )
        return SaturationPressureEvaluation(
            temperature,
            margin,
            finite,
            finite & (margin >= -tolerance),
            self.plan_id,
        )

    def temperature(self, pressure: ArrayLike, /) -> Array:
        return self.evaluate_temperature(pressure).value
