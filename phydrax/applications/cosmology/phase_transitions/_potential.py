#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState


class ThermalQuarticPotential(StrictModule, NonTrainableState):
    quadratic_coefficient: Array
    cubic_coefficient: Array
    quartic_coefficient: Array
    reference_temperature: Array
    potential_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        quadratic_coefficient: float,
        cubic_coefficient: float,
        quartic_coefficient: float,
        reference_temperature: float,
    ):
        values = tuple(
            map(
                float,
                (
                    quadratic_coefficient,
                    cubic_coefficient,
                    quartic_coefficient,
                    reference_temperature,
                ),
            )
        )
        if (
            any(not math.isfinite(value) for value in values)
            or values[0] <= 0.0
            or values[1] < 0.0
            or values[2] <= 0.0
            or values[3] <= 0.0
        ):
            raise ValueError("Thermal quartic potential coefficients are invalid.")
        (
            self.quadratic_coefficient,
            self.cubic_coefficient,
            self.quartic_coefficient,
            self.reference_temperature,
        ) = (jnp.asarray(value) for value in values)
        self.potential_id = canonical_fingerprint(
            {"kind": "thermal-quartic-potential", "coefficients": list(values)}
        )

    def evaluate(self, field: ArrayLike, temperature: ArrayLike, /) -> Array:
        value = jnp.asarray(field)
        temperature_ = jnp.asarray(temperature, dtype=value.dtype)
        return (
            self.quadratic_coefficient
            * (temperature_ * temperature_ - self.reference_temperature**2)
            * value**2
            - self.cubic_coefficient * temperature_ * value**3
            + 0.25 * self.quartic_coefficient * value**4
        )


class ThermalMinimaResult(StrictModule, NonTrainableState):
    stationary_fields: Array
    potential_values: Array
    local_minimum: Array
    discriminant: Array
    finite: Array
    potential_id: str = eqx.field(static=True)


def thermal_stationary_points(
    potential: ThermalQuarticPotential,
    temperature: ArrayLike,
    /,
) -> ThermalMinimaResult:
    """Return the origin and two analytic nonzero stationary roots."""
    if not isinstance(potential, ThermalQuarticPotential):
        raise TypeError("potential must be ThermalQuarticPotential.")
    temperature_ = jnp.asarray(temperature)
    linear = (
        2.0
        * potential.quadratic_coefficient
        * (temperature_**2 - potential.reference_temperature**2)
    )
    discriminant = (
        3.0 * potential.cubic_coefficient * temperature_
    ) ** 2 - 4.0 * potential.quartic_coefficient * linear
    root = jnp.sqrt(jnp.maximum(discriminant, 0.0))
    first = (3.0 * potential.cubic_coefficient * temperature_ - root) / (
        2.0 * potential.quartic_coefficient
    )
    second = (3.0 * potential.cubic_coefficient * temperature_ + root) / (
        2.0 * potential.quartic_coefficient
    )
    fields = jnp.stack((jnp.zeros_like(temperature_), first, second), axis=-1)
    values = potential.evaluate(fields, temperature_[..., None])
    second_derivative = (
        2.0
        * potential.quadratic_coefficient
        * (temperature_[..., None] ** 2 - potential.reference_temperature**2)
        - 6.0 * potential.cubic_coefficient * temperature_[..., None] * fields
        + 3.0 * potential.quartic_coefficient * fields**2
    )
    local_minimum = (second_derivative > 0.0) & (
        (discriminant >= 0.0)[..., None] | (fields == 0.0)
    )
    finite = jnp.all(jnp.isfinite(values), axis=-1)
    return ThermalMinimaResult(
        fields, values, local_minimum, discriminant, finite, potential.potential_id
    )


def thin_wall_thermal_action(
    surface_tension: ArrayLike,
    vacuum_energy_difference: ArrayLike,
    temperature: ArrayLike,
    /,
) -> Array:
    """Return the thin-wall S3/T action 16πσ³/(3 ΔV² T)."""
    sigma = jnp.asarray(surface_tension)
    delta_v = jnp.asarray(vacuum_energy_difference, dtype=sigma.dtype)
    temperature_ = jnp.asarray(temperature, dtype=sigma.dtype)
    valid = (sigma > 0.0) & (delta_v != 0.0) & (temperature_ > 0.0)
    action = 16.0 * jnp.pi * sigma**3 / (3.0 * delta_v**2 * temperature_)
    return jnp.where(valid, action, jnp.nan)


__all__ = [
    "ThermalMinimaResult",
    "ThermalQuarticPotential",
    "thermal_stationary_points",
    "thin_wall_thermal_action",
]
