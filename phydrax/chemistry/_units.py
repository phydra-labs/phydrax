#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact unit and physical-constant boundaries for molecular chemistry."""

from __future__ import annotations

import math

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..atomistic import AtomisticUnitSystem
from ..units import (
    conversion_factor,
    derived_unit,
    ENERGY,
    INVERSE_CENTIMETER,
    SI_REFERENCE_SYSTEM_ID,
    TEMPERATURE,
    UnitDefinition,
)


_SPEED_OF_LIGHT_M_PER_S = 299_792_458.0


class ChemistryPhysicalConstants(StrictModule, NonTrainableState):
    """CODATA-compatible constants represented in one atomistic unit system."""

    units: AtomisticUnitSystem
    speed_of_light: float = eqx.field(static=True)
    planck_constant: float = eqx.field(static=True)
    constants_id: str = eqx.field(static=True)

    def __init__(self, units: AtomisticUnitSystem, /):
        if not isinstance(units, AtomisticUnitSystem):
            raise TypeError("units must be AtomisticUnitSystem.")
        if (
            units.constant_set_id != "codata-2018"
            or units.scale.length_unit.reference_system_id != SI_REFERENCE_SYSTEM_ID
        ):
            raise ValueError(
                "Absolute molecular thermochemistry and spectroscopy require codata-2018 SI-referenced units."
            )
        length_scale = float(units.scale.length_unit.scale_to_reference)
        time_scale = float(units.time_unit.scale_to_reference)
        speed = _SPEED_OF_LIGHT_M_PER_S * time_scale / length_scale
        planck = 2.0 * math.pi * units.reduced_planck_constant
        if not math.isfinite(speed) or speed <= 0.0 or not math.isfinite(planck) or planck <= 0.0:
            raise ValueError("Chemistry physical constants are invalid in the selected units.")
        self.units = units
        self.speed_of_light = speed
        self.planck_constant = planck
        self.constants_id = canonical_fingerprint(
            {
                "kind": "chemistry-physical-constants",
                "unit_system": units.unit_system_id,
                "constant_set": units.constant_set_id,
                "speed_of_light": speed,
                "planck_constant": planck,
            }
        )


def hessian_unit(units: AtomisticUnitSystem, /) -> UnitDefinition:
    if not isinstance(units, AtomisticUnitSystem):
        raise TypeError("units must be AtomisticUnitSystem.")
    return derived_unit(
        f"{units.scale.energy_unit.symbol}/{units.scale.length_unit.symbol}^2",
        ((units.scale.energy_unit, 1), (units.scale.length_unit, -2)),
    )


def dipole_unit(units: AtomisticUnitSystem, /) -> UnitDefinition:
    if not isinstance(units, AtomisticUnitSystem):
        raise TypeError("units must be AtomisticUnitSystem.")
    return derived_unit(
        f"{units.charge_unit.symbol}*{units.scale.length_unit.symbol}",
        ((units.charge_unit, 1), (units.scale.length_unit, 1)),
    )


def dipole_derivative_unit(units: AtomisticUnitSystem, /) -> UnitDefinition:
    if not isinstance(units, AtomisticUnitSystem):
        raise TypeError("units must be AtomisticUnitSystem.")
    return units.charge_unit


def entropy_unit(units: AtomisticUnitSystem, /) -> UnitDefinition:
    if not isinstance(units, AtomisticUnitSystem):
        raise TypeError("units must be AtomisticUnitSystem.")
    result = derived_unit(
        f"{units.scale.energy_unit.symbol}/{units.temperature_unit.symbol}",
        ((units.scale.energy_unit, 1), (units.temperature_unit, -1)),
    )
    if result.dimension != ENERGY / TEMPERATURE:
        raise RuntimeError("Derived entropy unit has an invalid dimension.")
    return result


def angular_frequency_to_wavenumber(
    angular_frequency: ArrayLike,
    units: AtomisticUnitSystem,
    /,
) -> Array:
    """Convert angular frequency in inverse native time to signed cm^-1."""

    constants = ChemistryPhysicalConstants(units)
    inverse_length = derived_unit(
        f"1/{units.scale.length_unit.symbol}", ((units.scale.length_unit, -1),)
    )
    factor = float(conversion_factor(inverse_length, INVERSE_CENTIMETER))
    value = jnp.asarray(angular_frequency)
    return value * (factor / (2.0 * math.pi * constants.speed_of_light))


__all__ = [
    "ChemistryPhysicalConstants",
    "angular_frequency_to_wavenumber",
    "dipole_derivative_unit",
    "dipole_unit",
    "entropy_unit",
    "hessian_unit",
]
