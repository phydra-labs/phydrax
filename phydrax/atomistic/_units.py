#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Mapping
from fractions import Fraction
from typing import Any

import equinox as eqx

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..units import (
    AMOUNT,
    ANGSTROM,
    CHARGE,
    DALTON,
    derived_unit,
    ELECTRONVOLT,
    ELEMENTARY_CHARGE,
    ENERGY,
    FEMTOSECOND,
    KELVIN,
    LENGTH,
    MASS,
    SI_REFERENCE_SYSTEM_ID,
    TEMPERATURE,
    TIME,
    UnitDefinition,
)
from ._types import AtomisticScaleContract


_CODATA_2018 = "codata-2018"
_CANONICAL_REDUCED = "canonical-reduced"
_REDUCED_REFERENCE_SYSTEM = "phydrax:atomistic-reduced"
_AVOGADRO_BY_CONSTANT_SET = {
    _CODATA_2018: Fraction("6.02214076e23"),
}
_PHYSICAL_CONSTANTS_BY_SET = {
    _CODATA_2018: (
        Fraction("1.380649e-23"),
        Fraction("8.98755179226117e9"),
        Fraction("1.0545718176461565e-34"),
    ),
}


def _require_unit(
    value: UnitDefinition, dimension, name: str, reference_system_id: str
) -> None:
    if not isinstance(value, UnitDefinition):
        raise TypeError(f"{name} must be a UnitDefinition.")
    if value.dimension != dimension:
        raise ValueError(f"{name} has the wrong physical dimension.")
    if value.reference_system_id != reference_system_id:
        raise ValueError("All atomistic units must share one reference system.")


def molar_energy_to_single_system_factor(
    source: UnitDefinition,
    target: UnitDefinition,
    /,
    *,
    constant_set_id: str,
) -> float:
    """Resolve a host-only molar-energy to single-system energy conversion.

    This is an explicit semantic boundary: ordinary unit conversion intentionally
    rejects ``ENERGY / AMOUNT`` versus ``ENERGY``.
    """

    if not isinstance(source, UnitDefinition) or not isinstance(target, UnitDefinition):
        raise TypeError("Molar energy conversion requires UnitDefinition values.")
    if source.dimension != ENERGY / AMOUNT or target.dimension != ENERGY:
        raise ValueError(
            "Molar energy conversion requires ENERGY / AMOUNT source and ENERGY target."
        )
    if source.reference_system_id != target.reference_system_id:
        raise ValueError("Molar energy conversion requires one reference system.")
    avogadro = _AVOGADRO_BY_CONSTANT_SET.get(constant_set_id)
    if avogadro is None:
        raise ValueError("Unsupported physical constant-set identity.")
    factor = source.scale_to_reference / (avogadro * target.scale_to_reference)
    return float(factor)


class AtomisticUnitSystem(StrictModule, NonTrainableState):
    """Complete immutable host unit descriptor for atomistic dynamics."""

    scale: AtomisticScaleContract
    mass_unit: UnitDefinition
    time_unit: UnitDefinition
    charge_unit: UnitDefinition
    temperature_unit: UnitDefinition
    pressure_unit: UnitDefinition
    velocity_unit: UnitDefinition
    frequency_unit: UnitDefinition
    constant_set_id: str = eqx.field(static=True)
    kinetic_to_energy: float = eqx.field(static=True)
    boltzmann_constant: float = eqx.field(static=True)
    coulomb_constant: float = eqx.field(static=True)
    reduced_planck_constant: float = eqx.field(static=True)
    unit_system_id: str = eqx.field(static=True)

    def __init__(
        self,
        scale: AtomisticScaleContract,
        /,
        *,
        mass_unit: UnitDefinition,
        time_unit: UnitDefinition,
        charge_unit: UnitDefinition,
        temperature_unit: UnitDefinition,
        constant_set_id: str,
    ):
        if not isinstance(scale, AtomisticScaleContract):
            raise TypeError("scale must be an AtomisticScaleContract.")
        reference = scale.length_unit.reference_system_id
        _require_unit(mass_unit, MASS, "mass_unit", reference)
        _require_unit(time_unit, TIME, "time_unit", reference)
        _require_unit(charge_unit, CHARGE, "charge_unit", reference)
        _require_unit(temperature_unit, TEMPERATURE, "temperature_unit", reference)
        if not isinstance(constant_set_id, str) or not constant_set_id:
            raise ValueError("constant_set_id must be a non-empty string.")

        length_scale = scale.length_unit.scale_to_reference
        energy_scale = scale.energy_unit.scale_to_reference
        mass_scale = mass_unit.scale_to_reference
        time_scale = time_unit.scale_to_reference
        charge_scale = charge_unit.scale_to_reference
        temperature_scale = temperature_unit.scale_to_reference
        kinetic = float(mass_scale * length_scale**2 / (time_scale**2 * energy_scale))
        if constant_set_id == _CANONICAL_REDUCED:
            if reference != _REDUCED_REFERENCE_SYSTEM or any(
                value.scale_to_reference != 1
                for value in (
                    scale.length_unit,
                    scale.energy_unit,
                    mass_unit,
                    time_unit,
                    charge_unit,
                    temperature_unit,
                )
            ):
                raise ValueError(
                    "The canonical reduced constant set requires canonical unscaled "
                    "reduced units."
                )
            boltzmann = coulomb = hbar = 1.0
        else:
            if reference != SI_REFERENCE_SYSTEM_ID:
                raise ValueError("CODATA physical constants require SI-referenced units.")
            constants = _PHYSICAL_CONSTANTS_BY_SET.get(constant_set_id)
            if constants is None:
                raise ValueError("Unsupported physical constant-set identity.")
            boltzmann_si, coulomb_si, hbar_si = constants
            boltzmann = float(boltzmann_si * temperature_scale / energy_scale)
            coulomb = float(coulomb_si * charge_scale**2 / (energy_scale * length_scale))
            hbar = float(hbar_si / (energy_scale * time_scale))

        factors = (kinetic, boltzmann, coulomb, hbar)
        if any(not math.isfinite(value) or value <= 0.0 for value in factors):
            raise ValueError("Derived atomistic physical constants are invalid.")
        pressure_unit = derived_unit(
            f"{scale.energy_unit.symbol}/{scale.length_unit.symbol}^3",
            ((scale.energy_unit, 1), (scale.length_unit, -3)),
        )
        velocity_unit = derived_unit(
            f"{scale.length_unit.symbol}/{time_unit.symbol}",
            ((scale.length_unit, 1), (time_unit, -1)),
        )
        frequency_unit = derived_unit(
            f"1/{time_unit.symbol}",
            ((time_unit, -1),),
        )
        self.scale = scale
        self.mass_unit = mass_unit
        self.time_unit = time_unit
        self.charge_unit = charge_unit
        self.temperature_unit = temperature_unit
        self.pressure_unit = pressure_unit
        self.velocity_unit = velocity_unit
        self.frequency_unit = frequency_unit
        self.constant_set_id = constant_set_id
        self.kinetic_to_energy = kinetic
        self.boltzmann_constant = boltzmann
        self.coulomb_constant = coulomb
        self.reduced_planck_constant = hbar
        self.unit_system_id = canonical_fingerprint(
            {
                "kind": "atomistic-unit-system",
                "scale": scale.scale_id,
                "mass_unit": mass_unit.unit_id,
                "time_unit": time_unit.unit_id,
                "charge_unit": charge_unit.unit_id,
                "temperature_unit": temperature_unit.unit_id,
                "constant_set": constant_set_id,
            }
        )

    @classmethod
    def electronvolt_angstrom_dalton_femtosecond(cls) -> "AtomisticUnitSystem":
        """Return the eV-angstrom-dalton-femtosecond-K CODATA-2018 system."""

        return cls(
            AtomisticScaleContract(ANGSTROM, ELECTRONVOLT),
            mass_unit=DALTON,
            time_unit=FEMTOSECOND,
            charge_unit=ELEMENTARY_CHARGE,
            temperature_unit=KELVIN,
            constant_set_id=_CODATA_2018,
        )

    @classmethod
    def reduced(cls) -> "AtomisticUnitSystem":
        """Return the canonical uncalibrated, non-SI-convertible reduced system."""

        length = UnitDefinition("reduced_length", LENGTH, _REDUCED_REFERENCE_SYSTEM)
        energy = UnitDefinition("reduced_energy", ENERGY, _REDUCED_REFERENCE_SYSTEM)
        return cls(
            AtomisticScaleContract(length, energy),
            mass_unit=UnitDefinition("reduced_mass", MASS, _REDUCED_REFERENCE_SYSTEM),
            time_unit=UnitDefinition("reduced_time", TIME, _REDUCED_REFERENCE_SYSTEM),
            charge_unit=UnitDefinition(
                "reduced_charge", CHARGE, _REDUCED_REFERENCE_SYSTEM
            ),
            temperature_unit=UnitDefinition(
                "reduced_temperature", TEMPERATURE, _REDUCED_REFERENCE_SYSTEM
            ),
            constant_set_id=_CANONICAL_REDUCED,
        )

    @property
    def force_to_momentum_rate(self) -> float:
        """Convert an energy/length force to mass*length/time^2."""

        return 1.0 / self.kinetic_to_energy

    def to_dict(self) -> dict[str, Any]:
        return {
            "scale": self.scale.to_dict(),
            "mass_unit": self.mass_unit.to_dict(),
            "time_unit": self.time_unit.to_dict(),
            "charge_unit": self.charge_unit.to_dict(),
            "temperature_unit": self.temperature_unit.to_dict(),
            "constant_set_id": self.constant_set_id,
            "unit_system_id": self.unit_system_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AtomisticUnitSystem":
        if not isinstance(payload, Mapping):
            raise TypeError("Atomistic unit-system descriptor must be a mapping.")
        expected = {
            "scale",
            "mass_unit",
            "time_unit",
            "charge_unit",
            "temperature_unit",
            "constant_set_id",
            "unit_system_id",
        }
        if set(payload) != expected:
            raise ValueError(
                "Atomistic unit-system descriptor must use the canonical fields."
            )
        names = (
            "scale",
            "mass_unit",
            "time_unit",
            "charge_unit",
            "temperature_unit",
        )
        if any(not isinstance(payload.get(name), Mapping) for name in names):
            raise TypeError(
                "Atomistic unit-system descriptor must contain complete unit definitions."
            )
        constant_set_id = payload.get("constant_set_id")
        if not isinstance(constant_set_id, str):
            raise TypeError("Atomistic unit-system constant_set_id must be a string.")
        units = cls(
            AtomisticScaleContract.from_dict(payload["scale"]),
            mass_unit=UnitDefinition.from_dict(payload["mass_unit"]),
            time_unit=UnitDefinition.from_dict(payload["time_unit"]),
            charge_unit=UnitDefinition.from_dict(payload["charge_unit"]),
            temperature_unit=UnitDefinition.from_dict(payload["temperature_unit"]),
            constant_set_id=constant_set_id,
        )
        if payload.get("unit_system_id") != units.unit_system_id:
            raise ValueError("Atomistic unit-system descriptor identity is corrupt.")
        return units


__all__ = ["AtomisticUnitSystem", "molar_energy_to_single_system_factor"]
