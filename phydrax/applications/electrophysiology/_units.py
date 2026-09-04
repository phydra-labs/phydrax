#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Explicit unit definitions and conversions for electrophysiology quantities.

Compiled kernels use millivolts, milliseconds, nanoamperes, microsiemens,
nanofarads, micrometres, millimolar concentrations, and kelvin. Conversion is
kept at host-facing boundaries so every numerical field has an unambiguous unit.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, TypeAlias

import equinox as eqx
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...units import (
    AMPERE,
    CAPACITANCE,
    CENTIMETER,
    CONCENTRATION,
    conversion_factor as _conversion_factor,
    convert_value as _convert_value,
    CURRENT,
    derived_unit,
    FARAD,
    KELVIN,
    MEGOHM,
    METER,
    MICROAMPERE,
    MICROAMPERE_PER_SQUARE_CENTIMETER,
    MICROFARAD,
    MICROMETER,
    MICROSIEMENS,
    MILLIMETER,
    MILLIMOLAR,
    MILLISECOND,
    MILLISIEMENS,
    MILLISIEMENS_PER_SQUARE_CENTIMETER,
    MILLIVOLT,
    MOLE_PER_CUBIC_METER,
    NANOAMPERE,
    NANOFARAD,
    OHM,
    OHM_CENTIMETER,
    SECOND,
    SI_REFERENCE_SYSTEM_ID,
    SIEMENS,
    TIME,
    UnitDefinition,
    VOLT,
    VOLTAGE,
)


UnitLike: TypeAlias = str | UnitDefinition

_MICROSECOND = UnitDefinition("us", TIME, SI_REFERENCE_SYSTEM_ID, "1e-6")
_MICROVOLT = UnitDefinition("uV", VOLTAGE, SI_REFERENCE_SYSTEM_ID, "1e-6")
_MILLIAMPERE = UnitDefinition("mA", CURRENT, SI_REFERENCE_SYSTEM_ID, "1e-3")
_PICOAMPERE = UnitDefinition("pA", CURRENT, SI_REFERENCE_SYSTEM_ID, "1e-12")
_NANOSIEMENS = UnitDefinition("nS", SIEMENS.dimension, SI_REFERENCE_SYSTEM_ID, "1e-9")
_PICOFARAD = UnitDefinition("pF", CAPACITANCE, SI_REFERENCE_SYSTEM_ID, "1e-12")
_MICROMOLAR = UnitDefinition("uM", CONCENTRATION, SI_REFERENCE_SYSTEM_ID, "1e-3")
_KILOHM = UnitDefinition("kohm", OHM.dimension, SI_REFERENCE_SYSTEM_ID, "1e3")
_SQUARE_CENTIMETER = derived_unit("cm2", ((CENTIMETER, 2),))
_SQUARE_MICROMETER = derived_unit("um2", ((MICROMETER, 2),))
_AMPERE_PER_SQUARE_METER = derived_unit("A_per_m2", ((AMPERE, 1), (METER, -2)))
_MILLIAMPERE_PER_SQUARE_CENTIMETER = derived_unit(
    "mA_per_cm2", ((_MILLIAMPERE, 1), (CENTIMETER, -2))
)
_SIEMENS_PER_SQUARE_METER = derived_unit("S_per_m2", ((SIEMENS, 1), (METER, -2)))
_SIEMENS_PER_SQUARE_CENTIMETER = derived_unit(
    "S_per_cm2", ((SIEMENS, 1), (CENTIMETER, -2))
)
_MICROSIEMENS_PER_SQUARE_CENTIMETER = derived_unit(
    "uS_per_cm2", ((MICROSIEMENS, 1), (CENTIMETER, -2))
)

_UNIT_ALIASES = MappingProxyType(
    {
        "s": SECOND,
        "ms": MILLISECOND,
        "us": _MICROSECOND,
        "V": VOLT,
        "mV": MILLIVOLT,
        "uV": _MICROVOLT,
        "A": AMPERE,
        "mA": _MILLIAMPERE,
        "uA": MICROAMPERE,
        "nA": NANOAMPERE,
        "pA": _PICOAMPERE,
        "S": SIEMENS,
        "mS": MILLISIEMENS,
        "uS": MICROSIEMENS,
        "nS": _NANOSIEMENS,
        "F": FARAD,
        "uF": MICROFARAD,
        "nF": NANOFARAD,
        "pF": _PICOFARAD,
        "m": METER,
        "cm": CENTIMETER,
        "mm": MILLIMETER,
        "um": MICROMETER,
        "mol_per_m3": MOLE_PER_CUBIC_METER,
        "mM": MILLIMOLAR,
        "uM": _MICROMOLAR,
        "K": KELVIN,
        "ohm": OHM,
        "kohm": _KILOHM,
        "Mohm": MEGOHM,
        "ohm_cm": OHM_CENTIMETER,
        "cm2": _SQUARE_CENTIMETER,
        "um2": _SQUARE_MICROMETER,
        "A_per_m2": _AMPERE_PER_SQUARE_METER,
        "mA_per_cm2": _MILLIAMPERE_PER_SQUARE_CENTIMETER,
        "uA_per_cm2": MICROAMPERE_PER_SQUARE_CENTIMETER,
        "S_per_m2": _SIEMENS_PER_SQUARE_METER,
        "S_per_cm2": _SIEMENS_PER_SQUARE_CENTIMETER,
        "mS_per_cm2": MILLISIEMENS_PER_SQUARE_CENTIMETER,
        "uS_per_cm2": _MICROSIEMENS_PER_SQUARE_CENTIMETER,
    }
)


def _unit(value: UnitLike, /) -> UnitDefinition:
    if isinstance(value, UnitDefinition):
        return value
    if not isinstance(value, str) or value not in _UNIT_ALIASES:
        raise ValueError(f"Unknown electrophysiology unit {value!r}.")
    return _UNIT_ALIASES[value]


def conversion_factor(from_unit: UnitLike, to_unit: UnitLike, /) -> float:
    """Return the finite multiplier between two explicit compatible units."""
    return float(_conversion_factor(_unit(from_unit), _unit(to_unit)))


def convert_quantity(value: Any, from_unit: UnitLike, to_unit: UnitLike, /) -> Array:
    """Convert a scalar or array without hiding the source or target unit."""
    return _convert_value(value, source=_unit(from_unit), target=_unit(to_unit))


class ElectrophysiologyUnits(StrictModule, NonTrainableState):
    """Canonical compiled-kernel unit contract."""

    time: UnitDefinition
    voltage: UnitDefinition
    current: UnitDefinition
    conductance: UnitDefinition
    capacitance: UnitDefinition
    length: UnitDefinition
    concentration: UnitDefinition
    temperature: UnitDefinition
    units_id: str = eqx.field(static=True)

    def __init__(self) -> None:
        self.time = MILLISECOND
        self.voltage = MILLIVOLT
        self.current = NANOAMPERE
        self.conductance = MICROSIEMENS
        self.capacitance = NANOFARAD
        self.length = MICROMETER
        self.concentration = MILLIMOLAR
        self.temperature = KELVIN
        canonical = {
            "time": self.time.unit_id,
            "voltage": self.voltage.unit_id,
            "current": self.current.unit_id,
            "conductance": self.conductance.unit_id,
            "capacitance": self.capacitance.unit_id,
            "length": self.length.unit_id,
            "concentration": self.concentration.unit_id,
            "temperature": self.temperature.unit_id,
        }
        self.units_id = canonical_fingerprint(
            {"kind": "electrophysiology-units", "canonical": canonical}
        )

    def convert(self, value: Any, from_unit: UnitLike, to_unit: UnitLike, /) -> Array:
        """Convert a quantity under this explicit contract."""
        return convert_quantity(value, from_unit, to_unit)


ELECTROPHYSIOLOGY_UNITS = ElectrophysiologyUnits()


__all__ = [
    "ELECTROPHYSIOLOGY_UNITS",
    "ElectrophysiologyUnits",
    "conversion_factor",
    "convert_quantity",
]
