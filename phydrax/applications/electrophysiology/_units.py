#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Explicit units and conversions for electrophysiology quantities.

Compiled kernels use millivolts, milliseconds, nanoamperes, microsiemens,
nanofarads, micrometres, and millimolar concentrations. Conversion is kept at
host-facing boundaries so every numerical field has an unambiguous unit.
"""

from __future__ import annotations

from math import isfinite
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


_UNIT_TABLE: dict[str, tuple[str, float]] = {
    "s": ("time", 1.0),
    "ms": ("time", 1.0e-3),
    "us": ("time", 1.0e-6),
    "V": ("voltage", 1.0),
    "mV": ("voltage", 1.0e-3),
    "uV": ("voltage", 1.0e-6),
    "A": ("current", 1.0),
    "mA": ("current", 1.0e-3),
    "uA": ("current", 1.0e-6),
    "nA": ("current", 1.0e-9),
    "pA": ("current", 1.0e-12),
    "S": ("conductance", 1.0),
    "mS": ("conductance", 1.0e-3),
    "uS": ("conductance", 1.0e-6),
    "nS": ("conductance", 1.0e-9),
    "F": ("capacitance", 1.0),
    "uF": ("capacitance", 1.0e-6),
    "nF": ("capacitance", 1.0e-9),
    "pF": ("capacitance", 1.0e-12),
    "m": ("length", 1.0),
    "cm": ("length", 1.0e-2),
    "mm": ("length", 1.0e-3),
    "um": ("length", 1.0e-6),
    "mol_per_m3": ("concentration", 1.0),
    "mM": ("concentration", 1.0),
    "uM": ("concentration", 1.0e-3),
    "K": ("temperature", 1.0),
    "ohm": ("resistance", 1.0),
    "kohm": ("resistance", 1.0e3),
    "Mohm": ("resistance", 1.0e6),
    "cm2": ("area", 1.0e-4),
    "um2": ("area", 1.0e-12),
    "A_per_m2": ("current_density", 1.0),
    "mA_per_cm2": ("current_density", 10.0),
    "uA_per_cm2": ("current_density", 1.0e-2),
    "S_per_m2": ("conductance_density", 1.0),
    "S_per_cm2": ("conductance_density", 1.0e4),
    "mS_per_cm2": ("conductance_density", 10.0),
    "uS_per_cm2": ("conductance_density", 1.0e-2),
}

_CANONICAL_UNITS = (
    ("time", "ms"),
    ("voltage", "mV"),
    ("current", "nA"),
    ("conductance", "uS"),
    ("capacitance", "nF"),
    ("length", "um"),
    ("concentration", "mM"),
    ("temperature", "K"),
)


def _unit_record(name: str, /) -> tuple[str, float]:
    if not isinstance(name, str) or name not in _UNIT_TABLE:
        raise ValueError(f"Unknown electrophysiology unit {name!r}.")
    return _UNIT_TABLE[name]


def conversion_factor(from_unit: str, to_unit: str, /) -> float:
    """Return the finite multiplier converting between compatible units."""
    from_dimension, from_scale = _unit_record(from_unit)
    to_dimension, to_scale = _unit_record(to_unit)
    if from_dimension != to_dimension:
        raise ValueError(
            f"Cannot convert {from_dimension} unit {from_unit!r} to "
            f"{to_dimension} unit {to_unit!r}."
        )
    factor = from_scale / to_scale
    if not isfinite(factor) or factor <= 0.0:
        raise ValueError("Unit conversion factor must be finite and positive.")
    return factor


def convert_quantity(value: Any, from_unit: str, to_unit: str, /) -> Array:
    """Convert a scalar or array without hiding the source or target unit."""
    return jnp.asarray(value) * conversion_factor(from_unit, to_unit)


class ElectrophysiologyUnits(StrictModule, NonTrainableState):
    """Canonical compiled-kernel unit contract."""

    time: str = eqx.field(static=True)
    voltage: str = eqx.field(static=True)
    current: str = eqx.field(static=True)
    conductance: str = eqx.field(static=True)
    capacitance: str = eqx.field(static=True)
    length: str = eqx.field(static=True)
    concentration: str = eqx.field(static=True)
    temperature: str = eqx.field(static=True)
    units_id: str = eqx.field(static=True)

    def __init__(self) -> None:
        values = dict(_CANONICAL_UNITS)
        self.time = values["time"]
        self.voltage = values["voltage"]
        self.current = values["current"]
        self.conductance = values["conductance"]
        self.capacitance = values["capacitance"]
        self.length = values["length"]
        self.concentration = values["concentration"]
        self.temperature = values["temperature"]
        self.units_id = canonical_fingerprint(
            {"kind": "electrophysiology-units-v1", "canonical": values}
        )

    def convert(self, value: Any, from_unit: str, to_unit: str, /) -> Array:
        """Convert a quantity under this explicit contract."""
        return convert_quantity(value, from_unit, to_unit)


ELECTROPHYSIOLOGY_UNITS = ElectrophysiologyUnits()


__all__ = [
    "ELECTROPHYSIOLOGY_UNITS",
    "ElectrophysiologyUnits",
    "conversion_factor",
    "convert_quantity",
]
