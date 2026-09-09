#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact units at preparation boundaries; semiconductor kernels use SI.

Carrier density is a *number* per physical volume, not molar concentration.
Terminal current is positive into the device, and terminal charge is the
charge on the external electrode. Quasi-Fermi coordinates are energies/kT.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np

from ...measurement import resolve_quantity
from ...units import (
    AMPERE,
    convert_value,
    COULOMB,
    CUBIC_METER,
    derived_unit,
    FARAD,
    JOULE,
    KELVIN,
    METER,
    ONE,
    SECOND,
    UnitDefinition,
    VOLT,
)


ELEMENTARY_CHARGE_SI = 1.602176634e-19
BOLTZMANN_CONSTANT_SI = 1.380649e-23
VACUUM_PERMITTIVITY_SI = 8.8541878128e-12
SQUARE_METER = derived_unit("m2", ((METER, 2),))
HERTZ_UNIT = derived_unit("Hz", ((SECOND, -1),))
WATT_UNIT = derived_unit("W", ((JOULE, 1), (SECOND, -1)))
PER_CUBIC_METER = derived_unit("1/m3", ((METER, -3),))
PERMITTIVITY_UNIT = derived_unit("F/m", ((FARAD, 1), (METER, -1)))
MOBILITY_UNIT = derived_unit("m2/(V*s)", ((METER, 2), (VOLT, -1), (SECOND, -1)))
NUMBER_FLUX_UNIT = derived_unit("1/s", ((SECOND, -1),))
RECOMBINATION_UNIT = derived_unit("1/(m3*s)", ((METER, -3), (SECOND, -1)))

_REFERENCE_UNITS = MappingProxyType(
    {
        "length": METER,
        "area": SQUARE_METER,
        "volume": CUBIC_METER,
        "number_density": PER_CUBIC_METER,
        "permittivity": PERMITTIVITY_UNIT,
        "mobility": MOBILITY_UNIT,
        "temperature": KELVIN,
        "energy": JOULE,
        "voltage": VOLT,
        "current": AMPERE,
        "charge": COULOMB,
        "time": SECOND,
        "number_flux": NUMBER_FLUX_UNIT,
        "recombination": RECOMBINATION_UNIT,
        "coordinate": ONE,
        "number": ONE,
        "surface_number_density": derived_unit("1/m2", ((METER, -2),)),
        "frequency": HERTZ_UNIT,
        "power": WATT_UNIT,
        "electric_field": derived_unit("V/m", ((VOLT, 1), (METER, -1))),
        "energy_density": derived_unit("J/m3", ((JOULE, 1), (METER, -3))),
        "heat_capacity_density": derived_unit(
            "J/(m3*K)", ((JOULE, 1), (METER, -3), (KELVIN, -1))
        ),
        "thermal_conductivity": derived_unit(
            "W/(m*K)", ((WATT_UNIT, 1), (METER, -1), (KELVIN, -1))
        ),
        "particle_flux_density": derived_unit("1/(m2*s)", ((METER, -2), (SECOND, -1))),
        "heat_flux": derived_unit("W/m2", ((WATT_UNIT, 1), (METER, -2))),
    }
)


def _si(value: Any, unit: UnitDefinition, reference: UnitDefinition):
    if not isinstance(unit, UnitDefinition):
        raise TypeError("Physical units must be native UnitDefinition values.")
    return convert_value(value, source=unit, target=reference)


def _positive_scalar(value, unit, reference, name, *, nonnegative=False):
    array = _si(value, unit, reference)
    host = np.asarray(array)
    if (
        host.shape != ()
        or not np.isfinite(host)
        or (host < 0 if nonnegative else host <= 0)
    ):
        qualifier = "nonnegative" if nonnegative else "positive"
        raise ValueError(f"{name} must be one finite {qualifier} scalar.")
    return array


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be explicit nonempty text.")
    return value.strip()


@dataclass(frozen=True, slots=True)
class SemiconductorQuantitySpec:
    """Identity-bearing domain quantity using the native unit contract."""

    name: str
    quantity_kind: str
    unit: UnitDefinition
    axes: tuple[str, ...] = ()
    sign_convention: str = ""
    support_association: str = ""
    reference_configuration: str = ""
    quantity_id: str = field(init=False)

    def __post_init__(self):
        resolved = resolve_quantity(
            domain="semiconductor",
            reference_units=_REFERENCE_UNITS,
            name=self.name,
            quantity_kind=self.quantity_kind,
            unit=self.unit,
            axes=self.axes,
            sign_convention=self.sign_convention,
            support_association=self.support_association,
            reference_configuration=self.reference_configuration,
        )
        object.__setattr__(self, "axes", resolved.axes)
        object.__setattr__(self, "quantity_id", resolved.quantity_id)

    def to_si(self, value: Any):
        return _si(value, self.unit, _REFERENCE_UNITS[self.quantity_kind])


__all__ = [
    "BOLTZMANN_CONSTANT_SI",
    "ELEMENTARY_CHARGE_SI",
    "VACUUM_PERMITTIVITY_SI",
    "MOBILITY_UNIT",
    "HERTZ_UNIT",
    "NUMBER_FLUX_UNIT",
    "PER_CUBIC_METER",
    "PERMITTIVITY_UNIT",
    "RECOMBINATION_UNIT",
    "SQUARE_METER",
    "WATT_UNIT",
    "SemiconductorQuantitySpec",
]
