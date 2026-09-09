#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Tokamak quantity meanings on the shared measurement substrate."""

from __future__ import annotations

from types import MappingProxyType

from ...measurement import QuantitySpec, resolve_quantity
from ...units import (
    AMPERE,
    derived_unit,
    JOULE,
    KELVIN,
    METER,
    ONE,
    PASCAL,
    RADIAN,
    SECOND,
    TESLA,
    UnitDefinition,
    VOLT,
    WEBER,
)


_REFERENCE_UNITS = MappingProxyType(
    {
        "coil_current": AMPERE,
        "coil_voltage": VOLT,
        "electron_density": derived_unit("1/m3-tokamak", ((METER, -3),)),
        "length": METER,
        "magnetic_field": TESLA,
        "particle_source_density": derived_unit(
            "1/m3/s-tokamak", ((METER, -3), (SECOND, -1))
        ),
        "poloidal_flux": derived_unit("Wb/rad-tokamak", ((WEBER, 1), (RADIAN, -1))),
        "power": derived_unit("W-tokamak", ((JOULE, 1), (SECOND, -1))),
        "power_density": derived_unit(
            "W/m3-tokamak", ((JOULE, 1), (SECOND, -1), (METER, -3))
        ),
        "pressure": PASCAL,
        "safety_factor": ONE,
        "temperature": KELVIN,
        "thermal_energy": JOULE,
        "time": SECOND,
    }
)


def resolve_tokamak_quantity(
    name: str,
    quantity_kind: str,
    unit: UnitDefinition,
    /,
    *,
    axes: tuple[str, ...] = (),
    sign_convention: str = "positive",
    support_association: str = "unspecified",
    reference_configuration: str = "absolute",
) -> QuantitySpec:
    return resolve_quantity(
        domain="tokamak",
        reference_units=_REFERENCE_UNITS,
        name=name,
        quantity_kind=quantity_kind,
        unit=unit,
        axes=axes,
        sign_convention=sign_convention,
        support_association=support_association,
        reference_configuration=reference_configuration,
    )


__all__ = ["resolve_tokamak_quantity"]
