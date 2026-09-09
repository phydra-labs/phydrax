# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Physical meaning is separate from numerical storage and exchange compatibility."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from fractions import Fraction
from types import MappingProxyType
from typing import Any

from ..._fingerprint import canonical_fingerprint
from ...units import (
    conversion_factor,
    derived_unit,
    JOULE,
    KELVIN,
    KILOGRAM,
    METER,
    MOLE,
    ONE,
    PASCAL,
    SECOND,
    UnitDefinition,
)
from .._quantity_contract import resolve_application_quantity


_AREA = derived_unit("m2", ((METER, 2),))
_DENSITY = derived_unit("kg/m3", ((KILOGRAM, 1), (METER, -3)))
_SPECIFIC_ENERGY = derived_unit("J/kg", ((JOULE, 1), (KILOGRAM, -1)))
_HEAT_FLUX = derived_unit("W/m2", ((JOULE, 1), (SECOND, -1), (METER, -2)))
_MASS_FLUX = derived_unit("kg/m2/s", ((KILOGRAM, 1), (METER, -2), (SECOND, -1)))
_REFERENCE_UNITS = MappingProxyType(
    {
        "time": SECOND,
        "length": METER,
        "height": METER,
        "area": _AREA,
        "mass": KILOGRAM,
        "amount": MOLE,
        "pressure": PASCAL,
        "pressure_anomaly": PASCAL,
        "surface_pressure": PASCAL,
        "temperature": KELVIN,
        "temperature_anomaly": KELVIN,
        "mass_density": _DENSITY,
        "dry_air_density": _DENSITY,
        "moist_air_density": _DENSITY,
        "velocity": derived_unit("m/s", ((METER, 1), (SECOND, -1))),
        "acceleration": derived_unit("m/s2", ((METER, 1), (SECOND, -2))),
        "vorticity": derived_unit("1/s", ((SECOND, -1),)),
        "divergence": derived_unit("1/s", ((SECOND, -1),)),
        "geopotential": _SPECIFIC_ENERGY,
        "specific_energy": _SPECIFIC_ENERGY,
        "energy": JOULE,
        "heat_flux": _HEAT_FLUX,
        "radiative_flux": _HEAT_FLUX,
        "momentum_flux": PASCAL,
        "water_mass_flux": _MASS_FLUX,
        "precipitation_rate": _MASS_FLUX,
        "water_volume_flux": derived_unit("m/s", ((METER, 1), (SECOND, -1))),
        "precipitation_amount": derived_unit("kg/m2", ((KILOGRAM, 1), (METER, -2))),
        "column_mass": derived_unit("kg/m2", ((KILOGRAM, 1), (METER, -2))),
        "water_mass_per_area": derived_unit("kg/m2", ((KILOGRAM, 1), (METER, -2))),
        "enthalpy_per_area": derived_unit("J/m2", ((JOULE, 1), (METER, -2))),
        "impulse_per_area": derived_unit("Pa*s", ((PASCAL, 1), (SECOND, 1))),
        "specific_humidity": ONE,
        "total_water_fraction": ONE,
        "vapor_mixing_ratio": ONE,
        "liquid_water_fraction": ONE,
        "ice_water_fraction": ONE,
        "mole_fraction": ONE,
        "mass_fraction": ONE,
        "dimensionless": ONE,
    }
)


@dataclass(frozen=True, slots=True, init=False)
class GeophysicalQuantity:
    """Immutable domain quantity; equal dimensions alone never imply compatibility."""

    name: str
    quantity_kind: str
    unit: UnitDefinition
    axes: tuple[str, ...]
    sign_convention: str
    support_association: str
    reference_configuration: str
    quantity_id: str = field(init=False)
    compatibility_id: str = field(init=False)

    def __init__(
        self,
        name: str,
        quantity_kind: str,
        unit: UnitDefinition,
        *,
        axes: tuple[str, ...] = (),
        sign_convention: str = "positive",
        support_association: str = "unspecified",
        reference_configuration: str = "absolute",
    ):
        value = resolve_application_quantity(
            domain="geophysical",
            reference_units=_REFERENCE_UNITS,
            name=name,
            quantity_kind=quantity_kind,
            unit=unit,
            axes=axes,
            sign_convention=sign_convention,
            support_association=support_association,
            reference_configuration=reference_configuration,
        )
        object.__setattr__(self, "name", value.name)
        object.__setattr__(self, "quantity_kind", value.quantity_kind)
        object.__setattr__(self, "unit", value.unit)
        object.__setattr__(self, "axes", value.axes)
        object.__setattr__(self, "sign_convention", value.sign_convention)
        object.__setattr__(self, "support_association", value.support_association)
        object.__setattr__(self, "reference_configuration", value.reference_configuration)
        object.__setattr__(self, "quantity_id", value.quantity_id)
        object.__setattr__(
            self,
            "compatibility_id",
            canonical_fingerprint(
                {
                    "kind": "geophysical-physical-compatibility",
                    "quantity_kind": value.quantity_kind,
                    "reference_unit": self.reference_unit.unit_id,
                    "sign": value.sign_convention,
                    "reference": value.reference_configuration,
                }
            ),
        )

    @property
    def reference_unit(self) -> UnitDefinition:
        return _REFERENCE_UNITS[self.quantity_kind]

    @property
    def si_factor(self) -> Fraction:
        return conversion_factor(self.unit, self.reference_unit)

    def to_si(self, value: Any) -> Any:
        factor = self.si_factor
        return value * factor.numerator / factor.denominator

    def from_si(self, value: Any) -> Any:
        factor = self.si_factor
        return value * factor.denominator / factor.numerator

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "quantity_kind": self.quantity_kind,
            "unit": self.unit.to_dict(),
            "axes": list(self.axes),
            "sign_convention": self.sign_convention,
            "support_association": self.support_association,
            "reference_configuration": self.reference_configuration,
            "quantity_id": self.quantity_id,
            "compatibility_id": self.compatibility_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GeophysicalQuantity:
        expected = {
            "name",
            "quantity_kind",
            "unit",
            "axes",
            "sign_convention",
            "support_association",
            "reference_configuration",
            "quantity_id",
            "compatibility_id",
        }
        if set(payload) != expected:
            raise ValueError("Quantity descriptor must use exactly the canonical fields.")
        quantity = cls(
            payload["name"],
            payload["quantity_kind"],
            UnitDefinition.from_dict(payload["unit"]),
            axes=tuple(payload["axes"]),
            sign_convention=payload["sign_convention"],
            support_association=payload["support_association"],
            reference_configuration=payload["reference_configuration"],
        )
        if (
            quantity.quantity_id != payload["quantity_id"]
            or quantity.compatibility_id != payload["compatibility_id"]
        ):
            raise ValueError(
                "Quantity descriptor fingerprint does not match its content."
            )
        return quantity


__all__ = ["GeophysicalQuantity"]
