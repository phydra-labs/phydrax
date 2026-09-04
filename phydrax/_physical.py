#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import equinox as eqx

from ._fingerprint import canonical_fingerprint
from ._strict import StrictModule
from ._trainable import NonTrainableState
from .units import (
    derived_unit,
    KILOGRAM,
    LENGTH,
    MASS,
    METER,
    SECOND,
    TIME,
    UnitDefinition,
)


LengthCoordinateKind = Literal["physical", "comoving", "code"]


class DimensionalScaleContract(StrictModule, NonTrainableState):
    """Shared exact length, mass, time, and coordinate-kind identity."""

    length_unit: UnitDefinition = eqx.field(static=True)
    mass_unit: UnitDefinition = eqx.field(static=True)
    time_unit: UnitDefinition = eqx.field(static=True)
    length_coordinate_kind: LengthCoordinateKind = eqx.field(static=True)
    velocity_unit: UnitDefinition = eqx.field(static=True)
    acceleration_unit: UnitDefinition = eqx.field(static=True)
    gravitational_parameter_unit: UnitDefinition = eqx.field(static=True)
    gravitational_constant_unit: UnitDefinition = eqx.field(static=True)
    hubble_unit: UnitDefinition = eqx.field(static=True)
    wavenumber_unit: UnitDefinition = eqx.field(static=True)
    power_spectrum_unit: UnitDefinition = eqx.field(static=True)
    potential_unit: UnitDefinition = eqx.field(static=True)
    canonical_momentum_unit: UnitDefinition = eqx.field(static=True)
    scale_id: str = eqx.field(static=True)

    def __init__(
        self,
        length_unit: UnitDefinition,
        mass_unit: UnitDefinition,
        time_unit: UnitDefinition,
        /,
        *,
        length_coordinate_kind: LengthCoordinateKind = "physical",
    ):
        if not all(
            isinstance(unit, UnitDefinition)
            for unit in (length_unit, mass_unit, time_unit)
        ):
            raise TypeError("Dimensional scale units must be UnitDefinition values.")
        if (
            length_unit.dimension != LENGTH
            or mass_unit.dimension != MASS
            or time_unit.dimension != TIME
        ):
            raise ValueError(
                "Dimensional scale units must have length, mass, and time dimensions."
            )
        if (
            len(
                {
                    length_unit.reference_system_id,
                    mass_unit.reference_system_id,
                    time_unit.reference_system_id,
                }
            )
            != 1
        ):
            raise ValueError(
                "Dimensional scale units must share one explicit reference system."
            )
        kind = str(length_coordinate_kind).strip()
        if kind not in ("physical", "comoving", "code"):
            raise ValueError("Dimensional scale coordinate kind is invalid.")

        length_symbol = length_unit.symbol
        mass_symbol = mass_unit.symbol
        time_symbol = time_unit.symbol
        self.length_unit = length_unit
        self.mass_unit = mass_unit
        self.time_unit = time_unit
        self.length_coordinate_kind = kind
        self.velocity_unit = derived_unit(
            f"{length_symbol}/{time_symbol}",
            ((length_unit, 1), (time_unit, -1)),
        )
        self.acceleration_unit = derived_unit(
            f"{length_symbol}/{time_symbol}^2",
            ((length_unit, 1), (time_unit, -2)),
        )
        self.gravitational_parameter_unit = derived_unit(
            f"{length_symbol}^3/{time_symbol}^2",
            ((length_unit, 3), (time_unit, -2)),
        )
        self.gravitational_constant_unit = derived_unit(
            f"{length_symbol}^3/({mass_symbol}*{time_symbol}^2)",
            ((length_unit, 3), (mass_unit, -1), (time_unit, -2)),
        )
        self.hubble_unit = derived_unit(f"1/{time_symbol}", ((time_unit, -1),))
        self.wavenumber_unit = derived_unit(f"1/{length_symbol}", ((length_unit, -1),))
        self.power_spectrum_unit = derived_unit(f"{length_symbol}^3", ((length_unit, 3),))
        self.potential_unit = derived_unit(
            f"{length_symbol}^2/{time_symbol}^2",
            ((length_unit, 2), (time_unit, -2)),
        )
        self.canonical_momentum_unit = derived_unit(
            f"{mass_symbol}*{length_symbol}/{time_symbol}",
            ((mass_unit, 1), (length_unit, 1), (time_unit, -1)),
        )
        self.scale_id = canonical_fingerprint(
            {
                "kind": "dimensional-scale-contract",
                "length_unit": length_unit.unit_id,
                "mass_unit": mass_unit.unit_id,
                "time_unit": time_unit.unit_id,
                "length_coordinate_kind": kind,
            }
        )

    @classmethod
    def si(cls) -> DimensionalScaleContract:
        return cls(METER, KILOGRAM, SECOND, length_coordinate_kind="physical")

    def to_dict(self) -> dict[str, object]:
        return {
            "length_unit": self.length_unit.to_dict(),
            "mass_unit": self.mass_unit.to_dict(),
            "time_unit": self.time_unit.to_dict(),
            "length_coordinate_kind": self.length_coordinate_kind,
            "scale_id": self.scale_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DimensionalScaleContract:
        if not isinstance(payload, Mapping):
            raise TypeError("Dimensional scale payload must be a mapping.")
        expected = {
            "length_unit",
            "mass_unit",
            "time_unit",
            "length_coordinate_kind",
            "scale_id",
        }
        if set(payload) != expected:
            raise ValueError("Dimensional scale payload must use the canonical fields.")
        length_payload = payload.get("length_unit")
        mass_payload = payload.get("mass_unit")
        time_payload = payload.get("time_unit")
        if (
            not isinstance(length_payload, Mapping)
            or not isinstance(mass_payload, Mapping)
            or not isinstance(time_payload, Mapping)
        ):
            raise TypeError("Dimensional scale unit payloads must be mappings.")
        coordinate_kind = payload.get("length_coordinate_kind")
        if not isinstance(coordinate_kind, str):
            raise TypeError("Dimensional scale coordinate kind must be a string.")
        scale = cls(
            UnitDefinition.from_dict(length_payload),
            UnitDefinition.from_dict(mass_payload),
            UnitDefinition.from_dict(time_payload),
            length_coordinate_kind=coordinate_kind,
        )
        claimed_id = payload.get("scale_id")
        if not isinstance(claimed_id, str):
            raise TypeError("Dimensional scale payload scale_id must be a string.")
        if claimed_id != scale.scale_id:
            raise ValueError(
                "Dimensional scale payload fingerprint does not match its content."
            )
        return scale


__all__ = ["DimensionalScaleContract", "LengthCoordinateKind"]
