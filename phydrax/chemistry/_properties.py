#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Typed molecular electronic-property requests."""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum

import equinox as eqx

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


class ElectronicProperty(StrEnum):
    ENERGY = "energy"
    FORCES = "forces"
    HESSIAN = "hessian"
    DIPOLE = "dipole"


class ElectronicPropertyRequest(StrictModule, NonTrainableState):
    """One exact, sorted set of required electronic observables."""

    properties: tuple[ElectronicProperty, ...] = eqx.field(static=True)
    request_id: str = eqx.field(static=True)

    def __init__(self, properties: Sequence[ElectronicProperty], /):
        values = tuple(properties)
        if not values or any(not isinstance(value, ElectronicProperty) for value in values):
            raise TypeError("properties must contain ElectronicProperty values.")
        normalized = tuple(sorted(set(values), key=lambda value: value.value))
        if ElectronicProperty.ENERGY not in normalized:
            raise ValueError("Every electronic property request must include energy.")
        force_coupled = {
            ElectronicProperty.HESSIAN,
            ElectronicProperty.DIPOLE,
        }
        if force_coupled.intersection(normalized) and ElectronicProperty.FORCES not in normalized:
            raise ValueError("Hessian and dipole requests must also request forces.")
        self.properties = normalized
        self.request_id = canonical_fingerprint(
            {
                "kind": "electronic-property-request",
                "properties": [value.value for value in normalized],
            }
        )

    @classmethod
    def energy(cls) -> ElectronicPropertyRequest:
        return cls((ElectronicProperty.ENERGY,))

    @classmethod
    def energy_and_forces(cls) -> ElectronicPropertyRequest:
        return cls((ElectronicProperty.ENERGY, ElectronicProperty.FORCES))

    @classmethod
    def energy_forces_and_hessian(cls) -> ElectronicPropertyRequest:
        return cls(
            (
                ElectronicProperty.ENERGY,
                ElectronicProperty.FORCES,
                ElectronicProperty.HESSIAN,
            )
        )

    def requires(self, property_: ElectronicProperty, /) -> bool:
        if not isinstance(property_, ElectronicProperty):
            raise TypeError("property_ must be ElectronicProperty.")
        return property_ in self.properties


__all__ = ["ElectronicProperty", "ElectronicPropertyRequest"]
