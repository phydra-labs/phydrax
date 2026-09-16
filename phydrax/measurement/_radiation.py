#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Canonical radiation quantity meanings on the shared measurement substrate."""

from __future__ import annotations

from enum import StrEnum
from types import MappingProxyType

from ..units import (
    BECQUEREL,
    BECQUEREL_PER_CUBIC_METER,
    BECQUEREL_SECOND,
    BECQUEREL_SECOND_PER_CUBIC_METER,
    GRAY,
    GRAY_PER_SECOND,
    INVERSE_SQUARE_METER,
    JOULE,
    JOULE_PER_METER,
    JOULE_PER_SQUARE_METER,
    ONE,
    UnitDefinition,
)
from ._quantity import QuantitySpec, resolve_quantity


class RadiationQuantityKind(StrEnum):
    """Radiation meanings that remain distinct even when their dimensions agree."""

    DEPOSITED_ENERGY = "deposited_energy"
    ABSORBED_DOSE = "absorbed_dose"
    DOSE_TO_WATER = "dose_to_water"
    DOSE_TO_MEDIUM = "dose_to_medium"
    KERMA = "kerma"
    DOSE_RATE = "dose_rate"
    RELATIVE_DOSE = "relative_dose"
    PARTICLE_FLUENCE = "particle_fluence"
    ENERGY_FLUENCE = "energy_fluence"
    LET = "let"
    LINEAL_ENERGY = "lineal_energy"
    ACTIVITY = "activity"
    ACTIVITY_CONCENTRATION = "activity_concentration"
    TIME_INTEGRATED_ACTIVITY = "time_integrated_activity"
    TIME_INTEGRATED_ACTIVITY_CONCENTRATION = "time_integrated_activity_concentration"


_REFERENCE_UNITS = MappingProxyType(
    {
        RadiationQuantityKind.DEPOSITED_ENERGY.value: JOULE,
        RadiationQuantityKind.ABSORBED_DOSE.value: GRAY,
        RadiationQuantityKind.DOSE_TO_WATER.value: GRAY,
        RadiationQuantityKind.DOSE_TO_MEDIUM.value: GRAY,
        RadiationQuantityKind.KERMA.value: GRAY,
        RadiationQuantityKind.DOSE_RATE.value: GRAY_PER_SECOND,
        RadiationQuantityKind.RELATIVE_DOSE.value: ONE,
        RadiationQuantityKind.PARTICLE_FLUENCE.value: INVERSE_SQUARE_METER,
        RadiationQuantityKind.ENERGY_FLUENCE.value: JOULE_PER_SQUARE_METER,
        RadiationQuantityKind.LET.value: JOULE_PER_METER,
        RadiationQuantityKind.LINEAL_ENERGY.value: JOULE_PER_METER,
        RadiationQuantityKind.ACTIVITY.value: BECQUEREL,
        RadiationQuantityKind.ACTIVITY_CONCENTRATION.value: (BECQUEREL_PER_CUBIC_METER),
        RadiationQuantityKind.TIME_INTEGRATED_ACTIVITY.value: BECQUEREL_SECOND,
        RadiationQuantityKind.TIME_INTEGRATED_ACTIVITY_CONCENTRATION.value: (
            BECQUEREL_SECOND_PER_CUBIC_METER
        ),
    }
)
_KINDS_BY_VALUE = MappingProxyType({kind.value: kind for kind in RadiationQuantityKind})
_REFERENCE_REQUIRED = frozenset(
    {
        RadiationQuantityKind.ABSORBED_DOSE,
        RadiationQuantityKind.DOSE_TO_WATER,
        RadiationQuantityKind.DOSE_TO_MEDIUM,
        RadiationQuantityKind.KERMA,
        RadiationQuantityKind.DOSE_RATE,
        RadiationQuantityKind.RELATIVE_DOSE,
        RadiationQuantityKind.PARTICLE_FLUENCE,
        RadiationQuantityKind.ENERGY_FLUENCE,
        RadiationQuantityKind.LET,
        RadiationQuantityKind.LINEAL_ENERGY,
        RadiationQuantityKind.ACTIVITY,
        RadiationQuantityKind.ACTIVITY_CONCENTRATION,
        RadiationQuantityKind.TIME_INTEGRATED_ACTIVITY,
        RadiationQuantityKind.TIME_INTEGRATED_ACTIVITY_CONCENTRATION,
    }
)
_SUPPORT_REQUIRED = frozenset(
    {
        RadiationQuantityKind.DEPOSITED_ENERGY,
        RadiationQuantityKind.ABSORBED_DOSE,
        RadiationQuantityKind.DOSE_TO_WATER,
        RadiationQuantityKind.DOSE_TO_MEDIUM,
        RadiationQuantityKind.KERMA,
        RadiationQuantityKind.DOSE_RATE,
        RadiationQuantityKind.PARTICLE_FLUENCE,
        RadiationQuantityKind.ENERGY_FLUENCE,
        RadiationQuantityKind.ACTIVITY_CONCENTRATION,
        RadiationQuantityKind.TIME_INTEGRATED_ACTIVITY_CONCENTRATION,
    }
)
_GENERIC_REFERENCES = frozenset(
    {"", "absolute", "default", "generic", "unknown", "unspecified"}
)
_GENERIC_SUPPORTS = frozenset({"", "default", "generic", "unknown", "unspecified"})


def _radiation_kind(
    quantity_kind: RadiationQuantityKind | str, /
) -> RadiationQuantityKind:
    if isinstance(quantity_kind, RadiationQuantityKind):
        return quantity_kind
    if not isinstance(quantity_kind, str):
        raise TypeError("quantity_kind must be a RadiationQuantityKind or string.")
    if quantity_kind not in _KINDS_BY_VALUE:
        raise ValueError(f"Unsupported radiation quantity kind {quantity_kind!r}.")
    return _KINDS_BY_VALUE[quantity_kind]


def _require_specific(value: str, role: str, /, *, generic: frozenset[str]) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{role} must be a string.")
    if value.strip() != value or value.casefold() in generic:
        raise ValueError(f"{role} must declare a non-generic radiation meaning.")


def resolve_radiation_quantity(
    name: str,
    quantity_kind: RadiationQuantityKind | str,
    unit: UnitDefinition,
    /,
    *,
    axes: tuple[str, ...] = (),
    sign_convention: str = "nonnegative",
    support_association: str = "unspecified",
    reference_configuration: str = "absolute",
) -> QuantitySpec:
    """Resolve one radiation meaning with explicit score and support semantics."""

    kind = _radiation_kind(quantity_kind)
    if kind in _REFERENCE_REQUIRED:
        _require_specific(
            reference_configuration,
            "reference_configuration",
            generic=_GENERIC_REFERENCES,
        )
    if kind in _SUPPORT_REQUIRED:
        _require_specific(
            support_association,
            "support_association",
            generic=_GENERIC_SUPPORTS,
        )
    return resolve_quantity(
        domain="radiation",
        reference_units=_REFERENCE_UNITS,
        name=name,
        quantity_kind=kind.value,
        unit=unit,
        axes=axes,
        sign_convention=sign_convention,
        support_association=support_association,
        reference_configuration=reference_configuration,
    )


__all__ = ["RadiationQuantityKind", "resolve_radiation_quantity"]
