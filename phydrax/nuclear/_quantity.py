#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Nuclear quantity meanings built on the shared measurement substrate."""

from __future__ import annotations

from types import MappingProxyType

from ..measurement import (
    QuantitySpec,
    RadiationQuantityKind,
    resolve_quantity,
    resolve_radiation_quantity,
)
from ..units import (
    derived_unit,
    JOULE,
    KELVIN,
    KILOGRAM,
    METER,
    MOLE,
    ONE,
    SECOND,
    UnitDefinition,
)


_AREA = derived_unit("m2-nuclear", ((METER, 2),))
_VOLUME = derived_unit("m3-nuclear", ((METER, 3),))
_REFERENCE_UNITS = MappingProxyType(
    {
        "amount": MOLE,
        "cross_section": _AREA,
        "energy": JOULE,
        "fraction": ONE,
        "mass_density": derived_unit("kg/m3-nuclear", ((KILOGRAM, 1), (METER, -3))),
        "number_density": derived_unit("1/m3-nuclear", ((METER, -3),)),
        "particle_source_density": derived_unit(
            "1/m3/s-nuclear", ((METER, -3), (SECOND, -1))
        ),
        "power": derived_unit("W-nuclear", ((JOULE, 1), (SECOND, -1))),
        "power_density": derived_unit(
            "W/m3-nuclear", ((JOULE, 1), (SECOND, -1), (METER, -3))
        ),
        "reaction_rate_density": derived_unit(
            "1/m3/s-reaction", ((METER, -3), (SECOND, -1))
        ),
        "reactivity": derived_unit("m3/s-nuclear", ((METER, 3), (SECOND, -1))),
        "scalar_flux": derived_unit("1/m2/s-nuclear", ((METER, -2), (SECOND, -1))),
        "temperature": KELVIN,
        "volume": _VOLUME,
    }
)

_ACTIVITY_KINDS = frozenset(
    {
        RadiationQuantityKind.ACTIVITY.value,
        RadiationQuantityKind.ACTIVITY_CONCENTRATION.value,
        RadiationQuantityKind.TIME_INTEGRATED_ACTIVITY.value,
        RadiationQuantityKind.TIME_INTEGRATED_ACTIVITY_CONCENTRATION.value,
    }
)


def resolve_nuclear_quantity(
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
    """Resolve a nuclear quantity against its exact reference semantics."""

    shared_kind = (
        quantity_kind.value
        if isinstance(quantity_kind, RadiationQuantityKind)
        else quantity_kind
    )
    if shared_kind in _ACTIVITY_KINDS:
        return resolve_radiation_quantity(
            name,
            shared_kind,
            unit,
            axes=axes,
            sign_convention=sign_convention,
            support_association=support_association,
            reference_configuration=reference_configuration,
        )

    return resolve_quantity(
        domain="nuclear",
        reference_units=_REFERENCE_UNITS,
        name=name,
        quantity_kind=shared_kind,
        unit=unit,
        axes=axes,
        sign_convention=sign_convention,
        support_association=support_association,
        reference_configuration=reference_configuration,
    )


__all__ = ["resolve_nuclear_quantity"]
