#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Nuclear quantity meanings built on the shared measurement substrate."""

from __future__ import annotations

from types import MappingProxyType

from ..measurement import QuantitySpec, resolve_quantity
from ..units import (
    BECQUEREL,
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
        "activity": BECQUEREL,
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


def resolve_nuclear_quantity(
    name: str,
    quantity_kind: str,
    unit: UnitDefinition,
    /,
    *,
    axes: tuple[str, ...] = (),
    sign_convention: str = "nonnegative",
    support_association: str = "unspecified",
    reference_configuration: str = "absolute",
) -> QuantitySpec:
    """Resolve a nuclear quantity against its exact reference semantics."""

    return resolve_quantity(
        domain="nuclear",
        reference_units=_REFERENCE_UNITS,
        name=name,
        quantity_kind=quantity_kind,
        unit=unit,
        axes=axes,
        sign_convention=sign_convention,
        support_association=support_association,
        reference_configuration=reference_configuration,
    )


__all__ = ["resolve_nuclear_quantity"]
