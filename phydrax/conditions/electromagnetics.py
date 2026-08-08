#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from jaxtyping import ArrayLike

from phydrax.domain import DomainComponent, DomainFunction

from ..operators.differential import tangential_component
from ._base import Residual
from ._field_ops import cross, dot
from .boundary import _condition_value, ConditionValue


def PEC(
    electric_field: str,
    on: DomainComponent,
    /,
    *,
    var: str = "x",
    label: str | None = None,
) -> Residual:
    """Perfect electric conductor: zero tangential electric field."""
    return Residual(
        electric_field,
        on,
        lambda electric: tangential_component(electric, on, var=var),
        label=label,
    )


def PMC(
    magnetic_field: str,
    on: DomainComponent,
    /,
    *,
    var: str = "x",
    label: str | None = None,
) -> Residual:
    """Perfect magnetic conductor: zero tangential magnetic field."""
    return Residual(
        magnetic_field,
        on,
        lambda magnetic: tangential_component(magnetic, on, var=var),
        label=label,
    )


def Impedance(
    magnetic_field: str,
    electric_field: str,
    on: DomainComponent,
    /,
    *,
    admittance: DomainFunction | ArrayLike,
    var: str = "x",
    label: str | None = None,
) -> Residual:
    """Leontovich relation ``normal × H = admittance E_t``."""
    normal = on.normal(var=var)

    def residual(
        magnetic: DomainFunction,
        electric: DomainFunction,
        /,
    ) -> DomainFunction:
        tangential_electric = tangential_component(electric, on, var=var)
        return cross(normal, magnetic) - admittance * tangential_electric

    return Residual(
        (magnetic_field, electric_field),
        on,
        residual,
        label=label,
    )


def ElectricSurfaceCharge(
    electric_field: str,
    on: DomainComponent,
    /,
    *,
    permittivity: DomainFunction | ArrayLike,
    surface_charge: ConditionValue,
    var: str = "x",
    label: str | None = None,
) -> Residual:
    """Normal electric displacement equal to prescribed surface charge."""
    normal = on.normal(var=var)
    target = _condition_value(surface_charge, on, 0.0)
    return Residual(
        electric_field,
        on,
        lambda electric: permittivity * dot(electric, normal) - target,
        label=label,
    )


def MagneticSurfaceCurrent(
    magnetic_field: str,
    on: DomainComponent,
    /,
    *,
    surface_current: ConditionValue,
    var: str = "x",
    label: str | None = None,
) -> Residual:
    """Tangential magnetic field jump equal to surface current."""
    normal = on.normal(var=var)
    target = _condition_value(surface_current, on, 0.0)
    return Residual(
        magnetic_field,
        on,
        lambda magnetic: cross(normal, magnetic) - target,
        label=label,
    )


def InterfaceTangentialEContinuity(
    electric_field_1: str,
    electric_field_2: str,
    on: DomainComponent,
    /,
    *,
    var: str = "x",
    label: str | None = None,
) -> Residual:
    """Continuity of tangential electric field across an interface."""
    return Residual(
        (electric_field_1, electric_field_2),
        on,
        lambda first, second: tangential_component(second - first, on, var=var),
        label=label,
    )


def InterfaceNormalDJump(
    electric_field_1: str,
    electric_field_2: str,
    on: DomainComponent,
    /,
    *,
    permittivity_1: DomainFunction | ArrayLike,
    permittivity_2: DomainFunction | ArrayLike,
    surface_charge: ConditionValue | None = None,
    var: str = "x",
    label: str | None = None,
) -> Residual:
    """Normal electric-displacement jump across an interface."""
    normal = on.normal(var=var)
    target = _condition_value(surface_charge, on, 0.0)

    def residual(
        first: DomainFunction,
        second: DomainFunction,
        /,
    ) -> DomainFunction:
        return (
            permittivity_2 * dot(second, normal)
            - permittivity_1 * dot(first, normal)
            - target
        )

    return Residual(
        (electric_field_1, electric_field_2),
        on,
        residual,
        label=label,
    )


def InterfaceTangentialHJump(
    magnetic_field_1: str,
    magnetic_field_2: str,
    on: DomainComponent,
    /,
    *,
    surface_current: ConditionValue | None = None,
    var: str = "x",
    label: str | None = None,
) -> Residual:
    """Tangential magnetic-field jump across an interface."""
    normal = on.normal(var=var)
    target = _condition_value(surface_current, on, 0.0)
    return Residual(
        (magnetic_field_1, magnetic_field_2),
        on,
        lambda first, second: cross(normal, second - first) - target,
        label=label,
    )


def InterfaceNormalBContinuity(
    magnetic_field_1: str,
    magnetic_field_2: str,
    on: DomainComponent,
    /,
    *,
    permeability_1: DomainFunction | ArrayLike,
    permeability_2: DomainFunction | ArrayLike,
    var: str = "x",
    label: str | None = None,
) -> Residual:
    """Continuity of normal magnetic flux density across an interface."""
    normal = on.normal(var=var)
    return Residual(
        (magnetic_field_1, magnetic_field_2),
        on,
        lambda first, second: (
            permeability_2 * dot(second, normal)
            - permeability_1 * dot(first, normal)
        ),
        label=label,
    )


__all__ = [
    "ElectricSurfaceCharge",
    "Impedance",
    "InterfaceNormalBContinuity",
    "InterfaceNormalDJump",
    "InterfaceTangentialEContinuity",
    "InterfaceTangentialHJump",
    "MagneticSurfaceCurrent",
    "PEC",
    "PMC",
]
