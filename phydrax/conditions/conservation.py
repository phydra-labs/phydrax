#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from jaxtyping import ArrayLike

from phydrax.domain import DomainComponent, DomainFunction

from ..operators.differential import cauchy_stress
from ._base import Moment
from ._field_ops import cross, dot, matvec


def BoundaryCharge(
    displacement_field: str,
    on: DomainComponent,
    target: ArrayLike,
    /,
    *,
    var: str = "x",
    label: str | None = None,
) -> Moment:
    """Total outward electric-displacement flux."""
    normal = on.normal(var=var)
    return Moment(
        displacement_field,
        on,
        lambda displacement: dot(displacement, normal),
        target=target,
        label=label,
    )


def MagneticFlux(
    magnetic_flux_field: str,
    on: DomainComponent,
    /,
    *,
    target: ArrayLike = 0.0,
    var: str = "x",
    label: str | None = None,
) -> Moment:
    """Net outward magnetic flux, zero by default."""
    normal = on.normal(var=var)
    return Moment(
        magnetic_flux_field,
        on,
        lambda magnetic_flux: dot(magnetic_flux, normal),
        target=target,
        label=label,
    )


def FlowRate(
    velocity_field: str,
    on: DomainComponent,
    target: ArrayLike,
    /,
    *,
    var: str = "x",
    label: str | None = None,
) -> Moment:
    """Volumetric flow rate through a boundary component."""
    normal = on.normal(var=var)
    return Moment(
        velocity_field,
        on,
        lambda velocity: dot(velocity, normal),
        target=target,
        label=label,
    )


def KineticEnergyFlux(
    velocity_field: str,
    on: DomainComponent,
    target: ArrayLike,
    /,
    *,
    var: str = "x",
    label: str | None = None,
) -> Moment:
    """Kinetic energy flux ``0.5 |velocity|² velocity·normal``."""
    normal = on.normal(var=var)
    return Moment(
        velocity_field,
        on,
        lambda velocity: 0.5
        * dot(velocity, velocity)
        * dot(velocity, normal),
        target=target,
        label=label,
    )


def TotalReaction(
    displacement_field: str,
    on: DomainComponent,
    target: ArrayLike,
    /,
    *,
    lambda_: DomainFunction | ArrayLike,
    mu: DomainFunction | ArrayLike,
    var: str = "x",
    label: str | None = None,
) -> Moment:
    """Total Cauchy traction over a solid boundary component."""
    normal = on.normal(var=var)

    def traction(displacement: DomainFunction, /) -> DomainFunction:
        stress = cauchy_stress(
            displacement,
            lambda_=lambda_,
            mu=mu,
            var=var,
        )
        return matvec(stress, normal)

    return Moment(
        displacement_field,
        on,
        traction,
        target=target,
        label=label,
    )


def PressureIntegral(
    pressure_field: str,
    on: DomainComponent,
    target: ArrayLike,
    /,
    *,
    label: str | None = None,
) -> Moment:
    """Integral of pressure over a component."""
    return Moment(
        pressure_field,
        on,
        lambda pressure: pressure,
        target=target,
        label=label,
    )


def PoyntingFlux(
    electric_field: str,
    magnetic_field: str,
    on: DomainComponent,
    target: ArrayLike,
    /,
    *,
    var: str = "x",
    label: str | None = None,
) -> Moment:
    """Total electromagnetic power flux ``(electric × magnetic)·normal``."""
    normal = on.normal(var=var)
    return Moment(
        (electric_field, magnetic_field),
        on,
        lambda electric, magnetic: dot(cross(electric, magnetic), normal),
        target=target,
        label=label,
    )


__all__ = [
    "BoundaryCharge",
    "FlowRate",
    "KineticEnergyFlux",
    "MagneticFlux",
    "PoyntingFlux",
    "PressureIntegral",
    "TotalReaction",
]
