#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from typing import Literal

from jaxtyping import ArrayLike

from phydrax.domain import DomainComponent, DomainFunction

from ..operators.differential import directional_derivative
from ._base import Residual
from .boundary import _condition_value, ConditionValue


def HeatFlux(
    temperature_field: str,
    on: DomainComponent,
    /,
    *,
    conductivity: DomainFunction | ArrayLike,
    flux: ConditionValue | None = None,
    var: str = "x",
    mode: Literal["reverse", "forward"] = "reverse",
    label: str | None = None,
) -> Residual:
    """Prescribed outward conductive flux ``-k ∂T/∂n = flux``."""
    normal = on.normal(var=var)
    target = _condition_value(flux, on, 0.0)

    def residual(temperature: DomainFunction, /) -> DomainFunction:
        derivative = directional_derivative(
            temperature,
            normal,
            var=var,
            mode=mode,
        )
        return -(conductivity * derivative) - target

    return Residual(temperature_field, on, residual, label=label)


def Convection(
    temperature_field: str,
    on: DomainComponent,
    /,
    *,
    heat_transfer_coefficient: DomainFunction | ArrayLike,
    conductivity: DomainFunction | ArrayLike,
    ambient_temperature: ConditionValue | None = None,
    var: str = "x",
    mode: Literal["reverse", "forward"] = "reverse",
    label: str | None = None,
) -> Residual:
    """Convective boundary condition ``-k ∂T/∂n = h(T - T∞)``."""
    normal = on.normal(var=var)
    ambient = _condition_value(ambient_temperature, on, 0.0)

    def residual(temperature: DomainFunction, /) -> DomainFunction:
        derivative = directional_derivative(
            temperature,
            normal,
            var=var,
            mode=mode,
        )
        return -(conductivity * derivative) - heat_transfer_coefficient * (
            temperature - ambient
        )

    return Residual(temperature_field, on, residual, label=label)


__all__ = ["Convection", "HeatFlux"]
