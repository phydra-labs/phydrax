#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from typing import Literal

from jaxtyping import ArrayLike

from phydrax.domain import DomainComponent, DomainFunction

from ..operators.differential import directional_derivative, grad
from ..operators.linalg import einsum
from ._base import Residual
from ._field_ops import constant_field, dot, outer_scalar_vector
from .boundary import _condition_value, ConditionValue


def SymmetryVelocity(
    velocity_field: str,
    on: DomainComponent,
    /,
    *,
    var: str = "x",
    label: str | None = None,
) -> Residual:
    """No-penetration symmetry condition ``velocity · normal = 0``."""
    normal = on.normal(var=var)
    return Residual(
        velocity_field,
        on,
        lambda velocity: dot(velocity, normal),
        label=label,
    )


def NoPenetration(
    velocity_field: str,
    on: DomainComponent,
    /,
    *,
    wall_normal_velocity: ConditionValue | None = None,
    wall_velocity: ConditionValue | None = None,
    var: str = "x",
    label: str | None = None,
) -> Residual:
    """Impermeability condition for stationary or moving walls."""
    if wall_velocity is not None and wall_normal_velocity is not None:
        raise ValueError(
            "Provide either wall_velocity or wall_normal_velocity, not both."
        )
    normal = on.normal(var=var)
    if wall_velocity is None:
        target = _condition_value(wall_normal_velocity, on, 0.0)
    else:
        velocity = _condition_value(wall_velocity, on, 0.0)
        if not isinstance(velocity, DomainFunction):
            velocity = constant_field(velocity, normal)
        target = dot(velocity, normal)
    return Residual(
        velocity_field,
        on,
        lambda velocity: dot(velocity, normal) - target,
        label=label,
    )


def SlipWall(
    velocity_field: str,
    pressure_field: str,
    on: DomainComponent,
    /,
    *,
    viscosity: DomainFunction | ArrayLike,
    var: str = "x",
    mode: Literal["reverse", "forward"] = "reverse",
    label: str | None = None,
) -> Residual:
    """Free-slip condition defined by zero tangential traction."""
    normal = on.normal(var=var)

    def residual(
        velocity: DomainFunction,
        pressure: DomainFunction,
        /,
    ) -> DomainFunction:
        velocity_gradient = grad(velocity, var=var, mode=mode)
        viscous_stress = viscosity * (velocity_gradient + velocity_gradient.T)
        traction = -pressure * normal + einsum("...ij,...j->...i", viscous_stress, normal)
        normal_traction = outer_scalar_vector(dot(traction, normal), normal)
        return traction - normal_traction

    return Residual(
        (velocity_field, pressure_field),
        on,
        residual,
        label=label,
    )


def ZeroNormalGradientVelocity(
    velocity_field: str,
    on: DomainComponent,
    /,
    *,
    var: str = "x",
    mode: Literal["reverse", "forward"] = "reverse",
    label: str | None = None,
) -> Residual:
    """Componentwise zero normal derivative of velocity."""
    normal = on.normal(var=var)
    return Residual(
        velocity_field,
        on,
        lambda velocity: directional_derivative(
            velocity,
            normal,
            var=var,
            mode=mode,
        ),
        label=label,
    )


__all__ = [
    "NoPenetration",
    "SlipWall",
    "SymmetryVelocity",
    "ZeroNormalGradientVelocity",
]
