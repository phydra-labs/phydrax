#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

from jaxtyping import ArrayLike

from phydrax.domain import DomainComponent, DomainFunction

from ..operators.differential import (
    directional_derivative,
    dt,
    level_set_curvature,
    level_set_gradient_norm,
    level_set_normal,
    level_set_normal_velocity,
)
from ..operators.linalg import einsum
from ._base import Residual
from .boundary import _condition_value, ConditionValue


def InterfaceValueJump(
    inside_field: str,
    outside_field: str,
    on: DomainComponent,
    /,
    *,
    jump: ConditionValue | None = None,
    label: str | None = None,
) -> Residual:
    """Prescribe ``outside - inside = jump`` on an implicit interface."""

    target = _condition_value(jump, on, 0.0)
    return Residual(
        (inside_field, outside_field),
        on,
        lambda inside, outside: outside - inside - target,
        label=label,
    )


def InterfaceFluxJump(
    inside_field: str,
    outside_field: str,
    level_set_field: str,
    on: DomainComponent,
    /,
    *,
    inside_conductivity: DomainFunction | ArrayLike,
    outside_conductivity: DomainFunction | ArrayLike,
    jump: ConditionValue | None = None,
    spatial_var: str = "x",
    mode: Literal["reverse", "forward"] = "reverse",
    gradient_floor: float = 1.0e-12,
    label: str | None = None,
) -> Residual:
    """Prescribe the outward conductive-flux jump ``q_out - q_in``."""

    target = _condition_value(jump, on, 0.0)

    def residual(inside, outside, level_set):
        normal = level_set_normal(
            level_set,
            var=spatial_var,
            mode=mode,
            gradient_floor=gradient_floor,
        )
        inside_normal = directional_derivative(
            inside,
            normal,
            var=spatial_var,
            mode=mode,
        )
        outside_normal = directional_derivative(
            outside,
            normal,
            var=spatial_var,
            mode=mode,
        )
        inside_flux = -inside_conductivity * inside_normal
        outside_flux = -outside_conductivity * outside_normal
        return outside_flux - inside_flux - target

    return Residual(
        (inside_field, outside_field, level_set_field),
        on,
        residual,
        label=label,
    )


def InterfaceKinematic(
    level_set_field: str,
    normal_speed_field: str,
    on: DomainComponent,
    /,
    *,
    spatial_var: str = "x",
    time_var: str = "t",
    mode: Literal["reverse", "forward"] = "reverse",
    label: str | None = None,
) -> Residual:
    """Enforce ``partial_t(phi) + V_n |grad(phi)| = 0``."""

    def residual(level_set, normal_speed):
        return dt(level_set, var=time_var, mode=mode) + normal_speed * (
            level_set_gradient_norm(level_set, var=spatial_var, mode=mode)
        )

    return Residual(
        (level_set_field, normal_speed_field),
        on,
        residual,
        label=label,
    )


def StefanBalance(
    inside_temperature: str,
    outside_temperature: str,
    level_set_field: str,
    on: DomainComponent,
    /,
    *,
    inside_conductivity: DomainFunction | ArrayLike,
    outside_conductivity: DomainFunction | ArrayLike,
    volumetric_latent_heat: DomainFunction | ArrayLike,
    spatial_var: str = "x",
    time_var: str = "t",
    mode: Literal["reverse", "forward"] = "reverse",
    gradient_floor: float = 1.0e-12,
    label: str | None = None,
) -> Residual:
    """Enforce the two-phase Stefan balance under negative-inside convention.

    The residual is ``rho_L V_n + k_in d_n T_in - k_out d_n T_out``.
    """

    def residual(inside, outside, level_set):
        normal = level_set_normal(
            level_set,
            var=spatial_var,
            mode=mode,
            gradient_floor=gradient_floor,
        )
        speed = level_set_normal_velocity(
            level_set,
            spatial_var=spatial_var,
            time_var=time_var,
            mode=mode,
            gradient_floor=gradient_floor,
        )
        inside_gradient = directional_derivative(
            inside,
            normal,
            var=spatial_var,
            mode=mode,
        )
        outside_gradient = directional_derivative(
            outside,
            normal,
            var=spatial_var,
            mode=mode,
        )
        conductive_drive = (
            inside_conductivity * inside_gradient
            - outside_conductivity * outside_gradient
        )
        return volumetric_latent_heat * speed + conductive_drive

    return Residual(
        (inside_temperature, outside_temperature, level_set_field),
        on,
        residual,
        label=label,
    )


def YoungLaplaceJump(
    inside_pressure: str,
    outside_pressure: str,
    level_set_field: str,
    on: DomainComponent,
    /,
    *,
    surface_tension: DomainFunction | ArrayLike,
    spatial_var: str = "x",
    mode: Literal["reverse", "forward"] = "reverse",
    gradient_floor: float = 1.0e-12,
    label: str | None = None,
) -> Residual:
    """Enforce ``p_inside - p_outside = surface_tension * curvature``."""

    def residual(inside, outside, level_set):
        curvature = level_set_curvature(
            level_set,
            var=spatial_var,
            mode=mode,
            gradient_floor=gradient_floor,
        )
        return inside - outside - surface_tension * curvature

    return Residual(
        (inside_pressure, outside_pressure, level_set_field),
        on,
        residual,
        label=label,
    )


def GibbsThomson(
    temperature_field: str,
    level_set_field: str,
    on: DomainComponent,
    /,
    *,
    melting_temperature: ConditionValue,
    curvature_coefficient: DomainFunction | ArrayLike,
    spatial_var: str = "x",
    mode: Literal["reverse", "forward"] = "reverse",
    gradient_floor: float = 1.0e-12,
    label: str | None = None,
) -> Residual:
    """Enforce ``T = T_m - curvature_coefficient * curvature``."""

    target = _condition_value(melting_temperature, on, 0.0)

    def residual(temperature, level_set):
        curvature = level_set_curvature(
            level_set,
            var=spatial_var,
            mode=mode,
            gradient_floor=gradient_floor,
        )
        return temperature - target + curvature_coefficient * curvature

    return Residual(
        (temperature_field, level_set_field),
        on,
        residual,
        label=label,
    )


def InterfaceTractionJump(
    inside_stress: str,
    outside_stress: str,
    level_set_field: str,
    on: DomainComponent,
    /,
    *,
    surface_force: ConditionValue | None = None,
    spatial_var: str = "x",
    mode: Literal["reverse", "forward"] = "reverse",
    gradient_floor: float = 1.0e-12,
    label: str | None = None,
) -> Residual:
    """Prescribe ``traction_outside - traction_inside = surface_force``."""

    target = _condition_value(surface_force, on, 0.0)

    def residual(inside, outside, level_set):
        normal = level_set_normal(
            level_set,
            var=spatial_var,
            mode=mode,
            gradient_floor=gradient_floor,
        )
        inside_traction = einsum("...ij,...j->...i", inside, normal)
        outside_traction = einsum("...ij,...j->...i", outside, normal)
        return outside_traction - inside_traction - target

    return Residual(
        (inside_stress, outside_stress, level_set_field),
        on,
        residual,
        label=label,
    )


def LevelSetEikonal(
    level_set_field: str,
    on: DomainComponent,
    /,
    *,
    spatial_var: str = "x",
    mode: Literal["reverse", "forward"] = "reverse",
    label: str | None = None,
) -> Residual:
    """Enforce the signed-distance gauge ``|grad(phi)| = 1``."""

    return Residual(
        level_set_field,
        on,
        lambda level_set: (
            level_set_gradient_norm(
                level_set,
                var=spatial_var,
                mode=mode,
            )
            - 1.0
        ),
        label=label,
    )


__all__ = [
    "GibbsThomson",
    "InterfaceFluxJump",
    "InterfaceKinematic",
    "InterfaceTractionJump",
    "InterfaceValueJump",
    "LevelSetEikonal",
    "StefanBalance",
    "YoungLaplaceJump",
]
