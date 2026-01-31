#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import jax.numpy as jnp
from jaxtyping import ArrayLike

from ..domain._components import DomainComponent, DomainComponentUnion
from ..domain._function import DomainFunction
from ..operators.differential import directional_derivative, grad
from ..operators.linalg import einsum
from ._functional import FunctionalConstraint
from ._interpolate import idw_interpolant
from ._pointset import PointSetConstraint


def _normal(
    component: DomainComponent | DomainComponentUnion, *, var: str
) -> DomainFunction:
    if isinstance(component, DomainComponentUnion):
        raise TypeError("Boundary normal requires a single DomainComponent, not a union.")
    return component.normal(var=var)


def _dot(a: DomainFunction, b: DomainFunction) -> DomainFunction:
    return einsum("...i,...i->...", a, b)


def _outer_scalar_vec(s: DomainFunction, v: DomainFunction) -> DomainFunction:
    return einsum("...,...i->...i", s, v)


def _interp_target(
    component: DomainComponent,
    points: Mapping[str, ArrayLike] | ArrayLike,
    values: ArrayLike,
) -> DomainFunction:
    arr = jnp.asarray(values)
    if arr.ndim == 0:
        return DomainFunction(
            domain=component.domain,
            deps=(),
            func=arr.reshape(()),
            metadata={},
        )
    return idw_interpolant(
        component.domain,
        anchors=points,
        values=arr,
        labels=component.domain.labels,
    )


def _coerce_value(
    value: Any,
    component: DomainComponent | DomainComponentUnion,
    /,
) -> DomainFunction | ArrayLike:
    if isinstance(value, DomainFunction):
        return value
    if callable(value):
        return DomainFunction(
            domain=component.domain, deps=component.domain.labels, func=value, metadata={}
        )
    return value


# CFD Boundary Constraints (Continuous)


def ContinuousSymmetryVelocityBoundaryConstraint(
    velocity_var: str,
    component: DomainComponent,
    /,
    *,
    var: str = "x",
    num_points: int | tuple[Any, ...],
    structure: Any,
    sampler: str = "latin_hypercube",
    weight: ArrayLike = 1.0,
    label: str | None = None,
    over: str | tuple[str, ...] | None = None,
    reduction: Literal["mean", "integral"] = "mean",
) -> FunctionalConstraint:
    r"""Symmetry (no-penetration) condition on a symmetry plane.

    Enforces $u\cdot n = 0$ on the boundary component.
    """
    n = _normal(component, var=var)

    def operator(u: DomainFunction, /) -> DomainFunction:
        return _dot(u, n)

    return FunctionalConstraint.from_operator(
        component=component,
        operator=operator,
        constraint_vars=velocity_var,
        num_points=num_points,
        structure=structure,
        sampler=sampler,
        weight=weight,
        label=label,
        over=over,
        reduction=reduction,
    )


def ContinuousNoPenetrationBoundaryConstraint(
    velocity_var: str,
    component: DomainComponent,
    /,
    *,
    wall_normal_velocity: DomainFunction | ArrayLike | None = None,
    wall_velocity: DomainFunction | ArrayLike | None = None,
    var: str = "x",
    num_points: int | tuple[Any, ...],
    structure: Any,
    sampler: str = "latin_hypercube",
    weight: ArrayLike = 1.0,
    label: str | None = None,
    over: str | tuple[str, ...] | None = None,
    reduction: Literal["mean", "integral"] = "mean",
) -> FunctionalConstraint:
    r"""No-penetration (impermeability) wall boundary condition.

    Enforces the normal-velocity condition

    $$
    u\cdot n = u_n,
    $$

    where `u_n` can be provided via `wall_normal_velocity` or derived from a full
    `wall_velocity` by taking its normal component.
    """
    n = _normal(component, var=var)
    if wall_velocity is not None and wall_normal_velocity is not None:
        raise ValueError(
            "Provide either wall_velocity or wall_normal_velocity, not both."
        )
    if wall_velocity is not None:
        wall_velocity = _coerce_value(wall_velocity, component)
        if isinstance(wall_velocity, DomainFunction):
            target = _dot(wall_velocity, n)
        else:
            target_vec = DomainFunction(
                domain=component.domain, deps=(), func=wall_velocity, metadata={}
            )
            target = _dot(target_vec, n)
    elif wall_normal_velocity is None:
        target = 0.0
    else:
        target = _coerce_value(wall_normal_velocity, component)

    def operator(u: DomainFunction, /) -> DomainFunction:
        return _dot(u, n) - target

    return FunctionalConstraint.from_operator(
        component=component,
        operator=operator,
        constraint_vars=velocity_var,
        num_points=num_points,
        structure=structure,
        sampler=sampler,
        weight=weight,
        label=label,
        over=over,
        reduction=reduction,
    )


def ContinuousSlipWallBoundaryConstraint(
    velocity_var: str,
    pressure_var: str,
    component: DomainComponent,
    /,
    *,
    var: str = "x",
    viscosity: DomainFunction | ArrayLike,
    mode: Literal["reverse", "forward"] = "reverse",
    num_points: int | tuple[Any, ...],
    structure: Any,
    sampler: str = "latin_hypercube",
    weight: ArrayLike = 1.0,
    label: str | None = None,
    over: str | tuple[str, ...] | None = None,
    reduction: Literal["mean", "integral"] = "mean",
) -> FunctionalConstraint:
    r"""Free-slip wall boundary condition via zero tangential traction.

    Forms the (pressure + viscous) traction

    $$
    t = -p\,n + \tau\,n,\qquad \tau = \mu(\nabla u + \nabla u^\top),
    $$

    and enforces that its tangential component vanishes:

    $$
    (I - n\otimes n)\,t = 0.
    $$
    """
    n = _normal(component, var=var)

    def operator(u: DomainFunction, p: DomainFunction, /) -> DomainFunction:
        g = grad(u, var=var, mode=mode)
        tau = viscosity * (g + g.T)
        traction = -p * n + einsum("...ij,...j->...i", tau, n)
        tn = _outer_scalar_vec(_dot(traction, n), n)
        return traction - tn

    return FunctionalConstraint.from_operator(
        component=component,
        operator=operator,
        constraint_vars=(velocity_var, pressure_var),
        num_points=num_points,
        structure=structure,
        sampler=sampler,
        weight=weight,
        label=label,
        over=over,
        reduction=reduction,
    )


# CFD Boundary Constraints (Discrete)


def DiscreteNoPenetrationBoundaryConstraint(
    velocity_var: str,
    component: DomainComponent,
    /,
    *,
    points: Mapping[str, ArrayLike] | ArrayLike,
    wall_normal_velocity: ArrayLike | None = None,
    var: str = "x",
    weight: ArrayLike = 1.0,
    label: str | None = None,
    reduction: Literal["mean", "sum"] = "mean",
) -> PointSetConstraint:
    r"""Discrete no-penetration constraint at explicit boundary anchor points.

    Enforces $u\cdot n = u_n$ (default $0$) at the provided points.
    """
    n = _normal(component, var=var)
    if wall_normal_velocity is None:
        target: DomainFunction | ArrayLike = 0.0
    else:
        target = _interp_target(component, points, wall_normal_velocity)

    def residual(functions: Mapping[str, DomainFunction], /) -> DomainFunction:
        u = functions[velocity_var]
        u_dot_n = _dot(u, n)
        return u_dot_n - target

    return PointSetConstraint.from_points(
        component=component,
        points=points,
        residual=residual,
        weight=weight,
        label=label,
        reduction=reduction,
    )


def DiscreteZeroNormalGradientVelocityBoundaryConstraint(
    velocity_var: str,
    component: DomainComponent,
    /,
    *,
    points: Mapping[str, ArrayLike] | ArrayLike,
    var: str = "x",
    mode: Literal["reverse", "forward"] = "reverse",
    weight: ArrayLike = 1.0,
    label: str | None = None,
    reduction: Literal["mean", "sum"] = "mean",
) -> PointSetConstraint:
    r"""Discrete zero normal-gradient condition for velocity at explicit anchor points.

    Enforces $\partial u/\partial n = 0$ componentwise at the provided points.
    """
    n = _normal(component, var=var)

    def residual(functions: Mapping[str, DomainFunction], /) -> DomainFunction:
        return directional_derivative(functions[velocity_var], n, var=var, mode=mode)

    return PointSetConstraint.from_points(
        component=component,
        points=points,
        residual=residual,
        weight=weight,
        label=label,
        reduction=reduction,
    )
