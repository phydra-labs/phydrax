#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import jax.numpy as jnp
from jaxtyping import ArrayLike

from ..domain._components import DomainComponent
from ..domain._function import DomainFunction
from ..operators.differential import tangential_component
from ..operators.linalg import einsum
from ._functional import FunctionalConstraint
from ._interpolate import idw_interpolant
from ._pointset import PointSetConstraint


def _normal(component: DomainComponent, *, var: str) -> DomainFunction:
    return component.normal(var=var)


def _dot(a: DomainFunction, b: DomainFunction) -> DomainFunction:
    return einsum("...i,...i->...", a, b)


def _cross(a: DomainFunction, b: DomainFunction) -> DomainFunction:
    joined = a.domain.join(b.domain)
    a2 = a.promote(joined)
    b2 = b.promote(joined)

    deps = tuple(lbl for lbl in joined.labels if (lbl in a2.deps) or (lbl in b2.deps))
    idx = {lbl: i for i, lbl in enumerate(deps)}
    a_pos = tuple(idx[lbl] for lbl in a2.deps)
    b_pos = tuple(idx[lbl] for lbl in b2.deps)

    meta = a.metadata if a.metadata == b.metadata else {}

    def _op(*args, key=None, **kwargs):
        av = jnp.asarray(a2.func(*[args[i] for i in a_pos], key=key, **kwargs))
        bv = jnp.asarray(b2.func(*[args[i] for i in b_pos], key=key, **kwargs))
        return jnp.cross(av, bv)

    return DomainFunction(domain=joined, deps=deps, func=_op, metadata=meta)


def _interp_target(
    component: DomainComponent,
    points: Mapping[str, ArrayLike] | ArrayLike,
    values: ArrayLike,
) -> DomainFunction:
    arr = jnp.asarray(values)
    if arr.ndim == 0:
        return DomainFunction(
            domain=component.domain,
            deps=component.domain.labels,
            func=arr.reshape(()),
            metadata={},
        )
    return idw_interpolant(
        component.domain,
        anchors=points,
        values=arr,
        labels=component.domain.labels,
    )


# Electromagnetics Boundary Constraints (Continuous)


def ContinuousPECBoundaryConstraint(
    field_var: str,
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
    r"""Perfect electric conductor (PEC) boundary condition.

    For an electric field $E$, PEC implies the tangential component vanishes:

    $$
    E_t = (I-n\otimes n)\,E = 0,
    $$

    equivalently $n\times E = 0$.
    """

    def operator(u: DomainFunction, /) -> DomainFunction:
        return tangential_component(u, component, var=var)

    return FunctionalConstraint.from_operator(
        component=component,
        operator=operator,
        constraint_vars=field_var,
        num_points=num_points,
        structure=structure,
        sampler=sampler,
        weight=weight,
        label=label,
        over=over,
        reduction=reduction,
    )


def ContinuousImpedanceBoundaryConstraint(
    h_var: str,
    e_var: str,
    component: DomainComponent,
    /,
    *,
    admittance: DomainFunction | ArrayLike,
    var: str = "x",
    num_points: int | tuple[Any, ...],
    structure: Any,
    sampler: str = "latin_hypercube",
    weight: ArrayLike = 1.0,
    label: str | None = None,
    over: str | tuple[str, ...] | None = None,
    reduction: Literal["mean", "integral"] = "mean",
) -> FunctionalConstraint:
    r"""Impedance (Leontovich) boundary condition.

    Enforces a linear relation between tangential $E$ and $H$:

    $$
    n\times H = Y\,E_t,
    $$

    where `admittance` is $Y$ and $E_t$ is the tangential projection of $E$.
    """
    n = _normal(component, var=var)

    def operator(H: DomainFunction, E: DomainFunction, /) -> DomainFunction:
        Et = tangential_component(E, component, var=var)
        lhs = _cross(n, H)
        return lhs - admittance * Et

    return FunctionalConstraint.from_operator(
        component=component,
        operator=operator,
        constraint_vars=(h_var, e_var),
        num_points=num_points,
        structure=structure,
        sampler=sampler,
        weight=weight,
        label=label,
        over=over,
        reduction=reduction,
    )


def ContinuousPMCBoundaryConstraint(
    field_var: str,
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
    r"""Perfect magnetic conductor (PMC) boundary condition.

    For a magnetic field $H$, PMC implies the tangential component vanishes:

    $$
    H_t = (I-n\otimes n)\,H = 0,
    $$

    equivalently $n\times H = 0$.
    """

    def operator(u: DomainFunction, /) -> DomainFunction:
        return tangential_component(u, component, var=var)

    return FunctionalConstraint.from_operator(
        component=component,
        operator=operator,
        constraint_vars=field_var,
        num_points=num_points,
        structure=structure,
        sampler=sampler,
        weight=weight,
        label=label,
        over=over,
        reduction=reduction,
    )


def ContinuousElectricSurfaceChargeBoundaryConstraint(
    field_var: str,
    component: DomainComponent,
    /,
    *,
    epsilon: DomainFunction | ArrayLike,
    surface_charge: DomainFunction | ArrayLike,
    var: str = "x",
    num_points: int | tuple[Any, ...],
    structure: Any,
    sampler: str = "latin_hypercube",
    weight: ArrayLike = 1.0,
    label: str | None = None,
    over: str | tuple[str, ...] | None = None,
    reduction: Literal["mean", "integral"] = "mean",
) -> FunctionalConstraint:
    r"""Surface charge boundary condition for the electric field.

    Enforces a normal-flux condition

    $$
    \epsilon\,E\cdot n = \rho_s,
    $$

    where $\rho_s$ is a prescribed surface charge density.
    """
    n = _normal(component, var=var)

    def operator(u: DomainFunction, /) -> DomainFunction:
        return epsilon * _dot(u, n) - surface_charge

    return FunctionalConstraint.from_operator(
        component=component,
        operator=operator,
        constraint_vars=field_var,
        num_points=num_points,
        structure=structure,
        sampler=sampler,
        weight=weight,
        label=label,
        over=over,
        reduction=reduction,
    )


def ContinuousMagneticSurfaceCurrentBoundaryConstraint(
    field_var: str,
    component: DomainComponent,
    /,
    *,
    surface_current: DomainFunction | ArrayLike,
    var: str = "x",
    num_points: int | tuple[Any, ...],
    structure: Any,
    sampler: str = "latin_hypercube",
    weight: ArrayLike = 1.0,
    label: str | None = None,
    over: str | tuple[str, ...] | None = None,
    reduction: Literal["mean", "integral"] = "mean",
) -> FunctionalConstraint:
    r"""Surface current boundary condition for the magnetic field.

    Enforces the tangential jump condition

    $$
    n\times H = K_s,
    $$

    where $K_s$ is a prescribed surface current density.
    """
    n = _normal(component, var=var)

    def operator(u: DomainFunction, /) -> DomainFunction:
        return _cross(n, u) - surface_current

    return FunctionalConstraint.from_operator(
        component=component,
        operator=operator,
        constraint_vars=field_var,
        num_points=num_points,
        structure=structure,
        sampler=sampler,
        weight=weight,
        label=label,
        over=over,
        reduction=reduction,
    )


# Electromagnetics Interface Constraints (Continuous)


def ContinuousInterfaceTangentialEContinuityConstraint(
    e1_var: str,
    e2_var: str,
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
    r"""Tangential continuity of electric field across an interface.

    Enforces $(E_2 - E_1)_t = 0$, i.e. tangential $E$ is continuous across the interface.
    """

    def operator(E1: DomainFunction, E2: DomainFunction, /) -> DomainFunction:
        return tangential_component(E2 - E1, component, var=var)

    return FunctionalConstraint.from_operator(
        component=component,
        operator=operator,
        constraint_vars=(e1_var, e2_var),
        num_points=num_points,
        structure=structure,
        sampler=sampler,
        weight=weight,
        label=label,
        over=over,
        reduction=reduction,
    )


def ContinuousInterfaceNormalDJumpConstraint(
    e1_var: str,
    e2_var: str,
    component: DomainComponent,
    /,
    *,
    epsilon1: DomainFunction | ArrayLike,
    epsilon2: DomainFunction | ArrayLike,
    surface_charge: DomainFunction | ArrayLike | None = None,
    var: str = "x",
    num_points: int | tuple[Any, ...],
    structure: Any,
    sampler: str = "latin_hypercube",
    weight: ArrayLike = 1.0,
    label: str | None = None,
    over: str | tuple[str, ...] | None = None,
    reduction: Literal["mean", "integral"] = "mean",
) -> FunctionalConstraint:
    r"""Normal jump condition for electric displacement across an interface.

    For $D=\epsilon E$, enforces

    $$
    (D_2 - D_1)\cdot n = \rho_s,
    $$

    i.e. $\epsilon_2 E_2\cdot n - \epsilon_1 E_1\cdot n = \rho_s$.
    """
    n = _normal(component, var=var)
    rho = 0.0 if surface_charge is None else surface_charge

    def operator(E1: DomainFunction, E2: DomainFunction, /) -> DomainFunction:
        e1n = _dot(E1, n)
        e2n = _dot(E2, n)
        return epsilon2 * e2n - epsilon1 * e1n - rho

    return FunctionalConstraint.from_operator(
        component=component,
        operator=operator,
        constraint_vars=(e1_var, e2_var),
        num_points=num_points,
        structure=structure,
        sampler=sampler,
        weight=weight,
        label=label,
        over=over,
        reduction=reduction,
    )


def ContinuousInterfaceTangentialHJumpConstraint(
    h1_var: str,
    h2_var: str,
    component: DomainComponent,
    /,
    *,
    surface_current: DomainFunction | ArrayLike | None = None,
    var: str = "x",
    num_points: int | tuple[Any, ...],
    structure: Any,
    sampler: str = "latin_hypercube",
    weight: ArrayLike = 1.0,
    label: str | None = None,
    over: str | tuple[str, ...] | None = None,
    reduction: Literal["mean", "integral"] = "mean",
) -> FunctionalConstraint:
    r"""Tangential jump condition for magnetic field across an interface.

    Enforces

    $$
    n\times(H_2 - H_1) = K_s,
    $$

    where $K_s$ is an optional surface current density (defaults to $0$).
    """
    n = _normal(component, var=var)
    ks = 0.0 if surface_current is None else surface_current

    def operator(H1: DomainFunction, H2: DomainFunction, /) -> DomainFunction:
        return _cross(n, H2 - H1) - ks

    return FunctionalConstraint.from_operator(
        component=component,
        operator=operator,
        constraint_vars=(h1_var, h2_var),
        num_points=num_points,
        structure=structure,
        sampler=sampler,
        weight=weight,
        label=label,
        over=over,
        reduction=reduction,
    )


def ContinuousInterfaceNormalBContinuityConstraint(
    h1_var: str,
    h2_var: str,
    component: DomainComponent,
    /,
    *,
    mu1: DomainFunction | ArrayLike,
    mu2: DomainFunction | ArrayLike,
    var: str = "x",
    num_points: int | tuple[Any, ...],
    structure: Any,
    sampler: str = "latin_hypercube",
    weight: ArrayLike = 1.0,
    label: str | None = None,
    over: str | tuple[str, ...] | None = None,
    reduction: Literal["mean", "integral"] = "mean",
) -> FunctionalConstraint:
    r"""Normal continuity of magnetic flux density across an interface.

    For $B=\mu H$, enforces

    $$
    (B_2 - B_1)\cdot n = 0,
    $$

    i.e. $\mu_2 H_2\cdot n - \mu_1 H_1\cdot n = 0$.
    """
    n = _normal(component, var=var)

    def operator(H1: DomainFunction, H2: DomainFunction, /) -> DomainFunction:
        h1n = _dot(H1, n)
        h2n = _dot(H2, n)
        return mu2 * h2n - mu1 * h1n

    return FunctionalConstraint.from_operator(
        component=component,
        operator=operator,
        constraint_vars=(h1_var, h2_var),
        num_points=num_points,
        structure=structure,
        sampler=sampler,
        weight=weight,
        label=label,
        over=over,
        reduction=reduction,
    )


# Electromagnetics Boundary/Interface Constraints (Discrete)


def DiscretePECBoundaryConstraint(
    field_var: str,
    component: DomainComponent,
    /,
    *,
    points: Mapping[str, ArrayLike] | ArrayLike,
    var: str = "x",
    weight: ArrayLike = 1.0,
    label: str | None = None,
    reduction: Literal["mean", "sum"] = "mean",
) -> PointSetConstraint:
    r"""Discrete PEC constraint at explicit boundary anchor points."""

    def residual(functions: Mapping[str, DomainFunction], /) -> DomainFunction:
        return tangential_component(functions[field_var], component, var=var)

    return PointSetConstraint.from_points(
        component=component,
        points=points,
        residual=residual,
        weight=weight,
        label=label,
        reduction=reduction,
    )


def DiscretePMCBoundaryConstraint(
    field_var: str,
    component: DomainComponent,
    /,
    *,
    points: Mapping[str, ArrayLike] | ArrayLike,
    var: str = "x",
    weight: ArrayLike = 1.0,
    label: str | None = None,
    reduction: Literal["mean", "sum"] = "mean",
) -> PointSetConstraint:
    r"""Discrete PMC constraint at explicit boundary anchor points."""

    def residual(functions: Mapping[str, DomainFunction], /) -> DomainFunction:
        return tangential_component(functions[field_var], component, var=var)

    return PointSetConstraint.from_points(
        component=component,
        points=points,
        residual=residual,
        weight=weight,
        label=label,
        reduction=reduction,
    )


def DiscreteElectricSurfaceChargeBoundaryConstraint(
    field_var: str,
    component: DomainComponent,
    /,
    *,
    points: Mapping[str, ArrayLike] | ArrayLike,
    surface_charge_values: ArrayLike,
    epsilon: DomainFunction | ArrayLike,
    var: str = "x",
    weight: ArrayLike = 1.0,
    label: str | None = None,
    reduction: Literal["mean", "sum"] = "mean",
) -> PointSetConstraint:
    r"""Discrete surface charge constraint at explicit boundary anchor points."""
    n = _normal(component, var=var)
    target = _interp_target(component, points, surface_charge_values)

    def residual(functions: Mapping[str, DomainFunction], /) -> DomainFunction:
        u = functions[field_var]
        return epsilon * _dot(u, n) - target

    return PointSetConstraint.from_points(
        component=component,
        points=points,
        residual=residual,
        weight=weight,
        label=label,
        reduction=reduction,
    )


def DiscreteMagneticSurfaceCurrentBoundaryConstraint(
    field_var: str,
    component: DomainComponent,
    /,
    *,
    points: Mapping[str, ArrayLike] | ArrayLike,
    surface_current_values: ArrayLike,
    var: str = "x",
    weight: ArrayLike = 1.0,
    label: str | None = None,
    reduction: Literal["mean", "sum"] = "mean",
) -> PointSetConstraint:
    r"""Discrete surface current constraint at explicit boundary anchor points."""
    n = _normal(component, var=var)
    target = _interp_target(component, points, surface_current_values)

    def residual(functions: Mapping[str, DomainFunction], /) -> DomainFunction:
        u = functions[field_var]
        return _cross(n, u) - target

    return PointSetConstraint.from_points(
        component=component,
        points=points,
        residual=residual,
        weight=weight,
        label=label,
        reduction=reduction,
    )


def DiscreteInterfaceTangentialEContinuityConstraint(
    field_var: str,
    component: DomainComponent,
    /,
    *,
    points: Mapping[str, ArrayLike] | ArrayLike,
    tangential_values: ArrayLike,
    var: str = "x",
    weight: ArrayLike = 1.0,
    label: str | None = None,
    reduction: Literal["mean", "sum"] = "mean",
) -> PointSetConstraint:
    r"""Discrete tangential electric continuity constraint at explicit interface points."""
    target = _interp_target(component, points, tangential_values)

    def residual(functions: Mapping[str, DomainFunction], /) -> DomainFunction:
        u = functions[field_var]
        return tangential_component(u, component, var=var) - target

    return PointSetConstraint.from_points(
        component=component,
        points=points,
        residual=residual,
        weight=weight,
        label=label,
        reduction=reduction,
    )


def DiscreteInterfaceNormalDJumpConstraint(
    field_var: str,
    component: DomainComponent,
    /,
    *,
    points: Mapping[str, ArrayLike] | ArrayLike,
    values: ArrayLike,
    epsilon: DomainFunction | ArrayLike,
    var: str = "x",
    weight: ArrayLike = 1.0,
    label: str | None = None,
    reduction: Literal["mean", "sum"] = "mean",
) -> PointSetConstraint:
    r"""Discrete normal electric displacement jump constraint at explicit interface points."""
    n = _normal(component, var=var)
    target = _interp_target(component, points, values)

    def residual(functions: Mapping[str, DomainFunction], /) -> DomainFunction:
        u = functions[field_var]
        return epsilon * _dot(u, n) - target

    return PointSetConstraint.from_points(
        component=component,
        points=points,
        residual=residual,
        weight=weight,
        label=label,
        reduction=reduction,
    )


def DiscreteInterfaceTangentialHJumpConstraint(
    field_var: str,
    component: DomainComponent,
    /,
    *,
    points: Mapping[str, ArrayLike] | ArrayLike,
    Ks_values: ArrayLike,
    var: str = "x",
    weight: ArrayLike = 1.0,
    label: str | None = None,
    reduction: Literal["mean", "sum"] = "mean",
) -> PointSetConstraint:
    r"""Discrete tangential magnetic jump constraint at explicit interface points."""
    n = _normal(component, var=var)
    target = _interp_target(component, points, Ks_values)

    def residual(functions: Mapping[str, DomainFunction], /) -> DomainFunction:
        u = functions[field_var]
        return _cross(n, u) - target

    return PointSetConstraint.from_points(
        component=component,
        points=points,
        residual=residual,
        weight=weight,
        label=label,
        reduction=reduction,
    )


def DiscreteInterfaceNormalBContinuityConstraint(
    field_var: str,
    component: DomainComponent,
    /,
    *,
    points: Mapping[str, ArrayLike] | ArrayLike,
    values: ArrayLike,
    mu: DomainFunction | ArrayLike,
    var: str = "x",
    weight: ArrayLike = 1.0,
    label: str | None = None,
    reduction: Literal["mean", "sum"] = "mean",
) -> PointSetConstraint:
    r"""Discrete normal magnetic flux continuity constraint at explicit interface points."""
    n = _normal(component, var=var)
    target = _interp_target(component, points, values)

    def residual(functions: Mapping[str, DomainFunction], /) -> DomainFunction:
        u = functions[field_var]
        return mu * _dot(u, n) - target

    return PointSetConstraint.from_points(
        component=component,
        points=points,
        residual=residual,
        weight=weight,
        label=label,
        reduction=reduction,
    )
