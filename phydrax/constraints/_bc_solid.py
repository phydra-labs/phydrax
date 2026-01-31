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
from ..operators.differential import cauchy_stress
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


# Solid Mechanics Boundary Constraints (Continuous)


def ContinuousTractionBoundaryConstraint(
    displacement_var: str,
    component: DomainComponent,
    /,
    *,
    lambda_: DomainFunction | ArrayLike,
    mu: DomainFunction | ArrayLike,
    traction: DomainFunction | ArrayLike | None = None,
    var: str = "x",
    mode: Literal["reverse", "forward"] = "reverse",
    num_points: int | tuple[Any, ...],
    structure: Any,
    sampler: str = "latin_hypercube",
    weight: ArrayLike = 1.0,
    label: str | None = None,
    over: str | tuple[str, ...] | None = None,
    reduction: Literal["mean", "integral"] = "mean",
) -> FunctionalConstraint:
    r"""Prescribed traction (Neumann) boundary condition.

    Enforces $\sigma(u)\,n = t$ on the boundary component, where $n$ is the
    outward normal and $t$ is the prescribed traction (default $0$).

    **Arguments:**

    - `displacement_var`: Name of the displacement field.
    - `component`: Boundary component.
    - `lambda_`, `mu`: Lamé parameters.
    - `traction`: Target traction $t$ (defaults to 0).
    - `var`: Geometry variable used to compute normals.
    - `mode`: Differentiation mode (`"reverse"` or `"forward"`).
    - `num_points`: Number of boundary samples.
    - `structure`: Sampling structure over labels.
    - `sampler`: Sampling method.
    - `weight`: Scalar weight.
    - `label`: Optional label for logging.
    - `over`: Optional reduction axes.
    - `reduction`: `"mean"` or `"integral"`.

    **Returns:**

    - A `FunctionalConstraint` enforcing the traction condition.
    """
    n = _normal(component, var=var)
    target = 0.0 if traction is None else traction

    def operator(u: DomainFunction, /) -> DomainFunction:
        sigma = cauchy_stress(u, lambda_=lambda_, mu=mu, var=var, mode=mode)
        tr = einsum("...ij,...j->...i", sigma, n)
        return tr - target

    return FunctionalConstraint.from_operator(
        component=component,
        operator=operator,
        constraint_vars=displacement_var,
        num_points=num_points,
        structure=structure,
        sampler=sampler,
        weight=weight,
        label=label,
        over=over,
        reduction=reduction,
    )


def ContinuousNormalDisplacementBoundaryConstraint(
    displacement_var: str,
    component: DomainComponent,
    /,
    *,
    normal_displacement: DomainFunction | ArrayLike | None = None,
    var: str = "x",
    num_points: int | tuple[Any, ...],
    structure: Any,
    sampler: str = "latin_hypercube",
    weight: ArrayLike = 1.0,
    label: str | None = None,
    over: str | tuple[str, ...] | None = None,
    reduction: Literal["mean", "integral"] = "mean",
) -> FunctionalConstraint:
    r"""Prescribed normal displacement boundary condition.

    Enforces $u\cdot n = u_n$ on the boundary component (default $u_n=0$).
    """
    n = _normal(component, var=var)
    target = 0.0 if normal_displacement is None else normal_displacement

    def operator(u: DomainFunction, /) -> DomainFunction:
        un = _dot(u, n)
        return un - target

    return FunctionalConstraint.from_operator(
        component=component,
        operator=operator,
        constraint_vars=displacement_var,
        num_points=num_points,
        structure=structure,
        sampler=sampler,
        weight=weight,
        label=label,
        over=over,
        reduction=reduction,
    )


def ContinuousElasticFoundationBoundaryConstraint(
    displacement_var: str,
    component: DomainComponent,
    /,
    *,
    lambda_: DomainFunction | ArrayLike,
    mu: DomainFunction | ArrayLike,
    stiffness: DomainFunction | ArrayLike,
    foundation_displacement: DomainFunction | ArrayLike | None = None,
    var: str = "x",
    mode: Literal["reverse", "forward"] = "reverse",
    num_points: int | tuple[Any, ...],
    structure: Any,
    sampler: str = "latin_hypercube",
    weight: ArrayLike = 1.0,
    label: str | None = None,
    over: str | tuple[str, ...] | None = None,
    reduction: Literal["mean", "integral"] = "mean",
) -> FunctionalConstraint:
    r"""Elastic foundation boundary condition.

    Enforces a spring-like relation between traction and displacement:

    $$
    \sigma(u)\,n + k\,(u-u_0) = 0,
    $$

    where `stiffness` is the foundation stiffness $k$ and `foundation_displacement`
    is $u_0$ (default $0$).
    """
    n = _normal(component, var=var)
    u0 = 0.0 if foundation_displacement is None else foundation_displacement

    def operator(u: DomainFunction, /) -> DomainFunction:
        sigma = cauchy_stress(u, lambda_=lambda_, mu=mu, var=var, mode=mode)
        tr = einsum("...ij,...j->...i", sigma, n)
        return tr + stiffness * (u - u0)

    return FunctionalConstraint.from_operator(
        component=component,
        operator=operator,
        constraint_vars=displacement_var,
        num_points=num_points,
        structure=structure,
        sampler=sampler,
        weight=weight,
        label=label,
        over=over,
        reduction=reduction,
    )


def ContinuousElasticSymmetryBoundaryConstraint(
    displacement_var: str,
    component: DomainComponent,
    /,
    *,
    lambda_: DomainFunction | ArrayLike,
    mu: DomainFunction | ArrayLike,
    var: str = "x",
    mode: Literal["reverse", "forward"] = "reverse",
    num_points: int | tuple[Any, ...],
    structure: Any,
    sampler: str = "latin_hypercube",
    weight: ArrayLike = 1.0,
    label: str | None = None,
    over: str | tuple[str, ...] | None = None,
    reduction: Literal["mean", "integral"] = "mean",
) -> FunctionalConstraint:
    r"""Elastic symmetry boundary condition.

    Enforces symmetry by combining:
    - zero normal displacement: $u\cdot n = 0$,
    - zero tangential traction: $(I-n\otimes n)\,\sigma(u)\,n = 0$.
    """
    n = _normal(component, var=var)

    def operator(u: DomainFunction, /) -> DomainFunction:
        sigma = cauchy_stress(u, lambda_=lambda_, mu=mu, var=var, mode=mode)
        tr = einsum("...ij,...j->...i", sigma, n)
        un_vec = _outer_scalar_vec(_dot(u, n), n)
        tr_t = tr - _outer_scalar_vec(_dot(tr, n), n)
        return un_vec + tr_t

    return FunctionalConstraint.from_operator(
        component=component,
        operator=operator,
        constraint_vars=displacement_var,
        num_points=num_points,
        structure=structure,
        sampler=sampler,
        weight=weight,
        label=label,
        over=over,
        reduction=reduction,
    )


# Solid Mechanics Boundary Constraints (Discrete)


def DiscreteDisplacementBoundaryConstraint(
    displacement_var: str,
    component: DomainComponent,
    /,
    *,
    points: Mapping[str, ArrayLike] | ArrayLike,
    displacement_values: ArrayLike,
    weight: ArrayLike = 1.0,
    label: str | None = None,
    reduction: Literal["mean", "sum"] = "mean",
) -> PointSetConstraint:
    r"""Discrete displacement (Dirichlet) constraint at explicit anchor points."""
    target = _interp_target(component, points, displacement_values)

    def residual(functions: Mapping[str, DomainFunction], /) -> DomainFunction:
        return functions[displacement_var] - target

    return PointSetConstraint.from_points(
        component=component,
        points=points,
        residual=residual,
        weight=weight,
        label=label,
        reduction=reduction,
    )


def DiscreteTractionBoundaryConstraint(
    displacement_var: str,
    component: DomainComponent,
    /,
    *,
    points: Mapping[str, ArrayLike] | ArrayLike,
    values: ArrayLike,
    lambda_: DomainFunction | ArrayLike,
    mu: DomainFunction | ArrayLike,
    var: str = "x",
    mode: Literal["reverse", "forward"] = "reverse",
    weight: ArrayLike = 1.0,
    label: str | None = None,
    reduction: Literal["mean", "sum"] = "mean",
) -> PointSetConstraint:
    r"""Discrete traction (Neumann) constraint at explicit anchor points."""
    n = _normal(component, var=var)
    target = _interp_target(component, points, values)

    def residual(functions: Mapping[str, DomainFunction], /) -> DomainFunction:
        u = functions[displacement_var]
        sigma = cauchy_stress(u, lambda_=lambda_, mu=mu, var=var, mode=mode)
        tr = einsum("...ij,...j->...i", sigma, n)
        return tr - target

    return PointSetConstraint.from_points(
        component=component,
        points=points,
        residual=residual,
        weight=weight,
        label=label,
        reduction=reduction,
    )


def DiscreteNormalDisplacementBoundaryConstraint(
    displacement_var: str,
    component: DomainComponent,
    /,
    *,
    points: Mapping[str, ArrayLike] | ArrayLike,
    values: ArrayLike,
    var: str = "x",
    weight: ArrayLike = 1.0,
    label: str | None = None,
    reduction: Literal["mean", "sum"] = "mean",
) -> PointSetConstraint:
    r"""Discrete normal displacement constraint at explicit anchor points."""
    n = _normal(component, var=var)
    target = _interp_target(component, points, values)

    def residual(functions: Mapping[str, DomainFunction], /) -> DomainFunction:
        u = functions[displacement_var]
        return _dot(u, n) - target

    return PointSetConstraint.from_points(
        component=component,
        points=points,
        residual=residual,
        weight=weight,
        label=label,
        reduction=reduction,
    )
