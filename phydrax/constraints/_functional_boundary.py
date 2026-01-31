#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from jaxtyping import ArrayLike

from ..domain._components import (
    DomainComponent,
    DomainComponentUnion,
    Fixed,
    FixedEnd,
    FixedStart,
)
from ..domain._function import DomainFunction
from ..domain._structure import NumPoints, ProductStructure
from ..operators.differential._domain_ops import directional_derivative, dt
from ._functional import FunctionalConstraint
from ._interpolate import idw_interpolant
from ._pointset import PointSetConstraint


def _value_deps(component: DomainComponent | DomainComponentUnion, /) -> tuple[str, ...]:
    if isinstance(component, DomainComponent):
        return _non_fixed_labels(component)
    return component.domain.labels


def _coerce_value(
    value: Any,
    component: DomainComponent | DomainComponentUnion,
    /,
) -> DomainFunction | ArrayLike:
    if isinstance(value, DomainFunction):
        return value
    if callable(value):
        return DomainFunction(
            domain=component.domain, deps=_value_deps(component), func=value, metadata={}
        )
    return value


def _non_fixed_labels(component: DomainComponent, /) -> tuple[str, ...]:
    fixed = {
        lbl
        for lbl in component.domain.labels
        if isinstance(component.spec.component_for(lbl), (FixedStart, FixedEnd, Fixed))
    }
    return tuple(lbl for lbl in component.domain.labels if lbl not in fixed)


def ContinuousDirichletBoundaryConstraint(
    constraint_var: str,
    component: DomainComponent | DomainComponentUnion,
    /,
    *,
    target: DomainFunction | ArrayLike | None = None,
    num_points: NumPoints | tuple[Any, ...],
    structure: ProductStructure,
    sampler: str = "latin_hypercube",
    weight: ArrayLike = 1.0,
    label: str | None = None,
    over: str | tuple[str, ...] | None = None,
    reduction: Literal["mean", "integral"] = "mean",
) -> FunctionalConstraint:
    r"""Continuous Dirichlet boundary constraint.

    Enforces $u = g$ on the boundary component by minimizing the squared residual
    $u - g$.

    **Arguments:**

    - `constraint_var`: Name of the constrained field.
    - `component`: Boundary component or component union.
    - `target`: Target value/function $g$ (defaults to 0).
    - `num_points`: Number of boundary samples.
    - `structure`: Sampling structure over labels.
    - `sampler`: Sampling method.
    - `weight`: Scalar weight.
    - `label`: Optional label for logging.
    - `over`: Optional reduction axes.
    - `reduction`: `"mean"` or `"integral"`.

    **Returns:**

    - A `FunctionalConstraint` enforcing the Dirichlet condition.
    """
    value = 0.0 if target is None else _coerce_value(target, component)

    def operator(u: DomainFunction, /) -> DomainFunction:
        return u - value

    return FunctionalConstraint.from_operator(
        component=component,
        operator=operator,
        constraint_vars=constraint_var,
        num_points=num_points,
        structure=structure,
        sampler=sampler,
        weight=weight,
        label=label,
        over=over,
        reduction=reduction,
    )


def ContinuousNeumannBoundaryConstraint(
    constraint_var: str,
    component: DomainComponent,
    /,
    *,
    var: str = "x",
    target: DomainFunction | ArrayLike | None = None,
    mode: Literal["reverse", "forward"] = "reverse",
    num_points: NumPoints | tuple[Any, ...],
    structure: ProductStructure,
    sampler: str = "latin_hypercube",
    weight: ArrayLike = 1.0,
    label: str | None = None,
    over: str | tuple[str, ...] | None = None,
    reduction: Literal["mean", "integral"] = "mean",
) -> FunctionalConstraint:
    r"""Continuous Neumann boundary constraint.

    Enforces $\partial u/\partial n = g$ on the boundary component, where $n$ is
    the outward normal.

    **Arguments:**

    - `constraint_var`: Name of the constrained field.
    - `component`: Boundary component.
    - `var`: Geometry variable used to compute normals.
    - `target`: Target flux/function $g$ (defaults to 0).
    - `mode`: Differentiation mode (`"reverse"` or `"forward"`).
    - `num_points`: Number of boundary samples.
    - `structure`: Sampling structure over labels.
    - `sampler`: Sampling method.
    - `weight`: Scalar weight.
    - `label`: Optional label for logging.
    - `over`: Optional reduction axes.
    - `reduction`: `"mean"` or `"integral"`.

    **Returns:**

    - A `FunctionalConstraint` enforcing the Neumann condition.
    """
    n = component.normal(var=var)
    value = 0.0 if target is None else _coerce_value(target, component)

    def operator(u: DomainFunction, /) -> DomainFunction:
        dd = directional_derivative(u, n, var=var, mode=mode)
        return dd - value

    return FunctionalConstraint.from_operator(
        component=component,
        operator=operator,
        constraint_vars=constraint_var,
        num_points=num_points,
        structure=structure,
        sampler=sampler,
        weight=weight,
        label=label,
        over=over,
        reduction=reduction,
    )


def ContinuousRobinBoundaryConstraint(
    constraint_var: str,
    component: DomainComponent,
    /,
    *,
    var: str = "x",
    dirichlet_coeff: DomainFunction | ArrayLike | None = None,
    neumann_coeff: DomainFunction | ArrayLike | None = None,
    target: DomainFunction | ArrayLike | None = None,
    mode: Literal["reverse", "forward"] = "reverse",
    num_points: NumPoints | tuple[Any, ...],
    structure: ProductStructure,
    sampler: str = "latin_hypercube",
    weight: ArrayLike = 1.0,
    label: str | None = None,
    over: str | tuple[str, ...] | None = None,
    reduction: Literal["mean", "integral"] = "mean",
) -> FunctionalConstraint:
    r"""Continuous Robin boundary constraint.

    Enforces $c_0 u + c_1 \\, \\partial u/\\partial n = g$ on the boundary component.

    **Arguments:**

    - `constraint_var`: Name of the constrained field.
    - `component`: Boundary component.
    - `var`: Geometry variable used to compute normals.
    - `dirichlet_coeff`: Coefficient $c_0$ (defaults to 0).
    - `neumann_coeff`: Coefficient $c_1$ (defaults to 0).
    - `target`: Target value/function $g$ (defaults to 0).
    - `mode`: Differentiation mode (`"reverse"` or `"forward"`).
    - `num_points`: Number of boundary samples.
    - `structure`: Sampling structure over labels.
    - `sampler`: Sampling method.
    - `weight`: Scalar weight.
    - `label`: Optional label for logging.
    - `over`: Optional reduction axes.
    - `reduction`: `"mean"` or `"integral"`.

    **Returns:**

    - A `FunctionalConstraint` enforcing the Robin condition.
    """
    n = component.normal(var=var)
    c0 = 0.0 if dirichlet_coeff is None else _coerce_value(dirichlet_coeff, component)
    c1 = 0.0 if neumann_coeff is None else _coerce_value(neumann_coeff, component)
    g = 0.0 if target is None else _coerce_value(target, component)

    def operator(u: DomainFunction, /) -> DomainFunction:
        dd = directional_derivative(u, n, var=var, mode=mode)
        return c0 * u + c1 * dd - g

    return FunctionalConstraint.from_operator(
        component=component,
        operator=operator,
        constraint_vars=constraint_var,
        num_points=num_points,
        structure=structure,
        sampler=sampler,
        weight=weight,
        label=label,
        over=over,
        reduction=reduction,
    )


def AbsorbingBoundaryConstraint(
    constraint_var: str,
    component: DomainComponent,
    /,
    *,
    var: str = "x",
    time_var: str = "t",
    wavespeed: DomainFunction | ArrayLike | None = None,
    target: DomainFunction | ArrayLike | None = None,
    mode: Literal["reverse", "forward"] = "reverse",
    num_points: NumPoints | tuple[Any, ...],
    structure: ProductStructure,
    sampler: str = "latin_hypercube",
    weight: ArrayLike = 1.0,
    label: str | None = None,
    over: str | tuple[str, ...] | None = None,
    reduction: Literal["mean", "integral"] = "mean",
) -> FunctionalConstraint:
    r"""Absorbing/Sommerfeld boundary constraint

    $$
    \frac{\partial u}{\partial n} + \frac{1}{c} \frac{\partial u}{\partial t} = g.
    $$

    This is a common first-order radiation condition for wave-like PDEs, where `c`
    is a characteristic wave speed.

    **Arguments:**

    - `constraint_var`: Name of the constrained field.
    - `component`: Boundary `DomainComponent` to sample from.
    - `var`: Geometry label used for normals and spatial derivatives.
    - `time_var`: Time label used for $\partial/\partial t$.
    - `wavespeed`: Wave speed `c` as a `DomainFunction`, callable, or array-like. Defaults to `1.0`.
    - `target`: Target value as a `DomainFunction`, callable, or array-like. Defaults to `0.0`.
    - `mode`: Differentiation mode (`"reverse"` or `"forward"`).
    - `num_points`: Number of boundary points to sample (paired or structured; see `structure`).
    - `structure`: A `ProductStructure` describing how variables are sampled/blocked.
    - `sampler`: Sampling scheme (e.g. `"latin_hypercube"`).
    - `weight`: Scalar multiplier applied to this term.
    - `over`: Optional subset of labels to reduce/integrate over.
    - `reduction`: `"mean"` or `"integral"`.
    """
    if time_var not in component.domain.labels:
        raise KeyError(f"Label {time_var!r} not in domain {component.domain.labels}.")
    n = component.normal(var=var)
    c = 1.0 if wavespeed is None else _coerce_value(wavespeed, component)
    g = 0.0 if target is None else _coerce_value(target, component)

    def operator(u: DomainFunction, /) -> DomainFunction:
        du_dn = directional_derivative(u, n, var=var, mode=mode)
        du_dt = dt(u, var=time_var, mode=mode)
        return du_dn + (1.0 / c) * du_dt - g

    return FunctionalConstraint.from_operator(
        component=component,
        operator=operator,
        constraint_vars=constraint_var,
        num_points=num_points,
        structure=structure,
        sampler=sampler,
        weight=weight,
        label=label,
        over=over,
        reduction=reduction,
    )


def DiscreteDirichletBoundaryConstraint(
    constraint_var: str,
    component: DomainComponent | DomainComponentUnion,
    /,
    *,
    points: Mapping[str, ArrayLike] | ArrayLike,
    values: ArrayLike,
    idw_exponent: float = 2.0,
    eps_snap: float = 1e-12,
    lengthscales: Mapping[str, float] | None = None,
    weight: ArrayLike = 1.0,
    label: str | None = None,
    reduction: Literal["mean", "sum"] = "mean",
) -> PointSetConstraint:
    r"""Discrete Dirichlet constraint on a boundary component at explicit anchor points.

    Interpolates the provided `(points, values)` into a `DomainFunction` target via
    inverse-distance weighting (IDW), and then enforces `u(points) = target(points)`.

    **Arguments:**

    - `constraint_var`: Name of the constrained field.
    - `component`: Boundary `DomainComponent` associated with the points.
    - `points`: Anchor point coordinates (mapping by label, or stacked array).
    - `values`: Target values at the anchor points.
    - `idw_exponent`: Power $p$ in IDW weights $w_j \propto (\|z-z_j\|^2 + \varepsilon)^{-p/2}$.
    - `eps_snap`: Snap threshold $\varepsilon$ used to return exact anchors when very close.
    - `lengthscales`: Optional per-label lengthscales used inside the distance metric.
    - `weight`: Scalar multiplier applied to this term.
    - `reduction`: `"mean"` or `"sum"`.
    """
    if isinstance(component, DomainComponentUnion):
        raise TypeError(
            "DiscreteDirichletBoundaryConstraint requires a DomainComponent, not a union."
        )
    labels = _non_fixed_labels(component)
    target = idw_interpolant(
        component.domain,
        anchors=points,
        values=values,
        labels=labels,
        lengthscales=lengthscales,
        idw_exponent=idw_exponent,
        eps_snap=eps_snap,
    )

    def residual(functions: Mapping[str, DomainFunction], /) -> DomainFunction:
        return functions[constraint_var] - target

    pts_in = points if labels else {}
    return PointSetConstraint.from_points(
        component=component,
        points=pts_in,
        residual=residual,
        weight=weight,
        label=label,
        reduction=reduction,
    )


def DiscreteNeumannBoundaryConstraint(
    constraint_var: str,
    component: DomainComponent | DomainComponentUnion,
    /,
    *,
    points: Mapping[str, ArrayLike] | ArrayLike,
    values: ArrayLike,
    var: str = "x",
    mode: Literal["reverse", "forward"] = "reverse",
    idw_exponent: float = 2.0,
    eps_snap: float = 1e-12,
    lengthscales: Mapping[str, float] | None = None,
    weight: ArrayLike = 1.0,
    label: str | None = None,
    reduction: Literal["mean", "sum"] = "mean",
) -> PointSetConstraint:
    r"""Discrete Neumann constraint on a boundary component at explicit anchor points.

    Interpolates the provided `(points, values)` into a `DomainFunction` target via
    inverse-distance weighting (IDW), and then enforces $\partial u/\partial n = g$
    at the provided points.

    **Arguments:**

    - `constraint_var`: Name of the constrained field.
    - `component`: Boundary `DomainComponent` associated with the points.
    - `points`: Anchor point coordinates (mapping by label, or stacked array).
    - `values`: Target Neumann data $g$ at the anchor points.
    - `var`: Geometry label used to compute normals and differentiate.
    - `mode`: Differentiation mode (`"reverse"` or `"forward"`).
    - `idw_exponent`: Power $p$ in IDW weights $w_j \propto (\|z-z_j\|^2 + \varepsilon)^{-p/2}$.
    - `eps_snap`: Snap threshold $\varepsilon$ used to return exact anchors when very close.
    - `lengthscales`: Optional per-label lengthscales used inside the distance metric.
    - `weight`: Scalar multiplier applied to this term.
    - `reduction`: `"mean"` or `"sum"`.
    """
    if isinstance(component, DomainComponentUnion):
        raise TypeError(
            "DiscreteNeumannBoundaryConstraint requires a DomainComponent, not a union."
        )
    labels = _non_fixed_labels(component)
    target = idw_interpolant(
        component.domain,
        anchors=points,
        values=values,
        labels=labels,
        lengthscales=lengthscales,
        idw_exponent=idw_exponent,
        eps_snap=eps_snap,
    )

    n = component.normal(var=var)

    def residual(functions: Mapping[str, DomainFunction], /) -> DomainFunction:
        dd = directional_derivative(functions[constraint_var], n, var=var, mode=mode)
        return dd - target

    pts_in = points if labels else {}
    return PointSetConstraint.from_points(
        component=component,
        points=pts_in,
        residual=residual,
        weight=weight,
        label=label,
        reduction=reduction,
    )
