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
from ..operators.differential._domain_ops import dt_n
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


def ContinuousInitialConstraint(
    constraint_var: str,
    component: DomainComponent | DomainComponentUnion,
    /,
    *,
    evolution_var: str = "t",
    func: DomainFunction | ArrayLike | None = None,
    time_derivative_order: int = 0,
    mode: Literal["reverse", "forward"] = "reverse",
    time_derivative_backend: Literal["ad", "jet"] = "ad",
    num_points: NumPoints | tuple[Any, ...],
    structure: ProductStructure,
    coord_separable: Mapping[str, Any] | None = None,
    dense_structure: ProductStructure | None = None,
    sampler: str = "latin_hypercube",
    weight: ArrayLike = 1.0,
    label: str | None = None,
    over: str | tuple[str, ...] | None = None,
    reduction: Literal["mean", "integral"] = "mean",
) -> FunctionalConstraint:
    r"""Continuous initial-condition constraint on a fixed-start time slice.

    Enforces

    $$
    \left.\frac{\partial^n u}{\partial t^n}\right|_{t=t_0} = g
    $$

    by sampling the initial slice, where `n = time_derivative_order`.

    **Arguments:**

    - `constraint_var`: Name of the constrained field.
    - `component`: Domain component on the initial surface (or a union of such components).
    - `evolution_var`: Time label used for the initial slice (default: `"t"`).
    - `func`: Target value/function $g$ (defaults to 0).
    - `time_derivative_order`: Derivative order $n$ for $\partial^n/\partial t^n$.
    - `mode`: Differentiation mode (`"reverse"` or `"forward"`).
    - `time_derivative_backend`: Backend for time derivatives (`"ad"` or `"jet"`).
    - `num_points`: Number of initial-slice points to sample (paired or structured; see `structure`).
    - `structure`: A `ProductStructure` describing how variables are sampled/blocked.
    - `coord_separable`: Optional coord-separable sampling spec (per label).
    - `dense_structure`: Optional dense structure used when sampling produces dense batches.
    - `sampler`: Sampling scheme (e.g. `"latin_hypercube"`).
    - `weight`: Scalar multiplier applied to this term.
    - `label`: Optional label for logging.
    - `over`: Optional subset of labels to reduce/integrate over.
    - `reduction`: `"mean"` or `"integral"`.
    """
    if isinstance(component, DomainComponentUnion):
        raise TypeError(
            "ContinuousInitialConstraint requires a DomainComponent, not a union."
        )
    if evolution_var not in component.domain.labels:
        raise KeyError(
            f"Label {evolution_var!r} not in domain {component.domain.labels}."
        )
    if not isinstance(component.spec.component_for(evolution_var), FixedStart):
        raise ValueError(
            "ContinuousInitialConstraint requires a component with "
            f"{evolution_var!r}: FixedStart()."
        )

    order = int(time_derivative_order)
    if order < 0:
        raise ValueError("time_derivative_order must be non-negative.")

    value = 0.0 if func is None else _coerce_value(func, component)

    def operator(u: DomainFunction, /) -> DomainFunction:
        return (
            dt_n(
                u,
                var=evolution_var,
                order=order,
                mode=mode,
                backend=time_derivative_backend,
            )
            - value
        )

    return FunctionalConstraint.from_operator(
        component=component,
        operator=operator,
        constraint_vars=constraint_var,
        num_points=num_points,
        structure=structure,
        coord_separable=coord_separable,
        dense_structure=dense_structure,
        sampler=sampler,
        weight=weight,
        label=label,
        over=over,
        reduction=reduction,
    )


def DiscreteInitialConstraint(
    constraint_var: str,
    component: DomainComponent | DomainComponentUnion,
    /,
    *,
    evolution_var: str = "t",
    points: Mapping[str, ArrayLike] | ArrayLike,
    values: ArrayLike,
    time_derivative_order: int = 0,
    mode: Literal["reverse", "forward"] = "reverse",
    time_derivative_backend: Literal["ad", "jet"] = "ad",
    idw_exponent: float = 2.0,
    eps_snap: float = 1e-12,
    lengthscales: Mapping[str, float] | None = None,
    weight: ArrayLike = 1.0,
    label: str | None = None,
    reduction: Literal["mean", "sum"] = "mean",
) -> PointSetConstraint:
    r"""Discrete initial-condition constraint from anchor points on the initial surface.

    Enforces

    $$
    \left.\frac{\partial^n u}{\partial t^n}\right|_{t=t_0}(x_i) \approx y_i
    $$

    at provided anchor points, where `n = time_derivative_order` and $t_0$ is the
    start of the time interval.

    **Arguments:**

    - `constraint_var`: Name of the constrained field.
    - `component`: Domain component on the initial surface (or a union of such components).
    - `evolution_var`: Time label used for the initial slice (default: `"t"`).
    - `points`: Anchor point coordinates (mapping by label, or stacked array).
    - `values`: Target values $y_i$ at the anchor points.
    - `time_derivative_order`: Derivative order $n$ for $\partial^n/\partial t^n$.
    - `mode`: Differentiation mode (`"reverse"` or `"forward"`).
    - `time_derivative_backend`: Backend for time derivatives (`"ad"` or `"jet"`).
    - `idw_exponent`: Power $p$ in IDW weights $w_j \propto (\|z-z_j\|^2 + \varepsilon)^{-p/2}$.
    - `eps_snap`: Snap threshold $\varepsilon$ used to return exact anchors when very close.
    - `lengthscales`: Optional per-label lengthscales used inside the distance metric.
    - `weight`: Scalar multiplier applied to this term.
    - `reduction`: `"mean"` or `"sum"`.
    """
    if isinstance(component, DomainComponentUnion):
        raise TypeError(
            "DiscreteInitialConstraint requires a DomainComponent, not a union."
        )
    if evolution_var not in component.domain.labels:
        raise KeyError(
            f"Label {evolution_var!r} not in domain {component.domain.labels}."
        )
    if not isinstance(component.spec.component_for(evolution_var), FixedStart):
        raise ValueError(
            "DiscreteInitialConstraint requires a component with "
            f"{evolution_var!r}: FixedStart()."
        )

    order = int(time_derivative_order)
    if order < 0:
        raise ValueError("time_derivative_order must be non-negative.")

    labels = _non_fixed_labels(component)
    if labels:
        target = idw_interpolant(
            component.domain,
            anchors=points,
            values=values,
            labels=labels,
            lengthscales=lengthscales,
            idw_exponent=idw_exponent,
            eps_snap=eps_snap,
        )
    else:
        target = DomainFunction(
            domain=component.domain, deps=(), func=values, metadata={}
        )

    def residual(functions: Mapping[str, DomainFunction], /) -> DomainFunction:
        return (
            dt_n(
                functions[constraint_var],
                var=evolution_var,
                order=order,
                mode=mode,
                backend=time_derivative_backend,
            )
            - target
        )

    pts_in = points if labels else {}
    return PointSetConstraint.from_points(
        component=component,
        points=pts_in,
        residual=residual,
        weight=weight,
        label=label,
        reduction=reduction,
    )
