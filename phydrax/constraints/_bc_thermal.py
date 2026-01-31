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
from ..operators.differential import directional_derivative
from ._functional import FunctionalConstraint
from ._interpolate import idw_interpolant
from ._pointset import PointSetConstraint


def _normal(component: DomainComponent, *, var: str) -> DomainFunction:
    return component.normal(var=var)


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


# Heat Transfer Boundary Constraints (Continuous)


def ContinuousHeatFluxBoundaryConstraint(
    temperature_var: str,
    component: DomainComponent,
    /,
    *,
    k: DomainFunction | ArrayLike,
    flux: DomainFunction | ArrayLike | None = None,
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
    r"""Prescribed heat-flux (Neumann) boundary condition.

    Enforces $k\\,\\partial T/\\partial n = q$ on the boundary component, where
    $q$ is the heat flux (default $0$).

    **Arguments:**

    - `temperature_var`: Name of the temperature field.
    - `component`: Boundary component.
    - `k`: Thermal conductivity.
    - `flux`: Target flux $q$ (defaults to 0).
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

    - A `FunctionalConstraint` enforcing the heat-flux condition.
    """
    n = _normal(component, var=var)
    target = 0.0 if flux is None else flux

    def operator(u: DomainFunction, /) -> DomainFunction:
        dd = directional_derivative(u, n, var=var, mode=mode)
        return k * dd - target

    return FunctionalConstraint.from_operator(
        component=component,
        operator=operator,
        constraint_vars=temperature_var,
        num_points=num_points,
        structure=structure,
        sampler=sampler,
        weight=weight,
        label=label,
        over=over,
        reduction=reduction,
    )


def ContinuousConvectionBoundaryConstraint(
    temperature_var: str,
    component: DomainComponent,
    /,
    *,
    h: DomainFunction | ArrayLike,
    k: DomainFunction | ArrayLike,
    ambient_temp: DomainFunction | ArrayLike | None = None,
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
    r"""Convection (Robin) boundary condition.

    Enforces $k\\,\\partial T/\\partial n = h\\,(T - T_\\infty)$ on the boundary
    component, where $T_\\infty$ is the ambient temperature (default $0$).

    **Arguments:**

    - `temperature_var`: Name of the temperature field.
    - `component`: Boundary component.
    - `h`: Convection coefficient.
    - `k`: Thermal conductivity.
    - `ambient_temp`: Ambient temperature $T_\\infty$ (defaults to 0).
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

    - A `FunctionalConstraint` enforcing the convection condition.
    """
    n = _normal(component, var=var)
    ambient = 0.0 if ambient_temp is None else ambient_temp

    def operator(u: DomainFunction, /) -> DomainFunction:
        dd = directional_derivative(u, n, var=var, mode=mode)
        return k * dd - h * (u - ambient)

    return FunctionalConstraint.from_operator(
        component=component,
        operator=operator,
        constraint_vars=temperature_var,
        num_points=num_points,
        structure=structure,
        sampler=sampler,
        weight=weight,
        label=label,
        over=over,
        reduction=reduction,
    )


# Heat Transfer Boundary Constraints (Discrete)


def DiscreteRobinBoundaryConstraint(
    constraint_var: str,
    component: DomainComponent,
    /,
    *,
    points: Mapping[str, ArrayLike] | ArrayLike,
    values: ArrayLike,
    dirichlet_coeff: DomainFunction | ArrayLike,
    neumann_coeff: DomainFunction | ArrayLike,
    var: str = "x",
    mode: Literal["reverse", "forward"] = "reverse",
    weight: ArrayLike = 1.0,
    label: str | None = None,
    reduction: Literal["mean", "sum"] = "mean",
) -> PointSetConstraint:
    r"""Discrete Robin constraint evaluated at explicit boundary anchor points.

    Enforces $c_0 T + c_1\,\partial T/\partial n = g$ at the provided points.
    """
    n = _normal(component, var=var)
    target = _interp_target(component, points, values)

    def residual(functions: Mapping[str, DomainFunction], /) -> DomainFunction:
        u = functions[constraint_var]
        dudn = directional_derivative(u, n, var=var, mode=mode)
        return dirichlet_coeff * u + neumann_coeff * dudn - target

    return PointSetConstraint.from_points(
        component=component,
        points=points,
        residual=residual,
        weight=weight,
        label=label,
        reduction=reduction,
    )


def DiscreteHeatFluxBoundaryConstraint(
    temperature_var: str,
    component: DomainComponent,
    /,
    *,
    points: Mapping[str, ArrayLike] | ArrayLike,
    values: ArrayLike,
    k: DomainFunction | ArrayLike,
    var: str = "x",
    mode: Literal["reverse", "forward"] = "reverse",
    weight: ArrayLike = 1.0,
    label: str | None = None,
    reduction: Literal["mean", "sum"] = "mean",
) -> PointSetConstraint:
    r"""Discrete heat-flux (Neumann-type) constraint at explicit anchor points.

    Enforces $k\,\partial T/\partial n = q$ at the provided points.
    """
    n = _normal(component, var=var)
    target = _interp_target(component, points, values)

    def residual(functions: Mapping[str, DomainFunction], /) -> DomainFunction:
        u = functions[temperature_var]
        dudn = directional_derivative(u, n, var=var, mode=mode)
        return k * dudn - target

    return PointSetConstraint.from_points(
        component=component,
        points=points,
        residual=residual,
        weight=weight,
        label=label,
        reduction=reduction,
    )


def DiscreteConvectionBoundaryConstraint(
    temperature_var: str,
    component: DomainComponent,
    /,
    *,
    points: Mapping[str, ArrayLike] | ArrayLike,
    ambient_values: ArrayLike,
    h: DomainFunction | ArrayLike,
    k: DomainFunction | ArrayLike,
    var: str = "x",
    mode: Literal["reverse", "forward"] = "reverse",
    weight: ArrayLike = 1.0,
    label: str | None = None,
    reduction: Literal["mean", "sum"] = "mean",
) -> PointSetConstraint:
    r"""Discrete convection (Robin) constraint at explicit anchor points.

    Enforces $k\,\partial T/\partial n = h\,(T - T_\infty)$ at the provided points,
    where $T_\infty$ is given by `ambient_values` (interpolated when needed).
    """
    n = _normal(component, var=var)
    ambient = _interp_target(component, points, ambient_values)

    def residual(functions: Mapping[str, DomainFunction], /) -> DomainFunction:
        u = functions[temperature_var]
        dudn = directional_derivative(u, n, var=var, mode=mode)
        return k * dudn - h * (u - ambient)

    return PointSetConstraint.from_points(
        component=component,
        points=points,
        residual=residual,
        weight=weight,
        label=label,
        reduction=reduction,
    )
