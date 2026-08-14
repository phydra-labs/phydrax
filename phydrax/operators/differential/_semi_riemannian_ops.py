#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Literal

import equinox as eqx
import jax.numpy as jnp
import opt_einsum as oe

from phydrax.domain import AbstractGeometry, Domain, DomainFunction

from ..._strict import StrictModule
from ...metrix import (
    CoordinateChart,
    einstein_tensor,
    LeviCivitaConnection,
    LorentzianConvention,
    LorentzianMetric,
    ricci_tensor,
    riemann_tensor,
    scalar_curvature,
    SemiRiemannianMetric,
)
from ._domain_ops import _factor_and_dim, _resolve_var, grad, hessian


def _geometry_contract(
    function: DomainFunction,
    metric: SemiRiemannianMetric | LorentzianMetric,
    var: str | None,
    /,
) -> tuple[str, int]:
    if not isinstance(metric, (SemiRiemannianMetric, LorentzianMetric)):
        raise TypeError(
            "Signed differential operators require a SemiRiemannianMetric "
            "or LorentzianMetric."
        )
    variable = _resolve_var(function, var)
    _, dimension = _factor_and_dim(function, variable)
    if not isinstance(function.domain.factor(variable), AbstractGeometry):
        raise ValueError("Signed differential operators require a geometry variable.")
    if dimension != metric.chart.dimension:
        raise ValueError(
            f"Metric chart dimension {metric.chart.dimension} does not match "
            f"domain variable {variable!r} dimension {dimension}."
        )
    return variable, dimension


def _dependencies(
    domain_labels: tuple[str, ...],
    functions: tuple[DomainFunction, ...],
    var: str,
    /,
) -> tuple[str, ...]:
    return tuple(
        label
        for label in domain_labels
        if label == var or any(label in function.deps for function in functions)
    )


def _positions(deps: tuple[str, ...], function: DomainFunction, /) -> tuple[int, ...]:
    lookup = {label: position for position, label in enumerate(deps)}
    return tuple(lookup[label] for label in function.deps)


class _SignedGradientCallable(StrictModule):
    differential: DomainFunction
    metric: SemiRiemannianMetric | LorentzianMetric
    differential_positions: tuple[int, ...]
    coordinate_position: int

    def __init__(
        self,
        differential: DomainFunction,
        metric: SemiRiemannianMetric | LorentzianMetric,
        differential_positions: tuple[int, ...],
        coordinate_position: int,
        /,
    ):
        self.differential = differential
        self.metric = metric
        self.differential_positions = differential_positions
        self.coordinate_position = int(coordinate_position)

    def __call__(self, *args: Any, key=None, **kwargs: Any):
        differential = jnp.asarray(
            self.differential.func(
                *[args[position] for position in self.differential_positions],
                key=key,
                **kwargs,
            )
        )
        return oe.contract(
            "...ij,...j->...i",
            self.metric.inverse(args[self.coordinate_position]),
            differential,
        )


class _DalembertianCallable(StrictModule):
    differential: DomainFunction
    second_derivative: DomainFunction
    metric: LorentzianMetric
    differential_positions: tuple[int, ...]
    second_positions: tuple[int, ...]
    coordinate_position: int

    def __init__(
        self,
        differential: DomainFunction,
        second_derivative: DomainFunction,
        metric: LorentzianMetric,
        differential_positions: tuple[int, ...],
        second_positions: tuple[int, ...],
        coordinate_position: int,
        /,
    ):
        self.differential = differential
        self.second_derivative = second_derivative
        self.metric = metric
        self.differential_positions = differential_positions
        self.second_positions = second_positions
        self.coordinate_position = int(coordinate_position)

    def __call__(self, *args: Any, key=None, **kwargs: Any):
        differential = jnp.asarray(
            self.differential.func(
                *[args[position] for position in self.differential_positions],
                key=key,
                **kwargs,
            )
        )
        second = jnp.asarray(
            self.second_derivative.func(
                *[args[position] for position in self.second_positions],
                key=key,
                **kwargs,
            )
        )
        coordinates = args[self.coordinate_position]
        coefficients = LeviCivitaConnection(self.metric).coefficients(coordinates)
        covariant = second - oe.contract("...kij,...k->...ij", coefficients, differential)
        return oe.contract(
            "...ij,...ij->...", self.metric.inverse(coordinates), covariant
        )


def _domain_geometry_variable(
    domain: Domain,
    var: str | None,
    dimension: int,
    /,
) -> str:
    if not isinstance(domain, Domain):
        raise TypeError("Domain Lorentzian operators require a Domain.")
    if var is None:
        differentiable = tuple(
            label for label in domain.labels if domain.coordinate(label).differentiable
        )
        if len(differentiable) != 1:
            raise ValueError(
                "var=None is only valid when there is exactly one differentiable "
                f"variable in the domain; found {differentiable!r}."
            )
        variable = differentiable[0]
    else:
        if var not in domain.labels:
            raise ValueError(f"Unknown var {var!r}; expected one of {domain.labels}.")
        variable = var
    coordinate = domain.coordinate(variable)
    if not coordinate.differentiable:
        raise TypeError(
            "Domain Lorentzian operators require a differentiable coordinate."
        )
    if coordinate.kind == "scalar":
        coordinate_dimension = 1
    elif coordinate.kind == "array" and coordinate.event_shape is not None:
        if len(coordinate.event_shape) != 1:
            raise TypeError(
                "Domain Lorentzian operators require a rank-one coordinate event."
            )
        coordinate_dimension = int(coordinate.event_shape[0])
    else:
        raise TypeError(
            "Domain Lorentzian operators require a scalar or rank-one array coordinate."
        )
    if not isinstance(domain.factor(variable), AbstractGeometry):
        raise ValueError("Domain Lorentzian operators require a geometry variable.")
    if coordinate_dimension != dimension:
        raise ValueError(
            f"Metric chart dimension {dimension} does not match domain variable "
            f"{variable!r} dimension {coordinate_dimension}."
        )
    return variable


class _LorentzianMetricFieldCallable(StrictModule):
    metric: LorentzianMetric

    def __init__(self, metric: LorentzianMetric, /):
        self.metric = metric

    def __call__(self, coordinates: Any, /, *, key=None, **kwargs: Any):
        del key, kwargs
        return self.metric(coordinates)


class _FieldLorentzianMetricMap(StrictModule):
    field: DomainFunction

    def __init__(self, field: DomainFunction, /):
        self.field = field

    def __call__(self, coordinates: Any, /):
        return self.field.func(coordinates, key=None)


class _LorentzianCurvatureCallable(StrictModule):
    metric: LorentzianMetric
    operation: Literal["riemann", "ricci", "scalar", "einstein"] = eqx.field(static=True)

    def __init__(
        self,
        metric: LorentzianMetric,
        operation: Literal["riemann", "ricci", "scalar", "einstein"],
        /,
    ):
        self.metric = metric
        self.operation = operation

    def __call__(self, coordinates: Any, /, *, key=None, **kwargs: Any):
        del key, kwargs
        if self.operation == "riemann":
            return riemann_tensor(self.metric, coordinates)
        if self.operation == "ricci":
            return ricci_tensor(self.metric, coordinates)
        if self.operation == "scalar":
            return scalar_curvature(self.metric, coordinates)
        return einstein_tensor(self.metric, coordinates)


def _domain_lorentzian_curvature(
    domain: Domain,
    metric: LorentzianMetric,
    operation: Literal["riemann", "ricci", "scalar", "einstein"],
    var: str | None,
    /,
) -> DomainFunction:
    if not isinstance(metric, LorentzianMetric):
        raise TypeError("Domain Lorentzian curvature requires a LorentzianMetric.")
    variable = _domain_geometry_variable(
        domain,
        var,
        metric.chart.dimension,
    )
    return DomainFunction(
        domain=domain,
        deps=(variable,),
        func=_LorentzianCurvatureCallable(metric, operation),
    )


def as_lorentzian_metric_field(
    domain: Domain,
    metric: LorentzianMetric,
    /,
    *,
    var: str | None = None,
) -> DomainFunction:
    """Expose one trainable Lorentzian metric as a matrix-valued DomainFunction."""
    if not isinstance(metric, LorentzianMetric):
        raise TypeError("as_lorentzian_metric_field requires a LorentzianMetric.")
    variable = _domain_geometry_variable(
        domain,
        var,
        metric.chart.dimension,
    )
    return DomainFunction(
        domain=domain,
        deps=(variable,),
        func=_LorentzianMetricFieldCallable(metric),
    )


def lorentzian_metric_from_field(
    field: DomainFunction,
    /,
    *,
    chart: CoordinateChart,
    var: str | None = None,
    convention: LorentzianConvention = "mostly_plus",
) -> LorentzianMetric:
    """Recover pointwise metric calculus from a deterministic matrix field."""
    if not isinstance(field, DomainFunction):
        raise TypeError("lorentzian_metric_from_field requires a DomainFunction.")
    variable = _domain_geometry_variable(
        field.domain,
        var,
        chart.dimension,
    )
    if field.deps != (variable,):
        raise ValueError(
            "A Lorentzian metric field must depend only on its spacetime variable; "
            f"got dependencies {field.deps!r}."
        )
    return LorentzianMetric(
        _FieldLorentzianMetricMap(field),
        chart=chart,
        convention=convention,
    )


def domain_riemann_tensor(
    domain: Domain,
    metric: LorentzianMetric,
    /,
    *,
    var: str | None = None,
) -> DomainFunction:
    """Return the spacetime Riemann tensor as a trainable DomainFunction."""
    return _domain_lorentzian_curvature(domain, metric, "riemann", var)


def domain_ricci_tensor(
    domain: Domain,
    metric: LorentzianMetric,
    /,
    *,
    var: str | None = None,
) -> DomainFunction:
    """Return the spacetime Ricci tensor as a trainable DomainFunction."""
    return _domain_lorentzian_curvature(domain, metric, "ricci", var)


def domain_scalar_curvature(
    domain: Domain,
    metric: LorentzianMetric,
    /,
    *,
    var: str | None = None,
) -> DomainFunction:
    """Return spacetime scalar curvature as a trainable DomainFunction."""
    return _domain_lorentzian_curvature(domain, metric, "scalar", var)


def domain_einstein_tensor(
    domain: Domain,
    metric: LorentzianMetric,
    /,
    *,
    var: str | None = None,
) -> DomainFunction:
    """Return the spacetime Einstein tensor as a trainable DomainFunction."""
    return _domain_lorentzian_curvature(domain, metric, "einstein", var)


def semi_riemannian_grad(
    function: DomainFunction,
    metric: SemiRiemannianMetric | LorentzianMetric,
    /,
    *,
    var: str | None = None,
    mode: Literal["reverse", "forward"] = "reverse",
    ad_engine: Literal["auto", "reverse", "forward", "jvp"] = "auto",
) -> DomainFunction:
    """Raise a DomainFunction differential with a declared signed metric."""
    variable, _ = _geometry_contract(function, metric, var)
    differential = grad(function, var=variable, mode=mode, ad_engine=ad_engine)
    deps = _dependencies(function.domain.labels, (differential,), variable)
    return DomainFunction(
        domain=function.domain,
        deps=deps,
        func=_SignedGradientCallable(
            differential,
            metric,
            _positions(deps, differential),
            deps.index(variable),
        ),
        metadata=differential.metadata,
    )


def intrinsic_dalembertian(
    function: DomainFunction,
    metric: LorentzianMetric,
    /,
    *,
    var: str | None = None,
    mode: Literal["reverse", "forward"] = "forward",
    ad_engine: Literal["auto", "reverse", "forward", "jvp"] = "auto",
) -> DomainFunction:
    """Apply the Lorentzian wave operator to a scalar DomainFunction."""
    if not isinstance(metric, LorentzianMetric):
        raise TypeError("intrinsic_dalembertian requires a LorentzianMetric.")
    variable, _ = _geometry_contract(function, metric, var)
    differential = grad(function, var=variable, mode=mode, ad_engine=ad_engine)
    second = hessian(function, var=variable, ad_engine=ad_engine)
    deps = _dependencies(
        function.domain.labels,
        (differential, second),
        variable,
    )
    return DomainFunction(
        domain=function.domain,
        deps=deps,
        func=_DalembertianCallable(
            differential,
            second,
            metric,
            _positions(deps, differential),
            _positions(deps, second),
            deps.index(variable),
        ),
        metadata=second.metadata,
    )


__all__ = [
    "as_lorentzian_metric_field",
    "domain_einstein_tensor",
    "domain_ricci_tensor",
    "domain_riemann_tensor",
    "domain_scalar_curvature",
    "intrinsic_dalembertian",
    "semi_riemannian_grad",
    "lorentzian_metric_from_field",
]
