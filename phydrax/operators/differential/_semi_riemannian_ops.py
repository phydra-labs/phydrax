#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Literal

import jax.numpy as jnp
import opt_einsum as oe

from phydrax.domain import AbstractGeometry, DomainFunction

from ..._strict import StrictModule
from ...metrix import (
    LeviCivitaConnection,
    LorentzianMetric,
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


__all__ = ["intrinsic_dalembertian", "semi_riemannian_grad"]
