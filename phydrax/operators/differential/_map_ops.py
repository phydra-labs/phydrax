#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Literal

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from phydrax.domain import AbstractGeometry, DomainFunction

from ..._strict import StrictModule
from ...metrix import DifferentiableMap, RiemannianMapGeometry, RiemannianMetric
from ._domain_ops import _factor_and_dim, _resolve_var


MapOperation = Literal["energy", "tension", "isometry", "conformality"]


class _RiemannianMapCallable(StrictModule):
    function: DomainFunction
    source_metric: RiemannianMetric
    target_metric: RiemannianMetric
    function_positions: tuple[int, ...]
    coordinate_position: int = eqx.field(static=True)
    function_coordinate_position: int = eqx.field(static=True)
    operation: MapOperation = eqx.field(static=True)

    def __init__(
        self,
        function: DomainFunction,
        source_metric: RiemannianMetric,
        target_metric: RiemannianMetric,
        function_positions: tuple[int, ...],
        coordinate_position: int,
        function_coordinate_position: int,
        operation: MapOperation,
        /,
    ):
        self.function = function
        self.source_metric = source_metric
        self.target_metric = target_metric
        self.function_positions = function_positions
        self.coordinate_position = int(coordinate_position)
        self.function_coordinate_position = int(function_coordinate_position)
        self.operation = operation

    def __call__(self, *args: Any, key=None, **kwargs: Any) -> Array:
        bound = [args[position] for position in self.function_positions]

        def point_map(point: Array) -> Array:
            local = list(bound)
            if self.function_coordinate_position >= 0:
                local[self.function_coordinate_position] = point
            value = jnp.asarray(self.function.func(*local, key=key, **kwargs))
            expected = (self.target_metric.chart.dimension,)
            if value.shape != expected:
                raise ValueError(
                    "Riemannian map field must have pointwise target shape "
                    f"{expected}; got {value.shape}."
                )
            return value

        map = DifferentiableMap(
            self.source_metric.chart,
            self.target_metric.chart,
            point_map,
        )
        geometry = RiemannianMapGeometry(
            map,
            self.source_metric,
            self.target_metric,
        )
        coordinates = args[self.coordinate_position]
        if self.operation == "energy":
            return geometry.energy_density(coordinates)
        if self.operation == "tension":
            return geometry.tension_field(coordinates)
        if self.operation == "isometry":
            return geometry.isometry_residual(coordinates)
        return geometry.conformality_residual(coordinates)


def _map_operator(
    function: DomainFunction,
    source_metric: RiemannianMetric,
    target_metric: RiemannianMetric,
    operation: MapOperation,
    var: str | None,
    /,
) -> DomainFunction:
    if not isinstance(function, DomainFunction):
        raise TypeError("Riemannian map operators require a DomainFunction.")
    if not isinstance(source_metric, RiemannianMetric) or not isinstance(
        target_metric, RiemannianMetric
    ):
        raise TypeError("Riemannian map operators require source and target metrics.")
    var_ = _resolve_var(function, var)
    _, dimension = _factor_and_dim(function, var_)
    if not isinstance(function.domain.factor(var_), AbstractGeometry):
        raise ValueError("Riemannian map operators require a geometry variable.")
    if dimension != source_metric.chart.dimension:
        raise ValueError(
            f"Source metric dimension {source_metric.chart.dimension} does not match "
            f"domain variable {var_!r} dimension {dimension}."
        )
    deps = tuple(
        label
        for label in function.domain.labels
        if label == var_ or label in function.deps
    )
    positions = {label: index for index, label in enumerate(deps)}
    function_positions = tuple(positions[label] for label in function.deps)
    function_coordinate_position = (
        function.deps.index(var_) if var_ in function.deps else -1
    )
    return DomainFunction(
        domain=function.domain,
        deps=deps,
        func=_RiemannianMapCallable(
            function,
            source_metric,
            target_metric,
            function_positions,
            positions[var_],
            function_coordinate_position,
            operation,
        ),
        metadata=function.metadata,
    )


def riemannian_map_energy(
    function: DomainFunction,
    source_metric: RiemannianMetric,
    target_metric: RiemannianMetric,
    /,
    *,
    var: str | None = None,
) -> DomainFunction:
    """Return the pointwise Dirichlet energy density of a map field."""
    return _map_operator(function, source_metric, target_metric, "energy", var)


def riemannian_map_tension(
    function: DomainFunction,
    source_metric: RiemannianMetric,
    target_metric: RiemannianMetric,
    /,
    *,
    var: str | None = None,
) -> DomainFunction:
    """Return the harmonic-map tension field."""
    return _map_operator(function, source_metric, target_metric, "tension", var)


def riemannian_map_isometry_residual(
    function: DomainFunction,
    source_metric: RiemannianMetric,
    target_metric: RiemannianMetric,
    /,
    *,
    var: str | None = None,
) -> DomainFunction:
    """Return the maximum pointwise pullback-metric defect."""
    return _map_operator(function, source_metric, target_metric, "isometry", var)


def riemannian_map_conformality_residual(
    function: DomainFunction,
    source_metric: RiemannianMetric,
    target_metric: RiemannianMetric,
    /,
    *,
    var: str | None = None,
) -> DomainFunction:
    """Return the maximum trace-free pullback-metric defect."""
    return _map_operator(function, source_metric, target_metric, "conformality", var)


__all__ = [
    "riemannian_map_conformality_residual",
    "riemannian_map_energy",
    "riemannian_map_isometry_residual",
    "riemannian_map_tension",
]
