#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import equinox as eqx
import jax.numpy as jnp

from phydrax.domain import DomainFunction

from ..._strict import StrictModule
from ...metrix import (
    ComplexCoordinateConvention,
    KahlerPotentialGeometry,
    RiemannianMetric,
)
from ._domain_ops import _factor_and_dim, _resolve_var


Operation = Literal["monge-ampere", "positivity"]


class _KahlerPotentialCallable(StrictModule):
    potential: DomainFunction
    reference_metric: RiemannianMetric
    convention: ComplexCoordinateConvention
    target_log_volume: Callable[[jnp.ndarray], jnp.ndarray] | None
    potential_positions: tuple[int, ...]
    coordinate_position: int = eqx.field(static=True)
    potential_coordinate_position: int = eqx.field(static=True)
    operation: Operation = eqx.field(static=True)
    normalization: jnp.ndarray

    def __init__(
        self,
        potential: DomainFunction,
        reference_metric: RiemannianMetric,
        convention: ComplexCoordinateConvention,
        target_log_volume,
        potential_positions: tuple[int, ...],
        coordinate_position: int,
        potential_coordinate_position: int,
        operation: Operation,
        normalization,
        /,
    ):
        self.potential = potential
        self.reference_metric = reference_metric
        self.convention = convention
        self.target_log_volume = target_log_volume
        self.potential_positions = potential_positions
        self.coordinate_position = int(coordinate_position)
        self.potential_coordinate_position = int(potential_coordinate_position)
        self.operation = operation
        self.normalization = jnp.asarray(normalization)

    def __call__(self, *args: Any, key=None, **kwargs: Any):
        bound = [args[position] for position in self.potential_positions]

        def point_potential(point):
            local = list(bound)
            local[self.potential_coordinate_position] = point
            return self.potential.func(*local, key=key, **kwargs)

        geometry = KahlerPotentialGeometry(
            self.reference_metric,
            self.convention,
            point_potential,
        )
        coordinates = args[self.coordinate_position]
        if self.operation == "positivity":
            return geometry.positivity_margin(coordinates)
        assert self.target_log_volume is not None
        return geometry.monge_ampere_residual(
            self.target_log_volume,
            coordinates,
            normalization=self.normalization,
        )


def _kahler_operator(
    potential: DomainFunction,
    reference_metric: RiemannianMetric,
    convention: ComplexCoordinateConvention,
    operation: Operation,
    /,
    *,
    target_log_volume=None,
    normalization=0.0,
    var: str | None,
) -> DomainFunction:
    if not isinstance(potential, DomainFunction):
        raise TypeError("potential must be a DomainFunction.")
    if not isinstance(reference_metric, RiemannianMetric):
        raise TypeError("reference_metric must be a RiemannianMetric.")
    if not isinstance(convention, ComplexCoordinateConvention):
        raise TypeError("convention must be a ComplexCoordinateConvention.")
    if not reference_metric.chart.compatible_with(convention.chart):
        raise ValueError("Reference metric and complex convention must share a chart.")
    var_ = _resolve_var(potential, var)
    _, dimension = _factor_and_dim(potential, var_)
    if dimension != convention.chart.dimension or var_ not in potential.deps:
        raise ValueError("Potential must depend on one matching geometry variable.")
    deps = tuple(
        label
        for label in potential.domain.labels
        if label in potential.deps or label == var_
    )
    positions = {label: index for index, label in enumerate(deps)}
    return DomainFunction(
        domain=potential.domain,
        deps=deps,
        func=_KahlerPotentialCallable(
            potential,
            reference_metric,
            convention,
            target_log_volume,
            tuple(positions[label] for label in potential.deps),
            positions[var_],
            potential.deps.index(var_),
            operation,
            normalization,
        ),
        metadata=potential.metadata,
    )


def domain_monge_ampere_residual(
    potential: DomainFunction,
    reference_metric: RiemannianMetric,
    convention: ComplexCoordinateConvention,
    target_log_volume: Callable,
    /,
    *,
    normalization=0.0,
    var: str | None = None,
) -> DomainFunction:
    if not callable(target_log_volume):
        raise TypeError("target_log_volume must be callable.")
    return _kahler_operator(
        potential,
        reference_metric,
        convention,
        "monge-ampere",
        target_log_volume=target_log_volume,
        normalization=normalization,
        var=var,
    )


def domain_kahler_positivity_margin(
    potential: DomainFunction,
    reference_metric: RiemannianMetric,
    convention: ComplexCoordinateConvention,
    /,
    *,
    var: str | None = None,
) -> DomainFunction:
    return _kahler_operator(
        potential,
        reference_metric,
        convention,
        "positivity",
        var=var,
    )


__all__ = ["domain_kahler_positivity_margin", "domain_monge_ampere_residual"]
