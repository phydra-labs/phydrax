#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from jaxtyping import ArrayLike

from .._strict import StrictModule
from ..integration import (
    DiscreteMeasureTarget,
    IntegrationRealization,
    WeightedSampleTarget,
)
from ..transport import (
    AbstractGroundCost,
    BarycenterResult,
    fixed_support_barycenter_problem,
    FreeSupportBarycenter,
    FreeSupportBarycenterResult,
    SinkhornBarycenter,
)
from ..transport._measure import EventEncoder


FiniteAggregationMeasure = (
    DiscreteMeasureTarget | WeightedSampleTarget | IntegrationRealization
)


class TransportBarycenterAggregationResult(StrictModule):
    """A fixed-support aggregate measure and its complete transport solution."""

    measure: DiscreteMeasureTarget
    transport: BarycenterResult

    @property
    def converged(self):
        return self.transport.converged


class FreeSupportTransportBarycenterAggregationResult(StrictModule):
    """A locally optimized aggregate measure and all alternating solves."""

    measure: DiscreteMeasureTarget
    transport: FreeSupportBarycenterResult

    @property
    def converged(self):
        return self.transport.converged


def aggregate_transport_barycenter(
    measures: tuple[FiniteAggregationMeasure, ...],
    support: FiniteAggregationMeasure,
    /,
    *,
    measure_weights: ArrayLike,
    cost: AbstractGroundCost,
    solver: SinkhornBarycenter,
    encoders: tuple[EventEncoder | None, ...] | None = None,
    support_encoder: EventEncoder | None = None,
    mass_tolerance: float = 1e-8,
) -> TransportBarycenterAggregationResult:
    """Aggregate finite UQ laws on a declared support without losing diagnostics."""
    if not isinstance(solver, SinkhornBarycenter):
        raise TypeError("solver must be a SinkhornBarycenter.")
    problem = fixed_support_barycenter_problem(
        measures,
        support,
        measure_weights=measure_weights,
        cost=cost,
        encoders=encoders,
        support_encoder=support_encoder,
        mass_tolerance=mass_tolerance,
    )
    result = solver(problem)
    return TransportBarycenterAggregationResult(
        measure=result.as_target(provenance="uq-transport-barycenter"),
        transport=result,
    )


def aggregate_free_support_transport_barycenter(
    measures: tuple[FiniteAggregationMeasure, ...],
    initial_support: FiniteAggregationMeasure,
    /,
    *,
    measure_weights: ArrayLike,
    cost: AbstractGroundCost,
    solver: FreeSupportBarycenter,
    encoders: tuple[EventEncoder | None, ...] | None = None,
    support_encoder: EventEncoder | None = None,
    mass_tolerance: float = 1e-8,
) -> FreeSupportTransportBarycenterAggregationResult:
    """Aggregate finite UQ laws by an explicitly initialized local support search."""
    if not isinstance(solver, FreeSupportBarycenter):
        raise TypeError("solver must be a FreeSupportBarycenter.")
    problem = fixed_support_barycenter_problem(
        measures,
        initial_support,
        measure_weights=measure_weights,
        cost=cost,
        encoders=encoders,
        support_encoder=support_encoder,
        mass_tolerance=mass_tolerance,
    )
    result = solver(problem)
    return FreeSupportTransportBarycenterAggregationResult(
        measure=result.as_target(provenance="uq-free-support-transport-barycenter"),
        transport=result,
    )


__all__ = [
    "FreeSupportTransportBarycenterAggregationResult",
    "TransportBarycenterAggregationResult",
    "aggregate_free_support_transport_barycenter",
    "aggregate_transport_barycenter",
]
