#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from .._strict import StrictModule
from ..integration._targets import DiscreteMeasureTarget, WeightedSampleTarget
from ._costs import AbstractGroundCost
from ._divergence import SinkhornDivergenceResult
from ._measure import _FiniteTransportMeasure, EventEncoder, lower_transport_measure
from ._problem import DiscreteTransportProblem
from ._results import require_converged, SinkhornResult
from ._sinkhorn import Sinkhorn


class PreparedSinkhornReference(StrictModule):
    """Fixed target and validated target self-solve for repeated divergence."""

    target: _FiniteTransportMeasure
    cost: AbstractGroundCost
    solver: Sinkhorn
    target_self: SinkhornResult
    mass_tolerance: float


def prepare_sinkhorn_reference(
    target: DiscreteMeasureTarget | WeightedSampleTarget,
    /,
    *,
    cost: AbstractGroundCost,
    solver: Sinkhorn,
    encoder: EventEncoder | None = None,
    mass_tolerance: float = 1e-8,
) -> PreparedSinkhornReference:
    """Prepare an immutable target self-term for repeated Sinkhorn divergence."""
    if not isinstance(cost, AbstractGroundCost):
        raise TypeError("Prepared references require an AbstractGroundCost.")
    if not isinstance(solver, Sinkhorn):
        raise TypeError("solver must be a Sinkhorn solver.")
    target_measure = lower_transport_measure(
        target,
        encoder=encoder,
        name="target",
    )
    target_problem = DiscreteTransportProblem(
        target_measure,
        target_measure,
        cost,
        mass_tolerance=mass_tolerance,
    )
    target_self = require_converged(solver(target_problem))
    return PreparedSinkhornReference(
        target=target_measure,
        cost=cost,
        solver=solver,
        target_self=target_self,
        mass_tolerance=float(mass_tolerance),
    )


def sinkhorn_divergence_against(
    source: DiscreteMeasureTarget | WeightedSampleTarget,
    reference: PreparedSinkhornReference,
    /,
    *,
    encoder: EventEncoder | None = None,
) -> SinkhornDivergenceResult:
    """Evaluate Sinkhorn divergence against one prepared fixed target."""
    if not isinstance(reference, PreparedSinkhornReference):
        raise TypeError("reference must be a PreparedSinkhornReference.")
    source_measure = lower_transport_measure(
        source,
        encoder=encoder,
        name="source",
    )
    cross_problem = DiscreteTransportProblem(
        source_measure,
        reference.target,
        reference.cost,
        mass_tolerance=reference.mass_tolerance,
    )
    source_self_problem = DiscreteTransportProblem(
        source_measure,
        source_measure,
        reference.cost,
        mass_tolerance=reference.mass_tolerance,
    )
    cross = reference.solver(cross_problem)
    source_self = reference.solver(source_self_problem)
    value = (
        cross.regularized_cost
        - 0.5 * source_self.regularized_cost
        - 0.5 * reference.target_self.regularized_cost
    )
    return SinkhornDivergenceResult(
        value=value,
        cross=cross,
        source_self=source_self,
        target_self=reference.target_self,
    )


__all__ = [
    "PreparedSinkhornReference",
    "prepare_sinkhorn_reference",
    "sinkhorn_divergence_against",
]
