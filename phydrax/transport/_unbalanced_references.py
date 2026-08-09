#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ._costs import AbstractGroundCost
from ._measure import _FiniteTransportMeasure, EventEncoder, lower_transport_measure
from ._unbalanced_divergence import UnbalancedSinkhornDivergenceResult
from ._unbalanced_problem import _TransportMeasureInput, UnbalancedTransportProblem
from ._unbalanced_results import (
    require_unbalanced_converged,
    UnbalancedSinkhornResult,
)
from ._unbalanced_sinkhorn import UnbalancedSinkhorn


class PreparedUnbalancedSinkhornReference(StrictModule):
    """Fixed physical target and validated unbalanced target self-solve."""

    target: _FiniteTransportMeasure
    cost: AbstractGroundCost
    solver: UnbalancedSinkhorn
    target_self: UnbalancedSinkhornResult
    source_marginal_penalty: Array
    target_marginal_penalty: Array


def prepare_unbalanced_sinkhorn_reference(
    target: _TransportMeasureInput,
    /,
    *,
    cost: AbstractGroundCost,
    solver: UnbalancedSinkhorn,
    source_marginal_penalty: ArrayLike,
    target_marginal_penalty: ArrayLike,
    encoder: EventEncoder | None = None,
) -> PreparedUnbalancedSinkhornReference:
    """Prepare the fixed self term for repeated unbalanced divergence evaluation."""
    if not isinstance(cost, AbstractGroundCost):
        raise TypeError("Prepared references require an AbstractGroundCost.")
    if not isinstance(solver, UnbalancedSinkhorn):
        raise TypeError("solver must be an UnbalancedSinkhorn solver.")
    target_measure = lower_transport_measure(
        target,
        encoder=encoder,
        name="target",
    )
    target_problem = UnbalancedTransportProblem(
        target_measure,
        target_measure,
        cost,
        source_marginal_penalty=source_marginal_penalty,
        target_marginal_penalty=target_marginal_penalty,
    )
    target_self = require_unbalanced_converged(solver(target_problem))
    return PreparedUnbalancedSinkhornReference(
        target=target_measure,
        cost=cost,
        solver=solver,
        target_self=target_self,
        source_marginal_penalty=target_problem.source_marginal_penalty,
        target_marginal_penalty=target_problem.target_marginal_penalty,
    )


def unbalanced_sinkhorn_divergence_against(
    source: _TransportMeasureInput,
    reference: PreparedUnbalancedSinkhornReference,
    /,
    *,
    encoder: EventEncoder | None = None,
) -> UnbalancedSinkhornDivergenceResult:
    """Evaluate unbalanced divergence against one prepared physical target."""
    if not isinstance(reference, PreparedUnbalancedSinkhornReference):
        raise TypeError(
            "reference must be a PreparedUnbalancedSinkhornReference."
        )
    source_measure = lower_transport_measure(
        source,
        encoder=encoder,
        name="source",
    )
    cross_problem = UnbalancedTransportProblem(
        source_measure,
        reference.target,
        reference.cost,
        source_marginal_penalty=reference.source_marginal_penalty,
        target_marginal_penalty=reference.target_marginal_penalty,
    )
    source_problem = UnbalancedTransportProblem(
        source_measure,
        source_measure,
        reference.cost,
        source_marginal_penalty=reference.source_marginal_penalty,
        target_marginal_penalty=reference.target_marginal_penalty,
    )
    cross = reference.solver(cross_problem)
    source_self = reference.solver(source_problem)
    target_self = reference.target_self
    mass_difference = source_measure.mass - reference.target.mass
    correction = 0.5 * reference.solver.epsilon * mass_difference * mass_difference
    return UnbalancedSinkhornDivergenceResult(
        value=(
            cross.regularized_cost
            - 0.5 * source_self.regularized_cost
            - 0.5 * target_self.regularized_cost
            + correction
        ),
        mass_correction=correction,
        cross=cross,
        source_self=source_self,
        target_self=target_self,
    )


__all__ = [
    "PreparedUnbalancedSinkhornReference",
    "prepare_unbalanced_sinkhorn_reference",
    "unbalanced_sinkhorn_divergence_against",
]
