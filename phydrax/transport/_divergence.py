#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from jaxtyping import Array

from .._strict import StrictModule
from ._costs import PrecomputedCost
from ._problem import DiscreteTransportProblem
from ._results import AbstractBalancedTransportPlan, AbstractBalancedTransportSolver


class SinkhornDivergenceResult(StrictModule):
    """Debiased Sinkhorn divergence with all three native solves retained."""

    value: Array
    cross: AbstractBalancedTransportPlan
    source_self: AbstractBalancedTransportPlan
    target_self: AbstractBalancedTransportPlan

    @property
    def converged(self) -> Array:
        return (
            self.cross.converged & self.source_self.converged & self.target_self.converged
        )


def sinkhorn_divergence(
    problem: DiscreteTransportProblem,
    solver: AbstractBalancedTransportSolver,
    /,
    *,
    source_self_cost: PrecomputedCost | None = None,
    target_self_cost: PrecomputedCost | None = None,
) -> SinkhornDivergenceResult:
    """Compute the native debiased Sinkhorn divergence for one problem."""
    if not isinstance(problem, DiscreteTransportProblem):
        raise TypeError("problem must be a DiscreteTransportProblem.")
    if not isinstance(solver, AbstractBalancedTransportSolver):
        raise TypeError("solver must implement the balanced transport solver contract.")
    if isinstance(problem.cost, PrecomputedCost):
        if source_self_cost is None or target_self_cost is None:
            raise ValueError(
                "Precomputed cross costs require explicit source_self_cost and "
                "target_self_cost."
            )
        source_cost = source_self_cost
        target_cost = target_self_cost
    else:
        if source_self_cost is not None or target_self_cost is not None:
            raise ValueError(
                "Explicit self costs are only accepted with a precomputed cross cost."
            )
        source_cost = problem.cost
        target_cost = problem.cost
    source_self_problem = DiscreteTransportProblem(
        problem.source,
        problem.source,
        source_cost,
        mass_tolerance=problem.mass_tolerance,
    )
    target_self_problem = DiscreteTransportProblem(
        problem.target,
        problem.target,
        target_cost,
        mass_tolerance=problem.mass_tolerance,
    )
    cross = solver(problem)
    source_self = solver(source_self_problem)
    target_self = solver(target_self_problem)
    value = (
        cross.regularized_objective()
        - 0.5 * source_self.regularized_objective()
        - 0.5 * target_self.regularized_objective()
    )
    return SinkhornDivergenceResult(
        value=value,
        cross=cross,
        source_self=source_self,
        target_self=target_self,
    )


__all__ = ["SinkhornDivergenceResult", "sinkhorn_divergence"]
