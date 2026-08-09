#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from jaxtyping import Array

from .._strict import StrictModule
from ._costs import PrecomputedCost
from ._unbalanced_problem import UnbalancedTransportProblem
from ._unbalanced_results import UnbalancedSinkhornResult
from ._unbalanced_sinkhorn import UnbalancedSinkhorn


class UnbalancedSinkhornDivergenceResult(StrictModule):
    """Debiased unbalanced Sinkhorn divergence retaining all native solves."""

    value: Array
    mass_correction: Array
    cross: UnbalancedSinkhornResult
    source_self: UnbalancedSinkhornResult
    target_self: UnbalancedSinkhornResult

    @property
    def converged(self) -> Array:
        return (
            self.cross.converged
            & self.source_self.converged
            & self.target_self.converged
        )


def unbalanced_sinkhorn_divergence(
    problem: UnbalancedTransportProblem,
    solver: UnbalancedSinkhorn,
    /,
    *,
    source_self_cost: PrecomputedCost | None = None,
    target_self_cost: PrecomputedCost | None = None,
) -> UnbalancedSinkhornDivergenceResult:
    """Compute KL-unbalanced Sinkhorn divergence with its mass correction."""
    if not isinstance(problem, UnbalancedTransportProblem):
        raise TypeError("problem must be an UnbalancedTransportProblem.")
    if not isinstance(solver, UnbalancedSinkhorn):
        raise TypeError("solver must be an UnbalancedSinkhorn solver.")
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
    source_self_problem = UnbalancedTransportProblem(
        problem.source,
        problem.source,
        source_cost,
        source_marginal_penalty=problem.source_marginal_penalty,
        target_marginal_penalty=problem.target_marginal_penalty,
    )
    target_self_problem = UnbalancedTransportProblem(
        problem.target,
        problem.target,
        target_cost,
        source_marginal_penalty=problem.source_marginal_penalty,
        target_marginal_penalty=problem.target_marginal_penalty,
    )
    cross = solver(problem)
    source_self = solver(source_self_problem)
    target_self = solver(target_self_problem)
    correction = _mass_correction(problem, solver)
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


def _mass_correction(
    problem: UnbalancedTransportProblem,
    solver: UnbalancedSinkhorn,
    /,
) -> Array:
    difference = problem.source_mass - problem.target_mass
    return 0.5 * solver.epsilon * difference * difference


__all__ = [
    "UnbalancedSinkhornDivergenceResult",
    "unbalanced_sinkhorn_divergence",
]
