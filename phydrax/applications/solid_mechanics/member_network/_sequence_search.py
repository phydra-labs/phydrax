#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from math import inf, isfinite, isnan
from typing import Any

import equinox as eqx

from ....optim import (
    AbstractBranchAndBoundProblem,
    branch_and_bound,
    BranchAndBoundPolicy,
    BranchAndBoundResult,
    BranchBoundEvidence,
    BranchCandidate,
    BranchNodeEvaluation,
    PrecedenceNode,
    PrecedenceSpace,
)


class ConstructionSequenceSearchProblem(AbstractBranchAndBoundProblem):
    """Precedence-constrained exact construction-order search callbacks."""

    space: PrecedenceSpace
    evaluate_prefix: Callable = eqx.field(static=True)
    lower_bound_callback: Callable = eqx.field(static=True)
    complete_objective_callback: Callable = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        space: PrecedenceSpace,
        evaluate_prefix: Callable[[PrecedenceNode], tuple[bool, Any]],
        lower_bound: Callable[[PrecedenceNode], float],
        complete_objective: Callable[[PrecedenceNode], float],
        /,
        *,
        problem_id: str = "construction-sequence-search",
    ):
        if not isinstance(space, PrecedenceSpace):
            raise TypeError("space must be a PrecedenceSpace.")
        if not all(
            callable(value)
            for value in (evaluate_prefix, lower_bound, complete_objective)
        ):
            raise TypeError("Construction sequence callbacks must be callable.")
        self.space = space
        self.evaluate_prefix = evaluate_prefix
        self.lower_bound_callback = lower_bound
        self.complete_objective_callback = complete_objective
        self.problem_id = str(problem_id)

    def root(self, /) -> PrecedenceNode:
        return self.space.root()

    def node_id(self, node: PrecedenceNode, /) -> str:
        return node.node_id

    def evaluate(self, node: PrecedenceNode, /) -> BranchNodeEvaluation:
        feasible, state = self.evaluate_prefix(node)
        if not bool(feasible):
            return BranchNodeEvaluation.proven_infeasible(
                f"{self.problem_id}:{node.node_id}:prefix",
                state=state,
            )
        bound = float(self.lower_bound_callback(node))
        if isnan(bound):
            return BranchNodeEvaluation.failed(
                "nonfinite-sequence-bound",
                "The construction-sequence lower-bound callback returned NaN.",
                state=state,
            )
        if bound == inf:
            return BranchNodeEvaluation.proven_infeasible(
                f"{self.problem_id}:{node.node_id}:infinite-bound",
                state=state,
            )
        lower_bound = BranchBoundEvidence(
            bound,
            certified=True,
            certificate_id=f"{self.problem_id}:{node.node_id}:lower-bound",
        )
        if not self.space.complete(node):
            return BranchNodeEvaluation(lower_bound=lower_bound, state=state)

        objective = float(self.complete_objective_callback(node))
        if not isfinite(objective):
            return BranchNodeEvaluation.failed(
                "nonfinite-sequence-objective",
                "A complete construction sequence returned a nonfinite objective.",
                state=state,
            )
        if bound > objective:
            raise ValueError(
                "Construction-sequence lower bound exceeds a complete objective."
            )
        return BranchNodeEvaluation(
            lower_bound=lower_bound,
            candidate=BranchCandidate(
                node,
                objective,
                certificate_id=f"{self.problem_id}:{node.node_id}:complete",
            ),
            terminal=True,
            state=state,
        )

    def branch(self, node: PrecedenceNode, evaluation: BranchNodeEvaluation, /):
        del evaluation
        return self.space.branch(node)


def search_construction_sequences(
    problem: ConstructionSequenceSearchProblem,
    /,
    *,
    policy: BranchAndBoundPolicy | None = None,
) -> BranchAndBoundResult:
    return branch_and_bound(problem, policy=policy)


__all__ = [
    "ConstructionSequenceSearchProblem",
    "search_construction_sequences",
]
