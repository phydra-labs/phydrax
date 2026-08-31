#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx

from ....optim import (
    AbstractBranchAndBoundProblem,
    branch_and_bound,
    BranchAndBoundPolicy,
    BranchAndBoundResult,
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

    def lower_bound(self, node: PrecedenceNode, /) -> float:
        return float(self.lower_bound_callback(node))

    def feasible(self, node: PrecedenceNode, /) -> bool:
        feasible, _ = self.evaluate_prefix(node)
        return bool(feasible)

    def complete(self, node: PrecedenceNode, /) -> bool:
        return self.space.complete(node)

    def objective(self, node: PrecedenceNode, /) -> float:
        return float(self.complete_objective_callback(node))

    def branch(self, node: PrecedenceNode, /):
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
