#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
import heapq
from collections.abc import Sequence
from enum import IntEnum
from math import inf, isfinite
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from .._strict import StrictModule
from .._trainable import NonTrainableState


class BranchAndBoundStatus(IntEnum):
    OPTIMAL = 0
    GAP_REACHED = 1
    WORK_LIMIT = 2
    INFEASIBLE = 3


class BranchAndBoundPolicy(StrictModule, NonTrainableState):
    maximum_nodes: int = eqx.field(static=True)
    absolute_gap: float = eqx.field(static=True)
    relative_gap: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_nodes: int = 100_000,
        absolute_gap: float = 0.0,
        relative_gap: float = 0.0,
    ):
        if int(maximum_nodes) <= 0:
            raise ValueError("maximum_nodes must be positive.")
        if absolute_gap < 0.0 or relative_gap < 0.0:
            raise ValueError("Branch-and-bound gaps must be nonnegative.")
        self.maximum_nodes = int(maximum_nodes)
        self.absolute_gap = float(absolute_gap)
        self.relative_gap = float(relative_gap)


class AbstractBranchAndBoundProblem(StrictModule, NonTrainableState):
    problem_id: str = eqx.field(static=True)

    @abc.abstractmethod
    def root(self, /) -> Any:
        raise NotImplementedError

    @abc.abstractmethod
    def node_id(self, node: Any, /) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def lower_bound(self, node: Any, /) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def feasible(self, node: Any, /) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def complete(self, node: Any, /) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def objective(self, node: Any, /) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def branch(self, node: Any, /) -> Sequence[Any]:
        raise NotImplementedError


class BranchAndBoundResult(StrictModule):
    incumbent: Any
    objective: Array
    global_lower_bound: Array
    absolute_gap: Array
    relative_gap: Array
    explored_nodes: Array
    pruned_nodes: Array
    frontier_size: Array
    status: Array
    problem_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.status == int(BranchAndBoundStatus.OPTIMAL)


def branch_and_bound(
    problem: AbstractBranchAndBoundProblem,
    /,
    *,
    policy: BranchAndBoundPolicy | None = None,
) -> BranchAndBoundResult:
    """Deterministic exact best-bound search with certified optimality gap."""
    if not isinstance(problem, AbstractBranchAndBoundProblem):
        raise TypeError("problem must be an AbstractBranchAndBoundProblem.")
    policy_ = BranchAndBoundPolicy() if policy is None else policy
    root = problem.root()
    root_bound = float(problem.lower_bound(root))
    if not isfinite(root_bound):
        return BranchAndBoundResult(
            None,
            jnp.asarray(jnp.inf),
            jnp.asarray(jnp.inf),
            jnp.asarray(jnp.inf),
            jnp.asarray(jnp.inf),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(1, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(int(BranchAndBoundStatus.INFEASIBLE), dtype=jnp.int32),
            problem.problem_id,
        )
    frontier: list[tuple[float, str, int, Any]] = []
    counter = 0
    heapq.heappush(frontier, (root_bound, problem.node_id(root), counter, root))
    incumbent = None
    incumbent_value = inf
    explored = 0
    pruned = 0
    status = BranchAndBoundStatus.WORK_LIMIT
    while frontier and explored < policy_.maximum_nodes:
        bound, _, _, node = heapq.heappop(frontier)
        explored += 1
        if bound >= incumbent_value:
            pruned += 1
            continue
        if not problem.feasible(node):
            pruned += 1
            continue
        if problem.complete(node):
            value = float(problem.objective(node))
            if value < incumbent_value:
                incumbent, incumbent_value = node, value
        else:
            children = tuple(problem.branch(node))
            for child in sorted(children, key=problem.node_id):
                child_bound = float(problem.lower_bound(child))
                if not isfinite(child_bound) or child_bound >= incumbent_value:
                    pruned += 1
                    continue
                counter += 1
                heapq.heappush(
                    frontier,
                    (child_bound, problem.node_id(child), counter, child),
                )
        if incumbent is not None and frontier and frontier[0][0] >= incumbent_value:
            # The minimum certified bound dominates the incumbent, hence every
            # remaining node is prunable. Exhaust the frontier before deciding
            # whether termination is exact or merely within a requested gap.
            pruned += len(frontier)
            frontier.clear()
        lower = frontier[0][0] if frontier else incumbent_value
        absolute_gap = incumbent_value - lower
        relative_gap = absolute_gap / max(abs(incumbent_value), 1.0)
        if incumbent is not None and (
            absolute_gap <= policy_.absolute_gap or relative_gap <= policy_.relative_gap
        ):
            status = (
                BranchAndBoundStatus.OPTIMAL
                if not frontier
                else BranchAndBoundStatus.GAP_REACHED
            )
            break
    else:
        if incumbent is None:
            status = BranchAndBoundStatus.INFEASIBLE
        elif not frontier:
            status = BranchAndBoundStatus.OPTIMAL
    lower = frontier[0][0] if frontier else incumbent_value
    absolute_gap = incumbent_value - lower
    relative_gap = absolute_gap / max(abs(incumbent_value), 1.0)
    if not frontier and incumbent is not None:
        status = BranchAndBoundStatus.OPTIMAL
    return BranchAndBoundResult(
        incumbent,
        jnp.asarray(incumbent_value),
        jnp.asarray(lower),
        jnp.asarray(absolute_gap),
        jnp.asarray(relative_gap),
        jnp.asarray(explored, dtype=jnp.int32),
        jnp.asarray(pruned, dtype=jnp.int32),
        jnp.asarray(len(frontier), dtype=jnp.int32),
        jnp.asarray(int(status), dtype=jnp.int32),
        problem.problem_id,
    )


__all__ = [
    "AbstractBranchAndBoundProblem",
    "BranchAndBoundPolicy",
    "BranchAndBoundResult",
    "BranchAndBoundStatus",
    "branch_and_bound",
]
