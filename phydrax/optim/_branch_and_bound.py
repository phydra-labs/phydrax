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
    UNBOUNDED = 4
    EVALUATION_FAILURE = 5


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


class BranchBoundEvidence(StrictModule, NonTrainableState):
    """A lower bound together with its proof authority."""

    value: float = eqx.field(static=True)
    certified: bool = eqx.field(static=True)
    certificate_id: str = eqx.field(static=True)

    def __init__(
        self,
        value: float,
        /,
        *,
        certified: bool,
        certificate_id: str = "",
    ):
        value_ = float(value)
        if not isfinite(value_) and value_ != -inf:
            raise ValueError("A node lower bound must be finite or negative infinity.")
        certificate_id_ = str(certificate_id)
        if bool(certified) and not certificate_id_:
            raise ValueError("A certified lower bound requires a certificate_id.")
        self.value = value_
        self.certified = bool(certified)
        self.certificate_id = certificate_id_


class BranchCandidate(StrictModule, NonTrainableState):
    """An independently admitted incumbent candidate."""

    candidate: Any
    objective: float = eqx.field(static=True)
    certificate_id: str = eqx.field(static=True)

    def __init__(
        self,
        candidate: Any,
        objective: float,
        /,
        *,
        certificate_id: str,
    ):
        objective_ = float(objective)
        if not isfinite(objective_):
            raise ValueError("A branch-and-bound candidate objective must be finite.")
        certificate_id_ = str(certificate_id)
        if not certificate_id_:
            raise ValueError("A branch-and-bound candidate requires a certificate_id.")
        self.candidate = candidate
        self.objective = objective_
        self.certificate_id = certificate_id_


class BranchNodeFailure(StrictModule, NonTrainableState):
    """An unresolved node evaluation that prevents a global conclusion."""

    kind: str = eqx.field(static=True)
    message: str = eqx.field(static=True)

    def __init__(self, kind: str, message: str, /):
        kind_ = str(kind)
        if not kind_:
            raise ValueError("A branch node failure requires a nonempty kind.")
        self.kind = kind_
        self.message = str(message)


class BranchNodeEvaluation(StrictModule, NonTrainableState):
    """One complete, reusable node transaction."""

    lower_bound: BranchBoundEvidence | None
    candidate: BranchCandidate | None
    terminal: bool = eqx.field(static=True)
    infeasible: bool = eqx.field(static=True)
    unbounded: bool = eqx.field(static=True)
    certificate_id: str = eqx.field(static=True)
    failure: BranchNodeFailure | None
    state: Any

    def __init__(
        self,
        *,
        lower_bound: BranchBoundEvidence | None = None,
        candidate: BranchCandidate | None = None,
        terminal: bool = False,
        infeasible: bool = False,
        unbounded: bool = False,
        certificate_id: str = "",
        failure: BranchNodeFailure | None = None,
        state: Any = None,
    ):
        if lower_bound is not None and not isinstance(lower_bound, BranchBoundEvidence):
            raise TypeError("lower_bound must be BranchBoundEvidence or None.")
        if candidate is not None and not isinstance(candidate, BranchCandidate):
            raise TypeError("candidate must be BranchCandidate or None.")
        if failure is not None and not isinstance(failure, BranchNodeFailure):
            raise TypeError("failure must be BranchNodeFailure or None.")
        exclusive = (
            int(bool(infeasible)) + int(bool(unbounded)) + int(failure is not None)
        )
        if exclusive > 1:
            raise ValueError(
                "A node cannot be infeasible, unbounded, and failed simultaneously."
            )
        certificate_id_ = str(certificate_id)
        if (bool(infeasible) or bool(unbounded)) and not certificate_id_:
            raise ValueError(
                "Certified infeasibility or unboundedness requires a certificate_id."
            )
        if (bool(infeasible) or bool(unbounded)) and lower_bound is not None:
            raise ValueError(
                "Infeasible and unbounded nodes must not encode proof as a bound."
            )
        self.lower_bound = lower_bound
        self.candidate = candidate
        self.terminal = bool(terminal)
        self.infeasible = bool(infeasible)
        self.unbounded = bool(unbounded)
        self.certificate_id = certificate_id_
        self.failure = failure
        self.state = state

    @classmethod
    def proven_infeasible(
        cls, certificate_id: str, /, *, state: Any = None
    ) -> BranchNodeEvaluation:
        return cls(infeasible=True, certificate_id=certificate_id, state=state)

    @classmethod
    def proven_unbounded(
        cls, certificate_id: str, /, *, state: Any = None
    ) -> BranchNodeEvaluation:
        return cls(unbounded=True, certificate_id=certificate_id, state=state)

    @classmethod
    def failed(
        cls,
        kind: str,
        message: str,
        /,
        *,
        candidate: BranchCandidate | None = None,
        state: Any = None,
    ) -> BranchNodeEvaluation:
        return cls(
            candidate=candidate,
            failure=BranchNodeFailure(kind, message),
            state=state,
        )


class AbstractBranchAndBoundProblem(StrictModule, NonTrainableState):
    problem_id: str = eqx.field(static=True)

    @abc.abstractmethod
    def root(self, /) -> Any:
        raise NotImplementedError

    @abc.abstractmethod
    def node_id(self, node: Any, /) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate(self, node: Any, /) -> BranchNodeEvaluation:
        raise NotImplementedError

    @abc.abstractmethod
    def branch(self, node: Any, evaluation: BranchNodeEvaluation, /) -> Sequence[Any]:
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
    incumbent_available: Array
    global_lower_bound_certified: Array
    search_complete: Array
    failure: BranchNodeFailure | None
    problem_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return (
            (self.status == int(BranchAndBoundStatus.OPTIMAL))
            & self.incumbent_available
            & self.global_lower_bound_certified
            & self.search_complete
        )


def branch_and_bound(
    problem: AbstractBranchAndBoundProblem,
    /,
    *,
    policy: BranchAndBoundPolicy | None = None,
    initial_candidate: BranchCandidate | None = None,
) -> BranchAndBoundResult:
    """Deterministic best-bound search with explicit proof-bearing node effects."""
    if not isinstance(problem, AbstractBranchAndBoundProblem):
        raise TypeError("problem must be an AbstractBranchAndBoundProblem.")
    policy_ = BranchAndBoundPolicy() if policy is None else policy
    if not isinstance(policy_, BranchAndBoundPolicy):
        raise TypeError("policy must be a BranchAndBoundPolicy.")
    if initial_candidate is not None and not isinstance(
        initial_candidate, BranchCandidate
    ):
        raise TypeError("initial_candidate must be BranchCandidate or None.")
    root = problem.root()
    frontier: list[tuple[float, str, int, Any, BranchNodeEvaluation | None, bool]] = [
        (-inf, problem.node_id(root), 0, root, None, False)
    ]
    counter = 0
    incumbent = None if initial_candidate is None else initial_candidate.candidate
    incumbent_value = inf if initial_candidate is None else initial_candidate.objective
    explored = 0
    pruned = 0
    status = BranchAndBoundStatus.WORK_LIMIT
    failure = None
    unresolved_bound = inf
    terminal_without_candidate = False

    while frontier:
        frontier_bound = min(frontier[0][0], incumbent_value)
        if incumbent is not None:
            absolute_gap = incumbent_value - frontier_bound
            relative_gap = absolute_gap / max(abs(incumbent_value), 1.0)
            if (
                absolute_gap <= policy_.absolute_gap
                or relative_gap <= policy_.relative_gap
            ):
                if frontier[0][0] >= incumbent_value:
                    pruned += len(frontier)
                    frontier.clear()
                    status = BranchAndBoundStatus.OPTIMAL
                else:
                    status = BranchAndBoundStatus.GAP_REACHED
                break

        (
            inherited_bound,
            _,
            entry_counter,
            node,
            evaluation,
            counted,
        ) = heapq.heappop(frontier)
        fresh = not counted
        if fresh:
            if explored >= policy_.maximum_nodes:
                heapq.heappush(
                    frontier,
                    (
                        inherited_bound,
                        problem.node_id(node),
                        entry_counter,
                        node,
                        evaluation,
                        False,
                    ),
                )
                status = BranchAndBoundStatus.WORK_LIMIT
                break
            if evaluation is None:
                evaluation = problem.evaluate(node)
            explored += 1
            if not isinstance(evaluation, BranchNodeEvaluation):
                raise TypeError(
                    "AbstractBranchAndBoundProblem.evaluate must return "
                    "BranchNodeEvaluation."
                )
            if evaluation.candidate is not None:
                candidate = evaluation.candidate
                if candidate.objective < incumbent_value:
                    incumbent = candidate.candidate
                    incumbent_value = candidate.objective

        if evaluation.failure is not None:
            failure = evaluation.failure
            unresolved_bound = inherited_bound
            status = BranchAndBoundStatus.EVALUATION_FAILURE
            break
        if evaluation.unbounded:
            status = BranchAndBoundStatus.UNBOUNDED
            unresolved_bound = -inf
            break
        if evaluation.infeasible:
            pruned += 1
            continue

        node_bound = inherited_bound
        if evaluation.lower_bound is not None and evaluation.lower_bound.certified:
            node_bound = max(node_bound, evaluation.lower_bound.value)

        if node_bound >= incumbent_value:
            pruned += 1
            continue

        if frontier and node_bound > frontier[0][0]:
            counter += 1
            heapq.heappush(
                frontier,
                (
                    node_bound,
                    problem.node_id(node),
                    counter,
                    node,
                    evaluation,
                    True,
                ),
            )
            continue

        if evaluation.terminal:
            terminal_without_candidate |= evaluation.candidate is None
            continue

        children = tuple(problem.branch(node, evaluation))
        if not children:
            failure = BranchNodeFailure(
                "empty-branch",
                "A nonterminal feasible node produced no children.",
            )
            unresolved_bound = node_bound
            status = BranchAndBoundStatus.EVALUATION_FAILURE
            break
        unbounded_child = False
        for child in sorted(children, key=problem.node_id):
            child_evaluation = problem.evaluate(child)
            if not isinstance(child_evaluation, BranchNodeEvaluation):
                raise TypeError(
                    "AbstractBranchAndBoundProblem.evaluate must return "
                    "BranchNodeEvaluation."
                )
            if child_evaluation.unbounded:
                status = BranchAndBoundStatus.UNBOUNDED
                unresolved_bound = -inf
                frontier.clear()
                unbounded_child = True
                break
            if child_evaluation.infeasible:
                pruned += 1
                continue
            child_bound = node_bound
            if (
                child_evaluation.lower_bound is not None
                and child_evaluation.lower_bound.certified
            ):
                child_bound = max(
                    child_bound,
                    child_evaluation.lower_bound.value,
                )
            if child_bound >= incumbent_value:
                pruned += 1
                continue
            counter += 1
            heapq.heappush(
                frontier,
                (
                    child_bound,
                    problem.node_id(child),
                    counter,
                    child,
                    child_evaluation,
                    False,
                ),
            )
        if unbounded_child:
            break

    if status == BranchAndBoundStatus.WORK_LIMIT and not frontier:
        if incumbent is not None:
            status = BranchAndBoundStatus.OPTIMAL
        elif terminal_without_candidate:
            status = BranchAndBoundStatus.EVALUATION_FAILURE
            failure = BranchNodeFailure(
                "terminal-without-candidate",
                "Search closed a terminal region without an admitted candidate.",
            )
        else:
            status = BranchAndBoundStatus.INFEASIBLE

    search_complete = status in (
        BranchAndBoundStatus.OPTIMAL,
        BranchAndBoundStatus.INFEASIBLE,
        BranchAndBoundStatus.UNBOUNDED,
    )
    if status == BranchAndBoundStatus.UNBOUNDED:
        lower = -inf
    elif status == BranchAndBoundStatus.INFEASIBLE and incumbent is None:
        lower = inf
    elif frontier:
        lower = min(frontier[0][0], incumbent_value)
    elif failure is not None:
        remaining = min(
            frontier[0][0] if frontier else inf,
            unresolved_bound,
        )
        lower = min(remaining, incumbent_value)
    else:
        lower = incumbent_value

    if incumbent is None:
        absolute_gap = inf
        relative_gap = inf
    else:
        absolute_gap = incumbent_value - lower
        relative_gap = absolute_gap / max(abs(incumbent_value), 1.0)

    return BranchAndBoundResult(
        incumbent,
        jnp.asarray(incumbent_value),
        jnp.asarray(lower),
        jnp.asarray(absolute_gap),
        jnp.asarray(relative_gap),
        jnp.asarray(explored, dtype=jnp.int32),
        jnp.asarray(pruned, dtype=jnp.int32),
        jnp.asarray(len(frontier) + int(failure is not None), dtype=jnp.int32),
        jnp.asarray(int(status), dtype=jnp.int32),
        jnp.asarray(incumbent is not None),
        jnp.asarray(True),
        jnp.asarray(search_complete),
        failure,
        problem.problem_id,
    )


__all__ = [
    "AbstractBranchAndBoundProblem",
    "BranchAndBoundPolicy",
    "BranchAndBoundResult",
    "BranchAndBoundStatus",
    "BranchBoundEvidence",
    "BranchCandidate",
    "BranchNodeEvaluation",
    "BranchNodeFailure",
    "branch_and_bound",
]
