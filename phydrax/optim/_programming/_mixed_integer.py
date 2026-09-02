#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from math import isfinite
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from .._branch_and_bound import (
    AbstractBranchAndBoundProblem,
    branch_and_bound,
    BranchAndBoundPolicy,
    BranchAndBoundResult,
)
from ._lifecycle import (
    bind_convex_numeric,
    ConvexProgramExecution,
    ConvexProgramTemplate,
    prepare_convex_template,
    solve_prepared_convex_program,
)
from ._policy import ConvexSolvePolicy
from ._problem import ConicProgram, LinearProgram
from ._quadratic import ConvexProgramResult, QuadraticProgram
from ._types import ConvexProgramStatus


CanonicalMixedIntegerProgram: TypeAlias = LinearProgram | QuadraticProgram | ConicProgram
MixedIntegerBranchingRule: TypeAlias = Literal["most-fractional"]


class MixedIntegerStatus(IntEnum):
    OPTIMAL = 0
    GAP_REACHED = 1
    WORK_LIMIT = 2
    INFEASIBLE = 3
    RELAXATION_FAILURE = 4


class MixedIntegerProgram(StrictModule):
    """Bounded convex canonical program with immutable integral coordinates."""

    relaxation: CanonicalMixedIntegerProgram
    integer_indices: tuple[int, ...] = eqx.field(static=True)
    binary_indices: tuple[int, ...] = eqx.field(static=True)
    discrete_indices: tuple[int, ...] = eqx.field(static=True)
    program_id: str = eqx.field(static=True)
    structure_id: str = eqx.field(static=True)

    def __init__(
        self,
        relaxation: CanonicalMixedIntegerProgram,
        /,
        *,
        integer_indices: tuple[int, ...] = (),
        binary_indices: tuple[int, ...] = (),
        program_id: str = "bounded-mixed-integer-convex-program",
    ):
        if not isinstance(relaxation, (LinearProgram, QuadraticProgram, ConicProgram)):
            raise TypeError("relaxation must be a canonical convex program.")
        if isinstance(relaxation, LinearProgram):
            relaxation = relaxation.as_quadratic_program()
        if relaxation.batch_shape:
            raise ValueError("MixedIntegerProgram requires an unbatched program.")
        integer = _indices(integer_indices, relaxation.num_variables, "integer")
        binary = _indices(binary_indices, relaxation.num_variables, "binary")
        if set(integer) & set(binary):
            raise ValueError("integer_indices and binary_indices must be disjoint.")
        discrete = tuple(sorted((*integer, *binary)))
        if not discrete:
            raise ValueError("At least one discrete variable is required.")
        lower, upper = map(np.asarray, (relaxation.lower_bounds, relaxation.upper_bounds))
        selected = np.asarray(discrete, dtype=np.int64)
        lo, hi = lower[selected], upper[selected]
        if not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)):
            raise ValueError("Every discrete variable requires finite bounds.")
        if np.any(lo > hi) or np.any(lo != np.ceil(lo)) or np.any(hi != np.floor(hi)):
            raise ValueError("Discrete bounds must be consistent integer values.")
        if binary:
            b = np.asarray(binary, dtype=np.int64)
            if np.any(lower[b] < 0.0) or np.any(upper[b] > 1.0):
                raise ValueError("Binary bounds must be contained in [0, 1].")
        limit = float(2 ** (np.finfo(np.dtype(relaxation.linear.dtype)).nmant + 1))
        if np.any(np.abs(lo) > limit) or np.any(np.abs(hi) > limit):
            raise ValueError("Discrete bounds are not exact in the relaxation dtype.")
        identifier = str(program_id)
        if not identifier:
            raise ValueError("program_id must be non-empty.")
        self.relaxation = relaxation
        self.integer_indices = integer
        self.binary_indices = binary
        self.discrete_indices = discrete
        self.program_id = identifier
        self.structure_id = canonical_fingerprint(
            {
                "kind": "mixed-integer-program",
                "program": relaxation.structure_id,
                "integer": list(integer),
                "binary": list(binary),
                "id": identifier,
            }
        )


class MixedIntegerSolvePolicy(StrictModule):
    relaxation: ConvexSolvePolicy
    tree: BranchAndBoundPolicy
    integrality_tolerance: float = eqx.field(static=True)
    branching_rule: MixedIntegerBranchingRule = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        relaxation: ConvexSolvePolicy | None = None,
        /,
        *,
        tree: BranchAndBoundPolicy | None = None,
        integrality_tolerance: float = 1e-7,
        branching_rule: MixedIntegerBranchingRule = "most-fractional",
    ):
        relaxation_ = ConvexSolvePolicy() if relaxation is None else relaxation
        tree_ = BranchAndBoundPolicy() if tree is None else tree
        if not isinstance(relaxation_, ConvexSolvePolicy) or not isinstance(
            tree_, BranchAndBoundPolicy
        ):
            raise TypeError("relaxation and tree policies have incorrect types.")
        if relaxation_.failure.mode != "status":
            raise ValueError("Node relaxations require status failure mode.")
        tolerance = float(integrality_tolerance)
        if not isfinite(tolerance) or not 0.0 < tolerance < 0.5:
            raise ValueError("integrality_tolerance must lie in (0, 0.5).")
        if branching_rule != "most-fractional":
            raise ValueError("Only 'most-fractional' branching is supported.")
        self.relaxation, self.tree = relaxation_, tree_
        self.integrality_tolerance, self.branching_rule = tolerance, branching_rule
        self.policy_id = canonical_fingerprint(
            {
                "kind": "mixed-integer-policy",
                "relaxation": relaxation_.policy_id,
                "nodes": tree_.maximum_nodes,
                "absolute_gap": tree_.absolute_gap,
                "relative_gap": tree_.relative_gap,
                "integrality_tolerance": tolerance,
            }
        )


class MixedIntegerResult(StrictModule):
    primal: Array
    objective: Array
    global_lower_bound: Array
    absolute_gap: Array
    relative_gap: Array
    explored_nodes: Array
    pruned_nodes: Array
    frontier_size: Array
    status: Array
    integral: Array
    relaxation_result: ConvexProgramResult | None
    tree_result: BranchAndBoundResult
    program_id: str = eqx.field(static=True)
    structure_id: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        audited = (
            self.relaxation_result.successful
            if self.relaxation_result is not None
            else jnp.asarray(False)
        )
        return (self.status == int(MixedIntegerStatus.OPTIMAL)) & self.integral & audited


@dataclass(slots=True)
class _Node:
    lower: np.ndarray
    upper: np.ndarray
    path: str
    execution: ConvexProgramExecution | None = None


class _MixedIntegerBranchProblem(AbstractBranchAndBoundProblem):
    mixed: MixedIntegerProgram
    policy: MixedIntegerSolvePolicy
    template: ConvexProgramTemplate
    failed_relaxation: list[bool] = eqx.field(static=True)

    def __init__(self, mixed, policy):
        self.mixed, self.policy = mixed, policy
        self.template = prepare_convex_template(mixed.relaxation, policy.relaxation)
        self.failed_relaxation = [False]
        self.problem_id = mixed.structure_id

    def root(self, /):
        return _Node(
            np.asarray(self.mixed.relaxation.lower_bounds).copy(),
            np.asarray(self.mixed.relaxation.upper_bounds).copy(),
            "root",
        )

    def node_id(self, node, /):
        return node.path

    def _solve(self, node):
        if node.execution is None:
            numeric = _replace_bounds(self.mixed.relaxation, node.lower, node.upper)
            node.execution = solve_prepared_convex_program(
                bind_convex_numeric(self.template, numeric)
            )
            result = node.execution.result
            self.failed_relaxation[0] |= not bool(np.asarray(result.successful)) and int(
                np.asarray(result.status)
            ) != int(ConvexProgramStatus.PRIMAL_INFEASIBLE)
        return node.execution.result

    def lower_bound(self, node, /):
        result = self._solve(node)
        return (
            float(np.asarray(result.objective))
            if bool(np.asarray(result.successful))
            else np.inf
        )

    def feasible(self, node, /):
        return not np.any(node.lower > node.upper) and bool(
            np.asarray(self._solve(node).successful)
        )

    def complete(self, node, /):
        primal = np.asarray(self._solve(node).primal)
        values = primal[np.asarray(self.mixed.discrete_indices, dtype=np.int64)]
        return bool(
            np.all(np.isfinite(values))
            and np.all(
                np.abs(values - np.rint(values)) <= self.policy.integrality_tolerance
            )
        )

    def objective(self, node, /):
        result = self._solve(node)
        return (
            float(np.asarray(result.objective))
            if bool(np.asarray(result.successful)) and self.complete(node)
            else np.inf
        )

    def branch(self, node, /):
        primal = np.asarray(self._solve(node).primal)
        indices = np.asarray(self.mixed.discrete_indices, dtype=np.int64)
        values = primal[indices]
        position = int(np.argmax(np.abs(values - np.rint(values))))
        variable, value = int(indices[position]), float(values[position])
        floor, ceil = np.floor(value), np.ceil(value)
        llo, lhi, rlo, rhi = (
            node.lower.copy(),
            node.upper.copy(),
            node.lower.copy(),
            node.upper.copy(),
        )
        lhi[variable], rlo[variable] = min(lhi[variable], floor), max(rlo[variable], ceil)
        return (
            _Node(llo, lhi, f"{node.path}/x{variable}<={floor:g}"),
            _Node(rlo, rhi, f"{node.path}/x{variable}>={ceil:g}"),
        )


def solve_mixed_integer_program(
    program: MixedIntegerProgram, policy: MixedIntegerSolvePolicy | None = None, /
) -> MixedIntegerResult:
    """Solve a bounded mixed-integer convex program with audited relaxations."""
    if not isinstance(program, MixedIntegerProgram):
        raise TypeError("program must be a MixedIntegerProgram.")
    policy_ = MixedIntegerSolvePolicy() if policy is None else policy
    if not isinstance(policy_, MixedIntegerSolvePolicy):
        raise TypeError("policy must be a MixedIntegerSolvePolicy.")
    problem = _MixedIntegerBranchProblem(program, policy_)
    tree = branch_and_bound(problem, policy=policy_.tree)
    incumbent = tree.incumbent
    result = None if incumbent is None else incumbent.execution.result
    primal = (
        jnp.full(
            (program.relaxation.num_variables,),
            jnp.nan,
            dtype=program.relaxation.linear.dtype,
        )
        if result is None
        else result.primal
    )
    integral = jnp.asarray(
        incumbent is not None
        and problem.complete(incumbent)
        and bool(np.asarray(result.successful))
    )
    status = (
        int(MixedIntegerStatus.RELAXATION_FAILURE)
        if problem.failed_relaxation[0]
        else int(tree.status)
    )
    return MixedIntegerResult(
        primal,
        tree.objective,
        tree.global_lower_bound,
        tree.absolute_gap,
        tree.relative_gap,
        tree.explored_nodes,
        tree.pruned_nodes,
        tree.frontier_size,
        jnp.asarray(status, dtype=jnp.int32),
        integral,
        result,
        tree,
        program.program_id,
        program.structure_id,
        policy_.policy_id,
    )


def _indices(values, variables, name):
    original = tuple(values)
    if any(
        isinstance(value, bool) or not isinstance(value, (int, np.integer))
        for value in original
    ):
        raise TypeError(f"{name}_indices must contain integers.")
    resolved = tuple(int(value) for value in original)
    if len(set(resolved)) != len(resolved) or any(
        value < 0 or value >= variables for value in resolved
    ):
        raise ValueError(f"{name}_indices must be unique in-range coordinates.")
    return tuple(sorted(resolved))


def _replace_conic_bounds(program, lower, upper):
    return eqx.tree_at(
        lambda value: (value.lower_bounds, value.upper_bounds),
        program,
        (
            jnp.asarray(lower, dtype=program.linear.dtype),
            jnp.asarray(upper, dtype=program.linear.dtype),
        ),
    )


def _replace_bounds(program, lower, upper):
    dtype = program.linear.dtype
    lower_, upper_ = jnp.asarray(lower, dtype=dtype), jnp.asarray(upper, dtype=dtype)
    if isinstance(program, ConicProgram):
        return _replace_conic_bounds(program, lower, upper)
    if isinstance(program, LinearProgram):
        canonical = _replace_conic_bounds(program.canonical, lower, upper)
        return eqx.tree_at(
            lambda value: (value.lower_bounds, value.upper_bounds, value.canonical),
            program,
            (lower_, upper_, canonical),
        )
    fixed = jnp.asarray(program.fixed_bound_indices, dtype=jnp.int32)
    lo_idx = jnp.asarray(program.lower_bound_indices, dtype=jnp.int32)
    hi_idx = jnp.asarray(program.upper_bound_indices, dtype=jnp.int32)
    eq_rhs = jnp.concatenate(
        (program.equality_rhs[..., : program.num_user_equalities], lower_[fixed]), axis=-1
    )
    ineq_rhs = jnp.concatenate(
        (
            program.inequality_rhs[..., : program.num_user_inequalities],
            -lower_[lo_idx],
            upper_[hi_idx],
        ),
        axis=-1,
    )
    return eqx.tree_at(
        lambda value: (
            value.lower_bounds,
            value.upper_bounds,
            value.equality_rhs,
            value.inequality_rhs,
        ),
        program,
        (lower_, upper_, eq_rhs, ineq_rhs),
    )


__all__ = [
    "MixedIntegerBranchingRule",
    "MixedIntegerProgram",
    "MixedIntegerResult",
    "MixedIntegerSolvePolicy",
    "MixedIntegerStatus",
    "solve_mixed_integer_program",
]
