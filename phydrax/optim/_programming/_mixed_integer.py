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
from .._bounds import Bounds
from .._branch_and_bound import (
    AbstractBranchAndBoundProblem,
    branch_and_bound,
    BranchAndBoundPolicy,
    BranchAndBoundResult,
)
from ._audit import audit_dual_infeasibility_ray, DualRayAudit
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
    infeasibility_certificate: DualRayAudit | None = None


class _MixedIntegerBranchProblem(AbstractBranchAndBoundProblem):
    mixed: MixedIntegerProgram
    policy: MixedIntegerSolvePolicy
    templates: dict[str, ConvexProgramTemplate]
    failed_relaxation: list[bool] = eqx.field(static=True)

    def __init__(self, mixed, policy):
        self.mixed, self.policy = mixed, policy
        self.templates = {
            mixed.relaxation.structure_id: prepare_convex_template(
                mixed.relaxation, policy.relaxation
            )
        }
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
        if node.infeasibility_certificate is not None:
            return None
        if node.execution is None:
            numeric = _replace_bounds(self.mixed.relaxation, node.lower, node.upper)
            if isinstance(numeric, QuadraticProgram):
                node.infeasibility_certificate = _linear_bound_certificate(
                    numeric, self.policy.relaxation.termination.primal_infeasible
                )
                if node.infeasibility_certificate is not None:
                    return None
            if numeric.structure_id not in self.templates:
                self.templates[numeric.structure_id] = prepare_convex_template(
                    numeric, self.policy.relaxation
                )
            node.execution = solve_prepared_convex_program(
                bind_convex_numeric(self.templates[numeric.structure_id], numeric)
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
            if result is not None and bool(np.asarray(result.successful))
            else np.inf
        )

    def feasible(self, node, /):
        result = self._solve(node)
        return (
            not np.any(node.lower > node.upper)
            and result is not None
            and bool(np.asarray(result.successful))
        )

    def complete(self, node, /):
        result = self._solve(node)
        if result is None:
            return False
        primal = np.asarray(result.primal)
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
            if result is not None
            and bool(np.asarray(result.successful))
            and self.complete(node)
            else np.inf
        )

    def branch(self, node, /):
        result = self._solve(node)
        if result is None:
            raise ValueError("A certified infeasible node cannot be branched.")
        primal = np.asarray(result.primal)
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


def _linear_bound_certificate(problem: QuadraticProgram, tolerance: float):
    """Derive affine implications; prune only with an independently audited ray.

    Each inferred bound retains its nonnegative combination of the canonical
    rows. No bound is installed in the numerical problem. Exhausting this finite
    propagation pass is not a feasibility claim: the original solver still runs.
    """
    equality = np.asarray(problem.equality_matrix)
    inequality = np.asarray(problem.inequality_matrix)
    matrix = np.concatenate((equality, -equality, inequality))
    rhs = np.concatenate(
        (
            np.asarray(problem.equality_rhs),
            -np.asarray(problem.equality_rhs),
            np.asarray(problem.inequality_rhs),
        )
    )
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(rhs)):
        return None
    n, m = problem.num_variables, problem.num_equalities
    lower, upper = np.full(n, -np.inf), np.full(n, np.inf)
    lower_proof, upper_proof = [None] * n, [None] * n
    supports = tuple(np.flatnonzero(row) for row in matrix)

    def add(target, proof, scale):
        for index, value in proof.items():
            target[index] = target.get(index, 0.0) + scale * value

    def audit(proof):
        multipliers = np.zeros(len(rhs))
        for index, value in proof.items():
            multipliers[index] = value
        eq = multipliers[:m] - multipliers[m : 2 * m]
        iq = multipliers[2 * m :]
        lo, hi = np.zeros(n), np.zeros(n)
        fixed = np.asarray(problem.fixed_bound_indices, dtype=int)
        signed = eq[problem.num_user_equalities :]
        lo[fixed], hi[fixed] = np.maximum(-signed, 0), np.maximum(signed, 0)
        start = problem.num_user_inequalities
        middle = start + len(problem.lower_bound_indices)
        lo[np.asarray(problem.lower_bound_indices, dtype=int)] = iq[start:middle]
        hi[np.asarray(problem.upper_bound_indices, dtype=int)] = iq[middle:]
        result = audit_dual_infeasibility_ray(
            problem,
            jnp.asarray(eq[: problem.num_user_equalities]),
            jnp.asarray(iq[: problem.num_user_inequalities]),
            jnp.asarray(lo),
            jnp.asarray(hi),
            tolerance=tolerance,
        )
        return result if bool(np.asarray(result.valid)) else None

    # Affine implication cycles can converge only asymptotically. A dimension-
    # derived cap keeps presolve finite without asserting that a fixed point or
    # a numerically tiny improvement proves feasibility.
    for _ in range(n + len(rhs) + 1):
        changed = False
        for row, indices in enumerate(supports):
            coefficients = matrix[row, indices]
            selected = np.where(coefficients > 0, lower[indices], upper[indices])
            contributions = coefficients * selected
            if np.any(np.isnan(contributions)) or np.any(np.isposinf(contributions)):
                continue
            known = np.isfinite(contributions)
            unknown = int(np.count_nonzero(~known))
            minimum = float(np.sum(contributions[known]))
            proofs = tuple(
                lower_proof[index] if coefficient > 0 else upper_proof[index]
                for index, coefficient in zip(indices, coefficients, strict=True)
            )
            if unknown == 0 and minimum > rhs[row] + tolerance:
                proof = {row: 1.0}
                for coefficient, bound_proof in zip(coefficients, proofs, strict=True):
                    add(proof, bound_proof, abs(coefficient))
                certificate = audit(proof)
                if certificate is not None:
                    return certificate
            for position, index in enumerate(indices):
                if unknown > int(not known[position]):
                    continue
                other = minimum - (contributions[position] if known[position] else 0.0)
                coefficient = coefficients[position]
                candidate = (rhs[row] - other) / coefficient
                if not np.isfinite(candidate):
                    continue
                improves = (
                    candidate < upper[index] - tolerance
                    if coefficient > 0
                    else candidate > lower[index] + tolerance
                )
                if not improves:
                    continue
                proof = {row: 1.0 / abs(coefficient)}
                for other_position, bound_proof in enumerate(proofs):
                    if other_position != position:
                        add(
                            proof,
                            bound_proof,
                            abs(coefficients[other_position] / coefficient),
                        )
                if coefficient > 0:
                    upper[index], upper_proof[index] = candidate, proof
                else:
                    lower[index], lower_proof[index] = candidate, proof
                changed = True
                if lower[index] > upper[index] + tolerance:
                    contradiction = dict(lower_proof[index])
                    add(contradiction, upper_proof[index], 1.0)
                    certificate = audit(contradiction)
                    if certificate is not None:
                        return certificate
        if not changed:
            break
    return None


def _replace_bounds(program, lower, upper):
    if np.array_equal(lower, np.asarray(program.lower_bounds)) and np.array_equal(
        upper, np.asarray(program.upper_bounds)
    ):
        return program
    dtype = program.linear.dtype
    bounds = Bounds(jnp.asarray(lower, dtype=dtype), jnp.asarray(upper, dtype=dtype))
    # Branching changes a finite interval into a fixed coordinate. Rebuild its
    # canonical bound role: opposing inequalities have no strict interior and
    # must not masquerade as the root's unchanged numeric topology.
    if isinstance(program, ConicProgram):
        return ConicProgram(
            program.quadratic,
            program.linear,
            program.constraint_matrix,
            program.constraint_rhs,
            program.cone,
            bounds=bounds,
            problem_id=program.problem_id,
            convexity_evidence=program.convexity_evidence,
        )
    if isinstance(program, LinearProgram):
        return LinearProgram(
            program.linear,
            equality_matrix=program.equality_matrix,
            equality_rhs=program.equality_rhs,
            inequality_matrix=program.inequality_matrix,
            inequality_rhs=program.inequality_rhs,
            bounds=bounds,
            problem_id=program.problem_id,
        )
    return QuadraticProgram(
        program.quadratic,
        program.linear,
        equality_matrix=program.equality_matrix[..., : program.num_user_equalities, :],
        equality_rhs=program.equality_rhs[..., : program.num_user_equalities],
        inequality_matrix=program.inequality_matrix[
            ..., : program.num_user_inequalities, :
        ],
        inequality_rhs=program.inequality_rhs[..., : program.num_user_inequalities],
        bounds=bounds,
        problem_id=program.problem_id,
        convexity_evidence=program.convexity_evidence,
    )


__all__ = [
    "MixedIntegerBranchingRule",
    "MixedIntegerProgram",
    "MixedIntegerResult",
    "MixedIntegerSolvePolicy",
    "MixedIntegerStatus",
    "solve_mixed_integer_program",
]
