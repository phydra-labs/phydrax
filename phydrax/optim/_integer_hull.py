#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from math import isfinite
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, PyTree

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..combinatorial import (
    AbstractBoundableCombinatorialSpace,
    AbstractBoundableLinearCombinatorialMethod,
    CombinatorialCertification,
    CombinatorialFeatureRestriction,
    CombinatorialStatus,
    LinearCombinatorialProblem,
    solve_restricted_combinatorial,
)
from ._branch_and_bound import (
    AbstractBranchAndBoundProblem,
    branch_and_bound,
    BranchAndBoundPolicy,
    BranchAndBoundResult,
    BranchAndBoundStatus,
    BranchBoundEvidence,
    BranchCandidate,
    BranchNodeEvaluation,
)
from ._iterative._types import MinimizationProblem


ConvexObjectiveEvidenceKind: TypeAlias = Literal[
    "construction",
    "verified",
    "asserted",
]


class IntegerHullStatus(IntEnum):
    OPTIMAL = 0
    GAP_REACHED = 1
    WORK_LIMIT = 2
    INFEASIBLE = 3
    EVALUATION_FAILURE = 4


class ConvexObjectiveEvidence(StrictModule, NonTrainableState):
    """Declared authority for objective convexity on a combinatorial hull."""

    kind: ConvexObjectiveEvidenceKind = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: ConvexObjectiveEvidenceKind,
        evidence_id: str,
        /,
    ):
        if kind not in ("construction", "verified", "asserted"):
            raise ValueError("Unknown convex objective evidence kind.")
        identifier = str(evidence_id)
        if not identifier:
            raise ValueError("evidence_id must be nonempty.")
        self.kind = kind
        self.evidence_id = identifier


class IntegerHullProblem(StrictModule):
    """Smooth convex objective over one boundable combinatorial feature hull."""

    objective: MinimizationProblem
    space: AbstractBoundableCombinatorialSpace
    args: Any
    convexity: ConvexObjectiveEvidence
    problem_id: str = eqx.field(static=True)
    structure_id: str = eqx.field(static=True)

    def __init__(
        self,
        objective: MinimizationProblem,
        space: AbstractBoundableCombinatorialSpace,
        /,
        *,
        args: Any = None,
        convexity: ConvexObjectiveEvidence,
        problem_id: str = "integer-hull-convex-program",
    ):
        if not isinstance(objective, MinimizationProblem):
            raise TypeError("objective must be a MinimizationProblem.")
        if objective.bounds is not None or objective.constraints:
            raise ValueError(
                "IntegerHullProblem geometry must be owned entirely by its space."
            )
        if not isinstance(space, AbstractBoundableCombinatorialSpace):
            raise TypeError("space must be an AbstractBoundableCombinatorialSpace.")
        if not isinstance(convexity, ConvexObjectiveEvidence):
            raise TypeError("convexity must be ConvexObjectiveEvidence.")
        identifier = str(problem_id)
        if not identifier:
            raise ValueError("problem_id must be nonempty.")
        lower, upper = space.feature_bounds()
        if jax.tree.structure(lower) != jax.tree.structure(space.feature_spec()):
            raise ValueError("space feature bounds must match feature_spec().")
        if jax.tree.structure(upper) != jax.tree.structure(space.feature_spec()):
            raise ValueError("space feature bounds must match feature_spec().")
        self.objective = objective
        self.space = space
        self.args = args
        self.convexity = convexity
        self.problem_id = identifier
        self.structure_id = canonical_fingerprint(
            {
                "kind": "integer-hull-problem",
                "problem_id": identifier,
                "objective": objective.problem_id,
                "space": space.structure_id,
                "convexity": {
                    "kind": convexity.kind,
                    "id": convexity.evidence_id,
                },
                "args": array_tree_fingerprint(args),
            }
        )


class IntegerHullPolicy(StrictModule):
    """Tree, linear-oracle, and node-accuracy policy."""

    oracle: AbstractBoundableLinearCombinatorialMethod
    oracle_certification: CombinatorialCertification
    tree: BranchAndBoundPolicy
    maximum_fw_steps: int = eqx.field(static=True)
    maximum_active_atoms: int = eqx.field(static=True)
    fw_tolerance: float = eqx.field(static=True)
    integrality_tolerance: float = eqx.field(static=True)
    maximum_backtracks: int = eqx.field(static=True)
    armijo: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        oracle: AbstractBoundableLinearCombinatorialMethod,
        /,
        *,
        oracle_certification: CombinatorialCertification | None = None,
        tree: BranchAndBoundPolicy | None = None,
        maximum_fw_steps: int = 100,
        maximum_active_atoms: int | None = None,
        fw_tolerance: float = 1e-7,
        integrality_tolerance: float = 1e-7,
        maximum_backtracks: int = 20,
        armijo: float = 1e-4,
    ):
        if not isinstance(oracle, AbstractBoundableLinearCombinatorialMethod):
            raise TypeError(
                "oracle must be an AbstractBoundableLinearCombinatorialMethod."
            )
        capabilities = oracle.capabilities
        if not (
            capabilities.exact
            and capabilities.optimality_certificate
            and capabilities.bound_restrictions
        ):
            raise ValueError(
                "Integer-hull bounds require an exact certified restricted oracle."
            )
        certification = (
            CombinatorialCertification()
            if oracle_certification is None
            else oracle_certification
        )
        tree_ = BranchAndBoundPolicy() if tree is None else tree
        if not isinstance(certification, CombinatorialCertification):
            raise TypeError("oracle_certification has the wrong type.")
        if not isinstance(tree_, BranchAndBoundPolicy):
            raise TypeError("tree has the wrong type.")
        steps = int(maximum_fw_steps)
        active = steps + 1 if maximum_active_atoms is None else int(maximum_active_atoms)
        backtracks = int(maximum_backtracks)
        if steps < 1 or active < steps + 1 or backtracks < 1:
            raise ValueError(
                "maximum_fw_steps/backtracks must be positive and active capacity must hold every generated atom."
            )
        values = tuple(
            float(value) for value in (fw_tolerance, integrality_tolerance, armijo)
        )
        if any(not isfinite(value) or value <= 0.0 for value in values):
            raise ValueError(
                "Integer-hull tolerances and armijo must be positive finite."
            )
        if values[1] >= 0.5 or values[2] >= 1.0:
            raise ValueError("integrality_tolerance and armijo are out of range.")
        self.oracle = oracle
        self.oracle_certification = certification
        self.tree = tree_
        self.maximum_fw_steps = steps
        self.maximum_active_atoms = active
        self.fw_tolerance = values[0]
        self.integrality_tolerance = values[1]
        self.maximum_backtracks = backtracks
        self.armijo = values[2]
        self.policy_id = canonical_fingerprint(
            {
                "kind": "integer-hull-policy",
                "oracle": oracle.method_id,
                "configuration": list(oracle.configuration),
                "oracle_certification": {
                    "absolute": certification.absolute,
                    "relative": certification.relative,
                },
                "tree": {
                    "maximum_nodes": tree_.maximum_nodes,
                    "absolute_gap": tree_.absolute_gap,
                    "relative_gap": tree_.relative_gap,
                },
                "maximum_fw_steps": steps,
                "maximum_active_atoms": active,
                "fw_tolerance": values[0],
                "integrality_tolerance": values[1],
                "maximum_backtracks": backtracks,
                "armijo": values[2],
            }
        )


class IntegerHullCandidate(StrictModule, NonTrainableState):
    decision: Any
    features: PyTree[Array]
    objective: Array
    atom_id: str = eqx.field(static=True)
    restriction_id: str = eqx.field(static=True)


class IntegerHullCertificate(StrictModule, NonTrainableState):
    feasible: Array
    lower_bound_certified: Array
    search_complete: Array
    optimality_certified: Array
    fw_gap: Array
    convexity_kind: ConvexObjectiveEvidenceKind = eqx.field(static=True)
    convexity_evidence_id: str = eqx.field(static=True)


class IntegerHullWork(StrictModule, NonTrainableState):
    explored_nodes: Array
    pruned_nodes: Array
    frontier_size: Array
    oracle_calls: Array
    objective_evaluations: Array
    gradient_evaluations: Array
    fw_steps: Array
    maximum_active_atoms: Array


class IntegerHullResult(StrictModule):
    decision: Any
    features: Any
    relaxation: Any
    objective: Array
    global_lower_bound: Array
    absolute_gap: Array
    relative_gap: Array
    status: Array
    certificate: IntegerHullCertificate
    work: IntegerHullWork
    search: BranchAndBoundResult
    problem_id: str = eqx.field(static=True)
    structure_id: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return (
            (self.status == int(IntegerHullStatus.OPTIMAL))
            & self.certificate.feasible
            & self.certificate.optimality_certified
        )


@dataclass(slots=True)
class _Atom:
    decision: Any
    features: Any
    atom_id: str


@dataclass(slots=True)
class _IntegerHullNode:
    restriction: CombinatorialFeatureRestriction
    atoms: list[_Atom]
    weights: np.ndarray
    path: str


@dataclass(slots=True)
class _IntegerHullNodeState:
    relaxation: Any
    atoms: list[_Atom]
    weights: np.ndarray
    fw_gap: float


def _tree_dot(left, right, /) -> Array:
    products = tuple(
        jnp.sum(jnp.asarray(a) * jnp.asarray(b))
        for a, b in zip(
            jax.tree.leaves(left),
            jax.tree.leaves(right),
            strict=True,
        )
    )
    total = jnp.asarray(0.0)
    for product in products:
        total = total + product
    return total


def _tree_subtract(left, right, /):
    return jax.tree.map(lambda a, b: a - b, left, right)


def _tree_interpolate(current, atom, step: float, /):
    return jax.tree.map(
        lambda x, v: (1.0 - step) * x + step * v,
        current,
        atom,
    )


def _active_combination(atoms: list[_Atom], weights: np.ndarray, /):
    if not atoms or len(atoms) != len(weights):
        raise ValueError("An active set requires one weight per atom.")
    result = jax.tree.map(
        lambda value: jnp.asarray(weights[0], dtype=value.dtype) * value,
        atoms[0].features,
    )
    for atom, weight in zip(atoms[1:], weights[1:], strict=True):
        result = jax.tree.map(
            lambda total, value, weight=weight: (
                total + jnp.asarray(weight, dtype=value.dtype) * value
            ),
            result,
            atom.features,
        )
    return result


def _atom(features, decision, /) -> _Atom:
    identifier = canonical_fingerprint(
        {
            "kind": "integer-hull-atom",
            "features": array_tree_fingerprint(features),
        }
    )
    return _Atom(decision, features, identifier)


def _restriction_contains(
    atom: _Atom,
    restriction: CombinatorialFeatureRestriction,
    /,
) -> bool:
    for value, lower, upper in zip(
        jax.tree.leaves(atom.features),
        jax.tree.leaves(restriction.lower),
        jax.tree.leaves(restriction.upper),
        strict=True,
    ):
        host = np.asarray(value)
        if np.any(host < np.asarray(lower)) or np.any(host > np.asarray(upper)):
            return False
    return True


class _IntegerHullBranchProblem(AbstractBranchAndBoundProblem):
    problem: IntegerHullProblem
    policy: IntegerHullPolicy
    oracle_calls: list[int]
    objective_evaluations: list[int]
    gradient_evaluations: list[int]
    fw_steps: list[int]
    active_maximum: list[int]

    def __init__(self, problem: IntegerHullProblem, policy: IntegerHullPolicy, /):
        self.problem = problem
        self.policy = policy
        self.oracle_calls = [0]
        self.objective_evaluations = [0]
        self.gradient_evaluations = [0]
        self.fw_steps = [0]
        self.active_maximum = [0]
        self.problem_id = problem.structure_id

    def root(self, /) -> _IntegerHullNode:
        return _IntegerHullNode(
            CombinatorialFeatureRestriction.root(self.problem.space),
            [],
            np.empty((0,), dtype=np.float64),
            "root",
        )

    def node_id(self, node: _IntegerHullNode, /) -> str:
        return node.path

    def _oracle(self, costs, restriction, /):
        linear = LinearCombinatorialProblem(
            self.problem.space,
            costs,
            problem_id=f"{self.problem.problem_id}:linear-oracle",
        )
        execution = solve_restricted_combinatorial(
            linear,
            self.policy.oracle,
            restriction,
            certification=self.policy.oracle_certification,
        )
        self.oracle_calls[0] += 1
        return execution

    def _value(self, features, /) -> float:
        value, _ = self.problem.objective.value(features, self.problem.args)
        self.objective_evaluations[0] += 1
        return float(np.asarray(value))

    def _value_and_gradient(self, features, /):
        (value, _), gradient = self.problem.objective.value_and_gradient(
            features,
            self.problem.args,
        )
        self.objective_evaluations[0] += 1
        self.gradient_evaluations[0] += 1
        return float(np.asarray(value)), gradient

    def _candidate(self, atom: _Atom, restriction_id: str, /):
        objective = self._value(atom.features)
        if not np.isfinite(objective):
            return None
        candidate = IntegerHullCandidate(
            atom.decision,
            atom.features,
            jnp.asarray(objective),
            atom.atom_id,
            restriction_id,
        )
        return BranchCandidate(
            candidate,
            objective,
            certificate_id=atom.atom_id,
        )

    def evaluate(self, node: _IntegerHullNode, /) -> BranchNodeEvaluation:
        atoms = list(node.atoms)
        weights = np.asarray(node.weights, dtype=np.float64).copy()
        best_candidate = None
        if not atoms:
            zero_costs = jax.tree.map(
                lambda value: jnp.zeros_like(value),
                node.restriction.lower,
            )
            seed = self._oracle(zero_costs, node.restriction)
            seed_status = CombinatorialStatus(int(np.asarray(seed.result.status)))
            if seed_status == CombinatorialStatus.INFEASIBLE:
                return BranchNodeEvaluation.proven_infeasible(
                    node.restriction.restriction_id,
                    state=seed,
                )
            if not bool(
                np.asarray(seed.valid & seed.result.certificate.optimality_proven)
            ):
                return BranchNodeEvaluation.failed(
                    "integer-hull-seed-oracle",
                    "Restricted seed oracle did not return a certified atom.",
                    state=seed,
                )
            atoms = [_atom(seed.result.features, seed.result.decision)]
            weights = np.asarray([1.0])
        self.active_maximum[0] = max(self.active_maximum[0], len(atoms))
        for atom in atoms:
            candidate = self._candidate(
                atom,
                node.restriction.restriction_id,
            )
            if candidate is not None and (
                best_candidate is None or candidate.objective < best_candidate.objective
            ):
                best_candidate = candidate

        relaxation = _active_combination(atoms, weights)
        final_gap = float("inf")
        final_value = float("inf")
        for iteration in range(self.policy.maximum_fw_steps):
            value, gradient = self._value_and_gradient(relaxation)
            if not np.isfinite(value) or any(
                not np.all(np.isfinite(np.asarray(leaf)))
                for leaf in jax.tree.leaves(gradient)
            ):
                return BranchNodeEvaluation.failed(
                    "nonfinite-integer-hull-objective",
                    "The convex objective or gradient was nonfinite.",
                )
            oracle = self._oracle(gradient, node.restriction)
            if not bool(
                np.asarray(oracle.valid & oracle.result.certificate.optimality_proven)
            ):
                return BranchNodeEvaluation.failed(
                    "uncertified-integer-hull-oracle",
                    "A Frank-Wolfe linear oracle lacked an optimality certificate.",
                    state=oracle,
                )
            new_atom = _atom(oracle.result.features, oracle.result.decision)
            candidate = self._candidate(
                new_atom,
                node.restriction.restriction_id,
            )
            if candidate is not None and (
                best_candidate is None or candidate.objective < best_candidate.objective
            ):
                best_candidate = candidate
            direction = _tree_subtract(relaxation, new_atom.features)
            gap = float(np.asarray(_tree_dot(gradient, direction)))
            gap_tolerance = max(
                self.policy.fw_tolerance,
                float(np.asarray(self.policy.oracle_certification.threshold(value, gap))),
            )
            if not np.isfinite(gap) or gap < -gap_tolerance:
                return BranchNodeEvaluation.failed(
                    "invalid-frank-wolfe-gap",
                    f"The certified linear oracle produced Frank-Wolfe gap {gap}.",
                    state=oracle,
                )
            final_gap = max(gap, 0.0)
            final_value = value
            self.fw_steps[0] += 1
            if final_gap <= self.policy.fw_tolerance:
                break

            step = min(1.0, 2.0 / float(iteration + 2))
            accepted = False
            for _ in range(self.policy.maximum_backtracks):
                trial = _tree_interpolate(
                    relaxation,
                    new_atom.features,
                    step,
                )
                trial_value = self._value(trial)
                if np.isfinite(trial_value) and trial_value <= (
                    value - self.policy.armijo * step * final_gap
                ):
                    accepted = True
                    break
                step *= 0.5
            if not accepted:
                return BranchNodeEvaluation.failed(
                    "integer-hull-line-search",
                    "Frank-Wolfe backtracking found no finite descent step.",
                )
            weights *= 1.0 - step
            atom_index = next(
                (
                    index
                    for index, atom in enumerate(atoms)
                    if atom.atom_id == new_atom.atom_id
                ),
                None,
            )
            if atom_index is None:
                atoms.append(new_atom)
                weights = np.concatenate((weights, np.asarray([step])))
            else:
                weights[atom_index] += step
            relaxation = trial
            self.active_maximum[0] = max(self.active_maximum[0], len(atoms))

        if not np.isfinite(final_value) or not np.isfinite(final_gap):
            return BranchNodeEvaluation.failed(
                "missing-frank-wolfe-bound",
                "Integer-hull node did not produce a finite bound.",
            )
        lower_bound = final_value - final_gap
        flat, _ = ravel_pytree(relaxation)
        mask, _ = ravel_pytree(self.problem.space.integral_feature_mask())
        integral = np.asarray(mask, dtype=np.bool_)
        flat_host = np.asarray(flat)
        fractionality = np.zeros_like(flat_host)
        fractionality[integral] = np.abs(
            flat_host[integral] - np.rint(flat_host[integral])
        )
        terminal = bool(
            np.max(fractionality, initial=0.0) <= self.policy.integrality_tolerance
        )
        if terminal:
            lower_flat, lower_unravel = ravel_pytree(node.restriction.lower)
            upper_flat, upper_unravel = ravel_pytree(node.restriction.upper)
            fixed = jnp.rint(flat)
            fixed_lower = jnp.where(mask.astype("bool"), fixed, lower_flat)
            fixed_upper = jnp.where(mask.astype("bool"), fixed, upper_flat)
            fixed_restriction = CombinatorialFeatureRestriction(
                self.problem.space,
                lower=lower_unravel(fixed_lower),
                upper=upper_unravel(fixed_upper),
            )
            zero_costs = jax.tree.map(
                lambda value: jnp.zeros_like(value),
                fixed_restriction.lower,
            )
            reconstructed = self._oracle(zero_costs, fixed_restriction)
            if not bool(
                np.asarray(
                    reconstructed.valid
                    & reconstructed.result.certificate.optimality_proven
                )
            ):
                return BranchNodeEvaluation.failed(
                    "integer-hull-reconstruction",
                    "An integral hull point could not be reconstructed by its oracle.",
                    state=reconstructed,
                )
            reconstructed_atom = _atom(
                reconstructed.result.features,
                reconstructed.result.decision,
            )
            candidate = self._candidate(
                reconstructed_atom,
                fixed_restriction.restriction_id,
            )
            if candidate is None:
                return BranchNodeEvaluation.failed(
                    "integer-hull-candidate-objective",
                    "Reconstructed integer candidate had a nonfinite objective.",
                )
            if best_candidate is None or candidate.objective < best_candidate.objective:
                best_candidate = candidate

        state = _IntegerHullNodeState(
            relaxation,
            atoms,
            weights,
            final_gap,
        )
        return BranchNodeEvaluation(
            lower_bound=BranchBoundEvidence(
                lower_bound,
                certified=True,
                certificate_id=(
                    f"{self.problem.structure_id}:{node.path}:frank-wolfe-gap"
                ),
            ),
            candidate=best_candidate,
            terminal=terminal,
            state=state,
        )

    def branch(
        self,
        node: _IntegerHullNode,
        evaluation: BranchNodeEvaluation,
        /,
    ) -> tuple[_IntegerHullNode, _IntegerHullNode]:
        state = evaluation.state
        if not isinstance(state, _IntegerHullNodeState):
            raise TypeError("Integer-hull branching requires a solved node state.")
        flat, _ = ravel_pytree(state.relaxation)
        lower, lower_unravel = ravel_pytree(node.restriction.lower)
        upper, upper_unravel = ravel_pytree(node.restriction.upper)
        mask, _ = ravel_pytree(self.problem.space.integral_feature_mask())
        values = np.asarray(flat)
        integral = np.asarray(mask, dtype=np.bool_)
        fractionality = np.where(
            integral,
            np.abs(values - np.rint(values)),
            -1.0,
        )
        coordinate = int(np.argmax(fractionality))
        value = float(values[coordinate])
        floor = np.floor(value)
        ceil = np.ceil(value)
        left_upper = upper.at[coordinate].set(
            jnp.minimum(
                upper[coordinate],
                jnp.asarray(floor, dtype=upper.dtype),
            )
        )
        right_lower = lower.at[coordinate].set(
            jnp.maximum(
                lower[coordinate],
                jnp.asarray(ceil, dtype=lower.dtype),
            )
        )
        left_restriction = CombinatorialFeatureRestriction(
            self.problem.space,
            lower=lower_unravel(lower),
            upper=upper_unravel(left_upper),
        )
        right_restriction = CombinatorialFeatureRestriction(
            self.problem.space,
            lower=lower_unravel(right_lower),
            upper=upper_unravel(upper),
        )

        def child(restriction, suffix):
            selected = [
                (atom, weight)
                for atom, weight in zip(
                    state.atoms,
                    state.weights,
                    strict=True,
                )
                if weight > 0.0 and _restriction_contains(atom, restriction)
            ]
            if selected:
                atoms = [atom for atom, _ in selected]
                weights = np.asarray([weight for _, weight in selected])
                weights /= np.sum(weights)
            else:
                atoms = []
                weights = np.empty((0,), dtype=np.float64)
            return _IntegerHullNode(
                restriction,
                atoms,
                weights,
                f"{node.path}/f{coordinate}{suffix}",
            )

        return child(left_restriction, f"<={floor:g}"), child(
            right_restriction,
            f">={ceil:g}",
        )


def solve_integer_hull(
    problem: IntegerHullProblem,
    policy: IntegerHullPolicy,
    /,
) -> IntegerHullResult:
    """Solve one convex objective over a certified integer hull."""
    if not isinstance(problem, IntegerHullProblem):
        raise TypeError("problem must be an IntegerHullProblem.")
    if not isinstance(policy, IntegerHullPolicy):
        raise TypeError("policy must be an IntegerHullPolicy.")
    branch_problem = _IntegerHullBranchProblem(problem, policy)
    search = branch_and_bound(branch_problem, policy=policy.tree)
    candidate = search.incumbent
    feasible = candidate is not None
    branch_status = BranchAndBoundStatus(int(np.asarray(search.status)))
    status = {
        BranchAndBoundStatus.OPTIMAL: IntegerHullStatus.OPTIMAL,
        BranchAndBoundStatus.GAP_REACHED: IntegerHullStatus.GAP_REACHED,
        BranchAndBoundStatus.WORK_LIMIT: IntegerHullStatus.WORK_LIMIT,
        BranchAndBoundStatus.INFEASIBLE: IntegerHullStatus.INFEASIBLE,
        BranchAndBoundStatus.UNBOUNDED: IntegerHullStatus.EVALUATION_FAILURE,
        BranchAndBoundStatus.EVALUATION_FAILURE: IntegerHullStatus.EVALUATION_FAILURE,
    }[branch_status]
    objective = search.objective
    optimality = (
        status == IntegerHullStatus.OPTIMAL
        and feasible
        and bool(np.asarray(search.global_lower_bound_certified))
        and bool(np.asarray(search.search_complete))
    )
    certificate = IntegerHullCertificate(
        jnp.asarray(feasible),
        search.global_lower_bound_certified,
        search.search_complete,
        jnp.asarray(optimality),
        jnp.asarray(
            jnp.nan
            if candidate is None
            else candidate.objective - search.global_lower_bound
        ),
        problem.convexity.kind,
        problem.convexity.evidence_id,
    )
    work = IntegerHullWork(
        search.explored_nodes,
        search.pruned_nodes,
        search.frontier_size,
        jnp.asarray(branch_problem.oracle_calls[0], dtype=jnp.int32),
        jnp.asarray(branch_problem.objective_evaluations[0], dtype=jnp.int32),
        jnp.asarray(branch_problem.gradient_evaluations[0], dtype=jnp.int32),
        jnp.asarray(branch_problem.fw_steps[0], dtype=jnp.int32),
        jnp.asarray(branch_problem.active_maximum[0], dtype=jnp.int32),
    )
    result = IntegerHullResult(
        None if candidate is None else candidate.decision,
        None if candidate is None else candidate.features,
        None,
        objective,
        search.global_lower_bound,
        search.absolute_gap,
        search.relative_gap,
        jnp.asarray(int(status), dtype=jnp.int32),
        certificate,
        work,
        search,
        problem.problem_id,
        problem.structure_id,
        policy.policy_id,
    )
    return jax.tree.map(jax.lax.stop_gradient, result)


__all__ = [
    "ConvexObjectiveEvidence",
    "ConvexObjectiveEvidenceKind",
    "IntegerHullCandidate",
    "IntegerHullCertificate",
    "IntegerHullPolicy",
    "IntegerHullProblem",
    "IntegerHullResult",
    "IntegerHullStatus",
    "IntegerHullWork",
    "solve_integer_hull",
]
