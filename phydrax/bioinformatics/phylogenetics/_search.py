#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax._strict import StrictModule

from ..foundation import (
    BioinformaticsMethodContract,
    DifferentiationKind,
    ExecutionKind,
    MethodKind,
    OutputKind,
)
from ._pruning import (
    felsenstein_pruning,
    FelsensteinPruningResult,
    LikelihoodPartition,
)
from ._tree import tree_topology, TreeTopology


class NNISearchStatus(IntEnum):
    SUCCESS = 0
    INVALID_INPUT = 1
    CANDIDATE_CAPACITY_EXCEEDED = 2
    ITERATION_BOUND_REACHED = 3
    NO_VALID_CANDIDATE = 4


class NNISearchPlan(StrictModule):
    """Compile-time bounds for deterministic steepest-ascent rooted NNI."""

    max_iterations: int = eqx.field(static=True)
    candidate_capacity: int = eqx.field(static=True)
    improvement_tolerance: float = eqx.field(static=True)

    def __init__(
        self,
        max_iterations: int,
        candidate_capacity: int,
        /,
        *,
        improvement_tolerance: float = 0.0,
    ):
        iterations = int(max_iterations)
        candidates = int(candidate_capacity)
        tolerance = float(improvement_tolerance)
        if iterations < 0 or candidates < 1:
            raise ValueError(
                "NNI bounds require max_iterations >= 0 and candidate_capacity >= 1."
            )
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("improvement_tolerance must be finite and nonnegative.")
        self.max_iterations = iterations
        self.candidate_capacity = candidates
        self.improvement_tolerance = tolerance


class NNISearchEvidence(StrictModule):
    """Observable bounds and termination evidence for heuristic NNI search."""

    required_candidate_capacity: Array
    candidate_capacity: Array
    evaluated_candidates: Array
    accepted_moves: Array
    completed_iterations: Array
    converged_local_neighborhood: Array
    iteration_bound_reached: Array
    all_evaluations_valid: Array


class NNISearchResult(StrictModule):
    """Nondifferentiable bounded NNI heuristic result."""

    topology: TreeTopology
    branch_lengths: Array
    log_likelihood: Array
    likelihood: FelsensteinPruningResult
    valid: Array
    status: Array
    evidence: NNISearchEvidence
    method_contract: BioinformaticsMethodContract = eqx.field(static=True)
    claim_kind: str = eqx.field(static=True)


def _nni_contract(plan: NNISearchPlan, /) -> BioinformaticsMethodContract:
    return BioinformaticsMethodContract(
        "bounded_steepest_ascent_nni_topology_search",
        MethodKind.HEURISTIC,
        ExecutionKind.EXACT_DISCRETE,
        DifferentiationKind.NONE,
        OutputKind.GRAPH,
        conditioning_statement=(
            "Each candidate score is the fixed-branch Felsenstein likelihood under "
            "the supplied partitions; topology and candidate selection are discrete."
        ),
        truncation_statement=(
            f"Search stops after at most {plan.max_iterations} accepted/checked NNI "
            "iterations and does not claim global or local optimality when that bound is reached."
        ),
        capacity_semantics=(
            f"Every neighborhood is preflighted against candidate_capacity={plan.candidate_capacity}; "
            "an oversized neighborhood fails without evaluating a prefix."
        ),
        assumptions=(
            "Rooted NNI moves retain node identities and their attached branch lengths.",
        ),
        nondifferentiable_outputs=(
            "topology",
            "branch_lengths",
            "log_likelihood",
            "valid",
            "status",
            "evidence",
        ),
    )


def _nni_candidates(parent: np.ndarray, /) -> list[np.ndarray]:
    children: list[list[int]] = [[] for _ in range(parent.size)]
    root = int(np.flatnonzero(parent == -1)[0])
    for node, ancestor in enumerate(parent.tolist()):
        if node != root:
            children[ancestor].append(node)
    candidates: list[np.ndarray] = []
    for child in range(parent.size):
        ancestor = int(parent[child])
        if ancestor < 0 or len(children[child]) != 2 or len(children[ancestor]) != 2:
            continue
        sibling = children[ancestor][0]
        if sibling == child:
            sibling = children[ancestor][1]
        for grandchild in children[child]:
            candidate = parent.copy()
            candidate[sibling] = child
            candidate[grandchild] = ancestor
            candidates.append(candidate)
    return candidates


def nni_topology_search(
    topology: TreeTopology,
    tip_partials: ArrayLike,
    branch_lengths: ArrayLike,
    partitions: tuple[LikelihoodPartition, ...],
    plan: NNISearchPlan,
    /,
    *,
    pattern_weights: ArrayLike | None = None,
    method_contract: BioinformaticsMethodContract | None = None,
) -> NNISearchResult:
    """Run bounded deterministic steepest-ascent rooted NNI.

    This routine is intentionally eager and nondifferentiable. Candidate
    neighborhoods are complete or rejected before scoring; no capacity-limited
    prefix is ever mistaken for a complete neighborhood.
    """

    if not isinstance(topology, TreeTopology):
        raise TypeError("topology must be a TreeTopology.")
    if not isinstance(plan, NNISearchPlan):
        raise TypeError("plan must be an NNISearchPlan.")
    lengths = jnp.asarray(branch_lengths)
    values = jnp.asarray(tip_partials)
    current_likelihood = felsenstein_pruning(
        topology,
        values,
        lengths,
        partitions,
        pattern_weights=pattern_weights,
    )
    contract = _nni_contract(plan) if method_contract is None else method_contract
    parent = np.asarray(jax.device_get(topology.parent_indices), dtype=np.int32)
    input_valid = bool(np.asarray(jax.device_get(current_likelihood.valid)))
    evaluated = 0
    accepted = 0
    completed = 0
    maximum_required = 0
    all_evaluations_valid = input_valid
    converged = False
    iteration_bound_reached = False
    capacity_failure = False
    any_valid_candidate = False

    current_topology = topology
    current_score = float(np.asarray(jax.device_get(current_likelihood.log_likelihood)))
    if input_valid:
        for iteration in range(plan.max_iterations):
            candidates = _nni_candidates(parent)
            required = len(candidates)
            maximum_required = max(maximum_required, required)
            if required > plan.candidate_capacity:
                capacity_failure = True
                break
            completed = iteration + 1
            if not candidates:
                converged = True
                break
            best_score = current_score
            best_parent: np.ndarray | None = None
            best_topology: TreeTopology | None = None
            best_likelihood: FelsensteinPruningResult | None = None
            local_valid_candidate = False
            for candidate_parent in candidates:
                candidate_topology = tree_topology(
                    candidate_parent,
                    child_capacity=topology.child_capacity,
                    tip_indices=topology.tip_indices,
                )
                candidate_likelihood = felsenstein_pruning(
                    candidate_topology,
                    values,
                    lengths,
                    partitions,
                    pattern_weights=pattern_weights,
                )
                evaluated += 1
                candidate_valid = bool(
                    np.asarray(jax.device_get(candidate_likelihood.valid))
                )
                all_evaluations_valid = all_evaluations_valid and candidate_valid
                if not candidate_valid:
                    continue
                any_valid_candidate = True
                local_valid_candidate = True
                score = float(
                    np.asarray(jax.device_get(candidate_likelihood.log_likelihood))
                )
                if score > best_score + plan.improvement_tolerance:
                    best_score = score
                    best_parent = candidate_parent
                    best_topology = candidate_topology
                    best_likelihood = candidate_likelihood
            if not local_valid_candidate:
                break
            if best_parent is None or best_topology is None or best_likelihood is None:
                converged = True
                break
            parent = best_parent
            current_topology = best_topology
            current_likelihood = best_likelihood
            current_score = best_score
            accepted += 1
        else:
            iteration_bound_reached = not converged

    if not input_valid:
        status = NNISearchStatus.INVALID_INPUT
        valid = False
    elif capacity_failure:
        status = NNISearchStatus.CANDIDATE_CAPACITY_EXCEEDED
        valid = False
    elif evaluated > 0 and not any_valid_candidate:
        status = NNISearchStatus.NO_VALID_CANDIDATE
        valid = False
    elif iteration_bound_reached:
        status = NNISearchStatus.ITERATION_BOUND_REACHED
        valid = True
    else:
        status = NNISearchStatus.SUCCESS
        valid = True

    evidence = NNISearchEvidence(
        required_candidate_capacity=jnp.asarray(maximum_required, dtype=jnp.int32),
        candidate_capacity=jnp.asarray(plan.candidate_capacity, dtype=jnp.int32),
        evaluated_candidates=jnp.asarray(evaluated, dtype=jnp.int32),
        accepted_moves=jnp.asarray(accepted, dtype=jnp.int32),
        completed_iterations=jnp.asarray(completed, dtype=jnp.int32),
        converged_local_neighborhood=jnp.asarray(converged),
        iteration_bound_reached=jnp.asarray(iteration_bound_reached),
        all_evaluations_valid=jnp.asarray(all_evaluations_valid),
    )
    return NNISearchResult(
        topology=current_topology,
        branch_lengths=jax.lax.stop_gradient(lengths),
        log_likelihood=jax.lax.stop_gradient(current_likelihood.log_likelihood),
        likelihood=current_likelihood,
        valid=jnp.asarray(valid),
        status=jnp.asarray(int(status), dtype=jnp.int32),
        evidence=evidence,
        method_contract=contract,
        claim_kind="bounded_nni_heuristic",
    )


bounded_nni_search = nni_topology_search


__all__ = [
    "NNISearchEvidence",
    "NNISearchPlan",
    "NNISearchResult",
    "NNISearchStatus",
    "bounded_nni_search",
    "nni_topology_search",
]
