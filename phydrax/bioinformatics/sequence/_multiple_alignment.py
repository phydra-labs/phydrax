#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from numbers import Integral

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import core as jax_core
from jaxtyping import Array, ArrayLike

from phydrax._fingerprint import array_tree_fingerprint, canonical_fingerprint
from phydrax._strict import StrictModule

from ..foundation import (
    BioinformaticsMethodContract,
    DifferentiationKind,
    ExecutionKind,
    MethodKind,
    OutputKind,
)
from ._alphabet import AlphabetPlan
from ._batch import SequenceBatch
from ._motifs import _observation_support


MSA_STATUS_VALID = 0
MSA_STATUS_CAPACITY_EXCEEDED = 1
MSA_STATUS_INVALID_GUIDE_TREE = 2


def _msa_contract(
    guide_tree_method: str, profile_method: str
) -> BioinformaticsMethodContract:
    return BioinformaticsMethodContract(
        "progressive multiple-sequence alignment",
        MethodKind.HEURISTIC,
        ExecutionKind.EXACT_DISCRETE,
        DifferentiationKind.NONE,
        OutputKind.SEQUENCE,
        conditioning_statement=(
            f"Deterministic progressive alignment conditioned on the {guide_tree_method} "
            f"guide tree and {profile_method} profile-column score."
        ),
        truncation_statement="No profile column is pruned after capacity preflight.",
        capacity_semantics=(
            "Record count, input lengths, and the worst-case sum of active lengths "
            "must fit the declared bounds."
        ),
        assumptions=(
            "The guide tree and progressive merge order are heuristic choices.",
            "Pairwise profile merges optimize their local affine-gap objective exactly.",
        ),
        nondifferentiable_outputs=("alignment", "guide_tree", "status"),
    )


class GuideTree(StrictModule):
    """A rooted binary guide tree in chronological merge encoding."""

    merges: Array
    heights: Array
    leaf_count: int = eqx.field(static=True)
    method: str = eqx.field(static=True)
    tie_policy: str = eqx.field(static=True)
    fingerprint: str = eqx.field(static=True)

    def __init__(
        self,
        merges: ArrayLike,
        heights: ArrayLike,
        leaf_count: int,
        *,
        method: str = "supplied",
        tie_policy: str = "supplied-order",
    ):
        if isinstance(leaf_count, bool) or not isinstance(leaf_count, Integral):
            raise TypeError("leaf_count must be an integer.")
        leaves = int(leaf_count)
        if leaves <= 0:
            raise ValueError("leaf_count must be positive.")
        routes = jnp.asarray(merges)
        levels = jnp.asarray(heights)
        expected_merges = max(leaves - 1, 0)
        if routes.shape != (expected_merges, 2) or not jnp.issubdtype(
            routes.dtype, jnp.integer
        ):
            raise ValueError("merges must have shape (leaf_count - 1, 2).")
        if levels.shape != (expected_merges,) or not jnp.issubdtype(
            levels.dtype, jnp.floating
        ):
            raise ValueError("heights must have shape (leaf_count - 1,).")
        concrete_routes = (
            None if isinstance(routes, jax_core.Tracer) else np.asarray(routes)
        )
        concrete_levels = (
            None if isinstance(levels, jax_core.Tracer) else np.asarray(levels)
        )
        if concrete_routes is None or concrete_levels is None:
            raise TypeError("Guide-tree topology must be concrete at preparation.")
        active = set(range(leaves))
        for index, children in enumerate(concrete_routes):
            left, right = int(children[0]), int(children[1])
            if left == right or left not in active or right not in active:
                raise ValueError(
                    "Every guide-tree merge must combine two currently active clusters."
                )
            active.remove(left)
            active.remove(right)
            active.add(leaves + index)
        if len(active) != 1:
            raise ValueError("Guide-tree merges must form one rooted binary tree.")
        if np.any(~np.isfinite(concrete_levels)) or np.any(concrete_levels < 0.0):
            raise ValueError("Guide-tree heights must be finite and non-negative.")
        identifier = str(method).strip()
        policy = str(tie_policy).strip()
        if not identifier or not policy:
            raise ValueError("Guide-tree method and tie policy must be non-empty.")
        self.merges = routes.astype(jnp.int32)
        self.heights = levels
        self.leaf_count = leaves
        self.method = identifier
        self.tie_policy = policy
        self.fingerprint = canonical_fingerprint(
            {
                "kind": "guide-tree",
                "method": identifier,
                "tie_policy": policy,
                "leaf_count": leaves,
                "merges": array_tree_fingerprint(routes),
                "heights": array_tree_fingerprint(levels),
            }
        )


class ProgressiveMSAPlan(StrictModule):
    """Bounds and explicit guide-tree/profile heuristics for progressive MSA."""

    maximum_sequences: int = eqx.field(static=True)
    maximum_sequence_length: int = eqx.field(static=True)
    maximum_alignment_length: int = eqx.field(static=True)
    guide_tree_method: str = eqx.field(static=True)
    profile_alignment_method: str = eqx.field(static=True)
    match_score: float = eqx.field(static=True)
    mismatch_score: float = eqx.field(static=True)
    gap_open_score: float = eqx.field(static=True)
    gap_extend_score: float = eqx.field(static=True)
    tie_policy: str = eqx.field(static=True)
    globally_exact: bool = eqx.field(static=True)
    method_contract: BioinformaticsMethodContract = eqx.field(static=True)

    def __init__(
        self,
        maximum_sequences: int,
        maximum_sequence_length: int,
        maximum_alignment_length: int,
        *,
        guide_tree_method: str = "upgma-expected-identity",
        profile_alignment_method: str = "expected-sum-of-pairs",
        match_score: float = 2.0,
        mismatch_score: float = -1.0,
        gap_open_score: float = -2.0,
        gap_extend_score: float = -0.5,
    ):
        bounds = (
            maximum_sequences,
            maximum_sequence_length,
            maximum_alignment_length,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, Integral) for value in bounds
        ):
            raise TypeError("Progressive-MSA capacity bounds must be integers.")
        if any(int(value) <= 0 for value in bounds):
            raise ValueError("Progressive-MSA capacity bounds must be positive.")
        guide_method = str(guide_tree_method).strip()
        profile_method = str(profile_alignment_method).strip()
        if guide_method not in ("upgma-expected-identity", "supplied"):
            raise ValueError(
                "guide_tree_method must be 'upgma-expected-identity' or 'supplied'."
            )
        if profile_method != "expected-sum-of-pairs":
            raise ValueError(
                "Only the explicit expected-sum-of-pairs profile heuristic is supported."
            )
        scores = (
            float(match_score),
            float(mismatch_score),
            float(gap_open_score),
            float(gap_extend_score),
        )
        if any(not np.isfinite(value) for value in scores):
            raise ValueError("Progressive-MSA scores must be finite.")
        self.maximum_sequences = int(maximum_sequences)
        self.maximum_sequence_length = int(maximum_sequence_length)
        self.maximum_alignment_length = int(maximum_alignment_length)
        self.guide_tree_method = guide_method
        self.profile_alignment_method = profile_method
        self.match_score, self.mismatch_score = scores[:2]
        self.gap_open_score, self.gap_extend_score = scores[2:]
        self.tie_policy = "canonical-cluster-signature-then-match-insert-delete"
        self.globally_exact = False
        self.method_contract = _msa_contract(guide_method, profile_method)


class ProgressiveMSAEvidence(StrictModule):
    """Observable bounds, merge completion, and heuristic classification."""

    capacity_sufficient: Array
    guide_tree_complete: Array
    completed_profile_merges: Array
    expected_profile_merges: Array
    permutation_tie_count: Array
    sum_of_pairs_score: Array
    guide_tree_heuristic: Array
    profile_alignment_heuristic: Array
    globally_exact: Array


class ProgressiveMSAResult(StrictModule):
    alignment: SequenceBatch
    guide_tree: GuideTree | None
    column_mask: Array
    alignment_length: Array
    valid: Array
    status: Array
    evidence: ProgressiveMSAEvidence
    method_contract: BioinformaticsMethodContract = eqx.field(static=True)

    @property
    def globally_exact(self) -> Array:
        """The progressive objective is never a global-alignment exactness claim."""
        return self.evidence.globally_exact

    @property
    def guide_tree_heuristic(self) -> Array:
        return self.evidence.guide_tree_heuristic

    @property
    def profile_alignment_heuristic(self) -> Array:
        return self.evidence.profile_alignment_heuristic


def _concrete_batch(sequences: SequenceBatch, /) -> tuple[np.ndarray, ...]:
    arrays = (
        sequences.record_ids,
        sequences.token_codes,
        sequences.valid_mask,
        sequences.case_mask,
        sequences.soft_mask,
    )
    if any(isinstance(value, jax_core.Tracer) for value in arrays):
        raise TypeError(
            "Progressive guide-tree construction requires concrete encoded sequences."
        )
    return tuple(np.asarray(value) for value in arrays)


def _token_distributions(alphabet: AlphabetPlan, /) -> tuple[np.ndarray, np.ndarray]:
    support, scorable = _observation_support(alphabet)
    values = np.asarray(support, dtype=np.float64)
    totals = values.sum(axis=1, keepdims=True)
    distributions = np.divide(
        values, totals, out=np.zeros_like(values), where=totals > 0.0
    )
    return distributions, np.asarray(scorable, dtype=bool)


def _column_score(
    left: np.ndarray,
    right: np.ndarray,
    gap_code: int,
    distributions: np.ndarray,
    match_score: float,
    mismatch_score: float,
) -> float:
    left_residues = left[left != gap_code]
    right_residues = right[right != gap_code]
    if left_residues.size == 0 or right_residues.size == 0:
        return -np.inf
    compatibility = (
        distributions[left_residues] @ distributions[right_residues].T
    ).mean()
    return float(mismatch_score + (match_score - mismatch_score) * compatibility)


def _merge_profiles(
    left_tokens: np.ndarray,
    left_soft: np.ndarray,
    right_tokens: np.ndarray,
    right_soft: np.ndarray,
    gap_code: int,
    distributions: np.ndarray,
    plan: ProgressiveMSAPlan,
) -> tuple[np.ndarray, np.ndarray, float]:
    left_width = left_tokens.shape[1]
    right_width = right_tokens.shape[1]
    values = np.full((left_width + 1, right_width + 1, 3), -np.inf, dtype=np.float64)
    predecessor = np.full((left_width + 1, right_width + 1, 3), -1, dtype=np.int8)
    values[0, 0, 0] = 0.0
    for left_index in range(1, left_width + 1):
        candidates = values[left_index - 1, 0] + np.where(
            np.arange(3) == 1, plan.gap_extend_score, plan.gap_open_score
        )
        source = int(np.argmax(candidates))
        values[left_index, 0, 1] = candidates[source]
        predecessor[left_index, 0, 1] = source
    for right_index in range(1, right_width + 1):
        candidates = values[0, right_index - 1] + np.where(
            np.arange(3) == 2, plan.gap_extend_score, plan.gap_open_score
        )
        source = int(np.argmax(candidates))
        values[0, right_index, 2] = candidates[source]
        predecessor[0, right_index, 2] = source
    for left_index in range(1, left_width + 1):
        for right_index in range(1, right_width + 1):
            diagonal = values[left_index - 1, right_index - 1]
            source = int(np.argmax(diagonal))
            values[left_index, right_index, 0] = diagonal[source] + _column_score(
                left_tokens[:, left_index - 1],
                right_tokens[:, right_index - 1],
                gap_code,
                distributions,
                plan.match_score,
                plan.mismatch_score,
            )
            predecessor[left_index, right_index, 0] = source
            insertion = values[left_index - 1, right_index] + np.where(
                np.arange(3) == 1, plan.gap_extend_score, plan.gap_open_score
            )
            source = int(np.argmax(insertion))
            values[left_index, right_index, 1] = insertion[source]
            predecessor[left_index, right_index, 1] = source
            deletion = values[left_index, right_index - 1] + np.where(
                np.arange(3) == 2, plan.gap_extend_score, plan.gap_open_score
            )
            source = int(np.argmax(deletion))
            values[left_index, right_index, 2] = deletion[source]
            predecessor[left_index, right_index, 2] = source

    state = int(np.argmax(values[left_width, right_width]))
    score = float(values[left_width, right_width, state])
    left_index = left_width
    right_index = right_width
    operations: list[int] = []
    while left_index > 0 or right_index > 0:
        operations.append(state)
        previous = int(predecessor[left_index, right_index, state])
        if state == 0:
            left_index -= 1
            right_index -= 1
        elif state == 1:
            left_index -= 1
        else:
            right_index -= 1
        state = previous
    operations.reverse()

    merged_width = len(operations)
    merged_tokens = np.full(
        (left_tokens.shape[0] + right_tokens.shape[0], merged_width),
        gap_code,
        dtype=np.int32,
    )
    merged_soft = np.zeros(merged_tokens.shape, dtype=bool)
    left_index = 0
    right_index = 0
    for column, operation in enumerate(operations):
        if operation in (0, 1):
            merged_tokens[: left_tokens.shape[0], column] = left_tokens[:, left_index]
            merged_soft[: left_tokens.shape[0], column] = left_soft[:, left_index]
            left_index += 1
        if operation in (0, 2):
            merged_tokens[left_tokens.shape[0] :, column] = right_tokens[:, right_index]
            merged_soft[left_tokens.shape[0] :, column] = right_soft[:, right_index]
            right_index += 1
    informative = np.any(merged_tokens != gap_code, axis=0)
    return merged_tokens[:, informative], merged_soft[:, informative], score


def _single_distance(
    left: np.ndarray,
    right: np.ndarray,
    gap_code: int,
    distributions: np.ndarray,
    plan: ProgressiveMSAPlan,
) -> float:
    left_profile = left.reshape((1, -1))
    right_profile = right.reshape((1, -1))
    merged, _, _ = _merge_profiles(
        left_profile,
        np.zeros_like(left_profile, dtype=bool),
        right_profile,
        np.zeros_like(right_profile, dtype=bool),
        gap_code,
        distributions,
        plan,
    )
    comparable = (merged[0] != gap_code) & (merged[1] != gap_code)
    compatibility = np.zeros((merged.shape[1],), dtype=np.float64)
    compatibility[comparable] = np.sum(
        distributions[merged[0, comparable]] * distributions[merged[1, comparable]],
        axis=1,
    )
    return float(1.0 - compatibility.mean()) if compatibility.size else 1.0


def _ordered_pair(left: int, right: int, /) -> tuple[int, int]:
    return (left, right) if left < right else (right, left)


def _upgma_tree(
    sequences: list[np.ndarray],
    signatures: list[tuple[int, ...]],
    gap_code: int,
    distributions: np.ndarray,
    plan: ProgressiveMSAPlan,
) -> tuple[GuideTree, int]:
    leaves = len(sequences)
    if leaves == 1:
        return GuideTree(
            np.empty((0, 2), dtype=np.int32),
            np.empty((0,), dtype=np.float64),
            1,
            method="upgma-expected-identity",
            tie_policy=plan.tie_policy,
        ), 0
    base_distance: dict[tuple[int, int], float] = {}
    for left in range(leaves):
        for right in range(left + 1, leaves):
            base_distance[(left, right)] = _single_distance(
                sequences[left], sequences[right], gap_code, distributions, plan
            )
    members: dict[int, tuple[int, ...]] = {index: (index,) for index in range(leaves)}
    cluster_signature: dict[int, tuple[tuple[int, ...], ...]] = {
        index: (signatures[index],) for index in range(leaves)
    }
    active = set(range(leaves))
    merges: list[tuple[int, int]] = []
    heights: list[float] = []
    tie_count = 0
    for merge_index in range(leaves - 1):
        candidates: list[
            tuple[
                float, tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...], int, int
            ]
        ] = []
        ordered = sorted(active)
        for offset, first in enumerate(ordered):
            for second in ordered[offset + 1 :]:
                distance = np.mean(
                    [
                        base_distance[_ordered_pair(left, right)]
                        for left in members[first]
                        for right in members[second]
                    ]
                )
                left_node = first
                right_node = second
                first_signature = cluster_signature[left_node]
                second_signature = cluster_signature[right_node]
                if second_signature < first_signature:
                    left_node, right_node = right_node, left_node
                    first_signature, second_signature = second_signature, first_signature
                candidates.append(
                    (
                        float(distance),
                        first_signature,
                        second_signature,
                        left_node,
                        right_node,
                    )
                )
        candidates.sort()
        minimum = candidates[0][0]
        tied = sum(np.isclose(candidate[0], minimum) for candidate in candidates)
        tie_count += int(max(tied - 1, 0))
        distance, _, _, left, right = candidates[0]
        merges.append((left, right))
        heights.append(0.5 * distance)
        active.remove(left)
        active.remove(right)
        new_node = leaves + merge_index
        active.add(new_node)
        members[new_node] = tuple(sorted(members[left] + members[right]))
        cluster_signature[new_node] = tuple(
            sorted(cluster_signature[left] + cluster_signature[right])
        )
    return (
        GuideTree(
            np.asarray(merges, dtype=np.int32),
            np.asarray(heights, dtype=np.float64),
            leaves,
            method="upgma-expected-identity",
            tie_policy=plan.tie_policy,
        ),
        int(tie_count),
    )


def _empty_alignment(
    sequences: SequenceBatch,
    plan: ProgressiveMSAPlan,
    /,
) -> SequenceBatch:
    pad = sequences.alphabet.code(sequences.alphabet.pad_symbol)
    shape = (sequences.record_capacity, plan.maximum_alignment_length)
    return SequenceBatch(
        sequences.record_ids,
        jnp.full(shape, pad, dtype=jnp.int32),
        jnp.zeros(shape, dtype=bool),
        sequences.case_mask,
        jnp.zeros(shape, dtype=bool),
        sequences.alphabet,
    )


def _failure(
    sequences: SequenceBatch,
    plan: ProgressiveMSAPlan,
    status: int,
    guide_tree: GuideTree | None = None,
    /,
) -> ProgressiveMSAResult:
    evidence = ProgressiveMSAEvidence(
        jnp.asarray(False),
        jnp.asarray(guide_tree is not None),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.maximum(jnp.sum(sequences.case_mask, dtype=jnp.int32) - 1, 0),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(-jnp.inf),
        jnp.asarray(plan.guide_tree_method != "supplied"),
        jnp.asarray(True),
        jnp.asarray(False),
    )
    return ProgressiveMSAResult(
        _empty_alignment(sequences, plan),
        guide_tree,
        jnp.zeros((plan.maximum_alignment_length,), dtype=bool),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(False),
        jnp.asarray(status, dtype=jnp.int32),
        evidence,
        plan.method_contract,
    )


def progressive_multiple_alignment(
    sequences: SequenceBatch,
    plan: ProgressiveMSAPlan,
    guide_tree: GuideTree | None = None,
    /,
) -> ProgressiveMSAResult:
    """Build a deterministic progressive MSA and explicitly retain heuristic status."""
    if not isinstance(sequences, SequenceBatch):
        raise TypeError("sequences must be a SequenceBatch.")
    if not isinstance(plan, ProgressiveMSAPlan):
        raise TypeError("plan must be a ProgressiveMSAPlan.")
    if guide_tree is not None and not isinstance(guide_tree, GuideTree):
        raise TypeError("guide_tree must be a GuideTree or None.")
    ids, tokens, valid, cases, soft = _concrete_batch(sequences)
    active_indices = np.nonzero(cases)[0]
    record_count = int(active_indices.size)
    if record_count <= 0:
        raise ValueError("Progressive MSA requires at least one populated record.")
    if (plan.guide_tree_method == "supplied") != (guide_tree is not None):
        return _failure(sequences, plan, MSA_STATUS_INVALID_GUIDE_TREE, guide_tree)
    lengths = valid[active_indices].sum(axis=1).astype(np.int32)
    capacity_ok = (
        record_count <= plan.maximum_sequences
        and sequences.sequence_capacity <= plan.maximum_sequence_length
        and int(lengths.sum()) <= plan.maximum_alignment_length
    )
    if guide_tree is not None and guide_tree.leaf_count != record_count:
        return _failure(sequences, plan, MSA_STATUS_INVALID_GUIDE_TREE, guide_tree)
    if not capacity_ok:
        return _failure(sequences, plan, MSA_STATUS_CAPACITY_EXCEEDED, guide_tree)

    alphabet = sequences.alphabet
    gap_code = alphabet.code(alphabet.gap_symbol)
    distributions, scorable = _token_distributions(alphabet)
    initial_sequences = [
        tokens[slot, : lengths[index]].astype(np.int32)
        for index, slot in enumerate(active_indices)
    ]
    initial_soft = [
        soft[slot, : lengths[index]].astype(bool)
        for index, slot in enumerate(active_indices)
    ]
    if any(np.any(~scorable[value[value != gap_code]]) for value in initial_sequences):
        raise ValueError("MSA inputs contain a nonscorable active symbol.")
    signatures = [
        tuple(int(value) for value in sequence) for sequence in initial_sequences
    ]
    tie_count = 0
    if guide_tree is None:
        guide_tree, tie_count = _upgma_tree(
            initial_sequences, signatures, gap_code, distributions, plan
        )

    profiles: dict[int, tuple[tuple[int, ...], np.ndarray, np.ndarray]] = {
        index: (
            (index,),
            initial_sequences[index].reshape((1, -1)),
            initial_soft[index].reshape((1, -1)),
        )
        for index in range(record_count)
    }
    total_profile_score = 0.0
    for merge_index, children in enumerate(np.asarray(guide_tree.merges)):
        left_node, right_node = int(children[0]), int(children[1])
        left_members, left_tokens, left_soft = profiles.pop(left_node)
        right_members, right_tokens, right_soft = profiles.pop(right_node)
        left_signature = tuple(sorted(signatures[index] for index in left_members))
        right_signature = tuple(sorted(signatures[index] for index in right_members))
        if right_signature < left_signature:
            left_members, right_members = right_members, left_members
            left_tokens, right_tokens = right_tokens, left_tokens
            left_soft, right_soft = right_soft, left_soft
        merged_tokens, merged_soft, merge_score = _merge_profiles(
            left_tokens,
            left_soft,
            right_tokens,
            right_soft,
            gap_code,
            distributions,
            plan,
        )
        members = left_members + right_members
        order = np.argsort(np.asarray(members))
        profiles[record_count + merge_index] = (
            tuple(sorted(members)),
            merged_tokens[order],
            merged_soft[order],
        )
        total_profile_score += merge_score

    _, alignment_tokens, alignment_soft = next(iter(profiles.values()))
    alignment_length = int(alignment_tokens.shape[1])
    if alignment_length > plan.maximum_alignment_length:
        return _failure(sequences, plan, MSA_STATUS_CAPACITY_EXCEEDED, guide_tree)
    pad_code = alphabet.code(alphabet.pad_symbol)
    output_shape = (sequences.record_capacity, plan.maximum_alignment_length)
    output_tokens = np.full(output_shape, pad_code, dtype=np.int32)
    output_valid = np.zeros(output_shape, dtype=bool)
    output_soft = np.zeros(output_shape, dtype=bool)
    output_tokens[active_indices, :alignment_length] = alignment_tokens
    output_valid[active_indices, :alignment_length] = True
    output_soft[active_indices, :alignment_length] = alignment_soft
    alignment = SequenceBatch(
        ids,
        output_tokens,
        output_valid,
        cases,
        output_soft,
        alphabet,
    )
    expected_merges = record_count - 1
    evidence = ProgressiveMSAEvidence(
        jnp.asarray(True),
        jnp.asarray(True),
        jnp.asarray(expected_merges, dtype=jnp.int32),
        jnp.asarray(expected_merges, dtype=jnp.int32),
        jnp.asarray(tie_count, dtype=jnp.int32),
        jnp.asarray(total_profile_score),
        jnp.asarray(plan.guide_tree_method != "supplied"),
        jnp.asarray(True),
        jnp.asarray(False),
    )
    return ProgressiveMSAResult(
        alignment,
        guide_tree,
        jnp.arange(plan.maximum_alignment_length) < alignment_length,
        jnp.asarray(alignment_length, dtype=jnp.int32),
        jnp.asarray(True),
        jnp.asarray(MSA_STATUS_VALID, dtype=jnp.int32),
        evidence,
        plan.method_contract,
    )


__all__ = [
    "GuideTree",
    "MSA_STATUS_CAPACITY_EXCEEDED",
    "MSA_STATUS_INVALID_GUIDE_TREE",
    "MSA_STATUS_VALID",
    "ProgressiveMSAEvidence",
    "ProgressiveMSAPlan",
    "ProgressiveMSAResult",
    "progressive_multiple_alignment",
]
