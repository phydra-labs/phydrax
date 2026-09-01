#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..foundation import (
    BioinformaticsMethodContract,
    DifferentiationKind,
    ExecutionKind,
    MethodKind,
    OutputKind,
)
from ._taxonomy import resolve_taxon_ids, TaxonomyTree, TaxonomyVersion


class AssignmentStatus(IntEnum):
    ASSIGNED_UNIQUE = 0
    ASSIGNED_AMBIGUOUS = 1
    UNCLASSIFIED = 2
    VERSION_MISMATCH = 3
    INVALID_SCORE = 4
    CAPACITY_EXCEEDED = 5


def _candidate_contract() -> BioinformaticsMethodContract:
    return BioinformaticsMethodContract(
        "supplied taxonomic-candidate boundary",
        MethodKind.HEURISTIC,
        ExecutionKind.EXACT_DISCRETE,
        DifferentiationKind.NONE,
        OutputKind.RANKING,
        conditioning_statement=(
            "Candidate taxon IDs and scores are supplied by an external search or classifier."
        ),
        truncation_statement=(
            "No claim is made about retrieval sensitivity or candidates absent from the supplied "
            "fixed-width table."
        ),
        capacity_semantics="Every supplied candidate occupies an explicit table slot.",
        assumptions=("Scores are larger-is-better nonnegative support values.",),
        nondifferentiable_outputs=("taxon_ids", "candidate_valid"),
        input_dtype="int32/float/bool",
        compute_dtype="int32/float/bool",
        output_dtype="int32/float/bool",
    )


def _assignment_contract(relative_score_threshold: float) -> BioinformaticsMethodContract:
    return BioinformaticsMethodContract(
        "ambiguity-preserving taxonomic assignment",
        MethodKind.HEURISTIC,
        ExecutionKind.FLOATING_POINT_DIRECT,
        DifferentiationKind.NONE,
        OutputKind.PROBABILISTIC,
        conditioning_statement=(
            "All version-resolved candidates scoring at least "
            f"{relative_score_threshold:g} times the best score are retained and normalized."
        ),
        truncation_statement=(
            "If retained assignments exceed output capacity, the entire read is marked failed; "
            "no top-k truncation occurs."
        ),
        capacity_semantics="Candidate and output assignment widths are explicit fixed capacities.",
        assumptions=(
            "Supplied scores are comparable within a read.",
            "Normalized weights express assignment ambiguity, not a calibrated posterior claim.",
        ),
        nondifferentiable_outputs=("taxon_indices", "assigned_valid", "status"),
        input_dtype="int32/float/bool",
        compute_dtype="float32/int32/bool",
        output_dtype="float32/int32/bool",
    )


class SuppliedTaxonomicCandidates(StrictModule, NonTrainableState):
    """Fixed-width candidates from an explicitly external search/classifier boundary."""

    record_ids: Array
    taxon_ids: Array
    scores: Array
    candidate_valid: Array
    case_mask: Array
    database_version: TaxonomyVersion
    method_contract: BioinformaticsMethodContract
    provenance: str = eqx.field(static=True)
    candidate_id: str = eqx.field(static=True)

    def __init__(
        self,
        record_ids: ArrayLike,
        taxon_ids: ArrayLike,
        scores: ArrayLike,
        candidate_valid: ArrayLike,
        case_mask: ArrayLike,
        database_version: TaxonomyVersion,
        /,
        *,
        provenance: str = "caller-supplied",
        candidate_id: str | None = None,
    ):
        records = jnp.asarray(record_ids, dtype=jnp.int32)
        taxa = jnp.asarray(taxon_ids, dtype=jnp.int32)
        support = jnp.asarray(scores)
        valid = jnp.asarray(candidate_valid, dtype=bool)
        cases = jnp.asarray(case_mask, dtype=bool)
        if taxa.ndim != 2 or taxa.shape[1] < 1:
            raise ValueError(
                "taxon_ids must have shape (record_capacity, positive_width)."
            )
        if support.shape != taxa.shape or valid.shape != taxa.shape:
            raise ValueError("Candidate scores and masks must match taxon_ids.")
        if records.shape != (taxa.shape[0],) or cases.shape != records.shape:
            raise ValueError("record_ids and case_mask must match record capacity.")
        if not jnp.issubdtype(support.dtype, jnp.inexact):
            support = support.astype(jnp.float32)
        provenance_ = str(provenance).strip()
        if not provenance_:
            raise ValueError("Candidate provenance must be non-empty.")
        identity = candidate_id or canonical_fingerprint(
            {
                "kind": "supplied-taxonomic-candidates",
                "version": database_version.version_id,
                "provenance": provenance_,
                "arrays": array_tree_fingerprint((records, taxa, support, valid, cases)),
            }
        )
        if not identity:
            raise ValueError("candidate_id must be non-empty.")
        self.record_ids = records
        self.taxon_ids = taxa
        self.scores = support
        self.candidate_valid = valid
        self.case_mask = cases
        self.database_version = database_version
        self.method_contract = _candidate_contract()
        self.provenance = provenance_
        self.candidate_id = identity


class AmbiguousTaxonomicAssignment(StrictModule):
    """Per-read multi-assignment with a separately conserved unclassified mass."""

    record_ids: Array
    taxon_indices: Array
    taxon_ids: Array
    weights: Array
    assigned_valid: Array
    case_mask: Array
    unclassified_mass: Array
    valid: Array
    status: Array
    evidence: Array
    method_contract: BioinformaticsMethodContract
    taxonomy_id: str = eqx.field(static=True)
    version_id: str = eqx.field(static=True)
    candidate_id: str = eqx.field(static=True)


def assign_taxonomy(
    taxonomy: TaxonomyTree,
    candidates: SuppliedTaxonomicCandidates,
    /,
    *,
    relative_score_threshold: float = 1.0,
    assignment_capacity: int | None = None,
) -> AmbiguousTaxonomicAssignment:
    """Retain all near-best distinct assignments instead of forcing one label or LCA."""
    threshold = float(relative_score_threshold)
    if not (0.0 < threshold <= 1.0):
        raise ValueError("relative_score_threshold must lie in (0, 1].")
    record_capacity, candidate_width = candidates.taxon_ids.shape
    output_width = (
        candidate_width if assignment_capacity is None else int(assignment_capacity)
    )
    if output_width < 1 or output_width > candidate_width:
        raise ValueError("assignment_capacity must lie in [1, candidate_width].")

    resolution = resolve_taxon_ids(
        taxonomy,
        candidates.taxon_ids,
        database_version=candidates.database_version,
    )
    version_match = jnp.asarray(
        candidates.database_version.version_id == taxonomy.version.version_id,
        dtype=bool,
    )
    finite_nonnegative = jnp.isfinite(candidates.scores) & (candidates.scores >= 0.0)
    malformed_score = jnp.any(candidates.candidate_valid & (~finite_nonnegative), axis=1)
    usable = (
        candidates.candidate_valid
        & finite_nonnegative
        & resolution.valid
        & candidates.case_mask[:, None]
    )
    safe_scores = jnp.where(usable, candidates.scores, -jnp.inf)
    best_score = jnp.max(safe_scores, axis=1)
    has_positive_support = jnp.isfinite(best_score) & (best_score > 0.0)
    eligible = (
        usable
        & has_positive_support[:, None]
        & (candidates.scores >= best_score[:, None] * threshold)
    )

    same_resolved = (
        resolution.taxon_indices[:, :, None] == resolution.taxon_indices[:, None, :]
    )
    earlier = jnp.arange(candidate_width)[:, None] > jnp.arange(candidate_width)[None, :]
    first_distinct = ~jnp.any(
        same_resolved & eligible[:, :, None] & eligible[:, None, :] & earlier[None, :, :],
        axis=2,
    )
    distinct_eligible = eligible & first_distinct
    grouped_score = jnp.max(
        jnp.where(
            same_resolved & eligible[:, None, :],
            candidates.scores[:, None, :],
            -jnp.inf,
        ),
        axis=2,
    )
    grouped_score = jnp.where(distinct_eligible, grouped_score, -jnp.inf)
    selected_count = jnp.sum(distinct_eligible, axis=1, dtype=jnp.int32)
    overflow = selected_count > output_width
    ordering = jnp.argsort(-grouped_score, axis=1, stable=True)
    selected_slots = ordering[:, :output_width]
    selected_indices = jnp.take_along_axis(
        resolution.taxon_indices, selected_slots, axis=1
    )
    selected_ids = jnp.take_along_axis(
        resolution.resolved_taxon_ids, selected_slots, axis=1
    )
    selected_scores = jnp.take_along_axis(grouped_score, selected_slots, axis=1)
    selected_valid = jnp.arange(output_width)[None, :] < selected_count[:, None]
    read_usable = candidates.case_mask & version_match & (~malformed_score) & (~overflow)
    selected_valid = selected_valid & read_usable[:, None]
    selected_scores = jnp.where(selected_valid, selected_scores, 0.0)
    score_total = jnp.sum(selected_scores, axis=1)
    weights = jnp.where(
        selected_valid,
        selected_scores
        / jnp.maximum(score_total[:, None], jnp.finfo(selected_scores.dtype).tiny),
        0.0,
    )
    assigned_count = jnp.sum(selected_valid, axis=1, dtype=jnp.int32)
    classified = assigned_count > 0
    unclassified = jnp.where(candidates.case_mask & (~classified), 1.0, 0.0).astype(
        weights.dtype
    )
    selected_indices = jnp.where(selected_valid, selected_indices, -1)
    selected_ids = jnp.where(selected_valid, selected_ids, -1)
    status = jnp.where(
        ~version_match,
        int(AssignmentStatus.VERSION_MISMATCH),
        jnp.where(
            malformed_score,
            int(AssignmentStatus.INVALID_SCORE),
            jnp.where(
                overflow,
                int(AssignmentStatus.CAPACITY_EXCEEDED),
                jnp.where(
                    assigned_count > 1,
                    int(AssignmentStatus.ASSIGNED_AMBIGUOUS),
                    jnp.where(
                        assigned_count == 1,
                        int(AssignmentStatus.ASSIGNED_UNIQUE),
                        int(AssignmentStatus.UNCLASSIFIED),
                    ),
                ),
            ),
        ),
    ).astype(jnp.int32)
    status = jnp.where(
        candidates.case_mask,
        status,
        int(AssignmentStatus.UNCLASSIFIED),
    ).astype(jnp.int32)
    result_valid = candidates.case_mask & version_match & (~malformed_score) & (~overflow)
    contract = _assignment_contract(threshold)
    evidence = jnp.stack(
        (
            jnp.sum(candidates.candidate_valid, axis=1, dtype=jnp.int32),
            jnp.sum(usable, axis=1, dtype=jnp.int32),
            selected_count,
            assigned_count,
            jnp.sum(
                candidates.candidate_valid & resolution.was_deleted,
                axis=1,
                dtype=jnp.int32,
            ),
            jnp.sum(
                candidates.candidate_valid & resolution.was_merged,
                axis=1,
                dtype=jnp.int32,
            ),
        ),
        axis=1,
    )
    return AmbiguousTaxonomicAssignment(
        candidates.record_ids,
        selected_indices,
        selected_ids,
        weights,
        selected_valid,
        candidates.case_mask,
        unclassified,
        result_valid,
        status,
        evidence,
        contract,
        taxonomy.taxonomy_id,
        taxonomy.version.version_id,
        candidates.candidate_id,
    )


__all__ = [
    "AmbiguousTaxonomicAssignment",
    "AssignmentStatus",
    "SuppliedTaxonomicCandidates",
    "assign_taxonomy",
]
