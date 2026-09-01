#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum, IntFlag

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ..foundation import (
    BioinformaticsMethodContract,
    DifferentiationKind,
    ExecutionKind,
    MethodKind,
    OutputKind,
)


class CompetitionLevel(IntEnum):
    """Scientific entity level at which target-decoy competition occurs."""

    PSM = 0
    PEPTIDE = 1
    PROTEIN = 2


class ProteinInferenceStatus(IntEnum):
    """Status of target-decoy competition or protein inference."""

    SUCCESS = 0
    NO_TARGET_WINNERS = 1
    NONFINITE = 2
    INVALID_RELATION = 3


class ProteinInferenceEvidence(IntFlag):
    """Evidence retained by protein-level statistical inference."""

    NONE = 0
    TARGET_DECOY_COMPETITION = 1
    GROUP_SPECIFIC_FDR = 2
    SHARED_PEPTIDES = 4
    UNIQUE_PEPTIDES = 8
    RAZOR_ASSIGNMENT = 16
    INDISTINGUISHABLE_GROUPS = 32


_COMPETITION_CONTRACT = BioinformaticsMethodContract(
    "target-decoy competition and group FDR",
    MethodKind.APPROXIMATE_MODEL,
    ExecutionKind.EXACT_DISCRETE,
    DifferentiationKind.NONE,
    OutputKind.STRUCTURED,
    conditioning_statement=(
        "One deterministic winner is selected per competition identity; tail-area "
        "target-decoy estimates and monotone q-values are computed within each "
        "declared FDR group."
    ),
    truncation_statement="Every active entity participates; no score or FDR group is truncated.",
    capacity_semantics="Pairwise competition and q-value work are bounded by the entity capacity squared.",
    assumptions=(
        "Target and decoy scores are exchangeable under the null.",
        "Competition identities do not cross FDR groups.",
    ),
    nondifferentiable_outputs=("all outputs",),
)

_INFERENCE_CONTRACT = BioinformaticsMethodContract(
    "shared-peptide protein grouping and razor inference",
    MethodKind.HEURISTIC,
    ExecutionKind.EXACT_DISCRETE,
    DifferentiationKind.ALMOST_EVERYWHERE,
    OutputKind.STRUCTURED,
    conditioning_statement=(
        "Proteins with identical observed peptide support are grouped; shared "
        "peptides are assigned by unique-support count, then total evidence, then "
        "stable input order."
    ),
    truncation_statement="All protein-peptide relations and peptide evidence slots are evaluated without truncation.",
    capacity_semantics="Protein, peptide, and relation capacities are fixed by the input arrays.",
    assumptions=("Peptide identifiers are unique within the evidence batch.",),
    nondifferentiable_outputs=(
        "group_ids",
        "representative_mask",
        "razor_protein_index",
        "status",
        "evidence",
    ),
)


class TargetDecoyCompetitionBatch(StrictModule):
    """Scores and identities for bounded target-decoy competition."""

    scores: Array
    entity_ids: Array
    competition_ids: Array
    fdr_group_ids: Array
    is_decoy: Array
    active_mask: Array
    level: CompetitionLevel = eqx.field(static=True)
    entity_capacity: int = eqx.field(static=True)

    def __init__(
        self,
        scores: ArrayLike,
        entity_ids: ArrayLike,
        competition_ids: ArrayLike,
        is_decoy: ArrayLike,
        active_mask: ArrayLike,
        /,
        *,
        level: CompetitionLevel,
        fdr_group_ids: ArrayLike | None = None,
    ):
        score_host = np.asarray(scores)
        entities = np.asarray(entity_ids)
        competitions = np.asarray(competition_ids)
        decoys = np.asarray(is_decoy, dtype=bool)
        mask = np.asarray(active_mask, dtype=bool)
        if score_host.ndim != 1 or score_host.size == 0:
            raise ValueError("scores must be a non-empty vector.")
        if any(
            value.shape != score_host.shape
            for value in (entities, competitions, decoys, mask)
        ):
            raise ValueError("All competition arrays must have the scores shape.")
        if not np.issubdtype(entities.dtype, np.integer) or not np.issubdtype(
            competitions.dtype, np.integer
        ):
            raise TypeError("entity_ids and competition_ids must contain integers.")
        groups = (
            np.zeros(score_host.shape, dtype=np.int64)
            if fdr_group_ids is None
            else np.asarray(fdr_group_ids)
        )
        if groups.shape != score_host.shape or not np.issubdtype(
            groups.dtype, np.integer
        ):
            raise TypeError("fdr_group_ids must be an integer score-capacity vector.")
        if np.any(~np.isfinite(score_host[mask])):
            raise ValueError("Active competition scores must be finite.")
        if (
            np.any(entities[mask] < 0)
            or np.any(competitions[mask] < 0)
            or np.any(groups[mask] < 0)
        ):
            raise ValueError("Active identifiers must be nonnegative.")
        for competition in np.unique(competitions[mask]):
            competition_groups = np.unique(groups[mask & (competitions == competition)])
            if competition_groups.size != 1:
                raise ValueError("A competition identity cannot cross FDR groups.")
        for value in (score_host, entities, competitions, groups, decoys):
            if np.any(value[~mask] != 0):
                raise ValueError(
                    "Inactive competition entries must be zero/false padding."
                )
        self.scores = jnp.asarray(score_host)
        self.entity_ids = jnp.asarray(entities, dtype=jnp.int64)
        self.competition_ids = jnp.asarray(competitions, dtype=jnp.int64)
        self.fdr_group_ids = jnp.asarray(groups, dtype=jnp.int64)
        self.is_decoy = jnp.asarray(decoys)
        self.active_mask = jnp.asarray(mask)
        self.level = CompetitionLevel(level)
        self.entity_capacity = int(score_host.size)


class TargetDecoyCompetitionResult(StrictModule):
    """Competition winners, groupwise FDR estimates, and monotone q-values."""

    winner_mask: Array
    target_winner_mask: Array
    decoy_winner_mask: Array
    rank: Array
    false_discovery_rate: Array
    q_value: Array
    valid: Array
    status: Array
    evidence: Array
    level: CompetitionLevel = eqx.field(static=True)
    method_contract: BioinformaticsMethodContract = eqx.field(static=True)


def target_decoy_competition(
    batch: TargetDecoyCompetitionBatch,
    /,
    *,
    decoy_pseudocount: int = 1,
) -> TargetDecoyCompetitionResult:
    """Perform stable picked competition and group-specific monotone FDR control."""
    if not isinstance(batch, TargetDecoyCompetitionBatch):
        raise TypeError("batch must be TargetDecoyCompetitionBatch.")
    pseudocount = int(decoy_pseudocount)
    if pseudocount < 0:
        raise ValueError("decoy_pseudocount must be nonnegative.")
    capacity = batch.entity_capacity
    indices = jnp.arange(capacity)
    same_competition = (
        (batch.competition_ids[:, None] == batch.competition_ids[None, :])
        & batch.active_mask[:, None]
        & batch.active_mask[None, :]
    )
    score_i = batch.scores[:, None]
    score_j = batch.scores[None, :]
    index_i = indices[:, None]
    index_j = indices[None, :]
    competitor_better = same_competition & (
        (score_j > score_i) | ((score_j == score_i) & (index_j < index_i))
    )
    winner = batch.active_mask & ~jnp.any(competitor_better, axis=1)
    target_winner = winner & ~batch.is_decoy
    decoy_winner = winner & batch.is_decoy
    same_group = batch.fdr_group_ids[:, None] == batch.fdr_group_ids[None, :]
    winner_j = winner[None, :]
    ranks_before = (
        winner_j
        & same_group
        & ((score_j > score_i) | ((score_j == score_i) & (index_j < index_i)))
    )
    rank = jnp.sum(ranks_before, axis=1, dtype=jnp.int32)
    at_or_above = (
        winner_j
        & same_group
        & ((score_j > score_i) | ((score_j == score_i) & (index_j <= index_i)))
    )
    targets = jnp.sum(at_or_above & (~batch.is_decoy)[None, :], axis=1)
    decoys = jnp.sum(at_or_above & batch.is_decoy[None, :], axis=1)
    fdr = jnp.minimum((decoys + pseudocount) / jnp.maximum(targets, 1), 1.0)
    later_winner = (
        winner_j
        & same_group
        & ((score_j < score_i) | ((score_j == score_i) & (index_j >= index_i)))
    )
    q_value = jnp.min(jnp.where(later_winner, fdr[None, :], jnp.inf), axis=1)
    q_value = jnp.where(winner, jnp.minimum(q_value, 1.0), 1.0)
    fdr = jnp.where(winner, fdr, 1.0)
    target_count = jnp.sum(target_winner)
    valid = target_count > 0
    return TargetDecoyCompetitionResult(
        winner_mask=winner,
        target_winner_mask=target_winner,
        decoy_winner_mask=decoy_winner,
        rank=jnp.where(winner, rank, -1),
        false_discovery_rate=fdr,
        q_value=q_value,
        valid=valid,
        status=jnp.where(
            valid,
            int(ProteinInferenceStatus.SUCCESS),
            int(ProteinInferenceStatus.NO_TARGET_WINNERS),
        ).astype(jnp.int32),
        evidence=jnp.asarray(
            int(
                ProteinInferenceEvidence.TARGET_DECOY_COMPETITION
                | ProteinInferenceEvidence.GROUP_SPECIFIC_FDR
            ),
            dtype=jnp.uint32,
        ),
        level=batch.level,
        method_contract=_COMPETITION_CONTRACT,
    )


class ProteinPeptideRelation(StrictModule):
    """Fixed protein-by-peptide-slot relation for shared-peptide inference."""

    protein_ids: Array
    peptide_ids: Array
    relation_mask: Array
    protein_is_decoy: Array
    protein_mask: Array
    protein_capacity: int = eqx.field(static=True)
    peptide_slot_capacity: int = eqx.field(static=True)

    def __init__(
        self,
        protein_ids: ArrayLike,
        peptide_ids: ArrayLike,
        relation_mask: ArrayLike,
        /,
        *,
        protein_is_decoy: ArrayLike | None = None,
        protein_mask: ArrayLike | None = None,
    ):
        proteins = np.asarray(protein_ids)
        peptides = np.asarray(peptide_ids)
        relation = np.asarray(relation_mask, dtype=bool)
        if (
            proteins.ndim != 1
            or proteins.size == 0
            or peptides.ndim != 2
            or peptides.shape[0] != proteins.size
        ):
            raise ValueError(
                "protein_ids and peptide_ids require matching positive protein capacity."
            )
        if peptides.shape[1] == 0 or relation.shape != peptides.shape:
            raise ValueError("relation_mask must match a positive peptide slot capacity.")
        if not np.issubdtype(proteins.dtype, np.integer) or not np.issubdtype(
            peptides.dtype, np.integer
        ):
            raise TypeError("Protein and peptide identifiers must be integers.")
        active = (
            np.ones(proteins.shape, dtype=bool)
            if protein_mask is None
            else np.asarray(protein_mask, dtype=bool)
        )
        decoys = (
            np.zeros(proteins.shape, dtype=bool)
            if protein_is_decoy is None
            else np.asarray(protein_is_decoy, dtype=bool)
        )
        if active.shape != proteins.shape or decoys.shape != proteins.shape:
            raise ValueError("protein_mask and protein_is_decoy must match protein_ids.")
        count = int(np.count_nonzero(active))
        if not np.all(active[:count]) or np.any(active[count:]):
            raise ValueError("protein_mask must be a left-prefix mask.")
        if np.any(relation & ~active[:, None]):
            raise ValueError("Inactive proteins cannot carry peptide relations.")
        if np.any(proteins[active] < 0) or np.unique(proteins[active]).size != count:
            raise ValueError("Active protein identifiers must be unique and nonnegative.")
        if np.any(peptides[relation] < 0):
            raise ValueError("Active peptide identifiers must be nonnegative.")
        if (
            np.any(proteins[~active] != 0)
            or np.any(decoys[~active])
            or np.any(peptides[~relation] != 0)
        ):
            raise ValueError(
                "Inactive protein and relation entries must be zero/false padding."
            )
        self.protein_ids = jnp.asarray(proteins, dtype=jnp.int64)
        self.peptide_ids = jnp.asarray(peptides, dtype=jnp.int64)
        self.relation_mask = jnp.asarray(relation)
        self.protein_is_decoy = jnp.asarray(decoys)
        self.protein_mask = jnp.asarray(active)
        self.protein_capacity = int(proteins.size)
        self.peptide_slot_capacity = int(peptides.shape[1])


class PeptideEvidenceBatch(StrictModule):
    """Unique observed peptides and nonnegative evidence scores."""

    peptide_ids: Array
    scores: Array
    q_values: Array
    active_mask: Array
    peptide_capacity: int = eqx.field(static=True)

    def __init__(
        self,
        peptide_ids: ArrayLike,
        scores: ArrayLike,
        q_values: ArrayLike,
        active_mask: ArrayLike,
        /,
    ):
        peptides = np.asarray(peptide_ids)
        scores_host = np.asarray(scores)
        q_host = np.asarray(q_values)
        mask = np.asarray(active_mask, dtype=bool)
        if (
            peptides.ndim != 1
            or peptides.size == 0
            or any(value.shape != peptides.shape for value in (scores_host, q_host, mask))
        ):
            raise ValueError("Peptide evidence fields must be equal non-empty vectors.")
        if not np.issubdtype(peptides.dtype, np.integer):
            raise TypeError("peptide_ids must contain integers.")
        count = int(np.count_nonzero(mask))
        if not np.all(mask[:count]) or np.any(mask[count:]):
            raise ValueError("active_mask must be a left-prefix mask.")
        if np.any(peptides[mask] < 0) or np.unique(peptides[mask]).size != count:
            raise ValueError("Active peptide identifiers must be unique and nonnegative.")
        if np.any(~np.isfinite(scores_host[mask])) or np.any(scores_host[mask] < 0.0):
            raise ValueError(
                "Active peptide evidence scores must be finite and nonnegative."
            )
        if np.any(~np.isfinite(q_host[mask])) or np.any(
            (q_host[mask] < 0.0) | (q_host[mask] > 1.0)
        ):
            raise ValueError("Active peptide q-values must lie in [0, 1].")
        for value in (peptides, scores_host, q_host):
            if np.any(value[~mask] != 0):
                raise ValueError(
                    "Inactive peptide evidence entries must be zero padding."
                )
        self.peptide_ids = jnp.asarray(peptides, dtype=jnp.int64)
        self.scores = jnp.asarray(scores_host)
        self.q_values = jnp.asarray(q_host)
        self.active_mask = jnp.asarray(mask)
        self.peptide_capacity = int(peptides.size)


class ProteinInferenceResult(StrictModule):
    """Protein groups, razor assignments, and per-protein evidence scores."""

    protein_ids: Array
    group_ids: Array
    protein_scores: Array
    unique_peptide_count: Array
    total_peptide_count: Array
    representative_mask: Array
    razor_protein_index: Array
    shared_peptide_mask: Array
    peptide_supported_mask: Array
    protein_is_decoy: Array
    valid: Array
    status: Array
    evidence: Array
    method_contract: BioinformaticsMethodContract = eqx.field(static=True)


def infer_proteins_from_shared_peptides(
    relation: ProteinPeptideRelation,
    evidence: PeptideEvidenceBatch,
    /,
) -> ProteinInferenceResult:
    """Group indistinguishable proteins and assign shared peptides by razor rules."""
    if not isinstance(relation, ProteinPeptideRelation):
        raise TypeError("relation must be ProteinPeptideRelation.")
    if not isinstance(evidence, PeptideEvidenceBatch):
        raise TypeError("evidence must be PeptideEvidenceBatch.")
    relation_match = (
        (relation.peptide_ids[:, :, None] == evidence.peptide_ids[None, None, :])
        & relation.relation_mask[:, :, None]
        & evidence.active_mask[None, None, :]
    )
    incidence = jnp.any(relation_match, axis=1) & relation.protein_mask[:, None]
    peptide_protein_count = jnp.sum(incidence, axis=0, dtype=jnp.int32)
    supported = evidence.active_mask & (peptide_protein_count > 0)
    shared = supported & (peptide_protein_count > 1)
    unique_relation = incidence & (peptide_protein_count[None, :] == 1)
    unique_count = jnp.sum(unique_relation, axis=1, dtype=jnp.int32)
    total_count = jnp.sum(incidence, axis=1, dtype=jnp.int32)
    support_score = jnp.sum(incidence * evidence.scores[None, :], axis=1)
    scale = jnp.max(support_score, initial=0.0) + 1.0
    priority = unique_count.astype(support_score.dtype) * scale + support_score
    priority = jnp.where(
        relation.protein_mask[:, None] & incidence, priority[:, None], -jnp.inf
    )
    razor_index = jnp.argmax(priority, axis=0)
    razor_index = jnp.where(supported, razor_index, -1).astype(jnp.int32)
    safe_razor = jnp.maximum(razor_index, 0)
    assigned = (
        jnp.arange(relation.protein_capacity)[:, None] == safe_razor[None, :]
    ) & supported[None, :]
    protein_scores = jnp.sum(assigned * evidence.scores[None, :], axis=1)
    same_incidence = (
        jnp.all(incidence[:, None, :] == incidence[None, :, :], axis=-1)
        & relation.protein_mask[:, None]
        & relation.protein_mask[None, :]
    )
    indices = jnp.arange(relation.protein_capacity)
    representative_index = jnp.min(
        jnp.where(same_incidence, indices[None, :], relation.protein_capacity),
        axis=1,
    )
    representative = (
        relation.protein_mask & (indices == representative_index) & (total_count > 0)
    )
    group_ids = jnp.where(
        relation.protein_mask,
        relation.protein_ids[
            jnp.minimum(representative_index, relation.protein_capacity - 1)
        ],
        -1,
    )
    has_support = jnp.any(supported)
    valid = has_support
    evidence_bits = jnp.asarray(
        int(ProteinInferenceEvidence.RAZOR_ASSIGNMENT), dtype=jnp.uint32
    )
    evidence_bits = evidence_bits | jnp.where(
        jnp.any(shared), int(ProteinInferenceEvidence.SHARED_PEPTIDES), 0
    ).astype(jnp.uint32)
    evidence_bits = evidence_bits | jnp.where(
        jnp.any(unique_count > 0), int(ProteinInferenceEvidence.UNIQUE_PEPTIDES), 0
    ).astype(jnp.uint32)
    evidence_bits = evidence_bits | jnp.where(
        jnp.any(jnp.sum(same_incidence, axis=1) > 1),
        int(ProteinInferenceEvidence.INDISTINGUISHABLE_GROUPS),
        0,
    ).astype(jnp.uint32)
    return ProteinInferenceResult(
        protein_ids=relation.protein_ids,
        group_ids=group_ids,
        protein_scores=jnp.where(relation.protein_mask, protein_scores, 0.0),
        unique_peptide_count=jnp.where(relation.protein_mask, unique_count, 0),
        total_peptide_count=jnp.where(relation.protein_mask, total_count, 0),
        representative_mask=representative,
        razor_protein_index=razor_index,
        shared_peptide_mask=shared,
        peptide_supported_mask=supported,
        protein_is_decoy=relation.protein_is_decoy,
        valid=valid,
        status=jnp.where(
            valid,
            int(ProteinInferenceStatus.SUCCESS),
            int(ProteinInferenceStatus.INVALID_RELATION),
        ).astype(jnp.int32),
        evidence=evidence_bits,
        method_contract=_INFERENCE_CONTRACT,
    )


__all__ = [
    "CompetitionLevel",
    "PeptideEvidenceBatch",
    "ProteinInferenceEvidence",
    "ProteinInferenceResult",
    "ProteinInferenceStatus",
    "ProteinPeptideRelation",
    "TargetDecoyCompetitionBatch",
    "TargetDecoyCompetitionResult",
    "infer_proteins_from_shared_peptides",
    "target_decoy_competition",
]
