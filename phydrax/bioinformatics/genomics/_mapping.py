#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..foundation import (
    BioinformaticsMethodContract,
    DifferentiationKind,
    ExecutionKind,
    MethodKind,
    OutputKind,
)
from ..sequence import SequenceBatch
from ._alignment_events import AlignmentEventBatch
from ._reads import ReadBatch


MAPPING_STATUS_VALID = 0
MAPPING_STATUS_NO_CANDIDATES = 1
MAPPING_STATUS_CANDIDATE_TRUNCATED = 2
MAPPING_STATUS_MAPQ_UNCALIBRATED = 3
MAPPING_STATUS_INVALID_INPUT = 4
PILEUP_STATUS_VALID = 0
PILEUP_STATUS_REFERENCE_MISSING = 1
PILEUP_STATUS_REFERENCE_BOUNDS = 2
PILEUP_STATUS_CIGAR_REFERENCE_MISMATCH = 3
PILEUP_STATUS_INVALID_EVENTS = 4


_PILEUP_CONTRACT = BioinformaticsMethodContract(
    "reference_aware_pileup_likelihood",
    MethodKind.APPROXIMATE_MODEL,
    ExecutionKind.FLOATING_POINT_DIRECT,
    DifferentiationKind.ALMOST_EVERYWHERE,
    OutputKind.PROBABILISTIC,
    conditioning_statement=(
        "Likelihood conditions on the supplied alignment events, reference identity, "
        "Phred calls, and fixed indel/clip/splice penalties."
    ),
    truncation_statement=(
        "No partial event expansion is scored; an exceeded event capacity invalidates "
        "the complete candidate."
    ),
    capacity_semantics="Alignment-event and reference sequence axes are fixed capacities.",
    assumptions=(
        "Phred errors are conditionally independent across aligned read positions.",
        "Unknown base quality gives a uniform canonical-base observation law.",
    ),
    nondifferentiable_outputs=(
        "valid",
        "status",
        "reference_mismatch",
        "mismatch_count",
    ),
    compute_dtype="float32",
    output_dtype="float32",
)

_MAPPING_CONTRACT = BioinformaticsMethodContract(
    "candidate_mapping_evidence",
    MethodKind.HEURISTIC,
    ExecutionKind.FLOATING_POINT_DIRECT,
    DifferentiationKind.ALMOST_EVERYWHERE,
    OutputKind.PROBABILISTIC,
    conditioning_statement=(
        "Candidate probabilities are conditional on the supplied heuristic candidate "
        "set; pair scores and the explicit null/unmapped score are caller-supplied."
    ),
    truncation_statement=(
        "Candidate overflow is observable and makes MAPQ uncalibrated; candidates are "
        "never silently truncated by this kernel."
    ),
    capacity_semantics="The candidate axis is a fixed compile-time capacity.",
    assumptions=(
        "Pair-score inputs already account for overlapping mates and any intended UMI "
        "duplicate downweighting.",
        "Candidate retrieval is heuristic and is not evidence of exhaustive search.",
    ),
    nondifferentiable_outputs=(
        "best_candidate_index",
        "mapped",
        "valid",
        "status",
        "mapq_calibrated",
    ),
    input_dtype="float32",
    compute_dtype="float32",
    output_dtype="float32",
)


class MappingCandidateBatch(StrictModule):
    """Fixed-capacity candidate loci returned by a heuristic retrieval stage."""

    reference_id: Array
    reference_start: Array
    reverse_strand: Array
    candidate_mask: Array
    candidate_log_prior: Array
    retrieval_truncated: Array
    candidate_count: Array
    valid: Array
    status: Array
    evidence: Array

    def __init__(
        self,
        reference_id: ArrayLike,
        reference_start: ArrayLike,
        reverse_strand: ArrayLike,
        candidate_mask: ArrayLike,
        candidate_log_prior: ArrayLike,
        retrieval_truncated: ArrayLike,
        /,
    ):
        reference_id_ = jnp.asarray(reference_id, dtype=jnp.int32)
        reference_start_ = jnp.asarray(reference_start, dtype=jnp.int32)
        reverse_ = jnp.asarray(reverse_strand, dtype=bool)
        mask = jnp.asarray(candidate_mask, dtype=bool)
        prior = jnp.asarray(candidate_log_prior, dtype=float)
        if reference_id_.ndim != 2:
            raise ValueError("candidate arrays must have shape (read, candidate).")
        shape = reference_id_.shape
        if any(
            values.shape != shape for values in (reference_start_, reverse_, mask, prior)
        ):
            raise ValueError("All candidate arrays must have matching shapes.")
        truncated = jnp.asarray(retrieval_truncated, dtype=bool)
        if truncated.shape != shape[:1]:
            raise ValueError("retrieval_truncated must have shape (read,).")
        prefix_mask = jnp.cumprod(mask.astype(jnp.int8), axis=-1).astype(bool)
        prefix_valid = jnp.all(mask == prefix_mask, axis=-1)
        coordinate_valid = jnp.all(
            (~mask) | ((reference_id_ >= 0) & (reference_start_ >= 0)), axis=-1
        )
        prior_valid = jnp.all((~mask) | jnp.isfinite(prior), axis=-1)
        valid = prefix_valid & coordinate_valid & prior_valid
        count = jnp.sum(mask, axis=-1, dtype=jnp.int32)
        self.reference_id = reference_id_
        self.reference_start = reference_start_
        self.reverse_strand = reverse_
        self.candidate_mask = mask
        self.candidate_log_prior = prior
        self.retrieval_truncated = truncated
        self.candidate_count = count
        self.valid = valid
        self.status = jnp.where(
            ~valid,
            MAPPING_STATUS_INVALID_INPUT,
            jnp.where(
                truncated,
                MAPPING_STATUS_CANDIDATE_TRUNCATED,
                MAPPING_STATUS_VALID,
            ),
        ).astype(jnp.int8)
        self.evidence = count

    @property
    def candidate_capacity(self) -> int:
        return self.reference_id.shape[1]


class MappingExecutionPlan(StrictModule):
    """Static likelihood, candidate, and MAPQ semantics for one mapping bucket."""

    candidate_capacity: int = eqx.field(static=True)
    event_capacity: int = eqx.field(static=True)
    insertion_log_probability: float = eqx.field(static=True)
    deletion_log_probability: float = eqx.field(static=True)
    splice_log_probability: float = eqx.field(static=True)
    soft_clip_log_probability: float = eqx.field(static=True)
    probability_floor: float = eqx.field(static=True)
    score_temperature: float = eqx.field(static=True)
    mapq_cap: float = eqx.field(static=True)
    external_mapq_calibration: bool = eqx.field(static=True)
    fingerprint: str = eqx.field(static=True)

    def __init__(
        self,
        candidate_capacity: int,
        event_capacity: int,
        /,
        *,
        insertion_log_probability: float = -4.0,
        deletion_log_probability: float = -4.0,
        splice_log_probability: float = -1.0,
        soft_clip_log_probability: float = -0.25,
        probability_floor: float = 1.0e-7,
        score_temperature: float = 1.0,
        mapq_cap: float = 60.0,
        external_mapq_calibration: bool = False,
    ):
        candidate_capacity_ = int(candidate_capacity)
        event_capacity_ = int(event_capacity)
        probabilities = tuple(
            float(value)
            for value in (
                insertion_log_probability,
                deletion_log_probability,
                splice_log_probability,
                soft_clip_log_probability,
            )
        )
        floor = float(probability_floor)
        temperature = float(score_temperature)
        cap = float(mapq_cap)
        if candidate_capacity_ < 1 or event_capacity_ < 1:
            raise ValueError("Mapping capacities must be positive.")
        if any(not isfinite(value) or value > 0.0 for value in probabilities):
            raise ValueError("Event log probabilities must be finite and non-positive.")
        if not isfinite(floor) or not 0.0 < floor < 0.5:
            raise ValueError("probability_floor must be between zero and one half.")
        if (
            not isfinite(temperature)
            or not isfinite(cap)
            or temperature <= 0.0
            or cap <= 0.0
        ):
            raise ValueError(
                "score_temperature and mapq_cap must be finite and positive."
            )
        self.candidate_capacity = candidate_capacity_
        self.event_capacity = event_capacity_
        (
            self.insertion_log_probability,
            self.deletion_log_probability,
            self.splice_log_probability,
            self.soft_clip_log_probability,
        ) = probabilities
        self.probability_floor = floor
        self.score_temperature = temperature
        self.mapq_cap = cap
        self.external_mapq_calibration = bool(external_mapq_calibration)
        self.fingerprint = canonical_fingerprint(
            {
                "kind": "mapping-execution-plan",
                "candidate_capacity": candidate_capacity_,
                "event_capacity": event_capacity_,
                "event_log_probabilities": probabilities,
                "probability_floor": floor,
                "score_temperature": temperature,
                "mapq_cap": cap,
                "external_mapq_calibration": bool(external_mapq_calibration),
                "candidate_retrieval": "heuristic",
            }
        )


class PileupLikelihoodResult(StrictModule):
    """Reference-aware conditional log likelihood for each supplied candidate."""

    candidate_log_likelihood: Array
    aligned_base_count: Array
    mismatch_count: Array
    reference_mismatch: Array
    valid: Array
    status: Array
    evidence: Array
    method_contract: BioinformaticsMethodContract = eqx.field(static=True)


class MappingEvidenceResult(StrictModule):
    """Audited candidate/null probabilities and MAPQ interpretation status."""

    candidate_log_score: Array
    conditional_candidate_log_probability: Array
    candidate_log_probability: Array
    null_log_probability: Array
    best_candidate_index: Array
    mapped: Array
    mapping_quality: Array
    mapping_quality_known: Array
    mapq_calibrated: Array
    conditional_on_supplied_candidates: Array
    candidate_truncated: Array
    reference_mismatch: Array
    valid: Array
    status: Array
    evidence: Array
    method_contract: BioinformaticsMethodContract = eqx.field(static=True)


def _alphabet_observation_distributions(batch: SequenceBatch, /) -> Array:
    alphabet = batch.alphabet
    canonical = alphabet.canonical_symbols
    canonical_index = {symbol: index for index, symbol in enumerate(canonical)}
    ambiguity = alphabet.ambiguity_map
    uniform_symbols = {
        alphabet.unknown_symbol,
        alphabet.missing_symbol,
        alphabet.mask_symbol,
    }
    rows: list[list[float]] = []
    for symbol in alphabet.symbols:
        values = [0.0] * len(canonical)
        if symbol in canonical_index:
            values[canonical_index[symbol]] = 1.0
        elif symbol in ambiguity:
            probability = 1.0 / len(ambiguity[symbol])
            for value in ambiguity[symbol]:
                values[canonical_index[value]] = probability
        elif symbol in uniform_symbols:
            values = [1.0 / len(canonical)] * len(canonical)
        rows.append(values)
    return jnp.asarray(rows, dtype=float)


def reference_aware_pileup_likelihood(
    reads: ReadBatch,
    candidates: MappingCandidateBatch,
    events: AlignmentEventBatch,
    references: SequenceBatch,
    plan: MappingExecutionPlan,
    /,
) -> PileupLikelihoodResult:
    """Score supplied alignment events against explicitly identified references."""

    read_count, candidate_capacity = candidates.reference_id.shape
    if read_count != reads.layout.max_reads:
        raise ValueError("Candidate read capacity must match ReadBatch.")
    if candidate_capacity != plan.candidate_capacity:
        raise ValueError("Candidate capacity must match MappingExecutionPlan.")
    expected_event_shape = (read_count, candidate_capacity, plan.event_capacity)
    if events.operation.shape != expected_event_shape:
        raise ValueError("Alignment event shape must match mapping plan capacities.")
    if reads.sequence.alphabet.fingerprint != references.alphabet.fingerprint:
        raise ValueError("Read and reference sequences must use the same alphabet.")

    reference_match = (
        candidates.reference_id[..., None] == references.record_ids[None, None, :]
    ) & references.case_mask[None, None, :]
    reference_found = jnp.any(reference_match, axis=-1)
    reference_slot = jnp.argmax(reference_match, axis=-1).astype(jnp.int32)
    reference_rows = references.token_codes[reference_slot]
    reference_valid_rows = references.valid_mask[reference_slot]

    query_position = events.query_index
    reference_position = events.reference_position
    query_in_bounds = (query_position >= 0) & (
        query_position < reads.layout.max_read_length
    )
    reference_in_bounds = (reference_position >= 0) & (
        reference_position < references.sequence_capacity
    )
    safe_query = jnp.clip(query_position, 0, reads.layout.max_read_length - 1)
    safe_reference = jnp.clip(reference_position, 0, references.sequence_capacity - 1)
    read_rows = jnp.broadcast_to(
        reads.sequence.token_codes[:, None, :],
        (read_count, candidate_capacity, reads.layout.max_read_length),
    )
    read_valid_rows = jnp.broadcast_to(
        reads.sequence.valid_mask[:, None, :], read_rows.shape
    )
    quality_rows = jnp.broadcast_to(
        reads.quality.phred_scores[:, None, :], read_rows.shape
    )
    quality_valid_rows = jnp.broadcast_to(
        reads.quality.valid_mask[:, None, :], read_rows.shape
    )
    observed = jnp.take_along_axis(read_rows, safe_query, axis=-1)
    query_position_valid = query_in_bounds & jnp.take_along_axis(
        read_valid_rows, safe_query, axis=-1
    )
    qualities = jnp.take_along_axis(quality_rows, safe_query, axis=-1)
    quality_known = jnp.take_along_axis(quality_valid_rows, safe_query, axis=-1)
    reference_base = jnp.take_along_axis(reference_rows, safe_reference, axis=-1)
    reference_position_valid = jnp.take_along_axis(
        reference_valid_rows, safe_reference, axis=-1
    )

    distributions = _alphabet_observation_distributions(reads.sequence)
    observed_distribution = distributions[observed]
    reference_distribution = distributions[reference_base]
    same_probability = jnp.sum(observed_distribution * reference_distribution, axis=-1)
    canonical_count = len(reads.sequence.alphabet.canonical_symbols)
    error_probability = jnp.power(10.0, -qualities.astype(float) / 10.0)
    error_probability = jnp.clip(
        error_probability, plan.probability_floor, 1.0 - plan.probability_floor
    )
    known_probability = same_probability * (1.0 - error_probability) + (
        1.0 - same_probability
    ) * error_probability / max(canonical_count - 1, 1)
    base_probability = jnp.where(quality_known, known_probability, 1.0 / canonical_count)
    base_probability = jnp.clip(base_probability, plan.probability_floor, 1.0)

    aligned = events.aligned_base
    insertion = events.insertion
    deletion = events.deletion
    splice = events.reference_skip
    soft_clip = events.soft_clip
    aligned_coordinate_valid = (
        query_position_valid & reference_in_bounds & reference_position_valid
    )
    event_log_probability = jnp.where(aligned, jnp.log(base_probability), 0.0)
    event_log_probability = jnp.where(
        insertion, plan.insertion_log_probability, event_log_probability
    )
    event_log_probability = jnp.where(
        deletion, plan.deletion_log_probability, event_log_probability
    )
    event_log_probability = jnp.where(
        splice, plan.splice_log_probability, event_log_probability
    )
    event_log_probability = jnp.where(
        soft_clip, plan.soft_clip_log_probability, event_log_probability
    )
    event_log_probability = jnp.where(events.active, event_log_probability, 0.0)
    candidate_log_likelihood = jnp.sum(event_log_probability, axis=-1)

    semantic_mismatch = aligned & (
        ((events.operation == 7) & (observed != reference_base))
        | ((events.operation == 8) & (observed == reference_base))
    )
    literal_mismatch = aligned & (observed != reference_base)
    mismatch_count = jnp.sum(literal_mismatch, axis=-1, dtype=jnp.int32)
    query_consuming = aligned | insertion | soft_clip
    reference_consuming = aligned | deletion | splice
    coordinate_failure = jnp.any(
        events.active
        & (
            (query_consuming & ~query_position_valid)
            | (reference_consuming & ~(reference_in_bounds & reference_position_valid))
        ),
        axis=-1,
    )
    cigar_reference_mismatch = jnp.any(semantic_mismatch, axis=-1)
    event_valid = events.valid
    valid = (
        candidates.valid[:, None] & event_valid & reference_found & ~coordinate_failure
    )
    valid = valid & candidates.candidate_mask
    candidate_log_likelihood = jnp.where(valid, candidate_log_likelihood, -jnp.inf)
    status = jnp.where(
        ~event_valid,
        PILEUP_STATUS_INVALID_EVENTS,
        jnp.where(
            ~reference_found,
            PILEUP_STATUS_REFERENCE_MISSING,
            jnp.where(
                coordinate_failure,
                PILEUP_STATUS_REFERENCE_BOUNDS,
                jnp.where(
                    cigar_reference_mismatch,
                    PILEUP_STATUS_CIGAR_REFERENCE_MISMATCH,
                    PILEUP_STATUS_VALID,
                ),
            ),
        ),
    ).astype(jnp.int8)
    return PileupLikelihoodResult(
        candidate_log_likelihood,
        jnp.sum(aligned, axis=-1, dtype=jnp.int32),
        mismatch_count,
        (mismatch_count > 0)
        | cigar_reference_mismatch
        | coordinate_failure
        | ~reference_found,
        valid,
        status,
        candidate_log_likelihood,
        _PILEUP_CONTRACT,
    )


def candidate_mapping_evidence(
    candidates: MappingCandidateBatch,
    pileup: PileupLikelihoodResult,
    pair_log_score: ArrayLike,
    null_log_likelihood: ArrayLike,
    null_log_prior: ArrayLike,
    plan: MappingExecutionPlan,
    /,
) -> MappingEvidenceResult:
    """Normalize supplied-candidate and explicit null evidence without MAPQ overclaim."""

    shape = candidates.reference_id.shape
    pair_score = jnp.asarray(pair_log_score, dtype=float)
    if pair_score.shape != shape:
        raise ValueError("pair_log_score must have candidate shape.")
    if pileup.candidate_log_likelihood.shape != shape:
        raise ValueError("pileup candidate shape must match candidates.")
    if shape[1] != plan.candidate_capacity:
        raise ValueError("Candidate capacity must match MappingExecutionPlan.")
    null_likelihood = jnp.broadcast_to(
        jnp.asarray(null_log_likelihood, dtype=float), shape[:1]
    )
    null_prior = jnp.broadcast_to(jnp.asarray(null_log_prior, dtype=float), shape[:1])
    candidate_input_valid = (
        candidates.candidate_mask & pileup.valid & jnp.isfinite(pair_score)
    )
    score = (
        pileup.candidate_log_likelihood + pair_score + candidates.candidate_log_prior
    ) / plan.score_temperature
    score = jnp.where(candidate_input_valid, score, -jnp.inf)
    has_candidate = jnp.any(candidate_input_valid, axis=-1)
    candidate_log_normalizer = logsumexp(score, axis=-1)
    conditional = jnp.where(
        has_candidate[:, None], score - candidate_log_normalizer[:, None], -jnp.inf
    )
    null_score = (null_likelihood + null_prior) / plan.score_temperature
    null_valid = jnp.isfinite(null_score)
    joint_normalizer = jnp.logaddexp(candidate_log_normalizer, null_score)
    candidate_log_probability = score - joint_normalizer[:, None]
    null_log_probability = null_score - joint_normalizer

    best_index_raw = jnp.argmax(score, axis=-1).astype(jnp.int32)
    best_candidate_score = jnp.max(score, axis=-1)
    mapped = has_candidate & (best_candidate_score > null_score)
    best_index = jnp.where(mapped, best_index_raw, -1).astype(jnp.int32)
    best_candidate_log_probability = jnp.take_along_axis(
        candidate_log_probability, best_index_raw[:, None], axis=-1
    )[:, 0]
    best_state_log_probability = jnp.where(
        mapped, best_candidate_log_probability, null_log_probability
    )
    error_probability = jnp.maximum(
        -jnp.expm1(jnp.minimum(best_state_log_probability, 0.0)),
        plan.probability_floor,
    )
    mapq = jnp.minimum(-10.0 * jnp.log10(error_probability), plan.mapq_cap)
    supplied_candidate_inputs_valid = jnp.all(
        (~candidates.candidate_mask) | (pileup.valid & jnp.isfinite(pair_score)),
        axis=-1,
    )
    overall_valid = candidates.valid & null_valid & supplied_candidate_inputs_valid
    mapq_calibrated = (
        jnp.full(shape[:1], plan.external_mapq_calibration, dtype=bool)
        & ~candidates.retrieval_truncated
        & overall_valid
    )
    mapping_quality_known = mapq_calibrated & (mapped | null_valid)
    status = jnp.where(
        ~overall_valid,
        MAPPING_STATUS_INVALID_INPUT,
        jnp.where(
            candidates.retrieval_truncated,
            MAPPING_STATUS_CANDIDATE_TRUNCATED,
            jnp.where(
                ~has_candidate,
                MAPPING_STATUS_NO_CANDIDATES,
                jnp.where(
                    ~mapq_calibrated,
                    MAPPING_STATUS_MAPQ_UNCALIBRATED,
                    MAPPING_STATUS_VALID,
                ),
            ),
        ),
    ).astype(jnp.int8)
    return MappingEvidenceResult(
        score,
        conditional,
        candidate_log_probability,
        null_log_probability,
        best_index,
        mapped,
        mapq,
        mapping_quality_known,
        mapq_calibrated,
        jnp.ones(shape[:1], dtype=bool),
        candidates.retrieval_truncated,
        jnp.any(pileup.reference_mismatch & candidates.candidate_mask, axis=-1),
        overall_valid,
        status,
        joint_normalizer,
        _MAPPING_CONTRACT,
    )
