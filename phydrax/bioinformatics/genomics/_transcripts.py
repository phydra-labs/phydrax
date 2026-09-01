#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import core as jax_core, vmap
from jaxtyping import Array, ArrayLike

from phydrax._strict import StrictModule

from ..foundation import (
    BioinformaticsMethodContract,
    DifferentiationKind,
    ExecutionKind,
    MethodKind,
    OutputKind,
)
from ..sequence import (
    DNA_IUPAC,
    PROTEIN_IUPAC,
    SequenceBatch,
    STANDARD_GENETIC_CODE,
    translate,
    TranslationPlan,
    TranslationResult as SequenceTranslationResult,
)
from ._coordinates import LinearInterval, Strand
from ._reference import ReferenceGenome, ReferenceStatus


class TranscriptStatus(IntEnum):
    """Stable statuses for splicing, translation, and coordinate liftover."""

    SUCCESS = 0
    INVALID_MODEL = 1
    OUT_OF_REFERENCE = 2
    CAPACITY_EXCEEDED = 3
    PHASE_INCONSISTENT = 4
    AMBIGUOUS = 5
    PARTIAL_CODON = 6
    UNMAPPED = 7
    PARTIAL_MAPPING = 8
    INCOMPATIBLE_REFERENCE = 9


_SPLICING_CONTRACT = BioinformaticsMethodContract(
    "reference transcript splicing",
    MethodKind.EXACT_MODEL,
    ExecutionKind.EXACT_DISCRETE,
    DifferentiationKind.NONE,
    OutputKind.SEQUENCE,
    conditioning_statement="Exon intervals belong to one declared reference contig and are ordered 5′ to 3′.",
    truncation_statement="A transcript exceeding sequence capacity fails without returning a prefix.",
    capacity_semantics="Sequence and exon axes are fixed upper bounds with explicit masks.",
    assumptions=("Reference dictionary digests agree with the loaded bases.",),
    nondifferentiable_outputs=(
        "sequence",
        "exon_order",
        "exon_transcript_starts",
        "status",
        "evidence",
    ),
    input_dtype="int64",
    compute_dtype="int64/int32",
    output_dtype="int32-sequence-codes",
)

_CDS_CONTRACT = BioinformaticsMethodContract(
    "phased coding-sequence assembly",
    MethodKind.EXACT_MODEL,
    ExecutionKind.EXACT_DISCRETE,
    DifferentiationKind.NONE,
    OutputKind.SEQUENCE,
    conditioning_statement="CDS segments lie within transcript exons and GFF phases are strand-oriented.",
    truncation_statement="No segment is clipped; capacity and phase failures are observable.",
    capacity_semantics="The nucleotide sequence axis is a fixed upper bound.",
    assumptions=(
        "The first CDS phase denotes bases skipped before the first complete codon.",
    ),
    nondifferentiable_outputs=(
        "sequence",
        "phase_consistent",
        "lost_leading_bases",
        "status",
        "evidence",
    ),
    input_dtype="int64/int8",
    compute_dtype="int64/int32",
    output_dtype="int32-sequence-codes",
)

_TRANSLATION_CONTRACT = BioinformaticsMethodContract(
    "coding-sequence translation",
    MethodKind.EXACT_MODEL,
    ExecutionKind.EXACT_DISCRETE,
    DifferentiationKind.NONE,
    OutputKind.SEQUENCE,
    conditioning_statement="The input is an oriented CDS under a complete declared genetic code.",
    truncation_statement="Ambiguous codons, stops, and incomplete terminal codons are separately audited.",
    capacity_semantics="Protein capacity is derived from the bounded nucleotide capacity.",
    assumptions=("Phase handling has already oriented the CDS reading frame.",),
    nondifferentiable_outputs=("sequences", "report", "exact", "status", "evidence"),
    input_dtype="int32-sequence-codes",
    compute_dtype="int32",
    output_dtype="int32-protein-codes",
)

_LIFTOVER_CONTRACT = BioinformaticsMethodContract(
    "exon-aware coordinate liftover",
    MethodKind.EXACT_MODEL,
    ExecutionKind.EXACT_DISCRETE,
    DifferentiationKind.ALMOST_EVERYWHERE,
    OutputKind.STRUCTURED,
    conditioning_statement="Coordinates use zero-based half-open reference and transcript spaces.",
    truncation_statement="All mappings are retained or capacity failure is returned; no first-hit choice is made.",
    capacity_semantics="Route axes are fixed upper bounds and route_valid identifies exact mappings.",
    assumptions=("Transcript exon order is its biological 5′-to-3′ order.",),
    nondifferentiable_outputs=(
        "target_positions",
        "route_valid",
        "ambiguous",
        "lost",
        "status",
    ),
    input_dtype="int64",
    compute_dtype="int64",
    output_dtype="int64/bool",
)


def _concrete(value: Array) -> np.ndarray | None:
    if isinstance(value, jax_core.Tracer):
        return None
    return np.asarray(value)


def _scalar_integer(value: ArrayLike, name: str, *, dtype) -> Array:
    array = jnp.asarray(value)
    if array.shape != () or not jnp.issubdtype(array.dtype, jnp.integer):
        raise ValueError(f"{name} must be an integer scalar.")
    return array.astype(dtype)


def _segment_arrays(
    starts: ArrayLike,
    ends: ArrayLike,
    valid: ArrayLike,
    name: str,
) -> tuple[Array, Array, Array]:
    starts_ = jnp.asarray(starts)
    ends_ = jnp.asarray(ends)
    valid_ = jnp.asarray(valid)
    if starts_.ndim != 1 or ends_.shape != starts_.shape or valid_.shape != starts_.shape:
        raise ValueError(f"{name} starts, ends, and valid must be matching vectors.")
    if not jnp.issubdtype(starts_.dtype, jnp.integer) or not jnp.issubdtype(
        ends_.dtype, jnp.integer
    ):
        raise TypeError(f"{name} endpoints must be integer arrays.")
    if valid_.dtype != jnp.bool_:
        raise TypeError(f"{name} valid must be boolean.")
    starts_h = _concrete(starts_)
    ends_h = _concrete(ends_)
    valid_h = _concrete(valid_)
    if starts_h is not None and ends_h is not None and valid_h is not None:
        if np.any(valid_h & ((starts_h < 0) | (ends_h < starts_h))):
            raise ValueError(f"Valid {name} segments require 0 <= start <= end.")
    return starts_.astype(jnp.int64), ends_.astype(jnp.int64), valid_


def _default_order(starts: Array, ends: Array, valid: Array, strand: int) -> np.ndarray:
    starts_h = np.asarray(starts)
    ends_h = np.asarray(ends)
    valid_h = np.asarray(valid)
    indices = [index for index in range(starts.shape[0]) if bool(valid_h[index])]
    indices.sort(
        key=lambda index: (int(starts_h[index]), int(ends_h[index]), index),
        reverse=strand == int(Strand.REVERSE),
    )
    output = np.zeros((starts.shape[0],), dtype=np.int32)
    output[: len(indices)] = indices
    return output


def _validate_order(
    order: ArrayLike | None,
    starts: Array,
    ends: Array,
    valid: Array,
    strand: int,
    name: str,
) -> Array:
    resolved = (
        _default_order(starts, ends, valid, strand)
        if order is None
        else np.asarray(order)
    )
    if resolved.shape != starts.shape or not np.issubdtype(resolved.dtype, np.integer):
        raise ValueError(
            f"{name}_order must be an integer vector matching segment capacity."
        )
    valid_h = np.asarray(valid)
    count = int(valid_h.sum())
    expected = sorted(np.flatnonzero(valid_h).tolist())
    actual = sorted(int(value) for value in resolved[:count])
    if actual != expected:
        raise ValueError(
            f"The occupied {name}_order prefix must permute every valid segment once."
        )
    if np.any((resolved < 0) | (resolved >= starts.shape[0])):
        raise ValueError(f"{name}_order contains an out-of-bounds row.")
    return jnp.asarray(resolved, dtype=jnp.int32)


class TranscriptModel(StrictModule):
    """Fixed-capacity exon topology with explicit biological order."""

    transcript_index: Array
    reference_index: Array
    exon_starts: Array
    exon_ends: Array
    exon_valid: Array
    exon_order: Array
    strand: Array
    reference_length: Array
    circular: Array

    def __init__(
        self,
        transcript_index: ArrayLike,
        reference_index: ArrayLike,
        exon_starts: ArrayLike,
        exon_ends: ArrayLike,
        exon_valid: ArrayLike,
        /,
        *,
        strand: Strand | int,
        exon_order: ArrayLike | None = None,
        reference_length: ArrayLike = 0,
        circular: bool = False,
    ):
        strand_ = int(strand)
        if strand_ not in (-1, 1):
            raise ValueError("Transcript strand must be forward or reverse.")
        starts, ends, valid = _segment_arrays(exon_starts, exon_ends, exon_valid, "exon")
        reference_length_ = _scalar_integer(
            reference_length, "reference_length", dtype=jnp.int64
        )
        reference_length_host = int(np.asarray(reference_length_))
        circular_ = bool(circular)
        if reference_length_host < 0 or (circular_ and reference_length_host == 0):
            raise ValueError("Circular transcripts require a positive reference_length.")
        if circular_ and np.any(
            np.asarray(valid) & (np.asarray(ends - starts) > reference_length_host)
        ):
            raise ValueError(
                "A circular exon cannot traverse the reference more than once."
            )
        self.transcript_index = _scalar_integer(
            transcript_index, "transcript_index", dtype=jnp.int32
        )
        self.reference_index = _scalar_integer(
            reference_index, "reference_index", dtype=jnp.int32
        )
        if (
            int(np.asarray(self.transcript_index)) < 0
            or int(np.asarray(self.reference_index)) < 0
        ):
            raise ValueError("Transcript and reference indices must be non-negative.")
        self.exon_starts = starts
        self.exon_ends = ends
        self.exon_valid = valid
        self.exon_order = _validate_order(
            exon_order, starts, ends, valid, strand_, "exon"
        )
        self.strand = jnp.asarray(strand_, dtype=jnp.int8)
        self.reference_length = reference_length_
        self.circular = jnp.asarray(circular_)

    @property
    def exon_capacity(self) -> int:
        return int(self.exon_starts.shape[0])

    @property
    def exon_count(self) -> Array:
        return jnp.sum(self.exon_valid, dtype=jnp.int32)

    @property
    def length(self) -> Array:
        return jnp.sum(jnp.where(self.exon_valid, self.exon_ends - self.exon_starts, 0))


class CDSModel(StrictModule):
    """Phased CDS segments attached to one transcript topology."""

    transcript: TranscriptModel
    segment_starts: Array
    segment_ends: Array
    segment_valid: Array
    segment_order: Array
    phases: Array

    def __init__(
        self,
        transcript: TranscriptModel,
        segment_starts: ArrayLike,
        segment_ends: ArrayLike,
        segment_valid: ArrayLike,
        phases: ArrayLike,
        /,
        *,
        segment_order: ArrayLike | None = None,
    ):
        if not isinstance(transcript, TranscriptModel):
            raise TypeError("transcript must be a TranscriptModel.")
        starts, ends, valid = _segment_arrays(
            segment_starts, segment_ends, segment_valid, "CDS"
        )
        phases_ = jnp.asarray(phases)
        if phases_.shape != starts.shape or not jnp.issubdtype(
            phases_.dtype, jnp.integer
        ):
            raise ValueError("phases must be an integer vector matching CDS segments.")
        phases_h = _concrete(phases_)
        valid_h = _concrete(valid)
        if phases_h is not None and valid_h is not None:
            if np.any(valid_h & ~np.isin(phases_h, (0, 1, 2))):
                raise ValueError("Every valid CDS segment phase must be 0, 1, or 2.")
        strand_ = int(np.asarray(transcript.strand))
        self.transcript = transcript
        self.segment_starts = starts
        self.segment_ends = ends
        self.segment_valid = valid
        self.segment_order = _validate_order(
            segment_order, starts, ends, valid, strand_, "segment"
        )
        self.phases = phases_.astype(jnp.int8)

    @property
    def segment_capacity(self) -> int:
        return int(self.segment_starts.shape[0])

    @property
    def segment_count(self) -> Array:
        return jnp.sum(self.segment_valid, dtype=jnp.int32)


class TranscriptSplicingResult(StrictModule):
    sequence: SequenceBatch
    exon_transcript_starts: Array
    exon_transcript_ends: Array
    exon_order: Array
    exon_valid: Array
    valid: Array
    status: Array
    evidence: Array
    method_contract: BioinformaticsMethodContract = eqx.field(static=True)


class CDSAssemblyResult(StrictModule):
    sequence: SequenceBatch
    phase_consistent: Array
    lost_leading_bases: Array
    incomplete_terminal_bases: Array
    valid: Array
    status: Array
    evidence: Array
    method_contract: BioinformaticsMethodContract = eqx.field(static=True)


class TranscriptTranslationResult(StrictModule):
    translation: SequenceTranslationResult
    exact: Array
    valid: Array
    status: Array
    evidence: Array
    method_contract: BioinformaticsMethodContract = eqx.field(static=True)


class CoordinateLiftoverResult(StrictModule):
    source_positions: Array
    target_positions: Array
    route_valid: Array
    ambiguous: Array
    lost: Array
    valid: Array
    status: Array
    evidence: Array
    method_contract: BioinformaticsMethodContract = eqx.field(static=True)


class IntervalLiftoverResult(StrictModule):
    source_starts: Array
    source_ends: Array
    target_starts: Array
    target_ends: Array
    fragment_valid: Array
    reversed: Array
    ambiguous_bases: Array
    lost_bases: Array
    valid: Array
    status: Array
    evidence: Array
    method_contract: BioinformaticsMethodContract = eqx.field(static=True)


def _empty_sequence(record_id: int, capacity: int, alphabet=DNA_IUPAC) -> SequenceBatch:
    pad = alphabet.code(alphabet.pad_symbol)
    return SequenceBatch(
        jnp.asarray([record_id], dtype=jnp.int32),
        jnp.full((1, capacity), pad, dtype=jnp.int32),
        jnp.zeros((1, capacity), dtype=bool),
        jnp.asarray([True]),
        jnp.zeros((1, capacity), dtype=bool),
        alphabet,
    )


def _complement_codes(codes: np.ndarray) -> np.ndarray:
    mapping = DNA_IUPAC.symbol_to_code
    complement = DNA_IUPAC.complement_map
    lookup = np.asarray(
        [mapping[complement[symbol]] for symbol in DNA_IUPAC.symbols], dtype=np.int32
    )
    return lookup[codes[::-1]]


def _fetch_oriented_segment(
    genome: ReferenceGenome,
    reference_index: int,
    start: int,
    end: int,
    reverse: bool,
) -> tuple[np.ndarray | None, int]:
    span = end - start
    result = genome.fetch_window(reference_index, start, end, capacity=span)
    if not bool(np.asarray(result.valid)):
        return None, int(np.asarray(result.status))
    codes = np.asarray(result.window.sequence.token_codes[0, :span], dtype=np.int32)
    return (_complement_codes(codes) if reverse else codes), int(ReferenceStatus.SUCCESS)


def _model_segments(
    starts: Array,
    ends: Array,
    valid: Array,
    order: Array,
) -> tuple[tuple[int, int, int], ...]:
    starts_h = np.asarray(starts)
    ends_h = np.asarray(ends)
    valid_h = np.asarray(valid)
    order_h = np.asarray(order)
    count = int(valid_h.sum())
    return tuple(
        (int(row), int(starts_h[row]), int(ends_h[row])) for row in order_h[:count]
    )


def splice_transcript(
    genome: ReferenceGenome,
    transcript: TranscriptModel,
    /,
    *,
    capacity: int,
) -> TranscriptSplicingResult:
    """Splice all exons exactly in biological order, reverse-complementing as needed."""

    if not isinstance(genome, ReferenceGenome):
        raise TypeError("genome must be a ReferenceGenome.")
    if not isinstance(transcript, TranscriptModel):
        raise TypeError("transcript must be a TranscriptModel.")
    capacity_ = int(capacity)
    if capacity_ < 0:
        raise ValueError("capacity must be non-negative.")
    transcript_id = int(np.asarray(transcript.transcript_index))
    reference_id = int(np.asarray(transcript.reference_index))
    reverse = int(np.asarray(transcript.strand)) == int(Strand.REVERSE)
    segments = _model_segments(
        transcript.exon_starts,
        transcript.exon_ends,
        transcript.exon_valid,
        transcript.exon_order,
    )
    total_length = sum(end - start for _, start, end in segments)
    zero_length = sum(end == start for _, start, end in segments)
    duplicate_count = len(segments) - len({(start, end) for _, start, end in segments})
    starts_out = np.zeros((transcript.exon_capacity,), dtype=np.int32)
    ends_out = np.zeros((transcript.exon_capacity,), dtype=np.int32)
    valid_out = np.asarray(transcript.exon_valid, dtype=bool).copy()
    evidence = jnp.asarray(
        [len(segments), total_length, zero_length, duplicate_count, capacity_],
        dtype=jnp.int64,
    )
    if total_length > capacity_:
        return TranscriptSplicingResult(
            _empty_sequence(transcript_id, capacity_),
            jnp.asarray(starts_out),
            jnp.asarray(ends_out),
            transcript.exon_order,
            jnp.asarray(valid_out),
            jnp.asarray(False),
            jnp.asarray(int(TranscriptStatus.CAPACITY_EXCEEDED), dtype=jnp.int32),
            evidence,
            _SPLICING_CONTRACT,
        )

    pieces: list[np.ndarray] = []
    offset = 0
    status = TranscriptStatus.SUCCESS
    for row, start, end in segments:
        starts_out[row] = offset
        ends_out[row] = offset + end - start
        codes, fetch_status = _fetch_oriented_segment(
            genome, reference_id, start, end, reverse
        )
        if codes is None:
            status = TranscriptStatus.OUT_OF_REFERENCE
            valid_out[:] = False
            pieces = []
            break
        pieces.append(codes)
        offset += end - start
    if status is not TranscriptStatus.SUCCESS:
        sequence = _empty_sequence(transcript_id, capacity_)
    else:
        sequence = _empty_sequence(transcript_id, capacity_)
        payload = np.concatenate(pieces) if pieces else np.empty((0,), dtype=np.int32)
        tokens = np.asarray(sequence.token_codes).copy()
        valid = np.asarray(sequence.valid_mask).copy()
        tokens[0, :total_length] = payload
        valid[0, :total_length] = True
        sequence = SequenceBatch(
            sequence.record_ids,
            jnp.asarray(tokens),
            jnp.asarray(valid),
            sequence.case_mask,
            sequence.soft_mask,
            DNA_IUPAC,
        )
    return TranscriptSplicingResult(
        sequence,
        jnp.asarray(starts_out),
        jnp.asarray(ends_out),
        transcript.exon_order,
        jnp.asarray(valid_out),
        jnp.asarray(status is TranscriptStatus.SUCCESS),
        jnp.asarray(int(status), dtype=jnp.int32),
        evidence,
        _SPLICING_CONTRACT,
    )


def _segment_is_exonic(transcript: TranscriptModel, start: int, end: int) -> bool:
    starts = np.asarray(transcript.exon_starts)
    ends = np.asarray(transcript.exon_ends)
    valid = np.asarray(transcript.exon_valid)
    return any(
        bool(valid[index]) and int(starts[index]) <= start and end <= int(ends[index])
        for index in range(transcript.exon_capacity)
    )


def assemble_cds(
    genome: ReferenceGenome,
    cds: CDSModel,
    /,
    *,
    capacity: int,
) -> CDSAssemblyResult:
    """Assemble an oriented CDS and audit every declared GFF phase."""

    if not isinstance(genome, ReferenceGenome):
        raise TypeError("genome must be a ReferenceGenome.")
    if not isinstance(cds, CDSModel):
        raise TypeError("cds must be a CDSModel.")
    capacity_ = int(capacity)
    if capacity_ < 0:
        raise ValueError("capacity must be non-negative.")
    transcript = cds.transcript
    transcript_id = int(np.asarray(transcript.transcript_index))
    reference_id = int(np.asarray(transcript.reference_index))
    reverse = int(np.asarray(transcript.strand)) == int(Strand.REVERSE)
    segments = _model_segments(
        cds.segment_starts, cds.segment_ends, cds.segment_valid, cds.segment_order
    )
    phases = np.asarray(cds.phases)
    raw_length = sum(end - start for _, start, end in segments)
    first_phase = int(phases[segments[0][0]]) if segments else 0
    output_length = max(raw_length - first_phase, 0)
    exonic = all(_segment_is_exonic(transcript, start, end) for _, start, end in segments)
    phase_errors = 0
    cumulative = 0
    for index, (row, start, end) in enumerate(segments):
        phase = int(phases[row])
        expected = first_phase if index == 0 else (3 - (cumulative % 3)) % 3
        phase_errors += phase != expected
        cumulative += end - start - (first_phase if index == 0 else 0)
    phase_consistent = (
        bool(segments)
        and exonic
        and phase_errors == 0
        and first_phase <= (segments[0][2] - segments[0][1])
    )
    duplicate_count = len(segments) - len({(start, end) for _, start, end in segments})
    incomplete = output_length % 3
    evidence = jnp.asarray(
        [
            len(segments),
            raw_length,
            first_phase,
            incomplete,
            phase_errors,
            duplicate_count,
            int(exonic),
            capacity_,
        ],
        dtype=jnp.int64,
    )
    if output_length > capacity_:
        status = TranscriptStatus.CAPACITY_EXCEEDED
        sequence = _empty_sequence(transcript_id, capacity_)
        valid = False
    else:
        pieces: list[np.ndarray] = []
        fetch_ok = True
        for _, start, end in segments:
            codes, _ = _fetch_oriented_segment(genome, reference_id, start, end, reverse)
            if codes is None:
                fetch_ok = False
                pieces = []
                break
            pieces.append(codes)
        if not fetch_ok:
            status = TranscriptStatus.OUT_OF_REFERENCE
            sequence = _empty_sequence(transcript_id, capacity_)
            valid = False
        else:
            raw = np.concatenate(pieces) if pieces else np.empty((0,), dtype=np.int32)
            payload = raw[first_phase:]
            base = _empty_sequence(transcript_id, capacity_)
            tokens = np.asarray(base.token_codes).copy()
            valid_mask = np.asarray(base.valid_mask).copy()
            tokens[0, :output_length] = payload
            valid_mask[0, :output_length] = True
            sequence = SequenceBatch(
                base.record_ids,
                jnp.asarray(tokens),
                jnp.asarray(valid_mask),
                base.case_mask,
                base.soft_mask,
                DNA_IUPAC,
            )
            status = (
                TranscriptStatus.SUCCESS
                if phase_consistent
                else TranscriptStatus.PHASE_INCONSISTENT
            )
            valid = phase_consistent
    return CDSAssemblyResult(
        sequence,
        jnp.asarray(phase_consistent),
        jnp.asarray(first_phase, dtype=jnp.int32),
        jnp.asarray(incomplete, dtype=jnp.int32),
        jnp.asarray(valid),
        jnp.asarray(int(status), dtype=jnp.int32),
        evidence,
        _CDS_CONTRACT,
    )


def translate_cds(
    assembly: CDSAssemblyResult,
    /,
    *,
    plan: TranslationPlan | None = None,
) -> TranscriptTranslationResult:
    """Translate an oriented CDS while exposing ambiguity and incomplete-codon loss."""

    if not isinstance(assembly, CDSAssemblyResult):
        raise TypeError("assembly must be a CDSAssemblyResult.")
    plan_ = (
        TranslationPlan(
            frame=0,
            strand="forward",
            ambiguous_policy="consensus",
            incomplete_policy="drop",
            stop_policy="keep",
            genetic_code=STANDARD_GENETIC_CODE,
            output_alphabet=PROTEIN_IUPAC,
        )
        if plan is None
        else plan
    )
    if not isinstance(plan_, TranslationPlan):
        raise TypeError("plan must be a TranslationPlan.")
    translated = translate(assembly.sequence, plan_)
    ambiguous = translated.report.ambiguous_codon_counts[0]
    incomplete = translated.report.incomplete_base_counts[0]
    stops = translated.report.stop_codon_counts[0]
    assembly_valid = assembly.valid
    exact = assembly_valid & (ambiguous == 0) & (incomplete == 0)
    valid = assembly_valid
    status = jnp.where(
        ~assembly_valid,
        assembly.status,
        jnp.where(
            ambiguous > 0,
            int(TranscriptStatus.AMBIGUOUS),
            jnp.where(
                incomplete > 0,
                int(TranscriptStatus.PARTIAL_CODON),
                int(TranscriptStatus.SUCCESS),
            ),
        ),
    ).astype(jnp.int32)
    evidence = jnp.asarray(
        [
            assembly.sequence.lengths[0],
            translated.report.output_lengths[0],
            ambiguous,
            incomplete,
            stops,
        ],
        dtype=jnp.int32,
    )
    return TranscriptTranslationResult(
        translated,
        exact,
        valid,
        status,
        evidence,
        _TRANSLATION_CONTRACT,
    )


def _exon_offsets(transcript: TranscriptModel) -> Array:
    offsets = jnp.zeros((transcript.exon_capacity,), dtype=jnp.int64)
    cumulative = jnp.asarray(0, dtype=jnp.int64)
    for slot in range(transcript.exon_capacity):
        row = transcript.exon_order[slot]
        occupied = slot < transcript.exon_count
        offsets = offsets.at[row].set(jnp.where(occupied, cumulative, offsets[row]))
        length = transcript.exon_ends[row] - transcript.exon_starts[row]
        cumulative = cumulative + jnp.where(occupied, length, 0)
    return offsets


def genomic_to_transcript(
    transcript: TranscriptModel,
    positions: ArrayLike,
    /,
) -> CoordinateLiftoverResult:
    """Map each genomic position to all transcript occurrences, never choosing one hit."""

    if not isinstance(transcript, TranscriptModel):
        raise TypeError("transcript must be a TranscriptModel.")
    source = jnp.asarray(positions)
    if not jnp.issubdtype(source.dtype, jnp.integer):
        raise TypeError("positions must have an integer dtype.")
    source = source.astype(jnp.int64)
    expanded = source[..., None]
    lengths = transcript.exon_ends - transcript.exon_starts
    linear_local = expanded - transcript.exon_starts
    circular_local = jnp.mod(
        linear_local,
        jnp.maximum(transcript.reference_length, jnp.asarray(1, dtype=jnp.int64)),
    )
    local = jnp.where(transcript.circular, circular_local, linear_local)
    source_in_reference = (source >= 0) & (
        ~transcript.circular | (source < transcript.reference_length)
    )
    within = (
        transcript.exon_valid
        & source_in_reference[..., None]
        & (local >= 0)
        & (local < lengths)
    )
    offsets = _exon_offsets(transcript)
    forward = offsets + local
    reverse = offsets + lengths - 1 - local
    target = jnp.where(transcript.strand == int(Strand.REVERSE), reverse, forward)
    target = jnp.where(within, target, 0).astype(jnp.int64)
    counts = jnp.sum(within, axis=-1, dtype=jnp.int32)
    ambiguous = counts > 1
    lost = counts == 0
    valid = ~lost
    status = jnp.where(
        lost,
        int(TranscriptStatus.UNMAPPED),
        jnp.where(
            ambiguous, int(TranscriptStatus.AMBIGUOUS), int(TranscriptStatus.SUCCESS)
        ),
    ).astype(jnp.int32)
    evidence = jnp.stack(
        (counts, ambiguous.astype(jnp.int32), lost.astype(jnp.int32)), axis=-1
    )
    return CoordinateLiftoverResult(
        source,
        target,
        within,
        ambiguous,
        lost,
        valid,
        status,
        evidence,
        _LIFTOVER_CONTRACT,
    )


def transcript_to_genomic(
    transcript: TranscriptModel,
    positions: ArrayLike,
    /,
) -> CoordinateLiftoverResult:
    """Map transcript positions to genomic positions with an explicit route axis."""

    if not isinstance(transcript, TranscriptModel):
        raise TypeError("transcript must be a TranscriptModel.")
    source = jnp.asarray(positions)
    if not jnp.issubdtype(source.dtype, jnp.integer):
        raise TypeError("positions must have an integer dtype.")
    source = source.astype(jnp.int64)
    offsets = _exon_offsets(transcript)
    lengths = transcript.exon_ends - transcript.exon_starts
    expanded = source[..., None]
    within = (
        transcript.exon_valid & (expanded >= offsets) & (expanded < offsets + lengths)
    )
    local = expanded - offsets
    forward = transcript.exon_starts + local
    reverse = transcript.exon_ends - 1 - local
    raw_target = jnp.where(transcript.strand == int(Strand.REVERSE), reverse, forward)
    circular_target = jnp.mod(
        raw_target,
        jnp.maximum(transcript.reference_length, jnp.asarray(1, dtype=jnp.int64)),
    )
    target = jnp.where(transcript.circular, circular_target, raw_target)
    target = jnp.where(within, target, 0).astype(jnp.int64)
    counts = jnp.sum(within, axis=-1, dtype=jnp.int32)
    ambiguous = counts > 1
    lost = counts == 0
    valid = ~lost
    status = jnp.where(
        lost,
        int(TranscriptStatus.UNMAPPED),
        jnp.where(
            ambiguous, int(TranscriptStatus.AMBIGUOUS), int(TranscriptStatus.SUCCESS)
        ),
    ).astype(jnp.int32)
    evidence = jnp.stack(
        (counts, ambiguous.astype(jnp.int32), lost.astype(jnp.int32)), axis=-1
    )
    return CoordinateLiftoverResult(
        source,
        target,
        within,
        ambiguous,
        lost,
        valid,
        status,
        evidence,
        _LIFTOVER_CONTRACT,
    )


def liftover_transcript_coordinates(
    source_transcript: TranscriptModel,
    target_transcript: TranscriptModel,
    positions: ArrayLike,
    /,
    *,
    capacity: int,
) -> CoordinateLiftoverResult:
    """Lift transcript positions through genomic space, preserving one-to-many mappings."""

    if not isinstance(source_transcript, TranscriptModel) or not isinstance(
        target_transcript, TranscriptModel
    ):
        raise TypeError(
            "source_transcript and target_transcript must be TranscriptModel values."
        )
    capacity_ = int(capacity)
    if capacity_ < 0:
        raise ValueError("capacity must be non-negative.")
    incompatible = source_transcript.reference_index != target_transcript.reference_index
    genomic = transcript_to_genomic(source_transcript, positions)
    source_shape = genomic.source_positions.shape
    genomic_values = genomic.target_positions.reshape(source_shape + (-1,))
    genomic_routes = genomic.route_valid.reshape(source_shape + (-1,))
    target = genomic_to_transcript(target_transcript, genomic_values)
    candidates = target.target_positions.reshape(source_shape + (-1,))
    candidate_valid = (genomic_routes[..., :, None] & target.route_valid).reshape(
        source_shape + (-1,)
    )
    counts = jnp.sum(candidate_valid, axis=-1, dtype=jnp.int32)
    overflow = counts > capacity_
    if candidate_valid.shape[-1] == 0:
        selected_targets = jnp.zeros(source_shape + (capacity_,), dtype=jnp.int64)
    else:
        flat_masks = candidate_valid.reshape((-1, candidate_valid.shape[-1]))
        selected = vmap(
            lambda mask: jnp.nonzero(mask, size=capacity_, fill_value=0)[0]
        )(flat_masks).reshape(source_shape + (capacity_,))
        selected_targets = jnp.take_along_axis(candidates, selected, axis=-1)
    slot_valid = jnp.arange(capacity_, dtype=jnp.int32) < counts[..., None]
    route_valid = slot_valid & ~overflow[..., None] & ~incompatible
    target_positions = jnp.where(route_valid, selected_targets, 0)
    ambiguous = (counts > 1) & ~overflow & ~incompatible
    lost = (counts == 0) | overflow | incompatible
    valid = ~lost
    status = jnp.where(
        incompatible,
        int(TranscriptStatus.INCOMPATIBLE_REFERENCE),
        jnp.where(
            overflow,
            int(TranscriptStatus.CAPACITY_EXCEEDED),
            jnp.where(
                counts == 0,
                int(TranscriptStatus.UNMAPPED),
                jnp.where(
                    counts > 1,
                    int(TranscriptStatus.AMBIGUOUS),
                    int(TranscriptStatus.SUCCESS),
                ),
            ),
        ),
    ).astype(jnp.int32)
    evidence = jnp.stack(
        (
            counts,
            overflow.astype(jnp.int32),
            ambiguous.astype(jnp.int32),
            lost.astype(jnp.int32),
        ),
        axis=-1,
    )
    return CoordinateLiftoverResult(
        jnp.asarray(positions, dtype=jnp.int64),
        target_positions,
        route_valid,
        ambiguous,
        lost,
        valid,
        status,
        evidence,
        _LIFTOVER_CONTRACT,
    )


def _union_coverage(intervals: list[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    ordered = sorted(intervals)
    start, end = ordered[0]
    coverage = 0
    for next_start, next_end in ordered[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            coverage += end - start
            start, end = next_start, next_end
    return coverage + end - start


def liftover_interval_to_transcript(
    transcript: TranscriptModel,
    interval: LinearInterval,
    /,
    *,
    capacity: int | None = None,
) -> IntervalLiftoverResult:
    """Split a genomic interval across exons with exact loss and duplicate coverage."""

    if not isinstance(transcript, TranscriptModel):
        raise TypeError("transcript must be a TranscriptModel.")
    if not isinstance(interval, LinearInterval):
        raise TypeError("interval must be a LinearInterval.")
    output_capacity = transcript.exon_capacity if capacity is None else int(capacity)
    if output_capacity < 0:
        raise ValueError("capacity must be non-negative.")
    same_reference = int(np.asarray(interval.reference_index)) == int(
        np.asarray(transcript.reference_index)
    )
    starts = np.asarray(transcript.exon_starts)
    ends = np.asarray(transcript.exon_ends)
    valid_exons = np.asarray(transcript.exon_valid)
    offsets = np.asarray(_exon_offsets(transcript))
    query_start = int(np.asarray(interval.start))
    query_end = int(np.asarray(interval.end))
    reverse = int(np.asarray(transcript.strand)) == int(Strand.REVERSE)
    fragments: list[tuple[int, int, int, int]] = []
    if same_reference:
        for row in range(transcript.exon_capacity):
            if not bool(valid_exons[row]):
                continue
            start = max(query_start, int(starts[row]))
            end = min(query_end, int(ends[row]))
            if end <= start:
                continue
            if reverse:
                target_start = int(offsets[row]) + int(ends[row]) - end
                target_end = int(offsets[row]) + int(ends[row]) - start
            else:
                target_start = int(offsets[row]) + start - int(starts[row])
                target_end = int(offsets[row]) + end - int(starts[row])
            fragments.append((start, end, target_start, target_end))
    overflow = len(fragments) > output_capacity
    source_starts = np.zeros((output_capacity,), dtype=np.int64)
    source_ends = np.zeros((output_capacity,), dtype=np.int64)
    target_starts = np.zeros((output_capacity,), dtype=np.int64)
    target_ends = np.zeros((output_capacity,), dtype=np.int64)
    fragment_valid = np.zeros((output_capacity,), dtype=bool)
    if not overflow:
        for index, fragment in enumerate(fragments):
            (
                source_starts[index],
                source_ends[index],
                target_starts[index],
                target_ends[index],
            ) = fragment
            fragment_valid[index] = True
    covered_union = _union_coverage([(start, end) for start, end, _, _ in fragments])
    mapped_sum = sum(end - start for start, end, _, _ in fragments)
    query_length = query_end - query_start
    lost_bases = max(query_length - covered_union, 0)
    ambiguous_bases = max(mapped_sum - covered_union, 0)
    valid = same_reference and not overflow and (query_length == 0 or bool(fragments))
    if not same_reference:
        status = TranscriptStatus.INCOMPATIBLE_REFERENCE
    elif overflow:
        status = TranscriptStatus.CAPACITY_EXCEEDED
    elif not fragments and query_length > 0:
        status = TranscriptStatus.UNMAPPED
    elif lost_bases > 0:
        status = TranscriptStatus.PARTIAL_MAPPING
    elif ambiguous_bases > 0:
        status = TranscriptStatus.AMBIGUOUS
    else:
        status = TranscriptStatus.SUCCESS
    evidence = jnp.asarray(
        [
            query_length,
            mapped_sum,
            covered_union,
            lost_bases,
            ambiguous_bases,
            len(fragments),
            output_capacity,
        ],
        dtype=jnp.int64,
    )
    return IntervalLiftoverResult(
        jnp.asarray(source_starts),
        jnp.asarray(source_ends),
        jnp.asarray(target_starts),
        jnp.asarray(target_ends),
        jnp.asarray(fragment_valid),
        jnp.asarray(reverse),
        jnp.asarray(ambiguous_bases, dtype=jnp.int64),
        jnp.asarray(lost_bases, dtype=jnp.int64),
        jnp.asarray(valid),
        jnp.asarray(int(status), dtype=jnp.int32),
        evidence,
        _LIFTOVER_CONTRACT,
    )


__all__ = [
    "CDSAssemblyResult",
    "CDSModel",
    "CoordinateLiftoverResult",
    "IntervalLiftoverResult",
    "TranscriptModel",
    "TranscriptSplicingResult",
    "TranscriptStatus",
    "TranscriptTranslationResult",
    "assemble_cds",
    "genomic_to_transcript",
    "liftover_interval_to_transcript",
    "liftover_transcript_coordinates",
    "splice_transcript",
    "transcript_to_genomic",
    "translate_cds",
]
