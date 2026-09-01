#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..genomics._cigar import cigar_batch_from_tuples, CigarBatch
from ..genomics._mapping import MappingCandidateBatch
from ..genomics._reads import (
    ReadBatch,
    ReadLayout,
    ReadPairLayout,
    SAM_FLAG_FIRST_IN_PAIR,
    SAM_FLAG_MATE_REVERSE,
    SAM_FLAG_MATE_UNMAPPED,
    SAM_FLAG_PAIRED,
    SAM_FLAG_PROPER_PAIR,
    SAM_FLAG_REVERSE,
    SAM_FLAG_SECOND_IN_PAIR,
    SAM_FLAG_UNMAPPED,
)
from ..sequence import DNA_IUPAC, PhredEncoding, QualityBatch, SequenceBatch


@dataclass(frozen=True, slots=True)
class SAMLikeRecord:
    """Host-only SAM alignment data with normalized absence semantics.

    CIGAR tuples follow pysam/BAM order ``(operation_code, length)``. A mapping
    quality of ``None`` is unknown; numeric 255 is normalized to ``None`` by the
    pysam adapter. This host record is deliberately not a PyTree.
    """

    query_name: str | None
    query_sequence: str | None
    query_qualities: tuple[int, ...] | None
    flag: int
    reference_name: str | None
    reference_start: int | None
    mapping_quality: int | None
    cigar: tuple[tuple[int, int], ...] | None
    mate_reference_name: str | None = None
    mate_reference_start: int | None = None
    template_length: int = 0
    read_group: str | None = None
    umi: str | None = None
    reference_identity: str | None = None

    def __post_init__(self) -> None:
        flag = int(self.flag)
        if flag < 0 or flag > 0xFFFF:
            raise ValueError("SAM flag must fit an unsigned 16-bit integer.")
        if self.query_qualities is not None and self.query_sequence is not None:
            if len(self.query_qualities) != len(self.query_sequence):
                raise ValueError("SAM sequence and quality lengths must match.")
        if self.mapping_quality is not None and not 0 <= int(self.mapping_quality) <= 254:
            raise ValueError("Known SAM mapping quality must be between 0 and 254.")
        unmapped = bool(flag & SAM_FLAG_UNMAPPED)
        if unmapped and (
            self.reference_name is not None or self.reference_start is not None
        ):
            raise ValueError("An unmapped SAM-like record cannot name a mapped locus.")
        if not unmapped and (
            self.reference_name is None
            or self.reference_start is None
            or int(self.reference_start) < 0
        ):
            raise ValueError("A mapped SAM-like record requires a reference and start.")


def _optional_tag(record: Any, names: tuple[str, ...], /) -> str | None:
    for name in names:
        if record.has_tag(name):
            value = record.get_tag(name)
            return None if value is None else str(value)
    return None


def sam_like_from_pysam(
    record: Any,
    /,
    *,
    reference_identity: str | None = None,
    source_is_cram: bool = False,
) -> SAMLikeRecord:
    """Copy one pysam AlignedSegment into a dependency-free host record.

    pysam is intentionally neither imported nor retained here. CRAM-origin records
    require the caller's explicit reference identity rather than inferring identity
    from a filename or decoder state.
    """

    identity = None if reference_identity is None else str(reference_identity).strip()
    if source_is_cram and not identity:
        raise ValueError("CRAM conversion requires an explicit reference_identity.")
    flag = int(record.flag)
    mapping_quality = int(record.mapping_quality)
    if mapping_quality == 255 or flag & SAM_FLAG_UNMAPPED:
        mapping_quality_or_none: int | None = None
    else:
        mapping_quality_or_none = mapping_quality
    qualities = record.query_qualities
    cigar = record.cigartuples
    return SAMLikeRecord(
        query_name=None if record.query_name is None else str(record.query_name),
        query_sequence=(
            None if record.query_sequence is None else str(record.query_sequence)
        ),
        query_qualities=(
            None if qualities is None else tuple(int(value) for value in qualities)
        ),
        flag=flag,
        reference_name=(
            None if record.reference_name is None else str(record.reference_name)
        ),
        reference_start=(
            None
            if record.reference_start is None or int(record.reference_start) < 0
            else int(record.reference_start)
        ),
        mapping_quality=mapping_quality_or_none,
        cigar=(
            None
            if cigar is None
            else tuple((int(operation), int(length)) for operation, length in cigar)
        ),
        mate_reference_name=(
            None
            if record.next_reference_name is None
            else str(record.next_reference_name)
        ),
        mate_reference_start=(
            None
            if record.next_reference_start is None or int(record.next_reference_start) < 0
            else int(record.next_reference_start)
        ),
        template_length=int(record.template_length),
        read_group=_optional_tag(record, ("RG",)),
        umi=_optional_tag(record, ("RX", "UB", "UR")),
        reference_identity=identity,
    )


def sam_like_records_from_pysam(
    alignment_file: Any,
    /,
    *,
    reference_identity: str | None = None,
) -> tuple[SAMLikeRecord, ...]:
    """Materialize host records from an open pysam AlignmentFile."""

    source_is_cram = bool(alignment_file.is_cram)
    identity = None if reference_identity is None else str(reference_identity).strip()
    if source_is_cram and not identity:
        raise ValueError("Reading CRAM requires an explicit reference_identity.")
    return tuple(
        sam_like_from_pysam(
            record,
            reference_identity=identity,
            source_is_cram=source_is_cram,
        )
        for record in alignment_file.fetch(until_eof=True)
    )


def load_pysam_records(
    path: str | Path,
    /,
    *,
    reference_identity: str | None = None,
    reference_filename: str | Path | None = None,
) -> tuple[SAMLikeRecord, ...]:
    """Lazily import pysam and materialize SAM/BAM/CRAM as host records."""

    source = Path(path)
    is_cram = source.suffix.lower() == ".cram"
    identity = None if reference_identity is None else str(reference_identity).strip()
    if is_cram and not identity:
        raise ValueError("CRAM input requires an explicit reference_identity.")
    pysam = import_module("pysam")
    mode = "rc" if is_cram else "r"
    open_kwargs = (
        {}
        if reference_filename is None
        else {"reference_filename": str(reference_filename)}
    )
    with pysam.AlignmentFile(str(source), mode, **open_kwargs) as alignment_file:
        return sam_like_records_from_pysam(alignment_file, reference_identity=identity)


def _reference_span(record: SAMLikeRecord, /) -> int:
    if record.cigar is None:
        return 0
    reference_consuming = {0, 2, 3, 7, 8}
    return sum(
        length for operation, length in record.cigar if operation in reference_consuming
    )


def _pair_indices(records: Sequence[SAMLikeRecord], /) -> list[int]:
    indices = [-1] * len(records)
    by_name: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        if record.query_name is not None:
            by_name.setdefault(record.query_name, []).append(index)
    for index, record in enumerate(records):
        if not record.flag & SAM_FLAG_PAIRED or record.query_name is None:
            continue
        first = bool(record.flag & SAM_FLAG_FIRST_IN_PAIR)
        second = bool(record.flag & SAM_FLAG_SECOND_IN_PAIR)
        for candidate_index in by_name[record.query_name]:
            if candidate_index == index:
                continue
            candidate = records[candidate_index]
            complementary = (
                first and bool(candidate.flag & SAM_FLAG_SECOND_IN_PAIR)
            ) or (second and bool(candidate.flag & SAM_FLAG_FIRST_IN_PAIR))
            if complementary:
                indices[index] = candidate_index
                break
    return indices


def read_batch_from_sam_like(
    records: Sequence[SAMLikeRecord],
    layout: ReadLayout,
    /,
    *,
    reference_ids: Mapping[str, int],
    read_group_ids: Mapping[str, int] | None = None,
    record_ids: Sequence[int] | None = None,
    alphabet=DNA_IUPAC,
    phred_encoding: PhredEncoding = PhredEncoding.PHRED33,
) -> ReadBatch:
    """Lower host records into one capacity-preflighted array-only ReadBatch."""

    records_ = tuple(records)
    read_group_ids_ = {} if read_group_ids is None else read_group_ids
    if len(records_) > layout.max_reads:
        raise ValueError(
            f"Read capacity {layout.max_reads} is smaller than required {len(records_)}."
        )
    ids = tuple(range(len(records_))) if record_ids is None else tuple(record_ids)
    if len(ids) != len(records_):
        raise ValueError("record_ids must contain one integer per SAM-like record.")
    for record in records_:
        sequence = "" if record.query_sequence is None else record.query_sequence
        if len(sequence) > layout.max_read_length:
            raise ValueError(
                "A read exceeds max_read_length; no partial read was lowered."
            )
        cigar_count = 0 if record.cigar is None else len(record.cigar)
        if cigar_count > layout.max_cigar_ops:
            raise ValueError(
                "A CIGAR exceeds max_cigar_ops; no partial CIGAR was lowered."
            )
        umi_length = 0 if record.umi is None else len(record.umi)
        if umi_length > layout.max_umi_length:
            raise ValueError("A UMI exceeds max_umi_length; no partial UMI was lowered.")
        if (
            record.reference_name is not None
            and record.reference_name not in reference_ids
        ):
            raise ValueError("Every mapped reference name needs an explicit integer ID.")
        if record.read_group is not None and record.read_group not in read_group_ids_:
            raise ValueError("Every read-group tag needs an explicit integer ID.")

    pad_code = alphabet.code(alphabet.pad_symbol)
    sequence_codes = [
        [pad_code] * layout.max_read_length for _ in range(layout.max_reads)
    ]
    sequence_valid = [[False] * layout.max_read_length for _ in range(layout.max_reads)]
    sequence_soft = [[False] * layout.max_read_length for _ in range(layout.max_reads)]
    quality_scores = [[0] * layout.max_read_length for _ in range(layout.max_reads)]
    quality_valid = [[False] * layout.max_read_length for _ in range(layout.max_reads)]
    umi_codes = [[pad_code] * layout.max_umi_length for _ in range(layout.max_reads)]
    umi_valid = [[False] * layout.max_umi_length for _ in range(layout.max_reads)]
    flags = [0] * layout.max_reads
    reference_id = [-1] * layout.max_reads
    reference_start = [-1] * layout.max_reads
    mapping_quality = [0] * layout.max_reads
    mapping_quality_known = [False] * layout.max_reads
    mate_reference_id = [-1] * layout.max_reads
    mate_reference_start = [-1] * layout.max_reads
    template_length = [0] * layout.max_reads
    read_group_id = [-1] * layout.max_reads
    case_mask = [False] * layout.max_reads
    padded_record_ids = list(ids) + [-1] * (layout.max_reads - len(ids))

    for index, record in enumerate(records_):
        case_mask[index] = True
        sequence = "" if record.query_sequence is None else record.query_sequence
        for position, symbol in enumerate(sequence):
            sequence_codes[index][position] = alphabet.code(symbol)
            sequence_valid[index][position] = True
            sequence_soft[index][position] = symbol.islower()
        if record.query_qualities is not None:
            for position, quality in enumerate(record.query_qualities):
                quality_scores[index][position] = int(quality)
                quality_valid[index][position] = True
        if record.umi is not None:
            for position, symbol in enumerate(record.umi):
                umi_codes[index][position] = alphabet.code(symbol)
                umi_valid[index][position] = True
        flags[index] = int(record.flag)
        record_reference_name = record.reference_name
        if record_reference_name is not None:
            record_reference_start = record.reference_start
            if record_reference_start is None:
                raise ValueError("A mapped reference name requires a reference start.")
            reference_id[index] = int(reference_ids[record_reference_name])
            reference_start[index] = int(record_reference_start)
        if record.mapping_quality is not None:
            mapping_quality[index] = int(record.mapping_quality)
            mapping_quality_known[index] = True
        if record.mate_reference_name is not None:
            mate_reference_id[index] = int(reference_ids[record.mate_reference_name])
        if record.mate_reference_start is not None:
            mate_reference_start[index] = int(record.mate_reference_start)
        template_length[index] = int(record.template_length)
        if record.read_group is not None:
            read_group_id[index] = int(read_group_ids_[record.read_group])

    padded_cigars = [record.cigar for record in records_] + [None] * (
        layout.max_reads - len(records_)
    )
    cigar = cigar_batch_from_tuples(padded_cigars, layout.max_cigar_ops)
    mate_index = _pair_indices(records_) + [-1] * (layout.max_reads - len(records_))
    overlap_start = [-1] * layout.max_reads
    overlap_end = [-1] * layout.max_reads
    for index, mate in enumerate(mate_index[: len(records_)]):
        if mate < 0:
            continue
        record = records_[index]
        mate_record = records_[mate]
        if (
            record.reference_name is not None
            and record.reference_name == mate_record.reference_name
            and record.reference_start is not None
            and mate_record.reference_start is not None
        ):
            start = max(record.reference_start, mate_record.reference_start)
            end = min(
                record.reference_start + _reference_span(record),
                mate_record.reference_start + _reference_span(mate_record),
            )
            if end > start:
                overlap_start[index] = start
                overlap_end[index] = end
    pair = ReadPairLayout(
        mate_index,
        [(flag & SAM_FLAG_PAIRED) != 0 for flag in flags],
        [(flag & SAM_FLAG_PROPER_PAIR) != 0 for flag in flags],
        [(flag & SAM_FLAG_FIRST_IN_PAIR) != 0 for flag in flags],
        [(flag & SAM_FLAG_SECOND_IN_PAIR) != 0 for flag in flags],
        [(flag & SAM_FLAG_MATE_UNMAPPED) != 0 for flag in flags],
        [(flag & SAM_FLAG_MATE_REVERSE) != 0 for flag in flags],
        overlap_start,
        overlap_end,
    )
    sequence_batch = SequenceBatch(
        padded_record_ids,
        sequence_codes,
        sequence_valid,
        case_mask,
        sequence_soft,
        alphabet,
    )
    quality_batch = QualityBatch(
        padded_record_ids,
        quality_scores,
        quality_valid,
        case_mask,
        phred_encoding,
    )
    return ReadBatch(
        sequence_batch,
        quality_batch,
        cigar,
        pair,
        flags,
        reference_id,
        reference_start,
        mapping_quality,
        mapping_quality_known,
        mate_reference_id,
        mate_reference_start,
        template_length,
        read_group_id,
        umi_codes,
        umi_valid,
        layout,
    )


def mapping_candidates_from_sam_like(
    candidate_records: Sequence[Sequence[SAMLikeRecord]],
    candidate_capacity: int,
    cigar_operation_capacity: int,
    /,
    *,
    reference_ids: Mapping[str, int],
    candidate_log_prior: Sequence[Sequence[float]] | None = None,
) -> tuple[MappingCandidateBatch, CigarBatch]:
    """Lower heuristic SAM candidates and expose any bounded truncation."""

    rows = tuple(tuple(row) for row in candidate_records)
    capacity = int(candidate_capacity)
    if capacity < 1 or int(cigar_operation_capacity) < 1:
        raise ValueError("Candidate and CIGAR capacities must be positive.")
    prior_rows = (
        tuple(tuple(0.0 for _ in row) for row in rows)
        if candidate_log_prior is None
        else tuple(tuple(values) for values in candidate_log_prior)
    )
    if len(prior_rows) != len(rows) or any(
        len(priors) != len(row) for priors, row in zip(prior_rows, rows, strict=True)
    ):
        raise ValueError("candidate_log_prior must match the ragged candidate records.")
    reference_id: list[list[int]] = []
    reference_start: list[list[int]] = []
    reverse: list[list[bool]] = []
    mask: list[list[bool]] = []
    priors: list[list[float]] = []
    truncated: list[bool] = []
    flat_cigars: list[tuple[tuple[int, int], ...] | None] = []
    for row, prior_row in zip(rows, prior_rows, strict=True):
        retained = row[:capacity]
        retained_priors = prior_row[:capacity]
        truncated.append(len(row) > capacity)
        ids_row: list[int] = []
        starts_row: list[int] = []
        reverse_row: list[bool] = []
        mask_row: list[bool] = []
        priors_row: list[float] = []
        for record, prior in zip(retained, retained_priors, strict=True):
            if record.flag & SAM_FLAG_UNMAPPED:
                raise ValueError("An unmapped record is the null state, not a candidate.")
            candidate_reference_name = record.reference_name
            candidate_reference_start = record.reference_start
            if (
                candidate_reference_name is None
                or candidate_reference_start is None
                or candidate_reference_name not in reference_ids
            ):
                raise ValueError("Each candidate requires a mapped reference and start.")
            if record.cigar is not None and len(record.cigar) > cigar_operation_capacity:
                raise ValueError("A candidate CIGAR exceeds its operation capacity.")
            ids_row.append(int(reference_ids[candidate_reference_name]))
            starts_row.append(int(candidate_reference_start))
            reverse_row.append(bool(record.flag & SAM_FLAG_REVERSE))
            mask_row.append(True)
            priors_row.append(float(prior))
            flat_cigars.append(record.cigar)
        padding = capacity - len(retained)
        ids_row.extend([-1] * padding)
        starts_row.extend([-1] * padding)
        reverse_row.extend([False] * padding)
        mask_row.extend([False] * padding)
        priors_row.extend([0.0] * padding)
        flat_cigars.extend([None] * padding)
        reference_id.append(ids_row)
        reference_start.append(starts_row)
        reverse.append(reverse_row)
        mask.append(mask_row)
        priors.append(priors_row)
    candidates = MappingCandidateBatch(
        reference_id, reference_start, reverse, mask, priors, truncated
    )
    flat_cigar_batch = cigar_batch_from_tuples(flat_cigars, int(cigar_operation_capacity))
    cigar_shape = (len(rows), capacity, int(cigar_operation_capacity))
    candidate_cigar = CigarBatch(
        flat_cigar_batch.packed_ops.reshape(cigar_shape),
        flat_cigar_batch.op_count.reshape((len(rows), capacity)),
        source_valid=flat_cigar_batch.valid.reshape((len(rows), capacity)),
        source_status=flat_cigar_batch.status.reshape((len(rows), capacity)),
        evidence=flat_cigar_batch.evidence.reshape((len(rows), capacity)),
    )
    return candidates, candidate_cigar


__all__ = [
    "SAMLikeRecord",
    "load_pysam_records",
    "mapping_candidates_from_sam_like",
    "read_batch_from_sam_like",
    "sam_like_from_pysam",
    "sam_like_records_from_pysam",
]
