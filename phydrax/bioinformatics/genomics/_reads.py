#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..sequence import QualityBatch, SequenceBatch
from ._cigar import CigarBatch


SAM_FLAG_PAIRED = 0x1
SAM_FLAG_PROPER_PAIR = 0x2
SAM_FLAG_UNMAPPED = 0x4
SAM_FLAG_MATE_UNMAPPED = 0x8
SAM_FLAG_REVERSE = 0x10
SAM_FLAG_MATE_REVERSE = 0x20
SAM_FLAG_FIRST_IN_PAIR = 0x40
SAM_FLAG_SECOND_IN_PAIR = 0x80
SAM_FLAG_SECONDARY = 0x100
SAM_FLAG_QC_FAIL = 0x200
SAM_FLAG_DUPLICATE = 0x400
SAM_FLAG_SUPPLEMENTARY = 0x800

READ_STATUS_VALID = 0
READ_STATUS_INVALID_CIGAR = 1
READ_STATUS_SEQUENCE_CIGAR_LENGTH_MISMATCH = 2
READ_STATUS_INVALID_MAPPED_COORDINATE = 3
READ_STATUS_INVALID_UNMAPPED_STATE = 4
READ_STATUS_INVALID_PAIR = 5
READ_STATUS_INVALID_QUALITY = 6
READ_STATUS_INVALID_UMI = 7
READ_STATUS_INVALID_MAPQ = 8
READ_STATUS_IDENTITY_MISMATCH = 9


class ReadLayout(StrictModule):
    """Static capacities of one compiled read-evidence bucket."""

    max_reads: int = eqx.field(static=True)
    max_read_length: int = eqx.field(static=True)
    max_cigar_ops: int = eqx.field(static=True)
    max_umi_length: int = eqx.field(static=True)
    fingerprint: str = eqx.field(static=True)

    def __init__(
        self,
        max_reads: int,
        max_read_length: int,
        max_cigar_ops: int,
        max_umi_length: int = 0,
        /,
    ):
        values = tuple(
            int(value)
            for value in (
                max_reads,
                max_read_length,
                max_cigar_ops,
                max_umi_length,
            )
        )
        if values[0] < 1 or values[1] < 1 or values[2] < 1 or values[3] < 0:
            raise ValueError(
                "Read capacities must be positive except max_umi_length, which may be zero."
            )
        (
            self.max_reads,
            self.max_read_length,
            self.max_cigar_ops,
            self.max_umi_length,
        ) = values
        self.fingerprint = canonical_fingerprint(
            {
                "kind": "bioinformatics-read-layout",
                "max_reads": values[0],
                "max_read_length": values[1],
                "max_cigar_ops": values[2],
                "max_umi_length": values[3],
            }
        )


class ReadPairLayout(StrictModule):
    """Pair relations and reference overlap for every fixed read slot."""

    mate_index: Array
    paired: Array
    proper_pair: Array
    first_in_pair: Array
    second_in_pair: Array
    mate_unmapped: Array
    mate_reverse: Array
    overlap_start: Array
    overlap_end: Array
    valid: Array
    status: Array
    evidence: Array

    def __init__(
        self,
        mate_index: ArrayLike,
        paired: ArrayLike,
        proper_pair: ArrayLike,
        first_in_pair: ArrayLike,
        second_in_pair: ArrayLike,
        mate_unmapped: ArrayLike,
        mate_reverse: ArrayLike,
        overlap_start: ArrayLike,
        overlap_end: ArrayLike,
        /,
    ):
        mate = jnp.asarray(mate_index, dtype=jnp.int32)
        if mate.ndim != 1:
            raise ValueError("mate_index must be rank one.")
        shape = mate.shape

        def boolean(values: ArrayLike, name: str) -> Array:
            result = jnp.asarray(values, dtype=bool)
            if result.shape != shape:
                raise ValueError(f"{name} must have mate_index shape.")
            return result

        paired_ = boolean(paired, "paired")
        proper_ = boolean(proper_pair, "proper_pair")
        first_ = boolean(first_in_pair, "first_in_pair")
        second_ = boolean(second_in_pair, "second_in_pair")
        mate_unmapped_ = boolean(mate_unmapped, "mate_unmapped")
        mate_reverse_ = boolean(mate_reverse, "mate_reverse")
        overlap_start_ = jnp.asarray(overlap_start, dtype=jnp.int32)
        overlap_end_ = jnp.asarray(overlap_end, dtype=jnp.int32)
        if overlap_start_.shape != shape or overlap_end_.shape != shape:
            raise ValueError("overlap coordinates must have mate_index shape.")
        slot_count = shape[0]
        mate_in_range = (mate >= 0) & (mate < slot_count)
        pair_flag_valid = paired_ | ~(
            proper_ | first_ | second_ | mate_unmapped_ | mate_reverse_
        )
        valid = (
            (((~paired_) & (mate == -1)) | (paired_ & ((mate == -1) | mate_in_range)))
            & ~(first_ & second_)
            & pair_flag_valid
        )
        overlap_valid = ((overlap_start_ == -1) & (overlap_end_ == -1)) | (
            paired_
            & ~mate_unmapped_
            & (overlap_start_ >= 0)
            & (overlap_end_ > overlap_start_)
        )
        valid = valid & overlap_valid
        self.mate_index = mate
        self.paired = paired_
        self.proper_pair = proper_
        self.first_in_pair = first_
        self.second_in_pair = second_
        self.mate_unmapped = mate_unmapped_
        self.mate_reverse = mate_reverse_
        self.overlap_start = overlap_start_
        self.overlap_end = overlap_end_
        self.valid = valid
        self.status = jnp.where(
            valid, READ_STATUS_VALID, READ_STATUS_INVALID_PAIR
        ).astype(jnp.int8)
        self.evidence = jnp.maximum(overlap_end_ - overlap_start_, 0).astype(jnp.int32)

    @property
    def overlapping_mate(self) -> Array:
        return (
            self.valid
            & (self.overlap_start >= 0)
            & (self.overlap_end > self.overlap_start)
        )


class ReadBatch(StrictModule):
    """Fixed-capacity reads, qualities, CIGARs, flags, pairing, and provenance IDs."""

    sequence: SequenceBatch
    quality: QualityBatch
    cigar: CigarBatch
    pair: ReadPairLayout
    flags: Array
    reference_id: Array
    reference_start: Array
    mapping_quality: Array
    mapping_quality_known: Array
    mate_reference_id: Array
    mate_reference_start: Array
    template_length: Array
    read_group_id: Array
    umi_codes: Array
    umi_valid_mask: Array
    valid: Array
    status: Array
    evidence: Array
    layout: ReadLayout = eqx.field(static=True)

    def __init__(
        self,
        sequence: SequenceBatch,
        quality: QualityBatch,
        cigar: CigarBatch,
        pair: ReadPairLayout,
        flags: ArrayLike,
        reference_id: ArrayLike,
        reference_start: ArrayLike,
        mapping_quality: ArrayLike,
        mapping_quality_known: ArrayLike,
        mate_reference_id: ArrayLike,
        mate_reference_start: ArrayLike,
        template_length: ArrayLike,
        read_group_id: ArrayLike,
        umi_codes: ArrayLike,
        umi_valid_mask: ArrayLike,
        layout: ReadLayout,
        /,
    ):
        if not isinstance(layout, ReadLayout):
            raise TypeError("layout must be a ReadLayout.")
        expected_sequence_shape = (layout.max_reads, layout.max_read_length)
        if sequence.token_codes.shape != expected_sequence_shape:
            raise ValueError("sequence shape does not match the ReadLayout bucket.")
        if quality.phred_scores.shape != expected_sequence_shape:
            raise ValueError("quality shape does not match the ReadLayout bucket.")
        if cigar.packed_ops.shape != (layout.max_reads, layout.max_cigar_ops):
            raise ValueError("cigar shape does not match the ReadLayout bucket.")
        if pair.mate_index.shape != (layout.max_reads,):
            raise ValueError("pair shape does not match the ReadLayout bucket.")
        if quality.record_ids.shape != sequence.record_ids.shape:
            raise ValueError("sequence and quality record IDs must have matching shape.")
        vector_shape = (layout.max_reads,)

        def integer(values: ArrayLike, dtype: jnp.dtype, name: str) -> Array:
            result = jnp.asarray(values, dtype=dtype)
            if result.shape != vector_shape:
                raise ValueError(f"{name} must have shape {vector_shape}.")
            return result

        flags_ = integer(flags, jnp.uint16, "flags")
        reference_id_ = integer(reference_id, jnp.int32, "reference_id")
        reference_start_ = integer(reference_start, jnp.int32, "reference_start")
        mapping_quality_ = integer(mapping_quality, jnp.int16, "mapping_quality")
        mapq_known_ = jnp.asarray(mapping_quality_known, dtype=bool)
        if mapq_known_.shape != vector_shape:
            raise ValueError("mapping_quality_known must have the read-vector shape.")
        mate_reference_id_ = integer(mate_reference_id, jnp.int32, "mate_reference_id")
        mate_reference_start_ = integer(
            mate_reference_start, jnp.int32, "mate_reference_start"
        )
        template_length_ = integer(template_length, jnp.int32, "template_length")
        read_group_id_ = integer(read_group_id, jnp.int32, "read_group_id")
        umi_codes_ = jnp.asarray(umi_codes, dtype=jnp.int16)
        umi_valid_ = jnp.asarray(umi_valid_mask, dtype=bool)
        if umi_codes_.shape != (layout.max_reads, layout.max_umi_length):
            raise ValueError("umi_codes shape does not match the ReadLayout bucket.")
        if umi_valid_.shape != umi_codes_.shape:
            raise ValueError("umi_valid_mask must have umi_codes shape.")

        record_active = sequence.case_mask
        unmapped = (flags_ & SAM_FLAG_UNMAPPED) != 0
        mapped_coordinate_valid = unmapped | (
            (reference_id_ >= 0) & (reference_start_ >= 0)
        )
        unmapped_state_valid = (~unmapped) | (
            (reference_id_ == -1)
            & (reference_start_ == -1)
            & (cigar.op_count == 0)
            & ~mapq_known_
        )
        sequence_cigar_valid = unmapped | (
            cigar.query_length.astype(jnp.int32) == sequence.lengths
        )
        identity_valid = (quality.record_ids == sequence.record_ids) & (
            quality.case_mask == sequence.case_mask
        )
        quality_positions_valid = identity_valid & jnp.all(
            ~quality.valid_mask | sequence.valid_mask, axis=-1
        )
        mapq_valid = jnp.where(
            mapq_known_,
            (mapping_quality_ >= 0) & (mapping_quality_ <= 254),
            mapping_quality_ == 0,
        )
        umi_prefix = jnp.cumprod(umi_valid_.astype(jnp.int8), axis=-1).astype(bool)
        umi_valid = (
            jnp.all(umi_prefix == umi_valid_, axis=-1)
            & jnp.all(
                (~umi_valid_)
                | ((umi_codes_ >= 0) & (umi_codes_ < sequence.alphabet.size)),
                axis=-1,
            )
            & jnp.all(
                umi_valid_
                | (umi_codes_ == sequence.alphabet.code(sequence.alphabet.pad_symbol)),
                axis=-1,
            )
        )
        pair_flags_valid = (
            (pair.paired == ((flags_ & SAM_FLAG_PAIRED) != 0))
            & (pair.proper_pair == ((flags_ & SAM_FLAG_PROPER_PAIR) != 0))
            & (pair.first_in_pair == ((flags_ & SAM_FLAG_FIRST_IN_PAIR) != 0))
            & (pair.second_in_pair == ((flags_ & SAM_FLAG_SECOND_IN_PAIR) != 0))
            & (pair.mate_unmapped == ((flags_ & SAM_FLAG_MATE_UNMAPPED) != 0))
            & (pair.mate_reverse == ((flags_ & SAM_FLAG_MATE_REVERSE) != 0))
        )
        padding_valid = (~record_active) & (
            (flags_ == 0)
            & (reference_id_ == -1)
            & (reference_start_ == -1)
            & ~mapq_known_
            & (mapping_quality_ == 0)
            & (mate_reference_id_ == -1)
            & (mate_reference_start_ == -1)
            & (template_length_ == 0)
            & (read_group_id_ == -1)
            & (cigar.op_count == 0)
            & ~jnp.any(umi_valid_, axis=-1)
        )
        valid = (
            record_active
            & cigar.valid
            & pair.valid
            & pair_flags_valid
            & mapped_coordinate_valid
            & unmapped_state_valid
            & sequence_cigar_valid
            & quality_positions_valid
            & mapq_valid
            & umi_valid
        )
        valid = valid | padding_valid
        status = jnp.where(
            ~cigar.valid,
            READ_STATUS_INVALID_CIGAR,
            jnp.where(
                ~sequence_cigar_valid,
                READ_STATUS_SEQUENCE_CIGAR_LENGTH_MISMATCH,
                jnp.where(
                    ~mapped_coordinate_valid,
                    READ_STATUS_INVALID_MAPPED_COORDINATE,
                    jnp.where(
                        ~unmapped_state_valid,
                        READ_STATUS_INVALID_UNMAPPED_STATE,
                        jnp.where(
                            ~(pair.valid & pair_flags_valid),
                            READ_STATUS_INVALID_PAIR,
                            jnp.where(
                                ~identity_valid,
                                READ_STATUS_IDENTITY_MISMATCH,
                                jnp.where(
                                    ~quality_positions_valid,
                                    READ_STATUS_INVALID_QUALITY,
                                    jnp.where(
                                        ~umi_valid,
                                        READ_STATUS_INVALID_UMI,
                                        jnp.where(
                                            ~mapq_valid,
                                            READ_STATUS_INVALID_MAPQ,
                                            READ_STATUS_VALID,
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ).astype(jnp.int8)
        self.sequence = sequence
        self.quality = quality
        self.cigar = cigar
        self.pair = pair
        self.flags = flags_
        self.reference_id = reference_id_
        self.reference_start = reference_start_
        self.mapping_quality = mapping_quality_
        self.mapping_quality_known = mapq_known_
        self.mate_reference_id = mate_reference_id_
        self.mate_reference_start = mate_reference_start_
        self.template_length = template_length_
        self.read_group_id = read_group_id_
        self.umi_codes = umi_codes_
        self.umi_valid_mask = umi_valid_
        self.valid = valid
        self.status = jnp.where(
            record_active,
            status,
            jnp.where(padding_valid, READ_STATUS_VALID, READ_STATUS_IDENTITY_MISMATCH),
        ).astype(jnp.int8)
        self.evidence = sequence.lengths
        self.layout = layout

    @property
    def active(self) -> Array:
        return self.sequence.case_mask

    @property
    def paired(self) -> Array:
        return (self.flags & SAM_FLAG_PAIRED) != 0

    @property
    def unmapped(self) -> Array:
        return (self.flags & SAM_FLAG_UNMAPPED) != 0

    @property
    def reverse_strand(self) -> Array:
        return (self.flags & SAM_FLAG_REVERSE) != 0

    @property
    def secondary(self) -> Array:
        return (self.flags & SAM_FLAG_SECONDARY) != 0

    @property
    def supplementary(self) -> Array:
        return (self.flags & SAM_FLAG_SUPPLEMENTARY) != 0

    @property
    def duplicate(self) -> Array:
        return (self.flags & SAM_FLAG_DUPLICATE) != 0

    @property
    def umi_known(self) -> Array:
        return jnp.any(self.umi_valid_mask, axis=-1)

    @property
    def umi_duplicate(self) -> Array:
        return self.duplicate & self.umi_known

    @property
    def base_quality_known(self) -> Array:
        return self.quality.valid_mask


class ReadEvidenceProvenance(StrictModule):
    """Array-only audit facts that affect interpretation of read evidence."""

    record_ids: Array
    flags: Array
    read_group_id: Array
    quality_known: Array
    mapping_quality_known: Array
    umi_known: Array
    umi_duplicate: Array
    overlapping_mate: Array
    reverse_strand: Array
    unmapped: Array
    secondary: Array
    supplementary: Array
    valid: Array
    status: Array
    evidence: Array


def read_evidence_provenance(reads: ReadBatch, /) -> ReadEvidenceProvenance:
    """Extract explicit evidence-use provenance without host strings or objects."""

    quality_known = reads.active & jnp.all(
        (~reads.sequence.valid_mask) | reads.quality.valid_mask, axis=-1
    )
    return ReadEvidenceProvenance(
        reads.sequence.record_ids,
        reads.flags,
        reads.read_group_id,
        quality_known,
        reads.mapping_quality_known,
        reads.umi_known,
        reads.umi_duplicate,
        reads.pair.overlapping_mate,
        reads.reverse_strand,
        reads.unmapped,
        reads.secondary,
        reads.supplementary,
        reads.valid,
        reads.status,
        reads.evidence,
    )
