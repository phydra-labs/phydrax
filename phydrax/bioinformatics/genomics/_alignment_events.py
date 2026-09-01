#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ._cigar import (
    cigar_consumption_for_operation,
    CIGAR_STATUS_EVENT_CAPACITY_EXCEEDED,
    CigarBatch,
)


ALIGNMENT_EVENT_STATUS_VALID = 0
ALIGNMENT_EVENT_STATUS_QUERY_LENGTH_MISMATCH = 7
ALIGNMENT_EVENT_STATUS_INVALID_REFERENCE_START = 8


class AlignmentEventBatch(StrictModule):
    """Fixed-capacity per-unit expansion of a batch of CIGAR programs.

    Query-only and reference-only operations use ``-1`` on the axis they do not
    consume. Insertions additionally carry the preceding reference anchor, while
    leading insertions anchor at ``reference_start``. Hard clips and pads are
    represented, but consume neither coordinate.
    """

    operation: Array
    operation_index: Array
    offset_in_operation: Array
    query_index: Array
    read_cycle: Array
    reference_position: Array
    reference_anchor_position: Array
    active: Array
    event_count: Array
    valid: Array
    status: Array
    evidence: Array

    @property
    def event_capacity(self) -> int:
        return self.operation.shape[-1]

    @property
    def aligned_base(self) -> Array:
        return self.active & (
            (self.operation == 0) | (self.operation == 7) | (self.operation == 8)
        )

    @property
    def insertion(self) -> Array:
        return self.active & (self.operation == 1)

    @property
    def deletion(self) -> Array:
        return self.active & (self.operation == 2)

    @property
    def reference_skip(self) -> Array:
        return self.active & (self.operation == 3)

    @property
    def soft_clip(self) -> Array:
        return self.active & (self.operation == 4)

    @property
    def hard_clip(self) -> Array:
        return self.active & (self.operation == 5)

    @property
    def padding(self) -> Array:
        return self.active & (self.operation == 6)


def expand_alignment_events(
    cigar: CigarBatch,
    reference_start: ArrayLike,
    query_length: ArrayLike,
    reverse_strand: ArrayLike,
    event_capacity: int,
    /,
) -> AlignmentEventBatch:
    """Expand every CIGAR unit without returning usable partial overflow output."""

    capacity = int(event_capacity)
    if capacity < 1:
        raise ValueError("event_capacity must be positive.")
    leading_shape = cigar.op_count.shape
    reference_start_ = jnp.broadcast_to(
        jnp.asarray(reference_start, dtype=jnp.int32), leading_shape
    )
    query_length_ = jnp.broadcast_to(
        jnp.asarray(query_length, dtype=jnp.int32), leading_shape
    )
    reverse_ = jnp.broadcast_to(jnp.asarray(reverse_strand, dtype=bool), leading_shape)

    operation_mask = cigar.operation_mask
    lengths = jnp.where(operation_mask, cigar.lengths.astype(jnp.int32), 0)
    operations = cigar.operations
    query_consuming, reference_consuming = cigar_consumption_for_operation(operations)
    query_units = lengths * query_consuming
    reference_units = lengths * reference_consuming
    operation_ends = jnp.cumsum(lengths, axis=-1, dtype=jnp.int32)
    operation_starts = operation_ends - lengths
    query_starts = jnp.cumsum(query_units, axis=-1, dtype=jnp.int32) - query_units
    reference_starts = (
        jnp.cumsum(reference_units, axis=-1, dtype=jnp.int32) - reference_units
    )

    event_slots = jnp.arange(capacity, dtype=jnp.int32)
    event_grid = jnp.broadcast_to(event_slots, leading_shape + (capacity,))
    operation_index = jnp.sum(
        event_grid[..., :, None] >= operation_ends[..., None, :],
        axis=-1,
        dtype=jnp.int32,
    )
    safe_operation_index = jnp.clip(operation_index, 0, cigar.operation_capacity - 1)

    def gather(values: Array) -> Array:
        return jnp.take_along_axis(values, safe_operation_index, axis=-1)

    event_operation = gather(operations)
    event_operation_start = gather(operation_starts)
    event_offset = event_grid - event_operation_start
    event_query_consuming, event_reference_consuming = cigar_consumption_for_operation(
        event_operation
    )
    event_query_index = gather(query_starts) + event_offset
    event_reference_offset = gather(reference_starts) + event_offset
    event_reference_position = reference_start_[..., None] + event_reference_offset
    insertion_anchor = reference_start_[..., None] + jnp.maximum(
        gather(reference_starts) - 1, 0
    )

    capacity_valid = (cigar.evidence <= jnp.uint32(capacity)) & jnp.all(
        (~operation_mask) | (cigar.lengths <= jnp.uint32(capacity)), axis=-1
    )
    required_count = jnp.where(
        capacity_valid, cigar.evidence, jnp.uint32(capacity + 1)
    ).astype(jnp.int32)
    query_length_valid = cigar.query_length.astype(jnp.int32) == query_length_
    has_reference_consumption = cigar.reference_length > 0
    reference_start_valid = (~has_reference_consumption) | (reference_start_ >= 0)
    alignment_valid = (
        cigar.valid & capacity_valid & query_length_valid & reference_start_valid
    )
    active = (event_grid < required_count[..., None]) & alignment_valid[..., None]
    event_operation_valid = active & (operation_index < cigar.op_count[..., None])

    query_index = jnp.where(
        event_operation_valid & event_query_consuming,
        event_query_index,
        -1,
    ).astype(jnp.int32)
    read_cycle = jnp.where(
        event_operation_valid & event_query_consuming,
        jnp.where(
            reverse_[..., None], query_length_[..., None] - 1 - query_index, query_index
        ),
        -1,
    ).astype(jnp.int32)
    reference_position = jnp.where(
        event_operation_valid & event_reference_consuming,
        event_reference_position,
        -1,
    ).astype(jnp.int32)
    reference_anchor_position = jnp.where(
        event_operation_valid & (event_operation == 1), insertion_anchor, -1
    ).astype(jnp.int32)
    status = jnp.where(
        ~cigar.valid,
        cigar.status,
        jnp.where(
            ~capacity_valid,
            CIGAR_STATUS_EVENT_CAPACITY_EXCEEDED,
            jnp.where(
                ~query_length_valid,
                ALIGNMENT_EVENT_STATUS_QUERY_LENGTH_MISMATCH,
                jnp.where(
                    ~reference_start_valid,
                    ALIGNMENT_EVENT_STATUS_INVALID_REFERENCE_START,
                    ALIGNMENT_EVENT_STATUS_VALID,
                ),
            ),
        ),
    ).astype(jnp.int8)
    return AlignmentEventBatch(
        jnp.where(event_operation_valid, event_operation, 0).astype(jnp.uint8),
        jnp.where(event_operation_valid, operation_index, -1).astype(jnp.int32),
        jnp.where(event_operation_valid, event_offset, -1).astype(jnp.int32),
        query_index,
        read_cycle,
        reference_position,
        reference_anchor_position,
        event_operation_valid,
        required_count,
        alignment_valid,
        status,
        cigar.evidence,
    )
