#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import re
from enum import IntEnum
from typing import Iterable, Sequence

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule


class CigarOp(IntEnum):
    """The complete BAM packed-CIGAR operation code set."""

    MATCH = 0
    INSERTION = 1
    DELETION = 2
    REFERENCE_SKIP = 3
    SOFT_CLIP = 4
    HARD_CLIP = 5
    PADDING = 6
    SEQUENCE_MATCH = 7
    SEQUENCE_MISMATCH = 8


CIGAR_OPERATION_CHARS = "MIDNSHP=X"
CIGAR_STATUS_VALID = 0
CIGAR_STATUS_INVALID_COUNT = 1
CIGAR_STATUS_INVALID_OPERATION = 2
CIGAR_STATUS_INVALID_LENGTH = 3
CIGAR_STATUS_DIRTY_PADDING = 4
CIGAR_STATUS_OPERATION_CAPACITY_EXCEEDED = 5
CIGAR_STATUS_EVENT_CAPACITY_EXCEEDED = 6
_MAX_CIGAR_LENGTH = (1 << 28) - 1
_CIGAR_PATTERN = re.compile(r"([0-9]+)([MIDNSHP=X])")

_QUERY_CONSUMING = jnp.asarray(
    [True, True, False, False, True, False, False, True, True], dtype=bool
)
_REFERENCE_CONSUMING = jnp.asarray(
    [True, False, True, True, False, False, False, True, True], dtype=bool
)
_ALIGNED_BASE = jnp.asarray(
    [True, False, False, False, False, False, False, True, True], dtype=bool
)


class CigarBatch(StrictModule):
    """A fixed-operation-capacity batch of canonical BAM packed CIGARs.

    The final axis is operation capacity. ``op_count`` and every diagnostic have
    the leading batch shape. Inactive packed words must be zero; a zero-length
    active operation is invalid even though the packed representation can encode it.
    """

    packed_ops: Array
    op_count: Array
    valid: Array
    status: Array
    evidence: Array

    def __init__(
        self,
        packed_ops: ArrayLike,
        op_count: ArrayLike,
        /,
        *,
        source_valid: ArrayLike | None = None,
        source_status: ArrayLike | None = None,
        evidence: ArrayLike | None = None,
    ):
        packed = jnp.asarray(packed_ops, dtype=jnp.uint32)
        counts = jnp.asarray(op_count, dtype=jnp.int32)
        if packed.ndim < 1 or counts.shape != packed.shape[:-1]:
            raise ValueError(
                "op_count must have exactly the leading shape of packed_ops."
            )
        capacity = packed.shape[-1]
        slots = jnp.arange(capacity, dtype=jnp.int32)
        active = slots < counts[..., None]
        operations = packed & jnp.uint32(0xF)
        lengths = packed >> jnp.uint32(4)
        count_valid = (counts >= 0) & (counts <= capacity)
        operations_valid = jnp.all((~active) | (operations <= 8), axis=-1)
        lengths_valid = jnp.all((~active) | (lengths > 0), axis=-1)
        padding_valid = jnp.all(active | (packed == 0), axis=-1)
        structural_valid = count_valid & operations_valid & lengths_valid & padding_valid
        supplied_valid = (
            jnp.ones(counts.shape, dtype=bool)
            if source_valid is None
            else jnp.broadcast_to(jnp.asarray(source_valid, dtype=bool), counts.shape)
        )
        supplied_status = (
            jnp.zeros(counts.shape, dtype=jnp.int8)
            if source_status is None
            else jnp.broadcast_to(
                jnp.asarray(source_status, dtype=jnp.int8), counts.shape
            )
        )
        structural_status = jnp.where(
            ~count_valid,
            CIGAR_STATUS_INVALID_COUNT,
            jnp.where(
                ~operations_valid,
                CIGAR_STATUS_INVALID_OPERATION,
                jnp.where(
                    ~lengths_valid,
                    CIGAR_STATUS_INVALID_LENGTH,
                    jnp.where(
                        ~padding_valid,
                        CIGAR_STATUS_DIRTY_PADDING,
                        CIGAR_STATUS_VALID,
                    ),
                ),
            ),
        ).astype(jnp.int8)
        total_units = jnp.sum(jnp.where(active, lengths, 0), axis=-1, dtype=jnp.uint32)
        self.packed_ops = packed
        self.op_count = counts
        self.valid = structural_valid & supplied_valid
        self.status = jnp.where(
            structural_valid & ~supplied_valid, supplied_status, structural_status
        ).astype(jnp.int8)
        self.evidence = (
            total_units
            if evidence is None
            else jnp.broadcast_to(jnp.asarray(evidence, dtype=jnp.uint32), counts.shape)
        )

    @property
    def operation_capacity(self) -> int:
        return self.packed_ops.shape[-1]

    @property
    def lengths(self) -> Array:
        return self.packed_ops >> jnp.uint32(4)

    @property
    def operations(self) -> Array:
        return (self.packed_ops & jnp.uint32(0xF)).astype(jnp.uint8)

    @property
    def operation_mask(self) -> Array:
        slots = jnp.arange(self.operation_capacity, dtype=jnp.int32)
        return slots < self.op_count[..., None]

    @property
    def query_consumed(self) -> Array:
        operations = jnp.minimum(self.operations, 8)
        return self.lengths * _QUERY_CONSUMING[operations]

    @property
    def reference_consumed(self) -> Array:
        operations = jnp.minimum(self.operations, 8)
        return self.lengths * _REFERENCE_CONSUMING[operations]

    @property
    def query_length(self) -> Array:
        return jnp.sum(
            jnp.where(self.operation_mask, self.query_consumed, 0),
            axis=-1,
            dtype=jnp.uint32,
        )

    @property
    def reference_length(self) -> Array:
        return jnp.sum(
            jnp.where(self.operation_mask, self.reference_consumed, 0),
            axis=-1,
            dtype=jnp.uint32,
        )

    @property
    def aligned_base_count(self) -> Array:
        operations = jnp.minimum(self.operations, 8)
        counts = self.lengths * _ALIGNED_BASE[operations]
        return jnp.sum(
            jnp.where(self.operation_mask, counts, 0),
            axis=-1,
            dtype=jnp.uint32,
        )


def pack_cigar(
    lengths: ArrayLike, operations: ArrayLike, op_count: ArrayLike, /
) -> CigarBatch:
    """Pack lengths and operation codes without losing invalid-input diagnostics."""

    lengths_ = jnp.asarray(lengths)
    operations_ = jnp.asarray(operations)
    if lengths_.shape != operations_.shape or lengths_.ndim < 1:
        raise ValueError("lengths and operations must be matching non-scalar arrays.")
    lengths_i32 = lengths_.astype(jnp.int32)
    operations_i32 = operations_.astype(jnp.int32)
    representable = (lengths_i32 >= 0) & (lengths_i32 <= _MAX_CIGAR_LENGTH)
    safe_lengths = jnp.where(representable, lengths_i32, 0).astype(jnp.uint32)
    safe_operations = jnp.where(
        (operations_i32 >= 0) & (operations_i32 <= 15), operations_i32, 15
    ).astype(jnp.uint32)
    packed = (safe_lengths << jnp.uint32(4)) | safe_operations
    return CigarBatch(packed, op_count)


def cigar_batch_from_tuples(
    cigars: Sequence[Iterable[tuple[int | CigarOp, int]] | None],
    operation_capacity: int,
    /,
) -> CigarBatch:
    """Lower host CIGAR tuples ``(operation, length)`` with capacity preflight.

    An over-capacity record produces no partial CIGAR: its row is all zero, is
    invalid, reports status 5, and exposes its required operation count as evidence.
    Valid records expose their required expanded-event count as evidence. ``None``
    is the valid absent CIGAR used by an unmapped SAM record.
    """

    capacity = int(operation_capacity)
    if capacity < 1:
        raise ValueError("operation_capacity must be positive.")
    rows: list[list[int]] = []
    counts: list[int] = []
    valid: list[bool] = []
    status: list[int] = []
    evidence: list[int] = []
    for cigar in cigars:
        tuples = () if cigar is None else tuple(cigar)
        needed = len(tuples)
        expanded_units = 0
        if needed > capacity:
            rows.append([0] * capacity)
            counts.append(0)
            valid.append(False)
            status.append(CIGAR_STATUS_OPERATION_CAPACITY_EXCEEDED)
            evidence.append(needed)
            continue
        row = [0] * capacity
        row_valid = True
        row_status = CIGAR_STATUS_VALID
        for index, (operation, length) in enumerate(tuples):
            operation_i = int(operation)
            length_i = int(length)
            if 0 < length_i <= _MAX_CIGAR_LENGTH:
                expanded_units += length_i
            if operation_i < 0 or operation_i > 8:
                row_valid = False
                row_status = CIGAR_STATUS_INVALID_OPERATION
            elif length_i < 1 or length_i > _MAX_CIGAR_LENGTH:
                row_valid = False
                row_status = CIGAR_STATUS_INVALID_LENGTH
            else:
                row[index] = (length_i << 4) | operation_i
        rows.append(row)
        counts.append(needed)
        valid.append(row_valid)
        status.append(row_status)
        evidence.append(expanded_units)
    return CigarBatch(
        jnp.asarray(rows, dtype=jnp.uint32),
        jnp.asarray(counts, dtype=jnp.int32),
        source_valid=jnp.asarray(valid, dtype=bool),
        source_status=jnp.asarray(status, dtype=jnp.int8),
        evidence=jnp.asarray(evidence, dtype=jnp.uint32),
    )


def cigar_batch_from_strings(
    cigars: Sequence[str | None], operation_capacity: int, /
) -> CigarBatch:
    """Parse SAM CIGAR strings, treating ``None`` and ``*`` as absent CIGARs."""

    parsed: list[tuple[tuple[int, int], ...] | None] = []
    malformed: list[bool] = []
    for text in cigars:
        if text is None or text == "*":
            parsed.append(None)
            malformed.append(False)
            continue
        matches = tuple(_CIGAR_PATTERN.finditer(text))
        complete = bool(matches) and "".join(match.group(0) for match in matches) == text
        if not complete:
            parsed.append(((15, 1),))
            malformed.append(True)
            continue
        parsed.append(
            tuple(
                (CIGAR_OPERATION_CHARS.index(match.group(2)), int(match.group(1)))
                for match in matches
            )
        )
        malformed.append(False)
    batch = cigar_batch_from_tuples(parsed, operation_capacity)
    malformed_ = jnp.asarray(malformed, dtype=bool)
    return CigarBatch(
        batch.packed_ops,
        batch.op_count,
        source_valid=batch.valid & ~malformed_,
        source_status=jnp.where(malformed_, CIGAR_STATUS_INVALID_OPERATION, batch.status),
        evidence=batch.evidence,
    )


def cigar_consumption_for_operation(operations: ArrayLike, /) -> tuple[Array, Array]:
    """Return query/reference consumption for valid BAM operation codes."""

    operations_ = jnp.asarray(operations, dtype=jnp.int32)
    in_range = (operations_ >= 0) & (operations_ <= 8)
    safe = jnp.clip(operations_, 0, 8)
    return _QUERY_CONSUMING[safe] & in_range, _REFERENCE_CONSUMING[safe] & in_range
