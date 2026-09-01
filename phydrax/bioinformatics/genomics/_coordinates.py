#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import core as jax_core
from jaxtyping import Array, ArrayLike

from phydrax._strict import StrictModule

from ..foundation import (
    BioinformaticsMethodContract,
    DifferentiationKind,
    ExecutionKind,
    MethodKind,
    OutputKind,
)


class Strand(IntEnum):
    """Orientation relative to a linear reference."""

    REVERSE = -1
    UNKNOWN = 0
    FORWARD = 1


class CoordinateStatus(IntEnum):
    """Stable statuses for exact coordinate and interval operations."""

    SUCCESS = 0
    INCOMPATIBLE_REFERENCE = 1
    INCOMPATIBLE_STRAND = 2
    DISJOINT = 3
    OUT_OF_BOUNDS = 4
    CAPACITY_EXCEEDED = 5
    INVALID_INPUT = 6
    AMBIGUOUS = 7
    UNMAPPED = 8
    PARTIAL = 9


_INTERVAL_CONTRACT = BioinformaticsMethodContract(
    "zero-based half-open interval algebra",
    MethodKind.EXACT_MODEL,
    ExecutionKind.EXACT_DISCRETE,
    DifferentiationKind.ALMOST_EVERYWHERE,
    OutputKind.SET,
    conditioning_statement="Intervals use one reference and zero-based half-open endpoints.",
    truncation_statement="No interval is truncated; insufficient capacity invalidates the result.",
    capacity_semantics="valid marks occupied fixed-capacity interval slots.",
    assumptions=("Reference indices identify the same declared reference dictionary.",),
    nondifferentiable_outputs=(
        "reference_indices",
        "starts",
        "ends",
        "strands",
        "valid",
        "status",
    ),
    input_dtype="int64",
    compute_dtype="int64",
    output_dtype="int64",
)


def _concrete(value: Array) -> np.ndarray | None:
    if isinstance(value, jax_core.Tracer):
        return None
    return np.asarray(value)


def _integer_scalar(value: ArrayLike, name: str, *, dtype=jnp.int64) -> Array:
    array = jnp.asarray(value)
    if array.shape != () or not jnp.issubdtype(array.dtype, jnp.integer):
        raise ValueError(f"{name} must be an integer scalar.")
    return array.astype(dtype)


def _nonnegative_scalar(value: ArrayLike, name: str, *, dtype=jnp.int64) -> Array:
    array = _integer_scalar(value, name, dtype=dtype)
    concrete = _concrete(array)
    if concrete is not None and int(concrete) < 0:
        raise ValueError(f"{name} must be non-negative.")
    return array


def _strand_scalar(value: ArrayLike | Strand, name: str = "strand") -> Array:
    strand = _integer_scalar(value, name, dtype=jnp.int8)
    concrete = _concrete(strand)
    if concrete is not None and int(concrete) not in (-1, 0, 1):
        raise ValueError(f"{name} must be Strand.REVERSE, UNKNOWN, or FORWARD.")
    return strand


class LinearCoordinate(StrictModule):
    """A zero-based position on one linear reference sequence."""

    reference_index: Array
    position: Array

    def __init__(self, reference_index: ArrayLike, position: ArrayLike, /):
        self.reference_index = _nonnegative_scalar(
            reference_index, "reference_index", dtype=jnp.int32
        )
        self.position = _nonnegative_scalar(position, "position", dtype=jnp.int64)


class LinearInterval(StrictModule):
    """A zero-based half-open interval; zero-length intervals are representable."""

    reference_index: Array
    start: Array
    end: Array
    strand: Array

    def __init__(
        self,
        reference_index: ArrayLike,
        start: ArrayLike,
        end: ArrayLike,
        /,
        *,
        strand: ArrayLike | Strand = Strand.UNKNOWN,
    ):
        reference_ = _nonnegative_scalar(
            reference_index, "reference_index", dtype=jnp.int32
        )
        start_ = _nonnegative_scalar(start, "start", dtype=jnp.int64)
        end_ = _nonnegative_scalar(end, "end", dtype=jnp.int64)
        start_host = _concrete(start_)
        end_host = _concrete(end_)
        if (
            start_host is not None
            and end_host is not None
            and int(end_host) < int(start_host)
        ):
            raise ValueError("A half-open interval requires end >= start.")
        self.reference_index = reference_
        self.start = start_
        self.end = end_
        self.strand = _strand_scalar(strand)

    @property
    def length(self) -> Array:
        return self.end - self.start


class SourceAlleleCoordinate(StrictModule):
    """A position in one source allele with its replaced reference interval."""

    source_interval: LinearInterval
    allele_index: Array
    allele_position: Array

    def __init__(
        self,
        source_interval: LinearInterval,
        allele_index: ArrayLike,
        allele_position: ArrayLike,
        /,
    ):
        if not isinstance(source_interval, LinearInterval):
            raise TypeError("source_interval must be a LinearInterval.")
        self.source_interval = source_interval
        self.allele_index = _nonnegative_scalar(
            allele_index, "allele_index", dtype=jnp.int32
        )
        self.allele_position = _nonnegative_scalar(
            allele_position, "allele_position", dtype=jnp.int64
        )


class TranscriptCoordinate(StrictModule):
    """A zero-based position in a spliced transcript's 5′-to-3′ orientation."""

    transcript_index: Array
    position: Array

    def __init__(self, transcript_index: ArrayLike, position: ArrayLike, /):
        self.transcript_index = _nonnegative_scalar(
            transcript_index, "transcript_index", dtype=jnp.int32
        )
        self.position = _nonnegative_scalar(position, "position", dtype=jnp.int64)


class CDSCoordinate(StrictModule):
    """A zero-based nucleotide position in an oriented coding sequence."""

    transcript_index: Array
    position: Array

    def __init__(self, transcript_index: ArrayLike, position: ArrayLike, /):
        self.transcript_index = _nonnegative_scalar(
            transcript_index, "transcript_index", dtype=jnp.int32
        )
        self.position = _nonnegative_scalar(position, "position", dtype=jnp.int64)


class ProteinCoordinate(StrictModule):
    """A zero-based residue position in a translated protein."""

    protein_index: Array
    residue_index: Array

    def __init__(self, protein_index: ArrayLike, residue_index: ArrayLike, /):
        self.protein_index = _nonnegative_scalar(
            protein_index, "protein_index", dtype=jnp.int32
        )
        self.residue_index = _nonnegative_scalar(
            residue_index, "residue_index", dtype=jnp.int64
        )


class PhaseCoordinate(StrictModule):
    """GFF CDS phase attached to one oriented CDS segment."""

    cds_feature_index: Array
    phase: Array

    def __init__(self, cds_feature_index: ArrayLike, phase: ArrayLike, /):
        self.cds_feature_index = _nonnegative_scalar(
            cds_feature_index, "cds_feature_index", dtype=jnp.int32
        )
        phase_ = _integer_scalar(phase, "phase", dtype=jnp.int8)
        concrete = _concrete(phase_)
        if concrete is not None and int(concrete) not in (0, 1, 2):
            raise ValueError("CDS phase must be 0, 1, or 2.")
        self.phase = phase_


class GraphCoordinate(StrictModule):
    """A typed offset on an oriented node occurrence of a graph path."""

    graph_index: Array
    path_index: Array
    node_index: Array
    node_offset: Array
    orientation: Array

    def __init__(
        self,
        graph_index: ArrayLike,
        path_index: ArrayLike,
        node_index: ArrayLike,
        node_offset: ArrayLike,
        /,
        *,
        orientation: ArrayLike | Strand = Strand.FORWARD,
    ):
        self.graph_index = _nonnegative_scalar(
            graph_index, "graph_index", dtype=jnp.int32
        )
        self.path_index = _nonnegative_scalar(path_index, "path_index", dtype=jnp.int32)
        self.node_index = _nonnegative_scalar(node_index, "node_index", dtype=jnp.int32)
        self.node_offset = _nonnegative_scalar(
            node_offset, "node_offset", dtype=jnp.int64
        )
        orientation_ = _strand_scalar(orientation, "orientation")
        concrete = _concrete(orientation_)
        if concrete is not None and int(concrete) == 0:
            raise ValueError("A graph-node orientation must be forward or reverse.")
        self.orientation = orientation_


class IntervalSet(StrictModule):
    """A fixed-capacity collection of typed half-open linear intervals."""

    reference_indices: Array
    starts: Array
    ends: Array
    strands: Array
    valid: Array

    def __init__(
        self,
        reference_indices: ArrayLike,
        starts: ArrayLike,
        ends: ArrayLike,
        strands: ArrayLike,
        valid: ArrayLike,
        /,
    ):
        references = jnp.asarray(reference_indices)
        starts_ = jnp.asarray(starts)
        ends_ = jnp.asarray(ends)
        strands_ = jnp.asarray(strands)
        valid_ = jnp.asarray(valid)
        if any(
            value.ndim != 1 for value in (references, starts_, ends_, strands_, valid_)
        ):
            raise ValueError("IntervalSet fields must be one-dimensional.")
        if not all(
            value.shape == references.shape
            for value in (starts_, ends_, strands_, valid_)
        ):
            raise ValueError("IntervalSet fields must have matching shapes.")
        if not all(
            jnp.issubdtype(value.dtype, jnp.integer)
            for value in (references, starts_, ends_, strands_)
        ):
            raise TypeError("IntervalSet coordinate fields must be integer arrays.")
        if valid_.dtype != jnp.bool_:
            raise TypeError("IntervalSet valid must be boolean.")
        refs_h = _concrete(references)
        starts_h = _concrete(starts_)
        ends_h = _concrete(ends_)
        strands_h = _concrete(strands_)
        valid_h = _concrete(valid_)
        if (
            refs_h is not None
            and starts_h is not None
            and ends_h is not None
            and strands_h is not None
            and valid_h is not None
        ):
            if np.any(valid_h & ((refs_h < 0) | (starts_h < 0) | (ends_h < starts_h))):
                raise ValueError(
                    "Every valid interval must have a reference and 0 <= start <= end."
                )
            if np.any(valid_h & ~np.isin(strands_h, (-1, 0, 1))):
                raise ValueError("Every valid interval strand must be -1, 0, or 1.")
        self.reference_indices = references.astype(jnp.int32)
        self.starts = starts_.astype(jnp.int64)
        self.ends = ends_.astype(jnp.int64)
        self.strands = strands_.astype(jnp.int8)
        self.valid = valid_

    @classmethod
    def from_intervals(cls, intervals: tuple[LinearInterval, ...], /) -> "IntervalSet":
        if any(not isinstance(interval, LinearInterval) for interval in intervals):
            raise TypeError("intervals must contain only LinearInterval values.")
        if not intervals:
            empty = jnp.empty((0,), dtype=jnp.int32)
            return cls(
                empty,
                empty.astype(jnp.int64),
                empty.astype(jnp.int64),
                empty.astype(jnp.int8),
                empty.astype(bool),
            )
        return cls(
            jnp.stack([interval.reference_index for interval in intervals]),
            jnp.stack([interval.start for interval in intervals]),
            jnp.stack([interval.end for interval in intervals]),
            jnp.stack([interval.strand for interval in intervals]),
            jnp.ones((len(intervals),), dtype=bool),
        )

    @property
    def capacity(self) -> int:
        return int(self.starts.shape[0])

    @property
    def lengths(self) -> Array:
        return jnp.where(self.valid, self.ends - self.starts, 0)


class IntervalIntersectionResult(StrictModule):
    interval: LinearInterval
    valid: Array
    status: Array
    evidence: Array
    method_contract: BioinformaticsMethodContract = eqx.field(static=True)


class IntervalSetResult(StrictModule):
    intervals: IntervalSet
    valid: Array
    status: Array
    evidence: Array
    method_contract: BioinformaticsMethodContract = eqx.field(static=True)


def _compatible_strand(left: Array, right: Array) -> Array:
    return (left == right) | (left == 0) | (right == 0)


def _resolved_strand(left: Array, right: Array) -> Array:
    return jnp.where(left != 0, left, right).astype(jnp.int8)


def interval_contains(interval: LinearInterval, coordinate: LinearCoordinate, /) -> Array:
    """Return whether a position belongs to a zero-based half-open interval."""

    if not isinstance(interval, LinearInterval) or not isinstance(
        coordinate, LinearCoordinate
    ):
        raise TypeError("interval and coordinate must be typed coordinate records.")
    return (
        (interval.reference_index == coordinate.reference_index)
        & (coordinate.position >= interval.start)
        & (coordinate.position < interval.end)
    )


def interval_overlap_length(left: LinearInterval, right: LinearInterval, /) -> Array:
    """Return exact overlap length, or zero for incompatible/disjoint intervals."""

    if not isinstance(left, LinearInterval) or not isinstance(right, LinearInterval):
        raise TypeError("left and right must be LinearInterval values.")
    compatible = (left.reference_index == right.reference_index) & _compatible_strand(
        left.strand, right.strand
    )
    overlap = jnp.maximum(
        jnp.asarray(0, dtype=jnp.int64),
        jnp.minimum(left.end, right.end) - jnp.maximum(left.start, right.start),
    )
    return jnp.where(compatible, overlap, 0)


def interval_intersection(
    left: LinearInterval,
    right: LinearInterval,
    /,
) -> IntervalIntersectionResult:
    """Intersect two intervals, distinguishing incompatibility from empty overlap."""

    if not isinstance(left, LinearInterval) or not isinstance(right, LinearInterval):
        raise TypeError("left and right must be LinearInterval values.")
    same_reference = left.reference_index == right.reference_index
    same_strand = _compatible_strand(left.strand, right.strand)
    start = jnp.maximum(left.start, right.start)
    end = jnp.minimum(left.end, right.end)
    nonempty = end > start
    valid = same_reference & same_strand & nonempty
    status = jnp.where(
        ~same_reference,
        int(CoordinateStatus.INCOMPATIBLE_REFERENCE),
        jnp.where(
            ~same_strand,
            int(CoordinateStatus.INCOMPATIBLE_STRAND),
            jnp.where(
                nonempty, int(CoordinateStatus.SUCCESS), int(CoordinateStatus.DISJOINT)
            ),
        ),
    ).astype(jnp.int32)
    interval = LinearInterval(
        left.reference_index,
        jnp.where(valid, start, 0),
        jnp.where(valid, end, 0),
        strand=_resolved_strand(left.strand, right.strand),
    )
    evidence = jnp.asarray(
        [left.length, right.length, jnp.where(valid, end - start, 0)], dtype=jnp.int64
    )
    return IntervalIntersectionResult(
        interval, valid, status, evidence, _INTERVAL_CONTRACT
    )


def interval_union(left: LinearInterval, right: LinearInterval, /) -> IntervalSetResult:
    """Return exact set union in at most two slots, retaining disjoint components."""

    if not isinstance(left, LinearInterval) or not isinstance(right, LinearInterval):
        raise TypeError("left and right must be LinearInterval values.")
    same_reference = left.reference_index == right.reference_index
    same_strand = _compatible_strand(left.strand, right.strand)
    compatible = same_reference & same_strand
    left_first = (left.start < right.start) | (
        (left.start == right.start) & (left.end <= right.end)
    )
    first_start = jnp.where(left_first, left.start, right.start)
    first_end = jnp.where(left_first, left.end, right.end)
    second_start = jnp.where(left_first, right.start, left.start)
    second_end = jnp.where(left_first, right.end, left.end)
    connected = second_start <= first_end
    merged_end = jnp.maximum(first_end, second_end)
    strand = _resolved_strand(left.strand, right.strand)
    references = jnp.asarray(
        [left.reference_index, left.reference_index], dtype=jnp.int32
    )
    starts = jnp.stack((first_start, jnp.where(connected, 0, second_start)))
    ends = jnp.stack(
        (jnp.where(connected, merged_end, first_end), jnp.where(connected, 0, second_end))
    )
    strands = jnp.asarray([strand, strand], dtype=jnp.int8)
    slots = compatible & jnp.asarray([True, ~connected])
    intervals = IntervalSet(references, starts, ends, strands, slots)
    status = jnp.where(
        ~same_reference,
        int(CoordinateStatus.INCOMPATIBLE_REFERENCE),
        jnp.where(
            ~same_strand,
            int(CoordinateStatus.INCOMPATIBLE_STRAND),
            int(CoordinateStatus.SUCCESS),
        ),
    ).astype(jnp.int32)
    evidence = jnp.asarray(
        [left.length, right.length, interval_overlap_length(left, right), connected],
        dtype=jnp.int64,
    )
    return IntervalSetResult(intervals, compatible, status, evidence, _INTERVAL_CONTRACT)


def interval_difference(
    left: LinearInterval, right: LinearInterval, /
) -> IntervalSetResult:
    """Subtract right from left in at most two exact half-open components."""

    if not isinstance(left, LinearInterval) or not isinstance(right, LinearInterval):
        raise TypeError("left and right must be LinearInterval values.")
    same_reference = left.reference_index == right.reference_index
    same_strand = _compatible_strand(left.strand, right.strand)
    compatible = same_reference & same_strand
    overlap_start = jnp.maximum(left.start, right.start)
    overlap_end = jnp.minimum(left.end, right.end)
    overlap = compatible & (overlap_end > overlap_start)
    left_zero = left.end == left.start
    right_zero = right.end == right.start
    keep_whole = ~overlap | right_zero | left_zero
    prefix = overlap & (left.start < overlap_start)
    suffix = overlap & (overlap_end < left.end)
    first_start = jnp.where(keep_whole, left.start, left.start)
    first_end = jnp.where(keep_whole, left.end, overlap_start)
    second_start = overlap_end
    second_end = left.end
    valid = jnp.stack((keep_whole | prefix, suffix))
    intervals = IntervalSet(
        jnp.asarray([left.reference_index, left.reference_index], dtype=jnp.int32),
        jnp.stack((first_start, second_start)),
        jnp.stack((first_end, second_end)),
        jnp.asarray([left.strand, left.strand], dtype=jnp.int8),
        valid,
    )
    status = jnp.where(
        ~same_reference,
        int(CoordinateStatus.INCOMPATIBLE_REFERENCE),
        jnp.where(
            ~same_strand,
            int(CoordinateStatus.INCOMPATIBLE_STRAND),
            int(CoordinateStatus.SUCCESS),
        ),
    ).astype(jnp.int32)
    removed = jnp.where(
        overlap & ~left_zero & ~right_zero, overlap_end - overlap_start, 0
    )
    evidence = jnp.asarray(
        [left.length, right.length, removed, jnp.sum(valid)], dtype=jnp.int64
    )
    return IntervalSetResult(intervals, compatible, status, evidence, _INTERVAL_CONTRACT)


def merge_interval_set(
    intervals: IntervalSet, /, *, capacity: int | None = None
) -> IntervalSetResult:
    """Host-canonicalize overlaps/adjacency without discarding duplicates silently."""

    if not isinstance(intervals, IntervalSet):
        raise TypeError("intervals must be an IntervalSet.")
    output_capacity = intervals.capacity if capacity is None else int(capacity)
    if output_capacity < 0:
        raise ValueError("capacity must be non-negative.")
    references = _concrete(intervals.reference_indices)
    starts = _concrete(intervals.starts)
    ends = _concrete(intervals.ends)
    strands = _concrete(intervals.strands)
    valid = _concrete(intervals.valid)
    if (
        references is None
        or starts is None
        or ends is None
        or strands is None
        or valid is None
    ):
        raise TypeError("merge_interval_set requires concrete host interval arrays.")
    rows = sorted(
        (
            (
                int(references[index]),
                int(starts[index]),
                int(ends[index]),
                int(strands[index]),
            )
            for index in range(intervals.capacity)
            if bool(valid[index])
        ),
        key=lambda row: (row[0], row[3], row[1], row[2]),
    )
    merged: list[tuple[int, int, int, int]] = []
    for row in rows:
        if (
            merged
            and row[0] == merged[-1][0]
            and row[3] == merged[-1][3]
            and row[1] <= merged[-1][2]
        ):
            previous = merged[-1]
            merged[-1] = (previous[0], previous[1], max(previous[2], row[2]), previous[3])
        else:
            merged.append(row)
    overflow = len(merged) > output_capacity
    refs_out = np.zeros((output_capacity,), dtype=np.int32)
    starts_out = np.zeros((output_capacity,), dtype=np.int64)
    ends_out = np.zeros((output_capacity,), dtype=np.int64)
    strands_out = np.zeros((output_capacity,), dtype=np.int8)
    valid_out = np.zeros((output_capacity,), dtype=bool)
    if not overflow:
        for index, (reference, start, end, strand) in enumerate(merged):
            refs_out[index] = reference
            starts_out[index] = start
            ends_out[index] = end
            strands_out[index] = strand
            valid_out[index] = True
    result = IntervalSet(refs_out, starts_out, ends_out, strands_out, valid_out)
    status = CoordinateStatus.CAPACITY_EXCEEDED if overflow else CoordinateStatus.SUCCESS
    evidence = jnp.asarray(
        [len(rows), len(merged), output_capacity, len(rows) - len(merged)],
        dtype=jnp.int64,
    )
    return IntervalSetResult(
        result,
        jnp.asarray(not overflow),
        jnp.asarray(int(status), dtype=jnp.int32),
        evidence,
        _INTERVAL_CONTRACT,
    )


__all__ = [
    "CDSCoordinate",
    "CoordinateStatus",
    "GraphCoordinate",
    "IntervalIntersectionResult",
    "IntervalSet",
    "IntervalSetResult",
    "LinearCoordinate",
    "LinearInterval",
    "PhaseCoordinate",
    "ProteinCoordinate",
    "SourceAlleleCoordinate",
    "Strand",
    "TranscriptCoordinate",
    "interval_contains",
    "interval_difference",
    "interval_intersection",
    "interval_overlap_length",
    "interval_union",
    "merge_interval_set",
]
