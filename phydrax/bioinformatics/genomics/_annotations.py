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
from phydrax.sparse import EdgeRelation

from ..foundation import (
    BioinformaticsMethodContract,
    DifferentiationKind,
    ExecutionKind,
    FeatureDictionary,
    MethodKind,
    OutputKind,
)
from ._coordinates import LinearInterval, Strand


class AnnotationStatus(IntEnum):
    """Stable statuses for bounded annotation queries and relation audits."""

    SUCCESS = 0
    INVALID_INPUT = 1
    CAPACITY_EXCEEDED = 2
    NO_MATCH = 3
    CYCLIC_PARENT_RELATION = 4
    DUPLICATE_RELATION = 5


_ANNOTATION_QUERY_CONTRACT = BioinformaticsMethodContract(
    "bounded genomic annotation query",
    MethodKind.EXACT_MODEL,
    ExecutionKind.EXACT_DISCRETE,
    DifferentiationKind.NONE,
    OutputKind.SET,
    conditioning_statement="Features and queries use the same zero-based half-open reference space.",
    truncation_statement="A query with more matches than capacity fails and exposes no prefix.",
    capacity_semantics="row_valid marks occupied fixed-capacity result rows.",
    assumptions=(
        "Feature indices and reference indices use their declared dictionaries.",
    ),
    nondifferentiable_outputs=("rows", "row_valid", "status", "evidence"),
    input_dtype="int64",
    compute_dtype="int64",
    output_dtype="int32",
)

_PARENT_AUDIT_CONTRACT = BioinformaticsMethodContract(
    "feature-parent relation audit",
    MethodKind.EXACT_MODEL,
    ExecutionKind.EXACT_DISCRETE,
    DifferentiationKind.NONE,
    OutputKind.GRAPH,
    conditioning_statement="Every valid edge indexes a row in one finite feature table.",
    truncation_statement="All fixed-capacity relation edges are audited.",
    capacity_semantics="The edge-list capacity is fixed by the supplied sparse relation.",
    assumptions=(),
    nondifferentiable_outputs=("cyclic", "duplicate_edge_count", "status", "evidence"),
    input_dtype="int32",
    compute_dtype="int32",
    output_dtype="bool/int32",
)


def _concrete(value: Array) -> np.ndarray | None:
    if isinstance(value, jax_core.Tracer):
        return None
    return np.asarray(value)


class FeatureParentRelation(StrictModule):
    """Sparse child-to-parent feature rows with optional numeric relation kinds."""

    relation: EdgeRelation
    relation_codes: Array

    def __init__(
        self,
        child_rows: ArrayLike,
        parent_rows: ArrayLike,
        /,
        *,
        feature_count: int,
        valid: ArrayLike | None = None,
        relation_codes: ArrayLike | None = None,
    ):
        children = jnp.asarray(child_rows)
        parents = jnp.asarray(parent_rows)
        if children.ndim != 1 or parents.shape != children.shape:
            raise ValueError("child_rows and parent_rows must be matching vectors.")
        relation = EdgeRelation(
            children,
            parents,
            source_size=int(feature_count),
            target_size=int(feature_count),
            valid=valid,
        )
        if relation_codes is None:
            codes = jnp.zeros(children.shape, dtype=jnp.int32)
        else:
            codes = jnp.asarray(relation_codes)
            if codes.shape != children.shape or not jnp.issubdtype(
                codes.dtype, jnp.integer
            ):
                raise ValueError(
                    "relation_codes must be an integer vector matching edges."
                )
            codes = codes.astype(jnp.int32)
        self.relation = relation
        self.relation_codes = codes

    @property
    def feature_count(self) -> int:
        return self.relation.source_size

    @property
    def capacity(self) -> int:
        return self.relation.capacity


class GenomicAnnotation(StrictModule):
    """Fixed-capacity typed genomic features and their sparse parent relation."""

    features: FeatureDictionary
    reference_indices: Array
    starts: Array
    ends: Array
    strands: Array
    feature_type_ids: Array
    source_ids: Array
    scores: Array
    phases: Array
    valid: Array
    parents: FeatureParentRelation

    def __init__(
        self,
        features: FeatureDictionary,
        reference_indices: ArrayLike,
        starts: ArrayLike,
        ends: ArrayLike,
        strands: ArrayLike,
        feature_type_ids: ArrayLike,
        source_ids: ArrayLike,
        scores: ArrayLike,
        phases: ArrayLike,
        valid: ArrayLike,
        parents: FeatureParentRelation,
        /,
    ):
        if not isinstance(features, FeatureDictionary):
            raise TypeError("features must be a FeatureDictionary.")
        if not isinstance(parents, FeatureParentRelation):
            raise TypeError("parents must be a FeatureParentRelation.")
        arrays = tuple(
            jnp.asarray(value)
            for value in (
                reference_indices,
                starts,
                ends,
                strands,
                feature_type_ids,
                source_ids,
                scores,
                phases,
                valid,
            )
        )
        if any(array.ndim != 1 for array in arrays):
            raise ValueError("GenomicAnnotation fields must be one-dimensional.")
        if not all(array.shape == arrays[0].shape for array in arrays[1:]):
            raise ValueError("GenomicAnnotation fields must have matching shapes.")
        capacity = int(arrays[0].shape[0])
        if features.capacity != capacity:
            raise ValueError("FeatureDictionary capacity must match annotation rows.")
        if parents.feature_count != capacity:
            raise ValueError("Parent relation feature_count must match annotation rows.")
        integer_arrays = arrays[:6] + (arrays[7],)
        if not all(jnp.issubdtype(array.dtype, jnp.integer) for array in integer_arrays):
            raise TypeError(
                "Coordinate, strand, type, source, and phase fields must be integer."
            )
        if not jnp.issubdtype(arrays[6].dtype, jnp.floating):
            raise TypeError(
                "scores must have a floating dtype; use NaN for absent scores."
            )
        if arrays[8].dtype != jnp.bool_:
            raise TypeError("valid must be boolean.")
        refs_h = _concrete(arrays[0])
        starts_h = _concrete(arrays[1])
        ends_h = _concrete(arrays[2])
        strands_h = _concrete(arrays[3])
        types_h = _concrete(arrays[4])
        sources_h = _concrete(arrays[5])
        scores_h = _concrete(arrays[6])
        phases_h = _concrete(arrays[7])
        valid_h = _concrete(arrays[8])
        if (
            refs_h is not None
            and starts_h is not None
            and ends_h is not None
            and strands_h is not None
            and types_h is not None
            and sources_h is not None
            and scores_h is not None
            and phases_h is not None
            and valid_h is not None
        ):
            if np.any(valid_h & ((refs_h < 0) | (starts_h < 0) | (ends_h < starts_h))):
                raise ValueError(
                    "Valid features require a reference and 0 <= start <= end."
                )
            if np.any(valid_h & ~np.isin(strands_h, (-1, 0, 1))):
                raise ValueError("Valid feature strands must be -1, 0, or 1.")
            if np.any(valid_h & ((types_h < 0) | (sources_h < 0))):
                raise ValueError(
                    "Valid feature type and source IDs must be non-negative."
                )
            if np.any(valid_h & ~np.isin(phases_h, (-1, 0, 1, 2))):
                raise ValueError("Feature phases must be -1 (absent), 0, 1, or 2.")
            if np.any(valid_h & ~(np.isfinite(scores_h) | np.isnan(scores_h))):
                raise ValueError("Feature scores must be finite or NaN when absent.")
        self.features = features
        self.reference_indices = arrays[0].astype(jnp.int32)
        self.starts = arrays[1].astype(jnp.int64)
        self.ends = arrays[2].astype(jnp.int64)
        self.strands = arrays[3].astype(jnp.int8)
        self.feature_type_ids = arrays[4].astype(jnp.int32)
        self.source_ids = arrays[5].astype(jnp.int32)
        self.scores = arrays[6]
        self.phases = arrays[7].astype(jnp.int8)
        self.valid = arrays[8]
        self.parents = parents

    @property
    def capacity(self) -> int:
        return int(self.starts.shape[0])

    @property
    def lengths(self) -> Array:
        return jnp.where(self.valid, self.ends - self.starts, 0)


class FeatureQueryResult(StrictModule):
    """Bounded feature rows; capacity overflow never returns a truncated prefix."""

    rows: Array
    row_valid: Array
    valid: Array
    status: Array
    evidence: Array
    method_contract: BioinformaticsMethodContract = eqx.field(static=True)


class FeatureParentAuditResult(StrictModule):
    cyclic: Array
    duplicate_edge_count: Array
    valid: Array
    status: Array
    evidence: Array
    method_contract: BioinformaticsMethodContract = eqx.field(static=True)


def _bounded_rows(mask: Array, capacity: int, /) -> FeatureQueryResult:
    capacity_ = int(capacity)
    if capacity_ < 0:
        raise ValueError("capacity must be non-negative.")
    count = jnp.sum(mask, dtype=jnp.int32)
    overflow = count > capacity_
    rows = jnp.nonzero(mask, size=capacity_, fill_value=0)[0].astype(jnp.int32)
    slot = jnp.arange(capacity_, dtype=jnp.int32) < count
    row_valid = slot & ~overflow
    rows = jnp.where(row_valid, rows, 0)
    success = ~overflow
    status = jnp.where(
        overflow,
        int(AnnotationStatus.CAPACITY_EXCEEDED),
        jnp.where(
            count == 0, int(AnnotationStatus.NO_MATCH), int(AnnotationStatus.SUCCESS)
        ),
    ).astype(jnp.int32)
    evidence = jnp.asarray([count, capacity_, mask.shape[0]], dtype=jnp.int32)
    return FeatureQueryResult(
        rows,
        row_valid,
        success,
        status,
        evidence,
        _ANNOTATION_QUERY_CONTRACT,
    )


def query_overlapping_features(
    annotation: GenomicAnnotation,
    interval: LinearInterval,
    /,
    *,
    capacity: int,
    feature_type_id: int | None = None,
) -> FeatureQueryResult:
    """Query exact half-open overlaps, excluding zero-length intersections."""

    if not isinstance(annotation, GenomicAnnotation):
        raise TypeError("annotation must be a GenomicAnnotation.")
    if not isinstance(interval, LinearInterval):
        raise TypeError("interval must be a LinearInterval.")
    strand_compatible = (
        (annotation.strands == interval.strand)
        | (annotation.strands == int(Strand.UNKNOWN))
        | (interval.strand == int(Strand.UNKNOWN))
    )
    mask = (
        annotation.valid
        & (annotation.reference_indices == interval.reference_index)
        & strand_compatible
        & (annotation.starts < interval.end)
        & (annotation.ends > interval.start)
    )
    if feature_type_id is not None:
        mask = mask & (annotation.feature_type_ids == int(feature_type_id))
    return _bounded_rows(mask, capacity)


def query_feature_parents(
    annotation: GenomicAnnotation,
    feature_row: int,
    /,
    *,
    capacity: int,
) -> FeatureQueryResult:
    """Return every parent route, retaining duplicate and multi-parent edges."""

    if not isinstance(annotation, GenomicAnnotation):
        raise TypeError("annotation must be a GenomicAnnotation.")
    row = int(feature_row)
    if row < 0 or row >= annotation.capacity:
        raise IndexError("feature_row is outside the annotation table.")
    relation = annotation.parents.relation
    mask = relation.valid & (relation.source_indices == row)
    result = _bounded_rows(mask, capacity)
    parent_rows = relation.target_indices[result.rows]
    return FeatureQueryResult(
        jnp.where(result.row_valid, parent_rows, 0),
        result.row_valid,
        result.valid,
        result.status,
        result.evidence,
        result.method_contract,
    )


def query_feature_children(
    annotation: GenomicAnnotation,
    feature_row: int,
    /,
    *,
    capacity: int,
) -> FeatureQueryResult:
    """Return every child route, retaining duplicate and multi-parent edges."""

    if not isinstance(annotation, GenomicAnnotation):
        raise TypeError("annotation must be a GenomicAnnotation.")
    row = int(feature_row)
    if row < 0 or row >= annotation.capacity:
        raise IndexError("feature_row is outside the annotation table.")
    relation = annotation.parents.relation
    mask = relation.valid & (relation.target_indices == row)
    result = _bounded_rows(mask, capacity)
    child_rows = relation.source_indices[result.rows]
    return FeatureQueryResult(
        jnp.where(result.row_valid, child_rows, 0),
        result.row_valid,
        result.valid,
        result.status,
        result.evidence,
        result.method_contract,
    )


def audit_feature_parents(parents: FeatureParentRelation, /) -> FeatureParentAuditResult:
    """Audit all concrete parent edges for duplicate routes and directed cycles."""

    if not isinstance(parents, FeatureParentRelation):
        raise TypeError("parents must be a FeatureParentRelation.")
    relation = parents.relation
    children = _concrete(relation.source_indices)
    parent_rows = _concrete(relation.target_indices)
    valid = _concrete(relation.valid)
    if children is None or parent_rows is None or valid is None:
        raise TypeError("audit_feature_parents requires concrete relation arrays.")
    edges = [
        (int(children[index]), int(parent_rows[index]))
        for index in range(relation.capacity)
        if bool(valid[index])
    ]
    duplicate_count = len(edges) - len(set(edges))
    adjacency: list[list[int]] = [[] for _ in range(parents.feature_count)]
    for child, parent in set(edges):
        adjacency[child].append(parent)
    state = [0] * parents.feature_count
    cyclic = False
    for root in range(parents.feature_count):
        if state[root] != 0:
            continue
        stack: list[tuple[int, int]] = [(root, 0)]
        state[root] = 1
        while stack:
            node, edge_index = stack[-1]
            if edge_index == len(adjacency[node]):
                state[node] = 2
                stack.pop()
                continue
            target = adjacency[node][edge_index]
            stack[-1] = (node, edge_index + 1)
            if state[target] == 1:
                cyclic = True
            elif state[target] == 0:
                state[target] = 1
                stack.append((target, 0))
    status = (
        AnnotationStatus.CYCLIC_PARENT_RELATION
        if cyclic
        else AnnotationStatus.DUPLICATE_RELATION
        if duplicate_count
        else AnnotationStatus.SUCCESS
    )
    evidence = jnp.asarray(
        [len(edges), len(set(edges)), parents.feature_count], dtype=jnp.int32
    )
    return FeatureParentAuditResult(
        jnp.asarray(cyclic),
        jnp.asarray(duplicate_count, dtype=jnp.int32),
        jnp.asarray(not cyclic),
        jnp.asarray(int(status), dtype=jnp.int32),
        evidence,
        _PARENT_AUDIT_CONTRACT,
    )


__all__ = [
    "AnnotationStatus",
    "FeatureParentAuditResult",
    "FeatureParentRelation",
    "FeatureQueryResult",
    "GenomicAnnotation",
    "audit_feature_parents",
    "query_feature_children",
    "query_feature_parents",
    "query_overlapping_features",
]
