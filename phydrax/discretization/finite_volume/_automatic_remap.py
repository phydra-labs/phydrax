#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Host-side construction of conservative common-refinement artifacts.

The builder in this module deliberately does not attempt to repair a bad
intersection graph.  Candidate discovery and geometric predicates are done on
host NumPy arrays, and a plan is constructed only after both sides of the
measure ledger have been checked independently.
"""

import dataclasses
from enum import IntEnum
from typing import Any, Mapping

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ._unstructured import UnstructuredFiniteVolumeDiscretization
from ._unstructured_remap import UnstructuredConservativeRemapPlan


class UnstructuredConservativeRemapStatus(IntEnum):
    """Fail-closed result of automatic remap artifact construction."""

    SUCCESS = 0
    UNSUPPORTED_GEOMETRY = 1
    INVALID_INPUT = 2
    PREDICATE_UNCERTAIN = 3
    COVERAGE_FAILURE = 4
    RESOURCE_LIMIT = 5
    NUMERICAL_FAILURE = 6


# The longer name is useful to callers that treat the status as belonging to
# the build operation rather than to the resulting plan.
UnstructuredConservativeRemapBuildStatus = UnstructuredConservativeRemapStatus


class UnstructuredConservativeRemapEvidence(StrictModule):
    """Auditable host-side evidence attached to one remap build."""

    target_coverage_defects: Array
    source_coverage_defects: Array
    target_coverage_tolerance: Array
    source_coverage_tolerance: Array
    candidate_count: Array
    accepted_count: Array
    predicate_uncertain_count: Array
    passed: Array
    status: Array
    helper_statuses: tuple[str, ...] = eqx.field(static=True)
    helper_evidence_id: str = eqx.field(static=True)
    resource_evidence_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        target_coverage_defects: Any,
        source_coverage_defects: Any,
        target_coverage_tolerance: Any,
        source_coverage_tolerance: Any,
        candidate_count: int,
        accepted_count: int,
        passed: bool,
        status: UnstructuredConservativeRemapStatus,
        predicate_uncertain_count: int = 0,
        helper_statuses: tuple[str, ...] = (),
        helper_evidence_id: str = "",
        resource_evidence_id: str = "",
        evidence_id: str = "",
    ):
        self.target_coverage_defects = jnp.asarray(target_coverage_defects)
        self.source_coverage_defects = jnp.asarray(source_coverage_defects)
        self.target_coverage_tolerance = jnp.asarray(target_coverage_tolerance)
        self.source_coverage_tolerance = jnp.asarray(source_coverage_tolerance)
        self.candidate_count = jnp.asarray(int(candidate_count), dtype=jnp.int32)
        self.accepted_count = jnp.asarray(int(accepted_count), dtype=jnp.int32)
        self.predicate_uncertain_count = jnp.asarray(
            int(predicate_uncertain_count), dtype=jnp.int32
        )
        self.passed = jnp.asarray(bool(passed))
        self.status = jnp.asarray(int(status), dtype=jnp.int32)
        self.helper_statuses = tuple(str(value) for value in helper_statuses)
        self.helper_evidence_id = str(helper_evidence_id)
        self.resource_evidence_id = str(resource_evidence_id)
        self.evidence_id = str(evidence_id)
        self.status = jnp.asarray(int(status), dtype=jnp.int32)
        self.evidence_id = str(evidence_id)


@dataclasses.dataclass(frozen=True, slots=True)
class UnstructuredConservativeRemapBuildResult:
    """Result of automatic common-refinement generation.

    The CSR arrays are always present, including for a failed build, so a
    caller can persist diagnostics without accidentally applying a partial
    plan.  ``plan`` is non-``None`` only when ``status`` is ``SUCCESS``.
    """

    status: UnstructuredConservativeRemapStatus
    plan: UnstructuredConservativeRemapPlan | None
    target_offsets: Array
    source_indices: Array
    intersection_measures: Array
    target_routes: Array
    pair_source_global_ids: Array
    pair_target_global_ids: Array
    pair_ids: tuple[tuple[int, int], ...]
    target_coverage_defects: Array
    source_coverage_defects: Array
    candidate_count: int
    accepted_count: int
    evidence: UnstructuredConservativeRemapEvidence
    identity: str
    reason: str

    @property
    def build_id(self) -> str:
        return self.identity

    @property
    def artifact_id(self) -> str:
        return self.plan.plan_id if self.plan is not None else self.identity

    @property
    def passed(self) -> bool:
        return self.status is UnstructuredConservativeRemapStatus.SUCCESS

    @property
    def target_coverage_defect(self) -> Array:
        return self.target_coverage_defects

    @property
    def source_coverage_defect(self) -> Array:
        return self.source_coverage_defects


@dataclasses.dataclass(frozen=True, slots=True)
class _Limits:
    max_candidate_pairs: int
    max_accepted_pairs: int
    max_intersection_vertices: int


def _read_limit(limits: Any, names: tuple[str, ...], default: int) -> int:
    if limits is None:
        return default
    if isinstance(limits, Mapping):
        for name in names:
            if name in limits:
                return int(limits[name])
    elif isinstance(limits, (int, np.integer)):
        if names[0] in {"max_candidate_pairs", "max_accepted_pairs"}:
            return int(limits)
    else:
        for name in names:
            if hasattr(limits, name):
                return int(getattr(limits, name))
    return default


def _limits(limits: Any) -> _Limits:
    values = _Limits(
        max_candidate_pairs=_read_limit(
            limits,
            ("max_candidate_pairs", "candidate_pairs", "max_candidates"),
            10_000_000,
        ),
        max_accepted_pairs=_read_limit(
            limits,
            ("max_accepted_pairs", "accepted_pairs", "max_pairs"),
            10_000_000,
        ),
        max_intersection_vertices=_read_limit(
            limits,
            ("max_intersection_vertices", "intersection_vertices"),
            100_000_000,
        ),
    )
    if any(value < 0 for value in dataclasses.astuple(values)):
        raise ValueError("Remap resource limits must be nonnegative integers.")
    return values


def _ids(name: str, value: Any, count: int, fallback: Any) -> np.ndarray:
    raw = np.asarray(fallback if value is None else value)
    if raw.shape != (count,) or raw.dtype.kind not in "iu":
        raise ValueError(f"{name} must contain one integer ID per cell.")
    result = raw.astype(np.int64)
    if np.any(result < 0) or np.unique(result).size != count:
        raise ValueError(f"{name} must contain unique nonnegative IDs.")
    return result


def _result(
    *,
    status: UnstructuredConservativeRemapStatus,
    plan: UnstructuredConservativeRemapPlan | None,
    offsets: np.ndarray,
    indices: np.ndarray,
    measures: np.ndarray,
    target_routes: np.ndarray,
    source_ids: np.ndarray,
    target_ids: np.ndarray,
    pair_ids: tuple[tuple[int, int], ...],
    target_defects: np.ndarray,
    source_defects: np.ndarray,
    candidate_count: int,
    accepted_count: int,
    provenance: str,
    reason: str,
    identity_context: Any = None,
    target_tolerances: np.ndarray | None = None,
    source_tolerances: np.ndarray | None = None,
    predicate_uncertain_count: int = 0,
) -> UnstructuredConservativeRemapBuildResult:
    target_tol = (
        np.zeros_like(target_defects)
        if target_tolerances is None
        else np.asarray(target_tolerances)
    )
    source_tol = (
        np.zeros_like(source_defects)
        if source_tolerances is None
        else np.asarray(source_tolerances)
    )
    evidence_id = canonical_fingerprint(
        {
            "kind": "automatic-unstructured-remap-evidence",
            "status": int(status),
            "candidate_count": int(candidate_count),
            "accepted_count": int(accepted_count),
            "predicate_uncertain_count": int(predicate_uncertain_count),
            "target_defects": array_tree_fingerprint(target_defects),
            "source_defects": array_tree_fingerprint(source_defects),
            "target_tolerances": array_tree_fingerprint(target_tol),
            "source_tolerances": array_tree_fingerprint(source_tol),
            "provenance": str(provenance),
        }
    )
    evidence = UnstructuredConservativeRemapEvidence(
        target_coverage_defects=target_defects,
        source_coverage_defects=source_defects,
        target_coverage_tolerance=target_tol,
        source_coverage_tolerance=source_tol,
        candidate_count=candidate_count,
        accepted_count=accepted_count,
        predicate_uncertain_count=predicate_uncertain_count,
        passed=status is UnstructuredConservativeRemapStatus.SUCCESS,
        status=status,
        evidence_id=evidence_id,
    )
    identity = canonical_fingerprint(
        {
            "kind": "automatic-unstructured-conservative-remap-build",
            "status": int(status),
            "plan": None if plan is None else plan.plan_id,
            "offsets": array_tree_fingerprint(offsets),
            "indices": array_tree_fingerprint(indices),
            "measures": array_tree_fingerprint(measures),
            "source_ids": array_tree_fingerprint(source_ids),
            "target_ids": array_tree_fingerprint(target_ids),
            "provenance": str(provenance),
            "reason": str(reason),
            "context": identity_context,
        }
    )
    return UnstructuredConservativeRemapBuildResult(
        status=status,
        plan=plan,
        target_offsets=jnp.asarray(offsets, dtype=jnp.int32),
        source_indices=jnp.asarray(indices, dtype=jnp.int32),
        intersection_measures=jnp.asarray(measures),
        target_routes=jnp.asarray(target_routes, dtype=jnp.int32),
        pair_source_global_ids=jnp.asarray(source_ids, dtype=jnp.int64),
        pair_target_global_ids=jnp.asarray(target_ids, dtype=jnp.int64),
        pair_ids=pair_ids,
        target_coverage_defects=jnp.asarray(target_defects),
        source_coverage_defects=jnp.asarray(source_defects),
        candidate_count=int(candidate_count),
        accepted_count=int(accepted_count),
        evidence=evidence,
        identity=identity,
        reason=str(reason),
    )


def build_unstructured_conservative_remap(
    source: Any,
    target: Any,
    *,
    tolerance: float,
    limits: Any = None,
    provenance: str,
    source_global_ids: Any = None,
    target_global_ids: Any = None,
) -> UnstructuredConservativeRemapBuildResult:
    """Build a conservative CSR common refinement from certified geometry artifacts."""
    from ...geometry._bvh_overlap import (
        build_host_aabb_overlap_bvh,
        OverlapSearchLimits,
        OverlapSearchStatus,
        query_host_aabb_overlaps,
    )
    from ...geometry._convex_intersections import (
        intersect_convex_polygons,
        IntersectionStatus,
    )
    from ...geometry._tetra_intersections import (
        intersect_tetrahedra,
        TetraIntersectionLimits,
        TetraIntersectionStatus,
        TetraIntersectionTolerance,
    )

    empty_ids = np.empty((0,), dtype=np.int64)
    empty_defects = np.empty((0,), dtype=float)
    empty = np.asarray((0,), dtype=np.int32)
    try:
        tol = float(tolerance)
        if not np.isfinite(tol) or tol <= 0.0:
            raise ValueError("tolerance must be positive and finite")
        prov = str(provenance)
        if not prov:
            raise ValueError("provenance must be non-empty")
        resource_limits = _limits(limits)
    except Exception as error:
        return _result(
            status=UnstructuredConservativeRemapStatus.INVALID_INPUT,
            plan=None,
            offsets=empty,
            indices=np.empty((0,), dtype=np.int32),
            measures=np.empty((0,), dtype=float),
            target_routes=np.empty((0,), dtype=np.int32),
            source_ids=empty_ids,
            target_ids=empty_ids,
            pair_ids=(),
            target_defects=empty_defects,
            source_defects=empty_defects,
            candidate_count=0,
            accepted_count=0,
            provenance=str(provenance),
            reason=str(error),
        )
    if not isinstance(source, UnstructuredFiniteVolumeDiscretization) or not isinstance(
        target, UnstructuredFiniteVolumeDiscretization
    ):
        return _result(
            status=UnstructuredConservativeRemapStatus.UNSUPPORTED_GEOMETRY,
            plan=None,
            offsets=empty,
            indices=np.empty((0,), dtype=np.int32),
            measures=np.empty((0,), dtype=float),
            target_routes=np.empty((0,), dtype=np.int32),
            source_ids=empty_ids,
            target_ids=empty_ids,
            pair_ids=(),
            target_defects=empty_defects,
            source_defects=empty_defects,
            candidate_count=0,
            accepted_count=0,
            provenance=prov,
            reason="source and target must be prepared unstructured FV geometries",
        )
    if (
        source.cell_dimension not in (2, 3)
        or target.cell_dimension != source.cell_dimension
    ):
        return _result(
            status=UnstructuredConservativeRemapStatus.UNSUPPORTED_GEOMETRY,
            plan=None,
            offsets=np.zeros((target.cell_count + 1,), dtype=np.int32),
            indices=np.empty((0,), dtype=np.int32),
            measures=np.empty((0,), dtype=float),
            target_routes=np.empty((0,), dtype=np.int32),
            source_ids=empty_ids,
            target_ids=empty_ids,
            pair_ids=(),
            target_defects=np.empty((target.cell_count,), dtype=float),
            source_defects=np.empty((source.cell_count,), dtype=float),
            candidate_count=0,
            accepted_count=0,
            provenance=prov,
            reason="automatic remap requires matching 2D or 3D affine geometry",
        )
    try:
        source_ids = _ids(
            "source_global_ids",
            source_global_ids,
            source.cell_count,
            source.cell_global_ids,
        )
        target_ids = _ids(
            "target_global_ids",
            target_global_ids,
            target.cell_count,
            target.cell_global_ids,
        )
        vertices_source = np.asarray(source.vertices, dtype=float)
        vertices_target = np.asarray(target.vertices, dtype=float)
        source_cells = (
            [
                vertices_source[cell]
                for cell in np.asarray(source.triangles, dtype=np.int32)
            ]
            + [
                vertices_source[cell]
                for cell in np.asarray(source.quadrilaterals, dtype=np.int32)
            ]
            if source.cell_dimension == 2
            else [
                vertices_source[cell]
                for cell in np.asarray(source.tetrahedra, dtype=np.int32)
            ]
        )
        target_cells = (
            [
                vertices_target[cell]
                for cell in np.asarray(target.triangles, dtype=np.int32)
            ]
            + [
                vertices_target[cell]
                for cell in np.asarray(target.quadrilaterals, dtype=np.int32)
            ]
            if target.cell_dimension == 2
            else [
                vertices_target[cell]
                for cell in np.asarray(target.tetrahedra, dtype=np.int32)
            ]
        )
        source_min = np.asarray(
            [np.min(cell, axis=0) for cell in source_cells], dtype=float
        )
        source_max = np.asarray(
            [np.max(cell, axis=0) for cell in source_cells], dtype=float
        )
        target_min = np.asarray(
            [np.min(cell, axis=0) for cell in target_cells], dtype=float
        )
        target_max = np.asarray(
            [np.max(cell, axis=0) for cell in target_cells], dtype=float
        )
    except Exception as error:
        return _result(
            status=UnstructuredConservativeRemapStatus.INVALID_INPUT,
            plan=None,
            offsets=np.zeros((target.cell_count + 1,), dtype=np.int32),
            indices=np.empty((0,), dtype=np.int32),
            measures=np.empty((0,), dtype=float),
            target_routes=np.empty((0,), dtype=np.int32),
            source_ids=empty_ids,
            target_ids=empty_ids,
            pair_ids=(),
            target_defects=np.zeros((target.cell_count,), dtype=float),
            source_defects=np.zeros((source.cell_count,), dtype=float),
            candidate_count=0,
            accepted_count=0,
            provenance=prov,
            reason=str(error),
        )
    target_bvh = build_host_aabb_overlap_bvh(
        target_min,
        target_max,
        global_ids=target_ids,
        local_ids=np.arange(target.cell_count, dtype=np.int64),
        tolerance=(0.0, 0.0),
    )
    if target_bvh.status is not OverlapSearchStatus.SUCCESS:
        status = (
            UnstructuredConservativeRemapStatus.RESOURCE_LIMIT
            if target_bvh.status
            in {
                OverlapSearchStatus.CANDIDATE_LIMIT,
                OverlapSearchStatus.MEMORY_LIMIT,
                OverlapSearchStatus.TIME_LIMIT,
            }
            else UnstructuredConservativeRemapStatus.INVALID_INPUT
        )
        return _result(
            status=status,
            plan=None,
            offsets=np.zeros((target.cell_count + 1,), dtype=np.int32),
            indices=np.empty((0,), dtype=np.int32),
            measures=np.empty((0,), dtype=float),
            target_routes=np.empty((0,), dtype=np.int32),
            source_ids=empty_ids,
            target_ids=empty_ids,
            pair_ids=(),
            target_defects=np.zeros((target.cell_count,), dtype=float),
            source_defects=np.zeros((source.cell_count,), dtype=float),
            candidate_count=0,
            accepted_count=0,
            provenance=prov,
            reason=f"AABB helper {target_bvh.status.value}: {target_bvh.message}",
            identity_context={"aabb_status": target_bvh.status.value},
        )
    query = query_host_aabb_overlaps(
        target_bvh,
        source_min,
        source_max,
        source_global_ids=source_ids,
        source_local_ids=np.arange(source.cell_count, dtype=np.int64),
        tolerance=(0.0, 0.0),
        limits=OverlapSearchLimits(max_candidates=resource_limits.max_candidate_pairs),
    )
    candidate_count = int(query.candidate_count)
    if query.status is not OverlapSearchStatus.SUCCESS:
        status = (
            UnstructuredConservativeRemapStatus.RESOURCE_LIMIT
            if query.status
            in {
                OverlapSearchStatus.CANDIDATE_LIMIT,
                OverlapSearchStatus.MEMORY_LIMIT,
                OverlapSearchStatus.TIME_LIMIT,
            }
            else UnstructuredConservativeRemapStatus.INVALID_INPUT
        )
        return _result(
            status=status,
            plan=None,
            offsets=np.zeros((target.cell_count + 1,), dtype=np.int32),
            indices=np.empty((0,), dtype=np.int32),
            measures=np.empty((0,), dtype=float),
            target_routes=np.empty((0,), dtype=np.int32),
            source_ids=empty_ids,
            target_ids=empty_ids,
            pair_ids=(),
            target_defects=np.zeros((target.cell_count,), dtype=float),
            source_defects=np.zeros((source.cell_count,), dtype=float),
            candidate_count=candidate_count,
            accepted_count=0,
            provenance=prov,
            reason=f"AABB helper {query.status.value}: {query.message}",
            identity_context={
                "aabb_identity": query.content_identity,
                "aabb_status": query.status.value,
            },
        )
    candidates = [
        (int(tlocal), int(slocal), int(tglobal), int(sglobal))
        for sglobal, tglobal, slocal, tlocal in zip(
            query.source_global_ids,
            query.target_global_ids,
            query.source_local_ids,
            query.target_local_ids,
            strict=True,
        )
    ]
    candidates.sort(key=lambda row: (row[2], row[3], row[0], row[1]))
    source_volume_values = np.asarray(source.cell_volumes)
    target_volume_values = np.asarray(target.cell_volumes)
    ledger_dtype = np.result_type(source_volume_values.dtype, target_volume_values.dtype)
    records: list[tuple[int, int, float]] = []
    predicate_uncertain = 0
    intersection_vertices = 0
    helper_evidence: list[tuple[Any, ...]] = []
    narrow_failure: tuple[UnstructuredConservativeRemapStatus, str] | None = None
    try:
        for target_index, source_index, target_id, source_id in candidates:
            if source.cell_dimension == 2:
                artifact = intersect_convex_polygons(
                    source_cells[source_index],
                    target_cells[target_index],
                    source_pair_id=source_id,
                    target_pair_id=target_id,
                    tolerance=tol,
                )
                evidence = artifact.predicate_evidence
                helper_evidence.append(
                    (
                        "convex",
                        artifact.pair_id,
                        artifact.status.value,
                        evidence.minimum_abs_predicate,
                        evidence.predicate_scale,
                        evidence.tolerance,
                        evidence.evaluated,
                        evidence.uncertain_count,
                    )
                )
                intersection_vertices += int(artifact.vertices.shape[0])
                if intersection_vertices > resource_limits.max_intersection_vertices:
                    raise OverflowError("intersection vertex resource limit exceeded")
                if artifact.status is IntersectionStatus.SUCCESS:
                    measure = artifact.area
                elif artifact.status in {
                    IntersectionStatus.EMPTY,
                    IntersectionStatus.ZERO_MEASURE,
                }:
                    continue
                elif artifact.status is IntersectionStatus.UNCERTAIN_PREDICATE:
                    predicate_uncertain += max(1, int(evidence.uncertain_count))
                    continue
                else:
                    narrow_failure = (
                        UnstructuredConservativeRemapStatus.UNSUPPORTED_GEOMETRY,
                        f"convex helper {artifact.status.value}",
                    )
                    break
            else:
                artifact = intersect_tetrahedra(
                    source_cells[source_index],
                    target_cells[target_index],
                    source_id=source_id,
                    target_id=target_id,
                    tolerance=TetraIntersectionTolerance(relative=tol),
                    limits=TetraIntersectionLimits(
                        max_candidates=max(1, resource_limits.max_intersection_vertices),
                        max_vertices=max(1, resource_limits.max_intersection_vertices),
                        max_faces=16,
                    ),
                    volume_only=True,
                )
                evidence = artifact.evidence
                helper_evidence.append(
                    (
                        "tetra",
                        artifact.pair_id,
                        artifact.status.value,
                        evidence.candidate_count,
                        evidence.vertex_count,
                        evidence.face_count,
                        evidence.volume_error,
                        evidence.predicate_uncertain,
                    )
                )
                intersection_vertices += int(evidence.vertex_count)
                if intersection_vertices > resource_limits.max_intersection_vertices:
                    raise OverflowError("intersection vertex resource limit exceeded")
                if artifact.status is TetraIntersectionStatus.SUCCESS:
                    measure = artifact.volume
                elif artifact.status in {
                    TetraIntersectionStatus.DISJOINT,
                    TetraIntersectionStatus.ZERO_MEASURE_CONTACT,
                }:
                    continue
                elif artifact.status is TetraIntersectionStatus.UNCERTAIN_PREDICATE:
                    predicate_uncertain += 1
                    continue
                elif artifact.status is TetraIntersectionStatus.CANDIDATE_LIMIT:
                    raise OverflowError("tetra helper candidate limit exceeded")
                else:
                    narrow_failure = (
                        UnstructuredConservativeRemapStatus.UNSUPPORTED_GEOMETRY,
                        f"tetra helper {artifact.status.value}",
                    )
                    break
            measure = np.asarray(measure, dtype=ledger_dtype).item()
            if not np.isfinite(measure) or measure <= 0.0:
                raise FloatingPointError(
                    "intersection helper returned non-positive/non-finite measure"
                )
            scale = max(
                min(
                    float(source_volume_values[source_index]),
                    float(target_volume_values[target_index]),
                ),
                np.finfo(float).tiny,
            )
            if measure / scale < tol:
                predicate_uncertain += 1
                continue
            records.append((target_index, source_index, measure))
            if len(records) > resource_limits.max_accepted_pairs:
                raise OverflowError("accepted-pair resource limit exceeded")
    except OverflowError as error:
        return _result(
            status=UnstructuredConservativeRemapStatus.RESOURCE_LIMIT,
            plan=None,
            offsets=np.zeros((target.cell_count + 1,), dtype=np.int32),
            indices=np.empty((0,), dtype=np.int32),
            measures=np.empty((0,), dtype=float),
            target_routes=np.empty((0,), dtype=np.int32),
            source_ids=empty_ids,
            target_ids=empty_ids,
            pair_ids=(),
            target_defects=np.zeros((target.cell_count,), dtype=float),
            source_defects=np.zeros((source.cell_count,), dtype=float),
            candidate_count=candidate_count,
            accepted_count=0,
            provenance=prov,
            reason=str(error),
            identity_context={
                "aabb_identity": query.content_identity,
                "helper_evidence": helper_evidence,
            },
            predicate_uncertain_count=predicate_uncertain,
        )
    except Exception as error:
        return _result(
            status=UnstructuredConservativeRemapStatus.NUMERICAL_FAILURE,
            plan=None,
            offsets=np.zeros((target.cell_count + 1,), dtype=np.int32),
            indices=np.empty((0,), dtype=np.int32),
            measures=np.empty((0,), dtype=float),
            target_routes=np.empty((0,), dtype=np.int32),
            source_ids=empty_ids,
            target_ids=empty_ids,
            pair_ids=(),
            target_defects=np.zeros((target.cell_count,), dtype=float),
            source_defects=np.zeros((source.cell_count,), dtype=float),
            candidate_count=candidate_count,
            accepted_count=0,
            provenance=prov,
            reason=str(error),
            identity_context={
                "aabb_identity": query.content_identity,
                "helper_evidence": helper_evidence,
            },
            predicate_uncertain_count=predicate_uncertain,
        )
    records.sort(
        key=lambda record: (int(target_ids[record[0]]), int(source_ids[record[1]]))
    )
    target_coverage = np.zeros((target.cell_count,), dtype=float)
    source_coverage = np.zeros((source.cell_count,), dtype=float)
    for target_index, source_index, measure in records:
        target_coverage[target_index] += measure
        source_coverage[source_index] += measure
    target_volumes = np.asarray(target.cell_volumes, dtype=float)
    source_volumes = np.asarray(source.cell_volumes, dtype=float)
    target_defects = target_coverage - target_volumes
    source_defects = source_coverage - source_volumes
    target_scale = np.maximum(target_volumes, np.finfo(float).tiny)
    source_scale = np.maximum(source_volumes, np.finfo(float).tiny)
    coverage_ok = bool(
        np.all(np.isfinite(target_defects))
        and np.all(np.isfinite(source_defects))
        and np.all(np.abs(target_defects) <= tol * target_scale)
        and np.all(np.abs(source_defects) <= tol * source_scale)
    )
    if narrow_failure is not None:
        status, reason = narrow_failure
    elif predicate_uncertain:
        status = UnstructuredConservativeRemapStatus.PREDICATE_UNCERTAIN
        reason = (
            "one or more helper predicates are uncertain or below predicate tolerance"
        )
    elif not coverage_ok:
        status = UnstructuredConservativeRemapStatus.COVERAGE_FAILURE
        reason = "source or target coverage is incomplete or over-covered"
    else:
        status = UnstructuredConservativeRemapStatus.SUCCESS
        reason = "complete source and target conservative coverage"
    offsets = np.zeros((target.cell_count + 1,), dtype=np.int32)
    for target_index, _, _ in records:
        offsets[target_index + 1] += 1
    np.cumsum(offsets, out=offsets)
    indices = np.asarray([source_index for _, source_index, _ in records], dtype=np.int32)
    measures = np.asarray([measure for _, _, measure in records], dtype=float)
    target_routes = np.asarray(
        [target_index for target_index, _, _ in records], dtype=np.int32
    )
    pair_source_ids = np.asarray(
        [source_ids[source_index] for _, source_index, _ in records], dtype=np.int64
    )
    pair_target_ids = np.asarray(
        [target_ids[target_index] for target_index, _, _ in records], dtype=np.int64
    )
    pair_ids = tuple(
        (int(target_id), int(source_id))
        for target_id, source_id in zip(pair_target_ids, pair_source_ids, strict=True)
    )
    plan = None
    if status is UnstructuredConservativeRemapStatus.SUCCESS:
        try:
            plan = UnstructuredConservativeRemapPlan(
                source,
                target,
                offsets,
                indices,
                measures,
                method="public-geometry-common-refinement",
                provenance=prov,
                tolerance=tol,
                require_complete=True,
            )
        except Exception as error:
            status = UnstructuredConservativeRemapStatus.COVERAGE_FAILURE
            reason = f"remap plan constructor rejected validated ledger: {error}"
    return _result(
        status=status,
        plan=plan,
        offsets=offsets,
        indices=indices,
        measures=measures,
        target_routes=target_routes,
        source_ids=pair_source_ids,
        target_ids=pair_target_ids,
        pair_ids=pair_ids,
        target_defects=target_defects,
        source_defects=source_defects,
        candidate_count=candidate_count,
        identity_context={
            "tolerance": tol,
            "limits": dataclasses.asdict(resource_limits),
            "aabb_identity": query.content_identity,
            "helper_evidence": helper_evidence,
        },
        accepted_count=len(records),
        provenance=prov,
        reason=reason,
        target_tolerances=tol * target_scale,
        source_tolerances=tol * source_scale,
        predicate_uncertain_count=predicate_uncertain,
    )


__all__ = [
    "UnstructuredConservativeRemapBuildResult",
    "UnstructuredConservativeRemapBuildStatus",
    "UnstructuredConservativeRemapEvidence",
    "UnstructuredConservativeRemapStatus",
    "build_unstructured_conservative_remap",
]
