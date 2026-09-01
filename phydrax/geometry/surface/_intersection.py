#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import Enum

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._model import SurfaceRealization


class PlaneSectionStatus(str, Enum):
    """Resolved and fail-closed outcomes of a plane-surface section."""

    RESOLVED = "resolved"
    EMPTY = "empty"
    UNRESOLVED_VERTEX_CONTACT = "unresolved_vertex_contact"
    UNRESOLVED_OPEN_CHAIN = "unresolved_open_chain"
    UNRESOLVED_BRANCH = "unresolved_branch"


class PlaneSectionLoop(StrictModule, NonTrainableState):
    """One deterministic closed section loop with segment and edge provenance."""

    points: Array
    source_chart_ids: Array
    source_cell_global_ids: Array
    source_edge_vertex_global_ids: Array
    loop_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        points: ArrayLike,
        source_chart_ids: ArrayLike,
        source_cell_global_ids: ArrayLike,
        source_edge_vertex_global_ids: ArrayLike,
    ):
        points_ = np.asarray(points, dtype=float)
        charts = np.asarray(source_chart_ids, dtype=np.int32)
        cells = np.asarray(source_cell_global_ids, dtype=np.int64)
        edges = np.asarray(source_edge_vertex_global_ids, dtype=np.int64)
        if points_.ndim != 2 or points_.shape[0] < 3 or points_.shape[1] != 3:
            raise ValueError("Plane section loops require at least three 3D points.")
        count = points_.shape[0]
        if (
            charts.shape != (count,)
            or cells.shape != (count,)
            or edges.shape != (count, 2)
        ):
            raise ValueError(
                "Plane section provenance must contain one entry per segment."
            )
        if not np.all(np.isfinite(points_)):
            raise ValueError("Plane section loop points must be finite.")
        if np.any(charts < 0) or np.any(cells < 0) or np.any(edges < 0):
            raise ValueError("Plane section provenance IDs must be non-negative.")
        self.points = jnp.asarray(points_, dtype=float)
        self.source_chart_ids = jnp.asarray(charts, dtype=jnp.int32)
        self.source_cell_global_ids = jnp.asarray(cells, dtype=jnp.int64)
        self.source_edge_vertex_global_ids = jnp.asarray(edges, dtype=jnp.int64)
        self.loop_id = canonical_fingerprint(
            {
                "kind": "plane-section-loop-v1",
                "points": array_tree_fingerprint(points_),
                "source_chart_ids": array_tree_fingerprint(charts),
                "source_cell_global_ids": array_tree_fingerprint(cells),
                "source_edge_vertex_global_ids": array_tree_fingerprint(edges),
            }
        )


class PlaneSectionEvidence(StrictModule, NonTrainableState):
    """Immutable source, plane, tolerance, and route evidence for a section."""

    plane_origin: Array
    plane_normal: Array
    tolerance: Array
    intersected_chart_ids: Array
    intersected_cell_global_ids: Array
    status: PlaneSectionStatus = eqx.field(static=True)
    reason: str = eqx.field(static=True)
    source_realization_id: str = eqx.field(static=True)
    source_geometry_id: str = eqx.field(static=True)
    chart_mapping_id: str = eqx.field(static=True)
    considered_cell_count: int = eqx.field(static=True)
    segment_count: int = eqx.field(static=True)
    loop_count: int = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        plane_origin: ArrayLike,
        plane_normal: ArrayLike,
        tolerance: float,
        intersected_chart_ids: ArrayLike,
        intersected_cell_global_ids: ArrayLike,
        status: PlaneSectionStatus,
        reason: str,
        source_realization_id: str,
        source_geometry_id: str,
        chart_mapping_id: str,
        considered_cell_count: int,
        loop_count: int,
    ):
        origin = np.asarray(plane_origin, dtype=float)
        normal = np.asarray(plane_normal, dtype=float)
        tolerance_ = float(tolerance)
        charts = np.asarray(intersected_chart_ids, dtype=np.int32)
        cells = np.asarray(intersected_cell_global_ids, dtype=np.int64)
        if origin.shape != (3,) or normal.shape != (3,):
            raise ValueError("Plane section evidence requires 3D plane vectors.")
        if not np.all(np.isfinite(origin)) or not np.all(np.isfinite(normal)):
            raise ValueError("Plane section evidence vectors must be finite.")
        if not np.isfinite(tolerance_) or tolerance_ <= 0.0:
            raise ValueError("Plane section tolerance must be positive and finite.")
        if charts.ndim != 1 or cells.shape != charts.shape:
            raise ValueError("Plane section route evidence arrays must share one shape.")
        if not isinstance(status, PlaneSectionStatus):
            raise TypeError("status must be PlaneSectionStatus.")
        identifiers = (
            str(source_realization_id),
            str(source_geometry_id),
            str(chart_mapping_id),
        )
        reason_ = str(reason)
        if any(not value for value in identifiers) or not reason_:
            raise ValueError(
                "Plane section evidence requires complete identities and reason."
            )
        self.plane_origin = jnp.asarray(origin, dtype=float)
        self.plane_normal = jnp.asarray(normal, dtype=float)
        self.tolerance = jnp.asarray(tolerance_, dtype=float)
        self.intersected_chart_ids = jnp.asarray(charts, dtype=jnp.int32)
        self.intersected_cell_global_ids = jnp.asarray(cells, dtype=jnp.int64)
        self.status = status
        self.reason = reason_
        (
            self.source_realization_id,
            self.source_geometry_id,
            self.chart_mapping_id,
        ) = identifiers
        self.considered_cell_count = int(considered_cell_count)
        self.segment_count = int(charts.size)
        self.loop_count = int(loop_count)
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "plane-section-evidence-v1",
                "plane_origin": array_tree_fingerprint(origin),
                "plane_normal": array_tree_fingerprint(normal),
                "tolerance": tolerance_,
                "intersected_chart_ids": array_tree_fingerprint(charts),
                "intersected_cell_global_ids": array_tree_fingerprint(cells),
                "status": status.value,
                "reason": reason_,
                "source_realization_id": identifiers[0],
                "source_geometry_id": identifiers[1],
                "chart_mapping_id": identifiers[2],
                "considered_cell_count": int(considered_cell_count),
                "loop_count": int(loop_count),
            }
        )


class PlaneSurfaceSection(StrictModule, NonTrainableState):
    """A resolved set of loops or an explicit typed unresolved section."""

    loops: tuple[PlaneSectionLoop, ...]
    evidence: PlaneSectionEvidence
    section_id: str = eqx.field(static=True)

    def __init__(
        self,
        loops: tuple[PlaneSectionLoop, ...],
        evidence: PlaneSectionEvidence,
        /,
    ):
        if not all(isinstance(loop, PlaneSectionLoop) for loop in loops):
            raise TypeError("PlaneSurfaceSection loops must be PlaneSectionLoop values.")
        if not isinstance(evidence, PlaneSectionEvidence):
            raise TypeError("PlaneSurfaceSection requires PlaneSectionEvidence.")
        if evidence.status is PlaneSectionStatus.RESOLVED and not loops:
            raise ValueError("Resolved plane sections require at least one loop.")
        if evidence.status is not PlaneSectionStatus.RESOLVED and loops:
            raise ValueError("Unresolved or empty plane sections cannot expose loops.")
        if evidence.loop_count != len(loops):
            raise ValueError("Plane section evidence loop count is inconsistent.")
        self.loops = loops
        self.evidence = evidence
        self.section_id = canonical_fingerprint(
            {
                "kind": "plane-surface-section-v1",
                "evidence_id": evidence.evidence_id,
                "loop_ids": tuple(loop.loop_id for loop in loops),
            }
        )

    @property
    def status(self) -> PlaneSectionStatus:
        return self.evidence.status

    @property
    def resolved(self) -> bool:
        return self.status is PlaneSectionStatus.RESOLVED

    @property
    def unresolved(self) -> bool:
        return self.status not in (PlaneSectionStatus.RESOLVED, PlaneSectionStatus.EMPTY)


def _section_result(
    realization: SurfaceRealization,
    origin: np.ndarray,
    normal: np.ndarray,
    tolerance: float,
    status: PlaneSectionStatus,
    reason: str,
    chart_ids: np.ndarray,
    cell_ids: np.ndarray,
    /,
    *,
    loops: tuple[PlaneSectionLoop, ...] = (),
) -> PlaneSurfaceSection:
    evidence = PlaneSectionEvidence(
        plane_origin=origin,
        plane_normal=normal,
        tolerance=tolerance,
        intersected_chart_ids=chart_ids,
        intersected_cell_global_ids=cell_ids,
        status=status,
        reason=reason,
        source_realization_id=realization.realization_id,
        source_geometry_id=realization.mesh.geometry_id,
        chart_mapping_id=realization.chart_mapping.mapping_id,
        considered_cell_count=realization.audit.cell_count,
        loop_count=len(loops),
    )
    return PlaneSurfaceSection(loops, evidence)


def intersect_plane_surface(
    realization: SurfaceRealization,
    origin: ArrayLike,
    normal: ArrayLike,
    /,
    *,
    tolerance: float | None = None,
) -> PlaneSurfaceSection:
    """Intersect affine surface triangles with a plane, preserving exact routes.

    Plane contact with a source vertex and non-loop section graphs are returned as
    typed unresolved outcomes. No arbitrary perturbation or endpoint welding is used.
    """

    if not isinstance(realization, SurfaceRealization):
        raise TypeError("realization must be a SurfaceRealization.")
    origin_ = np.asarray(origin, dtype=float)
    normal_ = np.asarray(normal, dtype=float)
    if origin_.shape != (3,) or normal_.shape != (3,):
        raise ValueError("Plane origin and normal must have shape (3,).")
    if not np.all(np.isfinite(origin_)) or not np.all(np.isfinite(normal_)):
        raise ValueError("Plane origin and normal must be finite.")
    normal_norm = float(np.linalg.norm(normal_))
    if not np.isfinite(normal_norm) or normal_norm == 0.0:
        raise ValueError("Plane normal must be nonzero.")
    normal_unit = normal_ / normal_norm
    first_normal_axis = int(np.flatnonzero(normal_unit != 0.0)[0])
    if normal_unit[first_normal_axis] < 0.0:
        normal_unit = -normal_unit
    normal_unit = np.where(normal_unit == 0.0, 0.0, normal_unit)
    origin_ = normal_unit * float(np.dot(origin_, normal_unit))
    origin_ = np.where(origin_ == 0.0, 0.0, origin_)
    points = np.asarray(realization.mesh.coordinates, dtype=float)
    scale = max(float(np.linalg.norm(np.ptp(points, axis=0))), 1.0)
    tolerance_ = (
        128.0 * np.finfo(float).eps * scale if tolerance is None else float(tolerance)
    )
    if not np.isfinite(tolerance_) or tolerance_ <= 0.0:
        raise ValueError("Plane intersection tolerance must be positive and finite.")
    signed_distances = (points - origin_) @ normal_unit
    empty_charts = np.zeros((0,), dtype=np.int32)
    empty_cells = np.zeros((0,), dtype=np.int64)
    if np.any(np.abs(signed_distances) <= tolerance_):
        return _section_result(
            realization,
            origin_,
            normal_unit,
            tolerance_,
            PlaneSectionStatus.UNRESOLVED_VERTEX_CONTACT,
            "the tolerance slab contains one or more source vertices",
            empty_charts,
            empty_cells,
        )

    faces = np.asarray(realization.mesh.connectivity.cell_vertices[:, :3], dtype=np.int32)
    vertex_global_ids = np.asarray(realization.mesh.vertex_global_ids, dtype=np.int64)
    chart_cell_ids = np.asarray(realization.chart_mapping.cell_global_ids, dtype=np.int64)
    edge_points: dict[tuple[int, int], np.ndarray] = {}
    edge_global_ids: dict[tuple[int, int], tuple[int, int]] = {}
    segment_edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
    segment_charts: list[int] = []
    segment_cells: list[int] = []
    for chart, face in enumerate(faces):
        face_distances = signed_distances[face]
        if np.all(face_distances > tolerance_) or np.all(face_distances < -tolerance_):
            continue
        crossings: list[tuple[int, int]] = []
        for local in range(3):
            first = int(face[local])
            second = int(face[(local + 1) % 3])
            first_distance = float(signed_distances[first])
            second_distance = float(signed_distances[second])
            if (first_distance > 0.0) == (second_distance > 0.0):
                continue
            edge = (min(first, second), max(first, second))
            if edge not in edge_points:
                start, stop = edge
                start_distance = float(signed_distances[start])
                stop_distance = float(signed_distances[stop])
                parameter = start_distance / (start_distance - stop_distance)
                edge_points[edge] = points[start] + parameter * (
                    points[stop] - points[start]
                )
                edge_global_ids[edge] = tuple(
                    sorted((int(vertex_global_ids[start]), int(vertex_global_ids[stop])))
                )
            crossings.append(edge)
        if len(crossings) != 2 or crossings[0] == crossings[1]:
            charts = np.asarray((*segment_charts, chart), dtype=np.int32)
            cells = chart_cell_ids[charts]
            return _section_result(
                realization,
                origin_,
                normal_unit,
                tolerance_,
                PlaneSectionStatus.UNRESOLVED_BRANCH,
                "a triangle does not induce one unambiguous section segment",
                charts,
                cells,
            )
        segment_edges.append((crossings[0], crossings[1]))
        segment_charts.append(chart)
        segment_cells.append(int(chart_cell_ids[chart]))

    charts = np.asarray(segment_charts, dtype=np.int32)
    cells = np.asarray(segment_cells, dtype=np.int64)
    if not segment_edges:
        return _section_result(
            realization,
            origin_,
            normal_unit,
            tolerance_,
            PlaneSectionStatus.EMPTY,
            "the plane does not intersect the surface",
            charts,
            cells,
        )

    incident_segments: dict[tuple[int, int], list[int]] = {}
    for segment, (first, second) in enumerate(segment_edges):
        incident_segments.setdefault(first, []).append(segment)
        incident_segments.setdefault(second, []).append(segment)
    degrees = tuple(len(values) for values in incident_segments.values())
    if any(degree == 1 for degree in degrees):
        return _section_result(
            realization,
            origin_,
            normal_unit,
            tolerance_,
            PlaneSectionStatus.UNRESOLVED_OPEN_CHAIN,
            "the plane section contains one or more open chains",
            charts,
            cells,
        )
    if any(degree != 2 for degree in degrees):
        return _section_result(
            realization,
            origin_,
            normal_unit,
            tolerance_,
            PlaneSectionStatus.UNRESOLVED_BRANCH,
            "the plane section graph contains a branch or duplicate segment",
            charts,
            cells,
        )

    def node_order(edge: tuple[int, int]) -> tuple[float, float, float, int, int]:
        point = edge_points[edge]
        return (float(point[0]), float(point[1]), float(point[2]), edge[0], edge[1])

    unused = set(range(len(segment_edges)))
    loops: list[PlaneSectionLoop] = []
    while unused:
        active_nodes = {node for segment in unused for node in segment_edges[segment]}
        first_node = min(active_nodes, key=node_order)
        candidates = incident_segments[first_node]
        first_segment = min(
            candidates,
            key=lambda segment: node_order(
                segment_edges[segment][1]
                if segment_edges[segment][0] == first_node
                else segment_edges[segment][0]
            ),
        )
        loop_nodes: list[tuple[int, int]] = []
        loop_segments: list[int] = []
        current_node = first_node
        current_segment = first_segment
        while True:
            if current_segment not in unused:
                return _section_result(
                    realization,
                    origin_,
                    normal_unit,
                    tolerance_,
                    PlaneSectionStatus.UNRESOLVED_BRANCH,
                    "section traversal encountered an already consumed segment",
                    charts,
                    cells,
                )
            unused.remove(current_segment)
            loop_nodes.append(current_node)
            loop_segments.append(current_segment)
            first, second = segment_edges[current_segment]
            next_node = second if first == current_node else first
            if next_node == first_node:
                break
            following = tuple(
                segment
                for segment in incident_segments[next_node]
                if segment != current_segment
            )
            if len(following) != 1:
                return _section_result(
                    realization,
                    origin_,
                    normal_unit,
                    tolerance_,
                    PlaneSectionStatus.UNRESOLVED_BRANCH,
                    "section traversal did not have one continuation",
                    charts,
                    cells,
                )
            current_node = next_node
            current_segment = following[0]
        if len(loop_nodes) < 3:
            return _section_result(
                realization,
                origin_,
                normal_unit,
                tolerance_,
                PlaneSectionStatus.UNRESOLVED_BRANCH,
                "section graph contains a loop with fewer than three segments",
                charts,
                cells,
            )
        loop_points = np.stack(tuple(edge_points[node] for node in loop_nodes))
        loop_chart_ids = np.asarray(
            tuple(segment_charts[segment] for segment in loop_segments),
            dtype=np.int32,
        )
        loop_cell_ids = np.asarray(
            tuple(segment_cells[segment] for segment in loop_segments),
            dtype=np.int64,
        )
        loop_edge_ids = np.asarray(
            tuple(edge_global_ids[node] for node in loop_nodes),
            dtype=np.int64,
        )
        loops.append(
            PlaneSectionLoop(
                points=loop_points,
                source_chart_ids=loop_chart_ids,
                source_cell_global_ids=loop_cell_ids,
                source_edge_vertex_global_ids=loop_edge_ids,
            )
        )
    loops_ = tuple(sorted(loops, key=lambda loop: tuple(np.asarray(loop.points[0]))))
    return _section_result(
        realization,
        origin_,
        normal_unit,
        tolerance_,
        PlaneSectionStatus.RESOLVED,
        "all intersection segments form deterministic closed loops",
        charts,
        cells,
        loops=loops_,
    )


__all__ = [
    "PlaneSectionEvidence",
    "PlaneSectionLoop",
    "PlaneSectionStatus",
    "PlaneSurfaceSection",
    "intersect_plane_surface",
]
