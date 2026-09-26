#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact-predicate Delaunay, constrained Delaunay, Voronoi, and power diagrams.

All constructions are host-only immutable preparation executed by the native
meshcore library (adaptive expansion arithmetic with index-ordered symbolic
perturbation).  Every combinatorial decision is exact for inputs inside the
meshcore exact domain, results are canonically ordered and deterministic, and
the native status plus library identity is retained as
:class:`TriangulationEvidence`.

Voronoi and power cells are the generator's (weighted) bisector halfspaces
towards its Delaunay/regular-triangulation neighbors, intersected with an
axis-aligned box and optional convex-domain halfspaces ``{x : n . x <= h}``.
The two cells sharing a bisector use exactly opposite plane coefficients, and
every vertex classification in the clipper is exact, so cells cover the domain
without overlap up to the rounding of constructed vertex coordinates.  Cells are
stored as polygon/polyhedron CSR (:class:`DiagramCells`) with measures and
centroids.
"""

from __future__ import annotations

import math

import equinox as eqx
import numpy as np

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._geometry_predicates import orient2d, orient3d, PredicateMode
from .._meshcore import (
    clip_box_halfplanes,
    clip_box_halfspaces,
    constrained_delaunay_2d,
    delaunay_2d,
    delaunay_3d,
    meshcore_identity,
    MeshcoreError,
    MeshcoreStatus,
    regular_2d,
    regular_3d,
)
from .._strict import StrictModule
from .._trainable import NonTrainableState


_DEGENERATE_ALL_PAIRS_LIMIT = 512
_CLIP_PLANE_BUDGET = 1 << 18


class TriangulationEvidence(StrictModule, NonTrainableState):
    """Route, native status, library identity, and counts of a triangulation."""

    route: str = eqx.field(static=True)
    status: str = eqx.field(static=True)
    predicate_mode: PredicateMode = eqx.field(static=True)
    meshcore_identity: str = eqx.field(static=True)
    input_point_count: int = eqx.field(static=True)
    vertex_count: int = eqx.field(static=True)
    duplicate_count: int = eqx.field(static=True)
    redundant_count: int = eqx.field(static=True)
    steiner_count: int = eqx.field(static=True)
    simplex_count: int = eqx.field(static=True)
    minimum_angle_degrees: float = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        route: str,
        status: MeshcoreStatus,
        input_point_count: int,
        vertex_count: int,
        duplicate_count: int,
        redundant_count: int,
        steiner_count: int,
        simplex_count: int,
        minimum_angle_degrees: float,
        content: dict,
    ):
        if not isinstance(status, MeshcoreStatus):
            raise TypeError("status must be a MeshcoreStatus.")
        counts = (
            input_point_count,
            vertex_count,
            duplicate_count,
            redundant_count,
            steiner_count,
            simplex_count,
        )
        if any(count < 0 for count in counts):
            raise ValueError("Triangulation counts must be nonnegative.")
        identity = meshcore_identity()
        self.route = route
        self.status = status.name.lower()
        self.predicate_mode = PredicateMode.EXACT
        self.meshcore_identity = identity
        self.input_point_count = input_point_count
        self.vertex_count = vertex_count
        self.duplicate_count = duplicate_count
        self.redundant_count = redundant_count
        self.steiner_count = steiner_count
        self.simplex_count = simplex_count
        self.minimum_angle_degrees = float(minimum_angle_degrees)
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "triangulation-evidence",
                "route": route,
                "status": self.status,
                "meshcore": identity,
                "counts": list(counts),
                "content": content,
            }
        )


# ------------------------------------------------------------------ helpers


def _point_array(points: object, name: str, dimensions: tuple[int, ...], /) -> np.ndarray:
    array = np.asarray(points)
    if np.issubdtype(array.dtype, np.bool_) or not (
        np.issubdtype(array.dtype, np.floating) or np.issubdtype(array.dtype, np.integer)
    ):
        raise TypeError(f"{name} must be a real floating or integer array.")
    if array.ndim != 2 or array.shape[1] not in dimensions:
        shapes = " or ".join(f"(n, {dimension})" for dimension in dimensions)
        raise ValueError(f"{name} must have shape {shapes}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite.")
    converted = np.array(array, dtype=np.float64, copy=True)
    converted.setflags(write=False)
    return converted


def _frozen(array: np.ndarray, /) -> np.ndarray:
    result = np.ascontiguousarray(array)
    result.setflags(write=False)
    return result


def _minimum_angle_degrees(points: np.ndarray, triangles: np.ndarray, /) -> float:
    if triangles.shape[0] == 0:
        return math.nan
    corners = points[triangles]
    angles = []
    for vertex in range(3):
        first = corners[:, (vertex + 1) % 3] - corners[:, vertex]
        second = corners[:, (vertex + 2) % 3] - corners[:, vertex]
        cross = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
        dot = np.sum(first * second, axis=1)
        angles.append(np.arctan2(np.abs(cross), dot))
    return float(np.degrees(np.min(np.stack(angles, axis=1))))


def _spans_space(points: np.ndarray, /) -> bool:
    """Exact test that the points affinely span their ambient space."""

    dimension = points.shape[1]
    if points.shape[0] <= dimension:
        return False
    origin = points[0]
    distinct = np.flatnonzero(np.any(points != origin, axis=1))
    if distinct.size == 0:
        return False
    second = points[distinct[0]]
    if dimension == 2:
        turns = orient2d(origin, second, points, mode=PredicateMode.EXACT)
        return bool(np.any(turns.signs != 0))
    collinear = np.ones(points.shape[0], dtype=np.bool_)
    for axes in ((1, 2), (2, 0), (0, 1)):
        turns = orient2d(
            origin[list(axes)],
            second[list(axes)],
            points[:, list(axes)],
            mode=PredicateMode.EXACT,
        )
        collinear &= turns.signs == 0
    off_line = np.flatnonzero(~collinear)
    if off_line.size == 0:
        return False
    third = points[off_line[0]]
    volumes = orient3d(origin, second, third, points, mode=PredicateMode.EXACT)
    return bool(np.any(volumes.signs != 0))


def _simplex_edges(simplices: np.ndarray, /) -> np.ndarray:
    width = simplices.shape[1]
    pairs = [
        simplices[:, [first, second]]
        for first in range(width)
        for second in range(first + 1, width)
    ]
    if not pairs or simplices.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.int64)
    edges = np.sort(np.concatenate(pairs, axis=0).astype(np.int64), axis=1)
    return np.unique(edges, axis=0)


def _raise_for_items(status: np.ndarray, operation: str, /) -> None:
    failed = np.flatnonzero(status != MeshcoreStatus.OK)
    if failed.size == 0:
        return
    code = MeshcoreStatus(int(status[failed[0]]))
    match code:
        case MeshcoreStatus.CAPACITY_EXCEEDED | MeshcoreStatus.INTERNAL_ERROR:
            raise MeshcoreError(code, f"{operation}: cell {int(failed[0])} failed")
        case _:
            raise ValueError(
                f"{operation}: cell {int(failed[0])} has invalid input ({code.name})."
            )


# ------------------------------------------------------------------ triangulations


class DelaunayTriangulation(StrictModule, NonTrainableState):
    """Exact Delaunay triangulation (2D) or tetrahedralization (3D).

    ``simplices`` are positively oriented and canonically ordered; cocircular
    and cospherical ties are resolved by index-ordered symbolic perturbation.
    ``vertex_map`` sends each input point to its triangulation vertex (the
    smallest index of identical points).
    """

    points: np.ndarray
    simplices: np.ndarray
    vertex_map: np.ndarray
    evidence: TriangulationEvidence

    def __init__(self, points: object, /, *, max_simplices: int | None = None):
        point_array = _point_array(points, "points", (2, 3))
        dimension = point_array.shape[1]
        if dimension == 2:
            simplices, vertex_map = delaunay_2d(point_array, max_triangles=max_simplices)
        else:
            simplices, vertex_map = delaunay_3d(point_array, max_tetrahedra=max_simplices)
        count = point_array.shape[0]
        duplicates = int(np.count_nonzero(vertex_map != np.arange(count)))
        self.points = point_array
        self.simplices = _frozen(simplices)
        self.vertex_map = _frozen(vertex_map)
        self.evidence = TriangulationEvidence(
            route=f"delaunay_{dimension}d",
            status=MeshcoreStatus.OK,
            input_point_count=count,
            vertex_count=count - duplicates,
            duplicate_count=duplicates,
            redundant_count=0,
            steiner_count=0,
            simplex_count=simplices.shape[0],
            minimum_angle_degrees=(
                _minimum_angle_degrees(point_array, simplices)
                if dimension == 2
                else math.nan
            ),
            content={
                "points": array_tree_fingerprint(point_array),
                "simplices": array_tree_fingerprint(simplices),
            },
        )

    @property
    def dimension(self) -> int:
        return self.points.shape[1]


class ConstrainedDelaunayTriangulation(StrictModule, NonTrainableState):
    """Constrained Delaunay triangulation refined by Ruppert/Chew insertion.

    ``points`` holds the input points followed by Steiner points; every input
    segment is a union of mesh edges, and ``segment_ids[t, k]`` is the input
    segment carrying the edge opposite vertex ``k`` of triangle ``t`` (or -1).
    ``evidence.status`` is ``ok`` or ``refinement_limit`` (valid conforming mesh
    whose quality targets were not met within ``max_steiner`` insertions).
    """

    points: np.ndarray
    triangles: np.ndarray
    segment_ids: np.ndarray
    input_point_count: int = eqx.field(static=True)
    evidence: TriangulationEvidence

    def __init__(
        self,
        points: object,
        segments: object,
        /,
        *,
        holes: object | None = None,
        keep_convex_hull: bool = False,
        min_angle: float = 0.0,
        max_area: float = math.inf,
        max_steiner: int = 0,
        max_triangles: int | None = None,
    ):
        point_array = _point_array(points, "points", (2,))
        mesh_points, triangles, segment_ids, status = constrained_delaunay_2d(
            point_array,
            segments,
            holes=holes,
            keep_convex_hull=keep_convex_hull,
            min_angle=min_angle,
            max_area=max_area,
            max_steiner=max_steiner,
            max_triangles=max_triangles,
        )
        count = point_array.shape[0]
        used = np.unique(triangles)
        self.points = _frozen(mesh_points)
        self.triangles = _frozen(triangles)
        self.segment_ids = _frozen(segment_ids)
        self.input_point_count = count
        self.evidence = TriangulationEvidence(
            route="constrained_delaunay_2d",
            status=status,
            input_point_count=count,
            vertex_count=used.size,
            duplicate_count=count - np.unique(point_array, axis=0).shape[0],
            redundant_count=0,
            steiner_count=mesh_points.shape[0] - count,
            simplex_count=triangles.shape[0],
            minimum_angle_degrees=_minimum_angle_degrees(mesh_points, triangles),
            content={
                "points": array_tree_fingerprint(mesh_points),
                "triangles": array_tree_fingerprint(triangles),
                "segment_ids": array_tree_fingerprint(segment_ids),
            },
        )


# ------------------------------------------------------------------ diagrams


class DiagramCells(StrictModule, NonTrainableState):
    """Convex cells as polygon/polyhedron CSR with measures and centroids.

    Cell ``i`` owns vertex rows ``cell_vertex_offsets[i]:cell_vertex_offsets[i+1]``
    and faces ``cell_face_offsets[i]:cell_face_offsets[i+1]``.  Face ``f`` lists
    global vertex rows ``face_vertices[face_vertex_offsets[f]:face_vertex_offsets
    [f+1]]`` counterclockwise seen from outside (2D faces are the directed cell
    edges).  ``face_labels`` hold the neighboring generator index (>= 0) or
    ``-(1 + boundary)`` with boundary ``2 axis + side`` for box sides and
    ``2 dimension + k`` for convex-domain halfspace ``k``.  Empty cells have no
    vertices, zero measure, and NaN centroids.
    """

    vertices: np.ndarray
    cell_vertex_offsets: np.ndarray
    cell_face_offsets: np.ndarray
    face_vertex_offsets: np.ndarray
    face_vertices: np.ndarray
    face_labels: np.ndarray
    measures: np.ndarray
    centroids: np.ndarray

    def __init__(
        self,
        vertices: np.ndarray,
        cell_vertex_offsets: np.ndarray,
        cell_face_offsets: np.ndarray,
        face_vertex_offsets: np.ndarray,
        face_vertices: np.ndarray,
        face_labels: np.ndarray,
        measures: np.ndarray,
        centroids: np.ndarray,
    ):
        cell_count = measures.shape[0]
        if cell_vertex_offsets.shape != (cell_count + 1,) or cell_face_offsets.shape != (
            cell_count + 1,
        ):
            raise ValueError("Cell offsets must have shape (cells + 1,).")
        if face_vertex_offsets.shape != (face_labels.shape[0] + 1,):
            raise ValueError("Face offsets must have shape (faces + 1,).")
        if centroids.shape != (cell_count, vertices.shape[1]):
            raise ValueError("Centroids must have shape (cells, dimension).")
        self.vertices = _frozen(vertices)
        self.cell_vertex_offsets = _frozen(cell_vertex_offsets)
        self.cell_face_offsets = _frozen(cell_face_offsets)
        self.face_vertex_offsets = _frozen(face_vertex_offsets)
        self.face_vertices = _frozen(face_vertices)
        self.face_labels = _frozen(face_labels)
        self.measures = _frozen(measures)
        self.centroids = _frozen(centroids)

    @property
    def cell_count(self) -> int:
        return self.measures.shape[0]


def _ragged_order(counts: np.ndarray, order: np.ndarray, /) -> np.ndarray:
    """Flat indices that reorder ragged segments of ``counts`` into ``order``."""

    starts = np.concatenate(([0], np.cumsum(counts)[:-1])).astype(np.int64)
    selected = counts[order]
    total = int(np.sum(selected))
    within = np.arange(total, dtype=np.int64) - np.repeat(
        np.concatenate(([0], np.cumsum(selected)[:-1])).astype(np.int64), selected
    )
    return np.repeat(starts[order], selected) + within


def _offsets(counts: np.ndarray, /) -> np.ndarray:
    return np.concatenate(([0], np.cumsum(counts))).astype(np.int64)


class _Planes:
    """Halfspace CSR per cell with global face labels."""

    __slots__ = ("labels", "normals", "offsets", "values")

    def __init__(
        self,
        offsets: np.ndarray,
        normals: np.ndarray,
        values: np.ndarray,
        labels: np.ndarray,
    ):
        self.offsets = offsets
        self.normals = normals
        self.values = values
        self.labels = labels


def _cell_planes(
    points: np.ndarray,
    weights: np.ndarray | None,
    neighbors: np.ndarray,
    domain_normals: np.ndarray,
    domain_offsets: np.ndarray,
    active: np.ndarray,
    /,
) -> _Planes:
    """Bisector halfspaces towards neighbors plus the domain halfspaces.

    The (power) bisector of ``i`` towards ``j`` is ``(p_j - p_i) . x <=
    (p_j - p_i) . (p_i + p_j) / 2 - (w_j - w_i) / 2``; the reverse direction
    evaluates to the exact negation, so neighboring cells share one plane.
    """

    count, dimension = points.shape
    directed = np.concatenate((neighbors, neighbors[:, ::-1]), axis=0)
    order = np.lexsort((directed[:, 1], directed[:, 0]))
    directed = directed[order]
    owner = directed[:, 0]
    other = directed[:, 1]
    normal = points[other] - points[owner]
    midpoint = 0.5 * (points[owner] + points[other])
    value = np.sum(normal * midpoint, axis=1)
    if weights is not None:
        value = value - 0.5 * (weights[other] - weights[owner])
    neighbor_counts = np.bincount(owner, minlength=count).astype(np.int64)
    domain_count = domain_normals.shape[0]
    counts = np.where(active, neighbor_counts + domain_count, 0)
    offsets = _offsets(counts)
    total = int(offsets[-1])
    normals = np.zeros((total, dimension), dtype=np.float64)
    values = np.zeros((total,), dtype=np.float64)
    labels = np.zeros((total,), dtype=np.int32)
    keep = active[owner]
    neighbor_starts = _offsets(neighbor_counts)
    local = np.arange(owner.shape[0], dtype=np.int64) - neighbor_starts[owner]
    rows = offsets[owner[keep]] + local[keep]
    normals[rows] = normal[keep]
    values[rows] = value[keep]
    labels[rows] = other[keep].astype(np.int32)
    if domain_count:
        cells = np.flatnonzero(active)
        base = offsets[cells] + neighbor_counts[cells]
        rows = (base[:, None] + np.arange(domain_count)[None, :]).reshape((-1,))
        normals[rows] = np.tile(domain_normals, (cells.size, 1))
        values[rows] = np.tile(domain_offsets, cells.size)
        labels[rows] = np.tile(
            -(1 + 2 * dimension + np.arange(domain_count, dtype=np.int32)), cells.size
        )
    return _Planes(offsets, normals, values, labels)


def _clip_chunk(
    box_lower: np.ndarray,
    box_upper: np.ndarray,
    planes: _Planes,
    cells: np.ndarray,
    /,
) -> tuple[np.ndarray, ...]:
    """Clip one chunk of cells; returns ragged per-cell vertices and faces."""

    dimension = box_lower.shape[0]
    counts = (planes.offsets[cells + 1] - planes.offsets[cells]).astype(np.int64)
    width = max(1, int(np.max(counts, initial=0)))
    total = int(np.sum(counts))
    row = np.repeat(np.arange(cells.size, dtype=np.int64), counts)
    column = np.arange(total, dtype=np.int64) - np.repeat(_offsets(counts)[:-1], counts)
    source = np.repeat(planes.offsets[cells], counts) + column
    normals = np.zeros((cells.size, width, dimension), dtype=np.float64)
    values = np.zeros((cells.size, width), dtype=np.float64)
    normals[row, column] = planes.normals[source]
    values[row, column] = planes.values[source]
    label_table = np.zeros((cells.size, width), dtype=np.int32)
    label_table[row, column] = planes.labels[source]
    if dimension == 2:
        vertices, edge_labels, vertex_counts, measure, moment, status = (
            clip_box_halfplanes(box_lower, box_upper, normals, values, counts)
        )
        _raise_for_items(status, "VoronoiDiagram")
        mask = np.arange(vertices.shape[1])[None, :] < vertex_counts[:, None]
        flat_vertices = vertices[mask]
        local = edge_labels[mask]
        cell_rows = np.repeat(np.arange(cells.size), vertex_counts)
        face_labels = np.where(
            local >= 0, label_table[cell_rows, np.maximum(local, 0)], local
        ).astype(np.int32)
        position = np.arange(flat_vertices.shape[0]) - np.repeat(
            _offsets(vertex_counts)[:-1], vertex_counts
        )
        following = np.where(
            position + 1 < np.repeat(vertex_counts, vertex_counts), position + 1, 0
        )
        face_local = np.stack((position, following), axis=1).reshape((-1,))
        face_sizes = np.full(flat_vertices.shape[0], 2, dtype=np.int64)
        face_counts = vertex_counts.astype(np.int64)
        return (
            vertex_counts.astype(np.int64),
            flat_vertices,
            face_counts,
            face_sizes,
            face_labels,
            face_local.astype(np.int64),
            measure,
            moment,
        )
    (
        vertices,
        vertex_counts,
        face_offsets,
        face_label_table,
        face_vertices,
        face_counts,
        measure,
        moment,
        status,
    ) = clip_box_halfspaces(box_lower, box_upper, normals, values, counts)
    _raise_for_items(status, "VoronoiDiagram")
    vertex_mask = np.arange(vertices.shape[1])[None, :] < vertex_counts[:, None]
    face_mask = np.arange(face_label_table.shape[1])[None, :] < face_counts[:, None]
    local = face_label_table[face_mask]
    cell_rows = np.repeat(np.arange(cells.size), face_counts)
    face_labels = np.where(
        local >= 0, label_table[cell_rows, np.maximum(local, 0)], local
    ).astype(np.int32)
    face_sizes = np.diff(face_offsets, axis=1)[face_mask].astype(np.int64)
    used = face_offsets[np.arange(cells.size), face_counts]
    incidence_mask = np.arange(face_vertices.shape[1])[None, :] < used[:, None]
    return (
        vertex_counts.astype(np.int64),
        vertices[vertex_mask],
        face_counts.astype(np.int64),
        face_sizes,
        face_labels,
        face_vertices[incidence_mask].astype(np.int64),
        measure,
        moment,
    )


def _diagram_cells(
    box_lower: np.ndarray, box_upper: np.ndarray, planes: _Planes, active: np.ndarray, /
) -> DiagramCells:
    count = planes.offsets.shape[0] - 1
    dimension = box_lower.shape[0]
    plane_counts = np.diff(planes.offsets)
    # Only active generators own cells.  They are clipped in chunks of similar
    # plane count so the padded working set stays within an explicit budget.
    active_cells = np.flatnonzero(active)
    by_count = active_cells[np.argsort(plane_counts[active_cells], kind="stable")]
    chunks: list[np.ndarray] = []
    start = 0
    while start < by_count.size:
        stop = start + 1
        while (
            stop < by_count.size
            and (stop + 1 - start) * max(1, int(plane_counts[by_count[stop]]))
            <= _CLIP_PLANE_BUDGET
        ):
            stop += 1
        chunks.append(by_count[start:stop])
        start = stop
    parts = [_clip_chunk(box_lower, box_upper, planes, chunk) for chunk in chunks]
    inactive = np.flatnonzero(~active)
    empty = np.zeros((inactive.size,), dtype=np.int64)
    parts.append(
        (
            empty,
            np.zeros((0, dimension), dtype=np.float64),
            empty,
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((inactive.size,), dtype=np.float64),
            np.zeros((inactive.size, dimension), dtype=np.float64),
        )
    )
    chunk_cells = np.concatenate(chunks + [inactive])
    order = np.argsort(chunk_cells, kind="stable")
    vertex_counts = np.concatenate([part[0] for part in parts])
    vertices = np.concatenate([part[1] for part in parts]).reshape((-1, dimension))
    face_counts = np.concatenate([part[2] for part in parts])
    face_sizes = np.concatenate([part[3] for part in parts])
    face_labels = np.concatenate([part[4] for part in parts])
    face_local = np.concatenate([part[5] for part in parts])
    measures = np.concatenate([part[6] for part in parts])
    moments = np.concatenate([part[7] for part in parts]).reshape((-1, dimension))

    vertex_index = _ragged_order(vertex_counts, order)
    face_index = _ragged_order(face_counts, order)
    incidence_counts_by_face = face_sizes
    incidence_index = _ragged_order(incidence_counts_by_face, face_index)
    vertex_counts_ordered = vertex_counts[order]
    face_counts_ordered = face_counts[order]
    cell_vertex_offsets = _offsets(vertex_counts_ordered)
    cell_face_offsets = _offsets(face_counts_ordered)
    ordered_face_sizes = face_sizes[face_index]
    face_cells = np.repeat(np.arange(count, dtype=np.int64), face_counts_ordered)
    incidence_cells = np.repeat(face_cells, ordered_face_sizes)
    face_vertices = cell_vertex_offsets[incidence_cells] + face_local[incidence_index]
    measures_ordered = measures[order]
    moments_ordered = moments[order]
    centroids = np.full((count, dimension), np.nan, dtype=np.float64)
    positive = measures_ordered > 0.0
    centroids[positive] = moments_ordered[positive] / measures_ordered[positive, None]
    return DiagramCells(
        vertices[vertex_index],
        cell_vertex_offsets,
        cell_face_offsets,
        _offsets(ordered_face_sizes),
        face_vertices.astype(np.int64),
        face_labels[face_index],
        measures_ordered,
        centroids,
    )


def _domain(
    dimension: int,
    box_lower: object,
    box_upper: object,
    domain_normals: object | None,
    domain_offsets: object | None,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lower = np.asarray(box_lower, dtype=np.float64)
    upper = np.asarray(box_upper, dtype=np.float64)
    if lower.shape != (dimension,) or upper.shape != (dimension,):
        raise ValueError(f"box bounds must have shape ({dimension},).")
    if not (np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))):
        raise ValueError("box bounds must be finite.")
    if not np.all(lower < upper):
        raise ValueError("box_lower must be strictly below box_upper.")
    if (domain_normals is None) != (domain_offsets is None):
        raise ValueError("domain_normals and domain_offsets must be given together.")
    if domain_normals is None:
        return (
            lower,
            upper,
            np.zeros((0, dimension), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
        )
    normals = _point_array(domain_normals, "domain_normals", (dimension,))
    offsets = np.asarray(domain_offsets, dtype=np.float64)
    if offsets.shape != (normals.shape[0],) or not np.all(np.isfinite(offsets)):
        raise ValueError("domain_offsets must be a finite (k,) array.")
    if np.any(np.all(normals == 0.0, axis=1)):
        raise ValueError("domain_normals must be nonzero.")
    return lower, upper, normals, offsets


def _all_pairs(count: int, /) -> np.ndarray:
    if count > _DEGENERATE_ALL_PAIRS_LIMIT:
        raise ValueError(
            "Generators that do not span the space use the all-pairs bisector route, "
            f"limited to {_DEGENERATE_ALL_PAIRS_LIMIT} generators."
        )
    first, second = np.triu_indices(count, k=1)
    return np.stack((first, second), axis=1).astype(np.int64)


class VoronoiDiagram(StrictModule, NonTrainableState):
    """Bounded Voronoi cells of generators clipped to a box and convex domain.

    Cells are dual to the exact Delaunay triangulation (``dual``).  Generators
    that do not span the space use every generator pair as a bisector
    candidate (bounded route, ``dual is None``).  Duplicate generators keep one
    representative cell; the others are empty.
    """

    points: np.ndarray
    cells: DiagramCells
    dual: DelaunayTriangulation | None
    evidence: TriangulationEvidence

    def __init__(
        self,
        points: object,
        /,
        *,
        box_lower: object,
        box_upper: object,
        domain_normals: object | None = None,
        domain_offsets: object | None = None,
    ):
        point_array = _point_array(points, "points", (2, 3))
        count, dimension = point_array.shape
        lower, upper, normals, offsets = _domain(
            dimension, box_lower, box_upper, domain_normals, domain_offsets
        )
        if _spans_space(point_array):
            dual = DelaunayTriangulation(point_array)
            neighbors = _simplex_edges(dual.simplices)
            active = dual.vertex_map == np.arange(count)
            route = f"voronoi_{dimension}d"
        else:
            dual = None
            representative = np.unique(point_array, axis=0, return_index=True)[1]
            active = np.zeros(count, dtype=np.bool_)
            active[representative] = True
            candidates = np.sort(np.flatnonzero(active))
            neighbors = candidates[_all_pairs(candidates.size)]
            route = f"voronoi_{dimension}d_all_pairs"
        planes = _cell_planes(point_array, None, neighbors, normals, offsets, active)
        cells = _diagram_cells(lower, upper, planes, active)
        self.points = point_array
        self.cells = cells
        self.dual = dual
        self.evidence = TriangulationEvidence(
            route=route,
            status=MeshcoreStatus.OK,
            input_point_count=count,
            vertex_count=int(np.count_nonzero(active)),
            duplicate_count=count - int(np.count_nonzero(active)),
            redundant_count=0,
            steiner_count=0,
            simplex_count=0 if dual is None else dual.simplices.shape[0],
            minimum_angle_degrees=math.nan,
            content={
                "points": array_tree_fingerprint(point_array),
                "box": [lower.tolist(), upper.tolist()],
                "domain": array_tree_fingerprint((normals, offsets)),
                "measures": array_tree_fingerprint(cells.measures),
            },
        )


class PowerDiagram(StrictModule, NonTrainableState):
    """Bounded power (Laguerre) cells of weighted generators.

    The power distance of ``x`` to generator ``i`` is ``|x - p_i|^2 - w_i``.
    Cells are dual to the exact regular triangulation (``dual_simplices``);
    redundant generators (``dual_vertex_map == -1``) and lighter coincident
    generators have empty cells.  Generators that do not span the space use the
    bounded all-pairs route.
    """

    points: np.ndarray
    weights: np.ndarray
    cells: DiagramCells
    dual_simplices: np.ndarray | None
    dual_vertex_map: np.ndarray | None
    evidence: TriangulationEvidence

    def __init__(
        self,
        points: object,
        weights: object,
        /,
        *,
        box_lower: object,
        box_upper: object,
        domain_normals: object | None = None,
        domain_offsets: object | None = None,
        max_simplices: int | None = None,
    ):
        point_array = _point_array(points, "points", (2, 3))
        count, dimension = point_array.shape
        weight_array = np.asarray(weights)
        if not (
            np.issubdtype(weight_array.dtype, np.floating)
            or np.issubdtype(weight_array.dtype, np.integer)
        ) or np.issubdtype(weight_array.dtype, np.bool_):
            raise TypeError("weights must be a real floating or integer array.")
        if weight_array.shape != (count,) or not np.all(np.isfinite(weight_array)):
            raise ValueError("weights must be a finite (n,) array.")
        weight_array = np.array(weight_array, dtype=np.float64, copy=True)
        weight_array.setflags(write=False)
        lower, upper, normals, offsets = _domain(
            dimension, box_lower, box_upper, domain_normals, domain_offsets
        )
        if _spans_space(point_array):
            if dimension == 2:
                simplices, vertex_map = regular_2d(
                    point_array, weight_array, max_triangles=max_simplices
                )
            else:
                simplices, vertex_map = regular_3d(
                    point_array, weight_array, max_tetrahedra=max_simplices
                )
            neighbors = _simplex_edges(simplices)
            active = vertex_map == np.arange(count)
            dual_simplices: np.ndarray | None = _frozen(simplices)
            dual_vertex_map: np.ndarray | None = _frozen(vertex_map)
            route = f"power_{dimension}d"
        else:
            # Coincident generators: the heaviest (then smallest index) owns the cell.
            order = np.lexsort((np.arange(count), -weight_array, *point_array.T[::-1]))
            ordered = point_array[order]
            first = np.ones(count, dtype=np.bool_)
            first[1:] = np.any(ordered[1:] != ordered[:-1], axis=1)
            active = np.zeros(count, dtype=np.bool_)
            active[order[first]] = True
            candidates = np.sort(np.flatnonzero(active))
            neighbors = candidates[_all_pairs(candidates.size)]
            dual_simplices = None
            dual_vertex_map = None
            route = f"power_{dimension}d_all_pairs"
        planes = _cell_planes(
            point_array, weight_array, neighbors, normals, offsets, active
        )
        cells = _diagram_cells(lower, upper, planes, active)
        self.points = point_array
        self.weights = weight_array
        self.cells = cells
        self.dual_simplices = dual_simplices
        self.dual_vertex_map = dual_vertex_map
        redundant = (
            0 if dual_vertex_map is None else int(np.count_nonzero(dual_vertex_map < 0))
        )
        self.evidence = TriangulationEvidence(
            route=route,
            status=MeshcoreStatus.OK,
            input_point_count=count,
            vertex_count=int(np.count_nonzero(active)),
            duplicate_count=count - int(np.count_nonzero(active)) - redundant,
            redundant_count=redundant,
            steiner_count=0,
            simplex_count=0 if dual_simplices is None else dual_simplices.shape[0],
            minimum_angle_degrees=math.nan,
            content={
                "points": array_tree_fingerprint(point_array),
                "weights": array_tree_fingerprint(weight_array),
                "box": [lower.tolist(), upper.tolist()],
                "domain": array_tree_fingerprint((normals, offsets)),
                "measures": array_tree_fingerprint(cells.measures),
            },
        )


__all__ = [
    "ConstrainedDelaunayTriangulation",
    "DelaunayTriangulation",
    "DiagramCells",
    "PowerDiagram",
    "TriangulationEvidence",
    "VoronoiDiagram",
]
