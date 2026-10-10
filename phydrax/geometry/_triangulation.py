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

:class:`PeriodicDelaunayTriangulation` triangulates the lattice orbit of a
point set on a flat torus from certified bounded image neighborhoods and keeps
one simplex per orbit with the lattice shift of every corner.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from time import perf_counter
from typing import assert_never, Literal, TypeAlias

import equinox as eqx
import numpy as np
import scipy
from jax.typing import ArrayLike
from numpy.typing import NDArray
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

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
    periodic_delaunay,
    PERIODIC_DELAUNAY_EVIDENCE,
    PlanarExecutionEvidence,
    regular_2d,
    regular_3d,
    RestrictedPowerCells,
)
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._periodic_cell import PeriodicCell
from ..typing import parse


_DEGENERATE_ALL_PAIRS_LIMIT = 512
_CLIP_PLANE_BUDGET = 1 << 18


TriangulationProvider: TypeAlias = Literal["meshcore", "qhull"]

# Qhull's documented Delaunay defaults for two and three dimensions, passed
# explicitly so the construction is part of the recorded identity.
_QHULL_OPTIONS = "Qbb Qc Qz Q12"


class TriangulationEvidence(StrictModule, NonTrainableState):
    """Route, provider status and identity, and counts of a triangulation.

    ``provider`` is ``"meshcore"`` (exact predicates, ``predicate_mode``
    ``EXACT``) or ``"qhull"`` (SciPy's floating-point Qhull with recorded
    options, ``predicate_mode`` ``FILTERED``: ties are resolved by Qhull, not
    certified). ``provider_identity`` names the library build or options.
    """

    route: str = eqx.field(static=True)
    status: str = eqx.field(static=True)
    provider: TriangulationProvider = eqx.field(static=True)
    predicate_mode: PredicateMode = eqx.field(static=True)
    provider_identity: str = eqx.field(static=True)
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
        provider: TriangulationProvider = "meshcore",
    ) -> None:
        if not isinstance(status, MeshcoreStatus):
            raise TypeError("status must be a MeshcoreStatus.")
        provider_ = parse(provider, TriangulationProvider, "provider")
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
        match provider_:
            case "meshcore":
                identity = meshcore_identity()
                mode = PredicateMode.EXACT
            case "qhull":
                identity = f"scipy-qhull {scipy.__version__} {_QHULL_OPTIONS}"
                mode = PredicateMode.FILTERED
            case unreachable:
                assert_never(unreachable)
        self.route = route
        self.status = status.name.lower()
        self.provider = provider_
        self.predicate_mode = mode
        self.provider_identity = identity
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
                "provider": provider_,
                "identity": identity,
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


def _qhull_delaunay(
    points: np.ndarray, max_simplices: int | None, /
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """SciPy Qhull Delaunay in the canonical simplex convention.

    Rows are sorted, then the last two vertices swap where needed for positive
    orientation, and rows are ordered lexicographically. Points Qhull leaves
    out map to their nearest vertex (``Qc``); identical ones are duplicates.
    """
    from scipy.spatial import Delaunay, QhullError

    count, dimension = points.shape
    if count <= dimension:
        raise ValueError("Delaunay triangulation needs more points than dimensions.")
    try:
        triangulation = Delaunay(points, qhull_options=_QHULL_OPTIONS)
    except QhullError as error:
        raise ValueError(f"Qhull Delaunay triangulation failed: {error}") from error
    simplices = np.sort(np.asarray(triangulation.simplices, dtype=np.int32), axis=1)
    vertices = points[simplices]
    volumes = np.linalg.det(np.swapaxes(vertices[:, 1:] - vertices[:, :1], 1, 2))
    simplices[volumes < 0] = simplices[volumes < 0][
        :, [*range(dimension - 1), dimension, dimension - 1]
    ]
    simplices = simplices[np.lexsort(simplices.T[::-1])]
    if max_simplices is not None and simplices.shape[0] > max_simplices:
        raise MeshcoreError(
            MeshcoreStatus.CAPACITY_EXCEEDED,
            f"qhull delaunay: {simplices.shape[0]} simplices exceed max_simplices",
        )
    vertex_map = np.arange(count, dtype=np.int32)
    duplicates = redundant = 0
    for point, _, nearest in np.asarray(triangulation.coplanar, dtype=np.int64):
        vertex_map[point] = nearest
        if np.array_equal(points[point], points[nearest]):
            duplicates += 1
        else:
            redundant += 1
    return simplices, vertex_map, duplicates, redundant


# ------------------------------------------------------------------ triangulations


class DelaunayTriangulation(StrictModule, NonTrainableState):
    """Delaunay triangulation (2D) or tetrahedralization (3D).

    ``simplices`` are positively oriented and canonically ordered. The
    ``"meshcore"`` provider (default) is exact: cocircular and cospherical ties
    are resolved by index-ordered symbolic perturbation. The ``"qhull"``
    provider uses SciPy's Qhull (a core dependency) with the recorded options;
    its floating-point ties are deterministic but not certified, so it suits
    auxiliary constructions that need no exact predicates. ``vertex_map``
    sends each input point to its triangulation vertex (the smallest index of
    identical points; Qhull's nearest vertex for any other point it leaves out,
    counted as ``redundant_count``).
    """

    points: np.ndarray
    simplices: np.ndarray
    vertex_map: np.ndarray
    evidence: TriangulationEvidence

    def __init__(
        self,
        points: object,
        /,
        *,
        max_simplices: int | None = None,
        provider: TriangulationProvider = "meshcore",
    ) -> None:
        point_array = _point_array(points, "points", (2, 3))
        provider_ = parse(provider, TriangulationProvider, "provider")
        dimension = point_array.shape[1]
        count = point_array.shape[0]
        match provider_:
            case "meshcore":
                if dimension == 2:
                    simplices, vertex_map = delaunay_2d(
                        point_array, max_triangles=max_simplices
                    )
                else:
                    simplices, vertex_map = delaunay_3d(
                        point_array, max_tetrahedra=max_simplices
                    )
                duplicates = int(np.count_nonzero(vertex_map != np.arange(count)))
                redundant = 0
            case "qhull":
                simplices, vertex_map, duplicates, redundant = _qhull_delaunay(
                    point_array, max_simplices
                )
            case unreachable:
                assert_never(unreachable)
        self.points = point_array
        self.simplices = _frozen(simplices)
        self.vertex_map = _frozen(vertex_map)
        self.evidence = TriangulationEvidence(
            route=f"delaunay_{dimension}d",
            status=MeshcoreStatus.OK,
            input_point_count=count,
            vertex_count=count - duplicates - redundant,
            duplicate_count=duplicates,
            redundant_count=redundant,
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
            provider=provider_,
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

    ``max_cavity_cells``, ``maximum_work`` and ``max_scratch_bytes`` are hard
    nonnegative bounds (None selects signed-int64 maximum). Readonly uint64
    ``work_evidence`` and ``memory_evidence`` retain actual native execution
    measurements; resource refusals expose the same fields on
    ``NativeResourceFailure`` instead of publishing a successful partial mesh.
    """

    points: np.ndarray
    triangles: np.ndarray
    segment_ids: np.ndarray
    input_point_count: int = eqx.field(static=True)
    evidence: TriangulationEvidence
    work_evidence: NDArray[np.uint64]
    memory_evidence: NDArray[np.uint64]

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
        max_cavity_cells: int | None = None,
        maximum_work: int | None = None,
        max_scratch_bytes: int | None = None,
    ) -> None:
        point_array = _point_array(points, "points", (2,))
        measurements: list[PlanarExecutionEvidence] = []
        mesh_points, triangles, _, segment_ids, status = constrained_delaunay_2d(
            point_array,
            segments,
            holes=holes,
            keep_convex_hull=keep_convex_hull,
            min_angle=min_angle,
            max_area=max_area,
            max_steiner=max_steiner,
            max_triangles=max_triangles,
            max_cavity_cells=max_cavity_cells,
            max_work=maximum_work,
            max_scratch_bytes=max_scratch_bytes,
            record_native_resources=measurements.append,
        )
        count = point_array.shape[0]
        used = np.unique(triangles)
        self.points = _frozen(mesh_points)
        self.triangles = _frozen(triangles)
        self.segment_ids = _frozen(segment_ids)
        self.input_point_count = count
        self.work_evidence = measurements[0].work_evidence
        self.memory_evidence = measurements[0].memory_evidence
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


# ------------------------------------------------------------------ quality


# Normalized volume-length ratio at or below which a simplex has no usable
# measure (floating-point flat simplices of cocircular/cospherical ties).
_DEGENERATE_QUALITY = 1e-12


def _simplex_quality(
    points: np.ndarray, simplices: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    """Measures and normalized volume-length ratios ``|T| / v_d(l_rms)``."""
    dimension = points.shape[1]
    corners = points[simplices]
    measures = np.abs(
        np.linalg.det(np.swapaxes(corners[:, 1:] - corners[:, :1], 1, 2))
    ) / math.factorial(dimension)
    first, second = np.triu_indices(dimension + 1, k=1)
    squared = np.sum((corners[:, first] - corners[:, second]) ** 2, axis=-1)
    # Measure of the regular d-simplex whose edge is the RMS edge length.
    regular = (
        np.mean(squared, axis=1) ** (dimension / 2)
        / math.factorial(dimension)
        * math.sqrt((dimension + 1) / 2**dimension)
    )
    quality = np.divide(
        measures, regular, out=np.zeros_like(measures), where=regular > 0.0
    )
    return measures, quality


def _facet_neighbors(simplices: np.ndarray, /) -> tuple[np.ndarray, np.ndarray]:
    """Pairs of simplices that share a facet."""
    count, width = simplices.shape
    facets = np.sort(
        np.concatenate([np.delete(simplices, k, axis=1) for k in range(width)]), axis=1
    )
    owners = np.tile(np.arange(count), width)
    order = np.lexsort(facets.T[::-1])
    ordered = facets[order]
    shared = np.flatnonzero(np.all(ordered[1:] == ordered[:-1], axis=1))
    return owners[order][shared], owners[order][shared + 1]


def _facet_components(
    count: int, first: np.ndarray, second: np.ndarray, members: np.ndarray, /
) -> tuple[np.ndarray, int]:
    """Facet-connected component labels and the component count of ``members``."""
    inside = members[first] & members[second]
    graph = coo_matrix(
        (np.ones(np.count_nonzero(inside)), (first[inside], second[inside])),
        shape=(count, count),
    )
    _, labels = connected_components(graph, directed=False)
    return labels, np.unique(labels[members]).size


def _restorations(
    simplices: np.ndarray,
    quality: np.ndarray,
    retained: np.ndarray,
    usable: np.ndarray,
    neighbors: tuple[np.ndarray, np.ndarray],
    components: int,
    /,
) -> np.ndarray:
    """Excluded usable simplices to restore in one round (empty when complete).

    Uncovered vertices first receive their best incident simplex; once every
    vertex is covered, every component but the largest grows by its
    facet-adjacent excluded simplices until the count matches the input.
    """
    covered = np.zeros(int(simplices.max()) + 1, dtype=np.bool_)
    covered[simplices[retained]] = True
    candidates = np.flatnonzero(usable & ~retained)
    vertices = simplices[candidates].reshape(-1)
    owners = np.repeat(candidates, simplices.shape[1])
    open_ = ~covered[vertices]
    restore = np.zeros(simplices.shape[0], dtype=np.bool_)
    if np.any(open_):
        vertices, owners = vertices[open_], owners[open_]
        order = np.lexsort((-quality[owners], vertices))
        first = np.unique(vertices[order], return_index=True)[1]
        restore[owners[order][first]] = True
        return restore
    first, second = neighbors
    labels, count = _facet_components(simplices.shape[0], first, second, retained)
    if count <= components:
        return restore
    values, sizes = np.unique(labels[retained], return_counts=True)
    largest = values[np.argmax(sizes)]
    for inner, outer in ((first, second), (second, first)):
        grow = (
            retained[inner]
            & (labels[inner] != largest)
            & usable[outer]
            & ~retained[outer]
        )
        restore[outer[grow]] = True
    return restore


class SimplexQualityEvidence(StrictModule, NonTrainableState):
    """Threshold and counts of a quality-screened simplicial subcomplex."""

    minimum_quality: float = eqx.field(static=True)
    simplex_count: int = eqx.field(static=True)
    degenerate_count: int = eqx.field(static=True)
    excluded_count: int = eqx.field(static=True)
    restored_count: int = eqx.field(static=True)
    excluded_measure_fraction: float = eqx.field(static=True)
    minimum_retained_quality: float = eqx.field(static=True)


class SimplexQualitySubcomplex(StrictModule, NonTrainableState):
    """Quality-screened subcomplex of a simplicial triangulation.

    The quality of a simplex is its normalized volume-length ratio
    ``|T| / v_d(l_rms)`` (1 for the regular simplex, 0 when flat), with
    ``l_rms`` the root-mean-square edge length and ``v_d(l)`` the measure of
    the regular d-simplex of edge ``l``. Simplices of quality at most 1e-12
    carry no usable measure and are never retained (``degenerate_count``).
    Simplices below ``minimum_quality`` (3-D slivers, caps and needles) are
    excluded, except that excluded usable simplices are restored, best first,
    while a vertex of a usable simplex would be left uncovered or the retained
    simplices would split into more facet-connected components than the
    usable ones form. The result therefore covers every such vertex and is
    facet-connected exactly when the usable input is. ``retained`` marks the
    retained input rows (``simplices`` lists them in input order); the
    evidence's ``excluded_measure_fraction`` is the share of the usable
    measure carried by the excluded simplices.
    """

    simplices: np.ndarray
    quality: np.ndarray
    retained: np.ndarray
    evidence: SimplexQualityEvidence

    def __init__(
        self, points: object, simplices: object, /, *, minimum_quality: float
    ) -> None:
        point_array = _point_array(points, "points", (2, 3))
        dimension = point_array.shape[1]
        simplex_array = np.asarray(simplices)
        if not np.issubdtype(simplex_array.dtype, np.integer):
            raise TypeError("simplices must be an integer array.")
        if (
            simplex_array.ndim != 2
            or simplex_array.shape[1] != dimension + 1
            or simplex_array.shape[0] == 0
        ):
            raise ValueError("simplices must have shape (n, dimension + 1) with n >= 1.")
        if np.any(simplex_array < 0) or np.any(simplex_array >= point_array.shape[0]):
            raise ValueError("simplices must index the points.")
        threshold = float(minimum_quality)
        if not math.isfinite(threshold) or not 0.0 <= threshold < 1.0:
            raise ValueError("minimum_quality must lie in [0, 1).")
        simplex_array = simplex_array.astype(np.int64)
        measures, quality = _simplex_quality(point_array, simplex_array)
        usable = quality > _DEGENERATE_QUALITY
        if not np.any(usable):
            raise ValueError("No simplex carries a usable measure.")
        retained = usable & (quality >= threshold)
        neighbors = _facet_neighbors(simplex_array)
        _, components = _facet_components(simplex_array.shape[0], *neighbors, usable)
        excluded = np.count_nonzero(usable & ~retained)
        # Each round restores at least one simplex or ends the loop.
        while np.any(
            restore := _restorations(
                simplex_array, quality, retained, usable, neighbors, components
            )
        ):
            retained = retained | restore
        dropped = usable & ~retained
        self.simplices = _frozen(simplex_array[retained].astype(np.int32))
        self.quality = _frozen(quality)
        self.retained = _frozen(retained)
        self.evidence = SimplexQualityEvidence(
            minimum_quality=threshold,
            simplex_count=simplex_array.shape[0],
            # NumPy count scalars are converted once into static metadata.
            degenerate_count=int(np.count_nonzero(~usable)),
            excluded_count=int(np.count_nonzero(dropped)),
            restored_count=int(excluded - np.count_nonzero(dropped)),
            excluded_measure_fraction=float(
                np.sum(measures[dropped]) / np.sum(measures[usable])
            ),
            minimum_retained_quality=float(np.min(quality[retained])),
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
    ) -> None:
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
    ) -> None:
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
    ) -> None:
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
    ) -> None:
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


# ------------------------------------------------------------------ periodic


PeriodicImageLimit: TypeAlias = Literal["none", "images", "cells"]


class PeriodicTriangulationEvidence(StrictModule, NonTrainableState):
    """Certified image neighborhood and native work of a periodic triangulation.

    Every image with lattice coordinates in ``[-margin, 1 + margin]`` of the
    final round was triangulated. ``required_margin`` is the largest margin an
    extracted cell's circumball needs, including the construction slack, so a
    successful result has ``required_margin <= margin``: every extracted
    circumball is empty of all periodic points. ``uncertified_cell_count``
    counts extracted cells of the final round that were uncertified or not
    closed under adjacency. ``exhausted_limit`` names the budget that refused
    the next round (``"none"`` on success).
    """

    status: str = eqx.field(static=True)
    meshcore_identity: str = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    input_point_count: int = eqx.field(static=True)
    rounds: int = eqx.field(static=True)
    margin: float = eqx.field(static=True)
    image_count: int = eqx.field(static=True)
    maximum_images: int = eqx.field(static=True)
    uncertified_cell_count: int = eqx.field(static=True)
    required_margin: float = eqx.field(static=True)
    finite_cell_slots: int = eqx.field(static=True)
    maximum_cells: int = eqx.field(static=True)
    exact_evaluations: int = eqx.field(static=True)
    perturbed_decisions: int = eqx.field(static=True)
    exhausted_limit: PeriodicImageLimit = eqx.field(static=True)
    simplex_count: int = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        status: MeshcoreStatus,
        native: np.ndarray,
        /,
        *,
        dimension: int,
        input_point_count: int,
        maximum_images: int,
        maximum_cells: int,
        simplex_count: int,
        content: dict,
    ) -> None:
        if not isinstance(status, MeshcoreStatus):
            raise TypeError("status must be a MeshcoreStatus.")
        values = dict(zip(PERIODIC_DELAUNAY_EVIDENCE, native.tolist(), strict=True))
        limit: PeriodicImageLimit
        match int(values["exhausted_limit"]):
            case 0:
                limit = "none"
            case 1:
                limit = "images"
            case 2:
                limit = "cells"
            case code:
                raise ValueError(f"Unknown periodic image limit code {code}.")
        identity = meshcore_identity()
        self.status = status.name.lower()
        self.meshcore_identity = identity
        self.dimension = dimension
        self.input_point_count = input_point_count
        self.rounds = int(values["rounds"])
        self.margin = float(values["margin"])
        self.image_count = int(values["image_count"])
        self.maximum_images = maximum_images
        self.uncertified_cell_count = int(values["uncertified_cells"])
        self.required_margin = float(values["required_margin"])
        self.finite_cell_slots = int(values["finite_cell_slots"])
        self.maximum_cells = maximum_cells
        self.exact_evaluations = int(values["exact_evaluations"])
        self.perturbed_decisions = int(values["perturbed_decisions"])
        self.exhausted_limit = limit
        self.simplex_count = simplex_count
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "periodic-triangulation-evidence",
                "status": self.status,
                "meshcore": identity,
                "native": array_tree_fingerprint(native),
                "budgets": [maximum_images, maximum_cells],
                "content": content,
            }
        )


class PeriodicImageBudgetError(MeshcoreError):
    """A periodic triangulation needed more images or cells than its budget.

    ``evidence`` records the refused round: its margin, the images it would
    need, the uncertified cells of the previous round and the exhausted limit.
    """

    def __init__(self, evidence: PeriodicTriangulationEvidence, /) -> None:
        super().__init__(
            MeshcoreStatus.CAPACITY_EXCEEDED,
            f"periodic Delaunay round {evidence.rounds} at fractional margin "
            f"{evidence.margin:.6g} exceeds its {evidence.exhausted_limit} budget "
            f"(images {evidence.image_count} of {evidence.maximum_images}, cell "
            f"slots {evidence.finite_cell_slots} of {evidence.maximum_cells}; "
            f"{evidence.uncertified_cell_count} cells uncertified at margin "
            f"{evidence.required_margin:.6g})",
        )
        self.evidence = evidence


def _periodic_budget(value: int | None, default: int, name: str, /) -> int:
    if value is None:
        return default
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    if value < 1:
        raise ValueError(f"{name} must be positive.")
    return int(value)


class PeriodicDelaunayTriangulation(StrictModule, NonTrainableState):
    """Exact Delaunay triangulation of a translationally periodic point set.

    The point set is the lattice orbit of ``points`` under a fully periodic
    full-rank :class:`~phydrax.discretization.PeriodicCell`; representatives
    keep their input coordinates. ``simplices`` holds one positively oriented
    simplex per orbit of the periodic triangulation (representative indices)
    and ``simplex_shifts`` the lattice image of each corner, so corner ``k`` of
    simplex ``t`` lies at ``points[simplices[t, k]] + simplex_shifts[t, k] @
    cell.vectors``; the published member of each orbit is the one whose anchor
    corner (smallest representative, then smallest shift) is the anchor's
    image in the fundamental cell. Predicates are exact on translated positions
    and cospherical ties use a translation-invariant symbolic perturbation, so
    the triangulation is unique and lattice-invariant.

    Construction triangulates bounded image neighborhoods, starting at
    fractional margin ``initial_margin`` and doubling until every extracted
    circumball is certified inside the triangulated images and the extracted
    cells close up (:class:`PeriodicTriangulationEvidence`). A round needing
    more than ``maximum_images`` images or ``maximum_simplices`` finite cell
    slots raises :class:`PeriodicImageBudgetError`; two points in one lattice
    orbit raise ``ValueError``.
    """

    cell: PeriodicCell
    points: np.ndarray
    simplices: np.ndarray
    simplex_shifts: np.ndarray
    evidence: PeriodicTriangulationEvidence

    def __init__(
        self,
        points: object,
        cell: PeriodicCell,
        /,
        *,
        initial_margin: float | None = None,
        maximum_images: int | None = None,
        maximum_simplices: int | None = None,
    ) -> None:
        if not isinstance(cell, PeriodicCell):
            raise TypeError("cell must be a PeriodicCell.")
        point_array = _point_array(points, "points", (2, 3))
        count, dimension = point_array.shape
        if count == 0:
            raise ValueError("A periodic triangulation requires at least one point.")
        if (
            cell.ambient_dimension != dimension
            or cell.rank != dimension
            or not cell.fully_periodic
        ):
            raise ValueError(
                "Periodic Delaunay triangulation requires a fully periodic "
                "full-rank lattice in the point dimension."
            )
        # Delaunay circumradii scale like the mean spacing n^(-1/d) in lattice units.
        margin = (
            min(1.0, 2.0 * count ** (-1.0 / dimension))
            if initial_margin is None
            else float(initial_margin)
        )
        if not math.isfinite(margin) or margin <= 0.0:
            raise ValueError("initial_margin must be positive and finite.")
        images = _periodic_budget(maximum_images, 27 * count + 4096, "maximum_images")
        cells = _periodic_budget(maximum_simplices, 64 * images, "maximum_simplices")
        vectors = np.asarray(cell.vectors, dtype=np.float64)
        inverse = np.asarray(cell.inverse_vectors, dtype=np.float64)
        fractional = (point_array - np.asarray(cell.origin, dtype=np.float64)) @ inverse
        status, simplices, shifts, native = periodic_delaunay(
            point_array,
            fractional,
            vectors,
            inverse,
            initial_margin=margin,
            max_images=images,
            max_cells=cells,
        )
        evidence = PeriodicTriangulationEvidence(
            status,
            native,
            dimension=dimension,
            input_point_count=count,
            maximum_images=images,
            maximum_cells=cells,
            simplex_count=simplices.shape[0],
            content={
                "cell": cell.cell_id,
                "points": array_tree_fingerprint(point_array),
                "simplices": array_tree_fingerprint(simplices),
                "simplex_shifts": array_tree_fingerprint(shifts),
            },
        )
        match status:
            case MeshcoreStatus.OK:
                pass
            case MeshcoreStatus.CAPACITY_EXCEEDED:
                raise PeriodicImageBudgetError(evidence)
            case MeshcoreStatus.INVALID_INPUT:
                duplicate = int(
                    native[PERIODIC_DELAUNAY_EVIDENCE.index("duplicate_representative")]
                )
                raise ValueError(
                    f"points[{duplicate}] lies in the lattice orbit of another point."
                )
            case _:
                raise MeshcoreError(status, "periodic_delaunay: unexpected status")
        self.cell = cell
        self.points = point_array
        self.simplices = _frozen(simplices)
        self.simplex_shifts = _frozen(shifts)
        self.evidence = evidence

    @property
    def dimension(self) -> int:
        return self.points.shape[1]


__all__ = [
    "ConstrainedDelaunayTriangulation",
    "DelaunayTriangulation",
    "DiagramCells",
    "PeriodicDelaunayTriangulation",
    "PeriodicImageBudgetError",
    "PeriodicImageLimit",
    "PeriodicTriangulationEvidence",
    "PowerDiagram",
    "RestrictedPowerDiagram",
    "SimplexQualityEvidence",
    "SimplexQualitySubcomplex",
    "TriangulationEvidence",
    "VoronoiDiagram",
]


def _exact_power_translation_dual(vectors: np.ndarray, /) -> np.ndarray:
    """Rational dual of independent translation rows, including partial rank."""
    from fractions import Fraction

    from ..meshing._periodic import _periodic_exact_product, _reserve_periodic_exact_terms

    rank = len(vectors)
    gram = _periodic_exact_product(vectors, vectors.T)
    _reserve_periodic_exact_terms(2 * rank * rank)
    augmented = [
        list(row) + [Fraction(int(i == j)) for j in range(rank)]
        for i, row in enumerate(gram)
    ]
    for column in range(rank):
        pivot = None
        for candidate in range(column, rank):
            _reserve_periodic_exact_terms(1)
            if augmented[candidate][column]:
                pivot = candidate
                break
        if pivot is None:
            raise ValueError(
                "Periodic power translation generators are exactly dependent."
            )
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        _reserve_periodic_exact_terms(len(augmented[column]))
        augmented[column] = [x / divisor for x in augmented[column]]
        for row in range(rank):
            if row != column:
                factor = augmented[row][column]
                _reserve_periodic_exact_terms(1)
                if not factor:
                    continue
                _reserve_periodic_exact_terms(2 * len(augmented[row]))
                augmented[row] = [
                    x - factor * y
                    for x, y in zip(augmented[row], augmented[column], strict=True)
                ]
    inverse = np.asarray([row[rank:] for row in augmented], dtype=object)
    return _periodic_exact_product(vectors.T, inverse)


class PeriodicPowerImageCapacityRefusal(MeshcoreError):
    """Pre-materialization refusal retaining source and exact stage counts."""

    def __init__(
        self,
        points: np.ndarray,
        weights: np.ndarray,
        domain: np.ndarray,
        periodic_group: object,
        requested: int,
        maximum: int,
        stage: str,
        /,
    ) -> None:
        super().__init__(
            MeshcoreStatus.CAPACITY_EXCEEDED,
            f"Periodic power {stage} requires {requested} site images, exceeding {maximum}.",
        )
        self.points = _frozen(points.copy())
        self.weights = _frozen(weights.copy())
        self.domain_points = _frozen(domain.copy())
        self.periodic_group = periodic_group
        self.requested_images = requested
        self.completed_images = 0
        self.maximum_images = maximum
        self.stage = stage


class PeriodicPowerPreparation(StrictModule, NonTrainableState):
    """Complete power-relevant image bank, retaining the original source axes.

    ``image_sites`` indexes authored sites, never materialized coordinates.
    ``image_exponents`` is the exact group action on that source. Numerical
    image coordinates are deliberately not a substitute for this source law.
    The bank includes every image capable of winning anywhere in the carrier
    bounding box, including weighted sites outside that box.
    """

    points: np.ndarray
    weights: np.ndarray
    domain_points: np.ndarray
    periodic_group: object
    generators: np.ndarray
    image_sites: np.ndarray
    image_exponents: np.ndarray
    image_coordinate_offsets: np.ndarray
    image_coordinate_components: np.ndarray
    orders: tuple[int, ...] = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    carrier_id: str = eqx.field(static=True)
    identification_id: str = eqx.field(static=True)
    preparation_id: str = eqx.field(static=True)
    spent_work: int = eqx.field(static=True)
    native_charged_work: int = eqx.field(static=True)
    maximum_images: int = eqx.field(static=True)
    maximum_work_units: int = eqx.field(static=True)

    def __init__(
        self,
        points: object,
        weights: object,
        domain_points: object,
        periodic_group: object,
        /,
        *,
        maximum_images: int,
        maximum_work_units: int = 1 << 26,
    ) -> None:
        from .._meshcore import current_native_execution_budget
        from ..discretization._coordinate_enclosure import coordinate_enclosure_budget

        work_limit = _periodic_budget(maximum_work_units, 1, "maximum_work_units")
        self.maximum_images = _periodic_budget(maximum_images, 1, "maximum_images")
        self.maximum_work_units = work_limit
        native = current_native_execution_budget()
        memory_limit = (
            int(np.iinfo(np.intp).max)
            if native is None
            else native.remaining().remaining_scratch_bytes
        )
        ledger = coordinate_enclosure_budget(work_limit, memory_limit)
        starting_work = ledger.work_units
        starting_native = ledger.native_charged_work_units
        with (
            ledger.activate(),
            ledger.bound_stage(
                work_limit,
                memory_limit,
                starting_work_units=starting_work,
            ),
            ledger.temporary_scope(),
        ):
            try:
                self._prepare(
                    points,
                    weights,
                    domain_points,
                    periodic_group,
                    maximum_images=self.maximum_images,
                )
            finally:
                ledger.charge_native_work(ledger.work_units - starting_work)
        self.spent_work = ledger.work_units - starting_work
        self.native_charged_work = ledger.native_charged_work_units - starting_native

    def validate_restored(self) -> None:
        """Authenticate source/control/bank law without replaying past receipts.

        Revalidation performs real current work under the current original
        ledger. Its receipt is distinct from the archived preparation receipt.
        """
        from ..discretization._periodic_topology import PeriodicIsometryGroup

        if isinstance(self.periodic_group, (PeriodicCell, PeriodicIsometryGroup)):
            self.periodic_group.validate_restored()
        else:
            raise TypeError(
                "Restored periodic power requires its original periodic identification."
            )
        for value in (self.spent_work, self.native_charged_work):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or value < 0
            ):
                raise ValueError(
                    "Periodic power preparation receipts must be nonnegative integers."
                )
        if (
            self.native_charged_work > self.spent_work
            or self.spent_work > self.maximum_work_units
        ):
            raise ValueError(
                "Periodic power preparation receipt exceeds its authored allowance."
            )
        replay = PeriodicPowerPreparation(
            self.points,
            self.weights,
            self.domain_points,
            self.periodic_group,
            maximum_images=self.maximum_images,
            maximum_work_units=self.maximum_work_units,
        )
        for name in (
            "points",
            "weights",
            "domain_points",
            "generators",
            "image_sites",
            "image_exponents",
            "image_coordinate_offsets",
            "image_coordinate_components",
        ):
            if array_tree_fingerprint(getattr(self, name)) != array_tree_fingerprint(
                getattr(replay, name)
            ):
                raise ValueError(
                    f"Restored periodic power {name} violates its original source law."
                )
        for name in (
            "orders",
            "source_id",
            "carrier_id",
            "identification_id",
            "preparation_id",
            "maximum_images",
            "maximum_work_units",
        ):
            if getattr(self, name) != getattr(replay, name):
                raise ValueError(f"Restored periodic power {name} is not authenticated.")
        for name in (
            "points",
            "weights",
            "domain_points",
            "generators",
            "image_sites",
            "image_exponents",
            "image_coordinate_offsets",
            "image_coordinate_components",
        ):
            value = getattr(self, name)
            if isinstance(value, np.ndarray):
                value.setflags(write=False)

    def _prepare(
        self,
        points: object,
        weights: object,
        domain_points: object,
        periodic_group: object,
        /,
        *,
        maximum_images: int,
    ) -> None:
        from fractions import Fraction
        from itertools import product

        from ..discretization._periodic_topology import (
            _exact_periodic_generators,
            PeriodicIsometryGroup,
        )
        from ..meshing._periodic import (
            _exact_periodic_group_element,
            _periodic_exact_product,
            _prepare_periodic_image_frame,
            _reserve_periodic_exact_terms,
        )

        sites = _point_array(points, "points", (3,))
        domain = _point_array(domain_points, "domain_points", (3,))
        values = np.asarray(weights, dtype=np.float64)
        _reserve_periodic_exact_terms(sites.size + domain.size + values.size)
        if not len(sites) or not len(domain):
            raise ValueError("Periodic power preparation requires sites and a carrier.")
        if values.shape != (len(sites),) or not np.all(np.isfinite(values)):
            raise ValueError("weights must be finite with one value per authored site.")
        limit = _periodic_budget(maximum_images, 1, "maximum_images")
        if not isinstance(periodic_group, (PeriodicCell, PeriodicIsometryGroup)):
            raise TypeError(
                "periodic_group must be a PeriodicCell or PeriodicIsometryGroup."
            )
        if periodic_group.ambient_dimension != 3:
            raise ValueError(
                "Periodic restricted power requires ambient dimension three."
            )
        if len(sites) > limit:
            raise PeriodicPowerImageCapacityRefusal(
                sites,
                values,
                domain,
                periodic_group,
                len(sites),
                limit,
                "original-sites",
            )
        if isinstance(periodic_group, PeriodicCell):
            if not periodic_group.fully_periodic:
                raise ValueError(
                    "Periodic power requires every declared lattice axis periodic."
                )
            vectors = np.asarray(periodic_group.vectors, dtype=np.float64)
            generators = np.repeat(np.eye(4)[None], periodic_group.rank, axis=0)
            generators[:, :3, 3] = vectors
            orders = (0,) * periodic_group.rank
            linear_orders = (1,) * periodic_group.rank
            _reserve_periodic_exact_terms(sites.size)
            finite_sites = np.asarray(
                [[Fraction(float(x)) for x in site] for site in sites],
                dtype=object,
            )
        else:
            generators = np.asarray(periodic_group.generators, dtype=np.float64)
            orders = periodic_group.orders
            linear_orders = periodic_group.linear_orders
            finite_count = math.prod(linear_orders)
            if finite_count * len(sites) > limit:
                raise PeriodicPowerImageCapacityRefusal(
                    sites,
                    values,
                    domain,
                    periodic_group,
                    finite_count * len(sites),
                    limit,
                    "finite-orbits",
                )
            finite_sites_parts = []
            _reserve_periodic_exact_terms(sites.size)
            original = np.asarray(
                [[Fraction(float(x)) for x in site] for site in sites],
                dtype=object,
            )
            prepared_generators = _exact_periodic_generators(periodic_group)
            for finite_action in product(*(range(order) for order in linear_orders)):
                matrix = _exact_periodic_group_element(
                    periodic_group,
                    np.asarray(finite_action, dtype=np.int64),
                    prepared_generators=prepared_generators,
                )
                _reserve_periodic_exact_terms(len(sites) * 3)
                finite_sites_parts.append(
                    _periodic_exact_product(original, matrix[:3, :3].T) + matrix[:3, 3]
                )
            finite_sites = np.concatenate(finite_sites_parts)
        translations = [axis for axis, order in enumerate(orders) if order == 0]
        ranges = [range(order) for order in linear_orders]
        if translations:
            _reserve_periodic_exact_terms(len(translations) * 3)
            if isinstance(periodic_group, PeriodicCell):
                vectors = np.asarray(
                    [
                        [Fraction(float(x)) for x in generators[axis, :3, 3]]
                        for axis in translations
                    ],
                    dtype=object,
                )
            else:
                vectors = np.asarray(
                    [
                        _exact_periodic_group_element(
                            periodic_group,
                            np.asarray(
                                [
                                    linear_orders[index] if index == axis else 0
                                    for index in range(periodic_group.rank)
                                ],
                                dtype=np.int64,
                            ),
                            prepared_generators=prepared_generators,
                        )[:3, 3]
                        for axis in translations
                    ],
                    dtype=object,
                )
            dual = _exact_power_translation_dual(vectors)
            lower, upper = np.min(domain, axis=0), np.max(domain, axis=0)
            corners = tuple(product(*zip(lower, upper, strict=True)))
            reference = [Fraction(float(x)) for x in sites[0]]
            _reserve_periodic_exact_terms(len(corners) * 2 * 3 + len(values) + 2)
            ceiling = (
                max(
                    sum(
                        (Fraction(float(x)) - y) ** 2
                        for x, y in zip(corner, reference, strict=True)
                    )
                    for corner in corners
                )
                - Fraction(float(values[0]))
                + max(Fraction(float(w)) for w in values)
            )
            radius = math.isqrt(max(0, ceiling.numerator // ceiling.denominator))
            if Fraction(radius * radius) < ceiling:
                radius += 1
            for column, axis in enumerate(translations):
                coefficients = dual[:, column]
                _reserve_periodic_exact_terms(3 * 3)
                carrier_low = sum(
                    a * Fraction(float(lower[i] if a >= 0 else upper[i]))
                    for i, a in enumerate(coefficients)
                )
                carrier_high = sum(
                    a * Fraction(float(upper[i] if a >= 0 else lower[i]))
                    for i, a in enumerate(coefficients)
                )
                site_axis = _periodic_exact_product(finite_sites, coefficients)
                _reserve_periodic_exact_terms(2 * len(site_axis) + 5)
                reach = radius * sum(abs(a) for a in coefficients)
                low = carrier_low - max(site_axis) - reach
                high = carrier_high - min(site_axis) + reach
                period = linear_orders[axis]
                ranges[axis] = range(
                    period * math.ceil(low),
                    period * (math.floor(high) + 1),
                )
        actions = product(*ranges)
        action_count = math.prod(len(value) for value in ranges)
        required = action_count * len(sites)
        if required > limit:
            raise PeriodicPowerImageCapacityRefusal(
                sites,
                values,
                domain,
                periodic_group,
                required,
                limit,
                "power-images",
            )
        # Authenticate the same exact authored group frame as source/embedding.
        # Its overlap bank is not used as the power completeness bound above.
        action_rows = np.asarray(tuple(actions), dtype=np.int64).reshape(
            (action_count, periodic_group.rank)
        )
        frame = _prepare_periodic_image_frame(
            sites,
            np.arange(len(sites), dtype=np.int64),
            np.zeros((len(sites), periodic_group.rank), dtype=np.int64),
            periodic_group,
            limit,
            image_exponents=action_rows,
        )
        components: list[float] = []
        coordinate_offsets = [0]
        scale = Fraction(2) ** (
            frame.exponent if frame.lattice is not None else 3 * frame.exponent
        )
        for index, action in enumerate(action_rows):
            image = tuple(int(x) for x in action) if frame.lattice is not None else index
            exact_coordinates = frame.image_points(image)
            for integer in exact_coordinates.reshape(-1):
                _reserve_periodic_exact_terms(1)
                remaining = Fraction(int(integer)) * scale
                coordinate: list[float] = []
                while remaining:
                    _reserve_periodic_exact_terms(2)
                    component = float(remaining)
                    if not math.isfinite(component) or component == 0.0:
                        raise ValueError(
                            "Exact periodic image exceeds binary64 expansion range."
                        )
                    coordinate.append(component)
                    remaining -= Fraction(component)
                components.extend(reversed(coordinate))
                coordinate_offsets.append(len(components))
        self.points = _frozen(sites.copy())
        self.weights = _frozen(values.copy())
        self.domain_points = _frozen(domain.copy())
        self.periodic_group = periodic_group
        self.generators = _frozen(generators.copy())
        self.image_sites = _frozen(
            np.tile(np.arange(len(sites), dtype=np.int32), action_count)
        )
        self.image_exponents = _frozen(np.repeat(action_rows, len(sites), axis=0))
        self.image_coordinate_offsets = _frozen(
            np.asarray(coordinate_offsets, dtype=np.int64)
        )
        self.image_coordinate_components = _frozen(
            np.asarray(components, dtype=np.float64)
        )
        self.orders = orders
        self.carrier_id = canonical_fingerprint(array_tree_fingerprint(domain))
        self.identification_id = (
            periodic_group.cell_id
            if isinstance(periodic_group, PeriodicCell)
            else periodic_group.group_id
        )
        self.source_id = canonical_fingerprint(
            {
                "kind": "periodic-power-source",
                "points": array_tree_fingerprint(self.points),
                "weights": array_tree_fingerprint(self.weights),
                "generators": array_tree_fingerprint(self.generators),
                "orders": orders,
                "identification": self.identification_id,
            }
        )
        self.preparation_id = canonical_fingerprint(
            {
                "kind": "periodic-power-preparation",
                "source": self.source_id,
                "carrier": array_tree_fingerprint(domain),
                "site_axis": array_tree_fingerprint(self.image_sites),
                "action_axis": array_tree_fingerprint(self.image_exponents),
                "coordinate_offsets": array_tree_fingerprint(
                    self.image_coordinate_offsets
                ),
                "coordinate_components": array_tree_fingerprint(
                    self.image_coordinate_components
                ),
            }
        )


class PeriodicPowerSourceRefusal(MeshcoreError):
    """Native/domain refusal retaining the actual immutable scientific source."""

    def __init__(
        self,
        preparation: PeriodicPowerPreparation,
        refusal: MeshcoreError,
        maximum_work: int,
        /,
        *,
        requested_work: int | None = None,
        completed_work: int | None = None,
    ) -> None:
        super().__init__(
            refusal.status,
            "Exact periodic power source refused construction",
            work_evidence=refusal.work_evidence,
            memory_evidence=refusal.memory_evidence,
        )
        self.preparation = preparation
        self.native_refusal = refusal
        self.maximum_work = maximum_work
        self.requested_work = requested_work
        self.completed_work = completed_work


def _restricted_periodic_power_construction(
    preparation: PeriodicPowerPreparation,
    split: np.ndarray,
    domain: np.ndarray,
    tetrahedra: ArrayLike,
    regions: ArrayLike,
    facets: ArrayLike,
    /,
    *,
    max_pieces: int,
    max_vertices: int,
    work_limit: int,
    record_native_phase: Callable[[str, float, int | None, int], None] | None,
) -> tuple[np.ndarray, np.ndarray, RestrictedPowerCells, int]:
    from .._meshcore import current_native_execution_budget, restricted_power_cells_exact

    count = len(preparation.image_sites)
    adjacency_count = count * (count - 1)
    if adjacency_count >= work_limit:
        raise PeriodicPowerSourceRefusal(
            preparation,
            MeshcoreError(
                MeshcoreStatus.CAPACITY_EXCEEDED,
                "Exact periodic power adjacency exceeds the original work limit.",
            ),
            work_limit,
            requested_work=adjacency_count + 1,
            completed_work=0,
        )
    # A complete adjacency is an exact source candidate bank, not a rounded
    # regular-triangulation proxy. Its actual materialization is bounded before
    # allocation and charged as host work, never as native geometric predicates.
    budget = current_native_execution_budget()
    if budget is not None:
        budget.admit_work_bound(adjacency_count)
    offsets = np.arange(count + 1, dtype=np.int64) * (count - 1)
    neighbors = np.empty(adjacency_count, dtype=np.int32)
    row = np.arange(count, dtype=np.int32)
    materialized_work = 0
    for image in range(count):
        left, right = row[:image], row[image + 1 :]
        row_work = len(left) + len(right)
        if budget is not None:
            budget.charge(work=row_work)
        start = int(offsets[image])
        neighbors[start : start + len(left)] = left
        neighbors[start + len(left) : int(offsets[image + 1])] = right
        materialized_work += row_work
    construction = restricted_power_cells_exact(
        preparation.points,
        preparation.weights,
        preparation.image_sites,
        preparation.image_coordinate_offsets,
        preparation.image_coordinate_components,
        offsets,
        neighbors,
        split[preparation.image_sites],
        domain,
        tetrahedra,
        regions,
        facets,
        max_pieces=max_pieces,
        max_vertices=max_vertices,
        work_limit=work_limit - materialized_work,
        record_native_phase=record_native_phase,
    )
    return offsets, neighbors, construction, materialized_work


class RestrictedPowerDiagram(StrictModule, NonTrainableState):
    """Native connected power cells restricted to a constrained tet complex.

    Unlike the convex diagram contracts above this preserves disconnected site
    components and constrained material/sheet faces. ``construction`` carries
    reciprocal faces and the site/tet/cell piece witness. This is construction
    evidence, not an independent coverage certificate.
    """

    points: np.ndarray
    weights: np.ndarray
    construction: RestrictedPowerCells
    neighbor_offsets: np.ndarray
    neighbors: np.ndarray
    periodic_preparation: PeriodicPowerPreparation | None
    preparation_work_units: int = eqx.field(static=True)
    preparation_native_charged_work_units: int = eqx.field(static=True)
    adjacency_work_units: int = eqx.field(static=True)
    cell_original_sites: np.ndarray
    piece_original_sites: np.ndarray
    cell_image_exponents: np.ndarray
    piece_image_exponents: np.ndarray
    guard_count: int = eqx.field(static=True)
    meshcore_identity: str = eqx.field(static=True)

    def __init__(
        self,
        points: object,
        weights: object,
        domain_points: object,
        tetrahedra: object,
        tetrahedron_regions: object,
        tet_face_facets: object,
        /,
        *,
        split_sites: object | None = None,
        periodic_group: object | None = None,
        periodic_preparation: PeriodicPowerPreparation | None = None,
        maximum_images: int | None = None,
        max_pieces: int = 1 << 20,
        max_vertices: int = 1 << 22,
        work_limit: int = 1 << 26,
        record_native_phase: Callable[[str, float, int | None, int], None] | None = None,
    ) -> None:
        from fractions import Fraction
        from itertools import product

        from .._meshcore import restricted_power_cells

        sites = _point_array(points, "points", (3,))
        if sites.shape[0] == 0:
            raise ValueError("At least one site is required.")
        values = np.asarray(weights, dtype=np.float64)
        if values.shape != (sites.shape[0],) or not np.all(np.isfinite(values)):
            raise ValueError("weights must be a finite (site_count,) array.")
        domain = _point_array(domain_points, "domain_points", (3,))
        split = (
            np.zeros(sites.shape[0], dtype=np.int8)
            if split_sites is None
            else np.asarray(split_sites, dtype=np.int8)
        )
        tetrahedra_ = np.asarray(tetrahedra)
        tetrahedron_regions_ = np.asarray(tetrahedron_regions)
        tet_face_facets_ = np.asarray(tet_face_facets)
        if split.shape != (sites.shape[0],) or np.any((split != 0) & (split != 1)):
            raise ValueError("split_sites must be a binary (site_count,) array.")
        if periodic_group is not None or periodic_preparation is not None:
            preparation = periodic_preparation
            preparation_work_units = 0
            preparation_native_charged_work_units = 0
            if preparation is None:
                preparation = PeriodicPowerPreparation(
                    sites,
                    values,
                    domain,
                    periodic_group,
                    maximum_images=_periodic_budget(
                        maximum_images,
                        27 * len(sites) + 4096,
                        "maximum_images",
                    ),
                    maximum_work_units=work_limit,
                )
                preparation_work_units = preparation.spent_work
                preparation_native_charged_work_units = preparation.native_charged_work
            if (
                array_tree_fingerprint(sites)
                != array_tree_fingerprint(preparation.points)
                or array_tree_fingerprint(values)
                != array_tree_fingerprint(preparation.weights)
                or canonical_fingerprint(array_tree_fingerprint(domain))
                != preparation.carrier_id
            ):
                raise ValueError(
                    "Periodic power preparation requires its original source and carrier bytes."
                )
            if maximum_images is not None and len(
                preparation.image_sites
            ) > _periodic_budget(
                maximum_images,
                1,
                "maximum_images",
            ):
                raise PeriodicPowerSourceRefusal(
                    preparation,
                    PeriodicPowerImageCapacityRefusal(
                        sites,
                        values,
                        domain,
                        periodic_group,
                        len(preparation.image_sites),
                        maximum_images,
                        "retained-images",
                    ),
                    work_limit,
                )
            if periodic_group is not None:
                from ..discretization._periodic_topology import PeriodicIsometryGroup

                if not isinstance(periodic_group, (PeriodicCell, PeriodicIsometryGroup)):
                    raise TypeError(
                        "periodic_group must be a PeriodicCell or PeriodicIsometryGroup."
                    )
                identity = (
                    periodic_group.cell_id
                    if isinstance(periodic_group, PeriodicCell)
                    else periodic_group.group_id
                )
                if identity != preparation.identification_id:
                    raise ValueError(
                        "Periodic power preparation requires its original group identity."
                    )
            try:
                construction_work_limit = work_limit - preparation_work_units
                if construction_work_limit < 1:
                    raise PeriodicPowerSourceRefusal(
                        preparation,
                        MeshcoreError(
                            MeshcoreStatus.CAPACITY_EXCEEDED,
                            "Periodic power preparation exhausts the original construction allowance.",
                        ),
                        work_limit,
                        requested_work=preparation_work_units + 1,
                        completed_work=preparation_work_units,
                    )
                offsets, neighbors, construction, adjacency_work_units = (
                    _restricted_periodic_power_construction(
                        preparation,
                        split,
                        domain,
                        tetrahedra_,
                        tetrahedron_regions_,
                        tet_face_facets_,
                        max_pieces=max_pieces,
                        max_vertices=max_vertices,
                        work_limit=construction_work_limit,
                        record_native_phase=record_native_phase,
                    )
                )
            except PeriodicPowerSourceRefusal:
                raise
            except MeshcoreError as refusal:
                raise PeriodicPowerSourceRefusal(
                    preparation, refusal, work_limit
                ) from refusal
            self.points = preparation.points
            self.weights = preparation.weights
            self.construction = construction
            self.neighbor_offsets = _frozen(offsets)
            self.neighbors = _frozen(neighbors)
            self.periodic_preparation = preparation
            self.preparation_work_units = preparation_work_units
            self.preparation_native_charged_work_units = (
                preparation_native_charged_work_units
            )
            self.adjacency_work_units = adjacency_work_units
            self.cell_original_sites = _frozen(
                preparation.image_sites[construction.cell_sites]
            )
            self.piece_original_sites = _frozen(
                preparation.image_sites[construction.piece_sites]
            )
            self.cell_image_exponents = _frozen(
                preparation.image_exponents[construction.cell_sites]
            )
            self.piece_image_exponents = _frozen(
                preparation.image_exponents[construction.piece_sites]
            )
            self.guard_count = 0
            self.meshcore_identity = meshcore_identity()
            return
        guard_count = 0
        if sites.shape[0] == 1:
            prepared_sites, prepared_weights = sites, values
            edges = np.zeros((0, 2), dtype=np.int32)
        else:
            prepared_sites, prepared_weights = sites, values
            if not _spans_space(sites):
                # Four remote weighted guards make a full-dimensional regular
                # triangulation without a dense all-pairs degeneracy fallback.
                # Their power distance exceeds site zero everywhere in the
                # domain bounding box: the difference is affine, and its
                # minimum is evaluated exactly at all eight box corners.
                lower, upper = np.min(domain, axis=0), np.max(domain, axis=0)
                scale = max(1.0, float(np.max(upper - lower)))
                base = lower - 4.0 * scale
                guards = base + 8.0 * scale * np.asarray(
                    ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)),
                    dtype=np.float64,
                )
                if not np.all(np.isfinite(guards)) or not _spans_space(guards):
                    raise ValueError("Domain scale cannot represent spanning guards.")
                reference = [Fraction(float(x)) for x in sites[0]]
                reference_weight = Fraction(float(values[0]))
                corners = tuple(product(*zip(lower, upper, strict=True)))
                guard_weights = []
                for guard in guards:
                    g = [Fraction(float(x)) for x in guard]
                    minimum = min(
                        sum(
                            (a - Fraction(float(x))) ** 2 - (b - Fraction(float(x))) ** 2
                            for a, b, x in zip(g, reference, corner, strict=True)
                        )
                        + reference_weight
                        for corner in corners
                    )
                    # Keep a strict normal-scale margin: for positive minimum
                    # the old subtraction cancelled to zero, whose nextafter
                    # value lies outside meshcore's exact weight domain.
                    weight = float(minimum - max(Fraction(1), abs(minimum)) - 1)
                    weight = float(np.nextafter(weight, -np.inf))
                    if not math.isfinite(weight) or Fraction(weight) >= minimum:
                        raise ValueError("Unable to establish empty-guard power bound.")
                    guard_weights.append(weight)
                guard_count = 4
                prepared_sites = np.concatenate((sites, guards))
                prepared_weights = np.concatenate((values, guard_weights))
            started = 0.0 if record_native_phase is None else perf_counter()
            simplices, _ = regular_3d(
                prepared_sites, prepared_weights, max_tetrahedra=max_pieces
            )
            edges = _simplex_edges(simplices)
            if record_native_phase is not None:
                record_native_phase(
                    "regular_triangulation", perf_counter() - started, None, 1
                )
        directed = np.concatenate((edges, edges[:, ::-1]), axis=0)
        if directed.size:
            directed = directed[np.lexsort((directed[:, 1], directed[:, 0]))]
        counts = np.bincount(directed[:, 0], minlength=prepared_sites.shape[0])
        offsets = _offsets(counts)
        neighbors = directed[:, 1].astype(np.int32)
        construction = restricted_power_cells(
            prepared_sites,
            prepared_weights,
            offsets,
            neighbors,
            np.concatenate((split, np.zeros(guard_count, dtype=np.int8))),
            domain,
            tetrahedra_,
            tetrahedron_regions_,
            tet_face_facets_,
            max_pieces=max_pieces,
            max_vertices=max_vertices,
            work_limit=work_limit,
            record_native_phase=record_native_phase,
        )
        if np.any(construction.cell_sites >= sites.shape[0]):
            raise RuntimeError("A certified empty guard produced a restricted cell.")
        self.points = sites
        self.weights = _frozen(values.copy())
        self.construction = construction
        self.neighbor_offsets = _frozen(offsets)
        self.neighbors = _frozen(neighbors)
        self.guard_count = guard_count
        self.meshcore_identity = meshcore_identity()
        self.periodic_preparation = None
        self.preparation_work_units = 0
        self.preparation_native_charged_work_units = 0
        self.adjacency_work_units = 0
        self.cell_original_sites = construction.cell_sites
        self.piece_original_sites = construction.piece_sites
        self.cell_image_exponents = _frozen(
            np.empty((len(construction.cell_sites), 0), dtype=np.int64)
        )
        self.piece_image_exponents = _frozen(
            np.empty((len(construction.piece_sites), 0), dtype=np.int64)
        )
