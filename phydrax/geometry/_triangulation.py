#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Delaunay, constrained Delaunay, Voronoi, and power diagrams using exact predicates.

All constructions are host-only immutable preparation executed by the native
meshcore library (adaptive expansion arithmetic with index-ordered symbolic
perturbation). Every combinatorial decision is determined by exact predicates
for inputs inside the supported meshcore coordinate-exponent range. Results
are canonically ordered and deterministic, and the native status plus library
identity is retained as :class:`TriangulationEvidence`.

Voronoi and power cells are the generator's (weighted) bisector halfspaces
towards its Delaunay/regular-triangulation neighbors, intersected with an
axis-aligned box and optional convex-domain halfspaces ``{x : n . x <= h}``.
The two cells sharing a bisector use exactly opposite plane coefficients, and
clipper vertex classifications are determined by exact predicates, so cells
cover the domain without overlap up to the rounding of constructed vertex
coordinates. Cells are stored as polygon/polyhedron CSR
(:class:`DiagramCells`) with measures and centroids.

:class:`PeriodicDelaunayTriangulation` triangulates the lattice orbit of a
point set on a flat torus from certified bounded image neighborhoods and keeps
one simplex per orbit with the lattice shift of every corner.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from contextlib import nullcontext
from fractions import Fraction
from itertools import product
from time import perf_counter
from typing import assert_never, Literal, TypeAlias

import equinox as eqx
import numpy as np
import scipy
from numpy.typing import NDArray
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import Delaunay, QhullError

import phydrax._meshcore as meshcore_owner
import phydrax.geometry._periodic_power_preparation as power_preparation_owner

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._geometry_predicates import orient2d_host, orient3d_host, PredicateMode
from .._meshcore import (
    clip_box_halfplanes,
    clip_box_halfspaces,
    constrained_delaunay_2d,
    delaunay_2d,
    delaunay_3d,
    meshcore_identity,
    MeshcoreError,
    MeshcoreStatus,
    NativeExecutionLimits,
    NativeExecutionMemoryCounters,
    NativeExecutionScope,
    periodic_delaunay,
    PeriodicDelaunayEvidence,
    PlanarExecutionEvidence,
    regular_2d,
    regular_3d,
    RestrictedPowerCells,
)
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import positive_integer
from ..discretization._coordinate_budget import CoordinateEnclosureBudget
from ..discretization._periodic_cell import PeriodicCell
from ..discretization._periodic_exact import (
    identification_id,
    PeriodicMeshIdentification,
)
from ..typing import checked, ConvertibleToArray, parse
from ._triangulation_host import frozen_array, host_point_array


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
        content: dict[str, object],
        provider: TriangulationProvider = "meshcore",
    ) -> None:
        status = parse(status, MeshcoreStatus, "TriangulationEvidence.status")
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
            raise ValueError(
                f"TriangulationEvidence.counts must be nonnegative; got {counts}."
            )
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
    """Test affine spanning with host exact predicates."""

    dimension = points.shape[1]
    if points.shape[0] <= dimension:
        return False
    origin = points[0]
    distinct = np.flatnonzero(np.any(points != origin, axis=1))
    if distinct.size == 0:
        return False
    second = points[distinct[0]]
    if dimension == 2:
        turns = orient2d_host(origin, second, points, mode=PredicateMode.EXACT)
        return bool(np.any(turns.signs != 0))
    collinear = np.ones(points.shape[0], dtype=np.bool_)
    for axes in ((1, 2), (2, 0), (0, 1)):
        turns = orient2d_host(
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
    volumes = orient3d_host(origin, second, third, points, mode=PredicateMode.EXACT)
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
            raise MeshcoreError(
                code,
                f"{operation}.cells must finish native clipping; got cell {int(failed[0])} status {code.name}.",
            )
        case _:
            raise ValueError(
                f"{operation}.cells must have valid clipper inputs; got cell {int(failed[0])} status {code.name}."
            )


def _qhull_delaunay(
    points: np.ndarray, max_simplices: int | None, /
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Prepare SciPy Qhull Delaunay simplices with positive orientation.

    Rows are sorted, then the last two vertices swap where needed for positive
    orientation, and rows are ordered lexicographically. Points Qhull leaves
    out map to their nearest vertex (``Qc``); identical ones are duplicates.
    """

    count, dimension = points.shape
    if count <= dimension:
        raise ValueError(
            f"DelaunayTriangulation.points must contain more rows than dimensions; got {count} rows in dimension {dimension}."
        )
    try:
        triangulation = Delaunay(points, qhull_options=_QHULL_OPTIONS)
    except QhullError as error:
        raise ValueError(
            f"DelaunayTriangulation.points must admit Qhull construction; got {error}."
        ) from error
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
            f"DelaunayTriangulation.max_simplices must cover the Qhull output; got {simplices.shape[0]} simplices against {max_simplices}.",
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
    ``"meshcore"`` provider (default) uses exact predicates: cocircular and
    cospherical ties are resolved by index-ordered symbolic perturbation.
    The ``"qhull"`` provider uses SciPy's Qhull (a core dependency) with the recorded options;
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
        points: ConvertibleToArray,
        /,
        *,
        max_simplices: int | None = None,
        provider: TriangulationProvider = "meshcore",
    ) -> None:
        point_array = host_point_array(points, "points", (2, 3))
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
                duplicates = int(
                    np.count_nonzero(vertex_map != np.arange(count, dtype=np.int64))
                )
                redundant = 0
            case "qhull":
                simplices, vertex_map, duplicates, redundant = _qhull_delaunay(
                    point_array, max_simplices
                )
            case unreachable:
                assert_never(unreachable)
        self.points = point_array
        self.simplices = frozen_array(simplices)
        self.vertex_map = frozen_array(vertex_map)
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
    ``work_evidence`` retains the distinct nine planar work measurements;
    ``memory_counters`` owns native allocator measurements. Resource refusals
    expose the same values on ``NativeResourceFailure`` instead of a partial mesh.
    """

    points: np.ndarray
    triangles: np.ndarray
    segment_ids: np.ndarray
    input_point_count: int = eqx.field(static=True)
    evidence: TriangulationEvidence
    work_evidence: NDArray[np.uint64]
    memory_counters: NativeExecutionMemoryCounters
    constraint_parents: np.ndarray
    constraint_sources: np.ndarray
    constraint_ratios: np.ndarray

    def __init__(
        self,
        points: ConvertibleToArray,
        segments: ConvertibleToArray,
        /,
        *,
        holes: ConvertibleToArray | None = None,
        keep_convex_hull: bool = False,
        min_angle: float = 0.0,
        max_area: float = math.inf,
        max_steiner: int = 0,
        max_triangles: int | None = None,
        max_cavity_cells: int | None = None,
        maximum_work: int | None = None,
        max_scratch_bytes: int | None = None,
    ) -> None:
        point_array = host_point_array(points, "points", (2,))
        measurements: list[PlanarExecutionEvidence] = []
        witnesses: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

        def retain_witnesses(
            parents: np.ndarray, sources: np.ndarray, ratios: np.ndarray
        ) -> None:
            witnesses.append((parents, sources, ratios))

        result = constrained_delaunay_2d(
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
            record_constraint_witnesses=retain_witnesses,
        )
        mesh_points = result.points
        triangles = result.triangles
        segment_ids = result.segment_ids
        status = result.status
        count = point_array.shape[0]
        used = np.unique(triangles)
        self.points = frozen_array(mesh_points)
        self.triangles = frozen_array(triangles)
        self.segment_ids = frozen_array(segment_ids)
        self.input_point_count = count
        self.work_evidence = measurements[0].work_evidence
        self.memory_counters = measurements[0].memory_counters
        self.constraint_parents, self.constraint_sources, self.constraint_ratios = (
            witnesses[0]
        )
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
                "constraint_parents": array_tree_fingerprint(self.constraint_parents),
                "constraint_sources": array_tree_fingerprint(self.constraint_sources),
                "constraint_ratios": array_tree_fingerprint(self.constraint_ratios),
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
    owners = np.tile(np.arange(count, dtype=np.int64), width)
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
        (
            np.ones(np.count_nonzero(inside), dtype=np.float64),
            (first[inside], second[inside]),
        ),
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
    """Return usable simplices needed to restore vertex and component coverage.

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
        self,
        points: ConvertibleToArray,
        simplices: ConvertibleToArray,
        /,
        *,
        minimum_quality: float,
    ) -> None:
        point_array = host_point_array(points, "points", (2, 3))
        dimension = point_array.shape[1]
        simplex_array = np.asarray(simplices)
        if not np.issubdtype(simplex_array.dtype, np.integer):
            raise TypeError(
                f"SimplexQualitySubcomplex.simplices must be an integer array; got dtype {simplex_array.dtype}."
            )
        if (
            simplex_array.ndim != 2
            or simplex_array.shape[1] != dimension + 1
            or simplex_array.shape[0] == 0
        ):
            raise ValueError(
                f"SimplexQualitySubcomplex.simplices must have shape (n, dimension + 1) with n >= 1; got {simplex_array.shape}."
            )
        if np.any(simplex_array < 0) or np.any(simplex_array >= point_array.shape[0]):
            raise ValueError(
                "SimplexQualitySubcomplex.simplices must index the point bank; got out-of-range indices."
            )
        threshold = float(minimum_quality)
        if not math.isfinite(threshold) or not 0.0 <= threshold < 1.0:
            raise ValueError(
                f"SimplexQualitySubcomplex.minimum_quality must lie in [0, 1); got {threshold}."
            )
        simplex_array = simplex_array.astype(np.int64)
        measures, quality = _simplex_quality(point_array, simplex_array)
        usable = quality > _DEGENERATE_QUALITY
        if not np.any(usable):
            raise ValueError(
                "SimplexQualitySubcomplex.simplices must contain a positive usable measure; got only degenerate cells."
            )
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
        self.simplices = frozen_array(simplex_array[retained].astype(np.int32))
        self.quality = frozen_array(quality)
        self.retained = frozen_array(retained)
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
            raise ValueError(
                f"DiagramCells.cell_offsets must have shape (cells + 1,); got {cell_vertex_offsets.shape} and {cell_face_offsets.shape}."
            )
        if face_vertex_offsets.shape != (face_labels.shape[0] + 1,):
            raise ValueError(
                f"DiagramCells.face_vertex_offsets must have shape (faces + 1,); got {face_vertex_offsets.shape}."
            )
        if centroids.shape != (cell_count, vertices.shape[1]):
            raise ValueError(
                f"DiagramCells.centroids must have shape (cells, dimension); got {centroids.shape}."
            )
        self.vertices = frozen_array(vertices)
        self.cell_vertex_offsets = frozen_array(cell_vertex_offsets)
        self.cell_face_offsets = frozen_array(cell_face_offsets)
        self.face_vertex_offsets = frozen_array(face_vertex_offsets)
        self.face_vertices = frozen_array(face_vertices)
        self.face_labels = frozen_array(face_labels)
        self.measures = frozen_array(measures)
        self.centroids = frozen_array(centroids)

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


class _Planes(StrictModule):
    """Retain halfspace CSR per cell with global face labels."""

    offsets: np.ndarray
    normals: np.ndarray
    values: np.ndarray
    labels: np.ndarray


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
    uses sign reversal without further rounding, so neighboring cells share
    one plane.
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
        rows = (base[:, None] + np.arange(domain_count, dtype=np.int64)[None, :]).reshape(
            (-1,)
        )
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
        result = clip_box_halfplanes(box_lower, box_upper, normals, values, counts)
        vertices = result.vertices
        edge_labels = result.edge_labels
        vertex_counts = result.vertex_counts
        measure = result.area
        moment = result.first_moment
        status = result.status
        _raise_for_items(status, "VoronoiDiagram")
        mask = (
            np.arange(vertices.shape[1], dtype=np.int64)[None, :] < vertex_counts[:, None]
        )
        flat_vertices = vertices[mask]
        local = edge_labels[mask]
        cell_rows = np.repeat(np.arange(cells.size, dtype=np.int64), vertex_counts)
        face_labels = np.where(
            local >= 0, label_table[cell_rows, np.maximum(local, 0)], local
        ).astype(np.int32)
        position = np.arange(flat_vertices.shape[0], dtype=np.int64) - np.repeat(
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
    result = clip_box_halfspaces(box_lower, box_upper, normals, values, counts)
    vertices = result.vertices
    vertex_counts = result.vertex_counts
    face_offsets = result.face_offsets
    face_label_table = result.face_labels
    face_vertices = result.face_vertices
    face_counts = result.face_counts
    measure = result.volume
    moment = result.first_moment
    status = result.status
    _raise_for_items(status, "VoronoiDiagram")
    vertex_mask = (
        np.arange(vertices.shape[1], dtype=np.int64)[None, :] < vertex_counts[:, None]
    )
    face_mask = (
        np.arange(face_label_table.shape[1], dtype=np.int64)[None, :]
        < face_counts[:, None]
    )
    local = face_label_table[face_mask]
    cell_rows = np.repeat(np.arange(cells.size, dtype=np.int64), face_counts)
    face_labels = np.where(
        local >= 0, label_table[cell_rows, np.maximum(local, 0)], local
    ).astype(np.int32)
    face_sizes = np.diff(face_offsets, axis=1)[face_mask].astype(np.int64)
    used = face_offsets[np.arange(cells.size, dtype=np.int64), face_counts]
    incidence_mask = (
        np.arange(face_vertices.shape[1], dtype=np.int64)[None, :] < used[:, None]
    )
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
    box_lower: ConvertibleToArray,
    box_upper: ConvertibleToArray,
    domain_normals: ConvertibleToArray | None,
    domain_offsets: ConvertibleToArray | None,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lower = np.asarray(box_lower, dtype=np.float64)
    upper = np.asarray(box_upper, dtype=np.float64)
    if lower.shape != (dimension,) or upper.shape != (dimension,):
        raise ValueError(
            f"Diagram.box_bounds must have shape ({dimension},); got {lower.shape} and {upper.shape}."
        )
    if not (np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))):
        raise ValueError("Diagram.box_bounds must be finite; got nonfinite entries.")
    if not np.all(lower < upper):
        raise ValueError(
            "Diagram.box_lower must be strictly below box_upper; got a nonpositive extent."
        )
    if (domain_normals is None) != (domain_offsets is None):
        raise ValueError(
            "Diagram.domain_halfspaces must supply normals and offsets together; got one missing bank."
        )
    if domain_normals is None:
        return (
            lower,
            upper,
            np.zeros((0, dimension), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
        )
    normals = host_point_array(domain_normals, "domain_normals", (dimension,))
    offsets = np.asarray(domain_offsets, dtype=np.float64)
    if offsets.shape != (normals.shape[0],) or not np.all(np.isfinite(offsets)):
        raise ValueError(
            f"Diagram.domain_offsets must be a finite (k,) array; got shape {offsets.shape}."
        )
    if np.any(np.all(normals == 0.0, axis=1)):
        raise ValueError("Diagram.domain_normals must be nonzero; got a zero normal.")
    return lower, upper, normals, offsets


def _all_pairs(count: int, /) -> np.ndarray:
    if count > _DEGENERATE_ALL_PAIRS_LIMIT:
        raise ValueError(
            "Diagram.nonspanning_generators must fit the bounded all-pairs route; "
            f"got {count} generators against {_DEGENERATE_ALL_PAIRS_LIMIT}."
        )
    first, second = np.triu_indices(count, k=1)
    return np.stack((first, second), axis=1).astype(np.int64)


class VoronoiDiagram(StrictModule, NonTrainableState):
    """Bounded Voronoi cells of generators clipped to a box and convex domain.

    Cells are dual to Delaunay triangulation constructed with exact predicates
    (``dual``). Generators that do not span the space use every generator pair
    as a bisector candidate (bounded route, ``dual is None``). Duplicate
    generators keep one representative cell; the others are empty.
    """

    points: np.ndarray
    cells: DiagramCells
    dual: DelaunayTriangulation | None
    evidence: TriangulationEvidence

    def __init__(
        self,
        points: ConvertibleToArray,
        /,
        *,
        box_lower: ConvertibleToArray,
        box_upper: ConvertibleToArray,
        domain_normals: ConvertibleToArray | None = None,
        domain_offsets: ConvertibleToArray | None = None,
    ) -> None:
        point_array = host_point_array(points, "points", (2, 3))
        count, dimension = point_array.shape
        lower, upper, normals, offsets = _domain(
            dimension, box_lower, box_upper, domain_normals, domain_offsets
        )
        if _spans_space(point_array):
            dual = DelaunayTriangulation(point_array)
            neighbors = _simplex_edges(dual.simplices)
            active = dual.vertex_map == np.arange(count, dtype=np.int64)
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
    Cells are dual to regular triangulation constructed with exact predicates
    (``dual_simplices``); redundant generators (``dual_vertex_map == -1``)
    and lighter coincident generators have empty cells. Generators that do not
    span the space use the bounded all-pairs route.
    """

    points: np.ndarray
    weights: np.ndarray
    cells: DiagramCells
    dual_simplices: np.ndarray | None
    dual_vertex_map: np.ndarray | None
    evidence: TriangulationEvidence

    def __init__(
        self,
        points: ConvertibleToArray,
        weights: ConvertibleToArray,
        /,
        *,
        box_lower: ConvertibleToArray,
        box_upper: ConvertibleToArray,
        domain_normals: ConvertibleToArray | None = None,
        domain_offsets: ConvertibleToArray | None = None,
        max_simplices: int | None = None,
    ) -> None:
        point_array = host_point_array(points, "points", (2, 3))
        count, dimension = point_array.shape
        weight_array = np.asarray(weights)
        if not (
            np.issubdtype(weight_array.dtype, np.floating)
            or np.issubdtype(weight_array.dtype, np.integer)
        ) or np.issubdtype(weight_array.dtype, np.bool_):
            raise TypeError(
                f"PowerDiagram.weights must be a real floating or integer array; got dtype {weight_array.dtype}."
            )
        if weight_array.shape != (count,) or not np.all(np.isfinite(weight_array)):
            raise ValueError(
                f"PowerDiagram.weights must be a finite (n,) array; got shape {weight_array.shape}."
            )
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
            active = vertex_map == np.arange(count, dtype=np.int64)
            dual_simplices: np.ndarray | None = frozen_array(simplices)
            dual_vertex_map: np.ndarray | None = frozen_array(vertex_map)
            route = f"power_{dimension}d"
        else:
            # Coincident generators: the heaviest (then smallest index) owns the cell.
            order = np.lexsort(
                (np.arange(count, dtype=np.int64), -weight_array, *point_array.T[::-1])
            )
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

    @checked
    def __init__(
        self,
        status: MeshcoreStatus,
        native: PeriodicDelaunayEvidence,
        /,
        *,
        dimension: int,
        input_point_count: int,
        maximum_images: int,
        maximum_cells: int,
        simplex_count: int,
        content: dict[str, object],
    ) -> None:
        status = parse(status, MeshcoreStatus, "PeriodicTriangulationEvidence.status")
        limit: PeriodicImageLimit
        match int(native.exhausted_limit):
            case 0:
                limit = "none"
            case 1:
                limit = "images"
            case 2:
                limit = "cells"
            case code:
                raise ValueError(
                    f"PeriodicTriangulationEvidence.exhausted_limit must identify a known native limit; got code {code}."
                )
        identity = meshcore_identity()
        self.status = status.name.lower()
        self.meshcore_identity = identity
        self.dimension = dimension
        self.input_point_count = input_point_count
        self.rounds = int(native.rounds)
        self.margin = native.margin
        self.image_count = int(native.image_count)
        self.maximum_images = maximum_images
        self.uncertified_cell_count = int(native.uncertified_cells)
        self.required_margin = native.required_margin
        self.finite_cell_slots = int(native.finite_cell_slots)
        self.maximum_cells = maximum_cells
        self.exact_evaluations = int(native.exact_evaluations)
        self.perturbed_decisions = int(native.perturbed_decisions)
        self.exhausted_limit = limit
        self.simplex_count = simplex_count
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "periodic-triangulation-evidence",
                "status": self.status,
                "meshcore": identity,
                "native": array_tree_fingerprint(
                    np.asarray([value for _, value in native.items()], dtype=np.float64)
                ),
                "budgets": [maximum_images, maximum_cells],
                "content": content,
            }
        )


class PeriodicImageBudgetError(MeshcoreError):
    """A periodic triangulation needed more images or cells than its budget.

    ``evidence`` records the refused round: its margin, the images it would
    need, the uncertified cells of the previous round and the exhausted limit.
    """

    @checked
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


class PeriodicDelaunayTriangulation(StrictModule, NonTrainableState):
    """Triangulate a translationally periodic point set using exact predicates.

    The point set is the lattice orbit of ``points`` under a fully periodic
    full-rank :class:`~phydrax.discretization.PeriodicCell`; representatives
    keep their input coordinates. ``simplices`` holds one positively oriented
    simplex per orbit of the periodic triangulation (representative indices)
    and ``simplex_shifts`` the lattice image of each corner, so corner ``k`` of
    simplex ``t`` lies at ``points[simplices[t, k]] + simplex_shifts[t, k] @
    cell.vectors``; the published member of each orbit is the one whose anchor
    corner (smallest representative, then smallest shift) is the anchor's
    image in the fundamental cell. Translated positions use exact predicates,
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

    @checked
    def __init__(
        self,
        points: ConvertibleToArray,
        cell: PeriodicCell,
        /,
        *,
        initial_margin: float | None = None,
        maximum_images: int | None = None,
        maximum_simplices: int | None = None,
    ) -> None:
        point_array = host_point_array(points, "points", (2, 3))
        count, dimension = point_array.shape
        if count == 0:
            raise ValueError(
                "PeriodicDelaunayTriangulation.points must contain at least one row; got an empty bank."
            )
        if (
            cell.ambient_dimension != dimension
            or cell.rank != dimension
            or not cell.fully_periodic
        ):
            raise ValueError(
                "PeriodicDelaunayTriangulation.cell must be a fully periodic "
                "full-rank lattice in the point dimension; got incompatible lattice axes."
            )
        # Delaunay circumradii scale like the mean spacing n^(-1/d) in lattice units.
        margin = (
            min(1.0, 2.0 * count ** (-1.0 / dimension))
            if initial_margin is None
            else float(initial_margin)
        )
        if not math.isfinite(margin) or margin <= 0.0:
            raise ValueError(
                f"PeriodicDelaunayTriangulation.initial_margin must be positive and finite; got {margin}."
            )
        images = (
            27 * count + 4096
            if maximum_images is None
            else positive_integer(maximum_images, "maximum_images")
        )
        cells = (
            64 * images
            if maximum_simplices is None
            else positive_integer(maximum_simplices, "maximum_simplices")
        )
        vectors = np.asarray(cell.vectors, dtype=np.float64)
        inverse = np.asarray(cell.inverse_vectors, dtype=np.float64)
        fractional = (point_array - np.asarray(cell.origin, dtype=np.float64)) @ inverse
        result = periodic_delaunay(
            point_array,
            fractional,
            vectors,
            inverse,
            initial_margin=margin,
            max_images=images,
            max_cells=cells,
        )
        status = result.status
        simplices = result.vertices
        shifts = result.shifts
        native = result.evidence
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
                duplicate = int(native.duplicate_representative)
                raise ValueError(
                    f"PeriodicDelaunayTriangulation.points must have distinct lattice orbit representatives; got duplicate row {duplicate}."
                )
            case _:
                raise MeshcoreError(
                    status,
                    f"PeriodicDelaunayTriangulation.status must identify a supported native outcome; got {status.name}.",
                )
        self.cell = cell
        self.points = point_array
        self.simplices = frozen_array(simplices)
        self.simplex_shifts = frozen_array(shifts)
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


class PeriodicPowerSourceRefusal(MeshcoreError):
    """Retain the immutable power source and native refusal evidence."""

    @checked
    def __init__(
        self,
        preparation: power_preparation_owner.PeriodicPowerPreparation,
        refusal: MeshcoreError,
        maximum_work: int,
        /,
        *,
        requested_work: int | None = None,
        completed_work: int | None = None,
    ) -> None:
        super().__init__(
            refusal.status,
            "PeriodicPowerSourceRefusal.construction must finish within the source allowance; got a native refusal.",
            work_counters=refusal.work_counters,
            memory_counters=refusal.memory_counters,
        )
        self.preparation = preparation
        self.native_refusal = refusal
        self.maximum_work = maximum_work
        self.requested_work = requested_work
        self.completed_work = completed_work


def _restricted_periodic_power_construction(
    preparation: power_preparation_owner.PeriodicPowerPreparation,
    split: np.ndarray,
    domain: np.ndarray,
    tetrahedra: np.ndarray,
    regions: np.ndarray,
    facets: np.ndarray,
    /,
    *,
    max_pieces: int,
    max_vertices: int,
    work_limit: int,
    record_native_phase: Callable[[str, float, int | None, int], None] | None,
    scope: NativeExecutionScope,
) -> tuple[np.ndarray, np.ndarray, RestrictedPowerCells, int]:

    count = len(preparation.image_sites)
    adjacency_count = count * (count - 1)
    if adjacency_count >= work_limit:
        raise PeriodicPowerSourceRefusal(
            preparation,
            MeshcoreError(
                MeshcoreStatus.CAPACITY_EXCEEDED,
                f"RestrictedPowerDiagram.adjacency_work must stay below work_limit; got {adjacency_count} against {work_limit}.",
            ),
            work_limit,
            requested_work=adjacency_count + 1,
            completed_work=0,
        )
    # Full adjacency retains all source candidates before clipping; no rounded
    # regular triangulation determines the image bank. Materialization is bounded before
    # allocation and charged as host work, never as native geometric predicates.
    array_bytes = (count + 1) * np.dtype(np.int64).itemsize + (
        adjacency_count + count
    ) * np.dtype(np.int32).itemsize
    with scope.batch(adjacency_count, bytes_upper=array_bytes) as ticket:
        ticket.consume(managed_bytes=array_bytes)
        offsets = np.arange(count + 1, dtype=np.int64) * (count - 1)
        neighbors = np.empty(adjacency_count, dtype=np.int32)
        row = np.arange(count, dtype=np.int32)
        materialized_work = 0
        for image in range(count):
            left, right = row[:image], row[image + 1 :]
            row_work = len(left) + len(right)
            ticket.consume(row_work)
            start = int(offsets[image])
            neighbors[start : start + len(left)] = left
            neighbors[start + len(left) : int(offsets[image + 1])] = right
            materialized_work += row_work
    construction = meshcore_owner.restricted_power_cells_exact(
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


class _RestrictedSource(StrictModule):
    sites: np.ndarray
    values: np.ndarray
    domain: np.ndarray
    split: np.ndarray
    tetrahedra: np.ndarray
    regions: np.ndarray
    facets: np.ndarray


class _RestrictedControls(StrictModule):
    max_pieces: int = eqx.field(static=True)
    max_vertices: int = eqx.field(static=True)
    work_limit: int = eqx.field(static=True)
    record_native_phase: Callable[[str, float, int | None, int], None] | None


class _RestrictedConstruction(StrictModule):
    points: np.ndarray
    weights: np.ndarray
    construction: RestrictedPowerCells
    offsets: np.ndarray
    neighbors: np.ndarray
    preparation: power_preparation_owner.PeriodicPowerPreparation | None
    preparation_work: int = eqx.field(static=True)
    preparation_native_work: int = eqx.field(static=True)
    adjacency_work: int = eqx.field(static=True)
    guard_count: int = eqx.field(static=True)


def _restricted_source(
    points: ConvertibleToArray,
    weights: ConvertibleToArray,
    domain_points: ConvertibleToArray,
    tetrahedra: ConvertibleToArray,
    tetrahedron_regions: ConvertibleToArray,
    tet_face_facets: ConvertibleToArray,
    split_sites: ConvertibleToArray | None,
) -> _RestrictedSource:
    sites = host_point_array(points, "points", (3,))
    if sites.shape[0] == 0:
        raise ValueError(
            "RestrictedPowerDiagram.points must contain at least one site; got an empty bank."
        )
    values = np.asarray(weights, dtype=np.float64)
    if values.shape != (sites.shape[0],) or not np.all(np.isfinite(values)):
        raise ValueError(
            f"RestrictedPowerDiagram.weights must be a finite (site_count,) array; got shape {values.shape}."
        )
    domain = host_point_array(domain_points, "domain_points", (3,))
    split = (
        np.zeros(sites.shape[0], dtype=np.int8)
        if split_sites is None
        else np.asarray(split_sites, dtype=np.int8)
    )
    tetrahedra_ = np.asarray(tetrahedra)
    tetrahedron_regions_ = np.asarray(tetrahedron_regions)
    tet_face_facets_ = np.asarray(tet_face_facets)
    if split.shape != (sites.shape[0],) or np.any((split != 0) & (split != 1)):
        raise ValueError(
            f"RestrictedPowerDiagram.split_sites must be a binary (site_count,) array; got shape {split.shape}."
        )
    return _RestrictedSource(
        sites, values, domain, split, tetrahedra_, tetrahedron_regions_, tet_face_facets_
    )


def _prepare_restricted_periodic_source(
    source: _RestrictedSource,
    periodic_group: PeriodicMeshIdentification,
    maximum_images: int | None,
    controls: _RestrictedControls,
    scope: NativeExecutionScope,
    budget: CoordinateEnclosureBudget | None,
) -> power_preparation_owner.PeriodicPowerPreparation:
    sites, values, domain = source.sites, source.values, source.domain
    work_limit = controls.work_limit
    if budget is not None and not budget.ticket.settled:
        preparation = power_preparation_owner.PeriodicPowerPreparation(
            sites,
            values,
            domain,
            periodic_group,
            maximum_images=(
                27 * len(sites) + 4096
                if maximum_images is None
                else positive_integer(maximum_images, "maximum_images")
            ),
            maximum_work_units=work_limit,
            budget=budget,
        )
    else:
        remaining = scope.remaining()
        work_ticket = scope.batch(
            min(work_limit, remaining.remaining_work_units),
            queries_upper=remaining.remaining_geometry_queries,
            bytes_upper=remaining.remaining_managed_bytes,
        )
        if budget is None:
            view = CoordinateEnclosureBudget(
                work_ticket,
                maximum_work_units=work_limit,
                maximum_memory_bytes=int(np.iinfo(np.intp).max),
            )
            with work_ticket, view.activate():
                preparation = power_preparation_owner.PeriodicPowerPreparation(
                    sites,
                    values,
                    domain,
                    periodic_group,
                    maximum_images=(
                        27 * len(sites) + 4096
                        if maximum_images is None
                        else positive_integer(maximum_images, "maximum_images")
                    ),
                    maximum_work_units=work_limit,
                    budget=view,
                )
        else:
            with budget.bind_ticket(work_ticket):
                preparation = power_preparation_owner.PeriodicPowerPreparation(
                    sites,
                    values,
                    domain,
                    periodic_group,
                    maximum_images=(
                        27 * len(sites) + 4096
                        if maximum_images is None
                        else positive_integer(maximum_images, "maximum_images")
                    ),
                    maximum_work_units=work_limit,
                    budget=budget,
                )
    return preparation


def _authenticate_restricted_periodic_source(
    source: _RestrictedSource,
    preparation: power_preparation_owner.PeriodicPowerPreparation,
    periodic_group: PeriodicMeshIdentification | None,
    maximum_images: int | None,
    work_limit: int,
) -> None:
    sites, values, domain = source.sites, source.values, source.domain
    if (
        array_tree_fingerprint(sites) != array_tree_fingerprint(preparation.points)
        or array_tree_fingerprint(values) != array_tree_fingerprint(preparation.weights)
        or canonical_fingerprint(array_tree_fingerprint(domain)) != preparation.carrier_id
    ):
        raise ValueError(
            "RestrictedPowerDiagram.periodic_preparation.source must match the site, weight, and carrier bytes; got different content."
        )
    if maximum_images is not None and len(preparation.image_sites) > (
        1
        if maximum_images is None
        else positive_integer(maximum_images, "maximum_images")
    ):
        raise PeriodicPowerSourceRefusal(
            preparation,
            power_preparation_owner.PeriodicPowerImageCapacityRefusal(
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
        identity = identification_id(periodic_group)
        if identity != preparation.identification_id:
            raise ValueError(
                "RestrictedPowerDiagram.periodic_preparation.identification_id must match the group identity; got a different identification."
            )


def _execute_restricted_periodic_source(
    source: _RestrictedSource,
    preparation: power_preparation_owner.PeriodicPowerPreparation,
    preparation_work_units: int,
    preparation_native_charged_work_units: int,
    controls: _RestrictedControls,
    scope: NativeExecutionScope,
    budget: CoordinateEnclosureBudget | None,
) -> _RestrictedConstruction:
    split, domain, tetrahedra_, tetrahedron_regions_, tet_face_facets_ = (
        source.split,
        source.domain,
        source.tetrahedra,
        source.regions,
        source.facets,
    )
    max_pieces, max_vertices, work_limit, record_native_phase = (
        controls.max_pieces,
        controls.max_vertices,
        controls.work_limit,
        controls.record_native_phase,
    )
    try:
        with nullcontext() if budget is None else budget.measurement():
            construction_work_limit = work_limit - preparation_work_units
            if construction_work_limit < 1:
                raise PeriodicPowerSourceRefusal(
                    preparation,
                    MeshcoreError(
                        MeshcoreStatus.CAPACITY_EXCEEDED,
                        f"RestrictedPowerDiagram.preparation_work must leave construction allowance; got {preparation_work_units} against {work_limit}.",
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
                    scope=scope,
                )
            )
    except PeriodicPowerSourceRefusal:
        raise
    except MeshcoreError as refusal:
        raise PeriodicPowerSourceRefusal(preparation, refusal, work_limit) from refusal
    return _RestrictedConstruction(
        preparation.points,
        preparation.weights,
        construction,
        offsets,
        neighbors,
        preparation,
        preparation_work_units,
        preparation_native_charged_work_units,
        adjacency_work_units,
        0,
    )


def _execute_restricted_source(
    source: _RestrictedSource, controls: _RestrictedControls
) -> _RestrictedConstruction:
    sites, values, domain, split, tetrahedra_, tetrahedron_regions_, tet_face_facets_ = (
        source.sites,
        source.values,
        source.domain,
        source.split,
        source.tetrahedra,
        source.regions,
        source.facets,
    )
    max_pieces, max_vertices, work_limit, record_native_phase = (
        controls.max_pieces,
        controls.max_vertices,
        controls.work_limit,
        controls.record_native_phase,
    )
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
                raise ValueError(
                    "RestrictedPowerDiagram.domain_points must admit finite spanning guards; got an unrepresentable domain scale."
                )
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
                # value lies outside meshcore's supported weight-exponent range.
                weight = float(minimum - max(Fraction(1), abs(minimum)) - 1)
                weight = float(np.nextafter(weight, -np.inf))
                if not math.isfinite(weight) or Fraction(weight) >= minimum:
                    raise ValueError(
                        "RestrictedPowerDiagram.guard_weights must establish a strict empty-cell power bound; got an unrepresentable bound."
                    )
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
    construction = meshcore_owner.restricted_power_cells(
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
        raise RuntimeError(
            "RestrictedPowerDiagram.construction.cell_sites must exclude certified empty guards; got a guard cell."
        )
    return _RestrictedConstruction(
        sites,
        frozen_array(values.copy()),
        construction,
        offsets,
        neighbors,
        None,
        0,
        0,
        0,
        guard_count,
    )


class RestrictedPowerDiagram(StrictModule, NonTrainableState):
    """Native connected power cells restricted to a constrained tet complex.

    Unlike the convex diagram contracts above this preserves disconnected site
    components and constrained material/sheet faces. ``construction`` carries
    reciprocal faces and the site/tet/cell piece incidence. This is construction
    evidence, not an independent coverage certificate.
    """

    points: np.ndarray
    weights: np.ndarray
    construction: RestrictedPowerCells
    neighbor_offsets: np.ndarray
    neighbors: np.ndarray
    periodic_preparation: power_preparation_owner.PeriodicPowerPreparation | None
    preparation_work_units: int = eqx.field(static=True)
    preparation_native_charged_work_units: int = eqx.field(static=True)
    adjacency_work_units: int = eqx.field(static=True)
    cell_original_sites: np.ndarray
    piece_original_sites: np.ndarray
    cell_image_exponents: np.ndarray
    piece_image_exponents: np.ndarray
    guard_count: int = eqx.field(static=True)
    meshcore_identity: str = eqx.field(static=True)

    @staticmethod
    def native_limits(work_limit: int, /) -> NativeExecutionLimits:
        """Lower existing work/storage; no cavity control bounds this request."""
        return NativeExecutionLimits(
            work_limit,
            work_limit,
            int(np.iinfo(np.intp).max),
            int(np.iinfo(np.intp).max),
            math.inf,
        )

    @checked
    def __init__(
        self,
        points: ConvertibleToArray,
        weights: ConvertibleToArray,
        domain_points: ConvertibleToArray,
        tetrahedra: ConvertibleToArray,
        tetrahedron_regions: ConvertibleToArray,
        tet_face_facets: ConvertibleToArray,
        /,
        *,
        split_sites: ConvertibleToArray | None = None,
        periodic_group: PeriodicMeshIdentification | None = None,
        periodic_preparation: power_preparation_owner.PeriodicPowerPreparation
        | None = None,
        maximum_images: int | None = None,
        max_pieces: int = 1 << 20,
        max_vertices: int = 1 << 22,
        work_limit: int = 1 << 26,
        record_native_phase: Callable[[str, float, int | None, int], None] | None = None,
        scope: NativeExecutionScope | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> None:

        source = _restricted_source(
            points,
            weights,
            domain_points,
            tetrahedra,
            tetrahedron_regions,
            tet_face_facets,
            split_sites,
        )
        controls = _RestrictedControls(
            max_pieces, max_vertices, work_limit, record_native_phase
        )
        with (
            NativeExecutionScope(self.native_limits(work_limit))
            if scope is None
            else nullcontext(scope)
        ) as execution:
            preparation = periodic_preparation
            preparation_work = preparation_native_work = 0
            if preparation is None and periodic_group is not None:
                preparation = _prepare_restricted_periodic_source(
                    source, periodic_group, maximum_images, controls, execution, budget
                )
                preparation_work = preparation.spent_work
                preparation_native_work = preparation.native_charged_work
            if preparation is not None:
                _authenticate_restricted_periodic_source(
                    source, preparation, periodic_group, maximum_images, work_limit
                )
                result = _execute_restricted_periodic_source(
                    source,
                    preparation,
                    preparation_work,
                    preparation_native_work,
                    controls,
                    execution,
                    budget,
                )
            else:
                result = _execute_restricted_source(source, controls)
            self._commit(result)

    def _commit(self, result: _RestrictedConstruction) -> None:
        self.points = result.points
        self.weights = result.weights
        self.construction = result.construction
        self.neighbor_offsets = frozen_array(result.offsets)
        self.neighbors = frozen_array(result.neighbors)
        self.periodic_preparation = result.preparation
        self.preparation_work_units = result.preparation_work
        self.preparation_native_charged_work_units = result.preparation_native_work
        self.adjacency_work_units = result.adjacency_work
        preparation = result.preparation
        construction = result.construction
        if preparation is not None:
            self.cell_original_sites = frozen_array(
                preparation.image_sites[construction.cell_sites]
            )
            self.piece_original_sites = frozen_array(
                preparation.image_sites[construction.piece_sites]
            )
            self.cell_image_exponents = frozen_array(
                preparation.image_exponents[construction.cell_sites]
            )
            self.piece_image_exponents = frozen_array(
                preparation.image_exponents[construction.piece_sites]
            )
        else:
            self.cell_original_sites = construction.cell_sites
            self.piece_original_sites = construction.piece_sites
            self.cell_image_exponents = frozen_array(
                np.empty((len(construction.cell_sites), 0), dtype=np.int64)
            )
            self.piece_image_exponents = frozen_array(
                np.empty((len(construction.piece_sites), 0), dtype=np.int64)
            )
        self.guard_count = result.guard_count
        self.meshcore_identity = meshcore_identity()
