#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Host validation of labeled multiregion surfaces with exact predicates.

Four independent certificates are combined:

1. **Region cycles.** For every edge and finite label, the signed incidence of
   the label's oriented boundary faces must vanish (the boundary of the 2-chain
   is zero). An odd unsigned count means the region is open; an even count with
   a nonzero sum means inconsistent labels or orientation.
2. **Wedge labels.** The faces around each edge are ordered counterclockwise
   about ``v1 - v0`` with exact ``orient3d`` signs. Between consecutive faces
   ``a -> b`` the wedge region must be the same seen from both faces: it is
   ``right(a)`` when ``a`` traverses ``v0 -> v1`` and ``left(a)`` otherwise, and
   ``left(b)`` when ``b`` traverses ``v0 -> v1`` and ``right(b)`` otherwise. An
   edge with no mismatched wedge is interior; exactly one mismatched wedge
   between two boundary labels is a wire border; anything else is rejected.
3. **Vertex stars and region connectivity.** Every sheet meets a vertex in one
   fan (no singular pinch points); under the dry-foam profile every pair of
   regions meeting at an interior vertex shares a film there (complete region
   graph). Face sides joined across consistent wedges give the connected
   components of every label; a finite region must be one component.
4. **Embedding.** Exact face degeneracy, positive signed finite-region volumes
   and BVH broad phase plus exact triangle-pair narrow phase for
   self-intersection away from shared vertices and edges.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from ..._bvh import bvh_overlap_pair_blocks, prepare_bvh
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._geometry_predicates import (
    orient2d,
    orient3d,
    PredicateMode,
    resolve_host_predicate_mode,
)
from ._contracts import (
    DRY_FOAM_EDGE_VALENCE,
    DRY_FOAM_VERTEX_REGIONS,
    MultiRegionSurfaceEvidence,
    MultiRegionSurfaceStatus,
    MultiRegionSurfaceValidationPolicy,
)
from ._state import MultiRegionSurfaceState
from ._topology import MultiRegionSurfaceTopology


_PROJECTION_AXES = np.asarray(((1, 2), (0, 2), (0, 1)))


@dataclass(frozen=True, slots=True)
class _Combinatorial:
    consistent: bool
    inconsistent_edges: int
    watertight: bool
    open_edges: int


@dataclass(frozen=True, slots=True)
class _Wedges:
    consistent: bool
    inconsistent_edges: int
    border: np.ndarray
    uncertain: int
    links: np.ndarray


@dataclass(frozen=True, slots=True)
class _Stars:
    singular: np.ndarray
    incomplete: np.ndarray


@dataclass(frozen=True, slots=True)
class _Valence:
    supported: bool
    maximum_edge: int
    maximum_vertex_regions: int
    nonphysical_edges: int
    nonphysical_vertices: int


@dataclass(frozen=True, slots=True)
class _Intersections:
    checked: bool
    intersecting: int
    uncertain: int
    candidates: int
    exceeded: bool


def _region_cycles(
    labels: np.ndarray,
    face_edges: np.ndarray,
    face_edge_signs: np.ndarray,
    finite: np.ndarray,
    edge_count: int,
    region_count: int,
    /,
) -> _Combinatorial:
    """Signed/unsigned incidence of every finite label at every edge (exact)."""
    edges = face_edges.reshape(-1)
    signs = face_edge_signs.reshape(-1)
    left = np.repeat(labels[:, 0], 3)
    right = np.repeat(labels[:, 1], 3)
    keys = np.concatenate((edges * region_count + left, edges * region_count + right))
    contributions = np.concatenate((signs, -signs))
    signed = np.zeros((edge_count * region_count,), dtype=np.int64)
    counts = np.zeros((edge_count * region_count,), dtype=np.int64)
    np.add.at(signed, keys, contributions)
    np.add.at(counts, keys, 1)
    signed = signed.reshape((edge_count, region_count))[:, finite]
    counts = counts.reshape((edge_count, region_count))[:, finite]
    open_edges = np.any(counts % 2 == 1, axis=1)
    inconsistent = np.any((counts % 2 == 0) & (signed != 0), axis=1)
    return _Combinatorial(
        consistent=not bool(np.any(inconsistent)),
        inconsistent_edges=int(np.count_nonzero(inconsistent)),
        watertight=not bool(np.any(open_edges)),
        open_edges=int(np.count_nonzero(open_edges)),
    )


def _opposite_vertices(
    faces: np.ndarray, edges: np.ndarray, edge_faces: np.ndarray, /
) -> np.ndarray:
    """Vertex of each incident face not on its edge, shaped like ``edge_faces``."""
    rows = faces[np.maximum(edge_faces, 0)]
    on_edge = (rows == edges[:, None, 0:1]) | (rows == edges[:, None, 1:2])
    return np.take_along_axis(rows, np.argmax(~on_edge, axis=2)[..., None], axis=2)[
        ..., 0
    ]


def _cyclic_order(
    points: np.ndarray,
    edges: np.ndarray,
    opposite: np.ndarray,
    mode: PredicateMode,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact counterclockwise order of equal-valence faces about ``v1 - v0``.

    Rays are grouped against the first face (0: itself, 1: upper open half,
    2: antiparallel, 3: lower open half) and ranked inside their half by the
    number of rays preceding them; coincident keys mark overlapping faces.
    """
    count, valence = opposite.shape
    v0 = points[edges[:, 0]][:, None, None, :]
    v1 = points[edges[:, 1]][:, None, None, :]
    rays = points[opposite]
    first = np.broadcast_to(rays[:, :, None, :], (count, valence, valence, 3))
    second = np.broadcast_to(rays[:, None, :, :], (count, valence, valence, 3))
    sigma = orient3d(
        np.broadcast_to(v0, first.shape),
        np.broadcast_to(v1, first.shape),
        first,
        second,
        mode=mode,
    )
    signs = np.asarray(sigma.signs, dtype=np.int16)
    certain = np.all(np.asarray(sigma.certain), axis=(1, 2))
    reference = signs[:, 0, :]
    triangle = np.stack((points[edges[:, 0]], points[edges[:, 1]], rays[:, 0]), axis=1)
    normal = np.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0])
    axes = _PROJECTION_AXES[np.argmax(np.abs(normal), axis=1)]

    def project(values: np.ndarray, /) -> np.ndarray:
        return np.take_along_axis(values, axes[:, None, :], axis=-1)

    base = np.broadcast_to(
        project(points[edges[:, 0]][:, None, :]), rays.shape[:2] + (2,)
    )
    tip = np.broadcast_to(project(points[edges[:, 1]][:, None, :]), rays.shape[:2] + (2,))
    side = orient2d(base, tip, project(rays), mode=mode)
    side_signs = np.asarray(side.signs, dtype=np.int16)
    certain &= np.all(np.asarray(side.certain), axis=1)
    same_side = side_signs == side_signs[:, :1]
    group = np.where(
        reference > 0, 1, np.where(reference < 0, 3, np.where(same_side, 0, 2))
    )
    same_group = group[:, :, None] == group[:, None, :]
    precedes = same_group & (signs > 0)
    rank = np.sum(precedes, axis=1)
    key = group * valence + rank
    order = np.argsort(key, axis=1, kind="stable")
    ordered_keys = np.take_along_axis(key, order, axis=1)
    distinct = np.all(np.diff(ordered_keys, axis=1) != 0, axis=1)
    return order, certain & distinct


def _edge_wedges(
    edges: np.ndarray,
    edge_faces: np.ndarray,
    edge_signs: np.ndarray,
    finite: np.ndarray,
    points: np.ndarray,
    faces: np.ndarray,
    labels: np.ndarray,
    mode: PredicateMode,
    /,
) -> _Wedges:
    """Exact wedge-label consistency of the given edges.

    ``edge_faces``/``edge_signs`` are ``-1``/``0`` padded incidence rows. The
    returned ``links`` join face sides ``2 f + s`` (``s = 0`` left, ``1`` right)
    across every consistent wedge; they generate the label components.
    """
    valence = np.sum(edge_faces >= 0, axis=1)
    inconsistent = np.zeros((edges.shape[0],), dtype=np.bool_)
    border = np.zeros((edges.shape[0],), dtype=np.bool_)
    uncertain = 0
    links = [np.zeros((0, 2), dtype=np.int64)]
    for width in np.unique(valence):
        rows = np.flatnonzero(valence == width)
        local_faces = edge_faces[rows, :width]
        local_signs = edge_signs[rows, :width]
        if width <= 2:
            order = np.broadcast_to(np.arange(width), local_faces.shape)
            certain = np.ones((rows.size,), dtype=np.bool_)
        else:
            opposite = _opposite_vertices(faces, edges[rows], local_faces)
            order, certain = _cyclic_order(points, edges[rows], opposite, mode)
        ordered_faces = np.take_along_axis(local_faces, order, axis=1)
        ordered_signs = np.take_along_axis(local_signs, order, axis=1)
        face_left = labels[ordered_faces, 0]
        face_right = labels[ordered_faces, 1]
        after = np.where(ordered_signs > 0, face_right, face_left)
        before = np.where(ordered_signs > 0, face_left, face_right)
        following = np.roll(before, -1, axis=1)
        mismatch = after != following
        mismatches = np.sum(mismatch, axis=1)
        mismatch_boundary = np.all(
            ~mismatch | (~finite[after] & ~finite[following]), axis=1
        )
        is_border = (mismatches == 1) & mismatch_boundary & certain
        bad = ~certain | (mismatches >= 2) | ((mismatches == 1) & ~mismatch_boundary)
        inconsistent[rows] = bad
        border[rows] = is_border
        uncertain += int(np.count_nonzero(~certain))
        after_sides = 2 * ordered_faces + np.where(ordered_signs > 0, 1, 0)
        before_sides = 2 * ordered_faces + np.where(ordered_signs > 0, 0, 1)
        joined = ~mismatch & certain[:, None]
        links.append(
            np.stack(
                (after_sides[joined], np.roll(before_sides, -1, axis=1)[joined]), axis=1
            )
        )
    return _Wedges(
        consistent=not bool(np.any(inconsistent)),
        inconsistent_edges=int(np.count_nonzero(inconsistent)),
        border=border,
        uncertain=uncertain,
        links=np.concatenate(links).astype(np.int64),
    )


def _wedge_labels(
    topology: MultiRegionSurfaceTopology,
    points: np.ndarray,
    faces: np.ndarray,
    labels: np.ndarray,
    mode: PredicateMode,
    /,
) -> _Wedges:
    """Exact wedge-label consistency of every active edge."""
    return _edge_wedges(
        np.asarray(topology.edges[: topology.edge_count], dtype=np.int64),
        np.asarray(topology.edge_faces[: topology.edge_count], dtype=np.int64),
        np.asarray(topology.edge_face_signs[: topology.edge_count], dtype=np.int64),
        np.asarray(topology.region_finite),
        points,
        faces,
        labels,
        mode,
    )


def _region_components(
    labels: np.ndarray, links: np.ndarray, region_count: int, /
) -> tuple[np.ndarray, np.ndarray]:
    """Connected components of every label: counts per region, component per face side."""
    sides = 2 * labels.shape[0]
    graph = sp.coo_matrix(
        (np.ones((links.shape[0],)), (links[:, 0], links[:, 1])), shape=(sides, sides)
    )
    _, component = connected_components(graph, directed=False)
    keys = np.unique(np.stack((labels.reshape(-1), component), axis=1), axis=0)
    return np.bincount(keys[:, 0], minlength=region_count), component


def _component_volumes(
    points: np.ndarray, faces: np.ndarray, labels: np.ndarray, component: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    """Signed enclosed volume and region of every face-side component.

    A component of a finite label is a closed 2-cycle, so its signed volume is
    independent of the reference point: positive for an outer shell, negative
    for the boundary of an enclosed cavity. One spatially connected finite
    region has exactly one positive shell.
    """
    corners = points[faces] - np.mean(points, axis=0)
    volume = np.sum(corners[:, 0] * np.cross(corners[:, 1], corners[:, 2]), axis=1) / 6.0
    side_volume = np.stack((volume, -volume), axis=1).reshape(-1)
    count = int(np.max(component)) + 1
    region = np.zeros((count,), dtype=np.int64)
    region[component] = labels.reshape(-1)
    return np.bincount(component, weights=side_volume, minlength=count), region


def _vertex_stars(
    faces: np.ndarray,
    labels: np.ndarray,
    edge_faces: np.ndarray,
    edges: np.ndarray,
    border: np.ndarray,
    vertex_count: int,
    region_count: int,
    /,
) -> _Stars:
    """Singular sheet fans and incomplete region graphs at every vertex (exact).

    Corners of one sheet are joined across the manifold (two-face, same-pair)
    edges at both edge vertices; more than one component per
    ``(vertex, pair)`` is a singular vertex. The region graph of a vertex has
    its incident regions as nodes and the pairs of its incident faces as
    edges; it is complete when it has ``n (n - 1) / 2`` pairs. Border vertices
    (wires) are exempt from completeness.
    """
    ordered = np.sort(labels, axis=1)
    pair_key = ordered[:, 0] * region_count + ordered[:, 1]
    corner_vertices = faces.reshape(-1)
    corner_pairs = np.repeat(pair_key, 3)
    valence = np.sum(edge_faces >= 0, axis=1)
    padded = np.full(
        (edge_faces.shape[0], max(2, edge_faces.shape[1])), -1, dtype=np.int64
    )
    padded[:, : edge_faces.shape[1]] = edge_faces
    first, second = padded[:, 0], padded[:, 1]
    manifold = (valence == 2) & (pair_key[first] == pair_key[np.maximum(second, 0)])
    rows, cols = [], []
    for end in (0, 1):
        vertex = edges[manifold, end]
        left = first[manifold]
        right = second[manifold]
        rows.append(3 * left + np.argmax(faces[left] == vertex[:, None], axis=1))
        cols.append(3 * right + np.argmax(faces[right] == vertex[:, None], axis=1))
    corners = corner_vertices.size
    row = np.concatenate(rows)
    graph = sp.coo_matrix(
        (np.ones((row.size,)), (row, np.concatenate(cols))), shape=(corners, corners)
    )
    _, component = connected_components(graph, directed=False)
    slot_key = corner_vertices * (region_count * region_count) + corner_pairs
    fans = np.unique(np.stack((slot_key, component), axis=1), axis=0)
    slots, fan_count = np.unique(fans[:, 0], return_counts=True)
    singular = np.zeros((vertex_count,), dtype=np.bool_)
    singular[slots[fan_count > 1] // (region_count * region_count)] = True
    vertex_regions = np.unique(
        np.repeat(corner_vertices, 2) * region_count
        + np.repeat(labels, 3, axis=0).reshape(-1)
    )
    vertex_pairs = np.unique(
        corner_vertices * (region_count * region_count) + corner_pairs
    )
    regions = np.bincount(vertex_regions // region_count, minlength=vertex_count)
    pairs = np.bincount(
        vertex_pairs // (region_count * region_count), minlength=vertex_count
    )
    border_vertices = np.zeros((vertex_count,), dtype=np.bool_)
    border_vertices[edges[border].reshape(-1)] = True
    incomplete = (pairs != regions * (regions - 1) // 2) & ~border_vertices
    return _Stars(singular=singular, incomplete=incomplete)


def _valence_support(
    topology: MultiRegionSurfaceTopology,
    faces: np.ndarray,
    labels: np.ndarray,
    border: np.ndarray,
    profile: str,
    /,
) -> _Valence:
    """Edge valence and vertex region counts against the validation profile."""
    edge_faces = np.asarray(topology.edge_faces[: topology.edge_count], dtype=np.int64)
    edges = np.asarray(topology.edges[: topology.edge_count], dtype=np.int64)
    valence = np.sum(edge_faces >= 0, axis=1)
    vertex_count = topology.vertex_count
    corner_vertices = np.repeat(faces.reshape(-1), 2)
    corner_regions = np.tile(labels, (1, 3)).reshape(-1)
    region_keys = np.unique(corner_vertices * topology.region_count + corner_regions)
    regions_per_vertex = np.bincount(
        region_keys // topology.region_count, minlength=vertex_count
    )
    border_vertices = np.zeros((vertex_count,), dtype=np.bool_)
    border_vertices[edges[border].reshape(-1)] = True
    match profile:
        case "general":
            bad_edges = np.zeros_like(border)
            bad_vertices = np.zeros((vertex_count,), dtype=np.bool_)
        case "dry_foam":
            bad_edges = ~border & (valence != 2) & (valence != DRY_FOAM_EDGE_VALENCE)
            bad_vertices = ~border_vertices & (
                regions_per_vertex > DRY_FOAM_VERTEX_REGIONS
            )
        case "manifold_two_region":
            profile_admitted = (
                topology.region_count == 2
                and topology.region_pair_count == 1
                and len(topology.finite_region_indices) == 1
                and len(topology.boundary_region_indices) == 1
            )
            bad_edges = border | (valence != 2) | (not profile_admitted)
            bad_vertices = (regions_per_vertex != 2) | (not profile_admitted)
        case _:
            raise ValueError(f"Unknown validation profile {profile!r}.")
    return _Valence(
        supported=not (bool(np.any(bad_edges)) or bool(np.any(bad_vertices))),
        maximum_edge=int(np.max(valence)),
        maximum_vertex_regions=int(np.max(regions_per_vertex)),
        nonphysical_edges=int(np.count_nonzero(bad_edges)),
        nonphysical_vertices=int(np.count_nonzero(bad_vertices)),
    )


def _degenerate_faces(
    triangles: np.ndarray, mode: PredicateMode, /
) -> tuple[np.ndarray, np.ndarray]:
    """Exactly collinear faces: every coordinate projection has zero orientation.

    One certified nonzero projection proves a face nondegenerate; otherwise all
    three projections must be certified.
    """
    degenerate = np.ones((triangles.shape[0],), dtype=np.bool_)
    all_certain = np.ones((triangles.shape[0],), dtype=np.bool_)
    proven = np.zeros((triangles.shape[0],), dtype=np.bool_)
    for axes in _PROJECTION_AXES:
        projected = triangles[:, :, axes]
        result = orient2d(projected[:, 0], projected[:, 1], projected[:, 2], mode=mode)
        signs = np.asarray(result.signs)
        certain = np.asarray(result.certain)
        degenerate &= signs == 0
        all_certain &= certain
        proven |= certain & (signs != 0)
    return degenerate & ~proven, proven | all_certain


def _signed_volumes(
    points: np.ndarray,
    faces: np.ndarray,
    labels: np.ndarray,
    finite_indices: tuple[int, ...],
    region_count: int,
    /,
) -> np.ndarray:
    """Divergence-theorem volumes of finite regions about the vertex centroid."""
    centered = points - np.mean(points, axis=0)
    corners = centered[faces]
    contribution = (
        np.sum(corners[:, 0] * np.cross(corners[:, 1], corners[:, 2]), axis=1) / 6.0
    )
    volumes = np.zeros((region_count,), dtype=np.float64)
    np.add.at(volumes, labels[:, 0], contribution)
    np.add.at(volumes, labels[:, 1], -contribution)
    return volumes[list(finite_indices)]


def _euler_characteristics(
    faces: np.ndarray,
    labels: np.ndarray,
    face_edges: np.ndarray,
    finite_indices: tuple[int, ...],
    /,
) -> tuple[int, ...]:
    values = []
    for region in finite_indices:
        selected = np.any(labels == region, axis=1)
        vertices = np.unique(faces[selected]).size
        edges = np.unique(face_edges[selected]).size
        values.append(vertices - edges + int(np.count_nonzero(selected)))
    return tuple(values)


def _self_intersections(
    points: np.ndarray,
    faces: np.ndarray,
    valid: np.ndarray,
    capacity: int,
    /,
) -> _Intersections:
    """Bounded BVH broad phase plus exact narrow phase beyond shared features."""
    # The exact welded triangle-pair classifier is owned by the meshing audit;
    # importing it lazily keeps geometry importable before meshing.
    from ...meshing._audit_topology import _triangle_pairs_intersect

    selected = np.flatnonzero(valid)
    if selected.size < 2:
        return _Intersections(True, 0, 0, 0, False)
    triangles = points[faces[selected]]
    bvh = prepare_bvh(
        np.min(triangles, axis=1), np.max(triangles, axis=1), dtype=np.float64
    )
    intersecting = 0
    uncertain = 0
    candidates = 0
    exceeded = False
    for first, second in bvh_overlap_pair_blocks(bvh, bvh, include_touching=True):
        keep = first < second
        first = selected[first[keep]]
        second = selected[second[keep]]
        remaining = capacity - candidates
        if first.size > remaining:
            first, second = first[:remaining], second[:remaining]
            exceeded = True
        candidates += first.size
        if first.size:
            hit, certain = _triangle_pairs_intersect(points, faces[first], faces[second])
            intersecting += int(np.count_nonzero(hit & certain))
            uncertain += int(np.count_nonzero(~certain))
        if exceeded:
            break
    return _Intersections(True, intersecting, uncertain, candidates, exceeded)


@dataclass(frozen=True, slots=True)
class _Connectivity:
    singular_vertices: int
    incomplete_vertices: int
    completeness_required: bool
    disconnected_finite_regions: int
    component_counts: tuple[int, ...]


def _connectivity(
    topology: MultiRegionSurfaceTopology,
    points: np.ndarray,
    faces: np.ndarray,
    labels: np.ndarray,
    wedges: _Wedges,
    profile: str,
    /,
) -> _Connectivity:
    stars = _vertex_stars(
        faces,
        labels,
        np.asarray(topology.edge_faces[: topology.edge_count], dtype=np.int64),
        np.asarray(topology.edges[: topology.edge_count], dtype=np.int64),
        wedges.border,
        topology.vertex_count,
        topology.region_count,
    )
    counts, component = _region_components(labels, wedges.links, topology.region_count)
    volumes, owner = _component_volumes(points, faces, labels, component)
    pieces = np.bincount(owner[volumes > 0.0], minlength=topology.region_count)
    finite = np.asarray(topology.region_finite[: topology.region_count])
    match profile:
        case "general" | "manifold_two_region":
            required = False
        case "dry_foam":
            required = True
        case _:
            raise ValueError(f"Unknown validation profile {profile!r}.")
    return _Connectivity(
        singular_vertices=int(np.count_nonzero(stars.singular)),
        incomplete_vertices=int(np.count_nonzero(stars.incomplete)),
        completeness_required=required,
        disconnected_finite_regions=int(np.count_nonzero(finite & (pieces > 1))),
        component_counts=tuple(int(value) for value in counts),
    )


def _connectivity_status(connectivity: _Connectivity, /) -> MultiRegionSurfaceStatus:
    if connectivity.singular_vertices:
        return MultiRegionSurfaceStatus.SINGULAR_VERTEX
    if connectivity.completeness_required and connectivity.incomplete_vertices:
        return MultiRegionSurfaceStatus.REGION_GRAPH_INCOMPLETE
    if connectivity.disconnected_finite_regions:
        return MultiRegionSurfaceStatus.DISCONNECTED_FINITE_REGION
    return MultiRegionSurfaceStatus.ACCEPTED


def _status(
    finite: bool,
    combinatorial: _Combinatorial,
    wedges: _Wedges,
    valence: _Valence,
    connectivity: _Connectivity,
    degenerate: int,
    positive: bool,
    intersections: _Intersections,
    uncertain: int,
    /,
) -> MultiRegionSurfaceStatus:
    if not finite:
        return MultiRegionSurfaceStatus.NONFINITE_GEOMETRY
    if not (combinatorial.consistent and wedges.consistent):
        return MultiRegionSurfaceStatus.LABEL_ORIENTATION_INCONSISTENT
    if not combinatorial.watertight:
        return MultiRegionSurfaceStatus.REGION_NOT_WATERTIGHT
    if not valence.supported:
        return MultiRegionSurfaceStatus.NONPHYSICAL_VALENCE
    connected = _connectivity_status(connectivity)
    if connected is not MultiRegionSurfaceStatus.ACCEPTED:
        return connected
    if degenerate:
        return MultiRegionSurfaceStatus.DEGENERATE_FACE
    if not positive:
        return MultiRegionSurfaceStatus.NONPOSITIVE_VOLUME
    if intersections.intersecting:
        return MultiRegionSurfaceStatus.SELF_INTERSECTION
    if uncertain:
        return MultiRegionSurfaceStatus.UNCERTAIN_PREDICATE
    if intersections.exceeded:
        return MultiRegionSurfaceStatus.INTERSECTION_CANDIDATES_EXCEEDED
    return MultiRegionSurfaceStatus.ACCEPTED


def _geometry_id(topology: MultiRegionSurfaceTopology, points: np.ndarray, /) -> str:
    """Canonical identity of one topology epoch at the given active positions."""
    return canonical_fingerprint(
        {
            "kind": "multiregion-surface-geometry",
            "topology": topology.topology_id,
            "positions": array_tree_fingerprint(points),
        }
    )


def validate_multiregion_surface(
    topology: MultiRegionSurfaceTopology,
    state: MultiRegionSurfaceState,
    /,
    *,
    policy: MultiRegionSurfaceValidationPolicy | None = None,
) -> MultiRegionSurfaceEvidence:
    """Exact host validation of one topology epoch and its current geometry.

    Never raises for a scientifically invalid surface: every failure is
    reported in the returned evidence ``status``. This is a host preparation
    boundary (it reads device arrays once) and is not traced.
    """
    if not isinstance(topology, MultiRegionSurfaceTopology):
        raise TypeError("topology must be a MultiRegionSurfaceTopology.")
    if not isinstance(state, MultiRegionSurfaceState):
        raise TypeError("state must be a MultiRegionSurfaceState.")
    policy_ = MultiRegionSurfaceValidationPolicy() if policy is None else policy
    if not isinstance(policy_, MultiRegionSurfaceValidationPolicy):
        raise TypeError("policy must be a MultiRegionSurfaceValidationPolicy.")
    state.require_topology(topology)
    mode = resolve_host_predicate_mode(PredicateMode.EXACT)
    points = np.asarray(state.positions[: topology.vertex_count], dtype=np.float64)
    faces = topology.host_faces()
    labels = topology.host_face_labels()
    face_edges = np.asarray(topology.face_edges[: topology.face_count], dtype=np.int64)
    face_edge_signs = np.asarray(
        topology.face_edge_signs[: topology.face_count], dtype=np.int64
    )
    finite_regions = np.asarray(topology.region_finite[: topology.region_count])
    finite_indices = topology.finite_region_indices
    combinatorial = _region_cycles(
        labels,
        face_edges,
        face_edge_signs,
        finite_regions,
        topology.edge_count,
        topology.region_count,
    )
    finite = bool(np.all(np.isfinite(points)))
    empty_border = np.zeros((topology.edge_count,), dtype=np.bool_)
    wedges = (
        _wedge_labels(topology, points, faces, labels, mode)
        if finite
        else _Wedges(
            False, topology.edge_count, empty_border, 0, np.zeros((0, 2), dtype=np.int64)
        )
    )
    valence = _valence_support(topology, faces, labels, wedges.border, policy_.profile)
    connectivity = _connectivity(topology, points, faces, labels, wedges, policy_.profile)
    if finite:
        degenerate, degenerate_certain = _degenerate_faces(points[faces], mode)
        volumes = _signed_volumes(
            points, faces, labels, finite_indices, topology.region_count
        )
    else:
        degenerate = np.zeros((topology.face_count,), dtype=np.bool_)
        degenerate_certain = np.ones((topology.face_count,), dtype=np.bool_)
        volumes = np.full((len(finite_indices),), np.nan)
    positive = bool(np.all(volumes > 0.0))
    if finite and policy_.check_self_intersection:
        intersections = _self_intersections(
            points, faces, ~degenerate, policy_.intersection_candidate_capacity
        )
    else:
        intersections = _Intersections(False, 0, 0, 0, False)
    uncertain = (
        wedges.uncertain
        + int(np.count_nonzero(~degenerate_certain))
        + intersections.uncertain
    )
    status = _status(
        finite,
        combinatorial,
        wedges,
        valence,
        connectivity,
        int(np.count_nonzero(degenerate)),
        positive,
        intersections,
        uncertain,
    )
    geometry_id = _geometry_id(topology, points)
    euler = _euler_characteristics(faces, labels, face_edges, finite_indices)
    return MultiRegionSurfaceEvidence(
        status=status,
        accepted=status is MultiRegionSurfaceStatus.ACCEPTED,
        profile=policy_.profile,
        label_orientation_consistent=combinatorial.consistent and wedges.consistent,
        inconsistent_edge_count=max(
            combinatorial.inconsistent_edges, wedges.inconsistent_edges
        ),
        finite_regions_watertight=combinatorial.watertight,
        open_finite_edge_count=combinatorial.open_edges,
        border_edge_count=int(np.count_nonzero(wedges.border)),
        valence_supported=valence.supported,
        maximum_edge_valence=valence.maximum_edge,
        maximum_vertex_regions=valence.maximum_vertex_regions,
        nonphysical_edge_count=valence.nonphysical_edges,
        nonphysical_vertex_count=valence.nonphysical_vertices,
        degenerate_face_count=int(np.count_nonzero(degenerate)),
        finite=finite,
        finite_region_ids=topology.finite_region_ids,
        signed_volumes=tuple(float(value) for value in volumes),
        positive_volumes=positive,
        region_euler_characteristics=euler,
        region_component_counts=connectivity.component_counts,
        singular_vertex_count=connectivity.singular_vertices,
        incomplete_region_graph_vertex_count=connectivity.incomplete_vertices,
        self_intersection_checked=intersections.checked,
        intersecting_pair_count=intersections.intersecting,
        uncertain_pair_count=uncertain,
        candidate_pair_count=intersections.candidates,
        candidate_capacity_exceeded=intersections.exceeded,
        predicate_mode=mode.value,
        capacity=topology.plan.capacity_evidence(topology.counts),
        topology_id=topology.topology_id,
        lineage_id=topology.lineage_id,
        geometry_id=geometry_id,
        evidence_id=canonical_fingerprint(
            {
                "kind": "multiregion-surface-evidence",
                "geometry": geometry_id,
                "policy": policy_.policy_id,
                "status": status.name,
            }
        ),
    )


__all__ = ["validate_multiregion_surface"]
