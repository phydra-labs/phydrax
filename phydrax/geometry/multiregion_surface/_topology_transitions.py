#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Physical topology transitions of multiregion surfaces (T1, pinch, merge, region split).

Each transition is an explicit proposal class that owns its support set, its
local construction and its transfer groups; `apply_surface_events` dispatches
them with the remeshing events over the closed `SurfaceEventKind` set.

- **T1 pop** (Weaire and Rivier; Plateau): a vanishing film between regions
  ``D`` and ``E`` is collapsed to one vertex. The region graph of that vertex
  (regions as nodes, films as edges) then misses exactly the pair ``(D, E)``:
  ``D`` and ``E`` touch at a point. The pop pulls the vertex apart along the
  ``E -> D`` axis into ``v_D`` and ``v_E`` joined by a new edge; every film
  fan meeting both sides is split at its most equatorial spoke by one new
  triangle, so the new edge becomes a Plateau border carrying those films and
  both new vertices have complete region graphs (the triangle-to-edge T1 of
  three-dimensional dry foams).
- **Pinch** (neck criterion): a closed three-edge loop of one sheet that bounds
  no face is a topological neck. Collapsing the loop leaves one vertex with two
  disconnected incident fans; the fans are separated into two vertices moved
  apart along the neck axis.
- **Merge**: two facing films ``X|g`` and ``g|Y`` across a declared gap label
  ``g`` closer than the declared merge distance are zipped at one face pair
  into a new ``X|Y`` film bounded by three new Plateau borders; the liquid of
  both merged faces feeds the new film.
- **Region split**: a label whose face sides form several connected components
  (after a pinch, for example) is relabeled into children ``f"{id}/{k}"``
  ordered by their smallest face global id; extensive region fields split by
  component volume (finite regions) or bounding area (boundary labels).

Every motion is certified by inclusion CCD and finite-region volumes are
restored locally before the exact guards; no transition is differentiable.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from enum import IntEnum
from itertools import combinations
from typing import assert_never, final

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from ..._bvh import bvh_overlap_pair_blocks, refit_packed_bvh_bounds
from ..._fingerprint import canonical_fingerprint
from ..._geometry_predicates import PredicateMode
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._validation import (
    canonical_identifier,
    nonnegative_integer,
    positive_finite_float,
)
from ...typing import checked
from ._edits import (
    _CCDLeg,
    _Edit,
    _neighbors,
    _new_edit,
    _star,
    _WorkingMesh,
)
from ._events import SurfaceEventKind, SurfaceEventStatus
from ._geometry import PreparedMultiRegionSurface
from ._state import MultiRegionSurfaceState
from ._topology import _host_incidence
from ._validation import _edge_wedges, _region_components


def _ids(values: Sequence[int], name: str, minimum: int, /) -> tuple[int, ...]:
    if isinstance(values, str):
        raise TypeError(f"{name} must be a sequence of integers.")
    ids = tuple(sorted({nonnegative_integer(value, name) for value in values}))
    if len(ids) < minimum:
        raise ValueError(f"{name} needs at least {minimum} distinct ids.")
    return ids


def _fraction(value: float, name: str, /) -> float:
    fraction = positive_finite_float(value, name)
    if fraction >= 1.0:
        raise ValueError(f"{name} must lie in (0, 1).")
    return fraction


class _DiameterCertificate(IntEnum):
    """Outcome of the branch-and-bound film diameter certificate."""

    BELOW = 0
    NOT_BELOW = 1


_LEAF_SIZE = 4
# Relative roundoff guard of squared distances; ambiguous pairs fail closed.
_SQUARED_GUARD = 1.0e-12


def _box_tree(
    points: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Median-split bounding-box tree: order, ranges, boxes and child nodes."""
    order = np.arange(points.shape[0])
    starts, ends, lows, highs, children = [], [], [], [], []
    pending = [(0, points.shape[0], -1, 0)]
    while pending:
        start, end, parent, side = pending.pop()
        node = len(starts)
        chunk = points[order[start:end]]
        starts.append(start)
        ends.append(end)
        lows.append(np.min(chunk, axis=0))
        highs.append(np.max(chunk, axis=0))
        children.append([-1, -1])
        if parent >= 0:
            children[parent][side] = node
        if end - start > _LEAF_SIZE:
            axis = int(np.argmax(highs[-1] - lows[-1]))
            members = order[start:end]
            order[start:end] = members[np.argsort(points[members, axis], kind="stable")]
            middle = (start + end) // 2
            pending += [(middle, end, node, 1), (start, middle, node, 0)]
    return (
        order,
        np.asarray(starts),
        np.asarray(ends),
        np.asarray(lows),
        np.asarray(highs),
        np.asarray(children, dtype=np.int64),
    )


def _certify_diameter(
    points: np.ndarray, limit: float, /
) -> tuple[_DiameterCertificate, int]:
    """Certify ``max |p_i - p_j| < limit`` by branch and bound over box-node pairs.

    A node pair is accepted when the farthest corners of its two boxes are
    closer than ``limit`` and rejects the film when two of its actual points
    are at least ``limit`` apart; otherwise the larger node is split. Leaf
    pairs (at most ``4 x 4`` points) are decided exactly. Decisions within the
    roundoff guard fail closed (not below). Returns the outcome and work.
    """
    order, starts, ends, lows, highs, children = _box_tree(points)
    bound = limit * limit
    pairs = [(0, 0)]
    examined = 0
    while pairs:
        first, second = pairs.pop()
        examined += 1
        far = np.maximum(
            np.abs(highs[first] - lows[second]), np.abs(highs[second] - lows[first])
        )
        if float(far @ far) * (1.0 + _SQUARED_GUARD) < bound:
            continue
        probe = points[order[starts[first]]] - points[order[starts[second]]]
        if float(probe @ probe) * (1.0 + _SQUARED_GUARD) >= bound:
            return _DiameterCertificate.NOT_BELOW, examined
        leaf = (children[first, 0] < 0, children[second, 0] < 0)
        if all(leaf):
            left = points[order[starts[first] : ends[first]]]
            right = points[order[starts[second] : ends[second]]]
            gaps = left[:, None, :] - right[None, :, :]
            if (
                float(np.max(np.sum(gaps * gaps, axis=2))) * (1.0 + _SQUARED_GUARD)
                >= bound
            ):
                return _DiameterCertificate.NOT_BELOW, examined
            continue
        size = (
            -1.0 if leaf[0] else float(np.linalg.norm(highs[first] - lows[first])),
            -1.0 if leaf[1] else float(np.linalg.norm(highs[second] - lows[second])),
        )
        if first == second:
            left_child, right_child = children[first]
            pairs += [
                (left_child, left_child),
                (left_child, right_child),
                (right_child, right_child),
            ]
        elif size[0] >= size[1]:
            pairs += [(int(child), second) for child in children[first]]
        else:
            pairs += [(first, int(child)) for child in children[second]]
    return _DiameterCertificate.BELOW, examined


@final
class T1PopProposal(StrictModule, NonTrainableState):
    """Collapse the film ``region_ids`` spanned by ``film_vertex_ids`` and pop it.

    The pop executes only after the film's diameter is certified below
    ``maximum_film_diameter`` by a complete branch-and-bound over the prepared
    film vertices. ``pop_length_fraction`` sets the half length of the new
    Plateau border as a fraction of the shortest spoke at the collapsed vertex.
    """

    region_ids: tuple[str, str] = eqx.field(static=True)
    film_vertex_ids: tuple[int, ...] = eqx.field(static=True)
    maximum_film_diameter: float = eqx.field(static=True)
    pop_length_fraction: float = eqx.field(static=True)
    priority: float = eqx.field(static=True)

    def __init__(
        self,
        region_ids: Sequence[str],
        film_vertex_ids: Sequence[int],
        /,
        *,
        maximum_film_diameter: float,
        pop_length_fraction: float = 0.25,
        priority: float = 0.0,
    ) -> None:
        if isinstance(region_ids, str) or len(region_ids) != 2:
            raise ValueError("region_ids must name the two regions of the film.")
        first, second = (
            canonical_identifier(value, "region_ids") for value in region_ids
        )
        if first == second:
            raise ValueError("A film separates two distinct regions.")
        self.region_ids = (first, second) if first < second else (second, first)
        self.film_vertex_ids = _ids(film_vertex_ids, "film_vertex_ids", 3)
        self.maximum_film_diameter = positive_finite_float(
            maximum_film_diameter, "maximum_film_diameter"
        )
        self.pop_length_fraction = _fraction(pop_length_fraction, "pop_length_fraction")
        self.priority = float(priority)

    @property
    def kind(self) -> SurfaceEventKind:
        return SurfaceEventKind.T1_POP

    @property
    def support_vertex_ids(self) -> tuple[int, ...]:
        return self.film_vertex_ids


@final
class PinchProposal(StrictModule, NonTrainableState):
    """Pinch the neck bounded by the three-edge loop ``loop_vertex_ids``.

    ``separation_fraction`` sets the separation of the two new vertices as a
    fraction of the shortest spoke at the collapsed neck.
    """

    loop_vertex_ids: tuple[int, int, int] = eqx.field(static=True)
    separation_fraction: float = eqx.field(static=True)
    priority: float = eqx.field(static=True)

    def __init__(
        self,
        loop_vertex_ids: Sequence[int],
        /,
        *,
        separation_fraction: float = 0.25,
        priority: float = 0.0,
    ) -> None:
        ids = _ids(loop_vertex_ids, "loop_vertex_ids", 3)
        if len(ids) != 3:
            raise ValueError("A neck loop has exactly three vertices.")
        self.loop_vertex_ids = (ids[0], ids[1], ids[2])
        self.separation_fraction = _fraction(separation_fraction, "separation_fraction")
        self.priority = float(priority)

    @property
    def kind(self) -> SurfaceEventKind:
        return SurfaceEventKind.PINCH

    @property
    def support_vertex_ids(self) -> tuple[int, ...]:
        return self.loop_vertex_ids


@final
class SurfaceMergePolicy(StrictModule, NonTrainableState):
    """Declared film-merge law: which gap labels may close and when.

    Two films ``X|g`` and ``g|Y`` (``g`` in ``gap_region_ids``, ``X != Y``)
    merge at a face pair whose matched vertices are all closer than
    ``merge_distance`` and whose normals into ``g`` are antiparallel within
    ``maximum_normal_deviation`` (radians). The gap label loses the merged
    patch; the new ``X|Y`` film receives the extensive sheet content of both
    merged faces. The distance threshold stands for the continuum breakdown
    thickness of the draining gap; no drainage dynamics are implied.
    """

    gap_region_ids: tuple[str, ...] = eqx.field(static=True)
    merge_distance: float = eqx.field(static=True)
    maximum_normal_deviation: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        gap_region_ids: Sequence[str],
        /,
        *,
        merge_distance: float,
        maximum_normal_deviation: float = math.radians(30.0),
    ) -> None:
        if isinstance(gap_region_ids, str) or not gap_region_ids:
            raise ValueError("gap_region_ids must list at least one region id.")
        gaps = tuple(
            sorted({canonical_identifier(v, "gap_region_ids") for v in gap_region_ids})
        )
        distance = positive_finite_float(merge_distance, "merge_distance")
        deviation = positive_finite_float(
            maximum_normal_deviation, "maximum_normal_deviation"
        )
        if deviation >= 0.5 * math.pi:
            raise ValueError("maximum_normal_deviation must be below pi/2.")
        self.gap_region_ids = gaps
        self.merge_distance = distance
        self.maximum_normal_deviation = deviation
        self.policy_id = canonical_fingerprint(
            {
                "kind": "surface-merge-policy",
                "gap_region_ids": list(gaps),
                "merge_distance": float(distance).hex(),
                "maximum_normal_deviation": float(deviation).hex(),
            }
        )


@final
class MergeProposal(StrictModule, NonTrainableState):
    """Merge the facing faces ``face_ids`` under the declared merge policy."""

    face_ids: tuple[int, int] = eqx.field(static=True)
    policy: SurfaceMergePolicy
    priority: float = eqx.field(static=True)

    @checked
    def __init__(
        self,
        face_ids: Sequence[int],
        policy: SurfaceMergePolicy,
        /,
        *,
        priority: float = 0.0,
    ) -> None:
        ids = _ids(face_ids, "face_ids", 2)
        if len(ids) != 2:
            raise ValueError("A merge joins exactly two faces.")
        self.face_ids = (ids[0], ids[1])
        self.policy = policy
        self.priority = float(priority)

    @property
    def kind(self) -> SurfaceEventKind:
        return SurfaceEventKind.MERGE


@final
class RegionSplitProposal(StrictModule, NonTrainableState):
    """Relabel the connected components of one region into child regions."""

    region_id: str = eqx.field(static=True)

    def __init__(self, region_id: str, /) -> None:
        self.region_id = canonical_identifier(region_id, "region_id")

    @property
    def kind(self) -> SurfaceEventKind:
        return SurfaceEventKind.REGION_SPLIT

    @property
    def priority(self) -> float:
        return 0.0


# ---------------------------------------------------------------- detection


def _sheet_components(
    faces: np.ndarray, labels: np.ndarray, vertex_count: int, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Face components of every sheet, face edges and junction-or-interior edge flags.

    Faces join across two-face same-pair edges; an edge is admissible on a
    film outline when it is such an interior edge or a Plateau junction (three
    or more films), never a wire border.
    """
    incidence = _host_incidence(faces, labels, vertex_count)
    valence = np.sum(incidence.edge_faces >= 0, axis=1)
    width = incidence.edge_faces.shape[1]
    first = incidence.edge_faces[:, 0]
    second = incidence.edge_faces[:, min(1, width - 1)]
    joined = (valence == 2) & (
        incidence.face_pairs[first] == incidence.face_pairs[second]
    )
    graph = sp.coo_matrix(
        (np.ones((int(np.count_nonzero(joined)),)), (first[joined], second[joined])),
        shape=(faces.shape[0], faces.shape[0]),
    )
    _, component = connected_components(graph, directed=False)
    return component, incidence.face_edges, joined | (valence >= 3)


@final
class SurfaceEventSearch(StrictModule, NonTrainableState):
    """Complete proposals and observations of one physical transition search."""

    proposals: tuple[T1PopProposal | MergeProposal, ...]
    candidate_count: int = eqx.field(static=True)
    rejected_count: int = eqx.field(static=True)
    certificate_work: int = eqx.field(static=True)


def propose_t1_pops(
    prepared: PreparedMultiRegionSurface,
    state: MultiRegionSurfaceState,
    /,
    *,
    maximum_film_diameter: float,
    pop_length_fraction: float = 0.25,
) -> SurfaceEventSearch:
    """Propose every non-border film whose diameter is below its threshold.

    The bounding-box filter keeps every qualifying film; each remaining
    candidate is certified by complete branch-and-bound over its film vertices.
    """
    diameter_limit = positive_finite_float(maximum_film_diameter, "maximum_film_diameter")
    topology = prepared.topology
    state.require_topology(topology)
    faces = topology.host_faces()
    labels = topology.host_face_labels()
    points = np.asarray(state.positions[: topology.vertex_count], dtype=np.float64)
    component, face_edges, admissible = _sheet_components(
        faces, labels, topology.vertex_count
    )
    films = int(np.max(component)) + 1
    corners = points[faces]
    low = np.full((films, 3), np.inf)
    high = np.full((films, 3), -np.inf)
    np.minimum.at(low, component, np.min(corners, axis=1))
    np.maximum.at(high, component, np.max(corners, axis=1))
    bordered = np.zeros((films,), dtype=np.bool_)
    np.logical_or.at(bordered, component, ~np.all(admissible[face_edges], axis=1))
    extent = high - low
    small = np.flatnonzero((np.max(extent, axis=1) < diameter_limit) & ~bordered)
    ids = np.asarray(topology.vertex_global_ids[: topology.vertex_count], dtype=np.int64)
    order = np.argsort(component, kind="stable")
    starts = np.searchsorted(component[order], np.arange(films + 1))
    proposals = []
    rejected = work = 0
    for film in small.tolist():
        members = order[starts[film] : starts[film + 1]]
        vertices = np.unique(faces[members])
        certificate, examined = _certify_diameter(points[vertices], diameter_limit)
        work += examined
        match certificate:
            case _DiameterCertificate.BELOW:
                pair = labels[members[0]]
                proposals.append(
                    T1PopProposal(
                        (
                            topology.region_ids[int(pair[0])],
                            topology.region_ids[int(pair[1])],
                        ),
                        ids[vertices].tolist(),
                        maximum_film_diameter=diameter_limit,
                        pop_length_fraction=pop_length_fraction,
                        priority=float(np.linalg.norm(extent[film])),
                    )
                )
            case _DiameterCertificate.NOT_BELOW:
                rejected += 1
            case _:
                assert_never(certificate)
    return SurfaceEventSearch(
        proposals=tuple(sorted(proposals, key=lambda p: (p.priority, p.film_vertex_ids))),
        candidate_count=small.size,
        rejected_count=rejected,
        certificate_work=work,
    )


def propose_pinches(
    prepared: PreparedMultiRegionSurface,
    state: MultiRegionSurfaceState,
    /,
    *,
    maximum_neck_perimeter: float,
    separation_fraction: float = 0.25,
) -> tuple[PinchProposal, ...]:
    """Three-edge sheet loops bounding no face whose perimeter fell below the threshold."""
    limit = positive_finite_float(maximum_neck_perimeter, "maximum_neck_perimeter")
    topology = prepared.topology
    state.require_topology(topology)
    faces = topology.host_faces()
    points = np.asarray(state.positions[: topology.vertex_count], dtype=np.float64)
    ids = np.asarray(topology.vertex_global_ids[: topology.vertex_count], dtype=np.int64)
    neighbors: list[set[int]] = [set() for _ in range(topology.vertex_count)]
    for row in faces.tolist():
        for vertex in row:
            neighbors[vertex].update(row)
    triangles = {tuple(sorted(row)) for row in faces.tolist()}
    proposals = []
    for first in range(topology.vertex_count):
        for second in sorted(v for v in neighbors[first] if v > first):
            for third in sorted(
                v for v in neighbors[first] & neighbors[second] if v > second
            ):
                if (first, second, third) in triangles:
                    continue
                loop = points[[first, second, third]]
                perimeter = float(
                    np.sum(np.linalg.norm(loop - np.roll(loop, 1, axis=0), axis=1))
                )
                if perimeter < limit:
                    proposals.append(
                        PinchProposal(
                            ids[[first, second, third]].tolist(),
                            separation_fraction=separation_fraction,
                            priority=perimeter,
                        )
                    )
    return tuple(sorted(proposals, key=lambda p: (p.priority, p.loop_vertex_ids)))


def _triangle_distances(
    first: np.ndarray, second: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    """Minimum distances between disjoint triangle pairs and their roundoff bound.

    The distance between two non-intersecting triangles is attained between a
    vertex and the other triangle or between two edges, so it is the minimum
    of six point-triangle and nine edge-edge closest-point distances evaluated
    with the contact owner's robust kernels. The bound ``64 eps scale`` (scale
    = the largest coordinate magnitude of the pair, at least one) certifies a
    threshold decision whenever the computed distance lies outside it.
    """
    # The contact package imports geometry; its kernels resolve lazily.
    from ...discretization.contact import edge_edge_distance, point_triangle_distance

    left = jnp.asarray(first, dtype=jnp.float64)
    right = jnp.asarray(second, dtype=jnp.float64)
    squared = []
    for corner in range(3):
        squared.append(
            point_triangle_distance(
                left[:, corner], right[:, 0], right[:, 1], right[:, 2]
            ).squared_distance
        )
        squared.append(
            point_triangle_distance(
                right[:, corner], left[:, 0], left[:, 1], left[:, 2]
            ).squared_distance
        )
        for other in range(3):
            squared.append(
                edge_edge_distance(
                    left[:, corner],
                    left[:, (corner + 1) % 3],
                    right[:, other],
                    right[:, (other + 1) % 3],
                ).squared_distance
            )
    distance = np.sqrt(
        np.maximum(np.min(np.stack([np.asarray(s) for s in squared]), axis=0), 0.0)
    )
    scale = np.maximum(
        1.0,
        np.maximum(
            np.max(np.abs(first), axis=(1, 2)), np.max(np.abs(second), axis=(1, 2))
        ),
    )
    return distance, 64.0 * float(np.finfo(np.float64).eps) * scale


def _merge_candidates(
    prepared: PreparedMultiRegionSurface,
    corners: np.ndarray,
    margin: float,
    /,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Stream every face pair with overlapping inflated BVH bounds."""
    bvh = refit_packed_bvh_bounds(
        prepared.face_bvh,
        np.min(corners, axis=1) - margin,
        np.max(corners, axis=1) + margin,
    )
    firsts, seconds = [np.zeros((0,), dtype=np.int64)], [np.zeros((0,), dtype=np.int64)]
    count = 0
    for first, second in bvh_overlap_pair_blocks(bvh, bvh, include_touching=True):
        keep = first < second
        count += int(np.count_nonzero(keep))
        firsts.append(first[keep].astype(np.int64))
        seconds.append(second[keep].astype(np.int64))
    return np.concatenate(firsts), np.concatenate(seconds), count


def propose_merges(
    prepared: PreparedMultiRegionSurface,
    state: MultiRegionSurfaceState,
    policy: SurfaceMergePolicy,
    /,
) -> SurfaceEventSearch:
    """Enumerate every facing gap-label pair that meets the merge law.

    The prepared face BVH is streamed over all overlapping inflated bounds.
    Every candidate then receives the shared gap-label, orientation and
    roundoff-aware triangle-distance checks.
    """
    if not isinstance(policy, SurfaceMergePolicy):
        raise TypeError("policy must be a SurfaceMergePolicy.")
    topology = prepared.topology
    state.require_topology(topology)
    faces = topology.host_faces()
    labels = topology.host_face_labels()
    points = np.asarray(state.positions[: topology.vertex_count], dtype=np.float64)
    corners = points[faces]
    first, second, count = _merge_candidates(
        prepared, corners, 0.5 * policy.merge_distance
    )
    face_ids = np.asarray(topology.face_global_ids[: topology.face_count], dtype=np.int64)
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    centroids = np.mean(corners, axis=1)
    offset = centroids[second] - centroids[first]
    shared = np.any(faces[first][:, :, None] == faces[second][:, None, :], axis=(1, 2))
    distance, roundoff = _triangle_distances(corners[first], corners[second])
    limit = math.cos(policy.maximum_normal_deviation)
    proposals = []
    for gap_id in policy.gap_region_ids:
        if gap_id not in topology.region_ids:
            continue
        gap = topology.region_index(gap_id)
        into_first = np.where(
            (labels[first, 0] == gap)[:, None], -normals[first], normals[first]
        )
        into_second = np.where(
            (labels[second, 0] == gap)[:, None], -normals[second], normals[second]
        )
        outer_first = np.where(
            labels[first, 0] == gap, labels[first, 1], labels[first, 0]
        )
        outer_second = np.where(
            labels[second, 0] == gap, labels[second, 1], labels[second, 0]
        )
        eligible = (
            np.any(labels[first] == gap, axis=1)
            & np.any(labels[second] == gap, axis=1)
            & (outer_first != outer_second)
            & ~shared
            & (distance < policy.merge_distance + roundoff)
            & (np.sum(into_first * into_second, axis=1) < -limit)
            & (np.sum(offset * into_first, axis=1) > 0.0)
        )
        for index in np.flatnonzero(eligible).tolist():
            proposals.append(
                MergeProposal(
                    (int(face_ids[first[index]]), int(face_ids[second[index]])),
                    policy,
                    priority=float(distance[index]),
                )
            )
    return SurfaceEventSearch(
        proposals=tuple(sorted(proposals, key=lambda p: (p.priority, p.face_ids))),
        candidate_count=count,
        rejected_count=count - len(proposals),
        certificate_work=0,
    )


# ------------------------------------------------------------------ builders


def _collapse_star(
    mesh: _WorkingMesh, vertices: Sequence[int], point: int, /
) -> tuple[list[int], list[list[int]], list[list[int]], list[tuple[int, ...]]]:
    """Faces of the star of ``vertices`` reattached to one provisional ``point``.

    Faces with two or more collapsing vertices degenerate and are dropped.
    Returns removed faces, reattached rows, their labels and single parents.
    """
    group = set(vertices)
    removed = _star(mesh, group).tolist()
    rows, labels, parents = [], [], []
    for face in removed:
        row = mesh.faces[face].tolist()
        if sum(vertex in group for vertex in row) >= 2:
            continue
        rows.append([point if vertex in group else vertex for vertex in row])
        labels.append([int(value) for value in mesh.labels[face]])
        parents.append((face,))
    return removed, rows, labels, parents


def _fans(rows: Sequence[Sequence[int]], point: int, /) -> tuple[np.ndarray, int]:
    """Components of the faces at ``point`` joined across shared spokes."""
    count = len(rows)
    spokes: dict[int, list[int]] = {}
    for index, row in enumerate(rows):
        for vertex in row:
            if vertex != point:
                spokes.setdefault(vertex, []).append(index)
    first, second = [], []
    for members in spokes.values():
        for a, b in combinations(members, 2):
            first.append(a)
            second.append(b)
    graph = sp.coo_matrix(
        (
            np.ones((len(first),)),
            (np.asarray(first, dtype=np.int64), np.asarray(second, dtype=np.int64)),
        ),
        shape=(count, count),
    )
    fans, component = connected_components(graph, directed=False)
    return component, fans


def _stage_leg(
    mesh: _WorkingMesh,
    rows: Sequence[Sequence[int]],
    points: dict[int, np.ndarray],
    start_override: dict[int, np.ndarray],
    exclusions: Sequence[tuple[int, int]],
    /,
) -> _CCDLeg:
    """Candidate-topology leg over ``rows``: overridden keys start elsewhere."""
    faces = np.asarray(rows, dtype=np.int64).reshape((-1, 3))
    keys = np.unique(faces)
    end = np.stack(
        [
            points[int(key)] if int(key) in points else mesh.positions[int(key)]
            for key in keys
        ]
    )
    start = np.stack(
        [
            start_override.get(int(key), end[index])
            for index, key in enumerate(keys.tolist())
        ]
    )
    return _CCDLeg(
        keys=keys,
        faces=np.searchsorted(keys, faces),
        start=start,
        end=end,
        exclusions=tuple((int(a), int(b)) for a, b in exclusions),
    )


def _collapse_leg(
    mesh: _WorkingMesh, vertices: Sequence[int], point: np.ndarray, /
) -> _CCDLeg:
    """Source-topology leg: every collapsing vertex moves to ``point``."""
    star = _star(mesh, vertices)
    keys = np.unique(mesh.faces[star])
    start = mesh.positions[keys]
    end = start.copy()
    end[np.isin(keys, np.asarray(vertices))] = point
    return _CCDLeg(
        keys=keys,
        faces=np.searchsorted(keys, mesh.faces[star]),
        start=start,
        end=end,
        exclusions=tuple(combinations(sorted(int(v) for v in vertices), 2)),
    )


def _fan_path(
    rows: Sequence[Sequence[int]], members: Sequence[int], point: int, /
) -> tuple[list[int], list[int]] | None:
    """Ordered faces and spokes of one fan at ``point`` if it is a simple path."""
    spokes: dict[int, list[int]] = {}
    for index in members:
        for vertex in rows[index]:
            if vertex != point:
                spokes.setdefault(vertex, []).append(index)
    ends = sorted(vertex for vertex, owners in spokes.items() if len(owners) == 1)
    if len(ends) != 2 or any(len(owners) > 2 for owners in spokes.values()):
        return None
    order, path = [], [ends[0]]
    current = ends[0]
    previous = -1
    while len(order) < len(members):
        face = next(f for f in spokes[current] if f != previous)
        order.append(face)
        current = next(v for v in rows[face] if v not in (point, current))
        path.append(current)
        previous = face
    return order, path


def _directed(row: Sequence[int], first: int, second: int, /) -> bool:
    position = list(row).index(first)
    return row[(position + 1) % 3] == second


def _t1_edit(
    mesh: _WorkingMesh, proposal: T1PopProposal, film: Sequence[int], /
) -> _Edit | SurfaceEventStatus:
    topology = mesh.topology
    if not all(region in topology.region_ids for region in proposal.region_ids):
        return SurfaceEventStatus.SUPPORT_INVALID
    upper, lower = (topology.region_index(region) for region in proposal.region_ids)
    film_set = set(film)
    film_faces = [
        face
        for face in _star(mesh, film).tolist()
        if set(mesh.faces[face].tolist()) <= film_set
        and {int(v) for v in mesh.labels[face]} == {upper, lower}
    ]
    if not film_faces or set(mesh.faces[film_faces].reshape(-1).tolist()) != film_set:
        return SurfaceEventStatus.SUPPORT_INVALID
    if np.any(mesh.fixed[list(film)]):
        return SurfaceEventStatus.FEATURE_NOT_PRESERVED
    certificate, _ = _certify_diameter(
        mesh.positions[list(film)], proposal.maximum_film_diameter
    )
    match certificate:
        case _DiameterCertificate.BELOW:
            pass
        case _DiameterCertificate.NOT_BELOW:
            return SurfaceEventStatus.NOT_TRIGGERED
        case _:
            assert_never(certificate)
    corner = np.mean(mesh.positions[list(film)], axis=0)
    point = mesh.vertex_count
    removed, rows, labels, parents = _collapse_star(mesh, film, point)
    label_sets = [set(int(v) for v in label) for label in labels]
    if any({upper, lower} <= labels_ for labels_ in label_sets):
        return SurfaceEventStatus.NOT_TRIGGERED
    regions = set().union(*label_sets)
    pairs = {tuple(sorted(labels_)) for labels_ in label_sets}
    missing = {(a, b) for a, b in combinations(sorted(regions), 2) if (a, b) not in pairs}
    if missing != {tuple(sorted((upper, lower)))}:
        return SurfaceEventStatus.REGION_GRAPH_INCOMPLETE
    side_d = [i for i, labels_ in enumerate(label_sets) if upper in labels_]
    side_e = [i for i, labels_ in enumerate(label_sets) if lower in labels_]
    mixed = [i for i, labels_ in enumerate(label_sets) if not labels_ & {upper, lower}]
    spokes_d = {v for i in side_d for v in rows[i] if v != point}
    spokes_e = {v for i in side_e for v in rows[i] if v != point}
    positions_d = np.mean(mesh.positions[sorted(spokes_d)], axis=0)
    positions_e = np.mean(mesh.positions[sorted(spokes_e)], axis=0)
    axis = positions_d - positions_e
    if not np.linalg.norm(axis) > 0.0:
        return SurfaceEventStatus.FAN_STRUCTURE_INVALID
    axis /= np.linalg.norm(axis)
    spokes = sorted(
        spokes_d | spokes_e | {v for i in mixed for v in rows[i] if v != point}
    )
    reach = float(np.min(np.linalg.norm(mesh.positions[spokes] - corner, axis=1)))
    half = proposal.pop_length_fraction * reach
    vertex_d, vertex_e = point, point + 1
    new_rows = [list(row) for row in rows]
    for i in side_d:
        new_rows[i] = [vertex_d if v == point else v for v in rows[i]]
    for i in side_e:
        new_rows[i] = [vertex_e if v == point else v for v in rows[i]]
    extra_rows, extra_labels = [], []
    pair_groups: dict[tuple[int, int], list[int]] = {}
    for i in mixed:
        pair_groups.setdefault((min(label_sets[i]), max(label_sets[i])), []).append(i)
    for members in pair_groups.values():
        path = _fan_path(rows, members, point)
        if path is None:
            return SurfaceEventStatus.FAN_STRUCTURE_INVALID
        order, chain = path
        if chain[0] in spokes_e and chain[-1] in spokes_d:
            order, chain = order[::-1], chain[::-1]
        if not (chain[0] in spokes_d and chain[-1] in spokes_e):
            return SurfaceEventStatus.FAN_STRUCTURE_INVALID
        tilt = [
            abs(float(np.dot(mesh.positions[v] - corner, axis)))
            / max(float(np.linalg.norm(mesh.positions[v] - corner)), 1e-300)
            for v in chain
        ]
        split = min(
            (
                (value, int(mesh.vertex_ids[vertex]), index)
                for index, (value, vertex) in enumerate(zip(tilt, chain, strict=True))
            )
        )[2]
        for position, face in enumerate(order):
            owner = vertex_d if position < split else vertex_e
            new_rows[face] = [owner if v == point else v for v in rows[face]]
        spoke = chain[split]
        reference = order[split - 1] if split >= 1 else order[0]
        owner = vertex_d if split >= 1 else vertex_e
        other = vertex_e if split >= 1 else vertex_d
        if _directed(new_rows[reference], owner, spoke):
            extra_rows.append([spoke, owner, other])
        else:
            extra_rows.append([owner, spoke, other])
        extra_labels.append(labels[reference])
    positions = {vertex_d: corner + half * axis, vertex_e: corner - half * axis}
    stage = _stage_leg(
        mesh,
        new_rows,
        positions,
        {vertex_d: corner, vertex_e: corner},
        ((vertex_d, vertex_e),),
    )
    return _new_edit(
        SurfaceEventKind.T1_POP,
        removed_faces=removed,
        new_faces=new_rows + extra_rows,
        new_labels=labels + extra_labels,
        new_face_parents=parents + [()] * len(extra_rows),
        removed_vertices=film,
        new_positions=[positions[vertex_d], positions[vertex_e]],
        new_vertex_parents=[tuple(film), tuple(film)],
        legs=(_collapse_leg(mesh, film, corner), stage),
    )


def _pinch_edit(
    mesh: _WorkingMesh, proposal: PinchProposal, loop: Sequence[int], /
) -> _Edit | SurfaceEventStatus:
    first, second, third = loop
    faces = {tuple(sorted(mesh.faces[f].tolist())) for f in _star(mesh, loop).tolist()}
    edges_present = all(
        b in _neighbors(mesh, a)
        for a, b in ((first, second), (second, third), (first, third))
    )
    if not edges_present or tuple(sorted(loop)) in faces:
        return SurfaceEventStatus.SUPPORT_INVALID
    if np.any(mesh.fixed[list(loop)]):
        return SurfaceEventStatus.FEATURE_NOT_PRESERVED
    star = _star(mesh, loop)
    pairs = {tuple(sorted(int(v) for v in mesh.labels[f])) for f in star.tolist()}
    if len(pairs) != 1:
        return SurfaceEventStatus.FEATURE_NOT_PRESERVED
    corner = np.mean(mesh.positions[list(loop)], axis=0)
    point = mesh.vertex_count
    removed, rows, labels, parents = _collapse_star(mesh, loop, point)
    component, fans = _fans(rows, point)
    if fans != 2:
        return SurfaceEventStatus.FAN_STRUCTURE_INVALID
    centers = [
        np.mean(
            mesh.positions[
                sorted(
                    {
                        v
                        for i in np.flatnonzero(component == fan)
                        for v in rows[i]
                        if v != point
                    }
                )
            ],
            axis=0,
        )
        for fan in (0, 1)
    ]
    axis = centers[0] - centers[1]
    if not np.linalg.norm(axis) > 0.0:
        return SurfaceEventStatus.FAN_STRUCTURE_INVALID
    axis /= np.linalg.norm(axis)
    spokes = sorted({v for row in rows for v in row if v != point})
    reach = float(np.min(np.linalg.norm(mesh.positions[spokes] - corner, axis=1)))
    half = 0.5 * proposal.separation_fraction * reach
    new_rows = [
        [point + int(component[i]) if v == point else v for v in row]
        for i, row in enumerate(rows)
    ]
    positions = {point: corner + half * axis, point + 1: corner - half * axis}
    stage = _stage_leg(
        mesh,
        new_rows,
        positions,
        {point: corner, point + 1: corner},
        ((point, point + 1),),
    )
    return _new_edit(
        SurfaceEventKind.PINCH,
        removed_faces=removed,
        new_faces=new_rows,
        new_labels=labels,
        new_face_parents=parents,
        removed_vertices=loop,
        new_positions=[positions[point], positions[point + 1]],
        new_vertex_parents=[tuple(loop), tuple(loop)],
        legs=(_collapse_leg(mesh, loop, corner), stage),
    )


def _merge_edit(
    mesh: _WorkingMesh, proposal: MergeProposal, faces: Sequence[int], /
) -> _Edit | SurfaceEventStatus:
    one, other = faces
    topology = mesh.topology
    policy = proposal.policy
    gaps = [
        topology.region_index(g)
        for g in policy.gap_region_ids
        if g in topology.region_ids
    ]
    labels_one, labels_other = mesh.labels[one], mesh.labels[other]
    gap = next((g for g in gaps if g in labels_one and g in labels_other), None)
    if gap is None:
        return SurfaceEventStatus.SUPPORT_INVALID
    outer_one = int(labels_one[1] if labels_one[0] == gap else labels_one[0])
    outer_other = int(labels_other[1] if labels_other[0] == gap else labels_other[0])
    if outer_one == outer_other:
        return SurfaceEventStatus.SUPPORT_INVALID
    first, second = mesh.faces[one].tolist(), mesh.faces[other].tolist()
    if set(first) & set(second):
        return SurfaceEventStatus.LINK_CONDITION_VIOLATED
    if np.any(mesh.fixed[first + second]):
        return SurfaceEventStatus.FEATURE_NOT_PRESERVED
    ring_one = set().union(*(_neighbors(mesh, v) for v in first)) | set(first)
    ring_other = set().union(*(_neighbors(mesh, v) for v in second)) | set(second)
    if ring_one & ring_other:
        return SurfaceEventStatus.LINK_CONDITION_VIOLATED
    reversed_second = second[::-1]
    matches = [reversed_second[k:] + reversed_second[:k] for k in range(3)]
    distances = [
        float(
            np.sum(np.linalg.norm(mesh.positions[first] - mesh.positions[match], axis=1))
        )
        for match in matches
    ]
    match = matches[int(np.argmin(distances))]
    distance, roundoff = _triangle_distances(
        mesh.positions[first][None], mesh.positions[second][None]
    )
    normal_one = np.cross(*(mesh.positions[first[1:]] - mesh.positions[first[0]]))
    normal_other = np.cross(*(mesh.positions[second[1:]] - mesh.positions[second[0]]))
    into_one = -normal_one if labels_one[0] == gap else normal_one
    into_other = -normal_other if labels_other[0] == gap else normal_other
    cosine = float(np.dot(into_one, into_other)) / float(
        np.linalg.norm(into_one) * np.linalg.norm(into_other)
    )
    if float(distance[0]) >= policy.merge_distance + float(
        roundoff[0]
    ) or cosine > -math.cos(policy.maximum_normal_deviation):
        return SurfaceEventStatus.NOT_TRIGGERED
    base = mesh.vertex_count
    mapping = {u: base + k for k, u in enumerate(first)}
    mapping.update({w: base + k for k, w in enumerate(match)})
    midpoints = [
        0.5 * (mesh.positions[u] + mesh.positions[w])
        for u, w in zip(first, match, strict=True)
    ]
    removed = _star(mesh, first + second).tolist()
    rows, labels, parents = [], [], []
    for face in removed:
        if face in (one, other):
            continue
        rows.append([mapping.get(v, v) for v in mesh.faces[face].tolist()])
        labels.append(mesh.labels[face])
        parents.append((face,))
    merged = [mapping[u] for u in first]
    merged_labels = (
        (outer_one, outer_other) if labels_one[1] == gap else (outer_other, outer_one)
    )
    star = _star(mesh, first + second)
    keys = np.unique(mesh.faces[star])
    start = mesh.positions[keys]
    end = start.copy()
    for vertex, target in mapping.items():
        end[np.searchsorted(keys, vertex)] = midpoints[target - base]
    leg = _CCDLeg(
        keys=keys,
        faces=np.searchsorted(keys, mesh.faces[star]),
        start=start,
        end=end,
        exclusions=tuple((u, w) for u in first for w in second),
    )
    others = [face for face in removed if face not in (one, other)]
    return _new_edit(
        SurfaceEventKind.MERGE,
        removed_faces=removed,
        new_faces=rows + [merged],
        new_labels=labels + [merged_labels],
        new_face_parents=parents + [(one, other)],
        removed_vertices=first + second,
        new_positions=midpoints,
        new_vertex_parents=[(u, w) for u, w in zip(first, match, strict=True)],
        legs=(leg,),
        groups=((others, tuple(range(len(rows)))), ((one, other), (len(rows),))),
    )


# ------------------------------------------------------------- region split


def _label_components(
    points: np.ndarray,
    faces: np.ndarray,
    labels: np.ndarray,
    finite: np.ndarray,
    region_count: int,
    mode: PredicateMode,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Component counts per region and the component of every face side."""
    incidence = _host_incidence(faces, labels, points.shape[0])
    wedges = _edge_wedges(
        incidence.edges,
        incidence.edge_faces,
        incidence.edge_face_signs,
        finite,
        points,
        faces,
        labels,
        mode,
    )
    return _region_components(labels, wedges.links, region_count)


__all__ = [
    "MergeProposal",
    "PinchProposal",
    "RegionSplitProposal",
    "SurfaceMergePolicy",
    "SurfaceEventSearch",
    "T1PopProposal",
    "propose_merges",
    "propose_pinches",
    "propose_t1_pops",
]
