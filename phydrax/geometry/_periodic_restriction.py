#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Transfer a complete periodic embedding across an exact convex partition.

Let a source quotient mesh be certified embedded by a complete current
certificate bound to its mesh, geometry and validity, under an exact isometry
group, with closed convex full-dimensional lifted cells covering the closed
quotient manifold. A target with the same exact group whose every lifted cell
is a closed convex full-dimensional polytope inside one lifted parent, lying in
one closed sign cell of the exact restriction planes, with sign vectors unique
within each parent and children's exact volumes summing to the parent's
volume, partitions every parent exactly: distinct sign vectors separate
interiors, and a closed subset of full measure in a convex body is the body, so
each child is exactly its parent intersected with its closed sign cell.
Children of distinct parent images inherit disjoint interiors from the source
certificate, so all target images have disjoint interiors and cover the
quotient. Every cell boundary, source and target, is proved to be a
face-subdivision complex of its convex polytope boundary: each face is a simple
convex loop on an outward supporting plane of the vertex hull, every directed
edge has exactly one reversed partner (a 2-cycle on the hull sphere of constant
winding degree), each vertex link is one corner cycle, and on each supporting
plane the positive face areas equal the exact facet area, so the winding
degree is one: the faces tile each facet with disjoint interiors (no
overlapping coplanar subface, T-junction, duplicated sheet or pinched link).
Facet traces, reciprocal quotient facets and the group are proved
independently on the target: every quotient face has exactly two occurrences
whose lifted vertex keys carry exactly equal images. Chamber adjacency in the
link of any point then crosses common subdivision faces only; the minimal
subdivision entity containing a point is propagated across every crossed
face, so each intersection of two target cell images is a common subcomplex.
No allowed parent contact is reused for a child contact.

For inverse agglomeration the roles of the partition children and parents are
reversed, but the imported certificate is still the complete *current source*
certificate, never an ancestor certificate rebound by cell count or geometry
similarity. Each source cell lies in its declared target parent and a unique
closed sign cell; the same exact-volume argument proves each target is the
union of its declared source cells. Source-image interior disjointness then
implies target-image interior disjointness. The independent strong boundary
subcomplex, exact-group and target quotient-trace premises remain unchanged.
The inverse also requires the target's exact power source to select unchanged
current-source vertex rows and IDs. Within each group, all interior oriented
source faces cancel exactly; every remaining signed loop is one target face
with its complete subdivision. Region agreement remains the association
owner's independently checked obligation before this embedding admission.

Mesh certificate checks
-----------------------
[
  {
    "owner": "phydrax.geometry._periodic_restriction.certify_periodic_restriction_partition",
    "checks": [
      "periodic_restriction_source_embedding",
      "periodic_restriction_parent_partition",
      "periodic_restriction_boundary_subdivision"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "A complete current certified quotient embedding fully bound to source mesh, topology, geometry layout, geometry and positive source validity, with the same exact isometry group identification as the target; every source and target lifted cell is a closed convex full-dimensional polytope."
      },
      {
        "tag": "X",
        "statement": "In the recorded partition direction (refinement or inverse agglomeration), each child lies in its declared parent and in one closed sign cell of the exact restriction planes; sign vectors are unique within each parent. The source supplies parents for refinement and children for inverse agglomeration."
      },
      {
        "tag": "X",
        "statement": "Exact rational child volumes sum to each parent's exact volume, every partition parent has at least one child, and all partition children are included exactly once."
      },
      {
        "tag": "X",
        "statement": "Every source and target cell boundary is a face-subdivision complex of its convex hull boundary: each face a simple convex loop on an outward supporting plane, one consistent orientation, each directed edge with exactly one reversed partner, every vertex link one corner cycle, and per supporting plane positive face areas summing to the exact facet area (winding degree one)."
      },
      {
        "tag": "X",
        "statement": "For inverse agglomeration the target exact source is a pure vertex selection of the current source with matching global vertex IDs and exact source-query coordinates. In each declared group interior signed source-face loops cancel exactly and the remaining loops equal the complete target boundary subcomplex, with multiplicity one; association admission separately proves same-region source parents."
      },
      {
        "tag": "X",
        "statement": "Independently of the partition, the target's exact group, facet traces and reciprocal quotient facets are certified: every quotient face has exactly two occurrences whose lifted vertex keys agree exactly under the declared action, the target validity is positive, and the quotient has no boundary facet."
      }
    ],
    "conclusion": "The children partition every parent exactly in the recorded direction; refinement or inverse unions preserve source-image interior disjointness and quotient coverage. With independently proved strong boundary subdivisions and reciprocal lifted faces every target-image intersection is a common subcomplex, so the target quotient map is embedded without image-pair enumeration.",
    "citation": {
      "kind": "mathematical",
      "reference": "De Loera, Rambau and Santos, Triangulations: Structures for Algorithms and Applications",
      "locator": "Chapter 2: polyhedral subdivisions and their face-to-face intersection property, applied through chamber adjacency in the local link of a point",
      "url": "https://doi.org/10.1007/978-3-642-12971-1"
    },
    "failure_statuses": [
      "unresolved"
    ]
  }
]
End mesh certificate checks
"""

from __future__ import annotations

from fractions import Fraction
from typing import TYPE_CHECKING

import numpy as np

import phydrax._fingerprint as fingerprint_owner
import phydrax.geometry._planar_primitives as planar_primitives_owner
from phydrax._strict import StrictModule
from phydrax.discretization._cell_complex import PolyhedralConnectivity
from phydrax.discretization._cell_geometry_validity import (
    cell_geometry_id,
    CellValidityCertificate,
)
from phydrax.discretization._cell_mesh import CellMesh
from phydrax.discretization._cell_geometry_predecessor import CellGeometryPredecessor
from phydrax.discretization._cell_geometry_transition_types import CellGeometryTransition
from phydrax.discretization._exact_power_geometry import (
    ExactPowerCellGeometryLinearActionSource,
    ExactPowerCellGeometryRestrictionSource,
    ExactPowerCellGeometrySource,
)
from phydrax.discretization._periodic_exact import identification_id
from phydrax.geometry._embedding_state import (
    EmbeddingState,
    PeriodicRestrictionPartitionEvidence,
)

from ..discretization._cell_geometry import CellGeometrySpec
from ..discretization._coordinate_budget import CoordinateEnclosureBudget
from ..discretization._coordinate_enclosure import (
    CoordinateSourceBank,
    prepared_coordinate_source_bank,
)


if TYPE_CHECKING:
    from ._global_embedding import GlobalEmbeddingCertificate


# Three rational coordinates; planes are three normal entries and an offset.
type _Point = tuple[Fraction, ...]
type _Plane = tuple[Fraction, ...]

_POWER_SOURCES = (
    ExactPowerCellGeometrySource,
    ExactPowerCellGeometryRestrictionSource,
    ExactPowerCellGeometryLinearActionSource,
)


class PeriodicRestrictionPremise(StrictModule):
    """Authenticated source embedding and declared parents of a restricted target.

    ``parent_cell_ids`` pairs each partition child with its parent. For an
    inverse, the source supplies children and the current target supplies parents;
    the theorem still authenticates the complete current source certificate.
    """

    source_mesh: CellMesh
    source_geometry: CellGeometrySpec
    source_validity: CellValidityCertificate
    source_embedding: GlobalEmbeddingCertificate
    parent_cell_ids: tuple[tuple[int, int], ...]
    inverse: bool = False
    mapped_predecessor: CellGeometryPredecessor | None = None
    mapped_transition: CellGeometryTransition | None = None
    source_regions: tuple[int, ...] | None = None
    target_regions: tuple[int, ...] | None = None
    region_namespace_id: str | None = None
    source_domain: object | None = None


class _ConvexCell(StrictModule):
    cell_id: int
    points: tuple[_Point, ...]
    planes: tuple[_Plane, ...]
    volume: Fraction


def _rational_point(point: tuple[object, ...], /) -> _Point | None:
    values = tuple(value for value in point if isinstance(value, Fraction))
    return values if len(values) == len(point) == 3 else None


def _cell_faces(
    connectivity: PolyhedralConnectivity, /
) -> tuple[tuple[int, tuple[tuple[int, ...], ...]], ...]:
    """Return each cell's face loops in the cell's declared incidence orientation."""
    offsets = np.asarray(connectivity.cell_face_offsets, dtype=np.int64)
    values = np.asarray(connectivity.cell_face_values, dtype=np.int64)
    signs = np.asarray(connectivity.cell_face_sign_values)
    face_offsets = np.asarray(connectivity.face_vertex_offsets, dtype=np.int64)
    face_values = np.asarray(connectivity.face_vertex_values, dtype=np.int64)
    identifiers = np.asarray(connectivity.cell_global_ids, dtype=np.int64)
    return tuple(
        (
            int(identifiers[cell]),
            tuple(
                tuple(
                    int(vertex)
                    for vertex in face_values[
                        face_offsets[values[position]] : face_offsets[values[position] + 1]
                    ][:: 1 if signs[position] > 0 else -1]
                )
                for position in range(offsets[cell], offsets[cell + 1])
            ),
        )
        for cell in range(offsets.size - 1)
    )


def _evaluate(plane: _Plane, point: _Point, /, *, budget: CoordinateEnclosureBudget) -> Fraction:
    planar_primitives_owner.reserve_fraction_work((plane, point), 6, 1, 8, budget=budget)
    return plane[0] * point[0] + plane[1] * point[1] + plane[2] * point[2] + plane[3]


def _face_plane(
    points: tuple[_Point, ...], /, *, budget: CoordinateEnclosureBudget
) -> _Plane | None:
    anchor = points[0]
    for second in points[1:]:
        u = tuple(second[axis] - anchor[axis] for axis in range(3))
        if not any(u):
            continue
        for third in points[1:]:
            v = tuple(third[axis] - anchor[axis] for axis in range(3))
            planar_primitives_owner.reserve_fraction_work(
                (anchor, second, third), 15, 4, 12, budget=budget
            )
            normal = (
                u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0],
            )
            if any(normal):
                offset = -(normal[0] * anchor[0] + normal[1] * anchor[1] + normal[2] * anchor[2])
                return (*normal, offset)
        return None
    return None


def _plane_key(plane: _Plane, /) -> _Plane:
    leading = next(value for value in plane[:3] if value)
    scale = abs(leading)
    return tuple(value / scale for value in plane)


def _dot(first: tuple[Fraction, ...], second: tuple[Fraction, ...], /) -> Fraction:
    return first[0] * second[0] + first[1] * second[1] + first[2] * second[2]


def _cross(first: tuple[Fraction, ...], second: tuple[Fraction, ...], /) -> _Point:
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def _projection(plane: _Plane, /) -> tuple[tuple[int, int], int]:
    """Return the dominant-axis drop in cyclic order and the normal's handedness.

    Dropping axis ``a`` and keeping ``(a + 1, a + 2)`` in cyclic order maps a loop
    counter-clockwise about ``plane[:3]`` to a positively oriented planar loop
    exactly when ``plane[a] > 0``; the returned sign restores that orientation.
    """
    axis = max(range(3), key=lambda value: abs(plane[value]))
    return ((axis + 1) % 3, (axis + 2) % 3), 1 if plane[axis] > 0 else -1


def _simple_convex_loop(
    loop: tuple[tuple[Fraction, ...], ...], /, *, budget: CoordinateEnclosureBudget
) -> bool:
    """Return whether a positively oriented planar loop bounds a convex polygon once.

    Every vertex lies weakly left of every directed edge. A loop with positive
    area and this property has turning number one: a star loop or a backtracking
    collinear vertex puts some vertex strictly right of an edge.
    """
    count = len(loop)
    planar_primitives_owner.reserve_fraction_work(loop, 6 * count * count, 1, 8, budget=budget)
    for index in range(count):
        a, b = loop[index], loop[(index + 1) % count]
        dx, dy = b[0] - a[0], b[1] - a[1]
        if any(dx * (point[1] - a[1]) - dy * (point[0] - a[0]) < 0 for point in loop):
            return False
    return True


def _convex_cell(
    cell_id: int,
    faces: tuple[tuple[int, ...], ...],
    bank: CoordinateSourceBank,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> _ConvexCell | None:
    """Prove one closed convex full-dimensional cell and its boundary subdivision.

    Every face is a simple convex loop on a supporting plane of the vertex hull
    ``K`` with one consistent outward orientation, so the face 2-chain lies on
    the sphere ``∂K``. Each directed edge meets exactly one reversed edge, so the
    chain is a cycle and its winding degree is one constant ``m`` on ``∂K`` off
    the edges; every face contributes positively, so ``m >= 1`` and the degree at
    a point counts the faces covering it. On each supporting plane the face areas
    sum to ``m`` times the exact facet area ``area(K ∩ plane)``; equality with
    that area gives ``m = 1``: the faces tile every facet with disjoint
    interiors, overlapping coplanar faces are refused, and a vertex inside
    another face's edge leaves an unpartnered directed edge (T-junction). Every
    vertex link is checked to be one cycle of corners, so no pinched vertex
    survives. The cell is ``K`` and its exact volume is returned.
    """
    vertices = tuple(sorted({vertex for face in faces for vertex in face}))
    corners = sum(len(face) for face in faces)
    budget.reserve(len(vertices) + corners, 256 + 160 * (len(vertices) + corners))
    points: dict[int, _Point] = {}
    for vertex in vertices:
        point = _rational_point(bank[vertex])
        if point is None:
            return None
        points[vertex] = point
    if len(set(points.values())) != len(points):
        return None
    count = Fraction(len(vertices))
    planar_primitives_owner.reserve_fraction_work(
        tuple(points.values()), 3 * len(vertices), 3, 2 * len(vertices) + 2, budget=budget
    )
    centroid = tuple(sum(points[vertex][axis] for vertex in vertices) / count for axis in range(3))
    # Supporting-plane key -> (outward plane, hull vertices on it, outward loops, total area).
    facets: dict[_Plane, tuple[_Plane, tuple[int, ...], list[tuple[int, ...]], list[Fraction]]] = {}
    orientation = 0
    for face in faces:
        if len(face) < 3 or len(set(face)) != len(face):
            return None
        loop = tuple(points[vertex] for vertex in face)
        plane = _face_plane(loop, budget=budget)
        if plane is None:
            return None
        side = _evaluate(plane, centroid, budget=budget)
        if not side:
            return None
        if side > 0:
            plane = tuple(-value for value in plane)
        key = _plane_key(plane)
        record = facets.get(key)
        if record is None:
            # Each distinct outward plane is checked against every vertex once;
            # coplanar faces share the exact on-plane vertex set.
            on_plane: list[int] = []
            for vertex in vertices:
                value = _evaluate(plane, points[vertex], budget=budget)
                if value > 0:
                    return None
                if not value:
                    on_plane.append(vertex)
            record = facets[key] = (plane, tuple(on_plane), [], [])
        if not set(face) <= set(record[1]):
            return None
        axes, handedness = _projection(plane)
        projected = planar_primitives_owner.project(loop, axes, budget=budget)
        area = planar_primitives_owner.signed_measure(projected, budget=budget)
        if not isinstance(area, Fraction) or not area:
            return None
        sign = 1 if area * handedness > 0 else -1
        if orientation and sign != orientation:
            return None
        orientation = sign
        if not _simple_convex_loop(projected if area > 0 else projected[::-1], budget=budget):
            return None
        record[2].append(face if sign > 0 else face[::-1])
        record[3].append(abs(area))
    # Each outward directed edge has exactly one reversed partner in the cell.
    edges: dict[tuple[int, int], int] = {}
    following: dict[tuple[int, int], int] = {}
    for _, _, loops, _ in facets.values():
        for loop in loops:
            for index, vertex in enumerate(loop):
                edge = (vertex, loop[(index + 1) % len(loop)])
                edges[edge] = edges.get(edge, 0) + 1
                # At ``vertex`` the face entering from loop[index - 1] leaves to edge[1].
                following[(vertex, loop[index - 1])] = edge[1]
    budget.reserve(4 * len(edges), 256 * len(edges))
    if any(
        multiplicity != 1 or edges.get((edge[1], edge[0])) != 1
        for edge, multiplicity in edges.items()
    ):
        return None
    # Every vertex link is one cycle: the face leaving along v->w is followed by
    # the face entering along w->v. Edge pairing makes this a permutation of the
    # corners at v; one orbit covering every corner is a connected link.
    corners_at: dict[int, list[int]] = {}
    for vertex, previous in following:
        corners_at.setdefault(vertex, []).append(previous)
    for vertex, previous in corners_at.items():
        steps, current = 1, following[(vertex, previous[0])]
        while current != previous[0]:
            steps, current = steps + 1, following[(vertex, current)]
        if steps != len(previous):
            return None
    # Facet coverage: winding degree one, i.e. face areas sum to the facet area.
    for plane, supported, _, areas in facets.values():
        axes, _ = _projection(plane)
        support = planar_primitives_owner.project(
            tuple(points[vertex] for vertex in supported), axes, budget=budget
        )
        hull = planar_primitives_owner.convex_hull(support, budget=budget)
        hull_area = planar_primitives_owner.signed_measure(hull, budget=budget)
        if not isinstance(hull_area, Fraction) or abs(hull_area) != sum(areas, Fraction(0)):
            return None
    apex = points[vertices[0]]
    six_volume = Fraction(0)
    for _, _, loops, _ in facets.values():
        for loop in loops:
            if vertices[0] in loop:
                continue
            corners_ = tuple(points[vertex] for vertex in loop)
            for index in range(1, len(corners_) - 1):
                rows = tuple(
                    tuple(corner[axis] - apex[axis] for axis in range(3))
                    for corner in (corners_[0], corners_[index], corners_[index + 1])
                )
                planar_primitives_owner.reserve_fraction_work(rows, 23, 1, 18, budget=budget)
                determinant = _dot(rows[0], _cross(rows[1], rows[2]))
                if determinant < 0:
                    return None
                six_volume += determinant
    if six_volume <= 0:
        return None
    return _ConvexCell(
        cell_id,
        tuple(points[vertex] for vertex in vertices),
        tuple(plane for plane, _, _, _ in facets.values()),
        six_volume / 6,
    )


def _restriction_planes(
    geometry: CellGeometrySpec, source_geometry: CellGeometrySpec, /
) -> tuple[_Plane, ...] | None:
    """Collect the exact restriction planes between target and source sources."""
    current, original = geometry.exact_source, source_geometry.exact_source
    if not isinstance(current, _POWER_SOURCES) or not isinstance(original, _POWER_SOURCES):
        return None
    planes: list[_Plane] = []
    visited: set[int] = set()
    while current.source_id != original.source_id:
        if id(current) in visited or isinstance(current, ExactPowerCellGeometrySource):
            return None
        visited.add(id(current))
        if isinstance(current, ExactPowerCellGeometryRestrictionSource):
            for row in np.asarray(current.planes, dtype=np.float64):
                normal = tuple(Fraction(float(value)) for value in row[:3])
                plane = (*normal, -Fraction(float(row[3])))
                if plane not in planes:
                    planes.append(plane)
        current = current.parent
    return tuple(planes) if planes else None

def _paired_quotient_facets(
    facet_orbits: np.ndarray, /, *, budget: CoordinateEnclosureBudget
) -> bool:
    """Return whether every quotient facet orbit has exactly two occurrences."""
    budget.reserve(facet_orbits.size, 256 + 16 * facet_orbits.size)
    if facet_orbits.size == 0 or np.any(facet_orbits < 0):
        return False
    counts = np.bincount(facet_orbits)
    return bool(np.all(counts[np.unique(facet_orbits)] == 2))


def _parent_partition(
    parents: dict[int, _ConvexCell],
    children: tuple[tuple[int, _ConvexCell], ...],
    planes: tuple[_Plane, ...],
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[tuple[int, int, tuple[int, ...], Fraction], ...] | None:
    """Prove each declared parent is exactly partitioned by its children.

    ``children`` pairs each proved child cell with its declared parent ID. Each
    child lies in its parent and in one closed sign cell of ``planes``, sign
    vectors are unique per parent, and exact volumes sum to every parent's
    volume. Returns ``(child, parent, signs, volume)`` rows, else ``None``.
    """
    signs_by_parent: dict[int, set[tuple[int, ...]]] = {parent: set() for parent in parents}
    totals = dict.fromkeys(parents, Fraction(0))
    rows: list[tuple[int, int, tuple[int, ...], Fraction]] = []
    for parent_id, child in children:
        parent = parents.get(parent_id)
        if parent is None:
            return None
        points = child.points
        if any(
            _evaluate(plane, point, budget=budget) > 0
            for plane in parent.planes
            for point in points
        ):
            return None
        vector = []
        for plane in planes:
            values = tuple(_evaluate(plane, point, budget=budget) for point in points)
            if all(value >= 0 for value in values) and any(value > 0 for value in values):
                vector.append(1)
            elif all(value <= 0 for value in values) and any(value < 0 for value in values):
                vector.append(-1)
            else:
                return None
        signs = tuple(vector)
        if signs in signs_by_parent[parent_id]:
            return None
        signs_by_parent[parent_id].add(signs)
        totals[parent_id] += child.volume
        rows.append((child.cell_id, parent_id, signs, child.volume))
    if any(totals[cell_id] != parent.volume for cell_id, parent in parents.items()):
        return None
    return tuple(rows)



def _signed_boundary_faces(
    faces: tuple[tuple[int, ...], ...],
    bank: CoordinateSourceBank,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[tuple[tuple[_Point, ...], int], ...] | None:
    """Canonical exact outward loops, retaining orientation rather than area alone."""
    vertices = sorted({vertex for face in faces for vertex in face})
    points: dict[int, _Point] = {}
    for vertex in vertices:
        point = _rational_point(bank[vertex])
        if point is None:
            return None
        points[vertex] = point
    planar_primitives_owner.reserve_fraction_work(
        tuple(points.values()), 3 * len(vertices), 3, 2 * len(vertices) + 2, budget=budget
    )
    center = tuple(
        sum((point[axis] for point in points.values()), Fraction(0)) / len(vertices)
        for axis in range(3)
    )
    result = []
    for face in faces:
        loop = tuple(points[vertex] for vertex in face)
        plane = _face_plane(loop, budget=budget)
        if plane is None:
            return None
        side = _evaluate(plane, center, budget=budget)
        if not side:
            return None
        if side > 0:
            loop = loop[::-1]
        planar_primitives_owner.reserve_fraction_work(
            loop, 6 * len(loop) * len(loop), 1, 4, budget=budget
        )
        forward = min(loop[index:] + loop[:index] for index in range(len(loop)))
        reverse = loop[::-1]
        backward = min(reverse[index:] + reverse[:index] for index in range(len(loop)))
        result.append((min(forward, backward), 1 if forward < backward else -1))
    return tuple(result)


def _inverse_boundary_subcomplex(
    source: PolyhedralConnectivity,
    target: PolyhedralConnectivity,
    source_bank: CoordinateSourceBank,
    target_bank: CoordinateSourceBank,
    declared: dict[int, int],
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> bool:
    """Require exact signed face cancellation and the retained source subcomplex.

    Interior source faces cancel only with their exact reversed partner. Every
    remaining outward face occurs once and is exactly one target face, including
    all collinear subdivision vertices; an area match or equal cell count cannot
    substitute for this chain identity.
    """
    source_chains: dict[int, dict[tuple[_Point, ...], int]] = {}
    for child, faces in _cell_faces(source):
        rows = _signed_boundary_faces(faces, source_bank, budget=budget)
        if rows is None or child not in declared:
            return False
        chain = source_chains.setdefault(declared[child], {})
        budget.reserve(len(rows), 128 + sum(128 + 96 * len(loop) for loop, _ in rows))
        for loop, sign in rows:
            chain[loop] = chain.get(loop, 0) + sign
    target_cells = _cell_faces(target)
    if set(source_chains) != {cell for cell, _ in target_cells}:
        return False
    for cell, faces in target_cells:
        rows = _signed_boundary_faces(faces, target_bank, budget=budget)
        if rows is None or len({loop for loop, _ in rows}) != len(rows):
            return False
        remaining = {loop: sign for loop, sign in source_chains[cell].items() if sign}
        if any(abs(sign) != 1 for sign in remaining.values()) or remaining != dict(rows):
            return False
    return True


def certify_periodic_restriction_partition(
    state: EmbeddingState,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    validity: CellValidityCertificate | None,
    premise: PeriodicRestrictionPremise,
    facet_trace_certificate_id: str | None,
    facet_orbits: np.ndarray,
    target_bank: CoordinateSourceBank,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> PeriodicRestrictionPartitionEvidence | None:
    """Return earned partition evidence, or ``None`` outside admission.

    ``None`` leaves the caller's bounded image-pair proof responsible for the
    whole embedding; no partial premise is retained.
    """
    topology, source_topology = mesh.periodic_topology, premise.source_mesh.periodic_topology
    connectivity = mesh.connectivity
    source_connectivity = premise.source_mesh.connectivity
    embedding = premise.source_embedding
    if (
        validity is None
        or not validity.all_certified
        or facet_trace_certificate_id is None
        or state.findings
        or state.boundary_facets
        or topology is None
        or source_topology is None
        or mesh.storage is not None
        or mesh.topological_dimension != 3
        or mesh.ambient_dimension != 3
        or not isinstance(connectivity, PolyhedralConnectivity)
        or not isinstance(source_connectivity, PolyhedralConnectivity)
        or identification_id(topology.cell) != identification_id(source_topology.cell)
        or embedding.status != "certified"
        or embedding.findings
        or not premise.source_validity.all_certified
        or premise.source_mesh.storage is not None
        or (
            embedding.binding.mesh_id,
            embedding.binding.geometry_id,
            embedding.validity_certificate_id,
            premise.source_validity.geometry_id,
        )
        != (
            premise.source_mesh.mesh_id,
            cell_geometry_id(premise.source_geometry),
            premise.source_validity.certificate_id,
            cell_geometry_id(premise.source_geometry),
        )
    ):
        return None
    embedding.binding.require(premise.source_mesh, premise.source_geometry)
    premise.source_validity.require_bound(
        premise.source_geometry, mesh=premise.source_mesh
    )
    if not _paired_quotient_facets(facet_orbits, budget=budget):
        return None
    planes = _restriction_planes(geometry, premise.source_geometry)
    if premise.inverse:
        selection = geometry.exact_source
        if (
            not isinstance(selection, ExactPowerCellGeometryRestrictionSource)
            or premise.source_geometry.exact_source is None
            or selection.parent.source_id != premise.source_geometry.exact_source.source_id
            or selection.planes.shape[0] != 0
            or np.any(np.asarray(selection.vertex_parents)[:, 1] != -1)
            or np.any(np.asarray(selection.vertex_plane_ids) != -1)
        ):
            return None
        # Agglomeration's geometry is an exact vertex selection of the source.
        # Its parent's split planes provide the separating sign cells, not an
        # inferred geometric tolerance or a transplanted ancestor certificate.
        power = premise.source_geometry.exact_source
        if not isinstance(power, _POWER_SOURCES):
            return None
        planes_list: list[_Plane] = []
        visited: set[int] = set()
        while not isinstance(power, ExactPowerCellGeometrySource):
            if id(power) in visited:
                return None
            visited.add(id(power))
            if isinstance(power, ExactPowerCellGeometryRestrictionSource):
                for row in np.asarray(power.planes, dtype=np.float64):
                    plane = tuple(Fraction(float(value)) for value in row)
                    signed = (*plane[:3], -plane[3])
                    if signed not in planes_list:
                        planes_list.append(signed)
            power = power.parent
        planes = tuple(planes_list) if planes_list else None
    if planes is None:
        return None
    source_bank = prepared_coordinate_source_bank(premise.source_geometry, budget=budget)
    if premise.inverse:
        selection = geometry.exact_source
        if not isinstance(selection, ExactPowerCellGeometryRestrictionSource):
            return None
        rows = np.asarray(selection.vertex_parents)[:, 0]
        budget.reserve(4 * len(rows), 128 + 16 * len(rows))
        if (
            len(target_bank) != len(rows)
            or not np.array_equal(
                np.asarray(mesh.vertex_global_ids),
                np.asarray(premise.source_mesh.vertex_global_ids)[rows],
            )
            or any(
                point != source_bank[int(row)]
                for point, row in zip(target_bank, rows, strict=True)
            )
        ):
            return None
    parent_connectivity, child_connectivity = (
        (connectivity, source_connectivity)
        if premise.inverse
        else (source_connectivity, connectivity)
    )
    parent_bank, child_bank = (
        (target_bank, source_bank) if premise.inverse else (source_bank, target_bank)
    )
    parents: dict[int, _ConvexCell] = {}
    for cell_id, faces in _cell_faces(parent_connectivity):
        cell = _convex_cell(cell_id, faces, parent_bank, budget=budget)
        if cell is None:
            return None
        parents[cell_id] = cell
    declared = dict(premise.parent_cell_ids)
    cells = _cell_faces(child_connectivity)
    if len(declared) != len(premise.parent_cell_ids) or set(declared) != {
        cell_id for cell_id, _ in cells
    }:
        return None
    proved: list[tuple[int, _ConvexCell]] = []
    for cell_id, faces in cells:
        child = _convex_cell(cell_id, faces, child_bank, budget=budget)
        if child is None:
            return None
        proved.append((declared[cell_id], child))
    children = _parent_partition(parents, tuple(proved), planes, budget=budget)
    if children is None:
        return None
    if premise.inverse and not _inverse_boundary_subcomplex(
        source_connectivity, connectivity, source_bank, target_bank, declared, budget=budget
    ):
        return None
    face_records = tuple(
        (child.cell_id, tuple(_plane_key(plane) for plane in child.planes))
        for _, child in proved
    )

    def ratio(value: Fraction) -> tuple[int, int]:
        return value.numerator, value.denominator

    boundary_subdivision = fingerprint_owner.canonical_fingerprint(
        {
            "kind": "periodic-restriction-convex-boundary-subdivisions",
            "mesh": mesh.mesh_id,
            "direction": "coarsen" if premise.inverse else "refine",
            "signed_boundary_cancellation": premise.inverse,
            "source_mesh": premise.source_mesh.mesh_id,
            "source_facets": tuple(
                (cell_id, tuple(tuple(ratio(value) for value in _plane_key(plane)) for plane in cell.planes))
                for cell_id, cell in parents.items()
            ),
            "facets": tuple(
                (cell_id, tuple(tuple(ratio(value) for value in plane) for plane in keys))
                for cell_id, keys in face_records
            ),
        }
    )
    state.checks.extend(
        (
            "periodic_restriction_source_embedding",
            "periodic_restriction_parent_partition",
            "periodic_restriction_boundary_subdivision",
        )
    )
    return PeriodicRestrictionPartitionEvidence(
        mesh.mesh_id,
        validity.geometry_id,
        topology.periodic_topology_id,
        validity.certificate_id,
        facet_trace_certificate_id,
        boundary_subdivision,
        identification_id(topology.cell),
        premise.source_mesh.mesh_id,
        cell_geometry_id(premise.source_geometry),
        premise.source_validity.certificate_id,
        embedding.certificate_id,
        tuple(tuple(ratio(value) for value in plane) for plane in planes),
        tuple(row[0] for row in children),
        tuple(row[1] for row in children),
        tuple(row[2] for row in children),
        tuple(ratio(row[3]) for row in children),
        tuple(parents),
        tuple(ratio(parent.volume) for parent in parents.values()),
        direction="coarsen" if premise.inverse else "refine",
    )


__all__ = [
    "certify_periodic_restriction_partition",
    "PeriodicRestrictionPremise",
]
