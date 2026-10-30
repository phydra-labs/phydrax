#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Exact convex partition witnesses for periodic affine restrictions."""

from __future__ import annotations

from fractions import Fraction
from typing import TYPE_CHECKING

import numpy as np

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._validation import canonical_identifier
from ..discretization._cell_complex import PolyhedralConnectivity
from ..discretization._cell_geometry import (
    CellGeometryRestrictionSource,
    CellGeometrySpec,
    CellVertexGeometryElement,
    ScalarCoordinateElement,
)
from ..discretization._cell_geometry_transition_types import CellGeometryTransition
from ..discretization._cell_geometry_validity import (
    cell_geometry_id,
    CellValidityCertificate,
)
from ..discretization._cell_mesh import CellMesh
from ..discretization._coordinate_enclosure import (
    affine_map,
    coordinate_corner_images,
    coordinate_map_expressions,
)
from ..discretization._periodic_exact import identification_id
from ..discretization._periodic_topology import PeriodicMeshTopology
from ..discretization._reference_cell import reference_cell_topology
from ._embedding_state import EmbeddingState


if TYPE_CHECKING:
    from ._embedding_state import PeriodicRestrictionPartitionEvidence
    from ._global_embedding import GlobalEmbeddingCertificate


type _Point = tuple[Fraction, Fraction, Fraction]
type _Plane = tuple[Fraction, Fraction, Fraction, Fraction]
type _Ratio = tuple[int, int]
type _PlaneRecord = tuple[_Ratio, ...]
type _FaceRecord = tuple[tuple[_Ratio, ...], ...]
type _CellPlaneRecord = tuple[int, tuple[_PlaneRecord, ...], tuple[_FaceRecord, ...]]


class PeriodicRestrictionPremise(StrictModule):
    """Retain the current source proof and exact endpoint-bound cell transition."""

    source_mesh: CellMesh
    source_geometry: CellGeometrySpec
    source_validity: CellValidityCertificate
    source_embedding: GlobalEmbeddingCertificate
    transition: CellGeometryTransition | None

    def __init__(
        self,
        source_mesh: CellMesh,
        source_geometry: CellGeometrySpec,
        source_validity: CellValidityCertificate,
        source_embedding: GlobalEmbeddingCertificate,
        transition: CellGeometryTransition | None = None,
        /,
    ) -> None:
        from ._global_embedding import GlobalEmbeddingCertificate

        if not isinstance(source_mesh, CellMesh):
            raise TypeError("source_mesh must be CellMesh.")
        if not isinstance(source_geometry, CellGeometrySpec):
            raise TypeError("source_geometry must be CellGeometrySpec.")
        if not isinstance(source_validity, CellValidityCertificate):
            raise TypeError("source_validity must be CellValidityCertificate.")
        if not isinstance(source_embedding, GlobalEmbeddingCertificate):
            raise TypeError("source_embedding must be GlobalEmbeddingCertificate.")
        source_embedding.binding.require(source_mesh, source_geometry)
        source_validity.require_bound(source_geometry, mesh=source_mesh)
        if transition is not None and not isinstance(transition, CellGeometryTransition):
            raise TypeError("transition must be CellGeometryTransition or None.")
        self.source_mesh = source_mesh
        self.source_geometry = source_geometry
        self.source_validity = source_validity
        self.source_embedding = source_embedding
        self.transition = transition


class _ConvexCell(StrictModule):
    cell_id: int
    points: tuple[_Point, ...]
    planes: tuple[_Plane, ...]
    volume: Fraction
    boundary_faces: tuple[tuple[_Point, ...], ...]


def _cross(first: _Point, second: _Point) -> _Point:
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def _dot(first: _Point, second: _Point) -> Fraction:
    return first[0] * second[0] + first[1] * second[1] + first[2] * second[2]


def _face_plane(points: tuple[_Point, ...]) -> _Plane | None:
    anchor = points[0]
    for second in points[1:]:
        first_vector = (
            second[0] - anchor[0],
            second[1] - anchor[1],
            second[2] - anchor[2],
        )
        if not any(first_vector):
            continue
        for third in points[1:]:
            second_vector = (
                third[0] - anchor[0],
                third[1] - anchor[1],
                third[2] - anchor[2],
            )
            normal = _cross(first_vector, second_vector)
            if any(normal):
                return (*normal, -_dot(normal, anchor))
        return None
    return None


def _evaluate(plane: _Plane, point: _Point) -> Fraction:
    return plane[0] * point[0] + plane[1] * point[1] + plane[2] * point[2] + plane[3]


def _plane_key(plane: _Plane) -> _Plane:
    leading = next(value for value in plane[:3] if value)
    scale = abs(leading)
    return (
        plane[0] / scale,
        plane[1] / scale,
        plane[2] / scale,
        plane[3] / scale,
    )


def _partition_plane_key(plane: _Plane) -> _Plane:
    leading = next(value for value in plane[:3] if value)
    return (
        plane[0] / leading,
        plane[1] / leading,
        plane[2] / leading,
        plane[3] / leading,
    )


def _negate_plane(plane: _Plane) -> _Plane:
    return (-plane[0], -plane[1], -plane[2], -plane[3])


def _projection(plane: _Plane) -> tuple[tuple[int, int], int]:
    axis = max(range(3), key=lambda value: abs(plane[value]))
    return ((axis + 1) % 3, (axis + 2) % 3), 1 if plane[axis] > 0 else -1


def _signed_area2(points: tuple[tuple[Fraction, Fraction], ...]) -> Fraction:
    return sum(
        (
            points[index][0] * points[(index + 1) % len(points)][1]
            - points[index][1] * points[(index + 1) % len(points)][0]
            for index in range(len(points))
        ),
        Fraction(0),
    )


def _convex_hull(
    points: tuple[tuple[Fraction, Fraction], ...],
) -> tuple[tuple[Fraction, Fraction], ...]:
    ordered = sorted(set(points))
    if len(ordered) < 3:
        return tuple(ordered)

    def turn(
        first: tuple[Fraction, Fraction],
        second: tuple[Fraction, Fraction],
        third: tuple[Fraction, Fraction],
    ) -> Fraction:
        return (second[0] - first[0]) * (third[1] - first[1]) - (second[1] - first[1]) * (
            third[0] - first[0]
        )

    lower: list[tuple[Fraction, Fraction]] = []
    for point in ordered:
        while len(lower) >= 2 and turn(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[Fraction, Fraction]] = []
    for point in reversed(ordered):
        while len(upper) >= 2 and turn(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return tuple(lower[:-1] + upper[:-1])


def _simple_convex_loop(points: tuple[tuple[Fraction, Fraction], ...]) -> bool:
    if len(points) < 3 or _signed_area2(points) <= 0:
        return False
    for index, first in enumerate(points):
        second = points[(index + 1) % len(points)]
        dx, dy = second[0] - first[0], second[1] - first[1]
        if any(
            dx * (point[1] - first[1]) - dy * (point[0] - first[0]) < 0
            for point in points
        ):
            return False
    return True


def _cell_face_rows(
    mesh: CellMesh,
) -> tuple[tuple[int, tuple[tuple[int, ...], ...]], ...] | None:
    """Return each cell's oriented face loops in its mesh-vertex row indices."""
    connectivity = mesh.connectivity
    if isinstance(connectivity, PolyhedralConnectivity):
        face_offsets = np.asarray(connectivity.face_vertex_offsets, dtype=np.int64)
        face_values = np.asarray(connectivity.face_vertex_values, dtype=np.int64)
        cell_offsets = np.asarray(connectivity.cell_face_offsets, dtype=np.int64)
        cell_faces = np.asarray(connectivity.cell_face_values, dtype=np.int64)
        cell_signs = np.asarray(connectivity.cell_face_sign_values, dtype=np.int8)
        cell_ids = np.asarray(connectivity.cell_global_ids, dtype=np.int64)
        if (
            cell_ids.shape != (cell_offsets.size - 1,)
            or cell_signs.shape != cell_faces.shape
        ):
            return None
        result = []
        for row, identifier in enumerate(cell_ids.tolist()):
            loops = []
            for position in range(cell_offsets[row], cell_offsets[row + 1]):
                face = int(cell_faces[position])
                loop = tuple(
                    int(vertex)
                    for vertex in face_values[face_offsets[face] : face_offsets[face + 1]]
                )
                if cell_signs[position] < 0:
                    loop = loop[::-1]
                elif cell_signs[position] != 1:
                    return None
                loops.append(loop)
            result.append((int(identifier), tuple(loops)))
        return tuple(result)

    result = []
    for block in mesh.blocks:
        topology = reference_cell_topology(block.cell_kind)
        if topology.dimension != 3 or len(topology.entities) < 3:
            return None
        rows = np.asarray(block.vertices, dtype=np.int64)
        valid = np.asarray(block.vertex_valid, dtype=np.bool_)
        identifiers = np.asarray(block.global_ids, dtype=np.int64)
        if rows.shape != valid.shape or rows.shape[0] != identifiers.size:
            return None
        if not np.all(valid):
            return None
        for identifier, row in zip(identifiers.tolist(), rows.tolist(), strict=True):
            result.append(
                (
                    int(identifier),
                    tuple(
                        tuple(int(row[int(vertex)]) for vertex in face)
                        for face in topology.entities[2]
                    ),
                )
            )
    return tuple(result)


def _exact_vertex_points(
    mesh: CellMesh, geometry: CellGeometrySpec
) -> dict[int, _Point] | None:
    """Evaluate every affine coordinate corner in its retained exact source bank."""
    elements, routes, _ = geometry.resolve(mesh)
    source = geometry.source_coordinates()
    exact_source_rows: list[tuple[Fraction, ...]] = []
    for row in source:
        exact_row = []
        for value in row:
            if not isinstance(value, Fraction):
                return None
            exact_row.append(value)
        exact_source_rows.append(tuple(exact_row))
    exact_source = tuple(exact_source_rows)
    result: dict[int, _Point] = {}
    for block, element, route_value in zip(mesh.blocks, elements, routes, strict=True):
        routes_ = np.asarray(route_value, dtype=np.int64)
        rows = np.asarray(block.vertices, dtype=np.int64)
        valid = np.asarray(block.vertex_valid, dtype=np.bool_)
        if routes_.shape != rows.shape or valid.shape != rows.shape:
            return None
        for cell_row, cell in enumerate(rows):
            if isinstance(element, CellVertexGeometryElement):
                corners = tuple(exact_source[int(index)] for index in routes_[cell_row])
            elif isinstance(element, ScalarCoordinateElement):
                coefficients = tuple(
                    exact_source[int(index)] for index in routes_[cell_row]
                )
                expressions = coordinate_map_expressions(
                    element, coefficients, cell_row=cell_row
                )
                if affine_map(expressions, mesh.topological_dimension) is None:
                    return None
                corners = coordinate_corner_images(
                    element, coefficients, cell_row=cell_row
                )
            else:
                return None
            active_vertices = cell[valid[cell_row]]
            if len(corners) != active_vertices.size:
                return None
            for vertex, point in zip(active_vertices.tolist(), corners, strict=True):
                if len(point) != 3:
                    return None
                x, y, z = point
                if (
                    not isinstance(x, Fraction)
                    or not isinstance(y, Fraction)
                    or not isinstance(z, Fraction)
                ):
                    return None
                exact = (x, y, z)
                old = result.setdefault(int(vertex), exact)
                if old != exact:
                    return None
    return result


def _convex_cell(
    cell_id: int,
    faces: tuple[tuple[int, ...], ...],
    point_bank: dict[int, _Point],
) -> _ConvexCell | None:
    """Prove convexity and strong boundary subdivision of one exact cell."""
    vertices = tuple(sorted({vertex for face in faces for vertex in face}))
    if len(vertices) < 4 or not faces:
        return None
    points: dict[int, _Point] = {}
    for vertex in vertices:
        point = point_bank.get(vertex)
        if point is None:
            return None
        points[vertex] = point
    if len(set(points.values())) != len(points):
        return None
    count = Fraction(len(vertices))
    center: _Point = (
        sum((points[vertex][0] for vertex in vertices), Fraction(0)) / count,
        sum((points[vertex][1] for vertex in vertices), Fraction(0)) / count,
        sum((points[vertex][2] for vertex in vertices), Fraction(0)) / count,
    )
    facets: dict[
        _Plane,
        tuple[_Plane, tuple[int, ...], list[tuple[int, ...]], list[Fraction]],
    ] = {}
    orientation = 0
    for face in faces:
        if len(face) < 3 or len(set(face)) != len(face):
            return None
        loop = tuple(points[vertex] for vertex in face)
        plane = _face_plane(loop)
        if plane is None or any(_evaluate(plane, point) for point in loop):
            return None
        side = _evaluate(plane, center)
        if not side:
            return None
        outward = plane if side < 0 else _negate_plane(plane)
        key = _plane_key(outward)
        record = facets.get(key)
        if record is None:
            supported_vertices: list[int] = []
            for vertex in vertices:
                value = _evaluate(outward, points[vertex])
                if value > 0:
                    return None
                if not value:
                    supported_vertices.append(vertex)
            record = (outward, tuple(supported_vertices), [], [])
            facets[key] = record
        if not set(face) <= set(record[1]):
            return None
        axes, handedness = _projection(outward)
        projected = tuple((point[axes[0]], point[axes[1]]) for point in loop)
        area2 = _signed_area2(projected)
        if not area2:
            return None
        sign = 1 if area2 * handedness > 0 else -1
        if orientation and sign != orientation:
            return None
        orientation = sign
        if not _simple_convex_loop(projected if area2 > 0 else projected[::-1]):
            return None
        record[2].append(face if sign > 0 else face[::-1])
        record[3].append(abs(area2))

    directed_edges: dict[tuple[int, int], int] = {}
    following: dict[tuple[int, int], int] = {}
    for _, _, loops, _ in facets.values():
        for loop in loops:
            for index, vertex in enumerate(loop):
                edge = (vertex, loop[(index + 1) % len(loop)])
                directed_edges[edge] = directed_edges.get(edge, 0) + 1
                following[(vertex, loop[index - 1])] = edge[1]
    if any(
        multiplicity != 1 or directed_edges.get((edge[1], edge[0])) != 1
        for edge, multiplicity in directed_edges.items()
    ):
        return None
    corners_at: dict[int, list[int]] = {}
    for vertex, previous in following:
        corners_at.setdefault(vertex, []).append(previous)
    for vertex, previous in corners_at.items():
        start = previous[0]
        visited = {start}
        current = following[(vertex, start)]
        while current != start and current not in visited:
            visited.add(current)
            current = following[(vertex, current)]
        if current != start or len(visited) != len(previous):
            return None

    for plane, facet_vertices, _, areas in facets.values():
        axes, _ = _projection(plane)
        projected_support = tuple(
            (points[vertex][axes[0]], points[vertex][axes[1]])
            for vertex in facet_vertices
        )
        hull_area2 = abs(_signed_area2(_convex_hull(projected_support)))
        if hull_area2 != sum(areas, Fraction(0)):
            return None

    apex = points[vertices[0]]
    six_volume = Fraction(0)
    for _, _, loops, _ in facets.values():
        for loop in loops:
            if vertices[0] in loop:
                continue
            corners = tuple(points[vertex] for vertex in loop)
            for index in range(1, len(corners) - 1):
                first = (
                    corners[0][0] - apex[0],
                    corners[0][1] - apex[1],
                    corners[0][2] - apex[2],
                )
                second = (
                    corners[index][0] - apex[0],
                    corners[index][1] - apex[1],
                    corners[index][2] - apex[2],
                )
                third = (
                    corners[index + 1][0] - apex[0],
                    corners[index + 1][1] - apex[1],
                    corners[index + 1][2] - apex[2],
                )
                determinant = _dot(first, _cross(second, third))
                if determinant < 0:
                    return None
                six_volume += determinant
    if six_volume <= 0:
        return None
    return _ConvexCell(
        cell_id,
        tuple(points[vertex] for vertex in vertices),
        tuple(record[0] for record in facets.values()),
        six_volume / 6,
        tuple(
            tuple(points[vertex] for vertex in face)
            for _, _, loops, _ in facets.values()
            for face in loops
        ),
    )


def _oriented_edge_chain(
    faces: tuple[tuple[_Point, ...], ...],
) -> dict[tuple[_Point, _Point], int]:
    chain: dict[tuple[_Point, _Point], int] = {}
    for face in faces:
        for index, first in enumerate(face):
            second = face[(index + 1) % len(face)]
            edge = (first, second) if first < second else (second, first)
            sign = 1 if (first, second) == edge else -1
            chain[edge] = chain.get(edge, 0) + sign
    return {edge: sign for edge, sign in chain.items() if sign}


def _partition_boundary_subdivision(
    parents: dict[int, _ConvexCell],
    children: tuple[tuple[int, _ConvexCell], ...],
) -> bool:
    """Prove internal facet cancellation and exact parent-boundary subdivision."""
    grouped: dict[int, list[_ConvexCell]] = {identifier: [] for identifier in parents}
    for parent_id, child in children:
        if parent_id not in grouped:
            return False
        grouped[parent_id].append(child)
    for parent_id, parent in parents.items():
        parent_facets: dict[_Plane, list[tuple[_Point, ...]]] = {}
        for face in parent.boundary_faces:
            plane = _face_plane(face)
            if plane is None:
                return False
            parent_facets.setdefault(_partition_plane_key(plane), []).append(face)
        child_facets: dict[_Plane, list[tuple[_Point, ...]]] = {}
        internal_facets: dict[_Plane, list[tuple[_Point, ...]]] = {}
        for child in grouped[parent_id]:
            for face in child.boundary_faces:
                plane = _face_plane(face)
                if plane is None:
                    return False
                key = _partition_plane_key(plane)
                if key in parent_facets:
                    child_facets.setdefault(key, []).append(face)
                else:
                    internal_facets.setdefault(key, []).append(face)
        if set(child_facets) != set(parent_facets):
            return False
        for plane, faces in internal_facets.items():
            if _oriented_edge_chain(tuple(faces)):
                return False
        for plane, faces in parent_facets.items():
            if _oriented_edge_chain(tuple(faces)) != _oriented_edge_chain(
                tuple(child_facets[plane])
            ):
                return False
    return True


def _restriction_planes(
    parents: dict[int, _ConvexCell],
    children: tuple[tuple[int, _ConvexCell], ...],
) -> tuple[_Plane, ...] | None:
    """Derive exact internal cut planes from child faces; admit only full cuts."""
    candidates: set[_Plane] = set()
    for _, child in children:
        for plane in child.planes:
            candidates.add(_partition_plane_key(plane))
    cuts = []
    for plane in sorted(candidates):
        if any(
            any(_evaluate(plane, point) < 0 for point in parent.points)
            and any(_evaluate(plane, point) > 0 for point in parent.points)
            for parent in parents.values()
        ):
            cuts.append(plane)
    return tuple(cuts) if cuts else None


def _partition_side(plane: _Plane, child: _ConvexCell, /) -> int:
    values = tuple(_evaluate(plane, point) for point in child.points)
    if all(value <= 0 for value in values) and any(value < 0 for value in values):
        return -1
    if all(value >= 0 for value in values) and any(value > 0 for value in values):
        return 1
    return 0


def _parent_partition(
    parents: dict[int, _ConvexCell],
    children: tuple[tuple[int, _ConvexCell], ...],
    planes: tuple[_Plane, ...],
) -> (
    tuple[
        tuple[_Plane, ...],
        tuple[tuple[int, int, tuple[int, ...], Fraction], ...],
    ]
    | None
):
    """Prove containment, pairwise separation, exact volume, and boundary chains."""
    grouped: dict[int, list[_ConvexCell]] = {identifier: [] for identifier in parents}
    totals = dict.fromkeys(parents, Fraction(0))
    seen_children: set[int] = set()
    for parent_id, child in children:
        parent = parents.get(parent_id)
        if parent is None or child.cell_id in seen_children:
            return None
        seen_children.add(child.cell_id)
        if any(
            _evaluate(plane, point) > 0
            for plane in parent.planes
            for point in child.points
        ):
            return None
        grouped[parent_id].append(child)
        totals[parent_id] += child.volume
    if (
        not seen_children
        or any(not grouped[identifier] for identifier in parents)
        or any(
            totals[identifier] != parent.volume for identifier, parent in parents.items()
        )
        or not _partition_boundary_subdivision(parents, children)
    ):
        return None

    selected_planes: list[_Plane] = []
    side_by_plane: dict[_Plane, dict[int, int]] = {}
    for plane in planes:
        sides = {child.cell_id: _partition_side(plane, child) for _, child in children}
        separates_a_pair = any(
            sides[first.cell_id] == -sides[second.cell_id] and sides[first.cell_id] != 0
            for local_children in grouped.values()
            for first_index, first in enumerate(local_children)
            for second in local_children[first_index + 1 :]
        )
        if separates_a_pair:
            selected_planes.append(plane)
            side_by_plane[plane] = sides
    if not selected_planes:
        return None
    for local_children in grouped.values():
        for first_index, first in enumerate(local_children):
            for second in local_children[first_index + 1 :]:
                if not any(
                    side_by_plane[plane][first.cell_id]
                    == -side_by_plane[plane][second.cell_id]
                    and side_by_plane[plane][first.cell_id] != 0
                    for plane in selected_planes
                ):
                    return None

    signs_by_parent: dict[int, set[tuple[int, ...]]] = {
        identifier: set() for identifier in parents
    }
    rows = []
    for parent_id, child in children:
        vector = tuple(side_by_plane[plane][child.cell_id] for plane in selected_planes)
        if vector in signs_by_parent[parent_id]:
            return None
        signs_by_parent[parent_id].add(vector)
        rows.append((child.cell_id, parent_id, vector, child.volume))
    return tuple(selected_planes), tuple(sorted(rows, key=lambda row: row[0]))


def _periodic_facets_pair(mesh: CellMesh, topology: PeriodicMeshTopology) -> bool:
    offsets = np.asarray(topology.lifted_offsets, dtype=np.int64)
    if mesh.topological_dimension != 3 or offsets.shape != (5,):
        return False
    orbit_rows = np.asarray(topology.orbit_indices, dtype=np.int64)[
        offsets[2] : offsets[3]
    ]
    if orbit_rows.size == 0 or np.any(orbit_rows < 0):
        return False
    from ._periodic_embedding import _quotient_faces_pair_oriented

    return _quotient_faces_pair_oriented(topology) is not None


def _signed_boundary_faces(
    faces: tuple[tuple[int, ...], ...], point_bank: dict[int, _Point]
) -> tuple[tuple[tuple[_Point, ...], int], ...] | None:
    """Canonical exact outward loops with their oriented-chain coefficients."""
    vertices = sorted({vertex for face in faces for vertex in face})
    exact_points = []
    for vertex in vertices:
        point = point_bank.get(vertex)
        if point is None:
            return None
        exact_points.append(point)
    if not vertices:
        return None
    center: _Point = (
        sum((point[0] for point in exact_points), Fraction(0)) / len(vertices),
        sum((point[1] for point in exact_points), Fraction(0)) / len(vertices),
        sum((point[2] for point in exact_points), Fraction(0)) / len(vertices),
    )
    result = []
    for face in faces:
        exact_loop = tuple(point_bank[vertex] for vertex in face)
        plane = _face_plane(exact_loop)
        if plane is None:
            return None
        side = _evaluate(plane, center)
        if not side:
            return None
        if side > 0:
            exact_loop = exact_loop[::-1]
        forward = min(
            exact_loop[index:] + exact_loop[:index] for index in range(len(exact_loop))
        )
        reverse = exact_loop[::-1]
        backward = min(reverse[index:] + reverse[:index] for index in range(len(reverse)))
        result.append((min(forward, backward), 1 if forward < backward else -1))
    return tuple(result)


def _inverse_boundary_subcomplex(
    source_faces: tuple[tuple[int, tuple[tuple[int, ...], ...]], ...],
    target_faces: tuple[tuple[int, tuple[tuple[int, ...], ...]], ...],
    source_bank: dict[int, _Point],
    target_bank: dict[int, _Point],
    declared: dict[int, int],
) -> bool:
    """Require exact internal-face cancellation and the retained target boundary."""
    chains: dict[int, dict[tuple[_Point, ...], int]] = {}
    for child, faces in source_faces:
        rows = _signed_boundary_faces(faces, source_bank)
        if rows is None or child not in declared:
            return False
        chain = chains.setdefault(declared[child], {})
        for loop, sign in rows:
            chain[loop] = chain.get(loop, 0) + sign
    if set(chains) != {identifier for identifier, _ in target_faces}:
        return False
    for identifier, faces in target_faces:
        rows = _signed_boundary_faces(faces, target_bank)
        if rows is None or len({loop for loop, _ in rows}) != len(rows):
            return False
        remaining = {loop: sign for loop, sign in chains[identifier].items() if sign}
        if any(abs(sign) != 1 for sign in remaining.values()) or remaining != dict(rows):
            return False
    return True


def _relation(
    premise: PeriodicRestrictionPremise,
    target_mesh: CellMesh,
    target_geometry: CellGeometrySpec,
) -> tuple[bool, dict[int, int]] | None:
    if premise.transition is None:
        restriction = target_geometry.restriction_source
        if restriction is None:
            return None
        restriction = CellGeometryRestrictionSource.from_stored_fields(restriction)
        if (
            restriction.source_geometry_id != cell_geometry_id(premise.source_geometry)
            or restriction.source_topology_id != premise.source_mesh.topology_id
            or restriction.carrier_publication != "source_image"
        ):
            return None
        parent_tables = restriction.block_parent_cell_ids
        if set(parent_tables) != {block.name for block in target_mesh.blocks}:
            return None
        declared: dict[int, int] = {}
        for block in target_mesh.blocks:
            child_ids = np.asarray(block.global_ids, dtype=np.int64)
            parents = np.asarray(parent_tables[block.name], dtype=np.int64)
            if parents.shape != child_ids.shape or np.any(parents < 0):
                return None
            for child, parent in zip(child_ids.tolist(), parents.tolist(), strict=True):
                declared[int(child)] = int(parent)
        return False, declared
    transition = CellGeometryTransition.from_stored_fields(premise.transition)
    transition.require_bound(
        premise.source_mesh,
        premise.source_geometry,
        target_mesh,
        target_geometry,
    )
    if (transition.refinement is None) == (transition.coarsening is None):
        return None
    if transition.refinement is not None:
        witness = transition.refinement
        fine = np.asarray(witness.fine_cell_ids, dtype=np.int64)
        coarse = np.asarray(witness.coarse_cell_ids, dtype=np.int64)
        return False, dict(zip(fine.tolist(), coarse.tolist(), strict=True))
    witness = transition.coarsening
    if witness is None:
        return None
    fine = np.asarray(witness.fine_cell_ids, dtype=np.int64)
    coarse = np.asarray(witness.coarse_cell_ids, dtype=np.int64)
    return True, dict(zip(fine.tolist(), coarse.tolist(), strict=True))


def _inverse_vertex_selection(
    source_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    target_mesh: CellMesh,
    target_geometry: CellGeometrySpec,
    source_points: dict[int, _Point],
    target_points: dict[int, _Point],
) -> bool:
    restriction = target_geometry.restriction_source
    if restriction is None:
        return False
    restriction = CellGeometryRestrictionSource.from_stored_fields(restriction)
    if (
        restriction.source_geometry_id != cell_geometry_id(source_geometry)
        or restriction.source_topology_id != source_mesh.topology_id
    ):
        return False
    source_ids = np.asarray(source_mesh.vertex_global_ids, dtype=np.int64)
    target_ids = np.asarray(target_mesh.vertex_global_ids, dtype=np.int64)
    source_rows = {int(identifier): row for row, identifier in enumerate(source_ids)}
    if len(target_ids) > len(source_ids) or any(
        int(identifier) not in source_rows for identifier in target_ids
    ):
        return False
    for target_row, identifier_value in enumerate(target_ids):
        source_row = source_rows[int(identifier_value)]
        if target_points.get(target_row) != source_points.get(source_row):
            return False
    parent_vertices = restriction.block_parent_vertex_ids
    for block in target_mesh.blocks:
        declared = np.asarray(parent_vertices[block.name], dtype=np.int64)
        rows = np.asarray(block.vertices, dtype=np.int64)
        valid = np.asarray(block.vertex_valid, dtype=np.bool_)
        if declared.shape != rows.shape or valid.shape != rows.shape:
            return False
        for row in range(block.cell_count):
            selected = tuple(sorted(int(value) for value in declared[row] if value >= 0))
            actual = tuple(
                sorted(int(target_ids[int(vertex)]) for vertex in rows[row, valid[row]])
            )
            if selected != actual:
                return False
    return True


def _ratio(value: Fraction) -> tuple[int, int]:
    return value.numerator, value.denominator


def certify_periodic_restriction_partition(
    state: EmbeddingState,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    validity: CellValidityCertificate | None,
    premise: PeriodicRestrictionPremise,
    facet_trace_certificate_id: str | None,
) -> PeriodicRestrictionPartitionEvidence | None:
    """Return exact partition evidence only after every convex and quotient premise."""
    source_mesh, source_geometry = premise.source_mesh, premise.source_geometry
    source_validity, source_embedding = premise.source_validity, premise.source_embedding
    topology, source_topology = mesh.periodic_topology, source_mesh.periodic_topology
    if (
        validity is None
        or not validity.all_certified
        or not source_validity.all_certified
        or not isinstance(topology, PeriodicMeshTopology)
        or not isinstance(source_topology, PeriodicMeshTopology)
        or mesh.topological_dimension != 3
        or mesh.ambient_dimension != 3
        or source_mesh.topological_dimension != 3
        or source_mesh.ambient_dimension != 3
        or state.findings
        or facet_trace_certificate_id is None
        or not facet_trace_certificate_id
        or identification_id(topology.cell) != identification_id(source_topology.cell)
        or not _periodic_facets_pair(mesh, topology)
    ):
        return None
    trace_id = canonical_identifier(
        facet_trace_certificate_id, "facet_trace_certificate_id"
    )
    validity.require_current(geometry, mesh=mesh)
    source_validity.require_current(source_geometry, mesh=source_mesh)
    if source_embedding.status != "certified" or source_embedding.findings:
        return None
    from ._global_embedding import GlobalEmbeddingCertificate

    current_embedding = GlobalEmbeddingCertificate.from_stored_fields(source_embedding)
    current_embedding.binding.require(source_mesh, source_geometry)
    if (
        current_embedding.status != "certified"
        or current_embedding.findings
        or current_embedding.validity_certificate_id != source_validity.certificate_id
        or current_embedding.periodic is None
        or current_embedding.periodic.periodic_topology_id
        != source_topology.periodic_topology_id
    ):
        return None
    relation = _relation(premise, mesh, geometry)
    if relation is None:
        return None
    inverse, declared = relation
    source_faces = _cell_face_rows(source_mesh)
    target_faces = _cell_face_rows(mesh)
    source_points = _exact_vertex_points(source_mesh, source_geometry)
    target_points = _exact_vertex_points(mesh, geometry)
    if (
        source_faces is None
        or target_faces is None
        or source_points is None
        or target_points is None
    ):
        return None
    child_faces, parent_faces = (
        (source_faces, target_faces) if inverse else (target_faces, source_faces)
    )
    child_points, parent_points = (
        (source_points, target_points) if inverse else (target_points, source_points)
    )
    children_ids = {identifier for identifier, _ in child_faces}
    parent_ids = {identifier for identifier, _ in parent_faces}
    if set(declared) != children_ids or not set(declared.values()) <= parent_ids:
        return None
    if len(declared) != len(children_ids):
        return None
    parents: dict[int, _ConvexCell] = {}
    for identifier, faces in parent_faces:
        parent = _convex_cell(identifier, faces, parent_points)
        if parent is None:
            return None
        parents[identifier] = parent
    proved_children: list[tuple[int, _ConvexCell]] = []
    proved_child_cells: dict[int, _ConvexCell] = {}
    for identifier, faces in child_faces:
        child = _convex_cell(identifier, faces, child_points)
        if child is None:
            return None
        proved_child_cells[identifier] = child
        proved_children.append((declared[identifier], child))
    candidate_planes = _restriction_planes(parents, tuple(proved_children))
    if candidate_planes is None:
        return None
    partition = _parent_partition(parents, tuple(proved_children), candidate_planes)
    if partition is None:
        return None
    planes, child_rows = partition
    if inverse and (
        not _inverse_vertex_selection(
            source_mesh,
            source_geometry,
            mesh,
            geometry,
            source_points,
            target_points,
        )
        or not _inverse_boundary_subcomplex(
            source_faces,
            target_faces,
            source_points,
            target_points,
            declared,
        )
    ):
        return None

    def cell_plane_records(cells: dict[int, _ConvexCell]) -> tuple[_CellPlaneRecord, ...]:
        return tuple(
            (
                identifier,
                tuple(
                    tuple(_ratio(value) for value in _plane_key(plane))
                    for plane in cell.planes
                ),
                tuple(
                    tuple(tuple(_ratio(value) for value in point) for point in face)
                    for face in cell.boundary_faces
                ),
            )
            for identifier, cell in sorted(cells.items())
        )

    all_source_cells = proved_child_cells if inverse else parents
    all_target_cells = parents if inverse else proved_child_cells
    boundary_subdivision_id = canonical_fingerprint(
        {
            "kind": "periodic-restriction-convex-boundary-subdivisions",
            "mesh": mesh.mesh_id,
            "direction": "coarsen" if inverse else "refine",
            "source_mesh": source_mesh.mesh_id,
            "source_facets": cell_plane_records(all_source_cells),
            "target_facets": cell_plane_records(all_target_cells),
            "inverse_signed_boundary_cancellation": inverse,
            "restriction_planes": tuple(
                tuple(_ratio(value) for value in plane) for plane in planes
            ),
        }
    )
    parent_cells = tuple(sorted(parents))
    from ._embedding_state import PeriodicRestrictionPartitionEvidence

    evidence = PeriodicRestrictionPartitionEvidence(
        mesh.mesh_id,
        validity.geometry_id,
        topology.periodic_topology_id,
        validity.certificate_id,
        trace_id,
        boundary_subdivision_id,
        identification_id(topology.cell),
        source_mesh.mesh_id,
        cell_geometry_id(source_geometry),
        source_validity.certificate_id,
        current_embedding.certificate_id,
        tuple(tuple(_ratio(value) for value in plane) for plane in planes),
        tuple(row[0] for row in child_rows),
        tuple(row[1] for row in child_rows),
        tuple(row[2] for row in child_rows),
        tuple(_ratio(row[3]) for row in child_rows),
        parent_cells,
        tuple(_ratio(parents[identifier].volume) for identifier in parent_cells),
        direction="coarsen" if inverse else "refine",
    )
    state.add_check("periodic_restriction_source_embedding")
    state.add_check("periodic_restriction_parent_partition")
    state.add_check("periodic_restriction_boundary_subdivision")
    return evidence


__all__ = ["PeriodicRestrictionPremise", "certify_periodic_restriction_partition"]
