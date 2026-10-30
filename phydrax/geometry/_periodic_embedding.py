#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact affine-translation quotient contacts for periodic simplex meshes."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .._bvh import bvh_overlap_pair_blocks, prepare_bvh
from .._fingerprint import canonical_fingerprint
from ..discretization._cell_geometry import CellGeometrySpec
from ..discretization._cell_geometry_validity import (
    cell_geometry_id,
    CellValidityCertificate,
)
from ..discretization._cell_mesh import CellMesh
from ..discretization._exact_algebraic_geometry import (
    coefficient_endpoint,
    ExactScalar,
    rational_coefficient,
)
from ..discretization._periodic_cell import PeriodicCell
from ..discretization._periodic_exact import (
    exact_periodic_element,
    identification_orders,
    PeriodicIsometryGroup,
    reduce_periodic_exponents,
)
from ..discretization._periodic_topology import PeriodicMeshTopology
from ..discretization._topology import CellComplexTopology
from ._embedding_cells import EmbeddingCell
from ._embedding_state import (
    MeshCertificateFinding,
    PeriodicMappedTorusDegreeEvidence,
)
from ._exact_polyhedral_geometry import (
    _sign,
    contact_is_declared,
    determinant3,
    Point,
    Point2,
    Point3,
    tetrahedron_contact_points,
    triangle_contact_points2,
)
from ._periodic_image_frame import prepare_periodic_image_frame
from ._planar_chain_embedding import triangulate_planar_polygon


type PeriodicVertexKey = tuple[int, tuple[int, ...]]
type _PeriodicTetrahedron = tuple[
    tuple[Point3, Point3, Point3, Point3],
    tuple[PeriodicVertexKey, PeriodicVertexKey, PeriodicVertexKey, PeriodicVertexKey],
]

type _LinkGerm = tuple[int, tuple[int, ...]]
type _LinkEdge = tuple[_LinkGerm, _LinkGerm]


@dataclass(frozen=True, slots=True)
class PeriodicContactResult:
    status: Literal["certified", "violated", "unresolved"]
    findings: tuple[tuple[str, tuple[int, ...], str], ...]
    quotient_boundary: Literal["closed", "with_boundary"] | None = None


@dataclass(frozen=True, slots=True)
class PeriodicMappedDegreeResult:
    applicable: bool
    checks: tuple[str, ...] = ()
    findings: tuple[MeshCertificateFinding, ...] = ()
    evidence: PeriodicMappedTorusDegreeEvidence | None = None


def _contact_disposition(check: str, /) -> str:
    if check in (
        "periodic_source_vertex_trace",
        "periodic_source_trace_mismatch",
        "periodic_cell_lift_collision",
        "periodic_cell_contact",
    ):
        return "violated"
    return "unresolved"


def _boundary_cycles_oriented(
    quotient: CellComplexTopology,
    edge_boundary: NDArray[np.bool_],
    edge_orientations: NDArray[np.float64],
    /,
) -> Literal["with_boundary"] | None:
    """Require a 2D quotient boundary to be a disjoint union of oriented cycles."""
    vertex_count = quotient.entity_sets[0].count
    vertex_boundary = np.asarray(
        quotient.entity_sets[0].subset("boundary").mask, dtype=np.bool_
    )
    incidence = quotient.incidences[0]
    relation = incidence.relation
    valid = np.asarray(relation.valid, dtype=np.bool_)
    vertex_indices = np.asarray(relation.source_indices, dtype=np.int64)[valid]
    edge_indices = np.asarray(relation.target_indices, dtype=np.int64)[valid]
    signs = np.asarray(incidence.signs, dtype=np.float64)[valid]
    on_boundary = edge_boundary[edge_indices]
    boundary_counts = np.bincount(vertex_indices[on_boundary], minlength=vertex_count)
    if vertex_boundary.shape != (vertex_count,) or not np.array_equal(
        vertex_boundary, boundary_counts > 0
    ):
        return None
    for edge in np.flatnonzero(edge_boundary):
        endpoints = vertex_indices[on_boundary & (edge_indices == edge)]
        if endpoints.size != 2 or endpoints[0] == endpoints[1]:
            return None
    for vertex in np.flatnonzero(vertex_boundary):
        selected = on_boundary & (vertex_indices == vertex)
        coefficients = signs[selected] * edge_orientations[edge_indices[selected]]
        if (
            coefficients.size != 2
            or float(np.sum(coefficients)) != 0.0
            or np.any(np.abs(coefficients) != 1.0)
        ):
            return None
    return "with_boundary"


def _quotient_faces_pair_oriented(
    topology: PeriodicMeshTopology, /
) -> Literal["closed", "with_boundary"] | None:
    """Classify closed and manifold-boundary oriented 2D/3D periodic quotients."""
    quotient = topology.quotient
    dimension = topology.topological_dimension
    if dimension not in (2, 3) or len(quotient.entity_sets) < dimension + 1:
        return None
    face_count = quotient.entity_sets[dimension - 1].count
    face_boundary = np.asarray(
        quotient.entity_sets[dimension - 1].subset("boundary").mask, dtype=np.bool_
    )
    incidence = quotient.incidences[-1]
    relation = incidence.relation
    valid = np.asarray(relation.valid, dtype=np.bool_)
    face_indices = np.asarray(relation.source_indices, dtype=np.int64)[valid]
    cell_indices = np.asarray(relation.target_indices, dtype=np.int64)[valid]
    quotient_offsets = np.asarray(topology.quotient_offsets, dtype=np.int64)
    lifted_offsets = np.asarray(topology.lifted_offsets, dtype=np.int64)
    representatives = np.asarray(topology.representatives, dtype=np.int64)
    cell_representatives = representatives[
        quotient_offsets[dimension] : quotient_offsets[dimension + 1]
    ]
    orientations = np.asarray(topology.orbit_orientations, dtype=np.int8)
    cell_orientations = orientations[lifted_offsets[dimension] + cell_representatives]
    # Quotient cells use canonical vertex order, not necessarily positive
    # physical orientation. Form the boundary of the physical cell chain.
    face_signs = (
        np.asarray(incidence.signs, dtype=np.float64)[valid]
        * cell_orientations[cell_indices]
    )
    if face_boundary.shape != (face_count,):
        return None
    face_orientations = np.zeros((face_count,), dtype=np.float64)
    for face in range(face_count):
        selected = face_signs[face_indices == face]
        if selected.size == 1:
            if not face_boundary[face] or abs(float(selected[0])) != 1.0:
                return None
            face_orientations[face] = selected[0]
        elif (
            selected.size != 2
            or face_boundary[face]
            or float(np.sum(selected)) != 0.0
            or np.any(np.abs(selected) != 1.0)
        ):
            return None
    if not np.any(face_boundary):
        return "closed"
    if dimension == 2:
        return _boundary_cycles_oriented(quotient, face_boundary, face_orientations)

    edge_count = quotient.entity_sets[1].count
    edge_boundary = np.asarray(
        quotient.entity_sets[1].subset("boundary").mask, dtype=np.bool_
    )
    edge_incidence = quotient.incidences[1]
    edge_relation = edge_incidence.relation
    edge_valid = np.asarray(edge_relation.valid, dtype=np.bool_)
    edge_sources = np.asarray(edge_relation.source_indices, dtype=np.int64)[edge_valid]
    edge_faces = np.asarray(edge_relation.target_indices, dtype=np.int64)[edge_valid]
    edge_signs = np.asarray(edge_incidence.signs, dtype=np.float64)[edge_valid]
    on_boundary = face_boundary[edge_faces]
    boundary_edge_counts = np.bincount(edge_sources[on_boundary], minlength=edge_count)
    if edge_boundary.shape != (edge_count,) or not np.array_equal(
        edge_boundary, boundary_edge_counts > 0
    ):
        return None
    for edge in np.flatnonzero(edge_boundary):
        selected = on_boundary & (edge_sources == edge)
        boundary_coefficients = (
            edge_signs[selected] * face_orientations[edge_faces[selected]]
        )
        if (
            boundary_coefficients.size != 2
            or float(np.sum(boundary_coefficients)) != 0.0
            or np.any(np.abs(boundary_coefficients) != 1.0)
        ):
            return None

    vertex_count = quotient.entity_sets[0].count
    vertex_boundary = np.asarray(
        quotient.entity_sets[0].subset("boundary").mask, dtype=np.bool_
    )
    vertex_incidence = quotient.incidences[0]
    vertex_relation = vertex_incidence.relation
    vertex_valid = np.asarray(vertex_relation.valid, dtype=np.bool_)
    vertex_indices = np.asarray(vertex_relation.source_indices, dtype=np.int64)[
        vertex_valid
    ]
    edge_indices = np.asarray(vertex_relation.target_indices, dtype=np.int64)[
        vertex_valid
    ]
    vertex_signs = np.asarray(vertex_incidence.signs, dtype=np.float64)[vertex_valid]
    on_boundary_edge = edge_boundary[edge_indices]
    boundary_vertex_counts = np.bincount(
        vertex_indices[on_boundary_edge], minlength=vertex_count
    )
    if vertex_boundary.shape != (vertex_count,) or not np.array_equal(
        vertex_boundary, boundary_vertex_counts > 0
    ):
        return None
    edge_vertices: list[list[int]] = [[] for _ in range(edge_count)]
    for vertex, edge in zip(
        vertex_indices[on_boundary_edge], edge_indices[on_boundary_edge], strict=True
    ):
        edge_vertices[int(edge)].append(int(vertex))
    for edge in np.flatnonzero(edge_boundary):
        selected = vertex_signs[on_boundary_edge & (edge_indices == edge)]
        endpoints = edge_vertices[int(edge)]
        if (
            len(endpoints) != 2
            or endpoints[0] == endpoints[1]
            or selected.size != 2
            or float(np.sum(selected)) != 0.0
            or np.any(np.abs(selected) != 1.0)
        ):
            return None

    boundary_vertex_links: dict[int, dict[int, list[int]]] = {}
    for face in np.flatnonzero(face_boundary):
        face_edges = tuple(edge_sources[on_boundary & (edge_faces == face)])
        if len(face_edges) != 3:
            return None
        face_vertices = {
            vertex for edge in face_edges for vertex in edge_vertices[int(edge)]
        }
        if len(face_vertices) != 3:
            return None
        for vertex in face_vertices:
            adjacent = tuple(
                int(edge)
                for edge in face_edges
                if int(vertex) in edge_vertices[int(edge)]
            )
            if len(adjacent) != 2:
                return None
            link = boundary_vertex_links.setdefault(int(vertex), {})
            link.setdefault(adjacent[0], []).append(adjacent[1])
            link.setdefault(adjacent[1], []).append(adjacent[0])
    for vertex in np.flatnonzero(vertex_boundary):
        link = boundary_vertex_links.get(int(vertex))
        expected_edges = {
            edge
            for edge in np.flatnonzero(edge_boundary)
            if int(vertex) in edge_vertices[int(edge)]
        }
        if (
            link is None
            or set(link) != expected_edges
            or any(len(neighbors) != 2 for neighbors in link.values())
        ):
            return None
        pending = [next(iter(link))]
        visited: set[int] = set()
        while pending:
            edge = pending.pop()
            if edge in visited:
                continue
            visited.add(edge)
            pending.extend(neighbor for neighbor in link[edge] if neighbor not in visited)
        if visited != expected_edges:
            return None
    return "with_boundary"


def _point3(point: tuple[ExactScalar, ...], /) -> Point3:
    if len(point) != 3:
        raise ValueError(
            "Periodic affine contact requires three-dimensional source points."
        )
    return point[0], point[1], point[2]


def _point(point: tuple[ExactScalar, ...], dimension: int, /) -> Point:
    if len(point) != dimension:
        raise ValueError(
            "Periodic affine contact source points must match the mesh dimension."
        )
    if dimension == 2:
        return point[0], point[1]
    return _point3(point)


def _point2(point: Point, /) -> Point2:
    if len(point) != 2:
        raise ValueError("Planar periodic contact requires two-dimensional points.")
    return point[0], point[1]


def _simplex_contact_points(
    first: tuple[Point, ...], second: tuple[Point, ...], /
) -> tuple[Point, ...]:
    if len(first) == 4 and len(second) == 4:
        return tetrahedron_contact_points(
            tuple(_point3(point) for point in first),
            tuple(_point3(point) for point in second),
        )
    if len(first) == 3 and len(second) == 3:
        left = tuple(_point2(point) for point in first)
        right = tuple(_point2(point) for point in second)
        return triangle_contact_points2(
            (left[0], left[1], left[2]), (right[0], right[1], right[2])
        )
    raise ValueError("Periodic contact pieces must be matching affine simplices.")


def _image_winding(
    winding: tuple[int, ...],
    exponent: tuple[int, ...],
    orders: tuple[int, ...],
    fixed: tuple[bool, ...],
    /,
) -> tuple[int, ...]:
    """Compose an action with a lifted winding modulo the vertex stabilizer."""
    return tuple(
        0 if stationary else (value + step) % order if order > 0 else value + step
        for value, step, order, stationary in zip(
            winding, exponent, orders, fixed, strict=True
        )
    )


def _simplex_bounds(
    points: tuple[Point, ...], /
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    axes = range(len(points[0]))
    return (
        np.asarray(
            [
                coefficient_endpoint(min(point[axis] for point in points), upper=False)
                for axis in axes
            ],
            dtype=np.float64,
        ),
        np.asarray(
            [
                coefficient_endpoint(max(point[axis] for point in points), upper=True)
                for axis in axes
            ],
            dtype=np.float64,
        ),
    )


def _lifted_vertex_keys(
    cell: EmbeddingCell,
    mesh_vertex_ids: NDArray[np.int64],
    representatives: NDArray[np.int64],
    shifts: NDArray[np.int64],
    /,
) -> tuple[PeriodicVertexKey, ...]:
    return tuple(
        (
            int(mesh_vertex_ids[representatives[index]]),
            tuple(int(value) for value in shifts[index]),
        )
        for index in cell.vertex_indices
    )


def _lifted_cell_center_keys(
    mesh: CellMesh, topology: PeriodicMeshTopology, /
) -> dict[int, PeriodicVertexKey]:
    identifiers = np.asarray(mesh.entity_set(3).entity_ids, dtype=np.int64)
    stored = np.asarray(topology.lifted_entity_ids[3], dtype=np.int64)
    lifted_offsets = np.asarray(topology.lifted_offsets, dtype=np.int64)
    quotient_offsets = np.asarray(topology.quotient_offsets, dtype=np.int64)
    lifted_begin, lifted_end = map(int, lifted_offsets[3:5])
    quotient_begin, quotient_end = map(int, quotient_offsets[3:5])
    orbits = (
        np.asarray(topology.orbit_indices, dtype=np.int64)[lifted_begin:lifted_end]
        - quotient_begin
    )
    windings = np.asarray(topology.orbit_shifts, dtype=np.int64)[lifted_begin:lifted_end]
    quotient_ids = np.asarray(topology.quotient.entity_sets[3].entity_ids, dtype=np.int64)
    if (
        not np.array_equal(identifiers, stored)
        or identifiers.shape != (lifted_end - lifted_begin,)
        or quotient_ids.shape != (quotient_end - quotient_begin,)
        or orbits.shape != identifiers.shape
        or windings.shape != (identifiers.size, topology.cell.rank)
        or np.any((orbits < 0) | (orbits >= quotient_ids.size))
    ):
        raise ValueError("Periodic lifted-cell identity tables are malformed.")
    return {
        int(identifier): (
            -int(quotient_ids[int(orbits[row])]) - 1,
            tuple(int(value) for value in windings[row]),
        )
        for row, identifier in enumerate(identifiers)
    }


def _canonical_face_route(
    route: tuple[int, ...], vertex_keys: tuple[PeriodicVertexKey, ...], /
) -> tuple[int, ...]:
    windings = tuple(vertex_keys[index][1] for index in route)
    anchor = min(windings)
    normalized = {
        index: (
            vertex_keys[index][0],
            tuple(value - offset for value, offset in zip(winding, anchor, strict=True)),
        )
        for index, winding in zip(route, windings, strict=True)
    }
    candidates = []
    for direction in (route, tuple(reversed(route))):
        for shift in range(len(route)):
            order = direction[shift:] + direction[:shift]
            candidates.append((tuple(normalized[index] for index in order), order))
    return min(candidates)[1]


def _convex_polyhedron_tetrahedra(
    cell: EmbeddingCell,
    vertex_keys: tuple[PeriodicVertexKey, ...],
    center_key: PeriodicVertexKey,
    /,
) -> tuple[_PeriodicTetrahedron, ...]:
    points = tuple(_point3(value) for value in cell.corner_images)
    if (
        cell.kind != "polyhedron"
        or cell.dimension != 3
        or len(points) < 4
        or len(vertex_keys) != len(points)
        or len(set(vertex_keys)) != len(vertex_keys)
        or not cell.facet_routes
    ):
        raise ValueError(
            "Periodic polyhedron has malformed exact corner or face incidence."
        )
    center_values = tuple(
        sum((point[axis] for point in points), Fraction(0)) / len(points)
        for axis in range(3)
    )
    center: Point3 = (center_values[0], center_values[1], center_values[2])
    edge_counts: dict[tuple[int, int], int] = {}
    used_vertices: set[int] = set()
    tetrahedra: list[_PeriodicTetrahedron] = []
    for route in cell.facet_routes:
        if (
            len(route) < 3
            or len(set(route)) != len(route)
            or any(index < 0 or index >= len(points) for index in route)
        ):
            raise ValueError("Periodic polyhedron face incidence is malformed.")
        canonical_route = _canonical_face_route(route, vertex_keys)
        used_vertices.update(canonical_route)
        for index, vertex in enumerate(canonical_route):
            following = canonical_route[(index + 1) % len(canonical_route)]
            edge = (min(vertex, following), max(vertex, following))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
        face_points = tuple(points[index] for index in canonical_route)
        triangles = triangulate_planar_polygon(face_points)
        if not triangles:
            raise ValueError("Periodic polyhedron face has no exact triangulation.")
        first_triangle = triangles[0]
        plane = tuple(face_points[index] for index in first_triangle)
        first, second, third = plane
        if any(determinant3(first, second, third, point) != 0 for point in face_points):
            raise ValueError("Periodic polyhedron face is not exactly planar.")
        center_side = _sign(determinant3(first, second, third, center))
        if center_side == 0 or any(
            _sign(determinant3(first, second, third, point)) not in (0, center_side)
            for point in points
        ):
            raise ValueError(
                "Periodic polyhedron is not strictly convex and full-dimensional."
            )
        for triangle in triangles:
            triangle_points = tuple(face_points[index] for index in triangle)
            triangle_keys = tuple(
                vertex_keys[canonical_route[index]] for index in triangle
            )
            tetrahedron: tuple[Point3, Point3, Point3, Point3] = (
                center,
                triangle_points[0],
                triangle_points[1],
                triangle_points[2],
            )
            tetrahedron_keys: tuple[
                PeriodicVertexKey,
                PeriodicVertexKey,
                PeriodicVertexKey,
                PeriodicVertexKey,
            ] = (
                center_key,
                triangle_keys[0],
                triangle_keys[1],
                triangle_keys[2],
            )
            orientation = _sign(determinant3(*tetrahedron))
            if orientation == 0:
                raise ValueError(
                    "Periodic polyhedron has a degenerate exact face tetrahedron."
                )
            if orientation < 0:
                tetrahedron = (
                    tetrahedron[0],
                    tetrahedron[1],
                    tetrahedron[3],
                    tetrahedron[2],
                )
                tetrahedron_keys = (
                    tetrahedron_keys[0],
                    tetrahedron_keys[1],
                    tetrahedron_keys[3],
                    tetrahedron_keys[2],
                )
            tetrahedra.append((tetrahedron, tetrahedron_keys))
    if used_vertices != set(range(len(points))) or any(
        count != 2 for count in edge_counts.values()
    ):
        raise ValueError("Periodic polyhedron faces do not form one closed vertex shell.")
    return tuple(tetrahedra)


def _periodic_cell_tetrahedra(
    mesh: CellMesh,
    cells: tuple[EmbeddingCell, ...],
    topology: PeriodicMeshTopology,
    lifted_keys: tuple[tuple[PeriodicVertexKey, ...], ...],
    /,
) -> tuple[tuple[_PeriodicTetrahedron, ...], ...]:
    center_keys = (
        _lifted_cell_center_keys(mesh, topology)
        if any(cell.kind == "polyhedron" for cell in cells)
        else {}
    )
    result = []
    for cell, vertex_keys in zip(cells, lifted_keys, strict=True):
        if cell.kind in ("tetrahedron", "simplex:3"):
            points = tuple(_point3(value) for value in cell.corner_images)
            if len(points) != 4 or len(vertex_keys) != 4:
                raise ValueError("Periodic tetrahedron corner incidence is malformed.")
            tetrahedron: tuple[Point3, Point3, Point3, Point3] = (
                points[0],
                points[1],
                points[2],
                points[3],
            )
            tetrahedron_keys = (
                vertex_keys[0],
                vertex_keys[1],
                vertex_keys[2],
                vertex_keys[3],
            )
            result.append(((tetrahedron, tetrahedron_keys),))
            continue
        if cell.kind != "polyhedron":
            raise ValueError(f"Unsupported periodic volume cell kind {cell.kind!r}.")
        center_key = center_keys.get(cell.global_id)
        if center_key is None:
            raise ValueError(
                "Periodic polyhedron has no quotient-bound interior point identity."
            )
        result.append(_convex_polyhedron_tetrahedra(cell, vertex_keys, center_key))
    return tuple(result)


def _cycle_graph(vertices: set[_LinkGerm], edges: tuple[_LinkEdge, ...], /) -> bool:
    if len(vertices) < 3 or len(set(edges)) != len(edges):
        return False
    neighbors: dict[_LinkGerm, set[_LinkGerm]] = {vertex: set() for vertex in vertices}
    for first, second in edges:
        if first == second or first not in neighbors or second not in neighbors:
            return False
        neighbors[first].add(second)
        neighbors[second].add(first)
    if any(len(row) != 2 for row in neighbors.values()):
        return False
    pending = [next(iter(vertices))]
    visited: set[_LinkGerm] = set()
    while pending:
        vertex = pending.pop()
        if vertex not in visited:
            visited.add(vertex)
            pending.extend(neighbors[vertex] - visited)
    return visited == vertices


def _periodic_tetrahedron_manifold_id(
    mesh: CellMesh,
    cells: tuple[EmbeddingCell, ...],
    topology: PeriodicMeshTopology,
    /,
) -> str | None:
    if len(topology.quotient.entity_sets) != 4 or (
        topology.quotient.entity_sets[3].count != len(cells)
    ):
        return None
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    roots = np.asarray(topology.vertex_representatives, dtype=np.int64)
    shifts = np.asarray(topology.vertex_shifts, dtype=np.int64)
    if roots.shape != (len(vertex_ids),) or shifts.shape != (
        len(vertex_ids),
        topology.cell.rank,
    ):
        return None
    links: dict[int, list[tuple[_LinkGerm, _LinkGerm, _LinkGerm]]] = {}
    quotient_neighbors: dict[int, set[int]] = {}
    used_roots: set[int] = set()
    for cell in cells:
        if cell.kind not in ("tetrahedron", "simplex:3") or len(cell.vertex_indices) != 4:
            return None
        vertices = cell.vertex_indices
        for slot, center in enumerate(vertices):
            root_id = int(vertex_ids[int(roots[center])])
            germs = tuple(
                (
                    int(vertex_ids[int(roots[vertex])]),
                    tuple(int(value) for value in shifts[vertex] - shifts[center]),
                )
                for offset, vertex in enumerate(vertices)
                if offset != slot
            )
            if len(set(germs)) != 3:
                return None
            links.setdefault(root_id, []).append((germs[0], germs[1], germs[2]))
            neighbors = quotient_neighbors.setdefault(root_id, set())
            neighbors.update(germ[0] for germ in germs)
            used_roots.add(root_id)
    expected_roots = {int(vertex_ids[int(root)]) for root in np.unique(roots)}
    if used_roots != expected_roots:
        return None
    link_records = []
    for root_id, triangles in sorted(links.items()):
        triangle_keys = tuple(tuple(sorted(row)) for row in triangles)
        if len(set(triangle_keys)) != len(triangle_keys):
            return None
        link_vertices = {vertex for row in triangles for vertex in row}
        edge_owners: dict[_LinkEdge, list[int]] = {}
        for triangle_index, triangle in enumerate(triangles):
            for left, right in combinations(triangle, 2):
                edge: _LinkEdge = (left, right) if left <= right else (right, left)
                owners = edge_owners.get(edge)
                if owners is None:
                    edge_owners[edge] = [triangle_index]
                else:
                    owners.append(triangle_index)
        if any(len(owners) != 2 for owners in edge_owners.values()):
            return None
        if len(link_vertices) - len(edge_owners) + len(triangles) != 2:
            return None
        triangle_neighbors: list[set[int]] = [set() for _ in range(len(triangles))]
        for owners in edge_owners.values():
            first, second = owners
            triangle_neighbors[first].add(second)
            triangle_neighbors[second].add(first)
        pending_triangles = [0]
        visited_triangles: set[int] = set()
        while pending_triangles:
            triangle = pending_triangles.pop()
            if triangle not in visited_triangles:
                visited_triangles.add(triangle)
                pending_triangles.extend(triangle_neighbors[triangle] - visited_triangles)
        if len(visited_triangles) != len(triangles):
            return None
        for vertex in link_vertices:
            link_edges: list[_LinkEdge] = []
            link_vertices_at_vertex: set[_LinkGerm] = set()
            for triangle in triangles:
                if vertex not in triangle:
                    continue
                other = tuple(value for value in triangle if value != vertex)
                if len(other) != 2:
                    return None
                left, right = other
                link_vertices_at_vertex.update((left, right))
                link_edges.append((left, right) if left <= right else (right, left))
            if not _cycle_graph(link_vertices_at_vertex, tuple(link_edges)):
                return None
        link_records.append(
            (
                root_id,
                tuple(sorted(triangle_keys)),
                tuple(sorted(edge_owners)),
            )
        )
    if not links:
        return None
    pending_roots = [next(iter(links))]
    visited_roots: set[int] = set()
    while pending_roots:
        root = pending_roots.pop()
        if root not in visited_roots:
            visited_roots.add(root)
            pending_roots.extend(quotient_neighbors[root] - visited_roots)
    if visited_roots != set(links):
        return None
    return canonical_fingerprint(
        {
            "kind": "connected-closed-periodic-tetrahedron-links",
            "topology": topology.periodic_topology_id,
            "vertex_links": tuple(link_records),
        }
    )


def certify_periodic_mapped_torus_degree(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    validity: CellValidityCertificate,
    cells: tuple[EmbeddingCell, ...],
    topology: PeriodicMeshTopology,
    binding_id: str,
    local_injectivity_id: str,
    facet_trace_certificate_id: str,
    /,
) -> PeriodicMappedDegreeResult:
    """Transfer mapped local injectivity to a torus embedding by exact degree."""
    period = topology.cell
    if (
        not isinstance(period, PeriodicCell)
        or mesh.topological_dimension != 3
        or mesh.ambient_dimension != 3
        or period.rank != 3
        or not period.fully_periodic
        or not cells
        or any(
            cell.kind not in ("tetrahedron", "simplex:3")
            or cell.dimension != 3
            or len(cell.corner_images) != 4
            for cell in cells
        )
    ):
        return PeriodicMappedDegreeResult(False)
    checks = (
        "periodic_mapped_torus_manifold",
        "periodic_mapped_torus_trace",
        "periodic_mapped_torus_orientation",
        "periodic_mapped_torus_degree_one",
    )
    if (
        not validity.all_certified
        or validity.geometry_id != cell_geometry_id(geometry)
        or validity.mesh_id != mesh.mesh_id
        or not binding_id
        or not local_injectivity_id
        or not facet_trace_certificate_id
    ):
        return PeriodicMappedDegreeResult(
            True,
            checks,
            (
                MeshCertificateFinding(
                    "periodic_mapped_torus_trace",
                    "unresolved",
                    "mesh",
                    detail="The mapped degree proof is not bound to complete current local and facet traces.",
                ),
            ),
        )
    try:
        rebuilt = CellMesh.from_stored_fields(mesh)
    except (ArithmeticError, IndexError, KeyError, TypeError, ValueError) as error:
        return PeriodicMappedDegreeResult(
            True,
            checks,
            (
                MeshCertificateFinding(
                    "periodic_mapped_torus_manifold",
                    "unresolved",
                    "mesh",
                    detail=f"Current quotient reconstruction failed: {error}",
                ),
            ),
        )
    if (
        rebuilt.periodic_topology is None
        or rebuilt.periodic_topology.periodic_topology_id != topology.periodic_topology_id
        or _quotient_faces_pair_oriented(topology) != "closed"
    ):
        return PeriodicMappedDegreeResult(
            True,
            checks,
            (
                MeshCertificateFinding(
                    "periodic_mapped_torus_manifold",
                    "violated",
                    "mesh",
                    detail="Mapped torus degree requires a reconstructed closed oriented quotient.",
                ),
            ),
        )
    manifold_id = _periodic_tetrahedron_manifold_id(mesh, cells, topology)
    if manifold_id is None:
        return PeriodicMappedDegreeResult(
            True,
            checks,
            (
                MeshCertificateFinding(
                    "periodic_mapped_torus_manifold",
                    "violated",
                    "mesh",
                    detail="Periodic tetrahedron vertex links do not form connected 2-spheres.",
                ),
            ),
        )
    raw_lattice = np.asarray(period.vectors, dtype=np.float64)
    lattice = tuple(tuple(Fraction(float(value)) for value in row) for row in raw_lattice)
    lattice_determinant = (
        lattice[0][0] * (lattice[1][1] * lattice[2][2] - lattice[1][2] * lattice[2][1])
        - lattice[0][1] * (lattice[1][0] * lattice[2][2] - lattice[1][2] * lattice[2][0])
        + lattice[0][2] * (lattice[1][0] * lattice[2][1] - lattice[1][1] * lattice[2][0])
    )
    if not lattice_determinant:
        return PeriodicMappedDegreeResult(
            True,
            checks,
            (
                MeshCertificateFinding(
                    "periodic_mapped_torus_manifold",
                    "violated",
                    "mesh",
                    detail="The exact periodic lattice has zero covolume.",
                ),
            ),
        )
    pullback_six_volume: ExactScalar = Fraction(0)
    for cell in cells:
        corners = tuple(_point3(point) for point in cell.corner_images)
        if len(corners) != 4:
            return PeriodicMappedDegreeResult(
                True,
                checks,
                (
                    MeshCertificateFinding(
                        "periodic_mapped_torus_orientation",
                        "unresolved",
                        "cell",
                        (cell.global_id,),
                        detail="A mapped tetrahedron lacks four exact source corners.",
                    ),
                ),
            )
        pullback_six_volume += determinant3(*corners)
    lattice_volume = abs(lattice_determinant)
    if pullback_six_volume != 6 * lattice_volume:
        return PeriodicMappedDegreeResult(
            True,
            checks,
            (
                MeshCertificateFinding(
                    "periodic_mapped_torus_degree_one",
                    "violated",
                    "mesh",
                    detail="The exact mapped degree is not one lattice covolume.",
                ),
            ),
        )
    pullback_volume = rational_coefficient(
        pullback_six_volume / 6,
        role="periodic_mapped_torus.signed_pullback_volume",
    )
    local_id = canonical_fingerprint(
        {
            "kind": "periodic-mapped-local-injectivity",
            "binding": binding_id,
            "validity": validity.certificate_id,
            "local_proof": local_injectivity_id,
        }
    )
    evidence = PeriodicMappedTorusDegreeEvidence(
        mesh.mesh_id,
        validity.geometry_id,
        topology.periodic_topology_id,
        validity.certificate_id,
        binding_id,
        facet_trace_certificate_id,
        local_id,
        manifold_id,
        (pullback_volume.numerator, pullback_volume.denominator),
        (lattice_volume.numerator, lattice_volume.denominator),
    )
    return PeriodicMappedDegreeResult(True, checks, evidence=evidence)


def certify_periodic_affine_embedding(
    mesh: CellMesh,
    cells: tuple[EmbeddingCell, ...],
    topology: PeriodicMeshTopology,
    /,
) -> PeriodicContactResult:
    """Prove periodic cell injectivity, with closed and physical boundaries distinct.

    Exact image frames enumerate every lattice action whose source-vertex
    projections can meet. Each action is traversed through two BVHs; no
    cell-pair/image product is materialized. The route supports affine
    triangles in 2D and affine tetrahedra or convex vertex-defined polyhedra in
    3D, on closed quotients or oriented manifold subdomains. The identification
    may be a translational lattice of any rank up to the ambient dimension or a
    commuting proper isometry group (finite rotations and screws). Nonperiodic
    directions have no images; they are bounded by the quotient's physical
    boundary.
    """
    unresolved = (
        "periodic_quotient_unresolved",
        (),
        "Periodic quotient is outside the exact affine periodic simplex/convex-polyhedron route.",
    )
    period = topology.cell
    dimension = mesh.topological_dimension
    simplex_kinds = (
        ("tetrahedron", "simplex:3") if dimension == 3 else ("triangle", "simplex:2")
    )
    if (
        not isinstance(period, (PeriodicCell, PeriodicIsometryGroup))
        or dimension not in (2, 3)
        or mesh.ambient_dimension != dimension
        or period.ambient_dimension != dimension
        or len(cells) == 0
        or any(
            not (
                (cell.kind in simplex_kinds and cell.affine)
                or (dimension == 3 and cell.kind == "polyhedron" and cell.dimension == 3)
            )
            for cell in cells
        )
    ):
        return PeriodicContactResult("unresolved", (unresolved,))
    boundary_kind = _quotient_faces_pair_oriented(topology)
    if boundary_kind is None:
        finding = (
            "periodic_quotient_incidence_unresolved",
            (),
            "Periodic quotient facets do not define an oriented closed complex or manifold subdomain boundary.",
        )
        return PeriodicContactResult("unresolved", (finding,))

    orders = tuple(int(order) for order in identification_orders(period))

    try:
        representatives = np.asarray(topology.vertex_representatives, dtype=np.int64)
        shifts = np.asarray(topology.vertex_shifts, dtype=np.int64)
        fixed = np.asarray(topology.vertex_fixed_generators, dtype=np.bool_)
        vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
        source_by_index: dict[int, Point] = {}
        for cell in cells:
            for index, point in zip(cell.vertex_indices, cell.corner_images, strict=True):
                exact_point = _point(point, dimension)
                prior = source_by_index.get(index)
                if prior is not None and prior != exact_point:
                    finding = (
                        "periodic_source_vertex_trace",
                        (cell.global_id,),
                        f"Exact source map disagrees at lifted vertex index {index}.",
                    )
                    return PeriodicContactResult("violated", (finding,), boundary_kind)
                source_by_index[index] = exact_point
        for index, point in source_by_index.items():
            representative = int(representatives[index])
            representative_point = source_by_index.get(representative)
            if representative_point is None:
                finding = (
                    "periodic_source_representative_missing",
                    (int(vertex_ids[index]),),
                    "Exact source map does not cover the declared periodic representative.",
                )
                return PeriodicContactResult("unresolved", (finding,), boundary_kind)
            matrix = exact_periodic_element(period, shifts[index])
            expected = tuple(
                sum(
                    (
                        matrix[axis][column] * representative_point[column]
                        for column in range(dimension)
                    ),
                    matrix[axis][dimension],
                )
                for axis in range(dimension)
            )
            if point != expected:
                finding = (
                    "periodic_source_trace_mismatch",
                    (int(vertex_ids[index]),),
                    "Exact source corner does not match its declared orbit winding.",
                )
                return PeriodicContactResult("violated", (finding,), boundary_kind)
    except (ArithmeticError, IndexError, KeyError, TypeError, ValueError) as error:
        finding = ("periodic_source_identity_unresolved", (), str(error))
        return PeriodicContactResult("unresolved", (finding,), boundary_kind)

    mesh_vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    # Windings modulo each vertex stabilizer: a generator fixing an orbit does
    # not distinguish its images, matching the quotient topology convention.
    key_shifts = reduce_periodic_exponents(shifts, orders)
    key_shifts[fixed] = 0
    fixed_by_root = {
        int(mesh_vertex_ids[int(representatives[index])]): tuple(
            bool(value) for value in fixed[index]
        )
        for index in range(len(mesh_vertex_ids))
    }
    lifted_keys = tuple(
        _lifted_vertex_keys(cell, mesh_vertex_ids, representatives, key_shifts)
        for cell in cells
    )
    cell_pieces: tuple[
        tuple[tuple[tuple[Point, ...], tuple[PeriodicVertexKey, ...]], ...], ...
    ]
    if dimension == 3:
        try:
            cell_pieces = _periodic_cell_tetrahedra(mesh, cells, topology, lifted_keys)
        except (ValueError, ArithmeticError, IndexError) as error:
            polyhedron_ids = tuple(
                cell.global_id for cell in cells if cell.kind == "polyhedron"
            )
            finding = (
                "periodic_polyhedron_decomposition_unresolved",
                polyhedron_ids,
                str(error),
            )
            return PeriodicContactResult("unresolved", (finding,), boundary_kind)
    else:
        cell_pieces = tuple(
            ((tuple(_point(value, 2) for value in cell.corner_images), keys),)
            for cell, keys in zip(cells, lifted_keys, strict=True)
        )

    colliding_cells = tuple(
        cell.global_id
        for cell, keys in zip(cells, lifted_keys, strict=True)
        if len(set(keys)) != len(keys)
    )
    if colliding_cells:
        findings = tuple(
            (
                "periodic_cell_lift_collision",
                (identifier,),
                "A lifted cell contains repeated vertex identities at the same winding.",
            )
            for identifier in colliding_cells
        )
        return PeriodicContactResult("violated", findings, boundary_kind)

    piece_parent_rows: list[int] = []
    piece_points: list[tuple[Point, ...]] = []
    piece_keys: list[tuple[PeriodicVertexKey, ...]] = []
    piece_lowers: list[NDArray[np.float64]] = []
    piece_uppers: list[NDArray[np.float64]] = []
    for cell_row, pieces in enumerate(cell_pieces):
        for points, keys in pieces:
            lower, upper = _simplex_bounds(points)
            piece_parent_rows.append(cell_row)
            piece_points.append(points)
            piece_keys.append(keys)
            piece_lowers.append(lower)
            piece_uppers.append(upper)
    if not piece_points:
        finding = (
            "periodic_tetrahedralization_unresolved",
            (),
            "Periodic volume cells produced no exact contact simplices.",
        )
        return PeriodicContactResult("unresolved", (finding,), boundary_kind)
    parent_rows = np.asarray(piece_parent_rows, dtype=np.int64)
    frame_source_points = tuple(point for simplex in piece_points for point in simplex)
    frame_coordinates = np.asarray(
        tuple(tuple(float(value) for value in point) for point in frame_source_points),
        dtype=np.float64,
    )
    base_lower = np.asarray(np.stack(piece_lowers), dtype=np.float64)
    base_upper = np.asarray(np.stack(piece_uppers), dtype=np.float64)
    support_boxes = (
        ((np.min(base_lower, axis=0), np.max(base_upper, axis=0)),)
        if isinstance(period, PeriodicIsometryGroup)
        and any(order == 0 for order in period.orders)
        else None
    )
    try:
        image_frame = prepare_periodic_image_frame(
            frame_coordinates,
            np.arange(len(frame_source_points), dtype=np.int64),
            np.zeros((len(frame_source_points), period.rank), dtype=np.int64),
            period,
            source_support_boxes=support_boxes,
            exact_coordinates=frame_source_points,
        )
    except (ArithmeticError, OverflowError, TypeError, ValueError) as error:
        finding = ("periodic_image_frame_unresolved", (), str(error))
        return PeriodicContactResult("unresolved", (finding,), boundary_kind)
    base_tree = prepare_bvh(base_lower, base_upper)
    piece_offsets = np.cumsum(
        np.asarray((0, *(len(points) for points in piece_points)), dtype=np.int64)
    )
    no_fixed = (False,) * period.rank
    findings: list[tuple[str, tuple[int, ...], str]] = []

    for image in image_frame.images():
        if isinstance(image, tuple):
            exponent = tuple(int(value) for value in image)
            # Lattice images come in inverse pairs; one of each pair suffices.
            if any(exponent) and next(value for value in exponent if value) < 0:
                continue
        else:
            action_bank = image_frame.image_exponents
            if action_bank is None:
                raise TypeError(
                    "Periodic isometry image frames must retain action exponents."
                )
            exponent = tuple(int(value) for value in action_bank[int(image)])
        zero_shift = not any(exponent)
        if zero_shift:
            image_pieces = tuple(piece_points)
            shifted_tree = base_tree
        else:
            image_bank = image_frame.image_points(image)
            image_pieces = tuple(
                tuple(
                    _point(tuple(image_bank[row]), dimension)
                    for row in range(
                        int(piece_offsets[piece]), int(piece_offsets[piece + 1])
                    )
                )
                for piece in range(len(piece_points))
            )
            bounds = tuple(_simplex_bounds(points) for points in image_pieces)
            shifted_lower = np.stack(tuple(value[0] for value in bounds))
            shifted_upper = np.stack(tuple(value[1] for value in bounds))
            shifted_tree = prepare_bvh(shifted_lower, shifted_upper)
        for left_block, right_block in bvh_overlap_pair_blocks(
            base_tree, shifted_tree, include_touching=True
        ):
            for left, right in zip(
                left_block.tolist(), right_block.tolist(), strict=True
            ):
                first_piece, second_piece = int(left), int(right)
                first_row = int(parent_rows[first_piece])
                second_row = int(parent_rows[second_piece])
                if zero_shift and first_row >= second_row:
                    continue
                first_cell, second_cell = cells[first_row], cells[second_row]
                first_points = piece_points[first_piece]
                first_ids = piece_keys[first_piece]
                second_base_points = piece_points[second_piece]
                second_base_ids = piece_keys[second_piece]
                if zero_shift:
                    second_points, second_ids = second_base_points, second_base_ids
                else:
                    second_points = image_pieces[second_piece]
                    second_ids = tuple(
                        (
                            identifier,
                            _image_winding(
                                winding,
                                exponent,
                                orders,
                                fixed_by_root.get(identifier, no_fixed),
                            ),
                        )
                        for identifier, winding in second_base_ids
                    )
                if set(first_ids) == set(second_ids):
                    first_by_id = dict(zip(first_ids, first_points, strict=True))
                    second_by_id = dict(zip(second_ids, second_points, strict=True))
                    if all(
                        first_by_id[identifier] == second_by_id[identifier]
                        for identifier in first_by_id
                    ):
                        continue
                    findings.append(
                        (
                            "periodic_source_trace_mismatch",
                            (first_cell.global_id, second_cell.global_id),
                            "One quotient simplex has inconsistent exact image coordinates.",
                        )
                    )
                    continue
                try:
                    contacts = _simplex_contact_points(first_points, second_points)
                except (ValueError, ZeroDivisionError) as error:
                    findings.append(
                        (
                            "periodic_exact_contact_unresolved",
                            (first_cell.global_id, second_cell.global_id),
                            str(error),
                        )
                    )
                    continue
                if not contacts:
                    continue
                id_to_point = dict(zip(first_ids, first_points, strict=True))
                if not contact_is_declared(contacts, first_ids, second_ids, id_to_point):
                    findings.append(
                        (
                            "periodic_cell_contact",
                            (first_cell.global_id, second_cell.global_id),
                            "A cell intersects a periodic image beyond its declared quotient stratum.",
                        )
                    )
    dispositions = tuple(_contact_disposition(check) for check, _, _ in findings)
    status = (
        "violated"
        if "violated" in dispositions
        else "unresolved"
        if findings
        else "certified"
    )
    return PeriodicContactResult(status, tuple(findings), boundary_kind)


__all__ = [
    "PeriodicContactResult",
    "PeriodicMappedDegreeResult",
    "certify_periodic_affine_embedding",
    "certify_periodic_mapped_torus_degree",
]
