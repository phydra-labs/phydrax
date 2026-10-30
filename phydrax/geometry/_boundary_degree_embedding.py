#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact oriented-boundary and shell-degree proofs for volume-cell maps."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations, pairwise
from typing import cast, Literal

import numpy as np

from .._fingerprint import canonical_fingerprint
from ..discretization._cell_mesh import CellMesh
from ..discretization._coordinate_enclosure import (
    affine_map,
    bernstein_coefficients,
    compose_expression,
    constant,
    derivative,
    evaluate_expression,
    linear_combinations,
    polynomial_bounds,
    scale,
    subtract,
)
from ..discretization._exact_algebraic_geometry import ExactScalar, RadicalElement
from ..discretization._reference_cell import reference_cell_topology
from ._embedding_cells import EmbeddingCell
from ._exact_polyhedral_geometry import (
    _cross3,
    _dot3,
    _sign,
    _sub3,
    determinant3,
    Point3,
    point_in_triangle3,
    point_on_segment3,
    triangle_contact_points3,
)
from ._mapped_embedding import prove_mapped_cell_injective
from ._planar_chain_embedding import triangulate_planar_polygon
from ._restricted_embedding import _trace_arguments, restriction_trace_equal


type BoundaryStatus = Literal["certified", "violated", "unresolved"]


@dataclass(frozen=True, slots=True)
class BoundaryFinding:
    check: str
    entity_ids: tuple[int, ...]
    detail: str


@dataclass(frozen=True, slots=True)
class BoundaryTriangle:
    parent_face: int
    facet_id: int
    vertex_ids: tuple[int, int, int]
    points: tuple[Point3, Point3, Point3]
    shell: int = -1


@dataclass(frozen=True, slots=True)
class BoundaryFace:
    facet_id: int
    vertex_ids: tuple[int, ...]
    points: tuple[Point3, ...]
    triangles: tuple[BoundaryTriangle, ...]
    map_evidence: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class OrientedBoundary:
    status: BoundaryStatus
    faces: tuple[BoundaryFace, ...]
    triangles: tuple[BoundaryTriangle, ...]
    shell_face_indices: tuple[tuple[int, ...], ...]
    shell_signed_measures: tuple[ExactScalar, ...]
    shell_degrees: tuple[int, ...]
    boundary_degree_one: bool
    findings: tuple[BoundaryFinding, ...]


def _loop_relation(first: tuple[int, ...], second: tuple[int, ...]) -> int | None:
    if len(first) != len(second) or set(first) != set(second):
        return None
    count = len(first)
    for shift in range(count):
        if second == first[shift:] + first[:shift]:
            return 1
    reversed_first = tuple(reversed(first))
    for shift in range(count):
        if second == reversed_first[shift:] + reversed_first[:shift]:
            return -1
    return None


def _canonical_face_triangles(
    facet_id: int,
    vertex_ids: tuple[int, ...],
    points: tuple[Point3, ...],
    /,
    *,
    vertex_defined: bool = False,
) -> tuple[BoundaryTriangle, ...]:
    if len(points) != len(vertex_ids):
        raise ValueError("Boundary face points must align with their vertex identities.")
    if len(vertex_ids) == 3:
        return (
            BoundaryTriangle(
                -1,
                facet_id,
                (vertex_ids[0], vertex_ids[1], vertex_ids[2]),
                (points[0], points[1], points[2]),
            ),
        )
    if vertex_defined:
        # Vertex-defined polyhedral faces are exact planar loops that may carry
        # collinear vertices; their exact ear triangulation keeps the loop.
        routes = triangulate_planar_polygon(points)
    elif len(vertex_ids) != 4:
        raise ValueError(
            "Exact affine boundary route supports triangular and quadrilateral facets."
        )
    else:
        first_diagonal = tuple(sorted((vertex_ids[0], vertex_ids[2])))
        second_diagonal = tuple(sorted((vertex_ids[1], vertex_ids[3])))
        routes = (
            ((0, 1, 2), (0, 2, 3))
            if first_diagonal <= second_diagonal
            else ((0, 1, 3), (1, 2, 3))
        )
    return tuple(
        BoundaryTriangle(
            -1,
            facet_id,
            (vertex_ids[route[0]], vertex_ids[route[1]], vertex_ids[route[2]]),
            (points[route[0]], points[route[1]], points[route[2]]),
        )
        for route in routes
    )


def _connectivity_face_rows(mesh: CellMesh) -> tuple[tuple[int, ...], ...]:
    connectivity = mesh.connectivity
    rows = getattr(connectivity, "faces", None)
    if rows is not None:
        return tuple(tuple(int(value) for value in row) for row in np.asarray(rows))
    entity_rows = getattr(connectivity, "entities", None)
    if entity_rows is not None and len(entity_rows) > 2:
        return tuple(
            tuple(int(value) for value in row) for row in np.asarray(entity_rows[2])
        )
    offsets = getattr(connectivity, "face_vertex_offsets", None)
    values = getattr(connectivity, "face_vertex_values", None)
    if offsets is None or values is None:
        raise ValueError("Mesh connectivity does not retain canonical facet rows.")
    offsets_ = np.asarray(offsets, dtype=np.int64)
    values_ = np.asarray(values, dtype=np.int64)
    return tuple(
        tuple(int(value) for value in values_[start:stop])
        for start, stop in pairwise(offsets_)
    )


def _facet_identity_map(mesh: CellMesh) -> dict[frozenset[int], int]:
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    rows = _connectivity_face_rows(mesh)
    face_ids = np.asarray(mesh.topology.entity_sets[2].entity_ids, dtype=np.int64)
    if len(rows) != face_ids.size:
        raise ValueError(
            "Canonical facet rows do not align with their declared identities."
        )
    return {
        frozenset(int(vertex_ids[index]) for index in row): int(identifier)
        for row, identifier in zip(rows, face_ids, strict=True)
    }


def _cell_facets(cell: EmbeddingCell) -> tuple[tuple[int, ...], ...]:
    if cell.dimension != 3:
        raise ValueError("Boundary-degree cells must have topological dimension three.")
    if cell.kind == "polyhedron":
        return cell.facet_routes
    return tuple(
        tuple(int(value) for value in row)
        for row in reference_cell_topology(cell.kind).entities[2]
    )


def _shell_components(
    faces: tuple[BoundaryFace, ...],
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, int], ...]]:
    edge_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for face_index, face in enumerate(faces):
        closed_loop = (*face.vertex_ids[1:], face.vertex_ids[0])
        for left, right in zip(face.vertex_ids, closed_loop, strict=True):
            edge = (left, right) if left < right else (right, left)
            edge_faces[edge].append(face_index)
    bad = tuple(
        sorted(edge for edge, incident in edge_faces.items() if len(incident) != 2)
    )
    if bad:
        return (), bad
    adjacency: list[set[int]] = [set() for _ in faces]
    for incident in edge_faces.values():
        first, second = incident
        adjacency[first].add(second)
        adjacency[second].add(first)
    labels = [-1] * len(faces)
    components: list[tuple[int, ...]] = []
    for root in range(len(faces)):
        if labels[root] >= 0:
            continue
        label = len(components)
        queue = deque((root,))
        labels[root] = label
        component = []
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbor in adjacency[current]:
                if labels[neighbor] < 0:
                    labels[neighbor] = label
                    queue.append(neighbor)
        components.append(tuple(sorted(component)))
    return tuple(components), ()


def _shell_measure(faces: tuple[BoundaryFace, ...], /) -> ExactScalar:
    total: ExactScalar = Fraction(0)
    origin: Point3 = (Fraction(0), Fraction(0), Fraction(0))
    for face in faces:
        for triangle in face.triangles:
            total += determinant3(origin, *triangle.points)
    return total / 6


def _ray_winding(
    query: Point3,
    triangles: tuple[BoundaryTriangle, ...],
    /,
) -> tuple[int, bool]:
    """Return exact oriented ray winding with a finite generic-direction theorem."""
    if not triangles:
        return 0, True
    for exponent in range(8 * len(triangles) + 1):
        direction: Point3 = (
            Fraction(1),
            Fraction(exponent),
            Fraction(exponent * exponent + 1),
        )
        crossings = 0
        generic = True
        for triangle in triangles:
            first, second, third = triangle.points
            normal = _cross3(_sub3(second, first), _sub3(third, first))
            denominator = _dot3(normal, direction)
            if not denominator:
                generic = False
                break
            parameter = _dot3(normal, _sub3(first, query)) / denominator
            if parameter <= 0:
                continue
            hit_x, hit_y, hit_z = (
                query[axis] + parameter * direction[axis] for axis in range(3)
            )
            hit = (hit_x, hit_y, hit_z)
            if not point_in_triangle3(hit, triangle.points):
                continue
            if any(
                point_on_segment3(hit, triangle.points[left], triangle.points[right])
                for left, right in ((0, 1), (1, 2), (2, 0))
            ):
                generic = False
                break
            crossings += _sign(denominator)
        if generic:
            return crossings, True
    return 0, False


def _edge(left: int, right: int, /) -> tuple[int, int]:
    return (left, right) if left < right else (right, left)


def _loop_has_edge(loop: tuple[int, ...], left: int, right: int, /) -> bool:
    return any(
        _edge(loop[index], loop[(index + 1) % len(loop)]) == _edge(left, right)
        for index in range(len(loop))
    )


def _face_trace_is_affine(cell: EmbeddingCell, vertex_ids: tuple[int, ...], /) -> bool:
    if len(vertex_ids) not in (3, 4):
        return False
    if cell.affine:
        try:
            points = tuple(
                cell.corner_images[cell.vertex_ids.index(identifier)]
                for identifier in vertex_ids
            )
            if any(len(point) != 3 for point in points):
                return False
            first = (points[0][0], points[0][1], points[0][2])
            second = (points[1][0], points[1][1], points[1][2])
            third = (points[2][0], points[2][1], points[2][2])
        except (IndexError, ValueError):
            return False
        return any(_cross3(_sub3(second, first), _sub3(third, first)))
    if not cell.coordinates:
        return False
    arguments = _trace_arguments(cell, vertex_ids)
    if arguments is None:
        return False
    try:
        trace = tuple(
            compose_expression(expression, arguments) for expression in cell.coordinates
        )
        affine = affine_map(trace, 2)
    except (TypeError, ValueError, ZeroDivisionError):
        return False
    if affine is None:
        return False
    _, matrix = affine
    if len(matrix) != 3 or any(len(row) != 2 for row in matrix):
        return False
    first = (matrix[0][0], matrix[1][0], matrix[2][0])
    second = (matrix[0][1], matrix[1][1], matrix[2][1])
    return any(_cross3(first, second))


def _exact_scalar_identity(value: ExactScalar, /) -> tuple[object, ...]:
    if isinstance(value, Fraction):
        return ("rational", value.numerator, value.denominator)
    if isinstance(value, RadicalElement):
        return (
            "radical",
            tuple(
                (radicand, coefficient.numerator, coefficient.denominator)
                for radicand, coefficient in value.terms
            ),
        )
    raise TypeError(f"Unsupported boundary-face scalar {type(value).__name__}.")


def _vertex_defined_face_evidence(
    points: tuple[Point3, ...], /
) -> tuple[str, ...] | None:
    """Prove a vertex-defined face loop is exactly planar with nonzero area."""
    normal: Point3 = (Fraction(0), Fraction(0), Fraction(0))
    for start, stop in zip(points, (*points[1:], points[0]), strict=True):
        term = _cross3(start, stop)
        normal = (normal[0] + term[0], normal[1] + term[1], normal[2] + term[2])
    if not any(_sign(value) for value in normal):
        return None
    offset = _dot3(normal, points[0])
    if any(_sign(_dot3(normal, point) - offset) for point in points):
        return None
    return (
        "exact-vertex-defined-planar-face",
        f"plane:{tuple(_exact_scalar_identity(value) for value in normal)}:"
        f"offset:{_exact_scalar_identity(offset)}",
    )


def _face_trace_image_evidence(
    cell: EmbeddingCell,
    vertex_ids: tuple[int, ...],
    points: tuple[Point3, ...],
    /,
) -> tuple[str, ...] | None:
    """Prove a face image is its exact planar corner polygon."""
    if _face_trace_is_affine(cell, vertex_ids):
        return ("exact-affine-face-image",)
    if cell.kind == "polyhedron" and not cell.coordinates:
        # A vertex-defined face is by definition its corner polygon; only its
        # exact planarity and nonzero area remain to be proved.
        return _vertex_defined_face_evidence(points)
    if (
        len(vertex_ids) not in (3, 4)
        or cell.pyramid_normalizations is not None
        or not cell.coordinates
    ):
        return None
    face_kind = "triangle" if len(vertex_ids) == 3 else "quadrilateral"
    domain = "simplex" if face_kind == "triangle" else "box"
    arguments = _trace_arguments(cell, vertex_ids)
    if arguments is None:
        return None
    try:
        trace = tuple(
            compose_expression(expression, arguments) for expression in cell.coordinates
        )
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    if len(trace) != 3 or any(not isinstance(expression, dict) for expression in trace):
        return None
    trace_polynomials = tuple(
        cast(dict[tuple[int, ...], ExactScalar], expression) for expression in trace
    )
    if len(points) != len(vertex_ids):
        return None
    plane: tuple[ExactScalar, ExactScalar, ExactScalar] | None = None
    for first, second, third in combinations(range(len(points)), 3):
        candidate = _cross3(
            _sub3(points[second], points[first]),
            _sub3(points[third], points[first]),
        )
        if any(candidate):
            plane = candidate
            break
    if plane is None:
        return None
    offset = _dot3(plane, points[0])
    plane_expression = linear_combinations(trace_polynomials, (plane,))[0]
    plane_residual = subtract(plane_expression, constant(offset, 2))
    if not isinstance(plane_residual, dict) or plane_residual:
        return None
    if len(points) == 4:
        drop_axis = next(index for index, value in enumerate(plane) if value)
        projected_axes = tuple(index for index in range(3) if index != drop_axis)
        turns = tuple(
            (
                points[(index + 1) % 4][projected_axes[0]]
                - points[index][projected_axes[0]]
            )
            * (
                points[(index + 2) % 4][projected_axes[1]]
                - points[(index + 1) % 4][projected_axes[1]]
            )
            - (
                points[(index + 1) % 4][projected_axes[1]]
                - points[index][projected_axes[1]]
            )
            * (
                points[(index + 2) % 4][projected_axes[0]]
                - points[(index + 1) % 4][projected_axes[0]]
            )
            for index in range(4)
        )
        if not all(_sign(value) == _sign(turns[0]) and value for value in turns):
            return None
    image_points = {
        identifier: point for identifier, point in zip(vertex_ids, points, strict=True)
    }
    edge_evidence: list[str] = []
    for left, right in zip(vertex_ids, (*vertex_ids[1:], vertex_ids[0]), strict=True):
        edge_arguments = _trace_arguments(cell, (left, right))
        if edge_arguments is None:
            return None
        edge_trace = tuple(
            compose_expression(expression, edge_arguments)
            for expression in cell.coordinates
        )
        if len(edge_trace) != 3 or any(
            not isinstance(expression, dict) for expression in edge_trace
        ):
            return None
        edge_polynomials = tuple(
            cast(dict[tuple[int, ...], ExactScalar], expression)
            for expression in edge_trace
        )
        edge_start: Point3 = cast(
            Point3,
            tuple(
                evaluate_expression(expression, (Fraction(0),))
                for expression in edge_polynomials
            ),
        )
        edge_end: Point3 = cast(
            Point3,
            tuple(
                evaluate_expression(expression, (Fraction(1),))
                for expression in edge_polynomials
            ),
        )
        if not (
            (edge_start == image_points[left] and edge_end == image_points[right])
            or (edge_start == image_points[right] and edge_end == image_points[left])
        ):
            return None
        direction = _sub3(edge_end, edge_start)
        length_squared = _dot3(direction, direction)
        if not length_squared:
            return None
        differences = tuple(
            subtract(expression, constant(edge_start[axis], 1))
            for axis, expression in enumerate(edge_polynomials)
        )
        if any(not isinstance(expression, dict) for expression in differences):
            return None
        difference_polynomials = tuple(expression for expression in differences)
        numerator = linear_combinations(difference_polynomials, (direction,))[0]
        parameter = scale(numerator, Fraction(1) / length_squared)
        if not isinstance(parameter, dict):
            return None
        if (
            evaluate_expression(parameter, (Fraction(0),)) != 0
            or evaluate_expression(parameter, (Fraction(1),)) != 1
        ):
            return None
        for axis in range(3):
            line_residual = subtract(
                difference_polynomials[axis], scale(parameter, direction[axis])
            )
            if not isinstance(line_residual, dict) or line_residual:
                return None
        derivative_controls = bernstein_coefficients(derivative(parameter, 0), "box", 1)
        if not derivative_controls or any(
            _sign(value) <= 0 for value in derivative_controls
        ):
            return None
        edge_evidence.append(
            f"edge:{left}:{right}:strict-parameter-controls:"
            f"{tuple(_exact_scalar_identity(value) for value in derivative_controls)}"
        )
    topology = reference_cell_topology(face_kind)
    face_references = tuple(
        tuple(Fraction(float(value)) for value in row) for row in topology.vertices
    )
    face_corner_images = tuple(
        tuple(evaluate_expression(expression, point) for expression in trace_polynomials)
        for point in face_references
    )
    bounds = tuple(
        polynomial_bounds(expression, domain, 2) for expression in trace_polynomials
    )
    face_cell = EmbeddingCell(
        cell.global_id,
        cell.block_name,
        cell.local_index,
        face_kind,
        2,
        domain,
        tuple(range(len(vertex_ids))),
        vertex_ids,
        face_references,
        trace_polynomials,
        face_corner_images,
        None,
        None,
        tuple(lower for lower, _ in bounds),
        tuple(upper for _, upper in bounds),
        tuple(tuple(int(index) for index in route) for route in topology.entities[1]),
    )
    injectivity = prove_mapped_cell_injective(face_cell)
    if injectivity is None:
        return None
    trace_identity = canonical_fingerprint(
        {
            "kind": "planar-boundary-face-trace",
            "vertex_ids": vertex_ids,
            "coordinates": tuple(
                tuple(
                    (tuple(exponent), _exact_scalar_identity(coefficient))
                    for exponent, coefficient in sorted(expression.items())
                )
                for expression in trace_polynomials
            ),
        }
    )
    return (
        "exact-planar-injective-face-image",
        f"trace:{trace_identity}",
        f"plane:{tuple(_exact_scalar_identity(value) for value in plane)}:"
        f"offset:{_exact_scalar_identity(offset)}",
        f"projection_rows:{injectivity.projection_rows}:"
        f"matrix:{tuple(tuple(_exact_scalar_identity(value) for value in row) for row in injectivity.projection_matrix)}:"
        f"controls:{injectivity.principal_minor_controls}",
        *edge_evidence,
    )


def _vertex_images(
    cells: tuple[EmbeddingCell, ...], /
) -> tuple[dict[int, Point3], tuple[int, ...]]:
    images: dict[int, Point3] = {}
    mismatched: set[int] = set()
    for cell in cells:
        for identifier, point in zip(cell.vertex_ids, cell.corner_images, strict=True):
            if len(point) != 3:
                mismatched.add(identifier)
                continue
            image = (point[0], point[1], point[2])
            previous = images.setdefault(identifier, image)
            if previous != image:
                mismatched.add(identifier)
    return images, tuple(sorted(mismatched))


def _cycle_from_edges(edges: tuple[tuple[int, int], ...], /) -> tuple[int, ...] | None:
    adjacency: dict[int, list[int]] = defaultdict(list)
    for left, right in edges:
        if left == right:
            return None
        adjacency[left].append(right)
        adjacency[right].append(left)
    if len(adjacency) < 3 or any(len(neighbors) != 2 for neighbors in adjacency.values()):
        return None
    start = min(adjacency)
    cycle = [start]
    previous: int | None = None
    current = start
    while True:
        neighbors = adjacency[current]
        next_vertex = neighbors[0] if neighbors[0] != previous else neighbors[1]
        if next_vertex == start:
            break
        if next_vertex in cycle:
            return None
        cycle.append(next_vertex)
        previous, current = current, next_vertex
    return tuple(cycle) if len(cycle) == len(adjacency) else None


def _connected_components(adjacency: tuple[set[int], ...], /) -> int:
    labels = [-1] * len(adjacency)
    count = 0
    for root in range(len(adjacency)):
        if labels[root] >= 0:
            continue
        labels[root] = count
        queue = deque((root,))
        while queue:
            current = queue.popleft()
            for neighbor in adjacency[current]:
                if labels[neighbor] < 0:
                    labels[neighbor] = count
                    queue.append(neighbor)
        count += 1
    return count


def _surface_link_is_manifold(polygons: tuple[tuple[int, ...], ...], /) -> bool:
    if not polygons or any(
        len(polygon) < 3 or len(set(polygon)) != len(polygon) for polygon in polygons
    ):
        return False
    edge_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for face_index, polygon in enumerate(polygons):
        for index, left in enumerate(polygon):
            edge_faces[_edge(left, polygon[(index + 1) % len(polygon)])].append(
                face_index
            )
    if any(len(incident) not in (1, 2) for incident in edge_faces.values()):
        return False
    face_adjacency: list[set[int]] = [set() for _ in polygons]
    for incident in edge_faces.values():
        if len(incident) == 2:
            first, second = incident
            face_adjacency[first].add(second)
            face_adjacency[second].add(first)
    if _connected_components(tuple(face_adjacency)) != 1:
        return False
    boundary_edges = tuple(
        edge for edge, incident in edge_faces.items() if len(incident) == 1
    )
    vertices = {vertex for polygon in polygons for vertex in polygon}
    euler_characteristic = len(vertices) - len(edge_faces) + len(polygons)
    if not boundary_edges:
        return euler_characteristic == 2
    return _cycle_from_edges(boundary_edges) is not None and euler_characteristic == 1


def _volume_topology_findings(
    cells: tuple[EmbeddingCell, ...],
    cell_faces: tuple[tuple[tuple[int, ...], ...], ...],
    occurrences: dict[
        frozenset[int], list[tuple[int, tuple[int, ...], tuple[Point3, ...]]]
    ],
    /,
) -> tuple[list[BoundaryFinding], int]:
    findings: list[BoundaryFinding] = []
    cell_adjacency: list[set[int]] = [set() for _ in cells]
    for incident in occurrences.values():
        if len(incident) == 2:
            first, second = incident[0][0], incident[1][0]
            cell_adjacency[first].add(second)
            cell_adjacency[second].add(first)
    component_count = _connected_components(tuple(cell_adjacency))

    vertex_link_faces: dict[int, list[tuple[int, ...]]] = defaultdict(list)
    edge_link_cells: dict[
        tuple[int, int], list[tuple[frozenset[int], frozenset[int]]]
    ] = defaultdict(list)
    for cell, faces in zip(cells, cell_faces, strict=True):
        face_keys = tuple(frozenset(face) for face in faces)
        for identifier in set(cell.vertex_ids):
            vertex_link_edges: list[tuple[int, int]] = []
            for face in faces:
                if identifier not in face:
                    continue
                position = face.index(identifier)
                vertex_link_edges.append(
                    (
                        face[(position - 1) % len(face)],
                        face[(position + 1) % len(face)],
                    )
                )
            cycle = _cycle_from_edges(tuple(vertex_link_edges))
            if cycle is None:
                findings.append(
                    BoundaryFinding(
                        "nonmanifold_volume_vertex_link",
                        (identifier,),
                        "The cellwise link at this volume vertex is not one polygonal cycle.",
                    )
                )
                continue
            vertex_link_faces[identifier].append(cycle)
        cell_edges = {
            _edge(left, face[(index + 1) % len(face)])
            for face in faces
            for index, left in enumerate(face)
        }
        for edge in cell_edges:
            incident_faces = tuple(
                face_key for face_key in face_keys if set(edge) <= face_key
            )
            if len(incident_faces) != 2:
                findings.append(
                    BoundaryFinding(
                        "nonmanifold_volume_edge_link",
                        edge,
                        "Each volume-cell edge must meet exactly two cell facets.",
                    )
                )
                continue
            edge_link_cells[edge].append((incident_faces[0], incident_faces[1]))

    for identifier, polygons in vertex_link_faces.items():
        if not _surface_link_is_manifold(tuple(polygons)):
            findings.append(
                BoundaryFinding(
                    "nonmanifold_volume_vertex_link",
                    (identifier,),
                    "The exact volume-vertex link is not one disk or sphere.",
                )
            )
    for edge, edge_ring in edge_link_cells.items():
        adjacency: dict[frozenset[int], list[frozenset[int]]] = defaultdict(list)
        for first, second in edge_ring:
            adjacency[first].append(second)
            adjacency[second].append(first)
        nodes = set(adjacency)
        if not nodes:
            continue
        boundary_nodes = {
            face_key for face_key in nodes if len(occurrences.get(face_key, ())) == 1
        }
        degrees = tuple(len(adjacency[node]) for node in nodes)
        path_or_cycle = (
            all(degree == 2 for degree in degrees)
            if not boundary_nodes
            else (
                len(boundary_nodes) == 2
                and all(
                    len(adjacency[node]) == (1 if node in boundary_nodes else 2)
                    for node in nodes
                )
            )
        )
        reached = {next(iter(nodes))}
        queue = deque(reached)
        while queue:
            current = queue.popleft()
            for neighbor in adjacency[current]:
                if neighbor not in reached:
                    reached.add(neighbor)
                    queue.append(neighbor)
        if not path_or_cycle or reached != nodes:
            findings.append(
                BoundaryFinding(
                    "nonmanifold_volume_edge_link",
                    edge,
                    "The exact link of this volume edge is not one interval or circle.",
                )
            )
    return findings, component_count


def _boundary_vertex_links_are_manifold(
    faces: tuple[BoundaryFace, ...], /
) -> tuple[int, ...]:
    link_edges: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for face in faces:
        for position, identifier in enumerate(face.vertex_ids):
            link_edges[identifier].append(
                (
                    face.vertex_ids[position - 1],
                    face.vertex_ids[(position + 1) % len(face.vertex_ids)],
                )
            )
    return tuple(
        identifier
        for identifier, edges in link_edges.items()
        if _cycle_from_edges(tuple(edges)) is None
    )


def _shell_euler_characteristic(faces: tuple[BoundaryFace, ...], /) -> int:
    vertices = {identifier for face in faces for identifier in face.vertex_ids}
    edges = {
        _edge(face.vertex_ids[index], face.vertex_ids[(index + 1) % len(face.vertex_ids)])
        for face in faces
        for index in range(len(face.vertex_ids))
    }
    return len(vertices) - len(edges) + len(faces)


def _boundary_triangle_contacts(
    faces: tuple[BoundaryFace, ...],
    triangles: tuple[BoundaryTriangle, ...],
    vertex_images: dict[int, Point3],
    /,
) -> list[BoundaryFinding]:
    findings: list[BoundaryFinding] = []
    for first_index, first in enumerate(triangles):
        for second_index in range(first_index + 1, len(triangles)):
            second = triangles[second_index]
            if any(
                max(point[axis] for point in first.points)
                < min(point[axis] for point in second.points)
                or max(point[axis] for point in second.points)
                < min(point[axis] for point in first.points)
                for axis in range(3)
            ):
                continue
            try:
                contacts = triangle_contact_points3(first.points, second.points)
            except ValueError as error:
                findings.append(
                    BoundaryFinding(
                        "degenerate_boundary_triangle",
                        tuple(sorted((first.facet_id, second.facet_id))),
                        str(error),
                    )
                )
                continue
            if not contacts:
                continue
            shared = set(first.vertex_ids).intersection(second.vertex_ids)
            allowed = False
            if first.parent_face == second.parent_face and len(shared) == 2:
                allowed = True
            elif len(shared) == 1:
                allowed = True
            elif len(shared) == 2:
                left, right = tuple(sorted(shared))
                first_face = faces[first.parent_face].vertex_ids
                second_face = faces[second.parent_face].vertex_ids
                allowed = _loop_has_edge(first_face, left, right) and _loop_has_edge(
                    second_face, left, right
                )
            if allowed and len(shared) == 1:
                identifier = next(iter(shared))
                allowed = identifier in vertex_images and all(
                    point == vertex_images[identifier] for point in contacts
                )
            elif allowed and len(shared) == 2:
                identifiers = tuple(sorted(shared))
                if all(identifier in vertex_images for identifier in identifiers):
                    left, right = (
                        vertex_images[identifier] for identifier in identifiers
                    )
                    allowed = all(
                        point_on_segment3(point, left, right) for point in contacts
                    )
                else:
                    allowed = False
            else:
                allowed = False
            if not allowed:
                findings.append(
                    BoundaryFinding(
                        "exterior_boundary_self_contact",
                        tuple(sorted((first.facet_id, second.facet_id))),
                        "Exact exterior triangles intersect beyond their declared common vertex or edge.",
                    )
                )
    return findings


def build_oriented_boundary(
    mesh: CellMesh, cells: tuple[EmbeddingCell, ...], /
) -> OrientedBoundary:
    """Certify exact affine exterior traces and their oriented shell degree."""
    findings: list[BoundaryFinding] = []
    if mesh.topological_dimension != 3 or mesh.ambient_dimension != 3:
        finding = BoundaryFinding(
            "boundary_degree_dimension",
            (),
            "Boundary degree requires a full-dimensional 3-D complex.",
        )
        return OrientedBoundary("unresolved", (), (), (), (), (), False, (finding,))
    if not cells or any(cell.dimension != 3 for cell in cells):
        finding = BoundaryFinding(
            "curved_volume_boundary",
            tuple(cell.global_id for cell in cells),
            "The exact source complex is not a nonempty full-dimensional cell complex.",
        )
        return OrientedBoundary("unresolved", (), (), (), (), (), False, (finding,))
    try:
        facet_ids = _facet_identity_map(mesh)
        occurrences: dict[
            frozenset[int], list[tuple[int, tuple[int, ...], tuple[Point3, ...]]]
        ] = defaultdict(list)
        cell_faces_list: list[tuple[tuple[int, ...], ...]] = []
        for cell_index, cell in enumerate(cells):
            cell_faces = []
            for route in _cell_facets(cell):
                ids = tuple(cell.vertex_ids[index] for index in route)
                points = tuple(
                    (point[0], point[1], point[2])
                    for point in (cell.corner_images[index] for index in route)
                    if len(point) == 3
                )
                if len(points) != len(ids):
                    raise ValueError("A volume facet has non-spatial corner images.")
                cell_faces.append(ids)
                occurrences[frozenset(ids)].append((cell_index, ids, points))
            cell_faces_list.append(tuple(cell_faces))
    except (IndexError, KeyError, ValueError) as error:
        finding = BoundaryFinding("boundary_facet_preparation", (), str(error))
        return OrientedBoundary("unresolved", (), (), (), (), (), False, (finding,))

    vertex_images, mismatched_vertices = _vertex_images(cells)
    if mismatched_vertices:
        findings.append(
            BoundaryFinding(
                "boundary_vertex_image_mismatch",
                mismatched_vertices,
                "A shared global vertex has different exact mapped coordinates.",
            )
        )

    faces: list[BoundaryFace] = []
    for key, incident in occurrences.items():
        facet_id = facet_ids.get(key)
        if facet_id is None:
            findings.append(
                BoundaryFinding(
                    "missing_facet_identity",
                    (),
                    "A cell face has no matching declared topology facet.",
                )
            )
            continue
        if len(incident) > 2:
            findings.append(
                BoundaryFinding(
                    "nonmanifold_facet",
                    (facet_id,),
                    "More than two volume cells share a facet.",
                )
            )
            continue
        if len(incident) == 2:
            relation = _loop_relation(incident[0][1], incident[1][1])
            if relation != -1:
                findings.append(
                    BoundaryFinding(
                        "inconsistent_facet_orientation",
                        (facet_id,),
                        "Interior facet orientations do not cancel.",
                    )
                )
            first_index, second_index = incident[0][0], incident[1][0]
            trace = restriction_trace_equal(
                cells[first_index],
                cells[second_index],
                tuple(sorted(key)),
            )
            if trace is False:
                findings.append(
                    BoundaryFinding(
                        "mismatched_interior_trace",
                        (facet_id,),
                        "Exact coordinate maps disagree on an interior facet.",
                    )
                )
            elif trace is None:
                findings.append(
                    BoundaryFinding(
                        "interior_trace_unresolved",
                        (facet_id,),
                        "No exact whole-facet coordinate-map equality proof is available.",
                    )
                )
            continue
        cell_index, vertex_ids, points = incident[0]
        map_evidence = _face_trace_image_evidence(cells[cell_index], vertex_ids, points)
        if map_evidence is None:
            findings.append(
                BoundaryFinding(
                    "nonaffine_exterior_facet",
                    (facet_id,),
                    "The exact exterior image is not proved affine or planar-injective.",
                )
            )
        try:
            triangles = _canonical_face_triangles(
                facet_id,
                vertex_ids,
                points,
                vertex_defined=cells[cell_index].kind == "polyhedron",
            )
        except ValueError as error:
            findings.append(
                BoundaryFinding("unsupported_boundary_facet", (facet_id,), str(error))
            )
            continue
        faces.append(
            BoundaryFace(
                facet_id,
                vertex_ids,
                points,
                triangles,
                () if map_evidence is None else map_evidence,
            )
        )

    cell_faces = tuple(cell_faces_list)
    topology_findings, volume_component_count = _volume_topology_findings(
        cells, cell_faces, occurrences
    )
    findings.extend(topology_findings)
    face_tuple = tuple(faces)
    if not face_tuple:
        findings.append(
            BoundaryFinding(
                "boundary_empty",
                (),
                "A bounded embedded volume complex requires a nonempty closed boundary.",
            )
        )
    components, bad_edges = _shell_components(face_tuple)
    if bad_edges:
        findings.append(
            BoundaryFinding(
                "nonmanifold_boundary_ridge",
                (),
                "Boundary ridges do not have exactly two incident facets.",
            )
        )
    bad_boundary_vertices = _boundary_vertex_links_are_manifold(face_tuple)
    if bad_boundary_vertices:
        findings.append(
            BoundaryFinding(
                "nonmanifold_boundary_vertex_link",
                bad_boundary_vertices,
                "The exterior boundary link at this vertex is not one cycle.",
            )
        )
    planar_embedded_exterior = not any(
        finding.check in ("nonaffine_exterior_facet", "unsupported_boundary_facet")
        for finding in findings
    )
    degree_topology_blockers = {
        "boundary_empty",
        "boundary_facet_preparation",
        "boundary_vertex_image_mismatch",
        "curved_volume_boundary",
        "inconsistent_facet_orientation",
        "interior_trace_unresolved",
        "missing_facet_identity",
        "mismatched_interior_trace",
        "nonaffine_exterior_facet",
        "nonmanifold_boundary_ridge",
        "nonmanifold_boundary_vertex_link",
        "nonmanifold_facet",
        "nonmanifold_volume_edge_link",
        "nonmanifold_volume_vertex_link",
        "unsupported_boundary_facet",
    }
    degree_topology_available = planar_embedded_exterior and not any(
        finding.check in degree_topology_blockers for finding in findings
    )
    shell_measures = (
        tuple(
            _shell_measure(tuple(face_tuple[index] for index in shell))
            for shell in components
        )
        if degree_topology_available
        else ()
    )
    shell_euler = tuple(
        _shell_euler_characteristic(tuple(face_tuple[index] for index in shell))
        for shell in components
    )
    if degree_topology_available:
        for shell_index, measure in enumerate(shell_measures):
            if not measure:
                findings.append(
                    BoundaryFinding(
                        "degenerate_boundary_shell",
                        tuple(
                            face_tuple[index].facet_id
                            for index in components[shell_index]
                        ),
                        "An embedded closed shell must have nonzero oriented volume.",
                    )
                )

    triangles = tuple(
        BoundaryTriangle(
            face_index,
            triangle.facet_id,
            triangle.vertex_ids,
            triangle.points,
            shell_index,
        )
        for shell_index, shell in enumerate(components)
        for face_index in shell
        for triangle in face_tuple[face_index].triangles
    )
    if planar_embedded_exterior:
        findings.extend(_boundary_triangle_contacts(face_tuple, triangles, vertex_images))
    degree_geometry_available = degree_topology_available and not any(
        finding.check
        in (
            "degenerate_boundary_shell",
            "degenerate_boundary_triangle",
            "exterior_boundary_self_contact",
        )
        for finding in findings
    )
    degrees: list[int] = []
    if degree_geometry_available:
        for shell_index, shell in enumerate(components):
            query = face_tuple[shell[0]].points[0]
            other = tuple(
                triangle for triangle in triangles if triangle.shell != shell_index
            )
            winding, decided = _ray_winding(query, other)
            if not decided:
                findings.append(
                    BoundaryFinding(
                        "boundary_degree_ray_unresolved",
                        tuple(face_tuple[index].facet_id for index in shell),
                        "No exact general ray direction was certified.",
                    )
                )
                continue
            orientation = _sign(shell_measures[shell_index])
            exterior_degree = winding + (1 if orientation > 0 else 0) - 1
            degrees.append(exterior_degree)
            if exterior_degree != 0:
                findings.append(
                    BoundaryFinding(
                        "boundary_degree",
                        tuple(face_tuple[index].facet_id for index in shell),
                        f"Exterior degree is {exterior_degree}, not zero.",
                    )
                )

    unresolved_checks = {
        "boundary_degree_dimension",
        "boundary_degree_ray_unresolved",
        "boundary_empty",
        "boundary_facet_preparation",
        "curved_volume_boundary",
        "interior_trace_unresolved",
        "missing_facet_identity",
        "nonaffine_exterior_facet",
        "nonmanifold_boundary_ridge",
        "nonmanifold_boundary_vertex_link",
        "nonmanifold_volume_edge_link",
        "nonmanifold_volume_vertex_link",
        "unsupported_boundary_facet",
    }
    violated = any(finding.check not in unresolved_checks for finding in findings)
    status: BoundaryStatus = (
        "violated" if violated else "unresolved" if findings else "certified"
    )
    boundary_degree_one = (
        status == "certified"
        and volume_component_count == 1
        and len(components) == 1
        and shell_euler == (2,)
        and len(shell_measures) == 1
        and shell_measures[0] > 0
        and tuple(degrees) == (0,)
    )
    return OrientedBoundary(
        status,
        face_tuple,
        triangles,
        components,
        shell_measures,
        tuple(degrees),
        boundary_degree_one,
        tuple(findings),
    )


__all__ = [
    "BoundaryFace",
    "BoundaryFinding",
    "BoundaryStatus",
    "BoundaryTriangle",
    "OrientedBoundary",
    "build_oriented_boundary",
]
