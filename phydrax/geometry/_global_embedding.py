#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Input-bound global cell embedding certificates from exact contact proofs."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from fractions import Fraction
from itertools import chain, combinations, product
from typing import cast, Literal

import numpy as np
from numpy.typing import ArrayLike

from .._bvh import bvh_overlap_pair_blocks, prepare_bvh
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._cell_geometry import (
    _coordinate_coefficient_identity,
    CellGeometrySpec,
    PerCellRestrictedCellGeometryElement,
)
from ..discretization._cell_geometry_validity import (
    cell_geometry_id,
    CellValidityCertificate,
    CellValidityStatus,
    certify_cell_geometry_validity,
)
from ..discretization._cell_mesh import CellMesh
from ..discretization._coordinate_enclosure import (
    add,
    affine_map,
    axes,
    bernstein_coefficients,
    BernsteinDomain,
    compose_expression,
    constant,
    evaluate_expression,
    ExactScalar as CoordinateExactScalar,
    Expression,
    expression_equal,
    linear_combinations,
    multiply,
    normalize_pyramid_reference_expression,
    polynomial_bounds,
    pyramid_reference_to_box_arguments,
    RationalPolynomial,
    scale,
    subtract,
)
from ..discretization._exact_algebraic_geometry import ExactScalar, RadicalElement
from ..discretization._nested_reference import (
    nested_cell_reference_maps,
    validate_reference_cell_map_witnesses,
)
from ..discretization._periodic_topology import PeriodicMeshTopology
from ..discretization._reference_cell import reference_cell_topology
from ..linalg._exact import factor_exact_sparse, solve_exact
from ._boundary_degree_embedding import build_oriented_boundary
from ._embedding_cells import (
    EmbeddingCell,
    prepare_embedding_cells,
    UnsupportedEmbeddingMap,
)
from ._embedding_state import (
    CoordinateMapScope,
    EmbeddingCounts,
    EmbeddingState,
    MeshCertificateFinding,
    MeshEmbeddingWitness,
    PeriodicAffineTorusDegreeEvidence,
    PeriodicMappedTorusDegreeEvidence,
    PeriodicRestrictionPartitionEvidence,
)
from ._exact_polyhedral_geometry import (
    determinant3,
    Point2,
    Point3,
    point_on_segment2,
    point_on_segment3,
    segment_contact_points3,
    tetrahedron_contact_points,
    triangle_contact_points2,
    triangle_contact_points3,
)
from ._mapped_embedding import prove_mapped_cell_injective
from ._mesh_certificates import (
    certificate_status,
    MeshCertificateBinding,
    MeshCertificateStatus,
)
from ._periodic_restriction import (
    certify_periodic_restriction_partition,
    PeriodicRestrictionPremise,
)
from ._planar_chain_embedding import (
    point_in_convex_planar_hull,
    triangulate_planar_polygon,
)
from ._reference_chain_embedding import (
    certify_reference_chain_embedding,
    ReferenceChainEmbeddingCertificate,
)
from ._restricted_embedding import _trace_arguments, restriction_trace_equal


class _SingularReferenceFactorMap(ValueError):
    """Incident reference facets do not provide independent coordinates."""


def cell_global_ids(mesh: CellMesh, /) -> np.ndarray:
    """Return stable top-cell identities in canonical block order."""
    return np.concatenate(
        tuple(np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks)
    )


def _point2(point: tuple[ExactScalar, ...], /) -> Point2:
    if len(point) != 2:
        raise UnsupportedEmbeddingMap(
            "An exact planar simplex point must have two coordinates."
        )
    return point[0], point[1]


def _point3(point: tuple[ExactScalar, ...], /) -> Point3:
    if len(point) != 3:
        raise UnsupportedEmbeddingMap(
            "An exact spatial simplex point must have three coordinates."
        )
    return point[0], point[1], point[2]


def _triangle2(
    points: tuple[tuple[ExactScalar, ...], ...], /
) -> tuple[Point2, Point2, Point2]:
    if len(points) != 3:
        raise UnsupportedEmbeddingMap("Exact planar contact requires a triangle.")
    first, second, third = points
    return _point2(first), _point2(second), _point2(third)


def _triangle3(
    points: tuple[tuple[ExactScalar, ...], ...], /
) -> tuple[Point3, Point3, Point3]:
    if len(points) != 3:
        raise UnsupportedEmbeddingMap("Exact spatial contact requires a triangle.")
    first, second, third = points
    return _point3(first), _point3(second), _point3(third)


def _tetrahedron3(
    points: tuple[tuple[ExactScalar, ...], ...], /
) -> tuple[Point3, Point3, Point3, Point3]:
    if len(points) != 4:
        raise UnsupportedEmbeddingMap("Exact volume contact requires a tetrahedron.")
    first, second, third, fourth = points
    return _point3(first), _point3(second), _point3(third), _point3(fourth)


def _sign(value: ExactScalar, /) -> int:
    if isinstance(value, RadicalElement):
        return value.sign()
    return (value > 0) - (value < 0)


def _local_strata(cell: EmbeddingCell, /) -> tuple[frozenset[int], ...]:
    result: set[frozenset[int]] = {
        frozenset((identifier,)) for identifier in cell.vertex_ids
    }
    if cell.dimension == 1:
        return tuple(result)
    if cell.kind in ("polygon", "polyhedron"):
        for face in cell.facet_routes:
            ids = tuple(cell.vertex_ids[index] for index in face)
            result.add(frozenset(ids))
            if cell.dimension == 3:
                result.update(
                    frozenset((ids[index], ids[(index + 1) % len(ids)]))
                    for index in range(len(ids))
                )
        return tuple(result)
    topology = reference_cell_topology(cell.kind)
    for degree in range(cell.dimension):
        for row in topology.entities[degree]:
            result.add(frozenset(cell.vertex_ids[int(index)] for index in row))
    return tuple(result)


def _point_in_declared_stratum(
    point: tuple[ExactScalar, ...],
    shared_ids: tuple[int, ...],
    points_by_id: dict[int, tuple[ExactScalar, ...]],
    /,
) -> bool:
    if len(shared_ids) == 1:
        return point == points_by_id[shared_ids[0]]
    vertices = tuple(points_by_id[identifier] for identifier in shared_ids)
    dimension = len(vertices[0])
    if any(len(vertex) != dimension for vertex in vertices):
        raise ValueError("Shared vertices do not have a common ambient dimension.")
    if dimension == 2:
        if len(point) == 3:
            if point[2] != 0:
                return False
            point = point[0], point[1]
        if len(point) != 2:
            raise ValueError(
                "Exact contact point does not match the planar shared stratum."
            )
        planar_point = _point2(point)
        planar_vertices = tuple(_point2(vertex) for vertex in vertices)
        if len(planar_vertices) == 2:
            return point_on_segment2(planar_point, planar_vertices[0], planar_vertices[1])
        return point_in_convex_planar_hull(planar_point, planar_vertices)
    if dimension == 3:
        if len(point) != 3:
            raise ValueError(
                "Exact contact point does not match the spatial shared stratum."
            )
        spatial_point = _point3(point)
        spatial_vertices = tuple(_point3(vertex) for vertex in vertices)
        if len(spatial_vertices) == 2:
            return point_on_segment3(
                spatial_point, spatial_vertices[0], spatial_vertices[1]
            )
        return point_in_convex_planar_hull(spatial_point, spatial_vertices)
    raise ValueError(
        "Exact shared-stratum membership requires ambient dimension two or three."
    )


def _contact_is_declared(
    first: EmbeddingCell,
    second: EmbeddingCell,
    points: tuple[tuple[ExactScalar, ...], ...],
    first_ids: tuple[int, ...],
    second_ids: tuple[int, ...],
    /,
) -> bool | None:
    if not points:
        return True
    shared = tuple(sorted(set(first_ids).intersection(second_ids)))
    if not shared:
        return False
    common = frozenset(shared)
    if common not in _local_strata(first) or common not in _local_strata(second):
        return False
    point_by_id = dict(zip(first_ids, first.corner_images, strict=True))
    if any(identifier not in point_by_id for identifier in shared):
        return False
    try:
        return all(
            _point_in_declared_stratum(point, shared, point_by_id) for point in points
        )
    except (ValueError, ZeroDivisionError):
        return None


def _polygon_cycle(cell: EmbeddingCell, /) -> tuple[int, ...]:
    adjacency: dict[int, list[int]] = defaultdict(list)
    for edge in cell.facet_routes:
        if len(edge) != 2:
            raise UnsupportedEmbeddingMap(
                "A polygon boundary edge has invalid incidence."
            )
        first, second = edge
        adjacency[first].append(second)
        adjacency[second].append(first)
    if len(adjacency) != len(cell.corner_images) or any(
        len(values) != 2 for values in adjacency.values()
    ):
        raise UnsupportedEmbeddingMap(
            "Polygon edges do not form one closed boundary loop."
        )
    start = 0
    cycle = [start]
    previous = -1
    current = start
    while True:
        next_vertex = next(
            (value for value in adjacency[current] if value != previous), None
        )
        if next_vertex is None:
            raise UnsupportedEmbeddingMap(
                "Polygon edge incidence does not close into a loop."
            )
        if next_vertex == start:
            break
        if next_vertex in cycle:
            raise UnsupportedEmbeddingMap(
                "Polygon edge incidence contains multiple loops."
            )
        cycle.append(next_vertex)
        previous, current = current, next_vertex
    if len(cycle) != len(cell.corner_images):
        raise UnsupportedEmbeddingMap(
            "Polygon boundary does not cover every cell vertex."
        )
    return tuple(cycle)


def _polygon_triangles(cell: EmbeddingCell, /) -> tuple[tuple[int, ...], ...]:
    if cell.dimension != 2:
        raise UnsupportedEmbeddingMap(
            "Polygon triangulation requires a two-dimensional cell."
        )
    cycle = _polygon_cycle(cell)
    if len(cycle) == 3:
        return (cycle,)
    points = tuple(cell.corner_images[index] for index in cycle)
    return tuple(
        tuple(cycle[index] for index in triangle)
        for triangle in triangulate_planar_polygon(points)
    )


def _volume_tetrahedra(
    cell: EmbeddingCell, /
) -> tuple[tuple[Point3, Point3, Point3, Point3], ...]:
    if cell.dimension != 3 or any(len(point) != 3 for point in cell.corner_images):
        raise UnsupportedEmbeddingMap(
            "Exact volume contact requires a three-dimensional cell in R3."
        )
    if cell.kind in ("tetrahedron", "simplex:3"):
        return (_tetrahedron3(cell.corner_images),)
    if not cell.facet_routes:
        raise UnsupportedEmbeddingMap(
            "A linear polytope has no exact face triangulation."
        )
    edge_incidence: dict[frozenset[int], int] = defaultdict(int)
    center: Point3 = (
        sum((point[0] for point in cell.corner_images), Fraction(0))
        / len(cell.corner_images),
        sum((point[1] for point in cell.corner_images), Fraction(0))
        / len(cell.corner_images),
        sum((point[2] for point in cell.corner_images), Fraction(0))
        / len(cell.corner_images),
    )
    face_triangles: list[tuple[int, int, int]] = []
    for route in cell.facet_routes:
        if len(route) < 3 or len(set(route)) != len(route):
            raise UnsupportedEmbeddingMap("A polytope face has invalid vertex incidence.")
        ids = tuple(cell.vertex_ids[index] for index in route)
        for index in range(len(ids)):
            edge_incidence[frozenset((ids[index], ids[(index + 1) % len(ids)]))] += 1
        face_points = tuple(cell.corner_images[index] for index in route)
        triangles = triangulate_planar_polygon(face_points)
        origin, first, second = (
            _point3(face_points[0]),
            _point3(face_points[1]),
            _point3(face_points[2]),
        )
        normal = (
            (first[1] - origin[1]) * (second[2] - origin[2])
            - (first[2] - origin[2]) * (second[1] - origin[1]),
            (first[2] - origin[2]) * (second[0] - origin[0])
            - (first[0] - origin[0]) * (second[2] - origin[2]),
            (first[0] - origin[0]) * (second[1] - origin[1])
            - (first[1] - origin[1]) * (second[0] - origin[0]),
        )
        side_signs = tuple(
            _sign(
                sum(
                    (normal[axis] * (point[axis] - origin[axis]) for axis in range(3)),
                    Fraction(0),
                )
            )
            for point in cell.corner_images
        )
        if any(
            _sign(
                sum(
                    (normal[axis] * (point[axis] - origin[axis]) for axis in range(3)),
                    Fraction(0),
                )
            )
            for point in face_points
        ):
            raise UnsupportedEmbeddingMap("A polytope face is not exactly planar.")
        nonzero_sides = {value for value in side_signs if value}
        center_side = _sign(
            sum(
                (normal[axis] * (center[axis] - origin[axis]) for axis in range(3)),
                Fraction(0),
            )
        )
        if (
            len(nonzero_sides) != 1
            or center_side == 0
            or center_side not in nonzero_sides
        ):
            raise UnsupportedEmbeddingMap(
                "Polytope face planes do not prove a strict convex interior."
            )
        face_triangles.extend(
            (route[first_index], route[second_index], route[third_index])
            for first_index, second_index, third_index in triangles
        )
    if not edge_incidence or any(count != 2 for count in edge_incidence.values()):
        raise UnsupportedEmbeddingMap(
            "Polytope boundary is not a closed two-manifold face complex."
        )
    tetrahedra: list[tuple[Point3, Point3, Point3, Point3]] = []
    for first_index, second_index, third_index in face_triangles:
        tetrahedron = (
            center,
            _point3(cell.corner_images[first_index]),
            _point3(cell.corner_images[second_index]),
            _point3(cell.corner_images[third_index]),
        )
        if not determinant3(*tetrahedron):
            raise UnsupportedEmbeddingMap(
                "A polytope face and its interior point form a degenerate tetrahedron."
            )
        tetrahedra.append(tetrahedron)
    if not tetrahedra:
        raise UnsupportedEmbeddingMap(
            "A polytope has no nondegenerate volume decomposition."
        )
    return tuple(tetrahedra)


def _cell_simplices(
    cell: EmbeddingCell, /
) -> tuple[tuple[tuple[ExactScalar, ...], ...], ...]:
    if not cell.affine and cell.coordinates:
        raise UnsupportedEmbeddingMap(
            "A curved map does not have an exact affine simplex decomposition."
        )
    if cell.dimension == 1:
        if len(cell.corner_images) != 2:
            raise UnsupportedEmbeddingMap(
                "An interval map requires two exact endpoint images."
            )
        return ((cell.corner_images[0], cell.corner_images[1]),)
    if cell.dimension == 2:
        return tuple(
            tuple(cell.corner_images[index] for index in triangle)
            for triangle in _polygon_triangles(cell)
        )
    if cell.dimension == 3:
        return _volume_tetrahedra(cell)
    raise UnsupportedEmbeddingMap(
        "Global embedding supports cell dimensions one through three."
    )


def _piece_contacts(
    first: tuple[tuple[ExactScalar, ...], ...],
    second: tuple[tuple[ExactScalar, ...], ...],
    dimension: int,
    ambient: int,
    /,
) -> tuple[tuple[ExactScalar, ...], ...]:
    if dimension == 1:
        if len(first) != 2 or len(second) != 2:
            raise UnsupportedEmbeddingMap(
                "Exact segment contact requires two endpoints per cell."
            )
        first_start, first_end = first
        second_start, second_end = second
        if ambient == 2:
            first_start2, first_end2 = _point2(first_start), _point2(first_end)
            second_start2, second_end2 = _point2(second_start), _point2(second_end)
            return segment_contact_points3(
                (first_start2[0], first_start2[1], Fraction(0)),
                (first_end2[0], first_end2[1], Fraction(0)),
                (second_start2[0], second_start2[1], Fraction(0)),
                (second_end2[0], second_end2[1], Fraction(0)),
            )
        if ambient == 3:
            return segment_contact_points3(
                _point3(first_start),
                _point3(first_end),
                _point3(second_start),
                _point3(second_end),
            )
        raise ValueError(
            "Exact segment contacts require an ambient dimension of two or three."
        )
    if dimension == 2 and ambient == 2:
        return triangle_contact_points2(_triangle2(first), _triangle2(second))
    if dimension == 2 and ambient == 3:
        return triangle_contact_points3(_triangle3(first), _triangle3(second))
    if dimension == 3 and ambient == 3:
        return tetrahedron_contact_points(_tetrahedron3(first), _tetrahedron3(second))
    raise ValueError("Exact cell contact route is unavailable for these dimensions.")


def _junction_ids(mesh: CellMesh, values: ArrayLike | None, /) -> tuple[int, ...]:
    if values is None:
        return ()
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise TypeError("junction_vertices must be a one-dimensional integer array.")
    if raw.size == 0:
        return ()
    if raw.dtype.kind not in "iu":
        raise TypeError("junction_vertices must contain integer IDs.")
    requested = tuple(int(value) for value in raw.tolist())
    if any(value < 0 for value in requested):
        raise ValueError("junction vertex IDs must be nonnegative.")
    if len(set(requested)) != len(requested):
        raise ValueError("junction_vertices must contain unique global IDs.")
    present = set(
        int(value) for value in np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    )
    if not set(requested) <= present:
        raise ValueError("junction_vertices must name current mesh vertices.")
    return tuple(sorted(requested))


def _periodic_input(
    mesh: CellMesh, periodic: PeriodicMeshTopology | None, /
) -> PeriodicMeshTopology | None:
    current = mesh.periodic_topology
    if periodic is not None and not isinstance(periodic, PeriodicMeshTopology):
        raise TypeError("periodic must be PeriodicMeshTopology or None.")
    selected = current if periodic is None else periodic
    if selected is not current:
        if current is None or selected is None:
            raise ValueError(
                "periodic must be the quotient descriptor bound to this mesh."
            )
    if selected is not None:
        selected.require_lifted(mesh.topology, mesh.topology_id, mesh.blocks)
        selected.require_coordinates(mesh.coordinates)
    return selected


@dataclass(frozen=True, slots=True)
class _ProofOutcome:
    findings: tuple[MeshCertificateFinding, ...]
    checks: tuple[str, ...]
    counts: EmbeddingCounts
    witnesses: tuple[MeshEmbeddingWitness, ...]
    periodic_affine_degree: PeriodicAffineTorusDegreeEvidence | None = None
    periodic_restriction: PeriodicRestrictionPartitionEvidence | None = None
    periodic_mapped_degree: PeriodicMappedTorusDegreeEvidence | None = None


@dataclass(frozen=True, slots=True)
class _RootChart:
    source_geometry_id: str
    source_topology_id: str
    source_cell_id: int
    source_cell_kind: str
    arguments: tuple[Expression, ...]
    coordinates: tuple[Expression, ...]
    reference_cell: EmbeddingCell
    chart_cell: EmbeddingCell | None
    injective: bool


def _reference_domain(kind: str) -> BernsteinDomain | None:
    if kind in ("interval", "triangle", "tetrahedron") or kind.startswith("simplex:"):
        return "simplex"
    if kind == "prism":
        return "prism"
    if kind in ("quadrilateral", "hexahedron") or kind.startswith("tensor:"):
        return "box"
    return None


def _exact_reference_vertices(kind: str) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(
        tuple(Fraction(float(value)) for value in point)
        for point in reference_cell_topology(kind).vertices
    )


def _full_column_rank(matrix: tuple[tuple[ExactScalar, ...], ...], /) -> bool:
    if not matrix or not matrix[0]:
        return False
    rows = [list(row) for row in matrix]
    columns = len(rows[0])
    if any(len(row) != columns for row in rows):
        return False
    pivot_row = 0
    for column in range(columns):
        selected = next(
            (row for row in range(pivot_row, len(rows)) if rows[row][column] != 0),
            None,
        )
        if selected is None:
            continue
        rows[pivot_row], rows[selected] = rows[selected], rows[pivot_row]
        pivot = rows[pivot_row][column]
        for row in range(pivot_row + 1, len(rows)):
            if rows[row][column] == 0:
                continue
            ratio = rows[row][column] / pivot
            for index in range(column, columns):
                rows[row][index] -= ratio * rows[pivot_row][index]
        pivot_row += 1
        if pivot_row == len(rows):
            break
    return pivot_row == columns


def _root_reference_cell(
    kind: str, coordinates: tuple[Expression, ...], global_id: int, /
) -> EmbeddingCell | None:
    domain = _reference_domain(kind)
    topology = reference_cell_topology(kind)
    if domain is None or topology.dimension < 1:
        return None
    references = _exact_reference_vertices(kind)
    if any(
        len(expression) > 0
        and any(len(index) != topology.dimension for index in expression)
        for expression in coordinates
        if not isinstance(expression, RationalPolynomial)
    ):
        return None
    try:
        corners = tuple(
            tuple(evaluate_expression(expression, point) for expression in coordinates)
            for point in references
        )
        bounds = tuple(
            polynomial_bounds(expression, domain, topology.dimension)
            for expression in coordinates
        )
    except (ValueError, ZeroDivisionError):
        return None
    return EmbeddingCell(
        global_id,
        "root-reference",
        0,
        kind,
        topology.dimension,
        domain,
        tuple(range(len(references))),
        tuple(range(len(references))),
        references,
        coordinates,
        corners,
        None,
        None,
        tuple(value[0] for value in bounds),
        tuple(value[1] for value in bounds),
        tuple(
            tuple(int(index) for index in face)
            for face in topology.entities[topology.dimension - 1]
        ),
    )


def _root_map_injective(cell: EmbeddingCell, /) -> bool:
    dimension = cell.dimension
    try:
        affine = affine_map(cell.coordinates, dimension)
    except (ValueError, ZeroDivisionError):
        affine = None
    if affine is not None and _full_column_rank(affine[1]):
        return True
    try:
        return prove_mapped_cell_injective(cell) is not None
    except (ValueError, ZeroDivisionError):
        return False


def _affine_chart_cell(
    cell: EmbeddingCell, arguments: tuple[Expression, ...], /
) -> EmbeddingCell | None:
    try:
        affine = affine_map(arguments, cell.dimension)
    except (ValueError, ZeroDivisionError):
        return None
    if affine is None:
        return None
    try:
        corners = tuple(
            tuple(evaluate_expression(expression, point) for expression in arguments)
            for point in cell.reference_vertices
        )
        proof_arguments = tuple(
            _proof_polynomial(cell, expression) for expression in arguments
        )
        if len(proof_arguments) != len(arguments) or any(
            expression is None for expression in proof_arguments
        ):
            return None
        bounds = tuple(
            polynomial_bounds(expression, cell.proof_domain, cell.dimension)
            for expression in proof_arguments
            if expression is not None
        )
    except (ValueError, ZeroDivisionError):
        return None
    return EmbeddingCell(
        cell.global_id,
        cell.block_name,
        cell.local_index,
        cell.kind,
        cell.dimension,
        cell.domain,
        cell.vertex_indices,
        cell.vertex_ids,
        cell.reference_vertices,
        arguments,
        corners,
        affine[0],
        affine[1],
        tuple(value[0] for value in bounds),
        tuple(value[1] for value in bounds),
        cell.facet_routes,
    )


def _inverse_affine_arguments(
    arguments: tuple[Expression, ...], dimension: int, /
) -> tuple[Expression, ...] | None:
    affine = affine_map(arguments, dimension)
    if affine is None:
        return None
    offset, matrix = affine
    if (
        len(offset) != dimension
        or len(matrix) != dimension
        or any(len(row) != dimension for row in matrix)
        or not _full_column_rank(matrix)
    ):
        return None
    augmented = [
        list(row) + [Fraction(int(row_index == column)) for column in range(dimension)]
        for row_index, row in enumerate(matrix)
    ]
    for column in range(dimension):
        selected = next(
            (row for row in range(column, dimension) if augmented[row][column] != 0),
            None,
        )
        if selected is None:
            return None
        augmented[column], augmented[selected] = (
            augmented[selected],
            augmented[column],
        )
        pivot = augmented[column][column]
        augmented[column] = [value / pivot for value in augmented[column]]
        for row in range(dimension):
            if row == column or augmented[row][column] == 0:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(
                    augmented[row], augmented[column], strict=True
                )
            ]
    inverse = tuple(tuple(row[dimension:]) for row in augmented)
    zero = (0,) * dimension
    result = []
    for row in range(dimension):
        value: Expression = {
            zero: -sum(
                (inverse[row][column] * offset[column] for column in range(dimension)),
                Fraction(0),
            )
        }
        for column in range(dimension):
            coefficient = inverse[row][column]
            if coefficient:
                exponent = tuple(int(index == column) for index in range(dimension))
                value = add(value, {exponent: coefficient})
        result.append(value)
    return tuple(result)


def _retained_root_charts(
    mesh: CellMesh, geometry: CellGeometrySpec, cells: tuple[EmbeddingCell, ...], /
) -> tuple[_RootChart | None, ...]:
    """Revalidate source-cell charts and their exact physical compositions."""
    source = geometry.restriction_source
    if source is None:
        return tuple(None for _ in cells)
    try:
        elements, _, _ = geometry.resolve(mesh)
    except (ValueError, TypeError, IndexError):
        return tuple(None for _ in cells)
    chart_witnesses = dict(
        zip(source.block_names, source.parent_reference_maps, strict=True)
    )
    parent_ids = dict(
        zip(
            source.block_names,
            (np.asarray(value, dtype=np.int64) for value in source.parent_cell_ids),
            strict=True,
        )
    )
    cells_by_row = {
        (cell.block_name, cell.local_index): index for index, cell in enumerate(cells)
    }
    result: list[_RootChart | None] = [None] * len(cells)
    for block, element in zip(mesh.blocks, elements, strict=True):
        witness = chart_witnesses.get(block.name)
        if witness is None:
            continue
        try:
            validate_reference_cell_map_witnesses(witness)
            maps = nested_cell_reference_maps(witness)
        except (TypeError, ValueError, ZeroDivisionError):
            continue
        if (
            witness.target_cell_kind != block.cell_kind
            or not np.array_equal(
                np.asarray(witness.target_cell_ids, dtype=np.int64),
                np.arange(block.cell_count, dtype=np.int64),
            )
            or not np.array_equal(
                np.asarray(witness.source_cell_ids, dtype=np.int64),
                parent_ids[block.name],
            )
        ):
            continue
        maps_by_target = {value.target_cell_id: value for value in maps}
        for row, target_id in enumerate(np.asarray(block.global_ids, dtype=np.int64)):
            cell_index = cells_by_row.get((block.name, row))
            if cell_index is None or int(target_id) != cells[cell_index].global_id:
                continue
            chart = maps_by_target.get(row)
            if chart is None or chart.source_cell_id != int(parent_ids[block.name][row]):
                continue
            cell = cells[cell_index]
            if chart.arguments is None:
                continue
            arguments = cast(tuple[Expression, ...], tuple(chart.arguments))
            if isinstance(element, PerCellRestrictedCellGeometryElement):
                try:
                    element_arguments = element.chart_arguments(row)
                    if len(arguments) != len(element_arguments) or not all(
                        expression_equal(first, second)
                        for first, second in zip(
                            arguments, element_arguments, strict=True
                        )
                    ):
                        continue
                except (IndexError, TypeError, ValueError):
                    continue
            inverse = _inverse_affine_arguments(arguments, cell.dimension)
            if inverse is None:
                continue
            try:
                root_coordinates = tuple(
                    compose_expression(expression, inverse)
                    for expression in cell.coordinates
                )
                composed = tuple(
                    compose_expression(expression, arguments)
                    for expression in root_coordinates
                )
                if not all(
                    expression_equal(first, second)
                    for first, second in zip(cell.coordinates, composed, strict=True)
                ):
                    continue
            except (TypeError, ValueError, ZeroDivisionError):
                continue
            root_cell = _root_reference_cell(
                chart.source_cell_kind, root_coordinates, chart.source_cell_id
            )
            if root_cell is None:
                continue
            result[cell_index] = _RootChart(
                source.source_geometry_id,
                source.source_topology_id,
                chart.source_cell_id,
                chart.source_cell_kind,
                arguments,
                root_coordinates,
                root_cell,
                _affine_chart_cell(cell, arguments),
                _root_map_injective(root_cell),
            )
    return tuple(result)


class GlobalEmbeddingCertificate(StrictModule, NonTrainableState):
    """Computed global embedding theorem bound to its actual immutable inputs."""

    binding: MeshCertificateBinding
    validity: CellValidityCertificate
    status: MeshCertificateStatus
    findings: tuple[MeshCertificateFinding, ...]
    evaluated_checks: tuple[str, ...]
    counts: EmbeddingCounts
    witnesses: tuple[MeshEmbeddingWitness, ...]
    periodic: PeriodicMeshTopology | None
    periodic_affine_degree: PeriodicAffineTorusDegreeEvidence | None
    periodic_restriction: PeriodicRestrictionPartitionEvidence | None
    periodic_restriction_premise: PeriodicRestrictionPremise | None
    periodic_mapped_degree: PeriodicMappedTorusDegreeEvidence | None
    certificate_id: str

    def __init__(
        self,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        validity: CellValidityCertificate,
        /,
        *,
        junction_vertices: ArrayLike | None = None,
        periodic: PeriodicMeshTopology | None = None,
        periodic_restriction: PeriodicRestrictionPremise | None = None,
    ) -> None:
        """Compute every proof obligation from inputs; no caller-supplied verdict exists."""
        if not isinstance(mesh, CellMesh) or not isinstance(geometry, CellGeometrySpec):
            raise TypeError(
                "Global embedding requires CellMesh and CellGeometrySpec inputs."
            )
        if not isinstance(validity, CellValidityCertificate):
            raise TypeError("validity must be a CellValidityCertificate.")
        validity.require_current(geometry, mesh=mesh)
        junctions = _junction_ids(mesh, junction_vertices)
        periodic_topology = _periodic_input(mesh, periodic)
        if periodic_restriction is not None:
            if not isinstance(periodic_restriction, PeriodicRestrictionPremise):
                raise TypeError(
                    "periodic_restriction must be PeriodicRestrictionPremise or None."
                )
            if periodic_topology is None:
                raise ValueError(
                    "A periodic restriction premise requires a periodic target mesh."
                )
            transition = periodic_restriction.transition
            if transition is None:
                raise ValueError(
                    "A periodic restriction premise must retain an exact cell transition."
                )
            transition.require_bound(
                periodic_restriction.source_mesh,
                periodic_restriction.source_geometry,
                mesh,
                geometry,
            )
        prepared_cells: tuple[EmbeddingCell, ...] | None = None
        map_error: Exception | None = None
        if validity.all_certified:
            try:
                prepared_cells = prepare_embedding_cells(mesh, geometry)
            except (ValueError, ZeroDivisionError) as error:
                map_error = error
        scope: CoordinateMapScope = (
            "affine"
            if prepared_cells is not None
            and all(cell.affine or not cell.coordinates for cell in prepared_cells)
            else "mapped"
        )
        binding = MeshCertificateBinding(
            mesh, geometry, scope, junction_vertices=junctions
        )
        outcome = _proof(
            mesh,
            geometry,
            validity,
            binding,
            prepared_cells,
            map_error,
            periodic=periodic_topology,
            periodic_restriction=periodic_restriction,
        )
        degree_checks = {
            "periodic_affine_torus_manifold",
            "periodic_affine_torus_trace",
            "periodic_affine_torus_orientation",
            "periodic_affine_torus_degree_one",
        }
        degree_evidence = outcome.periodic_affine_degree
        if degree_evidence is not None:
            if (
                periodic_topology is None
                or (
                    degree_evidence.mesh_id,
                    degree_evidence.geometry_id,
                    degree_evidence.periodic_topology_id,
                )
                != (
                    mesh.mesh_id,
                    validity.geometry_id,
                    periodic_topology.periodic_topology_id,
                )
                or not degree_checks <= set(outcome.checks)
            ):
                raise ValueError(
                    "Periodic affine degree evidence must bind the current mesh, geometry, topology, and all theorem checks."
                )
            PeriodicAffineTorusDegreeEvidence.from_stored_fields(degree_evidence)
        elif (
            not outcome.findings and "periodic_affine_torus_degree_one" in outcome.checks
        ):
            raise ValueError(
                "A successful periodic affine degree route must retain its exact evidence."
            )
        restriction_checks = {
            "periodic_restriction_source_embedding",
            "periodic_restriction_parent_partition",
            "periodic_restriction_boundary_subdivision",
        }
        restriction_evidence = outcome.periodic_restriction
        if restriction_evidence is not None:
            if (
                periodic_topology is None
                or periodic_restriction is None
                or (
                    restriction_evidence.mesh_id,
                    restriction_evidence.geometry_id,
                    restriction_evidence.periodic_topology_id,
                    restriction_evidence.validity_certificate_id,
                    restriction_evidence.source_mesh_id,
                    restriction_evidence.source_geometry_id,
                    restriction_evidence.source_validity_certificate_id,
                    restriction_evidence.source_embedding_certificate_id,
                )
                != (
                    mesh.mesh_id,
                    validity.geometry_id,
                    periodic_topology.periodic_topology_id,
                    validity.certificate_id,
                    periodic_restriction.source_mesh.mesh_id,
                    periodic_restriction.source_validity.geometry_id,
                    periodic_restriction.source_validity.certificate_id,
                    periodic_restriction.source_embedding.certificate_id,
                )
                or not restriction_checks <= set(outcome.checks)
            ):
                raise ValueError(
                    "Periodic restriction evidence must bind the current target, source proof, topology, and all theorem checks."
                )
            PeriodicRestrictionPartitionEvidence.from_stored_fields(restriction_evidence)
        elif (
            not outcome.findings
            and "periodic_restriction_parent_partition" in outcome.checks
        ):
            raise ValueError(
                "A successful periodic restriction route must retain exact partition evidence."
            )
        mapped_degree_checks = {
            "periodic_mapped_torus_manifold",
            "periodic_mapped_torus_trace",
            "periodic_mapped_torus_orientation",
            "periodic_mapped_torus_degree_one",
        }
        mapped_degree_evidence = outcome.periodic_mapped_degree
        if mapped_degree_evidence is not None:
            if (
                periodic_topology is None
                or (
                    mapped_degree_evidence.mesh_id,
                    mapped_degree_evidence.geometry_id,
                    mapped_degree_evidence.periodic_topology_id,
                    mapped_degree_evidence.validity_certificate_id,
                )
                != (
                    mesh.mesh_id,
                    validity.geometry_id,
                    periodic_topology.periodic_topology_id,
                    validity.certificate_id,
                )
                or not mapped_degree_checks <= set(outcome.checks)
            ):
                raise ValueError(
                    "Mapped periodic degree evidence must bind the current mesh, geometry, topology, validity and all theorem checks."
                )
            PeriodicMappedTorusDegreeEvidence.from_stored_fields(mapped_degree_evidence)
        elif (
            not outcome.findings and "periodic_mapped_torus_degree_one" in outcome.checks
        ):
            raise ValueError(
                "A successful mapped periodic degree route must retain its exact evidence."
            )
        findings = outcome.findings
        self.binding = binding
        self.validity = validity
        self.status = certificate_status(findings)
        self.findings = findings
        self.evaluated_checks = outcome.checks
        self.counts = outcome.counts
        self.witnesses = outcome.witnesses
        self.periodic_affine_degree = outcome.periodic_affine_degree
        self.periodic_restriction = restriction_evidence
        self.periodic_restriction_premise = (
            periodic_restriction if restriction_evidence is not None else None
        )
        self.periodic = periodic_topology
        self.periodic_mapped_degree = mapped_degree_evidence
        identity = {
            "kind": "global-embedding-certificate",
            "binding": binding.binding_id,
            "validity": validity.certificate_id,
            "status": self.status,
            "findings": tuple(value.finding_id for value in findings),
            "checks": self.evaluated_checks,
            "inventory": (
                self.counts.cell_count,
                self.counts.source_bound_count,
                self.counts.boundary_facet_count,
                self.counts.shell_count,
            ),
            "witnesses": tuple(value.witness_id for value in self.witnesses),
            "periodic": (
                None
                if periodic_topology is None
                else periodic_topology.periodic_topology_id
            ),
        }
        if self.periodic_affine_degree is not None:
            identity["periodic_affine_degree"] = self.periodic_affine_degree.evidence_id
        if self.periodic_restriction is not None:
            identity["periodic_restriction"] = self.periodic_restriction.evidence_id
        if self.periodic_mapped_degree is not None:
            identity["periodic_mapped_degree"] = self.periodic_mapped_degree.evidence_id
        self.certificate_id = canonical_fingerprint(identity)

    @property
    def validity_certificate_id(self) -> str:
        return self.validity.certificate_id

    @property
    def cell_count(self) -> int:
        return self.counts.cell_count

    @property
    def source_bound_count(self) -> int:
        return self.counts.source_bound_count

    @property
    def boundary_facet_count(self) -> int:
        return self.counts.boundary_facet_count

    @property
    def shell_count(self) -> int:
        return self.counts.shell_count

    def require_current(
        self,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        /,
        *,
        validity: CellValidityCertificate | None = None,
    ) -> None:
        """Require the retained immutable proof to bind the current scientific inputs."""
        self.binding.require(mesh, geometry)
        local = self.validity if validity is None else validity
        local.require_bound(geometry, mesh=mesh)
        if local.certificate_id != self.validity_certificate_id:
            raise ValueError(
                "Global embedding certificate must retain the current local-validity proof."
            )
        current_periodic = _periodic_input(mesh, self.periodic)
        if (
            None if current_periodic is None else current_periodic.periodic_topology_id
        ) != (None if self.periodic is None else self.periodic.periodic_topology_id):
            raise ValueError(
                "Global embedding periodic topology must match its retained proof input."
            )
        if (self.periodic_restriction is None) != (
            self.periodic_restriction_premise is None
        ):
            raise ValueError(
                "Global embedding periodic restriction proof must retain its current transition premise."
            )

    @classmethod
    def from_stored_fields(
        cls, value: GlobalEmbeddingCertificate, /
    ) -> GlobalEmbeddingCertificate:
        """Restore by recomputing local validity and every global proof obligation."""
        if not isinstance(value, cls):
            raise TypeError(
                "Stored global embedding must be a GlobalEmbeddingCertificate."
            )
        value.binding.require(value.binding.mesh, value.binding.geometry)
        if value.periodic_affine_degree is not None:
            PeriodicAffineTorusDegreeEvidence.from_stored_fields(
                value.periodic_affine_degree
            )
        if value.periodic_restriction is not None:
            PeriodicRestrictionPartitionEvidence.from_stored_fields(
                value.periodic_restriction
            )
        if value.periodic_mapped_degree is not None:
            PeriodicMappedTorusDegreeEvidence.from_stored_fields(
                value.periodic_mapped_degree
            )
        rebuilt = certify_global_embedding(
            value.binding.mesh,
            value.binding.geometry,
            value.validity,
            junction_vertices=value.binding.junction_vertices,
            periodic=value.periodic,
            periodic_restriction=value.periodic_restriction_premise,
        )
        if (
            rebuilt.certificate_id != value.certificate_id
            or (
                None
                if rebuilt.periodic_affine_degree is None
                else rebuilt.periodic_affine_degree.evidence_id
            )
            != (
                None
                if value.periodic_affine_degree is None
                else value.periodic_affine_degree.evidence_id
            )
            or (
                None
                if rebuilt.periodic_restriction is None
                else rebuilt.periodic_restriction.evidence_id
            )
            != (
                None
                if value.periodic_restriction is None
                else value.periodic_restriction.evidence_id
            )
            or (
                None
                if rebuilt.periodic_mapped_degree is None
                else rebuilt.periodic_mapped_degree.evidence_id
            )
            != (
                None
                if value.periodic_mapped_degree is None
                else value.periodic_mapped_degree.evidence_id
            )
        ):
            raise ValueError(
                "Stored global embedding fields do not reproduce their mathematical proof."
            )
        return rebuilt


def _cell_map_findings(
    cells: tuple[EmbeddingCell, ...], /, *, binding: MeshCertificateBinding
) -> tuple[list[MeshCertificateFinding], list[MeshEmbeddingWitness]]:
    findings: list[MeshCertificateFinding] = []
    witnesses: list[MeshEmbeddingWitness] = []
    expressions: list[str] = []
    for cell in cells:
        if cell.affine or not cell.coordinates:
            continue
        try:
            witness = prove_mapped_cell_injective(cell)
        except (ValueError, ZeroDivisionError) as error:
            witness = None
            detail = str(error)
        else:
            detail = "Exact projected-Jacobian positivity was not decided by the available Bernstein controls."
        if witness is None:
            findings.append(
                MeshCertificateFinding(
                    "mapped_cell_injectivity",
                    "unresolved",
                    "cell",
                    (cell.global_id,),
                    detail=detail,
                )
            )
        else:
            projection_matrix = tuple(
                tuple(_coordinate_coefficient_identity(value) for value in row)
                for row in witness.projection_matrix
            )
            expressions.append(
                f"cell:{cell.global_id}:projection_rows:{witness.projection_rows}:"
                f"projection_matrix:{projection_matrix}:"
                f"controls:{witness.principal_minor_controls}"
            )
    if expressions:
        witnesses.append(
            MeshEmbeddingWitness(
                "mapped_cell_injective",
                (binding.binding_id,),
                tuple(
                    cell.global_id
                    for cell in cells
                    if not cell.affine and cell.coordinates
                ),
                tuple(expressions),
            )
        )
    return findings, witnesses


def _same_root_map(first: _RootChart, second: _RootChart, /) -> bool:
    return (
        first.source_geometry_id == second.source_geometry_id
        and first.source_topology_id == second.source_topology_id
        and first.source_cell_id == second.source_cell_id
        and first.source_cell_kind == second.source_cell_kind
        and len(first.coordinates) == len(second.coordinates)
        and all(
            expression_equal(left, right)
            for left, right in zip(first.coordinates, second.coordinates, strict=True)
        )
    )


def _one_dimensional_contacts(
    first: tuple[tuple[ExactScalar, ...], ...],
    second: tuple[tuple[ExactScalar, ...], ...],
    /,
) -> tuple[tuple[ExactScalar, ...], ...]:
    if len(first) != 2 or len(second) != 2:
        raise UnsupportedEmbeddingMap(
            "Exact interval contact requires two endpoints per chart."
        )
    lower = max(min(first[0][0], first[1][0]), min(second[0][0], second[1][0]))
    upper = min(max(first[0][0], first[1][0]), max(second[0][0], second[1][0]))
    if lower > upper:
        return ()
    if lower == upper:
        return ((lower,),)
    return ((lower,), (upper,))


def _root_contact_is_declared(
    first: EmbeddingCell,
    second: EmbeddingCell,
    points: tuple[tuple[ExactScalar, ...], ...],
    /,
) -> bool | None:
    if first.dimension != 1:
        return _contact_is_declared(
            first, second, points, first.vertex_ids, second.vertex_ids
        )
    if not points:
        return True
    shared = tuple(sorted(set(first.vertex_ids).intersection(second.vertex_ids)))
    if len(shared) != 1:
        return False
    first_point = first.corner_images[first.vertex_ids.index(shared[0])]
    second_point = second.corner_images[second.vertex_ids.index(shared[0])]
    if first_point != second_point:
        return False
    return all(point == first_point for point in points)


def _root_chart_contact(
    first: _RootChart, second: _RootChart, /
) -> Literal["declared", "outside", "unresolved"] | None:
    if not (
        first.injective
        and second.injective
        and _same_root_map(first, second)
        and first.chart_cell is not None
        and second.chart_cell is not None
    ):
        return None
    first_cell, second_cell = first.chart_cell, second.chart_cell
    if first_cell.dimension != second_cell.dimension or len(first.coordinates) != len(
        second.coordinates
    ):
        return None
    try:
        first_pieces = _cell_simplices(first_cell)
        second_pieces = _cell_simplices(second_cell)
    except (ValueError, ZeroDivisionError, UnsupportedEmbeddingMap):
        return None
    unresolved = False
    for left in first_pieces:
        for right in second_pieces:
            try:
                contacts = (
                    _one_dimensional_contacts(left, right)
                    if first_cell.dimension == 1
                    else _piece_contacts(
                        left,
                        right,
                        first_cell.dimension,
                        len(first.coordinates),
                    )
                )
                declared = _root_contact_is_declared(first_cell, second_cell, contacts)
            except (ValueError, ZeroDivisionError, UnsupportedEmbeddingMap):
                unresolved = True
                continue
            if declared is None:
                unresolved = True
            elif not declared:
                return "outside"
    return "unresolved" if unresolved else "declared"


def _candidate_normals(
    first: EmbeddingCell, second: EmbeddingCell, /
) -> Iterator[tuple[ExactScalar, ...]]:
    ambient = len(first.corner_images[0])
    for axis in range(ambient):
        yield tuple(Fraction(int(index == axis)) for index in range(ambient))
    if ambient not in (2, 3):
        return
    first_points = first.corner_images
    second_points = second.corner_images
    edge_directions = tuple(
        tuple(right[axis] - left[axis] for axis in range(ambient))
        for points in (first_points, second_points)
        for left, right in combinations(points, 2)
    )
    if ambient == 2:
        for direction in edge_directions:
            yield -direction[1], direction[0]
        return
    for left, right in combinations(edge_directions, 2):
        normal = (
            left[1] * right[2] - left[2] * right[1],
            left[2] * right[0] - left[0] * right[2],
            left[0] * right[1] - left[1] * right[0],
        )
        if any(_sign(value) for value in normal):
            yield normal
    for left in first_points:
        for right in second_points:
            direction = tuple(right[axis] - left[axis] for axis in range(ambient))
            if any(_sign(value) for value in direction):
                yield direction


def _unique_projective_normals(
    normals: Iterator[tuple[ExactScalar, ...]], /
) -> Iterator[tuple[ExactScalar, ...]]:
    """Yield one exactly normalized representative of each candidate plane."""
    seen: list[tuple[ExactScalar, ...]] = []
    for normal in normals:
        leading = next((value for value in normal if _sign(value)), None)
        if leading is None:
            continue
        normalized = tuple(value / leading for value in normal)
        if any(normalized == previous for previous in seen):
            continue
        seen.append(normalized)
        yield normalized


def _projected_expression(
    cell: EmbeddingCell,
    normal: tuple[ExactScalar, ...],
    offset: ExactScalar,
    /,
) -> Expression:
    return subtract(
        linear_combinations(cell.coordinates, (normal,))[0],
        constant(offset, cell.dimension),
    )


def _strict_image_separation(first: EmbeddingCell, second: EmbeddingCell, /) -> bool:
    if (
        not first.coordinates
        or not second.coordinates
        or len(first.corner_images[0]) != len(second.corner_images[0])
    ):
        return False
    for normal in _unique_projective_normals(_candidate_normals(first, second)):
        if not any(_sign(value) for value in normal):
            continue
        try:
            first_range = polynomial_bounds(
                linear_combinations(first.proof_coordinates, (normal,))[0],
                first.proof_domain,
                first.dimension,
            )
            second_range = polynomial_bounds(
                linear_combinations(second.proof_coordinates, (normal,))[0],
                second.proof_domain,
                second.dimension,
            )
        except (ValueError, ZeroDivisionError):
            continue
        if first_range[1] < second_range[0] or second_range[1] < first_range[0]:
            return True
    return False


def _solve_affine_values(
    points: tuple[tuple[Fraction, ...], ...],
    values: tuple[Fraction, ...],
    /,
) -> tuple[Fraction, ...] | None:
    if not points or len(points) != len(values):
        return None
    dimension = len(points[0])
    rows = [
        [Fraction(1), *point, value] for point, value in zip(points, values, strict=True)
    ]
    if any(len(point) != dimension for point in points):
        return None
    pivot_row = 0
    for column in range(dimension + 1):
        selected = next(
            (row for row in range(pivot_row, len(rows)) if rows[row][column]),
            None,
        )
        if selected is None:
            return None
        rows[pivot_row], rows[selected] = rows[selected], rows[pivot_row]
        pivot = rows[pivot_row][column]
        rows[pivot_row] = [value / pivot for value in rows[pivot_row]]
        for row in range(len(rows)):
            if row == pivot_row or not rows[row][column]:
                continue
            factor = rows[row][column]
            rows[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(rows[row], rows[pivot_row], strict=True)
            ]
        pivot_row += 1
    if any(not any(row[: dimension + 1]) and row[-1] for row in rows):
        return None
    coefficients = [Fraction(0) for _ in range(dimension + 1)]
    for row in rows:
        pivot = next(
            (column for column, value in enumerate(row[: dimension + 1]) if value),
            None,
        )
        if pivot is not None:
            coefficients[pivot] = row[-1]
    if any(
        coefficients[0]
        + sum(
            (
                coefficient * point[axis]
                for axis, coefficient in enumerate(coefficients[1:])
            ),
            Fraction(0),
        )
        != value
        for point, value in zip(points, values, strict=True)
    ):
        return None
    return tuple(coefficients)


def _reference_facet_factor(
    cell: EmbeddingCell, shared_ids: tuple[int, ...], /
) -> dict[tuple[int, ...], Fraction] | None:
    dimension = cell.dimension
    face_indices = next(
        (
            tuple(face)
            for face in cell.facet_routes
            if {cell.vertex_ids[index] for index in face} == set(shared_ids)
        ),
        None,
    )
    if face_indices is None:
        return None
    values = tuple(
        Fraction(0 if index in face_indices else 1)
        for index in range(len(cell.reference_vertices))
    )
    coefficients = _solve_affine_values(cell.reference_vertices, values)
    if coefficients is None:
        return None
    factor: dict[tuple[int, ...], Fraction] = {}
    zero = (0,) * dimension
    if coefficients[0]:
        factor[zero] = coefficients[0]
    for axis, coefficient in enumerate(coefficients[1:]):
        if coefficient:
            exponent = tuple(int(index == axis) for index in range(dimension))
            factor[exponent] = coefficient
    return factor or None


def _reference_edge_factors(
    cell: EmbeddingCell, shared_ids: tuple[int, ...], /
) -> (
    tuple[
        dict[tuple[int, ...], Fraction],
        dict[tuple[int, ...], Fraction],
    ]
    | None
):
    """Return the two nonnegative reference-face factors that define an edge."""
    if len(shared_ids) != 2:
        return None
    try:
        shared_vertices = {cell.vertex_ids.index(identifier) for identifier in shared_ids}
    except ValueError:
        return None
    common_vertices = set(range(len(cell.vertex_ids)))
    factors: list[dict[tuple[int, ...], Fraction]] = []
    for face in cell.facet_routes:
        face_vertices = set(face)
        if not shared_vertices <= face_vertices:
            continue
        face_ids = tuple(cell.vertex_ids[index] for index in face)
        factor = _reference_facet_factor(cell, face_ids)
        if factor is None:
            return None
        factors.append(factor)
        common_vertices.intersection_update(face_vertices)
    if len(factors) != 2 or common_vertices != shared_vertices:
        return None
    return factors[0], factors[1]


def _reference_vertex_factors(
    cell: EmbeddingCell, vertex_id: int, /
) -> (
    tuple[
        dict[tuple[int, ...], Fraction],
        dict[tuple[int, ...], Fraction],
        dict[tuple[int, ...], Fraction],
    ]
    | None
):
    if cell.dimension != 3:
        return None
    try:
        shared_vertex = cell.vertex_ids.index(vertex_id)
    except ValueError:
        return None
    common_vertices = set(range(len(cell.vertex_ids)))
    factors: list[dict[tuple[int, ...], Fraction]] = []
    for face in cell.facet_routes:
        face_vertices = set(face)
        if shared_vertex not in face_vertices:
            continue
        face_ids = tuple(cell.vertex_ids[index] for index in face)
        factor = _reference_facet_factor(cell, face_ids)
        if factor is None:
            return None
        factors.append(factor)
        common_vertices.intersection_update(face_vertices)
    if len(factors) != 3 or common_vertices != {shared_vertex}:
        return None
    return factors[0], factors[1], factors[2]


def _divide_by_linear_factor(
    polynomial: Mapping[tuple[int, ...], ExactScalar],
    factor: Mapping[tuple[int, ...], ExactScalar],
    /,
) -> (
    tuple[
        dict[tuple[int, ...], ExactScalar],
        dict[tuple[int, ...], ExactScalar],
    ]
    | None
):
    if not factor:
        return None
    dimension = len(next(iter(factor)))
    axis = next(
        (
            index
            for index in range(dimension)
            if factor.get(tuple(int(item == index) for item in range(dimension)), 0)
        ),
        None,
    )
    if axis is None:
        return None
    leading = factor[tuple(int(index == axis) for index in range(dimension))]
    remainder = dict(polynomial)
    quotient: dict[tuple[int, ...], ExactScalar] = {}
    while True:
        degree = max((index[axis] for index in remainder), default=0)
        if degree == 0:
            break
        quotient_term = {
            tuple(
                value - int(index == axis)
                for value, index in zip(exponent, range(dimension), strict=True)
            ): coefficient / leading
            for exponent, coefficient in remainder.items()
            if exponent[axis] == degree
        }
        quotient = add(quotient, quotient_term)
        remainder = subtract(remainder, multiply(quotient_term, factor))
    return quotient, remainder


def _strict_control_sign(
    polynomial: Mapping[tuple[int, ...], ExactScalar],
    domain: BernsteinDomain,
    dimension: int,
    /,
) -> int | None:
    controls = bernstein_coefficients(polynomial, domain, dimension)
    if controls and all(_sign(value) > 0 for value in controls):
        return 1
    if controls and all(_sign(value) < 0 for value in controls):
        return -1
    return None


def _nonnegative_control_sign(
    polynomial: Mapping[tuple[int, ...], ExactScalar],
    domain: BernsteinDomain,
    dimension: int,
    /,
) -> tuple[int, bool] | None:
    """Prove a weak polynomial sign and whether its exact controls are strict."""
    controls = bernstein_coefficients(polynomial, domain, dimension)
    if not controls:
        return None
    signs = tuple(_sign(value) for value in controls)
    if all(value >= 0 for value in signs):
        if not any(signs):
            return 0, False
        return 1, all(value > 0 for value in signs)
    if all(value <= 0 for value in signs):
        if not any(signs):
            return 0, False
        return -1, all(value < 0 for value in signs)
    return None


def _proof_polynomial(
    cell: EmbeddingCell,
    expression: Expression,
    /,
) -> Expression | None:
    if cell.pyramid_normalizations is None:
        return None if cell.kind == "pyramid" else expression
    arguments = cell.pyramid_reference_arguments
    if (
        arguments is None
        or len(arguments) != 3
        or any(
            normalization.apex_trace is None or normalization.apex_value is None
            for normalization in cell.pyramid_normalizations
        )
        or not all(
            expression_equal(actual, expected)
            for actual, expected in zip(
                arguments,
                pyramid_reference_to_box_arguments(),
                strict=True,
            )
        )
    ):
        return None
    normalization = normalize_pyramid_reference_expression(expression)
    if normalization.apex_trace is None or normalization.apex_value is None:
        return None
    return normalization.expression


def _stratum_side_sign(
    cell: EmbeddingCell,
    residual: Expression,
    factor: dict[tuple[int, ...], Fraction],
    /,
) -> int | None:
    if isinstance(residual, RationalPolynomial):
        numerator = dict(residual.numerator)
        denominator = dict(residual.denominator)
    else:
        numerator = dict(residual)
        denominator = constant(1, cell.dimension)
    proof_denominator = _proof_polynomial(cell, denominator)
    if not isinstance(proof_denominator, dict):
        return None
    domain = cell.proof_domain
    denominator_sign = _strict_control_sign(proof_denominator, domain, cell.dimension)
    if denominator_sign is None:
        return None
    while numerator:
        proof_numerator = _proof_polynomial(cell, numerator)
        if not isinstance(proof_numerator, dict):
            return None
        sign = _strict_control_sign(proof_numerator, domain, cell.dimension)
        if sign is not None:
            return sign * denominator_sign
        division = _divide_by_linear_factor(numerator, factor)
        if division is None:
            return None
        numerator, remainder = division
        if remainder:
            return None
    return None


def _edge_residual_side_sign(
    cell: EmbeddingCell,
    residual: Expression,
    first_factor: dict[tuple[int, ...], Fraction],
    second_factor: dict[tuple[int, ...], Fraction],
    /,
) -> tuple[int, bool] | None:
    """Certify residual sign via exact two-facet quotient controls."""
    if isinstance(residual, RationalPolynomial):
        numerator = dict(residual.numerator)
        denominator = dict(residual.denominator)
    else:
        numerator = dict(residual)
        denominator = constant(1, cell.dimension)
    proof_denominator = _proof_polynomial(cell, denominator)
    if not isinstance(proof_denominator, dict):
        return None
    domain = cell.proof_domain
    denominator_sign = _strict_control_sign(proof_denominator, domain, cell.dimension)
    if denominator_sign is None:
        return None
    for first, second in (
        (first_factor, second_factor),
        (second_factor, first_factor),
    ):
        first_axis = next(
            (
                axis
                for axis in range(cell.dimension)
                if first.get(tuple(int(index == axis) for index in range(cell.dimension)))
            ),
            None,
        )
        if first_axis is None:
            continue
        first_exponent = tuple(
            int(index == first_axis) for index in range(cell.dimension)
        )
        alpha = second.get(first_exponent, Fraction(0)) / first[first_exponent]
        restricted_second = subtract(second, scale(first, alpha))
        if not isinstance(restricted_second, dict) or not restricted_second:
            continue
        first_division = _divide_by_linear_factor(numerator, first)
        if first_division is None:
            continue
        first_quotient, remainder = first_division
        second_division = _divide_by_linear_factor(remainder, restricted_second)
        if second_division is None:
            continue
        second_quotient, residual_remainder = second_division
        if residual_remainder:
            continue
        adjusted_first = subtract(first_quotient, scale(second_quotient, alpha))
        if not isinstance(adjusted_first, dict):
            continue
        proof_first = _proof_polynomial(cell, adjusted_first)
        proof_second = _proof_polynomial(cell, second_quotient)
        if not isinstance(proof_first, dict) or not isinstance(proof_second, dict):
            continue
        first_result = _nonnegative_control_sign(proof_first, domain, cell.dimension)
        second_result = _nonnegative_control_sign(proof_second, domain, cell.dimension)
        if first_result is None or second_result is None:
            continue
        directions = tuple(
            direction for direction, _ in (first_result, second_result) if direction
        )
        if not directions or any(value != directions[0] for value in directions[1:]):
            continue
        return (
            directions[0] * denominator_sign,
            first_result[1] and second_result[1],
        )
    return None


def _vertex_residual_side_sign(
    cell: EmbeddingCell,
    residual: Expression,
    factors: tuple[
        dict[tuple[int, ...], Fraction],
        dict[tuple[int, ...], Fraction],
        dict[tuple[int, ...], Fraction],
    ],
    /,
) -> tuple[int, bool] | None:
    if isinstance(residual, RationalPolynomial):
        numerator = dict(residual.numerator)
        denominator = dict(residual.denominator)
    else:
        numerator = dict(residual)
        denominator = constant(1, cell.dimension)
    proof_denominator = _proof_polynomial(cell, denominator)
    if not isinstance(proof_denominator, dict):
        return None
    domain = cell.proof_domain
    denominator_sign = _strict_control_sign(proof_denominator, domain, cell.dimension)
    if denominator_sign is None:
        return None
    factor_expressions: tuple[Expression, ...] = tuple(
        scale(factor, Fraction(1)) for factor in factors
    )
    affine_factors = affine_map(factor_expressions, cell.dimension)
    if affine_factors is None:
        return None
    offsets, matrix = affine_factors
    dimension = cell.dimension
    identity: tuple[tuple[ExactScalar, ...], ...] = tuple(
        tuple(Fraction(int(row == column)) for column in range(dimension))
        for row in range(dimension)
    )
    try:
        inverse = solve_exact(
            matrix,
            identity,
            singular_error=_SingularReferenceFactorMap,
        )
    except _SingularReferenceFactorMap:
        return None
    shifted_variables = tuple(
        subtract(variable, constant(offset, dimension))
        for variable, offset in zip(axes(dimension), offsets, strict=True)
    )
    inverse_arguments = linear_combinations(shifted_variables, inverse)
    transformed = compose_expression(numerator, inverse_arguments)
    if not isinstance(transformed, dict):
        return None
    zero = (0,) * dimension
    if transformed.get(zero, Fraction(0)):
        return None
    quotients: list[dict[tuple[int, ...], ExactScalar]] = [{} for _ in range(dimension)]
    for exponent, coefficient in transformed.items():
        axis = next(
            (index for index, power in enumerate(exponent) if power),
            None,
        )
        if axis is None:
            return None
        quotient_exponent = tuple(
            power - int(index == axis) for index, power in enumerate(exponent)
        )
        quotients[axis] = add(quotients[axis], {quotient_exponent: coefficient})
    reference_quotients = tuple(
        compose_expression(quotient, factors) for quotient in quotients
    )
    coefficient_polynomials = tuple(
        expression for expression in reference_quotients if isinstance(expression, dict)
    )
    if len(coefficient_polynomials) != dimension:
        return None
    proof_coefficients = tuple(
        _proof_polynomial(cell, expression) for expression in coefficient_polynomials
    )
    if any(not isinstance(expression, dict) for expression in proof_coefficients):
        return None
    results = tuple(
        _nonnegative_control_sign(expression, domain, dimension)
        for expression in proof_coefficients
        if isinstance(expression, dict)
    )
    verified: list[tuple[int, bool]] = []
    for result in results:
        if result is None:
            return None
        verified.append(result)
    directions = tuple(result[0] for result in verified if result[0])
    if not directions or any(value != directions[0] for value in directions[1:]):
        return None
    return directions[0] * denominator_sign, all(result[1] for result in verified)


def _implicit_facet_relation(
    trace: tuple[Expression, ...], /
) -> dict[tuple[int, ...], ExactScalar] | None:
    coordinates = tuple(
        expression for expression in trace if isinstance(expression, dict)
    )
    if len(coordinates) != 3 or any(
        len(exponent) != 2 for expression in coordinates for exponent in expression
    ):
        return None
    degree = max(
        (sum(exponent) for expression in coordinates for exponent in expression),
        default=0,
    )
    if degree == 0:
        return None
    degree_u = max(
        (exponent[0] for expression in coordinates for exponent in expression),
        default=0,
    )
    degree_v = max(
        (exponent[1] for expression in coordinates for exponent in expression),
        default=0,
    )
    relation_degree_bound = degree * degree
    if degree_u and degree_v:
        relation_degree_bound = min(relation_degree_bound, 2 * degree_u * degree_v)
    for relation_degree in range(1, relation_degree_bound + 1):
        monomials = tuple(
            exponent
            for exponent in product(range(relation_degree + 1), repeat=3)
            if sum(exponent) <= relation_degree
        )
        compositions = tuple(
            compose_expression({exponent: Fraction(1)}, coordinates)
            for exponent in monomials
        )
        parameter_polynomials = tuple(
            expression for expression in compositions if isinstance(expression, dict)
        )
        if len(parameter_polynomials) != len(monomials):
            return None
        parameter_exponents = tuple(
            sorted(
                {
                    exponent
                    for expression in parameter_polynomials
                    for exponent in expression
                }
            )
        )
        if not parameter_exponents:
            return None
        rows = tuple(
            {
                column: coefficient
                for column, expression in enumerate(parameter_polynomials)
                if (coefficient := expression.get(parameter_exponent, Fraction(0)))
            }
            for parameter_exponent in parameter_exponents
        )
        factorization = factor_exact_sparse(rows, len(monomials))
        if factorization.evidence.rank != len(monomials) - 1:
            continue
        nullspace = factorization.nullspace_basis()
        if len(nullspace) != 1:
            continue
        relation = {
            exponent: coefficient
            for exponent, coefficient in zip(monomials, nullspace[0], strict=True)
            if coefficient
        }
        composed_relation = compose_expression(relation, coordinates)
        if isinstance(composed_relation, dict) and not composed_relation:
            return relation
    return None


def _facet_residual_side_sign(
    cell: EmbeddingCell,
    residual: Expression,
    factor: dict[tuple[int, ...], Fraction],
    /,
) -> tuple[int, bool] | None:
    if isinstance(residual, RationalPolynomial):
        numerator = dict(residual.numerator)
        denominator = dict(residual.denominator)
    else:
        numerator = dict(residual)
        denominator = constant(1, cell.dimension)
    proof_denominator = _proof_polynomial(cell, denominator)
    if not isinstance(proof_denominator, dict):
        return None
    domain = cell.proof_domain
    denominator_sign = _strict_control_sign(proof_denominator, domain, cell.dimension)
    if denominator_sign is None:
        return None
    while numerator:
        division = _divide_by_linear_factor(numerator, factor)
        if division is None:
            return None
        quotient, remainder = division
        if remainder or not quotient:
            return None
        proof_quotient = _proof_polynomial(cell, quotient)
        if not isinstance(proof_quotient, dict):
            return None
        sign = _strict_control_sign(proof_quotient, domain, cell.dimension)
        if sign is not None:
            return sign * denominator_sign, True
        numerator = quotient
    return None


def _shared_curved_facet_separation(
    first: EmbeddingCell,
    second: EmbeddingCell,
    shared_ids: tuple[int, ...],
    /,
) -> tuple[dict[tuple[int, ...], ExactScalar], int, int] | None:
    if (
        first.dimension != 3
        or second.dimension != 3
        or len(shared_ids) not in (3, 4)
        or len(first.coordinates) != 3
        or len(second.coordinates) != 3
    ):
        return None
    ordered_ids = next(
        (
            tuple(first.vertex_ids[index] for index in face)
            for face in first.facet_routes
            if {first.vertex_ids[index] for index in face} == set(shared_ids)
        ),
        None,
    )
    if ordered_ids is None or not any(
        {second.vertex_ids[index] for index in face} == set(shared_ids)
        for face in second.facet_routes
    ):
        return None
    first_factor = _reference_facet_factor(first, shared_ids)
    second_factor = _reference_facet_factor(second, shared_ids)
    first_arguments = _trace_arguments(first, ordered_ids)
    second_arguments = _trace_arguments(second, ordered_ids)
    if (
        first_factor is None
        or second_factor is None
        or first_arguments is None
        or second_arguments is None
    ):
        return None
    first_trace = tuple(
        compose_expression(expression, first_arguments)
        for expression in first.coordinates
    )
    second_trace = tuple(
        compose_expression(expression, second_arguments)
        for expression in second.coordinates
    )
    if not all(
        expression_equal(left, right)
        for left, right in zip(first_trace, second_trace, strict=True)
    ):
        return None
    relation = _implicit_facet_relation(first_trace)
    if relation is None:
        return None
    try:
        first_sign = _facet_residual_side_sign(
            first,
            compose_expression(relation, first.coordinates),
            first_factor,
        )
        second_sign = _facet_residual_side_sign(
            second,
            compose_expression(relation, second.coordinates),
            second_factor,
        )
    except (ValueError, ZeroDivisionError):
        return None
    if (
        first_sign is None
        or second_sign is None
        or first_sign[0] != -second_sign[0]
        or not first_sign[1]
        or not second_sign[1]
    ):
        return None
    return relation, first_sign[0], second_sign[0]


def _shared_facet_separation(
    first: EmbeddingCell,
    second: EmbeddingCell,
    shared_ids: tuple[int, ...],
    /,
) -> bool:
    if not shared_ids or not first.coordinates or not second.coordinates:
        return False
    ordered_ids = next(
        (
            tuple(first.vertex_ids[index] for index in face)
            for face in first.facet_routes
            if {first.vertex_ids[index] for index in face} == set(shared_ids)
        ),
        None,
    )
    first_facet = _reference_facet_factor(first, shared_ids)
    second_facet = _reference_facet_factor(second, shared_ids)
    if (
        ordered_ids is None
        or first_facet is None
        or second_facet is None
        or not any(
            {second.vertex_ids[index] for index in face} == set(shared_ids)
            for face in second.facet_routes
        )
    ):
        return False
    common_point = first.corner_images[first.vertex_ids.index(min(shared_ids))]
    ambient = len(common_point)
    for normal in _unique_projective_normals(_candidate_normals(first, second)):
        if len(normal) != ambient or not any(_sign(value) for value in normal):
            continue
        offset = sum(
            (normal[axis] * common_point[axis] for axis in range(ambient)),
            Fraction(0),
        )
        first_args = _trace_arguments(first, ordered_ids)
        second_args = _trace_arguments(second, ordered_ids)
        if first_args is None or second_args is None:
            continue
        try:
            first_trace = compose_expression(
                _projected_expression(first, normal, offset), first_args
            )
            second_trace = compose_expression(
                _projected_expression(second, normal, offset), second_args
            )
            if not expression_equal(first_trace, {}) or not expression_equal(
                second_trace, {}
            ):
                continue
            first_sign = _stratum_side_sign(
                first,
                _projected_expression(first, normal, offset),
                first_facet,
            )
            second_sign = _stratum_side_sign(
                second,
                _projected_expression(second, normal, offset),
                second_facet,
            )
        except (ValueError, ZeroDivisionError):
            continue
        if first_sign is not None and second_sign == -first_sign:
            return True
    return False


def _shared_edge_separation(
    first: EmbeddingCell,
    second: EmbeddingCell,
    shared_ids: tuple[int, ...],
    /,
) -> tuple[tuple[ExactScalar, ...], int, int, bool, bool] | None:
    if (
        first.dimension != 3
        or second.dimension != 3
        or len(shared_ids) != 2
        or not first.coordinates
        or not second.coordinates
        or not first.corner_images
        or not second.corner_images
        or len(first.corner_images[0]) != 3
        or len(second.corner_images[0]) != 3
    ):
        return None
    first_factors = _reference_edge_factors(first, shared_ids)
    second_factors = _reference_edge_factors(second, shared_ids)
    if first_factors is None or second_factors is None:
        return None
    ordered_ids = tuple(sorted(shared_ids))
    first_arguments = _trace_arguments(first, ordered_ids)
    second_arguments = _trace_arguments(second, ordered_ids)
    if first_arguments is None or second_arguments is None:
        return None
    common_point = first.corner_images[first.vertex_ids.index(ordered_ids[0])]
    edge_end = first.corner_images[first.vertex_ids.index(ordered_ids[1])]
    edge_direction = tuple(edge_end[axis] - common_point[axis] for axis in range(3))
    if not any(_sign(value) for value in edge_direction):
        return None
    first_trace = tuple(
        compose_expression(expression, first_arguments)
        for expression in first.coordinates
    )
    curve_polynomials = tuple(
        expression for expression in first_trace if isinstance(expression, dict)
    )
    if len(curve_polynomials) != 3:
        return None
    curve_degree = max(
        (exponent[0] for expression in curve_polynomials for exponent in expression),
        default=0,
    )
    curve_directions = tuple(
        tuple(expression.get((degree,), Fraction(0)) for expression in curve_polynomials)
        for degree in range(1, curve_degree + 1)
    )
    trace_normals: list[tuple[ExactScalar, ...]] = []
    for left, right in combinations(curve_directions, 2):
        normal = (
            left[1] * right[2] - left[2] * right[1],
            left[2] * right[0] - left[0] * right[2],
            left[0] * right[1] - left[1] * right[0],
        )
        if any(_sign(value) for value in normal):
            trace_normals.append(normal)
    normals = _unique_projective_normals(
        chain(_candidate_normals(first, second), trace_normals)
    )
    # Normals only propose a plane; exact traces and full-polynomial signs prove it.
    for normal in normals:
        if (
            len(normal) != 3
            or not any(_sign(value) for value in normal)
            or sum(
                (normal[axis] * edge_direction[axis] for axis in range(3)),
                Fraction(0),
            )
            != 0
        ):
            continue
        offset = sum(
            (normal[axis] * common_point[axis] for axis in range(3)),
            Fraction(0),
        )
        first_projection = _projected_expression(first, normal, offset)
        second_projection = _projected_expression(second, normal, offset)
        try:
            if not expression_equal(
                compose_expression(first_projection, first_arguments), {}
            ) or not expression_equal(
                compose_expression(second_projection, second_arguments), {}
            ):
                continue
            first_sign = _edge_residual_side_sign(
                first, first_projection, first_factors[0], first_factors[1]
            )
            second_sign = _edge_residual_side_sign(
                second, second_projection, second_factors[0], second_factors[1]
            )
        except (ValueError, ZeroDivisionError):
            continue
        if first_sign is not None and second_sign is not None:
            first_direction, first_strict = first_sign
            second_direction, second_strict = second_sign
            if first_direction == -second_direction and first_strict and second_strict:
                return (
                    normal,
                    first_direction,
                    second_direction,
                    first_strict,
                    second_strict,
                )
    return None


def _physical_linear_form(
    coefficients: tuple[
        CoordinateExactScalar, CoordinateExactScalar, CoordinateExactScalar
    ],
    /,
) -> dict[tuple[int, ...], CoordinateExactScalar]:
    return {
        tuple(int(index == axis) for index in range(3)): coefficient
        for axis, coefficient in enumerate(coefficients)
        if coefficient
    }


def _quadratic_edge_residuals(
    trace: tuple[Expression, ...], /
) -> (
    tuple[
        dict[tuple[int, ...], CoordinateExactScalar],
        dict[tuple[int, ...], CoordinateExactScalar],
    ]
    | None
):
    coordinates = tuple(
        expression for expression in trace if isinstance(expression, dict)
    )
    if len(coordinates) != 3 or any(
        len(exponent) != 1 for expression in coordinates for exponent in expression
    ):
        return None
    degree = max(
        (exponent[0] for expression in coordinates for exponent in expression),
        default=0,
    )
    if degree != 2:
        return None
    first_coordinate, second_coordinate, third_coordinate = coordinates
    offset: tuple[
        CoordinateExactScalar,
        CoordinateExactScalar,
        CoordinateExactScalar,
    ] = (
        first_coordinate.get((0,), Fraction(0)),
        second_coordinate.get((0,), Fraction(0)),
        third_coordinate.get((0,), Fraction(0)),
    )
    tangent: tuple[
        CoordinateExactScalar,
        CoordinateExactScalar,
        CoordinateExactScalar,
    ] = (
        first_coordinate.get((1,), Fraction(0)),
        second_coordinate.get((1,), Fraction(0)),
        third_coordinate.get((1,), Fraction(0)),
    )
    curvature: tuple[
        CoordinateExactScalar,
        CoordinateExactScalar,
        CoordinateExactScalar,
    ] = (
        first_coordinate.get((2,), Fraction(0)),
        second_coordinate.get((2,), Fraction(0)),
        third_coordinate.get((2,), Fraction(0)),
    )
    normal: tuple[
        CoordinateExactScalar,
        CoordinateExactScalar,
        CoordinateExactScalar,
    ] = (
        tangent[1] * curvature[2] - tangent[2] * curvature[1],
        tangent[2] * curvature[0] - tangent[0] * curvature[2],
        tangent[0] * curvature[1] - tangent[1] * curvature[0],
    )
    if not any(_sign(value) for value in normal):
        return None
    curvature_squared = sum((value * value for value in curvature), Fraction(0))
    if not _sign(curvature_squared):
        return None
    projection = (
        sum(
            (first * second for first, second in zip(tangent, curvature, strict=True)),
            Fraction(0),
        )
        / curvature_squared
    )
    linear_parameter: tuple[
        CoordinateExactScalar,
        CoordinateExactScalar,
        CoordinateExactScalar,
    ] = (
        tangent[0] - projection * curvature[0],
        tangent[1] - projection * curvature[1],
        tangent[2] - projection * curvature[2],
    )
    parameter_slope = sum(
        (first * second for first, second in zip(linear_parameter, tangent, strict=True)),
        Fraction(0),
    )
    if not _sign(parameter_slope):
        return None
    parameter_offset = sum(
        (first * second for first, second in zip(linear_parameter, offset, strict=True)),
        Fraction(0),
    )
    curve_axis = curvature
    curve_constant = sum(
        (first * second for first, second in zip(curve_axis, offset, strict=True)),
        Fraction(0),
    )
    curve_linear = sum(
        (first * second for first, second in zip(curve_axis, tangent, strict=True)),
        Fraction(0),
    )
    curve_quadratic = sum(
        (first * second for first, second in zip(curve_axis, curvature, strict=True)),
        Fraction(0),
    )
    normal_residual = subtract(
        _physical_linear_form(normal),
        constant(
            sum(
                (first * second for first, second in zip(normal, offset, strict=True)),
                Fraction(0),
            ),
            3,
        ),
    )
    parameter_residual = subtract(
        _physical_linear_form(linear_parameter),
        constant(parameter_offset, 3),
    )
    curve_residual = subtract(
        _physical_linear_form(curve_axis),
        constant(curve_constant, 3),
    )
    curve_residual = subtract(
        curve_residual,
        scale(parameter_residual, curve_linear / parameter_slope),
    )
    curve_residual = subtract(
        curve_residual,
        scale(
            multiply(parameter_residual, parameter_residual),
            curve_quadratic / (parameter_slope * parameter_slope),
        ),
    )
    if not isinstance(normal_residual, dict) or not isinstance(curve_residual, dict):
        return None
    return normal_residual, curve_residual


def _edge_factor_quotients(
    polynomial: Mapping[tuple[int, ...], CoordinateExactScalar],
    first_factor: Mapping[tuple[int, ...], CoordinateExactScalar],
    second_factor: Mapping[tuple[int, ...], CoordinateExactScalar],
    dimension: int,
    /,
) -> (
    tuple[
        dict[tuple[int, ...], CoordinateExactScalar],
        dict[tuple[int, ...], CoordinateExactScalar],
    ]
    | None
):
    first_axis = next(
        (
            axis
            for axis in range(dimension)
            if first_factor.get(
                tuple(int(index == axis) for index in range(dimension)), Fraction(0)
            )
        ),
        None,
    )
    if first_axis is None:
        return None
    first_exponent = tuple(int(index == first_axis) for index in range(dimension))
    alpha = second_factor.get(first_exponent, Fraction(0)) / first_factor[first_exponent]
    restricted_second = subtract(second_factor, scale(first_factor, alpha))
    if not isinstance(restricted_second, dict) or not restricted_second:
        return None
    first_division = _divide_by_linear_factor(polynomial, first_factor)
    if first_division is None:
        return None
    first_quotient, remainder = first_division
    second_division = _divide_by_linear_factor(remainder, restricted_second)
    if second_division is None:
        return None
    second_quotient, residual_remainder = second_division
    if residual_remainder:
        return None
    adjusted_first = subtract(first_quotient, scale(second_quotient, alpha))
    if not isinstance(adjusted_first, dict):
        return None
    return adjusted_first, second_quotient


def _bernstein_control_range(
    polynomial: Mapping[tuple[int, ...], CoordinateExactScalar],
    domain: BernsteinDomain,
    dimension: int,
    /,
) -> tuple[CoordinateExactScalar, CoordinateExactScalar]:
    controls = bernstein_coefficients(polynomial, domain, dimension)
    if not controls:
        return Fraction(0), Fraction(0)
    lower = controls[0]
    upper = controls[0]
    for value in controls[1:]:
        if _sign(value - lower) < 0:
            lower = value
        if _sign(value - upper) > 0:
            upper = value
    return lower, upper


def _strict_two_variable_solution(
    constraints: tuple[tuple[CoordinateExactScalar, CoordinateExactScalar], ...],
    /,
) -> tuple[CoordinateExactScalar, CoordinateExactScalar] | None:
    for second in (Fraction(1), Fraction(-1)):
        lower: list[CoordinateExactScalar] = []
        upper: list[CoordinateExactScalar] = []
        feasible = True
        for first_coefficient, second_coefficient in constraints:
            first_sign = _sign(first_coefficient)
            if first_sign > 0:
                lower.append(-second_coefficient * second / first_coefficient)
            elif first_sign < 0:
                upper.append(-second_coefficient * second / first_coefficient)
            elif _sign(second_coefficient * second) <= 0:
                feasible = False
                break
        if not feasible:
            continue
        lower_bound: CoordinateExactScalar | None = lower[0] if lower else None
        if lower_bound is not None:
            for value in lower[1:]:
                if _sign(value - lower_bound) > 0:
                    lower_bound = value
        upper_bound: CoordinateExactScalar | None = upper[0] if upper else None
        if upper_bound is not None:
            for value in upper[1:]:
                if _sign(upper_bound - value) > 0:
                    upper_bound = value
        if (
            lower_bound is not None
            and upper_bound is not None
            and _sign(upper_bound - lower_bound) <= 0
        ):
            continue
        first = (
            (lower_bound + upper_bound) / 2
            if lower_bound is not None and upper_bound is not None
            else lower_bound + 1
            if lower_bound is not None
            else upper_bound - 1
            if upper_bound is not None
            else Fraction(0)
        )
        if all(
            _sign(first_coefficient * first + second_coefficient * second) > 0
            for first_coefficient, second_coefficient in constraints
        ):
            return first, second
    for first in (Fraction(1), Fraction(-1)):
        if all(
            _sign(first_coefficient * first) > 0
            for first_coefficient, _second_coefficient in constraints
        ):
            return first, Fraction(0)
    return None


def _edge_cone_control_vectors(
    first_quotients: tuple[
        dict[tuple[int, ...], CoordinateExactScalar],
        dict[tuple[int, ...], CoordinateExactScalar],
    ],
    second_quotients: tuple[
        dict[tuple[int, ...], CoordinateExactScalar],
        dict[tuple[int, ...], CoordinateExactScalar],
    ],
    cell: EmbeddingCell,
    denominator_sign: int,
    /,
) -> (
    tuple[
        tuple[tuple[CoordinateExactScalar, CoordinateExactScalar], ...],
        tuple[str, ...],
    ]
    | None
):
    domain = cell.proof_domain
    vectors: list[tuple[ExactScalar, ExactScalar]] = []
    ranges: list[str] = []
    for factor_index in range(2):
        first_polynomial = _proof_polynomial(cell, first_quotients[factor_index])
        second_polynomial = _proof_polynomial(cell, second_quotients[factor_index])
        if not isinstance(first_polynomial, dict) or not isinstance(
            second_polynomial, dict
        ):
            return None
        first_range = _bernstein_control_range(first_polynomial, domain, cell.dimension)
        second_range = _bernstein_control_range(second_polynomial, domain, cell.dimension)
        if denominator_sign < 0:
            first_range = -first_range[1], -first_range[0]
            second_range = -second_range[1], -second_range[0]
        ranges.append(
            f"factor:{factor_index}:controls:"
            f"{tuple(_coordinate_coefficient_identity(value) for value in first_range)},"
            f"{tuple(_coordinate_coefficient_identity(value) for value in second_range)}"
        )
        vectors.extend(product(first_range, second_range))
    return tuple(vectors), tuple(ranges)


def _shared_edge_cone_separation(
    first: EmbeddingCell,
    second: EmbeddingCell,
    shared_ids: tuple[int, ...],
    /,
) -> tuple[str, ...] | None:
    if (
        first.dimension != 3
        or second.dimension != 3
        or len(shared_ids) != 2
        or len(first.coordinates) != 3
        or len(second.coordinates) != 3
    ):
        return None
    first_factors = _reference_edge_factors(first, shared_ids)
    second_factors = _reference_edge_factors(second, shared_ids)
    ordered_ids = tuple(sorted(shared_ids))
    first_arguments = _trace_arguments(first, ordered_ids)
    second_arguments = _trace_arguments(second, ordered_ids)
    if (
        first_factors is None
        or second_factors is None
        or first_arguments is None
        or second_arguments is None
    ):
        return None
    first_trace = tuple(
        compose_expression(expression, first_arguments)
        for expression in first.coordinates
    )
    second_trace = tuple(
        compose_expression(expression, second_arguments)
        for expression in second.coordinates
    )
    if not all(
        expression_equal(left, right)
        for left, right in zip(first_trace, second_trace, strict=True)
    ):
        return None
    physical_residuals = _quadratic_edge_residuals(first_trace)
    if physical_residuals is None:
        return None

    cell_data: list[
        tuple[
            tuple[
                tuple[tuple[CoordinateExactScalar, CoordinateExactScalar], ...],
                tuple[str, ...],
            ]
            | None,
            tuple[
                tuple[tuple[CoordinateExactScalar, CoordinateExactScalar], ...],
                tuple[str, ...],
            ]
            | None,
        ]
    ] = []
    for cell, factors in ((first, first_factors), (second, second_factors)):
        residual_parts: list[
            tuple[
                dict[tuple[int, ...], CoordinateExactScalar],
                dict[tuple[int, ...], CoordinateExactScalar],
            ]
        ] = []
        for residual in physical_residuals:
            composed = compose_expression(residual, cell.coordinates)
            if isinstance(composed, RationalPolynomial):
                numerator = dict(composed.numerator)
                denominator = dict(composed.denominator)
            elif isinstance(composed, dict):
                numerator = dict(composed)
                denominator = constant(1, cell.dimension)
            else:
                return None
            residual_parts.append((numerator, denominator))
        common_denominator = multiply(residual_parts[0][1], residual_parts[1][1])
        if not isinstance(common_denominator, dict):
            return None
        proof_denominator = _proof_polynomial(cell, common_denominator)
        if not isinstance(proof_denominator, dict):
            return None
        domain = cell.proof_domain
        denominator_sign = _strict_control_sign(proof_denominator, domain, cell.dimension)
        if denominator_sign is None:
            return None
        first_common_numerator = multiply(residual_parts[0][0], residual_parts[1][1])
        second_common_numerator = multiply(residual_parts[1][0], residual_parts[0][1])
        if not isinstance(first_common_numerator, dict) or not isinstance(
            second_common_numerator, dict
        ):
            return None
        common_numerators = first_common_numerator, second_common_numerator
        order_data: list[
            tuple[
                tuple[tuple[ExactScalar, ExactScalar], ...],
                tuple[str, ...],
            ]
            | None
        ] = []
        for factor_order in (factors, (factors[1], factors[0])):
            first_quotients = _edge_factor_quotients(
                common_numerators[0], factor_order[0], factor_order[1], cell.dimension
            )
            second_quotients = _edge_factor_quotients(
                common_numerators[1], factor_order[0], factor_order[1], cell.dimension
            )
            if first_quotients is None or second_quotients is None:
                order_data.append(None)
                continue
            order_data.append(
                _edge_cone_control_vectors(
                    first_quotients,
                    second_quotients,
                    cell,
                    denominator_sign,
                )
            )
        cell_data.append((order_data[0], order_data[1]))

    for first_order in range(2):
        first_data = cell_data[0][first_order]
        if first_data is None:
            continue
        for second_order in range(2):
            second_data = cell_data[1][second_order]
            if second_data is None:
                continue
            first_vectors, first_ranges = first_data
            second_vectors, second_ranges = second_data
            for first_direction in (1, -1):
                constraints = tuple(
                    (first_direction * first, first_direction * second)
                    for first, second in first_vectors
                ) + tuple(
                    (-first_direction * first, -first_direction * second)
                    for first, second in second_vectors
                )
                separator = _strict_two_variable_solution(constraints)
                if separator is None:
                    continue
                trace_values = tuple(
                    (exponent, _coordinate_coefficient_identity(value))
                    for expression in first_trace
                    if isinstance(expression, dict)
                    for exponent, value in sorted(expression.items())
                )
                residual_values = tuple(
                    tuple(
                        (exponent, _coordinate_coefficient_identity(value))
                        for exponent, value in sorted(residual.items())
                    )
                    for residual in physical_residuals
                )
                return (
                    f"trace:{trace_values}",
                    f"implicit_edge_residuals:{residual_values}",
                    f"separator:{tuple(_coordinate_coefficient_identity(value) for value in separator)}:"
                    f"sides:{first_direction},{-first_direction}",
                    f"factor_orders:{first_order},{second_order}",
                    f"first_control_ranges:{first_ranges}",
                    f"second_control_ranges:{second_ranges}",
                )
    return None


def _shared_vertex_separation(
    first: EmbeddingCell,
    second: EmbeddingCell,
    shared_ids: tuple[int, ...],
    /,
) -> tuple[tuple[ExactScalar, ...], int, int, bool, bool] | None:
    if (
        first.dimension != 3
        or second.dimension != 3
        or len(shared_ids) != 1
        or not first.coordinates
        or not second.coordinates
        or not first.corner_images
        or not second.corner_images
        or len(first.corner_images[0]) != 3
        or len(second.corner_images[0]) != 3
    ):
        return None
    vertex_id = shared_ids[0]
    try:
        first_vertex = first.vertex_ids.index(vertex_id)
        second_vertex = second.vertex_ids.index(vertex_id)
    except ValueError:
        return None
    common_point = first.corner_images[first_vertex]
    if second.corner_images[second_vertex] != common_point:
        return None
    first_factors = _reference_vertex_factors(first, vertex_id)
    second_factors = _reference_vertex_factors(second, vertex_id)
    if first_factors is None or second_factors is None:
        return None
    for normal in _unique_projective_normals(_candidate_normals(first, second)):
        if len(normal) != 3 or not any(_sign(value) for value in normal):
            continue
        offset = sum(
            (normal[axis] * common_point[axis] for axis in range(3)),
            Fraction(0),
        )
        try:
            first_sign = _vertex_residual_side_sign(
                first,
                _projected_expression(first, normal, offset),
                first_factors,
            )
            second_sign = _vertex_residual_side_sign(
                second,
                _projected_expression(second, normal, offset),
                second_factors,
            )
        except (ValueError, ZeroDivisionError):
            continue
        if first_sign is None or second_sign is None:
            continue
        first_direction, first_strict = first_sign
        second_direction, second_strict = second_sign
        if first_direction == -second_direction and first_strict and second_strict:
            return (
                normal,
                first_direction,
                second_direction,
                first_strict,
                second_strict,
            )
    return None


def _shared_vertex_findings(
    cells: tuple[EmbeddingCell, ...],
    /,
) -> list[MeshCertificateFinding]:
    incidents: dict[int, tuple[ExactScalar, ...]] = {}
    findings = []
    for cell in cells:
        for identifier, point in zip(cell.vertex_ids, cell.corner_images, strict=True):
            previous = incidents.get(identifier)
            if previous is not None and previous != point:
                findings.append(
                    MeshCertificateFinding(
                        "shared_vertex_trace",
                        "violated",
                        "cell",
                        (cell.global_id,),
                        detail=f"Cell vertex {identifier} has conflicting exact source images.",
                    )
                )
            else:
                incidents[identifier] = point
    return findings


def _nonperiodic_contacts(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    cells: tuple[EmbeddingCell, ...],
    proof_values: list[str] | None = None,
    /,
    *,
    skip_internal_quadrilateral_separator: bool = False,
) -> list[MeshCertificateFinding]:
    if not cells:
        return [
            MeshCertificateFinding(
                "empty_embedding", "unresolved", "mesh", detail="No cells were supplied."
            )
        ]
    lower = np.asarray([cell.image_lower for cell in cells], dtype=np.float64)
    upper = np.asarray([cell.image_upper for cell in cells], dtype=np.float64)
    tree = prepare_bvh(lower, upper)
    root_charts = _retained_root_charts(mesh, geometry, cells)
    simplices: list[tuple[tuple[tuple[ExactScalar, ...], ...], ...] | None] = []
    findings: list[MeshCertificateFinding] = []
    for cell in cells:
        if not cell.affine and cell.coordinates:
            simplices.append(None)
            continue
        try:
            simplices.append(_cell_simplices(cell))
        except (ValueError, ZeroDivisionError) as error:
            simplices.append(None)
            findings.append(
                MeshCertificateFinding(
                    "exact_cell_contact_preparation",
                    "unresolved",
                    "cell",
                    (cell.global_id,),
                    detail=str(error),
                )
            )
    for first_block, second_block in bvh_overlap_pair_blocks(
        tree, tree, include_touching=True
    ):
        for first_row, second_row in zip(
            first_block.tolist(), second_block.tolist(), strict=True
        ):
            first_index, second_index = int(first_row), int(second_row)
            if first_index >= second_index:
                continue
            first_cell, second_cell = cells[first_index], cells[second_index]
            first_pieces, second_pieces = simplices[first_index], simplices[second_index]
            if (
                first_cell.dimension != second_cell.dimension
                or first_cell.dimension != mesh.topological_dimension
            ):
                findings.append(
                    MeshCertificateFinding(
                        "mixed_cell_contact_dimension",
                        "unresolved",
                        "cell",
                        (first_cell.global_id, second_cell.global_id),
                        detail="Exact contact comparison requires a common topological cell dimension.",
                    )
                )
                continue
            shared = tuple(
                sorted(set(first_cell.vertex_ids).intersection(second_cell.vertex_ids))
            )
            trace = restriction_trace_equal(first_cell, second_cell, shared)
            if trace is False:
                findings.append(
                    MeshCertificateFinding(
                        "mapped_shared_trace",
                        "violated",
                        "cell",
                        (first_cell.global_id, second_cell.global_id),
                        detail="Exact coordinate-map restrictions disagree on a declared shared stratum.",
                    )
                )
                continue
            curved = not (
                (first_cell.affine or not first_cell.coordinates)
                and (second_cell.affine or not second_cell.coordinates)
            )
            if curved:
                first_root = root_charts[first_index]
                second_root = root_charts[second_index]
                if first_root is not None and second_root is not None:
                    root_contact = _root_chart_contact(first_root, second_root)
                    if root_contact == "declared":
                        if proof_values is not None:
                            proof_values.append(
                                f"cell-pair:{first_cell.global_id}:{second_cell.global_id}:same-root:{first_root.source_geometry_id}:{first_root.source_cell_id}:injective-chart-contact"
                            )
                        continue
                    if root_contact == "outside":
                        findings.append(
                            MeshCertificateFinding(
                                "mapped_cell_contact",
                                "violated",
                                "cell",
                                (first_cell.global_id, second_cell.global_id),
                                detail="An exact same-root injective-map contact lies outside the declared reference stratum.",
                            )
                        )
                        continue
                shared_corner_contact = any(
                    left_point == right_point and left_id != right_id
                    for left_id, left_point in zip(
                        first_cell.vertex_ids, first_cell.corner_images, strict=True
                    )
                    for right_id, right_point in zip(
                        second_cell.vertex_ids, second_cell.corner_images, strict=True
                    )
                )
                if shared_corner_contact:
                    findings.append(
                        MeshCertificateFinding(
                            "mapped_cell_contact",
                            "violated",
                            "cell",
                            (first_cell.global_id, second_cell.global_id),
                            detail="Distinct mapped corner identities have the same exact physical image.",
                        )
                    )
                    continue
                if trace is True and _shared_facet_separation(
                    first_cell, second_cell, shared
                ):
                    if proof_values is not None:
                        proof_values.append(
                            f"cell-pair:{first_cell.global_id}:{second_cell.global_id}:exact-opposite-side-shared-facet"
                        )
                    continue
                if trace is True and len(shared) in (3, 4):
                    if skip_internal_quadrilateral_separator and len(shared) == 4:
                        continue
                    facet_witness = _shared_curved_facet_separation(
                        first_cell, second_cell, shared
                    )
                    if facet_witness is not None:
                        relation, first_side, second_side = facet_witness
                        if proof_values is not None:
                            relation_values = tuple(
                                (
                                    exponent,
                                    _coordinate_coefficient_identity(value),
                                )
                                for exponent, value in sorted(relation.items())
                            )
                            proof_values.append(
                                f"cell-pair:{first_cell.global_id}:{second_cell.global_id}:"
                                f"exact-opposite-side-curved-shared-facet:"
                                f"relation:{relation_values}:sides:{first_side},{second_side}"
                            )
                        continue
                if trace is True:
                    edge_witness = _shared_edge_separation(
                        first_cell, second_cell, shared
                    )
                    if edge_witness is not None:
                        (
                            normal,
                            first_side,
                            second_side,
                            first_strict,
                            second_strict,
                        ) = edge_witness
                        if proof_values is not None:
                            proof_values.append(
                                f"cell-pair:{first_cell.global_id}:{second_cell.global_id}:"
                                f"exact-opposite-side-shared-edge:normal:"
                                f"{tuple(_coordinate_coefficient_identity(value) for value in normal)}:"
                                f"sides:{first_side},{second_side}:"
                                f"strict:{first_strict},{second_strict}"
                            )
                        continue
                    edge_cone_witness = _shared_edge_cone_separation(
                        first_cell, second_cell, shared
                    )
                    if edge_cone_witness is not None:
                        if proof_values is not None:
                            proof_values.append(
                                f"cell-pair:{first_cell.global_id}:{second_cell.global_id}:"
                                f"exact-opposite-side-two-residual-edge-cone:"
                                f"{edge_cone_witness}"
                            )
                        continue
                if trace is True and len(shared) == 1:
                    vertex_witness = _shared_vertex_separation(
                        first_cell, second_cell, shared
                    )
                    if vertex_witness is not None:
                        (
                            normal,
                            first_side,
                            second_side,
                            first_strict,
                            second_strict,
                        ) = vertex_witness
                        if proof_values is not None:
                            proof_values.append(
                                f"cell-pair:{first_cell.global_id}:{second_cell.global_id}:"
                                f"exact-opposite-side-shared-vertex:normal:"
                                f"{tuple(_coordinate_coefficient_identity(value) for value in normal)}:"
                                f"sides:{first_side},{second_side}:"
                                f"strict:{first_strict},{second_strict}"
                            )
                        continue
                if _strict_image_separation(first_cell, second_cell):
                    if proof_values is not None:
                        proof_values.append(
                            f"cell-pair:{first_cell.global_id}:{second_cell.global_id}:outward-bernstein-projection-separation"
                        )
                    continue
                if trace is None and shared:
                    findings.append(
                        MeshCertificateFinding(
                            "mapped_shared_trace",
                            "unresolved",
                            "cell",
                            (first_cell.global_id, second_cell.global_id),
                            detail="The shared mapped stratum has no completed exact restriction proof.",
                        )
                    )
                findings.append(
                    MeshCertificateFinding(
                        "mapped_cell_contact",
                        "unresolved",
                        "cell",
                        (first_cell.global_id, second_cell.global_id),
                        detail="Exact bounded image projections do not exclude contact, and no authoritative common-root restriction theorem applies.",
                    )
                )
                continue
            if trace is None and shared:
                findings.append(
                    MeshCertificateFinding(
                        "mapped_shared_trace",
                        "unresolved",
                        "cell",
                        (first_cell.global_id, second_cell.global_id),
                        detail="The shared mapped stratum has no completed exact restriction proof.",
                    )
                )
            if first_pieces is None or second_pieces is None:
                continue
            offending: tuple[tuple[ExactScalar, ...], ...] = ()
            contact_error: Exception | None = None
            for left in first_pieces:
                for right in second_pieces:
                    try:
                        contacts = _piece_contacts(
                            left,
                            right,
                            first_cell.dimension,
                            mesh.ambient_dimension,
                        )
                    except (ValueError, ZeroDivisionError) as error:
                        contact_error = error
                        break
                    declared = _contact_is_declared(
                        first_cell,
                        second_cell,
                        contacts,
                        first_cell.vertex_ids,
                        second_cell.vertex_ids,
                    )
                    if declared is None:
                        contact_error = UnsupportedEmbeddingMap(
                            "Exact intersection membership in the shared stratum was not decided."
                        )
                        break
                    if not declared:
                        offending = contacts
                        break
                if offending or contact_error is not None:
                    break
            if contact_error is not None:
                findings.append(
                    MeshCertificateFinding(
                        "exact_cell_contact",
                        "unresolved",
                        "cell",
                        (first_cell.global_id, second_cell.global_id),
                        detail=str(contact_error),
                    )
                )
            elif offending:
                findings.append(
                    MeshCertificateFinding(
                        "cell_contact",
                        "violated",
                        "cell",
                        (first_cell.global_id, second_cell.global_id),
                        detail=(
                            "Exact intersection vertices outside the declared shared stratum: "
                            f"{tuple(tuple(_coordinate_coefficient_identity(value) for value in point) for point in offending)}."
                        ),
                    )
                )
    return findings


def _periodic_contacts(
    mesh: CellMesh,
    cells: tuple[EmbeddingCell, ...],
    topology: PeriodicMeshTopology,
    /,
) -> tuple[list[MeshCertificateFinding], Literal["closed", "with_boundary"] | None]:
    from ._periodic_embedding import (
        _contact_disposition,
        certify_periodic_affine_embedding,
    )

    result = certify_periodic_affine_embedding(mesh, cells, topology)
    findings = [
        MeshCertificateFinding(
            check,
            "violated" if _contact_disposition(check) == "violated" else "unresolved",
            "cell" if ids else "mesh",
            ids,
            detail=detail,
        )
        for check, ids, detail in result.findings
    ]
    return findings, result.quotient_boundary


def _global_degree_contact_premises(findings: list[MeshCertificateFinding], /) -> bool:
    unresolved_contacts = {
        "exact_cell_contact",
        "exact_cell_contact_preparation",
        "mapped_cell_contact",
        "mapped_shared_trace",
    }
    return all(
        finding.status == "unresolved" and finding.check in unresolved_contacts
        for finding in findings
    )


def _boundary_findings(
    mesh: CellMesh,
    cells: tuple[EmbeddingCell, ...],
    /,
) -> tuple[list[MeshCertificateFinding], int, int, list[str], bool]:
    if (
        mesh.topological_dimension != 3
        or mesh.ambient_dimension != 3
        or not cells
        or any(cell.kind in ("polygon", "polyhedron") for cell in cells)
    ):
        return [], 0, 0, [], False
    boundary = build_oriented_boundary(mesh, cells)
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
    findings = [
        MeshCertificateFinding(
            finding.check,
            "violated"
            if boundary.status == "violated" and finding.check not in unresolved_checks
            else "unresolved",
            "facet",
            finding.entity_ids,
            detail=finding.detail,
        )
        for finding in boundary.findings
    ]
    # Boundary degree is optional when exact cell-contact proofs suffice; retain
    # boundary findings here only when they certify a contradiction.
    if boundary.status != "violated":
        findings = []
    values = []
    if boundary.status == "certified":
        values.extend(
            (
                f"shell_measures:{tuple(_coordinate_coefficient_identity(value) for value in boundary.shell_signed_measures)}",
                f"exterior_degrees:{boundary.shell_degrees}",
            )
        )
        values.extend(
            f"exterior_face_image:{face.facet_id}:vertices:{face.vertex_ids}:"
            f"map_evidence:{face.map_evidence}"
            for face in boundary.faces
        )
    if boundary.boundary_degree_one:
        values.extend(
            (
                "boundary_degree_one:single-connected-source-component",
                "boundary_topology:one-embedded-sphere",
                f"exterior_planar_embedded_facets:{tuple(sorted(face.facet_id for face in boundary.faces))}",
                f"embedded_exterior_triangles:{len(boundary.triangles)}",
            )
        )
    return (
        findings,
        len(boundary.faces),
        len(boundary.shell_face_indices),
        values,
        boundary.boundary_degree_one,
    )


def _reference_chain_replay(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    /,
) -> tuple[ReferenceChainEmbeddingCertificate, GlobalEmbeddingCertificate] | None:
    source_map = geometry.source_map
    restriction = geometry.restriction_source
    if source_map is None or restriction is None:
        return None
    parent_mesh = getattr(source_map, "source_mesh", None)
    parent_geometry = getattr(source_map, "source_geometry", None)
    if not isinstance(parent_mesh, CellMesh) or not isinstance(
        parent_geometry, CellGeometrySpec
    ):
        return None
    target_mesh_id = getattr(source_map, "target_mesh_id", mesh.mesh_id)
    target_topology_id = getattr(source_map, "target_topology_id", mesh.topology_id)
    if target_mesh_id != mesh.mesh_id or target_topology_id != mesh.topology_id:
        return None
    try:
        parent_validity = certify_cell_geometry_validity(
            parent_geometry, mesh=parent_mesh
        )
        parent_embedding = certify_global_embedding(
            parent_mesh, parent_geometry, parent_validity
        )
        reference_maps = restriction.block_parent_reference_maps
        chain = certify_reference_chain_embedding(
            parent_mesh,
            parent_geometry,
            mesh,
            geometry,
            reference_maps,
            parent_embedding,
        )
    except (TypeError, ValueError, IndexError, ZeroDivisionError):
        return None
    return chain, parent_embedding


def _boundary_inventory(cells: tuple[EmbeddingCell, ...], /) -> tuple[int, int]:
    face_counts: dict[frozenset[int], int] = defaultdict(int)
    face_loops: dict[frozenset[int], tuple[int, ...]] = {}
    for cell in cells:
        for route in reference_cell_topology(cell.kind).entities[2]:
            loop = tuple(cell.vertex_ids[int(index)] for index in route)
            key = frozenset(loop)
            face_counts[key] += 1
            face_loops.setdefault(key, loop)
    boundary_faces = tuple(
        face_loops[key] for key, count in face_counts.items() if count == 1
    )
    edge_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for face_index, loop in enumerate(boundary_faces):
        for left, right in zip(loop, (*loop[1:], loop[0]), strict=True):
            edge_faces[(left, right) if left < right else (right, left)].append(
                face_index
            )
    adjacency: list[set[int]] = [set() for _ in boundary_faces]
    for incident in edge_faces.values():
        for first, second in combinations(incident, 2):
            adjacency[first].add(second)
            adjacency[second].add(first)
    visited: set[int] = set()
    shell_count = 0
    for root in range(len(boundary_faces)):
        if root in visited:
            continue
        shell_count += 1
        visited.add(root)
        queue = [root]
        while queue:
            current = queue.pop()
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
    return len(boundary_faces), shell_count


def _reference_chain_exact_values(
    chain: ReferenceChainEmbeddingCertificate, /
) -> tuple[str, ...]:
    values = [
        f"reference_chain_certificate:{chain.certificate_id}",
        f"reference_chain_map_bank:{chain.reference_map_bank_id}",
        f"reference_chain_parent_embedding:{chain.parent_embedding_id}",
        f"reference_chain_parent:{chain.parent_mesh_id}:"
        f"{chain.parent_topology_id}:{chain.parent_geometry_id}",
        f"reference_chain_target:{chain.target_mesh_id}:"
        f"{chain.target_topology_id}:{chain.target_geometry_id}",
        f"reference_chain_parent_count:{len(chain.premises)}",
    ]
    for parent in chain.premises:
        values.append(
            f"reference_chain_parent:{parent.parent_cell_id}:"
            f"targets:{parent.target_cell_ids}:degree:{parent.oriented_degree}:"
            f"premise:{parent.premise_id}"
        )
        values.extend(
            f"reference_chain_chart:{parent.parent_cell_id}:{chart.target_cell_id}:"
            f"jacobian_sign:{chart.jacobian_sign}:"
            f"bernstein:{chart.jacobian_bernstein_controls}:"
            f"composition:{chart.composition_id}"
            for chart in parent.charts
        )
        values.extend(
            f"reference_chain_internal_face:{parent.parent_cell_id}:"
            f"vertices:{face.target_face_vertex_ids}:cells:{face.target_cell_ids}:"
            f"orientations:{face.orientation_signs}:trace:{face.trace_id}"
            for face in parent.internal_faces
        )
        values.extend(
            f"reference_chain_boundary_face:{parent.parent_cell_id}:"
            f"facet:{face.parent_facet_index}:vertices:{face.target_face_vertex_ids}:"
            f"area:{face.projected_area}:bernstein:{face.face_jacobian_bernstein_controls}:"
            f"trace:{face.trace_id}"
            for face in parent.boundary_faces
        )
        values.extend(
            f"reference_chain_boundary_chain:{parent.parent_cell_id}:"
            f"facet:{facet.parent_facet_index}:degree:{facet.degree}:"
            f"mapped_area:{facet.mapped_projected_area}:"
            f"parent_area:{facet.parent_projected_area}:"
            f"chain:{facet.boundary_chain_id}"
            for facet in parent.boundary_facets
        )
    values.append(
        f"reference_chain_cross_parent_coverage_complete:{chain.cross_parent_coverage_complete}:"
        f"premises:{len(chain.cross_parent_premises)}"
    )
    values.extend(
        f"reference_chain_cross_parent:{premise.source_dimension}:"
        f"target_dimension:{premise.target_entity_dimension}:"
        f"source_vertices:{premise.source_vertex_ids}:"
        f"parents:{premise.incident_parent_cell_ids}:"
        f"target_entities:{premise.target_entity_ids_by_parent}:"
        f"root_traces:{premise.root_trace_ids_by_parent}:"
        f"physical_trace:{premise.physical_trace_id}:"
        f"orientations:{premise.orientation_signs}:premise:{premise.premise_id}"
        for premise in chain.cross_parent_premises
    )
    return tuple(values)


def _periodic_scalar_record(value: ExactScalar, /) -> tuple[tuple[int, int, int], ...]:
    terms = value.terms if isinstance(value, RadicalElement) else ((1, value),)
    return tuple(
        (root, coefficient.numerator, coefficient.denominator)
        for root, coefficient in terms
        if coefficient
    )


def _periodic_quotient_face_key(
    topology: PeriodicMeshTopology,
    quotient_face: int,
    orders: np.ndarray,
    /,
) -> tuple[tuple[int, tuple[int, ...]], ...] | None:
    quotient_offsets = np.asarray(topology.quotient_offsets, dtype=np.int64)
    key_offsets = np.asarray(topology.entity_key_offsets, dtype=np.int64)
    key_values = np.asarray(topology.entity_key_values, dtype=np.int64)
    row = int(quotient_offsets[2]) + quotient_face
    if row < 0 or row + 1 >= len(key_offsets):
        return None
    start, stop = int(key_offsets[row]), int(key_offsets[row + 1])
    width = orders.size + 1
    if width < 2 or (stop - start) % width:
        return None
    entries = tuple(
        (
            int(key_values[offset]),
            tuple(int(value) for value in key_values[offset + 1 : offset + width]),
        )
        for offset in range(start, stop, width)
    )
    return (
        entries if len(entries) in (3, 4) and len(set(entries)) == len(entries) else None
    )


def _periodic_face_arguments(
    cell: EmbeddingCell, local_vertices: tuple[int, ...], /
) -> tuple[Expression, ...] | None:
    if (
        len(cell.coordinates) != 3
        or len(cell.reference_vertices) != len(cell.vertex_indices)
        or len(local_vertices) not in (3, 4)
    ):
        return None
    references = tuple(cell.reference_vertices[index] for index in local_vertices)
    from ._restricted_embedding import _affine_arguments

    if len(references) == 3:
        return _affine_arguments(references)
    tensor = (references[0], references[1], references[3], references[2])
    if any(
        tensor[3][axis] != tensor[1][axis] + tensor[2][axis] - tensor[0][axis]
        for axis in range(3)
    ):
        return None
    return _affine_arguments(tensor[:3])


def _periodic_face_trace(
    cell: EmbeddingCell, local_vertices: tuple[int, ...], /
) -> tuple[Expression, ...] | None:
    arguments = _periodic_face_arguments(cell, local_vertices)
    if arguments is None:
        return None
    try:
        return tuple(
            compose_expression(expression, arguments) for expression in cell.coordinates
        )
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _periodic_transform_face_trace(
    trace: tuple[Expression, ...],
    matrix: tuple[tuple[Fraction, ...], ...],
    /,
) -> tuple[Expression, ...]:
    linear = linear_combinations(
        trace,
        tuple(
            tuple(matrix[axis][coordinate] for coordinate in range(3))
            for axis in range(3)
        ),
    )
    return tuple(
        add(component, constant(matrix[axis][3], 2))
        for axis, component in enumerate(linear)
    )


def _periodic_facet_trace_certificate_id(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    cells: tuple[EmbeddingCell, ...],
    topology: PeriodicMeshTopology,
    /,
) -> str | None:
    """Reprove exact lifted vertex and whole mapped facet equivariance."""
    if (
        mesh.topological_dimension != 3
        or mesh.ambient_dimension != 3
        or mesh.periodic_topology is None
        or topology.periodic_topology_id != mesh.periodic_topology.periodic_topology_id
        or len(cells) != sum(block.cell_count for block in mesh.blocks)
        or any(len(cell.coordinates) != 3 for cell in cells)
    ):
        return None
    try:
        rebuilt = CellMesh.from_stored_fields(mesh)
    except (ArithmeticError, IndexError, KeyError, TypeError, ValueError):
        return None
    if (
        rebuilt.periodic_topology is None
        or rebuilt.periodic_topology.periodic_topology_id != topology.periodic_topology_id
    ):
        return None
    representatives = np.asarray(topology.vertex_representatives, dtype=np.int64)
    shifts = np.asarray(topology.vertex_shifts, dtype=np.int64)
    fixed = np.asarray(topology.vertex_fixed_generators, dtype=np.bool_)
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    vertex_count = len(vertex_ids)
    if (
        representatives.shape != (vertex_count,)
        or shifts.shape != (vertex_count, topology.cell.rank)
        or fixed.shape != shifts.shape
        or np.any(representatives < 0)
        or np.any(representatives >= vertex_count)
        or np.any(representatives[representatives] != representatives)
        or np.any(fixed)
    ):
        return None
    points: dict[int, tuple[ExactScalar, ExactScalar, ExactScalar]] = {}
    for cell in cells:
        if len(cell.corner_images) != len(cell.vertex_indices):
            return None
        for index, image in zip(cell.vertex_indices, cell.corner_images, strict=True):
            if len(image) != 3:
                return None
            point = (image[0], image[1], image[2])
            previous = points.get(index)
            if previous is not None and previous != point:
                return None
            points[index] = point
    if len(points) != vertex_count:
        return None

    from ..discretization._periodic_exact import (
        exact_periodic_element,
        identification_orders,
        PeriodicIsometryGroup,
        reduce_periodic_exponents,
    )
    from ..discretization._periodic_topology import _entity_vertex_rows
    from ._periodic_embedding import _quotient_faces_pair_oriented

    boundary_kind = _quotient_faces_pair_oriented(topology)
    if boundary_kind is None:
        return None
    orders = identification_orders(topology.cell)
    linear_orders = np.asarray(
        topology.cell.linear_orders
        if isinstance(topology.cell, PeriodicIsometryGroup)
        else (1,) * topology.cell.rank,
        dtype=np.int64,
    )
    vertex_records = []
    for index, point in sorted(points.items()):
        root = int(representatives[index])
        root_point = points.get(root)
        if root_point is None:
            return None
        try:
            matrix = exact_periodic_element(topology.cell, shifts[index])
        except (ArithmeticError, IndexError, TypeError, ValueError):
            return None
        expected = tuple(
            sum(
                (
                    matrix[axis][coordinate] * root_point[coordinate]
                    for coordinate in range(3)
                ),
                matrix[axis][3],
            )
            for axis in range(3)
        )
        if point != expected:
            return None
        vertex_records.append(
            (
                int(vertex_ids[index]),
                int(vertex_ids[root]),
                tuple(int(value) for value in shifts[index]),
                tuple(_periodic_scalar_record(value) for value in point),
            )
        )

    face_rows = _entity_vertex_rows(mesh)[2]
    lifted_offsets = np.asarray(topology.lifted_offsets, dtype=np.int64)
    quotient_offsets = np.asarray(topology.quotient_offsets, dtype=np.int64)
    lifted_begin, lifted_end = int(lifted_offsets[2]), int(lifted_offsets[3])
    quotient_begin, quotient_end = (
        int(quotient_offsets[2]),
        int(quotient_offsets[3]),
    )
    face_count = len(face_rows)
    quotient_face_count = quotient_end - quotient_begin
    orbit_indices = (
        np.asarray(topology.orbit_indices, dtype=np.int64)[lifted_begin:lifted_end]
        - quotient_begin
    )
    face_actions = np.asarray(topology.orbit_shifts, dtype=np.int64)[
        lifted_begin:lifted_end
    ]
    if (
        lifted_end - lifted_begin != face_count
        or orbit_indices.shape != (face_count,)
        or face_actions.shape != (face_count, topology.cell.rank)
        or np.any((orbit_indices < 0) | (orbit_indices >= quotient_face_count))
    ):
        return None
    incidence = mesh.topology.incidences[-1]
    relation = incidence.relation
    valid = np.asarray(relation.valid, dtype=np.bool_)
    source_faces = np.asarray(relation.source_indices, dtype=np.int64)[valid]
    target_cells = np.asarray(relation.target_indices, dtype=np.int64)[valid]
    cell_ids = np.asarray(mesh.entity_set(3).entity_ids, dtype=np.int64)
    if (
        source_faces.shape != target_cells.shape
        or np.any((source_faces < 0) | (source_faces >= face_count))
        or np.any((target_cells < 0) | (target_cells >= cell_ids.size))
    ):
        return None
    incident_cells: dict[int, list[int]] = {row: [] for row in range(face_count)}
    for face_row, cell_row in zip(source_faces, target_cells, strict=True):
        incident_cells[int(face_row)].append(int(cell_row))
    if any(
        not incident_cells[row] or len(incident_cells[row]) > 2
        for row in range(face_count)
    ):
        return None
    cells_by_id = {cell.global_id: cell for cell in cells}
    if len(cells_by_id) != len(cells) or set(cells_by_id) != set(cell_ids.tolist()):
        return None
    quotient_rows: list[list[int]] = [[] for _ in range(quotient_face_count)]
    for face_row, orbit in enumerate(orbit_indices):
        quotient_rows[int(orbit)].append(face_row)
    face_trace_records = []
    for quotient_face, lifted_face_rows in enumerate(quotient_rows):
        face_key = _periodic_quotient_face_key(topology, quotient_face, orders)
        if face_key is None:
            return None
        occurrences: list[tuple[int, np.ndarray, tuple[Expression, ...]]] = []
        for face_row in lifted_face_rows:
            face_vertices = tuple(int(value) for value in face_rows[face_row])
            if len(face_vertices) not in (3, 4):
                return None
            face_vertex_ids = {int(vertex_ids[vertex]) for vertex in face_vertices}
            for cell_row in incident_cells[face_row]:
                cell_id = int(cell_ids[cell_row])
                cell = cells_by_id[cell_id]
                if not any(
                    {cell.vertex_ids[local_vertex] for local_vertex in route}
                    == face_vertex_ids
                    for route in cell.facet_routes
                ):
                    return None
                local_by_key: dict[tuple[int, tuple[int, ...]], int] = {}
                for vertex in face_vertices:
                    root = int(representatives[vertex])
                    relative = reduce_periodic_exponents(
                        shifts[vertex] - face_actions[face_row], orders
                    )
                    relative = np.asarray(relative, dtype=np.int64)
                    relative[fixed[vertex]] = 0
                    key = (
                        int(vertex_ids[root]),
                        tuple(int(value) for value in relative),
                    )
                    if key in local_by_key:
                        return None
                    try:
                        local_by_key[key] = cell.vertex_indices.index(vertex)
                    except ValueError:
                        return None
                face_action = np.asarray(face_actions[face_row], dtype=np.int64)
                if set(local_by_key) != set(face_key):
                    alignment_offset = None
                    for local_root, local_winding in local_by_key:
                        for canonical_root, canonical_winding in face_key:
                            if local_root != canonical_root:
                                continue
                            candidate = reduce_periodic_exponents(
                                np.asarray(local_winding, dtype=np.int64)
                                - np.asarray(canonical_winding, dtype=np.int64),
                                orders,
                            )
                            if any(
                                (int(order) > 0 and int(candidate[axis]) != 0)
                                or (
                                    int(order) == 0
                                    and (
                                        int(linear_orders[axis]) <= 0
                                        or int(candidate[axis]) % int(linear_orders[axis])
                                        != 0
                                    )
                                )
                                for axis, order in enumerate(orders)
                            ):
                                continue
                            aligned: dict[tuple[int, tuple[int, ...]], int] = {}
                            for (root, winding), local_vertex in local_by_key.items():
                                normalized = reduce_periodic_exponents(
                                    np.asarray(winding, dtype=np.int64)
                                    - np.asarray(candidate, dtype=np.int64),
                                    orders,
                                )
                                aligned_key = (
                                    root,
                                    tuple(int(value) for value in normalized),
                                )
                                if aligned_key in aligned:
                                    break
                                aligned[aligned_key] = local_vertex
                            if set(aligned) == set(face_key):
                                alignment_offset = candidate
                                local_by_key = aligned
                                break
                        if alignment_offset is not None:
                            break
                    if alignment_offset is None:
                        return None
                    face_action = np.asarray(
                        reduce_periodic_exponents(
                            face_action + np.asarray(alignment_offset, dtype=np.int64),
                            orders,
                        ),
                        dtype=np.int64,
                    )
                local_vertices = tuple(local_by_key[key] for key in face_key)
                trace = _periodic_face_trace(cell, local_vertices)
                if trace is None:
                    return None
                occurrences.append((cell_id, face_action, trace))
        if len(occurrences) == 1:
            face_trace_records.append((quotient_face, face_key, "physical-boundary"))
            continue
        if len(occurrences) != 2:
            return None
        first_id, first_action, first_trace = occurrences[0]
        second_id, second_action, second_trace = occurrences[1]
        action_delta = reduce_periodic_exponents(second_action - first_action, orders)
        try:
            action = exact_periodic_element(topology.cell, action_delta)
            transformed = _periodic_transform_face_trace(first_trace, action)
        except (ArithmeticError, IndexError, TypeError, ValueError, ZeroDivisionError):
            return None
        if not all(
            expression_equal(first, second)
            for first, second in zip(transformed, second_trace, strict=True)
        ):
            return None
        face_trace_records.append(
            (
                quotient_face,
                face_key,
                first_id,
                second_id,
                tuple(int(value) for value in action_delta),
            )
        )
    return canonical_fingerprint(
        {
            "kind": "periodic-current-oriented-facet-traces",
            "mesh": mesh.mesh_id,
            "geometry": cell_geometry_id(geometry),
            "periodic_topology": topology.periodic_topology_id,
            "quotient": array_tree_fingerprint(topology.quotient),
            "boundary_kind": boundary_kind,
            "vertices": tuple(vertex_records),
            "facets": tuple(face_trace_records),
        }
    )


def _proof(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    validity: CellValidityCertificate,
    binding: MeshCertificateBinding,
    cells: tuple[EmbeddingCell, ...] | None,
    map_error: Exception | None,
    /,
    *,
    periodic: PeriodicMeshTopology | None,
    periodic_restriction: PeriodicRestrictionPremise | None,
) -> _ProofOutcome:
    cell_ids = cell_global_ids(mesh)
    statuses = np.asarray(validity.status, dtype=np.int32)
    if statuses.shape != cell_ids.shape:
        raise ValueError(
            "Validity certificate rows do not align with current mesh cells."
        )
    checks = ["cell_validity"]
    state_findings: list[MeshCertificateFinding] = []
    periodic_affine_degree: PeriodicAffineTorusDegreeEvidence | None = None
    periodic_restriction_evidence: PeriodicRestrictionPartitionEvidence | None = None
    periodic_mapped_degree: PeriodicMappedTorusDegreeEvidence | None = None
    for index, status in enumerate(statuses.tolist()):
        if status == CellValidityStatus.INVALID:
            state_findings.append(
                MeshCertificateFinding(
                    "cell_validity",
                    "violated",
                    "cell",
                    (int(cell_ids[index]),),
                    detail="The input-bound local coordinate-map validity proof found a nonpositive or degenerate cell.",
                )
            )
        elif status == CellValidityStatus.UNRESOLVED:
            state_findings.append(
                MeshCertificateFinding(
                    "cell_validity",
                    "unresolved",
                    "cell",
                    (int(cell_ids[index]),),
                    detail="The input-bound local coordinate-map validity proof did not decide this cell.",
                )
            )
    if not validity.all_certified:
        counts = EmbeddingCounts(cell_count=len(cell_ids))
        return _ProofOutcome(tuple(state_findings), tuple(checks), counts, ())
    if map_error is not None or cells is None:
        error = map_error or UnsupportedEmbeddingMap(
            "Exact embedding cells were not prepared."
        )
        state_findings.append(
            MeshCertificateFinding(
                "exact_embedding_map",
                "unresolved",
                "mesh",
                detail=str(error),
            )
        )
        return _ProofOutcome(
            tuple(state_findings),
            tuple((*checks, "exact_embedding_map")),
            EmbeddingCounts(cell_count=len(cell_ids)),
            (),
        )
    checks.append("exact_embedding_map")
    state_findings.extend(_shared_vertex_findings(cells))
    checks.append("shared_vertex_trace")
    boundary_facet_count = 0
    shell_count = 0
    boundary_values: list[str] = []
    degree_one_boundary = False
    global_degree_used = False
    reference_chain_attempt = (
        _reference_chain_replay(mesh, geometry) if periodic is None else None
    )
    reference_chain_certificate = (
        reference_chain_attempt[0] if reference_chain_attempt is not None else None
    )
    parent_embedding = (
        reference_chain_attempt[1] if reference_chain_attempt is not None else None
    )
    reference_chain_ready = (
        reference_chain_certificate is not None
        and parent_embedding is not None
        and reference_chain_certificate.status == "certified"
        and not reference_chain_certificate.findings
        and reference_chain_certificate.cross_parent_coverage_complete
        and parent_embedding.status == "certified"
        and reference_chain_certificate.parent_embedding_id
        == parent_embedding.certificate_id
        and not state_findings
    )
    if reference_chain_attempt is not None:
        checks.append("forward_reference_chain_attempt")
    if reference_chain_ready:
        map_findings: list[MeshCertificateFinding] = []
        map_witnesses: list[MeshEmbeddingWitness] = []
        checks.append("forward_reference_chain_cell_injectivity")
    else:
        map_findings, map_witnesses = _cell_map_findings(cells, binding=binding)
        state_findings.extend(map_findings)
    checks.append("cell_map_injectivity")
    reference_chain_used = reference_chain_ready and not state_findings
    if periodic is not None:
        from ._periodic_affine_degree import certify_affine_torus_degree

        degree_result = certify_affine_torus_degree(
            mesh, geometry, validity, cells, binding.binding_id
        )
        if degree_result.applicable:
            checks.extend(degree_result.checks)
            state_findings.extend(degree_result.findings)
            periodic_affine_degree = degree_result.evidence
            checks.append("periodic_exact_cell_contacts")
            if degree_result.evidence is not None:
                checks.append("periodic_closed_quotient")
                boundary_values.append(
                    f"periodic_affine_torus_degree_one:{degree_result.evidence.evidence_id}"
                )
            if any(
                finding.check == "periodic_affine_torus_degree_one"
                and finding.status == "violated"
                for finding in degree_result.findings
            ):
                state_findings.append(
                    MeshCertificateFinding(
                        "periodic_cell_contact",
                        "violated",
                        "mesh",
                        detail=(
                            "A positive connected torus map with degree greater than one "
                            "cannot be injective on the quotient."
                        ),
                    )
                )
        else:
            mapped_result = None
            if any(not cell.affine for cell in cells) and not state_findings:
                trace_id = _periodic_facet_trace_certificate_id(
                    mesh, geometry, cells, periodic
                )
                if trace_id is not None:
                    from ._periodic_embedding import (
                        certify_periodic_mapped_torus_degree,
                    )

                    local_injectivity_id = canonical_fingerprint(
                        {
                            "kind": "periodic-mapped-local-injectivity",
                            "binding": binding.binding_id,
                            "validity": validity.certificate_id,
                            "witnesses": tuple(
                                witness.witness_id for witness in map_witnesses
                            ),
                            "reference_chain": (
                                None
                                if reference_chain_certificate is None
                                else reference_chain_certificate.certificate_id
                            ),
                        }
                    )
                    mapped_result = certify_periodic_mapped_torus_degree(
                        mesh,
                        geometry,
                        validity,
                        cells,
                        periodic,
                        binding.binding_id,
                        local_injectivity_id,
                        trace_id,
                    )
            if mapped_result is not None and mapped_result.applicable:
                checks.extend(mapped_result.checks)
                state_findings.extend(mapped_result.findings)
                periodic_mapped_degree = mapped_result.evidence
                checks.append("periodic_exact_cell_contacts")
                if mapped_result.evidence is not None:
                    checks.append("periodic_closed_quotient")
                    boundary_values.append(
                        "periodic_mapped_torus_degree_one:"
                        f"{mapped_result.evidence.evidence_id}"
                    )
                if any(
                    finding.check == "periodic_mapped_torus_degree_one"
                    and finding.status == "violated"
                    for finding in mapped_result.findings
                ):
                    state_findings.append(
                        MeshCertificateFinding(
                            "periodic_cell_contact",
                            "violated",
                            "mesh",
                            detail=(
                                "A positive local homeomorphism with degree greater "
                                "than one cannot be injective on the quotient."
                            ),
                        )
                    )
            else:
                restriction_state = EmbeddingState(tuple(state_findings), tuple(checks))
                restriction_evidence = None
                if periodic_restriction is not None and not state_findings:
                    facet_trace_id = _periodic_facet_trace_certificate_id(
                        mesh, geometry, cells, periodic
                    )
                    if facet_trace_id is not None:
                        restriction_evidence = certify_periodic_restriction_partition(
                            restriction_state,
                            mesh,
                            geometry,
                            validity,
                            periodic_restriction,
                            facet_trace_id,
                        )
                if restriction_evidence is not None:
                    periodic_restriction_evidence = restriction_evidence
                    checks.extend(
                        check for check in restriction_state.checks if check not in checks
                    )
                    checks.extend(
                        check
                        for check in (
                            "periodic_exact_cell_contacts",
                            "periodic_closed_quotient",
                        )
                        if check not in checks
                    )
                    boundary_values.append(
                        "periodic_restriction_partition:"
                        f"{restriction_evidence.evidence_id}"
                    )
                else:
                    contact_findings, quotient_boundary = _periodic_contacts(
                        mesh, cells, periodic
                    )
                    state_findings.extend(contact_findings)
                    checks.append("periodic_exact_cell_contacts")
                    if quotient_boundary == "closed":
                        checks.append("periodic_closed_quotient")
                    elif quotient_boundary == "with_boundary":
                        checks.append("periodic_subdomain_boundary")
    elif reference_chain_used:
        assert reference_chain_certificate is not None
        assert parent_embedding is not None
        boundary_facet_count, shell_count = _boundary_inventory(cells)
        boundary_values.extend(_reference_chain_exact_values(reference_chain_certificate))
        checks.extend(("parent_global_embedding", "forward_reference_chain_embedding"))
    else:
        (
            boundary,
            boundary_facet_count,
            shell_count,
            boundary_values,
            degree_one_boundary,
        ) = _boundary_findings(mesh, cells)
        state_findings.extend(boundary)
        checks.extend(("oriented_boundary", "exterior_degree"))
        degree_boundary_candidate = (
            degree_one_boundary and not state_findings and not map_findings
        )
        contact_values: list[str] = []
        contact_findings = _nonperiodic_contacts(
            mesh,
            geometry,
            cells,
            contact_values,
            skip_internal_quadrilateral_separator=degree_boundary_candidate,
        )
        boundary_values.extend(contact_values)
        checks.append("exact_cell_contacts")
        # Degree one discharges unresolved contacts only, never a violation or a
        # missing local-map, continuity, or boundary premise.
        global_degree_used = (
            degree_boundary_candidate
            and _global_degree_contact_premises(contact_findings)
        )
        if global_degree_used:
            boundary_values.extend(
                (
                    "global_degree_one:positive-local-jacobians-and-cancelled-interior-traces",
                    "global_degree_argument:embedded-oriented-sphere-has-winding-one-inside",
                    "injectivity_argument:positive-local-degree-makes-each-regular-preimage-contribute-one",
                )
            )
            checks.append("global_degree_one_injectivity")
            state_findings.extend(
                finding
                for finding in contact_findings
                if finding.status != "unresolved"
                or finding.check
                not in {
                    "exact_cell_contact",
                    "exact_cell_contact_preparation",
                    "mapped_cell_contact",
                    "mapped_shared_trace",
                }
            )
        else:
            state_findings.extend(contact_findings)
    witnesses: list[MeshEmbeddingWitness] = []
    if validity.all_certified:
        witnesses.append(
            MeshEmbeddingWitness(
                "positive_local_validity",
                (binding.binding_id, validity.certificate_id),
                tuple(int(value) for value in cell_ids),
            )
        )
    witnesses.extend(map_witnesses)
    if reference_chain_used:
        assert reference_chain_certificate is not None
        assert parent_embedding is not None
        witnesses.append(
            MeshEmbeddingWitness(
                "forward_reference_chain_injectivity",
                (
                    binding.binding_id,
                    validity.certificate_id,
                    parent_embedding.certificate_id,
                    reference_chain_certificate.certificate_id,
                ),
                tuple(int(value) for value in cell_ids),
                tuple(boundary_values),
            )
        )
    if global_degree_used:
        witnesses.append(
            MeshEmbeddingWitness(
                "global_degree_one_injectivity",
                (
                    binding.binding_id,
                    validity.certificate_id,
                    "embedded_planar_exterior_boundary",
                    "exact_internal_trace_cancellation",
                    "single_sphere_oriented_degree_one",
                ),
                tuple(int(value) for value in cell_ids),
                tuple(boundary_values),
            )
        )
    if not any(value.status in ("violated", "unresolved") for value in state_findings):
        premises = [binding.binding_id, validity.certificate_id]
        if periodic is not None:
            premises.append(periodic.periodic_topology_id)
        witnesses.append(
            MeshEmbeddingWitness(
                "complete_global_cell_contact_theorem",
                tuple(premises),
                tuple(int(value) for value in cell_ids),
                tuple(boundary_values),
            )
        )
    counts = EmbeddingCounts(
        cell_count=len(cell_ids),
        source_bound_count=len(cells),
        boundary_facet_count=boundary_facet_count,
        shell_count=shell_count,
    )
    return _ProofOutcome(
        tuple(state_findings),
        tuple(dict.fromkeys(checks)),
        counts,
        tuple(witnesses),
        periodic_affine_degree,
        periodic_restriction_evidence,
        periodic_mapped_degree,
    )


def certify_global_embedding(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    validity: CellValidityCertificate,
    /,
    *,
    junction_vertices: ArrayLike | None = None,
    periodic: PeriodicMeshTopology | None = None,
    periodic_restriction: PeriodicRestrictionPremise | None = None,
) -> GlobalEmbeddingCertificate:
    """Certify exact global cell contacts after local coordinate-map validation."""
    return GlobalEmbeddingCertificate(
        mesh,
        geometry,
        validity,
        junction_vertices=junction_vertices,
        periodic=periodic,
        periodic_restriction=periodic_restriction,
    )


__all__ = [
    "EmbeddingCounts",
    "GlobalEmbeddingCertificate",
    "MeshCertificateStatus",
    "MeshEmbeddingWitness",
    "cell_global_ids",
    "certify_global_embedding",
]
