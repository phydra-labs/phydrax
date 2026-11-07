#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Construct the oriented exterior/interface chain of planar-faced target cells."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from typing import Literal, TYPE_CHECKING

import numpy as np

from ..discretization._cell_complex import PolyhedralConnectivity
from ..discretization._cell_geometry import CellGeometrySpec
from ..discretization._cell_mesh import CellMesh
from ..discretization._coordinate_enclosure import (
    affine_map,
    axes,
    compose_expression,
    constant,
    ExactScalar,
    Expression,
    multiply,
    normalize_pyramid_reference_expression,
    Polynomial,
    scale,
    subtract,
    sum_polynomials,
)
from ..discretization._reference_cell import reference_cell_topology
from ._domain_numerics import evaluate_map, polyhedral_vertex_coordinates
from ._planar_coverage import (
    CellCorners,
    OrientedFacet,
    PlcSourcePoints,
    SourceFacetIndex,
    SourcePointRefusal,
)
from ._planar_primitives import cross, plane_key, projection_axes, signed_area2


if TYPE_CHECKING:
    from ._affine_plc_domain import AffinePlcDomain
    from ._domain_coverage import PiecewiseLinearDomain


@dataclass(frozen=True, slots=True)
class BoundaryIssue:
    """A target-chain defect found while forming oriented facet pieces."""

    check: str
    entity_ids: tuple[int, ...]
    detail: str
    status: Literal["violated", "unresolved"] = "violated"


def _fraction_point(point: tuple[object, ...]) -> tuple[Fraction, ...] | None:
    result: list[Fraction] = []
    for value in point:
        if isinstance(value, Fraction):
            result.append(value)
        elif isinstance(value, int) and not isinstance(value, bool):
            result.append(Fraction(value))
        else:
            return None
    return tuple(result)


type _FaceTrace = tuple[tuple[Expression, ...], tuple[tuple[Fraction, ...], ...]]


def _reference_corner_image(
    expressions: tuple[Expression, ...],
    cell_family: str,
    point: tuple[Fraction, ...],
    /,
) -> tuple[ExactScalar | None, ...]:
    """Evaluate a cell map at a reference corner, using the unique pyramid apex limit."""
    if cell_family == "pyramid" and point[-1] == 1:
        return tuple(
            normalize_pyramid_reference_expression(expression).apex_value
            for expression in expressions
        )
    return evaluate_map(expressions, point)


def _face_weights(count: int, /) -> tuple[Polynomial, ...]:
    """Return the multilinear corner weights of a unit edge, triangle or square."""
    match count:
        case 2:
            (s,) = axes(1)
            return (subtract(constant(1, 1), s), s)
        case 3:
            s, t = axes(2)
            return (subtract(subtract(constant(1, 2), s), t), s, t)
        case 4:
            s, t = axes(2)
            s_complement = subtract(constant(1, 2), s)
            t_complement = subtract(constant(1, 2), t)
            return (
                multiply(s_complement, t_complement),
                multiply(s, t_complement),
                multiply(s, t),
                multiply(s_complement, t),
            )
        case _:
            raise ValueError("Exact face traces require two, three or four corners.")


def _corner_interpolant(
    weights: tuple[Polynomial, ...], corners: tuple[tuple[Fraction, ...], ...], /
) -> tuple[Polynomial, ...]:
    parameters = 1 if len(weights) == 2 else 2
    return tuple(
        sum_polynomials(
            tuple(
                scale(weight, corner[axis])
                for weight, corner in zip(weights, corners, strict=True)
            ),
            parameters,
        )
        for axis in range(len(corners[0]))
    )


def _strictly_convex_planar_quadrilateral(
    points: tuple[tuple[Fraction, ...], ...], /
) -> bool:
    def difference(
        left: tuple[Fraction, ...], right: tuple[Fraction, ...], /
    ) -> tuple[Fraction, ...]:
        return tuple(a - b for a, b in zip(left, right, strict=True))

    def cross3(
        left: tuple[Fraction, ...], right: tuple[Fraction, ...], /
    ) -> tuple[Fraction, ...]:
        return (
            left[1] * right[2] - left[2] * right[1],
            left[2] * right[0] - left[0] * right[2],
            left[0] * right[1] - left[1] * right[0],
        )

    def dot(left: tuple[Fraction, ...], right: tuple[Fraction, ...], /) -> Fraction:
        return sum((a * b for a, b in zip(left, right, strict=True)), Fraction(0))

    if len(points) != 4 or any(len(point) != 3 for point in points):
        return False
    edges = tuple(
        difference(points[(index + 1) % 4], points[index]) for index in range(4)
    )
    turns = tuple(cross3(edges[index], edges[(index + 1) % 4]) for index in range(4))
    normal = turns[0]
    return (
        any(normal)
        and all(dot(normal, difference(point, points[0])) == 0 for point in points)
        and all(dot(normal, turn) > 0 for turn in turns)
    )


def _face_image_is_corner_polygon(
    trace: _FaceTrace, points: tuple[tuple[Fraction, ...], ...], /
) -> bool:
    """Prove a face image is exactly its corner segment, triangle or convex quadrilateral.

    The restricted cell map must equal the multilinear interpolant of its exact
    corner images. A bilinear interpolant of a planar strictly convex
    quadrilateral is a bijection onto that quadrilateral, so the face image is
    the polygon itself and the corner triangulation partitions it exactly.
    """
    expressions, reference_corners = trace
    weights = _face_weights(len(points))
    arguments = _corner_interpolant(weights, reference_corners)
    expected = _corner_interpolant(weights, points)
    if len(expected) != len(expressions):
        return False
    try:
        for expression, target in zip(expressions, expected, strict=True):
            if subtract(compose_expression(expression, arguments), target):
                return False
    except (TypeError, ValueError, ZeroDivisionError):
        return False
    return len(points) != 4 or _strictly_convex_planar_quadrilateral(points)


def _matching_source_pairs(
    source_index: SourceFacetIndex,
    points: tuple[tuple[Fraction, ...], ...],
    dimension: int,
    allowed_pairs: tuple[tuple[int, int], ...],
    /,
) -> set[tuple[int, int]]:
    from ._planar_coverage import _overlap

    pieces = (
        ((0, 1, 2), (0, 2, 3))
        if dimension == 3 and len(points) == 4
        else (tuple(range(len(points))),)
    )
    pairs: set[tuple[int, int]] = set()
    for piece in pieces:
        piece_points = tuple(points[index] for index in piece)
        for row in source_index.candidates(piece_points, dimension):
            source = source_index.facets[row]
            if (
                source.region_pair in allowed_pairs
                and source.region_pair not in pairs
                and _overlap(
                    source,
                    OrientedFacet(-1, source.region_pair, piece_points),
                    dimension,
                )
                > 0
            ):
                pairs.add(source.region_pair)
    return pairs


def _segments_intersect(
    first: tuple[Fraction, Fraction],
    second: tuple[Fraction, Fraction],
    third: tuple[Fraction, Fraction],
    fourth: tuple[Fraction, Fraction],
    /,
) -> bool:
    def on_segment(
        start: tuple[Fraction, Fraction],
        stop: tuple[Fraction, Fraction],
        point: tuple[Fraction, Fraction],
        /,
    ) -> bool:
        return min(start[0], stop[0]) <= point[0] <= max(start[0], stop[0]) and min(
            start[1], stop[1]
        ) <= point[1] <= max(start[1], stop[1])

    first_side = cross(first, second, third)
    second_side = cross(first, second, fourth)
    third_side = cross(third, fourth, first)
    fourth_side = cross(third, fourth, second)
    if first_side == 0 and on_segment(first, second, third):
        return True
    if second_side == 0 and on_segment(first, second, fourth):
        return True
    if third_side == 0 and on_segment(third, fourth, first):
        return True
    if fourth_side == 0 and on_segment(third, fourth, second):
        return True
    return (
        (first_side > 0 and second_side < 0) or (first_side < 0 and second_side > 0)
    ) and ((third_side > 0 and fourth_side < 0) or (third_side < 0 and fourth_side > 0))


def _triangulate_planar_loop(
    points: tuple[tuple[Fraction, Fraction, Fraction], ...], /
) -> tuple[tuple[tuple[Fraction, Fraction, Fraction], ...], ...]:
    """Triangulate a simple exact planar loop with an ear-clipping proof."""
    if len(set(points)) != len(points) or len(points) < 3:
        raise ValueError("Polyhedral faces require distinct nondegenerate vertices.")
    plane = None
    for candidate in combinations(points, 3):
        try:
            plane = plane_key(candidate)
        except ValueError:
            continue
        break
    if plane is None:
        raise ValueError("Polyhedral face requires three non-collinear exact vertices.")
    if any(
        sum(
            coefficient * value
            for coefficient, value in zip(plane[:3], point, strict=True)
        )
        + plane[3]
        != 0
        for point in points
    ):
        raise ValueError("Polyhedral face vertices are not exactly coplanar.")
    axes_2d = projection_axes(plane)
    projected = [(point[axes_2d[0]], point[axes_2d[1]]) for point in points]
    count = len(projected)
    for first in range(count):
        for second in range(first + 1, count):
            if second == first + 1 or (first == 0 and second == count - 1):
                continue
            if _segments_intersect(
                projected[first],
                projected[(first + 1) % count],
                projected[second],
                projected[(second + 1) % count],
            ):
                raise ValueError("Polyhedral face boundary is not a simple planar loop.")
    vertices = list(zip(points, projected, strict=True))
    changed = True
    while changed and len(vertices) > 3:
        changed = False
        for index, (_, point) in enumerate(vertices):
            previous = vertices[index - 1][1]
            following = vertices[(index + 1) % len(vertices)][1]
            if cross(previous, point, following) == 0:
                if not (
                    min(previous[0], following[0])
                    <= point[0]
                    <= max(previous[0], following[0])
                    and min(previous[1], following[1])
                    <= point[1]
                    <= max(previous[1], following[1])
                ):
                    raise ValueError("Polyhedral face contains a collinear reversal.")
                del vertices[index]
                changed = True
                break
    projected = [point for _, point in vertices]
    orientation = 1 if signed_area2(tuple(projected)) > 0 else -1
    if signed_area2(tuple(projected)) == 0 or len(vertices) < 3:
        raise ValueError("Polyhedral face has zero exact projected area.")
    triangles: list[tuple[tuple[Fraction, Fraction, Fraction], ...]] = []
    while len(vertices) > 3:
        ear = None
        for index, (_, middle) in enumerate(vertices):
            previous_row = (index - 1) % len(vertices)
            next_row = (index + 1) % len(vertices)
            first_2d = vertices[previous_row][1]
            third_2d = vertices[next_row][1]
            if orientation * cross(first_2d, middle, third_2d) <= 0:
                continue
            contains_vertex = False
            for other, (_, point) in enumerate(vertices):
                if other in (previous_row, index, next_row):
                    continue
                if all(
                    orientation * cross(start, stop, point) >= 0
                    for start, stop in (
                        (first_2d, middle),
                        (middle, third_2d),
                        (third_2d, first_2d),
                    )
                ):
                    contains_vertex = True
                    break
            if not contains_vertex:
                ear = (previous_row, index, next_row)
                break
        if ear is None:
            raise ValueError("Exact ear clipping could not prove a face triangulation.")
        triangles.append(tuple(vertices[index][0] for index in ear))
        del vertices[ear[1]]
    final_2d = tuple(point for _, point in vertices)
    if orientation * cross(final_2d[0], final_2d[1], final_2d[2]) <= 0:
        raise ValueError("Exact face triangulation ended in a degenerate triangle.")
    triangles.append(tuple(point for point, _ in vertices))
    return tuple(triangles)


def _polyhedral_target_facets(
    mesh: CellMesh,
    cell_regions: np.ndarray,
    source_index: SourceFacetIndex | None,
    /,
    *,
    geometry: CellGeometrySpec | None,
) -> tuple[tuple[OrientedFacet, ...], tuple[BoundaryIssue, ...]]:
    connectivity = mesh.connectivity
    if not isinstance(connectivity, PolyhedralConnectivity):
        return (), (
            BoundaryIssue(
                "unsupported_polyhedral_connectivity",
                (),
                "Variable-topology target cells require PolyhedralConnectivity.",
                "unresolved",
            ),
        )
    if geometry is None:
        return (), (
            BoundaryIssue(
                "polyhedral_target_geometry_missing",
                (),
                "Exact polyhedral boundary extraction requires the retained CellGeometrySpec.",
                "unresolved",
            ),
        )
    try:
        coordinates = polyhedral_vertex_coordinates(mesh, geometry)
    except (TypeError, ValueError, IndexError, ArithmeticError) as error:
        return (), (
            BoundaryIssue(
                "polyhedral_target_geometry_unresolved", (), str(error), "unresolved"
            ),
        )
    cell_count = sum(block.cell_count for block in mesh.blocks)
    if cell_regions.shape != (cell_count,) or connectivity.cell_count != cell_count:
        return (), (
            BoundaryIssue(
                "polyhedral_target_assignment",
                (),
                "Polyhedral incidence and material assignments do not align with mesh cells.",
            ),
        )
    face_offsets = np.asarray(connectivity.face_vertex_offsets, dtype=np.int64)
    face_vertices = np.asarray(connectivity.face_vertex_values, dtype=np.int64)
    owners = np.asarray(connectivity.face_owner, dtype=np.int64)
    neighbors = np.asarray(connectivity.face_neighbor, dtype=np.int64)
    owner_local = np.asarray(connectivity.face_owner_local, dtype=np.int64)
    cell_face_offsets = np.asarray(connectivity.cell_face_offsets, dtype=np.int64)
    cell_face_signs = np.asarray(connectivity.cell_face_sign_values, dtype=np.int64)
    result: list[OrientedFacet] = []
    issues: list[BoundaryIssue] = []
    target_id = 0
    for face in range(connectivity.face_count):
        owner = int(owners[face])
        neighbor = int(neighbors[face])
        if owner < 0 or owner >= cell_count or neighbor >= cell_count or neighbor < -1:
            issues.append(
                BoundaryIssue(
                    "polyhedral_face_incidence",
                    (face,),
                    "Polyhedral face has an invalid owner or neighbor.",
                )
            )
            continue
        owner_region = int(cell_regions[owner])
        if neighbor >= 0:
            neighbor_region = int(cell_regions[neighbor])
            if owner_region == neighbor_region:
                continue
            pair = (owner_region, neighbor_region)
        else:
            pair = (owner_region, -1)
        start, stop = int(face_offsets[face]), int(face_offsets[face + 1])
        loop = tuple(coordinates[int(vertex)] for vertex in face_vertices[start:stop])
        if any(point is None for point in loop):
            issues.append(
                BoundaryIssue(
                    "polyhedral_face_coordinates",
                    (face,),
                    "Polyhedral face is missing exact physical vertices.",
                    "unresolved",
                )
            )
            continue
        points = tuple(point for point in loop if point is not None)
        owner_row = int(cell_face_offsets[owner]) + int(owner_local[face])
        outward_sign = int(cell_face_signs[owner_row])
        if outward_sign not in (-1, 1):
            issues.append(
                BoundaryIssue(
                    "polyhedral_face_orientation",
                    (face,),
                    "Polyhedral face has no exact signed owner incidence.",
                )
            )
            continue
        if outward_sign < 0:
            points = tuple(reversed(points))
        try:
            triangles = _triangulate_planar_loop(points)
        except (TypeError, ValueError, ArithmeticError) as error:
            issues.append(
                BoundaryIssue(
                    "polyhedral_face_triangulation", (face,), str(error), "unresolved"
                )
            )
            continue
        source_pairs: set[tuple[int, int]] = set()
        if source_index is not None:
            allowed = (pair, pair[::-1])
            for triangle in triangles:
                source_pairs.update(
                    _matching_source_pairs(source_index, triangle, 3, allowed)
                )
        if len(source_pairs) > 1:
            issues.append(
                BoundaryIssue(
                    "polyhedral_face_source_pair_ambiguous",
                    (face,),
                    "One target face overlaps conflicting source material-pair orientations.",
                    "unresolved",
                )
            )
            continue
        if source_pairs:
            source_pair = next(iter(source_pairs))
            if source_pair != pair:
                triangles = tuple(tuple(reversed(triangle)) for triangle in triangles)
            pair = source_pair
        for triangle in triangles:
            result.append(OrientedFacet(target_id, pair, triangle))
            target_id += 1
    return tuple(result), tuple(issues)


def oriented_target_facets(
    mesh: CellMesh,
    cell_regions: np.ndarray,
    coordinate_maps: tuple[tuple[Expression, ...] | None, ...],
    /,
    *,
    coordinate_space: Literal["physical", "reference"] = "physical",
    source_domain: PiecewiseLinearDomain | AffinePlcDomain | None = None,
    geometry: CellGeometrySpec | None = None,
    exact_face_images: bool = False,
    source_points: PlcSourcePoints | None = None,
) -> tuple[tuple[OrientedFacet, ...], tuple[BoundaryIssue, ...]]:
    """Build a complete planar boundary/interface chain without coordinate welding.

    Facets are corner polygons. With ``exact_face_images`` a non-affine cell
    contributes a boundary or interface face only after its exact trace is
    proved to be that corner polygon; otherwise the chain is the vertex skeleton
    used by bounded (non-exact) source comparisons. With ``source_points`` the
    chain uses each witnessed vertex's exact source point instead of its
    carrier, and a facet on a declared-plane chart is compared in that chart.
    """
    if coordinate_space not in ("physical", "reference"):
        raise ValueError("coordinate_space must be physical or reference.")
    if cell_regions.shape != (sum(block.cell_count for block in mesh.blocks),):
        raise ValueError("cell_regions must align with every target cell.")
    if source_points is not None:
        if source_domain is None:
            raise ValueError("Exact source points require their source domain.")
        source_index = SourceFacetIndex(source_points.sources)
    elif source_domain is None:
        source_index = None
    else:
        from ._planar_coverage import source_facets

        source_index = SourceFacetIndex(source_facets(source_domain))
    if any(expressions is None for expressions in coordinate_maps):
        if source_points is not None:
            raise ValueError("Exact source points substitute only affine cell corners.")
        return _polyhedral_target_facets(
            mesh, cell_regions, source_index, geometry=geometry
        )
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    cells: list[tuple[int, int, tuple[Expression, ...], str, np.ndarray]] = []
    global_cell = 0
    for block in mesh.blocks:
        vertices = np.asarray(block.vertices, dtype=np.int64)
        for row in range(block.cell_count):
            expressions = coordinate_maps[global_cell]
            if expressions is None:
                raise ValueError(
                    "Affine target maps must be present after polyhedral dispatch."
                )
            cells.append(
                (
                    global_cell,
                    int(cell_regions[global_cell]),
                    expressions,
                    block.cell_kind,
                    vertices[row],
                )
            )
            global_cell += 1

    grouped: dict[
        tuple[int, ...],
        list[tuple[int, int, tuple[int, ...], tuple[tuple[Fraction, ...], ...]]],
    ] = {}
    # Non-affine cells keep their face traces until a facet is emitted; only
    # boundary/interface facets need the exact corner-polygon image proof.
    face_traces: dict[tuple[int, tuple[int, ...]], _FaceTrace] = {}
    issues: list[BoundaryIssue] = []
    facet_sequence = 0
    for cell_index, region, expressions, cell_kind, cell_vertices in cells:
        try:
            reference = reference_cell_topology(cell_kind)
        except (KeyError, ValueError) as error:
            issues.append(
                BoundaryIssue(
                    "unsupported_target_boundary_topology",
                    (cell_index,),
                    f"No exact standard-cell boundary table is available for {cell_kind!r}: {error}.",
                    "unresolved",
                )
            )
            continue
        if reference.dimension != mesh.topological_dimension:
            issues.append(
                BoundaryIssue(
                    "target_boundary_dimension",
                    (cell_index,),
                    "Target cell reference dimension differs from its mesh topology.",
                )
            )
            continue
        reference_vertices = tuple(
            tuple(Fraction(value) for value in point) for point in reference.vertices
        )
        prove_faces = (
            exact_face_images and affine_map(expressions, reference.dimension) is None
        )
        for face in reference.entities[reference.dimension - 1]:
            face_slots = tuple(int(value) for value in face)
            mapped = tuple(
                _fraction_point(
                    _reference_corner_image(
                        expressions, reference.family, reference_vertices[slot]
                    )
                )
                for slot in face_slots
            )
            if any(point is None for point in mapped):
                issues.append(
                    BoundaryIssue(
                        "unsupported_exact_boundary_coefficients",
                        (cell_index,),
                        "Target boundary map uses coefficients outside the rational exact plane owner.",
                        "unresolved",
                    )
                )
                continue
            points = tuple(point for point in mapped if point is not None)
            global_vertices = tuple(
                int(vertex_ids[int(cell_vertices[slot])]) for slot in face_slots
            )
            if source_points is not None:
                points = source_points.substitute(global_vertices, points)
            key = tuple(sorted(global_vertices))
            grouped.setdefault(key, []).append(
                (cell_index, region, global_vertices, points)
            )
            if prove_faces:
                face_traces[(cell_index, key)] = (
                    expressions,
                    tuple(reference_vertices[slot] for slot in face_slots),
                )

    result: list[OrientedFacet] = []
    for key, owners in grouped.items():
        if len(owners) > 2:
            issues.append(
                BoundaryIssue(
                    "nonmanifold_target_facet",
                    tuple(sorted(owner[0] for owner in owners)),
                    "A target facet has more than two incident cells.",
                )
            )
            continue
        first = owners[0]
        if len(owners) == 2 and owners[0][1] == owners[1][1]:
            continue
        unproven = tuple(
            owner[0]
            for owner in owners
            if (owner[0], key) in face_traces
            and not _face_image_is_corner_polygon(face_traces[(owner[0], key)], owner[3])
        )
        if unproven:
            issues.append(
                BoundaryIssue(
                    "nonplanar_target_face_image",
                    tuple(sorted(unproven)),
                    "A non-affine boundary or interface face trace is not proved to be its exact corner polygon.",
                    "unresolved",
                )
            )
            continue
        if source_points is not None:
            try:
                owners = [
                    (
                        owner[0],
                        owner[1],
                        owner[2],
                        source_points.chart(owner[2], owner[3]),
                    )
                    for owner in owners
                ]
            except SourcePointRefusal as error:
                issues.append(
                    BoundaryIssue(
                        error.check,
                        tuple(sorted(owner[0] for owner in owners)),
                        str(error),
                        error.status,
                    )
                )
                continue
            first = owners[0]
        if len(owners) == 1:
            pair = (first[1], -1)
            if source_index is not None:
                source_pairs = _matching_source_pairs(
                    source_index,
                    first[3],
                    mesh.topological_dimension,
                    (pair, (-1, first[1])),
                )
                if len(source_pairs) > 1:
                    issues.append(
                        BoundaryIssue(
                            "target_boundary_source_orientation_ambiguous",
                            (first[0],),
                            "A target boundary facet overlaps source facets with conflicting region orientation.",
                            "unresolved",
                        )
                    )
                    continue
                if source_pairs:
                    pair = next(iter(source_pairs))
                    if pair[0] == -1:
                        first = (first[0], first[1], first[2], tuple(reversed(first[3])))
        else:
            second = owners[1]
            first_by_vertex = dict(zip(first[2], first[3], strict=True))
            second_by_vertex = dict(zip(second[2], second[3], strict=True))
            if first_by_vertex != second_by_vertex:
                issues.append(
                    BoundaryIssue(
                        "target_interface_coordinate_discontinuity",
                        tuple(sorted((first[0], second[0]))),
                        "Adjacent cells disagree on exact coordinates at their shared facet vertices.",
                    )
                )
                continue
            pair = (first[1], second[1])
            if source_index is not None:
                source_pairs = _matching_source_pairs(
                    source_index,
                    first[3],
                    mesh.topological_dimension,
                    (pair, pair[::-1]),
                )
                if len(source_pairs) > 1:
                    issues.append(
                        BoundaryIssue(
                            "target_interface_source_orientation_ambiguous",
                            tuple(sorted((first[0], second[0]))),
                            "A target interface overlaps source facets with conflicting region orientation.",
                            "unresolved",
                        )
                    )
                    continue
                if source_pairs:
                    pair = next(iter(source_pairs))
                    first = next(owner for owner in owners if owner[1] == pair[0])
        points = first[3]
        if mesh.topological_dimension == 3 and len(points) == 4:
            triangles = ((0, 1, 2), (0, 2, 3))
        elif mesh.topological_dimension == 3 and len(points) != 3:
            issues.append(
                BoundaryIssue(
                    "unsupported_target_facet_arity",
                    (first[0],),
                    "A 3-D affine target face must be triangular or quadrilateral.",
                    "unresolved",
                )
            )
            continue
        else:
            triangles = (tuple(range(len(points))),)
        for indices in triangles:
            selected = tuple(points[index] for index in indices)
            if mesh.topological_dimension == 3 and len(selected) != 3:
                raise AssertionError(
                    "Triangular boundary subdivision produced a nontriangle."
                )
            result.append(OrientedFacet(facet_sequence, pair, selected))
            facet_sequence += 1
    return tuple(result), tuple(issues)


def affine_cell_corners(
    mesh: CellMesh,
    coordinate_maps: tuple[tuple[Expression, ...] | None, ...],
    /,
) -> tuple[CellCorners, ...]:
    """Exact carrier corners of every affine target cell, in cell order."""
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    result: list[CellCorners] = []
    global_cell = 0
    for block in mesh.blocks:
        reference = reference_cell_topology(block.cell_kind)
        reference_vertices = tuple(
            tuple(Fraction(value) for value in point) for point in reference.vertices
        )
        vertices = np.asarray(block.vertices, dtype=np.int64)
        for row in range(block.cell_count):
            expressions = coordinate_maps[global_cell]
            if (
                expressions is None
                or affine_map(expressions, reference.dimension) is None
            ):
                raise ValueError("Exact cell corners require affine cell maps.")
            corners = tuple(
                _fraction_point(
                    _reference_corner_image(expressions, reference.family, point)
                )
                for point in reference_vertices
            )
            exact = tuple(corner for corner in corners if corner is not None)
            if len(exact) != len(corners):
                raise ValueError("Cell corner coordinates are not exact rationals.")
            result.append(
                (
                    reference.family,
                    tuple(int(vertex_ids[int(value)]) for value in vertices[row]),
                    exact,
                )
            )
            global_cell += 1
    return tuple(result)


__all__ = ["affine_cell_corners", "BoundaryIssue", "oriented_target_facets"]
