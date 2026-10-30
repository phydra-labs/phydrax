#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Construct the oriented exterior/interface chain of affine target cells."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from typing import Literal, TYPE_CHECKING

import numpy as np

from ..discretization._cell_complex import PolyhedralConnectivity
from ..discretization._cell_geometry import CellGeometrySpec
from ..discretization._cell_mesh import CellMesh
from ..discretization._coordinate_enclosure import Expression
from ..discretization._reference_cell import reference_cell_topology
from ._domain_numerics import evaluate_map, polyhedral_vertex_coordinates
from ._planar_coverage import OrientedFacet
from ._planar_primitives import cross, plane_key, projection_axes, signed_area2


if TYPE_CHECKING:
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


def _matching_source_pairs(
    source_chain: tuple[OrientedFacet, ...],
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
    return {
        source.region_pair
        for source in source_chain
        if source.region_pair in allowed_pairs
        and any(
            _overlap(
                source,
                OrientedFacet(
                    -1,
                    source.region_pair,
                    tuple(points[index] for index in piece),
                ),
                dimension,
            )
            > 0
            for piece in pieces
        )
    }


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
    source_chain: tuple[OrientedFacet, ...],
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
        if source_chain:
            allowed = (pair, pair[::-1])
            for triangle in triangles:
                source_pairs.update(
                    _matching_source_pairs(source_chain, triangle, 3, allowed)
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
    source_domain: PiecewiseLinearDomain | None = None,
    geometry: CellGeometrySpec | None = None,
) -> tuple[tuple[OrientedFacet, ...], tuple[BoundaryIssue, ...]]:
    """Build a complete affine boundary/interface chain without coordinate welding."""
    if coordinate_space not in ("physical", "reference"):
        raise ValueError("coordinate_space must be physical or reference.")
    if cell_regions.shape != (sum(block.cell_count for block in mesh.blocks),):
        raise ValueError("cell_regions must align with every target cell.")
    if source_domain is None:
        source_chain: tuple[OrientedFacet, ...] = ()
    else:
        from ._planar_coverage import source_facets

        source_chain = source_facets(source_domain)
    if any(expressions is None for expressions in coordinate_maps):
        return _polyhedral_target_facets(
            mesh, cell_regions, source_chain, geometry=geometry
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
        for face in reference.entities[reference.dimension - 1]:
            face_slots = tuple(int(value) for value in face)
            mapped = tuple(
                _fraction_point(evaluate_map(expressions, reference_vertices[slot]))
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
            key = tuple(sorted(global_vertices))
            grouped.setdefault(key, []).append(
                (cell_index, region, global_vertices, points)
            )

    result: list[OrientedFacet] = []
    for owners in grouped.values():
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
        if len(owners) == 1:
            pair = (first[1], -1)
            if source_chain:
                source_pairs = _matching_source_pairs(
                    source_chain,
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
            if source_chain:
                source_pairs = _matching_source_pairs(
                    source_chain,
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


__all__ = ["BoundaryIssue", "oriented_target_facets"]
