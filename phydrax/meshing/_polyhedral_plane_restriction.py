#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact rational plane clipping for straight-sided polyhedral cells."""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations
from typing import NamedTuple

import numpy as np

from ..discretization._cell_complex import PolyhedralConnectivity
from ..discretization._cell_mesh import CellMesh
from ._polyhedral_adaptation_shared import (
    PolyhedralAdaptationError,
    polyhedral_face_loops,
)


ExactPoint = tuple[Fraction, Fraction, Fraction]
VertexSupport = dict[int, dict[int, Fraction]]


class PolyhedralPlaneRestriction(NamedTuple):
    mesh: CellMesh
    exact_coordinates: tuple[ExactPoint, ...]
    child_parent_ids: tuple[tuple[int, int], ...]
    maximum_coordinate_rounding_error: float

def _point_bank(coordinates: np.ndarray, /) -> list[ExactPoint]:
    return [
        (Fraction(float(row[0])), Fraction(float(row[1])), Fraction(float(row[2])))
        for row in np.asarray(coordinates, dtype=np.float64)
    ]


def _plane_value(point: ExactPoint, plane: ExactPoint, offset: Fraction, /) -> Fraction:
    return sum(
        (plane[axis] * point[axis] for axis in range(3)), -offset
    )


def _cross(first: ExactPoint, second: ExactPoint, /) -> ExactPoint:
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def _difference(first: ExactPoint, second: ExactPoint, /) -> ExactPoint:
    return (
        first[0] - second[0],
        first[1] - second[1],
        first[2] - second[2],
    )


def _dot(first: ExactPoint, second: ExactPoint, /) -> Fraction:
    return sum((first[axis] * second[axis] for axis in range(3)), Fraction(0))


def _face_normal(face: tuple[int, ...], points: list[ExactPoint], /) -> ExactPoint:
    anchor = points[face[0]]
    for first, second in combinations(face[1:], 2):
        normal = _cross(_difference(points[first], anchor), _difference(points[second], anchor))
        if normal != (Fraction(0), Fraction(0), Fraction(0)):
            return normal
    raise PolyhedralAdaptationError("A cut polyhedral cell contains a degenerate face.")


def _projection_axes(normal: ExactPoint, /) -> tuple[int, int, int]:
    pivot = next(axis for axis, value in enumerate(normal) if value != 0)
    if pivot == 0:
        return pivot, 1, 2
    if pivot == 1:
        return pivot, 2, 0
    return pivot, 0, 1


def _turn(
    first: tuple[Fraction, Fraction],
    second: tuple[Fraction, Fraction],
    third: tuple[Fraction, Fraction],
    /,
) -> Fraction:
    return (second[0] - first[0]) * (third[1] - first[1]) - (
        second[1] - first[1]
    ) * (third[0] - first[0])


def _require_convex_cell(
    cell: tuple[np.ndarray, ...],
    points: list[ExactPoint],
    cell_id: int,
    /,
) -> None:
    vertices = tuple(
        sorted(set(int(value) for face in cell for value in np.asarray(face)))
    )
    if len(vertices) < 4:
        raise PolyhedralAdaptationError(
            "A plane-cut cell must have at least four distinct vertices.",
            entity_ids=(cell_id,),
        )
    for face_array in cell:
        face = tuple(int(value) for value in np.asarray(face))
        normal = _face_normal(face, points)
        pivot, first_axis, second_axis = _projection_axes(normal)
        anchor = points[face[0]]
        for vertex in face:
            if _dot(normal, _difference(points[vertex], anchor)) != 0:
                raise PolyhedralAdaptationError(
                    "Plane splitting requires planar polyhedral faces.",
                    entity_ids=(cell_id,),
                )
        projected = tuple(
            (points[vertex][first_axis], points[vertex][second_axis])
            for vertex in face
        )
        turns = tuple(
            _turn(projected[index - 1], projected[index], projected[(index + 1) % len(face)])
            for index in range(len(face))
        )
        if not any(value != 0 for value in turns) or any(
            value * normal[pivot] < 0 for value in turns
        ):
            raise PolyhedralAdaptationError(
                "Plane splitting requires convex, consistently oriented face loops.",
                entity_ids=(cell_id,),
            )
        if any(
            _dot(normal, _difference(points[vertex], anchor)) > 0
            for vertex in vertices
        ):
            raise PolyhedralAdaptationError(
                "Plane splitting currently requires each edited polyhedron to be convex.",
                entity_ids=(cell_id,),
            )


def _intersection(
    first: int,
    second: int,
    values: list[Fraction],
    points: list[ExactPoint],
    vertex_ids: list[int],
    next_vertex: int,
    vertex_support: VertexSupport,
    intersections: dict[tuple[int, int], int],
    /,
) -> tuple[int, int]:
    first_value, second_value = values[first], values[second]
    if first_value == 0:
        return first, next_vertex
    if second_value == 0:
        return second, next_vertex
    edge = tuple(sorted((vertex_ids[first], vertex_ids[second])))
    existing = intersections.get(edge)
    if existing is not None:
        return existing, next_vertex
    denominator = first_value - second_value
    parameter = first_value / denominator
    if not 0 < parameter < 1:
        raise PolyhedralAdaptationError("A plane-edge intersection left its source edge.")
    start, end = points[first], points[second]
    point = tuple(
        start[axis] + parameter * (end[axis] - start[axis])
        for axis in range(3)
    )
    exact_point: ExactPoint = (point[0], point[1], point[2])
    if next_vertex > np.iinfo(np.int64).max:
        raise OverflowError("Polyhedral vertex global IDs exceed int64.")
    row = len(points)
    intersections[edge] = row
    values.append(Fraction(0))
    points.append(exact_point)
    vertex_ids.append(next_vertex)
    first_weights = vertex_support.get(vertex_ids[first], {vertex_ids[first]: Fraction(1)})
    second_weights = vertex_support.get(vertex_ids[second], {vertex_ids[second]: Fraction(1)})
    composed: dict[int, Fraction] = {}
    for source_id, weight in first_weights.items():
        composed[source_id] = composed.get(source_id, Fraction(0)) + (1 - parameter) * weight
    for source_id, weight in second_weights.items():
        composed[source_id] = composed.get(source_id, Fraction(0)) + parameter * weight
    vertex_support[next_vertex] = {
        source_id: weight for source_id, weight in composed.items() if weight != 0
    }
    return row, next_vertex + 1


def _clip_face(
    face: np.ndarray,
    side: int,
    values: list[Fraction],
    points: list[ExactPoint],
    vertex_ids: list[int],
    next_vertex: int,
    vertex_support: VertexSupport,
    intersections: dict[tuple[int, int], int],
    /,
) -> tuple[np.ndarray, int]:
    result: list[int] = []
    row = tuple(int(value) for value in np.asarray(face, dtype=np.int64))
    previous = row[-1]
    for current in row:
        if (side * values[previous] <= 0) != (side * values[current] <= 0):
            crossing, next_vertex = _intersection(
                previous,
                current,
                values,
                points,
                vertex_ids,
                next_vertex,
                vertex_support,
                intersections,
            )
            result.append(crossing)
        if side * values[current] <= 0:
            result.append(current)
        previous = current
    cleaned: list[int] = []
    for vertex in result:
        if not cleaned or cleaned[-1] != vertex:
            cleaned.append(vertex)
    if len(cleaned) > 1 and cleaned[0] == cleaned[-1]:
        cleaned.pop()
    return np.asarray(cleaned, dtype=np.int32), next_vertex


def _hull(points: tuple[tuple[Fraction, Fraction], ...], /) -> tuple[tuple[Fraction, Fraction], ...]:
    unique = tuple(sorted(set(points)))
    if len(unique) < 3:
        return ()
    lower: list[tuple[Fraction, Fraction]] = []
    for point in unique:
        while len(lower) >= 2 and _turn(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[Fraction, Fraction]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and _turn(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return tuple(lower[:-1] + upper[:-1])


def _ordered_section(
    section_vertices: set[int],
    side: int,
    plane: ExactPoint,
    points: list[ExactPoint],
    /,
) -> np.ndarray:
    pivot, first_axis, second_axis = _projection_axes(plane)
    ordered_vertices = tuple(sorted(section_vertices))
    projected = tuple(
        (points[row][first_axis], points[row][second_axis])
        for row in ordered_vertices
    )
    projected_to_vertex = dict(zip(projected, ordered_vertices, strict=True))
    if len(projected_to_vertex) != len(projected):
        raise PolyhedralAdaptationError("A plane section has coincident distinct vertices.")
    hull = _hull(projected)
    if len(hull) < 3:
        raise PolyhedralAdaptationError("A cut cell has a degenerate plane section.")
    ordered: list[int] = []
    for first, second in zip(hull, (*hull[1:], hull[0]), strict=True):
        axis = next(index for index in range(2) if first[index] != second[index])
        members = []
        for point in projected:
            if _turn(first, second, point) == 0 and all(
                min(left, right) <= value <= max(left, right)
                for value, left, right in zip(point, first, second, strict=True)
            ):
                parameter = (point[axis] - first[axis]) / (second[axis] - first[axis])
                members.append((parameter, point))
        ordered.extend(
            projected_to_vertex[point]
            for parameter, point in sorted(members)
            if parameter < 1
        )
    if side * plane[pivot] < 0:
        ordered.reverse()
    if len(set(ordered)) < 3:
        raise PolyhedralAdaptationError("A cut cell has a degenerate oriented section.")
    return np.asarray(ordered, dtype=np.int32)


def _signed_cell_volume(points: list[ExactPoint], faces: tuple[np.ndarray, ...], /) -> Fraction:
    total = Fraction(0)
    for face in faces:
        loop = tuple(int(value) for value in face)
        anchor = points[loop[0]]
        for index in range(1, len(loop) - 1):
            first, second = points[loop[index]], points[loop[index + 1]]
            total += (
                anchor[0] * (first[1] * second[2] - first[2] * second[1])
                + anchor[1] * (first[2] * second[0] - first[0] * second[2])
                + anchor[2] * (first[0] * second[1] - first[1] * second[0])
            )
    return total / 6


def restrict_polyhedral_plane(
    mesh: CellMesh,
    plane: np.ndarray,
    protected_face_ids: set[int],
    vertex_support: VertexSupport,
    /,
    *,
    exact_coordinates: tuple[ExactPoint, ...] | None = None,
) -> PolyhedralPlaneRestriction:
    """Split every crossed convex cell while reusing exact shared cut vertices."""
    connectivity = mesh.connectivity
    if not isinstance(connectivity, PolyhedralConnectivity):
        raise TypeError("plane restriction requires packed polyhedral connectivity.")
    plane_values = np.asarray(plane, dtype=np.float64)
    if plane_values.shape != (4,) or not np.all(np.isfinite(plane_values)):
        raise ValueError("plane must contain four finite coefficients.")
    normal = tuple(Fraction(float(value)) for value in plane_values[:3])
    exact_normal: ExactPoint = (normal[0], normal[1], normal[2])
    if exact_normal == (Fraction(0), Fraction(0), Fraction(0)):
        raise ValueError("plane normal must be nonzero.")
    offset = Fraction(float(plane_values[3]))
    points = (
        _point_bank(np.asarray(mesh.coordinates, dtype=np.float64))
        if exact_coordinates is None
        else list(exact_coordinates)
    )
    if len(points) != mesh.coordinates.shape[0]:
        raise ValueError("Exact coordinates must align with the source mesh vertices.")
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64).tolist()
    cell_ids = np.asarray(connectivity.cell_global_ids, dtype=np.int64)
    face_ids = np.asarray(connectivity.face_global_ids, dtype=np.int64)
    cell_face_offsets = np.asarray(connectivity.cell_face_offsets, dtype=np.int64)
    cell_face_values = np.asarray(connectivity.cell_face_values, dtype=np.int64)
    loops = polyhedral_face_loops(mesh)
    values = [_plane_value(point, exact_normal, offset) for point in points]
    cut_cells: set[int] = set()
    for cell, faces in enumerate(loops):
        vertices = tuple(sorted(set(int(value) for face in faces for value in face)))
        signed = tuple(values[vertex] for vertex in vertices)
        if min(signed) < 0 < max(signed):
            cut_cells.add(cell)
    if not cut_cells:
        return PolyhedralPlaneRestriction(mesh, tuple(points), (), (), 0.0)

    for cell in sorted(cut_cells):
        cell_id = int(cell_ids[cell])
        _require_convex_cell(loops[cell], points, cell_id)
        face_start = int(cell_face_offsets[cell])
        for local_face, face in enumerate(loops[cell]):
            face_values = tuple(values[int(vertex)] for vertex in face)
            if min(face_values) < 0 < max(face_values) and int(
                face_ids[int(cell_face_values[face_start + local_face])]
            ) in protected_face_ids:
                raise PolyhedralAdaptationError(
                    "A protected polyhedral face would be subdivided by the plane.",
                    entity_ids=(cell_id,),
                )

    global_ids = list(vertex_ids)
    next_vertex = max(global_ids, default=-1) + 1
    next_cell = int(np.max(cell_ids)) + 1
    intersection_rows: dict[tuple[int, int], int] = {}
    result_cells: list[tuple[np.ndarray, ...]] = []
    result_ids: list[int] = []
    direct_parent: list[tuple[int, int]] = []
    original_cell_volumes: dict[int, Fraction] = {}
    child_volumes: dict[int, list[Fraction]] = {}
    maximum_error = 0.0

    for cell, original_faces in enumerate(loops):
        original_id = int(cell_ids[cell])
        if cell not in cut_cells:
            result_cells.append(original_faces)
            result_ids.append(original_id)
            continue
        original_cell_volumes[original_id] = _signed_cell_volume(points, original_faces)
        child_volumes[original_id] = []
        for side in (1, -1):
            child_faces: list[np.ndarray] = []
            section_vertices: set[int] = set()
            for face in original_faces:
                clipped, next_vertex = _clip_face(
                    face,
                    side,
                    values,
                    points,
                    global_ids,
                    next_vertex,
                    vertex_support,
                    intersection_rows,
                )
                if clipped.size >= 3:
                    child_faces.append(clipped)
                    section_vertices.update(
                        int(vertex)
                        for vertex in clipped
                        if values[int(vertex)] == 0
                    )
            section = _ordered_section(section_vertices, side, exact_normal, points)
            child_faces.append(section)
            child = tuple(child_faces)
            volume = _signed_cell_volume(points, child)
            if volume <= 0:
                raise PolyhedralAdaptationError(
                    "Plane clipping produced a nonpositive whole-cell volume.",
                    entity_ids=(original_id,),
                )
            child_volumes[original_id].append(volume)
            if next_cell > np.iinfo(np.int64).max:
                raise OverflowError("Polyhedral cell global IDs exceed int64.")
            result_cells.append(child)
            result_ids.append(next_cell)
            direct_parent.append((next_cell, original_id))
            next_cell += 1
    if any(
        sum(child_volumes[parent], Fraction(0)) != original_cell_volumes[parent]
        for parent in original_cell_volumes
    ):
        raise PolyhedralAdaptationError("Plane clipping failed exact cell-volume conservation.")

    rounded = np.asarray(
        [[float(value) for value in point] for point in points], dtype=np.float64
    )
    if not np.all(np.isfinite(rounded)):
        raise PolyhedralAdaptationError("Plane intersections exceed finite coordinates.")
    for point, row in zip(points, rounded, strict=True):
        maximum_error = max(
            maximum_error,
            max(
                abs(float(Fraction(float(value)) - exact))
                for value, exact in zip(row, point, strict=True)
            ),
        )
    target = CellMesh.from_polyhedra(
        rounded,
        result_cells,
        block_name=mesh.blocks[0].name,
        vertex_global_ids=np.asarray(global_ids, dtype=np.int64),
        cell_global_ids=np.asarray(result_ids, dtype=np.int64),
        numeric_version=mesh.numeric_version,
    )
    return PolyhedralPlaneRestriction(
        target,
        tuple(points),
        tuple(direct_parent),
        maximum_error,
    )


__all__ = ["PolyhedralPlaneRestriction", "restrict_polyhedral_plane"]
