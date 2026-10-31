#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Complete point location in authored planar-face polyhedral cell complexes."""

from __future__ import annotations

from fractions import Fraction
from typing import Any, cast, final, Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.core import Tracer

from .._bvh import PackedBVH, point_select_leaf_items, prepare_bvh
from .._fingerprint import canonical_fingerprint
from .._geometry_predicates import (
    orient2d,
    orient3d,
    PredicateMode,
    PredicateResult,
)
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..geometry.simplicial import MeshRegion
from ..typing import checked, ConvertibleToArray, Dim, Float
from ._cell_complex import PolyhedralConnectivity
from ._cell_geometry import CellGeometrySpec
from ._cell_geometry_validity import cell_geometry_id
from ._cell_locator import AbstractCellLocator, CellLocationResult, CellLocationStatus
from ._cell_mesh import CellMesh
from ._coordinate_enclosure import CoordinateSourceBank
from ._exact_algebraic_geometry import coefficient_payload


class _PolyhedralCoordinateDim(Dim):
    """Vertex-coordinate rows retained by one authored-face locator."""


type _ExactPoint2 = tuple[Fraction, Fraction]
type _ExactPoint3 = tuple[Fraction, Fraction, Fraction]
type _ExactCoordinateBank = tuple[_ExactPoint3, ...]


def _exact_coordinate_bank(values: CoordinateSourceBank, /) -> _ExactCoordinateBank:
    if any(len(row) != 3 for row in values):
        raise ValueError(
            "Polyhedral coordinate maps must have three coordinates per vertex."
        )
    if any(not isinstance(value, Fraction) for row in values for value in row):
        raise TypeError("Polyhedral point location requires rational coordinate maps.")
    return cast(_ExactCoordinateBank, values)


def _exact_sign(value: Fraction, /) -> int:
    return int(value > 0) - int(value < 0)


def _exact_orient2d(
    first: _ExactPoint2,
    second: _ExactPoint2,
    third: _ExactPoint2,
    /,
) -> int:
    return _exact_sign(
        (second[0] - first[0]) * (third[1] - first[1])
        - (second[1] - first[1]) * (third[0] - first[0])
    )


def _exact_orient3d(
    first: _ExactPoint3,
    second: _ExactPoint3,
    third: _ExactPoint3,
    fourth: _ExactPoint3,
    /,
) -> int:
    u = tuple(second[axis] - first[axis] for axis in range(3))
    v = tuple(third[axis] - first[axis] for axis in range(3))
    w = tuple(fourth[axis] - first[axis] for axis in range(3))
    determinant = (
        u[0] * (v[1] * w[2] - v[2] * w[1])
        - u[1] * (v[0] * w[2] - v[2] * w[0])
        + u[2] * (v[0] * w[1] - v[1] * w[0])
    )
    return _exact_sign(determinant)


def _rational_normal(points: tuple[_ExactPoint3, ...], /) -> _ExactPoint3 | None:
    origin = points[0]
    for first in range(1, len(points) - 1):
        u = tuple(points[first][axis] - origin[axis] for axis in range(3))
        for second in range(first + 1, len(points)):
            v = tuple(points[second][axis] - origin[axis] for axis in range(3))
            normal = (
                u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0],
            )
            if any(normal):
                return normal
    return None


def _exact_segments_intersect(
    first: _ExactPoint2,
    second: _ExactPoint2,
    third: _ExactPoint2,
    fourth: _ExactPoint2,
    /,
) -> bool:
    signs = (
        _exact_orient2d(first, second, third),
        _exact_orient2d(first, second, fourth),
        _exact_orient2d(third, fourth, first),
        _exact_orient2d(third, fourth, second),
    )
    if signs[0] * signs[1] < 0 and signs[2] * signs[3] < 0:
        return True

    def on_segment(point: _ExactPoint2, start: _ExactPoint2, end: _ExactPoint2) -> bool:
        return all(
            min(start[axis], end[axis]) <= point[axis] <= max(start[axis], end[axis])
            for axis in range(2)
        )

    return bool(
        signs[0] == 0
        and on_segment(third, first, second)
        or signs[1] == 0
        and on_segment(fourth, first, second)
        or signs[2] == 0
        and on_segment(first, third, fourth)
        or signs[3] == 0
        and on_segment(second, third, fourth)
    )


def _exact_point_in_triangle(
    point: _ExactPoint2,
    first: _ExactPoint2,
    second: _ExactPoint2,
    third: _ExactPoint2,
    orientation: int,
    /,
) -> bool:
    return all(
        orientation * _exact_orient2d(start, end, point) >= 0
        for start, end in ((first, second), (second, third), (third, first))
    )


def _triangulate_face_exact(
    coordinates: _ExactCoordinateBank, loop: np.ndarray, /
) -> tuple[tuple[int, int, int], ...]:
    if loop.ndim != 1 or loop.size < 3 or np.unique(loop).size != loop.size:
        raise ValueError(
            "Polyhedral face loops require at least three distinct vertices."
        )
    points = tuple(coordinates[int(vertex)] for vertex in loop)
    normal = _rational_normal(points)
    if normal is None:
        raise ValueError("Polyhedral face requires three non-collinear vertices.")
    origin = points[0]
    if any(
        sum(
            (normal[axis] * (point[axis] - origin[axis]) for axis in range(3)),
            Fraction(0),
        )
        != 0
        for point in points
    ):
        raise ValueError("Polyhedral face vertices must lie in one exact plane.")
    drop = max(range(3), key=lambda axis: abs(normal[axis]))
    keep = tuple(axis for axis in range(3) if axis != drop)
    projected: tuple[_ExactPoint2, ...] = tuple(
        (point[keep[0]], point[keep[1]]) for point in points
    )
    area = sum(
        (
            projected[index][0] * projected[(index + 1) % loop.size][1]
            - projected[(index + 1) % loop.size][0] * projected[index][1]
            for index in range(loop.size)
        ),
        Fraction(0),
    )
    if not area:
        raise ValueError("Polyhedral face projection is degenerate.")
    orientation = _exact_sign(area)
    for first_edge in range(loop.size):
        first_next = (first_edge + 1) % loop.size
        for second_edge in range(first_edge + 1, loop.size):
            second_next = (second_edge + 1) % loop.size
            if first_edge in (second_edge, second_next) or first_next in (
                second_edge,
                second_next,
            ):
                continue
            if _exact_segments_intersect(
                projected[first_edge],
                projected[first_next],
                projected[second_edge],
                projected[second_next],
            ):
                raise ValueError("Polyhedral face loops must be simple polygons.")
    remaining = list(range(loop.size))
    triangles: list[tuple[int, int, int]] = []
    while len(remaining) > 3:
        ear: int | None = None
        for slot, current in enumerate(remaining):
            previous = remaining[(slot - 1) % len(remaining)]
            following = remaining[(slot + 1) % len(remaining)]
            if (
                orientation
                * _exact_orient2d(
                    projected[previous], projected[current], projected[following]
                )
                <= 0
            ):
                continue
            if any(
                _exact_point_in_triangle(
                    projected[candidate],
                    projected[previous],
                    projected[current],
                    projected[following],
                    orientation,
                )
                for candidate in remaining
                if candidate not in (previous, current, following)
            ):
                continue
            ear = slot
            triangles.append(
                (
                    int(loop[previous]),
                    int(loop[current]),
                    int(loop[following]),
                )
            )
            break
        if ear is None:
            raise ValueError(
                "Polyhedral face loop cannot be triangulated without changing its boundary."
            )
        remaining.pop(ear)
    first, second, third = remaining
    triangles.append((int(loop[first]), int(loop[second]), int(loop[third])))
    return tuple(triangles)


def _perturbed_side_exact(
    first: _ExactPoint2,
    second: _ExactPoint2,
    point: _ExactPoint2,
    /,
) -> int:
    exact = _exact_orient2d(first, second, point)
    if exact:
        return exact
    if second[1] != first[1]:
        return _exact_sign(first[1] - second[1])
    return _exact_sign(second[0] - first[0])


def _triangle_membership_exact(
    point: _ExactPoint3,
    triangle_vertices: tuple[int, int, int],
    coordinates: _ExactCoordinateBank,
    /,
) -> tuple[bool, bool]:
    triangle = (
        coordinates[triangle_vertices[0]],
        coordinates[triangle_vertices[1]],
        coordinates[triangle_vertices[2]],
    )
    first, second, third = triangle
    projected = (point[1], point[2])
    projected_vertices: tuple[_ExactPoint2, _ExactPoint2, _ExactPoint2] = (
        (first[1], first[2]),
        (second[1], second[2]),
        (third[1], third[2]),
    )
    side = tuple(
        _perturbed_side_exact(start, end, projected)
        for start, end in zip(
            projected_vertices,
            (*projected_vertices[1:], projected_vertices[0]),
            strict=True,
        )
    )
    normal = _exact_orient2d(*projected_vertices)
    volume = _exact_orient3d(first, second, third, point)
    crossing = (
        normal != 0
        and all(value == side[0] for value in side)
        and side[0] != 0
        and volume != 0
        and volume == -normal
    )
    touching = False
    if volume == 0:
        exact_normal = _rational_normal(triangle)
        if exact_normal is None:
            raise ValueError("Polyhedral boundary triangles must be nondegenerate.")
        drop = max(range(3), key=lambda axis: abs(exact_normal[axis]))
        keep = tuple(axis for axis in range(3) if axis != drop)
        point_2d: _ExactPoint2 = (point[keep[0]], point[keep[1]])
        vertices_2d = tuple((vertex[keep[0]], vertex[keep[1]]) for vertex in triangle)
        orientation = _exact_orient2d(*vertices_2d)
        touching = orientation != 0 and _exact_point_in_triangle(
            point_2d,
            vertices_2d[0],
            vertices_2d[1],
            vertices_2d[2],
            orientation,
        )
    return crossing, touching


def _sign(value: PredicateResult, /) -> int:
    if not np.all(np.asarray(value.certain)):
        raise RuntimeError(
            "Exact polyhedral location predicate returned an uncertain sign."
        )
    return int(np.asarray(value.signs).item())


def _orient2d(first: np.ndarray, second: np.ndarray, third: np.ndarray, /) -> int:
    return _sign(orient2d(first, second, third, mode=PredicateMode.EXACT))


def _orient3d(
    first: np.ndarray,
    second: np.ndarray,
    third: np.ndarray,
    fourth: np.ndarray,
    /,
) -> int:
    return _sign(orient3d(first, second, third, fourth, mode=PredicateMode.EXACT))


def _exact_normal(vertices: np.ndarray, /) -> tuple[Fraction, Fraction, Fraction] | None:
    origin = tuple(Fraction(float(value)) for value in vertices[0])
    for first in range(1, vertices.shape[0] - 1):
        u = tuple(
            Fraction(float(value)) - start
            for value, start in zip(vertices[first], origin, strict=True)
        )
        for second in range(first + 1, vertices.shape[0]):
            v = tuple(
                Fraction(float(value)) - start
                for value, start in zip(vertices[second], origin, strict=True)
            )
            normal = (
                u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0],
            )
            if any(normal):
                return normal
    return None


def _segments_intersect(
    first: np.ndarray,
    second: np.ndarray,
    third: np.ndarray,
    fourth: np.ndarray,
    /,
) -> bool:
    a = _orient2d(first, second, third)
    b = _orient2d(first, second, fourth)
    c = _orient2d(third, fourth, first)
    d = _orient2d(third, fourth, second)
    if a * b < 0 and c * d < 0:
        return True
    return bool(
        a == 0
        and np.all(third >= np.minimum(first, second))
        and np.all(third <= np.maximum(first, second))
        or b == 0
        and np.all(fourth >= np.minimum(first, second))
        and np.all(fourth <= np.maximum(first, second))
        or c == 0
        and np.all(first >= np.minimum(third, fourth))
        and np.all(first <= np.maximum(third, fourth))
        or d == 0
        and np.all(second >= np.minimum(third, fourth))
        and np.all(second <= np.maximum(third, fourth))
    )


def _point_in_triangle(
    point: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    third: np.ndarray,
    orientation: int,
    /,
) -> bool:
    return all(
        orientation * _orient2d(a, b, point) >= 0
        for a, b in ((first, second), (second, third), (third, first))
    )


def _triangulate_face(
    coordinates: np.ndarray, loop: np.ndarray, /
) -> tuple[tuple[int, int, int], ...]:
    if loop.ndim != 1 or loop.size < 3 or np.unique(loop).size != loop.size:
        raise ValueError(
            "Polyhedral face loops require at least three distinct vertices."
        )
    points = coordinates[loop]
    normal = _exact_normal(points)
    if normal is None:
        raise ValueError("Polyhedral face requires three non-collinear vertices.")
    plane: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    for first in range(1, points.shape[0] - 1):
        for second in range(first + 1, points.shape[0]):
            if _exact_normal(points[[0, first, second]]) is not None:
                plane = (points[0], points[first], points[second])
                break
        if plane is not None:
            break
    if plane is None:
        raise ValueError("Polyhedral face requires three non-collinear vertices.")
    if any(_orient3d(*plane, point) != 0 for point in points):
        raise ValueError("Polyhedral face vertices must lie in one exact plane.")
    drop = max(range(3), key=lambda axis: abs(normal[axis]))
    keep = tuple(axis for axis in range(3) if axis != drop)
    projected = points[:, keep]
    area = sum(
        Fraction(float(projected[index, 0]))
        * Fraction(float(projected[(index + 1) % loop.size, 1]))
        - Fraction(float(projected[(index + 1) % loop.size, 0]))
        * Fraction(float(projected[index, 1]))
        for index in range(loop.size)
    )
    if not area:
        raise ValueError("Polyhedral face projection is degenerate.")
    orientation = 1 if area > 0 else -1
    for first_edge in range(loop.size):
        first_next = (first_edge + 1) % loop.size
        for second_edge in range(first_edge + 1, loop.size):
            second_next = (second_edge + 1) % loop.size
            if first_edge in (second_edge, second_next) or first_next in (
                second_edge,
                second_next,
            ):
                continue
            if _segments_intersect(
                projected[first_edge],
                projected[first_next],
                projected[second_edge],
                projected[second_next],
            ):
                raise ValueError("Polyhedral face loops must be simple polygons.")
    remaining = list(range(loop.size))
    triangles: list[tuple[int, int, int]] = []
    while len(remaining) > 3:
        ear: int | None = None
        for slot, current in enumerate(remaining):
            previous = remaining[(slot - 1) % len(remaining)]
            following = remaining[(slot + 1) % len(remaining)]
            if (
                orientation
                * _orient2d(projected[previous], projected[current], projected[following])
                <= 0
            ):
                continue
            if any(
                _point_in_triangle(
                    projected[candidate],
                    projected[previous],
                    projected[current],
                    projected[following],
                    orientation,
                )
                for candidate in remaining
                if candidate not in (previous, current, following)
            ):
                continue
            ear = slot
            triangles.append(
                (int(loop[previous]), int(loop[current]), int(loop[following]))
            )
            break
        if ear is None:
            raise ValueError(
                "Polyhedral face loop cannot be triangulated without changing its boundary."
            )
        remaining.pop(ear)
    first, second, third = remaining
    triangles.append((int(loop[first]), int(loop[second]), int(loop[third])))
    return tuple(triangles)


def _perturbed_side(first: np.ndarray, second: np.ndarray, point: np.ndarray, /) -> int:
    exact = _orient2d(first, second, point)
    if exact:
        return exact
    if second[1] != first[1]:
        return int(np.sign(first[1] - second[1]))
    return int(np.sign(second[0] - first[0]))


def _triangle_membership(point: np.ndarray, triangle: np.ndarray, /) -> tuple[bool, bool]:
    a, b, c = triangle
    projected = point[1:]
    projected_vertices = (a[1:], b[1:], c[1:])
    side = tuple(
        _perturbed_side(first, second, projected)
        for first, second in zip(
            projected_vertices,
            (*projected_vertices[1:], projected_vertices[0]),
            strict=True,
        )
    )
    normal = _orient2d(a[1:], b[1:], c[1:])
    volume = _orient3d(a, b, c, point)
    crossing = (
        normal != 0
        and all(value == side[0] for value in side)
        and side[0] != 0
        and volume != 0
        and volume == -normal
    )
    touching = False
    if volume == 0:
        exact_normal = _exact_normal(triangle)
        if exact_normal is None:
            raise ValueError("Polyhedral boundary triangles must be nondegenerate.")
        drop = max(range(3), key=lambda axis: abs(exact_normal[axis]))
        keep = tuple(axis for axis in range(3) if axis != drop)
        point_2d = point[list(keep)]
        vertices_2d = triangle[:, list(keep)]
        orientation = _orient2d(vertices_2d[0], vertices_2d[1], vertices_2d[2])
        touching = orientation != 0 and _point_in_triangle(
            point_2d, vertices_2d[0], vertices_2d[1], vertices_2d[2], orientation
        )
    return crossing, touching


class PreparedPolyhedralCellMap(StrictModule, NonTrainableState):
    """Physical Cartesian map of each authored planar-face cell."""

    mesh: CellMesh
    coordinates: Array
    cell_lower: Array
    cell_upper: Array
    exact_coordinates: _ExactCoordinateBank = eqx.field(static=True)
    cell_vertices: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    cell_triangles: tuple[tuple[tuple[int, int, int], ...], ...] = eqx.field(static=True)
    face_triangles: tuple[tuple[tuple[int, int, int], ...], ...] = eqx.field(static=True)
    cell_count: int = eqx.field(static=True)
    coordinate_count: int = eqx.field(static=True)
    ambient_dimension: int = eqx.field(static=True)
    reference_dimension: int = eqx.field(static=True)
    cell_map_id: str = eqx.field(static=True)
    cell_global_ids: Array
    face_global_ids: Array
    source_mesh_id: str = eqx.field(static=True)
    source_geometry_id: str = eqx.field(static=True)
    source_map_id: str | None = eqx.field(static=True)
    local_view_id: str | None = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    exact_geometry: bool = eqx.field(static=True)

    @checked
    def __init__(
        self,
        mesh: CellMesh,
        /,
        *,
        geometry: CellGeometrySpec | None = None,
        exact_source_coordinates: CoordinateSourceBank | None = None,
        source_mesh_id: str | None = None,
        source_topology_id: str | None = None,
        source_geometry_id: str | None = None,
        source_map_id: str | None = None,
        local_view_id: str | None = None,
        face_global_ids: ConvertibleToArray | None = None,
    ) -> None:
        if not isinstance(mesh, CellMesh) or not isinstance(
            mesh.connectivity, PolyhedralConnectivity
        ):
            raise TypeError(
                "Polyhedral location requires a CellMesh with PolyhedralConnectivity."
            )
        if mesh.ambient_dimension != 3 or mesh.topological_dimension != 3:
            raise ValueError("Polyhedral location requires a full-dimensional 3-D mesh.")
        coordinates = np.asarray(mesh.coordinates, dtype=np.float64)
        if not np.all(np.isfinite(coordinates)):
            raise ValueError("Polyhedral coordinates must be finite.")
        geometry_ = CellGeometrySpec.affine(mesh) if geometry is None else geometry
        if not isinstance(geometry_, CellGeometrySpec):
            raise TypeError("geometry must be a CellGeometrySpec.")
        if not np.array_equal(np.asarray(geometry_.coordinates), coordinates):
            raise ValueError(
                "Polyhedral coordinate geometry does not match the mesh realization."
            )
        if exact_source_coordinates is not None:
            exact_coordinates = _exact_coordinate_bank(exact_source_coordinates)
            exact_execution_coordinates = np.asarray(exact_coordinates, dtype=np.float64)
            if not np.array_equal(exact_execution_coordinates, coordinates):
                raise ValueError(
                    "Exact polyhedral coordinates do not match the mesh realization."
                )
            exact_geometry = True
        elif geometry_.source_map is not None:
            if geometry_.exact_coordinate_bank is None:
                raise ValueError(
                    "Source-mapped polyhedral location requires an exact coordinate bank."
                )
            exact_coordinates = _exact_coordinate_bank(geometry_.exact_coordinate_bank)
            exact_execution_coordinates = np.asarray(exact_coordinates, dtype=np.float64)
            if not np.array_equal(exact_execution_coordinates, coordinates):
                raise ValueError(
                    "Exact polyhedral coordinates do not match the mesh realization."
                )
            exact_geometry = True
        else:
            exact_coordinates = ()
            exact_execution_coordinates = coordinates
            exact_geometry = False
        elements, _, _ = geometry_.resolve(mesh)
        if any(element.cell_kind != "polyhedron" for element in elements):
            raise ValueError(
                "Every polyhedral block must retain its native polyhedron coordinate element."
            )
        connectivity = mesh.connectivity
        face_offsets = np.asarray(connectivity.face_vertex_offsets, dtype=np.int64)
        face_values = np.asarray(connectivity.face_vertex_values, dtype=np.int64)
        cell_offsets = np.asarray(connectivity.cell_face_offsets, dtype=np.int64)
        cell_faces = np.asarray(connectivity.cell_face_values, dtype=np.int64)
        cell_vertex_offsets = np.asarray(connectivity.cell_vertex_offsets, dtype=np.int64)
        cell_vertex_values = np.asarray(connectivity.cell_vertex_values, dtype=np.int64)
        faces: list[tuple[tuple[int, int, int], ...]] = []
        for face in range(connectivity.face_count):
            loop = face_values[face_offsets[face] : face_offsets[face + 1]]
            triangulation = (
                _triangulate_face_exact(exact_coordinates, loop)
                if exact_geometry
                else _triangulate_face(coordinates, loop)
            )
            faces.append(triangulation)
        cells: list[tuple[int, ...]] = []
        triangles_by_cell: list[tuple[tuple[int, int, int], ...]] = []
        lower = np.empty((connectivity.cell_count, 3), dtype=np.float64)
        upper = np.empty_like(lower)
        for cell in range(connectivity.cell_count):
            vertices = tuple(
                int(value)
                for value in cell_vertex_values[
                    cell_vertex_offsets[cell] : cell_vertex_offsets[cell + 1]
                ]
            )
            cell_faces_ = cell_faces[cell_offsets[cell] : cell_offsets[cell + 1]]
            cells.append(vertices)
            triangles_by_cell.append(
                tuple(triangle for face in cell_faces_ for triangle in faces[int(face)])
            )
            vertex_coordinates = exact_execution_coordinates[
                np.asarray(vertices, dtype=np.int64)
            ]
            lower[cell] = np.nextafter(np.min(vertex_coordinates, axis=0), -np.inf)
            upper[cell] = np.nextafter(np.max(vertex_coordinates, axis=0), np.inf)
        identifiers = np.asarray(connectivity.cell_global_ids, dtype=np.int64)
        face_ids = (
            np.asarray(connectivity.face_global_ids, dtype=np.int64)
            if face_global_ids is None
            else np.asarray(face_global_ids)
        )
        if (
            identifiers.shape != (connectivity.cell_count,)
            or face_ids.shape != (connectivity.face_count,)
            or face_ids.dtype.kind not in "iu"
        ):
            raise ValueError(
                "Polyhedral global cell/face identities do not match the compact mesh."
            )
        source_mesh_id_ = mesh.mesh_id if source_mesh_id is None else str(source_mesh_id)
        source_topology_id_ = (
            mesh.topology_id if source_topology_id is None else str(source_topology_id)
        )
        source_geometry_id_ = (
            cell_geometry_id(geometry_)
            if source_geometry_id is None
            else str(source_geometry_id)
        )
        source_map_id_ = (
            geometry_.source_map_id if source_map_id is None else str(source_map_id)
        )
        if any(
            not value
            for value in (source_mesh_id_, source_topology_id_, source_geometry_id_)
        ) or (source_map_id is not None and not source_map_id_):
            raise ValueError("Polyhedral source binding identities must be non-empty.")
        self.mesh = mesh
        self.coordinates = mesh.coordinates
        self.exact_coordinates = exact_coordinates
        self.exact_geometry = exact_geometry
        self.cell_lower, self.cell_upper = jnp.asarray(lower), jnp.asarray(upper)
        self.cell_vertices = tuple(cells)
        self.cell_triangles = tuple(triangles_by_cell)
        self.face_triangles = tuple(faces)
        self.cell_count = connectivity.cell_count
        self.coordinate_count = coordinates.shape[0]
        self.ambient_dimension = self.reference_dimension = 3
        self.cell_global_ids = jnp.asarray(identifiers)
        self.face_global_ids = jnp.asarray(face_ids, dtype=jnp.int64)
        self.source_mesh_id = source_mesh_id_
        self.source_geometry_id = source_geometry_id_
        self.source_map_id = source_map_id_
        self.local_view_id = local_view_id
        self.topology_id = source_topology_id_
        self.cell_map_id = canonical_fingerprint(
            {
                "kind": "authored-polyhedral-cell-map",
                "mesh": source_mesh_id_,
                "topology": source_topology_id_,
                "geometry": source_geometry_id_,
                "source_map": source_map_id_,
                "local_view": local_view_id,
                "cell_ids": identifiers,
                "face_ids": face_ids,
                "exact_coordinates": tuple(
                    tuple(coefficient_payload(value) for value in point)
                    for point in exact_coordinates
                ),
                "triangles": faces,
            }
        )

    def support_geometry(self, support_id: str, /) -> object:
        connectivity = self.mesh.connectivity
        if not isinstance(connectivity, PolyhedralConnectivity):
            raise TypeError("Polyhedral map lost its canonical connectivity.")
        neighbor = np.asarray(connectivity.face_neighbor, dtype=np.int64)
        owner = np.asarray(connectivity.face_owner, dtype=np.int64)
        cell_offsets = np.asarray(connectivity.cell_face_offsets, dtype=np.int64)
        cell_faces = np.asarray(connectivity.cell_face_values, dtype=np.int64)
        cell_signs = np.asarray(connectivity.cell_face_sign_values, dtype=np.int8)
        boundary: list[tuple[int, int, int]] = []
        for face in np.flatnonzero(neighbor < 0):
            cell = int(owner[face])
            local = np.flatnonzero(
                cell_faces[cell_offsets[cell] : cell_offsets[cell + 1]] == face
            )[0]
            sign = int(cell_signs[cell_offsets[cell] + local])
            for triangle in self.face_triangles[int(face)]:
                boundary.append(
                    triangle if sign > 0 else (triangle[0], triangle[2], triangle[1])
                )
        if not boundary:
            raise ValueError("Polyhedral support has no represented exterior boundary.")
        faces = np.asarray(boundary, dtype=np.int32)
        used, compact = np.unique(faces, return_inverse=True)
        vertices: Array = self.coordinates[jnp.asarray(used)]
        triangle_faces: Array = jnp.asarray(compact.reshape(faces.shape).astype(np.int32))
        return MeshRegion(
            vertices,
            triangle_faces,
            feature_id=f"polyhedral-field-support:{support_id}",
        ).compile()


@final
class PreparedPolyhedralCellLocator(AbstractCellLocator, NonTrainableState):
    """Locate every authored polyhedral cell with exact boundary predicates."""

    __strict_contract__ = True

    cell_map: PreparedPolyhedralCellMap
    coordinates: Float[_PolyhedralCoordinateDim, Literal[3]]
    bvh: PackedBVH
    locator_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        mesh: CellMesh,
        /,
        *,
        geometry: CellGeometrySpec | None = None,
        exact_source_coordinates: CoordinateSourceBank | None = None,
        source_mesh_id: str | None = None,
        source_topology_id: str | None = None,
        source_geometry_id: str | None = None,
        source_map_id: str | None = None,
        local_view_id: str | None = None,
        face_global_ids: ConvertibleToArray | None = None,
    ) -> None:
        cell_map = PreparedPolyhedralCellMap(
            mesh,
            geometry=geometry,
            exact_source_coordinates=exact_source_coordinates,
            source_mesh_id=source_mesh_id,
            source_topology_id=source_topology_id,
            source_geometry_id=source_geometry_id,
            source_map_id=source_map_id,
            local_view_id=local_view_id,
            face_global_ids=face_global_ids,
        )
        self.cell_map, self.coordinates = cell_map, cell_map.coordinates
        self.bvh = prepare_bvh(
            cell_map.cell_lower, cell_map.cell_upper, dtype=jnp.float64
        )
        self.locator_id = canonical_fingerprint(
            {"kind": "prepared-polyhedral-cell-locator", "map": cell_map.cell_map_id}
        )

    @classmethod
    def from_owner_local_geometry(
        cls, view: Any, block_name: str, /
    ) -> PreparedPolyhedralCellLocator:
        """Prepare the canonical authored-face locator from one compact owner view."""
        from ._view_support import OwnerLocalCellGeometry

        if not isinstance(view, OwnerLocalCellGeometry):
            raise TypeError("view must be a canonical OwnerLocalCellGeometry.")
        matches = tuple(block for block in view.blocks if block.name == str(block_name))
        if len(matches) != 1 or matches[0].cell_kind != "polyhedron":
            raise ValueError(
                "Owner-local polyhedron name must identify one polyhedron block."
            )
        block = matches[0]
        if not np.asarray(block.cell_ids).size:
            raise ValueError("An ownerless polyhedron block has no local point locator.")
        old_vertices = np.asarray(block.vertices, dtype=np.int64)
        vertex_valid = np.asarray(block.vertex_valid, dtype=np.bool_)
        coordinate_routes = np.asarray(block.coordinate_routes, dtype=np.int64)
        local_shifts = (
            None
            if block.cell_vertex_shifts is None
            else np.asarray(block.cell_vertex_shifts, dtype=np.int64)
        )
        lattice = (
            None
            if view.periodic_lift_matrix is None
            else np.asarray(view.periodic_lift_matrix, dtype=np.float64)
        )
        periodic_rank = int(np.asarray(view.vertex_absolute_windings).shape[1])
        if (
            not view.coordinates_are_lifted
            and periodic_rank
            and (local_shifts is None or lattice is None)
        ):
            raise ValueError(
                "Unlifted periodic polyhedra require source windings and a lattice."
            )
        coordinates_by_vertex = np.asarray(view.vertex_coordinates, dtype=np.float64)
        exact_vertices = (
            None
            if view.exact_vertex_coordinates is None
            else tuple(
                tuple(value for value in row) for row in view.exact_vertex_coordinates
            )
        )
        exact_bank = view.exact_coordinates
        coordinates_rows: list[np.ndarray] = []
        exact_rows: list[tuple[Any, ...]] = []
        cells = []
        for cell in range(old_vertices.shape[0]):
            valid_slots = np.flatnonzero(vertex_valid[cell])
            row_by_vertex: dict[int, int] = {}
            cell_points = []
            cell_exact = []
            for slot in valid_slots:
                vertex = int(old_vertices[cell, slot])
                route = int(coordinate_routes[cell, slot])
                shift = (
                    np.zeros((periodic_rank,), dtype=np.int64)
                    if view.coordinates_are_lifted or local_shifts is None
                    else local_shifts[cell, slot]
                )
                base = coordinates_by_vertex[vertex]
                if exact_vertices is not None:
                    exact_base = exact_vertices[vertex]
                else:
                    exact_base = exact_bank[route]
                if periodic_rank and not view.coordinates_are_lifted:
                    if lattice is None:
                        raise ValueError("Periodic polyhedron lost its accepted lattice.")
                    displacement = shift @ lattice
                    exact_displacement = tuple(
                        sum(
                            (
                                int(shift[rank]) * Fraction(float(lattice[rank, axis]))
                                for rank in range(periodic_rank)
                            ),
                            Fraction(0),
                        )
                        for axis in range(coordinates_by_vertex.shape[1])
                    )
                    exact_point = tuple(
                        value + exact_displacement[axis]
                        for axis, value in enumerate(exact_base)
                    )
                    point = base + displacement
                else:
                    point, exact_point = base, exact_base
                row_by_vertex[vertex] = len(coordinates_rows) + len(cell_points)
                cell_points.append(np.asarray(point, dtype=np.float64))
                cell_exact.append(tuple(exact_point))
            coordinates_rows.extend(cell_points)
            exact_rows.extend(cell_exact)
            face_loops = block.polyhedron_faces[cell]
            face_signs = np.asarray(block.polyhedron_face_signs[cell], dtype=np.int8)
            oriented_faces = []
            for loop, sign in zip(face_loops, face_signs, strict=True):
                face = np.asarray(loop, dtype=np.int64)
                try:
                    local_face = np.asarray(
                        [row_by_vertex[int(vertex)] for vertex in face],
                        dtype=np.int32,
                    )
                except KeyError as error:
                    raise ValueError(
                        "Owner-local polyhedral face references an absent cell vertex."
                    ) from error
                oriented_faces.append(local_face if sign > 0 else local_face[::-1])
            cells.append(tuple(oriented_faces))
        coordinates = np.asarray(coordinates_rows, dtype=np.float64).reshape((-1, 3))
        exact_coordinates = tuple(exact_rows)
        if not np.array_equal(
            np.asarray(exact_coordinates, dtype=np.float64), coordinates
        ):
            raise ValueError(
                "Owner-local exact polyhedral source coefficients differ from their lifted shadows."
            )
        mesh = CellMesh.from_polyhedra(
            coordinates,
            cells,
            vertex_global_ids=np.arange(coordinates.shape[0], dtype=np.int64),
            cell_global_ids=np.asarray(block.cell_ids, dtype=np.int64),
        )
        return cls(
            mesh,
            exact_source_coordinates=exact_coordinates,
            source_mesh_id=view.source_mesh_id,
            source_topology_id=view.source_topology_id,
            source_geometry_id=view.source_geometry_id,
            source_map_id=view.source_map_id,
            local_view_id=view.local_view_id,
        )

    def support_geometry(self, support_id: str, /) -> object:
        return self.cell_map.support_geometry(support_id)

    def locate(
        self,
        points: ConvertibleToArray,
        /,
        *,
        cell_mask: ConvertibleToArray | None = None,
    ) -> CellLocationResult:
        if isinstance(points, Tracer):
            raise TypeError("Polyhedral cell location is a host exact-predicate query.")
        values = np.asarray(points, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 3:
            raise ValueError("Polyhedral locator points must have shape (points, 3).")
        if cell_mask is None:
            mask = np.ones((self.cell_map.cell_count,), dtype=np.bool_)
        else:
            mask = np.asarray(cell_mask)
            if mask.shape != (self.cell_map.cell_count,) or mask.dtype != np.dtype(
                np.bool_
            ):
                raise ValueError(
                    "cell_mask must contain one boolean per polyhedral cell."
                )
        finite = np.all(np.isfinite(values), axis=1)
        located: list[list[int]] = []
        complete = np.ones((values.shape[0],), dtype=np.bool_)
        coordinates = np.asarray(self.coordinates, dtype=np.float64)
        for point_index, point in enumerate(values):
            cells: list[int] = []
            if not finite[point_index]:
                complete[point_index] = False
                located.append(cells)
                continue
            target = np.asarray(point, dtype=np.float64)
            exact_target: _ExactPoint3 | None = (
                None
                if not self.cell_map.exact_geometry
                else (
                    Fraction(float(target[0])),
                    Fraction(float(target[1])),
                    Fraction(float(target[2])),
                )
            )
            candidates, valid, search_complete = point_select_leaf_items(
                jnp.asarray(target[None]),
                bvh=self.bvh,
                maximum_candidates=self.cell_map.cell_count,
                tolerance=0.0,
            )
            complete[point_index] = bool(np.asarray(search_complete[0]))
            for cell in np.asarray(candidates[0])[np.asarray(valid[0])]:
                cell_index = int(cell)
                if not mask[cell_index]:
                    continue
                crossings = 0
                touching = False
                for triangle_ids in self.cell_map.cell_triangles[cell_index]:
                    if self.cell_map.exact_geometry:
                        if exact_target is None:
                            raise RuntimeError(
                                "Exact polyhedral coordinates require an exact query point."
                            )
                        crossing, contact = _triangle_membership_exact(
                            exact_target,
                            triangle_ids,
                            self.cell_map.exact_coordinates,
                        )
                    else:
                        crossing, contact = _triangle_membership(
                            target, coordinates[np.asarray(triangle_ids)]
                        )
                    crossings += int(crossing)
                    touching |= contact
                if touching or crossings % 2:
                    cells.append(cell_index)
            located.append(cells)
        width = max((len(cells) for cells in located), default=0)
        candidate_cells = np.full((values.shape[0], width), -1, dtype=np.int32)
        candidate_reference = np.zeros((values.shape[0], width, 3), dtype=np.float64)
        count = np.asarray([len(cells) for cells in located], dtype=np.int32)
        cell_ids = np.full((values.shape[0],), -1, dtype=np.int32)
        for point_index, cells in enumerate(located):
            if not cells:
                continue
            cells.sort()
            candidate_cells[point_index, : len(cells)] = cells
            candidate_reference[point_index, : len(cells)] = values[point_index]
            cell_ids[point_index] = cells[0]
        inside = (count > 0) & complete & finite
        status = np.where(
            ~finite,
            int(CellLocationStatus.NONFINITE),
            np.where(
                ~complete,
                int(CellLocationStatus.CANDIDATE_SEARCH_INCOMPLETE),
                np.where(
                    inside,
                    int(CellLocationStatus.LOCATED),
                    int(CellLocationStatus.OUTSIDE),
                ),
            ),
        ).astype(np.int32)
        return CellLocationResult(
            jnp.asarray(np.where(inside, cell_ids, -1)),
            jnp.asarray(np.where(inside[:, None], values, 0.0)),
            jnp.empty((values.shape[0], 0), dtype=jnp.float64),
            jnp.asarray(np.where(inside, 0.0, np.inf)),
            jnp.zeros((values.shape[0],), dtype=jnp.int32),
            jnp.asarray(np.where(inside, 1.0, np.inf)),
            jnp.asarray(inside),
            jnp.zeros((values.shape[0],), dtype=jnp.bool_),
            jnp.asarray(count),
            jnp.asarray(status),
            jnp.asarray(inside),
            jnp.asarray(candidate_cells),
            jnp.asarray(candidate_reference),
            jnp.asarray(complete),
            self.locator_id,
        )


__all__ = ["PreparedPolyhedralCellMap", "PreparedPolyhedralCellLocator"]
