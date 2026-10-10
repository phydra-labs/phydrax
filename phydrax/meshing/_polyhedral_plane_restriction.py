#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

from fractions import Fraction

import numpy as np

from .._meshcore import current_native_execution_budget
from ..discretization._cell_complex import PolyhedralConnectivity
from ..discretization._cell_geometry import CellGeometrySpec
from ..discretization._cell_mesh import CellMesh
from ..discretization._exact_power_geometry import ExactPowerCellGeometryRestrictionSource
from ..geometry._planar_coverage import convex_hull, plane_key, project, turn
from ..linalg._small_batched import prepare_exact_small_linear_actions
from ._contracts import MeshingFailureCategory
from ._polyhedral_adaptation import _loops, _power_source, PolyhedralAdaptationError


class _PlaneCut:
    def __init__(
        self,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        plane: np.ndarray,
        maximum_vertices: int,
        maximum_work: int,
        maximum_bytes: int,
    ) -> None:
        parent = _power_source(geometry)
        if parent is None:
            raise ValueError("Exact plane cuts require the current source construction.")
        self.parent = parent
        self.points = list(geometry.source_coordinates())
        self.parent_count = len(self.points)
        self.plane = tuple(Fraction(float(value)) for value in plane)
        self.signs = [
            sum((self.plane[axis] * point[axis] for axis in range(3)), -self.plane[3])
            for point in self.points
        ]
        self.witnesses = [(index, -1) for index in range(self.parent_count)]
        self.plane_ids = [-1] * self.parent_count
        self.vertex_ids = np.asarray(mesh.vertex_global_ids).tolist()
        self.next_vertex = max(self.vertex_ids) + 1
        self.keys = {point: index for index, point in enumerate(self.points)}
        self.maximum_vertices, self.maximum_work, self.maximum_bytes = (
            maximum_vertices,
            maximum_work,
            maximum_bytes,
        )
        self.work = 0
        self.check_storage()

    def charge(self, work: int) -> None:
        if self.work + work > self.maximum_work:
            raise PolyhedralAdaptationError(
                "Exact plane restriction exceeds its original work budget.",
                category=MeshingFailureCategory.RESOURCE_EXHAUSTED,
            )
        execution = current_native_execution_budget()
        if execution is not None:
            execution.charge(work=work)
        self.work += work

    def check_storage(self) -> None:
        per_vertex = 256 + 3 * (128 + 2 * ((self.parent.maximum_bits + 7) // 8))
        if (
            len(self.points) > self.maximum_vertices
            or len(self.points) * per_vertex > self.maximum_bytes
        ):
            raise PolyhedralAdaptationError(
                "Exact plane restriction exceeds its vertex or rational scratch budget.",
                category=MeshingFailureCategory.RESOURCE_EXHAUSTED,
            )

    def intersect(self, first: int, second: int) -> int:
        self.charge(40)
        if self.signs[first] == 0:
            return first
        if self.signs[second] == 0:
            return second
        a, b = self.points[first], self.points[second]
        divisor = self.signs[first] - self.signs[second]
        solve = prepare_exact_small_linear_actions(((divisor,),), ((self.signs[first],),))
        if solve.actions is None:
            raise PolyhedralAdaptationError(
                "Exact plane-edge intersection has deficient rank."
            )
        parameter = solve.actions[0][0]
        if not 0 < parameter < 1:
            raise PolyhedralAdaptationError(
                "Exact plane-edge witness does not intersect its parent edge interior."
            )
        point = tuple(x + parameter * (y - x) for x, y in zip(a, b, strict=True))
        point = (point[0], point[1], point[2])
        if (
            max(
                max(value.numerator.bit_length(), value.denominator.bit_length())
                for value in point
            )
            > self.parent.maximum_bits
        ):
            raise PolyhedralAdaptationError(
                "Exact plane restriction exceeds its rational integer-bit budget.",
                category=MeshingFailureCategory.RESOURCE_EXHAUSTED,
            )
        if point in self.keys:
            return self.keys[point]
        if self.next_vertex >= 2**63:
            raise PolyhedralAdaptationError(
                "Exact plane restriction exhausts scientific vertex IDs.",
                category=MeshingFailureCategory.RESOURCE_EXHAUSTED,
            )
        index = len(self.points)
        self.points.append(point)
        self.signs.append(Fraction(0))
        self.witnesses.append((min(first, second), max(first, second)))
        self.plane_ids.append(0)
        self.vertex_ids.append(self.next_vertex)
        self.next_vertex += 1
        self.keys[point] = index
        self.check_storage()
        return index

    def clip(self, row: np.ndarray, sign: int) -> np.ndarray:
        self.charge(4 * row.size)
        result = []
        previous = int(row[-1])
        for value in row:
            current = int(value)
            if (sign * self.signs[previous] <= 0) != (sign * self.signs[current] <= 0):
                result.append(self.intersect(previous, current))
            if sign * self.signs[current] <= 0:
                result.append(current)
            previous = current
        unique = []
        for vertex in result:
            if not unique or unique[-1] != vertex:
                unique.append(vertex)
        if len(unique) > 1 and unique[0] == unique[-1]:
            unique.pop()
        return np.asarray(unique, dtype=np.int32)

    def section(self, indices: set[int], sign: int) -> np.ndarray:
        pivot = next(axis for axis in range(3) if self.plane[axis])
        axes = ((1, 2), (2, 0), (0, 1))[pivot]
        original = tuple(self.points[index] for index in sorted(indices))
        projected = project(original, axes)
        hull = convex_hull(projected)
        if len(hull) < 3:
            raise PolyhedralAdaptationError(
                "Exact plane restriction has a deficient section face."
            )
        lookup = dict(zip(projected, sorted(indices), strict=True))
        ordered = []
        for a, b in zip(hull, (*hull[1:], hull[0]), strict=True):
            axis = next(index for index in range(2) if a[index] != b[index])
            members = [
                ((point[axis] - a[axis]) / (b[axis] - a[axis]), point)
                for point in projected
                if turn(a, b, point) == 0
                and all(
                    min(x, y) <= p <= max(x, y)
                    for p, x, y in zip(point, a, b, strict=True)
                )
            ]
            ordered.extend(
                lookup[point] for parameter, point in sorted(members) if parameter < 1
            )
        if sign * self.plane[pivot] < 0:
            ordered.reverse()
        return np.asarray(ordered, dtype=np.int32)


def restrict_polyhedral_plane(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    plane: np.ndarray,
    protected: set[int],
    vertex_support: dict[int, set[int]],
    /,
    *,
    maximum_cells: int,
    maximum_vertices: int,
    maximum_work: int,
    maximum_bytes: int,
) -> tuple[CellMesh, CellGeometrySpec]:
    geometry.resolve(mesh)
    construction = _PlaneCut(
        mesh, geometry, plane, maximum_vertices, maximum_work, maximum_bytes
    )
    connectivity = mesh.connectivity
    if not isinstance(connectivity, PolyhedralConnectivity):
        raise TypeError("Exact plane cuts require packed polyhedral connectivity.")
    loops = _loops(mesh)
    cell_ids = np.asarray(connectivity.cell_global_ids).tolist()
    face_ids, face_offsets, face_values = (
        np.asarray(connectivity.face_global_ids),
        np.asarray(connectivity.cell_face_offsets),
        np.asarray(connectivity.cell_face_values),
    )
    cut = [
        cell
        for cell, rows in enumerate(loops)
        if min(construction.signs[index] for index in np.unique(np.concatenate(rows)))
        < 0
        < max(construction.signs[index] for index in np.unique(np.concatenate(rows)))
    ]
    if len(loops) + len(cut) > maximum_cells:
        raise PolyhedralAdaptationError(
            "Exact plane restriction exceeds its target cell budget.",
            category=MeshingFailureCategory.RESOURCE_EXHAUSTED,
        )
    if not cut:
        return mesh, geometry
    result, result_ids, next_cell = [], [], max(cell_ids) + 1
    for cell, rows in enumerate(loops):
        if cell not in cut:
            result.append(rows)
            result_ids.append(cell_ids[cell])
            continue
        vertices = tuple(
            construction.points[index] for index in np.unique(np.concatenate(rows))
        )
        for local, row in enumerate(rows):
            face = tuple(construction.points[index] for index in row)
            key = next(
                (
                    plane_key((face[0], face[index], face[index + 1]))
                    for index in range(1, len(face) - 1)
                    if plane_key((face[0], face[index], face[index + 1])) is not None
                ),
                None,
            )
            if key is None or any(
                key[3] * sum((key[0][axis] * point[axis] for axis in range(3)), key[0][3])
                > 0
                for point in vertices
            ):
                raise PolyhedralAdaptationError(
                    "Exact plane subdivision requires convex source cells with proven ideal facets."
                )
            if (
                min(construction.signs[index] for index in row)
                < 0
                < max(construction.signs[index] for index in row)
                and int(face_ids[face_values[face_offsets[cell] + local]]) in protected
            ):
                raise PolyhedralAdaptationError(
                    "Exact plane closure would subdivide a protected source face."
                )
        for sign in (1, -1):
            child, section = [], set()
            for row in rows:
                face = construction.clip(row, sign)
                if face.size >= 3:
                    child.append(face)
                    section.update(
                        int(index) for index in face if construction.signs[index] == 0
                    )
            child.append(construction.section(section, sign))
            if next_cell >= 2**63:
                raise PolyhedralAdaptationError(
                    "Exact plane restriction exhausts scientific cell IDs.",
                    category=MeshingFailureCategory.RESOURCE_EXHAUSTED,
                )
            result.append(tuple(child))
            result_ids.append(next_cell)
            next_cell += 1
    source = ExactPowerCellGeometryRestrictionSource(
        construction.parent,
        plane[None, :],
        np.asarray(construction.witnesses, dtype=np.int64),
        np.asarray(construction.plane_ids, dtype=np.int64),
    )
    exact = source.prepare()
    target = CellMesh.from_polyhedra(
        exact.rounded_vertices,
        result,
        vertex_global_ids=np.asarray(construction.vertex_ids, dtype=np.int64),
        cell_global_ids=np.asarray(result_ids, dtype=np.int64),
    )
    for index in range(construction.parent_count, len(construction.vertex_ids)):
        first, second = construction.witnesses[index]
        vertex_support[construction.vertex_ids[index]] = (
            vertex_support[construction.vertex_ids[first]]
            | vertex_support[construction.vertex_ids[second]]
        )
    return target, CellGeometrySpec.power(target, source)
