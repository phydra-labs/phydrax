#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

from fractions import Fraction

import numpy as np

from phydrax.geometry._planar_primitives import convex_hull, plane_key, project, turn
from phydrax.meshing._failures import MeshingFailureCategory

from ..discretization._cell_complex import PolyhedralConnectivity
from ..discretization._cell_geometry import CellGeometrySpec
from ..discretization._cell_mesh import CellMesh
from ..discretization._coordinate_budget import CoordinateEnclosureBudget
from ..discretization._exact_power_geometry import ExactPowerCellGeometryRestrictionSource
from ..linalg._small_batched import prepare_exact_small_linear_actions
from ..typing import checked
from ._polyhedral_adaptation_shared import (
    bound_power_source,
    polyhedral_face_loops,
    PolyhedralAdaptationError,
)


# Mutable host scratch owns incremental vertex discovery; it is not retained
# as a module or placed in a device PyTree.
class _PlaneCut:
    def __init__(
        self,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        plane: np.ndarray,
        maximum_vertices: int,
        *,
        budget: CoordinateEnclosureBudget,
    ) -> None:
        parent = bound_power_source(geometry)
        if parent is None:
            raise ValueError(
                "PolyhedralPlane.geometry must retain its source construction; got missing power source"
            )
        self.parent = parent
        self.budget = budget
        self.points = list(geometry.rational_source_coordinates(budget=budget))
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
        self.maximum_vertices = maximum_vertices
        self.vertex_bytes_upper = 256 + 3 * (
            128 + 2 * ((self.parent.maximum_bits + 7) // 8)
        )
        self.check_vertices()
        self.budget.reserve(0, len(self.points) * self.vertex_bytes_upper)

    def check_vertices(self) -> None:
        if len(self.points) > self.maximum_vertices:
            raise PolyhedralAdaptationError(
                "PolyhedralPlane.vertices must fit vertex and rational scratch budgets; got exhausted vertex budget",
                category=MeshingFailureCategory.RESOURCE_EXHAUSTED,
            )

    def intersect(self, first: int, second: int) -> int:
        self.budget.reserve(40)
        if self.signs[first] == 0:
            return first
        if self.signs[second] == 0:
            return second
        a, b = self.points[first], self.points[second]
        divisor = self.signs[first] - self.signs[second]
        solve = prepare_exact_small_linear_actions(
            ((divisor,),), ((self.signs[first],),), budget=self.budget
        )
        if solve.actions is None:
            raise PolyhedralAdaptationError(
                "PolyhedralPlane.edge_solve must have full rank; got deficient intersection rank"
            )
        parameter = solve.actions[0][0]
        if not 0 < parameter < 1:
            raise PolyhedralAdaptationError(
                "PolyhedralPlane.edge_parameter must lie strictly inside its parent edge; got out-of-interior intersection"
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
                "PolyhedralPlane.rational_bits must fit the integer-bit budget; got excessive rational bits",
                category=MeshingFailureCategory.RESOURCE_EXHAUSTED,
            )
        if point in self.keys:
            return self.keys[point]
        if self.next_vertex >= 2**63:
            raise PolyhedralAdaptationError(
                "PolyhedralPlane.vertex_ids must fit signed-int64 scientific IDs; got exhausted vertex IDs",
                category=MeshingFailureCategory.RESOURCE_EXHAUSTED,
            )
        self.budget.reserve(0, self.vertex_bytes_upper)
        index = len(self.points)
        self.points.append(point)
        self.signs.append(Fraction(0))
        self.witnesses.append((min(first, second), max(first, second)))
        self.plane_ids.append(0)
        self.vertex_ids.append(self.next_vertex)
        self.next_vertex += 1
        self.keys[point] = index
        self.check_vertices()
        return index

    def clip(self, row: np.ndarray, sign: int) -> np.ndarray:
        self.budget.reserve(4 * row.size)
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
        projected = project(original, axes, budget=self.budget)
        hull = convex_hull(projected, budget=self.budget)
        if len(hull) < 3:
            raise PolyhedralAdaptationError(
                "PolyhedralPlane.section must contain at least three hull vertices; got deficient section hull"
            )
        lookup = dict(zip(projected, sorted(indices), strict=True))
        ordered = []
        for a, b in zip(hull, (*hull[1:], hull[0]), strict=True):
            axis = next(index for index in range(2) if a[index] != b[index])
            members = [
                ((point[axis] - a[axis]) / (b[axis] - a[axis]), point)
                for point in projected
                if turn(a, b, point, budget=self.budget) == 0
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


@checked
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
    budget: CoordinateEnclosureBudget,
) -> tuple[CellMesh, CellGeometrySpec]:
    with budget.bound_stage(maximum_work, maximum_bytes):
        geometry.resolve(mesh, budget=budget)
        construction = _PlaneCut(mesh, geometry, plane, maximum_vertices, budget=budget)
        connectivity = mesh.connectivity
        if not isinstance(connectivity, PolyhedralConnectivity):
            raise TypeError(
                "PolyhedralPlane.connectivity must be packed polyhedral connectivity; got different connectivity kind"
            )
        local_loops = polyhedral_face_loops(mesh)
        cell_ids = np.asarray(connectivity.cell_global_ids).tolist()
        face_ids, face_offsets, face_values = (
            np.asarray(connectivity.face_global_ids),
            np.asarray(connectivity.cell_face_offsets),
            np.asarray(connectivity.cell_face_values),
        )
        cut = [
            cell
            for cell, rows in enumerate(local_loops)
            if min(construction.signs[index] for index in np.unique(np.concatenate(rows)))
            < 0
            < max(construction.signs[index] for index in np.unique(np.concatenate(rows)))
        ]
        if len(local_loops) + len(cut) > maximum_cells:
            raise PolyhedralAdaptationError(
                "PolyhedralPlane.cells must fit the target cell budget; got excessive child cells",
                category=MeshingFailureCategory.RESOURCE_EXHAUSTED,
            )
        if not cut:
            return mesh, geometry
        result, result_ids, next_cell = [], [], max(cell_ids) + 1
        for cell, rows in enumerate(local_loops):
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
                        plane_key((face[0], face[index], face[index + 1]), budget=budget)
                        for index in range(1, len(face) - 1)
                        if plane_key(
                            (face[0], face[index], face[index + 1]), budget=budget
                        )
                        is not None
                    ),
                    None,
                )
                if key is None or any(
                    key[3]
                    * sum((key[0][axis] * point[axis] for axis in range(3)), key[0][3])
                    > 0
                    for point in vertices
                ):
                    raise PolyhedralAdaptationError(
                        "PolyhedralPlane.source_cells must be convex with proved ideal facets; got nonconvex or deficient source facets"
                    )
                if (
                    min(construction.signs[index] for index in row)
                    < 0
                    < max(construction.signs[index] for index in row)
                    and int(face_ids[face_values[face_offsets[cell] + local]])
                    in protected
                ):
                    raise PolyhedralAdaptationError(
                        "PolyhedralPlane.protected_faces must remain unsubdivided; got an intersected protected face"
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
                        "PolyhedralPlane.cell_ids must fit signed-int64 scientific IDs; got exhausted cell IDs",
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
        exact = source.prepare(budget=budget)
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
        return target, CellGeometrySpec.power(target, source, budget=budget)
