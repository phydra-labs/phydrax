#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact represented coordinate maps and cell integrals for domain proofs."""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations

import numpy as np
from jax.typing import ArrayLike

from ..discretization._cell_complex import PolyhedralConnectivity
from ..discretization._cell_geometry import (
    CellGeometrySpec,
    CellVertexGeometryElement,
    coordinate_effective_basis,
    PerCellRestrictedCellGeometryElement,
    ScalarCoordinateElement,
)
from ..discretization._cell_mesh import CellMesh
from ..discretization._coordinate_enclosure import (
    affine_map,
    evaluate_expression,
    ExactScalar,
    Expression,
    linear_combinations,
    Polynomial,
    RestrictedCoordinateMapPreparation,
)
from ..discretization._reference_cell import reference_cell_topology
from ._mapped_coverage import integrate_jacobian, map_domain
from ._planar_primitives import plane_key


def _coordinate_route_matrix(value: ArrayLike, /) -> np.ndarray:
    rows = np.asarray(value, dtype=np.int64)
    if rows.ndim != 2:
        raise ValueError("Coordinate DOF routes must be a rank-two matrix.")
    return rows


def cell_coordinate_expressions(
    mesh: CellMesh, geometry: CellGeometrySpec, /
) -> tuple[tuple[Expression, ...] | None, ...]:
    """Return each exact physical map; vertex-defined polyhedra have no reference map."""
    elements, routes, _ = geometry.resolve(mesh)
    coefficients = geometry.source_coordinates()
    maps: list[tuple[Expression, ...] | None] = []
    for block, element, route_bank in zip(mesh.blocks, elements, routes, strict=True):
        if isinstance(element, CellVertexGeometryElement):
            maps.extend((None,) * block.cell_count)
            continue
        if not isinstance(element, ScalarCoordinateElement):
            raise TypeError(
                "Exact domain maps require a supported scalar coordinate element."
            )
        routes_host = _coordinate_route_matrix(route_bank)
        preparation = (
            RestrictedCoordinateMapPreparation(
                element, coefficients, routes_host, mode="source"
            )
            if routes_host.shape[0] and isinstance(element, PerCellRestrictedCellGeometryElement)
            else None
        )
        for row_index in range(routes_host.shape[0]):
            route = routes_host[row_index]
            local = tuple(coefficients[int(index)] for index in route)
            expressions = None if preparation is None else preparation.expressions(row_index)
            if expressions is None:
                basis = coordinate_effective_basis(element, cell_row=row_index)
                if len(basis) != len(local):
                    raise ValueError(
                        "Exact coordinate basis and local source bank widths differ."
                    )
                components = tuple(
                    tuple(local[dof][axis] for dof in range(len(local)))
                    for axis in range(len(local[0]))
                )
                expressions = linear_combinations(basis, components)
            maps.append(expressions)
    return tuple(maps)


def polyhedral_vertex_coordinates(
    mesh: CellMesh, geometry: CellGeometrySpec, /
) -> tuple[tuple[Fraction, Fraction, Fraction] | None, ...]:
    """Recover exact physical coordinates of every topological polyhedral vertex."""
    connectivity = mesh.connectivity
    if (
        mesh.topological_dimension != 3
        or mesh.ambient_dimension != 3
        or not isinstance(connectivity, PolyhedralConnectivity)
    ):
        raise ValueError(
            "Exact polygonal target facets require 3-D PolyhedralConnectivity."
        )
    elements, routes, _ = geometry.resolve(mesh)
    coefficients = geometry.source_coordinates()
    coordinate_maps = cell_coordinate_expressions(mesh, geometry)
    vertex_count = np.asarray(mesh.coordinates).shape[0]
    points: list[tuple[Fraction, Fraction, Fraction] | None] = [None] * vertex_count
    cursor = 0

    def retain(vertex: int, values: tuple[object, ...], /) -> None:
        if len(values) != 3:
            raise ValueError(
                "Polyhedral coordinates must have three physical components."
            )
        exact: list[Fraction] = []
        for value in values:
            if isinstance(value, Fraction):
                exact.append(value)
            elif isinstance(value, (int, np.integer)) and not isinstance(
                value, (bool, np.bool_)
            ):
                exact.append(Fraction(int(value)))
            else:
                raise ValueError(
                    "Polyhedral exact coverage requires rational vertex coordinates."
                )
        point = (exact[0], exact[1], exact[2])
        previous = points[vertex]
        if previous is not None and previous != point:
            raise ValueError(
                "Incident polyhedral cells disagree on exact shared-vertex coordinates."
            )
        points[vertex] = point

    for block, element, route_bank in zip(mesh.blocks, elements, routes, strict=True):
        vertices = np.asarray(block.vertices, dtype=np.int64)
        valid = np.asarray(block.vertex_valid, dtype=np.bool_)
        routes_host = np.asarray(route_bank, dtype=np.int64)
        if isinstance(element, CellVertexGeometryElement):
            for row in range(block.cell_count):
                for slot in np.flatnonzero(valid[row]):
                    vertex = int(vertices[row, slot])
                    coefficient = coefficients[int(routes_host[row, slot])]
                    retain(vertex, coefficient)
                cursor += 1
            continue
        if block.cell_kind in ("polygon", "polyhedron"):
            raise ValueError(
                "Variable-topology cells require their native vertex geometry element."
            )
        reference = reference_cell_topology(block.cell_kind)
        reference_points_list = []
        for point in reference.vertices:
            exact_point = []
            for value in point:
                if value == 0.0:
                    exact_point.append(Fraction(0))
                elif value == 1.0:
                    exact_point.append(Fraction(1))
                else:
                    raise ValueError(
                        "Polyhedral coverage requires canonical rational reference vertices."
                    )
            reference_points_list.append(tuple(exact_point))
        reference_points = tuple(reference_points_list)
        vertex_interpolated = routes_host.shape[1] == len(reference_points)
        for row in range(block.cell_count):
            expressions = coordinate_maps[cursor]
            # Vertex-interpolated multilinear maps are fixed by their corners,
            # so every face image is its corner interpolation; face planarity is
            # proved where the face loops are consumed.
            if expressions is None or (
                not vertex_interpolated and affine_map(expressions, 3) is None
            ):
                raise ValueError(
                    "Polyhedral coverage requires vertex-interpolated or affine maps on fixed-cell blocks."
                )
            for slot, reference_point in enumerate(reference_points):
                retain(
                    int(vertices[row, slot]),
                    evaluate_map(expressions, reference_point),
                )
            cursor += 1
    connected_vertices = np.asarray(connectivity.cell_vertex_values, dtype=np.int64)
    if cursor != connectivity.cell_count or any(
        points[int(vertex)] is None for vertex in connected_vertices
    ):
        raise ValueError("Polyhedral exact vertex maps do not cover the connected mesh.")
    return tuple(points)


def polyhedral_cell_measures(
    mesh: CellMesh, geometry: CellGeometrySpec, /
) -> tuple[Fraction | None, ...]:
    """Integrate oriented planar face loops as exact signed polyhedron volumes.

    Fixed-reference cells have exact Jacobian integrals instead and stay ``None``.
    """
    connectivity = mesh.connectivity
    if not isinstance(connectivity, PolyhedralConnectivity):
        raise ValueError("Exact polyhedral cell measures require PolyhedralConnectivity.")
    points = polyhedral_vertex_coordinates(mesh, geometry)
    face_offsets = np.asarray(connectivity.face_vertex_offsets, dtype=np.int64)
    face_vertices = np.asarray(connectivity.face_vertex_values, dtype=np.int64)
    cell_face_offsets = np.asarray(connectivity.cell_face_offsets, dtype=np.int64)
    cell_faces = np.asarray(connectivity.cell_face_values, dtype=np.int64)
    face_signs = np.asarray(connectivity.cell_face_sign_values, dtype=np.int64)
    cell_kinds = tuple(
        block.cell_kind for block in mesh.blocks for _ in range(block.cell_count)
    )
    measures: list[Fraction | None] = []
    for cell in range(connectivity.cell_count):
        if cell_kinds[cell] != "polyhedron":
            measures.append(None)
            continue
        volume = Fraction(0)
        for local_face, face in enumerate(
            cell_faces[cell_face_offsets[cell] : cell_face_offsets[cell + 1]]
        ):
            start, stop = int(face_offsets[face]), int(face_offsets[face + 1])
            face_points = tuple(points[int(index)] for index in face_vertices[start:stop])
            if any(point is None for point in face_points) or len(face_points) < 3:
                raise ValueError(
                    "Polyhedral face lacks exact nondegenerate vertex coordinates."
                )
            polygon = tuple(point for point in face_points if point is not None)
            plane = None
            for candidate in combinations(polygon, 3):
                try:
                    plane = plane_key(candidate)
                except ValueError:
                    continue
                break
            if plane is None or any(
                sum(
                    coefficient * value
                    for coefficient, value in zip(plane[:3], point, strict=True)
                )
                + plane[3]
                != 0
                for point in polygon
            ):
                raise ValueError("Polyhedral face vertices are not exactly coplanar.")
            p0 = polygon[0]
            oriented = Fraction(0)
            for index in range(1, len(polygon) - 1):
                p1, p2 = polygon[index], polygon[index + 1]
                oriented += (
                    p0[0] * (p1[1] * p2[2] - p1[2] * p2[1])
                    - p0[1] * (p1[0] * p2[2] - p1[2] * p2[0])
                    + p0[2] * (p1[0] * p2[1] - p1[1] * p2[0])
                ) / 6
            volume += int(face_signs[cell_face_offsets[cell] + local_face]) * oriented
        measures.append(volume)
    return tuple(measures)


def affine_cell_measure(
    expressions: tuple[Expression, ...], cell_kind: str, /
) -> ExactScalar:
    """Integrate one represented polynomial Jacobian exactly."""
    return integrate_jacobian(expressions, cell_kind)


def reference_cell_measure(cell_kind: str, /) -> ExactScalar:
    """Exact measure of a supported standard 2-D/3-D reference cell."""
    dimension = (
        2 if cell_kind in ("triangle", "quadrilateral") or cell_kind == "simplex:2" else 3
    )
    unit: Polynomial = {(0,) * dimension: Fraction(1)}
    from ._mapped_coverage import integrate

    return integrate(unit, map_domain(cell_kind), dimension)


def evaluate_map(
    expressions: tuple[Expression, ...], point: tuple[ExactScalar, ...], /
) -> tuple[ExactScalar, ...]:
    return tuple(evaluate_expression(expression, point) for expression in expressions)


__all__ = [
    "affine_cell_measure",
    "cell_coordinate_expressions",
    "evaluate_map",
    "reference_cell_measure",
    "polyhedral_cell_measures",
    "polyhedral_vertex_coordinates",
]
