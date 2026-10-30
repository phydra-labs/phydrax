#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from fractions import Fraction
from itertools import permutations
from math import factorial

import numpy as np

from ..typing import checked
from ._cell_geometry import CellGeometrySpec
from ._cell_mesh import CellMesh
from ._periodic_exact import identification_id
from ._periodic_topology import PeriodicMeasureReport, PeriodicMeshTopology


def _determinant_exact(matrix: tuple[tuple[Fraction, ...], ...], /) -> Fraction:
    dimension = len(matrix)
    result = Fraction()
    for order in permutations(range(dimension)):
        inversions = sum(
            order[first] > order[second]
            for first in range(dimension)
            for second in range(first + 1, dimension)
        )
        term = Fraction(-1 if inversions % 2 else 1)
        for row, column in enumerate(order):
            term *= matrix[row][column]
        result += term
    return result


def _affine_simplex_measure(points: np.ndarray, /) -> float:
    dimension = points.shape[0] - 1
    if points.shape != (dimension + 1, dimension) or dimension not in (1, 2, 3):
        raise ValueError(
            "Periodic measures support full-dimensional 1D, 2D, or 3D simplices."
        )
    origin = tuple(Fraction.from_float(float(value)) for value in points[0])
    jacobian = tuple(
        tuple(
            Fraction.from_float(float(points[row + 1, column])) - origin[column]
            for column in range(dimension)
        )
        for row in range(dimension)
    )
    determinant = abs(_determinant_exact(jacobian))
    return float(determinant / factorial(dimension))


@checked
def periodic_orbit_measures(
    mesh: CellMesh,
    /,
    *,
    geometry: CellGeometrySpec | None = None,
) -> PeriodicMeasureReport:
    """Integrate affine simplex measures once per quotient top-cell orbit."""
    if not isinstance(mesh, CellMesh):
        raise TypeError("mesh must be a CellMesh.")
    topology = mesh.periodic_topology
    if not isinstance(topology, PeriodicMeshTopology):
        raise ValueError("mesh must retain a PeriodicMeshTopology descriptor.")
    topology.require_lifted(mesh.topology, mesh.topology_id, mesh.blocks)
    topology.require_coordinates(mesh.coordinates)
    selected_geometry = CellGeometrySpec.affine(mesh) if geometry is None else geometry
    if not isinstance(selected_geometry, CellGeometrySpec):
        raise TypeError("geometry must be a CellGeometrySpec.")
    elements, routes, coordinates = selected_geometry.resolve(mesh)
    if (
        selected_geometry.source_map is not None
        or selected_geometry.restriction_source is not None
    ):
        raise ValueError(
            "Periodic affine measures do not approximate source-mapped coordinate geometry."
        )
    dimension = mesh.topological_dimension
    block_rows = []
    for block, element, route in zip(mesh.blocks, elements, routes, strict=True):
        if block.cell_kind not in (
            f"simplex:{dimension}",
            {1: "interval", 2: "triangle", 3: "tetrahedron"}.get(dimension),
        ):
            raise ValueError("Periodic affine measures require simplex cell blocks.")
        if route.shape != (block.cell_count, dimension + 1):
            raise ValueError(
                "Periodic affine measures require degree-one simplex coordinate routes."
            )
        if not np.array_equal(np.asarray(route), np.asarray(block.vertices)):
            raise ValueError(
                "Periodic affine measures require vertex-indexed simplex coordinates."
            )
        block_rows.append(np.asarray(route, dtype=np.int64))
    cells = np.concatenate(block_rows, axis=0)
    if cells.shape[0] != int(topology.lifted_offsets[-1] - topology.lifted_offsets[-2]):
        raise ValueError("Periodic top-cell rows do not match the quotient descriptor.")
    quotient_begin, quotient_end = map(int, np.asarray(topology.quotient_offsets)[-2:])
    lifted_representatives = np.asarray(topology.representatives, dtype=np.int64)[
        quotient_begin:quotient_end
    ]
    points = np.asarray(coordinates, dtype=np.float64)
    measures = np.asarray(
        [_affine_simplex_measure(points[cells[row]]) for row in lifted_representatives],
        dtype=np.float64,
    )
    return PeriodicMeasureReport(
        measures,
        identification_id(topology.cell),
        topology.periodic_topology_id,
        selected_geometry.geometry_layout_id,
    )


__all__ = ["periodic_orbit_measures"]
