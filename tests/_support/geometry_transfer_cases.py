# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Analytic polynomial-shear sources with stable simplex fixture row identities."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

import phydrax as phx
from phydrax.discretization import CellGeometrySpec, CellMesh
from phydrax.meshing import CellMeshingResult
from phydrax.meshing._curving import straight_geometry
from tests._support.cell_meshes import unit_square


_SHEAR = 0.1
_CONTRACT = phx.SpatialCoordinateContract.si()


def shear_term(degree: int, dimension: int) -> Callable[[np.ndarray], np.ndarray]:
    """``x + a q`` with ``q`` a monomial of total ``degree`` in the other axes."""

    def term(points: np.ndarray) -> np.ndarray:
        if dimension == 2:
            return points[..., 1] ** degree
        return points[..., 1] ** (degree - 1) * points[..., 2]

    return term


def apply_shear(
    points: np.ndarray, term: Callable[[np.ndarray], np.ndarray], sign: float
) -> np.ndarray:
    moved = np.array(points, dtype=np.float64, copy=True)
    moved[..., 0] += sign * _SHEAR * term(points)
    return moved


def curved_shear_source(
    mesh: CellMesh, degree: int, term: Callable[[np.ndarray], np.ndarray]
) -> CellMeshingResult:
    """Certified result whose coordinate map is the shear of the straight mesh."""

    mesh = phx.meshing.canonicalize_cell_mesh(mesh)
    straight = straight_geometry(mesh, degree)
    names = [block.name for block in mesh.blocks]
    geometry = CellGeometrySpec(
        dict(zip(names, straight.elements, strict=True)),
        dict(zip(names, straight.geometry_dofs, strict=True)),
        apply_shear(np.asarray(straight.coordinates), term, 1.0),
    )
    moved = mesh.with_coordinates(
        apply_shear(np.asarray(mesh.coordinates), term, 1.0), numeric_version="sheared"
    )
    return phx.meshing.certify_cell_mesh(moved, _CONTRACT, geometry=geometry)


def transfer_triangles(size: int) -> CellMesh:
    """Return the xy-indexed square grid used by the shear-transfer contracts."""
    mesh = unit_square(size)
    cells = np.asarray(mesh.blocks[0].vertices).reshape((-1, 2, 3))
    ordered = cells[:, ::-1, :][:, :, (0, 2, 1)].reshape((-1, 3))
    return CellMesh.from_triangles(np.asarray(mesh.coordinates)[:, ::-1], ordered)
