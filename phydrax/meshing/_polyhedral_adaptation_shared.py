#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Polyhedral failure, incidence, and source boundaries shared by edit routes."""

from __future__ import annotations

from typing import assert_never

import numpy as np

import phydrax.discretization._cell_geometry as _dependency_phydrax_discretization__cell_geometry
import phydrax.discretization._exact_plc_geometry as _dependency_phydrax_discretization__exact_plc_geometry
import phydrax.discretization._exact_power_geometry as _dependency_phydrax_discretization__exact_power_geometry

from ..discretization import CellMesh, PolyhedralConnectivity
from ..discretization._exact_algebraic_geometry import ExactAlgebraicCellGeometrySource
from ..discretization._exact_rational_geometry import ExactRationalCellGeometrySource
from ..typing import checked
from ._failures import MeshingFailure, MeshingFailureCategory


class PolyhedralAdaptationError(MeshingFailure):
    """A rejected candidate retains scientific failure status and accepted state."""

    def __init__(
        self,
        message: str,
        /,
        *,
        category: MeshingFailureCategory = MeshingFailureCategory.COMPLIANCE_FAILED,
    ) -> None:
        super().__init__(category, message, stage="polyhedral_adaptation")


@checked
def polyhedral_face_loops(mesh: CellMesh) -> tuple[tuple[np.ndarray, ...], ...]:
    connectivity = mesh.connectivity
    match connectivity:
        case PolyhedralConnectivity():
            pass
        case _:
            raise ValueError(
                "polyhedral_adaptation.connectivity must use packed oriented incidence; got another connectivity kind"
            )
    co = np.asarray(connectivity.cell_face_offsets, dtype=np.int64)
    cf = np.asarray(connectivity.cell_face_values, dtype=np.int64)
    signs = np.asarray(connectivity.cell_face_sign_values, dtype=np.int32)
    fo = np.asarray(connectivity.face_vertex_offsets, dtype=np.int64)
    fv = np.asarray(connectivity.face_vertex_values, dtype=np.int32)
    return tuple(
        tuple(
            fv[fo[cf[position]] : fo[cf[position] + 1]][:: int(signs[position])]
            for position in range(co[row], co[row + 1])
        )
        for row in range(co.size - 1)
    )


@checked
def bound_power_source(
    geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec, /
) -> (
    _dependency_phydrax_discretization__exact_power_geometry.ExactPowerCellGeometrySource
    | _dependency_phydrax_discretization__exact_power_geometry.ExactPowerCellGeometryRestrictionSource
    | _dependency_phydrax_discretization__exact_power_geometry.ExactPowerCellGeometryLinearActionSource
    | None
):
    """Return the power construction associated with a polyhedral geometry."""
    match geometry.exact_source:
        case (
            None
            | _dependency_phydrax_discretization__exact_power_geometry.ExactPowerCellGeometrySource()
            | _dependency_phydrax_discretization__exact_power_geometry.ExactPowerCellGeometryRestrictionSource()
            | _dependency_phydrax_discretization__exact_power_geometry.ExactPowerCellGeometryLinearActionSource() as source
        ):
            return source
        case ExactAlgebraicCellGeometrySource():
            raise ValueError(
                "polyhedral_adaptation.geometry.exact_source must be a power-polyhedron source; got an algebraic nodal source"
            )
        case (
            _dependency_phydrax_discretization__exact_plc_geometry.ExactPlcCellGeometrySource()
            | _dependency_phydrax_discretization__exact_plc_geometry.ExactPlcCellGeometryConvexSource()
            | ExactRationalCellGeometrySource()
        ):
            raise ValueError(
                "polyhedral_adaptation.geometry.exact_source must be a power-polyhedron source; got a PLC source"
            )
        case invalid:
            assert_never(invalid)
