# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Rational quarter-cylinder sources and their authored quadrilateral partition."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from phydrax.discretization import CellBlock, CellGeometrySpec, CellMesh
from phydrax.discretization._cell_geometry import (
    CellGeometryRestrictionSource,
    RestrictedCellGeometryElement,
    SplineCellGeometryElement,
)
from phydrax.discretization._cell_geometry_validity import cell_geometry_id
from phydrax.meshing import MeshingLimits
from phydrax.meshing._quad_generation import extract_surface_quads
from tests._support.coordinate_references import native_meshing_request


def cylinder_source() -> tuple[
    CellMesh, CellGeometrySpec, SplineCellGeometryElement, NDArray[np.float64]
]:
    controls = np.asarray(
        [[(x, 1.0, 0.0), (x, 1.0, 1.0), (x, 0.0, 1.0)] for x in (0.0, 1.0)],
        dtype=np.float64,
    ).reshape(-1, 3)
    u = np.asarray((0.0, 0.0, 1.0, 1.0), dtype=np.float64)
    v = np.asarray((0.0, 0.0, 0.0, 1.0, 1.0, 1.0), dtype=np.float64)
    weights = np.tile(np.asarray((1.0, math.sqrt(0.5), 1.0), dtype=np.float64), (2, 1))
    element = SplineCellGeometryElement(
        u, v, weights, 1, 2, (1, 2), "spline", "original-r1"
    )
    corners = np.asarray(
        ((0.0, 1.0, 0.0), (1.0, 1.0, 0.0), (1.0, 0.0, 1.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    mesh = CellMesh(
        corners,
        (
            CellBlock(
                "root", "quadrilateral", np.asarray(((0, 1, 2, 3),), dtype=np.int64)
            ),
        ),
    )
    geometry = CellGeometrySpec(
        {"root": element}, {"root": np.arange(6, dtype=np.int64)[None]}, controls
    )
    return mesh, geometry, element, controls


def six_quad_source(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    root: SplineCellGeometryElement,
    controls: NDArray[np.float64],
) -> tuple[CellMesh, CellGeometrySpec]:
    triangles = CellMesh(
        mesh.coordinates,
        (
            CellBlock(
                "a",
                "triangle",
                np.asarray(((0, 1, 2),), dtype=np.int64),
                global_ids=np.asarray((0,), dtype=np.int64),
            ),
            CellBlock(
                "b",
                "triangle",
                np.asarray(((0, 2, 3),), dtype=np.int64),
                global_ids=np.asarray((1,), dtype=np.int64),
            ),
        ),
    )
    elements = {
        "a": RestrictedCellGeometryElement(
            root,
            "triangle",
            np.asarray(((1.0, 1.0), (0.0, 1.0)), dtype=np.float64),
            np.zeros(2, dtype=np.float64),
        ),
        "b": RestrictedCellGeometryElement(
            root,
            "triangle",
            np.asarray(((1.0, 0.0), (1.0, 1.0)), dtype=np.float64),
            np.zeros(2, dtype=np.float64),
        ),
    }
    ancestry = CellGeometryRestrictionSource(
        cell_geometry_id(geometry),
        mesh.topology_id,
        {name: np.asarray((0,), dtype=np.int64) for name in elements},
        {name: np.asarray(((0, 1, 2, 3),), dtype=np.int64) for name in elements},
    )
    charts = CellGeometrySpec(
        elements,
        {name: np.arange(6, dtype=np.int64)[None] for name in elements},
        controls,
        restriction_source=ancestry,
    )
    limits = MeshingLimits()
    with native_meshing_request(limits) as (scope, extraction_budget, _workspace):
        result = extract_surface_quads(
            triangles,
            limits,
            source_geometry=charts,
            scope=scope,
            budget=extraction_budget,
        )
    if result.geometry is None:
        raise ValueError(
            "rational_mapped_cells.geometry must retain the extracted coordinate source; got absent geometry"
        )
    return result.mesh, result.geometry
