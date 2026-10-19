"""Retained quarter-cylinder sources and triangle chart restrictions for quad tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from phydrax.discretization import (
    CellBlock,
    CellGeometryRestrictionSource,
    CellGeometrySpec,
    CellMesh,
    RestrictedCellGeometryElement,
    SplineCellGeometryElement,
)
from phydrax.discretization._cell_geometry_validity import cell_geometry_id
from phydrax.geometry import BSplineSurfacePatch
from tests._support.mapped_cells import rational_cylinder


@dataclass(frozen=True)
class RationalQuadSource:
    root: SplineCellGeometryElement
    root_geometry: CellGeometrySpec
    mesh: CellMesh
    geometry: CellGeometrySpec
    patch: BSplineSurfacePatch
    controls: np.ndarray


def rational_quad_source() -> RationalQuadSource:
    """Retain the source controls, weights, and declared chart ancestry.

    The circular-arc controls come from mapped_cells.rational_cylinder. Block
    names and ordered vertex IDs preserve the quad-generation source fixture.
    """
    carrier, _, root, controls = rational_cylinder()
    root_mesh = CellMesh(
        carrier.coordinates,
        (
            CellBlock(
                "original", "quadrilateral", np.asarray(((0, 1, 2, 3),), dtype=np.int64)
            ),
        ),
    )
    root_geometry = CellGeometrySpec(
        {"original": root},
        {"original": np.arange(6, dtype=np.int64)[None, :]},
        controls,
    )
    source = CellMesh(
        carrier.coordinates,
        (
            CellBlock(
                "chart/0",
                "triangle",
                np.asarray(((0, 1, 2),), dtype=np.int64),
                global_ids=np.asarray((0,), dtype=np.int64),
            ),
            CellBlock(
                "chart/1",
                "triangle",
                np.asarray(((0, 2, 3),), dtype=np.int64),
                global_ids=np.asarray((1,), dtype=np.int64),
            ),
        ),
    )
    elements = {
        "chart/0": RestrictedCellGeometryElement(
            root,
            "triangle",
            np.asarray(((1.0, 1.0), (0.0, 1.0)), dtype=np.float64),
            np.zeros(2, dtype=np.float64),
        ),
        "chart/1": RestrictedCellGeometryElement(
            root,
            "triangle",
            np.asarray(((1.0, 0.0), (1.0, 1.0)), dtype=np.float64),
            np.zeros(2, dtype=np.float64),
        ),
    }
    ancestry = CellGeometryRestrictionSource(
        cell_geometry_id(root_geometry),
        root_mesh.topology_id,
        {name: np.asarray((0,), dtype=np.int64) for name in elements},
        {name: np.asarray(((0, 1, 2, 3),), dtype=np.int64) for name in elements},
    )
    geometry = CellGeometrySpec(
        elements,
        {name: np.arange(6, dtype=np.int64)[None, :] for name in elements},
        controls,
        restriction_source=ancestry,
    )
    # The reference patch uses the analytic quarter-circle inputs, not metadata
    # returned by the coordinate element constructor under test.
    weights = np.tile(np.asarray((1.0, np.sqrt(0.5), 1.0), dtype=np.float64), (2, 1))
    u = np.asarray((0.0, 0.0, 1.0, 1.0), dtype=np.float64)
    v = np.asarray((0.0, 0.0, 0.0, 1.0, 1.0, 1.0), dtype=np.float64)
    patch = BSplineSurfacePatch(controls.reshape((2, 3, 3)), weights, u, v, 1, 2)
    return RationalQuadSource(root, root_geometry, source, geometry, patch, controls)


def tangent_triangle(height: float) -> CellMesh:
    """Build the triangle family spanning degenerate and thin validity cases."""
    return CellMesh.from_triangles(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (0.5, height)), dtype=np.float64),
        np.asarray(((0, 1, 2),), dtype=np.int64),
    )
