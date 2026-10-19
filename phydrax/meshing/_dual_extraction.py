# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Immutable field evidence and subdivision ancestry for mesh-family extraction."""

from __future__ import annotations

from typing import final

import equinox as eqx
import numpy as np

from .._strict import StrictModule
from ..discretization import CellGeometrySpec, CellMesh
from ..discretization._cell_geometry_validity import CellValidityCertificate
from ..linalg._small_batched import SmallLinearSolveResult
from ..optim import MinimizationResult


@final
class SurfaceCrossField(StrictModule):
    """C4 angles in feature-separated face patches with native solve evidence."""

    frames: np.ndarray
    angles: np.ndarray
    transport_edges: np.ndarray
    patch_ids: np.ndarray
    feature_edges: np.ndarray
    transport_angles: np.ndarray
    feature_rows: np.ndarray
    feature_angles: np.ndarray
    feature_residuals: np.ndarray
    optimization: MinimizationResult
    frame_solve: SmallLinearSolveResult
    source_topology_id: str = eqx.field(static=True)
    source_geometry_id: str = eqx.field(static=True)


@final
class DualExtraction(StrictModule):
    """Deterministic candidate plus combinatorial subdivision ancestry.

    ``vertex_supports`` is a padded row of source vertex indices for each new
    vertex (``-1`` padding). ``entity_parent_dimensions`` and
    ``entity_parent_rows`` identify the smallest source entity containing each
    target entity. These maps are suitable for source association and material
    propagation; they do not certify global embedding of the input complex.
    """

    mesh: CellMesh
    source: CellMesh
    vertex_supports: np.ndarray
    parent_cells: np.ndarray
    entity_parent_dimensions: tuple[np.ndarray, ...]
    entity_parent_rows: tuple[np.ndarray, ...]
    validity: CellValidityCertificate
    valences: np.ndarray
    boundary_vertices: np.ndarray
    singular_vertices: np.ndarray
    cross_field: SurfaceCrossField | None
    work_units: int = eqx.field(static=True)
    geometry: CellGeometrySpec | None = None
    source_geometry: CellGeometrySpec | None = None


__all__ = ["SurfaceCrossField", "DualExtraction"]
