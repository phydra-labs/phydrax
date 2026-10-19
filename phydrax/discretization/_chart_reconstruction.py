#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Store reconstructed chart geometry with its retained target certificates."""

from __future__ import annotations

import equinox as eqx

from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..typing import Float64, Int32, Int64
from ._cell_geometry import CellGeometrySpec
from ._cell_geometry_transition_types import (
    GeometryAmbientDim,
    GeometryCornerDim,
    GeometryNodeDim,
    GeometryReferenceDim,
    GeometryTargetCellDim,
    GeometryVertexDim,
)
from ._cell_mesh import CellMesh


class SurfaceGeometryReconstruction(StrictModule, NonTrainableState):
    """Bounded target map, not an accepted old-to-new transition.

    Polynomial reconstruction bounds encoded-basis reproduction, point
    enclosures and shared-node placement. Retained spline reconstruction uses
    source source-expression composition and has no interpolation defects.
    ``target_mesh`` retains the coefficient-element block layout.
    Old-map inverse/declared-chart coverage, trim coverage, validity, embedding
    and transfer remain separate gates; zero source fidelity is not an accepted
    zero-displacement material correspondence.
    """

    __strict_contract__ = True
    source_geometry_id: str = eqx.field(static=True)
    source_topology_id: str = eqx.field(static=True)
    target_topology_id: str = eqx.field(static=True)
    domain_id: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    cell_geometry_entity_ids: tuple[str, ...] = eqx.field(static=True)
    cell_occurrence_paths: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    geometry: CellGeometrySpec
    target_mesh: CellMesh
    vertex_coordinates: Float64[GeometryVertexDim, GeometryAmbientDim]
    cell_global_ids: Int64[GeometryTargetCellDim]
    cell_patches: Int32[GeometryTargetCellDim]
    cell_charts: Float64[GeometryTargetCellDim, GeometryCornerDim, GeometryReferenceDim]
    fidelity_bounds: Float64[GeometryTargetCellDim]
    lebesgue_bounds: Float64[GeometryTargetCellDim]
    affine_reproduction_bounds: Float64[GeometryTargetCellDim]
    node_enclosure_bounds: Float64[GeometryTargetCellDim]
    node_owner_cells: Int64[GeometryNodeDim] | None
    node_owner_locals: Int32[GeometryNodeDim] | None
    continuity_residual: float = eqx.field(static=True)
    evaluation_count: int = eqx.field(static=True)
    reconstruction_id: str = eqx.field(static=True)

    @property
    def exact(self) -> bool:
        return False

    @property
    def source_inverse_coverage(self) -> None:
        return None
