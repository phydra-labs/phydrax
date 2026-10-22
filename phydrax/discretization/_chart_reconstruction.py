#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Store reconstructed chart geometry with its retained target certificates."""

from __future__ import annotations

import equinox as eqx
import numpy as np

from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..typing import checked, Float64, Int32, Int64
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
from ._coordinate_budget import CoordinateEnclosureBudget
from .fem._reference_types import FiniteElementSpec


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

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[SurfaceGeometryReconstruction],
        value: SurfaceGeometryReconstruction,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> SurfaceGeometryReconstruction:
        """Revalidate retained coordinates and numeric metadata without changing IDs.

        This is not renewed source-fidelity, validity or embedding evidence.
        Those scientific acceptance gates remain the restoring caller's job.
        Any exact coordinate-source preparation uses the caller's ledger.
        """
        geometry = CellGeometrySpec.from_stored_fields(value.geometry, budget=budget)
        elements, routes, _ = geometry.resolve(value.target_mesh, budget=budget)
        if (
            geometry.geometry_layout_id != value.geometry.geometry_layout_id
            or value.target_topology_id != value.target_mesh.topology_id
        ):
            raise ValueError(
                "surface_reconstruction.binding must retain its coordinate layout and target topology; got inconsistent restored identity"
            )
        identifiers = np.concatenate(
            [np.asarray(block.global_ids) for block in value.target_mesh.blocks]
        )
        count = identifiers.size
        if (
            not np.array_equal(np.asarray(value.cell_global_ids), identifiers)
            or np.asarray(value.cell_patches).shape != (count,)
            or np.any(np.asarray(value.cell_patches) < 0)
            or np.asarray(value.cell_charts).shape != (count, 3, 2)
            or not np.all(np.isfinite(np.asarray(value.cell_charts)))
            or len(value.cell_geometry_entity_ids) != count
            or len(value.cell_occurrence_paths) != count
        ):
            raise ValueError(
                "surface_reconstruction.cells must retain aligned target cells, charts and source metadata; got inconsistent restored content"
            )
        vertices = np.asarray(value.vertex_coordinates)
        if (
            vertices.shape != value.target_mesh.coordinates.shape
            or not np.all(np.isfinite(vertices))
            or not np.isfinite(value.continuity_residual)
            or value.continuity_residual < 0
            or value.evaluation_count < 0
        ):
            raise ValueError(
                "surface_reconstruction.coordinates must retain finite vertices and nonnegative work and continuity evidence; got invalid restored content"
            )
        for bounds in (
            value.fidelity_bounds,
            value.lebesgue_bounds,
            value.affine_reproduction_bounds,
            value.node_enclosure_bounds,
        ):
            if budget is not None:
                # Revisit each retained bound for finiteness and nonnegativity.
                budget.reserve(2 * bounds.size)
            numeric = np.asarray(bounds)
            if (
                numeric.shape != (count,)
                or not np.all(np.isfinite(numeric))
                or np.any(numeric < 0)
            ):
                raise ValueError(
                    "surface_reconstruction.bounds must retain finite nonnegative bounds for every target cell; got invalid restored evidence"
                )
        owner_cells, owner_locals = value.node_owner_cells, value.node_owner_locals
        if owner_cells is None or owner_locals is None:
            if owner_cells is not None or owner_locals is not None:
                raise ValueError(
                    "surface_reconstruction.node_owners must be present together; got incomplete restored ownership"
                )
            if not np.array_equal(vertices, np.asarray(value.target_mesh.coordinates)):
                raise ValueError(
                    "surface_reconstruction.coordinates must retain the source-composed carrier vertices; got changed restored representatives"
                )
        else:
            cells, locals_ = np.asarray(owner_cells), np.asarray(owner_locals)
            if (
                cells.shape != (geometry.coordinates.shape[0],)
                or locals_.shape != cells.shape
                or not np.all(np.isin(cells, identifiers))
                or np.any(locals_ < 0)
            ):
                raise ValueError(
                    "surface_reconstruction.node_owners must identify a target-cell node for every coordinate; got inconsistent restored ownership"
                )
            for block, element, route in zip(
                value.target_mesh.blocks, elements, routes, strict=True
            ):
                if not isinstance(element, FiniteElementSpec):
                    raise ValueError(
                        "surface_reconstruction.node_owners must bind a nodal coordinate element; got incompatible restored geometry"
                    )
                vertex_dofs = [entity[0] for entity in element.entity_dofs[0]]
                if not np.array_equal(
                    vertices[np.asarray(block.vertices)],
                    np.asarray(geometry.coordinates)[np.asarray(route)[:, vertex_dofs]],
                ):
                    raise ValueError(
                        "surface_reconstruction.coordinates must retain nodal coordinate vertex images; got changed restored representatives"
                    )
        return value

    @property
    def exact(self) -> bool:
        return False

    @property
    def source_inverse_coverage(self) -> None:
        return None
