from __future__ import annotations

import numpy as np

from ..discretization._cell_geometry import CellGeometrySpec
from ..discretization._cell_mesh import CellMesh
from ..geometry._domain_certification import DomainCoverageCertificate
from ..geometry._volume_material_map import MaterialSourceMapCapability
from ._association import (
    GeometryAssociation,
    GeometryAssociationKind,
    GeometrySourceEntityRole,
)


def material_source_map_from_cell_association(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    coverage: DomainCoverageCertificate,
    association: GeometryAssociation,
    /,
    *,
    chart_parent: tuple[CellMesh, CellGeometrySpec] | None = None,
) -> MaterialSourceMapCapability:
    """Bind complete cell-region lineage to an accepted physical source domain.

    This adapter is source-family agnostic: CAD, PLC, mapped and multi-region
    producers provide their actual per-cell REGION associations, while the
    domain certificate independently establishes the covered material regions.
    No row position or reconstructed source ID is promoted to source identity.
    """
    if not isinstance(mesh, CellMesh) or not isinstance(geometry, CellGeometrySpec):
        raise TypeError(
            "Projected material maps require canonical mesh and geometry owners."
        )
    if not isinstance(coverage, DomainCoverageCertificate):
        raise TypeError("coverage must be DomainCoverageCertificate.")
    if not isinstance(association, GeometryAssociation):
        raise TypeError("association must be GeometryAssociation.")
    if mesh.topological_dimension != 3:
        raise ValueError(
            "Projected material source maps require volume-cell associations."
        )
    if coverage.status != "certified":
        raise ValueError(
            "Projected material source maps require certified domain coverage."
        )
    coverage.require_current(
        mesh,
        geometry,
        coverage.domain,
        coverage.cell_regions,
        embedding=coverage.embedding,
        source=coverage.source,
        fidelity=coverage.fidelity,
        source_coverage_witness=coverage.source_coverage_witness,
        chart_parent=chart_parent,
    )
    if (
        association.source_id != coverage.source_id
        or association.source_revision != coverage.source_revision
        or association.target_entity_set_id != mesh.entity_set(3).entity_set_id
    ):
        raise ValueError(
            "Cell source associations do not bind the certified material domain."
        )
    cell_ids = tuple(
        int(identifier)
        for block in mesh.blocks
        for identifier in np.asarray(block.global_ids, dtype=np.int64)
    )
    association_ids = np.asarray(association.target_global_ids, dtype=np.int64)
    if (
        association_ids.ndim != 1
        or np.unique(association_ids).size != association_ids.size
        or set(int(identifier) for identifier in association_ids) != set(cell_ids)
    ):
        raise ValueError("Cell source associations must cover the exact target cell IDs.")
    association_rows = {
        int(identifier): row for row, identifier in enumerate(association_ids)
    }
    ordered_rows = tuple(association_rows[identifier] for identifier in cell_ids)
    resolved = np.asarray(association.resolved, dtype=np.bool_)
    ambiguous = np.asarray(association.ambiguous, dtype=np.bool_)
    if (
        resolved.shape != association_ids.shape
        or ambiguous.shape != association_ids.shape
        or not np.all(resolved[list(ordered_rows)])
        or np.any(ambiguous[list(ordered_rows)])
        or any(
            association.source_roles[row] is not GeometrySourceEntityRole.REGION
            for row in ordered_rows
        )
    ):
        raise ValueError(
            "Projected material maps require one uniquely resolved source REGION per cell."
        )

    if association.association_kind is GeometryAssociationKind.BREP:
        source_dimensions = np.asarray(association.source_dimensions, dtype=np.int64)
        if source_dimensions.shape != association_ids.shape or np.any(
            source_dimensions[np.asarray(ordered_rows, dtype=np.int64)] != 3
        ):
            raise ValueError("B-Rep material cells must bind resolved source regions.")
    region_indices = np.asarray(coverage.cell_regions, dtype=np.int64)
    if region_indices.shape != (len(cell_ids),) or np.any(
        (region_indices < 0) | (region_indices >= len(coverage.region_ids))
    ):
        raise ValueError("Certified material regions do not align with the volume cells.")
    cell_regions = tuple(coverage.region_ids[int(index)] for index in region_indices)
    cell_entities = tuple(association.source_entity_ids[row] for row in ordered_rows)
    occurrence_paths = tuple(
        association.source_occurrence_paths[row] for row in ordered_rows
    )
    source_map = MaterialSourceMapCapability(
        coverage.source_id,
        coverage.source_revision,
        coverage.domain_id,
        cell_entities,
        cell_regions,
        occurrence_paths=occurrence_paths,
    )
    if geometry.source_map is not None and (
        geometry.source_map.source_id != source_map.source_id
        or geometry.source_map.source_revision != source_map.source_revision
    ):
        raise ValueError(
            "Coordinate source-map identity differs from the material source."
        )
    return source_map


__all__ = ["material_source_map_from_cell_association"]
