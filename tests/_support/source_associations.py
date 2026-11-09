from __future__ import annotations

import numpy as np

from phydrax.meshing._association import (
    GeometryAssociation,
    SourceBoundaryTargetPieceRow,
)


def replace_association_support(
    table: GeometryAssociation,
    /,
    *,
    pieces: tuple[SourceBoundaryTargetPieceRow | None, ...] | None = None,
    exact: bool | None = None,
    sides: tuple[tuple[str | None, str | None], ...] | None = None,
    witness_id: str | None = None,
) -> GeometryAssociation:
    return GeometryAssociation(
        table.association_kind,
        table.source_id,
        table.source_revision,
        table.target_entity_set_id,
        table.target_global_ids,
        table.source_entity_ids,
        table.residuals,
        resolved=table.resolved,
        ambiguous=table.ambiguous,
        exact=table.exact if exact is None else exact,
        source_dimensions=(
            table.source_dimensions
            if np.any(np.asarray(table.source_dimensions) >= 0)
            else None
        ),
        source_indices=(
            table.source_indices
            if np.any(np.asarray(table.source_indices) >= 0)
            else None
        ),
        source_roles=table.source_roles,
        source_occurrence_paths=table.source_occurrence_paths,
        source_region_sides=table.source_region_sides if sides is None else sides,
        source_parameter_intervals=table.source_parameter_intervals,
        source_parameter_points=table.source_parameter_points,
        target_parameter_intervals=table.target_parameter_intervals,
        target_reference_pieces=table.target_reference_pieces
        if pieces is None
        else pieces,
        source_boundary_witness_id=table.source_boundary_witness_id
        if witness_id is None
        else witness_id,
        parameters=table.parameters,
        orientations=table.orientations,
        parent_dimensions=table.parent_dimensions,
        parent_ids=table.parent_ids,
        parent_association_id=table.parent_association_id,
        provenance=table.provenance,
    )
