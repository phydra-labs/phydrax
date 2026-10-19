#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Prepare source-faithful coordinate maps shared by native PLC routes."""

from __future__ import annotations

import phydrax.discretization._cell_geometry as cell_geometry_owner

from ...discretization._coordinate_budget import CoordinateEnclosureBudget
from ...discretization._exact_plc_geometry import ExactPlcCellGeometrySource
from ...typing import checked
from .._failures import MeshingFailure, MeshingFailureCategory
from .._trace import MeshingStageKind
from .._volume_generation import VolumeConstruction
from . import _native_sources as sources_owner


@checked
def exact_plc_geometry(
    construction: VolumeConstruction,
    source: sources_owner.NativePlcSource,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> cell_geometry_owner.CellGeometrySpec | None:
    """Return source coordinates for a mesh with ancestry-backed PLC carriers.

    Zero deviation changes no coordinate but retains the closed source rows
    and parameters. Bounded carriers retain their declared source S.
    """
    fidelity = construction.source_fidelity
    if fidelity is None:
        return None
    declared = fidelity.declared_source
    if declared is None:
        raise MeshingFailure(
            MeshingFailureCategory.COMPLIANCE_FAILED,
            "VolumeConstruction.source_fidelity.declared_source must retain source rows for the bounded carrier; got None.",
            stage=MeshingStageKind.GEOMETRY_AUDIT.value,
        )
    exact = ExactPlcCellGeometrySource(
        declared.points,
        declared.faces,
        declared.segments,
        fidelity.witness_strata,
        fidelity.witness_entities,
        fidelity.witness_parameters,
        domain_source_id=source.source_id,
        domain_source_revision=source.source_revision,
        source_triangle_ids=declared.face_ids[declared.face_group_rows],
        source_triangle_bounds=declared.face_tolerances,
        source_segment_ids=declared.segment_ids[declared.segment_group_rows],
        source_segment_bounds=declared.segment_tolerances,
        budget=budget,
    )
    exact.require_domain(
        declared.points, declared.faces, source.source_id, source.source_revision
    )
    return cell_geometry_owner.CellGeometrySpec.plc(
        construction.mesh, exact, budget=budget
    )
