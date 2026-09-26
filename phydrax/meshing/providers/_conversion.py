#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Shared checks for converting external provider meshes into CellMesh."""

from __future__ import annotations

import numpy as np

from ...discretization import CellMesh
from ...interchange import AdapterLoss, AdapterReport, AdapterStatus
from .._contracts import MeshingFailure, MeshingFailureCategory, MeshingLimits
from .._result import CellMeshingResult


def _fresh_ids(source_ids: np.ndarray, count: int) -> np.ndarray:
    start = int(np.max(source_ids, initial=-1)) + 1
    if start + count > np.iinfo(np.int64).max:
        raise MeshingFailure(
            MeshingFailureCategory.CONVERSION_FAILED,
            "Generated mesh identities overflow int64.",
        )
    return np.arange(start, start + count, dtype=np.int64)


def _check_arrays(points: np.ndarray, cells: np.ndarray, limits: MeshingLimits) -> None:
    if (
        points.ndim != 2
        or points.shape[0] == 0
        or not np.all(np.isfinite(points))
        or cells.ndim != 2
        or cells.shape[0] == 0
        or not np.issubdtype(cells.dtype, np.integer)
        or np.any(cells < 0)
        or np.any(cells >= len(points))
    ):
        raise MeshingFailure(
            MeshingFailureCategory.CONVERSION_FAILED,
            "Backend returned invalid vertices or connectivity.",
        )
    if (
        len(points) > limits.maximum_vertices
        or len(cells) > limits.maximum_cells
        or cells.size > limits.maximum_connectivity_entries
        or points.nbytes + cells.nbytes > limits.maximum_data_bytes
    ):
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "Backend arrays exceed the requested meshing limits.",
        )


def _check_result_limits(result: CellMeshingResult, limits: MeshingLimits) -> None:
    counts = result.audit.entity_counts
    if (
        counts[1] > limits.maximum_edges
        or (len(counts) > 2 and counts[2] > limits.maximum_faces)
        or result.audit.connectivity_entries > limits.maximum_connectivity_entries
    ):
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "Canonical mesh incidence exceeds the requested meshing limits.",
        )


def _identity_report(source: CellMesh, mesh: CellMesh, provider: str) -> AdapterReport:
    return AdapterReport(
        AdapterStatus.DECLARED_LOSS,
        provider,
        "phydrax-cell-mesh",
        source_id=source.mesh_id,
        target_id=mesh.mesh_id,
        coordinate_mapping=("identity",),
        assumptions=(
            "No entity correspondence or field-transfer map is supplied by this backend.",
            "Only unlabeled simplex geometry is converted; provider-generated feature flags are not imported.",
        ),
        losses=(
            AdapterLoss(
                "entity_global_ids",
                "import",
                "synthesized",
                "Generated output IDs are not source row/region identities; lineage is unknown.",
                changes_interpretation=False,
            ),
        ),
    )
