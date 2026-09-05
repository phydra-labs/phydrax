#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import numpy as np

from .._identity import SemanticProvenance
from .._physical import SpatialCoordinateContract
from ..discretization import (
    CellBlock,
    CellGeometrySpec,
    CellMesh,
    PolyhedralBlock,
    PolyhedralConnectivity,
)
from ..discretization._cell_complex import PolygonalConnectivity, TetrahedralConnectivity
from ..discretization._hexahedral import HexahedralConnectivity
from ._association import GeometryAssociation
from ._audit import audit_cell_mesh, CellMeshAuditPolicy
from ._contracts import (
    MeshingCapability,
    MeshingDerivativeMode,
    MeshingExecutionMode,
    MeshingFailure,
    MeshingFailureCategory,
    MeshingOperation,
    MeshingProviderInfo,
    MeshingSourceKind,
)
from ._organization import MeshAttribute, MeshLabel, MeshPatch, MeshZone
from ._quality import evaluate_cell_quality
from ._result import CellMeshingResult, MeshingComplianceReport, MeshingRuntimeInfo
from ._trace import MeshingStageKind, MeshingStageReport, MeshingStageStatus, MeshingTrace


_NATIVE_PROVIDER = MeshingProviderInfo(
    "phydrax-native",
    "current",
    "Proprietary",
    operations=(MeshingOperation.OPTIMIZE_MESH,),
    source_kinds=(MeshingSourceKind.CELL_MESH,),
    capabilities=(MeshingCapability.DETERMINISTIC,),
    cell_kinds=(
        "interval",
        "triangle",
        "quadrilateral",
        "polygon",
        "tetrahedron",
        "hexahedron",
        "prism",
        "pyramid",
        "polyhedron",
    ),
    dimensions=(1, 2, 3),
    execution_modes=(MeshingExecutionMode.IN_PROCESS,),
)


def _entity_vertex_keys(mesh: CellMesh, dimension: int, /) -> tuple[tuple[int, ...], ...]:
    """Identify lower-dimensional entities independently of traversal ordering."""
    connectivity = mesh.connectivity
    if dimension == 1:
        if not isinstance(
            connectivity,
            (
                PolygonalConnectivity,
                TetrahedralConnectivity,
                HexahedralConnectivity,
                PolyhedralConnectivity,
            ),
        ):
            raise TypeError("Edge entities require polygonal or volume connectivity.")
        rows = np.asarray(connectivity.edges)
    elif dimension == 2:
        if isinstance(connectivity, PolyhedralConnectivity):
            offsets = np.asarray(connectivity.face_vertex_offsets)
            values = np.asarray(connectivity.face_vertex_values)
            rows = tuple(
                values[start:stop]
                for start, stop in zip(offsets[:-1], offsets[1:], strict=True)
            )
        elif isinstance(connectivity, (TetrahedralConnectivity, HexahedralConnectivity)):
            rows = np.asarray(connectivity.faces)
        else:
            raise TypeError("Face entities require volume connectivity.")
    else:
        raise ValueError("Entity vertex keys require edge or face dimension.")
    vertex_ids = np.asarray(mesh.vertex_global_ids)
    return tuple(tuple(sorted(int(value) for value in vertex_ids[row])) for row in rows)


def canonicalize_cell_mesh(mesh: CellMesh, /) -> CellMesh:
    """Return deterministic block/cell ordering without changing mesh geometry."""

    if not isinstance(mesh, CellMesh):
        raise TypeError("mesh must be CellMesh.")
    if any(isinstance(block, PolyhedralBlock) for block in mesh.blocks):
        for block in mesh.blocks:
            identifiers = np.asarray(block.global_ids, dtype=np.int64)
            if np.any(identifiers[:-1] > identifiers[1:]):
                raise ValueError(
                    "Face-defined polyhedral blocks must be globally ID ordered at construction."
                )
        return mesh
    blocks = []
    for block in mesh.blocks:
        identifiers = np.asarray(block.global_ids, dtype=np.int64)
        order = np.argsort(identifiers, kind="stable")
        blocks.append(
            CellBlock(
                block.name,
                block.cell_kind,
                np.asarray(block.vertices)[order],
                vertex_valid=np.asarray(block.vertex_valid)[order],
                global_ids=identifiers[order],
            )
        )
    ordered = tuple(
        sorted(
            blocks,
            key=(
                (lambda block: (block.arity, block.name))
                if mesh.topological_dimension == 2
                else (lambda block: block.name)
            ),
        )
    )
    if tuple(block.block_id for block in ordered) == tuple(
        block.block_id for block in mesh.blocks
    ):
        return mesh
    rebuilt = CellMesh(
        mesh.coordinates,
        ordered,
        vertex_global_ids=mesh.vertex_global_ids,
        numeric_version=mesh.numeric_version,
    )
    entity_ids = {}
    for dimension in range(1, mesh.topological_dimension):
        identifiers = dict(
            zip(
                _entity_vertex_keys(mesh, dimension),
                np.asarray(mesh.entity_set(dimension).entity_ids),
                strict=True,
            )
        )
        entity_ids[dimension] = np.asarray(
            [identifiers[key] for key in _entity_vertex_keys(rebuilt, dimension)],
            dtype=np.int64,
        )
    if all(
        np.array_equal(values, np.asarray(rebuilt.entity_set(dimension).entity_ids))
        for dimension, values in entity_ids.items()
    ):
        return rebuilt
    return CellMesh(
        mesh.coordinates,
        ordered,
        vertex_global_ids=mesh.vertex_global_ids,
        entity_global_ids=entity_ids,
        numeric_version=mesh.numeric_version,
    )


def certify_cell_mesh(
    mesh: CellMesh,
    coordinate_contract: SpatialCoordinateContract,
    /,
    *,
    geometry: CellGeometrySpec | None = None,
    audit_policy: CellMeshAuditPolicy | None = None,
    patches: tuple[MeshPatch, ...] = (),
    zones: tuple[MeshZone, ...] = (),
    labels: tuple[MeshLabel, ...] = (),
    attributes: tuple[MeshAttribute, ...] = (),
    associations: tuple[GeometryAssociation, ...] = (),
) -> CellMeshingResult:
    """Canonicalize and certify one existing CellMesh through native substrates."""

    if not isinstance(coordinate_contract, SpatialCoordinateContract):
        raise TypeError("coordinate_contract must be SpatialCoordinateContract.")
    canonical = canonicalize_cell_mesh(mesh)
    if geometry is not None and canonical is not mesh:
        raise ValueError(
            "Canonicalization would reorder supplied geometry DOF rows; canonicalize "
            "the mesh before constructing CellGeometrySpec."
        )
    geometry_ = CellGeometrySpec.affine(canonical) if geometry is None else geometry
    if not isinstance(geometry_, CellGeometrySpec):
        raise TypeError("geometry must be CellGeometrySpec or None.")
    quality_evaluation = evaluate_cell_quality(canonical)
    audit = audit_cell_mesh(
        canonical,
        geometry_,
        quality_evaluation,
        policy=audit_policy,
        patches=patches,
        associations=associations,
        attributes=attributes,
        zones=zones,
        labels=labels,
    )
    if audit.quality_scope != "vertex_geometry":
        raise MeshingFailure(
            MeshingFailureCategory.AUDIT_FAILED,
            "Native certification does not certify high-order geometry from corner-only quality.",
            stage=MeshingStageKind.GEOMETRY_AUDIT.value,
        )
    if not audit.passed:
        raise MeshingFailure(
            MeshingFailureCategory.AUDIT_FAILED,
            "; ".join(audit.issues),
            stage=MeshingStageKind.GEOMETRY_AUDIT.value,
            entity_ids=audit.quality.worst_cell_global_ids,
        )
    compliance = MeshingComplianceReport(f"existing-cell-mesh:{canonical.mesh_id}")
    stages = (
        MeshingStageReport(
            MeshingStageKind.CANONICALIZATION,
            MeshingStageStatus.PASSED,
            input_ids=(mesh.mesh_id,),
            output_ids=(canonical.mesh_id,),
        ),
        MeshingStageReport(
            MeshingStageKind.QUALITY_EVALUATION,
            MeshingStageStatus.PASSED,
            input_ids=(canonical.mesh_id,),
            output_ids=(audit.quality.report_id,),
        ),
        MeshingStageReport(
            MeshingStageKind.TOPOLOGY_AUDIT,
            MeshingStageStatus.PASSED,
            input_ids=(canonical.topology_id,),
            output_ids=(audit.report_id,),
        ),
        MeshingStageReport(
            MeshingStageKind.SPECIFICATION_COMPLIANCE,
            MeshingStageStatus.PASSED,
            input_ids=(canonical.mesh_id,),
            output_ids=(compliance.report_id,),
        ),
    )
    trace = MeshingTrace(stages)
    provenance = SemanticProvenance(
        {
            "kind": "native-cell-mesh-certification",
            "mesh_id": canonical.mesh_id,
            "geometry_layout_id": geometry_.geometry_layout_id,
            "coordinate_contract": coordinate_contract.spatial_id,
            "audit": audit.report_id,
        }
    )
    runtime = MeshingRuntimeInfo(
        _NATIVE_PROVIDER.provider_id,
        _NATIVE_PROVIDER.version,
        MeshingExecutionMode.IN_PROCESS,
        deterministic=True,
        enforced_limits=("canonical_connectivity",),
    )
    return CellMeshingResult(
        canonical,
        geometry_,
        coordinate_contract,
        audit,
        audit.quality,
        compliance,
        trace,
        _NATIVE_PROVIDER,
        runtime,
        MeshingDerivativeMode.FIXED_TOPOLOGY_EXACT,
        provenance,
        patches=patches,
        zones=zones,
        labels=labels,
        attributes=attributes,
        associations=associations,
    )


__all__ = ["canonicalize_cell_mesh", "certify_cell_mesh"]
