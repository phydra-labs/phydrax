#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx

from .._fingerprint import canonical_fingerprint
from .._identity import SemanticProvenance
from .._physical import SpatialCoordinateContract
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import CellGeometrySpec, CellMesh
from ..discretization._cell_geometry_validity import cell_geometry_id
from ..geometry._mesh_certificates import _same_tree_content
from ..geometry._planar_embedding import PlanarEmbedding
from ..geometry._surface_source_support import SurfaceSourceCharts
from ..geometry.surface import SurfaceModel
from ..interchange import AdapterReport
from ..typing import checked
from ._association import GeometryAssociation
from ._audit import (
    _evidence_id,
    _mesh_evidence_issues,
    CellMeshAuditReport,
)
from ._certification import MeshCertificationReport
from ._contracts import (
    MeshingDerivativeMode,
    MeshingExecutionMode,
    MeshingProviderInfo,
)
from ._layer_curving_records import LayerCurvingSourceEvidence
from ._organization import (
    MeshAttribute,
    MeshLabel,
    MeshPatch,
    MeshZone,
    RegionBoundaryEvidence,
    validate_mesh_labels,
    validate_mesh_zones,
)
from ._quality import CellQualityReport
from ._trace import MeshingTrace


class MeshingComplianceReport(StrictModule, NonTrainableState):
    specification_id: str = eqx.field(static=True)
    passed: bool = eqx.field(static=True)
    issues: tuple[str, ...] = eqx.field(static=True)
    requested: tuple[tuple[str, float], ...] = eqx.field(static=True)
    achieved: tuple[tuple[str, float], ...] = eqx.field(static=True)
    report_id: str = eqx.field(static=True)

    def __init__(
        self,
        specification_id: str,
        /,
        *,
        issues: tuple[str, ...] = (),
        requested: tuple[tuple[str, float], ...] = (),
        achieved: tuple[tuple[str, float], ...] = (),
    ) -> None:
        specification = str(specification_id).strip()
        if not specification:
            raise ValueError("Compliance specification_id must be non-empty.")
        issues_ = tuple(str(value) for value in issues)
        requested_ = tuple((str(name), float(value)) for name, value in requested)
        achieved_ = tuple((str(name), float(value)) for name, value in achieved)
        self.specification_id = specification
        self.passed = not issues_
        self.issues = issues_
        self.requested = requested_
        self.achieved = achieved_
        self.report_id = canonical_fingerprint(
            {
                "kind": "meshing-compliance-report",
                "specification": specification,
                "issues": issues_,
                "requested": requested_,
                "achieved": achieved_,
            }
        )


class MeshingRuntimeInfo(StrictModule, NonTrainableState):
    provider_id: str = eqx.field(static=True)
    actual_version: str = eqx.field(static=True)
    execution_mode: MeshingExecutionMode = eqx.field(static=True)
    deterministic: bool = eqx.field(static=True)
    runtime_id: str = eqx.field(static=True)

    def __init__(
        self,
        provider_id: str,
        actual_version: str,
        execution_mode: MeshingExecutionMode,
        /,
        *,
        deterministic: bool,
    ) -> None:
        provider = str(provider_id).strip()
        version = str(actual_version).strip()
        if not provider or not version:
            raise ValueError("Runtime provider and version identities must be non-empty.")
        if not isinstance(execution_mode, MeshingExecutionMode):
            raise TypeError("execution_mode must be MeshingExecutionMode.")
        self.provider_id = provider
        self.actual_version = version
        self.execution_mode = execution_mode
        self.deterministic = bool(deterministic)
        self.runtime_id = canonical_fingerprint(
            {
                "kind": "meshing-runtime-info",
                "provider": provider,
                "actual_version": version,
                "execution_mode": execution_mode.value,
                "deterministic": bool(deterministic),
            }
        )


def _require_result_reports(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    audit: CellMeshAuditReport,
    quality: CellQualityReport,
    compliance: MeshingComplianceReport,
    trace: MeshingTrace,
    provider: MeshingProviderInfo,
    runtime: MeshingRuntimeInfo,
    derivative_mode: MeshingDerivativeMode,
    /,
) -> None:
    if audit.mesh_id != mesh.mesh_id:
        raise ValueError("Audit must be bound to the result mesh.")
    if (
        audit.geometry_layout_id != geometry.geometry_layout_id
        or audit.geometry_id != cell_geometry_id(geometry)
    ):
        raise ValueError("Audit must be bound to the result geometry values and layout.")
    audit.require_passed()
    if quality.report_id != audit.quality.report_id:
        raise ValueError("Quality report must match the mesh audit.")
    if not compliance.passed:
        raise ValueError("Successful meshing results require passed compliance.")
    if not trace.successful:
        raise ValueError("Successful meshing results require a successful trace.")
    if runtime.provider_id != provider.provider_id:
        raise ValueError("Meshing runtime must match the provider.")
    if not isinstance(derivative_mode, MeshingDerivativeMode):
        raise TypeError("derivative_mode must be MeshingDerivativeMode.")


def _validated_result_organization(
    mesh: CellMesh,
    coordinate_contract: SpatialCoordinateContract,
    audit: CellMeshAuditReport,
    boundary: SurfaceModel | None,
    patches: tuple[MeshPatch, ...],
    zones: tuple[MeshZone, ...],
    labels: tuple[MeshLabel, ...],
    attributes: tuple[MeshAttribute, ...],
    associations: tuple[GeometryAssociation, ...],
    adapter_reports: tuple[AdapterReport, ...],
    /,
) -> tuple[tuple[MeshPatch, ...], tuple[MeshZone, ...], tuple[MeshLabel, ...]]:
    patches_ = tuple(patches)
    zones_ = validate_mesh_zones(tuple(zones))
    labels_ = validate_mesh_labels(tuple(labels))
    if not all(isinstance(value, MeshPatch) for value in patches_):
        raise TypeError("patches must contain MeshPatch values.")
    if not all(isinstance(value, MeshAttribute) for value in attributes):
        raise TypeError("attributes must contain MeshAttribute values.")
    if not all(isinstance(value, GeometryAssociation) for value in associations):
        raise TypeError("associations must contain GeometryAssociation values.")
    if not all(isinstance(value, AdapterReport) for value in adapter_reports):
        raise TypeError("adapter_reports must contain AdapterReport values.")
    if (
        boundary is not None
        and boundary.metadata.coordinate_contract.spatial_id
        != coordinate_contract.spatial_id
    ):
        raise ValueError("Boundary coordinate contract must match the result.")
    evidence_issues = _mesh_evidence_issues(
        mesh,
        boundary,
        patches_,
        zones_,
        labels_,
        attributes,
        associations,
    )
    if evidence_issues:
        raise ValueError(
            "Result evidence is not bound to the mesh: " + "; ".join(evidence_issues)
        )
    if audit.evidence_id != _evidence_id(
        patches_, zones_, labels_, attributes, associations
    ):
        raise ValueError(
            "Result organization and associations must match the audited evidence."
        )
    return patches_, zones_, labels_


def _validated_source_evidence(
    mesh: CellMesh,
    coordinate_contract: SpatialCoordinateContract,
    surface_source: SurfaceSourceCharts | None,
    region_boundary_evidence: tuple[RegionBoundaryEvidence, ...],
    planar_embedding: PlanarEmbedding | None,
    labels: tuple[MeshLabel, ...],
    patches: tuple[MeshPatch, ...],
    /,
) -> tuple[RegionBoundaryEvidence, ...]:
    if planar_embedding is not None and mesh.ambient_dimension != 2:
        raise ValueError("A planar embedding requires a two-dimensional coordinate mesh.")
    if surface_source is not None:
        surface_source.require_current()
        if (
            surface_source.coordinate_contract.spatial_id
            != coordinate_contract.spatial_id
        ):
            raise ValueError("Surface source coordinates must match the result frame.")
    evidence = tuple(region_boundary_evidence)
    if not all(isinstance(value, RegionBoundaryEvidence) for value in evidence):
        raise TypeError(
            "region_boundary_evidence must contain RegionBoundaryEvidence values."
        )
    identities = {
        (
            value.source_scope.scope_id,
            value.source_region_id,
            value.source_occurrence_path,
        )
        for value in evidence
    }
    if len(identities) != len(evidence):
        raise ValueError(
            "Each scoped source-region occurrence may have only one boundary evidence record."
        )
    evidence = tuple(sorted(evidence, key=lambda value: value.evidence_id))
    for value in evidence:
        value.require_current(mesh, labels, patches)
    return evidence


def _require_result_certification(
    certification: MeshCertificationReport | None,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    audit: CellMeshAuditReport,
    boundary: SurfaceModel | None,
    /,
) -> None:
    if certification is None:
        return
    certification.inputs.require_current(
        mesh,
        geometry,
        domain=certification.inputs.domain,
        source=certification.inputs.source,
    )
    certification.require_passed()
    if not _same_tree_content(audit, certification.audit):
        raise ValueError("Result audit must match its computed scientific acceptance.")
    if not _same_tree_content(boundary, certification.boundary):
        raise ValueError("Result boundary must match its computed scientific acceptance.")


class CellMeshingResult(StrictModule, NonTrainableState):
    mesh: CellMesh
    geometry: CellGeometrySpec
    coordinate_contract: SpatialCoordinateContract
    boundary: SurfaceModel | None
    patches: tuple[MeshPatch, ...]
    zones: tuple[MeshZone, ...]
    labels: tuple[MeshLabel, ...]
    attributes: tuple[MeshAttribute, ...]
    associations: tuple[GeometryAssociation, ...]
    surface_source: SurfaceSourceCharts | None
    region_boundary_evidence: tuple[RegionBoundaryEvidence, ...]
    planar_embedding: PlanarEmbedding | None = eqx.field(static=True)
    audit: CellMeshAuditReport
    certification: MeshCertificationReport | None
    quality: CellQualityReport
    compliance: MeshingComplianceReport
    trace: MeshingTrace
    adapter_reports: tuple[AdapterReport, ...]
    provider: MeshingProviderInfo
    runtime: MeshingRuntimeInfo
    derivative_mode: MeshingDerivativeMode = eqx.field(static=True)
    provenance: SemanticProvenance
    result_id: str = eqx.field(static=True)
    layer_curving_evidence: LayerCurvingSourceEvidence | None

    @checked
    def __init__(
        self,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        coordinate_contract: SpatialCoordinateContract,
        audit: CellMeshAuditReport,
        quality: CellQualityReport,
        compliance: MeshingComplianceReport,
        trace: MeshingTrace,
        provider: MeshingProviderInfo,
        runtime: MeshingRuntimeInfo,
        derivative_mode: MeshingDerivativeMode,
        provenance: SemanticProvenance,
        /,
        *,
        boundary: SurfaceModel | None = None,
        patches: tuple[MeshPatch, ...] = (),
        zones: tuple[MeshZone, ...] = (),
        labels: tuple[MeshLabel, ...] = (),
        attributes: tuple[MeshAttribute, ...] = (),
        associations: tuple[GeometryAssociation, ...] = (),
        adapter_reports: tuple[AdapterReport, ...] = (),
        surface_source: SurfaceSourceCharts | None = None,
        region_boundary_evidence: tuple[RegionBoundaryEvidence, ...] = (),
        planar_embedding: PlanarEmbedding | None = None,
        certification: MeshCertificationReport | None = None,
        layer_curving_evidence: LayerCurvingSourceEvidence | None = None,
    ) -> None:
        geometry.resolve(mesh)
        _require_result_reports(
            mesh,
            geometry,
            audit,
            quality,
            compliance,
            trace,
            provider,
            runtime,
            derivative_mode,
        )
        patches_, zones_, labels_ = _validated_result_organization(
            mesh,
            coordinate_contract,
            audit,
            boundary,
            patches,
            zones,
            labels,
            attributes,
            associations,
            adapter_reports,
        )
        boundary_evidence = _validated_source_evidence(
            mesh,
            coordinate_contract,
            surface_source,
            region_boundary_evidence,
            planar_embedding,
            labels_,
            patches_,
        )
        _require_result_certification(certification, mesh, geometry, audit, boundary)
        if layer_curving_evidence is not None:
            if layer_curving_evidence.frame_id != coordinate_contract.spatial_id:
                raise ValueError(
                    "Layer curving evidence belongs to a different physical frame."
                )
            layer_curving_evidence.require_current(mesh, geometry)
        self.mesh = mesh
        self.geometry = geometry
        self.coordinate_contract = coordinate_contract
        self.boundary = boundary
        self.patches = patches_
        self.zones = zones_
        self.labels = labels_
        self.attributes = tuple(attributes)
        self.associations = tuple(associations)
        self.surface_source = surface_source
        self.region_boundary_evidence = boundary_evidence
        self.planar_embedding = planar_embedding
        self.audit = audit
        self.certification = certification
        self.quality = quality
        self.compliance = compliance
        self.trace = trace
        self.adapter_reports = tuple(adapter_reports)
        self.provider = provider
        self.runtime = runtime
        self.derivative_mode = derivative_mode
        self.provenance = provenance
        self.layer_curving_evidence = layer_curving_evidence
        self.result_id = canonical_fingerprint(
            {
                "kind": "cell-meshing-result",
                "mesh": mesh.mesh_id,
                "geometry_layout": geometry.geometry_layout_id,
                "geometry": audit.geometry_id,
                "coordinate_contract": coordinate_contract.spatial_id,
                "boundary": None if boundary is None else boundary.model_id,
                "patches": [value.patch_id for value in patches_],
                "zones": [value.zone_id for value in zones_],
                "labels": [value.label_id for value in labels_],
                "attributes": [value.attribute_id for value in attributes],
                "associations": [value.association_id for value in associations],
                "surface_source": None
                if surface_source is None
                else surface_source.source_chart_id,
                "region_boundaries": tuple(
                    value.evidence_id for value in boundary_evidence
                ),
                "planar_embedding": None
                if planar_embedding is None
                else planar_embedding.embedding_id,
                "audit": audit.report_id,
                "certification": None
                if certification is None
                else certification.report_id,
                "quality": quality.report_id,
                "compliance": compliance.report_id,
                "adapter_reports": [value.report_id for value in adapter_reports],
                "derivative_mode": derivative_mode.value,
                "provenance": provenance.semantic_id,
                "layer_curving_evidence": None
                if layer_curving_evidence is None
                else layer_curving_evidence.evidence_id,
            }
        )


__all__ = ["CellMeshingResult", "MeshingComplianceReport", "MeshingRuntimeInfo"]
