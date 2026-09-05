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
from ..geometry.surface import SurfaceModel
from ..interchange import AdapterReport
from ._association import GeometryAssociation
from ._audit import (
    _evidence_id,
    _geometry_id,
    _mesh_evidence_issues,
    CellMeshAuditReport,
)
from ._contracts import (
    MeshingDerivativeMode,
    MeshingExecutionMode,
    MeshingProviderInfo,
)
from ._organization import (
    MeshAttribute,
    MeshLabel,
    MeshPatch,
    MeshZone,
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
    ):
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
    enforced_limits: tuple[str, ...] = eqx.field(static=True)
    unenforced_limits: tuple[str, ...] = eqx.field(static=True)
    runtime_id: str = eqx.field(static=True)

    def __init__(
        self,
        provider_id: str,
        actual_version: str,
        execution_mode: MeshingExecutionMode,
        /,
        *,
        deterministic: bool,
        enforced_limits: tuple[str, ...] = (),
        unenforced_limits: tuple[str, ...] = (),
    ):
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
        self.enforced_limits = tuple(str(value) for value in enforced_limits)
        self.unenforced_limits = tuple(str(value) for value in unenforced_limits)
        self.runtime_id = canonical_fingerprint(
            {
                "kind": "meshing-runtime-info",
                "provider": provider,
                "actual_version": version,
                "execution_mode": execution_mode.value,
                "deterministic": bool(deterministic),
                "enforced_limits": self.enforced_limits,
                "unenforced_limits": self.unenforced_limits,
            }
        )


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
    audit: CellMeshAuditReport
    quality: CellQualityReport
    compliance: MeshingComplianceReport
    trace: MeshingTrace
    adapter_reports: tuple[AdapterReport, ...]
    provider: MeshingProviderInfo
    runtime: MeshingRuntimeInfo
    derivative_mode: MeshingDerivativeMode = eqx.field(static=True)
    provenance: SemanticProvenance
    result_id: str = eqx.field(static=True)

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
    ):
        if not isinstance(mesh, CellMesh):
            raise TypeError("mesh must be CellMesh.")
        if not isinstance(geometry, CellGeometrySpec):
            raise TypeError("geometry must be CellGeometrySpec.")
        geometry.resolve(mesh)
        if not isinstance(coordinate_contract, SpatialCoordinateContract):
            raise TypeError("coordinate_contract must be SpatialCoordinateContract.")
        if boundary is not None and not isinstance(boundary, SurfaceModel):
            raise TypeError("boundary must be SurfaceModel or None.")
        if not isinstance(audit, CellMeshAuditReport) or audit.mesh_id != mesh.mesh_id:
            raise ValueError("Audit must be bound to the result mesh.")
        if (
            audit.geometry_layout_id != geometry.geometry_layout_id
            or audit.geometry_id != _geometry_id(geometry)
        ):
            raise ValueError(
                "Audit must be bound to the result geometry values and layout."
            )
        audit.require_passed()
        if (
            not isinstance(quality, CellQualityReport)
            or quality.report_id != audit.quality.report_id
        ):
            raise ValueError("Quality report must match the mesh audit.")
        if not isinstance(compliance, MeshingComplianceReport) or not compliance.passed:
            raise ValueError("Successful meshing results require passed compliance.")
        if not isinstance(trace, MeshingTrace) or not trace.successful:
            raise ValueError("Successful meshing results require a successful trace.")
        if not isinstance(provider, MeshingProviderInfo):
            raise TypeError("provider must be MeshingProviderInfo.")
        if (
            not isinstance(runtime, MeshingRuntimeInfo)
            or runtime.provider_id != provider.provider_id
        ):
            raise ValueError("Meshing runtime must match the provider.")
        if not isinstance(derivative_mode, MeshingDerivativeMode):
            raise TypeError("derivative_mode must be MeshingDerivativeMode.")
        if not isinstance(provenance, SemanticProvenance):
            raise TypeError("provenance must be SemanticProvenance.")
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
            patches_,
            zones_,
            labels_,
            attributes,
            associations,
        ):
            raise ValueError(
                "Result organization and associations must match the audited evidence."
            )
        self.mesh = mesh
        self.geometry = geometry
        self.coordinate_contract = coordinate_contract
        self.boundary = boundary
        self.patches = patches_
        self.zones = zones_
        self.labels = labels_
        self.attributes = tuple(attributes)
        self.associations = tuple(associations)
        self.audit = audit
        self.quality = quality
        self.compliance = compliance
        self.trace = trace
        self.adapter_reports = tuple(adapter_reports)
        self.provider = provider
        self.runtime = runtime
        self.derivative_mode = derivative_mode
        self.provenance = provenance
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
                "audit": audit.report_id,
                "quality": quality.report_id,
                "compliance": compliance.report_id,
                "trace": trace.trace_id,
                "adapter_reports": [value.report_id for value in adapter_reports],
                "provider": provider.provider_id,
                "runtime": runtime.runtime_id,
                "derivative_mode": derivative_mode.value,
                "provenance": provenance.semantic_id,
            }
        )


__all__ = ["CellMeshingResult", "MeshingComplianceReport", "MeshingRuntimeInfo"]
