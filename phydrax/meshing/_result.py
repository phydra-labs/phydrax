#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import equinox as eqx

from .._fingerprint import canonical_fingerprint
from .._identity import SemanticProvenance
from .._physical import SpatialCoordinateContract
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import CellGeometrySpec, CellMesh
from ..discretization._cell_geometry_validity import cell_geometry_id
from ..geometry._global_embedding import ValidatedGlobalEmbedding
from ..geometry._mesh_certificates import _same_tree_content
from ..geometry._planar_embedding import PlanarEmbedding
from ..geometry._source_fidelity import ValidatedSourceFidelityWitness
from ..geometry._surface_source_support import SurfaceSourceCharts
from ..geometry.surface import SurfaceModel
from ..interchange import AdapterReport
from ..typing import checked
from ._association import GeometryAssociation
from ._audit import (
    _evidence_id,
    _mesh_evidence_issues,
    CellMeshAuditPolicy,
    CellMeshAuditReport,
)
from ._certification import MeshCertificationReport
from ._contracts import (
    MeshingDerivativeMode,
    MeshingExecutionMode,
    MeshingProviderInfo,
)
from ._dual_extraction import SurfaceQuadFieldEvent
from ._layer_core_carrier_witness import NativeLayerCoreCarrierWitness
from ._layer_core_root_geometry import ComputedLayerCoreRoot, LayerCoreRootGeometry
from ._layer_core_subdivision import OverlapSourceStencils, RefinedColumnCertification
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


if TYPE_CHECKING:
    from ..geometry._global_embedding import GlobalEmbeddingCertificate
    from ..geometry._nested_restriction_embedding import NestedRestrictionPremise
    from ._canonical import _CellMeshAcceptance
    from .providers._native_sources import NativeLayerCoreSource


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
    validated: tuple[ValidatedSourceFidelityWitness, ...],
    /,
) -> None:
    if certification is None:
        if validated:
            raise ValueError(
                "Validated fidelity witnesses are reused only by certified results."
            )
        return
    certification.inputs.require_current(
        mesh,
        geometry,
        domain=certification.inputs.domain,
        source=certification.inputs.source,
        validated=validated,
    )
    certification.require_passed()
    if not _same_tree_content(audit, certification.audit):
        raise ValueError("Result audit must match its computed scientific acceptance.")
    if not _same_tree_content(boundary, certification.boundary):
        raise ValueError("Result boundary must match its computed scientific acceptance.")


def _restored_audit_policy(stored: CellMeshAuditPolicy, /) -> CellMeshAuditPolicy:
    """Rebuild a stored audit policy and require its stored identity."""
    from ..discretization._cell_geometry_validity import CellValidityPolicy

    if type(stored) is not CellMeshAuditPolicy:
        raise TypeError("Stored audit policy must be CellMeshAuditPolicy.")
    validity = CellValidityPolicy(
        relative_determinant_floor=stored.validity_policy.relative_determinant_floor,
        relative_planarity_tolerance=stored.validity_policy.relative_planarity_tolerance,
    )
    names = (
        "unused_entities",
        "require_complete_association",
        "minimum_measure",
        "minimum_mean_ratio",
        "maximum_aspect_ratio",
        "coincident_vertices",
        "coincident_vertex_tolerance",
        "duplicate_cells",
        "nonmanifold",
        "inconsistent_orientation",
        "watertight_boundary",
        "self_intersection",
        "invalid_geometry",
        "unresolved",
    )
    policy = CellMeshAuditPolicy(
        **{name: getattr(stored, name) for name in names},
        validity_policy=validity,
    )
    if policy.policy_id != stored.policy_id or not _same_tree_content(policy, stored):
        raise ValueError("Stored audit policy changed.")
    return policy


def _restored_adapter_report(stored: AdapterReport, /) -> AdapterReport:
    """Rebuild one stored conversion report and its negotiation by constructor.

    Losses, profiles, requirements, capabilities, waivers, and nested stages are
    rebuilt by their owners; the report identity must reproduce. The external
    conversion itself is never replayed.
    """
    from ..interchange._report import (
        AdapterCapability,
        AdapterFormatProfile,
        AdapterLoss,
        AdapterRequirement,
        AdapterWaiver,
    )

    if type(stored) is not AdapterReport:
        raise TypeError("Stored adapter reports must be AdapterReport values.")

    def profile(value: AdapterFormatProfile, /) -> AdapterFormatProfile:
        rebuilt = AdapterFormatProfile(value.format, qualifiers=value.qualifiers)
        if rebuilt.profile_id != value.profile_id:
            raise ValueError("Stored adapter format profile changed.")
        return rebuilt

    rebuilt = AdapterReport(
        stored.status,
        stored.source_format,
        stored.target_format,
        source_id=stored.source_id,
        target_id=stored.target_id,
        coordinate_mapping=stored.coordinate_mapping,
        preserved_fields=stored.preserved_fields,
        assumptions=stored.assumptions,
        losses=tuple(
            AdapterLoss(
                item.path,
                item.direction,
                item.category,
                item.rationale,
                changes_interpretation=item.changes_interpretation,
                affected_capability_ids=item.affected_capability_ids,
            )
            for item in stored.losses
        ),
        stage=stored.stage,
        source_profile=profile(stored.source_profile),
        target_profile=profile(stored.target_profile),
        requirements=tuple(
            AdapterRequirement(
                item.semantic_id, required=item.required, rationale=item.rationale
            )
            for item in stored.requirements
        ),
        capabilities=tuple(
            AdapterCapability(item.semantic_id, detail=item.detail)
            for item in stored.capabilities
        ),
        waivers=tuple(
            AdapterWaiver(item.loss_id, item.rationale) for item in stored.waivers
        ),
        stages=tuple(_restored_adapter_report(item) for item in stored.stages),
    )
    if rebuilt.report_id != stored.report_id or not _same_tree_content(rebuilt, stored):
        raise ValueError("Stored adapter report changed.")
    return rebuilt


def _restored_global_embedding(
    stored: GlobalEmbeddingCertificate, mesh: CellMesh, geometry: CellGeometrySpec, /
) -> GlobalEmbeddingCertificate:
    """Recompute a stored global embedding over restored mesh and coordinate map.

    Local validity and every global proof obligation are recomputed (through a
    restored nested-restriction premise when the proof used one); the recomputed
    certificates must reproduce the stored identities.
    """
    from ..discretization._cell_geometry_validity import (
        CellValidityPolicy,
        certify_cell_geometry_validity,
    )
    from ..geometry._global_embedding import (
        certify_global_embedding,
        GlobalEmbeddingCertificate,
    )

    if (
        type(stored) is not GlobalEmbeddingCertificate
        or stored.periodic is not None
        or stored.periodic_restriction_premise is not None
    ):
        raise TypeError("Restored global embeddings require a nonperiodic proof.")
    stored_validity = stored.validity
    validity = certify_cell_geometry_validity(
        geometry,
        mesh=None if stored_validity.mesh is None else mesh,
        policy=CellValidityPolicy(
            relative_determinant_floor=stored_validity.policy.relative_determinant_floor,
            relative_planarity_tolerance=stored_validity.policy.relative_planarity_tolerance,
        ),
    )
    if validity.certificate_id != stored_validity.certificate_id:
        raise ValueError("Stored cell validity does not reproduce its proof.")
    nested = stored.nested_restriction_premise
    embedding = certify_global_embedding(
        mesh,
        geometry,
        validity,
        junction_vertices=stored.binding.junction_vertices,
        nested_restriction=(
            None if nested is None else _restored_nested_restriction_premise(nested)
        ),
    )
    if embedding.certificate_id != stored.certificate_id:
        raise ValueError("Stored global embedding does not reproduce its proof.")
    return embedding


def _restored_nested_restriction_premise(
    stored: NestedRestrictionPremise, /
) -> NestedRestrictionPremise:
    """Rebuild a stored nested-restriction premise from its restored owners.

    The source embedding is recomputed over the restored source mesh and
    coordinate map and the coordinate transition is rebuilt by its owner;
    acceptance then re-proves the nested restriction theorem over them.
    """
    from ..discretization._cell_geometry_transition_types import CellGeometryTransition
    from ..geometry._nested_restriction_embedding import NestedRestrictionPremise
    from ..geometry._surface_source_archive import _geometry
    from ._model_codec import _same_model_content, restore_model_archive_value

    if type(stored) is not NestedRestrictionPremise:
        raise TypeError("Stored embedding premise must be NestedRestrictionPremise.")
    source_mesh = CellMesh.from_stored_fields(stored.source_mesh)
    source_geometry = _geometry(stored.source_geometry, source_mesh)
    source_embedding = _restored_global_embedding(
        stored.source_embedding, source_mesh, source_geometry
    )
    transition = restore_model_archive_value(stored.transition)
    if type(transition) is not CellGeometryTransition:
        raise TypeError("Stored embedding premise lost its geometry transition.")
    premise = NestedRestrictionPremise(
        source_mesh, source_geometry, source_embedding, transition
    )
    if not _same_model_content(premise, stored):
        raise ValueError("Stored nested-restriction premise changed.")
    return premise


def _with_stored_quality_evidence(
    value: CellMeshingResult,
    result: CellMeshingResult | _CellMeshAcceptance,
    stored_quality: CellQualityReport,
    /,
) -> tuple[CellMeshAuditReport, MeshCertificationReport | None]:
    """Carry the stored quality diagnostics into a reproved acceptance.

    The caller has already required the recomputed quality, audit and
    certification identities to equal the stored ones; they bind only the
    evaluation request and the exactly decided quality gates. The sampled XLA
    quality arrays and their float summaries are platform-dependent
    diagnostics, so the restored reports keep the stored evidence instead of
    this platform's re-evaluation. The stored evaluation must have the
    recomputed structure (block layout, cell IDs, array shapes and dtypes), and
    every stored copy of it must be the same evidence.
    """
    from copy import copy

    import jax
    import numpy as np

    from ._model_codec import _same_model_content

    stored_reports = (value.audit.quality, value.quality) + (
        () if value.certification is None else (value.certification.audit.quality,)
    )
    for stored in stored_reports:
        if stored is not stored_quality and not _same_model_content(
            CellQualityReport.from_stored_fields(stored), stored_quality
        ):
            raise ValueError("Stored quality reports retain different evidence.")
    recomputed = result.audit.quality
    if result.quality is not recomputed or (
        result.certification is not None
        and result.certification.audit is not result.audit
    ):
        raise ValueError(
            "Recomputed acceptance must retain one audit and quality report."
        )
    stored_leaves, stored_structure = jax.tree_util.tree_flatten(
        stored_quality.evaluation
    )
    recomputed_leaves, recomputed_structure = jax.tree_util.tree_flatten(
        recomputed.evaluation
    )
    if stored_structure != recomputed_structure:
        raise ValueError("Stored quality evaluation has another structure.")
    for stored_leaf, recomputed_leaf in zip(
        stored_leaves, recomputed_leaves, strict=True
    ):
        stored_array = np.asarray(stored_leaf)
        recomputed_array = np.asarray(recomputed_leaf)
        if stored_array.shape != recomputed_array.shape or (
            stored_array.dtype != recomputed_array.dtype
        ):
            raise ValueError("Stored quality evaluation has another layout.")
        if np.issubdtype(recomputed_array.dtype, np.integer) and not np.array_equal(
            stored_array, recomputed_array
        ):
            raise ValueError("Stored quality evaluation names other cells.")
    audit = copy(result.audit)
    object.__setattr__(audit, "quality", stored_quality)
    certification = result.certification
    if certification is not None:
        certification = copy(certification)
        object.__setattr__(certification, "audit", audit)
    return audit, certification


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
    surface_quad_field_event: SurfaceQuadFieldEvent | None
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
    layer_core_carrier_witness: NativeLayerCoreCarrierWitness | None
    layer_core_overlap_stencils: OverlapSourceStencils | None
    layer_core_column_certification: RefinedColumnCertification | None
    layer_core_root_geometry: LayerCoreRootGeometry | None

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
        surface_quad_field_event: SurfaceQuadFieldEvent | None = None,
        region_boundary_evidence: tuple[RegionBoundaryEvidence, ...] = (),
        planar_embedding: PlanarEmbedding | None = None,
        certification: MeshCertificationReport | None = None,
        layer_curving_evidence: LayerCurvingSourceEvidence | None = None,
        layer_core_carrier_witness: NativeLayerCoreCarrierWitness | None = None,
        layer_core_overlap_stencils: OverlapSourceStencils | None = None,
        layer_core_column_certification: RefinedColumnCertification | None = None,
        layer_core_root_geometry: LayerCoreRootGeometry | None = None,
        computed_root: ComputedLayerCoreRoot | None = None,
        validated_witnesses: tuple[ValidatedSourceFidelityWitness, ...] = (),
    ) -> None:
        """Bind one meshing result to its reports and computed acceptance.

        ``validated_witnesses`` are fidelity-witness validations already made
        for exactly these mesh, coordinate map and source objects; checking the
        retained acceptance does not repeat them.
        ``computed_root`` is a fresh operation-local proof for the exact
        retained layer source, specification, mesh, geometry and root; it is
        checked and never stored in the result.
        """
        geometry.resolve(mesh)
        if surface_quad_field_event is not None:
            surface_quad_field_event.require_target(mesh, geometry)
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
        _require_result_certification(
            certification, mesh, geometry, audit, boundary, validated_witnesses
        )
        if layer_curving_evidence is not None:
            if layer_curving_evidence.frame_id != coordinate_contract.spatial_id:
                raise ValueError(
                    "Layer curving evidence belongs to a different physical frame."
                )
            layer_curving_evidence.require_current(mesh, geometry)
        mapped_source = None
        mapped_layer_root = False
        if layer_core_root_geometry is not None:
            root = layer_core_root_geometry
            mapped_layer_root = root.reference_mesh is not None
            if (
                layer_core_carrier_witness is None
                or root.source_binding_id != layer_core_carrier_witness.source_binding_id
                or root.specification_id != layer_core_carrier_witness.specification_id
            ):
                raise ValueError(
                    "Layer root geometry and retained evidence have different source bindings."
                )
            if mapped_layer_root:
                from .providers._native_sources import NativeLayerCoreSource

                if certification is None:
                    raise ValueError(
                        "Mapped physical layer results require certification."
                    )
                source = certification.inputs.feature_source
                if (
                    not isinstance(source, NativeLayerCoreSource)
                    or source.mapped_domain is None
                    or certification.inputs.layer_core_root_geometry is not root
                    or layer_core_overlap_stencils is not None
                    or layer_core_column_certification is not None
                ):
                    raise ValueError(
                        "Mapped physical layer results require their original mapped root."
                    )
                mapped_source = source
            elif (
                layer_core_overlap_stencils is None
                or layer_core_column_certification is None
                or root.overlap_stencils.witness_id
                != layer_core_overlap_stencils.witness_id
                or layer_core_column_certification.source_binding_id
                != root.source_binding_id
                or layer_core_column_certification.specification_id
                != root.specification_id
            ):
                raise ValueError(
                    "Layer root geometry and retained evidence have different source bindings."
                )
        if computed_root is not None:
            from .providers._native_sources import NativeLayerCoreSource

            if (
                layer_core_root_geometry is None
                or certification is None
                or type(certification.inputs.feature_source) is not NativeLayerCoreSource
            ):
                raise ValueError("A computed root requires its actual certified source.")
            computed_root.require(
                certification.inputs.feature_source,
                layer_core_root_geometry.specification,
                layer_core_root_geometry.mesh,
                layer_core_root_geometry.geometry,
                layer_core_root_geometry,
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
        self.surface_source = surface_source
        self.surface_quad_field_event = surface_quad_field_event
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
        self.layer_core_carrier_witness = layer_core_carrier_witness
        self.layer_core_overlap_stencils = layer_core_overlap_stencils
        self.layer_core_column_certification = layer_core_column_certification
        self.layer_core_root_geometry = layer_core_root_geometry
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
                "surface_quad_field_event": None
                if surface_quad_field_event is None
                else surface_quad_field_event.event_id,
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
                **(
                    {}
                    if layer_core_carrier_witness is None
                    else {
                        "layer_core_carrier_witness": layer_core_carrier_witness.witness_id
                    }
                ),
                **(
                    {}
                    if layer_core_overlap_stencils is None
                    else {
                        "layer_core_overlap_stencils": layer_core_overlap_stencils.witness_id
                    }
                ),
                **(
                    {}
                    if layer_core_column_certification is None
                    else {
                        "layer_core_column_certification": layer_core_column_certification.certification_id
                    }
                ),
                **(
                    {}
                    if layer_core_root_geometry is None
                    else {"layer_core_root_geometry": layer_core_root_geometry.root_id}
                ),
            }
        )
        if mapped_layer_root:
            from ._layer_core_association_transfer import (
                NativeLayerCoreAssociationTransfer,
            )

            if mapped_source is None or layer_core_root_geometry is None:
                raise ValueError("Mapped layer result lost its source/root authority.")
            transfer = NativeLayerCoreAssociationTransfer(
                mapped_source, layer_core_root_geometry.specification
            )
            if computed_root is None:
                transfer.validate_source(self)
            else:
                transfer._validate_source(self, computed_root=computed_root)

    @classmethod
    def from_stored_fields(
        cls: type[CellMeshingResult],
        value: CellMeshingResult,
        /,
        *,
        layer_curving_evidence: LayerCurvingSourceEvidence | None = None,
        layer_core_carrier_witness: NativeLayerCoreCarrierWitness | None = None,
        layer_core_overlap_stencils: OverlapSourceStencils | None = None,
        layer_core_column_certification: RefinedColumnCertification | None = None,
        layer_core_root_geometry: LayerCoreRootGeometry | None = None,
        layer_core_feature_source: NativeLayerCoreSource | None = None,
        computed_root: ComputedLayerCoreRoot | None = None,
        _computed_embedding_out: list[ValidatedGlobalEmbedding] | None = None,
        _validated_chart_parent_embedding: ValidatedGlobalEmbedding | None = None,
    ) -> CellMeshingResult:
        """Reconstruct an accepted result and independently recompute its proof.

        Layer evidence, when present, must be supplied by the lifecycle owner
        after replaying its source and generation; archived witness claims are
        never themselves authority for a carrier.
        A supplied ``computed_root`` is a fresh, operand-bound proof from the
        same restore operation, not a stored authority.
        An uncertified result (the direct cell-mesh audit of ``certify_cell_mesh``
        without certification inputs) is re-audited under the audit policy its
        stored audit report retains.
        """
        from copy import copy

        from .._model._structure import (
            AdaptiveImplicitBoundarySourceArchiveRecipe,
            LayerCoreFidelitySourceReferenceArchiveRecipe,
            LayerCoreSourceReferenceArchiveRecipe,
            MappedCoreFacetAuthorityArchiveRecipe,
            MappedReferenceBoundarySourceArchiveRecipe,
            model_archive_projection,
            NativeImplicitBoundaryQueryArchiveRecipe,
        )
        from ..discretization._cell_geometry_validity import CellValidityPolicy
        from ..geometry._surface_source_archive import (
            _geometry,
            restore_surface_source_charts,
        )
        from ..interchange._cad_archive import _reconstruct_brep_graph
        from ._association import GeometryAssociationArchiveRecipe
        from ._audit import CellMeshManifoldScope
        from ._canonical import _certify_cell_mesh_acceptance
        from ._certification_inputs import (
            MeshCertificationInputs,
            validate_fidelity_witnesses,
        )
        from ._certification_schedule import MeshCertificationSchedule
        from ._controls import ProtectedFeature
        from ._model_codec import (
            _register_model_archive_types,
            _same_implicit_archive,
            _same_model_content,
            restore_mapped_layer_core_boundary_source,
            restore_model_archive_value,
        )
        from ._native_construction import NativeMeshOrganization
        from ._scope import MeshingScope
        from .providers._native_sources import NativeLayerCoreSource

        if type(value) is not cls or (
            value.certification is not None
            and type(value.certification) is not MeshCertificationReport
        ):
            raise TypeError(
                "Stored result certification must be a MeshCertificationReport or absent."
            )
        _register_model_archive_types()
        for stored, regenerated, name in (
            (value.layer_curving_evidence, layer_curving_evidence, "layer curving"),
            (
                value.layer_core_carrier_witness,
                layer_core_carrier_witness,
                "layer core carrier",
            ),
            (
                value.layer_core_overlap_stencils,
                layer_core_overlap_stencils,
                "layer core overlap source stencils",
            ),
            (
                value.layer_core_column_certification,
                layer_core_column_certification,
                "layer core column certification",
            ),
            (
                value.layer_core_root_geometry,
                layer_core_root_geometry,
                "layer core root geometry",
            ),
        ):
            if (stored is None) != (regenerated is None) or (
                stored is not None and not _same_model_content(stored, regenerated)
            ):
                raise ValueError(
                    f"Stored {name} evidence requires independent lifecycle replay."
                )

        mesh = CellMesh.from_stored_fields(value.mesh)
        geometry = _geometry(value.geometry, mesh)
        if computed_root is not None:
            root = layer_core_root_geometry
            if (
                type(root) is not LayerCoreRootGeometry
                or type(layer_core_feature_source) is not NativeLayerCoreSource
            ):
                raise ValueError("A computed root requires its actual source and root.")
            computed_root.require(
                layer_core_feature_source,
                root.specification,
                root.mesh,
                root.geometry,
                root,
            )
            if mesh.mesh_id == root.mesh.mesh_id:
                if (
                    cell_geometry_id(geometry) != cell_geometry_id(root.geometry)
                    or not _same_tree_content(mesh, root.mesh)
                    or not _same_tree_content(geometry, root.geometry)
                    or not _same_model_content(mesh, root.mesh)
                    or not _same_model_content(geometry, root.geometry)
                ):
                    raise ValueError(
                        "Stored root endpoint differs from its proved operands."
                    )
                mesh, geometry = root.mesh, root.geometry
        contract = SpatialCoordinateContract.from_dict(
            value.coordinate_contract.to_dict()
        )
        if contract.spatial_id != value.coordinate_contract.spatial_id:
            raise ValueError("Stored result coordinate contract changed.")

        def scope(item: MeshingScope) -> MeshingScope:
            return MeshingScope.from_stored_fields(item)

        patches = tuple(
            MeshPatch(
                item.name,
                scope(item.scope),
                connected=item.connected,
                adjacent_zone_ids=item.adjacent_zone_ids,
                source_adjacent_region_ids=item.source_adjacent_region_ids,
                source_adjacent_region_occurrence_paths=item.source_adjacent_region_occurrence_paths,
            )
            for item in value.patches
        )
        zones = tuple(
            MeshZone(
                item.name,
                item.role,
                scope(item.scope),
                material_id=item.material_id,
                region_role=item.region_role,
            )
            for item in value.zones
        )
        labels = tuple(MeshLabel(item.name, scope(item.scope)) for item in value.labels)
        attributes = tuple(
            MeshAttribute(
                item.name,
                item.role,
                scope(item.scope),
                item.values,
                unit=None if item.unit is None else _reconstruct_brep_graph(item.unit),
            )
            for item in value.attributes
        )
        associations = tuple(
            GeometryAssociationArchiveRecipe.from_association(item).restore()
            for item in value.associations
        )
        for original, rebuilt, identity in (
            (value.patches, patches, "patch_id"),
            (value.zones, zones, "zone_id"),
            (value.labels, labels, "label_id"),
            (value.attributes, attributes, "attribute_id"),
        ):
            if any(
                getattr(a, identity) != getattr(b, identity)
                for a, b in zip(original, rebuilt, strict=True)
            ):
                raise ValueError("Stored mesh organization differs from its constructor.")

        boundary = None
        if value.boundary is not None:
            from ..geometry.surface._contracts import (
                SurfaceInterface,
                SurfaceMetadata,
                SurfaceOrientationRepair,
                SurfaceSelection,
            )

            stored_boundary = value.boundary
            stored_metadata = stored_boundary.metadata
            metadata = SurfaceMetadata(
                source_id=stored_metadata.source_id,
                source_revision=stored_metadata.source_revision,
                coordinate_contract=SpatialCoordinateContract.from_dict(
                    stored_metadata.coordinate_contract.to_dict()
                ),
                provenance=stored_metadata.provenance,
                cell_tags=stored_metadata.cell_tags,
            )
            if metadata.metadata_id != stored_metadata.metadata_id:
                raise ValueError("Stored surface metadata changed.")

            def selection(item: SurfaceSelection) -> SurfaceSelection:
                rebuilt = SurfaceSelection(
                    item.name,
                    item.cell_global_ids,
                    cell_entity_set_id=item.cell_entity_set_id,
                    role=item.role,
                )
                if rebuilt.selection_id != item.selection_id:
                    raise ValueError("Stored surface selection changed.")
                return rebuilt

            selections = tuple(selection(item) for item in stored_boundary.selections)
            interfaces = tuple(
                SurfaceInterface(
                    item.name,
                    selection(item.support),
                    minus_region=item.minus_region,
                    plus_region=item.plus_region,
                )
                for item in stored_boundary.interfaces
            )
            if any(
                a.interface_id != b.interface_id
                for a, b in zip(stored_boundary.interfaces, interfaces, strict=True)
            ):
                raise ValueError("Stored surface interface changed.")
            repair = stored_boundary.orientation_repair
            if repair is not None:
                rebuilt_repair = SurfaceOrientationRepair(
                    source_face_indices=repair.source_face_indices,
                    orientation_signs=repair.orientation_signs,
                    component_ids=repair.component_ids,
                    source_topology_id=repair.source_topology_id,
                    repaired_topology_id=repair.repaired_topology_id,
                )
                if rebuilt_repair.repair_id != repair.repair_id:
                    raise ValueError("Stored surface orientation repair changed.")
                repair = rebuilt_repair
            boundary = SurfaceModel(
                CellMesh.from_stored_fields(stored_boundary.mesh),
                metadata,
                selections=selections,
                interfaces=interfaces,
                orientation_repair=repair,
            )
            if boundary.model_id != stored_boundary.model_id:
                raise ValueError("Stored boundary differs from its reconstructed owners.")
        surface_source = (
            None
            if value.surface_source is None
            else restore_surface_source_charts(
                model_archive_projection(value.surface_source)
            )
        )
        region_boundaries = tuple(
            RegionBoundaryEvidence(
                mesh,
                scope(item.source_scope),
                item.source_region_id,
                item.control_id,
                item.material_id,
                item.role,
                MeshLabel(item.boundary_label.name, scope(item.boundary_label.scope)),
                item.patch_sides,
                source_occurrence_path=item.source_occurrence_path,
            )
            for item in value.region_boundary_evidence
        )
        if any(
            a.evidence_id != b.evidence_id
            for a, b in zip(
                value.region_boundary_evidence, region_boundaries, strict=True
            )
        ):
            raise ValueError(
                "Stored region boundaries differ from their source evidence."
            )
        embedding = None
        if value.planar_embedding is not None:
            item = value.planar_embedding
            embedding = PlanarEmbedding(
                item.origin, item.x_axis, item.y_axis, item.normal
            )
            if embedding.embedding_id != item.embedding_id:
                raise ValueError("Stored planar embedding changed.")

        if value.certification is None:
            if (
                computed_root is not None
                or layer_core_feature_source is not None
                or _computed_embedding_out is not None
                or _validated_chart_parent_embedding is not None
            ):
                raise ValueError(
                    "An uncertified stored result has no certified source authority."
                )
            audit_policy = _restored_audit_policy(value.audit.policy)
            stored_scope = value.audit.manifold_scope
            manifold_scope = None
            if stored_scope is not None:
                manifold_scope = CellMeshManifoldScope(
                    stored_scope.kind,
                    stored_scope.vertex_ids,
                    star_cells=tuple(
                        zip(
                            stored_scope.star_cell_kinds,
                            stored_scope.star_cell_ids,
                            stored_scope.star_cell_vertex_ids,
                            strict=True,
                        )
                    ),
                )
                if manifold_scope.scope_id != stored_scope.scope_id:
                    raise ValueError("Stored audit manifold scope changed.")
            audited = _certify_cell_mesh_acceptance(
                mesh,
                contract,
                geometry=geometry,
                audit_policy=audit_policy,
                boundary=boundary,
                patches=patches,
                zones=zones,
                labels=labels,
                attributes=attributes,
                associations=associations,
                surface_source=surface_source,
                region_boundary_evidence=region_boundaries,
                planar_embedding=embedding,
                layer_curving_evidence=layer_curving_evidence,
                layer_core_carrier_witness=layer_core_carrier_witness,
                layer_core_overlap_stencils=layer_core_overlap_stencils,
                layer_core_column_certification=layer_core_column_certification,
                layer_core_root_geometry=layer_core_root_geometry,
                manifold_scope=manifold_scope,
            )
            return cls._complete_reproved_acceptance(
                value, audited, CellQualityReport.from_stored_fields(value.audit.quality)
            )
        report = value.certification
        original = report.inputs
        input_mesh = CellMesh.from_stored_fields(original.mesh)
        input_geometry = _geometry(original.geometry, input_mesh)
        if input_mesh.mesh_id != mesh.mesh_id or not _same_model_content(
            input_geometry, geometry
        ):
            raise ValueError("Stored certification inputs bind another mesh or geometry.")
        schedule = MeshCertificationSchedule(original.schedule.route)
        if not _same_tree_content(schedule, original.schedule):
            raise ValueError("Stored certification schedule changed.")
        # One owner often plays several certification roles (an analytic profile
        # is domain, fidelity query and feature source); restore each distinct
        # stored owner once so the roles keep one exact object.
        restored_owners: list[tuple[Any, Any]] = []

        def restored_owner(stored: Any, /) -> Any:
            for previous, owner in restored_owners:
                if previous is stored or _same_model_content(previous, stored):
                    return owner
            owner = restore_model_archive_value(stored)
            restored_owners.append((stored, owner))
            return owner

        implicit_query = (
            original.source.query
            if type(original.source) is NativeImplicitBoundaryQueryArchiveRecipe
            else original.source
        )
        shared_implicit = type(
            implicit_query
        ) is AdaptiveImplicitBoundarySourceArchiveRecipe and _same_implicit_archive(
            original.domain, implicit_query.surface
        )
        if shared_implicit:
            # The complete stored discovery recipes agree byte-for-byte. Restore
            # their shared scientific owner once through the query's normal
            # constructor and full discovery replay, rather than independently
            # rebuilding callback-bearing interval owners that compare by identity.
            source = restored_owner(original.source)
            adaptive = (
                source.adaptive_source
                if type(original.source) is NativeImplicitBoundaryQueryArchiveRecipe
                else source
            )
            from ..geometry.implicit._adaptive_discovery import (
                AdaptiveImplicitBoundarySource,
            )

            if type(adaptive) is not AdaptiveImplicitBoundarySource:
                raise TypeError("Restored implicit query lost its adaptive source owner.")
            domain = adaptive.surface
            restored_owners.append((original.domain, domain))
        else:
            domain = None if original.domain is None else restored_owner(original.domain)
            source = (
                None
                if original.source is None
                or type(original.source) is LayerCoreFidelitySourceReferenceArchiveRecipe
                else restored_owner(original.source)
            )
        if type(original.feature_source) is LayerCoreSourceReferenceArchiveRecipe:
            reference = original.feature_source
            if (
                type(layer_core_feature_source) is not NativeLayerCoreSource
                or reference.binding_id != layer_core_feature_source.binding_id
                or reference.source_id != layer_core_feature_source.source_id
                or reference.source_revision != layer_core_feature_source.source_revision
            ):
                raise ValueError(
                    "Restored layer core feature source differs from the stored source binding."
                )
            restored_feature_source = layer_core_feature_source
            actual_source = layer_core_feature_source.fidelity_source
            if type(original.source) is LayerCoreFidelitySourceReferenceArchiveRecipe:
                from ..geometry._mapped_reference_boundary import (
                    MappedReferenceBoundarySource,
                )

                stored_source = original.source
                if (
                    type(actual_source) is not MappedReferenceBoundarySource
                    or stored_source.source_id != layer_core_feature_source.source_id
                    or stored_source.source_revision
                    != layer_core_feature_source.source_revision
                    or stored_source.binding_id != layer_core_feature_source.binding_id
                    or actual_source.source_id != stored_source.source_id
                    or actual_source.source_revision != stored_source.source_revision
                    or stored_source.query_id != actual_source.query_id
                ):
                    raise ValueError(
                        "Stored fidelity query reference differs from the restored layer core source."
                    )
            elif not _same_model_content(source, actual_source):
                raise ValueError(
                    "Stored fidelity query differs from the restored layer core source."
                )
            source = actual_source
        else:
            if type(original.source) is LayerCoreFidelitySourceReferenceArchiveRecipe:
                raise ValueError(
                    "Layer core fidelity reference requires its restored source owner."
                )
            if layer_core_feature_source is not None:
                raise ValueError(
                    "Layer core source override requires a stored source reference."
                )
            restored_feature_source = (
                None
                if original.feature_source is None
                else restored_owner(original.feature_source)
            )
            from .providers._native_sources import (
                NativeStructuredSource,
                NativeSweepSource,
            )

            if (
                type(restored_feature_source) is NativeStructuredSource
                or type(restored_feature_source) is NativeSweepSource
            ):
                if not _same_model_content(
                    source, restored_feature_source.fidelity_source
                ) or not _same_model_content(domain, restored_feature_source.domain):
                    raise ValueError(
                        "Stored structured or sweep certification differs from its "
                        "restored source authority."
                    )
                source = restored_feature_source.fidelity_source
                domain = restored_feature_source.domain
        if not _same_model_content(
            original.layer_core_root_geometry, value.layer_core_root_geometry
        ):
            raise ValueError(
                "Stored certification and result name different layer core root geometry."
            )
        features = (
            None
            if original.protected_features is None
            else tuple(
                ProtectedFeature.from_stored_fields(item)
                for item in original.protected_features
            )
        )
        from ..geometry._source_boundary_records import SurfaceEnvelopeDistanceWitness

        def stored_witness[W](witness: W, /) -> W:
            # An envelope witness retains its derived chart source and witness
            # as archive recipes; they are rebuilt through their owners.
            return (
                restore_model_archive_value(witness)
                if type(witness) is SurfaceEnvelopeDistanceWitness
                else witness
            )

        fidelity_witness = stored_witness(original.fidelity_witness)

        def restore_scoped(
            query: Any, witness: Any, tolerance: float
        ) -> tuple[Any, Any, float]:
            same_query = shared_implicit and _same_implicit_archive(
                query, original.source
            )
            if same_query:
                restored_query = source
            elif (
                type(query) is MappedReferenceBoundarySourceArchiveRecipe
                and type(query.authority) is MappedCoreFacetAuthorityArchiveRecipe
            ):
                restored_query = restore_mapped_layer_core_boundary_source(
                    query, layer_core_feature_source
                )
            else:
                restored_query = restore_model_archive_value(query)
            restored_witness = (
                fidelity_witness
                if same_query
                and _same_implicit_archive(witness, original.fidelity_witness)
                else stored_witness(witness)
            )
            return restored_query, restored_witness, tolerance

        scoped = tuple(
            restore_scoped(query, witness, tolerance)
            for query, witness, tolerance in original.scoped_fidelity
        )
        validated = validate_fidelity_witnesses(
            mesh, geometry, source, fidelity_witness, scoped
        )
        inputs = MeshCertificationInputs(
            mesh,
            geometry,
            schedule,
            source_id=original.source_id,
            source_revision=original.source_revision,
            domain=domain,
            cell_regions=original.cell_regions,
            source=source,
            fidelity_tolerance=original.fidelity_tolerance,
            maximum_normal_deviation=original.maximum_normal_deviation,
            junction_vertices=original.junction_vertices,
            protected_features=features,
            scoped_fidelity=scoped,
            fidelity_witness=fidelity_witness,
            feature_source=restored_feature_source,
            layer_core_root_geometry=layer_core_root_geometry,
            computed_root=computed_root,
            validated=validated,
            source_coverage_witness=(
                None
                if original.source_coverage_witness is None
                else restore_model_archive_value(original.source_coverage_witness)
            ),
        )
        if inputs.request_id != original.request_id:
            raise ValueError("Stored certification request changed.")
        from ..geometry._source_boundary_records import (
            SourceBoundaryChartWitness,
            SourceBoundaryFaceImageWitness,
        )

        witness_sources = ((inputs.source, inputs.fidelity_witness),) + tuple(
            (query, witness) for query, witness, _ in inputs.scoped_fidelity
        )
        checked_associations: set[tuple[str, str, str]] = set()
        for association in associations:
            if association.source_boundary_witness_id is None:
                continue
            key = (
                association.source_boundary_witness_id,
                association.source_id,
                association.source_revision,
            )
            if key in checked_associations:
                continue
            matches = []
            for query, witness in witness_sources:
                if (
                    query is None
                    or not isinstance(
                        witness,
                        (SourceBoundaryChartWitness, SourceBoundaryFaceImageWitness),
                    )
                    or witness.witness_id != association.source_boundary_witness_id
                    or witness.source_id != association.source_id
                    or witness.source_revision != association.source_revision
                ):
                    continue
                if not any(
                    _same_model_content(query, existing_query)
                    and _same_model_content(witness, existing_witness)
                    for existing_query, existing_witness in matches
                ):
                    matches.append((query, witness))
            if len(matches) != 1:
                raise ValueError(
                    "Partial source association requires one independently restored "
                    "matching boundary query and source witness."
                )
            # Reproof of each exact association row and its source-side support
            # is performed by canonical recertification using this same token bank.
            checked_associations.add(key)
        policy = _restored_audit_policy(report.audit_policy)
        organization = NativeMeshOrganization(
            boundary=boundary,
            patches=patches,
            zones=zones,
            labels=labels,
            attributes=attributes,
            associations=associations,
            surface_source=surface_source,
            region_boundary_evidence=region_boundaries,
        )
        stored_report = report
        if report.coverage is not None:
            stored_report = copy(report)
            coverage = copy(report.coverage)
            object.__setattr__(
                coverage, "source", restore_model_archive_value(coverage.source)
            )
            object.__setattr__(stored_report, "coverage", coverage)
        stored_report.require_premises(
            inputs, audit_policy=policy, organization=organization
        )
        stored_quality = CellQualityReport.from_stored_fields(report.audit.quality)
        stored_embedding = report.embedding
        embedding_premise = (
            None
            if stored_embedding is None
            or stored_embedding.nested_restriction_premise is None
            else _restored_nested_restriction_premise(
                stored_embedding.nested_restriction_premise
            )
        )
        computed_embedding = None
        if _computed_embedding_out is not None:
            if not isinstance(_computed_embedding_out, list) or _computed_embedding_out:
                raise ValueError(
                    "Computed embedding output must be an empty operation-local list."
                )
            root = layer_core_root_geometry
            if (
                computed_root is not None
                and root is not None
                and mesh is root.mesh
                and geometry is root.geometry
                and inputs.junction_vertices is None
                and policy.validity_policy.policy_id == CellValidityPolicy().policy_id
            ):
                from ..discretization._cell_geometry_validity import ComputedCellValidity

                computed_validity = ComputedCellValidity(
                    geometry, mesh, policy.validity_policy, ()
                )
                computed_embedding = ValidatedGlobalEmbedding(
                    mesh,
                    geometry,
                    computed_validity.certificate,
                    validated_validity=computed_validity,
                )
        result = _certify_cell_mesh_acceptance(
            mesh,
            contract,
            geometry=geometry,
            audit_policy=policy,
            boundary=boundary,
            certification_inputs=inputs,
            patches=patches,
            zones=zones,
            labels=labels,
            attributes=attributes,
            associations=associations,
            surface_source=surface_source,
            region_boundary_evidence=region_boundaries,
            planar_embedding=embedding,
            layer_curving_evidence=layer_curving_evidence,
            layer_core_carrier_witness=layer_core_carrier_witness,
            layer_core_overlap_stencils=layer_core_overlap_stencils,
            layer_core_column_certification=layer_core_column_certification,
            layer_core_root_geometry=layer_core_root_geometry,
            computed_root=computed_root,
            _validated_embedding=computed_embedding,
            _validated_chart_parent_embedding=_validated_chart_parent_embedding,
            embedding_premise=embedding_premise,
            validated_witnesses=validated,
            # Certified fidelity bounds are stored evidence: each recomputed
            # certificate must reproduce the stored one, which is then kept.
            stored_fidelity=(report.fidelity, report.scoped_fidelity),
        )
        rebuilt = cls._complete_reproved_acceptance(
            value, result, stored_quality, computed_root=computed_root
        )
        if computed_embedding is not None:
            assert _computed_embedding_out is not None
            _computed_embedding_out.append(computed_embedding)
        return rebuilt

    @classmethod
    def _complete_reproved_acceptance(
        cls: type[CellMeshingResult],
        value: CellMeshingResult,
        result: CellMeshingResult | _CellMeshAcceptance,
        stored_quality: CellQualityReport,
        /,
        *,
        computed_root: ComputedLayerCoreRoot | None = None,
    ) -> CellMeshingResult:
        """Finalize original publication metadata after actual acceptance reproof."""
        report = value.certification
        if (report is None) != (result.certification is None):
            raise ValueError("Stored result requires its recomputed acceptance.")
        mesh, geometry, contract = (
            result.mesh,
            result.geometry,
            result.coordinate_contract,
        )
        inputs = None if result.certification is None else result.certification.inputs
        boundary, patches, zones, labels = (
            result.boundary,
            result.patches,
            result.zones,
            result.labels,
        )
        attributes, associations = result.attributes, result.associations
        surface_source, region_boundaries = (
            result.surface_source,
            result.region_boundary_evidence,
        )
        embedding = result.planar_embedding
        layer_curving_evidence = result.layer_curving_evidence
        layer_core_carrier_witness = result.layer_core_carrier_witness
        layer_core_overlap_stencils = result.layer_core_overlap_stencils
        layer_core_column_certification = result.layer_core_column_certification
        layer_core_root_geometry = result.layer_core_root_geometry
        from .._model._structure import model_archive_projection
        from ._model_codec import _same_model_content, restore_model_archive_value

        stored_event = value.surface_quad_field_event
        surface_quad_field_event = (
            None
            if stored_event is None
            else restore_model_archive_value(model_archive_projection(stored_event))
        )
        if (
            surface_quad_field_event is not None
            and type(surface_quad_field_event) is not SurfaceQuadFieldEvent
        ):
            raise TypeError("Stored quad field event lost its canonical owner.")

        if not _same_model_content(mesh, value.mesh) or not _same_model_content(
            geometry, value.geometry
        ):
            raise ValueError("Recomputed acceptance binds different stored operands.")
        if (
            (None if report is None else report.report_id)
            != (None if result.certification is None else result.certification.report_id)
            or result.quality.report_id != stored_quality.report_id
            or result.audit.report_id != value.audit.report_id
            or result.quality.report_id != value.quality.report_id
        ):
            raise ValueError(
                "Stored result differs from its independently recomputed acceptance."
            )
        audit, certification = _with_stored_quality_evidence(
            value, result, stored_quality
        )

        # Scientific acceptance is recomputed above. Reconstruct publication
        # metadata separately: it records the actual provider operation rather
        # than the canonical certifier's own lifecycle.
        from ._contracts import MeshingFailureEvidence
        from ._trace import MeshingDiagnostic, MeshingStageReport

        adapter_reports = tuple(
            _restored_adapter_report(item) for item in value.adapter_reports
        )
        stored_compliance = value.compliance
        compliance = MeshingComplianceReport(
            stored_compliance.specification_id,
            issues=stored_compliance.issues,
            requested=stored_compliance.requested,
            achieved=stored_compliance.achieved,
        )
        if not _same_tree_content(compliance, stored_compliance):
            raise ValueError("Stored meshing compliance changed.")
        stored_provider = value.provider
        provider = MeshingProviderInfo(
            stored_provider.name,
            stored_provider.version,
            stored_provider.license_spdx,
            operations=stored_provider.operations,
            source_kinds=stored_provider.source_kinds,
            capabilities=stored_provider.capabilities,
            cell_kinds=stored_provider.cell_kinds,
            dimensions=stored_provider.dimensions,
            execution_modes=stored_provider.execution_modes,
        )
        if not _same_tree_content(provider, stored_provider):
            raise ValueError("Stored provider identity changed.")
        stored_runtime = value.runtime
        runtime = MeshingRuntimeInfo(
            stored_runtime.provider_id,
            stored_runtime.actual_version,
            stored_runtime.execution_mode,
            deterministic=stored_runtime.deterministic,
        )
        if not _same_tree_content(runtime, stored_runtime):
            raise ValueError("Stored provider runtime changed.")
        stages = []
        for stage in value.trace.stages:
            diagnostics = []
            for item in stage.diagnostics:
                failure = None
                if item.evidence_id:
                    failure = MeshingFailureEvidence(
                        requested_quantities=item.requested_quantities,
                        achieved_quantities=item.achieved_quantities,
                        checkpoint=item.checkpoint,
                        logical_findings=item.logical_findings,
                    )
                diagnostic = MeshingDiagnostic(
                    item.severity,
                    item.message,
                    provider_code=item.provider_code,
                    failure_category=item.failure_category,
                    entity_ids=item.entity_ids,
                    locations=item.locations,
                    failure_evidence=failure,
                )
                if not _same_tree_content(diagnostic, item):
                    raise ValueError("Stored meshing diagnostic changed.")
                diagnostics.append(diagnostic)
            rebuilt_stage = MeshingStageReport(
                stage.stage,
                stage.status,
                input_ids=stage.input_ids,
                output_ids=stage.output_ids,
                diagnostics=tuple(diagnostics),
                created_count=stage.created_count,
                modified_count=stage.modified_count,
                deleted_count=stage.deleted_count,
            )
            if not _same_tree_content(rebuilt_stage, stage):
                raise ValueError("Stored meshing stage changed.")
            stages.append(rebuilt_stage)
        trace = MeshingTrace(tuple(stages))
        if not _same_tree_content(trace, value.trace):
            raise ValueError("Stored meshing trace changed.")
        provenance = SemanticProvenance.from_stored_fields(value.provenance)
        if provenance.matches_content_fields({"kind": "native-mesh-construction"}) and (
            inputs is None
            or report is None
            or not provenance.matches_content_fields(
                {
                    "source": (inputs.source_id, inputs.source_revision),
                    "specification": compliance.specification_id,
                    "certification": report.report_id,
                    "surface_quad_field_event": None
                    if surface_quad_field_event is None
                    else surface_quad_field_event.event_id,
                }
            )
        ):
            raise ValueError(
                "Stored native provenance differs from its revalidated premises."
            )
        if provenance.matches_content_fields(
            {"kind": "native-cell-mesh-certification"}
        ) and not provenance.matches_content_fields(
            {
                "mesh_id": mesh.mesh_id,
                "geometry_layout_id": geometry.geometry_layout_id,
                "coordinate_contract": contract.spatial_id,
                "audit": result.audit.report_id,
            }
        ):
            raise ValueError(
                "Stored certification provenance differs from its revalidated premises."
            )
        rebuilt = cls(
            mesh,
            geometry,
            contract,
            audit,
            audit.quality,
            compliance,
            trace,
            provider,
            runtime,
            value.derivative_mode,
            provenance,
            boundary=boundary,
            patches=patches,
            zones=zones,
            labels=labels,
            attributes=attributes,
            associations=associations,
            surface_source=surface_source,
            surface_quad_field_event=surface_quad_field_event,
            adapter_reports=adapter_reports,
            region_boundary_evidence=region_boundaries,
            planar_embedding=embedding,
            certification=certification,
            layer_curving_evidence=layer_curving_evidence,
            layer_core_carrier_witness=layer_core_carrier_witness,
            layer_core_overlap_stencils=layer_core_overlap_stencils,
            layer_core_column_certification=layer_core_column_certification,
            layer_core_root_geometry=layer_core_root_geometry,
            computed_root=computed_root,
        )
        if rebuilt.result_id != value.result_id:
            raise ValueError("Stored meshing result identity changed.")
        return rebuilt


__all__ = ["CellMeshingResult", "MeshingComplianceReport", "MeshingRuntimeInfo"]
