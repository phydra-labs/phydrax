#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass
from typing import NoReturn, SupportsIndex

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
from ..discretization._cell_geometry import (
    CellVertexGeometryElement,
    ScalarCoordinateElement,
)
from ..discretization._cell_geometry_validity import (
    CellValidityCertificate,
    CellValidityStatus,
    ValidatedCellValidityParent,
)
from ..discretization._hexahedral import HexahedralConnectivity
from ..discretization._reference_cell import reference_cell_topology
from ..geometry._global_embedding import (
    GlobalEmbeddingCertificate,
    ValidatedGlobalEmbedding,
)
from ..geometry._nested_restriction_embedding import NestedRestrictionPremise
from ..geometry._planar_embedding import PlanarEmbedding
from ..geometry._source_fidelity import ValidatedSourceFidelityWitness
from ..geometry._surface_source_support import SurfaceSourceCharts
from ..geometry.implicit._chart_restriction import (
    ImplicitChartPremise,
    ValidatedImplicitChartParent,
)
from ..geometry.surface import SurfaceModel
from ._association import GeometryAssociation
from ._audit import (
    audit_cell_mesh,
    CellMeshAuditPolicy,
    CellMeshAuditReport,
    CellMeshManifoldScope,
)
from ._certification import _ComputedSurfaceAcceptance, MeshCertificationReport
from ._certification_inputs import (
    MeshCertificationInputs,
    validate_fidelity_witnesses,
)
from ._contracts import (
    CurveMeshingSpec,
    MeshingCapability,
    MeshingDerivativeMode,
    MeshingExecutionMode,
    MeshingFailure,
    MeshingFailureCategory,
    MeshingOperation,
    MeshingProviderInfo,
    MeshingSourceKind,
    SurfaceMeshingSpec,
    VolumeMeshingSpec,
)
from ._layer_core_carrier_witness import NativeLayerCoreCarrierWitness
from ._layer_core_root_geometry import ComputedLayerCoreRoot, LayerCoreRootGeometry
from ._layer_core_subdivision import OverlapSourceStencils, RefinedColumnCertification
from ._layer_curving_records import LayerCurvingSourceEvidence
from ._native_construction import NativeMeshConstruction, NativeMeshOrganization
from ._organization import (
    MeshAttribute,
    MeshLabel,
    MeshPatch,
    MeshZone,
    RegionBoundaryEvidence,
)
from ._quality import CellQualityReport
from ._result import CellMeshingResult, MeshingComplianceReport, MeshingRuntimeInfo
from ._trace import (
    MeshingDiagnostic,
    MeshingDiagnosticSeverity,
    MeshingStageKind,
    MeshingStageReport,
    MeshingStageStatus,
    MeshingTrace,
)


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
        canonical_lifted = rebuilt
    else:
        canonical_lifted = CellMesh(
            mesh.coordinates,
            ordered,
            vertex_global_ids=mesh.vertex_global_ids,
            entity_global_ids=entity_ids,
            numeric_version=mesh.numeric_version,
        )
    if mesh.periodic_topology is None:
        return canonical_lifted
    from ._periodic import _periodic_topology_after_reorder

    periodic_topology = _periodic_topology_after_reorder(mesh, canonical_lifted)
    return CellMesh(
        mesh.coordinates,
        ordered,
        vertex_global_ids=mesh.vertex_global_ids,
        entity_global_ids=entity_ids,
        numeric_version=mesh.numeric_version,
        periodic_topology=periodic_topology,
    )


@dataclass(frozen=True, slots=True)
class _CellMeshAcceptance:
    """Operation-local canonical acceptance before result publication."""

    mesh: CellMesh
    geometry: CellGeometrySpec
    coordinate_contract: SpatialCoordinateContract
    audit: CellMeshAuditReport
    certification: MeshCertificationReport | None
    boundary: SurfaceModel | None
    patches: tuple[MeshPatch, ...]
    zones: tuple[MeshZone, ...]
    labels: tuple[MeshLabel, ...]
    attributes: tuple[MeshAttribute, ...]
    associations: tuple[GeometryAssociation, ...]
    surface_source: SurfaceSourceCharts | None
    region_boundary_evidence: tuple[RegionBoundaryEvidence, ...]
    planar_embedding: PlanarEmbedding | None
    layer_curving_evidence: LayerCurvingSourceEvidence | None
    layer_core_carrier_witness: NativeLayerCoreCarrierWitness | None
    layer_core_overlap_stencils: OverlapSourceStencils | None
    layer_core_column_certification: RefinedColumnCertification | None
    layer_core_root_geometry: LayerCoreRootGeometry | None
    validated_witnesses: tuple[ValidatedSourceFidelityWitness, ...]

    @property
    def quality(self) -> CellQualityReport:
        return self.audit.quality

    def __reduce_ex__(self, protocol: SupportsIndex, /) -> NoReturn:
        raise TypeError("Computed cell acceptance cannot be serialized.")


def _certify_cell_mesh_acceptance(
    mesh: CellMesh,
    coordinate_contract: SpatialCoordinateContract,
    /,
    *,
    geometry: CellGeometrySpec | None = None,
    audit_policy: CellMeshAuditPolicy | None = None,
    boundary: SurfaceModel | None = None,
    certification_inputs: MeshCertificationInputs | None = None,
    patches: tuple[MeshPatch, ...] = (),
    zones: tuple[MeshZone, ...] = (),
    labels: tuple[MeshLabel, ...] = (),
    attributes: tuple[MeshAttribute, ...] = (),
    associations: tuple[GeometryAssociation, ...] = (),
    surface_source: SurfaceSourceCharts | None = None,
    region_boundary_evidence: tuple[RegionBoundaryEvidence, ...] = (),
    planar_embedding: PlanarEmbedding | None = None,
    layer_curving_evidence: LayerCurvingSourceEvidence | None = None,
    layer_core_carrier_witness: NativeLayerCoreCarrierWitness | None = None,
    layer_core_overlap_stencils: OverlapSourceStencils | None = None,
    layer_core_column_certification: RefinedColumnCertification | None = None,
    layer_core_root_geometry: LayerCoreRootGeometry | None = None,
    computed_root: ComputedLayerCoreRoot | None = None,
    _validated_embedding: ValidatedGlobalEmbedding | None = None,
    _validated_chart_parent_embedding: ValidatedGlobalEmbedding | None = None,
    embedding: GlobalEmbeddingCertificate | None = None,
    embedding_premise: NestedRestrictionPremise | None = None,
    fidelity_premise: ImplicitChartPremise | None = None,
    validated_chart_parent: ValidatedImplicitChartParent | None = None,
    manifold_scope: CellMeshManifoldScope | None = None,
    validated_witnesses: tuple[ValidatedSourceFidelityWitness, ...] = (),
    validity: CellValidityCertificate | None = None,
    validity_parent: CellValidityCertificate | ValidatedCellValidityParent
    | tuple[CellValidityCertificate | ValidatedCellValidityParent, ...]
    | None = None,
) -> _CellMeshAcceptance:
    """Compute canonical acceptance without publishing a meshing result."""

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
    if (
        layer_core_root_geometry is not None
        and certification_inputs is None
    ) or (
        certification_inputs is not None
        and certification_inputs.layer_core_root_geometry is not layer_core_root_geometry
    ):
        raise ValueError("Layer root geometry must match its certified source authority.")
    if computed_root is not None and (
        certification_inputs is None or layer_core_root_geometry is None
    ):
        raise ValueError("A computed root requires its certified layer source authority.")
    if (
        _validated_embedding is not None or _validated_chart_parent_embedding is not None
    ) and certification_inputs is None:
        raise ValueError("Validated embeddings require certified acceptance.")
    if (
        embedding is not None
        or embedding_premise is not None
        or fidelity_premise is not None
    ) and certification_inputs is None:
        raise ValueError("A prepared embedding is reused only by certified acceptance.")
    if validated_witnesses and certification_inputs is None:
        raise ValueError(
            "Validated fidelity witnesses are reused only by certified acceptance."
        )
    if (
        validity is not None or validity_parent is not None
    ) and certification_inputs is None:
        raise ValueError("Validity proofs are reused only by certified acceptance.")
    if manifold_scope is not None and certification_inputs is not None:
        raise ValueError("A manifold scope applies only to the direct cell-mesh audit.")
    certification = None
    validated: tuple[ValidatedSourceFidelityWitness, ...] = ()
    if certification_inputs is None:
        audit = audit_cell_mesh(
            canonical,
            geometry_,
            policy=audit_policy,
            boundary=boundary,
            patches=patches,
            associations=associations,
            attributes=attributes,
            zones=zones,
            labels=labels,
            manifold_scope=manifold_scope,
        )
    else:
        validated = certification_inputs.require_current(
            canonical,
            geometry_,
            domain=certification_inputs.domain,
            source=certification_inputs.source,
            validated=validated_witnesses,
        )
        certification = MeshCertificationReport(
            certification_inputs,
            audit_policy=audit_policy,
            organization=NativeMeshOrganization(
                boundary=boundary,
                patches=patches,
                zones=zones,
                labels=labels,
                attributes=attributes,
                associations=associations,
                surface_source=surface_source,
                region_boundary_evidence=region_boundary_evidence,
            ),
            embedding=embedding,
            embedding_premise=embedding_premise,
            fidelity_premise=fidelity_premise,
            validated_chart_parent=validated_chart_parent,
            validated=validated,
            validity=validity,
            validity_parent=validity_parent,
            computed_root=computed_root,
            validated_embedding=_validated_embedding,
            _validated_chart_parent_embedding=_validated_chart_parent_embedding,
        )
        certification.require_passed()
        audit = certification.audit
    if not audit.passed:
        uncertified = (
            np.asarray(audit.validity.status) != CellValidityStatus.CERTIFIED_VALID
        )
        cell_ids = np.concatenate(
            [np.asarray(block.global_ids, dtype=np.int64) for block in canonical.blocks]
        )
        raise MeshingFailure(
            MeshingFailureCategory.AUDIT_FAILED,
            "; ".join(audit.issues),
            stage=MeshingStageKind.GEOMETRY_AUDIT.value,
            entity_ids=(
                tuple(int(value) for value in cell_ids[uncertified])
                if np.any(uncertified)
                else audit.quality.worst_cell_global_ids
            ),
        )
    return _CellMeshAcceptance(
        canonical, geometry_, coordinate_contract, audit, certification, boundary,
        patches, zones, labels, attributes, associations, surface_source,
        region_boundary_evidence, planar_embedding, layer_curving_evidence,
        layer_core_carrier_witness, layer_core_overlap_stencils,
        layer_core_column_certification, layer_core_root_geometry, validated,
    )


def certify_cell_mesh(
    mesh: CellMesh,
    coordinate_contract: SpatialCoordinateContract,
    /,
    *,
    geometry: CellGeometrySpec | None = None,
    audit_policy: CellMeshAuditPolicy | None = None,
    boundary: SurfaceModel | None = None,
    certification_inputs: MeshCertificationInputs | None = None,
    patches: tuple[MeshPatch, ...] = (),
    zones: tuple[MeshZone, ...] = (),
    labels: tuple[MeshLabel, ...] = (),
    attributes: tuple[MeshAttribute, ...] = (),
    associations: tuple[GeometryAssociation, ...] = (),
    surface_source: SurfaceSourceCharts | None = None,
    region_boundary_evidence: tuple[RegionBoundaryEvidence, ...] = (),
    planar_embedding: PlanarEmbedding | None = None,
    layer_curving_evidence: LayerCurvingSourceEvidence | None = None,
    layer_core_carrier_witness: NativeLayerCoreCarrierWitness | None = None,
    layer_core_overlap_stencils: OverlapSourceStencils | None = None,
    layer_core_column_certification: RefinedColumnCertification | None = None,
    layer_core_root_geometry: LayerCoreRootGeometry | None = None,
    computed_root: ComputedLayerCoreRoot | None = None,
    _validated_embedding: ValidatedGlobalEmbedding | None = None,
    _validated_chart_parent_embedding: ValidatedGlobalEmbedding | None = None,
    embedding: GlobalEmbeddingCertificate | None = None,
    embedding_premise: NestedRestrictionPremise | None = None,
    fidelity_premise: ImplicitChartPremise | None = None,
    validated_chart_parent: ValidatedImplicitChartParent | None = None,
    manifold_scope: CellMeshManifoldScope | None = None,
    validated_witnesses: tuple[ValidatedSourceFidelityWitness, ...] = (),
    validity: CellValidityCertificate | None = None,
    validity_parent: CellValidityCertificate | ValidatedCellValidityParent
    | tuple[CellValidityCertificate | ValidatedCellValidityParent, ...]
    | None = None,
) -> CellMeshingResult:
    """Canonicalize and certify one existing CellMesh through native substrates.

    ``embedding`` is a prepared global-embedding certificate of the canonical
    mesh; acceptance with ``certification_inputs`` checks and reuses it.
    ``embedding_premise`` offers acceptance a certified source embedding and
    the exact nested transition to this mesh, from which the global embedding
    follows when the nested-restriction theorem's premises hold.
    ``fidelity_premise`` offers implicit-chart fidelity a certified parent
    certificate: rows equal to the parent's are reused by identity, and a
    nested restriction inherits the parent's bounds.
    ``validated_witnesses`` are fidelity-witness validations the caller made
    for exactly these mesh, coordinate map and source objects; acceptance with
    ``certification_inputs`` does not repeat them.
    ``validity`` (an in-hand proof of exactly these inputs) or
    ``validity_parent`` (earlier proofs whose identical cell maps are reused)
    serve acceptance with ``certification_inputs``; see
    ``MeshCertificationReport``.
    ``computed_root`` is an operation-local proof bound to the exact original
    layer source, request and root objects; it is never retained in the result.
    ``manifold_scope`` declares the vertex stars a resident patch of a
    distributed mesh certifies; it applies to the direct audit only.
    """

    accepted = _certify_cell_mesh_acceptance(
        mesh, coordinate_contract, geometry=geometry, audit_policy=audit_policy,
        boundary=boundary, certification_inputs=certification_inputs, patches=patches,
        zones=zones, labels=labels, attributes=attributes, associations=associations,
        surface_source=surface_source, region_boundary_evidence=region_boundary_evidence,
        planar_embedding=planar_embedding, layer_curving_evidence=layer_curving_evidence,
        layer_core_carrier_witness=layer_core_carrier_witness,
        layer_core_overlap_stencils=layer_core_overlap_stencils,
        layer_core_column_certification=layer_core_column_certification,
        layer_core_root_geometry=layer_core_root_geometry, computed_root=computed_root,
        _validated_embedding=_validated_embedding,
        _validated_chart_parent_embedding=_validated_chart_parent_embedding,
        embedding=embedding, embedding_premise=embedding_premise,
        fidelity_premise=fidelity_premise, validated_chart_parent=validated_chart_parent,
        manifold_scope=manifold_scope, validated_witnesses=validated_witnesses,
        validity=validity, validity_parent=validity_parent,
    )
    canonical, geometry_, audit, certification, validated = (
        accepted.mesh, accepted.geometry, accepted.audit,
        accepted.certification, accepted.validated_witnesses,
    )
    findings = audit.recorded
    audit_status = MeshingStageStatus.WARNING if findings else MeshingStageStatus.PASSED
    audit_diagnostics = (
        (
            MeshingDiagnostic(
                MeshingDiagnosticSeverity.WARNING,
                "Accepted audit findings: " + "; ".join(findings),
            ),
        )
        if findings
        else ()
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
            MeshingStageKind.GEOMETRY_AUDIT,
            audit_status,
            input_ids=(geometry_.geometry_layout_id,),
            output_ids=(audit.validity.certificate_id,),
            diagnostics=audit_diagnostics,
        ),
        MeshingStageReport(
            MeshingStageKind.TOPOLOGY_AUDIT,
            audit_status,
            input_ids=(canonical.topology_id,),
            output_ids=(audit.report_id,),
            diagnostics=audit_diagnostics,
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
        boundary=boundary,
        certification=certification,
        patches=patches,
        zones=zones,
        labels=labels,
        attributes=attributes,
        associations=associations,
        surface_source=surface_source,
        region_boundary_evidence=region_boundary_evidence,
        planar_embedding=planar_embedding,
        layer_curving_evidence=layer_curving_evidence,
        layer_core_carrier_witness=layer_core_carrier_witness,
        layer_core_overlap_stencils=layer_core_overlap_stencils,
        layer_core_column_certification=layer_core_column_certification,
        layer_core_root_geometry=layer_core_root_geometry,
        computed_root=computed_root,
        validated_witnesses=validated,
    )


def _require_native_target(
    construction: NativeMeshConstruction,
    specification: CurveMeshingSpec | SurfaceMeshingSpec | VolumeMeshingSpec,
    /,
) -> None:
    mesh, target = construction.mesh, specification.target
    issues: list[str] = []
    if (mesh.topological_dimension, mesh.ambient_dimension) != (
        target.topological_dimension,
        target.ambient_dimension,
    ):
        issues.append("Constructed mesh dimensions differ from the requested target.")
    kinds = {
        block.cell_kind
        if block.cell_kind in ("polygon", "polyhedron")
        else reference_cell_topology(block.cell_kind).family
        for block in mesh.blocks
        if block.cell_count
    }
    families = target.cell_families
    allowed = set(
        (*families.required, *families.preferred, *families.allowed_transitions)
    )
    if not kinds or not set(families.required) <= kinds or not kinds <= allowed:
        issues.append(
            "Constructed cell families do not satisfy the requested family policy."
        )
    if not families.allow_mixed and len(kinds) != 1:
        issues.append("The requested target does not permit mixed cell families.")
    elements, _, _ = construction.geometry.resolve(mesh)
    for element in elements:
        if isinstance(element, CellVertexGeometryElement):
            degree = 1
        elif isinstance(element, ScalarCoordinateElement):
            degree = element.degree
        else:
            raise TypeError(
                "Native target geometry must use a canonical coordinate element."
            )
        if degree != target.geometry_order:
            issues.append(
                "Constructed coordinate geometry order differs from the requested target."
            )
            break
    if construction.compliance.specification_id != specification.specification_id:
        raise ValueError(
            "Native compliance must bind the actual requested specification."
        )
    issues.extend(construction.compliance.issues)
    if issues:
        raise MeshingFailure(
            MeshingFailureCategory.COMPLIANCE_FAILED,
            "; ".join(issues),
            stage=MeshingStageKind.SPECIFICATION_COMPLIANCE.value,
        )


def _native_acceptance_trace(
    construction: NativeMeshConstruction, certification: MeshCertificationReport, /
) -> MeshingTrace:
    diagnostics = tuple(
        MeshingDiagnostic(
            MeshingDiagnosticSeverity.WARNING,
            f"{finding.check}: {finding.status}",
            entity_ids=finding.entity_ids,
        )
        for finding in certification.recorded
    )
    status = MeshingStageStatus.WARNING if diagnostics else MeshingStageStatus.PASSED
    return MeshingTrace(
        (
            *construction.stages,
            MeshingStageReport(
                MeshingStageKind.GEOMETRY_AUDIT,
                status,
                input_ids=(certification.inputs.request_id,),
                output_ids=(certification.report_id,),
                diagnostics=diagnostics,
            ),
            MeshingStageReport(
                MeshingStageKind.SPECIFICATION_COMPLIANCE,
                MeshingStageStatus.PASSED,
                input_ids=(construction.compliance.specification_id,),
                output_ids=(construction.compliance.report_id,),
            ),
        )
    )


def _accept_native_construction(
    construction: NativeMeshConstruction,
    specification: CurveMeshingSpec | SurfaceMeshingSpec | VolumeMeshingSpec,
    provider: MeshingProviderInfo,
    actual_version: str,
    plan_id: str,
    route: str,
    /,
    *,
    computed_surface_out: list[_ComputedSurfaceAcceptance] | None = None,
) -> CellMeshingResult:
    """Publish only a candidate satisfying its actual source and target obligations."""
    _require_native_target(construction, specification)
    event = construction.surface_quad_field_event
    requested_field = (
        specification.cross_field if isinstance(specification, SurfaceMeshingSpec) else None
    )
    if event is not None:
        from ..geometry._mesh_certificates import _same_tree_content

        if (
            route != "planar_dual_quad"
            or construction.source_mesh is None
            or construction.source_geometry is None
            or not _same_tree_content(event.source, construction.source_mesh)
            or not _same_tree_content(event.source_geometry, construction.source_geometry)
        ):
            raise ValueError("Quad field event requires its actual supported source route.")
        if requested_field is not None and (
            event.field.field_id != requested_field.field_id
            or not _same_tree_content(event.field, requested_field)
        ):
            raise ValueError("Quad field event differs from the requested source field.")
    elif requested_field is not None:
        raise ValueError("A requested cross field requires its realized quad field event.")
    if (
        construction.layer_core_carrier_witness is not None
        and construction.layer_core_carrier_witness.specification_id
        != specification.specification_id
    ):
        raise ValueError("Native carrier inventory does not bind the accepted request.")
    if construction.layer_core_overlap_stencils is not None:
        if (
            construction.layer_core_carrier_witness is None
            or construction.layer_core_overlap_stencils.source_binding_id
            != construction.layer_core_carrier_witness.source_binding_id
            or construction.layer_core_overlap_stencils.target_mesh_id
            != construction.mesh.mesh_id
        ):
            raise ValueError("Overlap source stencils do not bind the accepted layer/core construction.")
    if construction.layer_core_column_certification is not None and (
        construction.layer_core_carrier_witness is None
        or construction.layer_core_column_certification.source_binding_id
        != construction.layer_core_carrier_witness.source_binding_id
        or construction.layer_core_column_certification.specification_id
        != specification.specification_id
    ):
        raise ValueError("Layer column certification does not bind the accepted source and request.")
    if construction.layer_core_root_geometry is not None:
        root = construction.layer_core_root_geometry
        mapped = root.reference_mesh is not None
        if (
            construction.layer_core_carrier_witness is None
            or root.source_binding_id != construction.layer_core_carrier_witness.source_binding_id
            or root.specification_id != specification.specification_id
            or root.mesh.mesh_id != construction.mesh.mesh_id
            or root.geometry.geometry_layout_id != construction.geometry.geometry_layout_id
        ):
            raise ValueError("Layer root geometry does not bind the accepted construction.")
        if mapped:
            if (
                construction.layer_core_overlap_stencils is not None
                or construction.layer_core_column_certification is not None
            ):
                raise ValueError("Mapped physical results cannot relabel reference layer evidence.")
        elif (
            construction.layer_core_overlap_stencils is None
            or construction.layer_core_column_certification is None
            or root.overlap_stencils.witness_id != construction.layer_core_overlap_stencils.witness_id
            or root.column_certification.certification_id != construction.layer_core_column_certification.certification_id
        ):
            raise ValueError("Layer root geometry does not bind the accepted construction.")
    mesh, geometry = construction.mesh, construction.geometry
    if canonicalize_cell_mesh(mesh) is not mesh:
        raise MeshingFailure(
            MeshingFailureCategory.CANONICALIZATION_FAILED,
            "Native construction must canonicalize topology before binding coordinate routes.",
            stage=MeshingStageKind.CANONICALIZATION.value,
        )
    request = construction.certification
    # Validate each fidelity witness once and hand the validations on to the
    # inputs, their currency check and the acceptance proofs.
    validated = validate_fidelity_witnesses(
        mesh,
        geometry,
        request.fidelity_source,
        request.fidelity_witness,
        request.scoped_fidelity,
        validated=construction.validated_fidelity,
    )
    inputs = MeshCertificationInputs(
        mesh,
        geometry,
        request.schedule,
        source_id=request.source_id,
        source_revision=request.source_revision,
        domain=request.domain,
        cell_regions=request.cell_regions,
        source=request.fidelity_source,
        fidelity_tolerance=request.fidelity_tolerance,
        maximum_normal_deviation=request.maximum_normal_deviation,
        junction_vertices=request.junction_vertices,
        protected_features=request.protected_features,
        feature_source=request.feature_source,
        layer_core_root_geometry=construction.layer_core_root_geometry,
        computed_root=construction.computed_layer_core_root,
        scoped_fidelity=request.scoped_fidelity,
        fidelity_witness=request.fidelity_witness,
        source_coverage_witness=request.source_coverage_witness,
        validated=validated,
    )
    organization = construction.organization
    validated = inputs.require_current(
        mesh, geometry, domain=inputs.domain, source=inputs.source, validated=validated
    )
    computed_surface = None
    if computed_surface_out is None:
        certification = MeshCertificationReport(
            inputs,
            organization=organization,
            validity_parent=request.validity_parent,
            computed_validity=construction.computed_validity,
            validated_embedding=construction.validated_global_embedding,
            computed_root=construction.computed_layer_core_root,
            validated=validated,
        )
        certification.require_passed()
    else:
        computed_surface = _ComputedSurfaceAcceptance(
            inputs,
            organization,
            validity_parent=request.validity_parent,
            validated=validated,
            computed_validity=construction.computed_validity,
        )
        certification = computed_surface.report
    runtime = MeshingRuntimeInfo(
        provider.provider_id,
        actual_version,
        MeshingExecutionMode.IN_PROCESS,
        deterministic=True,
    )
    provenance = SemanticProvenance(
        {
            "kind": "native-mesh-construction",
            "plan": plan_id,
            "route": route,
            "source": (request.source_id, request.source_revision),
            "specification": specification.specification_id,
            "certification": certification.report_id,
            "surface_quad_field_event": None if event is None else event.event_id,
        }
    )
    result = CellMeshingResult(
        mesh,
        geometry,
        construction.coordinate_contract,
        certification.audit,
        certification.audit.quality,
        construction.compliance,
        _native_acceptance_trace(construction, certification),
        provider,
        runtime,
        MeshingDerivativeMode.NONDIFFERENTIABLE,
        provenance,
        boundary=organization.boundary,
        patches=organization.patches,
        zones=organization.zones,
        labels=organization.labels,
        attributes=organization.attributes,
        associations=organization.associations,
        surface_source=organization.surface_source,
        region_boundary_evidence=organization.region_boundary_evidence,
        planar_embedding=construction.planar_embedding,
        surface_quad_field_event=event,
        certification=certification,
        layer_curving_evidence=construction.layer_curving_evidence,
        layer_core_carrier_witness=construction.layer_core_carrier_witness,
        layer_core_overlap_stencils=construction.layer_core_overlap_stencils,
        layer_core_column_certification=construction.layer_core_column_certification,
        layer_core_root_geometry=construction.layer_core_root_geometry,
        computed_root=construction.computed_layer_core_root,
        validated_witnesses=validated,
    )
    if computed_surface_out is not None:
        assert computed_surface is not None
        computed_surface_out.append(computed_surface)
    return result


__all__ = ["canonicalize_cell_mesh", "certify_cell_mesh"]
