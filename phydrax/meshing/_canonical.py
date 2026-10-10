#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import numpy as np
from jax.typing import ArrayLike

from .._identity import SemanticProvenance
from .._physical import SpatialCoordinateContract
from ..discretization import (
    CellBlock,
    CellGeometrySpec,
    CellMesh,
    PeriodicMeshTopology,
    PolyhedralBlock,
    PolyhedralConnectivity,
)
from ..discretization._cell_complex import PolygonalConnectivity, TetrahedralConnectivity
from ..discretization._cell_geometry_validity import (
    CellValidityCertificate,
    CellValidityStatus,
)
from ..discretization._exact_power_geometry import (
    ExactPowerCellGeometryLinearActionSource,
    ExactPowerCellGeometryRestrictionSource,
    ExactPowerCellGeometrySource,
)
from ..discretization._hexahedral import HexahedralConnectivity
from ..geometry._mesh_certificates import (
    DomainCoverageCertificate,
    GlobalEmbeddingCertificate,
    SourceFidelityCertificate,
)
from ..geometry._surface_source_support import SurfaceSourceCharts
from ..geometry.surface import SurfaceModel
from ._association import GeometryAssociation
from ._audit import audit_cell_mesh, CellMeshAuditPolicy
from ._certification import acceptance_stage_report, MeshCertificationPreparedEvidence
from ._certification_inputs import MeshCertificationInputs
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
from ._lineage import MeshLineage
from ._organization import (
    MeshAttribute,
    MeshAttributeProjection,
    MeshLabel,
    MeshPatch,
    MeshZone,
    RegionBoundaryEvidence,
    RegionMeshingEvidence,
)
from ._result import (
    CellMeshingResult,
    CollectiveMeshEvidence,
    CollectiveMeshStorageBinding,
    MeshingComplianceReport,
    MeshingRuntimeInfo,
    require_original_meshing_source,
)
from ._scope import MeshScopeProjection
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
    """Identify lifted lower-dimensional entities independently of traversal ordering.

    Lifted vertices keep distinct global IDs even when they share a periodic
    quotient representative, so these keys never merge distinct winding
    entities; quotient identity uses the relative-shift keys of
    `_quotient_entity_ids`.
    """
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


def _identifiers_by_key(
    keys: tuple[tuple[int, ...], ...], identifiers: np.ndarray, /
) -> dict[tuple[int, ...], int]:
    mapping = {
        key: int(identifier) for key, identifier in zip(keys, identifiers, strict=True)
    }
    if len(mapping) != len(keys):
        raise ValueError("Canonical entity keys must identify distinct entities.")
    return mapping


def _quotient_entity_ids(
    source: PeriodicMeshTopology, target: PeriodicMeshTopology, /
) -> dict[int, np.ndarray]:
    """Carry intermediate quotient IDs to a rebuilt descriptor by winding keys."""

    entity_ids = {}
    for degree in range(1, source.topological_dimension):
        identifiers = _identifiers_by_key(
            source.entity_keys(degree),
            np.asarray(source.quotient.entities(degree).entity_ids),
        )
        entity_ids[degree] = np.asarray(
            [identifiers[key] for key in target.entity_keys(degree)], dtype=np.int64
        )
    return entity_ids


def _periodic_rebuild(
    mesh: CellMesh, lifted: CellMesh, entity_ids: dict[int, np.ndarray], /
) -> CellMesh:
    """Rebind the periodic descriptor of `mesh` to its reordered lifted carrier."""

    periodic = mesh.periodic_topology
    if periodic is None:
        return lifted
    actual_geometry = periodic.actual_geometry
    if actual_geometry is not None:
        source = actual_geometry.exact_source
        if not isinstance(
            source,
            (
                ExactPowerCellGeometrySource,
                ExactPowerCellGeometryRestrictionSource,
                ExactPowerCellGeometryLinearActionSource,
            ),
        ):
            raise TypeError(
                "Periodic canonicalization requires the retained exact vertex source owner."
            )
        actual_geometry = CellGeometrySpec.power(lifted, source)
    arguments = (
        lifted,
        periodic.cell,
        periodic.vertex_representatives,
        periodic.vertex_shifts,
    )
    candidate = PeriodicMeshTopology(*arguments, actual_geometry=actual_geometry)
    quotient_ids = _quotient_entity_ids(periodic, candidate)
    if candidate.allocator_next_ids != periodic.allocator_next_ids or not all(
        np.array_equal(values, np.asarray(candidate.quotient.entities(degree).entity_ids))
        for degree, values in quotient_ids.items()
    ):
        # Unresolved (-1) cursors stay unresolved: explicit IDs without history.
        candidate = PeriodicMeshTopology(
            *arguments,
            actual_geometry=actual_geometry,
            entity_global_ids=quotient_ids,
            entity_allocator_next_ids={
                degree: cursor
                for degree, cursor in enumerate(periodic.allocator_next_ids)
                if 0 < degree < periodic.topological_dimension and cursor >= 0
            },
        )
    return CellMesh(
        lifted.coordinates,
        lifted.blocks,
        vertex_global_ids=lifted.vertex_global_ids,
        entity_global_ids=entity_ids,
        periodic_topology=candidate,
        numeric_version=lifted.numeric_version,
    )


def canonicalize_cell_mesh(mesh: CellMesh, /) -> CellMesh:
    """Return deterministic block/cell ordering without changing mesh geometry."""

    if not isinstance(mesh, CellMesh):
        raise TypeError("mesh must be CellMesh.")
    if mesh.storage is not None:
        # These are prepared local slots of an already canonical logical mesh.
        # Reordering them as a complete serial mesh would destroy routing.
        return mesh
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
        identifiers = _identifiers_by_key(
            _entity_vertex_keys(mesh, dimension),
            np.asarray(mesh.entity_set(dimension).entity_ids),
        )
        entity_ids[dimension] = np.asarray(
            [identifiers[key] for key in _entity_vertex_keys(rebuilt, dimension)],
            dtype=np.int64,
        )
    if not all(
        np.array_equal(values, np.asarray(rebuilt.entity_set(dimension).entity_ids))
        for dimension, values in entity_ids.items()
    ):
        rebuilt = CellMesh(
            mesh.coordinates,
            ordered,
            vertex_global_ids=mesh.vertex_global_ids,
            entity_global_ids=entity_ids,
            numeric_version=mesh.numeric_version,
        )
    return _periodic_rebuild(mesh, rebuilt, entity_ids)


def certify_cell_mesh(
    mesh: CellMesh,
    coordinate_contract: SpatialCoordinateContract,
    /,
    *,
    geometry: CellGeometrySpec | None = None,
    audit_policy: CellMeshAuditPolicy | None = None,
    boundary: SurfaceModel | None = None,
    patches: tuple[MeshPatch, ...] = (),
    zones: tuple[MeshZone, ...] = (),
    labels: tuple[MeshLabel, ...] = (),
    attributes: tuple[MeshAttribute, ...] = (),
    associations: tuple[GeometryAssociation, ...] = (),
    region_evidence: RegionMeshingEvidence | None = None,
    region_boundary_evidence: tuple[RegionBoundaryEvidence, ...] = (),
    certification_inputs: MeshCertificationInputs | None = None,
    lineage: MeshLineage | None = None,
    certification_cell_regions: ArrayLike | None = None,
    certification_junction_vertices: ArrayLike | None = None,
    certification_prepared: MeshCertificationPreparedEvidence | None = None,
    certification_prepared_embedding: GlobalEmbeddingCertificate | None = None,
    certification_prepared_fidelity: SourceFidelityCertificate | None = None,
    audit_prepared_validity: CellValidityCertificate | None = None,
    surface_source: SurfaceSourceCharts | None = None,
) -> CellMeshingResult:
    """Canonicalize and certify one existing CellMesh through native substrates."""

    mesh.require_dense("certify_cell_mesh")
    if not isinstance(coordinate_contract, SpatialCoordinateContract):
        raise TypeError("coordinate_contract must be SpatialCoordinateContract.")
    if boundary is not None:
        if not isinstance(boundary, SurfaceModel):
            raise TypeError("boundary must be SurfaceModel or None.")
        if (
            boundary.metadata.coordinate_contract.spatial_id
            != coordinate_contract.spatial_id
        ):
            raise ValueError(
                "Boundary and target carrier require the exact same physical coordinate contract."
            )
    if certification_inputs is not None and not isinstance(
        certification_inputs, MeshCertificationInputs
    ):
        raise TypeError("certification_inputs must be MeshCertificationInputs or None.")
    if lineage is not None and not isinstance(lineage, MeshLineage):
        raise TypeError("lineage must be MeshLineage or None.")
    if (certification_inputs is None) != (lineage is None):
        raise ValueError(
            "Certification transition inputs and lineage must be supplied together."
        )
    if certification_prepared is not None and (
        not isinstance(certification_prepared, MeshCertificationPreparedEvidence)
        or certification_inputs is None
    ):
        raise ValueError(
            "Prepared target evidence requires its owning transition request."
        )
    if certification_prepared_embedding is not None and (
        not isinstance(certification_prepared_embedding, GlobalEmbeddingCertificate)
        or certification_inputs is None
    ):
        raise ValueError(
            "Prepared target embedding requires its owning transition request."
        )
    if certification_prepared_fidelity is not None and (
        not isinstance(certification_prepared_fidelity, SourceFidelityCertificate)
        or certification_inputs is None
    ):
        raise ValueError(
            "Prepared target fidelity requires its owning transition request."
        )
    if audit_prepared_validity is not None and not isinstance(
        audit_prepared_validity, CellValidityCertificate
    ):
        raise TypeError(
            "audit_prepared_validity must be CellValidityCertificate or None."
        )
    if (
        certification_prepared is not None
        and audit_prepared_validity is not None
        and certification_prepared.validity.certificate_id
        != audit_prepared_validity.certificate_id
    ):
        raise ValueError("Audit and certification prepared validity evidence must agree.")
    if certification_inputs is None and (
        certification_cell_regions is not None
        or certification_junction_vertices is not None
    ):
        raise ValueError("Target certification assignments require transition inputs.")
    canonical = canonicalize_cell_mesh(mesh)
    if region_evidence is not None and canonical is not mesh:
        raise ValueError(
            "Canonicalization would reorder supplied region evidence; canonicalize "
            "the mesh before constructing RegionMeshingEvidence."
        )
    if geometry is not None and canonical is not mesh:
        raise ValueError(
            "Canonicalization would reorder supplied geometry DOF rows; canonicalize "
            "the mesh before constructing CellGeometrySpec."
        )
    geometry_ = CellGeometrySpec.affine(canonical) if geometry is None else geometry
    if not isinstance(geometry_, CellGeometrySpec):
        raise TypeError("geometry must be CellGeometrySpec or None.")
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
        prepared_validity=(
            certification_prepared.validity
            if certification_prepared is not None
            else audit_prepared_validity
        ),
    )
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
    certification = None
    certification_stages: tuple[MeshingStageReport, ...] = ()
    if certification_inputs is not None and lineage is not None:
        certification = certification_inputs.recertify_transition(
            canonical,
            geometry_,
            audit,
            lineage=lineage,
            cell_regions=certification_cell_regions,
            junction_vertices=certification_junction_vertices,
            prepared=certification_prepared,
            prepared_embedding=certification_prepared_embedding,
            prepared_fidelity=certification_prepared_fidelity,
        )
        certification.require_passed()
        certification_stages = (acceptance_stage_report(certification),)
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
        *certification_stages,
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
            **({"boundary_model": boundary.model_id} if boundary is not None else {}),
            **(
                {"certification": certification.report_id}
                if certification is not None
                else {}
            ),
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
        boundary=boundary,
        patches=patches,
        zones=zones,
        labels=labels,
        attributes=attributes,
        associations=associations,
        region_evidence=region_evidence,
        region_boundary_evidence=region_boundary_evidence,
        certification=certification,
        surface_source=surface_source,
    )


def certify_owner_local_cell_mesh(
    mesh: CellMesh,
    source: CellMeshingResult,
    collective_evidence: CollectiveMeshEvidence,
    /,
    *,
    audit_policy: CellMeshAuditPolicy | None = None,
    geometry: CellGeometrySpec | None = None,
    patches: tuple[MeshPatch, ...] = (),
    zones: tuple[MeshZone, ...] = (),
    labels: tuple[MeshLabel, ...] = (),
    attributes: tuple[MeshAttribute, ...] = (),
    associations: tuple[GeometryAssociation, ...] = (),
    region_evidence: RegionMeshingEvidence | None = None,
    region_boundary_evidence: tuple[RegionBoundaryEvidence, ...] = (),
    scope_projections: tuple[tuple[str, MeshScopeProjection], ...] | None = None,
    attribute_projections: tuple[tuple[str, MeshAttributeProjection], ...] | None = None,
    storage_binding: CollectiveMeshStorageBinding | None = None,
    collective_certificates: tuple[GlobalEmbeddingCertificate, DomainCoverageCertificate]
    | None = None,
) -> CellMeshingResult:
    """Publish a consumed affine-subdivision proof and audited local closure.

    The local audit remains explicitly local. Global embedding/domain meaning
    comes from the accepted source plus the distributed subdivision witnesses;
    no copied source certificate is relabeled as a target global certificate.
    ``collective_certificates`` are the target-bound embedding and coverage
    recertified collectively for this exact publication.
    """
    if not isinstance(mesh, CellMesh) or mesh.storage is None:
        raise ValueError(
            "Owner-local certification requires canonical owner-local storage."
        )
    uniform = (
        collective_evidence.uniform_refinement
        if isinstance(collective_evidence, CollectiveMeshEvidence)
        else None
    )
    original = require_original_meshing_source(
        source if uniform is None else uniform.source
    )
    from ._initial_certification import InitialCollectiveMeshEvidence

    if isinstance(original, InitialCollectiveMeshEvidence):
        original.require_passed()
        source_audit_id = original.evidence_id
        scientific_theorem_id = original.source_evidence_id
    else:
        certification = original.certification
        if certification is None or certification.embedding is None:
            raise ValueError(
                "Owner-local publication requires the original scientific embedding theorem."
            )
        certification.require_passed()
        source_audit_id = original.audit.report_id
        scientific_theorem_id = certification.report_id
    if (
        not isinstance(collective_evidence, CollectiveMeshEvidence)
        or collective_evidence.source_mesh_id != source.mesh.mesh_id
        or collective_evidence.source_audit_id != source_audit_id
        or collective_evidence.mesh_id != mesh.mesh_id
        or collective_evidence.evidence_id != mesh.storage.evidence_id
    ):
        raise ValueError(
            "Collective subdivision witnesses do not join the source and target."
        )
    collective_evidence.require_passed()
    obligations = (
        (source.patches, patches, "patch"),
        (source.zones, zones, "zone"),
        (source.labels, labels, "label"),
        (source.attributes, attributes, "attribute"),
        (source.associations, associations, "geometry association"),
        (
            source.region_boundary_evidence,
            region_boundary_evidence,
            "source region boundary evidence",
        ),
    )
    for before, after, name in obligations:
        if before and not after:
            raise ValueError(
                f"Owner-local publication requires the inherited {name} coverage, not silent removal."
            )
    if source.region_evidence is not None and region_evidence is None:
        raise ValueError(
            "Owner-local publication requires inherited authoritative region evidence."
        )
    geometry = CellGeometrySpec.affine(mesh) if geometry is None else geometry
    audit = audit_cell_mesh(
        mesh,
        geometry,
        policy=audit_policy,
        patches=patches,
        zones=zones,
        labels=labels,
        attributes=attributes,
        associations=associations,
    )
    audit.require_passed()
    audit.require_decided()
    counts = collective_evidence.global_entity_counts
    limits = collective_evidence.preparation.policy.limits
    observations = (
        ("vertices", counts[0], limits.maximum_vertices),
        ("cells", counts[-1], limits.maximum_cells),
        (
            "connectivity_entries",
            counts[-1] * len(counts),
            limits.maximum_connectivity_entries,
        ),
    )
    compliance = MeshingComplianceReport(
        source.compliance.specification_id,
        issues=tuple(
            f"maximum_{name}"
            for name, actual, maximum in observations
            if actual > maximum
        ),
        requested=tuple(
            (f"maximum_{name}", float(maximum)) for name, _, maximum in observations
        ),
        achieved=tuple((name, float(actual)) for name, actual, _ in observations),
    )
    trace = MeshingTrace(
        (
            MeshingStageReport(
                MeshingStageKind.SOURCE_INSPECTION,
                MeshingStageStatus.PASSED,
                input_ids=(source.result_id,),
                output_ids=(
                    scientific_theorem_id,
                    collective_evidence.source_evidence_id,
                ),
            ),
            MeshingStageReport(
                MeshingStageKind.GEOMETRY_AUDIT,
                MeshingStageStatus.PASSED,
                input_ids=(mesh.mesh_id, mesh.storage.storage_id),
                output_ids=(audit.validity.certificate_id,),
            ),
            MeshingStageReport(
                MeshingStageKind.TOPOLOGY_AUDIT,
                MeshingStageStatus.PASSED,
                input_ids=(source.mesh.topology_id, mesh.topology_id),
                output_ids=(audit.report_id, collective_evidence.evidence_id),
            ),
        )
    )
    runtime = MeshingRuntimeInfo(
        source.provider.provider_id,
        source.runtime.actual_version,
        MeshingExecutionMode.IN_PROCESS,
        deterministic=True,
        enforced_limits=("local_closure", "subdivision_witness", "collective_acceptance"),
        unenforced_limits=source.runtime.unenforced_limits,
    )
    provenance = SemanticProvenance(
        {
            "kind": "owner-local-affine-bisection-publication",
            "source": source.result_id,
            "mesh": mesh.mesh_id,
            "local_storage": mesh.storage.storage_id,
            "local_audit": audit.report_id,
            "collective_evidence": collective_evidence.evidence_id,
            "collective_certificates": None
            if collective_certificates is None
            else [certificate.certificate_id for certificate in collective_certificates],
        }
    )
    return CellMeshingResult(
        mesh,
        geometry,
        source.coordinate_contract,
        audit,
        audit.quality,
        compliance,
        trace,
        source.provider,
        runtime,
        MeshingDerivativeMode.FIXED_TOPOLOGY_EXACT,
        provenance,
        patches=patches,
        zones=zones,
        labels=labels,
        attributes=attributes,
        associations=associations,
        region_evidence=region_evidence,
        region_boundary_evidence=region_boundary_evidence,
        collective_evidence=collective_evidence,
        scope_projections=scope_projections,
        attribute_projections=attribute_projections,
        storage_binding=storage_binding,
        collective_certificates=collective_certificates,
    )


__all__ = [
    "canonicalize_cell_mesh",
    "certify_cell_mesh",
    "certify_owner_local_cell_mesh",
]
