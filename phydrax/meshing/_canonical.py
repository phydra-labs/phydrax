#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import assert_never, final

import equinox as eqx
import numpy as np
from jax.typing import ArrayLike

import phydrax.discretization._cell_complex as cell_complex_owner
import phydrax.discretization._cell_geometry as cell_geometry_owner
import phydrax.discretization._cell_mesh as cell_mesh_owner
import phydrax.discretization._periodic_topology as periodic_topology_owner
import phydrax.discretization._reference_cell_types as reference_cell_types_owner
from phydrax.meshing._association_state import GeometryAssociation
from phydrax.meshing._contracts import (
    MeshingCapability,
    MeshingDerivativeMode,
    MeshingExecutionMode,
    MeshingOperation,
    MeshingProviderInfo,
    MeshingSourceKind,
)
from phydrax.meshing._failures import MeshingFailure, MeshingFailureCategory

from .._identity import SemanticProvenance
from .._physical import SpatialCoordinateContract
from .._strict import StrictModule
from ..discretization._cell_complex import PolygonalConnectivity, TetrahedralConnectivity
from ..discretization._cell_geometry_validity import (
    CellValidityCertificate,
    CellValidityStatus,
)
from ..discretization._coordinate_budget import CoordinateEnclosureBudget
from ..discretization._exact_power_geometry import (
    ExactPowerCellGeometryLinearActionSource,
    ExactPowerCellGeometryRestrictionSource,
    ExactPowerCellGeometrySource,
)
from ..discretization._hexahedral import HexahedralConnectivity
from ..geometry._domain_coverage import DomainCoverageCertificate
from ..geometry._global_embedding import GlobalEmbeddingCertificate
from ..geometry._surface_source_support import SurfaceSourceCharts
from ..geometry.surface import SurfaceModel
from ..typing import checked
from ._audit import (
    audit_cell_mesh,
    CellMeshAuditPolicy,
    CellMeshAuditReport,
    required_audit_policy,
)
from ._certification import (
    acceptance_stage_report,
    MeshCertificationReport,
    PreparedBoundaryAcceptance,
    PreparedMeshingAcceptance,
    PreparedVolumeAcceptance,
)
from ._certification_inputs import MeshCertificationInputs
from ._lineage_records import MeshLineage
from ._organization_records import (
    MeshAttribute,
    MeshAttributeProjection,
    MeshLabel,
    MeshPatch,
    MeshPatchBank,
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


@checked
def entity_vertex_keys(
    mesh: cell_mesh_owner.CellMesh, dimension: int, /
) -> tuple[tuple[int, ...], ...]:
    """Identify lifted lower-dimensional entities independently of traversal ordering.

    Lifted vertices keep distinct global IDs even when they share a periodic
    quotient representative, so these keys never merge distinct winding
    entities; quotient identity uses the relative-shift keys of
    `_quotient_entity_ids`.
    """
    connectivity = mesh.connectivity
    match (dimension, connectivity):
        case (
            1,
            PolygonalConnectivity()
            | TetrahedralConnectivity()
            | HexahedralConnectivity()
            | cell_complex_owner.PolyhedralConnectivity() as connectivity,
        ):
            rows = np.asarray(connectivity.edges)
        case (1, _):
            raise TypeError(
                "mesh.connectivity must be polygonal or volumetric for edge keys; "
                f"got {type(connectivity).__name__}"
            )
        case (2, cell_complex_owner.PolyhedralConnectivity() as connectivity):
            offsets = np.asarray(connectivity.face_vertex_offsets)
            values = np.asarray(connectivity.face_vertex_values)
            rows = tuple(
                values[start:stop]
                for start, stop in zip(offsets[:-1], offsets[1:], strict=True)
            )
        case (2, TetrahedralConnectivity() | HexahedralConnectivity() as connectivity):
            rows = np.asarray(connectivity.faces)
        case (2, _):
            raise TypeError(
                "mesh.connectivity must be volumetric for face keys; "
                f"got {type(connectivity).__name__}"
            )
        case _:
            raise ValueError(
                f"entity_vertex_keys.dimension must be 1 or 2; got {dimension!r}"
            )
    vertex_ids = np.asarray(mesh.vertex_global_ids)
    return tuple(tuple(sorted(int(value) for value in vertex_ids[row])) for row in rows)


def _identifiers_by_key(
    keys: tuple[tuple[int, ...], ...], identifiers: np.ndarray, /
) -> dict[tuple[int, ...], int]:
    mapping = {
        key: int(identifier) for key, identifier in zip(keys, identifiers, strict=True)
    }
    if len(mapping) != len(keys):
        raise ValueError(
            "entity_vertex_keys.keys must identify distinct entities; "
            f"got {len(mapping)} distinct keys for {len(keys)} entities"
        )
    return mapping


def _quotient_entity_ids(
    source: periodic_topology_owner.PeriodicMeshTopology,
    target: periodic_topology_owner.PeriodicMeshTopology,
    /,
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
    mesh: cell_mesh_owner.CellMesh,
    lifted: cell_mesh_owner.CellMesh,
    entity_ids: dict[int, np.ndarray],
    /,
    *,
    budget: CoordinateEnclosureBudget | None,
) -> cell_mesh_owner.CellMesh:
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
                "periodic.actual_geometry.exact_source must retain a power-geometry vertex source; "
                f"got {type(source).__name__}"
            )
        actual_geometry = cell_geometry_owner.CellGeometrySpec.power(
            lifted, source, budget=budget
        )
    arguments = (
        lifted,
        periodic.cell,
        periodic.vertex_representatives,
        periodic.vertex_shifts,
    )
    candidate = periodic_topology_owner.PeriodicMeshTopology(
        *arguments, actual_geometry=actual_geometry, budget=budget
    )
    quotient_ids = _quotient_entity_ids(periodic, candidate)
    if candidate.allocator_next_ids != periodic.allocator_next_ids or not all(
        np.array_equal(values, np.asarray(candidate.quotient.entities(degree).entity_ids))
        for degree, values in quotient_ids.items()
    ):
        # Unresolved (-1) cursors stay unresolved: explicit IDs without history.
        candidate = periodic_topology_owner.PeriodicMeshTopology(
            *arguments,
            actual_geometry=actual_geometry,
            entity_global_ids=quotient_ids,
            entity_allocator_next_ids={
                degree: cursor
                for degree, cursor in enumerate(periodic.allocator_next_ids)
                if 0 < degree < periodic.topological_dimension and cursor >= 0
            },
            budget=budget,
        )
    return cell_mesh_owner.CellMesh(
        lifted.coordinates,
        lifted.blocks,
        vertex_global_ids=lifted.vertex_global_ids,
        entity_global_ids=entity_ids,
        periodic_topology=candidate,
        numeric_version=lifted.numeric_version,
        budget=budget,
    )


@checked
def canonicalize_cell_mesh(
    mesh: cell_mesh_owner.CellMesh, /, *, budget: CoordinateEnclosureBudget | None = None
) -> cell_mesh_owner.CellMesh:
    """Return deterministic block/cell ordering without changing mesh geometry."""

    if mesh.storage is not None:
        # These are prepared local slots of an established logical mesh.
        # Reordering them as a serial mesh would destroy routing.
        return mesh
    if any(
        isinstance(block, reference_cell_types_owner.PolyhedralBlock)
        for block in mesh.blocks
    ):
        for block in mesh.blocks:
            identifiers = np.asarray(block.global_ids, dtype=np.int64)
            if np.any(identifiers[:-1] > identifiers[1:]):
                raise ValueError(
                    "block.global_ids must be ascending at construction for face-defined polyhedra; "
                    f"got block={block.name!r} with out-of-order IDs"
                )
        return mesh
    blocks = []
    for block in mesh.blocks:
        identifiers = np.asarray(block.global_ids, dtype=np.int64)
        order = np.argsort(identifiers, kind="stable")
        blocks.append(
            reference_cell_types_owner.CellBlock(
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
    rebuilt = cell_mesh_owner.CellMesh(
        mesh.coordinates,
        ordered,
        vertex_global_ids=mesh.vertex_global_ids,
        numeric_version=mesh.numeric_version,
    )
    entity_ids = {}
    for dimension in range(1, mesh.topological_dimension):
        identifiers = _identifiers_by_key(
            entity_vertex_keys(mesh, dimension),
            np.asarray(mesh.entity_set(dimension).entity_ids),
        )
        entity_ids[dimension] = np.asarray(
            [identifiers[key] for key in entity_vertex_keys(rebuilt, dimension)],
            dtype=np.int64,
        )
    if not all(
        np.array_equal(values, np.asarray(rebuilt.entity_set(dimension).entity_ids))
        for dimension, values in entity_ids.items()
    ):
        rebuilt = cell_mesh_owner.CellMesh(
            mesh.coordinates,
            ordered,
            vertex_global_ids=mesh.vertex_global_ids,
            entity_global_ids=entity_ids,
            numeric_version=mesh.numeric_version,
        )
    return _periodic_rebuild(mesh, rebuilt, entity_ids, budget=budget)


@final
class _CertificationTransition(StrictModule):
    inputs: MeshCertificationInputs | None
    lineage: MeshLineage | None
    cell_regions: ArrayLike | None
    junction_vertices: ArrayLike | None
    prepared: PreparedMeshingAcceptance | None
    audit_validity: CellValidityCertificate | None


@final
class _CertificationCarrier(StrictModule):
    mesh: cell_mesh_owner.CellMesh
    geometry: cell_geometry_owner.CellGeometrySpec
    transition: _CertificationTransition


@final
class _CertificationVerdict(StrictModule):
    report: MeshCertificationReport | None
    stages: tuple[MeshingStageReport, ...]


@final
class _CertificationPublication(StrictModule):
    compliance: MeshingComplianceReport
    trace: MeshingTrace
    provenance: SemanticProvenance
    runtime: MeshingRuntimeInfo


def _prepare_certification_transition(
    inputs: MeshCertificationInputs | None,
    lineage: MeshLineage | None,
    cell_regions: ArrayLike | None,
    junction_vertices: ArrayLike | None,
    prepared: PreparedMeshingAcceptance | None,
    audit_validity: CellValidityCertificate | None,
    /,
) -> _CertificationTransition:
    """Join transition inputs with prepared evidence before mesh preparation."""
    if (inputs is None) != (lineage is None):
        raise ValueError(
            "certify_cell_mesh.certification_inputs must accompany lineage; "
            f"got inputs={type(inputs).__name__}, lineage={type(lineage).__name__}"
        )
    if prepared is not None and inputs is None:
        raise ValueError(
            "certify_cell_mesh.certification_prepared must accompany certification_inputs; "
            "got an absent transition request"
        )
    match prepared:
        case PreparedVolumeAcceptance(volume=volume):
            if (
                audit_validity is not None
                and volume.validity.certificate_id != audit_validity.certificate_id
            ):
                raise ValueError(
                    "certify_cell_mesh.audit_prepared_validity must agree with certification_prepared.volume.validity; "
                    f"got audit={audit_validity.certificate_id!r}, volume={volume.validity.certificate_id!r}"
                )
            audit_validity = volume.validity
        case PreparedBoundaryAcceptance() | None:
            pass
        case invalid:
            assert_never(invalid)
    for name, supplied in (
        ("certification_cell_regions", cell_regions is not None),
        ("certification_junction_vertices", junction_vertices is not None),
    ):
        if supplied and inputs is None:
            raise ValueError(
                f"certify_cell_mesh.{name} must accompany certification_inputs; got an absent transition request"
            )
    return _CertificationTransition(
        inputs=inputs,
        lineage=lineage,
        cell_regions=cell_regions,
        junction_vertices=junction_vertices,
        prepared=prepared,
        audit_validity=audit_validity,
    )


def _prepare_certification_carrier(
    mesh: cell_mesh_owner.CellMesh,
    geometry: cell_geometry_owner.CellGeometrySpec | None,
    region_evidence: RegionMeshingEvidence | None,
    transition: _CertificationTransition,
    budget: CoordinateEnclosureBudget | None,
    /,
) -> _CertificationCarrier:
    """Canonicalize only carriers whose supplied rows will remain aligned."""
    canonical = canonicalize_cell_mesh(mesh, budget=budget)
    if region_evidence is not None and canonical is not mesh:
        raise ValueError(
            "certify_cell_mesh.region_evidence must bind a mesh sorted before constructing RegionMeshingEvidence; "
            "got canonicalization that would reorder supplied region evidence"
        )
    if geometry is not None and canonical is not mesh:
        raise ValueError(
            "certify_cell_mesh.geometry must bind a mesh sorted before constructing CellGeometrySpec; "
            "got canonicalization that would reorder supplied geometry DOF rows"
        )
    geometry_ = (
        cell_geometry_owner.CellGeometrySpec.affine(canonical, budget=budget)
        if geometry is None
        else geometry
    )
    return _CertificationCarrier(
        mesh=canonical, geometry=geometry_, transition=transition
    )


def _audit_certification_carrier(
    carrier: _CertificationCarrier,
    policy: CellMeshAuditPolicy | None,
    boundary: SurfaceModel | None,
    patches: tuple[MeshPatch, ...] | MeshPatchBank,
    zones: tuple[MeshZone, ...],
    labels: tuple[MeshLabel, ...],
    attributes: tuple[MeshAttribute, ...],
    associations: tuple[GeometryAssociation, ...],
    budget: CoordinateEnclosureBudget | None,
    /,
) -> CellMeshAuditReport:
    """Audit the ordered carrier and preserve failing-cell diagnostics."""
    inputs = carrier.transition.inputs
    if inputs is not None:
        # A volume's boundary must close even when its coordinate map is curved.
        # Execute the retained schedule's checks before renewing its acceptance.
        policy = required_audit_policy(
            CellMeshAuditPolicy() if policy is None else policy,
            inputs.schedule.required_audit_checks,
        )
    audit = audit_cell_mesh(
        carrier.mesh,
        carrier.geometry,
        policy=policy,
        boundary=boundary,
        patches=patches,
        associations=associations,
        attributes=attributes,
        zones=zones,
        labels=labels,
        prepared_validity=carrier.transition.audit_validity,
        budget=budget,
    )
    if not audit.passed:
        uncertified = (
            np.asarray(audit.validity.status) != CellValidityStatus.CERTIFIED_VALID
        )
        cell_ids = np.concatenate(
            [
                np.asarray(block.global_ids, dtype=np.int64)
                for block in carrier.mesh.blocks
            ]
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
    return audit


def _certify_transition(
    carrier: _CertificationCarrier,
    audit: CellMeshAuditReport,
    budget: CoordinateEnclosureBudget | None,
    /,
) -> _CertificationVerdict:
    """Renew retained acceptance after the ordered carrier passes its audit."""
    transition = carrier.transition
    if transition.inputs is None or transition.lineage is None:
        return _CertificationVerdict(report=None, stages=())
    certification = transition.inputs.recertify_transition(
        carrier.mesh,
        carrier.geometry,
        audit,
        lineage=transition.lineage,
        cell_regions=transition.cell_regions,
        junction_vertices=transition.junction_vertices,
        prepared=transition.prepared,
        budget=budget,
    )
    certification.require_passed()
    return _CertificationVerdict(
        report=certification, stages=(acceptance_stage_report(certification),)
    )


def _prepare_certification_publication(
    mesh: cell_mesh_owner.CellMesh,
    carrier: _CertificationCarrier,
    coordinate_contract: SpatialCoordinateContract,
    boundary: SurfaceModel | None,
    audit: CellMeshAuditReport,
    verdict: _CertificationVerdict,
    /,
) -> _CertificationPublication:
    """Build trace and provenance without changing stage or fingerprint order."""
    canonical = carrier.mesh
    geometry_ = carrier.geometry
    certification = verdict.report
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
        *verdict.stages,
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
    return _CertificationPublication(
        compliance=compliance, trace=trace, provenance=provenance, runtime=runtime
    )


@checked
def certify_cell_mesh(
    mesh: cell_mesh_owner.CellMesh,
    coordinate_contract: SpatialCoordinateContract,
    /,
    *,
    budget: CoordinateEnclosureBudget | None = None,
    geometry: cell_geometry_owner.CellGeometrySpec | None = None,
    audit_policy: CellMeshAuditPolicy | None = None,
    boundary: SurfaceModel | None = None,
    patches: tuple[MeshPatch, ...] | MeshPatchBank = (),
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
    certification_prepared: PreparedMeshingAcceptance | None = None,
    audit_prepared_validity: CellValidityCertificate | None = None,
    surface_source: SurfaceSourceCharts | None = None,
) -> CellMeshingResult:
    """Canonicalize a cell mesh and publish its audited coordinate map.

    ``certification_prepared`` selects volume or independent boundary evidence
    for retained acceptance. Volume validity also feeds the audit; separately
    supplied audit validity must identify that same certificate.
    Retained schedules execute their required boundary-closure audit even when
    ``audit_policy`` leaves that check skipped.
    """

    mesh.require_dense("certify_cell_mesh")
    if boundary is not None and (
        boundary.metadata.coordinate_contract.spatial_id != coordinate_contract.spatial_id
    ):
        raise ValueError(
            "boundary.coordinate_contract must match coordinate_contract; "
            f"got spatial_id={boundary.metadata.coordinate_contract.spatial_id!r}"
        )
    transition = _prepare_certification_transition(
        certification_inputs,
        lineage,
        certification_cell_regions,
        certification_junction_vertices,
        certification_prepared,
        audit_prepared_validity,
    )
    carrier = _prepare_certification_carrier(
        mesh, geometry, region_evidence, transition, budget
    )
    audit = _audit_certification_carrier(
        carrier,
        audit_policy,
        boundary,
        patches,
        zones,
        labels,
        attributes,
        associations,
        budget,
    )
    verdict = _certify_transition(carrier, audit, budget)
    publication = _prepare_certification_publication(
        mesh, carrier, coordinate_contract, boundary, audit, verdict
    )
    return CellMeshingResult(
        carrier.mesh,
        carrier.geometry,
        coordinate_contract,
        audit,
        audit.quality,
        publication.compliance,
        publication.trace,
        _NATIVE_PROVIDER,
        publication.runtime,
        MeshingDerivativeMode.FIXED_TOPOLOGY_EXACT,
        publication.provenance,
        boundary=boundary,
        patches=patches,
        zones=zones,
        labels=labels,
        attributes=attributes,
        associations=associations,
        region_evidence=region_evidence,
        region_boundary_evidence=region_boundary_evidence,
        certification=verdict.report,
        surface_source=surface_source,
        budget=budget,
    )


@final
class _OwnerLocalScientificSource(StrictModule):
    source_audit_id: str = eqx.field(static=True)
    scientific_theorem_id: str = eqx.field(static=True)
    storage_id: str = eqx.field(static=True)


def _prepare_owner_local_scientific_source(
    mesh: cell_mesh_owner.CellMesh,
    source: CellMeshingResult,
    collective_evidence: CollectiveMeshEvidence,
    /,
) -> _OwnerLocalScientificSource:
    """Join source acceptance, subdivision evidence, and target storage."""
    storage = mesh.storage
    if storage is None:
        raise ValueError("mesh.storage must provide owner-local storage; got None")
    uniform = collective_evidence.uniform_refinement
    original = require_original_meshing_source(
        source if uniform is None else uniform.source
    )
    # Lazy: _canonical -> _initial_certification -> _adaptation -> _assembly import cycle.
    from . import _initial_certification as initial_certification_owner

    match original:
        case initial_certification_owner.InitialCollectiveMeshEvidence():
            type(original).require_passed(original)
            source_audit_id = original.evidence_id
            scientific_theorem_id = original.source_evidence_id
        case CellMeshingResult():
            embedding = collective_evidence.source_embedding
            if embedding is None:
                raise ValueError(
                    "owner_local_source.embedding must carry its prepared proof; got absent evidence"
                )
            type(embedding.binding).require(embedding.binding, original.mesh, original.geometry)
            if (
                embedding.status != "certified"
                or embedding.validity_certificate_id
                != original.audit.validity.certificate_id
            ):
                raise ValueError(
                    "owner_local_source.embedding must certify the audited map; got mismatched evidence"
                )
            source_audit_id = original.audit.report_id
            scientific_theorem_id = collective_evidence.source_premise_id
        case invalid:
            assert_never(invalid)
    if (
        collective_evidence.source_mesh_id != source.mesh.mesh_id
        or collective_evidence.source_audit_id != source_audit_id
        or collective_evidence.mesh_id != mesh.mesh_id
        or collective_evidence.evidence_id != storage.evidence_id
    ):
        raise ValueError(
            "certify_owner_local_cell_mesh.collective_evidence must join source acceptance and target storage; "
            f"got source_mesh_id={collective_evidence.source_mesh_id!r}, "
            f"target_mesh_id={collective_evidence.mesh_id!r}, evidence_id={collective_evidence.evidence_id!r}"
        )
    type(collective_evidence).require_passed(collective_evidence)
    return _OwnerLocalScientificSource(
        source_audit_id=source_audit_id,
        scientific_theorem_id=scientific_theorem_id,
        storage_id=storage.storage_id,
    )


@checked
def certify_owner_local_cell_mesh(
    mesh: cell_mesh_owner.CellMesh,
    source: CellMeshingResult,
    collective_evidence: CollectiveMeshEvidence,
    /,
    *,
    budget: CoordinateEnclosureBudget | None = None,
    audit_policy: CellMeshAuditPolicy | None = None,
    geometry: cell_geometry_owner.CellGeometrySpec | None = None,
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
    recertified collectively for this publication.
    """
    scientific = _prepare_owner_local_scientific_source(mesh, source, collective_evidence)
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
                f"certify_owner_local_cell_mesh.organization must retain inherited {name} coverage; "
                "got an empty target collection"
            )
    if source.region_evidence is not None and region_evidence is None:
        raise ValueError(
            "certify_owner_local_cell_mesh.region_evidence must retain source.region_evidence; got None"
        )
    geometry = (
        cell_geometry_owner.CellGeometrySpec.affine(mesh, budget=budget)
        if geometry is None
        else geometry
    )
    audit = audit_cell_mesh(
        mesh,
        geometry,
        policy=audit_policy,
        patches=patches,
        zones=zones,
        labels=labels,
        attributes=attributes,
        associations=associations,
        budget=budget,
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
                    scientific.scientific_theorem_id,
                    collective_evidence.source_evidence_id,
                ),
            ),
            MeshingStageReport(
                MeshingStageKind.GEOMETRY_AUDIT,
                MeshingStageStatus.PASSED,
                input_ids=(mesh.mesh_id, scientific.storage_id),
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
            "local_storage": scientific.storage_id,
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
        budget=budget,
    )


__all__ = [
    "canonicalize_cell_mesh",
    "certify_cell_mesh",
    "certify_owner_local_cell_mesh",
]
