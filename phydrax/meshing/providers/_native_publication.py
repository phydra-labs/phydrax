#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Shared acceptance and publication of native meshing routes.

Every native route constructs a ``CellMesh`` with its coordinate geometry, organization and geometry associations, measures compliance,
and publishes through `publish_native_result`: the mesh is audited under the
route's declared output checks, then certified under the route's
`MeshCertificationSchedule` (global embedding plus the route's domain coverage
or source fidelity), and a `CellMeshingResult` exists only when the audit, the
certification, and compliance all pass. Failures raise `MeshingFailure` with
stage, entity, and requested-versus-achieved evidence.

Native planar publication retains a completed positive validity and global
embedding proof across audit and certification, under the same request limits.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager, nullcontext
from typing import Any, final

import equinox as eqx
import jax
import numpy as np
from jax.typing import ArrayLike

import phydrax.discretization as cell_geometry_owner
import phydrax.discretization as cell_mesh_owner
from phydrax.geometry._embedding_state import MeshCertificateLimits
from phydrax.geometry._global_embedding import GlobalEmbeddingCertificate
from phydrax.geometry._source_boundary_records import SourceBoundaryQuery
from phydrax.meshing._association_state import (
    GeometryAssociation,
    GeometryAssociationKind,
)
from phydrax.meshing._certification import (
    acceptance_stage_report,
    certify_meshing_acceptance,
    MeshCertificationPreparedEvidence,
    MeshCertificationReport,
    PreparedBoundaryAcceptance,
    PreparedVolumeAcceptance,
)
from phydrax.meshing._certification_schedule import MeshCertificationSchedule
from phydrax.meshing._contracts import (
    MeshingExecutionMode,
    MeshingLimits,
    MeshingProviderInfo,
)
from phydrax.meshing._failures import MeshingFailure, MeshingFailureCategory

from ..._identity import SemanticProvenance
from ..._meshcore import (
    NativeExecutionEvidence,
    NativeExecutionScope,
    NativeHostStorageWorkspace,
)
from ..._model._structure import model_array_bytes
from ..._physical import SpatialCoordinateContract
from ..._resource_ticket import HostBatchTicket
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization._cell_geometry import affine_geometry_construction_bounds
from ...discretization._cell_geometry_validity import cell_geometry_id, CellValidityStatus
from ...discretization._coordinate_budget import CoordinateEnclosureBudget
from ...geometry._domain_coverage import PiecewiseLinearDomain
from ...geometry._domain_planar_boundary import CadSourceDomain
from ...geometry._mapped_reference_domain import MappedReferenceDomain
from ...geometry._surface_source_support import SurfaceSourceCharts
from ...geometry.surface import SurfaceModel
from ...typing import checked
from .._audit import audit_cell_mesh, CellMeshAuditReport
from .. import _audit_topology as audit_topology_owner
from .._certification_inputs import MeshCertificationInputs
from .._measurements import (
    _NATIVE_EXECUTION_RECORD_ARRAY_BYTES,
    measure_phase,
    NativeExecutionRecord,
    NativeMeshingPhaseRecorder,
)
from .._organization_records import (
    MeshAttribute,
    MeshLabel,
    MeshPatch,
    MeshPatchBank,
    MeshZone,
    RegionBoundaryEvidence,
    RegionMeshingEvidence,
)
from .._result import CellMeshingResult, MeshingComplianceReport, MeshingRuntimeInfo
from .._trace import (
    MeshingEvidenceBinding,
    MeshingStageKind,
    MeshingStageReport,
    MeshingStageStatus,
    MeshingTrace,
)
from .._volume_generation import native_volume_checkpoint
from ._native_route_checks import (
    check_deadline,
    published_entity_limits,
    require_compliance,
)
from ._native_route_policy import NativePublicationPolicy


@checked
def coordinate_budget(limits: MeshingLimits, /) -> CoordinateEnclosureBudget:
    """Create a reusable coordinate view with the request's existing domain maxima."""
    ticket = HostBatchTicket(
        limits.maximum_work_units,
        queries_upper=limits.maximum_geometry_queries,
        bytes_upper=limits.maximum_scratch_bytes,
    )
    ticket.settle()
    return CoordinateEnclosureBudget(
        ticket,
        maximum_work_units=limits.maximum_work_units,
        maximum_memory_bytes=limits.maximum_scratch_bytes,
    )


@contextmanager
@checked
def coordinate_phase(
    scope: NativeExecutionScope,
    limits: MeshingLimits,
    budget: CoordinateEnclosureBudget,
    /,
) -> Iterator[CoordinateEnclosureBudget]:
    """Bind a coordinate-source phase; settle before independent native operations."""
    remaining = scope.remaining()
    ticket = scope.batch(
        min(
            limits.maximum_work_units - budget.work_units, remaining.remaining_work_units
        ),
        queries_upper=min(
            limits.maximum_geometry_queries, remaining.remaining_geometry_queries
        ),
        bytes_upper=min(limits.maximum_scratch_bytes, remaining.remaining_managed_bytes),
    )
    with budget.bind_ticket(ticket), ticket.native_target():
        yield budget


@final
class NativeCertificationRequest(StrictModule, NonTrainableState):
    """Route acceptance request: schedule, evidence inputs, and source identity.

    ``domain``/``cell_regions`` feed domain coverage, ``fidelity_source`` and
    ``fidelity_tolerance`` source fidelity, and ``junction_vertices`` declares
    curve-network junctions. Every scheduled check must certify; unresolved
    or violated evidence refuses publication.
    ``prepared`` retains positive volume evidence; its bindings
    are validated before reuse, including the unchanged certificate limits.
    Authored revisions come from source records or every source
    association row, never from the distinct numerical domain revision.
    ``certificate_limits`` is normalized once: an explicit value is the
    request's own certificate caps and is kept unchanged; without it the caps
    are derived from ``limits``. Either way publication runs certification on
    tickets of the request's outer ``limits`` execution scope, so those work,
    query and scratch maxima still refuse any certificate work they do not admit.
    """

    schedule: MeshCertificationSchedule
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    limits: MeshingLimits
    certificate_limits: MeshCertificateLimits
    domain: PiecewiseLinearDomain | MappedReferenceDomain | CadSourceDomain | None = None
    cell_regions: np.ndarray | None = None
    fidelity_source: SourceBoundaryQuery | None = None
    fidelity_tolerance: float | None = None
    junction_vertices: ArrayLike | None = None
    prepared: MeshCertificationPreparedEvidence | None = None
    scoped_fidelity: tuple[tuple[SourceBoundaryQuery, ArrayLike, float], ...] = ()

    @checked
    def __init__(
        self,
        schedule: MeshCertificationSchedule,
        source_id: str,
        source_revision: str,
        limits: MeshingLimits,
        domain: PiecewiseLinearDomain | MappedReferenceDomain | CadSourceDomain | None = None,
        cell_regions: np.ndarray | None = None,
        fidelity_source: SourceBoundaryQuery | None = None,
        fidelity_tolerance: float | None = None,
        junction_vertices: ArrayLike | None = None,
        prepared: MeshCertificationPreparedEvidence | None = None,
        scoped_fidelity: tuple[tuple[SourceBoundaryQuery, ArrayLike, float], ...] = (),
        certificate_limits: MeshCertificateLimits | None = None,
    ) -> None:
        self.schedule = schedule
        self.source_id = source_id
        self.source_revision = source_revision
        self.limits = limits
        self.domain = domain
        self.cell_regions = cell_regions
        self.fidelity_source = fidelity_source
        self.fidelity_tolerance = fidelity_tolerance
        self.junction_vertices = junction_vertices
        self.prepared = prepared
        self.scoped_fidelity = scoped_fidelity
        self.certificate_limits = (
            derived_certificate_limits(limits)
            if certificate_limits is None
            else certificate_limits
        )


def derived_certificate_limits(limits: MeshingLimits, /) -> MeshCertificateLimits:
    """Lower the default certificate caps to the request's maxima.

    Routes that certify before publication use these same caps so publication
    reuses their request-earned certificates.
    """
    defaults = MeshCertificateLimits()
    query_capacity = min(limits.maximum_geometry_queries, limits.maximum_work_units)
    pair_capacity = max(
        1, min(limits.maximum_work_units, limits.maximum_scratch_bytes // 256)
    )
    return MeshCertificateLimits(
        maximum_candidate_pairs=min(defaults.maximum_candidate_pairs, pair_capacity),
        maximum_ray_tests=min(defaults.maximum_ray_tests, limits.maximum_work_units),
        maximum_source_samples=max(
            1,
            min(
                defaults.maximum_source_samples,
                query_capacity,
                limits.maximum_scratch_bytes // 256,
            ),
        ),
        maximum_distance_evaluations=min(
            defaults.maximum_distance_evaluations, query_capacity
        ),
        maximum_subdivision_depth=defaults.maximum_subdivision_depth,
        maximum_subdivision_pieces=max(
            1,
            min(
                defaults.maximum_subdivision_pieces,
                limits.maximum_work_units,
                limits.maximum_scratch_bytes // 512,
            ),
        ),
        maximum_bernstein_nodes=max(
            1,
            min(
                defaults.maximum_bernstein_nodes,
                limits.maximum_work_units,
                limits.maximum_scratch_bytes // 128,
            ),
        ),
        maximum_periodic_images=max(
            1,
            min(
                defaults.maximum_periodic_images,
                limits.maximum_work_units // 32,
                limits.maximum_scratch_bytes // 4096,
            ),
        ),
        maximum_work_units=limits.maximum_work_units,
        maximum_scratch_bytes=limits.maximum_scratch_bytes,
    )


def _certify(
    mesh: cell_mesh_owner.CellMesh,
    geometry: cell_geometry_owner.CellGeometrySpec,
    audit: CellMeshAuditReport,
    request: NativeCertificationRequest,
    /,
    *,
    budget: CoordinateEnclosureBudget,
    prepared_embedding: GlobalEmbeddingCertificate | None = None,
) -> tuple[MeshCertificationReport, MeshingStageReport]:
    """Certify the audited mesh; refuse unless every scheduled check certifies."""

    report = certify_meshing_acceptance(
        mesh,
        geometry,
        audit,
        schedule=request.schedule,
        domain=request.domain,
        cell_regions=request.cell_regions,
        source=request.fidelity_source,
        fidelity_tolerance=request.fidelity_tolerance,
        limits=request.certificate_limits,
        junction_vertices=request.junction_vertices,
        prepared=PreparedVolumeAcceptance(request.prepared)
        if request.prepared is not None
        else PreparedBoundaryAcceptance(embedding=prepared_embedding)
        if prepared_embedding is not None
        else None,
        scoped_fidelity=request.scoped_fidelity,
        budget=budget,
    )
    stage = acceptance_stage_report(report)
    report.require_passed()
    return report, stage


@final
class NativePublicationOrganization(StrictModule):
    """Source associations and authored organization retained by publication."""

    boundary: SurfaceModel | None = None
    patches: tuple[MeshPatch, ...] | MeshPatchBank = ()
    zones: tuple[MeshZone, ...] = ()
    labels: tuple[MeshLabel, ...] = ()
    attributes: tuple[MeshAttribute, ...] = ()
    associations: tuple[GeometryAssociation, ...] = ()
    region_evidence: RegionMeshingEvidence | None = None
    region_boundary_evidence: tuple[RegionBoundaryEvidence, ...] = ()
    surface_source: SurfaceSourceCharts | None = None

    @checked
    def __init__(
        self,
        boundary: SurfaceModel | None = None,
        patches: tuple[MeshPatch, ...] | MeshPatchBank = (),
        zones: tuple[MeshZone, ...] = (),
        labels: tuple[MeshLabel, ...] = (),
        attributes: tuple[MeshAttribute, ...] = (),
        associations: tuple[GeometryAssociation, ...] = (),
        region_evidence: RegionMeshingEvidence | None = None,
        region_boundary_evidence: tuple[RegionBoundaryEvidence, ...] = (),
        surface_source: SurfaceSourceCharts | None = None,
    ) -> None:
        self.boundary = boundary
        self.patches = patches
        self.zones = zones
        self.labels = labels
        self.attributes = attributes
        self.associations = associations
        self.region_evidence = region_evidence
        self.region_boundary_evidence = region_boundary_evidence
        self.surface_source = surface_source


@final
class NativePublicationExecution(StrictModule):
    """Execution scope and retained-storage owner for publication phases."""

    scope: NativeExecutionScope
    workspace: NativeHostStorageWorkspace
    record_phase: NativeMeshingPhaseRecorder | None = None
    budget: CoordinateEnclosureBudget | None = None

    @checked
    def __init__(
        self,
        scope: NativeExecutionScope,
        workspace: NativeHostStorageWorkspace,
        record_phase: NativeMeshingPhaseRecorder | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> None:
        self.scope = scope
        self.workspace = workspace
        self.record_phase = record_phase
        self.budget = budget


@final
class _PublicationRequest(StrictModule):
    mesh: cell_mesh_owner.CellMesh
    coordinate_contract: SpatialCoordinateContract
    compliance: MeshingComplianceReport
    construction: tuple[MeshingStageReport, ...]
    provider: MeshingProviderInfo
    provenance: Mapping[str, Any]
    certification: NativeCertificationRequest
    policy: NativePublicationPolicy
    organization: NativePublicationOrganization
    execution: NativePublicationExecution
    budget: CoordinateEnclosureBudget


@final
class _PublicationGeometry(StrictModule):
    geometry: cell_geometry_owner.CellGeometrySpec


@final
class _PublicationAudit(StrictModule):
    audit: CellMeshAuditReport
    embedding: GlobalEmbeddingCertificate | None = None


@final
class _PublicationCertification(StrictModule):
    report: MeshCertificationReport
    stage: MeshingStageReport


@final
class _PublicationResult(StrictModule):
    result: CellMeshingResult


def _prepare_publication(
    request: _PublicationRequest, geometry: cell_geometry_owner.CellGeometrySpec | None, /
) -> _PublicationGeometry:
    mesh = request.mesh
    certification = request.certification
    scope = request.execution.scope
    workspace = request.execution.workspace
    record_phase = request.execution.record_phase
    associations = request.organization.associations
    with measure_phase(record_phase, "publication"):
        native_volume_checkpoint(
            certification.limits, MeshingStageKind.CANONICALIZATION, scope=scope
        )
        if geometry is None:
            mesh_shapes = tuple(
                (block.cell_kind, block.cell_count, block.arity) for block in mesh.blocks
            )
            with scope.host_workspace() as phase_storage:
                phase_storage.set_bound(
                    affine_geometry_construction_bounds(
                        mesh.coordinates.shape[0], mesh.ambient_dimension, mesh_shapes
                    )
                )
                geometry = cell_geometry_owner.CellGeometrySpec.affine(mesh)
        if workspace is not None:
            workspace.retain_owner(geometry)
        if certification.prepared is not None:
            certification.prepared.require(
                mesh,
                geometry,
                MeshCertificationInputs(
                    mesh,
                    geometry,
                    certification.schedule,
                    domain=certification.domain,
                    cell_regions=certification.cell_regions,
                    source=certification.fidelity_source,
                    fidelity_tolerance=certification.fidelity_tolerance,
                    limits=certification.certificate_limits,
                    junction_vertices=certification.junction_vertices,
                ),
            )
            bound_source = certification.prepared.request.source
            bound_domain = certification.prepared.request.domain
            if bound_domain is None:
                raise ValueError(
                    "certification.prepared.request.domain must be bound; got None"
                )
            if bound_source is not None:
                if (
                    certification.source_id != bound_source.source_id
                    or certification.source_revision != bound_source.source_revision
                ):
                    raise ValueError(
                        "certification.source_revision must match the prepared source; got a different source binding"
                    )
            elif associations:
                if certification.source_id != bound_domain.source_id or any(
                    value.source_id != certification.source_id
                    or value.source_revision != certification.source_revision
                    or not value.complete
                    or not value.exact
                    or (
                        isinstance(bound_domain, PiecewiseLinearDomain)
                        and value.association_kind
                        is not GeometryAssociationKind.PIECEWISE_LINEAR
                    )
                    for value in associations
                ):
                    raise ValueError(
                        "publication.associations must match the prepared source binding; got inconsistent associations"
                    )
            else:
                raise ValueError(
                    "publication.associations must bind the prepared source; got no source evidence"
                )
    return _PublicationGeometry(geometry)


def _audit_publication(
    request: _PublicationRequest, prepared: _PublicationGeometry, /
) -> _PublicationAudit:
    mesh = request.mesh
    certification = request.certification
    scope = request.execution.scope
    record_phase = request.execution.record_phase
    budget = request.budget
    geometry = prepared.geometry
    audit_policy = request.policy.audit_policy
    boundary = request.organization.boundary
    patches = request.organization.patches
    associations = request.organization.associations
    zones = request.organization.zones
    labels = request.organization.labels
    attributes = request.organization.attributes
    with measure_phase(record_phase, "audit"):
        native_volume_checkpoint(
            certification.limits, MeshingStageKind.GEOMETRY_AUDIT, scope=scope
        )
        with coordinate_phase(scope, certification.limits, budget):
            embedding = None
            if certification.prepared is None:
                validity, embedding = audit_topology_owner.prepare_planar_audit_embedding(
                    mesh,
                    geometry,
                    policy=audit_policy,
                    limits=certification.certificate_limits,
                    junction_vertices=certification.junction_vertices,
                    budget=budget,
                )
            else:
                validity = certification.prepared.validity
            audit = audit_cell_mesh(
                mesh,
                geometry,
                policy=audit_policy,
                prepared_validity=validity,
                boundary=boundary,
                patches=patches,
                associations=associations,
                zones=zones,
                labels=labels,
                attributes=attributes,
                budget=budget,
            )
    native_volume_checkpoint(
        certification.limits, MeshingStageKind.GEOMETRY_AUDIT, scope=scope
    )
    if not audit.passed:
        evaluation = audit.quality.evaluation
        status, measures, mean_ratios, aspect_ratios, host_points, blocks = (
            jax.device_get(
                (
                    audit.validity.status,
                    evaluation.measures,
                    evaluation.mean_ratios,
                    evaluation.aspect_ratios,
                    mesh.coordinates,
                    tuple(
                        (value.global_ids, value.vertices, value.vertex_valid)
                        for value in mesh.blocks
                    ),
                )
            )
        )
        uncertified = np.asarray(status) != CellValidityStatus.CERTIFIED_VALID
        cell_ids = np.concatenate(
            [np.asarray(ids, dtype=np.int64) for ids, _, _ in blocks]
        )
        failure_policy = audit_policy
        quality_failed = (
            (np.asarray(measures) <= failure_policy.minimum_measure)
            | (np.asarray(mean_ratios) < failure_policy.minimum_mean_ratio)
            | (np.asarray(aspect_ratios) > failure_policy.maximum_aspect_ratio)
        )
        offending = tuple(int(value) for value in cell_ids[uncertified | quality_failed])
        if not offending:
            offending = audit.quality.worst_cell_global_ids
        selected = set(offending)
        points = np.asarray(host_points, dtype=np.float64)
        locations = tuple(
            tuple(float(value) for value in np.mean(points[vertices[valid]], axis=0))
            for ids, vertices_, valid_ in blocks
            for identifier, vertices, valid in zip(
                np.asarray(ids).tolist(),
                np.asarray(vertices_),
                np.asarray(valid_, dtype=np.bool_),
                strict=True,
            )
            if identifier in selected
        )
        raise MeshingFailure(
            MeshingFailureCategory.AUDIT_FAILED,
            "audit.issues must be empty; got "
            + "; ".join(audit.issues)
            + (
                f" (audit={audit.report_id}, geometry={audit.geometry_id}, "
                f"source={certification.source_id}@{certification.source_revision})"
            ),
            stage=MeshingStageKind.GEOMETRY_AUDIT.value,
            entity_ids=offending,
            locations=locations,
            requested=(
                ("minimum_measure", failure_policy.minimum_measure),
                ("minimum_mean_ratio", failure_policy.minimum_mean_ratio),
                ("maximum_aspect_ratio", failure_policy.maximum_aspect_ratio),
            ),
            achieved=(
                ("minimum_measure", audit.quality.minimum_measure),
                ("minimum_mean_ratio", audit.quality.minimum_mean_ratio),
                ("maximum_aspect_ratio", audit.quality.maximum_aspect_ratio),
            ),
            logical_findings=(
                ("audit:quality:cell_ids", evaluation.cell_global_ids),
                ("audit:quality:measures", evaluation.measures),
                (
                    "audit:validity:determinant_lower",
                    audit.validity.determinant_lower,
                ),
                (
                    "audit:validity:determinant_upper",
                    audit.validity.determinant_upper,
                ),
                ("audit:validity:status", audit.validity.status),
            ),
        )
    return _PublicationAudit(audit, embedding)


def _certify_publication(
    request: _PublicationRequest,
    prepared: _PublicationGeometry,
    audited: _PublicationAudit,
    /,
) -> _PublicationCertification:
    certification = request.certification
    scope = request.execution.scope
    record_phase = request.execution.record_phase
    budget = request.budget
    with measure_phase(record_phase, "certification"):
        native_volume_checkpoint(
            certification.limits, MeshingStageKind.CERTIFICATION, scope=scope
        )
        with coordinate_phase(scope, certification.limits, budget):
            report, stage = _certify(
                request.mesh,
                prepared.geometry,
                audited.audit,
                certification,
                budget=budget,
                prepared_embedding=audited.embedding,
            )
    native_volume_checkpoint(
        certification.limits, MeshingStageKind.CERTIFICATION, scope=scope
    )
    return _PublicationCertification(report, stage)


@final
class _PublicationTrace(StrictModule):
    stages: tuple[MeshingStageReport, ...]
    binding: MeshingEvidenceBinding
    runtime: MeshingRuntimeInfo


def _prepare_publication_trace(
    request: _PublicationRequest,
    prepared: _PublicationGeometry,
    audited: _PublicationAudit,
    certified: _PublicationCertification,
    /,
) -> _PublicationTrace:
    mesh, geometry, audit = request.mesh, prepared.geometry, audited.audit
    certification, certification_stage = request.certification, certified.stage
    construction, compliance, provider = (
        request.construction,
        request.compliance,
        request.provider,
    )
    associations = request.organization.associations
    enforced_limits = request.policy.enforced_limits
    if "data_bytes" not in enforced_limits:
        enforced_limits = (*enforced_limits, "data_bytes")
    unenforced_limits = tuple(
        value for value in request.policy.unenforced_limits if value != "data_bytes"
    )
    audit_status = (
        MeshingStageStatus.WARNING if audit.recorded else MeshingStageStatus.PASSED
    )
    stages = (
        *construction,
        *(
            (
                MeshingStageReport(
                    MeshingStageKind.GEOMETRY_ASSOCIATION,
                    MeshingStageStatus.PASSED,
                    input_ids=(mesh.mesh_id,),
                    output_ids=tuple(value.association_id for value in associations),
                ),
            )
            if associations
            else ()
        ),
        MeshingStageReport(
            MeshingStageKind.GEOMETRY_AUDIT,
            audit_status,
            input_ids=(geometry.geometry_layout_id,),
            output_ids=(audit.validity.certificate_id,),
        ),
        MeshingStageReport(
            MeshingStageKind.TOPOLOGY_AUDIT,
            audit_status,
            input_ids=(mesh.topology_id,),
            output_ids=(audit.report_id,),
        ),
        certification_stage,
        MeshingStageReport(
            MeshingStageKind.SPECIFICATION_COMPLIANCE,
            MeshingStageStatus.PASSED,
            input_ids=(compliance.specification_id,),
            output_ids=(compliance.report_id,),
        ),
    )
    runtime = MeshingRuntimeInfo(
        provider.provider_id,
        provider.version,
        MeshingExecutionMode.IN_PROCESS,
        deterministic=True,
        enforced_limits=enforced_limits,
        unenforced_limits=unenforced_limits,
    )
    binding = MeshingEvidenceBinding(
        source_id=certification.source_id,
        source_revision=certification.source_revision,
        topology_id=mesh.topology_id,
        geometry_id=cell_geometry_id(geometry),
        geometry_layout_id=geometry.geometry_layout_id,
        policy_ids=(audit.policy_id, certification.schedule.schedule_id),
        runtime_id=runtime.runtime_id,
    )
    return _PublicationTrace(stages, binding, runtime)


def _primitive_provenance_storage(
    value: object, /, *, path_length: int = len("semantic_content")
) -> tuple[int, int, int] | None:
    """Bound static normalization/JSON for a primitive provenance tree.

    Return normalized graph bytes, ASCII JSON characters and normalization
    scratch. Unsupported semantic programs keep their existing publication
    path; this proof is used only where the actual provenance tree satisfies it.
    """
    path_storage = 256 + 4 * path_length
    if value is None or isinstance(value, bool):
        return 0, 5, path_storage
    if isinstance(value, int):
        return 0, value.bit_length() * 30103 // 100000 + 3, path_storage
    if isinstance(value, float):
        return 0, 32, path_storage
    if isinstance(value, str):
        # ensure_ascii emits two six-character surrogate escapes for an
        # astral code point; this also covers quotes and control characters.
        return 0, 2 + 12 * len(value), path_storage
    if isinstance(value, np.generic):
        scalar = _primitive_provenance_storage(value.item(), path_length=path_length)
        if scalar is None:
            return None
        return 256 + scalar[0], scalar[1], scalar[2]
    if isinstance(value, Mapping):
        count = len(value)
        graph = 2 * 256 + 2 * 64 + (16 + 256 + 32) * count
        characters = len('{"items":[],"kind":"mapping"}')
        # _named_records owns raw/normalized pairs, keys, distinct-key slots
        # and sorting input/output before normalization consumes its records.
        scratch = path_storage + 6 * 256 + count * (2 * (256 + 32) + 64 + 5 * 16)
        for key, child in value.items():
            if not isinstance(key, str):
                return None
            child_bound = _primitive_provenance_storage(
                child, path_length=path_length + 1 + len(key)
            )
            if child_bound is None:
                return None
            graph += child_bound[0]
            characters += 4 + 2 + 12 * len(key) + child_bound[1]
            scratch += child_bound[2] + 256 + 4 * len(key)
        return graph, characters, scratch
    if isinstance(value, (tuple, list)):
        graph = 2 * 256 + 2 * 64 + 16 * len(value)
        characters = len('{"items":[],"kind":"tuple"}')
        scratch = path_storage
        for index, child in enumerate(value):
            child_bound = _primitive_provenance_storage(
                child, path_length=path_length + 2 + len(str(index))
            )
            if child_bound is None:
                return None
            graph += child_bound[0]
            characters += 1 + child_bound[1]
            scratch += child_bound[2]
        return graph, characters, scratch
    return None


def _publication_provenance_bytes(provenance: Mapping[str, Any], /) -> int | None:
    bound = _primitive_provenance_storage(provenance)
    if bound is None:
        return None
    graph, characters, scratch = bound
    # The three publication IDs are 64-character strings. Include their
    # mapping entries and semantic-content hash wrapper before constructing
    # the merged mapping. Normalized/canonical graph copies coexist, as do
    # retained JSON, hash JSON text and UTF-8 bytes (4+4+1 bytes/character).
    graph += 3 * (16 + 256 + 32)
    characters += 3 * (4 + 2 + 12 * 13 + 2 + 12 * 64) + 64
    scratch += 6 * 256 + 3 * (2 * (256 + 32) + 64 + 5 * 16)
    return 2 * graph + scratch + 9 * characters + 32 * 256


def _assemble_publication(
    request: _PublicationRequest,
    prepared: _PublicationGeometry,
    audited: _PublicationAudit,
    certified: _PublicationCertification,
    /,
) -> _PublicationResult:
    if not request.execution.workspace.has_covered_retained_storage:
        return _assemble_publication_active(request, prepared, audited, certified)
    provenance_bytes = _publication_provenance_bytes(request.provenance)
    if provenance_bytes is None:
        return _assemble_publication_active(request, prepared, audited, certified)
    scope, owner = request.execution.scope, request.execution.workspace
    rows = (
        len(request.construction) + 6 + len(request.organization.associations)
        + len(request.organization.patches) + len(request.organization.zones)
        + len(request.organization.labels)
    )
    patches = request.organization.patches
    bank_bytes = (
        0
        if isinstance(patches, MeshPatchBank)
        else sum(
            patch.scope.entity_ids.size * np.dtype(patch.scope.entity_ids.dtype).itemsize
            for patch in patches
        ) + 17 * len(patches) + 8
    )
    phase = NativeHostStorageWorkspace(scope).open()
    adopted = False
    try:
        phase.set_bound(
            provenance_bytes + _RECORD_ROW_BYTES * rows + 2 * bank_bytes + 4096
        )
        result = _assemble_publication_active(request, prepared, audited, certified)
        # All new result banks/records were covered before construction.
        # Complete the same token with its exact new-owner union; neither the
        # live input coverage nor this token is released during adoption.
        phase.retain_owner(
            result, existing=owner.retained_owner_objects, replace_bound=True
        )
        owner.adopt_workspace(phase)
        adopted = True
        return result
    finally:
        if not adopted:
            phase.close()


def _assemble_publication_active(
    request: _PublicationRequest,
    prepared: _PublicationGeometry,
    audited: _PublicationAudit,
    certified: _PublicationCertification,
    /,
) -> _PublicationResult:
    mesh = request.mesh
    certification = request.certification
    scope = request.execution.scope
    geometry = prepared.geometry
    audit = audited.audit
    report = certified.report
    certification_stage = certified.stage
    compliance = request.compliance
    provider = request.provider
    coordinate_contract = request.coordinate_contract
    provenance = request.provenance
    derivative_mode = request.policy.derivative_mode
    boundary = request.organization.boundary
    patches = request.organization.patches
    zones = request.organization.zones
    labels = request.organization.labels
    attributes = request.organization.attributes
    associations = request.organization.associations
    surface_source = request.organization.surface_source
    region_evidence = request.organization.region_evidence
    region_boundary_evidence = request.organization.region_boundary_evidence
    traced = _prepare_publication_trace(request, prepared, audited, certified)
    native_volume_checkpoint(
        certification.limits, MeshingStageKind.CANONICALIZATION, scope=scope
    )
    trace = MeshingTrace(traced.stages, binding=traced.binding)
    retained = (
        mesh,
        geometry,
        audit,
        compliance,
        trace,
        traced.runtime,
        boundary,
        patches,
        zones,
        labels,
        attributes,
        associations,
        surface_source,
        report,
        region_evidence,
        region_boundary_evidence,
    )
    bounds = _result_construction_bounds(retained, patches, zones, labels)
    # The supplied components are distinct-buffer subsets of the result; with the
    # bank the constructor may build, they bound its retained data before any
    # allocation. Execution receipts are bound later by the provider owner.
    _require_retained_bytes(bounds.retained_bytes, certification.limits)
    with scope.host_workspace() as workspace:
        workspace.set_bound(bounds.temporary_bytes)
        result = CellMeshingResult(
            mesh,
            geometry,
            coordinate_contract,
            audit,
            audit.quality,
            compliance,
            trace,
            provider,
            traced.runtime,
            derivative_mode,
            SemanticProvenance(
                {
                    **provenance,
                    "mesh": mesh.mesh_id,
                    "audit": audit.report_id,
                    "certification": certification_stage.report_id,
                }
            ),
            boundary=boundary,
            patches=patches,
            zones=zones,
            labels=labels,
            attributes=attributes,
            associations=associations,
            surface_source=surface_source,
            certification=report,
            region_evidence=region_evidence,
            region_boundary_evidence=region_boundary_evidence,
        )
    _require_retained_data(result, certification.limits)
    return _PublicationResult(result)


@final
class _ResultConstructionBounds(StrictModule):
    retained_bytes: int = eqx.field(static=True)
    temporary_bytes: int = eqx.field(static=True)


# Host bytes for one Python record row (object header, slots, and tuple entry).
_RECORD_ROW_BYTES = 1024


def _result_construction_bounds(
    retained: tuple[object, ...],
    patches: tuple[MeshPatch, ...] | MeshPatchBank,
    zones: tuple[MeshZone, ...],
    labels: tuple[MeshLabel, ...],
    /,
) -> _ResultConstructionBounds:
    """Bound result retention and constructor temporaries.

    The constructor materializes each patch row once (its scope slice may copy
    the entity IDs) and builds a column bank from tuple input (codes plus the
    concatenated scope IDs and offsets).
    """
    rows = len(patches) + len(zones) + len(labels)
    if isinstance(patches, MeshPatchBank):
        entity_bytes = model_array_bytes(patches.scopes)
        bank_bytes = 0
    else:
        entity_bytes = sum(np.asarray(patch.scope.entity_ids).nbytes for patch in patches)
        bank_bytes = entity_bytes + 17 * len(patches) + 8
    return _ResultConstructionBounds(
        retained_bytes=model_array_bytes(retained) + bank_bytes,
        temporary_bytes=entity_bytes + _RECORD_ROW_BYTES * rows + 4096,
    )


def _require_retained_bytes(retained_bytes: int, limits: MeshingLimits, /) -> None:
    if retained_bytes > limits.maximum_data_bytes:
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "publication.retained_data_bytes must not exceed limits.maximum_data_bytes; "
            f"got {retained_bytes} > {limits.maximum_data_bytes}",
            stage=MeshingStageKind.CANONICALIZATION.value,
            requested=(("maximum_data_bytes", limits.maximum_data_bytes),),
            achieved=(("retained_data_bytes", retained_bytes),),
        )


def _require_retained_data(result: object, limits: MeshingLimits, /) -> None:
    _require_retained_bytes(model_array_bytes(result), limits)


@checked
def publish_native_result(
    mesh: cell_mesh_owner.CellMesh,
    coordinate_contract: SpatialCoordinateContract,
    compliance: MeshingComplianceReport,
    construction: tuple[MeshingStageReport, ...],
    provider: MeshingProviderInfo,
    provenance: Mapping[str, Any],
    certification: NativeCertificationRequest,
    /,
    *,
    policy: NativePublicationPolicy,
    organization: NativePublicationOrganization = NativePublicationOrganization(),
    execution: NativePublicationExecution,
    geometry: cell_geometry_owner.CellGeometrySpec | None = None,
    covered_coordinate_storage: bool = False,
) -> CellMeshingResult:
    """Audit, certify, and publish a mesh with its supplied source geometry.

    Failed certification raises ``MeshingFailure`` with its entities and quantities.
    """
    scope, workspace = execution.scope, execution.workspace
    budget = (
        coordinate_budget(certification.limits)
        if execution.budget is None
        else execution.budget
    )
    request = _PublicationRequest(
        mesh,
        coordinate_contract,
        compliance,
        construction,
        provider,
        provenance,
        certification,
        policy,
        organization,
        execution,
        budget,
    )
    with (
        (
            budget.retained_storage(lambda size: scope.batch(0, bytes_upper=size))
            if budget.storage_ticket is None
            else nullcontext()
        ),
        (
            workspace.cover_retained_storage(budget.retained_owner_objects)
            if covered_coordinate_storage
            else nullcontext()
        ),
    ):
        workspace.retain_owner(
            (
                mesh,
                geometry,
                organization.boundary,
                compliance,
                construction,
                certification,
                organization.patches,
                organization.zones,
                organization.labels,
                organization.attributes,
                organization.associations,
                organization.region_evidence,
                organization.region_boundary_evidence,
                organization.surface_source,
            )
        )
        native_volume_checkpoint(
            certification.limits, MeshingStageKind.SPECIFICATION_COMPLIANCE, scope=scope
        )
        require_compliance(compliance)
        published_entity_limits(mesh, certification.limits)
        prepared = _prepare_publication(request, geometry)
        audited = _audit_publication(request, prepared)
        certified = _certify_publication(request, prepared, audited)
        with measure_phase(execution.record_phase, "publication"):
            result = _assemble_publication(request, prepared, audited, certified).result
        native_volume_checkpoint(
            certification.limits, MeshingStageKind.CANONICALIZATION, scope=scope
        )
        return result


@checked
def bind_native_execution_result(
    result: CellMeshingResult,
    evidence: NativeExecutionEvidence,
    limits: MeshingLimits,
    started: float,
    /,
    *,
    source_work_units: int = 0,
    source_geometry_queries: int = 0,
    preparation_seconds: float = 0.0,
    preparation_evidence: NativeExecutionRecord | None = None,
    history_evidence: tuple[NativeExecutionRecord, ...] = (),
) -> CellMeshingResult:
    """Bind completed execution-scope measurements without copying scientific arrays.

    ``preparation_evidence`` is this request's charged chain; ``history_evidence``
    names completed earlier requests retained as provenance only.
    """
    _require_retained_bytes(
        model_array_bytes((result, preparation_evidence, history_evidence))
        + _NATIVE_EXECUTION_RECORD_ARRAY_BYTES
        - (
            np.dtype(np.uint64).itemsize
            if evidence.managed_admission_peak_bytes_upper is None
            else 0
        ),
        limits,
    )
    record = NativeExecutionRecord(
        evidence,
        source_preparation_work_units=source_work_units
        if preparation_evidence is None
        else None,
        source_preparation_geometry_queries=source_geometry_queries
        if preparation_evidence is None
        else None,
        preparation_seconds=preparation_seconds if preparation_evidence is None else None,
        preparation_evidence=preparation_evidence,
        consumer_evidence=result.execution_evidence,
        history_evidence=history_evidence,
        owner_id=(
            preparation_evidence.owner_id
            if preparation_evidence is not None
            else (
                result.execution_evidence.owner_id
                if result.execution_evidence is not None
                else result.result_id
            )
        ),
    )
    check_deadline(started, limits, MeshingStageKind.CERTIFICATION)
    bound = result.with_execution_evidence(record)

    _require_retained_data(bound, limits)
    check_deadline(started, limits, MeshingStageKind.CANONICALIZATION)
    return bound


__all__ = [
    "NativeCertificationRequest",
    "NativePublicationOrganization",
    "NativePublicationExecution",
    "publish_native_result",
]
