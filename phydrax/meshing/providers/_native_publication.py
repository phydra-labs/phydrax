#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Shared acceptance and publication of native meshing routes.

Every native route constructs a canonical ``CellMesh`` with its actual
coordinate geometry, organization and geometry associations, measures compliance,
and publishes through `publish_native_result`: the mesh is audited under the
route's declared output checks, then certified under the route's
`MeshCertificationSchedule` (global embedding plus the route's domain coverage
or source fidelity), and a `CellMeshingResult` exists only when the audit, the
certification, and compliance all pass. Failures raise `MeshingFailure` with
stage, entity, and requested-versus-achieved evidence.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from time import monotonic
from typing import Any

import numpy as np
from jax.typing import ArrayLike

from ..._identity import SemanticProvenance
from ..._meshcore import NativeExecutionEvidence
from ..._physical import SpatialCoordinateContract
from ...discretization import CellGeometrySpec, CellMesh
from ...discretization._cell_geometry_validity import cell_geometry_id, CellValidityStatus
from ...discretization._reference_cell import reference_cell_topology
from ...geometry._mapped_reference_domain import MappedReferenceDomain
from ...geometry._mesh_certificates import (
    MeshCertificateLimits,
    PiecewiseLinearDomain,
    SourceBoundaryQuery,
)
from ...geometry._surface_source_support import SurfaceSourceCharts
from ...geometry.surface import SurfaceModel
from .._association import GeometryAssociation, GeometryAssociationKind
from .._audit import audit_cell_mesh, CellMeshAuditPolicy, CellMeshAuditReport
from .._certification import (
    acceptance_stage_report,
    certify_meshing_acceptance,
    MeshCertificationPreparedEvidence,
    MeshCertificationReport,
    MeshCertificationSchedule,
)
from .._certification_inputs import MeshCertificationInputs
from .._contracts import (
    MeshingDerivativeMode,
    MeshingExecutionMode,
    MeshingFailure,
    MeshingFailureCategory,
    MeshingLimits,
    MeshingProviderInfo,
)
from .._measurements import (
    measure_phase,
    NativeExecutionRecord,
    NativeMeshingPhaseRecorder,
    phase_started,
    record_elapsed,
)
from .._organization import (
    MeshAttribute,
    MeshLabel,
    MeshPatch,
    MeshZone,
    RegionBoundaryEvidence,
    RegionMeshingEvidence,
)
from .._result import CellMeshingResult, MeshingComplianceReport, MeshingRuntimeInfo
from .._sizing import SizeCompliancePolicy, SizeControlStrength, UniformSizeControl
from .._trace import (
    MeshingEvidenceBinding,
    MeshingStageKind,
    MeshingStageReport,
    MeshingStageStatus,
    MeshingTrace,
)


def check_deadline(
    started: float, limits: MeshingLimits, stage: MeshingStageKind, /
) -> None:
    """Refuse publication once the enforced wall-time budget has elapsed."""

    elapsed = monotonic() - started
    if elapsed > limits.maximum_wall_seconds:
        raise MeshingFailure(
            MeshingFailureCategory.TIMED_OUT,
            "Native meshing exceeded its enforced wall-time limit.",
            stage=stage.value,
            requested=(("maximum_wall_seconds", limits.maximum_wall_seconds),),
            achieved=(("elapsed_seconds", elapsed),),
        )


def unique_edges(cells: np.ndarray, cell_kind: str, /) -> np.ndarray:
    """Canonical edge pairs of one explicitly declared reference-cell family."""
    topology = reference_cell_topology(cell_kind)
    if cells.ndim != 2 or cells.shape[1] != len(topology.vertices):
        raise ValueError("Cell rows must match their declared reference topology.")
    local = np.asarray(topology.entities[1], dtype=np.int32)
    pairs = cells[:, local].reshape((-1, 2))
    return np.unique(np.sort(pairs, axis=1), axis=0)


def edge_size_evidence(
    vertices: np.ndarray, edges: np.ndarray, /
) -> tuple[np.ndarray, float]:
    """Edge lengths and the largest ratio of incident edge lengths at a vertex."""

    lengths = np.linalg.norm(vertices[edges[:, 1]] - vertices[edges[:, 0]], axis=1)
    return lengths, edge_growth_evidence(lengths, edges, vertices.shape[0])


def edge_growth_evidence(
    lengths: np.ndarray, edges: np.ndarray, vertex_count: int, /
) -> float:
    """Incident-edge growth of an actual finite positive physical length bank."""
    if not lengths.size or np.any(~np.isfinite(lengths)) or np.any(lengths <= 0.0):
        raise MeshingFailure(
            MeshingFailureCategory.COMPLIANCE_FAILED,
            "Native meshing produced no finite positive edge-size evidence.",
            stage=MeshingStageKind.SPECIFICATION_COMPLIANCE.value,
        )
    minimum = np.full((vertex_count,), np.inf, dtype=np.float64)
    maximum = np.zeros((vertex_count,), dtype=np.float64)
    np.minimum.at(minimum, edges[:, 0], lengths)
    np.minimum.at(minimum, edges[:, 1], lengths)
    np.maximum.at(maximum, edges[:, 0], lengths)
    np.maximum.at(maximum, edges[:, 1], lengths)
    active = np.isfinite(minimum) & (minimum > 0.0)
    return float(np.max(maximum[active] / minimum[active], initial=1.0))


def uniform_size_compliance(
    control: UniformSizeControl,
    policy: SizeCompliancePolicy,
    lengths: np.ndarray,
    growth: float,
    /,
) -> tuple[list[tuple[str, float]], list[tuple[str, float]], list[str]]:
    """Requested, achieved, and failed quantities of one uniform size control.

    Soft controls are recorded; hard controls fail when a requested target
    statistic, bound, or growth rate is missed beyond the compliance tolerance.
    """

    key = f"size:{control.control_id}"
    requested = [(f"{key}:target_size", control.target_size)]
    optional = (
        ("minimum_size", control.minimum_size),
        ("maximum_size", control.maximum_size),
        ("maximum_growth_rate", control.maximum_growth_rate),
    )
    requested.extend(
        (f"{key}:{name}", value) for name, value in optional if value is not None
    )
    achieved = [
        (f"{key}:minimum_edge", float(np.min(lengths))),
        (f"{key}:maximum_edge", float(np.max(lengths))),
        (f"{key}:maximum_local_edge_ratio", growth),
    ]
    statistics: dict[str, float] = {}
    for statistic in policy.target_statistics:
        quantile = {"p50": 0.5, "p95": 0.95}[statistic]
        value = float(np.quantile(lengths, quantile))
        statistics[statistic] = value
        achieved.append((f"{key}:{statistic}_edge", value))
    issues: list[str] = []
    if control.strength is SizeControlStrength.HARD:
        tolerance = policy.tolerance

        for statistic, value in statistics.items():
            if abs(value - control.target_size) > tolerance(control.target_size):
                issues.append(f"target_size_{statistic}:{control.control_id}")
        if control.minimum_size is not None and float(
            np.min(lengths)
        ) < control.minimum_size - tolerance(control.minimum_size):
            issues.append(f"minimum_size:{control.control_id}")
        if control.maximum_size is not None and float(
            np.max(lengths)
        ) > control.maximum_size + tolerance(control.maximum_size):
            issues.append(f"maximum_size:{control.control_id}")
        if (
            control.maximum_growth_rate is not None
            and growth
            > control.maximum_growth_rate + tolerance(control.maximum_growth_rate)
        ):
            issues.append(f"maximum_growth_rate:{control.control_id}")
    return requested, achieved, issues


def require_compliance(compliance: MeshingComplianceReport, /) -> None:
    """Refuse a construction that misses a hard request, with its evidence."""

    if not compliance.passed:
        raise MeshingFailure(
            MeshingFailureCategory.COMPLIANCE_FAILED,
            "; ".join(compliance.issues),
            stage=MeshingStageKind.SPECIFICATION_COMPLIANCE.value,
            requested=compliance.requested,
            achieved=compliance.achieved,
        )


def simplex_entity_limits(
    vertices: np.ndarray,
    cells: np.ndarray,
    limits: MeshingLimits,
    stage: MeshingStageKind,
    /,
    *,
    cell_kind: str,
) -> None:
    """Refuse constructed simplices beyond the entity and data budgets.

    Runs on the host construction arrays before any mesh carrier is built.
    """

    edges = unique_edges(cells, cell_kind).shape[0]
    topology = reference_cell_topology(cell_kind)
    faces = 0
    if topology.dimension >= 2:
        for width in sorted({len(face) for face in topology.entities[2]}):
            local = np.asarray(
                tuple(face for face in topology.entities[2] if len(face) == width),
                dtype=np.int32,
            )
            rows = cells[:, local].reshape((-1, width))
            faces += np.unique(np.sort(rows, axis=1), axis=0).shape[0]
    checks = (
        ("vertices", vertices.shape[0], limits.maximum_vertices),
        ("edges", edges, limits.maximum_edges),
        ("faces", faces, limits.maximum_faces),
        ("cells", cells.shape[0], limits.maximum_cells),
        ("connectivity_entries", cells.size, limits.maximum_connectivity_entries),
        ("data_bytes", vertices.nbytes + cells.nbytes, limits.maximum_data_bytes),
    )
    exceeded = tuple(name for name, count, limit in checks if count > limit)
    if exceeded:
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "Native meshing exceeded its declared budgets: " + ", ".join(exceeded) + ".",
            stage=stage.value,
            requested=tuple((f"maximum_{name}", limit) for name, _, limit in checks),
            achieved=tuple((name, count) for name, count, _ in checks),
        )


@dataclass(frozen=True, slots=True)
class NativeCertificationRequest:
    """Route acceptance request: schedule, evidence inputs, and source identity.

    ``domain``/``cell_regions`` feed domain coverage, ``fidelity_source`` and
    ``fidelity_tolerance`` source fidelity, and ``junction_vertices`` declares
    curve-network junctions. Every scheduled check must certify; unresolved
    or violated evidence refuses publication.
    ``prepared`` retains actual positive volume premises; its complete bindings
    are validated before reuse, including the unchanged certificate limits.
    Authored revisions come from actual owning source records or every source
    association row, never from the distinct numerical domain revision.
    """

    schedule: MeshCertificationSchedule
    source_id: str
    source_revision: str
    limits: MeshingLimits
    domain: PiecewiseLinearDomain | MappedReferenceDomain | None = None
    cell_regions: np.ndarray | None = None
    fidelity_source: SourceBoundaryQuery | None = None
    fidelity_tolerance: float | None = None
    junction_vertices: ArrayLike | None = None
    prepared: MeshCertificationPreparedEvidence | None = None
    scoped_fidelity: tuple[tuple[SourceBoundaryQuery, ArrayLike, float], ...] = ()

    @property
    def certificate_limits(self) -> MeshCertificateLimits:
        """Lower original native work/query/scratch caps without expanding them."""
        defaults = MeshCertificateLimits()
        limits = self.limits
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
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    audit: CellMeshAuditReport,
    request: NativeCertificationRequest,
    /,
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
        prepared=request.prepared,
        scoped_fidelity=request.scoped_fidelity,
    )
    stage = acceptance_stage_report(report)
    if report.passed:
        return report, stage
    report.require_passed()
    raise RuntimeError("An unpassed certification report did not refuse publication.")


def _published_entity_limits(mesh: CellMesh, limits: MeshingLimits, /) -> None:
    """Admit every canonical cell and oriented-incidence index buffer."""
    dimension = mesh.topological_dimension
    entries = sum(block.vertices.size for block in mesh.blocks) + sum(
        incidence.relation.source_indices.size + incidence.relation.target_indices.size
        for incidence in mesh.topology.incidences
    )
    counts = (
        ("vertices", mesh.coordinates.shape[0], limits.maximum_vertices),
        (
            "edges",
            mesh.entity_set(1).entity_ids.size if dimension >= 1 else 0,
            limits.maximum_edges,
        ),
        (
            "faces",
            mesh.entity_set(2).entity_ids.size if dimension >= 2 else 0,
            limits.maximum_faces,
        ),
        (
            "cells",
            sum(block.vertices.shape[0] for block in mesh.blocks),
            limits.maximum_cells,
        ),
        ("connectivity_entries", entries, limits.maximum_connectivity_entries),
    )
    if any(actual > maximum for _, actual, maximum in counts):
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "Canonical native publication exceeds its complete entity or connectivity budget.",
            stage=MeshingStageKind.CANONICALIZATION.value,
            requested=tuple((f"maximum_{name}", maximum) for name, _, maximum in counts),
            achieved=tuple((name, actual) for name, actual, _ in counts),
        )


def publish_native_result(
    mesh: CellMesh,
    coordinate_contract: SpatialCoordinateContract,
    compliance: MeshingComplianceReport,
    construction: tuple[MeshingStageReport, ...],
    provider: MeshingProviderInfo,
    provenance: Mapping[str, Any],
    certification: NativeCertificationRequest,
    /,
    *,
    audit_policy: CellMeshAuditPolicy,
    derivative_mode: MeshingDerivativeMode,
    enforced_limits: tuple[str, ...],
    unenforced_limits: tuple[str, ...],
    boundary: SurfaceModel | None = None,
    patches: tuple[MeshPatch, ...] = (),
    zones: tuple[MeshZone, ...] = (),
    labels: tuple[MeshLabel, ...] = (),
    attributes: tuple[MeshAttribute, ...] = (),
    associations: tuple[GeometryAssociation, ...] = (),
    region_evidence: RegionMeshingEvidence | None = None,
    region_boundary_evidence: tuple[RegionBoundaryEvidence, ...] = (),
    record_phase: NativeMeshingPhaseRecorder | None = None,
    geometry: CellGeometrySpec | None = None,
    operation_started: float | None = None,
    surface_source: SurfaceSourceCharts | None = None,
) -> CellMeshingResult:
    """Audit and certify one constructed mesh without discarding supplied geometry.

    A failed certification raises ``MeshingFailure`` (``AUDIT_FAILED``, stage
    ``certification``) with the failing entities and quantities.
    """

    from ..._meshcore import current_native_host_workspace
    from .._volume_generation import native_volume_checkpoint

    workspace = current_native_host_workspace()
    if workspace is not None:
        workspace.retain_owner(
            (
                mesh,
                geometry,
                boundary,
                compliance,
                construction,
                certification,
                patches,
                zones,
                labels,
                attributes,
                associations,
                region_evidence,
                region_boundary_evidence,
                surface_source,
            )
        )

    native_volume_checkpoint(
        certification.limits,
        MeshingStageKind.SPECIFICATION_COMPLIANCE,
        operation_started=operation_started,
    )
    require_compliance(compliance)
    _published_entity_limits(mesh, certification.limits)
    if "data_bytes" not in enforced_limits:
        enforced_limits = (*enforced_limits, "data_bytes")
    unenforced_limits = tuple(
        value for value in unenforced_limits if value != "data_bytes"
    )
    with measure_phase(record_phase, "publication"):
        native_volume_checkpoint(
            certification.limits,
            MeshingStageKind.CANONICALIZATION,
            operation_started=operation_started,
        )
        geometry = CellGeometrySpec.affine(mesh) if geometry is None else geometry
        if not isinstance(geometry, CellGeometrySpec):
            raise TypeError("geometry must be CellGeometrySpec or None.")
        if workspace is not None:
            workspace.retain_owner(geometry)
        if certification.prepared is not None:
            if not isinstance(certification.prepared, MeshCertificationPreparedEvidence):
                raise TypeError(
                    "prepared must be MeshCertificationPreparedEvidence or None."
                )
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
            assert bound_domain is not None
            if bound_source is not None:
                if (
                    certification.source_id != bound_source.source_id
                    or certification.source_revision != bound_source.source_revision
                ):
                    raise ValueError(
                        "Prepared publication differs from its owning source authority."
                    )
            elif associations:
                if not all(
                    isinstance(value, GeometryAssociation) for value in associations
                ):
                    raise TypeError(
                        "associations must contain GeometryAssociation values."
                    )
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
                        "Prepared publication differs from its actual source association authority."
                    )
            else:
                raise ValueError(
                    "Prepared publication requires actual owning source authority evidence."
                )
    with measure_phase(record_phase, "audit"):
        native_volume_checkpoint(
            certification.limits,
            MeshingStageKind.GEOMETRY_AUDIT,
            operation_started=operation_started,
        )
        audit = audit_cell_mesh(
            mesh,
            geometry,
            policy=audit_policy,
            prepared_validity=None
            if certification.prepared is None
            else certification.prepared.validity,
            boundary=boundary,
            patches=patches,
            associations=associations,
            zones=zones,
            labels=labels,
            attributes=attributes,
        )
    native_volume_checkpoint(
        certification.limits,
        MeshingStageKind.GEOMETRY_AUDIT,
        operation_started=operation_started,
    )
    if not audit.passed:
        uncertified = (
            np.asarray(audit.validity.status) != CellValidityStatus.CERTIFIED_VALID
        )
        cell_ids = np.concatenate(
            [np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks]
        )
        failure_policy = CellMeshAuditPolicy() if audit_policy is None else audit_policy
        evaluation = audit.quality.evaluation
        quality_failed = (
            (np.asarray(evaluation.measures) <= failure_policy.minimum_measure)
            | (np.asarray(evaluation.mean_ratios) < failure_policy.minimum_mean_ratio)
            | (np.asarray(evaluation.aspect_ratios) > failure_policy.maximum_aspect_ratio)
        )
        offending = tuple(int(value) for value in cell_ids[uncertified | quality_failed])
        if not offending:
            offending = audit.quality.worst_cell_global_ids
        selected = set(offending)
        points = np.asarray(mesh.coordinates, dtype=np.float64)
        locations = tuple(
            tuple(float(value) for value in np.mean(points[vertices[valid]], axis=0))
            for block in mesh.blocks
            for identifier, vertices, valid in zip(
                np.asarray(block.global_ids).tolist(),
                np.asarray(block.vertices),
                np.asarray(block.vertex_valid, dtype=np.bool_),
                strict=True,
            )
            if identifier in selected
        )
        raise MeshingFailure(
            MeshingFailureCategory.AUDIT_FAILED,
            "; ".join(audit.issues)
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
                ("audit:validity:determinant_lower", audit.validity.determinant_lower),
                ("audit:validity:determinant_upper", audit.validity.determinant_upper),
                ("audit:validity:status", audit.validity.status),
            ),
        )
    with measure_phase(record_phase, "certification"):
        native_volume_checkpoint(
            certification.limits,
            MeshingStageKind.CERTIFICATION,
            operation_started=operation_started,
        )
        report, certification_stage = _certify(mesh, geometry, audit, certification)
    native_volume_checkpoint(
        certification.limits,
        MeshingStageKind.CERTIFICATION,
        operation_started=operation_started,
    )
    publication_started = phase_started(record_phase)
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
    native_volume_checkpoint(
        certification.limits,
        MeshingStageKind.CANONICALIZATION,
        operation_started=operation_started,
    )
    result = CellMeshingResult(
        mesh,
        geometry,
        coordinate_contract,
        audit,
        audit.quality,
        compliance,
        MeshingTrace(stages, binding=binding),
        provider,
        runtime,
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
    from ..._model._structure import model_array_bytes

    retained_bytes = model_array_bytes(result)
    if retained_bytes > certification.limits.maximum_data_bytes:
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "Native publication exceeds its complete retained scientific-data budget.",
            stage=MeshingStageKind.CANONICALIZATION.value,
            requested=(
                ("maximum_data_bytes", float(certification.limits.maximum_data_bytes)),
            ),
            achieved=(("retained_data_bytes", float(retained_bytes)),),
        )
    record_elapsed(record_phase, "publication", publication_started)
    native_volume_checkpoint(
        certification.limits,
        MeshingStageKind.CANONICALIZATION,
        operation_started=operation_started,
    )
    return result


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
) -> CellMeshingResult:
    """Bind completed original-scope measurements without copying scientific arrays."""
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
    original_started = started - evidence.prior_elapsed_seconds
    check_deadline(original_started, limits, MeshingStageKind.CERTIFICATION)
    bound = result.with_execution_evidence(record)
    from ..._model._structure import model_array_bytes

    retained_bytes = model_array_bytes(bound)
    if retained_bytes > limits.maximum_data_bytes:
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "Native publication exceeds its complete retained scientific-data budget.",
            stage=MeshingStageKind.CANONICALIZATION.value,
            requested=(("maximum_data_bytes", limits.maximum_data_bytes),),
            achieved=(("retained_data_bytes", retained_bytes),),
        )
    check_deadline(original_started, limits, MeshingStageKind.CANONICALIZATION)
    return bound


__all__ = [
    "NativeCertificationRequest",
    "check_deadline",
    "edge_size_evidence",
    "publish_native_result",
    "require_compliance",
    "simplex_entity_limits",
    "uniform_size_compliance",
    "unique_edges",
]
