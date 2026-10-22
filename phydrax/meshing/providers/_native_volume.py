#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Generate constrained PLC tetrahedra and publish their acceptance evidence.

The route admits a `NativePlcSource` with a `VolumeMeshingSpec`, recovers its
fixed or conforming boundary, classifies regions, runs the scheduled refinement
and improvement, and constructs organization and source associations.
Publication requires size-compliance evidence and independent ``volume_plc``
checks for per-cell validity, topology, global embedding, and coverage of the
declared source polygons and region measures.
"""

from __future__ import annotations

from typing import final

import equinox as eqx
import numpy as np

import phydrax.discretization._cell_complex as cell_complex_owner
from phydrax.meshing._certification_schedule import MeshCertificationSchedule
from phydrax.meshing._contracts import (
    MeshingProviderInfo,
    VolumeFillStrategy,
    VolumeMeshingSpec,
)
from phydrax.meshing._failures import MeshingFailure, MeshingFailureCategory
from phydrax.meshing._host_banks import entity_key
from phydrax.meshing.providers._native_route_policy import publication_policy

from ..._fingerprint import canonical_fingerprint
from ..._meshcore import (
    NativeBatchTicket,
    NativeExecutionScope,
    NativeHostStorageWorkspace,
)
from ..._physical import SpatialCoordinateContract
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization._coordinate_budget import CoordinateEnclosureBudget
from ...discretization._coordinate_operands import (
    CoordinateOperandCollector,
    PlcSourceCoordinateOperands,
    PlcSourceInputWitness,
    seal_coordinate_operand_array,
)
from ...geometry._domain_coverage import PiecewiseLinearDomain
from ...optim import OptimizationStatus
from ...typing import checked
from .._audit import CellMeshAuditDisposition, CellMeshAuditPolicy
from .._controls import FeatureKind, RegionControl
from .._measurements import NativeMeshingPhaseRecorder
from .._native_preparation import native_preparation_ledger
from .._result import CellMeshingResult, MeshingComplianceReport
from .._sizing import UniformSizeControl
from .._tetra_metric import MetricRemeshingEvidence
from .._trace import MeshingStageKind, MeshingStageReport, MeshingStageStatus
from .._volume_generation import (
    declared_plc_domain,
    generate_plc_volume,
    native_volume_coordinate_budget,
    NativeVolumeSchedule,
    require_native_preparation_allowance,
    VolumeConstruction,
)
from . import _native_plc_geometry as plc_geometry_owner, _native_sources as sources_owner
from ._native_publication import (
    NativeCertificationRequest,
    NativePublicationExecution,
    NativePublicationOrganization,
    coordinate_phase,
    publish_native_result,
)
from ._native_route_checks import (
    check_deadline,
    edge_size_evidence,
    simplex_entity_limits,
    uniform_size_compliance,
    unique_edges,
)


@final
class _VolumeSupportPhase(StrictModule):
    issues: tuple[str, ...] = eqx.field(static=True)


def _plc_request_support(
    source: sources_owner.NativePlcSource | sources_owner.NativePolyhedralSource,
    specification: VolumeMeshingSpec,
    /,
) -> _VolumeSupportPhase:
    """Check the tetrahedral target, all boundary facets, and PLC size controls."""
    unsupported: list[str] = []
    target = specification.target
    families = target.cell_families
    complex_ = source.complex
    if set((*families.required, *families.preferred)) != {"tetrahedron"}:
        unsupported.append("a tetrahedral volume target")
    if (
        target.topological_dimension != 3
        or target.ambient_dimension != 3
        or target.geometry_order != 1
    ):
        unsupported.append("affine tetrahedra in ambient dimension three")
    if families.allowed_transitions or families.allow_mixed:
        unsupported.append("mixed-cell transition policies")
    if specification.fill_strategy is not VolumeFillStrategy.SIMPLEX:
        unsupported.append("the simplex fill strategy")
    boundary = specification.boundary_scope
    if boundary.entity_dimension != 2 or not np.array_equal(
        np.sort(np.asarray(boundary.entity_ids)), np.arange(complex_.facet_count)
    ):
        unsupported.append("a boundary scope of every PLC facet")
    for control in specification.size_controls:
        match control:
            case UniformSizeControl():
                if control.scope.entity_dimension != 2 or np.any(
                    np.asarray(control.scope.entity_ids) >= complex_.facet_count
                ):
                    unsupported.append("uniform size controls bound to PLC facets")
            case _:
                unsupported.append("nonuniform curvature/proximity PLC size fields")
    return _VolumeSupportPhase(tuple(unsupported))


def _plc_feature_support(
    source: sources_owner.NativePlcSource | sources_owner.NativePolyhedralSource,
    specification: VolumeMeshingSpec,
    /,
) -> _VolumeSupportPhase:
    """Check protected entity incidence and region seed names."""
    unsupported: list[str] = []
    complex_ = source.complex
    counts = {
        0: complex_.vertices.shape[0],
        1: complex_.segments.shape[0],
        2: complex_.facet_count,
    }
    for feature in specification.protected_features:
        dimension = feature.scope.entity_dimension
        identifiers = np.asarray(feature.scope.entity_ids)
        if dimension not in counts or np.any(identifiers >= counts[dimension]):
            unsupported.append(
                "protected features outside the PLC vertices, curves and facets"
            )
        elif feature.feature_kind is FeatureKind.MATERIAL_INTERFACE and dimension != 2:
            unsupported.append("material interfaces that are not PLC facets")
    names = set(complex_.region_ids)
    if any(seed.region_name not in names for seed in specification.region_seeds):
        unsupported.append("region seeds naming regions of the complex")
    return _VolumeSupportPhase(tuple(unsupported))


def _plc_region_support(
    source: sources_owner.NativePlcSource | sources_owner.NativePolyhedralSource,
    specification: VolumeMeshingSpec,
    /,
) -> _VolumeSupportPhase:
    """Check enabled PLC regions and consistent material/role declarations."""
    unsupported: list[str] = []
    complex_ = source.complex
    names = set(complex_.region_ids)
    region_requests: dict[str, RegionControl] = {}
    for control in specification.region_controls:
        identifiers = np.asarray(control.scope.entity_ids)
        if (
            control.scope.entity_dimension != 3
            or control.region_name not in names
            or not np.array_equal(
                identifiers, [complex_.region_ids.index(control.region_name)]
            )
        ):
            unsupported.append("region controls bound to their declared PLC region")
            continue
        if not control.meshing_enabled:
            unsupported.append("disabled regions declared as void in the PLC incidence")
        previous = region_requests.get(control.region_name)
        if previous is not None and (
            previous.material_id != control.material_id or previous.role != control.role
        ):
            unsupported.append("contradictory region material/role controls")
        region_requests[control.region_name] = control
        for seed in specification.region_seeds:
            if seed.region_name == control.region_name and (
                seed.material_id != control.material_id or seed.role != control.role
            ):
                unsupported.append("contradictory region seed material/role controls")
    return _VolumeSupportPhase(tuple(unsupported))


def _plc_patch_support(
    source: sources_owner.NativePlcSource | sources_owner.NativePolyhedralSource,
    specification: VolumeMeshingSpec,
    /,
) -> _VolumeSupportPhase:
    """Check requested facet scopes and their PLC region adjacency."""
    unsupported: list[str] = []
    complex_ = source.complex
    for control in specification.patch_controls:
        identifiers = np.asarray(control.scope.entity_ids)
        if control.scope.entity_dimension != 2 or np.any(
            (identifiers < 0) | (identifiers >= complex_.facet_count)
        ):
            unsupported.append("patch controls bound to PLC facets")
            continue
        for facet in identifiers.tolist():
            adjacency = {
                complex_.region_ids[int(region)]
                for region in complex_.facet_regions[facet]
                if region >= 0
            }
            if adjacency != set(control.adjacent_region_names):
                unsupported.append("patch control adjacency matching the PLC incidence")
    return _VolumeSupportPhase(tuple(unsupported))


@checked
def volume_support_issues(
    source: sources_owner.NativePlcSource | sources_owner.NativePolyhedralSource,
    specification: VolumeMeshingSpec,
    /,
) -> list[str]:
    """Return unmet PLC volume requirements in validation order."""
    unsupported = [
        *_plc_request_support(source, specification).issues,
        *_plc_feature_support(source, specification).issues,
        *_plc_region_support(source, specification).issues,
        *_plc_patch_support(source, specification).issues,
    ]
    if specification.periodic_constraints:
        unsupported.append("periodic constraints")
    if specification.layer_controls:
        unsupported.append("boundary-layer controls")
    return unsupported


@final
class PreparedPlcVolume(StrictModule, NonTrainableState):
    """Admitted PLC volume request and its numerical schedule."""

    schedule: NativeVolumeSchedule
    source_binding_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        source: sources_owner.NativePlcSource,
        specification: VolumeMeshingSpec,
        schedule: NativeVolumeSchedule,
        /,
    ) -> None:
        self.schedule = schedule
        self.source_binding_id = source.binding_id
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "native-plc-volume",
                "source": source.binding_id,
                "specification": specification.specification_id,
                "schedule": schedule.schedule_id,
            }
        )


def _append_metric_evidence(
    metric: MetricRemeshingEvidence,
    requested: list[tuple[str, float]],
    achieved: list[tuple[str, float]],
    /,
) -> None:
    """Publish the optimization goal, outcome, work, and native resources."""
    guards = (
        ("metric_quality_floor", metric.metric_quality_floor),
        ("shape_radius_edge_bound", metric.shape_radius_edge_bound),
        ("shape_minimum_dihedral_degrees", metric.shape_minimum_dihedral_degrees),
    )
    requested.extend(
        (f"metric_optimization:{name}", value)
        for name, value in guards
        if value is not None
    )
    requested.extend(
        (f"metric_optimization:resource:{name}", value)
        for name, value in metric.resource_requested
    )
    achieved.extend(
        (f"metric_optimization:resource:{name}", value)
        for name, value in metric.resource_achieved
    )
    requested.extend(
        (f"metric_optimization:{name}", value) for name, value in metric.size_requested
    )
    achieved.extend(
        (
            (f"metric_optimization:criterion:{metric.criterion.value}", 1.0),
            (f"metric_optimization:status:{metric.status.value}", 1.0),
            ("metric_optimization:passes", metric.passes),
            ("metric_optimization:splits", metric.splits),
            ("metric_optimization:compound_splits", metric.compound_splits),
            ("metric_optimization:expanded_insertions", metric.expanded_insertions),
            ("metric_optimization:collapses", metric.collapses),
            ("metric_optimization:flips", metric.flips),
            ("metric_optimization:relocations", metric.relocations),
            ("metric_optimization:rejected_operations", metric.rejected_operations),
            ("metric_optimization:operation_attempts", metric.operation_attempts),
            ("metric_optimization:work_units", metric.work_units),
            ("metric_optimization:source_candidate_pairs", metric.source_candidate_pairs),
            ("metric_optimization:lower_metric_length", metric.lower_metric_length),
            ("metric_optimization:upper_metric_length", metric.upper_metric_length),
            ("metric_optimization:minimum_metric_length", metric.minimum_metric_length),
            ("metric_optimization:maximum_metric_length", metric.maximum_metric_length),
            ("metric_optimization:minimum_metric_quality", metric.minimum_metric_quality),
            ("metric_optimization:maximum_fidelity_bound", metric.maximum_fidelity_bound),
        )
    )
    achieved.extend(
        (f"metric_optimization:{name}", value) for name, value in metric.size_achieved
    )
    achieved.extend(
        (f"metric_optimization:issue:{issue}", 1.0) for issue in metric.size_issues
    )
    if metric.memory.native_counters is not None:
        achieved.extend(
            (f"metric_optimization:native_memory:{name}", value)
            for name, value in metric.memory.native_counters.items()
        )
    if metric.metric_arc_bounds is not None:
        bounds = np.asarray(metric.metric_arc_bounds, dtype=np.float64)
        achieved.extend(
            (
                ("metric_optimization:arc_minimum_lower", float(np.min(bounds[:, 0]))),
                ("metric_optimization:arc_maximum_upper", float(np.max(bounds[:, 1]))),
                (
                    "metric_optimization:arc_maximum_width",
                    float(np.max(bounds[:, 1] - bounds[:, 0])),
                ),
            )
        )
    _append_optimizer_evidence(metric, requested, achieved)


def _append_optimizer_evidence(
    metric: MetricRemeshingEvidence,
    requested: list[tuple[str, float]],
    achieved: list[tuple[str, float]],
    /,
) -> None:
    """Retain solver termination and charged host proposal resources."""
    if metric.optimizer.method_id is None:
        return
    achieved.extend(
        (
            (f"metric_optimization:optimizer:{metric.optimizer.method_id}", 1.0),
            ("metric_optimization:native_work_units", metric.native_work_units),
            ("metric_optimization:optimizer_iterations", metric.optimizer.iterations),
            (
                "metric_optimization:optimizer_objective_evaluations",
                metric.optimizer.evaluations,
            ),
            (
                "metric_optimization:optimizer_constraint_evaluations",
                metric.optimizer.constraint_evaluations,
            ),
            (
                "metric_optimization:optimizer_proposal_evaluations",
                metric.optimizer.proposal_evaluations,
            ),
            (
                "metric_optimization:optimizer_refused_proposals",
                metric.optimizer.refused_proposals,
            ),
            (
                "metric_optimization:optimizer_host_work_units",
                metric.optimizer.host_work_units,
            ),
            (
                "metric_optimization:optimizer_host_table_bytes",
                metric.memory.optimizer_scratch_required_bytes,
            ),
        )
    )
    requested.extend(
        (f"metric_optimization:optimizer_tolerance:{index}", value)
        for index, value in enumerate(metric.optimizer.tolerances)
    )
    if metric.optimizer.status_counts:
        achieved.extend(
            (f"metric_optimization:optimizer_status:{status.name.lower()}", count)
            for status, count in zip(
                OptimizationStatus,
                metric.optimizer.status_counts,
                strict=True,
            )
        )
    for index, (steps, evaluations) in enumerate(metric.optimizer.terminations):
        requested.extend(
            (
                (f"metric_optimization:optimizer:{index}:maximum_steps", steps),
                (
                    f"metric_optimization:optimizer:{index}:maximum_evaluations",
                    evaluations,
                ),
            )
        )
    achieved.extend(
        (f"metric_optimization:optimizer_refusal:{name}", count)
        for name, count in metric.optimizer.refusal_counts
    )


@checked
def volume_compliance(
    specification: VolumeMeshingSpec,
    schedule: NativeVolumeSchedule,
    construction: VolumeConstruction,
    vertices: np.ndarray,
    cells: np.ndarray,
    /,
    *,
    scope: NativeExecutionScope | None = None,
) -> MeshingComplianceReport:
    """Enforce physical size and metric floors; retain soft shape AIM evidence."""

    requested: list[tuple[str, float]] = [
        ("radius_edge_bound", schedule.radius_edge_bound),
        ("minimum_dihedral_degrees", schedule.minimum_dihedral_degrees),
    ]
    quality = construction.quality
    if not np.isfinite(quality.maximum_radius_edge):
        raise MeshingFailure(
            MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
            "VolumeConstruction.quality.maximum_radius_edge must be finite; got unresolved tetrahedral circumcenter quality.",
            stage=MeshingStageKind.SPECIFICATION_COMPLIANCE.value,
            requested=(("radius_edge_bound", schedule.radius_edge_bound),),
            achieved=(("native_radius_edge_quality_resolved", 0),),
            provider_code="NONFINITE_NATIVE_RADIUS_EDGE_QUALITY",
        )
    achieved: list[tuple[str, float]] = [
        ("maximum_radius_edge", quality.maximum_radius_edge),
        ("minimum_dihedral_degrees", quality.minimum_dihedral),
        ("slivers", quality.slivers),
        ("steiner_points", construction.steiner_points),
        ("work_units", construction.work_units),
        *(
            (f"refinement:{name}", int(value))
            for name, value in (
                ()
                if construction.refinement is None
                else construction.refinement.counters.items()
            )
        ),
        *(
            (f"improvement:{name}", int(value))
            for name, value in (
                ()
                if construction.improvement is None
                else construction.improvement.counters.items()
            )
        ),
        *(
            ()
            if construction.exudation is None
            else (
                (f"exudation:{name}", int(value))
                for name, value in construction.exudation.counters.items()
            )
        ),
        *(
            (f"construction:{name}", value)
            for name, value in construction.construction_counters
        ),
        *(
            (f"unmet:{criterion}:{reason}", count)
            for criterion, reason, count in construction.unmet
        ),
    ]
    requested.extend(construction.aim_requested)
    achieved.extend(construction.aim_achieved)
    issues: list[str] = []
    metric = construction.metric_optimization
    if metric is not None:
        _append_metric_evidence(metric, requested, achieved)
        # A stalled low-level outcome may retain bad input cells. Merely
        # reporting its declared hard floor must not certify that output.
        if metric.metric_quality_floor is not None and (
            not np.isfinite(metric.minimum_metric_quality)
            or metric.minimum_metric_quality < metric.metric_quality_floor
        ):
            issues.append("metric_optimization:metric_quality_floor")
    for control in specification.size_controls:
        match control:
            case UniformSizeControl():
                pass
            case _:
                raise TypeError(
                    "VolumeMeshingSpec.size_controls must contain UniformSizeControl after admission; got a different size control."
                )
        if np.array_equal(
            control.scope.entity_ids, specification.boundary_scope.entity_ids
        ):
            edges = unique_edges(cells, "tetrahedron")
        else:
            mesh = construction.mesh
            face_set = mesh.entity_set(2)
            association = next(
                value
                for value in construction.associations
                if value.target_entity_set_id == face_set.entity_set_id
            )
            source_entities = {
                entity_key(mesh.numeric_version, "facet", facet)
                for facet in np.asarray(control.scope.entity_ids).tolist()
            }
            selected = np.asarray(
                [
                    entity in source_entities
                    for entity in association.source_entities.entity_ids()
                ],
                dtype=np.bool_,
            )
            identifiers = np.asarray(association.target_global_ids)[selected]
            connectivity = mesh.connectivity
            if not isinstance(connectivity, cell_complex_owner.TetrahedralConnectivity):
                raise TypeError(
                    "VolumeConstruction.mesh.connectivity must be TetrahedralConnectivity; got a different connectivity kind."
                )
            rows = np.asarray(connectivity.faces, dtype=np.int64)[
                np.isin(np.asarray(face_set.entity_ids), identifiers)
            ]
            edges = unique_edges(rows, "triangle")
        if construction.geometry is None:
            lengths, growth = edge_size_evidence(vertices, edges)
        else:
            if scope is None:
                raise ValueError(
                    "volume_compliance.geometry must share the request coordinate budget; got missing execution scope"
                )
            with native_volume_coordinate_budget(
                specification.limits, scope=scope
            ) as budget:
                lengths, growth = edge_size_evidence(
                    vertices,
                    edges,
                    geometry=construction.geometry,
                    mesh=construction.mesh,
                    budget=budget,
                )
        size_requested, size_achieved, size_issues = uniform_size_compliance(
            control, specification.size_compliance, lengths, growth
        )
        requested.extend(size_requested)
        achieved.extend(size_achieved)
        issues.extend(size_issues)
    fidelity = construction.source_fidelity
    for feature in specification.protected_features:
        key = f"protected:{feature.feature_id}:maximum_deviation"
        identifiers = np.asarray(feature.scope.entity_ids, dtype=np.int64)
        # Natively certified deviation of the published vertices on the
        # feature's PLC facets or explicit curves; PLC vertices never move.
        match feature.scope.entity_dimension:
            case 2 if fidelity is not None:
                deviation = float(
                    np.max(fidelity.facet_achieved[identifiers], initial=0.0)
                )
            case 1 if fidelity is not None:
                deviation = float(
                    np.max(fidelity.segment_achieved[identifiers], initial=0.0)
                )
            case _:
                deviation = 0.0
        requested.append((key, feature.maximum_deviation))
        achieved.append((key, deviation))
        if feature.hard and deviation > feature.maximum_deviation:
            issues.append(key)
    return MeshingComplianceReport(
        specification.specification_id,
        issues=tuple(issues),
        requested=tuple(requested),
        achieved=tuple(achieved),
    )


def plc_volume_audit_policy() -> CellMeshAuditPolicy:
    """Publication audit of native PLC volumes.

    Its validity policy also owns the native determinant-floor repair, so the
    generated cells and the independent audit share one floor.
    """

    return CellMeshAuditPolicy(
        require_complete_association=True,
        watertight_boundary=CellMeshAuditDisposition.REJECT,
    )


def _source_input_witness(
    construction: VolumeConstruction,
    source: sources_owner.NativePlcSource,
    budget: CoordinateEnclosureBudget,
    /,
) -> PlcSourceInputWitness | None:
    """Detach only actual source-prefix inputs, never carrier membership verdicts."""
    actual = construction.source_input_points
    fidelity = construction.source_fidelity
    declared = None if fidelity is None else fidelity.declared_source
    if actual is None or declared is None:
        return None
    points = source.complex.vertices
    count = points.shape[0]
    budget.reserve(3 * points.size + 4 * count)
    if (
        actual.shape != points.shape or actual.dtype != points.dtype
        or declared.points.shape != points.shape or declared.points.dtype != points.dtype
        or actual.tobytes(order="C") != points.tobytes(order="C")
        or declared.points.tobytes(order="C") != points.tobytes(order="C")
        or declared.witness_strata.shape[0] < count
        or declared.witness_entities.shape[0] < count
        or declared.witness_parameters.shape[0] < count
    ):
        raise ValueError("PLC source input witnesses require the actual unchanged source-point prefix.")
    # Native carrier witnesses are indexed by every initial/generated point.
    # Association source inputs need only the admitted original-point prefix.
    # Keep all original source strata/groups, including declared diagonals,
    # while detaching these input rows from the full carrier-sized arrays.
    return PlcSourceInputWitness(
        points=declared.points,
        faces=declared.faces,
        face_group_rows=declared.face_group_rows,
        face_tolerances=declared.face_tolerances,
        segments=declared.segments,
        segment_group_rows=declared.segment_group_rows,
        segment_tolerances=declared.segment_tolerances,
        face_ids=declared.face_ids,
        segment_ids=declared.segment_ids,
        witness_strata=seal_coordinate_operand_array(
            declared.witness_strata[:count], budget=budget, copy=True,
        ),
        witness_entities=seal_coordinate_operand_array(
            declared.witness_entities[:count], budget=budget, copy=True,
        ),
        witness_parameters=seal_coordinate_operand_array(
            declared.witness_parameters[:count], budget=budget, copy=True,
        ),
    )


@checked
def execute_volume_route(
    source: sources_owner.NativePlcSource,
    specification: VolumeMeshingSpec,
    prepared: PreparedPlcVolume,
    coordinate_contract: SpatialCoordinateContract,
    provider: MeshingProviderInfo,
    plan_id: str,
    /,
    *,
    scope: NativeExecutionScope,
    workspace: NativeHostStorageWorkspace,
    construction: VolumeConstruction | None = None,
    record_phase: NativeMeshingPhaseRecorder | None = None,
    coordinate_operands: CoordinateOperandCollector | None = None,
) -> CellMeshingResult:
    """Generate and publish in the explicit request scope."""
    if construction is not None and construction.native_execution_evidence is not None:
        preparation = construction.native_execution_record
        if preparation is None:
            raise ValueError(
                "VolumeConstruction.native_execution_record must retain ended execution evidence; got None for a supplied construction."
            )
        if prepared.source_binding_id != source.binding_id or any(
            association.source_id != source.source_id
            or association.source_revision != source.source_revision
            for association in construction.associations
        ):
            raise ValueError(
                "VolumeConstruction.associations must bind NativePlcSource.source_id and source_revision; got a different source binding."
            )
        require_native_preparation_allowance(specification.limits, preparation)
        native_preparation_ledger(scope).import_receipt(scope, preparation)
    workspace.retain_owner((source, specification, prepared, construction))
    return _execute_volume_route_bound(
        source,
        specification,
        prepared,
        coordinate_contract,
        provider,
        plan_id,
        scope=scope,
        workspace=workspace,
        construction=construction,
        record_phase=record_phase,
        coordinate_operands=coordinate_operands,
    )


def _execute_volume_route_bound(
    source: sources_owner.NativePlcSource,
    specification: VolumeMeshingSpec,
    prepared: PreparedPlcVolume,
    coordinate_contract: SpatialCoordinateContract,
    provider: MeshingProviderInfo,
    plan_id: str,
    /,
    *,
    scope: NativeExecutionScope,
    workspace: NativeHostStorageWorkspace,
    construction: VolumeConstruction | None = None,
    record_phase: NativeMeshingPhaseRecorder | None = None,
    coordinate_operands: CoordinateOperandCollector | None = None,
) -> CellMeshingResult:
    """Generate, certify, and publish within the request allowance."""

    if prepared.source_binding_id != source.binding_id:
        raise ValueError(
            "PreparedPlcVolume.source_binding_id must match NativePlcSource.binding_id; got a different source binding."
        )
    started = scope.operation_started
    limits = specification.limits
    audit_policy = plc_volume_audit_policy()
    if construction is None:
        construction = generate_plc_volume(
            source.complex,
            specification,
            prepared.schedule,
            validity_policy=audit_policy.validity_policy,
            source_id=source.source_id,
            source_revision=source.source_revision,
            input_id=prepared.prepared_id,
            scope=scope,
            record_phase=record_phase,
        )
    if construction.validity_policy_id != audit_policy.validity_policy.policy_id:
        raise ValueError(
            "VolumeConstruction.validity_policy_id must match the publication audit policy; got a different validity policy."
        )
    if any(
        association.source_id != source.source_id
        or association.source_revision != source.source_revision
        for association in construction.associations
    ):
        raise ValueError(
            "VolumeConstruction.associations must bind NativePlcSource.source_id and source_revision; got a different source revision."
        )
    check_deadline(started, limits, MeshingStageKind.VOLUME_FILL)
    mesh = construction.mesh
    vertices = np.asarray(mesh.coordinates, dtype=np.float64)
    cells = np.asarray(mesh.blocks[0].vertices, dtype=np.int64)
    simplex_entity_limits(
        vertices,
        cells,
        limits,
        MeshingStageKind.VOLUME_FILL,
        cell_kind=mesh.blocks[0].cell_kind,
    )
    compliance = volume_compliance(
        specification, prepared.schedule, construction, vertices, cells, scope=scope
    )
    check_deadline(started, limits, MeshingStageKind.GEOMETRY_AUDIT)
    domain = declared_plc_domain(source.complex, source.source_id)
    allowance = scope.remaining()
    with scope.batch(
        min(limits.maximum_work_units, allowance.remaining_work_units),
        queries_upper=min(
            limits.maximum_geometry_queries, allowance.remaining_geometry_queries
        ),
        bytes_upper=min(limits.maximum_scratch_bytes, allowance.remaining_managed_bytes),
    ) as ticket:
        budget = CoordinateEnclosureBudget(
            ticket,
            maximum_work_units=limits.maximum_work_units,
            maximum_memory_bytes=limits.maximum_scratch_bytes,
        )
        budget.capture_coordinate_operands = coordinate_operands is not None
        budget.capture_signed_determinant_operands = (
            coordinate_operands is not None and coordinate_operands.capture_signed_determinants
        )
        with budget.activate():
            geometry = (
                plc_geometry_owner.exact_plc_geometry(construction, source, budget=budget)
                if construction.geometry is None
                else construction.geometry
            )
    construction_stages = (
        MeshingStageReport(
            MeshingStageKind.SOURCE_INSPECTION,
            MeshingStageStatus.PASSED,
            input_ids=(source.binding_id,),
            output_ids=(prepared.prepared_id,),
        ),
        *construction.stages,
    )

    def retain_coordinate_storage(bytes_upper: int) -> NativeBatchTicket:
        return scope.batch(0, bytes_upper=bytes_upper)

    with budget.retained_storage(retain_coordinate_storage):
        result = publish_native_result(
            mesh,
            coordinate_contract,
            compliance,
            construction_stages,
            provider,
            {
                "kind": "native-plc-volume-mesh",
                "route": "plc_tetrahedral",
                "source": source.binding_id,
                "plan": plan_id,
                "specification": specification.specification_id,
                "schedule": prepared.schedule.schedule_id,
            },
            NativeCertificationRequest(
                MeshCertificationSchedule("volume_plc"),
                source.source_id,
                source.source_revision,
                limits,
                domain=domain,
                cell_regions=construction.cell_regions,
            ),
            policy=publication_policy("volume", audit_policy=audit_policy),
            organization=NativePublicationOrganization(
                patches=construction.patches,
                zones=construction.zones,
                labels=construction.labels,
                associations=construction.associations,
            ),
            execution=NativePublicationExecution(scope, workspace, record_phase, budget),
            geometry=geometry,
        )
        if coordinate_operands is not None:
            # Source recovery's numeric strata are retained with the same
            # source/frame/policy and ended earning receipt as coordinate banks.
            # The receiving consumer still validates every association freshly.
            with coordinate_phase(scope, limits, budget):
                constraints = construction.source_constraints
                if constraints is None:
                    raise ValueError("PLC operand capture requires the actual recovered source tables.")
                certificate = result.certification
                if certificate is None or not isinstance(certificate.request.domain, PiecewiseLinearDomain):
                    raise ValueError("PLC operand capture requires the actual published declared domain owner.")
                with budget.live_storage() as storage:
                    storage.set_bound(
                        65536 + 4096 * (
                            constraints.points.shape[0]
                            + constraints.input_triangles.shape[0]
                            + constraints.plc_edges.shape[0]
                            + source.complex.polygon_facets.size
                        )
                    )
                    source_operands = PlcSourceCoordinateOperands(
                        source, specification, coordinate_contract, constraints,
                        certificate.request.domain,
                        _source_input_witness(construction, source, budget),
                        prepared.schedule, audit_policy,
                    )
                    budget.retain_owner(source_operands, temporary_storage_credit=storage)
                coordinate_operands.capture(
                    result.mesh, result.geometry, budget, scope,
                    plc_source=source_operands,
                    coordinate_contract=result.coordinate_contract,
                )
        return result


__all__ = [
    "PreparedPlcVolume",
    "execute_volume_route",
    "plc_volume_audit_policy",
    "volume_support_issues",
]
