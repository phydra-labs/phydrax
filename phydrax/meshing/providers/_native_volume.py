#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native PLC volume route: constrained tetrahedral generation and acceptance.

The route admits a `NativePlcSource` with a `VolumeMeshingSpec`, runs the
phases of `phydrax.meshing._volume_generation` (exact boundary recovery under
the source's fixed or conforming boundary policy, region classification,
refinement and improvement, organization and exact associations), measures
size compliance, and publishes only after the independent ``volume_plc``
acceptance: per-cell validity, topology, global embedding and exact coverage
of the declared source polygons and region measures.
"""

from __future__ import annotations

from contextlib import nullcontext
from time import monotonic
from typing import final

import equinox as eqx
import numpy as np

from ..._fingerprint import canonical_fingerprint
from ..._meshcore import (
    current_native_execution_budget,
    current_native_host_workspace,
    TET_MESH_EXUDE_COUNTERS,
    TET_MESH_IMPROVE_COUNTERS,
    TET_MESH_REFINE_COUNTERS,
)
from ..._physical import SpatialCoordinateContract
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization import CellGeometrySpec, TetrahedralConnectivity
from ...discretization._exact_plc_geometry import ExactPlcCellGeometrySource
from ...optim import OptimizationStatus
from .._audit import CellMeshAuditDisposition, CellMeshAuditPolicy
from .._certification import MeshCertificationSchedule
from .._contracts import (
    MeshingDerivativeMode,
    MeshingFailure,
    MeshingFailureCategory,
    MeshingProviderInfo,
    VolumeFillStrategy,
    VolumeMeshingSpec,
)
from .._controls import FeatureKind, RegionControl
from .._measurements import NativeMeshingPhaseRecorder
from .._result import CellMeshingResult, MeshingComplianceReport
from .._sizing import UniformSizeControl
from .._tetra_metric import MetricRemeshingEvidence
from .._trace import MeshingStageKind, MeshingStageReport, MeshingStageStatus
from .._volume_generation import (
    _entity,
    _import_native_preparation,
    _native_volume_operation_started,
    _require_native_preparation_allowance,
    declared_plc_domain,
    generate_plc_volume,
    native_volume_execution_budget,
    NativeVolumeSchedule,
    VolumeConstruction,
)
from ._native_publication import (
    bind_native_execution_result,
    check_deadline,
    edge_size_evidence,
    NativeCertificationRequest,
    publish_native_result,
    simplex_entity_limits,
    uniform_size_compliance,
    unique_edges,
)
from ._native_sources import NativePlcSource, NativePolyhedralSource


def volume_support_issues(
    source: NativePlcSource | NativePolyhedralSource, specification: VolumeMeshingSpec, /
) -> list[str]:
    """Physical requests of one volume specification this route cannot enforce."""

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
    controls = specification.size_controls
    for control in controls:
        if not isinstance(control, UniformSizeControl):
            unsupported.append("nonuniform curvature/proximity PLC size fields")
        elif control.scope.entity_dimension != 2 or np.any(
            np.asarray(control.scope.entity_ids) >= complex_.facet_count
        ):
            unsupported.append("uniform size controls bound to PLC facets")
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

    def __init__(
        self,
        source: NativePlcSource,
        specification: VolumeMeshingSpec,
        schedule: NativeVolumeSchedule,
        /,
    ) -> None:
        if not isinstance(source, NativePlcSource):
            raise TypeError("source must be NativePlcSource.")
        if not isinstance(schedule, NativeVolumeSchedule):
            raise TypeError("schedule must be NativeVolumeSchedule.")
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
    """Publish the actual optimization goal, outcome, work and native resources."""
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
    memory_names = (
        "limit",
        "live",
        "peak",
        "largest_request",
        "allocations",
        "refusals",
    )
    achieved.extend(
        (f"metric_optimization:native_memory:{name}", value)
        for name, value in zip(memory_names, metric.native_memory_evidence, strict=True)
    )
    _append_optimizer_evidence(metric, requested, achieved)


def _append_optimizer_evidence(
    metric: MetricRemeshingEvidence,
    requested: list[tuple[str, float]],
    achieved: list[tuple[str, float]],
    /,
) -> None:
    """Retain canonical solver termination and charged host proposal resources."""
    if metric.optimizer_method_id is None:
        return
    achieved.extend(
        (
            (f"metric_optimization:optimizer:{metric.optimizer_method_id}", 1.0),
            ("metric_optimization:native_work_units", metric.native_work_units),
            ("metric_optimization:optimizer_iterations", metric.optimizer_iterations),
            (
                "metric_optimization:optimizer_objective_evaluations",
                metric.optimizer_evaluations,
            ),
            (
                "metric_optimization:optimizer_constraint_evaluations",
                metric.optimizer_constraint_evaluations,
            ),
            (
                "metric_optimization:optimizer_proposal_evaluations",
                metric.optimizer_proposal_evaluations,
            ),
            (
                "metric_optimization:optimizer_refused_proposals",
                metric.optimizer_refused_proposals,
            ),
            (
                "metric_optimization:optimizer_host_work_units",
                metric.optimizer_host_work_units,
            ),
            (
                "metric_optimization:optimizer_host_table_bytes",
                metric.optimizer_scratch_required_bytes,
            ),
        )
    )
    requested.extend(
        (f"metric_optimization:optimizer_tolerance:{index}", value)
        for index, value in enumerate(metric.optimizer_tolerances)
    )
    if metric.optimizer_status_counts:
        achieved.extend(
            (f"metric_optimization:optimizer_status:{status.name.lower()}", count)
            for status, count in zip(
                OptimizationStatus,
                metric.optimizer_status_counts,
                strict=True,
            )
        )
    for index, (steps, evaluations) in enumerate(metric.optimizer_terminations):
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
        for name, count in metric.optimizer_refusal_counts
    )


def _volume_compliance(
    specification: VolumeMeshingSpec,
    schedule: NativeVolumeSchedule,
    construction: VolumeConstruction,
    vertices: np.ndarray,
    cells: np.ndarray,
    /,
) -> MeshingComplianceReport:
    """Size requests are hard or recorded; refinement and sliver aims are recorded."""

    requested: list[tuple[str, float]] = [
        ("radius_edge_bound", schedule.radius_edge_bound),
        ("minimum_dihedral_degrees", schedule.minimum_dihedral_degrees),
    ]
    quality = construction.quality
    if not np.isfinite(quality.maximum_radius_edge):
        raise MeshingFailure(
            MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
            "Native tetrahedral circumcenter construction did not resolve radius-edge quality.",
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
            for name, value in zip(
                TET_MESH_REFINE_COUNTERS,
                construction.refinement.counters.tolist(),
                strict=True,
            )
        ),
        *(
            (f"improvement:{name}", int(value))
            for name, value in zip(
                TET_MESH_IMPROVE_COUNTERS,
                construction.improvement.counters.tolist(),
                strict=True,
            )
        ),
        *(
            ()
            if construction.exudation is None
            else (
                (f"exudation:{name}", int(value))
                for name, value in zip(
                    TET_MESH_EXUDE_COUNTERS,
                    construction.exudation.counters.tolist(),
                    strict=True,
                )
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
    issues: list[str] = []
    metric = construction.metric_optimization
    if metric is not None:
        _append_metric_evidence(metric, requested, achieved)
    for control in specification.size_controls:
        if not isinstance(control, UniformSizeControl):
            raise TypeError("Admitted volume size controls must be UniformSizeControl.")
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
                _entity(mesh.numeric_version, "facet", facet)
                for facet in np.asarray(control.scope.entity_ids).tolist()
            }
            selected = np.asarray(
                [entity in source_entities for entity in association.source_entity_ids],
                dtype=np.bool_,
            )
            identifiers = np.asarray(association.target_global_ids)[selected]
            connectivity = mesh.connectivity
            if not isinstance(connectivity, TetrahedralConnectivity):
                raise TypeError("A PLC volume requires tetrahedral connectivity.")
            rows = np.asarray(connectivity.faces, dtype=np.int64)[
                np.isin(np.asarray(face_set.entity_ids), identifiers)
            ]
            edges = unique_edges(rows, "triangle")
        lengths, growth = edge_size_evidence(vertices, edges)
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


def execute_volume_route(
    source: NativePlcSource,
    specification: VolumeMeshingSpec,
    prepared: PreparedPlcVolume,
    coordinate_contract: SpatialCoordinateContract,
    provider: MeshingProviderInfo,
    plan_id: str,
    /,
    *,
    construction: VolumeConstruction | None = None,
    record_phase: NativeMeshingPhaseRecorder | None = None,
) -> CellMeshingResult:
    """Own standalone generation through publication, or borrow the public root."""
    active = current_native_execution_budget()
    if active is not None and (
        construction is None or construction.native_execution_evidence is None
    ):
        return _execute_volume_route_bound(
            source,
            specification,
            prepared,
            coordinate_contract,
            provider,
            plan_id,
            construction=construction,
            record_phase=record_phase,
        )
    preparation = None
    work = queries = 0
    seconds = 0.0
    if construction is not None:
        preparation = construction.native_execution_record
        if preparation is None:
            raise ValueError(
                "Standalone supplied construction requires its actual ended native receipt."
            )
        if prepared.source_binding_id != source.binding_id or any(
            association.source_id != source.source_id
            or association.source_revision != source.source_revision
            for association in construction.associations
        ):
            raise ValueError(
                "The retained construction receipt binds another original source."
            )
        _require_native_preparation_allowance(specification.limits, preparation)
        work = int(np.asarray(preparation.total_work_units))
        queries = int(np.asarray(preparation.total_geometry_queries))
        seconds = float(np.asarray(preparation.total_elapsed_seconds))
    started = monotonic() - seconds
    workspace = current_native_host_workspace()
    scope = (
        active.host_workspace()
        if active is not None and workspace is None
        else nullcontext(workspace)
    )
    with scope as storage:
        if active is not None and preparation is not None:
            if storage is None:
                raise RuntimeError("Construction import lost its owning host workspace.")
            _import_native_preparation(active, storage, preparation)
        parent_started = _native_volume_operation_started()
        if parent_started is not None:
            started = min(started, parent_started)
        with native_volume_execution_budget(
            specification.limits,
            source_work_units=work,
            source_geometry_queries=queries,
            operation_started=started,
            borrow_active=False,
        ) as budget:
            storage = current_native_host_workspace()
            if storage is None:
                raise RuntimeError(
                    "Native volume publication lost its owning host workspace."
                )
            storage.retain_owner((source, specification, prepared, construction))
            result = _execute_volume_route_bound(
                source,
                specification,
                prepared,
                coordinate_contract,
                provider,
                plan_id,
                construction=construction,
                record_phase=record_phase,
            )
        if budget.evidence is None:
            raise RuntimeError(
                "Native volume publication has no actual ended scope evidence."
            )
        return bind_native_execution_result(
            result,
            budget.evidence,
            specification.limits,
            started,
            source_work_units=work,
            source_geometry_queries=queries,
            preparation_seconds=seconds,
            preparation_evidence=preparation,
        )


def _execute_volume_route_bound(
    source: NativePlcSource,
    specification: VolumeMeshingSpec,
    prepared: PreparedPlcVolume,
    coordinate_contract: SpatialCoordinateContract,
    provider: MeshingProviderInfo,
    plan_id: str,
    /,
    *,
    construction: VolumeConstruction | None = None,
    record_phase: NativeMeshingPhaseRecorder | None = None,
) -> CellMeshingResult:
    """Generate, certify and publish while borrowing the original allowance."""

    if prepared.source_binding_id != source.binding_id:
        raise ValueError("The prepared volume route binds another source.")
    started = monotonic()
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
            record_phase=record_phase,
        )
    elif not isinstance(construction, VolumeConstruction):
        raise TypeError("construction must be an authoritative VolumeConstruction.")
    if construction.validity_policy_id != audit_policy.validity_policy.policy_id:
        raise ValueError(
            "The native construction was improved under another cell validity policy."
        )
    if any(
        association.source_id != source.source_id
        or association.source_revision != source.source_revision
        for association in construction.associations
    ):
        raise ValueError("The native construction binds a different source revision.")
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
    compliance = _volume_compliance(
        specification, prepared.schedule, construction, vertices, cells
    )
    check_deadline(started, limits, MeshingStageKind.GEOMETRY_AUDIT)
    domain = declared_plc_domain(source.complex, source.source_id)
    geometry = _exact_plc_geometry(construction, source)
    construction_stages = (
        MeshingStageReport(
            MeshingStageKind.SOURCE_INSPECTION,
            MeshingStageStatus.PASSED,
            input_ids=(source.binding_id,),
            output_ids=(prepared.prepared_id,),
        ),
        *construction.stages,
    )
    return publish_native_result(
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
        audit_policy=audit_policy,
        geometry=geometry,
        derivative_mode=MeshingDerivativeMode.NONDIFFERENTIABLE,
        enforced_limits=(
            "vertices",
            "edges",
            "faces",
            "cells",
            "connectivity_entries",
            "data_bytes",
            "scratch_bytes",
            "work_units",
            "wall_seconds",
        ),
        unenforced_limits=("cavity_cells", "geometry_queries"),
        patches=construction.patches,
        zones=construction.zones,
        labels=construction.labels,
        associations=construction.associations,
        record_phase=record_phase,
    )


def _exact_plc_geometry(
    construction: VolumeConstruction,
    source: NativePlcSource,
    /,
) -> CellGeometrySpec | None:
    """Exact source coordinates of a mesh with ancestry-backed carriers.

    Zero deviation changes no coordinate but still retains the original closed
    source rows and parameters. Bounded carriers publish the witness source S,
    never an approximate affine carrier in its place.
    """
    fidelity = construction.source_fidelity
    if fidelity is None:
        return None
    declared = fidelity.declared_source
    if declared is None:
        raise MeshingFailure(
            MeshingFailureCategory.COMPLIANCE_FAILED,
            "A bounded PLC carrier has no declared source rows for its exact coordinates.",
            stage=MeshingStageKind.GEOMETRY_AUDIT.value,
        )
    exact = ExactPlcCellGeometrySource(
        declared.points,
        declared.faces,
        declared.segments,
        fidelity.witness_strata,
        fidelity.witness_entities,
        fidelity.witness_parameters,
        domain_source_id=source.source_id,
        domain_source_revision=source.source_revision,
        source_triangle_ids=declared.face_ids[declared.face_group_rows],
        source_triangle_bounds=declared.face_tolerances,
        source_segment_ids=declared.segment_ids[declared.segment_group_rows],
        source_segment_bounds=declared.segment_tolerances,
    )
    exact.require_domain(
        declared.points, declared.faces, source.source_id, source.source_revision
    )
    return CellGeometrySpec.plc(construction.mesh, exact)


__all__ = [
    "PreparedPlcVolume",
    "execute_volume_route",
    "plc_volume_audit_policy",
    "volume_support_issues",
]
