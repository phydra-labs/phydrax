#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Cold reconstruction of retained native adaptations."""

from __future__ import annotations

import inspect
import math
from typing import Any, TYPE_CHECKING

import numpy as np

from .._fingerprint import canonical_fingerprint
from ._bisection import BisectionEvidence, BisectionHierarchy
from ._device_metric import DeviceMetricEvidence
from ._device_tetra_metric import DeviceTetraMetricEvidence
from ._lineage import (
    CellMeshTransition,
    EntityLineage,
    EntityLineageKind,
    MeshLineage,
    VertexInterpolationStencil,
)
from ._local_metric import LocalMetricEvidence
from ._metric import MeshMetricField
from ._mixed_adaptation import (
    MixedAdaptationEvidence,
    MixedAdaptationHierarchy,
    MixedLayerColumns,
)
from ._mixed_hierarchy_reproof import validate_mixed_restoration
from ._periodic import PeriodicQuotientHistory
from ._polyhedral_adaptation import (
    _hierarchy as _polyhedral_hierarchy_from_parents,
    PolyhedralAdaptationEvidence,
    PolyhedralAdaptationHierarchy,
    PolyhedralAdaptationOperation,
)
from ._scope import MeshingEntityKind, MeshingScope
from ._surface_metric import (
    SurfaceMetricGeometryRelation,
    SurfaceMetricOutcome,
    SurfaceMetricStatus,
)
from ._tetra_metric import TetraMetricEvidence, TetraMetricStatus
from .providers._mmg import (
    MmgAdaptationResult,
    MmgProvider,
    MmgReference,
    MmgReferenceRetention,
)
from .providers._omega_h import (
    _entity_scope,
    _GatheredTarget,
    _RankArrays,
    OmegaHAdaptationEvidence,
    OmegaHAdaptationResult,
    OmegaHClassification,
    OmegaHFieldEvidence,
    OmegaHOptions,
    OmegaHPartition,
    OmegaHTransferredField,
)


if TYPE_CHECKING:
    from ..discretization._cell_geometry_transition_types import CellGeometryTransition
    from ..discretization._cell_mesh import CellMesh
    from ._adaptation import (
        MarkedMeshAdaptation,
        MeshAdaptationPolicy,
        MeshAdaptationResult,
        MeshAdaptationStatus,
    )
    from ._layer_core_root_geometry import ComputedLayerCoreRoot
    from ._lineage import MeshTransitionKind
    from ._result import CellMeshingResult


def _hierarchy(
    value: MixedAdaptationHierarchy | None, /
) -> MixedAdaptationHierarchy | None:
    if value is None:
        return None
    if not isinstance(value, MixedAdaptationHierarchy):
        raise TypeError("Stored mixed adaptation requires a mixed hierarchy.")
    history = value.periodic_quotient_history
    if history is not None:
        rebuilt_history = PeriodicQuotientHistory(
            history.identification_id,
            history.source_mesh_id,
            history.entity_keys_by_degree,
            history.allocator_next_ids,
            history.occurrences_by_degree,
            history.vertex_actions_by_id,
        )
        if rebuilt_history.history_id != history.history_id:
            raise ValueError("Stored periodic quotient history identity changed.")
        history = rebuilt_history
    rebuilt = MixedAdaptationHierarchy(
        value.siblings,
        value.active_cells,
        value.forest_roots,
        value.vertex_points,
        value.retired_entities,
        value.layer_policy,
        value.next_cell_id,
        value.next_vertex_id,
        history,
        value.active_lifted_topology_id,
    )
    if rebuilt.hierarchy_id != value.hierarchy_id:
        raise ValueError("Stored mixed hierarchy identity changed.")
    return rebuilt


def _lineage(value: MeshLineage, /) -> MeshLineage:
    records = []
    for record in value.entities:
        rebuilt = EntityLineage(
            record.dimension,
            record.source_entity_set_id,
            record.target_entity_set_id,
            record.source_global_ids,
            record.target_global_ids,
            record.relation_kinds,
            created_target_ids=record.created_target_ids,
            deleted_source_ids=record.deleted_source_ids,
        )
        if rebuilt.lineage_id != record.lineage_id:
            raise ValueError("Stored entity lineage identity changed.")
        records.append(rebuilt)
    rebuilt = MeshLineage(
        value.source_topology_id, value.target_topology_id, tuple(records)
    )
    if rebuilt.lineage_id != value.lineage_id:
        raise ValueError("Stored mesh lineage identity changed.")
    return rebuilt


def _stencil(value: VertexInterpolationStencil, /) -> VertexInterpolationStencil:
    rebuilt = VertexInterpolationStencil(
        value.source_entity_set_id,
        value.target_entity_set_id,
        value.target_global_ids,
        value.source_global_ids,
        value.weights,
        value.valid,
        preserves_constants=value.preserves_constants,
        exact_weights=value.exact_weights,
    )
    if rebuilt.stencil_id != value.stencil_id:
        raise ValueError("Stored interpolation stencil identity changed.")
    return rebuilt


def _evidence(
    value: MixedAdaptationEvidence,
    request: MarkedMeshAdaptation,
    hierarchy: MixedAdaptationHierarchy,
    lineage: MeshLineage,
) -> MixedAdaptationEvidence:
    if not isinstance(value, MixedAdaptationEvidence):
        raise TypeError("Stored mixed adaptation lacks its evidence.")
    names = (
        "requested_refine_cell_ids",
        "refined_cell_ids",
        "coarsened_cell_ids",
        "rejected_coarsening_ids",
        "closure_cell_ids",
        "schedule_changed_cell_ids",
    )
    arrays = []
    for name in names:
        raw = np.asarray(getattr(value, name))
        if raw.ndim != 1 or raw.dtype.kind not in "iu":
            raise ValueError("Mixed evidence IDs must be integer vectors.")
        array = np.asarray(raw, dtype=np.int64).copy()
        if np.any(array < 0) or not np.array_equal(array, np.unique(array)):
            raise ValueError("Mixed evidence IDs must be sorted and unique.")
        array.setflags(write=False)
        arrays.append(array)
    requested, refined, coarsened, rejected, closure, schedule = arrays
    cell = lineage.entity_lineage(3)
    sources = np.asarray(cell.source_global_ids, dtype=np.int64)
    kinds = np.asarray(cell.relation_kinds, dtype=np.int32)
    actual_refined = set(map(int, sources[kinds == int(EntityLineageKind.REFINED_FROM)]))
    actual_coarsened = set(
        map(int, sources[kinds == int(EntityLineageKind.COARSENED_INTO)])
    )
    marked = set(map(int, np.asarray(request.refine_cell_ids)))
    source_cells = set(map(int, sources)) | set(
        map(int, np.asarray(cell.deleted_source_ids))
    )
    if (
        not np.array_equal(requested, np.asarray(request.refine_cell_ids))
        or set(map(int, refined)) != actual_refined & marked
        or set(map(int, closure)) != actual_refined - marked
        or set(map(int, coarsened)) != actual_coarsened
        or not set(map(int, rejected)) <= source_cells
        or not set(map(int, schedule))
        <= {item.cell_id for item in hierarchy.active_cells}
        or any(
            type(number) is not int or number < 0
            for number in (value.closure_steps, value.interface_count, value.work_units)
        )
    ):
        raise ValueError(
            "Mixed adaptation evidence conflicts with its accepted lineage or marks."
        )
    identity = canonical_fingerprint(
        {
            "kind": "mixed-adaptation-evidence",
            "requested_refine": requested,
            "refined": refined,
            "coarsened": coarsened,
            "rejected_coarsening": rejected,
            "closure": closure,
            "schedule_changed": schedule,
            "closure_steps": value.closure_steps,
            "interfaces": value.interface_count,
            "work_units": value.work_units,
            "hierarchy": hierarchy.hierarchy_id,
        }
    )
    if identity != value.evidence_id:
        raise ValueError("Stored mixed adaptation evidence identity changed.")
    return MixedAdaptationEvidence(
        requested,
        refined,
        coarsened,
        rejected,
        closure,
        schedule,
        value.closure_steps,
        value.interface_count,
        value.work_units,
        identity,
    )


def restore_mixed_adaptation_result(
    raw: MeshAdaptationResult,
    /,
    *,
    source: CellMeshingResult,
    target: CellMeshingResult,
    policy: MeshAdaptationPolicy,
    request: MarkedMeshAdaptation,
    geometry_transition: CellGeometryTransition,
    hierarchy: MixedAdaptationHierarchy,
    expected_result_id: str,
    computed_layer_root: ComputedLayerCoreRoot | None = None,
) -> MeshAdaptationResult:
    """Reconstruct actual retained owners and reprove their mathematical relations."""
    from ._adaptation import (
        _compliance,
        _RouteOutcome,
        MarkedMeshAdaptation,
        MeshAdaptationPolicy,
        MeshAdaptationResult,
        MeshAdaptationRoute,
        MeshAdaptationStatus,
        prepare_mesh_adaptation,
        SourceBoundaryChartWitness,
        SourceBoundaryFaceImageWitness,
        transfer_cell_markers,
    )
    from ._model_codec import _register_model_archive_types, _same_model_content

    _register_model_archive_types()

    if not isinstance(raw, MeshAdaptationResult) or not isinstance(
        request, MarkedMeshAdaptation
    ):
        raise TypeError(
            "Mixed restoration requires the retained result and marked request."
        )
    if (
        not isinstance(expected_result_id, str)
        or not expected_result_id
        or raw.result_id != expected_result_id
    ):
        raise ValueError(
            "Mixed restoration requires the trusted expected result identity."
        )
    if (
        policy.route is not MeshAdaptationRoute.NATIVE_MIXED
        or raw.route is not policy.route
    ):
        raise ValueError("Stored result is not a native mixed adaptation.")
    if (
        raw.source.result_id != source.result_id
        or raw.target.result_id != target.result_id
        or not _same_model_content(raw.source, source)
        or not _same_model_content(raw.target, target)
    ):
        raise ValueError("Stored mixed result names different accepted endpoints.")
    policy_copy = MeshAdaptationPolicy(
        policy.route,
        protected_scopes=policy.protected_scopes,
        compatibility=policy.compatibility,
        predicate_mode=policy.predicate_mode,
        relocation=policy.relocation,
        audit_policy=policy.audit_policy,
        device_policy=policy.device_policy,
        provider=policy.provider,
        provider_options=policy.provider_options,
        distribution=policy.distribution,
        partition_policy=policy.partition_policy,
        association_transfer=policy.association_transfer,
    )
    if policy_copy.policy_id != policy.policy_id:
        raise ValueError("Stored mixed policy identity changed.")
    if computed_layer_root is not None:
        from ._layer_core_association_transfer import NativeLayerCoreAssociationTransfer
        from ._layer_core_root_geometry import ComputedLayerCoreRoot

        transfer = policy_copy.association_transfer
        root = source.layer_core_root_geometry
        if (
            type(computed_layer_root) is not ComputedLayerCoreRoot
            or type(transfer) is not NativeLayerCoreAssociationTransfer
            or root is None
            or target.layer_core_root_geometry is not root
            or not source.associations
        ):
            raise ValueError("Computed root requires its exact native layer transition.")
        computed_layer_root.require(
            transfer.source, transfer.specification, root.mesh, root.geometry, root
        )
    if request.hierarchy is not None and not isinstance(
        request.hierarchy, MixedAdaptationHierarchy
    ):
        raise TypeError("Stored mixed request requires a mixed hierarchy.")
    old_hierarchy = _hierarchy(request.hierarchy)
    columns = request.layer_columns
    if columns is not None:
        rebuilt_columns = MixedLayerColumns(
            columns.cell_ids,
            columns.column_ids,
            columns.interval_indices,
            hard_first_thickness=columns.hard_first_thickness,
            axial_refinement=columns.axial_refinement,
            allow_schedule_change=columns.allow_schedule_change,
        )
        if rebuilt_columns.columns_id != columns.columns_id:
            raise ValueError("Stored mixed layer controls changed.")
        columns = rebuilt_columns
    lock = request.periodic_edge_lock
    if lock is not None:
        from ._periodic import prepare_periodic_refinement_edge_lock

        rebuilt_lock = prepare_periodic_refinement_edge_lock(
            source.mesh, lock.cell_global_ids, lock.selected_edge_global_ids
        )
        if rebuilt_lock.lock_id != lock.lock_id:
            raise ValueError("Stored periodic edge lock identity changed.")
        lock = rebuilt_lock
    request_copy = MarkedMeshAdaptation(
        request.refine_cell_ids,
        request.coarsen_cell_ids,
        hierarchy=old_hierarchy,
        periodic_edge_lock=lock,
        layer_columns=columns,
    )
    if request_copy.request_id != request.request_id:
        raise ValueError("Stored mixed request identity changed.")
    prepared = prepare_mesh_adaptation(
        source, request_copy, policy=policy_copy, _computed_layer_root=computed_layer_root
    )
    if prepared.prepared_id != raw.prepared_id:
        raise ValueError(
            "Stored mixed request was prepared under different source authority."
        )
    restored_hierarchy = _hierarchy(hierarchy)
    if (
        restored_hierarchy is None
        or not isinstance(raw.hierarchy, MixedAdaptationHierarchy)
        or raw.hierarchy.hierarchy_id != restored_hierarchy.hierarchy_id
        or not _same_model_content(raw.hierarchy, restored_hierarchy)
    ):
        raise ValueError("Stored mixed result hierarchy differs from supplied hierarchy.")
    if raw.lineage is None or raw.stencil is None or raw.transition is None:
        raise ValueError("Stored accepted mixed transition lacks lineage or stencil.")
    lineage = _lineage(raw.lineage)
    stencil = _stencil(raw.stencil)
    if (
        raw.transition.vertex_stencil is None
        or raw.transition.lineage.lineage_id != lineage.lineage_id
        or raw.transition.vertex_stencil.stencil_id != stencil.stencil_id
        or not _same_model_content(raw.transition.lineage, lineage)
        or not _same_model_content(raw.transition.vertex_stencil, stencil)
        or raw.transition.geometry_transition is None
        or not _same_model_content(
            raw.transition.geometry_transition, geometry_transition
        )
    ):
        raise ValueError("Stored mixed transition has inconsistent lineage or stencil.")
    cell_transfer = validate_mixed_restoration(
        source=source,
        target=target,
        geometry_transition=geometry_transition,
        hierarchy=restored_hierarchy,
        request=request_copy,
        lineage=lineage,
        stencil=stencil,
        constraints=prepared.constraints,
    )
    markers = transfer_cell_markers(source.mesh, source.attributes, lineage, target.mesh)
    if not _same_model_content(markers, target.attributes):
        raise ValueError("Mixed target material markers differ from source lineage.")
    association_transfer = policy_copy.association_transfer
    if association_transfer is not None and source.associations:
        from ._layer_core_association_transfer import _semantic_rows

        certification = target.certification
        if certification is None:
            raise ValueError(
                "Mixed target associations require certified source fidelity."
            )
        inputs = certification.inputs
        witness = inputs.fidelity_witness
        target_witness = (
            witness
            if isinstance(
                witness, (SourceBoundaryChartWitness, SourceBoundaryFaceImageWitness)
            )
            else None
        )
        validated = certification.require_current(target.mesh, target.geometry)
        if computed_layer_root is None:
            associations = association_transfer.propagate(
                source,
                lineage,
                target.mesh,
                vertex_stencil=stencil,
                geometry_transition=geometry_transition,
                target_fidelity_witness=target_witness,
                target_scoped_fidelity=inputs.scoped_fidelity,
                validated_fidelity=validated,
            )
        else:
            from ._layer_core_association_transfer import (
                NativeLayerCoreAssociationTransfer,
            )

            if type(association_transfer) is not NativeLayerCoreAssociationTransfer:
                raise TypeError(
                    "Computed root requires its native layer association owner."
                )
            associations = association_transfer._propagate_computed(
                source,
                lineage,
                target.mesh,
                vertex_stencil=stencil,
                geometry_transition=geometry_transition,
                target_fidelity_witness=target_witness,
                target_scoped_fidelity=inputs.scoped_fidelity,
                validated_fidelity=validated,
                computed_root=computed_layer_root,
            )
        if _semantic_rows(target.mesh, associations) != _semantic_rows(
            target.mesh, target.associations
        ):
            raise ValueError(
                "Mixed target associations differ from source-bound lineage."
            )
    elif target.associations:
        raise ValueError(
            "Mixed target has associations without source transfer authority."
        )
    if raw.cell_content_transfer is None or not _same_model_content(
        cell_transfer, raw.cell_content_transfer
    ):
        raise ValueError(
            "Stored mixed cell-content action differs from its exact partition."
        )
    if not isinstance(raw.evidence, MixedAdaptationEvidence):
        raise TypeError("Stored mixed adaptation lacks its evidence.")
    evidence = _evidence(raw.evidence, request_copy, restored_hierarchy, lineage)
    partial = bool(evidence.rejected_coarsening_ids.size) or (
        evidence.refined_cell_ids.size != evidence.requested_refine_cell_ids.size
    )
    expected_status = (
        MeshAdaptationStatus.PARTIAL if partial else MeshAdaptationStatus.COMPLETE
    )
    if raw.status is not expected_status:
        raise ValueError("Stored mixed status differs from the accepted evidence.")
    from ._lineage import MeshTransitionKind

    has_refinement = bool(
        evidence.refined_cell_ids.size or evidence.closure_cell_ids.size
    )
    has_coarsening = bool(evidence.coarsened_cell_ids.size)
    expected_kind = (
        MeshTransitionKind.REMESH
        if has_refinement and has_coarsening
        else MeshTransitionKind.REFINE
        if has_refinement
        else MeshTransitionKind.COARSEN
    )
    if (
        not (has_refinement or has_coarsening)
        or raw.transition.transition_kind is not expected_kind
    ):
        raise ValueError(
            "Stored mixed transition kind differs from its exact cell lineage."
        )
    if (
        raw.transition.volume_material_chain is not None
        or raw.transition.material_reference is not None
    ):
        raise ValueError(
            "Mixed restoration requires independently reconstructed material transition authority."
        )
    transition = CellMeshTransition(
        source.mesh.mesh_id,
        source.mesh.topology_id,
        target,
        lineage,
        raw.transition.transition_kind,
        vertex_stencil=stencil,
        geometry_transition=geometry_transition,
    )
    if transition.transition_id != raw.transition.transition_id:
        raise ValueError("Stored mixed transition identity changed.")
    transfer = None
    if stencil.preserves_constants and np.all(np.any(np.asarray(stencil.valid), axis=1)):
        try:
            transfer = stencil.as_transfer(
                source.mesh.vertex_global_ids,
                source_topology_id=source.mesh.topology_id,
                target_topology_id=target.mesh.topology_id,
                preserves_linear=True,
                conservative=False,
                source_coordinates=source.mesh.coordinates,
                target_coordinates=target.mesh.coordinates,
            )
        except ValueError as error:
            if str(error) not in (
                "Transfer claims linear preservation but moves coordinates.",
                "Transfer claims conservation but changes DOF measures.",
            ):
                raise
    if (transfer is None) != (raw.transfer is None) or (
        transfer is not None and not _same_model_content(transfer, raw.transfer)
    ):
        raise ValueError("Stored finite-element transfer differs from its exact stencil.")
    if (
        raw.metric is not None
        or raw.conservative_remap is not None
        or raw.surface_chart_witness is not None
    ):
        raise ValueError("Stored mixed result has unsupported foreign-route evidence.")
    distribution = None
    if policy_copy.distribution is not None:
        from ._adaptation import MeshPart, prepare_distribution_transition

        if policy_copy.partition_policy is None:
            raise ValueError("Mixed distribution requires a partition policy.")

        distribution = prepare_distribution_transition(
            policy_copy.distribution,
            MeshPart(policy_copy.distribution.part.name, target),
            lineage,
            policy=policy_copy.partition_policy,
        )
    if (distribution is None) != (raw.distribution is None) or (
        distribution is not None
        and not _same_model_content(distribution, raw.distribution)
    ):
        raise ValueError("Stored mixed distribution differs from its source and lineage.")
    if raw.elapsed_seconds is not None:
        raise ValueError("Model archives do not persist adaptation wall time.")
    outcome = _RouteOutcome(
        raw.status,
        target,
        transition,
        lineage,
        stencil,
        transfer,
        None,
        evidence,
        restored_hierarchy,
        cell_transfer,
    )
    rebuilt = MeshAdaptationResult(
        prepared,
        outcome,
        _compliance(prepared, target),
        distribution,
        None,
    )
    if rebuilt.result_id != expected_result_id or not _same_model_content(
        rebuilt.compliance, raw.compliance
    ):
        raise ValueError("Stored mixed result identity or compliance changed.")
    return rebuilt


_RouteEvidence = (
    LocalMetricEvidence
    | DeviceMetricEvidence
    | TetraMetricEvidence
    | DeviceTetraMetricEvidence
    | BisectionEvidence
    | PolyhedralAdaptationEvidence
    | MmgAdaptationResult
    | OmegaHAdaptationResult
    | SurfaceMetricOutcome
)

_ROUTE_EVIDENCE: dict[str, type] = {
    "native_metric_2d": LocalMetricEvidence,
    "device_metric_2d": DeviceMetricEvidence,
    "native_metric_3d": TetraMetricEvidence,
    "device_metric_3d": DeviceTetraMetricEvidence,
    "native_bisection": BisectionEvidence,
    "device_bisection": BisectionEvidence,
    "native_polyhedral": PolyhedralAdaptationEvidence,
    "mmg": MmgAdaptationResult,
    "omega_h": OmegaHAdaptationResult,
    "native_metric_surface": SurfaceMetricOutcome,
}


def _identifier_vector(value: Any, name: str, /) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1 or (raw.size and raw.dtype.kind not in "iu"):
        raise ValueError(f"Stored {name} must be an integer vector.")
    array = raw.astype(np.int64, copy=True)
    array.setflags(write=False)
    return array


def _polyhedral_evidence(value: PolyhedralAdaptationEvidence, /) -> Any:
    """Rebuild polyhedral evidence; its identity binds the unretained request.

    The evidence identity is hashed as stored. Its exact members are bound to
    the restored meshes, hierarchy, and conservative remap by the route restore.
    """
    from ._model_codec import _same_model_content

    names = (
        value.source_mesh_id,
        value.target_mesh_id,
        value.common_refinement_id,
        value.conservative_remap_id,
        value.evidence_id,
    )
    measures = (value.maximum_measure_residual, value.maximum_coordinate_rounding_error)
    if (
        type(value.operation) is not PolyhedralAdaptationOperation
        or any(not isinstance(item, str) or not item for item in names)
        or any(
            not isinstance(item, float) or not math.isfinite(item) or item < 0.0
            for item in measures
        )
        or any(
            item is not None and (not isinstance(item, str) or not item)
            for item in (value.construction_id, value.coverage_renewal_id)
        )
        or any(not isinstance(item, str) for item in value.site_ids)
        or (value.site_points is None) != (value.site_weights is None)
    ):
        raise ValueError("Stored polyhedral evidence fields are malformed.")
    site_points = site_weights = None
    if value.site_points is not None and value.site_weights is not None:
        site_points = np.asarray(value.site_points, dtype=np.float64)
        site_weights = np.asarray(value.site_weights, dtype=np.float64)
        if (
            site_points.shape != (len(value.site_ids), 3)
            or site_weights.shape != (len(value.site_ids),)
            or not np.all(np.isfinite(site_points))
            or not np.all(np.isfinite(site_weights))
        ):
            raise ValueError("Stored polyhedral sites are malformed.")
    elif value.site_ids:
        raise ValueError("Stored polyhedral site IDs lack their sites.")
    rebuilt = PolyhedralAdaptationEvidence(
        value.operation,
        value.source_mesh_id,
        value.target_mesh_id,
        _identifier_vector(value.changed_cell_ids, "changed polyhedral cells"),
        _identifier_vector(value.target_region_classes, "polyhedral region classes"),
        value.common_refinement_id,
        value.conservative_remap_id,
        value.maximum_measure_residual,
        value.maximum_coordinate_rounding_error,
        value.construction_id,
        tuple(value.site_ids),
        site_points,
        site_weights,
        value.coverage_renewal_id,
        value.evidence_id,
    )
    if not _same_model_content(rebuilt, value):
        raise ValueError("Stored polyhedral evidence changed.")
    return rebuilt


def _polyhedral_hierarchy(
    value: Any, source: CellMesh, target: CellMesh, /
) -> PolyhedralAdaptationHierarchy:
    """Recompute the hierarchy owner from its stored exact parent map."""
    from ._model_codec import _same_model_content

    if type(value) is not PolyhedralAdaptationHierarchy:
        raise TypeError("Stored polyhedral adaptation lacks its hierarchy.")
    target_ids = _identifier_vector(value.target_cell_ids, "polyhedral target cells")
    offsets = _identifier_vector(value.parent_offsets, "polyhedral parent offsets")
    parents = _identifier_vector(value.parent_cell_ids, "polyhedral parent cells")
    classes = _identifier_vector(value.region_classes, "polyhedral region classes")
    if (
        offsets.size != target_ids.size + 1
        or classes.size != target_ids.size
        or (offsets.size and offsets[0] != 0)
        or np.any(np.diff(offsets) < 0)
        or offsets[-1] != parents.size
    ):
        raise ValueError("Stored polyhedral hierarchy offsets are malformed.")
    rebuilt = _polyhedral_hierarchy_from_parents(
        source,
        target,
        {
            int(identifier): tuple(int(item) for item in parents[start:stop])
            for identifier, start, stop in zip(
                target_ids, offsets[:-1], offsets[1:], strict=True
            )
        },
        {
            int(identifier): int(region)
            for identifier, region in zip(target_ids, classes, strict=True)
        },
    )
    source_ids = set(map(int, np.asarray(rebuilt.source_cell_ids)))
    if (
        rebuilt.hierarchy_id != value.hierarchy_id
        or not set(map(int, parents)) <= source_ids
        or not _same_model_content(rebuilt, value)
    ):
        raise ValueError("Stored polyhedral hierarchy identity changed.")
    return rebuilt


def _polyhedral_remap(
    value: Any,
    source: CellMesh,
    target: CellMesh,
    evidence: PolyhedralAdaptationEvidence,
    /,
) -> Any:
    """Rebuild the physical-overlap remap over the restored endpoint meshes.

    Overlap measures and the common-refinement provenance are hashed as stored;
    the remap owner recomputes its routes, layout, and coverage over them.
    """
    from ..discretization.finite_volume._unstructured import (
        UnstructuredFiniteVolumePlan,
    )
    from ..discretization.finite_volume._unstructured_remap import (
        UnstructuredConservativeRemapPlan,
    )
    from ._model_codec import _same_model_content

    if type(value) is not UnstructuredConservativeRemapPlan:
        raise TypeError("Stored polyhedral adaptation lacks its conservative remap.")
    rebuilt = UnstructuredConservativeRemapPlan(
        UnstructuredFiniteVolumePlan.from_cell_mesh(source).prepare(
            numeric_version=source.numeric_version
        ),
        UnstructuredFiniteVolumePlan.from_cell_mesh(target).prepare(
            numeric_version=target.numeric_version
        ),
        value.target_offsets,
        value.source_indices,
        value.intersection_measures,
        method=value.method,
        provenance=value.provenance,
        tolerance=float(np.asarray(value.report.tolerance)),
        require_complete=value.require_complete,
    )
    if (
        rebuilt.plan_id != value.plan_id
        or value.method != "physical-overlap"
        or value.provenance != evidence.common_refinement_id
        or value.plan_id != evidence.conservative_remap_id
        or not _same_model_content(rebuilt, value)
    ):
        raise ValueError("Stored conservative remap differs from its endpoints.")
    return rebuilt


def _reconstructed(value: Any, identity: str | None, label: str, /, **fields: Any) -> Any:
    """Rebuild one record through its constructor from its stored fields.

    Constructor parameters are named after the fields they set; ``fields``
    replaces stored nested owners by their reconstructed values. Floating-point
    measurements are hashed as stored, never recomputed. Derived fields outside
    the identity must also reproduce.
    """
    from ._model_codec import _same_model_content

    record_type = type(value)
    arguments: list[Any] = []
    keywords: dict[str, Any] = {}
    parameters = tuple(inspect.signature(record_type.__init__).parameters.values())
    for parameter in parameters[1:]:
        stored = fields.get(parameter.name, getattr(value, parameter.name))
        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
            arguments.append(stored)
        else:
            keywords[parameter.name] = stored
    rebuilt = record_type(*arguments, **keywords)
    if (
        identity is not None and getattr(rebuilt, identity) != getattr(value, identity)
    ) or not _same_model_content(rebuilt, value):
        raise ValueError(f"Stored {label} identity changed.")
    return rebuilt


def _route_evidence(value: Any, route: str, /) -> _RouteEvidence:
    expected = _ROUTE_EVIDENCE[route]
    if type(value) is not expected:
        raise TypeError("Stored adaptation evidence does not belong to its route.")
    if type(value) in (MmgAdaptationResult, OmegaHAdaptationResult):
        # Rebound to the restored target after endpoint recertification.
        return value
    if type(value) is SurfaceMetricOutcome:
        return _surface_outcome(value)
    if type(value) is PolyhedralAdaptationEvidence:
        return _polyhedral_evidence(value)
    if type(value) is DeviceTetraMetricEvidence:
        nested = value.metric_evidence
        if nested is not None and type(nested) is not TetraMetricEvidence:
            raise TypeError("Stored device tetra evidence lost its native evidence.")
        return _reconstructed(
            value,
            "evidence_id",
            "adaptation evidence",
            metric_evidence=(
                None
                if nested is None
                else _reconstructed(nested, "evidence_id", "tetra metric evidence")
            ),
        )
    return _reconstructed(value, "evidence_id", "adaptation evidence")


def _route_decision(
    evidence: _RouteEvidence, /
) -> tuple[bool, MeshAdaptationStatus, MeshTransitionKind, bool]:
    """Exact change, status, transition kind, and conservation of one route."""
    from ._adaptation import _metric_status, MeshAdaptationStatus
    from ._lineage import MeshTransitionKind

    tetra_status = {
        TetraMetricStatus.COMPLETE: MeshAdaptationStatus.COMPLETE,
        TetraMetricStatus.STALLED: MeshAdaptationStatus.STALLED,
        TetraMetricStatus.NUMERICAL_STOP: MeshAdaptationStatus.NUMERICAL_STOP,
    }
    match evidence:
        case LocalMetricEvidence() | DeviceMetricEvidence():
            applied = (
                evidence.splits
                + evidence.collapses
                + evidence.flips
                + evidence.relocations
            )
            status = _metric_status(evidence, evidence.topology_operations)
            return applied > 0, status, MeshTransitionKind.REMESH, False
        case TetraMetricEvidence():
            applied = (
                evidence.splits
                + evidence.collapses
                + evidence.flips23
                + evidence.flips32
                + evidence.ring_reconnections
                + evidence.relocations
            )
            status = tetra_status[evidence.status]
            return applied > 0, status, MeshTransitionKind.REMESH, False
        case DeviceTetraMetricEvidence():
            status = tetra_status[evidence.status]
            return evidence.committed, status, MeshTransitionKind.REMESH, False
        case BisectionEvidence():
            refined = evidence.bisections > 0 or evidence.uniform_refinement_applied
            coarsened = evidence.coarsened_vertices > 0
            partial = (
                np.asarray(evidence.rejected_refinement_ids).size > 0
                or np.asarray(evidence.rejected_coarsening_ids).size > 0
            )
            if not refined and not coarsened:
                status = (
                    MeshAdaptationStatus.PARTIAL
                    if partial
                    else MeshAdaptationStatus.UNCHANGED
                )
            else:
                status = (
                    MeshAdaptationStatus.PARTIAL
                    if partial
                    else MeshAdaptationStatus.COMPLETE
                )
            kind = (
                MeshTransitionKind.REMESH
                if refined and coarsened
                else MeshTransitionKind.REFINE
                if refined
                else MeshTransitionKind.COARSEN
            )
            return refined or coarsened, status, kind, not coarsened
        case PolyhedralAdaptationEvidence():
            changed = np.asarray(evidence.changed_cell_ids).size > 0
            status = (
                MeshAdaptationStatus.COMPLETE
                if changed
                else MeshAdaptationStatus.UNCHANGED
            )
            return changed, status, MeshTransitionKind.REMESH, False
        case MmgAdaptationResult() | OmegaHAdaptationResult():
            return True, MeshAdaptationStatus.COMPLETE, MeshTransitionKind.REMESH, False
        case SurfaceMetricOutcome():
            applied = (
                evidence.splits
                + evidence.collapses
                + evidence.flips
                + evidence.relocations
            )
            surface_status = {
                SurfaceMetricStatus.COMPLETE: MeshAdaptationStatus.COMPLETE,
                SurfaceMetricStatus.STALLED: MeshAdaptationStatus.STALLED,
                SurfaceMetricStatus.NUMERICAL_STOP: MeshAdaptationStatus.NUMERICAL_STOP,
            }
            return (
                applied > 0,
                surface_status[evidence.status],
                MeshTransitionKind.REMESH,
                False,
            )
        case _:
            raise TypeError("Unsupported route adaptation evidence.")


def _surface_outcome(value: SurfaceMetricOutcome, /) -> SurfaceMetricOutcome:
    """Rebuild the surface-metric evidence and its exactly derived decisions.

    The local metric evidence is rebuilt by its constructor. The outcome copies
    its counts and measurements and decides its status from it exactly; float
    measurements are hashed as stored. The edit, metric, and chart witness are
    bound to the restored endpoints by the route restore.
    """
    evidence = value.evidence
    if type(evidence) is not LocalMetricEvidence or not isinstance(
        value.status, SurfaceMetricStatus
    ):
        raise TypeError("Stored surface metric outcome lost its local evidence.")
    evidence = _reconstructed(evidence, "evidence_id", "surface metric evidence")
    status = (
        SurfaceMetricStatus.COMPLETE
        if evidence.converged
        else SurfaceMetricStatus.NUMERICAL_STOP
        if evidence.rejected_uncertain
        else SurfaceMetricStatus.STALLED
        if evidence.stalled
        else SurfaceMetricStatus.NUMERICAL_STOP
    )
    if (
        value.status is not status
        or (
            value.passes,
            value.splits,
            value.collapses,
            value.flips,
            value.relocations,
            value.out_of_range_edges,
            value.minimum_metric_length,
            value.maximum_metric_length,
            value.unit_fraction,
        )
        != (
            evidence.passes,
            evidence.splits,
            evidence.collapses,
            evidence.flips,
            evidence.relocations,
            evidence.out_of_range_edges,
            evidence.minimum_metric_length,
            evidence.maximum_metric_length,
            evidence.unit_fraction,
        )
        or not isinstance(value.geometry.relation, SurfaceMetricGeometryRelation)
    ):
        raise ValueError("Stored surface metric outcome differs from its evidence.")
    for maps in value.geometry.reference_maps:
        from ..discretization._nested_reference_types import (
            validate_reference_cell_map_witnesses,
        )

        validate_reference_cell_map_witnesses(maps)
    return value._replace(evidence=evidence)


def _surface_chart_witness(stored: Any, target: CellMeshingResult, /) -> Any:
    """The restored owner of the stored target chart witness.

    The witness is certified by the recertified target, either as its fidelity
    witness or as the chart witness of its refreshed source charts.
    """
    from ._model_codec import _same_model_content

    candidates = []
    if target.certification is not None:
        candidates.append(target.certification.inputs.fidelity_witness)
    if target.surface_source is not None:
        candidates.append(target.surface_source.chart_witness)
    for candidate in candidates:
        if candidate is not None and _same_model_content(candidate, stored):
            return candidate
    raise ValueError("Stored surface chart witness is not certified by the target.")


def _bind_surface_outcome(
    value: SurfaceMetricOutcome,
    source: CellMeshingResult,
    target: CellMeshingResult,
    lineage: MeshLineage,
    witness: Any,
    metric: MeshMetricField,
    /,
) -> SurfaceMetricOutcome:
    """Bind a changed surface outcome's edit, metric, and witness to the endpoints."""
    from ._model_codec import _same_model_content
    from ._topology_edit import assemble_topology_edit

    assembled, edit_lineage, _ = assemble_topology_edit(
        source.mesh, value.edit, numeric_version=target.mesh.numeric_version
    )
    order = np.argsort(np.asarray(target.mesh.vertex_global_ids), kind="stable")
    values = np.asarray(value.metric, dtype=np.float64)
    if (
        assembled.topology_id != target.mesh.topology_id
        or edit_lineage.lineage_id != lineage.lineage_id
        or values.shape[:1] != order.shape
        or not np.array_equal(np.asarray(metric.values), values[order])
        or not _same_model_content(value.geometry.target_chart_witness, witness)
    ):
        raise ValueError("Stored surface metric outcome does not bind its target.")
    return value._replace(geometry=value.geometry._replace(target_chart_witness=witness))


def _require_endpoint_lineage(
    source: CellMesh, lineage: MeshLineage, target: CellMesh, /
) -> None:
    if (
        lineage.source_topology_id != source.topology_id
        or lineage.target_topology_id != target.topology_id
        or any(
            record.source_entity_set_id
            != source.entity_set(record.dimension).entity_set_id
            or record.target_entity_set_id
            != target.entity_set(record.dimension).entity_set_id
            for record in lineage.entities
        )
    ):
        raise ValueError("Stored adaptation lineage does not bind its endpoints.")


def _require_hierarchy_mesh(hierarchy: BisectionHierarchy, mesh: CellMesh, /) -> None:
    """A hierarchy applies to the mesh with its active cells and vertex sets."""
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    cell_ids = np.concatenate(
        tuple(np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks)
    )
    rows = np.concatenate(
        tuple(
            vertex_ids[np.asarray(block.vertices, dtype=np.int64)]
            for block in mesh.blocks
        )
    )
    order = np.argsort(cell_ids, kind="stable")
    active = np.asarray(hierarchy.cell_global_ids, dtype=np.int64)
    ordered = np.asarray(hierarchy.ordered_vertices, dtype=np.int64)
    if (
        active.shape != cell_ids.shape
        or ordered.shape != rows.shape
        or not np.array_equal(active, cell_ids[order])
        or not np.array_equal(np.sort(ordered, axis=1), np.sort(rows[order], axis=1))
    ):
        raise ValueError("Stored bisection hierarchy does not apply to its mesh.")


def _target_metric(
    value: MeshMetricField, target: CellMesh, /, *, scope: MeshingScope | None = None
) -> MeshMetricField:
    """Rebind stored metric values (hashed as stored) to the target vertices."""
    if type(value) is not MeshMetricField:
        raise TypeError("Stored metric adaptation lacks its target metric.")
    rebuilt = MeshMetricField(
        MeshingScope(
            target.mesh_id,
            target.numeric_version,
            MeshingEntityKind.MESH,
            0,
            target.entity_set(0).entity_set_id,
            np.asarray(target.vertex_global_ids, dtype=np.int64),
        )
        if scope is None
        else scope,
        value.values,
        minimum_size=value.minimum_size,
        maximum_size=value.maximum_size,
        maximum_anisotropy=value.maximum_anisotropy,
    )
    from ._model_codec import _same_model_content

    if rebuilt.metric_id != value.metric_id or not _same_model_content(rebuilt, value):
        raise ValueError("Stored target metric does not bind the adapted target.")
    return rebuilt


def _mmg_evidence(
    value: MmgAdaptationResult, target: CellMeshingResult, /
) -> MmgAdaptationResult:
    """Rebind stored Mmg evidence to the recertified target; Mmg is not re-run."""
    from ._model_codec import _same_model_content

    if type(value) is not MmgAdaptationResult or (
        value.mesh.result_id != target.result_id
    ):
        raise ValueError("Stored Mmg evidence does not bind the adapted target.")
    references = value.references
    if type(references) is not MmgReferenceRetention or any(
        type(item) is not MmgReference for item in references.references
    ):
        raise TypeError("Stored Mmg evidence lost its reference retention.")
    if value.session.elapsed_seconds is not None:
        raise ValueError("Model archives do not persist Mmg session wall time.")
    rebuilt = MmgAdaptationResult(
        value.plan_id,
        target,
        None
        if value.metric is None
        else _target_metric(
            value.metric, target.mesh, scope=MmgProvider.vertex_scope(target.mesh)
        ),
        value.metric_representation,
        None
        if value.fields is None
        else _reconstructed(value.fields, "transfer_id", "Mmg field transfer"),
        _reconstructed(
            references,
            "evidence_id",
            "Mmg reference retention",
            references=tuple(
                _reconstructed(item, None, "Mmg reference")
                for item in references.references
            ),
        ),
        _reconstructed(value.session, "evidence_id", "Mmg session evidence"),
    )
    if rebuilt.result_id != value.result_id or not _same_model_content(rebuilt, value):
        raise ValueError("Stored Mmg evidence identity changed.")
    return rebuilt


def _omega_h_target_scope(stored: Any, target: CellMesh, /) -> MeshingScope:
    """Rebind a stored Omega_h vertex or cell scope to the restored target mesh."""
    if type(stored) is not MeshingScope:
        raise TypeError("Stored Omega_h scope must be MeshingScope.")
    dimension = int(stored.entity_dimension)
    if dimension not in (0, target.topological_dimension):
        raise ValueError("Stored Omega_h scope is neither vertices nor cells.")
    identifiers = np.asarray(stored.entity_ids, dtype=np.int64)
    expected = np.asarray(target.entity_set(dimension).entity_ids, dtype=np.int64)
    if identifiers.shape != expected.shape or not np.array_equal(
        np.sort(identifiers), np.sort(expected)
    ):
        raise ValueError("Stored Omega_h scope does not cover the adapted target.")
    return _entity_scope(target, dimension, identifiers)


def _omega_h_partition(stored: Any, /) -> OmegaHPartition:
    """Rebuild one rank partition from its stored native arrays."""
    from ._model_codec import _same_model_content

    if type(stored) is not OmegaHPartition:
        raise TypeError("Stored Omega_h partitions must be OmegaHPartition.")
    arrays = _RankArrays(
        vertex_global_ids=np.asarray(stored.vertex_global_ids),
        coordinates=np.asarray(stored.coordinates),
        metric=np.asarray(stored.metric),
        vertex_owner_ranks=np.asarray(stored.vertex_owner_ranks),
        vertex_owner_indices=np.asarray(stored.vertex_owner_indices),
        cell_global_ids=np.asarray(stored.cell_global_ids),
        cells=np.asarray(stored.cells),
        cell_class_ids=np.asarray(stored.cell_class_ids),
        cell_owner_ranks=np.asarray(stored.cell_owner_ranks),
        cell_owner_indices=np.asarray(stored.cell_owner_indices),
        facet_global_ids=np.asarray(stored.facet_global_ids),
        facet_vertices=np.asarray(stored.facet_vertices),
        facet_class_ids=np.asarray(stored.facet_class_ids),
        facet_owner_ranks=np.asarray(stored.facet_owner_ranks),
        fields=tuple(np.asarray(values) for values in stored.fields),
    )
    rebuilt = OmegaHPartition(stored.rank, arrays)
    if rebuilt.partition_id != stored.partition_id or not _same_model_content(
        rebuilt, stored
    ):
        raise ValueError("Stored Omega_h partition identity changed.")
    return rebuilt


def _omega_h_evidence(
    value: OmegaHAdaptationResult,
    source: CellMeshingResult,
    target: CellMeshingResult,
    /,
) -> OmegaHAdaptationResult:
    """Rebuild stored Omega_h evidence over the recertified endpoints.

    Omega_h is not re-run. Every identity is recomputed from its stored inputs,
    including the canonical worker record and the source field identities.
    """
    from ._model_codec import _same_model_content

    if (
        type(value) is not OmegaHAdaptationResult
        or value.target is None
        or value.metric is None
        or value.source_id != source.result_id
        or value.target.result_id != target.result_id
    ):
        raise ValueError("Stored Omega_h evidence does not bind the adapted endpoints.")
    stored = value.evidence
    if (
        type(stored) is not OmegaHAdaptationEvidence
        or type(stored.options) is not OmegaHOptions
        or any(type(item) is not OmegaHFieldEvidence for item in stored.fields)
        or type(value.classification) is not OmegaHClassification
        or any(type(item) is not OmegaHTransferredField for item in value.fields)
    ):
        raise TypeError("Stored Omega_h evidence lost its registered records.")
    if (value.distribution is None) != (stored.ranks == 1):
        raise ValueError("Stored Omega_h distribution differs from its rank count.")
    evidence = _reconstructed(
        stored,
        "evidence_id",
        "Omega_h adaptation evidence",
        options=_reconstructed(stored.options, "options_id", "Omega_h options"),
        fields=tuple(
            _reconstructed(item, "evidence_id", "Omega_h field evidence")
            for item in stored.fields
        ),
    )
    vertex_scope = _omega_h_target_scope(value.metric.scope, target.mesh)
    if vertex_scope.entity_dimension != 0:
        raise ValueError("The stored Omega_h metric is not a vertex field.")
    gathered = _GatheredTarget(
        target,
        _target_metric(value.metric, target.mesh, scope=vertex_scope),
        tuple(
            _reconstructed(
                item,
                "field_id",
                "Omega_h transferred field",
                scope=_omega_h_target_scope(item.scope, target.mesh),
            )
            for item in value.fields
        ),
        # The result stores neither; they only build a fresh distribution.
        np.zeros((0,), dtype=np.int32),
        np.zeros((0,), dtype=np.int64),
    )
    rebuilt = OmegaHAdaptationResult(
        source,
        value.metric_id,
        tuple(_omega_h_partition(item) for item in value.partitions),
        _reconstructed(
            value.classification, "classification_id", "Omega_h classification"
        ),
        evidence,
        gathered,
        None
        if value.distribution is None
        else _restored_distribution(value.distribution, target),
        target.provider,
        target.runtime,
        target.provenance,
    )
    if rebuilt.result_id != value.result_id or not _same_model_content(rebuilt, value):
        raise ValueError("Stored Omega_h evidence identity changed.")
    return rebuilt


def _restored_distribution(stored: Any, carrier: CellMeshingResult, /) -> Any:
    """Rebuild one stored distribution over its independently restored part."""
    from ..discretization import CellPartition
    from ._assembly import MeshPart
    from ._distribution import MeshDistribution
    from ._model_codec import _same_model_content

    if type(stored) is not MeshDistribution:
        raise TypeError("Stored distribution must be MeshDistribution.")
    part = MeshPart(stored.part.name, carrier)
    if part.part_id != stored.part.part_id:
        raise ValueError("Stored distribution names another part revision.")

    def rebuilt(halos: tuple[np.ndarray, ...] | None, /) -> MeshDistribution:
        return MeshDistribution(
            part,
            CellPartition(
                np.asarray(stored.partition.cell_owner), stored.partition.part_count
            ),
            cell_global_ids=np.asarray(stored.cell_global_ids),
            halo_global_ids=halos,
            split_factors=stored.split_factors,
            halo_width=stored.halo_width,
            cell_weights=np.asarray(stored.cell_weights),
            partition_kind=stored.evidence.kind,
            partition_provenance=stored.evidence.provenance,
            partition_feasible=stored.evidence.feasible,
        )

    # Halos are the exact adjacency reach unless explicit residency was supplied;
    # supplied residency is revalidated against that reach.
    distribution = rebuilt(None)
    if distribution.distribution_id != stored.distribution_id:
        distribution = rebuilt(
            tuple(np.asarray(value, dtype=np.int64) for value in stored.halo_global_ids)
        )
    if distribution.distribution_id != stored.distribution_id or not (
        _same_model_content(distribution, stored)
    ):
        raise ValueError("Stored distribution identity changed.")
    return distribution


def _restored_distribution_transition(
    stored: Any,
    source: CellMeshingResult,
    target: CellMeshingResult,
    lineage: MeshLineage,
    /,
) -> Any:
    """Rebuild ownership carried through the lineage onto the adapted target.

    Lineage routes and inherited ownership are recomputed exactly. A rebalanced
    target ownership came from a partition policy the result does not retain;
    it is hashed as stored and revalidated as a distribution of the target.
    """
    import sys

    from ._distribution import (
        _lineage_routes,
        _transition_owners,
        MeshDistributionTransition,
        MeshPartitionPolicy,
    )
    from ._model_codec import _same_model_content

    if type(stored) is not MeshDistributionTransition:
        raise TypeError("Stored distribution transition has the wrong owner.")
    source_distribution = _restored_distribution(stored.source, source)
    target_distribution = _restored_distribution(stored.target, target)
    routes = _lineage_routes(source_distribution, target_distribution.part, lineage)
    provenance = target_distribution.evidence.provenance
    if stored.rebalanced:
        if not provenance.startswith("transition-"):
            raise ValueError("Stored rebalanced ownership lost its transition origin.")
    else:
        # The inherited branch accepts exactly when every target cell inherits an
        # owner and no part is empty; the retained imbalance bound only selects
        # between this branch and a rebalance, which the stored flag records.
        owner, rebalanced, inherited = _transition_owners(
            source_distribution,
            target_distribution.part,
            routes,
            np.asarray(target_distribution.cell_weights, dtype=np.float64),
            MeshPartitionPolicy(
                target_distribution.evidence.kind,
                target_distribution.partition.part_count,
                halo_width=target_distribution.halo_width,
                maximum_imbalance=sys.float_info.max,
            ),
            None,
        )
        if (
            rebalanced
            or inherited != provenance
            or not np.array_equal(
                owner, np.asarray(target_distribution.partition.cell_owner)
            )
        ):
            raise ValueError("Stored inherited ownership differs from its lineage.")
    transition = MeshDistributionTransition(
        source_distribution,
        target_distribution,
        routes[0],
        routes[1],
        lineage,
        rebalanced=stored.rebalanced,
    )
    if transition.transition_id != stored.transition_id or not _same_model_content(
        transition, stored
    ):
        raise ValueError("Stored distribution transition identity changed.")
    return transition


def _restored_material_state(
    stored_mesh: Any, stored_geometry: Any, stored_coverage: Any, /
) -> tuple[Any, Any, Any]:
    """Restore one material-chain endpoint retained only inside the chain."""
    from ..discretization._cell_mesh import CellMesh
    from ..geometry._domain_certification import DomainCoverageCertificate
    from ..geometry._surface_source_archive import _geometry
    from ._model_codec import restore_model_archive_value
    from ._result import _restored_global_embedding

    if type(stored_coverage) is not DomainCoverageCertificate:
        raise TypeError("Stored material state lost its coverage certificate.")
    mesh = CellMesh.from_stored_fields(stored_mesh)
    geometry = _geometry(stored_geometry, mesh)
    coverage = DomainCoverageCertificate.from_restored_inputs(
        stored_coverage,
        mesh,
        geometry,
        _restored_global_embedding(stored_coverage.embedding, mesh, geometry),
        restore_model_archive_value(stored_coverage.domain),
        source=(
            None
            if stored_coverage.source is None
            else restore_model_archive_value(stored_coverage.source)
        ),
    )
    return mesh, geometry, coverage


def _restored_material_transition(
    stored: CellMeshTransition,
    source: CellMeshingResult,
    target: CellMeshingResult,
    geometry_transition: CellGeometryTransition | None,
    /,
) -> tuple[Any, Any]:
    """Re-prove every epoch map of a material chain over restored endpoints.

    The last map joins the restored adaptation endpoints; earlier epochs are
    restored from the chain with their coverages recomputed. Each map is
    revalidated by its owner against the restored reference root.
    """
    from ..geometry._volume_material_deformation import MaterialReferenceDeclaration
    from ..geometry._volume_material_map import (
        MaterialSourceMapCapability,
        VolumeMaterialChain,
        VolumeMaterialMap,
    )
    from ._adaptation import _material_layer_chart_parent
    from ._model_codec import _same_model_content, restore_model_archive_value

    chain = stored.volume_material_chain
    reference = (
        stored.material_reference
        if chain is None
        else chain.maps[0].reference
        if type(chain) is VolumeMaterialChain
        else None
    )
    if type(reference) is not MaterialReferenceDeclaration:
        raise TypeError("Stored material transition lost its reference declaration.")
    if (
        source.certification is None
        or target.certification is None
        or source.certification.coverage is None
        or target.certification.coverage is None
    ):
        raise ValueError("Material transitions require certified endpoint coverage.")
    restored_states: dict[str, tuple[Any, Any, Any, Any]] = {
        source.mesh.mesh_id: (
            source.mesh,
            source.geometry,
            source.certification.coverage,
            _material_layer_chart_parent(source, source.geometry),
        ),
        target.mesh.mesh_id: (
            target.mesh,
            target.geometry,
            target.certification.coverage,
            _material_layer_chart_parent(target, target.geometry),
        ),
    }

    def state(mesh: Any, geometry: Any, coverage: Any, /) -> tuple[Any, Any, Any, Any]:
        restored = restored_states.get(mesh.mesh_id)
        if restored is None:
            restored = (*_restored_material_state(mesh, geometry, coverage), None)
            restored_states[mesh.mesh_id] = restored
        if restored[2].certificate_id != coverage.certificate_id:
            raise ValueError("Stored material coverage differs from its endpoint.")
        return restored

    root_mesh, root_geometry, root_coverage, root_parent = state(
        reference.root_mesh, reference.root_geometry, reference.root_coverage
    )
    if chain is None:
        material_reference = MaterialReferenceDeclaration.revalidate(
            reference,
            root_mesh,
            root_geometry,
            MaterialSourceMapCapability.revalidate(reference.source_map),
            root_coverage,
            chart_parent=root_parent,
        )
        if not _same_model_content(material_reference, reference):
            raise ValueError("Stored material reference changed.")
        return None, material_reference
    if (
        stored.material_reference is not None
        and stored.material_reference.declaration_id != reference.declaration_id
    ):
        raise ValueError("A transition reference must be its chain's reference.")
    maps = []
    for index, item in enumerate(chain.maps):
        if type(item) is not VolumeMaterialMap:
            raise TypeError("Stored material chain lost a material map.")
        source_state = state(item.source_mesh, item.source_geometry, item.source_coverage)
        target_state = state(item.target_mesh, item.target_geometry, item.target_coverage)
        stored_transition = item.deformation.coordinate_transition
        if index == len(chain.maps) - 1:
            if (
                source_state[0] is not source.mesh
                or target_state[0] is not target.mesh
                or (stored_transition is None) != (geometry_transition is None)
            ):
                raise ValueError("The last material map must join the adaptation.")
            coordinate_transition = geometry_transition
        else:
            coordinate_transition = (
                None
                if stored_transition is None
                else restore_model_archive_value(stored_transition)
            )
        maps.append(
            VolumeMaterialMap.revalidate(
                item,
                source_state[0],
                source_state[1],
                target_state[0],
                target_state[1],
                root_mesh=root_mesh,
                root_geometry=root_geometry,
                root_coverage=root_coverage,
                source_coverage=source_state[2],
                target_coverage=target_state[2],
                coordinate_transition=coordinate_transition,
                root_chart_parent=root_parent,
                source_chart_parent=source_state[3],
                target_chart_parent=target_state[3],
            )
        )
    return VolumeMaterialChain.revalidate(chain, maps), None


def restore_route_adaptation_result(raw: MeshAdaptationResult, /) -> MeshAdaptationResult:
    """Restore one native or device bisection or metric adaptation from its fields.

    Both endpoints are recertified. Lineage, stencil, geometry and mesh
    transition, P1 transfer, target metric, hierarchy, and compliance are rebuilt
    by their owners and bound to the endpoints. Change, status, and transition
    kind are decided exactly from the evidence counts and status. Floating-point
    evidence is hashed as stored. The request and policy are not retained, so
    the prepared identity is hashed as stored and bound by the result identity.
    """
    from ..discretization._cell_geometry_transition_types import CellGeometryTransition
    from ._adaptation import (
        _compliance,
        _native_vertex_transfer,
        _RouteOutcome,
        _StoredAdaptationBinding,
        MeshAdaptationResult,
    )
    from ._model_codec import (
        _register_model_archive_types,
        _same_model_content,
        restore_model_archive_value,
    )
    from ._result import CellMeshingResult

    _register_model_archive_types()
    if type(raw) is not MeshAdaptationResult:
        raise TypeError("Route restoration requires a stored MeshAdaptationResult.")
    route = raw.route
    if route.value not in _ROUTE_EVIDENCE:
        raise ValueError(
            f"Stored {route.value} adaptation has no canonical restore owner."
        )
    if not isinstance(raw.prepared_id, str) or not raw.prepared_id:
        raise ValueError("Stored adaptation lost its prepared identity.")
    if raw.elapsed_seconds is not None:
        raise ValueError("Model archives do not persist adaptation wall time.")
    if raw.cell_content_transfer is not None or (
        raw.surface_chart_witness is not None
        and type(raw.evidence) is not SurfaceMetricOutcome
    ):
        raise ValueError("Stored adaptation retains evidence foreign to its route.")
    evidence = _route_evidence(raw.evidence, route.value)
    changed, status, kind, conservative = _route_decision(evidence)
    if raw.status is not status:
        raise ValueError("Stored adaptation status differs from its exact evidence.")
    source = CellMeshingResult.from_stored_fields(raw.source)
    transition = lineage = stencil = transfer = metric = remap = None
    if not changed:
        if (
            raw.target.result_id != raw.source.result_id
            or not _same_model_content(raw.target, raw.source)
            or raw.transition is not None
            or raw.lineage is not None
            or raw.stencil is not None
            or raw.transfer is not None
            or raw.metric is not None
            or raw.surface_chart_witness is not None
            or raw.conservative_remap is not None
        ):
            raise ValueError(
                "A source-preserving adaptation retains no transition, transfer, or metric."
            )
        target = source
    else:
        target = CellMeshingResult.from_stored_fields(raw.target)
        stored = raw.transition
        if raw.lineage is None or stored is None:
            raise ValueError("Stored adaptation transition lacks its lineage.")
        provider_route = type(evidence) in (MmgAdaptationResult, OmegaHAdaptationResult)
        if provider_route:
            from ._adaptation import _unknown_lineage

            # Provider remeshing has unknown lineage: every entity is created
            # or deleted, and no vertex stencil exists.
            lineage = _unknown_lineage(source.mesh, target.mesh)
            if (
                raw.stencil is not None
                or stored.vertex_stencil is not None
                or not _same_model_content(raw.lineage, lineage)
                or not _same_model_content(stored.lineage, lineage)
            ):
                raise ValueError("Stored provider transition claims a known lineage.")
        else:
            if raw.stencil is None:
                raise ValueError("Stored adaptation transition lacks its stencil.")
            lineage = _lineage(raw.lineage)
            _require_endpoint_lineage(source.mesh, lineage, target.mesh)
            stencil = _stencil(raw.stencil)
            if (
                stencil.source_entity_set_id != source.mesh.entity_set(0).entity_set_id
                or stencil.target_entity_set_id != target.mesh.entity_set(0).entity_set_id
                or stored.vertex_stencil is None
                or not _same_model_content(stored.vertex_stencil, stencil)
                or not _same_model_content(stored.lineage, lineage)
            ):
                raise ValueError("Stored adaptation transition has inconsistent lineage.")
        geometry_transition = None
        if stored.geometry_transition is not None:
            geometry_transition = restore_model_archive_value(stored.geometry_transition)
            if (
                type(geometry_transition) is not CellGeometryTransition
                or not _same_model_content(geometry_transition.source_mesh, source.mesh)
                or not _same_model_content(
                    geometry_transition.source_geometry, source.geometry
                )
                or not _same_model_content(geometry_transition.target_mesh, target.mesh)
                or not _same_model_content(
                    geometry_transition.target_geometry, target.geometry
                )
            ):
                raise ValueError(
                    "Stored geometry transition does not bind the adaptation endpoints."
                )
        material_chain = material_reference = None
        if (
            stored.volume_material_chain is not None
            or stored.material_reference is not None
        ):
            material_chain, material_reference = _restored_material_transition(
                stored, source, target, geometry_transition
            )
        transition = CellMeshTransition(
            source.mesh.mesh_id,
            source.mesh.topology_id,
            target,
            lineage,
            kind,
            vertex_stencil=stencil,
            geometry_transition=geometry_transition,
            volume_material_chain=material_chain,
            material_reference=material_reference,
        )
        if transition.transition_id != stored.transition_id:
            raise ValueError("Stored adaptation transition identity changed.")
        # Republished polyhedral regeneration and exact surface projections bind
        # only constant-preserving transfer; other native routes claim linear
        # preservation.
        linear_transfer = not (
            type(evidence) is PolyhedralAdaptationEvidence
            and evidence.operation is PolyhedralAdaptationOperation.REGENERATE
            and bool(source.attributes or source.associations)
        ) and not (
            type(evidence) is SurfaceMetricOutcome
            and evidence.geometry.relation
            is SurfaceMetricGeometryRelation.EXACT_PROJECTION
        )
        transfer = (
            None
            if stencil is None
            else _native_vertex_transfer(
                source.mesh,
                target.mesh,
                lineage,
                stencil,
                conservative=conservative,
                linear_transfer=linear_transfer,
            )
        )
        if (transfer is None) != (raw.transfer is None) or (
            transfer is not None and not _same_model_content(transfer, raw.transfer)
        ):
            raise ValueError("Stored adaptation transfer differs from its exact stencil.")
    surface_chart_witness = None
    if type(evidence) is BisectionEvidence:
        if type(raw.hierarchy) is not BisectionHierarchy or raw.metric is not None:
            raise ValueError("Stored bisection lacks its hierarchy or retains a metric.")
        hierarchy = _reconstructed(raw.hierarchy, "hierarchy_id", "bisection hierarchy")
        _require_hierarchy_mesh(hierarchy, target.mesh)
    elif type(evidence) is SurfaceMetricOutcome:
        if (
            raw.hierarchy is not None
            or raw.conservative_remap is not None
            or (raw.metric is None) == changed
            or (raw.surface_chart_witness is None) == changed
        ):
            raise ValueError(
                "Stored surface adaptation must retain exactly its metric and witness."
            )
        hierarchy = None
        if changed:
            assert lineage is not None and raw.metric is not None
            metric = _target_metric(raw.metric, target.mesh)
            surface_chart_witness = _surface_chart_witness(
                raw.surface_chart_witness, target
            )
            evidence = _bind_surface_outcome(
                evidence, source, target, lineage, surface_chart_witness, metric
            )
        if not _same_model_content(evidence, raw.evidence):
            raise ValueError("Stored surface metric outcome changed.")
    elif type(evidence) is MmgAdaptationResult:
        if raw.hierarchy is not None or raw.conservative_remap is not None:
            raise ValueError("Stored Mmg adaptation retains foreign evidence.")
        hierarchy = None
        evidence = _mmg_evidence(evidence, target)
        metric = evidence.metric
        stored_metric = raw.metric
        if (metric is None) != (stored_metric is None) or (
            metric is not None
            and stored_metric is not None
            and metric.metric_id != stored_metric.metric_id
        ):
            raise ValueError("Stored Mmg adaptation metric differs from its evidence.")
    elif type(evidence) is OmegaHAdaptationResult:
        if raw.hierarchy is not None or raw.conservative_remap is not None:
            raise ValueError("Stored Omega_h adaptation retains foreign evidence.")
        hierarchy = None
        evidence = _omega_h_evidence(evidence, source, target)
        metric = evidence.metric
        if (
            metric is None
            or raw.metric is None
            or metric.metric_id != raw.metric.metric_id
        ):
            raise ValueError(
                "Stored Omega_h adaptation metric differs from its evidence."
            )
    elif type(evidence) is PolyhedralAdaptationEvidence:
        if raw.metric is not None or (raw.conservative_remap is None) == changed:
            raise ValueError(
                "Stored polyhedral adaptation must retain exactly its conservative remap."
            )
        hierarchy = _polyhedral_hierarchy(raw.hierarchy, source.mesh, target.mesh)
        if (
            evidence.source_mesh_id != source.mesh.mesh_id
            or hierarchy.source_mesh_id != source.mesh.mesh_id
        ):
            raise ValueError("Stored polyhedral evidence names another source mesh.")
        if changed:
            if (
                evidence.target_mesh_id != target.mesh.mesh_id
                or not np.array_equal(
                    evidence.target_region_classes, hierarchy.region_classes
                )
                or not set(map(int, evidence.changed_cell_ids))
                <= set(map(int, np.asarray(hierarchy.target_cell_ids)))
            ):
                raise ValueError(
                    "Stored polyhedral evidence differs from its target hierarchy."
                )
            remap = _polyhedral_remap(
                raw.conservative_remap, source.mesh, target.mesh, evidence
            )
    else:
        if (
            raw.hierarchy is not None
            or raw.conservative_remap is not None
            or (raw.metric is None) == changed
        ):
            raise ValueError(
                "Stored metric adaptation must retain exactly its target metric."
            )
        hierarchy = None
        if raw.metric is not None:
            metric = _target_metric(raw.metric, target.mesh)
    distribution = None
    if raw.distribution is not None:
        if lineage is None:
            raise ValueError("A stored distribution transition requires its lineage.")
        distribution = _restored_distribution_transition(
            raw.distribution, source, target, lineage
        )
    binding = _StoredAdaptationBinding(route, source, raw.prepared_id)
    rebuilt = MeshAdaptationResult(
        binding,
        _RouteOutcome(
            status,
            target,
            transition,
            lineage,
            stencil,
            transfer,
            metric,
            evidence,
            hierarchy,
            None,
            remap,
            surface_chart_witness,
        ),
        _compliance(binding, target),
        distribution,
        None,
    )
    if rebuilt.result_id != raw.result_id or not _same_model_content(
        rebuilt.compliance, raw.compliance
    ):
        raise ValueError("Stored adaptation result identity or compliance changed.")
    return rebuilt
