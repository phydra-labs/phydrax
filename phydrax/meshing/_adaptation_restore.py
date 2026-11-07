#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Cold reconstruction of a retained native mixed adaptation."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from .._fingerprint import canonical_fingerprint
from ._lineage import (
    CellMeshTransition,
    EntityLineage,
    EntityLineageKind,
    MeshLineage,
    VertexInterpolationStencil,
)
from ._mixed_adaptation import (
    MixedAdaptationEvidence,
    MixedAdaptationHierarchy,
    MixedLayerColumns,
)
from ._mixed_hierarchy_reproof import validate_mixed_restoration
from ._periodic import PeriodicQuotientHistory


if TYPE_CHECKING:
    from ..discretization._cell_geometry_transition_types import CellGeometryTransition
    from ._adaptation import (
        MarkedMeshAdaptation,
        MeshAdaptationPolicy,
        MeshAdaptationResult,
    )
    from ._layer_core_root_geometry import ComputedLayerCoreRoot
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
            from ._layer_core_association_transfer import NativeLayerCoreAssociationTransfer

            if type(association_transfer) is not NativeLayerCoreAssociationTransfer:
                raise TypeError("Computed root requires its native layer association owner.")
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
    if not math.isfinite(raw.elapsed_seconds) or raw.elapsed_seconds < 0:
        raise ValueError("Stored mixed elapsed time is invalid.")
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
        raw.elapsed_seconds,
    )
    if rebuilt.result_id != expected_result_id or not _same_model_content(
        rebuilt.compliance, raw.compliance
    ):
        raise ValueError("Stored mixed result identity or compliance changed.")
    return rebuilt
