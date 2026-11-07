#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Canonical owner-local finite-volume archive context reconstruction."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, TYPE_CHECKING

import jax
import numpy as np

import phydrax as phx
from phydrax.meshing._overset import OversetPartSpec
from phydrax.meshing._overset_finite_volume import (
    prepare_owner_local_finite_volume_overset,
)


if TYPE_CHECKING:
    from phydrax.discretization._fv_precision import FiniteVolumePrecisionPolicy


def _rebuild_finite_volume_precision_policy(
    value: object, /
) -> FiniteVolumePrecisionPolicy:
    from phydrax._precision import PrecisionResolution
    from phydrax.discretization._fv_precision import FiniteVolumePrecisionPolicy

    if not isinstance(value, FiniteVolumePrecisionPolicy):
        raise TypeError("Stored FV precision policy has the wrong owner type.")
    if type(value) is not FiniteVolumePrecisionPolicy:
        raise TypeError("Stored FV precision policy has the wrong concrete owner type.")
    resolution = None
    if value.resolution_id is not None:
        resolution = PrecisionResolution(
            value.request,
            value.provider,
            dict(value.request.requested),
        )
        if resolution.resolution_id != value.resolution_id:
            raise ValueError("Stored FV precision resolution identity changed.")
    policy = FiniteVolumePrecisionPolicy(
        value.storage_dtype,
        reconstruction_dtype=value.reconstruction_dtype,
        flux_dtype=value.flux_dtype,
        reduction_dtype=value.reduction_dtype,
        output_dtype=value.output_dtype,
        checkpoint_dtype=value.checkpoint_dtype,
        resolution=resolution,
    )
    if policy.policy_id != value.policy_id:
        raise ValueError("Stored FV precision policy does not rebuild canonically.")
    return policy


def _rebuild_finite_volume_runtime_state_archive_recipe(
    value: object, source_runtime: object, /
) -> Any:
    """Revalidate a stored accepted state against its rebuilt source runtime."""
    from phydrax._model._structure import FiniteVolumeRuntimeStateArchiveRecipe
    from phydrax.solver._finite_volume_content import FiniteVolumeConservativeContentState
    from phydrax.solver._finite_volume_runtime import (
        FiniteVolumeRuntimeState,
        FiniteVolumeValidatedRuntimeState,
        PreparedFiniteVolumeRuntime,
    )
    from phydrax.solver._finite_volume_topology_events import (
        FiniteVolumeTopologyEventJournal,
    )

    if type(value) is not FiniteVolumeRuntimeStateArchiveRecipe:
        raise TypeError("Stored FV runtime state must use its canonical archive recipe.")
    if not isinstance(source_runtime, PreparedFiniteVolumeRuntime):
        raise TypeError(
            "Restoring an FV state requires its current prepared source runtime."
        )
    stored_content = value.content_state
    if type(stored_content) is not FiniteVolumeConservativeContentState:
        raise TypeError("Stored FV archive lost its canonical conservative content.")
    precision = _rebuild_finite_volume_precision_policy(stored_content.precision)
    content = FiniteVolumeConservativeContentState(
        stored_content.conservative_content,
        stored_content.effective_cell_volumes,
        stored_content.active_cell_mask,
        stored_content.time,
        topology_epoch_id=stored_content.topology_epoch_id,
        geometry_family_id=stored_content.geometry_family_id,
        geometry_layout_id=stored_content.geometry_layout_id,
        geometry_version=stored_content.geometry_version,
        evidence_policy_id=stored_content.evidence_policy_id,
        evidence_version=stored_content.evidence_version,
        precision=precision,
        owned_cell_mask=stored_content.owned_cell_mask,
    )
    journal = FiniteVolumeTopologyEventJournal.from_archive_record(
        dict(value.topology_journal_record), dict(value.topology_journal_arrays)
    )
    sliding_plan = source_runtime.sliding_plan
    sliding_coupling = None
    if sliding_plan is None:
        if value.sliding_coupling_id is not None:
            raise ValueError("Stored FV state requires a missing source sliding plan.")
    else:
        if value.sliding_coupling_id is None:
            raise ValueError(
                "Stored sliding FV state omitted its exact coupling identity."
            )
        sliding_coupling = sliding_plan.coupling(value.sliding_shift)
        if sliding_coupling.coupling_id != value.sliding_coupling_id:
            raise ValueError(
                "Stored FV sliding coupling differs from the current source plan."
            )
    state = FiniteVolumeRuntimeState(
        content,
        journal,
        value.step_size,
        accepted_step=value.accepted_step,
        last_status=value.last_status,
        controller_state=value.controller_state,
        integrator_state=value.integrator_state,
        output_cursor=value.output_cursor,
        sliding_coupling=sliding_coupling,
        sliding_shift=value.sliding_shift,
        sliding_event_id=value.sliding_event_id,
    )
    restored = source_runtime.validate_restored_accepted_state(
        state,
        value.source_cell_ids,
        expected_runtime_id=value.expected_runtime_id,
        expected_epoch_id=value.expected_epoch_id,
        expected_state_id=value.expected_state_id,
    )
    if not isinstance(restored, FiniteVolumeValidatedRuntimeState):
        raise TypeError("FV source-state validation returned no production token.")
    return restored


def _resolve_finite_volume_archive_source(
    source_id: str | None,
    sources: Mapping[str, Callable[..., Any]] | None,
    /,
) -> Callable[..., Any] | None:
    """Resolve the archived semantic identity without serializing executable code."""
    if source_id is None:
        return None
    if sources is None or source_id not in sources:
        raise ValueError(f"Unregistered FV source: {source_id!r}.")
    source = sources[source_id]
    if not callable(source):
        raise TypeError("Registered FV source must be callable.")
    return source


def _finite_volume_archive_target_receptor_spec(
    source_spec: Any, source_part: Any, target_part: Any, execution_group: Any, /
) -> Any:
    """Rebind scopes through sharded scientific-identity directories.

    Compact facet ordinals are local; cells retain global IDs. All three scope
    kinds use the same registration, coverage, and reply-validation protocol.
    """
    from phydrax._fingerprint import canonical_fingerprint
    from phydrax.meshing._distribution import (
        _owner_exchange_message_banks,
        _owner_group_allgather,
        _owner_group_fail_if_any,
    )

    rank = execution_group.spec.process_indices.index(jax.process_index())
    parts = len(execution_group.spec.process_indices)
    source_view = source_part.carrier
    target_view = target_part.carrier
    facet_dimension = target_view.topological_dimension - 1
    local_error = None
    source_keys: list[tuple[int, ...]] = []
    target_keys: list[tuple[int, ...]] = []
    target_ids: list[np.ndarray] = []
    dimensions = (facet_dimension, facet_dimension, target_view.topological_dimension)
    scope_names = ("boundary", "wall", "excluded")

    def keys(view: Any, dimension: int, identifiers: np.ndarray) -> list[tuple[int, ...]]:
        if dimension == view.topological_dimension:
            return [(int(identifier),) for identifier in identifiers]
        vertices = np.asarray(view.vertex_ids, dtype=np.int64)
        return [
            tuple(sorted(map(int, vertices[np.asarray(loop, dtype=np.int64)])))
            for loop in view.entity_vertex_loops(dimension, identifiers)
        ]

    try:
        if source_view.topological_dimension != target_view.topological_dimension:
            raise ValueError("Repartition changed receptor facet dimension.")
        for kind, (name, dimension) in enumerate(
            zip(scope_names, dimensions, strict=True)
        ):
            scope = getattr(source_spec, name)
            scoped_keys: list[tuple[int, ...]] = []
            if scope is not None:
                source_part.require_scope(scope)
                if scope.entity_dimension != dimension:
                    raise ValueError(f"Receptor {name} scope has the wrong dimension.")
                scoped_keys = keys(
                    source_view, dimension, np.asarray(scope.entity_ids, dtype=np.int64)
                )
            _, identifiers = target_view.entity_binding(dimension)
            identifiers = np.asarray(identifiers, dtype=np.int64)
            resident_keys = keys(target_view, dimension, identifiers)
            if (
                any(
                    not key or len(set(key)) != len(key)
                    for key in (*scoped_keys, *resident_keys)
                )
                or len(set(scoped_keys)) != len(scoped_keys)
                or len(set(resident_keys)) != len(resident_keys)
            ):
                raise ValueError(f"Receptor {name} identities are malformed or repeated.")
            source_keys.extend((kind, *key) for key in scoped_keys)
            target_keys.extend((kind, *key) for key in resident_keys)
            target_ids.append(identifiers)
    except Exception as error:
        local_error = f"{type(error).__name__}: {error}"
    _owner_group_fail_if_any(
        local_error, execution_group, "fv_archive_scope_keys", "receptor scopes"
    )
    width = int(
        np.max(
            np.asarray(
                _owner_group_allgather(
                    np.asarray(
                        (
                            max(
                                (len(key) for key in (*source_keys, *target_keys)),
                                default=0,
                            ),
                        ),
                        dtype=np.int64,
                    ),
                    execution_group,
                    "fv_archive_scope_key_width",
                )
            )
        )
    )
    width = max(width, 1)

    def padded(key: tuple[int, ...]) -> tuple[int, ...]:
        return (*key, *(-1 for _ in range(width - len(key))))

    def bucket(key: tuple[int, ...]) -> int:
        return (
            int(
                canonical_fingerprint({"kind": "fv-archive-receptor-scope", "key": key})[
                    :16
                ],
                16,
            )
            % parts
        )

    def exchange(
        rows: list[list[tuple[int, ...]]], columns: int, axis: str
    ) -> dict[int, np.ndarray]:
        banks = tuple(
            np.asarray(values if destination != rank else [], dtype=np.int64).reshape(
                (-1, columns)
            )
            for destination, values in enumerate(rows)
        )
        received = _owner_exchange_message_banks(banks, rank, execution_group, axis)
        if rows[rank]:
            received[rank] = np.asarray(rows[rank], dtype=np.int64).reshape((-1, columns))
        return received

    registrations: list[list[tuple[int, ...]]] = [[] for _ in range(parts)]
    for key in source_keys:
        registrations[bucket(key)].append(padded(key))
    registered = exchange(registrations, width, "fv_archive_scope_register")
    shard = {
        tuple(int(value) for value in row) for bank in registered.values() for row in bank
    }
    queries: list[list[tuple[int, ...]]] = [[] for _ in range(parts)]
    for row, key in enumerate(target_keys):
        queries[bucket(key)].append((*padded(key), row))
    incoming = exchange(queries, width + 1, "fv_archive_scope_queries")
    replies: list[list[tuple[int, ...]]] = [[] for _ in range(parts)]
    found: set[tuple[int, ...]] = set()
    for sender, bank in incoming.items():
        for row in bank:
            key = tuple(int(value) for value in row[:-1])
            selected = key in shard
            if selected:
                found.add(key)
            replies[sender].append((int(row[-1]), int(selected)))
    _owner_group_fail_if_any(
        None
        if found == shard
        else "A scoped source entity is absent from the target mesh.",
        execution_group,
        "fv_archive_scope_coverage",
        "receptor scopes",
    )
    returned = exchange(replies, 2, "fv_archive_scope_replies")
    selected = np.zeros((len(target_keys),), dtype=np.bool_)
    seen = np.zeros((len(target_keys),), dtype=np.bool_)
    local_error = None
    for bank in returned.values():
        for row, value in bank:
            if row < 0 or row >= len(target_keys) or seen[row] or value not in (0, 1):
                local_error = "Receptor scope replies are malformed."
                continue
            seen[row] = True
            selected[row] = bool(value)
    if not np.all(seen):
        local_error = "Receptor scope replies are incomplete."
    _owner_group_fail_if_any(
        local_error, execution_group, "fv_archive_scope_validate", "receptor scopes"
    )
    rebound = []
    offset = 0
    local_error = None
    try:
        for dimension, identifiers in zip(dimensions, target_ids, strict=True):
            resident = np.sort(identifiers[selected[offset : offset + identifiers.size]])
            offset += identifiers.size
            rebound.append(
                None if resident.size == 0 else target_part.scope(dimension, resident)
            )
        result = OversetPartSpec(
            source_spec.part_name,
            boundary=rebound[0],
            wall=rebound[1],
            excluded=rebound[2],
            solid_query=source_spec.solid_query,
            solid_index=source_spec.solid_index,
            priority=source_spec.priority,
            owner_rank=source_spec.owner_rank,
            image_rotation=source_spec.image_rotation,
            image_translation=source_spec.image_translation,
        )
    except Exception as error:
        local_error = f"{type(error).__name__}: {error}"
    _owner_group_fail_if_any(
        local_error, execution_group, "fv_archive_scope_rebind", "receptor scopes"
    )
    return result


def _finite_volume_archive_overset_part_spec(value: Any, /) -> Any:
    from phydrax._model._structure import FiniteVolumeOversetPartSpecArchiveRecipe
    from phydrax.meshing._overset import OversetPartSpec

    if type(value) is FiniteVolumeOversetPartSpecArchiveRecipe:
        rebuilt = OversetPartSpec(
            value.part_name,
            wall=value.wall,
            boundary=value.boundary,
            excluded=value.excluded,
            solid_index=value.solid_index,
            priority=value.priority,
            owner_rank=value.owner_rank,
            image_rotation=value.image_rotation,
            image_translation=value.image_translation,
        )
        if rebuilt.spec_id != value.expected_spec_id:
            raise ValueError("Stored FV overset part spec identity changed.")
        return rebuilt
    if type(value) is OversetPartSpec:
        if value.solid_query is not None:
            raise TypeError("Cold FV context cannot persist an unregistered solid query.")
        return OversetPartSpec(
            value.part_name,
            wall=value.wall,
            boundary=value.boundary,
            excluded=value.excluded,
            solid_index=value.solid_index,
            priority=value.priority,
            owner_rank=value.owner_rank,
            image_rotation=value.image_rotation,
            image_translation=value.image_translation,
        )
    raise TypeError("FV overset context requires its canonical part-spec record.")


def _finite_volume_archive_overset_policy(value: Any, /) -> Any:
    from phydrax._model._structure import FiniteVolumeOversetPolicyArchiveRecipe
    from phydrax.meshing._overset import OversetPolicy

    if type(value) is FiniteVolumeOversetPolicyArchiveRecipe:
        rebuilt = OversetPolicy(
            fringe_layers=value.fringe_layers,
            donor_priority=value.donor_priority,
        )
        if rebuilt.policy_id != value.expected_policy_id:
            raise ValueError("Stored FV overset policy identity changed.")
        return rebuilt
    if type(value) is OversetPolicy:
        return OversetPolicy(
            fringe_layers=value.fringe_layers,
            donor_priority=value.donor_priority,
        )
    raise TypeError("FV overset context requires its canonical policy record.")


def _finite_volume_archive_execution_group(context: Any, /) -> Any:
    from phydrax._execution_resources import ExecutionGroupSpec
    from phydrax._execution_runtime import ExecutionGroup

    devices = tuple(sorted(jax.devices(), key=lambda device: device.process_index))
    process_indices = tuple(device.process_index for device in devices)
    if process_indices != context.process_indices or len(devices) != len(
        context.process_indices
    ):
        raise ValueError(
            "Fresh FV worker placement differs from the archived group ranks."
        )
    specification = ExecutionGroupSpec(
        context.group_id,
        context.process_indices,
        context.device_keys,
        mesh_axes=context.mesh_axes,
        parent_group_id=context.parent_group_id,
        process_host_ids=context.process_host_ids,
    )
    return ExecutionGroup(specification, devices)


def _finite_volume_archive_rebuild_context_views(
    context: Any,
    execution_group: Any,
    /,
) -> Any:
    from phydrax.discretization._view_support import OwnerLocalCellGeometry
    from phydrax.graph._partition import GraphPartitionPlan
    from phydrax.meshing._contracts import SurfaceMeshingSpec

    if len(context.source_views) != len(context.part_names) or len(
        context.source_specifications
    ) != len(context.part_names):
        raise ValueError("FV archive source views/specifications are not aligned.")
    source_specifications = tuple(
        SurfaceMeshingSpec.from_stored_fields(value)
        for value in context.source_specifications
    )
    source_views = tuple(
        OwnerLocalCellGeometry.from_stored_fields(
            value,
            execution_group,
            current_specification=specification,
            axis_name=f"{context.owner_local_axis_name}_archive_source_{name}",
        )
        for name, value, specification in zip(
            context.part_names,
            context.source_views,
            source_specifications,
            strict=True,
        )
    )
    target_specification = SurfaceMeshingSpec.from_stored_fields(
        context.target_specification
    )
    target_plan = GraphPartitionPlan.from_stored_fields(
        context.target_graph_partition_plan
    )
    # The stored target view is the expected result of the graph repartition,
    # which the context build replays from the restored source views and the
    # stored plan; the replayed view must reproduce its exact local identity.
    target_view = context.target_view
    if type(target_view) is not OwnerLocalCellGeometry:
        raise TypeError("FV archive target view must be a canonical owner-local view.")
    return context._replace(
        source_views=source_views,
        source_specifications=source_specifications,
        source_overset_specs=tuple(
            _finite_volume_archive_overset_part_spec(value)
            for value in context.source_overset_specs
        ),
        source_overset_policy=_finite_volume_archive_overset_policy(
            context.source_overset_policy
        ),
        target_overset_specs=tuple(
            _finite_volume_archive_overset_part_spec(value)
            for value in context.target_overset_specs
        ),
        target_graph_partition_plan=target_plan,
        target_specification=target_specification,
        target_view=target_view,
    )


def _finite_volume_archive_build_context(
    context: Any,
    execution_group: Any,
    /,
    *,
    validate_expected: bool,
    sources: Mapping[str, Callable[..., Any]] | None = None,
) -> dict[str, Any]:
    from phydrax._execution_runtime import ExecutionGroup
    from phydrax.discretization._topology_epoch import TopologyEpoch
    from phydrax.discretization._view_support import OwnerLocalCellGeometry
    from phydrax.discretization.finite_volume._coupling import (
        UnstructuredFiniteVolumeCouplingPlan,
    )
    from phydrax.discretization.finite_volume._unstructured_dynamics import (
        UnstructuredFiniteVolumeBoundarySet,
        UnstructuredFiniteVolumeMethodPlan,
    )
    from phydrax.equations._conservation import (
        compile_conservation_problem,
        ConservationProblemIR,
    )
    from phydrax.equations._hyperbolic_systems import EulerSystem
    from phydrax.equations._materials import IdealGasMaterial
    from phydrax.graph._partition import GraphPartitionPlan
    from phydrax.meshing._assembly import MeshAssembly, MeshPart, MeshPartManifest
    from phydrax.meshing._distribution import partition_owner_local_graph
    from phydrax.meshing._overset import (
        prepare_overset_connectivity,
        prepare_overset_field_transfer,
    )
    from phydrax.meshing.providers._native_periodic import (
        _repartition_owner_local_periodic_assembly,
    )
    from phydrax.solver._finite_volume_runtime import (
        FiniteVolumeStepPolicy,
        PreparedFiniteVolumeRuntime,
    )
    from phydrax.solver._finite_volume_topology_events import (
        FiniteVolumeTopologyArtifacts,
    )

    if not isinstance(validate_expected, bool):
        raise TypeError("validate_expected must be bool.")
    if not isinstance(execution_group, ExecutionGroup):
        raise TypeError("FV archive replay requires its explicit live ExecutionGroup.")
    if (
        len(context.part_names) != len(context.source_views)
        or len(context.part_names) != len(context.source_specifications)
        or len(context.process_indices) != len(execution_group.spec.process_indices)
    ):
        raise ValueError("FV archive source or group declarations are incomplete.")
    if (
        context.system_kind != "euler"
        or context.reconstruction_kind != "piecewise-constant"
        or context.flux_kind != "rusanov"
        or context.boundary_condition_kind != "extrapolation"
        or context.positivity_fallback_kind != "einfeldt-hll"
    ):
        raise ValueError(
            "FV archive runtime recipe selects an unregistered callable path."
        )
    source = _resolve_finite_volume_archive_source(context.source_id, sources)
    donor_name = context.donor_part_name
    receptor_name = context.receptor_part_name
    if (
        donor_name not in context.part_names
        or receptor_name not in context.part_names
        or donor_name == receptor_name
    ):
        raise ValueError("FV archive runtime recipe has invalid donor/receptor roles.")

    source_views: dict[str, OwnerLocalCellGeometry] = {}
    for name, stored_view in zip(context.part_names, context.source_views, strict=True):
        if type(stored_view) is not OwnerLocalCellGeometry:
            raise TypeError(
                "FV archive source views must be canonical owner-local geometries."
            )
        source_views[name] = stored_view
    if any(source_views[name].part_name != name for name in context.part_names):
        raise ValueError(
            "FV archive source view names differ from their typed authority."
        )
    if any(
        view.specification_id is not None
        and specification.specification_id != view.specification_id
        for view, specification in zip(
            (source_views[name] for name in context.part_names),
            context.source_specifications,
            strict=True,
        )
    ):
        raise ValueError("FV archive view/specification binding changed.")
    source_parts = tuple(
        MeshPart(
            name,
            source_views[name],
            coordinate_contract=source_views[name].coordinate_contract,
        )
        for name in context.part_names
    )
    source_manifests = tuple(MeshPartManifest.from_part(part) for part in source_parts)
    local_rank = execution_group.spec.process_indices.index(jax.process_index())
    source_assembly = MeshAssembly(
        source_parts,
        global_parts=source_manifests,
        local_rank=local_rank,
    )
    source_overset_specs = tuple(
        _finite_volume_archive_overset_part_spec(value)
        for value in context.source_overset_specs
    )
    source_policy = _finite_volume_archive_overset_policy(context.source_overset_policy)
    if tuple(spec.part_name for spec in source_overset_specs) != (
        source_assembly.global_part_names
    ):
        raise ValueError("FV archive source overset specs differ from part order.")
    source_receptor_spec = next(
        spec for spec in source_overset_specs if spec.part_name == receptor_name
    )
    source_candidate = prepare_overset_connectivity(
        source_assembly,
        source_overset_specs,
        policy=source_policy,
        execution_group=execution_group,
    )
    donor_view = source_views[donor_name]
    receptor_view = source_views[receptor_name]
    system = EulerSystem(
        context.spatial_dimension,
        material=IdealGasMaterial(
            context.material_gamma,
            context.material_gas_constant,
            density_floor=context.material_density_floor,
            pressure_floor=context.material_pressure_floor,
        ),
    )

    def require_neighbor_carriers(
        view: OwnerLocalCellGeometry,
        part_name: str,
        /,
    ) -> None:
        from jax.experimental import multihost_utils

        cell_ids = np.asarray(view.cell_ids, dtype=np.int64)
        owned = np.asarray(view.owned, dtype=np.bool_)
        graph_ids = np.asarray(view.owned_cell_ids, dtype=np.int64)
        offsets = np.asarray(view.cell_neighbor_offsets, dtype=np.int64)
        neighbor_ids = np.asarray(view.cell_neighbor_ids, dtype=np.int64)
        neighbor_owner_ranks = np.asarray(
            view.cell_neighbor_owner_ranks,
            dtype=np.int32,
        )
        if (
            owned.shape != cell_ids.shape
            or offsets.shape != (graph_ids.size + 1,)
            or offsets[-1] != neighbor_ids.size
            or neighbor_owner_ranks.shape != neighbor_ids.shape
            or set(graph_ids.tolist()) != set(cell_ids[owned].tolist())
        ):
            raise ValueError("FV archive source neighbor metadata is malformed.")
        local_cells = set(int(identifier) for identifier in cell_ids.tolist())
        missing_records: list[tuple[int, int, int]] = []
        for row, owned_cell_id in enumerate(graph_ids.tolist()):
            for neighbor_row in range(int(offsets[row]), int(offsets[row + 1])):
                neighbor_id = int(neighbor_ids[neighbor_row])
                if neighbor_id not in local_cells:
                    missing_records.append(
                        (
                            int(owned_cell_id),
                            neighbor_id,
                            int(neighbor_owner_ranks[neighbor_row]),
                        )
                    )
        process_indices = execution_group.spec.process_indices
        rank_counts = np.asarray(
            multihost_utils.process_allgather(
                np.asarray((len(missing_records),), dtype=np.int32)
            ),
            dtype=np.int64,
        ).reshape((-1,))
        if not np.any(rank_counts):
            return
        record_width = max(1, int(np.max(rank_counts)))
        padded_records = np.full((record_width, 3), -1, dtype=np.int64)
        if missing_records:
            padded_records[: len(missing_records)] = np.asarray(
                missing_records,
                dtype=np.int64,
            )
        gathered_records = np.asarray(
            multihost_utils.process_allgather(padded_records, tiled=False),
            dtype=np.int64,
        ).reshape((len(process_indices), record_width, 3))
        cell_counts = np.asarray(
            multihost_utils.process_allgather(
                np.asarray((cell_ids.size,), dtype=np.int32)
            ),
            dtype=np.int64,
        ).reshape((-1,))
        cell_width = max(1, int(np.max(cell_counts)))
        padded_cell_ids = np.full((cell_width,), -1, dtype=np.int64)
        padded_cell_ids[: cell_ids.size] = cell_ids
        gathered_cell_ids = np.asarray(
            multihost_utils.process_allgather(padded_cell_ids, tiled=False),
            dtype=np.int64,
        ).reshape((len(process_indices), cell_width))
        missing_summary: list[tuple[int, int, int, int, tuple[int, ...]]] = []
        for owner_position, count in enumerate(rank_counts):
            process_index = process_indices[owner_position]
            for record in gathered_records[owner_position, : int(count)]:
                owned_cell_id, neighbor_id, expected_owner = (
                    int(value) for value in record
                )
                present_ranks = tuple(
                    process_indices[position]
                    for position, local_count in enumerate(cell_counts)
                    if neighbor_id in gathered_cell_ids[position, : int(local_count)]
                )
                missing_summary.append(
                    (
                        process_index,
                        owned_cell_id,
                        neighbor_id,
                        expected_owner,
                        present_ranks,
                    )
                )
        raise ValueError(
            f"FV archive source {part_name!r} lacks owned-cell neighbor carriers "
            f"(process, owned cell, missing neighbor, expected owner, local carrier "
            f"processes): {missing_summary!r}"
        )

    require_neighbor_carriers(donor_view, donor_name)
    require_neighbor_carriers(receptor_view, receptor_name)
    donor_blocks = tuple(
        block for block in donor_view.blocks if np.asarray(block.cell_ids).size
    )
    if len(donor_blocks) > 1:
        raise ValueError("FV archive source requires one donor block per owner.")
    local_reconstructions: dict[str, Any] = {}
    coefficient_ids: dict[str, np.ndarray] = {}
    if donor_blocks:
        donor_block = donor_blocks[0]
        local_reconstructions[donor_name] = (
            phx.discretization.prepare_owner_local_finite_volume_field_reconstruction(
                donor_view,
                donor_block.name,
                context.field_name,
            )
        )
        coefficient_ids[donor_name] = np.asarray(donor_block.cell_ids, dtype=np.int64)
    source_transfer = prepare_overset_field_transfer(
        source_candidate, local_reconstructions, context.field_name
    )
    source_connectivity = source_transfer.prepare_owner_local_exchange(
        coefficient_ids,
        execution_group,
        source_views,
    ).connectivity
    if not source_connectivity.owner_local or not source_connectivity.complete:
        raise ValueError("FV archive source connectivity did not replay completely.")
    source_coupling = prepare_owner_local_finite_volume_overset(
        source_connectivity,
        {donor_name: donor_view, receptor_name: receptor_view},
        donor_name,
        receptor_name,
        execution_group,
        component_names=system.component_names,
        field_name=context.field_name,
        axis_name=f"{context.owner_local_axis_name}_source_overset",
    )
    receptor_geometry = source_coupling.geometry
    source_overset = source_coupling.overset

    precision = _rebuild_finite_volume_precision_policy(context.precision_policy)
    if validate_expected and (
        precision.policy_id != context.expected_precision_policy_id
        or system.system_id != context.expected_system_id
    ):
        raise ValueError("FV archive system or precision identity changed.")

    def build_dynamics(
        geometry: Any,
        overset: Any,
        expected_patches: tuple[str, ...],
        expected_boundary_set_id: str,
        expected_dynamics_id: str,
        /,
    ) -> tuple[Any, Any, Any]:
        patch_names = tuple(geometry.boundary_patch_names)
        if validate_expected and patch_names != expected_patches:
            raise ValueError("FV geometry boundary patch list changed.")
        boundaries = UnstructuredFiniteVolumeBoundarySet(
            patch_names,
            {name: phx.discretization.ExtrapolationBoundary() for name in patch_names},
        )
        if validate_expected and boundaries.boundary_set_id != expected_boundary_set_id:
            raise ValueError("FV boundary policy identity changed.")
        method = UnstructuredFiniteVolumeMethodPlan(
            phx.discretization.PiecewiseConstantReconstruction(),
            phx.discretization.RusanovFluxPlan(
                smooth_epsilon=context.flux_smooth_epsilon
            ),
        )
        if validate_expected and method.method_id != context.expected_method_id:
            raise ValueError("FV numerical method identity changed.")
        problem = ConservationProblemIR(
            context.problem_name,
            context.field_name,
            system,
            boundaries,
            source=source,
            source_id=context.source_id,
        )
        coupling = UnstructuredFiniteVolumeCouplingPlan(
            overset=overset,
            topology_event_capacity=context.topology_event_capacity,
            topology_event_policy=context.topology_event_policy,
        )
        dynamics = compile_conservation_problem(
            problem,
            geometry,
            method,
            precision=precision,
            coupling=coupling,
        ).dynamics
        if validate_expected and dynamics.dynamics_id != expected_dynamics_id:
            raise ValueError("FV compiled dynamics identity changed.")
        return dynamics, boundaries, method

    source_dynamics, source_boundaries, method = build_dynamics(
        receptor_geometry,
        source_overset,
        context.source_boundary_patch_names,
        context.expected_source_boundary_set_id,
        context.expected_source_dynamics_id,
    )
    positivity = phx.discretization.FluxPositivityPlan(
        context.positivity_iterations,
        fallback_flux=phx.discretization.EinfeldtHLLFluxPlan(),
    )
    step_policy = FiniteVolumeStepPolicy(
        cfl=context.step_cfl,
        maximum_retries=context.maximum_retries,
        reduction_factor=context.step_reduction_factor,
        minimum_step_size=context.step_minimum_step_size,
    )
    if validate_expected and (
        positivity.plan_id != context.expected_positivity_id
        or step_policy.policy_id != context.expected_step_policy_id
    ):
        raise ValueError("FV time-step or positivity policy identity changed.")
    source_runtime = PreparedFiniteVolumeRuntime(
        source_dynamics,
        positivity,
        step_policy,
        execution_group=execution_group,
        owner_local_geometry=source_coupling.composite,
        owner_local_axis_name=context.owner_local_axis_name,
    )
    if validate_expected:
        archived_source_epoch = TopologyEpoch.from_archive_record(
            context.source_topology_epoch_record
        )
        archived_source_artifacts = FiniteVolumeTopologyArtifacts.from_archive_record(
            context.source_topology_artifacts_record,
            archived_source_epoch,
        )
        if (
            source_runtime.runtime_id != context.expected_source_runtime_id
            or source_runtime.topology_epoch_id != context.expected_source_epoch_id
            or source_runtime.initial_topology_epoch.epoch_id
            != archived_source_epoch.epoch_id
            or source_runtime.initial_topology_artifacts.artifacts_id
            != context.expected_source_artifacts_id
            or source_runtime.initial_topology_artifacts.artifacts_id
            != archived_source_artifacts.artifacts_id
            or source_connectivity.connectivity_id
            != context.expected_source_connectivity_id
        ):
            raise ValueError("FV source runtime/topology authority changed.")

    target_plan = GraphPartitionPlan.from_stored_fields(
        context.target_graph_partition_plan
    )
    target_specification = phx.meshing.SurfaceMeshingSpec.from_stored_fields(
        context.target_specification
    )
    receptor_part = source_assembly.part(receptor_name)
    receptor_manifest = next(
        item for item in source_assembly.part_manifest if item.name == receptor_name
    )
    receptor_assembly = MeshAssembly(
        (receptor_part,),
        global_parts=(receptor_manifest,),
        local_rank=local_rank,
    )
    migration = partition_owner_local_graph(
        receptor_assembly,
        target_plan,
        execution_group,
        axis_name=f"{context.owner_local_axis_name}_target_partition",
    )
    target_assembly = _repartition_owner_local_periodic_assembly(
        receptor_assembly,
        migration,
        execution_group,
        current_specification=target_specification,
        graph_partition_plan=target_plan,
        axis_name=f"{context.owner_local_axis_name}_target_view",
    )
    target_part = target_assembly.part(receptor_name)
    replayed_target_view = target_part.carrier
    if not isinstance(replayed_target_view, OwnerLocalCellGeometry):
        raise TypeError("Graph repartition did not rebuild its owner-local target view.")
    if validate_expected and len(context.target_overset_specs) != 1:
        raise ValueError("FV archive must retain its exact target overset declaration.")
    if context.target_overset_specs:
        stored_target_spec = _finite_volume_archive_overset_part_spec(
            context.target_overset_specs[0]
        )
        derived_target_spec = _finite_volume_archive_target_receptor_spec(
            source_receptor_spec, receptor_part, target_part, execution_group
        )
        target_overset_spec = _finite_volume_archive_overset_part_spec(
            derived_target_spec
        )
        if stored_target_spec.spec_id != target_overset_spec.spec_id:
            raise ValueError("Stored target overset spec differs from owner replay.")
    else:
        target_overset_spec = _finite_volume_archive_target_receptor_spec(
            source_receptor_spec, receptor_part, target_part, execution_group
        )
    target_candidate = source_connectivity.reregister(
        {receptor_name: target_part},
        successor_specs={receptor_name: target_overset_spec},
        execution_group=execution_group,
        axis_name=f"{context.owner_local_axis_name}_target_reregister",
    )
    target_transfer = prepare_overset_field_transfer(
        target_candidate, local_reconstructions, context.field_name
    )
    target_views: dict[str, OwnerLocalCellGeometry] = {}
    for part in target_candidate.assembly.parts:
        if not isinstance(part.carrier, OwnerLocalCellGeometry):
            raise TypeError("Target overset search requires owner-local geometry.")
        target_views[part.name] = part.carrier
    target_connectivity = target_transfer.prepare_owner_local_exchange(
        coefficient_ids,
        execution_group,
        target_views,
    ).connectivity
    target_view = target_connectivity.assembly.part(receptor_name).carrier
    if not isinstance(target_view, OwnerLocalCellGeometry):
        raise TypeError("Candidate connectivity lost its owner-local target view.")
    if validate_expected and (
        target_view.distribution_id != context.expected_target_distribution_id
        or context.target_view.distribution_id != replayed_target_view.distribution_id
        or context.target_view.local_view_id != replayed_target_view.local_view_id
        or not np.array_equal(
            np.asarray(context.target_view.cell_ids, dtype=np.int64),
            np.asarray(replayed_target_view.cell_ids, dtype=np.int64),
        )
        or not np.array_equal(
            np.asarray(context.target_view.owned, dtype=np.bool_),
            np.asarray(replayed_target_view.owned, dtype=np.bool_),
        )
        or not np.array_equal(
            np.asarray(context.target_view.vertex_ids, dtype=np.int64),
            np.asarray(replayed_target_view.vertex_ids, dtype=np.int64),
        )
        or target_plan.plan_id != context.expected_target_plan_id
        or target_connectivity.connectivity_id != context.expected_target_connectivity_id
    ):
        raise ValueError("FV target view, plan, or connectivity identity changed.")
    target_coupling = prepare_owner_local_finite_volume_overset(
        target_connectivity,
        {donor_name: donor_view, receptor_name: target_view},
        donor_name,
        receptor_name,
        execution_group,
        component_names=system.component_names,
        field_name=context.field_name,
        axis_name=f"{context.owner_local_axis_name}_target_overset",
    )
    target_geometry = target_coupling.geometry
    target_overset = target_coupling.overset
    target_dynamics, target_boundaries, _ = build_dynamics(
        target_geometry,
        target_overset,
        context.target_boundary_patch_names,
        context.expected_target_boundary_set_id,
        context.expected_target_dynamics_id,
    )
    source_epoch = source_runtime.initial_topology_epoch
    graph_migration = target_view.graph_migration_evidence
    if graph_migration is None:
        raise ValueError("Replayed target view lost its graph migration evidence.")
    target_epoch = TopologyEpoch(
        source_epoch.index + 1,
        source_epoch.geometry_id,
        source_epoch.topology_id,
        target_coupling.composite.distribution_id,
    )
    target_artifacts = FiniteVolumeTopologyArtifacts(
        target_epoch,
        target_geometry.prepared_id,
        topology_artifact_id=target_connectivity.connectivity_id,
        metrics_artifact_id=graph_migration.evidence_id,
        operators_artifact_id=target_dynamics.dynamics_id,
    )
    if validate_expected:
        archived_target_epoch = TopologyEpoch.from_archive_record(
            context.target_topology_epoch_record
        )
        archived_target_artifacts = FiniteVolumeTopologyArtifacts.from_archive_record(
            context.target_topology_artifacts_record,
            archived_target_epoch,
        )
        if (
            target_epoch.epoch_id != context.expected_target_epoch_id
            or target_artifacts.artifacts_id != context.expected_target_artifacts_id
            or target_epoch.epoch_id != archived_target_epoch.epoch_id
            or target_artifacts.artifacts_id != archived_target_artifacts.artifacts_id
        ):
            raise ValueError("Replayed FV target epoch/artifact authority changed.")
    rebuilt_context = context._replace(
        source_views=tuple(source_views[name] for name in context.part_names),
        source_overset_specs=source_overset_specs,
        source_overset_policy=source_policy,
        target_overset_specs=(target_overset_spec,),
        target_graph_partition_plan=target_plan,
        target_specification=target_specification,
        target_view=target_view,
        source_boundary_patch_names=tuple(source_boundaries.patch_names),
        target_boundary_patch_names=tuple(target_boundaries.patch_names),
        expected_target_plan_id=target_plan.plan_id,
        expected_source_dynamics_id=source_dynamics.dynamics_id,
        expected_source_boundary_set_id=source_boundaries.boundary_set_id,
        expected_target_boundary_set_id=target_boundaries.boundary_set_id,
        expected_source_runtime_id=source_runtime.runtime_id,
        expected_source_epoch_id=source_runtime.topology_epoch_id,
        expected_source_artifacts_id=source_runtime.initial_topology_artifacts.artifacts_id,
        source_topology_epoch_record=source_runtime.initial_topology_epoch.to_archive_record(),
        source_topology_artifacts_record=source_runtime.initial_topology_artifacts.to_archive_record(),
        target_topology_epoch_record=target_epoch.to_archive_record(),
        target_topology_artifacts_record=target_artifacts.to_archive_record(),
        expected_source_connectivity_id=source_connectivity.connectivity_id,
        expected_target_distribution_id=target_view.distribution_id,
        expected_target_connectivity_id=target_connectivity.connectivity_id,
        expected_target_dynamics_id=target_dynamics.dynamics_id,
        expected_system_id=system.system_id,
        expected_method_id=method.method_id,
        expected_positivity_id=positivity.plan_id,
        expected_precision_policy_id=precision.policy_id,
        expected_step_policy_id=step_policy.policy_id,
        expected_target_epoch_id=target_epoch.epoch_id,
        expected_target_artifacts_id=target_artifacts.artifacts_id,
    )
    return {
        "context": rebuilt_context,
        "group": execution_group,
        "source_assembly": source_assembly,
        "source_views": source_views,
        "source_connectivity": source_connectivity,
        "source_runtime": source_runtime,
        "source_geometry": receptor_geometry,
        "source_dynamics": source_dynamics,
        "target_view": target_view,
        "target_connectivity": target_connectivity,
        "target_dynamics": target_dynamics,
        "system": system,
        "source_cell_ids": np.asarray(
            source_runtime.owner_local_cell_ids(), dtype=np.int64
        ),
        "receptor_name": receptor_name,
        "source_epoch": source_runtime.initial_topology_epoch,
        "source_artifacts": source_runtime.initial_topology_artifacts,
        "target_epoch": target_epoch,
        "target_artifacts": target_artifacts,
    }
