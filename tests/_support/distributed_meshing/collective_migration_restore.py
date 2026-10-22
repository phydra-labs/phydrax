"""Fresh-process replay of topology-neutral collective migration snapshots."""

from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._array_archive import ArrayArchiveLimits
from phydrax._execution_array import _place_process_local_array
from phydrax._fingerprint import tree_content_equal
from phydrax._meshcore import NativeExecutionLimits, NativeExecutionScope
from phydrax._model._structure import restore_registered_fields
from phydrax.discretization._adaptive_simplex import (
    adaptive_simplex_state,
    AdaptiveSimplexLayout,
    refine_adaptive_simplex_parts,
)
from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.graph._distributed_partition import replay_logical_partition
from phydrax.graph._partition import GraphPartitionPlan
from phydrax.lifecycle._meshing_sources import (
    read_meshing_source_closure,
    reconstruct_meshing_source,
    write_meshing_source_closure,
)
from phydrax.lifecycle._restart_topology import TopologyRestartPolicy
from phydrax.meshing import MeshingLimits
from phydrax.meshing._device_adaptation import (
    commit_partitioned_adaptive_simplex,
    execute_partitioned_mesh_adaptation,
    validate_partitioned_mesh_evidence,
)
from phydrax.meshing._distribution import SimplexNeighborhoodWorkset
from phydrax.meshing._distribution_migration import (
    _migration_placed_pairs_equal,
    _replay_logical_solver_owners,
    migrate_partitioned_epoch,
    place_logical_simplex_graph_weights,
    prepare_ownerlocal_migration_state,
    replay_graph_restart_proposal,
    replay_logical_simplex_graph,
    SimplexGraphRestartProposal,
)
from phydrax.meshing._failures import MeshingFailure
from phydrax.meshing._migration_record import CollectiveMigrationRecord
from phydrax.meshing._migration_state_record import (
    MigrationTransportEvidence,
    MigrationWorkState,
)
from phydrax.meshing._result import CollectiveMeshEvidence
from tests._support.distributed_meshing.device_adaptation import _public_preparation
from tests._support.distributed_meshing.native_distributed_proof import (
    _dense_predecessor_oracle,
)
from tests._support.distributed_meshing.native_distributed_records import (
    _require_instance,
    _require_present,
)


def mixed_migration_comparison(permuted: str, empty: str) -> dict[str, Any]:
    """Keep resident records local while comparing the common distributed bank."""
    rank = jax.process_index()
    devices = tuple(jax.devices())
    if permuted == "true":
        devices = tuple(reversed(devices))
    mesh = jax.sharding.Mesh(np.asarray(devices), ("owner",))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("owner"))
    shared = jax.make_array_from_callback(
        (2, 3), sharding,
        lambda index: np.arange(6, dtype=np.float64).reshape(2, 3)[index],
    )
    replicated = jax.make_array_from_callback(
        (3,), jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec()),
        lambda index: np.arange(3, dtype=np.int64)[index],
    )
    local = jax.device_put(
        np.empty((0, 3), dtype=np.float64)
        if empty == "true" and rank == 1
        else np.full((2, 3), rank + 1.0, dtype=np.float64),
        jax.local_devices()[0],
    )
    left: tuple[jax.Array, ...] = (local, shared, replicated)
    assert _migration_placed_pairs_equal(left, left)
    refused = []
    for fault in ("local-value", "missing-local", "missing-global", "foreign-owner"):
        right = left
        before = left
        if rank == 0:
            if fault == "local-value":
                right = (local + 1.0, shared, replicated)
            elif fault == "missing-local":
                right = (shared, replicated)
            elif fault == "missing-global":
                before, right = (local, replicated), (local, replicated)
            else:
                right = (shared, shared, replicated)
        try:
            same = _migration_placed_pairs_equal(before, right)
        except (ValueError, MeshingFailure):
            same = False
        assert not same
        refused.append(fault)
        assert _migration_placed_pairs_equal(left, left)
    assert local.sharding.device_set == {jax.local_devices()[0]}
    return {"rank": rank, "refused": refused, "retry": True, "unchanged": True}


def replay_graph_weight_placement(permuted: str, empty: str) -> dict[str, Any]:
    """Exercise dense logical rows on current, optionally permuted, devices."""
    devices = jax.devices()
    if permuted == "true":
        devices = list(reversed(devices))
    mesh = jax.sharding.Mesh(np.asarray(devices), ("owner",))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("owner"))
    ids = jnp.arange(4, dtype=jnp.int64).reshape(4, 1) + 2**54
    vertices = jnp.arange(12, dtype=jnp.int64).reshape(4, 1, 3) + 2**54
    work = SimplexNeighborhoodWorkset(
        ids,
        vertices,
        jnp.zeros((4, 1, 3, 2), jnp.float64),
        jnp.arange(4, dtype=jnp.int32).reshape(4, 1),
        jnp.zeros((4, 1), jnp.int32),
        jnp.ones((4, 1, 3), jnp.bool_),
        jnp.full((4, 1), empty != "true", jnp.bool_),
        ids,
        ids[..., None],
        jnp.broadcast_to(vertices[..., None], (4, 1, 3, 3)),
        jnp.broadcast_to(jnp.eye(3), (4, 1, 3, 3)),
        jnp.zeros(4, jnp.int32),
    )
    vertex_weights = jnp.ones(ids.shape, jnp.int64)
    edge_weights = jnp.ones(vertices.shape, jnp.int64)
    expected = replay_logical_simplex_graph(work, edge_weights, axis_name="owner")
    placed_work = jax.tree.map(
        lambda value: _place_process_local_array(value, sharding), work
    )
    for invalid in (
        edge_weights[:2],
        edge_weights.astype(jnp.int32),
    ):
        with pytest.raises(MeshingFailure):
            place_logical_simplex_graph_weights(
                placed_work,
                vertex_weights,
                invalid if jax.process_index() == 0 else edge_weights,
            )
    if jax.process_count() > 1:
        foreign = _place_process_local_array(
            edge_weights,
            jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec()),
        )
        with pytest.raises(MeshingFailure):
            place_logical_simplex_graph_weights(
                placed_work,
                vertex_weights,
                foreign if jax.process_index() == 0 else edge_weights,
            )
    # Admission failures must not poison a subsequent retry.
    with pytest.MonkeyPatch.context() as patch:

        def forbid_fetch(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("graph placement must not fetch source rows")

        patch.setattr(jax, "device_get", forbid_fetch)
        placed_vertex, placed_edge = place_logical_simplex_graph_weights(
            placed_work, vertex_weights, edge_weights
        )
        actual = replay_logical_simplex_graph(
            placed_work, placed_edge, axis_name="owner"
        )
    for reference, placed in (
        (vertex_weights, placed_vertex),
        (edge_weights, placed_edge),
    ):
        assert placed.sharding.is_equivalent_to(sharding, ndim=placed.ndim)
        for shard in placed.addressable_shards:
            np.testing.assert_array_equal(
                np.asarray(reference[shard.index]), np.asarray(shard.data)
            )
    for reference, placed in zip(expected, actual, strict=True):
        for shard in placed.addressable_shards:
            np.testing.assert_array_equal(
                np.asarray(reference[shard.index]), np.asarray(shard.data)
            )
    for reference, placed in zip(
        jax.tree.leaves(work), jax.tree.leaves(placed_work), strict=True
    ):
        for shard in placed.addressable_shards:
            np.testing.assert_array_equal(
                np.asarray(reference[shard.index]), np.asarray(shard.data)
            )
    # Exercise the actual receipt-restore entry point, not just graph lowering.
    # Archived weights remain local while the restored workset/raw state are global.
    layout = AdaptiveSimplexLayout(
        2, 2, vertex_capacity=3, cell_capacity=1, protected_edge_capacity=1,
        maximum_closure_iterations=4, maximum_coarsening_passes=4,
    )
    states = []
    for rank in range(4):
        states.append(
            adaptive_simplex_state(
                layout,
                coordinates=np.asarray([[0., 0.], [1., 0.], [0., 1.]]),
                vertex_ids=np.asarray(vertices[rank, 0]),
                vertex_active=np.ones(3, np.bool_),
                vertex_parents=np.full((3, 2), -1, np.int32),
                vertex_protected=np.zeros(3, np.bool_),
                cells=np.asarray([[0, 1, 2]], np.int32),
                tuples=np.asarray([[0, 1, 2]], np.int32),
                tags=np.zeros(1, np.int32),
                blocks=np.zeros(1, np.int32),
                generations=np.zeros(1, np.int32),
                parents=np.full(1, -1, np.int32),
                children=np.full((1, 2), -1, np.int32),
                bisection_vertices=np.full(1, -1, np.int32),
                cell_ids=np.asarray(ids[rank]),
                cell_active=np.asarray(work.cell_valid[rank]),
                cell_classes=np.zeros(1, np.int32),
                facet_classes=np.zeros((1, 3), np.int32),
                protected_edges=np.empty((0, 2), np.int32),
                next_vertex_id=int(vertices[rank, 0, -1]) + 1,
                next_cell_id=int(ids[rank, 0]) + 1,
            )
        )
    saved = jax.tree.map(lambda *rows: jnp.stack(rows), *states)
    offsets, neighbors, weights, valid = expected
    parts, _, measured = replay_logical_partition(
        offsets, neighbors, weights, vertex_weights, ids, work.cell_owner, valid,
        GraphPartitionPlan(4), axis_name="owner",
    )
    assert (int(measured.resource_status) != 0) == (empty == "true")
    requested, status = _replay_logical_solver_owners(
        work, parts, saved.mesh.cell_ids, saved.mesh.cell_active, 4, "owner"
    )
    assert not np.any(np.asarray(status))
    proposal = SimplexGraphRestartProposal(
        placed_work, vertex_weights, edge_weights,
        _place_process_local_array(parts, sharding),
        _place_process_local_array(requested, sharding),
        GraphPartitionPlan(4).maximum_imbalance,
    )
    placed_saved = jax.tree.map(
        lambda value: _place_process_local_array(value, sharding), saved
    )
    if empty == "true":
        with pytest.raises(ValueError, match="must match an accepted replayed native partition"):
            replay_graph_restart_proposal(
                proposal, placed_saved, axis_name="owner", collective=True
            )
    else:
        replay_graph_restart_proposal(
            proposal, placed_saved, axis_name="owner", collective=True
        )
    return {"graph_bytes_unchanged": True, "retry": True}


def _assert_device_free(value: Any) -> None:
    visited: set[int] = set()
    pending = [value]
    while pending:
        node = pending.pop()
        if id(node) in visited:
            continue
        visited.add(id(node))
        if isinstance(node, (jax.Device, jax.sharding.Mesh)):
            raise AssertionError(
                "MigrationCheckpoint.payload must exclude device handles; got live execution object"
            )
        if isinstance(node, (jax.Array, np.ndarray)):
            continue
        if is_dataclass(node):
            pending.extend(getattr(node, field.name) for field in fields(node))
        elif isinstance(node, Mapping):
            pending.extend(node.values())
        elif isinstance(node, (tuple, list)):
            pending.extend(node)


def save_migration(path: str) -> dict[str, Any]:
    _, _, _, _, partitioned, first = _public_preparation(False)
    marks = np.asarray(first.state.mesh.cell_active).copy()
    marks[1:] = False
    second = refine_adaptive_simplex_parts(
        partitioned.layout, type(partitioned).execution_parts(partitioned), first.state, marks
    )
    target = (
        commit_partitioned_adaptive_simplex(partitioned, second.state)
        .local_results[0]
        .target
    )
    evidence = _require_instance(target.collective_evidence, CollectiveMeshEvidence)
    migration = _require_present(evidence.migration_record)
    _assert_device_free(migration)
    restored_migration = reconstruct_meshing_source(migration)
    if not tree_content_equal(restored_migration, migration):
        raise AssertionError(
            "MigrationCheckpoint.record must restore through validated standalone fields; got changed record"
        )
    altered = eqx.tree_at(
        lambda value: value.requested_target_owners,
        migration,
        migration.requested_target_owners + 1,
    )
    with pytest.raises(
        ValueError, match=r"CollectiveMigrationRecord\.migration_id must bind"
    ):
        reconstruct_meshing_source(altered)
    receipt = write_meshing_source_closure(
        Path(path), target, limits=ArrayArchiveLimits()
    )
    historical = _save_historical_migration(
        partitioned,
        second.state,
        migration.requested_target_owners,
        f"{path}.historical",
    )
    return {
        "content_id": receipt.content_id,
        "result_id": target.result_id,
        "evidence_id": evidence.evidence_id,
        "migration_id": migration.migration_id,
        "device_free": True,
        "standalone_validated": True,
        "tampered_record_refused": True,
        **historical,
    }


def replay_migration(path: str, content_id: str) -> dict[str, Any]:
    with NativeExecutionScope(NativeExecutionLimits(1, 0, 0, 1 << 20, 120.0)) as scope:
        with scope.batch(1) as ticket:
            ticket.consume(1)
            with ticket.native_target():
                target = read_meshing_source_closure(
                    Path(path),
                    expected_content_id=content_id,
                    limits=ArrayArchiveLimits(),
                )
            consumed_work = ticket.consumed_work_units
    restore_usage = _require_present(scope.evidence)
    evidence = _require_present(target.collective_evidence)
    migration = _require_present(evidence.migration_record)
    _assert_device_free(migration)
    validate_partitioned_mesh_evidence(_require_present(target.mesh.storage), evidence)
    return {
        "result_id": target.result_id,
        "evidence_id": evidence.evidence_id,
        "migration_id": migration.migration_id,
        "device_free": True,
        "replayed": True,
        "consumed_work": consumed_work,
        "restore_native_work": restore_usage.work_counters.construction_work
        - restore_usage.externally_charged_work,
        "restore_native_queries": restore_usage.native_primitive_queries,
        "restore_external_work": restore_usage.externally_charged_work,
    }


def refuse_topology_change(path: str, content_id: str) -> dict[str, Any]:
    migration = _require_instance(
        read_meshing_source_closure(
            Path(path), expected_content_id=content_id, limits=ArrayArchiveLimits()
        ),
        CollectiveMigrationRecord,
    )
    try:
        migration.placement.rebind()
    except ValueError as failure:
        if "MigrationPlacement.topology" not in str(failure):
            raise
        return {"refused": True, "error": str(failure)}
    raise AssertionError(
        "MigrationCheckpoint.topology must refuse an unadmitted device-count change; got admission"
    )


def _save_historical_migration(
    partitioned: Any, states: Any, owners: jax.Array, path: str
) -> dict[str, Any]:
    """Issue a real canonical receipt with nonempty scientific packet histories."""
    mesh = states.mesh
    cell_ids = jnp.where(mesh.cell_active, mesh.cell_ids, -1)
    vertex_ids = jnp.where(mesh.vertex_active, mesh.vertex_ids, -1)
    keys = jax.vmap(
        lambda value: jax.random.fold_in(jax.random.key(20261020), value)
    )(mesh.cell_ids.reshape(-1).astype(jnp.uint32))
    # The distributed field-role/archive contract stores unsigned key words,
    # not JAX's extended typed-key dtype.
    keys = jax.random.key_data(keys).reshape((*mesh.cell_ids.shape, 2))
    keys = jax.device_put(keys, mesh.cell_ids.sharding)
    counters = jnp.where(
        mesh.cell_ids >= 0, 2**54 + mesh.cell_ids, 0
    ).astype(jnp.uint64)
    vertex_history = jnp.where(
        mesh.vertex_ids >= 0, 2**54 + mesh.vertex_ids, -1
    ).astype(jnp.int64)
    bound = prepare_ownerlocal_migration_state(
        partitioned,
        states,
        cell_field_ids=cell_ids,
        cell_fields=jnp.stack(
            (cell_ids.astype(jnp.float64) / 16, -cell_ids.astype(jnp.float64) / 32),
            axis=-1,
        ),
        vertex_field_ids=vertex_ids,
        vertex_fields=jnp.stack(
            (vertex_ids.astype(jnp.float64) / 8, -vertex_ids.astype(jnp.float64) / 16),
            axis=-1,
        ),
        cell_epochs=jnp.where(mesh.cell_active, 41 + mesh.cell_ids % 5, 0).astype(
            jnp.int64
        ),
        cell_history=(
            keys, counters,
            jnp.where(mesh.cell_active, mesh.cell_ids.astype(jnp.float64) / 64, 0),
            mesh.cell_active & ((mesh.cell_ids % 2) == 0),
        ),
        vertex_history=(vertex_history,),
        # Scalar and boolean histories accompany the vector cell/vertex fields
        # and exact unsigned RNG key/counter words through the same real route.
    )
    issued = migrate_partitioned_epoch(partitioned, states, bound, owners)
    record = CollectiveMigrationRecord(issued)
    assert record.part_count == 4
    assert bool(jnp.all(record.transport_evidence.accepted))
    assert int(jnp.sum(record.transport_evidence.sent_cells)) > 0
    assert record.saved_migration_state.cell_history[0].dtype == keys.dtype
    assert record.saved_migration_state.cell_history[1].dtype == jnp.uint64
    assert record.saved_migration_state.workset.source_cell_ids.shape[-1] == 1
    _assert_original_predecessors(record)
    shape_only = eqx.tree_at(
        lambda value: value.saved_migration_state.workset.source_cell_ids,
        record,
        jnp.pad(
            record.saved_migration_state.workset.source_cell_ids,
            ((0, 0), (0, 0), (0, 1)),
            constant_values=-1,
        ),
    )
    assert shape_only._identity() != record.migration_id
    # Identity framing changes while every entity, coefficient, scalar/vector
    # field, history, raw clock and root-seeded RNG word stays byte-identical.
    _assert_tree_bytes(
        (
            shape_only.saved_states,
            shape_only.saved_migration_state.cell_fields,
            shape_only.saved_migration_state.vertex_fields,
            shape_only.saved_migration_state.cell_history,
            shape_only.saved_migration_state.vertex_history,
            shape_only.saved_migration_state.workset.cell_ids,
            shape_only.saved_migration_state.workset.source_vertex_ids,
            shape_only.saved_migration_state.workset.source_weights,
        ),
        (
            record.saved_states,
            record.saved_migration_state.cell_fields,
            record.saved_migration_state.vertex_fields,
            record.saved_migration_state.cell_history,
            record.saved_migration_state.vertex_history,
            record.saved_migration_state.workset.cell_ids,
            record.saved_migration_state.workset.source_vertex_ids,
            record.saved_migration_state.workset.source_weights,
        ),
    )
    _assert_device_free(record)
    receipt = write_meshing_source_closure(
        Path(path), record, limits=ArrayArchiveLimits()
    )
    packet_receipt = write_meshing_source_closure(
        Path(f"{path}.packets"),
        (
            MigrationWorkState.from_live(issued.migration.original),
            MigrationWorkState.from_live(issued.migration.candidate),
        ),
        limits=ArrayArchiveLimits(),
    )
    return {
        "historical_content_id": receipt.content_id,
        "historical_migration_id": record.migration_id,
        "historical_placement_id": record.placement.placement_id,
        "historical_packets_content_id": packet_receipt.content_id,
        "compact_source_width": 1,
        "source_shape_changes_migration_id": True,
        "shape_change_preserves_entity_physics_rng_bytes": True,
    }


def _replay_historical(record: CollectiveMigrationRecord) -> Any:
    work = 40960000
    memory = 81920000
    with NativeExecutionScope(
        NativeExecutionLimits(
            maximum_work_units=work,
            maximum_geometry_queries=200000,
            maximum_cavity_cells=MeshingLimits().maximum_cavity_cells,
            maximum_managed_bytes=memory,
            maximum_seconds=120,
        )
    ) as scope:
        with (
            scope.host_workspace(),
            scope.batch(work, bytes_upper=memory) as ticket,
            CoordinateEnclosureBudget(
                ticket, maximum_work_units=work, maximum_memory_bytes=memory
            ).activate() as budget,
        ):
            return record.replay(
                scope=scope,
                budget=budget,
                policy=TopologyRestartPolicy(allow_topology_change=True),
            )


def _assert_tree_bytes(actual: Any, expected: Any) -> None:
    assert jax.tree_util.tree_structure(actual) == jax.tree_util.tree_structure(expected)
    for left, right in zip(
        jax.tree_util.tree_leaves(actual),
        jax.tree_util.tree_leaves(expected),
        strict=True,
    ):
        assert left.shape == right.shape and left.dtype == right.dtype
        if jax.dtypes.issubdtype(left.dtype, jax.dtypes.prng_key):
            left, right = jax.random.key_data(left), jax.random.key_data(right)
        assert np.asarray(left).tobytes() == np.asarray(right).tobytes()



def _assert_original_predecessors(record: CollectiveMigrationRecord) -> None:
    """Check the complete raw-state predicate, not merely a self-consistent hash."""
    work = record.saved_migration_state.workset
    expected = {}
    for rank in range(record.part_count):
        current = jax.tree_util.tree_map(lambda value: value[rank], record.saved_states)
        initial = jax.tree_util.tree_map(
            lambda value: value[rank], record.source.states
        )
        support, dense = _dense_predecessor_oracle(current, initial)
        for slot in np.flatnonzero(np.asarray(current.mesh.cell_active)):
            expected[int(current.mesh.cell_ids[slot])] = np.sort(
                dense[slot, support[slot]], kind="stable"
            )
    actual_width = max(1, *(values.size for values in expected.values()))
    assert work.source_cell_ids.shape[-1] == actual_width
    for rank, slot in np.argwhere(np.asarray(work.cell_valid)):
        identifiers = expected[int(work.cell_ids[rank, slot])]
        np.testing.assert_array_equal(
            work.source_cell_ids[rank, slot, : identifiers.size], identifiers
        )
        np.testing.assert_array_equal(
            work.source_cell_ids[rank, slot, identifiers.size :], -1
        )

def replay_historical_migration(
    path: str, content_id: str, packets_content_id: str
) -> dict[str, Any]:
    record = _require_instance(
        read_meshing_source_closure(
            Path(path), expected_content_id=content_id, limits=ArrayArchiveLimits()
        ),
        CollectiveMigrationRecord,
    )
    original, candidate = read_meshing_source_closure(
        Path(f"{path}.packets"),
        expected_content_id=packets_content_id,
        limits=ArrayArchiveLimits(),
    )
    _assert_device_free(record)
    _assert_original_predecessors(record)
    assert record.saved_migration_state.workset.source_cell_ids.shape[-1] == 1
    replayed = _replay_historical(record)
    renewed = CollectiveMigrationRecord(replayed)
    _assert_tree_bytes(renewed, record)
    _assert_tree_bytes(
        MigrationWorkState.from_live(replayed.saved_migration_state),
        record.saved_migration_state,
    )
    _assert_tree_bytes(
        MigrationWorkState.from_live(replayed.migration.original), original
    )
    _assert_tree_bytes(
        MigrationWorkState.from_live(replayed.migration.candidate), candidate
    )
    _assert_tree_bytes(
        MigrationTransportEvidence.from_live(replayed.migration.evidence),
        record.transport_evidence,
    )
    assert replayed.parts.part_count == record.part_count == 4
    assert replayed.parts.mesh.size == jax.device_count()
    assert renewed.placement.placement_id == record.placement.placement_id
    assert renewed.migration_id == record.migration_id
    live_refusals = {}
    if record.part_count > jax.device_count():
        source = replayed.source_partitioned

        def unexpected_resource_work(*args: Any, **kwargs: Any) -> None:
            raise AssertionError(
                "Historical logical owners must be refused before live resource admission"
            )

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(
                "phydrax.meshing._device_adaptation._phase_allowance",
                unexpected_resource_work,
            )
            for name, publish in (
                ("execute", lambda: execute_partitioned_mesh_adaptation(source)),
                (
                    "commit",
                    lambda: commit_partitioned_adaptive_simplex(
                        source, replayed.saved_states
                    ),
                ),
            ):
                with pytest.raises(ValueError) as failure:
                    publish()
                error = str(failure.value)
                assert "logical" in error.lower()
                live_refusals[name] = error
    return {
        "physical_devices": jax.device_count(),
        "logical_owners": replayed.parts.part_count,
        "migration_id": renewed.migration_id,
        "placement_id": renewed.placement.placement_id,
        "bytewise_records": True,
        "bytewise_fields_history_rng_clock": True,
        "bytewise_transport": True,
        "compact_source_width": record.saved_migration_state.workset.source_cell_ids.shape[-1],
        "complete_source_multisets": True,
        "sent_cells": int(jnp.sum(renewed.transport_evidence.sent_cells)),
        "live_publication_refusals": live_refusals,
    }


def refuse_historical_mutations(path: str, content_id: str) -> dict[str, Any]:
    record = _require_instance(
        read_meshing_source_closure(
            Path(path), expected_content_id=content_id, limits=ArrayArchiveLimits()
        ),
        CollectiveMigrationRecord,
    )
    saved = record.saved_migration_state
    row, slot = np.argwhere(np.asarray(saved.workset.cell_valid))[0]
    forest = _require_present(saved.forest)
    history_row, history_slot = np.argwhere(np.asarray(forest.mesh.cell_active))[0]
    vertex_row, vertex_slot = np.argwhere(np.asarray(forest.mesh.vertex_active))[0]
    keys, counters, *_ = saved.cell_history
    changed_keys = keys.at[history_row, history_slot, 0].add(jnp.uint32(1))
    mutations = {
        "fields": eqx.tree_at(
            lambda value: value.saved_migration_state.cell_fields,
            record,
            saved.cell_fields.at[row, slot, 0].add(1),
        ),
        "owner": eqx.tree_at(
            lambda value: value.saved_migration_state.workset.cell_owner,
            record,
            saved.workset.cell_owner.at[row, slot].set(
                (saved.workset.cell_owner[row, slot] + 1) % record.part_count
            ),
        ),
        "cell_history": eqx.tree_at(
            lambda value: value.saved_migration_state.cell_history[1],
            record,
            counters.at[history_row, history_slot].add(jnp.uint64(1)),
        ),
        "vertex_history": eqx.tree_at(
            lambda value: value.saved_migration_state.vertex_history[0],
            record,
            saved.vertex_history[0].at[vertex_row, vertex_slot].add(1),
        ),
        "rng": eqx.tree_at(
            lambda value: value.saved_migration_state.cell_history[0],
            record,
            changed_keys,
        ),
        "raw_clock": eqx.tree_at(
            lambda value: value.saved_states.clocks,
            record,
            record.saved_states.clocks.at[0, 0].add(1),
        ),
        "transport": eqx.tree_at(
            lambda value: value.transport_evidence.sent_cells,
            record,
            record.transport_evidence.sent_cells.at[0].add(1),
        ),
        "transport_control": eqx.tree_at(
            lambda value: value.transport_evidence,
            record,
            replace(
                record.transport_evidence,
                halo_width=record.transport_evidence.halo_width + 1,
            ),
        ),
        "source_missing": eqx.tree_at(
            lambda value: value.saved_migration_state.workset.source_cell_ids,
            record,
            jnp.full_like(saved.workset.source_cell_ids, -1),
        ),
        "source_negative": eqx.tree_at(
            lambda value: value.saved_migration_state.workset.source_cell_ids,
            record,
            saved.workset.source_cell_ids.at[row, slot, 0].set(-7),
        ),
        "source_duplicate": eqx.tree_at(
            lambda value: value.saved_migration_state.workset.source_cell_ids,
            record,
            jnp.repeat(saved.workset.source_cell_ids, 2, axis=-1),
        ),
    }
    refusals = {}
    for name, changed in mutations.items():
        if name.startswith("source_"):
            # A recomputed receipt hash is not authority for predecessor
            # completeness: replay must derive it from actual initial/current.
            changed_fields = {
                field.name: getattr(changed, field.name) for field in fields(changed)
            }
            changed_fields["migration_id"] = changed._identity()
            changed = restore_registered_fields(CollectiveMigrationRecord, changed_fields)
        with pytest.raises(ValueError) as failure:
            _replay_historical(changed)
        error = str(failure.value)
        assert (
            "predecessor" in error.lower()
            if name.startswith("source_")
            else "CollectiveMigrationRecord." in error
        )
        refusals[name] = error
    return {"refusals": refusals, "physical_devices": jax.device_count()}
