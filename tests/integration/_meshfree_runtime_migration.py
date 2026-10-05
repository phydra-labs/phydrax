# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Dedicated four-device process for the ownership-migration restart scenario.

Run as ``python -m tests.integration._meshfree_runtime_migration ROOT`` with
``XLA_FLAGS=--xla_force_host_platform_device_count=4``; it prints one JSON
report consumed by ``test_meshfree_runtime_restart.py``. Forced CPU devices
prove the restart relation and partition parity only, not accelerator
performance.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from phydrax import execution, solver
from phydrax.discretization.meshfree import (
    DistributedMeshfreeOperator,
    LocalStencilPolicy,
    meshfree_runtime_inventory,
    MeshfreeFunctional,
    MeshfreeNeighborhoodPlan,
    MeshfreeOperator,
    ownership_migration_relation,
    prepare_local_stencils,
)
from phydrax.discretization.spatial import (
    DistributedOwnershipPlan,
    DistributedPointLayout,
    MortonAddressPlan,
)
from phydrax.lifecycle import (
    ArtifactManifest,
    HPCFilesystemProfile,
    POSIXArtifactRepository,
    POSIXRepositoryPolicy,
    ResolvedRunSpec,
)
from phydrax.qualification import SupportDependency
from phydrax.sparse import SparseCoordinateOperator


_KAPPA = 0.01
_STEP = 0.01
_STEPS = 20
_CAPACITY = 24


def _operator(points: np.ndarray, /) -> SparseCoordinateOperator:
    cloud = jnp.asarray(points)
    neighborhood = MeshfreeNeighborhoodPlan(cloud, 12).prepare()
    stencils = prepare_local_stencils(
        neighborhood,
        cloud,
        cloud,
        (MeshfreeFunctional(((2, 0), (0, 2)), (1.0, 1.0), name="laplacian"),),
        LocalStencilPolicy(polynomial_degree=2),
    )
    return MeshfreeOperator(stencils).operator


def _method(operator: DistributedMeshfreeOperator, /) -> solver.CallableFixedStepMethod:
    """Explicit diffusion on owner-blocked values; one identity on every partition."""

    def step(
        step_index: jax.Array,
        time: jax.Array,
        state: jax.Array,
        step_size: jax.Array,
        args: Any,
    ) -> solver.FixedStepResult:
        del step_index, time, args
        candidate = state + step_size * _KAPPA * operator.apply(state)
        finite = jnp.all(jnp.isfinite(candidate))
        return solver.FixedStepResult(
            candidate,
            jnp.where(finite, candidate, state),
            finite,
            jnp.zeros((), dtype=state.dtype),
            jnp.asarray(1, dtype=jnp.int32),
            jnp.asarray(1, dtype=jnp.int32),
            jnp.asarray(False),
            jnp.zeros((), dtype=state.dtype),
        )

    return solver.CallableFixedStepMethod(
        step, f"distributed-explicit-diffusion:{operator.operator_id}:{_KAPPA}"
    )


class _Bindings:
    def __init__(self, root: Path, /) -> None:
        root.mkdir(mode=0o700)
        self.repository = POSIXArtifactRepository(
            root,
            POSIXRepositoryPolicy(
                HPCFilesystemProfile(
                    "migration-posix",
                    "test-filesystem",
                    atomic_rename_same_filesystem=True,
                    file_fsync=True,
                    directory_fsync=True,
                    advisory_locking=True,
                    attempt_private_staging=True,
                ),
                maximum_chunk_bytes=1 << 16,
                maximum_metadata_bytes=1 << 20,
            ),
        )
        self.request = execution.ResourceRequest(
            cpu_cores=1,
            memory_bytes=64 << 20,
            maximum_checkpoint_staging_bytes=16 << 20,
            maximum_output_backlog_bytes=8 << 20,
        )
        self.policy = solver.CheckpointGenerationPolicy(3)
        dependency = SupportDependency(
            "migration-repository", self.repository.support_tuple.support_tuple_id
        )
        self.resolved = ResolvedRunSpec(
            (),
            (dependency,),
            release_index_id="release-index",
            profile_ids=(dependency.profile_id,),
            trust_policy_id="trust-policy",
            valid_at=10,
            valid_from=0,
            valid_until=20,
            prepared_configuration_id="migration-configuration",
            precision_policy_id="float64",
            resource_policy_id=self.request.resource_id,
            checkpoint_policy_id=self.policy.policy_id,
            output_policy_id="ordered-outbox",
            repository_id=self.repository.provider_id,
            scheduler_id="scheduler",
            auth_policy_id="auth-policy",
        )

    def runtime(
        self,
        operator: DistributedMeshfreeOperator,
        /,
        *,
        relation: solver.RuntimeRestartRelation | None = None,
        publisher: solver.ByteBoundedAsyncPublisher | None = None,
    ) -> tuple[solver.PreparedProductionRun, solver.RuntimeIdentityInventory]:
        method = _method(operator)
        retry = solver.RobustRetryPolicy(maximum_retries=0)
        plan = solver.ProductionRunPlan(
            method,
            retry,
            step_size=_STEP,
            end_time=_STEPS * _STEP,
            maximum_steps=_STEPS + 2,
            checkpoint_interval=4,
            segment_steps=4,
        )
        inventory = meshfree_runtime_inventory(
            operator,
            source="ownership-migration-worker",
            program=plan.plan_id,
            method=method,
            controller=retry.policy_id,
            precision="float64",
        )
        manifest = solver.ProductionCaseManifest.from_inventory(
            inventory, problem_id="distributed-diffusion", dtype="float64"
        )
        store = solver.ArtifactCheckpointStore(
            self.repository,
            manifest,
            self.policy,
            self.resolved,
            writer_id="migration-worker",
            resource_request=self.request,
            artifact_id="distributed-diffusion",
        )
        runtime = solver.PreparedProductionRun(
            manifest,
            plan,
            store,
            resolved_run_spec=self.resolved,
            restart_relation=relation,
            publisher=publisher,
        )
        return runtime, inventory


def _publisher(
    layout: DistributedPointLayout,
    outputs: dict[str, np.ndarray],
    calls: list[str],
    /,
) -> solver.ByteBoundedAsyncPublisher:
    def write(event_id: str, snapshot: Any) -> None:
        logical = np.asarray(layout.collect(jnp.asarray(snapshot)))
        if event_id in outputs:
            np.testing.assert_array_equal(outputs[event_id], logical)
        else:
            outputs[event_id] = logical.copy()
        calls.append(event_id)

    return solver.ByteBoundedAsyncPublisher(
        write,
        maximum_pending=2,
        maximum_pending_bytes=2 * layout.active.size * np.dtype(np.float64).itemsize,
    )


def _store(runtime: solver.PreparedProductionRun, /) -> solver.ArtifactCheckpointStore:
    store = runtime.checkpoint_store
    if not isinstance(store, solver.ArtifactCheckpointStore):
        raise RuntimeError("The migration scenario requires a repository outbox.")
    return store


def _outbox_content(
    manifest: ArtifactManifest, /
) -> tuple[tuple[str, int, int, int, str], ...]:
    return tuple(
        sorted(
            (
                chunk.logical_name,
                chunk.index,
                chunk.offset,
                chunk.plaintext_size,
                chunk.plaintext_sha256,
            )
            for chunk in manifest.chunks
            if chunk.logical_name.startswith("outbox-")
        )
    )


def _identity_restart(
    bindings: _Bindings,
    operator: DistributedMeshfreeOperator,
    layout: DistributedPointLayout,
    expected_outputs: dict[str, np.ndarray],
    resumed: solver.ProductionRunState,
    destination_manifest: ArtifactManifest,
    /,
) -> tuple[dict[str, Any], solver.PreparedProductionRun, solver.ProductionRunState]:
    outputs: dict[str, np.ndarray] = {}
    calls: list[str] = []
    with _publisher(layout, outputs, calls) as publisher:
        runtime, _ = bindings.runtime(operator, publisher=publisher)
        restored = runtime.resume(runtime.initial_state(layout.distribute(jnp.zeros(48))))
    store = _store(runtime)
    assert calls == []
    assert outputs == {}
    assert int(restored.output_cursor) == 2
    assert runtime.last_replay_classification == "bitwise"
    assert tuple(event.event_id for event in store._events) == tuple(expected_outputs)
    assert tuple(event.cursor for event in store._events) == (0, 1)
    assert all(event.delivered for event in store._events)
    for event in store._events:
        np.testing.assert_array_equal(
            layout.collect(event.state), expected_outputs[event.event_id]
        )
    np.testing.assert_array_equal(restored.accepted_state, resumed.accepted_state)
    identity_manifest = bindings.repository.get_manifest(store.artifact_id)
    destination_metadata = dict(destination_manifest.metadata)
    identity_metadata = dict(identity_manifest.metadata)
    # Leaving the admitted migration relation changes the prepared runtime ID.
    # Its restart binding may commit; that is not an output acknowledgement.
    assert runtime.run_id != destination_metadata["runtime_id"]
    assert identity_metadata["runtime_id"] == runtime.run_id
    assert identity_metadata["phase"] == "restart-lineage"
    no_reack = (
        _outbox_content(identity_manifest) == _outbox_content(destination_manifest)
        and identity_metadata["generation"] == destination_metadata["generation"]
        and identity_metadata["accepted_step"] == destination_metadata["accepted_step"]
    )
    assert no_reack
    lineage = [record.to_record() for record in store.migration_lineage()]
    with _publisher(layout, outputs, calls) as publisher:
        repeated, _ = bindings.runtime(operator, publisher=publisher)
        repeated_state = repeated.resume(
            repeated.initial_state(layout.distribute(jnp.zeros(48)))
        )
    repeated_store = _store(repeated)
    assert calls == []
    assert outputs == {}
    assert repeated.run_id == runtime.run_id
    assert int(repeated_state.output_cursor) == 2
    np.testing.assert_array_equal(repeated_state.accepted_state, restored.accepted_state)
    assert repeated_state.last_checkpoint_id == restored.last_checkpoint_id
    repeated_manifest = bindings.repository.get_manifest(store.artifact_id)
    repeated_unchanged = repeated_manifest.manifest_id == identity_manifest.manifest_id
    assert repeated_unchanged
    lineage_unchanged = [
        record.to_record() for record in repeated_store.migration_lineage()
    ] == lineage
    assert lineage_unchanged
    for event in repeated_store._events:
        np.testing.assert_array_equal(
            layout.collect(event.state), expected_outputs[event.event_id]
        )
    report = {
        "identity_retained_output_bitwise": all(
            np.array_equal(layout.collect(event.state), expected_outputs[event.event_id])
            for event in repeated_store._events
        ),
        "identity_published_events": calls,
        "identity_output_cursor": int(repeated_state.output_cursor),
        "identity_restart_no_reack": no_reack,
        "identity_checkpoint_phase": identity_metadata["phase"],
        "identity_repeat_manifest_unchanged": repeated_unchanged,
        "identity_lineage_unchanged": lineage_unchanged,
    }
    return report, repeated, repeated_state


def main(root: Path, /) -> dict[str, Any]:
    devices = jax.devices()
    if len(devices) != 4:
        raise RuntimeError("The migration worker requires exactly four devices.")
    group = execution.ExecutionRuntime.current().child_groups(1)[0]
    rng = np.random.default_rng(5)
    points = rng.uniform(0.0, 1.0, (48, 2))
    by_x = np.minimum((4 * points[:, 0]).astype(np.int32), 3)
    by_y = np.minimum((4 * points[:, 1]).astype(np.int32), 3)
    ownership = DistributedOwnershipPlan(
        MortonAddressPlan((0.0, 0.0), (1.0, 1.0), 10), group, _CAPACITY
    )
    source = DistributedPointLayout.from_global(ownership, points, by_x)
    reference_operator = _operator(points)

    def bind(layout: DistributedPointLayout) -> DistributedMeshfreeOperator:
        return DistributedMeshfreeOperator.bind(
            reference_operator, layout, layout, halo_capacity=_CAPACITY
        )

    values = np.sin(2.0 * np.pi * points[:, 0]) * np.cos(2.0 * np.pi * points[:, 1])
    initial = source.distribute(jnp.asarray(values))

    reference_runtime, _ = _Bindings(root / "reference").runtime(bind(source))
    reference = reference_runtime.run(reference_runtime.initial_state(initial))
    logical_reference = np.asarray(source.collect(reference.state.accepted_state))

    bindings = _Bindings(root / "migrated")
    runtime, source_inventory = bindings.runtime(bind(source))
    state = runtime.initial_state(initial)
    source_store = _store(runtime)
    source_outputs: dict[str, np.ndarray] = {}
    source_calls: list[str] = []
    delivered_id = "delivered-initial"
    pending_id = "pending-interruption"
    source_store.stage_output(delivered_id, 0, state.accepted_state)
    state = eqx.tree_at(
        lambda value: value.output_cursor,
        state,
        jnp.asarray(1, dtype=jnp.int64),
    )
    state = runtime.checkpoint(state)
    with _publisher(source, source_outputs, source_calls) as publisher:
        delivery_receipt = source_store.dispatch_outbox(publisher)
    if delivery_receipt is None:
        raise RuntimeError("The initial output was not durably acknowledged.")
    source_store.verify_commit(delivery_receipt)
    for _ in range(8):
        state, _ = runtime.step(state)
    source_store.stage_output(pending_id, 1, state.accepted_state)
    state = eqx.tree_at(
        lambda value: value.output_cursor,
        state,
        jnp.asarray(2, dtype=jnp.int64),
    )
    state = runtime.checkpoint(state)
    interrupted = np.asarray(source.collect(state.accepted_state))
    expected_outputs = {
        delivered_id: source_outputs[delivered_id],
        pending_id: interrupted,
    }
    assert source_calls == [delivered_id]
    assert tuple(event.delivered for event in source_store._events) == (True, False)

    logical = np.asarray(jax.device_get(source.logical_indices))
    active = np.asarray(jax.device_get(source.active))
    destinations = np.where(active, by_y[np.where(active, logical, 0)], 0).astype(
        np.int32
    )
    migrated = source.migrate(jnp.asarray(destinations), packet_capacity=_CAPACITY)
    committed = bool(np.asarray(jax.device_get(migrated.evidence.committed)))
    target = migrated.layout
    target_operator = bind(target)

    unrelated, target_inventory = bindings.runtime(target_operator)
    refused: list[str] = []
    try:
        unrelated.resume(unrelated.initial_state(target.distribute(jnp.zeros(48))))
    except solver.StaleRuntimeCheckpointError as error:
        refused = list(error.roles)

    relation = ownership_migration_relation(
        source_inventory,
        target_inventory,
        source,
        destinations,
        packet_capacity=_CAPACITY,
        source_template=state.accepted_state,
    )
    target_outputs: dict[str, np.ndarray] = {}
    target_calls: list[str] = []
    with _publisher(target, target_outputs, target_calls) as publisher:
        restarted, _ = bindings.runtime(
            target_operator, relation=relation, publisher=publisher
        )
        resumed = restarted.resume(
            restarted.initial_state(target.distribute(jnp.zeros(48)))
        )
    destination_store = _store(restarted)
    assert source_calls == [delivered_id]
    assert target_calls == [pending_id]
    np.testing.assert_array_equal(target_outputs[pending_id], interrupted)
    assert int(resumed.output_cursor) == 2
    assert tuple(event.cursor for event in destination_store._events) == (0, 1)
    assert all(event.delivered for event in destination_store._events)
    for event in destination_store._events:
        np.testing.assert_array_equal(
            target.collect(event.state), expected_outputs[event.event_id]
        )
    destination_manifest = bindings.repository.get_manifest(destination_store.artifact_id)
    destination_lineage = [
        record.to_record() for record in destination_store.migration_lineage()
    ]
    identity_report, final_runtime, final_start = _identity_restart(
        bindings, target_operator, target, expected_outputs, resumed, destination_manifest
    )
    assert [
        record.to_record() for record in destination_store.migration_lineage()
    ] == destination_lineage
    restored = np.asarray(target.collect(resumed.accepted_state))
    final = final_runtime.run(final_start)
    logical_final = np.asarray(target.collect(final.state.accepted_state))
    migration = relation.migration
    if migration is None:
        raise RuntimeError("An ownership relation carries its migration receipt.")
    return {
        "devices": len(devices),
        "migration_committed": committed,
        "source_partition": source.partition_fingerprint(),
        "target_partition": target.partition_fingerprint(),
        "migrated_roles": list(migration.migrated_roles),
        "identity_restart_refused_roles": refused,
        "replay_classification": restarted.last_replay_classification,
        "restored_bitwise": bool(np.array_equal(restored, interrupted)),
        "pending_output_bitwise": bool(
            np.array_equal(target_outputs[pending_id], interrupted)
        ),
        "retained_output_bitwise": all(
            np.array_equal(target.collect(event.state), expected_outputs[event.event_id])
            for event in destination_store._events
        ),
        "source_published_events": source_calls,
        "destination_published_events": target_calls,
        **identity_report,
        "final_status": final.state.status,
        "step_index": int(final.state.step_index),
        "reference_step_index": int(reference.state.step_index),
        "final_maximum_difference": float(
            np.max(np.abs(logical_final - logical_reference))
        ),
    }


if __name__ == "__main__":
    print(json.dumps(main(Path(sys.argv[1]).resolve())), flush=True)
