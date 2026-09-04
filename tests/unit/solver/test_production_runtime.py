#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec

import phydrax as phx
from phydrax.lifecycle._archive import migrate_configuration, rollback_configuration
from phydrax.lifecycle._migration import CompatibilityRegistry, MigrationEdge
from phydrax.lifecycle._repository import (
    HPCFilesystemProfile,
    POSIXArtifactRepository,
    POSIXRepositoryPolicy,
)
from phydrax.lifecycle._resolved_run import ResolvedRunSpec
from phydrax.qualification._evidence import SupportDependency
from phydrax.solver._production_runtime import ArtifactCheckpointStore
from phydrax.solver._runtime_lifecycle import (
    RuntimeRestartRelation,
    UnsupportedReplayError,
)


def _manifest(method):
    return phx.solver.ProductionCaseManifest(
        problem_id="constant-growth",
        method_id=method.method_id,
        precision_id="float64",
        topology_id="one-state",
        geometry_layout_id="static",
        dtype="float64",
    )


def test_production_run_checkpoints_observes_triggers_and_resumes(tmp_path):
    method = phx.solver.SSPRK33FixedStepMethod(
        lambda time, state, args: jnp.ones_like(state)
    )
    manifest = _manifest(method)
    store = phx.solver.DurableCheckpointStore(
        tmp_path / "checkpoints",
        manifest,
        phx.solver.CheckpointGenerationPolicy(2),
    )
    moment = phx.solver.StreamingMomentPlan(
        lambda time, state, args: jnp.mean(state),
        value_shape=(),
        histogram_edges=jnp.asarray((0.0, 0.15, 0.3, 1.0)),
        plan_id="state-mean",
    )
    trigger = phx.solver.AcceptedStepTriggerGraph(
        (phx.solver.AcceptedStepTrigger(0.12),), debounce_steps=0
    )
    published = []
    publisher = phx.solver.ByteBoundedAsyncPublisher(
        lambda event_id, snapshot: published.append((event_id, snapshot)),
        maximum_pending=2,
        maximum_pending_bytes=1024,
    )
    plan = phx.solver.ProductionRunPlan(
        method,
        phx.solver.RobustRetryPolicy(maximum_retries=2),
        step_size=0.1,
        end_time=0.25,
        maximum_steps=3,
        checkpoint_interval=2,
        output_schedule=phx.solver.ExactTimeSchedule(jnp.asarray((0.1, 0.2, 0.25))),
        moments=(moment,),
        trigger_bindings=(
            phx.solver.ProductionTriggerBinding(
                "state-threshold-checkpoint",
                trigger,
                (0,),
                "checkpoint",
                "checkpoint-on-state-threshold",
            ),
        ),
    )
    prepared = phx.solver.PreparedProductionRun(
        manifest, plan, store, publisher=publisher
    )
    initial = prepared.initial_state(jnp.asarray((0.0,)))
    result = prepared.run(initial)
    assert result.successful
    assert result.state.status == "completed"
    np.testing.assert_allclose(result.state.time, 0.25, atol=2.0e-12)
    np.testing.assert_allclose(result.state.accepted_state, (0.25,), atol=2.0e-12)
    assert result.state.moment_states[0].weight > 0.0
    assert result.state.trigger_states[0].fire_count == 1
    assert len(published) == 3
    assert len(publisher.acknowledged_event_ids) == 3
    assert json.loads((store.root / "terminal.json").read_text())["status"] == "completed"

    resumed = prepared.resume(initial)
    np.testing.assert_allclose(resumed.accepted_state, result.state.accepted_state)
    np.testing.assert_allclose(resumed.time, result.state.time)
    assert resumed.output_cursor == result.state.output_cursor
    assert (
        resumed.trigger_states[0].fire_count == result.state.trigger_states[0].fire_count
    )


def test_production_device_resident_execution_preserves_state_and_default(
    tmp_path, monkeypatch
):
    method = phx.solver.SSPRK33FixedStepMethod(
        lambda time, state, args: jnp.ones_like(state)
    )
    manifest = _manifest(method)
    policy = phx.solver.CheckpointGenerationPolicy(1)
    common = {
        "step_size": 0.1,
        "end_time": 0.2,
        "maximum_steps": 2,
        "checkpoint_interval": 10,
        "segment_steps": 1,
    }
    resident_plan = phx.solver.ProductionRunPlan(
        method,
        phx.solver.RobustRetryPolicy(maximum_retries=0),
        device_resident=True,
        **common,
    )
    default_plan = phx.solver.ProductionRunPlan(
        method,
        phx.solver.RobustRetryPolicy(maximum_retries=0),
        **common,
    )
    explicit_default_plan = phx.solver.ProductionRunPlan(
        method,
        phx.solver.RobustRetryPolicy(maximum_retries=0),
        device_resident=False,
        **common,
    )
    assert explicit_default_plan.plan_id == default_plan.plan_id
    assert resident_plan.plan_id != default_plan.plan_id
    device = jax.devices("cpu")[0]
    mesh = Mesh(np.asarray((device,), dtype=object), ("state",))
    sharding = NamedSharding(mesh, PartitionSpec("state"))
    initial_array = jax.device_put(jnp.zeros((4,)), sharding)
    resident = phx.solver.PreparedProductionRun(
        manifest,
        resident_plan,
        phx.solver.DurableCheckpointStore(tmp_path / "resident", manifest, policy),
    )
    resident_state = resident.initial_state(initial_array)
    with monkeypatch.context() as guard:
        guard.setattr(
            jax,
            "device_get",
            lambda *_args, **_kwargs: pytest.fail(
                "device-resident execution gathered its state"
            ),
        )
        following, transition = resident.step(resident_state)
    assert following.accepted_state.sharding == sharding
    assert transition.accepted_state.sharding == sharding

    default = phx.solver.PreparedProductionRun(
        manifest,
        default_plan,
        phx.solver.DurableCheckpointStore(tmp_path / "default", manifest, policy),
    )
    default_following, _ = default.step(default.initial_state(initial_array))
    assert isinstance(default_following.accepted_state, np.ndarray)


class _AlwaysReject(phx.solver.AbstractAcceptedStepTransform):
    transform_id = "always-reject"

    def __init__(self):
        self.transform_id = "always-reject"

    def apply(self, step_index, time, previous_state, candidate_state, args, /):
        del step_index, time, previous_state, args
        return phx.solver.AcceptedStepTransformResult(
            candidate_state,
            jnp.asarray(False),
            jnp.asarray(False),
            jnp.asarray(0.0),
        )


def test_production_run_writes_terminal_failure_manifest(tmp_path):
    method = phx.solver.SSPRK33FixedStepMethod(
        lambda time, state, args: jnp.ones_like(state),
        transform=_AlwaysReject(),
    )
    manifest = _manifest(method)
    store = phx.solver.DurableCheckpointStore(
        tmp_path / "failed",
        manifest,
        phx.solver.CheckpointGenerationPolicy(1),
    )
    plan = phx.solver.ProductionRunPlan(
        method,
        phx.solver.RobustRetryPolicy(maximum_retries=1),
        step_size=0.1,
        end_time=0.2,
        maximum_steps=2,
        checkpoint_interval=1,
    )
    result = phx.solver.PreparedProductionRun(manifest, plan, store).run(
        phx.solver.PreparedProductionRun(manifest, plan, store).initial_state(
            jnp.asarray((0.0,))
        )
    )
    assert not result.successful
    assert result.failure is not None
    terminal = json.loads((store.root / "terminal.json").read_text())
    assert terminal["status"] == "failed"
    assert terminal["failure_id"] == result.failure.failure_id


def test_vector_moment_triggers_require_explicit_components(tmp_path):
    method = phx.solver.SSPRK33FixedStepMethod(
        lambda time, state, args: jnp.ones_like(state)
    )
    manifest = _manifest(method)
    store = phx.solver.DurableCheckpointStore(
        tmp_path / "vector-trigger",
        manifest,
        phx.solver.CheckpointGenerationPolicy(1),
    )
    moment = phx.solver.StreamingMomentPlan(
        lambda time, state, args: state,
        value_shape=(2,),
        plan_id="vector-state",
    )
    graph = phx.solver.AcceptedStepTriggerGraph(
        (phx.solver.AcceptedStepTrigger(0.05),),
        debounce_steps=0,
    )
    missing_component = phx.solver.ProductionTriggerBinding(
        "vector-trigger",
        graph,
        (0,),
        "checkpoint",
        "vector-trigger-checkpoint",
    )
    with pytest.raises(ValueError, match="explicit trigger components"):
        phx.solver.ProductionRunPlan(
            method,
            phx.solver.RobustRetryPolicy(),
            step_size=0.1,
            end_time=0.1,
            maximum_steps=1,
            checkpoint_interval=1,
            moments=(moment,),
            trigger_bindings=(missing_component,),
        )

    binding = phx.solver.ProductionTriggerBinding(
        "vector-trigger",
        graph,
        (0,),
        "checkpoint",
        "vector-trigger-checkpoint",
        moment_components=(1,),
    )
    plan = phx.solver.ProductionRunPlan(
        method,
        phx.solver.RobustRetryPolicy(),
        step_size=0.1,
        end_time=0.1,
        maximum_steps=1,
        checkpoint_interval=1,
        moments=(moment,),
        trigger_bindings=(binding,),
    )
    prepared = phx.solver.PreparedProductionRun(manifest, plan, store)
    result = prepared.run(prepared.initial_state(jnp.zeros((2,))))
    assert result.successful
    assert result.state.trigger_states[0].fire_count == 1


def test_output_failure_never_checkpoints_advanced_cursor(tmp_path):
    method = phx.solver.SSPRK33FixedStepMethod(
        lambda time, state, args: jnp.ones_like(state)
    )
    manifest = _manifest(method)
    store = phx.solver.DurableCheckpointStore(
        tmp_path / "output-failure",
        manifest,
        phx.solver.CheckpointGenerationPolicy(1),
    )

    def fail_writer(event_id, snapshot):
        del event_id, snapshot
        raise RuntimeError("writer failed")

    publisher = phx.solver.ByteBoundedAsyncPublisher(
        fail_writer,
        maximum_pending=1,
        maximum_pending_bytes=1024,
    )
    plan = phx.solver.ProductionRunPlan(
        method,
        phx.solver.RobustRetryPolicy(),
        step_size=0.1,
        end_time=0.1,
        maximum_steps=1,
        checkpoint_interval=2,
        output_schedule=phx.solver.ExactTimeSchedule(jnp.asarray((0.1,))),
    )
    prepared = phx.solver.PreparedProductionRun(
        manifest,
        plan,
        store,
        publisher=publisher,
    )
    result = prepared.run(prepared.initial_state(jnp.zeros((1,))))

    assert not result.successful
    assert result.failure is not None
    assert result.failure.category == "output-failed"
    assert not (store.root / "latest.json").exists()
    terminal = json.loads((store.root / "terminal.json").read_text())
    assert terminal["status"] == "failed"


def _repository_policy(provider_id="production-posix"):
    profile = HPCFilesystemProfile(
        provider_id,
        "test-filesystem",
        atomic_rename_same_filesystem=True,
        file_fsync=True,
        directory_fsync=True,
        advisory_locking=True,
        attempt_private_staging=True,
    )
    return POSIXRepositoryPolicy(
        profile,
        maximum_chunk_bytes=128,
        maximum_metadata_bytes=1024 * 1024,
    )


def _artifact_bindings(
    root,
    method,
    /,
    *,
    topology_id="repository-topology",
    geometry_layout_id="repository-layout",
    artifact_id="production-checkpoint",
    repository_policy=None,
    prepared_configuration_id="prepared-configuration",
    failure_injector=None,
):
    repository_policy = (
        _repository_policy() if repository_policy is None else repository_policy
    )
    repository = POSIXArtifactRepository(
        root,
        repository_policy,
        failure_injector=failure_injector,
    )
    checkpoint_policy = phx.solver.CheckpointGenerationPolicy(3)
    dependency = SupportDependency(
        "repository-profile", repository.support_tuple.support_tuple_id
    )
    resolved = ResolvedRunSpec(
        (),
        (dependency,),
        release_index_id="release-index",
        profile_ids=(dependency.profile_id,),
        trust_policy_id="trust-policy",
        valid_at=10,
        valid_from=0,
        valid_until=20,
        prepared_configuration_id=prepared_configuration_id,
        precision_policy_id="precision-policy",
        resource_policy_id="resource-policy",
        checkpoint_policy_id=checkpoint_policy.policy_id,
        output_policy_id="output-policy",
        repository_id=repository.provider_id,
        scheduler_id="scheduler",
        auth_policy_id="auth-policy",
    )
    manifest = phx.solver.ProductionCaseManifest(
        problem_id="repository-growth",
        method_id=method.method_id,
        precision_id="native-precision",
        topology_id=topology_id,
        geometry_layout_id=geometry_layout_id,
        dtype=str(jnp.asarray(0.0).dtype),
    )
    store = ArtifactCheckpointStore(
        repository,
        manifest,
        checkpoint_policy,
        resolved,
        writer_id="production-worker",
        artifact_id=artifact_id,
    )
    return repository, manifest, store, resolved, checkpoint_policy, repository_policy


def _repository_plan(method, *, end_time=0.2, output_schedule=None):
    return phx.solver.ProductionRunPlan(
        method,
        phx.solver.RobustRetryPolicy(maximum_retries=0),
        step_size=0.1,
        end_time=end_time,
        maximum_steps=max(1, int(round(end_time / 0.1))),
        checkpoint_interval=1,
        segment_steps=1,
        output_schedule=output_schedule,
    )


def test_artifact_repository_checkpoint_outbox_resume_and_cache_rebuild(tmp_path):
    method = phx.solver.SSPRK33FixedStepMethod(
        lambda time, state, args: jnp.ones_like(state)
    )
    repository, manifest, store, resolved, _, repository_policy = _artifact_bindings(
        tmp_path / "repository", method
    )
    plan = _repository_plan(
        method,
        output_schedule=phx.solver.ExactTimeSchedule(jnp.asarray((0.1, 0.2))),
    )
    published = []
    publisher = phx.solver.ByteBoundedAsyncPublisher(
        lambda event_id, snapshot: published.append(
            (event_id, np.asarray(snapshot).copy())
        ),
        maximum_pending=2,
        maximum_pending_bytes=1024,
    )
    prepared = phx.solver.PreparedProductionRun(
        manifest,
        plan,
        store,
        publisher=publisher,
        resolved_run_spec=resolved,
    )
    initial = prepared.initial_state(jnp.asarray((0.0,)))
    result = prepared.run(initial)
    assert result.successful
    assert result.state.output_cursor == 2
    assert tuple(event.cursor for event in store._events) == (0, 1)
    assert all(event.delivered for event in store._events)
    assert len({event.event_id for event in store._events}) == 2
    assert len(published) == 2

    committed = repository.get_manifest(store.artifact_id)
    logical_names = {chunk.logical_name for chunk in committed.chunks}
    assert "runtime" in logical_names
    assert "state-manifest" in logical_names
    assert "outbox-manifest" in logical_names
    assert any(name.startswith("state-") for name in logical_names)
    assert any(name.startswith("outbox-") for name in logical_names)
    assert all("cache" not in name for name in logical_names)

    duplicate = store._events[0]
    store.stage_output(duplicate.event_id, duplicate.cursor, jnp.asarray((99.0,)))
    assert len(store._events) == 2

    reopened, _, resumed_store, reopened_spec, _, _ = _artifact_bindings(
        tmp_path / "repository",
        method,
        repository_policy=repository_policy,
    )
    replayed = []
    resumed_publisher = phx.solver.ByteBoundedAsyncPublisher(
        lambda event_id, snapshot: replayed.append(event_id),
        maximum_pending=2,
        maximum_pending_bytes=1024,
    )
    resumed_runtime = phx.solver.PreparedProductionRun(
        resumed_store.manifest,
        plan,
        resumed_store,
        publisher=resumed_publisher,
        resolved_run_spec=reopened_spec,
    )
    resumed = resumed_runtime.resume(resumed_runtime.initial_state(jnp.asarray((0.0,))))
    np.testing.assert_allclose(resumed.accepted_state, result.state.accepted_state)
    assert resumed.output_cursor == 2
    assert resumed_runtime.last_replay_classification == "bitwise"
    assert replayed == []
    assert reopened.get_manifest(resumed_store.artifact_id).complete


@pytest.mark.parametrize(
    "failure_point", ("before_manifest", "after_manifest", "before_pointer")
)
def test_artifact_repository_crash_never_exposes_partial_checkpoint(
    tmp_path, failure_point
):
    method = phx.solver.SSPRK33FixedStepMethod(
        lambda time, state, args: jnp.ones_like(state)
    )
    repository, manifest, store, resolved, _, repository_policy = _artifact_bindings(
        tmp_path / failure_point, method
    )
    plan = _repository_plan(method, end_time=0.3)
    prepared = phx.solver.PreparedProductionRun(
        manifest, plan, store, resolved_run_spec=resolved
    )
    initial = prepared.initial_state(jnp.asarray((0.0,)))
    committed_state, _ = prepared.step(initial)
    stable_manifest_id = repository.get_manifest(store.artifact_id).manifest_id

    def fail(point):
        if point == failure_point:
            raise RuntimeError(f"crash at {point}")

    repository.failure_injector = fail
    with pytest.raises(RuntimeError, match="crash at"):
        prepared.step(committed_state)

    _, _, reopened_store, reopened_spec, _, _ = _artifact_bindings(
        tmp_path / failure_point,
        method,
        repository_policy=repository_policy,
    )
    reopened_runtime = phx.solver.PreparedProductionRun(
        reopened_store.manifest,
        plan,
        reopened_store,
        resolved_run_spec=reopened_spec,
    )
    restored = reopened_runtime.resume(initial)
    assert int(restored.step_index) == 1
    assert (
        reopened_store.repository.get_manifest(reopened_store.artifact_id).manifest_id
        == stable_manifest_id
    )


def test_artifact_repository_admitted_topology_restart_and_rejections(tmp_path):
    method = phx.solver.SSPRK33FixedStepMethod(
        lambda time, state, args: jnp.ones_like(state)
    )
    repository, source_manifest, source_store, resolved, _, repository_policy = (
        _artifact_bindings(
            tmp_path / "topology",
            method,
            topology_id="topology-a",
            geometry_layout_id="layout-a",
            artifact_id="topology-restart",
        )
    )
    plan = _repository_plan(method, end_time=0.1)
    source_runtime = phx.solver.PreparedProductionRun(
        source_manifest, plan, source_store, resolved_run_spec=resolved
    )
    source = source_runtime.initial_state(jnp.asarray((1.0, 2.0)))
    source_runtime.checkpoint(source)
    parent_manifest_id = repository.get_manifest(source_store.artifact_id).manifest_id

    _, target_manifest, target_store, target_spec, _, _ = _artifact_bindings(
        tmp_path / "topology",
        method,
        topology_id="topology-b",
        geometry_layout_id="layout-b",
        artifact_id="topology-restart",
        repository_policy=repository_policy,
    )
    unsupported_relation = RuntimeRestartRelation(
        "topology-a",
        "topology-b",
        classification="unsupported",
        relation_id="unsupported-topology-replay",
    )
    unsupported_runtime = phx.solver.PreparedProductionRun(
        target_manifest,
        plan,
        target_store,
        resolved_run_spec=target_spec,
        restart_relation=unsupported_relation,
    )
    with pytest.raises(UnsupportedReplayError):
        unsupported_runtime.resume(unsupported_runtime.initial_state(jnp.asarray((0.0,))))
    _, target_manifest, target_store, target_spec, _, _ = _artifact_bindings(
        tmp_path / "topology",
        method,
        topology_id="topology-b",
        geometry_layout_id="layout-b",
        artifact_id="topology-restart",
        repository_policy=repository_policy,
    )

    def aggregate(source_arrays, source_specification, template, encoding):
        del encoding
        source_leaf = np.asarray(source_arrays[source_specification["arrays"][0]])
        return jnp.asarray((source_leaf.sum(),), dtype=jnp.asarray(template).dtype)

    relation = RuntimeRestartRelation(
        "topology-a",
        "topology-b",
        classification="tolerance",
        tolerance=1.0e-12,
        relation_id="admitted-topology-aggregation",
        restorer=aggregate,
    )
    target_runtime = phx.solver.PreparedProductionRun(
        target_manifest,
        plan,
        target_store,
        resolved_run_spec=target_spec,
        restart_relation=relation,
    )
    resumed = target_runtime.resume(target_runtime.initial_state(jnp.asarray((0.0,))))
    np.testing.assert_allclose(resumed.accepted_state, (3.0,))
    assert target_runtime.last_replay_classification == "tolerance"
    lineage_manifest = repository.get_manifest(target_store.artifact_id)
    assert lineage_manifest.base_manifest_id == parent_manifest_id

    _, rejected_manifest, rejected_store, rejected_spec, _, _ = _artifact_bindings(
        tmp_path / "topology",
        method,
        topology_id="topology-c",
        geometry_layout_id="layout-c",
        artifact_id="topology-restart",
        repository_policy=repository_policy,
    )
    rejected_relation = RuntimeRestartRelation(
        "wrong-source",
        "topology-c",
        classification="unsupported",
        relation_id="rejected-topology-relation",
    )
    rejected_runtime = phx.solver.PreparedProductionRun(
        rejected_manifest,
        plan,
        rejected_store,
        resolved_run_spec=rejected_spec,
        restart_relation=rejected_relation,
    )
    with pytest.raises((ValueError, UnsupportedReplayError)):
        rejected_runtime.resume(rejected_runtime.initial_state(jnp.asarray((0.0,))))


def test_configuration_migration_commits_lineage_and_rollback_selects_parent(tmp_path):
    method = phx.solver.SSPRK33FixedStepMethod(
        lambda time, state, args: jnp.ones_like(state)
    )
    repository, _, _, _, _, _ = _artifact_bindings(tmp_path / "configuration", method)
    edge = MigrationEdge(
        "configuration-v1",
        "configuration-v2",
        lambda record: {"coefficient": record["coefficient"], "scheme": "current"},
        migration_id="configuration-v1-to-v2",
    )
    registry = CompatibilityRegistry("configuration-v2", (edge,))
    artifact = migrate_configuration(
        repository,
        registry,
        {"coefficient": 2},
        source_format_id="configuration-v1",
        writer_id="configuration-writer",
    )
    assert artifact.manifest.artifact_id == artifact.report.output_digest
    assert artifact.manifest.base_manifest_id is None
    parent = rollback_configuration(registry, artifact)
    assert parent["artifact_id"] == artifact.report.input_digest
    assert parent["record"] == {"coefficient": 2}
    assert parent["lineage"] == [artifact.report.input_digest]


def test_runtime_configuration_migration_requires_lineage_and_commits_child(tmp_path):
    method = phx.solver.SSPRK33FixedStepMethod(
        lambda time, state, args: jnp.ones_like(state)
    )
    edge = MigrationEdge(
        "runtime-configuration-v1",
        "runtime-configuration-v2",
        lambda record: {"coefficient": record["coefficient"], "current": True},
        migration_id="runtime-configuration-upgrade",
    )
    registry = CompatibilityRegistry("runtime-configuration-v2", (edge,))
    report = registry.resolve(
        {"coefficient": 3},
        source_format_id="runtime-configuration-v1",
    )
    repository, source_manifest, source_store, source_spec, _, repository_policy = (
        _artifact_bindings(
            tmp_path / "runtime-configuration",
            method,
            artifact_id="runtime-configuration-checkpoint",
            prepared_configuration_id=report.input_digest,
        )
    )
    plan = _repository_plan(method, end_time=0.1)
    source_runtime = phx.solver.PreparedProductionRun(
        source_manifest,
        plan,
        source_store,
        resolved_run_spec=source_spec,
    )
    source_runtime.checkpoint(source_runtime.initial_state(jnp.asarray((4.0,))))
    parent_manifest_id = repository.get_manifest(source_store.artifact_id).manifest_id

    _, target_manifest, target_store, target_spec, _, _ = _artifact_bindings(
        tmp_path / "runtime-configuration",
        method,
        artifact_id="runtime-configuration-checkpoint",
        repository_policy=repository_policy,
        prepared_configuration_id=report.output_digest,
    )
    rejected_runtime = phx.solver.PreparedProductionRun(
        target_manifest,
        plan,
        target_store,
        resolved_run_spec=target_spec,
    )
    with pytest.raises(ValueError, match="without an explicit migration"):
        rejected_runtime.resume(rejected_runtime.initial_state(jnp.asarray((0.0,))))
    _, target_manifest, target_store, target_spec, _, _ = _artifact_bindings(
        tmp_path / "runtime-configuration",
        method,
        artifact_id="runtime-configuration-checkpoint",
        repository_policy=repository_policy,
        prepared_configuration_id=report.output_digest,
    )
    target_runtime = phx.solver.PreparedProductionRun(
        target_manifest,
        plan,
        target_store,
        resolved_run_spec=target_spec,
        migration_report=report,
    )
    resumed = target_runtime.resume(target_runtime.initial_state(jnp.asarray((0.0,))))
    np.testing.assert_allclose(resumed.accepted_state, (4.0,))
    child = repository.get_manifest(target_store.artifact_id)
    assert child.base_manifest_id == parent_manifest_id
    assert dict(child.metadata)["phase"] == "restart-lineage"
