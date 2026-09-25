#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import hashlib
import json
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec

import phydrax as phx
from phydrax._array_archive import ArrayArchiveCorruptionError
from phydrax.lifecycle._archive import migrate_configuration, rollback_configuration
from phydrax.lifecycle._chunk_repository import CheckpointResourcePolicy
from phydrax.lifecycle._migration import CompatibilityRegistry, MigrationEdge
from phydrax.lifecycle._repository import (
    HPCFilesystemProfile,
    POSIXArtifactRepository,
    POSIXRepositoryPolicy,
)
from phydrax.lifecycle._resolved_run import ResolvedRunSpec
from phydrax.qualification._evidence import SupportDependency
from phydrax.solver._production_runtime import (
    ArtifactCheckpointStore,
    CheckpointCommitReceipt,
)
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


def test_production_resource_forecast_rejects_invalid_multipliers():
    mesh = phx.discretization.CellMesh.from_triangles(
        jnp.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))),
        jnp.asarray(((0, 1, 2),), dtype=jnp.int32),
    )
    field = phx.discretization.FiniteElementFieldSpec(
        "u", phx.discretization.lagrange_element("triangle", 1)
    )
    discretization = phx.discretization.FiniteElementPlan(mesh, field).prepare()
    form = phx.equations.FiniteElementForm(
        "resource-forecast",
        "u",
        (phx.equations.DiffusionAction("u"),),
    )
    action_ir = phx.equations.fem.lower_finite_element_form(form, discretization)
    worksets = phx.equations.fem.compile_workset_program(action_ir, form, discretization)
    budget = phx.solver.ProductionResourceBudget(
        maximum_compile_units=100,
        maximum_host_bytes=1_000_000,
        maximum_device_bytes=1_000_000,
        maximum_output_queue_bytes=1_000_000,
    )
    with pytest.raises(ValueError, match="ad_multiplier"):
        phx.solver.prepare_production_resource_forecast(
            worksets,
            jnp.zeros((3,)),
            budget,
            ad_multiplier=-1.0,
        )
    with pytest.raises(ValueError, match="output_snapshots"):
        phx.solver.prepare_production_resource_forecast(
            worksets,
            jnp.zeros((3,)),
            budget,
            output_snapshots=-1,
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


def test_production_iteration_session_stops_and_restores_exact_cursor(tmp_path):
    method = phx.solver.SSPRK33FixedStepMethod(
        lambda time, state, args: jnp.ones_like(state)
    )
    manifest = _manifest(method)
    store = phx.solver.DurableCheckpointStore(
        tmp_path / "observed-checkpoints",
        manifest,
        phx.solver.CheckpointGenerationPolicy(2),
    )
    plan = phx.solver.ProductionRunPlan(
        method,
        phx.solver.RobustRetryPolicy(maximum_retries=1),
        step_size=0.1,
        end_time=0.3,
        maximum_steps=3,
        checkpoint_interval=1,
        segment_steps=3,
    )
    events = []

    def phase(event):
        return int(event.record.coordinates.phase)

    session = phx.execution.IterationSession(
        "production-observation-test",
        sinks=(
            phx.execution.CallableIterationSink(
                lambda event: events.append(event), "collector"
            ),
        ),
        control=phx.execution.CallableIterationHostControl(
            lambda event: phase(event) == int(phx.execution.IterationPhase.COMMIT),
            "stop-after-first-step",
        ),
    )
    prepared = phx.solver.PreparedProductionRun(
        manifest,
        plan,
        store,
        session=session,
    )
    initial = prepared.initial_state(jnp.asarray((0.0,)))
    result = prepared.run(initial)

    assert result.state.status == "canceled"
    assert not result.successful
    assert int(result.state.step_index) == 1
    assert [phase(event) for event in events] == [
        int(phx.execution.IterationPhase.START),
        int(phx.execution.IterationPhase.COMMIT),
        int(phx.execution.IterationPhase.TERMINAL),
    ]
    assert result.iteration_session_state is not None
    assert result.iteration_session_state.cursor == 3

    resumed_session = phx.execution.IterationSession(
        "production-observation-test",
        control=phx.execution.CallableIterationHostControl(
            lambda event: phase(event) == int(phx.execution.IterationPhase.COMMIT),
            "stop-after-first-step",
        ),
    )
    resumed_runtime = phx.solver.PreparedProductionRun(
        manifest,
        plan,
        store,
        session=resumed_session,
    )
    template = resumed_runtime.initial_state(jnp.asarray((0.0,)))
    resumed = resumed_runtime.resume(template)
    assert int(resumed.step_index) == 1
    assert resumed_session.cursor == 3
    assert resumed_session.stop_requested


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
    transform_id: str = "always-reject"

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
    assert result.failure.error_code == "PRODUCTION_OUTPUT_DRAIN_FAILED"
    assert (store.root / "committed.json").is_file()
    resumed = prepared.resume(prepared.initial_state(jnp.zeros((1,))))
    assert int(resumed.output_cursor) == 0
    terminal = json.loads((store.root / "terminal.json").read_text())
    assert terminal["status"] == "failed"
    assert terminal["failure_error_code"] == "PRODUCTION_OUTPUT_DRAIN_FAILED"
    assert terminal["failure_id"] == result.failure.failure_id
    assert terminal["last_checkpoint_id"] == result.state.last_checkpoint_id
    assert result.failure.last_checkpoint_id == result.state.last_checkpoint_id


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
    maximum_output_backlog_bytes=8 * 1024 * 1024,
):
    repository_policy = (
        _repository_policy() if repository_policy is None else repository_policy
    )
    repository = POSIXArtifactRepository(
        root,
        repository_policy,
        failure_injector=failure_injector,
    )
    resource_request = phx.execution.ResourceRequest(
        cpu_cores=1,
        memory_bytes=32 * 1024 * 1024,
        maximum_checkpoint_staging_bytes=16 * 1024 * 1024,
        maximum_output_backlog_bytes=maximum_output_backlog_bytes,
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
        resource_policy_id=resource_request.resource_id,
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
        resource_request=resource_request,
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
        "source-configuration",
        "current-configuration",
        lambda record: {"coefficient": record["coefficient"], "scheme": "current"},
        migration_id="configuration-upgrade",
    )
    registry = CompatibilityRegistry("current-configuration", (edge,))
    artifact = migrate_configuration(
        repository,
        registry,
        {"coefficient": 2},
        source_format_id="source-configuration",
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
        "source-runtime-configuration",
        "current-runtime-configuration",
        lambda record: {"coefficient": record["coefficient"], "current": True},
        migration_id="runtime-configuration-upgrade",
    )
    registry = CompatibilityRegistry("current-runtime-configuration", (edge,))
    report = registry.resolve(
        {"coefficient": 3},
        source_format_id="source-runtime-configuration",
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


def test_checkpoint_commit_receipt_defeats_forged_last_checkpoint_id(tmp_path):
    method = phx.solver.SSPRK33FixedStepMethod(
        lambda time, state, args: jnp.ones_like(state)
    )
    manifest = _manifest(method)
    store = phx.solver.DurableCheckpointStore(
        tmp_path / "receipt-store",
        manifest,
        phx.solver.CheckpointGenerationPolicy(1),
    )
    plan = phx.solver.ProductionRunPlan(
        method,
        phx.solver.RobustRetryPolicy(),
        step_size=0.1,
        end_time=0.1,
        maximum_steps=1,
        checkpoint_interval=1,
    )
    prepared = phx.solver.PreparedProductionRun(manifest, plan, store)
    state = prepared.initial_state(jnp.asarray((0.0,)))
    envelope = prepared._envelope(state)
    forged = phx.solver.ProductionRunState(
        state.step_index,
        state.time,
        state.accepted_state,
        state.controller_state,
        state.rng_state,
        state.schedule_cursor,
        state.moment_states,
        state.trigger_states,
        state.output_cursor,
        "canceled",
        envelope.checkpoint_id,
    )
    fake_receipt = CheckpointCommitReceipt(
        store.store_id,
        envelope.checkpoint_id,
        envelope.content_digest,
        envelope.runtime_id,
        0,
        0,
        "0" * 64,
        "generation-00000000.phx",
        1,
        "0" * 64,
    )

    with pytest.raises(FileNotFoundError):
        store.verify_commit(fake_receipt)
    preserved, receipt = prepared.commit_checkpoint(forged)

    assert preserved.last_checkpoint_id == envelope.checkpoint_id
    assert store.verify_commit(receipt) == receipt
    assert (store.root / receipt.commit_locator).is_file()
    sequenced_store = phx.solver.DurableCheckpointStore(
        tmp_path / "sequenced-receipt-store",
        manifest,
        phx.solver.CheckpointGenerationPolicy(1),
    )
    sequenced_receipt = sequenced_store.commit(7, envelope)
    assert sequenced_receipt.generation == 7
    assert sequenced_receipt.accepted_step == 0
    assert sequenced_store.receipt_for(envelope) == sequenced_receipt
    archive_path = store.root / receipt.commit_locator
    corrupted = bytearray(archive_path.read_bytes())
    corrupted[len(corrupted) // 2] ^= 0x01
    archive_path.write_bytes(corrupted)
    with pytest.raises(ValueError, match="integrity"):
        store.verify_commit(receipt)


def test_repository_generation_is_independent_of_accepted_step(tmp_path):
    method = phx.solver.SSPRK33FixedStepMethod(
        lambda time, state, args: jnp.ones_like(state)
    )
    repository, manifest, store, resolved, _, _ = _artifact_bindings(
        tmp_path / "repository-generation", method
    )
    plan = _repository_plan(method, end_time=0.1)
    prepared = phx.solver.PreparedProductionRun(
        manifest, plan, store, resolved_run_spec=resolved
    )
    state = prepared.initial_state(jnp.asarray((0.0,)))
    receipt = store.commit(9, prepared._envelope(state))
    metadata = dict(repository.get_manifest(store.artifact_id).metadata)

    assert receipt.generation == 9
    assert receipt.accepted_step == 0
    assert metadata["generation"] == "9"
    assert metadata["accepted_step"] == "0"
    resumed = prepared.resume(state)
    assert int(resumed.step_index) == 0
    _, repeated = prepared.commit_checkpoint(resumed)
    assert repeated.generation == 9
    assert repeated.accepted_step == 0


def test_durable_checkpoint_store_rejects_symlink_root_and_fifo_pointer(tmp_path):
    method = phx.solver.SSPRK33FixedStepMethod(
        lambda time, state, args: jnp.ones_like(state)
    )
    manifest = _manifest(method)
    real_root = tmp_path / "real-root"
    real_root.mkdir()
    linked_root = tmp_path / "linked-root"
    linked_root.symlink_to(real_root, target_is_directory=True)
    with pytest.raises(OSError):
        phx.solver.DurableCheckpointStore(
            linked_root,
            manifest,
            phx.solver.CheckpointGenerationPolicy(1),
        )

    store = phx.solver.DurableCheckpointStore(
        tmp_path / "fifo-store",
        manifest,
        phx.solver.CheckpointGenerationPolicy(1),
    )
    os.mkfifo(store.root / "committed.json")
    with pytest.raises(ValueError, match="regular file"):
        store.latest(jnp.asarray((0.0,)))


def test_durable_checkpoint_generation_read_never_follows_symlink(tmp_path):
    method = phx.solver.SSPRK33FixedStepMethod(
        lambda time, state, args: jnp.ones_like(state)
    )
    manifest = _manifest(method)
    store = phx.solver.DurableCheckpointStore(
        tmp_path / "generation-store",
        manifest,
        phx.solver.CheckpointGenerationPolicy(1),
    )
    plan = phx.solver.ProductionRunPlan(
        method,
        phx.solver.RobustRetryPolicy(),
        step_size=0.1,
        end_time=0.1,
        maximum_steps=1,
        checkpoint_interval=1,
    )
    prepared = phx.solver.PreparedProductionRun(manifest, plan, store)
    state, receipt = prepared.commit_checkpoint(
        prepared.initial_state(jnp.asarray((0.0,)))
    )
    generation = store.root / receipt.commit_locator
    unrelated = tmp_path / "unrelated"
    unrelated.write_bytes(b"not a checkpoint")
    generation.unlink()
    generation.symlink_to(unrelated)

    with pytest.raises(ArrayArchiveCorruptionError):
        prepared.resume(state)


def test_repository_aggregate_limits_precede_chunk_reads(tmp_path, monkeypatch):
    method = phx.solver.SSPRK33FixedStepMethod(
        lambda time, state, args: jnp.ones_like(state)
    )
    repository, _, store, _, _, _ = _artifact_bindings(
        tmp_path / "bounded-repository", method
    )
    resources = CheckpointResourcePolicy(
        store.resource_request.resource_id,
        maximum_manifest_bytes=1024 * 1024,
        maximum_chunks=2,
        maximum_logical_payloads=2,
        maximum_logical_payload_bytes=1024,
        maximum_total_plaintext_bytes=1024,
        maximum_total_encoded_bytes=1024,
        maximum_outbox_records=1,
        maximum_outbox_bytes=1024,
    )
    store.checkpoint_resources = resources
    transaction_id = hashlib.sha256(b"malicious-transaction").hexdigest()
    chunks = tuple(
        phx.lifecycle.ChunkRecord(
            transaction_id,
            "runtime",
            index,
            0,
            0,
            0,
            hashlib.sha256(b"").hexdigest(),
            hashlib.sha256(b"").hexdigest(),
            "identity",
            f"roots/malicious/chunks/runtime/{index}",
        )
        for index in range(resources.maximum_chunks + 1)
    )
    malicious = phx.lifecycle.ArtifactManifest(
        repository.provider_id,
        store.artifact_id,
        transaction_id,
        None,
        chunks,
        committed_at=1,
    )
    monkeypatch.setattr(
        repository,
        "read_chunk",
        lambda *_args, **_kwargs: pytest.fail(
            "chunk read occurred before aggregate preflight"
        ),
    )

    with pytest.raises(phx.lifecycle.RepositoryCorruptionError, match="chunk-count"):
        store._read_payloads(malicious)

    payload_digest = hashlib.sha256(b"x" * 600).hexdigest()
    aggregate_chunks = tuple(
        phx.lifecycle.ChunkRecord(
            transaction_id,
            "runtime",
            index,
            index * 600,
            600,
            600,
            payload_digest,
            payload_digest,
            "identity",
            f"roots/malicious/chunks/aggregate/{index}",
        )
        for index in range(2)
    )
    aggregate = phx.lifecycle.ArtifactManifest(
        repository.provider_id,
        store.artifact_id,
        transaction_id,
        None,
        aggregate_chunks,
        committed_at=1,
    )
    with pytest.raises(phx.lifecycle.RepositoryCorruptionError, match="plaintext-byte"):
        store._read_payloads(aggregate)


def test_publisher_exception_detail_is_not_durable_or_public(tmp_path):
    secret = "https://user:credential@example.invalid/private/checkpoint"
    method = phx.solver.SSPRK33FixedStepMethod(
        lambda time, state, args: jnp.ones_like(state)
    )
    manifest = _manifest(method)
    store = phx.solver.DurableCheckpointStore(
        tmp_path / "redacted-output-failure",
        manifest,
        phx.solver.CheckpointGenerationPolicy(1),
    )
    publisher = phx.solver.ByteBoundedAsyncPublisher(
        lambda event_id, snapshot: (_ for _ in ()).throw(RuntimeError(secret)),
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
        manifest, plan, store, publisher=publisher
    )
    result = prepared.run(prepared.initial_state(jnp.zeros((1,))))
    terminal_text = (store.root / "terminal.json").read_text()

    assert result.failure is not None
    assert result.failure.error_code == "PRODUCTION_OUTPUT_DRAIN_FAILED"
    assert secret not in repr(result.failure)
    assert secret not in terminal_text
    assert (
        json.loads(terminal_text)["failure_error_code"]
        == "PRODUCTION_OUTPUT_DRAIN_FAILED"
    )


def test_first_repository_output_failure_commits_bounded_terminal_transaction(
    tmp_path,
):
    method = phx.solver.SSPRK33FixedStepMethod(
        lambda time, state, args: jnp.ones_like(state)
    )
    repository, manifest, store, resolved, _, _ = _artifact_bindings(
        tmp_path / "first-output-failure",
        method,
        maximum_output_backlog_bytes=1,
    )
    plan = _repository_plan(
        method,
        end_time=0.1,
        output_schedule=phx.solver.ExactTimeSchedule(jnp.asarray((0.1,))),
    )
    publisher = phx.solver.ByteBoundedAsyncPublisher(
        lambda event_id, snapshot: None,
        maximum_pending=1,
        maximum_pending_bytes=1024,
    )
    prepared = phx.solver.PreparedProductionRun(
        manifest,
        plan,
        store,
        publisher=publisher,
        resolved_run_spec=resolved,
    )

    result = prepared.run(prepared.initial_state(jnp.zeros((1,))))
    committed = repository.get_manifest(store.artifact_id)

    assert result.failure is not None
    assert result.failure.error_code == "PRODUCTION_OUTPUT_PUBLISH_FAILED"
    assert result.state.last_checkpoint_id
    assert result.failure.last_checkpoint_id == result.state.last_checkpoint_id
    assert dict(committed.metadata)["phase"] == "terminal"
