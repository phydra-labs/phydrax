#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import json

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


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
