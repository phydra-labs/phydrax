#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import json

import jax.numpy as jnp
import numpy as np

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
        maximum_steps=3,
        checkpoint_interval=2,
        schedule=phx.solver.ExactTimeSchedule(jnp.asarray((0.15, 0.3))),
        moments=(moment,),
        triggers=(trigger,),
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
