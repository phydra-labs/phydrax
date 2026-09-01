#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

from phydrax.solver._runtime_lifecycle import (
    AcceptedStepTrigger,
    BoundedAsyncPublisher,
    ExactTimeSchedule,
    read_runtime_checkpoint,
    RuntimeCheckpointEnvelope,
    StreamingObservablePlan,
    write_runtime_checkpoint,
)


def test_runtime_checkpoint_roundtrip_binds_all_compatibility_ids(tmp_path):
    state = {"solution": jnp.arange(6.0).reshape((3, 2))}
    controller = {"previous_error": jnp.asarray(0.25)}
    observer = ({"sum": jnp.asarray((1.0, 2.0))},)
    rng = {"key": jnp.asarray((3, 7), dtype=jnp.uint32)}
    envelope = RuntimeCheckpointEnvelope(
        state,
        time=1.25,
        step_index=12,
        schedule_cursor=3,
        mesh_id="mesh-a",
        method_id="method-a",
        precision_id="float64",
        topology_epoch_id="epoch-a",
        controller_state=controller,
        observer_states=observer,
        rng_state=rng,
        partition_id="partition-a",
    )
    path = write_runtime_checkpoint(tmp_path / "runtime.phx", envelope)
    restored = read_runtime_checkpoint(
        path,
        state_template=state,
        mesh_id="mesh-a",
        method_id="method-a",
        precision_id="float64",
        topology_epoch_id="epoch-a",
        controller_template=controller,
        observer_templates=observer,
        rng_template=rng,
        partition_id="partition-a",
    )
    np.testing.assert_array_equal(restored.state["solution"], state["solution"])
    assert restored.step_index == 12
    assert restored.schedule_cursor == 3
    assert restored.checkpoint_id == envelope.checkpoint_id


def test_exact_schedule_observable_and_trigger_are_restartable():
    schedule = ExactTimeSchedule(jnp.asarray((0.25, 0.5, 1.0)))
    np.testing.assert_allclose(schedule.clamp_step(0.2, 0.2, 0), 0.05)
    assert schedule.advance_cursor(0.5, 0) == 2

    observable = StreamingObservablePlan(
        "mass",
        lambda time, state, args: jnp.sum(state),
        "mean",
    )
    state = observable.initial_state(())
    state = observable.update(0.1, state, jnp.asarray((1.0, 2.0)))
    state = observable.update(0.2, state, jnp.asarray((3.0, 4.0)))
    np.testing.assert_allclose(observable.value(state), 5.0)
    merged = observable.merge(state, observable.initial_state(()))
    np.testing.assert_allclose(observable.value(merged), 5.0)

    trigger = AcceptedStepTrigger(4.0, hysteresis=0.5)
    trigger_state = trigger.initial_state()
    fire, trigger_state = trigger.evaluate(4.2, trigger_state, accepted=True)
    assert fire
    fire, trigger_state = trigger.evaluate(4.3, trigger_state, accepted=True)
    assert not fire
    _fire, trigger_state = trigger.evaluate(3.4, trigger_state, accepted=True)
    fire, trigger_state = trigger.evaluate(4.1, trigger_state, accepted=True)
    assert fire
    assert trigger_state.fire_count == 2


def test_bounded_async_publisher_snapshots_and_drains():
    published = []

    def writer(value):
        published.append(value)

    source = np.asarray((1.0, 2.0))
    with BoundedAsyncPublisher(writer, maximum_pending=1) as publisher:
        publisher.publish({"value": source})
        source[:] = -1.0
        publisher.drain()
        assert publisher.pending_count == 0
    np.testing.assert_array_equal(published[0]["value"], (1.0, 2.0))
