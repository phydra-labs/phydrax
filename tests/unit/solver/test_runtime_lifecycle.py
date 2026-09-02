#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax.solver._runtime_lifecycle import (
    AcceptedStepTrigger,
    BoundedAsyncPublisher,
    ExactTimeSchedule,
    read_runtime_checkpoint,
    RuntimeCheckpointEncodingPlan,
    RuntimeCheckpointEnvelope,
    RuntimeCheckpointLeafBinding,
    StreamingMomentPlan,
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


def test_runtime_checkpoint_identity_is_content_derived():
    first = RuntimeCheckpointEnvelope(
        {"state": jnp.asarray((1.0, 2.0))},
        time=0.0,
        step_index=0,
        schedule_cursor=0,
        mesh_id="mesh",
        method_id="method",
        precision_id="precision",
        topology_epoch_id="epoch",
    )
    repeated = RuntimeCheckpointEnvelope(
        {"state": jnp.asarray((1.0, 2.0))},
        time=0.0,
        step_index=0,
        schedule_cursor=0,
        mesh_id="mesh",
        method_id="method",
        precision_id="precision",
        topology_epoch_id="epoch",
    )
    changed = RuntimeCheckpointEnvelope(
        {"state": jnp.asarray((1.0, 3.0))},
        time=0.0,
        step_index=0,
        schedule_cursor=0,
        mesh_id="mesh",
        method_id="method",
        precision_id="precision",
        topology_epoch_id="epoch",
    )
    assert first.checkpoint_id == repeated.checkpoint_id
    assert first.checkpoint_id != changed.checkpoint_id


def test_windowed_time_moments_and_batch_means_restart_exactly(tmp_path):
    plan = StreamingMomentPlan(
        lambda time, state, args: state,
        weighting="time",
        window_start=0.2,
        window_end=0.6,
        batch_duration=0.2,
        maximum_batches=2,
        plan_id="windowed-batches",
    )
    state = plan.initial_state()
    state = plan.update(0.1, jnp.asarray(1.0), state, previous_time=0.0)
    assert state.weight == 0.0
    state = plan.update(0.3, jnp.asarray(2.0), state, previous_time=0.1)
    envelope = RuntimeCheckpointEnvelope(
        jnp.asarray(0.0),
        time=0.3,
        step_index=2,
        schedule_cursor=0,
        mesh_id="mesh",
        method_id="method",
        precision_id="precision",
        topology_epoch_id="epoch",
        observer_states=(state,),
    )
    path = write_runtime_checkpoint(tmp_path / "batch.phx", envelope)
    restored = read_runtime_checkpoint(
        path,
        state_template=jnp.asarray(0.0),
        mesh_id="mesh",
        method_id="method",
        precision_id="precision",
        topology_epoch_id="epoch",
        observer_templates=(plan.initial_state(),),
    ).observer_states[0]
    restored = plan.update(0.5, jnp.asarray(4.0), restored, previous_time=0.3)
    restored = plan.update(0.7, jnp.asarray(6.0), restored, previous_time=0.5)
    np.testing.assert_allclose(restored.weight, 0.4)
    np.testing.assert_allclose(restored.mean, 4.0)
    np.testing.assert_allclose(plan.batch_mean_standard_error(restored), 1.0)


def test_runtime_checkpoint_leafwise_hermitian_encoding_roundtrip(tmp_path):
    space = phx.discretization.TensorSpectralPlan(
        (phx.discretization.FourierBasisPlan(8),),
        axis_names=("x",),
    ).prepare((phx.discretization.AxisDomain.periodic(0.0, 1.0),))
    coordinates = phx.discretization.HermitianSpectralCoordinates(space)
    spectral = space.project(jnp.sin(2.0 * jnp.pi * space.axes[0].nodes))
    state = {"native": jnp.asarray((3.0,)), "spectral": spectral}
    encoding = RuntimeCheckpointEncodingPlan(
        (RuntimeCheckpointLeafBinding(1, coordinates, coordinates.evidence),)
    )
    envelope = RuntimeCheckpointEnvelope(
        state,
        time=0.0,
        step_index=0,
        schedule_cursor=0,
        mesh_id="mesh",
        method_id="method",
        precision_id="precision",
        topology_epoch_id="epoch",
        encoding_plan=encoding,
    )
    path = write_runtime_checkpoint(tmp_path / "coordinates.phx", envelope)
    restored = read_runtime_checkpoint(
        path,
        state_template=state,
        mesh_id="mesh",
        method_id="method",
        precision_id="precision",
        topology_epoch_id="epoch",
        encoding_plan=encoding,
    )
    np.testing.assert_array_equal(restored.state["native"], state["native"])
    np.testing.assert_allclose(restored.state["spectral"], spectral, atol=1e-12)
    assert restored.encoding_plan.encoding_id == encoding.encoding_id
