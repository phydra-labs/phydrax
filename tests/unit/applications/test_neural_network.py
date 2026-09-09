#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications import electrophysiology as ep


jax.config.update("jax_enable_x64", True)


def _lif(*, capacitance=1.0, leak=0.0, threshold=-64.0, refractory=10.0):
    return ep.LeakyIntegrateAndFire(
        capacitance, leak, -65.0, threshold, -65.0, refractory_ms=refractory
    )


def _inputs(runtime, current):
    return eqx.tree_at(
        lambda value: value.injected_current_nA,
        ep.zero_neural_inputs(runtime),
        jnp.asarray(current),
    )


def _assert_same_state(actual, expected):
    for actual_leaf, expected_leaf in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        if jax.dtypes.issubdtype(actual_leaf.dtype, jax.dtypes.prng_key):
            actual_leaf = jax.random.key_data(actual_leaf)
            expected_leaf = jax.random.key_data(expected_leaf)
        np.testing.assert_array_equal(actual_leaf, expected_leaf)


def _single_cable(*mechanisms, dt=0.05):
    diameter = np.sqrt(100_000.0 / np.pi)
    morphology = ep.CellMorphologyPlan(
        "soma", (ep.CompartmentSpec("soma", None, diameter, diameter),)
    ).prepare()
    return ep.CableSolverPlan(dt, residual_tolerance=1.0e-9).prepare(
        morphology, ep.MembraneProgram(mechanisms)
    )


def _source_runtime(
    *,
    queue_capacity=4,
    spike_capacity=4,
    recording_capacity=4,
    maximum_events_per_step=8,
    external_spikes=((0.25, "source"),),
    delay=2.0,
    fanout=1,
):
    cells = (ep.NeuralCellPlan("source", ep.SpikeSource()),) + tuple(
        ep.NeuralCellPlan(f"target-{index}", _lif(threshold=0.0))
        for index in range(fanout)
    )
    connections = tuple(
        ep.SynapseConnection(
            f"edge-{index}",
            0,
            0,
            index + 1,
            0,
            ep.CurrentSynapse(5.0, -1.0),
            weight=0.4,
            delay_ms=delay,
        )
        for index in range(fanout)
    )
    return ep.NeuralNetworkPlan(
        cells,
        ep.SynapseNetworkPlan(
            (1,) * len(cells),
            fanout,
            2.0,
            1.0,
            connections=connections,
            execution="event",
        ),
        queue_capacity=queue_capacity,
        spike_capacity=spike_capacity,
        recording_capacity=recording_capacity,
        maximum_events_per_step=maximum_events_per_step,
        root_subdivisions=1,
        external_spikes=external_spikes,
        learning=ep.PairSTDPPlan(20.0, 25.0, 0.1, 0.05, 0.0, 1.0),
    ).prepare()


def test_endogenous_lif_delay_keeps_emission_weight_during_inflight_learning():
    runtime = ep.NeuralNetworkPlan(
        (ep.NeuralCellPlan("pre", _lif()), ep.NeuralCellPlan("post", _lif())),
        ep.SynapseNetworkPlan(
            (1, 1),
            1,
            0.7,
            1.0,
            execution="event",
            connections=(
                ep.SynapseConnection(
                    "edge",
                    0,
                    0,
                    1,
                    0,
                    ep.CurrentSynapse(5.0, -1.0),
                    weight=0.4,
                    delay_ms=0.7,
                ),
            ),
        ),
        queue_capacity=2,
        spike_capacity=4,
        recording_capacity=2,
        maximum_events_per_step=8,
        root_subdivisions=1,
        learning=ep.PairSTDPPlan(20.0, 25.0, 0.1, 0.05, 0.0, 1.0),
    ).prepare()
    inputs = _inputs(runtime, [2.0, 1.25])
    first = ep.step_neural_network(runtime, ep.initialize_neural_network(runtime), inputs)
    assert bool(first.evidence.successful)
    assert bool(first.evidence.sensitivity_valid)
    assert int(first.state.spikes.count) == 2
    np.testing.assert_allclose(first.state.spikes.time_ms[:2], [0.5, 0.8], atol=1.0e-7)
    np.testing.assert_array_equal(first.state.spikes.endpoint[:2], [0, 1])
    np.testing.assert_allclose(ep.neural_voltage(runtime, first.state), [-65.0, -65.0])
    assert int(first.evidence.delivered_messages) == 0
    np.testing.assert_array_equal(first.state.relations.activation, [0.0])
    learned_weight = 0.4 + 0.1 * np.exp(-0.3 / 20.0)
    np.testing.assert_allclose(
        first.state.relations.weight, [learned_weight], atol=1.0e-8
    )

    second = ep.step_neural_network(runtime, first.state, inputs)
    assert bool(second.evidence.successful)
    assert int(second.evidence.delivered_messages) == 1
    assert int(second.state.queue.size) == 0
    # Emission at 0.5 ms arrives at 1.2 ms; later potentiation cannot rewrite it.
    np.testing.assert_allclose(
        second.state.relations.activation, [0.4 * np.exp(-0.8 / 5.0)], atol=1.0e-8
    )
    np.testing.assert_allclose(
        second.state.relations.weight, [learned_weight], atol=1.0e-8
    )


def test_heterogeneous_point_and_cable_cells_preserve_endpoint_charge_and_parameters():
    morphology = ep.CellMorphologyPlan(
        "two-compartment",
        (
            ep.CompartmentSpec("soma", None, 20.0, 20.0),
            ep.CompartmentSpec("dendrite", "soma", 50.0, 2.0),
        ),
    ).prepare()
    cable = ep.CableSolverPlan(0.1, residual_tolerance=1.0e-9).prepare(
        morphology, ep.MembraneProgram((ep.PassiveLeak(0.0, -65.0),))
    )
    runtime = ep.NeuralNetworkPlan(
        (
            ep.NeuralCellPlan("left", _lif(leak=0.1, threshold=0.0)),
            ep.NeuralCellPlan("cable", cable),
            ep.NeuralCellPlan("right", _lif(capacitance=0.5, leak=0.2, threshold=0.0)),
        ),
        ep.SynapseNetworkPlan((1, 2, 1), 1, 0.0, 0.1, execution="event"),
        queue_capacity=1,
        spike_capacity=1,
        recording_capacity=1,
        maximum_events_per_step=2,
        root_subdivisions=1,
    ).prepare()
    result = ep.step_neural_network(
        runtime,
        ep.initialize_neural_network(runtime),
        _inputs(runtime, [2.0, 0.0, 0.001, 3.0]),
    )
    assert bool(result.evidence.successful)
    voltage = ep.neural_voltage(runtime, result.state)
    np.testing.assert_allclose(
        voltage[jnp.asarray([0, 3])],
        [-65.0 + 20.0 * -np.expm1(-0.01), -65.0 + 15.0 * -np.expm1(-0.04)],
        atol=1.0e-11,
    )
    assert -65.0 < float(voltage[1]) < float(voltage[2])
    np.testing.assert_allclose(
        jnp.sum(morphology.capacitance_nF * (voltage[1:3] + 65.0)),
        0.001 * 0.1,
        atol=1.0e-13,
    )
    assert int(result.state.spikes.count) == 0


def test_hh_spike_observation_never_resets_or_truncates_the_action_potential():
    cable = _single_cable(ep.HodgkinHuxleyNaK())
    runtime = ep.NeuralNetworkPlan(
        (ep.NeuralCellPlan("hh", cable, threshold_mV=0.0, rearm_mV=-40.0),),
        ep.SynapseNetworkPlan((1,), 1, 0.0, 0.05, execution="clock"),
        queue_capacity=1,
        spike_capacity=4,
        recording_capacity=160,
        root_subdivisions=1,
        current_clamps=(("hh", ep.CurrentClamp("pulse", "soma", 10.0, 0.0, 4.0)),),
    ).prepare()
    result = ep.run_neural_network(runtime, ep.initialize_neural_network(runtime), 160)
    np.testing.assert_array_equal(result.status, np.zeros(160, dtype=np.int32))
    assert int(result.state.spikes.count) == 1
    spike_time = float(result.state.spikes.time_ms[0])
    peak_index = int(jnp.argmax(result.voltage_mV[:, 0]))
    assert float(result.state.recording.time_ms[peak_index]) > spike_time
    assert float(result.voltage_mV[peak_index, 0]) > 20.0
    assert float(result.voltage_mV[-1, 0]) < -40.0


def test_voltage_recording_overflow_rolls_back_already_pending_transport():
    runtime = _source_runtime(
        recording_capacity=1, external_spikes=((0.25, "source"), (1.25, "source"))
    )
    first = ep.step_neural_network(runtime, ep.initialize_neural_network(runtime))
    assert bool(first.evidence.successful)
    assert int(first.state.queue.size) == 1
    rejected = ep.step_neural_network(runtime, first.state)
    assert int(rejected.evidence.status) & int(ep.NeuralStatus.RECORDING_CAPACITY)
    assert not bool(rejected.evidence.successful)
    _assert_same_state(rejected.state, first.state)


def test_queue_overflow_cannot_partially_deliver_a_source_fanout():
    runtime = _source_runtime(queue_capacity=1, fanout=2)
    state = ep.initialize_neural_network(runtime)
    rejected = ep.step_neural_network(runtime, state)
    assert int(rejected.evidence.status) & int(ep.NeuralStatus.EVENT_CAPACITY)
    assert not bool(rejected.evidence.successful)
    _assert_same_state(rejected.state, state)


def test_spike_recording_overflow_cannot_commit_same_time_emissions():
    runtime = _source_runtime(
        spike_capacity=1, external_spikes=((0.25, "source"), (0.25, "source"))
    )
    state = ep.initialize_neural_network(runtime)
    rejected = ep.step_neural_network(runtime, state)
    assert int(rejected.evidence.status) & int(ep.NeuralStatus.RECORDING_CAPACITY)
    assert not bool(rejected.evidence.successful)
    _assert_same_state(rejected.state, state)


def test_event_work_exhaustion_rolls_back_delivery_and_learning_together():
    runtime = _source_runtime(maximum_events_per_step=1, delay=0.0)
    state = ep.initialize_neural_network(runtime)
    rejected = ep.step_neural_network(runtime, state)
    assert int(rejected.evidence.status) & int(ep.NeuralStatus.EVENT_WORK_EXHAUSTED)
    assert not bool(rejected.evidence.successful)
    _assert_same_state(rejected.state, state)


def test_relation_deletion_and_slot_reuse_cancel_old_arrival_and_rebuild_source_route():
    runtime = ep.NeuralNetworkPlan(
        (
            ep.NeuralCellPlan("old", ep.SpikeSource()),
            ep.NeuralCellPlan("new", ep.SpikeSource()),
            ep.NeuralCellPlan("target", _lif(threshold=0.0)),
        ),
        ep.SynapseNetworkPlan(
            (1, 1, 1),
            1,
            2.0,
            1.0,
            execution="event",
            connections=(
                ep.SynapseConnection(
                    "edge",
                    0,
                    0,
                    2,
                    0,
                    ep.CurrentSynapse(5.0, -1.0),
                    weight=0.4,
                    delay_ms=2.0,
                ),
            ),
        ),
        external_spikes=((0.25, "old"), (1.25, "new")),
        learning=ep.PairSTDPPlan(20.0, 25.0, 0.1, 0.05, 0.0, 1.0),
        queue_capacity=2,
        spike_capacity=2,
        recording_capacity=3,
        maximum_events_per_step=8,
        root_subdivisions=1,
    ).prepare()
    emitted = ep.step_neural_network(runtime, ep.initialize_neural_network(runtime))
    assert bool(emitted.evidence.successful)
    assert int(emitted.state.queue.size) == 1
    assert float(emitted.state.learning.pre_trace[0]) > 0.0
    deletion = ep.SynapseRelationEvent(
        int(ep.SynapseRelationEventKind.DEACTIVATE),
        0,
        0,
        0,
        2,
        0,
        int(ep.SynapseKind.CURRENT),
        0.4,
        -1.0,
        0.0,
        5.0,
        2.0,
    )
    deleted = ep.apply_neural_relation_event(runtime, emitted.state, deletion)
    assert bool(deleted.evidence.successful)
    assert int(deleted.state.queue.size) == 0
    np.testing.assert_array_equal(deleted.state.learning.pre_trace, [0.0])
    np.testing.assert_array_equal(deleted.state.learning.post_trace, [0.0])
    activation = ep.SynapseRelationEvent(
        int(ep.SynapseRelationEventKind.ACTIVATE),
        0,
        1,
        0,
        2,
        0,
        int(ep.SynapseKind.CURRENT),
        0.7,
        -1.0,
        0.0,
        5.0,
        0.5,
    )
    reused = ep.apply_neural_relation_event(runtime, deleted.state, activation)
    assert bool(reused.evidence.successful)
    arrived = ep.step_neural_network(runtime, reused.state)
    assert bool(arrived.evidence.successful)
    assert int(arrived.evidence.delivered_messages) == 1
    np.testing.assert_allclose(
        arrived.state.relations.activation, [0.7 * np.exp(-0.25 / 5.0)], atol=1.0e-10
    )
    after_old_deadline = ep.step_neural_network(runtime, arrived.state)
    assert bool(after_old_deadline.evidence.successful)
    assert int(after_old_deadline.evidence.delivered_messages) == 0
    np.testing.assert_allclose(
        after_old_deadline.state.relations.activation,
        [0.7 * np.exp(-1.25 / 5.0)],
        atol=1.0e-10,
    )


def test_same_time_zero_delay_events_and_boundary_checkpoint_continue_exactly_once():
    runtime = ep.NeuralNetworkPlan(
        (
            ep.NeuralCellPlan("a", ep.SpikeSource()),
            ep.NeuralCellPlan("b", ep.SpikeSource()),
            ep.NeuralCellPlan("target", _lif(threshold=0.0)),
        ),
        ep.SynapseNetworkPlan(
            (1, 1, 1),
            2,
            0.0,
            1.0,
            execution="event",
            connections=tuple(
                ep.SynapseConnection(
                    name,
                    index,
                    0,
                    2,
                    0,
                    ep.CurrentSynapse(5.0, -1.0),
                    weight=weight,
                    delay_ms=0.0,
                )
                for index, (name, weight) in enumerate((("a-edge", 0.4), ("b-edge", 0.7)))
            ),
        ),
        external_spikes=((0.0, "a"), (1.0, "b"), (1.0, "a"), (2.0, "b")),
        queue_capacity=2,
        spike_capacity=4,
        recording_capacity=2,
        maximum_events_per_step=4,
        root_subdivisions=1,
    ).prepare()
    initial = ep.initialize_neural_network(runtime)
    whole = ep.run_neural_network(runtime, initial, 2)
    first = ep.run_neural_network(runtime, initial, 1)
    checkpoint = ep.checkpoint_neural_network(runtime, first.state)
    resumed = ep.run_neural_network(
        runtime, ep.restore_neural_network(runtime, checkpoint), 1
    )
    np.testing.assert_array_equal(whole.status, [0, 0])
    np.testing.assert_array_equal(first.status, [0])
    np.testing.assert_array_equal(resumed.status, [0])
    _assert_same_state(resumed.state, whole.state)
    np.testing.assert_array_equal(first.state.spikes.endpoint[:3], [0, 0, 1])
    np.testing.assert_allclose(first.state.spikes.time_ms[:3], [0.0, 1.0, 1.0])
    assert int(whole.state.spikes.count) == 4
    np.testing.assert_allclose(whole.state.spikes.time_ms, [0.0, 1.0, 1.0, 2.0])
    np.testing.assert_array_equal(whole.delivered_messages, [3, 1])
    expected_charge = 0.4 * 5.0 * -np.expm1(-2.0 / 5.0)
    expected_charge += (0.4 + 0.7) * 5.0 * -np.expm1(-1.0 / 5.0)
    np.testing.assert_allclose(
        ep.neural_voltage(runtime, whole.state)[2], -65.0 + expected_charge, atol=1.0e-11
    )


def test_isolated_physical_lif_numerical_root_has_conditional_current_gradient():
    runtime = ep.NeuralNetworkPlan(
        (ep.NeuralCellPlan("lif", _lif(leak=0.2)),),
        ep.SynapseNetworkPlan((1,), 1, 0.0, 1.0, execution="event"),
        queue_capacity=1,
        spike_capacity=1,
        recording_capacity=1,
        maximum_events_per_step=4,
        root_subdivisions=1,
        root_tolerance_ms=1.0e-10,
    ).prepare()
    state = ep.initialize_neural_network(runtime)

    def first_spike_time(current):
        result = ep.step_neural_network(runtime, state, _inputs(runtime, [current]))
        return result.state.spikes.time_ms[0]

    result = ep.step_neural_network(runtime, state, _inputs(runtime, [2.0]))
    assert bool(result.evidence.successful)
    assert bool(result.evidence.sensitivity_valid)
    assert int(result.state.spikes.count) == 1
    expected_time = -np.log1p(-0.2 / 2.0) / 0.2
    # dt/dI = -C * (threshold - rest) / (I * (I - g * (threshold - rest))).
    expected_gradient = -1.0 / (2.0 * (2.0 - 0.2))
    np.testing.assert_allclose(result.state.spikes.time_ms[0], expected_time, atol=1.0e-9)
    np.testing.assert_allclose(
        jax.grad(first_spike_time)(jnp.asarray(2.0)), expected_gradient, atol=1.0e-8
    )


def test_clock_initial_time_must_share_the_prepared_grid():
    runtime = ep.NeuralNetworkPlan(
        (ep.NeuralCellPlan("cell", _lif(threshold=0.0)),),
        ep.SynapseNetworkPlan((1,), 1, 0.0, 1.0, execution="clock"),
        queue_capacity=1,
        spike_capacity=1,
        recording_capacity=1,
    ).prepare()
    with pytest.raises(ValueError, match="grid-aligned"):
        ep.initialize_neural_network(runtime, time_ms=0.25)


def test_checkpoint_identity_binds_fixed_capacities():
    first = _source_runtime(recording_capacity=2)
    second = _source_runtime(recording_capacity=3)
    assert first.runtime_id != second.runtime_id
    checkpoint = ep.checkpoint_neural_network(first, ep.initialize_neural_network(first))
    with pytest.raises(ValueError, match="runtime"):
        ep.restore_neural_network(second, checkpoint)


def test_failed_ion_coupling_rolls_back_channel_draw_keys_and_membrane_state():
    cable = _single_cable(ep.PassiveLeak(0.3, -65.0), dt=1.0)
    ions = ep.IonDynamicsPlan((ep.IonSpecies("Na", 1),), (1.0,), (1.0,)).prepare()
    channels = ep.MarkovChannelPlan([[-0.2, 0.2], [0.1, -0.1]], 1).prepare(0.5)

    def outward_leak_current(old_cell, new_cell):
        del old_cell
        return 0.3 * (new_cell.voltage_mV[None, :] + 65.0)

    runtime = ep.NeuralNetworkPlan(
        (ep.NeuralCellPlan("cell", cable, threshold_mV=0.0),),
        ep.SynapseNetworkPlan((1,), 1, 0.0, 1.0, execution="event"),
        ion_couplings=(
            ep.NeuralIonCoupling(
                "cell",
                ions,
                outward_leak_current,
                coupling_id="outward-leak-current-v1",
            ),
        ),
        channel_couplings=(ep.NeuralChannelCoupling("cell", channels, 1, 0.001, -65.0),),
        queue_capacity=1,
        spike_capacity=1,
        recording_capacity=1,
        maximum_events_per_step=4,
        root_subdivisions=1,
    ).prepare()

    def initialize(inside):
        return ep.initialize_neural_network(
            runtime,
            jnp.asarray([-40.0]),
            intracellular_mM=(jnp.asarray([[inside]]),),
            extracellular_mM=(jnp.asarray([[145.0]]),),
            channel_counts=(jnp.asarray([[80, 20]], dtype=jnp.int32),),
            key=jax.random.key(91),
        )

    depleted = initialize(1.0e-6)
    rejected = ep.step_neural_network(runtime, depleted)
    assert not bool(rejected.evidence.successful)
    _assert_same_state(rejected.state, depleted)
    healthy = initialize(10.0)
    accepted = ep.step_neural_network(runtime, healthy)
    assert bool(accepted.evidence.successful)
    assert int(accepted.state.channels[0].step_index) == 2
    assert not np.array_equal(
        jax.random.key_data(accepted.state.channels[0].key),
        jax.random.key_data(healthy.channels[0].key),
    )
    assert float(accepted.state.ions[0].intracellular_mM[0, 0]) < 10.0
