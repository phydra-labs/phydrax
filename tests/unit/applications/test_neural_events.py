#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import heapq

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from phydrax.applications.electrophysiology._events import (
    build_source_fanout,
    cancel_neural_events,
    enqueue_neural_event,
    initialize_event_queue,
    peek_neural_event_time,
    pop_neural_event,
)
from phydrax.applications.electrophysiology._synapses import (
    CurrentSynapse,
    initialize_synapse_network,
    SynapseConnection,
    SynapseNetworkPlan,
)


jax.config.update("jax_enable_x64", True)


_enqueue = jax.jit(enqueue_neural_event)
_pop = jax.jit(pop_neural_event)


def _assert_unchanged(before, after):
    for original, result in zip(
        jax.tree.leaves(before), jax.tree.leaves(after), strict=True
    ):
        np.testing.assert_array_equal(original, result)


def test_heap_same_time_slot_order_and_same_relation_emission_order():
    queue = initialize_event_queue(6)
    events = [
        (2.0, 5, 10.0),
        (1.0, 4, 20.0),
        (1.0, 2, 30.0),
        (1.0, 4, 40.0),
        (0.0, 9, 50.0),
        (1.0, 2, -60.0),
    ]
    for time, slot, amplitude in events:
        queue, accepted = _enqueue(queue, time, slot, 7, amplitude)
        assert bool(accepted)
    delivered = []
    for _ in events:
        queue, time, slot, generation, amplitude, count, accepted = _pop(queue)
        assert bool(accepted)
        delivered.append(
            (float(time), int(slot), int(generation), float(amplitude), int(count))
        )
    assert delivered == [
        (0.0, 9, 7, 50.0, 1),
        (1.0, 2, 7, 30.0, 1),
        (1.0, 2, 7, -60.0, 1),
        (1.0, 4, 7, 20.0, 1),
        (1.0, 4, 7, 40.0, 1),
        (2.0, 5, 7, 10.0, 1),
    ]
    assert np.isposinf(float(peek_neural_event_time(queue)))
    empty, time, slot, generation, amplitude, count, accepted = _pop(queue)
    _assert_unchanged(queue, empty)
    assert not bool(accepted)
    assert np.isposinf(float(time))
    assert (int(slot), int(generation), float(amplitude), int(count)) == (-1, -1, 0.0, 0)


def test_heap_interleaved_push_pop_matches_independent_priority_queue():
    queue = initialize_event_queue(13)
    reference = []
    random = np.random.default_rng(941)
    sequence = 0
    for _ in range(70):
        if reference and (len(reference) == 13 or random.random() < 0.4):
            expected_time, expected_slot, _, expected_amplitude = heapq.heappop(reference)
            queue, time, slot, _, amplitude, _, accepted = _pop(queue)
            assert bool(accepted)
            assert (float(time), int(slot), float(amplitude)) == (
                expected_time,
                expected_slot,
                expected_amplitude,
            )
        else:
            time = float(random.integers(0, 5))
            slot = int(random.integers(0, 20))
            amplitude = float(sequence - 35)
            queue, accepted = _enqueue(queue, time, slot, 0, amplitude)
            assert bool(accepted)
            heapq.heappush(reference, (time, slot, sequence, amplitude))
            sequence += 1
        if reference:
            assert float(peek_neural_event_time(queue)) == reference[0][0]
    while reference:
        expected_time, expected_slot, _, expected_amplitude = heapq.heappop(reference)
        queue, time, slot, _, amplitude, _, accepted = _pop(queue)
        assert bool(accepted)
        assert (float(time), int(slot), float(amplitude)) == (
            expected_time,
            expected_slot,
            expected_amplitude,
        )


def test_capacity_rejection_preserves_all_events_and_emission_counter():
    queue = initialize_event_queue(2)
    queue, _ = _enqueue(queue, 2.0, 1, 0, 0.5)
    queue, _ = _enqueue(queue, 1.0, 2, 0, 0.25)
    rejected, accepted = _enqueue(queue, 0.0, 0, 0, 100.0)
    assert not bool(accepted)
    _assert_unchanged(queue, rejected)
    queue, time, slot, _, _, _, _ = _pop(rejected)
    assert (float(time), int(slot)) == (1.0, 2)
    queue, accepted = _enqueue(queue, 1.5, 3, 0, 0.0, count=4)
    assert bool(accepted)
    _, time, slot, _, amplitude, count, _ = _pop(queue)
    assert (float(time), int(slot), float(amplitude), int(count)) == (1.5, 3, 0.0, 4)
    empty = initialize_event_queue(0)
    rejected, accepted = _enqueue(empty, 0.0, 0, 0, 1.0)
    assert not bool(accepted)
    _assert_unchanged(empty, rejected)
    popped = _pop(empty)
    assert not bool(popped[-1])
    _assert_unchanged(empty, popped[0])


def test_invalid_events_cannot_consume_capacity_or_overflow_identifiers():
    queue, _ = _enqueue(initialize_event_queue(3), 2.0, 0, 1, 0.25)
    invalid = [
        (np.nan, 0, 1, 1.0, 1),
        (1.0, 0, 1, np.inf, 1),
        (-1.0, 0, 1, 1.0, 1),
        (1.0, -1, 1, 1.0, 1),
        (1.0, 0.5, 1, 1.0, 1),
        (1.0, 0, -1, 1.0, 1),
        (1.0, 0, 1, jnp.asarray([1.0, 2.0]), 1),
        (1.0, jnp.asarray(2**32, dtype=jnp.int64), 1, 1.0, 1),
        (1.0, 0, jnp.asarray(2**63, dtype=jnp.uint64), 1.0, 1),
        (1.0, 0, 1, 1.0, 0),
    ]
    for event in invalid:
        rejected, accepted = _enqueue(queue, *event)
        assert not bool(accepted)
        _assert_unchanged(queue, rejected)
    queue, accepted = _enqueue(queue, 1.0, jnp.asarray(2, dtype=jnp.int8), 1, -0.5)
    assert bool(accepted)
    _, time, slot, _, amplitude, _, _ = _pop(queue)
    assert (float(time), int(slot), float(amplitude)) == (1.0, 2, -0.5)


def test_generation_cancellation_compacts_and_preserves_surviving_event_order():
    queue = initialize_event_queue(8)
    events = [
        (0.0, 0, 0, 10.0, 1),
        (4.0, 2, 3, 20.0, 1),
        (2.0, 0, 1, 0.0, 3),
        (5.0, 1, 2, 30.0, 1),
        (6.0, 0, 1, 50.0, 1),
        (3.0, 2, 3, 70.0, 2),
        (7.0, 2, 2, 60.0, 1),
    ]
    for event in events:
        queue, accepted = _enqueue(queue, *event)
        assert bool(accepted)
    compacted = jax.jit(cancel_neural_events)(
        queue, jnp.asarray([True, False, True]), jnp.asarray([1, 2, 3])
    )
    compacted, accepted = _enqueue(compacted, 2.0, 0, 1, 80.0)
    assert bool(accepted)
    deliveries = []
    for _ in range(5):
        compacted, time, slot, generation, amplitude, count, accepted = _pop(compacted)
        assert bool(accepted)
        deliveries.append(
            (float(time), int(slot), int(generation), float(amplitude), int(count))
        )
    assert deliveries == [
        (2.0, 0, 1, 0.0, 3),
        (2.0, 0, 1, 80.0, 1),
        (3.0, 2, 3, 70.0, 2),
        (4.0, 2, 3, 20.0, 1),
        (6.0, 0, 1, 50.0, 1),
    ]
    assert np.isposinf(float(peek_neural_event_time(compacted)))


def test_source_fanout_indexes_only_active_relations_with_stable_row_order():
    synapse = CurrentSynapse(5.0, -0.1)
    prepared = SynapseNetworkPlan(
        (2, 1),
        5,
        3.0,
        0.1,
        connections=(
            SynapseConnection("last-a", 1, 0, 0, 0, synapse),
            SynapseConnection("first", 0, 0, 1, 0, synapse),
            SynapseConnection("last-b", 1, 0, 0, 1, synapse),
            SynapseConnection("middle", 0, 1, 1, 0, synapse),
        ),
        execution="event",
    ).prepare()
    relations = initialize_synapse_network(prepared).relations
    initial = jax.jit(build_source_fanout, static_argnums=(1,))(relations, 3)
    np.testing.assert_array_equal(initial.offsets, [0, 1, 2, 4])
    np.testing.assert_array_equal(
        initial.slots[: int(initial.active_count)], [1, 3, 0, 2]
    )
    changed = eqx.tree_at(
        lambda value: (value.active, value.pre_endpoint),
        relations,
        (jnp.asarray([False, True, True, True, True]), jnp.asarray([2, 0, 2, 1, 0])),
    )
    rebuilt = jax.jit(build_source_fanout, static_argnums=(1,))(changed, 3)
    np.testing.assert_array_equal(rebuilt.offsets, [0, 2, 3, 4])
    np.testing.assert_array_equal(
        rebuilt.slots[: int(rebuilt.active_count)], [1, 4, 3, 2]
    )
