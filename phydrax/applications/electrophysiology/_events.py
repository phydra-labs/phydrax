#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fixed-capacity deterministic neural-event transport and source indexing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from ..._strict import StrictModule


if TYPE_CHECKING:
    from ._synapses import SynapseRelationState


class NeuralEventQueue(StrictModule):
    """Binary min-heap ordered by ``(time, relation slot, emission sequence)``.

    Only the first ``size`` entries are live. Amplitudes are sampled at emission;
    generations identify relation lifetimes, not versions of mutable weights.
    All operations are pure and preserve the fixed allocated capacity.
    """

    times: Array
    slots: Array
    generations: Array
    amplitudes: Array
    counts: Array
    sequence: Array
    size: Array
    next_sequence: Array


def initialize_event_queue(capacity: int, dtype=None) -> NeuralEventQueue:
    """Allocate an empty heap; zero capacity is valid and always rejects input."""
    if isinstance(capacity, bool) or not isinstance(capacity, int):
        raise TypeError("capacity must be an integer.")
    if capacity < 0 or capacity > jnp.iinfo(jnp.int32).max:
        raise ValueError("capacity must be a nonnegative int32-sized integer.")
    resolved_dtype = jnp.asarray(0.0, dtype=dtype).dtype
    if not jnp.issubdtype(resolved_dtype, jnp.floating):
        raise TypeError("Event times and amplitudes require a floating dtype.")
    sequence_dtype = jnp.int64 if jax.config.jax_enable_x64 else jnp.int32
    return NeuralEventQueue(
        jnp.full((capacity,), jnp.inf, dtype=resolved_dtype),
        jnp.full((capacity,), -1, dtype=jnp.int32),
        jnp.full((capacity,), -1, dtype=sequence_dtype),
        jnp.zeros((capacity,), dtype=resolved_dtype),
        jnp.zeros((capacity,), dtype=jnp.int32),
        jnp.full((capacity,), -1, dtype=sequence_dtype),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=sequence_dtype),
    )


def _key_less(time, slot, sequence, other_time, other_slot, other_sequence):
    return (time < other_time) | (
        (time == other_time)
        & ((slot < other_slot) | ((slot == other_slot) & (sequence < other_sequence)))
    )


def _set_entry(queue, index, time, slot, generation, amplitude, count, sequence):
    return NeuralEventQueue(
        queue.times.at[index].set(time),
        queue.slots.at[index].set(slot),
        queue.generations.at[index].set(generation),
        queue.amplitudes.at[index].set(amplitude),
        queue.counts.at[index].set(count),
        queue.sequence.at[index].set(sequence),
        queue.size,
        queue.next_sequence,
    )


def _copy_entry(queue, destination, source):
    return _set_entry(
        queue,
        destination,
        queue.times[source],
        queue.slots[source],
        queue.generations[source],
        queue.amplitudes[source],
        queue.counts[source],
        queue.sequence[source],
    )


def _sift_down(queue: NeuralEventQueue, root: Array) -> NeuralEventQueue:
    """Move one hole down at most the statically bounded heap height."""
    time = queue.times[root]
    slot = queue.slots[root]
    generation = queue.generations[root]
    amplitude = queue.amplitudes[root]
    count = queue.counts[root]
    sequence = queue.sequence[root]
    capacity = queue.times.shape[0]

    def move(_, carry):
        current, index, moving = carry
        left = 2 * index + 1
        right = left + 1
        safe_left = jnp.minimum(left, capacity - 1)
        safe_right = jnp.minimum(right, capacity - 1)
        right_first = (right < current.size) & _key_less(
            current.times[safe_right],
            current.slots[safe_right],
            current.sequence[safe_right],
            current.times[safe_left],
            current.slots[safe_left],
            current.sequence[safe_left],
        )
        child = jnp.where(right_first, safe_right, safe_left)
        descend = (
            moving
            & (left < current.size)
            & _key_less(
                current.times[child],
                current.slots[child],
                current.sequence[child],
                time,
                slot,
                sequence,
            )
        )
        current = jax.lax.cond(
            descend, lambda q: _copy_entry(q, index, child), lambda q: q, current
        )
        return current, jnp.where(descend, child, index), descend

    queue, index, _ = jax.lax.fori_loop(
        0, capacity.bit_length(), move, (queue, root, jnp.asarray(True))
    )
    return _set_entry(queue, index, time, slot, generation, amplitude, count, sequence)


def enqueue_neural_event(
    queue: NeuralEventQueue, time_ms, slot, generation, amplitude, count=1
) -> tuple[NeuralEventQueue, Array]:
    """Insert in bounded O(log Q) work, or return the entire original queue.

    Invalid means a nonscalar/nonfinite time or amplitude, negative time,
    noninteger/negative/unrepresentable slot or generation, nonpositive or
    noninteger event count, exhausted sequence numbers, or insufficient capacity.
    Slots need not be smaller than queue capacity. Signed/zero amplitudes are
    permitted; unweighted spike counts remain independent of those amplitudes.
    """
    time = jnp.asarray(time_ms)
    value = jnp.asarray(amplitude)
    relation = jnp.asarray(slot)
    lifetime = jnp.asarray(generation)
    event_count = jnp.asarray(count)
    if (
        queue.times.shape[0] == 0
        or any(item.ndim != 0 for item in (time, value, relation, lifetime, event_count))
        or not jnp.issubdtype(relation.dtype, jnp.integer)
        or not jnp.issubdtype(lifetime.dtype, jnp.integer)
        or not jnp.issubdtype(event_count.dtype, jnp.integer)
        or not (
            jnp.issubdtype(time.dtype, jnp.floating)
            or jnp.issubdtype(time.dtype, jnp.integer)
        )
        or not (
            jnp.issubdtype(value.dtype, jnp.floating)
            or jnp.issubdtype(value.dtype, jnp.integer)
        )
    ):
        return queue, jnp.asarray(False)
    time = time.astype(queue.times.dtype)
    value = value.astype(queue.amplitudes.dtype)
    converted_slot = relation.astype(queue.slots.dtype)
    converted_generation = lifetime.astype(queue.generations.dtype)
    converted_count = event_count.astype(queue.counts.dtype)
    accepted = (
        (queue.size < queue.times.shape[0])
        & jnp.isfinite(time)
        & (time >= 0)
        & jnp.isfinite(value)
        & (relation >= 0)
        & (converted_slot >= 0)
        & (converted_slot.astype(relation.dtype) == relation)
        & (lifetime >= 0)
        & (converted_generation >= 0)
        & (converted_generation.astype(lifetime.dtype) == lifetime)
        & (event_count > 0)
        & (converted_count > 0)
        & (converted_count.astype(event_count.dtype) == event_count)
        & (queue.next_sequence < jnp.iinfo(queue.sequence.dtype).max)
    )
    relation = converted_slot
    lifetime = converted_generation
    event_count = converted_count

    def insert(current):
        sequence = current.next_sequence

        def move(_, carry):
            heap, index, moving = carry
            parent = jnp.maximum((index - 1) // 2, 0)
            ascend = (
                moving
                & (index > 0)
                & _key_less(
                    time,
                    relation,
                    sequence,
                    heap.times[parent],
                    heap.slots[parent],
                    heap.sequence[parent],
                )
            )
            heap = jax.lax.cond(
                ascend, lambda q: _copy_entry(q, index, parent), lambda q: q, heap
            )
            return heap, jnp.where(ascend, parent, index), ascend

        current, index, _ = jax.lax.fori_loop(
            0,
            current.times.shape[0].bit_length(),
            move,
            (current, current.size, jnp.asarray(True)),
        )
        current = _set_entry(
            current, index, time, relation, lifetime, value, event_count, sequence
        )
        return NeuralEventQueue(
            current.times,
            current.slots,
            current.generations,
            current.amplitudes,
            current.counts,
            current.sequence,
            current.size + 1,
            sequence + 1,
        )

    return jax.lax.cond(accepted, insert, lambda current: current, queue), accepted


def peek_neural_event_time(queue: NeuralEventQueue) -> Array:
    """Return the earliest event time, or positive infinity for an empty heap."""
    if queue.times.shape[0] == 0:
        return jnp.asarray(jnp.inf, dtype=queue.times.dtype)
    return jnp.where(queue.size > 0, queue.times[0], jnp.inf)


def pop_neural_event(
    queue: NeuralEventQueue,
) -> tuple[NeuralEventQueue, Array, Array, Array, Array, Array, Array]:
    """Remove the minimum in bounded O(log Q) work without reallocating capacity.

    Empty pops return ``(+inf, -1, -1, 0, 0, False)`` and the unchanged queue.
    """
    empty = (
        queue,
        jnp.asarray(jnp.inf, dtype=queue.times.dtype),
        jnp.asarray(-1, dtype=queue.slots.dtype),
        jnp.asarray(-1, dtype=queue.generations.dtype),
        jnp.asarray(0, dtype=queue.amplitudes.dtype),
        jnp.asarray(0, dtype=queue.counts.dtype),
        jnp.asarray(False),
    )
    if queue.times.shape[0] == 0:
        return empty

    def remove(current):
        time, slot, generation, amplitude, count = (
            current.times[0],
            current.slots[0],
            current.generations[0],
            current.amplitudes[0],
            current.counts[0],
        )
        size = current.size - 1
        current = _copy_entry(current, 0, size)
        current = _set_entry(current, size, jnp.inf, -1, -1, 0, 0, -1)
        current = NeuralEventQueue(
            current.times,
            current.slots,
            current.generations,
            current.amplitudes,
            current.counts,
            current.sequence,
            size,
            current.next_sequence,
        )
        current = jax.lax.cond(
            size > 0,
            lambda q: _sift_down(q, jnp.asarray(0, dtype=jnp.int32)),
            lambda q: q,
            current,
        )
        return current, time, slot, generation, amplitude, count, jnp.asarray(True)

    return jax.lax.cond(queue.size > 0, remove, lambda _: empty, queue)


def cancel_neural_events(
    queue: NeuralEventQueue, relations_active: Array, relations_generation: Array
) -> NeuralEventQueue:
    """Compact surviving lifetimes and rebuild once after structural changes.

    Deleted relations and reused slots cannot deliver old events. Surviving
    sequence numbers and the emission counter are retained, so cancellation
    cannot change the ordering of later equal-time/equal-slot deliveries.
    """
    active = jnp.asarray(relations_active, dtype=jnp.bool_)
    generations = jnp.asarray(relations_generation)
    if active.ndim != 1 or generations.shape != active.shape:
        raise ValueError("Relation active and generation arrays must be equal vectors.")
    capacity = queue.times.shape[0]
    if capacity == 0:
        return queue
    if active.shape[0] == 0:
        keep = jnp.zeros((capacity,), dtype=jnp.bool_)
    else:
        valid_slot = (queue.slots >= 0) & (queue.slots < active.shape[0])
        safe_slots = jnp.clip(queue.slots, 0, active.shape[0] - 1)
        keep = (
            (jnp.arange(capacity) < queue.size)
            & valid_slot
            & active[safe_slots]
            & (generations[safe_slots] == queue.generations)
        )
    count = jnp.sum(keep, dtype=queue.size.dtype)
    indices = jnp.nonzero(keep, size=capacity, fill_value=0)[0]
    live = jnp.arange(capacity) < count
    compacted = NeuralEventQueue(
        jnp.where(live, queue.times[indices], jnp.inf),
        jnp.where(live, queue.slots[indices], -1),
        jnp.where(live, queue.generations[indices], -1),
        jnp.where(live, queue.amplitudes[indices], 0),
        jnp.where(live, queue.counts[indices], 0),
        jnp.where(live, queue.sequence[indices], -1),
        count,
        queue.next_sequence,
    )

    def heapify(index, current):
        root = capacity // 2 - 1 - index
        return jax.lax.cond(
            root < current.size // 2,
            lambda q: _sift_down(q, root),
            lambda q: q,
            current,
        )

    return jax.lax.fori_loop(0, capacity // 2, heapify, compacted)


class SourceFanout(StrictModule):
    """CSR source rows containing ascending stable relation-slot indices."""

    offsets: Array
    slots: Array
    active_count: Array


def build_source_fanout(
    relations: SynapseRelationState, endpoint_count: int
) -> SourceFanout:
    """Build deterministic O(endpoints + capacity) CSR after structural changes.

    Emission reads only the source row; no sort or full relation scan is needed
    per spike. Inactive slots occupy an unused suffix marked with -1.
    """
    if isinstance(endpoint_count, bool) or not isinstance(endpoint_count, int):
        raise TypeError("endpoint_count must be an integer.")
    if endpoint_count <= 0:
        raise ValueError("endpoint_count must be positive.")
    source = relations.pre_endpoint
    active = relations.active
    if source.ndim != 1 or source.shape != active.shape:
        raise ValueError("Relation source and active arrays must be equal vectors.")
    active = eqx.error_if(
        active,
        jnp.any(active & ((source < 0) | (source >= endpoint_count))),
        "An active source endpoint is outside endpoint_count.",
    )
    safe_source = jnp.clip(source, 0, endpoint_count - 1)
    counts = (
        jnp.zeros((endpoint_count,), dtype=jnp.int32)
        .at[safe_source]
        .add(active.astype(jnp.int32))
    )
    offsets = jnp.concatenate((jnp.zeros((1,), dtype=jnp.int32), jnp.cumsum(counts)))
    slots = jnp.full(active.shape, -1, dtype=jnp.int32)

    def place(slot, carry):

        def insert(values):
            cursor, result = values
            endpoint = safe_source[slot]
            result = result.at[cursor[endpoint]].set(slot)
            return cursor.at[endpoint].add(jnp.asarray(1, cursor.dtype)), result

        return jax.lax.cond(active[slot], insert, lambda values: values, carry)

    _, slots = jax.lax.fori_loop(0, active.shape[0], place, (offsets[:-1], slots))
    return SourceFanout(offsets, slots, offsets[-1])


__all__ = [
    "NeuralEventQueue",
    "SourceFanout",
    "build_source_fanout",
    "cancel_neural_events",
    "enqueue_neural_event",
    "initialize_event_queue",
    "peek_neural_event_time",
    "pop_neural_event",
]
