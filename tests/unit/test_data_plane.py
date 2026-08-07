#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import sys
import threading
import time

import jax
import jax.numpy as jnp
import pytest

from phydrax._data_plane import (
    BoundedPrefetchIterator,
    IndexEpochPlan,
    StatelessIndexPermutation,
)


def _wait_until(predicate, *, timeout=2.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("condition was not satisfied before timeout")
        time.sleep(0.005)


@pytest.mark.parametrize(
    ("population", "batch_size", "drop_last"),
    [
        (1, 1, False),
        (2, 4, False),
        (7, 3, False),
        (8, 4, False),
        (9, 4, False),
        (17, 5, True),
        (257, 16, False),
    ],
)
@pytest.mark.parametrize("seed", [0, 1, 2**31 - 1])
def test_index_epoch_plan_is_deterministic_bijective_and_directly_resumable(
    population,
    batch_size,
    drop_last,
    seed,
):
    plan = IndexEpochPlan(
        population,
        batch_size,
        True,
        seed,
        4,
        drop_last,
    )
    batches = tuple(plan)
    repeated = tuple(
        IndexEpochPlan(
            population,
            batch_size,
            True,
            seed,
            4,
            drop_last,
        )
    )
    flattened = tuple(index for batch in batches for index in batch)
    expected_count = population // batch_size * batch_size if drop_last else population

    assert batches == repeated
    assert len(flattened) == expected_count
    assert len(set(flattened)) == expected_count
    assert all(0 <= index < population for index in flattened)
    assert all(len(batch) == batch_size for batch in batches[:-1])
    if batches:
        expected_last = batch_size if drop_last else population % batch_size or batch_size
        assert len(batches[-1]) == expected_last
    suffix_start = min(2, plan.batch_count)
    assert tuple(plan.iter_batches(start_batch=suffix_start)) == tuple(
        enumerate(batches[suffix_start:], suffix_start)
    )
    if not drop_last:
        assert sorted(flattened) == list(range(population))


def test_index_epoch_plan_sequential_boundaries_and_validation():
    plan = IndexEpochPlan(5, 2, False, 0, 3, False)

    assert tuple(plan) == ((0, 1), (2, 3), (4,))
    assert tuple(plan.iter_batches(start_batch=3)) == ()
    with pytest.raises(IndexError, match="batch index"):
        plan.batch(-1)
    with pytest.raises(IndexError, match="batch index"):
        plan.batch(3)
    with pytest.raises(ValueError, match="start_batch"):
        tuple(plan.iter_batches(start_batch=-1))
    with pytest.raises(ValueError, match="start_batch"):
        tuple(plan.iter_batches(start_batch=4))

    dropped = IndexEpochPlan(2, 4, False, 0, 0, True)
    assert dropped.batch_count == 0
    assert tuple(dropped) == ()

    invalid = [
        ((0, 1, False, 0, 0, False), "source_size"),
        ((1, 0, False, 0, 0, False), "batch_size"),
        ((1, 1, False, -1, 0, False), "seed"),
        ((1, 1, False, 0, -1, False), "epoch"),
    ]
    for arguments, message in invalid:
        with pytest.raises(ValueError, match=message):
            IndexEpochPlan(*arguments)


def test_epoch_order_matches_jax_mapping_and_preserves_golden_vector():
    permutation = StatelessIndexPermutation(127, 23, 6)
    positions = jnp.arange(127, dtype=jnp.int32)
    compiled = jax.jit(jax.vmap(permutation.jax))(positions)
    host = jnp.asarray([permutation(int(position)) for position in range(127)])
    other_epoch = jnp.asarray(
        [StatelessIndexPermutation(127, 23, 7)(position) for position in range(127)]
    )

    assert jnp.array_equal(compiled, host)
    assert not jnp.array_equal(host, other_epoch)
    assert jnp.array_equal(jnp.sort(compiled), positions)
    assert tuple(StatelessIndexPermutation(17, 23, 6)(index) for index in range(17)) == (
        10,
        4,
        2,
        7,
        15,
        9,
        0,
        8,
        11,
        12,
        14,
        1,
        16,
        6,
        13,
        3,
        5,
    )


def test_epoch_order_and_plan_reject_invalid_bounds_without_population_storage():
    for arguments, message in [
        ((0, 0, 0), "population"),
        ((2**31 + 1, 0, 0), "must not exceed"),
        ((1, -1, 0), "seed"),
        ((1, 0, -1), "epoch"),
    ]:
        with pytest.raises(ValueError, match=message):
            StatelessIndexPermutation(*arguments)

    permutation = StatelessIndexPermutation(3, 0, 0)
    with pytest.raises(IndexError, match="out of range"):
        permutation(-1)
    with pytest.raises(IndexError, match="out of range"):
        permutation(3)

    small = IndexEpochPlan(8, 4, True, 1, 0, False)
    large = IndexEpochPlan(2**30 + 1, 4, True, 1, 0, False)
    assert sys.getsizeof(small) == sys.getsizeof(large)
    assert len(large.batch(1)) == 4


def test_bounded_prefetch_is_lazy_and_capacity_zero_runs_synchronously():
    calls = []
    main_thread = threading.current_thread().name

    def prepare(value):
        calls.append((value, threading.current_thread().name))
        return value * 2

    iterator = BoundedPrefetchIterator(
        range(3),
        prepare,
        capacity=0,
        thread_name="unused-data-plane-thread",
    )

    assert calls == []
    with iterator as prepared:
        assert next(prepared) == 0
        assert calls == [(0, main_thread)]
    assert iterator.closed
    iterator.close()


def test_bounded_prefetch_preserves_order_and_bounded_read_ahead():
    calls = []
    thread_name = "phydrax-data-plane-bounded-test"

    def prepare(value):
        calls.append((value, threading.current_thread().name))
        return value * 10

    iterator = BoundedPrefetchIterator(
        range(6),
        prepare,
        capacity=2,
        thread_name=thread_name,
    )

    assert calls == []
    with iterator as prepared:
        _wait_until(lambda: len(calls) == 2)
        assert [value for value, _ in calls] == [0, 1]
        assert next(prepared) == 0
        _wait_until(lambda: len(calls) == 3)
        assert [value for value, _ in calls] == [0, 1, 2]
        assert tuple(prepared) == (10, 20, 30, 40, 50)

    assert iterator.closed
    assert all(name == thread_name for _, name in calls)
    assert not any(thread.name == thread_name for thread in threading.enumerate())


def test_bounded_prefetch_propagates_preparation_and_input_errors():
    prepare_thread = "phydrax-data-plane-prepare-error"

    def fail_prepare(value):
        if value == 1:
            raise ValueError("preparation failed")
        return value

    failing_prepare = BoundedPrefetchIterator(
        range(3),
        fail_prepare,
        capacity=2,
        thread_name=prepare_thread,
    )
    with pytest.raises(ValueError, match="preparation failed"):
        with failing_prepare as prepared:
            tuple(prepared)
    assert failing_prepare.closed
    assert not any(thread.name == prepare_thread for thread in threading.enumerate())

    input_thread = "phydrax-data-plane-input-error"

    def failing_items():
        yield 4
        raise RuntimeError("input iteration failed")

    failing_input = BoundedPrefetchIterator(
        failing_items(),
        lambda value: value,
        capacity=1,
        thread_name=input_thread,
    )
    with failing_input as prepared:
        assert next(prepared) == 4
        with pytest.raises(RuntimeError, match="input iteration failed"):
            next(prepared)
    assert failing_input.closed
    assert not any(thread.name == input_thread for thread in threading.enumerate())


def test_bounded_prefetch_closes_empty_full_and_synchronous_failure_paths():
    empty_thread = "phydrax-data-plane-empty"
    empty = BoundedPrefetchIterator(
        (),
        lambda value: value,
        capacity=1,
        thread_name=empty_thread,
    )
    with empty as prepared:
        with pytest.raises(StopIteration):
            next(prepared)
    assert empty.closed
    assert not any(thread.name == empty_thread for thread in threading.enumerate())

    full_thread = "phydrax-data-plane-full"
    calls = []
    full = BoundedPrefetchIterator(
        range(10),
        lambda value: calls.append(value) or value,
        capacity=1,
        thread_name=full_thread,
    )
    full.__enter__()
    _wait_until(lambda: calls == [0])
    full.close()
    full.close()
    assert full.closed
    assert calls == [0]
    assert not any(thread.name == full_thread for thread in threading.enumerate())

    synchronous = BoundedPrefetchIterator(
        (1,),
        lambda value: (_ for _ in ()).throw(ValueError(f"bad value {value}")),
        capacity=0,
        thread_name="unused-synchronous-error",
    )
    with pytest.raises(ValueError, match="bad value 1"):
        next(synchronous)
    assert synchronous.closed

    with pytest.raises(ValueError, match="capacity"):
        BoundedPrefetchIterator((), lambda value: value, capacity=-1, thread_name="x")
    with pytest.raises(ValueError, match="thread_name"):
        BoundedPrefetchIterator((), lambda value: value, capacity=0, thread_name=" ")
