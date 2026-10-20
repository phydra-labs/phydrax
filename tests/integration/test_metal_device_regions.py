"""Bounded ordered maps/recurrences and their general-path admission boundary.

The existing streamed-relation forward workload's 64-event coordinate/weight
maps use the same dynamic-slice/update lowering as ``gather_routes`` below.
Profiler evidence must additionally show ``phydrax_bounded_ordered_region_*``;
semantic equality alone does not establish admission or a performance claim.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax.experimental import io_callback

from phydrax import sparse
from phydrax._jax_config import default_device
from tests._support.metal_qualification import metal_device, record_metal_execution


pytestmark = pytest.mark.metal_native

type RecordProperty = Callable[[str, object], None]


@pytest.fixture
def device() -> Iterator[jax.Device]:
    selected = metal_device()
    with default_device(selected):
        yield selected


@pytest.mark.parametrize("dtype", [np.float32, np.int64, np.uint64])
def test_existing_sparse_ordered_gather_replays_dynamic_values(
    device: jax.Device, record_property: RecordProperty, dtype: type[np.generic]
) -> None:
    # Duplicate indices, invalid routes and signed zero belong to the actual
    # sparse API, not a synthetic empty loop. Same executable, different inputs.
    indices = np.asarray([3, 1, 3, 0, 2, 1, 0, 3] * 8, np.int32)
    valid = np.ones(64, np.bool_)
    valid[::7] = False
    relation = sparse.EdgeRelation(
        jax.device_put(indices, device),
        jnp.arange(64, dtype=jnp.int32),
        source_size=4,
        target_size=64,
        valid=jax.device_put(valid, device),
    )
    program = jax.jit(
        lambda x: sparse.gather_routes(relation, x, accumulation="deterministic")
    )
    if dtype == np.float32:
        host = np.asarray([[0.0, -0.0, np.nan], [1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype)
    elif dtype == np.int64:
        host = np.asarray(
            [[2**60 + 10 * row + k for k in range(3)] for row in range(4)], dtype
        )
    else:
        host = np.asarray(
            [[2**63 + 10 * row + k for k in range(3)] for row in range(4)], dtype
        )
    results = []
    for current in (host, host[::-1].copy()):
        result = program(jax.device_put(current, device))
        expected = np.where(valid[:, None], current[indices], np.zeros((), dtype))
        actual = np.asarray(result)
        assert actual.dtype == dtype and actual.shape == (64, 3)
        if dtype == np.float32:
            np.testing.assert_array_equal(
                actual.view(np.uint32), expected.view(np.uint32)
            )
        else:
            np.testing.assert_array_equal(actual, expected)
        results.append(result)
    record_metal_execution(record_property, *results)


@pytest.mark.parametrize("steps", [0, 1, 17])
def test_ordered_recurrence_preserves_event_rounding_and_selected_branch(
    device: jax.Device, record_property: RecordProperty, steps: int
) -> None:
    increments = np.asarray([1e8, 1, -1e8, 3, -0.0] * 4, np.float32)
    enabled = np.asarray([True, False, True, True, False] * 4)

    @jax.jit
    def program(values: jax.Array, mask: jax.Array, start: jax.Array) -> jax.Array:
        def step(index: jax.Array, total: jax.Array) -> jax.Array:
            return lax.cond(mask[index], lambda x: x + values[index], lambda x: x, total)

        return lax.fori_loop(0, steps, step, start)

    initial = np.asarray(-0.0, np.float32)
    expected = initial
    for index in range(steps):
        if enabled[index]:
            expected = np.asarray(expected + increments[index], np.float32)
    result = program(
        jax.device_put(increments, device),
        jax.device_put(enabled, device),
        jax.device_put(initial, device),
    )
    np.testing.assert_array_equal(
        np.asarray(result).view(np.uint32), expected.view(np.uint32)
    )
    record_metal_execution(record_property, result)


@pytest.mark.parametrize("index_dtype", [np.int64, np.uint64])
def test_indexed_device_map_clamps_full_width_starts(
    device: jax.Device, record_property: RecordProperty, index_dtype: type[np.generic]
) -> None:
    starts = np.asarray(
        [-(2**63), -1, 0, 2, 2**63 - 1]
        if index_dtype == np.int64
        else [0, 1, 2, 2**63, 2**64 - 1],
        index_dtype,
    )
    host = np.asarray([-(2**60) + 3, 2**59 + 1, 7, 11], np.int64)

    @jax.jit
    def program(source: jax.Array, indices: jax.Array) -> jax.Array:
        return lax.map(
            lambda index: lax.dynamic_slice(
                source, (index,), (2,), allow_negative_indices=False
            ),
            indices,
        )

    result = program(jax.device_put(host, device), jax.device_put(starts, device))
    expected = np.stack(
        [host[max(0, min(int(i), 2)) : max(0, min(int(i), 2)) + 2] for i in starts]
    )
    np.testing.assert_array_equal(np.asarray(result), expected)
    record_metal_execution(record_property, result)


@pytest.mark.parametrize("ordered", [False, True])
def test_callbacks_in_selected_branches_keep_general_loop_contract(
    device: jax.Device, record_property: RecordProperty, ordered: bool
) -> None:
    calls: list[int] = []
    signature = jax.ShapeDtypeStruct((), np.int64)

    def callback(value: np.ndarray) -> np.ndarray:
        calls.append(int(value))
        return np.asarray(value + np.int64(7), np.int64)

    @jax.jit
    def program(start: jax.Array, selected: jax.Array) -> jax.Array:
        def step(index: jax.Array, total: jax.Array) -> jax.Array:
            def branch(value: jax.Array) -> jax.Array:
                if ordered:
                    return io_callback(callback, signature, value, ordered=True)
                return jax.pure_callback(callback, signature, value)

            return lax.cond(selected, branch, lambda value: value + np.int64(1), total)

        return lax.fori_loop(0, 4, step, start)

    results = []
    for selected in (False, True):
        calls.clear()
        result = jax.block_until_ready(
            program(
                jax.device_put(np.int64(2), device),
                jax.device_put(np.bool_(selected), device),
            )
        )
        assert int(np.asarray(result)) == (30 if selected else 6)
        assert calls == ([2, 9, 16, 23] if selected else [])
        results.append(result)
    record_metal_execution(record_property, *results)


def test_dynamic_bounds_remain_correct_general_gpu_loop(
    device: jax.Device, record_property: RecordProperty
) -> None:
    @jax.jit
    def program(limit: jax.Array, initial: jax.Array) -> jax.Array:
        return lax.fori_loop(0, limit, lambda i, carry: carry + np.int64(3), initial)

    results = []
    for limit in (0, 3, 9):
        result = program(
            jax.device_put(np.int64(limit), device),
            jax.device_put(np.int64(2**55 + 1), device),
        )
        assert int(np.asarray(result)) == 2**55 + 1 + 3 * limit
        results.append(result)
    record_metal_execution(record_property, *results)


def test_device_region_reads_strided_inputs_and_parallel_carried_assignments(
    device: jax.Device, record_property: RecordProperty
) -> None:
    host = np.arange(48, dtype=np.float32).reshape(3, 16)
    indices = np.asarray([15, 1, 8, 1, 0, 4, 2, 7], np.int32)

    @jax.jit
    def mapped(source: jax.Array, indices: jax.Array) -> jax.Array:
        transposed = source.T
        return lax.map(lambda index: transposed[index], indices)

    @jax.jit
    def swapped(a: jax.Array, b: jax.Array) -> tuple[jax.Array, jax.Array]:
        return lax.fori_loop(0, 7, lambda i, carry: (carry[1], carry[0]), (a, b))

    result = mapped(jax.device_put(host, device), jax.device_put(indices, device))
    np.testing.assert_array_equal(np.asarray(result), host.T[indices])
    a, b = swapped(
        jax.device_put(np.int64(2**60 + 1), device),
        jax.device_put(np.int64(-(2**60) + 3), device),
    )
    assert int(np.asarray(a)) == -(2**60) + 3
    assert int(np.asarray(b)) == 2**60 + 1
    record_metal_execution(record_property, result, a, b)
