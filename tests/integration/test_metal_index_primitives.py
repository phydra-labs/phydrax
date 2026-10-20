#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Real-device bounds, duplicate-update, and ordering contracts for Metal.

Run in an MPS-only, x64-enabled process. References are host NumPy operations,
closed-form slices, or an independent JAX CPU process, never another GPU index
lowering. Shapes are deliberately small and fixed; full-width values are not.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

from phydrax._jax_config import default_device
from tests._support.metal_process import json_result, run_python
from tests._support.metal_qualification import metal_device, record_metal_execution


pytestmark = pytest.mark.metal_native

type RecordProperty = Callable[[str, object], None]


@pytest.fixture
def device() -> Iterator[jax.Device]:
    selected = metal_device()
    with default_device(selected):
        yield selected


def _put(value: np.ndarray, device: jax.Device) -> jax.Array:
    result = jax.device_put(value, device)
    assert result.dtype == value.dtype
    return result


def _extreme_indices(index_kind: str) -> np.ndarray:
    match index_kind:
        case "int64":
            return np.asarray(
                [-(2**63), -(2**32) - 1, -1, 0, 1, 3, 4, 2**32 + 1, 2**63 - 1],
                np.int64,
            )
        case "uint64":
            return np.asarray([0, 1, 3, 4, 2**31, 2**32, 2**63, 2**64 - 1], np.uint64)
        case _:
            raise AssertionError(index_kind)


@pytest.mark.parametrize("index_kind", ["int64", "uint64"])
def test_metal_gather_clamps_full_width_indices_before_narrowing(
    device: jax.Device, record_property: RecordProperty, index_kind: str
) -> None:
    operand = np.asarray([101, -(2**55) + 3, 2**54 + 1, 7, 11], np.int64)
    indices = _extreme_indices(index_kind)
    dimensions = lax.GatherDimensionNumbers(
        offset_dims=(1,), collapsed_slice_dims=(), start_index_map=(0,)
    )

    @jax.jit
    def program(values: jax.Array, starts: jax.Array) -> jax.Array:
        return lax.gather(
            values, starts[:, None], dimensions, slice_sizes=(2,), mode="clip"
        )

    result = program(_put(operand, device), _put(indices, device))
    # A clipped length-two window has only four possible origins. Python
    # integers retain every high bit when choosing the independent host slice.
    expected = np.stack(
        [
            operand[max(0, min(int(index), 3)) : max(0, min(int(index), 3)) + 2]
            for index in indices
        ]
    )
    assert result.dtype == np.int64
    np.testing.assert_array_equal(np.asarray(result), expected)
    record_metal_execution(record_property, result)


@pytest.mark.parametrize("index_kind", ["int64", "uint64"])
def test_metal_dynamic_slices_clamp_full_width_starts_before_narrowing(
    device: jax.Device, record_property: RecordProperty, index_kind: str
) -> None:
    operand = np.asarray([101, -(2**55) + 3, 2**54 + 1, 7, 11], np.int64)
    replacement = np.asarray([2**61 + 3, -(2**61) + 5], np.int64)

    @jax.jit
    def program(
        values: jax.Array, update: jax.Array, start: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        return (
            lax.dynamic_slice(values, (start,), (2,), allow_negative_indices=False),
            lax.dynamic_update_slice(
                values, update, (start,), allow_negative_indices=False
            ),
        )

    values = _put(operand, device)
    update = _put(replacement, device)
    results: list[jax.Array] = []
    # Do not vmap this program: JAX's dynamic-slice batching rule can replace
    # the primitive with gather, leaving the scalar dynamic-slice path untested.
    for index in _extreme_indices(index_kind):
        sliced, replaced = program(values, update, _put(np.asarray(index), device))
        origin = max(0, min(int(index), 3))
        expected_update = operand.copy()
        expected_update[origin : origin + 2] = replacement
        np.testing.assert_array_equal(np.asarray(sliced), operand[origin : origin + 2])
        np.testing.assert_array_equal(np.asarray(replaced), expected_update)
        assert sliced.dtype == np.int64 and replaced.dtype == np.int64
        results.extend((sliced, replaced))
    record_metal_execution(record_property, *results)


_SCATTER_CPU_REFERENCE = """
import json
import sys
import jax
import numpy as np
from jax import lax
from phydrax import config

assert {device.platform for device in jax.devices()} == {"cpu"}
assert config.x64_enabled()
payload = json.loads(sys.stdin.read())
operand = np.asarray(payload["operand"], np.float32)
indices = np.asarray(payload["indices"], np.dtype(payload["index_kind"]))
updates = np.asarray(payload["updates"], np.float32)
dimensions = lax.ScatterDimensionNumbers(
    **{name: tuple(axes) for name, axes in payload["dimensions"].items()}
)
result = jax.jit(
    lambda a, i, u: lax.scatter_add(
        a, i, u, dimensions, mode=lax.GatherScatterMode.FILL_OR_DROP
    )
)(operand, indices, updates)
print(json.dumps({"values": np.asarray(result).tolist()}))
"""


def _cpu_scatter_reference(payload: dict[str, object]) -> np.ndarray:
    reference = json_result(
        run_python(
            _SCATTER_CPU_REFERENCE,
            environment={"JAX_PLATFORMS": "cpu", "JAX_ENABLE_X64": "1"},
            stdin=json.dumps(payload),
        )
    )
    return np.asarray(reference["values"], np.float32)


@pytest.mark.parametrize("index_kind", ["int64", "uint64"])
def test_metal_scatter_partial_windows_match_independent_cpu(
    device: jax.Device, record_property: RecordProperty, index_kind: str
) -> None:
    operand = np.asarray([10, 20, 30, 40, 50], np.float32)
    starts = [-1, 3, 5, 2**32] if index_kind == "int64" else [0, 3, 5, 2**32]
    indices = np.asarray(starts, np.dtype(index_kind))[:, None]
    updates = np.asarray(
        [[1, 2, 4], [8, 16, 32], [64, 128, 256], [512, 1024, 2048]], np.float32
    )
    dimension_arguments = {
        "update_window_dims": (1,),
        "inserted_window_dims": (),
        "scatter_dims_to_operand_dims": (0,),
    }
    dimensions = lax.ScatterDimensionNumbers(**dimension_arguments)

    @jax.jit
    def program(a: jax.Array, i: jax.Array, u: jax.Array) -> jax.Array:
        return lax.scatter_add(
            a, i, u, dimensions, mode=lax.GatherScatterMode.FILL_OR_DROP
        )

    expected = _cpu_scatter_reference(
        {
            "operand": operand.tolist(),
            "indices": indices.tolist(),
            "updates": updates.tolist(),
            "index_kind": index_kind,
            "dimensions": dimension_arguments,
        }
    )
    result = program(_put(operand, device), _put(indices, device), _put(updates, device))
    np.testing.assert_array_equal(np.asarray(result), expected)
    record_metal_execution(record_property, result)


@pytest.mark.parametrize("index_kind", ["int64", "uint64"])
def test_metal_scatter_batched_interleaved_windows_match_independent_cpu(
    device: jax.Device, record_property: RecordProperty, index_kind: str
) -> None:
    operand = np.arange(40, dtype=np.float32).reshape(2, 4, 5)
    starts = (
        [[[-1, 3], [3, -1], [2**32, 0]], [[0, 0], [2, 4], [1, 1]]]
        if index_kind == "int64"
        else [[[0, 3], [3, 0], [2**64 - 1, 0]], [[0, 0], [2, 4], [1, 1]]]
    )
    indices = np.asarray(starts, np.dtype(index_kind))
    # Window axes 0/2 and index-batch axes 1/3 are intentionally interleaved.
    updates = np.arange(1, 37, dtype=np.float32).reshape(2, 2, 3, 3)
    dimension_arguments = {
        "update_window_dims": (0, 2),
        "inserted_window_dims": (),
        "scatter_dims_to_operand_dims": (1, 2),
        "operand_batching_dims": (0,),
        "scatter_indices_batching_dims": (0,),
    }
    dimensions = lax.ScatterDimensionNumbers(**dimension_arguments)

    @jax.jit
    def program(a: jax.Array, i: jax.Array, u: jax.Array) -> jax.Array:
        return lax.scatter_add(
            a, i, u, dimensions, mode=lax.GatherScatterMode.FILL_OR_DROP
        )

    expected = _cpu_scatter_reference(
        {
            "operand": operand.tolist(),
            "indices": indices.tolist(),
            "updates": updates.tolist(),
            "index_kind": index_kind,
            "dimensions": dimension_arguments,
        }
    )
    result = program(_put(operand, device), _put(indices, device), _put(updates, device))
    np.testing.assert_array_equal(np.asarray(result), expected)
    record_metal_execution(record_property, result)


@pytest.mark.parametrize("value_kind", ["int64", "uint64", "complex64"])
def test_metal_duplicate_scatter_updates_preserve_full_width_values(
    device: jax.Device, record_property: RecordProperty, value_kind: str
) -> None:
    match value_kind:
        case "int64":
            operand = np.asarray([-(2**63) + 9, 2**62 + 5, 7, -(2**62) + 1, 33], np.int64)
            updates = np.asarray(
                [2**62 + 3, 2**62 + 7, 2**40 + 1, -(2**61) + 1, 43, 99, 2**60 + 9, -3],
                np.int64,
            )
        case "uint64":
            operand = np.asarray([2**64 - 9, 2**63 + 5, 7, 2**62 + 1, 33], np.uint64)
            updates = np.asarray(
                [2**63 + 3, 2**63 + 7, 2**53 + 1, 2**62 + 1, 43, 99, 2**60 + 9, 3],
                np.uint64,
            )
        case "complex64":
            operand = np.asarray([1 + 2j, -2 + 1j, 3 - 1j, 1 + 0j, 0 + 0j], np.complex64)
            updates = np.asarray(
                [1 + 1j, 2 - 1j, -1 + 2j, 3 + 0j, 43 + 2j, 99 - 1j, 2 + 3j, -3 + 1j],
                np.complex64,
            )
        case _:
            raise AssertionError(value_kind)
    indices = np.asarray([1, 1, 3, 1, 6, -1, 0, 3], np.int64)[:, None]
    dimensions = lax.ScatterDimensionNumbers(
        update_window_dims=(),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,),
    )

    @jax.jit
    def program(a: jax.Array, i: jax.Array, u: jax.Array) -> tuple[jax.Array, ...]:
        mode = lax.GatherScatterMode.FILL_OR_DROP
        return (
            lax.scatter_add(a, i, u, dimensions, mode=mode),
            lax.scatter_sub(a, i, u, dimensions, mode=mode),
            lax.scatter_mul(a, i, u, dimensions, mode=mode),
            lax.scatter(a, i, u, dimensions, mode=mode),
        )

    results = program(_put(operand, device), _put(indices, device), _put(updates, device))
    # NumPy's indexed ufuncs independently specify repeated updates and modular
    # integer arithmetic. The valid positions are explicit, not GPU-derived.
    valid_positions = np.asarray([0, 1, 2, 3, 6, 7], np.intp)
    valid_indices = np.asarray([1, 1, 3, 1, 0, 3], np.intp)
    for result, operation in zip(
        results[:3], (np.add, np.subtract, np.multiply), strict=True
    ):
        expected = operand.copy()
        operation.at(expected, valid_indices, updates[valid_positions])
        assert result.dtype == operand.dtype
        np.testing.assert_array_equal(np.asarray(result), expected)
    replaced = np.asarray(results[3])
    assert results[3].dtype == operand.dtype
    np.testing.assert_array_equal(replaced[[2, 4]], operand[[2, 4]])
    assert replaced[0] == updates[6]
    # StableHLO permits any serialization of duplicate replacement writes; it
    # does not require the CPU/GPU to pick the same winner.
    assert replaced[1] in updates[[0, 1, 3]]
    assert replaced[3] in updates[[2, 7]]
    record_metal_execution(record_property, *results)


def test_metal_out_of_bounds_nan_scatter_updates_do_not_contaminate_valid_duplicates(
    device: jax.Device, record_property: RecordProperty
) -> None:
    operand = np.asarray([3, 4], np.float32)
    indices = np.asarray([0, 7, -1, 2**32, 0], np.int64)[:, None]
    updates = np.asarray([1, np.nan, np.nan, np.nan, 2], np.float32)
    dimensions = lax.ScatterDimensionNumbers(
        update_window_dims=(),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,),
    )

    @jax.jit
    def program(a: jax.Array, i: jax.Array, u: jax.Array) -> tuple[jax.Array, ...]:
        mode = lax.GatherScatterMode.FILL_OR_DROP
        return (
            lax.scatter_add(a, i, u, dimensions, mode=mode),
            lax.scatter_min(a, i, u, dimensions, mode=mode),
            lax.scatter_max(a, i, u, dimensions, mode=mode),
        )

    results = program(_put(operand, device), _put(indices, device), _put(updates, device))
    for result, expected in zip(results, ([6, 4], [1, 4], [3, 4]), strict=True):
        np.testing.assert_array_equal(
            np.asarray(result), np.asarray(expected, np.float32)
        )
    record_metal_execution(record_property, *results)


def test_metal_scatter_min_max_propagate_nan_and_preserve_signed_zero(
    device: jax.Device, record_property: RecordProperty
) -> None:
    operand = np.asarray([0.0, -0.0, 1.0, np.nan, 0.0, -0.0], np.float32)
    indices = np.asarray([0, 1, 2, 3, 4, 5, 4, 5, 7], np.int64)[:, None]
    updates = np.asarray(
        [-0.0, 0.0, np.nan, 2.0, 0.0, -0.0, -0.0, 0.0, np.nan], np.float32
    )
    dimensions = lax.ScatterDimensionNumbers(
        update_window_dims=(),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,),
    )

    @jax.jit
    def program(a: jax.Array, i: jax.Array, u: jax.Array) -> tuple[jax.Array, jax.Array]:
        mode = lax.GatherScatterMode.FILL_OR_DROP
        return (
            lax.scatter_min(a, i, u, dimensions, mode=mode),
            lax.scatter_max(a, i, u, dimensions, mode=mode),
        )

    minimum, maximum = program(
        _put(operand, device), _put(indices, device), _put(updates, device)
    )
    zero_positions = np.asarray([0, 1, 4, 5], np.intp)
    for result, zero_bits in ((minimum, 0x80000000), (maximum, 0x00000000)):
        observed = np.asarray(result)
        np.testing.assert_array_equal(
            np.isnan(observed), [False, False, True, True, False, False]
        )
        np.testing.assert_array_equal(
            observed[zero_positions].view(np.uint32), np.full(4, zero_bits, np.uint32)
        )
    record_metal_execution(record_property, minimum, maximum)


@pytest.mark.parametrize("index_kind", ["int64", "uint64"])
@pytest.mark.parametrize("value_kind", ["int64", "uint64", "complex64", "float32"])
@pytest.mark.parametrize("batched_index", [False, True])
def test_metal_single_point_scatter_full_width_bounds_and_reducers(
    device: jax.Device,
    record_property: RecordProperty,
    index_kind: str,
    value_kind: str,
    batched_index: bool,
) -> None:
    dtype = np.dtype(value_kind)
    if value_kind == "int64":
        operand = np.asarray([2**63 - 3, -(2**63) + 5, 2**54 + 1, 7], dtype)
        update = np.asarray(2**62 + 9, dtype)
    elif value_kind == "uint64":
        operand = np.asarray([2**64 - 3, 2**63 + 5, 2**54 + 1, 7], dtype)
        update = np.asarray(2**63 + 9, dtype)
    elif value_kind == "complex64":
        operand = np.asarray([1 + 2j, -2 + 3j, 4 - 1j, 7], dtype)
        update = np.asarray(2 - 3j, dtype)
    else:
        operand = np.asarray([1, -2, 4, 7], dtype)
        update = np.asarray(-3, dtype)
    dimensions = lax.ScatterDimensionNumbers(
        update_window_dims=(),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,),
    )
    operations = [lax.scatter, lax.scatter_add, lax.scatter_sub, lax.scatter_mul]
    references = [None, np.add, np.subtract, np.multiply]
    if value_kind != "complex64":
        operations.extend((lax.scatter_min, lax.scatter_max))
        references.extend((np.minimum, np.maximum))

    @jax.jit
    def program(a: jax.Array, i: jax.Array, u: jax.Array) -> tuple[jax.Array, ...]:
        return tuple(
            operation(a, i, u, dimensions, mode=lax.GatherScatterMode.FILL_OR_DROP)
            for operation in operations
        )

    values = _put(operand, device)
    replacement = _put(update.reshape((1,) if batched_index else ()), device)
    results = []
    for index in _extreme_indices(index_kind):
        starts = np.asarray(index).reshape((1, 1) if batched_index else (1,))
        observed = program(values, _put(starts, device), replacement)
        for result, reference in zip(observed, references, strict=True):
            expected = operand.copy()
            if 0 <= int(index) < operand.size:
                if reference is None:
                    expected[int(index)] = update
                else:
                    reference.at(expected, int(index), update)
            assert result.dtype == dtype
            np.testing.assert_array_equal(np.asarray(result), expected)
        results.extend(observed)
    record_metal_execution(record_property, *results)


@pytest.mark.parametrize("index_kind", ["int64", "uint64"])
def test_metal_single_interleaved_window_scatter_drop_nan_zero_and_reducers(
    device: jax.Device, record_property: RecordProperty, index_kind: str
) -> None:
    operand = np.resize(np.asarray([0.0, -0.0, np.nan, 2.0], np.float32), (3, 4, 5))
    updates = np.asarray([-0.0, 0.0, 1.0, np.nan, 3.0, -2.0], np.float32).reshape(2, 1, 3)
    dimensions = lax.ScatterDimensionNumbers(
        update_window_dims=(0, 2),
        inserted_window_dims=(1,),
        scatter_dims_to_operand_dims=(0, 1, 2),
    )
    operations = (
        lax.scatter,
        lax.scatter_add,
        lax.scatter_sub,
        lax.scatter_mul,
        lax.scatter_min,
        lax.scatter_max,
    )
    references = (None, np.add, np.subtract, np.multiply, np.minimum, np.maximum)

    @jax.jit
    def program(a: jax.Array, i: jax.Array, u: jax.Array) -> tuple[jax.Array, ...]:
        return tuple(
            operation(a, i, u, dimensions, mode=lax.GatherScatterMode.FILL_OR_DROP)
            for operation in operations
        )

    starts = [[0, 2, 1], [1, 3, 2], [0, 2, 3], [2, 2, 0], [0, 4, 0]]
    starts.extend([int(index), 2, 1] for index in _extreme_indices(index_kind))
    results = []
    for start in starts:
        observed = program(
            _put(operand, device),
            _put(np.asarray([start], np.dtype(index_kind)), device),
            _put(updates, device),
        )
        valid = 0 <= start[0] <= 1 and 0 <= start[1] < 4 and 0 <= start[2] <= 2
        for result, reference in zip(observed, references, strict=True):
            expected = operand.copy()
            if valid:
                region = np.s_[start[0] : start[0] + 2, start[1], start[2] : start[2] + 3]
                old, new = operand[region], updates[:, 0, :]
                folded = new.copy() if reference is None else reference(old, new)
                if reference in (np.minimum, np.maximum):
                    zeros = (old == 0) & (new == 0)
                    bits = (
                        old.view(np.uint32) | new.view(np.uint32)
                        if reference is np.minimum
                        else old.view(np.uint32) & new.view(np.uint32)
                    )
                    folded[zeros] = bits.view(np.float32)[zeros]
                expected[region] = folded
            actual = np.asarray(result)
            assert result.dtype == np.float32
            np.testing.assert_array_equal(actual, expected)
            finite = ~np.isnan(expected)
            np.testing.assert_array_equal(
                actual.view(np.uint32)[finite], expected.view(np.uint32)[finite]
            )
        results.extend(observed)
    record_metal_execution(record_property, *results)


def test_metal_single_window_scatter_without_index_components(
    device: jax.Device, record_property: RecordProperty
) -> None:
    operand = np.asarray([[2**63 + 1, 2**64 - 1, 3], [4, 5, 6]], np.uint64)
    updates = np.asarray([[2**63 + 3, 7, 8], [9, 10, 11]], np.uint64)
    dimensions = lax.ScatterDimensionNumbers(
        update_window_dims=(0, 1),
        inserted_window_dims=(),
        scatter_dims_to_operand_dims=(),
    )

    @jax.jit
    def program(a: jax.Array, i: jax.Array, u: jax.Array) -> tuple[jax.Array, ...]:
        return (
            lax.scatter(a, i, u, dimensions, mode="drop"),
            lax.scatter_add(a, i, u, dimensions, mode="drop"),
        )

    replaced, added = program(
        _put(operand, device),
        _put(np.empty((0,), np.int64), device),
        _put(updates, device),
    )
    for result, expected in ((replaced, updates), (added, operand + updates)):
        assert result.dtype == np.uint64
        np.testing.assert_array_equal(np.asarray(result), expected)
    record_metal_execution(record_property, replaced, added)


@pytest.mark.parametrize("index_kind", ["int64", "uint64"])
@pytest.mark.parametrize("value_kind", ["int64", "uint64", "float32", "complex64"])
def test_metal_fill_gather_preserves_full_width_bounds_and_complete_windows(
    device: jax.Device,
    record_property: RecordProperty,
    index_kind: str,
    value_kind: str,
) -> None:
    dtype = np.dtype(value_kind)
    if value_kind == "int64":
        operand = np.asarray([2**62 + 1, -(2**61) + 3, 7, 9, 11], dtype)
        fill = -(2**60) + 5
    elif value_kind == "uint64":
        operand = np.asarray([2**63 + 1, 2**64 - 3, 7, 9, 11], dtype)
        fill = 2**63 + 5
    elif value_kind == "complex64":
        operand = np.asarray([1 + 2j, -3 + 4j, 7, 9, 11], dtype)
        fill = -7 + 3j
    else:
        operand = np.asarray([1, -3, 7, 9, 11], dtype)
        fill = -17.0
    indices = _extreme_indices(index_kind)
    dimensions = lax.GatherDimensionNumbers(
        offset_dims=(1,), collapsed_slice_dims=(), start_index_map=(0,)
    )

    @jax.jit
    def program(values: jax.Array, starts: jax.Array) -> jax.Array:
        return lax.gather(
            values,
            starts[:, None],
            dimensions,
            slice_sizes=(2,),
            mode="fill",
            fill_value=fill,
        )

    result = program(_put(operand, device), _put(indices, device))
    expected = np.stack(
        [
            operand[int(index) : int(index) + 2]
            if 0 <= int(index) <= 3
            else np.full(2, fill, dtype)
            for index in indices
        ]
    )
    assert result.dtype == dtype
    np.testing.assert_array_equal(np.asarray(result), expected)
    record_metal_execution(record_property, result)


@pytest.mark.parametrize("index_kind", ["int64", "uint64"])
def test_metal_fill_gather_batched_windows_keep_batch_and_offset_axes(
    device: jax.Device, record_property: RecordProperty, index_kind: str
) -> None:
    operand = np.arange(40, dtype=np.float32).reshape(2, 4, 5)
    extreme = -(2**63) if index_kind == "int64" else 2**64 - 1
    indices = np.asarray(
        [[[0], [2], [3]], [[extreme], [1], [2**32]]], np.dtype(index_kind)
    )
    dimensions = lax.GatherDimensionNumbers(
        offset_dims=(1, 3),
        collapsed_slice_dims=(),
        start_index_map=(2,),
        operand_batching_dims=(0,),
        start_indices_batching_dims=(0,),
    )

    @jax.jit
    def program(a: jax.Array, i: jax.Array) -> jax.Array:
        return lax.gather(
            a, i, dimensions, slice_sizes=(1, 2, 3), mode="fill", fill_value=-17.0
        )

    # Batch output dimensions 0/2 deliberately interleave the two window axes.
    expected = np.full((2, 2, 3, 3), -17, np.float32)
    for batch in range(2):
        for window in range(3):
            start = int(indices[batch, window, 0])
            if 0 <= start <= 2:
                expected[batch, :, window, :] = operand[batch, :2, start : start + 3]
    result = program(_put(operand, device), _put(indices, device))
    assert result.dtype == np.float32
    np.testing.assert_array_equal(np.asarray(result), expected)
    record_metal_execution(record_property, result)


@pytest.mark.parametrize("index_kind", ["int64", "uint64"])
@pytest.mark.parametrize("window_size", [1, 3])
@pytest.mark.parametrize("reducer", ["update", "add", "mul"])
def test_metal_single_scatter_derivatives(
    device: jax.Device,
    record_property: RecordProperty,
    window_size: int,
    reducer: str,
    index_kind: str,
) -> None:
    operand = np.arange(1, 7, dtype=np.float32)
    updates = np.arange(2, window_size + 2, dtype=np.float32)
    dimensions = lax.ScatterDimensionNumbers(
        update_window_dims=(0,),
        inserted_window_dims=(),
        scatter_dims_to_operand_dims=(0,),
    )
    operation = {"update": lax.scatter, "add": lax.scatter_add, "mul": lax.scatter_mul}[
        reducer
    ]

    @jax.jit
    def program(a: jax.Array, i: jax.Array, u: jax.Array) -> tuple[jax.Array, jax.Array]:
        def loss(x: jax.Array, y: jax.Array) -> jax.Array:
            # A single indexed window is injective; scatter_mul's transpose
            # requires this true uniqueness promise.
            return jnp.sum(
                operation(x, i, y, dimensions, mode="drop", unique_indices=True)
            )

        return jax.grad(loss, argnums=(0, 1))(a, u)

    results = []
    # Reverse-mode scatter lowers its update cotangent through fill-gather.
    # Check that lowering before it can discard high index bits.
    starts = [*_extreme_indices(index_kind), 6]
    for index in starts:
        start = int(index)
        da, du = program(
            _put(operand, device),
            _put(np.asarray([start], np.dtype(index_kind)), device),
            _put(updates, device),
        )
        expected_a = np.ones_like(operand)
        expected_u = np.zeros_like(updates)
        if 0 <= start <= operand.size - window_size:
            region = slice(start, start + window_size)
            expected_a[region] = (
                updates if reducer == "mul" else 0 if reducer == "update" else 1
            )
            expected_u[:] = operand[region] if reducer == "mul" else 1
        np.testing.assert_array_equal(np.asarray(da), expected_a)
        np.testing.assert_array_equal(np.asarray(du), expected_u)
        results.extend((da, du))
    record_metal_execution(record_property, *results)


@pytest.mark.parametrize("key_kind", ["int64", "uint64"])
def test_metal_stable_descending_sort_keeps_full_width_key_payload_ties(
    device: jax.Device, record_property: RecordProperty, key_kind: str
) -> None:
    values = (
        [2**63 - 1, -(2**63), 2**53 + 1, 2**53, 2**53 + 1, -1, 2**63 - 1, -(2**63)]
        if key_kind == "int64"
        else [2**64 - 1, 2**63, 2**53 + 1, 2**53, 2**53 + 1, 0, 2**64 - 1, 2**63]
    )
    keys = np.asarray(values, np.dtype(key_kind))
    identifiers = np.asarray(
        [2**64 - 1, 2**63 + 9, 2**53 + 1, 2**53, 2**63, 2**32 + 1, 17, 2**64 - 2],
        np.uint64,
    )

    @jax.jit
    def program(k: jax.Array, payload: jax.Array) -> tuple[jax.Array, ...]:
        ascending = jnp.argsort(k, stable=True)
        descending = jnp.argsort(k, stable=True, descending=True)
        _, _, lexicographic = lax.sort(
            (k, payload, lax.broadcasted_iota(np.int64, k.shape, 0)),
            num_keys=2,
            is_stable=True,
        )
        return (
            k[ascending],
            payload[ascending],
            k[descending],
            payload[descending],
            lexicographic,
        )

    results = program(_put(keys, device), _put(identifiers, device))
    ascending = sorted(range(8), key=lambda i: values[i])
    descending = sorted(range(8), key=lambda i: -values[i])
    lexicographic = sorted(range(8), key=lambda i: (values[i], int(identifiers[i])))
    for result, expected in zip(
        results,
        (
            keys[ascending],
            identifiers[ascending],
            keys[descending],
            identifiers[descending],
            lexicographic,
        ),
        strict=True,
    ):
        np.testing.assert_array_equal(np.asarray(result), expected)
    assert results[0].dtype == keys.dtype and results[2].dtype == keys.dtype
    assert results[1].dtype == np.uint64 and results[3].dtype == np.uint64
    record_metal_execution(record_property, *results)


def test_metal_unrecognized_sort_comparator_is_refused_without_single_key_fallback(
    device: jax.Device, record_property: RecordProperty
) -> None:
    # Public lax.sort constructs only its standard comparator. A test-local JAX
    # extension emits a real, valid StableHLO sort by squared magnitude, which
    # cannot be approximated by sorting its raw input in either direction.
    from jax import core
    from jax.extend.core import Primitive
    from jax.interpreters import mlir
    from jaxlib.mlir import ir
    from jaxlib.mlir.dialects import stablehlo

    primitive = Primitive("metal_regression_sort_squared_magnitude")

    def abstract_eval(operand: core.AbstractValue) -> core.AbstractValue:
        return operand

    def lower(_context: mlir.LoweringRuleContext, operand: ir.Value) -> list[ir.Value]:
        operation = stablehlo.SortOp(
            [operand.type], [operand], dimension=0, is_stable=True
        )
        scalar = ir.RankedTensorType.get((), ir.IntegerType.get_signless(32))
        block = operation.comparator.blocks.append(scalar, scalar)
        with ir.InsertionPoint(block):
            lhs = stablehlo.MulOp(block.arguments[0], block.arguments[0]).result
            rhs = stablehlo.MulOp(block.arguments[1], block.arguments[1]).result
            comparison = stablehlo.CompareOp(
                lhs,
                rhs,
                stablehlo.ComparisonDirectionAttr.get("LT"),
                compare_type=stablehlo.ComparisonTypeAttr.get("SIGNED"),
            )
            stablehlo.ReturnOp([comparison.result])
        return list(operation.results)

    primitive.def_abstract_eval(abstract_eval)
    mlir.register_lowering(primitive, lower, platform="mps")

    @jax.jit
    def unsupported(operand: jax.Array) -> jax.Array:
        return primitive.bind(operand)

    operand = _put(np.asarray([-3, -1, 2], np.int32), device)
    with pytest.raises(jax.errors.JaxRuntimeError, match=r"stablehlo.sort"):
        jax.block_until_ready(unsupported(operand))
    # Failure must leave admitted GPU work executable, not poison the client.
    recovered = jax.jit(lambda x: x + np.int32(1))(operand)
    np.testing.assert_array_equal(np.asarray(recovered), np.asarray([-2, 0, 3], np.int32))
    record_metal_execution(record_property, recovered)
