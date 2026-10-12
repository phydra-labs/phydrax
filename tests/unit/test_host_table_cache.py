"""Retention and scientific-value contracts of host preparation caches."""

from __future__ import annotations

import dataclasses
import gc
import importlib
import weakref
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pytest

from benchmarks.memory_host_caches import fem_hot_sets, HOT_SETS
from phydrax._host_table_cache import host_table_cache, logical_table_bytes
from phydrax._trainable import _field_roles, ArrayRole, fixed_field
from phydrax.applications.conformal_bootstrap._virasoro_generic import (
    VirasoroEllipticBlockPlan,
)


_TABLE_CASES = (*HOT_SETS, *fem_hot_sets())


def test_count_eviction_follows_access_order() -> None:
    calls: list[int] = []

    @host_table_cache(max_entries=2, max_bytes=1024)
    def table(key: int) -> tuple[int, ...]:
        calls.append(key)
        return (key,)

    table(1)
    table(2)
    table(1)
    table(3)
    table(1)
    table(2)
    assert calls == [1, 2, 3, 2]
    assert table.cache_info().currsize == 2


def test_byte_eviction_is_independent_of_count() -> None:
    calls: list[int] = []

    @host_table_cache(max_entries=20, max_bytes=64)
    def table(key: int) -> np.ndarray:
        calls.append(key)
        return np.full((4,), key, dtype=np.float64)

    table(1)
    table(2)
    table(1)
    table(3)
    assert table.cache_info().current_bytes == 64
    table(2)
    assert calls == [1, 2, 3, 2]
    assert table.cache_info().currsize == 2


def test_oversized_value_does_not_flush_or_enter_cache() -> None:
    calls: list[int] = []

    @host_table_cache(max_entries=3, max_bytes=64)
    def table(count: int) -> np.ndarray:
        calls.append(count)
        return np.arange(count, dtype=np.float64)

    small = table(4)
    np.testing.assert_array_equal(table(9), np.arange(9, dtype=np.float64))
    table(9)
    assert table(4) is small
    assert calls == [4, 9, 9]
    assert table.cache_info().current_bytes == 32
    assert table.cache_info().currsize == 1


def test_concurrent_reads_preserve_values_and_budget() -> None:
    @host_table_cache(max_entries=4, max_bytes=128)
    def table(key: int) -> np.ndarray:
        return np.full((4,), key, dtype=np.float64)

    keys = tuple(range(16)) * 4
    with ThreadPoolExecutor(max_workers=4) as pool:
        values = tuple(pool.map(table, keys))
    for key, value in zip(keys, values, strict=True):
        np.testing.assert_array_equal(value, np.full((4,), key, dtype=np.float64))
    info = table.cache_info()
    assert info.current_bytes <= 128
    assert info.currsize <= 4
    table.cache_clear()
    assert table.cache_info().current_bytes == 0
    assert table.cache_info().currsize == 0


@pytest.mark.parametrize(
    "max_entries,max_bytes",
    [(0, 8), (1, 0), (-1, 8)],
    ids=["zero-count", "zero-bytes", "negative-count"],
)
def test_invalid_cache_limits_refuse(max_entries: int, max_bytes: int) -> None:
    def table(key: int) -> int:
        return key

    with pytest.raises(ValueError, match="limits must be positive"):
        host_table_cache(max_entries=max_entries, max_bytes=max_bytes)(table)


def test_structural_weight_does_not_materialize_arrays() -> None:
    array = np.arange(32, dtype=np.float64)[::2]
    assert (
        logical_table_bytes((array, (1, 2), "abc", None, 1.0j))
        == array.nbytes + 16 + 3 + 16
    )
    assert logical_table_bytes(1 << 128) == 17


def test_field_role_metadata_does_not_retain_class() -> None:
    cls = dataclasses.make_dataclass("Transient", [("value", np.ndarray, fixed_field())])
    reference = weakref.ref(cls)
    roles = _field_roles(cls)
    assert roles == {"value": ArrayRole.FIXED}
    del cls
    gc.collect()
    assert reference() is None
    assert roles == {"value": ArrayRole.FIXED}


def _assert_same(left: Any, right: Any, /) -> None:
    if isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif dataclasses.is_dataclass(left) and not isinstance(left, type):
        assert type(left) is type(right)
        for field in dataclasses.fields(left):
            _assert_same(getattr(left, field.name), getattr(right, field.name))
    elif isinstance(left, tuple):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            _assert_same(a, b)
    else:
        assert left == right


@pytest.mark.parametrize(
    "module,symbol,arguments", _TABLE_CASES, ids=[row[1] for row in _TABLE_CASES]
)
def test_migrated_tables_equal_uncached_computation(
    module: str, symbol: str, arguments: tuple[tuple[Any, ...], ...]
) -> None:
    function = getattr(importlib.import_module(f"phydrax.{module}"), symbol)
    function.cache_clear()
    args = arguments[-1]
    expected = function.__wrapped__(*args)
    actual = function(*args)
    _assert_same(actual, expected)
    _assert_same(function(*args), expected)
    info = function.cache_info()
    assert info.current_bytes <= info.max_bytes
    assert info.currsize <= info.max_entries


@pytest.mark.parametrize(
    "level,capacity", [(8, 66), (1000000000, 4096)], ids=["exact-boundary", "huge-level"]
)
def test_virasoro_partition_cap_refuses_before_enumeration(
    monkeypatch: pytest.MonkeyPatch, level: int, capacity: int
) -> None:
    def unexpected_enumeration(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Partition enumeration happened before capacity refusal")

    monkeypatch.setattr(
        "phydrax.applications.conformal_bootstrap._virasoro_generic._partitions",
        unexpected_enumeration,
    )
    with pytest.raises(ValueError, match="descendant count exceeds"):
        VirasoroEllipticBlockPlan(
            25.0,
            (0.2, 0.3, 0.4, 0.5),
            0.7,
            maximum_level=level,
            maximum_partition_count=capacity,
        )


def test_virasoro_partition_admission_accepts_exact_capacity() -> None:
    # Independent partition numbers p(0)..p(8) sum to 67.
    plan = VirasoroEllipticBlockPlan(
        25.0, (0.2, 0.3, 0.4, 0.5), 0.7, maximum_level=8, maximum_partition_count=67
    )
    assert plan.maximum_level == 8
    assert plan.maximum_partition_count == 67
