"""Deterministic, byte-admitted LRU storage for immutable host preparation tables.

Weights count logical NumPy payload bytes (including views), not allocator/RSS
usage. Structural tuples count their scalar payloads explicitly. Oversized
results are still computed and returned, but never retained. In-flight results
and caller-owned references are outside the cache retention budget.

Preparation-site ceilings are selected from measured representative hot sets:
max(1 MiB, next power of two >= 8 * measured logical hot-set payload). This
reserve accommodates larger supported workloads without making a small smoke
workload a silent recomputation threshold. Dynamic entry limits are at least
256; the byte ceiling remains authoritative for large tables.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from functools import update_wrapper
from threading import RLock
from typing import NamedTuple, ParamSpec, TypeVar

import numpy as np


P = ParamSpec("P")
R = TypeVar("R")
_KEY_SEPARATOR = object()


def logical_table_bytes(value: object, /) -> int:
    """Weight array/tuple payloads without copying or transferring array data.

    Owners of custom records supply an explicit weight of their retained fields.
    Integer metadata uses an eight-byte logical slot, independent of Python's
    allocator; arbitrary-precision integers charge additional magnitude bytes.
    """
    if isinstance(value, np.ndarray):
        return value.nbytes
    if isinstance(value, tuple):
        return sum(logical_table_bytes(item) for item in value)
    if value is None:
        return 0
    if isinstance(value, str):
        return len(value.encode("utf-8"))
    if isinstance(value, bytes):
        return len(value)
    if isinstance(value, complex):
        return 16
    if isinstance(value, int):
        return max(8, (value.bit_length() + 7) // 8)
    if isinstance(value, (float, np.number)):
        return 8
    raise TypeError("Host-table records require an explicit logical byte weight.")


class HostTableCacheInfo(NamedTuple):
    hits: int
    misses: int
    max_entries: int
    max_bytes: int
    currsize: int
    current_bytes: int


class HostTableCache[**P, R]:
    """Callable host table cache; concurrent misses may compute independently."""

    def __init__(
        self,
        function: Callable[P, R],
        /,
        *,
        max_entries: int,
        max_bytes: int,
        weight: Callable[[R], int],
    ) -> None:
        if max_entries < 1 or max_bytes < 1:
            raise ValueError("Host-table cache limits must be positive.")
        self.__wrapped__ = function
        self._max_entries = max_entries
        self._max_bytes = max_bytes
        self._weight = weight
        self._entries: OrderedDict[tuple[object, ...], tuple[R, int]] = OrderedDict()
        self._bytes = 0
        self._hits = 0
        self._misses = 0
        self._lock = RLock()
        update_wrapper(self, function)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        # Preserve keyword insertion order, matching functools' key convention.
        key = args + (_KEY_SEPARATOR,) + tuple(kwargs.items())
        with self._lock:
            if key in self._entries:
                self._entries.move_to_end(key)
                self._hits += 1
                return self._entries[key][0]
            self._misses += 1
        value = self.__wrapped__(*args, **kwargs)
        size = self._weight(value)
        if not isinstance(size, int) or isinstance(size, bool):
            raise TypeError("Host-table weights must be integer byte counts.")
        if size < 0:
            raise ValueError("Host-table weights must be nonnegative.")
        if size > self._max_bytes:
            return value
        with self._lock:
            if key not in self._entries:
                while self._entries and (
                    len(self._entries) >= self._max_entries
                    or self._bytes + size > self._max_bytes
                ):
                    _, (_, evicted_bytes) = self._entries.popitem(last=False)
                    self._bytes -= evicted_bytes
                self._entries[key] = value, size
                self._bytes += size
        return value

    def cache_info(self) -> HostTableCacheInfo:
        with self._lock:
            return HostTableCacheInfo(
                self._hits,
                self._misses,
                self._max_entries,
                self._max_bytes,
                len(self._entries),
                self._bytes,
            )

    def cache_clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._bytes = 0
            self._hits = 0
            self._misses = 0


def host_table_cache(
    *,
    max_entries: int,
    max_bytes: int,
    weight: Callable[[R], int] = logical_table_bytes,
) -> Callable[[Callable[P, R]], HostTableCache[P, R]]:
    """Decorate one host preparation function with count and byte ceilings."""

    def decorate(function: Callable[P, R]) -> HostTableCache[P, R]:
        return HostTableCache(
            function, max_entries=max_entries, max_bytes=max_bytes, weight=weight
        )

    return decorate
