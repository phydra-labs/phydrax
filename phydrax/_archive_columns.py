#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Archive columns for per-row static values of native model records.

Model recipes encode every static value as one manifest node. Records that keep
one static value per mesh entity (source IDs, roles, tags, exact witness text)
would otherwise put one JSON node per row into the manifest. These columns move
the row data into array members: repeated values become a first-occurrence table
plus one row-index array, and unique text becomes UTF-8 bytes plus offsets.
Both decode to exactly the original tuple.
"""

from __future__ import annotations

from collections.abc import Hashable
from typing import Any, NamedTuple

import numpy as np


def _column_key(value: Any, /) -> Hashable:
    """Type-exact key so equal-comparing values of different types stay distinct."""
    if isinstance(value, tuple):
        return (type(value), tuple(_column_key(item) for item in value))
    return (type(value), value)


class StaticColumnArchiveRecipe(NamedTuple):
    """Per-row static values as a first-occurrence table and int64 row indices."""

    table: tuple[Any, ...]
    rows: np.ndarray

    @classmethod
    def encode(cls, values: tuple[Any, ...], /) -> StaticColumnArchiveRecipe:
        indices: dict[Hashable, int] = {}
        table: list[Any] = []
        rows = np.empty((len(values),), dtype=np.int64)
        for row, value in enumerate(values):
            key = _column_key(value)
            index = indices.get(key)
            if index is None:
                index = indices[key] = len(table)
                table.append(value)
            rows[row] = index
        return cls(tuple(table), rows)

    def decode(self) -> tuple[Any, ...]:
        rows = np.asarray(self.rows)
        if not isinstance(self.table, tuple):
            raise TypeError("Static column table must be a tuple.")
        if rows.ndim != 1 or rows.dtype != np.int64:
            raise TypeError("Static column rows must be one int64 vector.")
        if rows.size and (int(rows.min()) < 0 or int(rows.max()) >= len(self.table)):
            raise ValueError("Static column rows must index their value table.")
        # The table is the canonical first-occurrence order of the rows: each
        # row names a seen entry or the next new one, and every entry is used.
        if rows.size:
            running = np.maximum.accumulate(rows)
            previous = np.concatenate((np.full((1,), -1, np.int64), running[:-1]))
            canonical = bool(np.all(rows <= previous + 1)) and int(running[-1]) == (
                len(self.table) - 1
            )
        else:
            canonical = not self.table
        if not canonical:
            raise ValueError("Static column table is not in first-occurrence order.")
        table = self.table
        return tuple(table[index] for index in rows.tolist())


class TextColumnArchiveRecipe(NamedTuple):
    """Per-row text as concatenated UTF-8 bytes and int64 row end offsets."""

    data: np.ndarray
    ends: np.ndarray

    @classmethod
    def encode(cls, values: tuple[str, ...], /) -> TextColumnArchiveRecipe:
        if any(not isinstance(value, str) for value in values):
            raise TypeError("Text columns hold only strings.")
        encoded = [value.encode("utf-8") for value in values]
        ends = np.cumsum(
            np.asarray([len(item) for item in encoded], dtype=np.int64), dtype=np.int64
        )
        data = np.frombuffer(b"".join(encoded), dtype=np.uint8).copy()
        return cls(data, ends)

    def decode(self) -> tuple[str, ...]:
        data = np.asarray(self.data)
        ends = np.asarray(self.ends)
        if data.ndim != 1 or data.dtype != np.uint8:
            raise TypeError("Text column data must be one uint8 vector.")
        if ends.ndim != 1 or ends.dtype != np.int64:
            raise TypeError("Text column ends must be one int64 vector.")
        if (ends.size == 0 and data.size) or (
            ends.size
            and (
                int(ends[0]) < 0
                or np.any(np.diff(ends) < 0)
                or int(ends[-1]) != data.size
            )
        ):
            raise ValueError("Text column ends must partition their UTF-8 bytes.")
        payload = data.tobytes()
        starts = np.concatenate((np.zeros((1,), np.int64), ends))[:-1]
        return tuple(
            payload[start:end].decode("utf-8")
            for start, end in zip(starts.tolist(), ends.tolist(), strict=True)
        )


def integer_column(values: tuple[int, ...], /) -> np.ndarray:
    """Return exact int64 rows; refuses values outside the int64 range."""
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise TypeError("Integer columns hold only Python integers.")
    try:
        return np.asarray(values, dtype=np.int64).reshape((len(values),))
    except OverflowError as error:
        raise TypeError("Integer column values must fit int64.") from error


def integer_column_values(rows: Any, /) -> tuple[int, ...]:
    array = np.asarray(rows)
    if array.ndim != 1 or array.dtype != np.int64:
        raise TypeError("Integer column rows must be one int64 vector.")
    return tuple(array.tolist())


__all__ = [
    "integer_column",
    "integer_column_values",
    "StaticColumnArchiveRecipe",
    "TextColumnArchiveRecipe",
]
