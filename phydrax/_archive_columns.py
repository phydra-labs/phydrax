#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Archive columns for per-row static values of native model records.

Model recipes encode every static value as one manifest node. Records that keep
one static value per mesh entity (source IDs, roles, tags, exact witness text)
would otherwise put one JSON node per row into the manifest. These columns move
the row data into array members: repeated values become a first-occurrence table
plus one row-index array, unique text becomes UTF-8 bytes plus offsets,
variable-length integer rows become int64 values plus row ends, and regular
exact integer or rational tensors become canonical decimal text. All decode to
exactly the original tuple.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Iterator
from fractions import Fraction
from itertools import repeat
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


def _encode_text_data(
    values: Iterable[str], count: int, /
) -> tuple[np.ndarray, np.ndarray]:
    """Own one UTF-8 payload and its exact row offsets, without flat text banks."""
    payload = bytearray()
    ends = np.empty((count,), dtype=np.int64)
    for index, value in zip(range(count), values, strict=True):
        payload.extend(value.encode("utf-8"))
        ends[index] = len(payload)
    return np.frombuffer(payload, dtype=np.uint8), ends


class TextColumnArchiveRecipe(NamedTuple):
    """Per-row text as concatenated UTF-8 bytes and int64 row end offsets."""

    data: np.ndarray
    ends: np.ndarray

    @classmethod
    def encode(cls, values: tuple[str, ...], /) -> TextColumnArchiveRecipe:
        if any(not isinstance(value, str) for value in values):
            raise TypeError("Text columns hold only strings.")
        return cls(*_encode_text_data(values, len(values)))

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
                or np.any(ends[1:] < ends[:-1])
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


class IntegerRowsArchiveRecipe(NamedTuple):
    """Variable-length integer rows as concatenated int64 values and row ends."""

    values: np.ndarray
    ends: np.ndarray

    @classmethod
    def encode(cls, rows: tuple[tuple[int, ...], ...], /) -> IntegerRowsArchiveRecipe:
        if any(not isinstance(row, tuple) for row in rows):
            raise TypeError("Integer rows must be tuples.")
        ends = np.cumsum(
            np.asarray([len(row) for row in rows], dtype=np.int64), dtype=np.int64
        ).reshape((len(rows),))
        return cls(integer_column(tuple(value for row in rows for value in row)), ends)

    def decode(self) -> tuple[tuple[int, ...], ...]:
        values = integer_column_values(self.values)
        ends = integer_column_values(self.ends)
        if (not ends and values) or (
            ends
            and (
                ends[0] < 0
                or any(last < first for first, last in zip(ends, ends[1:]))
                or ends[-1] != len(values)
            )
        ):
            raise ValueError("Integer row ends must partition their values.")
        return tuple(
            values[start:end] for start, end in zip((0, *ends)[:-1], ends, strict=True)
        )


class TextRowsArchiveRecipe(NamedTuple):
    """Variable-length text rows with a first-occurrence UTF-8 value table."""

    table: TextColumnArchiveRecipe
    rows: IntegerRowsArchiveRecipe

    @classmethod
    def encode(cls, rows: tuple[tuple[str, ...], ...], /) -> TextRowsArchiveRecipe:
        if any(not isinstance(row, tuple) for row in rows):
            raise TypeError("Text rows must be tuples.")
        column = StaticColumnArchiveRecipe.encode(
            tuple(value for row in rows for value in row)
        )
        ends = np.cumsum(
            np.asarray([len(row) for row in rows], dtype=np.int64), dtype=np.int64
        )
        return cls(
            TextColumnArchiveRecipe.encode(column.table),
            IntegerRowsArchiveRecipe(column.rows, ends),
        )

    def decode(self) -> tuple[tuple[str, ...], ...]:
        if (
            type(self.table) is not TextColumnArchiveRecipe
            or type(self.rows) is not IntegerRowsArchiveRecipe
        ):
            raise TypeError("Text rows require their exact table and row recipes.")
        rows = self.rows.decode()
        values = StaticColumnArchiveRecipe(self.table.decode(), self.rows.values).decode()
        position = 0
        result = []
        for row in rows:
            result.append(values[position : position + len(row)])
            position += len(row)
        return tuple(result)


class ExactTensorArchiveRecipe(NamedTuple):
    """Regular nested tuples of exact integers or rationals as canonical decimal text.

    Each leaf is stored as ``str(value)`` (``n`` or ``n/d`` in lowest terms), so
    integers of any size and exact rationals round-trip without binary64 rounding.
    """

    text: TextColumnArchiveRecipe
    shape: tuple[int, ...]
    rational: bool

    @classmethod
    def encode(
        cls, value: tuple[Any, ...], /, *, rational: bool
    ) -> ExactTensorArchiveRecipe:
        leaf_type = Fraction if rational else int
        shape: list[int] = []
        node: Any = value
        while isinstance(node, tuple):
            shape.append(len(node))
            if not node:
                break
            node = node[0]

        def visit(item: Any, depth: int, /) -> Iterator[str]:
            if depth == len(shape):
                if type(item) is not leaf_type:
                    raise TypeError("Exact tensor leaves must share one exact type.")
                yield str(item)
                return
            if not isinstance(item, tuple) or len(item) != shape[depth]:
                raise ValueError("Exact tensor rows must have one regular shape.")
            for child in item:
                yield from visit(child, depth + 1)

        extent = 1
        containers = 0
        for size in shape:
            containers += extent
            extent *= size
        if extent:
            text = TextColumnArchiveRecipe(*_encode_text_data(visit(value, 0), extent))
        else:
            # Validate even empty rows before encoding their allocation evidence.
            for _ in visit(value, 0):
                raise ValueError("An empty exact tensor cannot contain scalar leaves.")
            text = TextColumnArchiveRecipe(
                *_encode_text_data(repeat("", containers), containers)
            )
        return cls(text, tuple(shape), rational)

    def decode(self) -> tuple[Any, ...]:
        if not isinstance(self.shape, tuple) or any(
            isinstance(size, bool) or not isinstance(size, int) or size < 0
            for size in self.shape
        ):
            raise TypeError("Exact tensor shape must be a tuple of sizes.")
        if not self.shape or type(self.rational) is not bool:
            raise TypeError("Exact tensor must be a tuple with one exact leaf type.")
        texts = self.text.decode()
        extent = 1
        containers = 0
        for size in self.shape:
            containers += extent
            extent *= size
        if extent == 0:
            if len(texts) != containers or any(texts):
                raise ValueError("Empty exact tensor has invalid container evidence.")
            leaves = []
        else:
            if len(texts) != extent:
                raise ValueError("Exact tensor text does not match its shape.")
            leaves = [Fraction(text) if self.rational else int(text) for text in texts]
            if any(str(leaf) != text for leaf, text in zip(leaves, texts, strict=True)):
                raise ValueError("Exact tensor text must be canonical.")
        position = 0

        def build(depth: int, /) -> Any:
            nonlocal position
            if depth == len(self.shape):
                position += 1
                return leaves[position - 1]
            return tuple(build(depth + 1) for _ in range(self.shape[depth]))

        return build(0)


__all__ = [
    "ExactTensorArchiveRecipe",
    "integer_column",
    "integer_column_values",
    "IntegerRowsArchiveRecipe",
    "StaticColumnArchiveRecipe",
    "TextColumnArchiveRecipe",
    "TextRowsArchiveRecipe",
]
