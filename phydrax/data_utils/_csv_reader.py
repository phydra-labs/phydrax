#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, overload

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .._strict import StrictModule
from .._trainable import NonTrainableState


CSVValue = Array | list[Any]
CSVSelection = Array | dict[str, CSVValue]


class CSVReadPolicy(StrictModule, NonTrainableState):
    """Explicit host CSV syntax and header policy."""

    delimiter: str = eqx.field(static=True)
    quote_character: str = eqx.field(static=True)
    escape_character: str | None = eqx.field(static=True)
    has_header: bool = eqx.field(static=True)
    column_names: tuple[str, ...] | None = eqx.field(static=True)
    encoding: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        delimiter: str = ",",
        quote_character: str = '"',
        escape_character: str | None = None,
        has_header: bool = True,
        column_names: tuple[str, ...] | None = None,
        encoding: str = "utf-8",
    ):
        if not isinstance(delimiter, str) or len(delimiter) != 1:
            raise ValueError("delimiter must be one character.")
        if not isinstance(quote_character, str) or len(quote_character) != 1:
            raise ValueError("quote_character must be one character.")
        if escape_character is not None and (
            not isinstance(escape_character, str) or len(escape_character) != 1
        ):
            raise ValueError("escape_character must be one character or None.")
        if not isinstance(has_header, bool):
            raise TypeError("has_header must be a boolean.")
        names = None if column_names is None else tuple(column_names)
        if names is not None and (
            not names
            or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError("column_names must be nonempty unique strings.")
        if not isinstance(encoding, str) or not encoding:
            raise ValueError("encoding must be a nonempty string.")
        self.delimiter = delimiter
        self.quote_character = quote_character
        self.escape_character = escape_character
        self.has_header = has_header
        self.column_names = names
        self.encoding = encoding


def _numeric_column(values: tuple[str, ...], /) -> tuple[bool, tuple[Any, ...]]:
    try:
        return True, tuple(int(value) for value in values)
    except ValueError:
        pass
    try:
        converted = tuple(float(value) for value in values)
    except ValueError:
        return False, values
    if any(not np.isfinite(value) for value in converted):
        raise ValueError("Numeric CSV columns must contain only finite values.")
    return True, converted


class CSVReader(StrictModule, NonTrainableState):
    """Read a bounded CSV table into JAX-friendly immutable columns."""

    _columns: tuple[str, ...] = eqx.field(static=True)
    _values: tuple[tuple[Any, ...], ...] = eqx.field(static=True)
    _numeric: tuple[bool, ...] = eqx.field(static=True)

    def __init__(
        self,
        filepath: str | Path,
        /,
        *,
        policy: CSVReadPolicy | None = None,
    ):
        policy_ = CSVReadPolicy() if policy is None else policy
        if not isinstance(policy_, CSVReadPolicy):
            raise TypeError("policy must be a CSVReadPolicy or None.")
        path = Path(filepath).resolve(strict=True)
        with path.open("r", encoding=policy_.encoding, newline="") as stream:
            rows = tuple(
                tuple(row)
                for row in csv.reader(
                    stream,
                    delimiter=policy_.delimiter,
                    quotechar=policy_.quote_character,
                    escapechar=policy_.escape_character,
                    strict=True,
                )
            )
        if not rows:
            raise ValueError("CSV input must contain at least one row.")
        width = len(rows[0])
        if width == 0 or any(len(row) != width for row in rows):
            raise ValueError("Every CSV row must contain the same positive column count.")
        if policy_.has_header:
            parsed_names = rows[0]
            data_rows = rows[1:]
        else:
            parsed_names = tuple(f"column_{index + 1}" for index in range(width))
            data_rows = rows
        names = policy_.column_names or parsed_names
        if len(names) != width:
            raise ValueError("column_names must match the CSV column count.")
        if any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError("CSV column names must be nonempty and unique.")
        raw_columns = tuple(
            tuple(row[index] for row in data_rows) for index in range(width)
        )
        converted = tuple(_numeric_column(values) for values in raw_columns)
        self._columns = tuple(names)
        self._numeric = tuple(item[0] for item in converted)
        self._values = tuple(item[1] for item in converted)

    def _column_values(self, column: str, /) -> CSVValue:
        try:
            index = self._columns.index(column)
        except ValueError as error:
            raise KeyError(column) from error
        values = self._values[index]
        return jnp.asarray(values) if self._numeric[index] else list(values)

    @overload
    def __getitem__(self, key: str) -> CSVValue: ...

    @overload
    def __getitem__(self, key: list[str] | tuple[str, ...]) -> CSVSelection: ...

    def __getitem__(
        self,
        key: str | list[str] | tuple[str, ...],
    ) -> CSVValue | CSVSelection:
        if isinstance(key, str):
            return self._column_values(key)
        columns = tuple(key)
        if not columns:
            raise ValueError("CSV column selections must not be empty.")
        indices = tuple(self._columns.index(column) for column in columns)
        if all(self._numeric[index] for index in indices):
            return jnp.column_stack(
                tuple(jnp.asarray(self._values[index]) for index in indices)
            )
        values = tuple(self._column_values(column) for column in columns)
        return dict(zip(columns, values, strict=True))

    def __len__(self) -> int:
        return len(self._values[0])

    @property
    def columns(self) -> list[str]:
        """Return the CSV column names."""
        return list(self._columns)

    def to_array(self) -> Array:
        """Convert the complete numeric table to a rank-two JAX array."""
        if not all(self._numeric):
            raise TypeError(
                "Cannot convert non-numeric CSV data to a JAX array. "
                "Use to_dict() for mixed data."
            )
        return jnp.column_stack(tuple(jnp.asarray(values) for values in self._values))

    def to_dict(self) -> dict[str, CSVValue]:
        """Convert the CSV table to a column dictionary."""
        return {column: self._column_values(column) for column in self._columns}


__all__ = ["CSVReadPolicy", "CSVReader", "CSVSelection", "CSVValue"]
