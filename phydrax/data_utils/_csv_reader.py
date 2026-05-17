#
#  Copyright 2026 PHYDRA, Inc. All rights reserved.
#

from pathlib import Path
from typing import Any, overload

import equinox as eqx
import jax.numpy as jnp
import polars as pl
from jaxtyping import Array

from .._strict import StrictModule
from ._utils import _is_numeric_dtype


CSVValue = Array | list[Any]
CSVSelection = Array | dict[str, CSVValue]


class CSVReader(StrictModule):
    """Read CSV files into convenient JAX-friendly column accessors.

    Numeric columns are returned as JAX arrays. Non-numeric columns are returned
    as Python lists. Multiple numeric columns are returned as a two-dimensional
    JAX array with shape `(num_rows, num_columns)`.
    """

    _data: pl.DataFrame = eqx.field(static=True)

    def __init__(self, filepath: str | Path, **read_csv_kwargs: Any):
        """Construct a CSV reader.

        **Arguments:**

        - `filepath`: Path to the CSV file.
        - `**read_csv_kwargs`: Additional keyword arguments forwarded to
          `polars.read_csv`.
        """
        path = Path(filepath).resolve(strict=True)
        self._data = pl.read_csv(path, **read_csv_kwargs)

    def _column_values(self, column: str, /) -> CSVValue:
        series = self._data.get_column(column)
        if _is_numeric_dtype(series):
            return jnp.asarray(series.to_numpy())
        return series.to_list()

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

        columns = list(key)
        frame = self._data.select(columns)
        if _is_numeric_dtype(frame):
            return jnp.asarray(frame.to_numpy())
        return {column: self._column_values(column) for column in columns}

    def __len__(self) -> int:
        return self._data.height

    @property
    def columns(self) -> list[str]:
        """Return the CSV column names."""
        return list(self._data.columns)

    def to_array(self) -> Array:
        """Convert the full CSV table to a JAX array.

        Raises:
            TypeError: If any column is non-numeric.
        """
        if _is_numeric_dtype(self._data):
            return jnp.asarray(self._data.to_numpy())
        raise TypeError(
            "Cannot convert non-numeric CSV data to a JAX array. "
            "Use `to_dict()` for mixed data."
        )

    def to_dict(self) -> dict[str, CSVValue]:
        """Convert the CSV table to a column dictionary."""
        return {column: self._column_values(column) for column in self.columns}
