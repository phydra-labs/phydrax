#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

import equinox as eqx
import polars as pl

from ..._fingerprint import canonical_fingerprint, canonical_json
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


def _text(value: str, name: str, /) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty canonical identifier.")
    return value


def _names(values: Sequence[str], name: str, /) -> tuple[str, ...]:
    if not isinstance(values, Sequence) or isinstance(values, str):
        raise TypeError(f"{name} must be a sequence of strings.")
    result = tuple(_text(value, name) for value in values)
    if not result:
        raise ValueError(f"{name} must not be empty.")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must be unique.")
    return result


def _decode_cell(value: str, /) -> object:
    return json.loads(value, parse_constant=_reject_nonfinite)


def _reject_nonfinite(value: str, /) -> object:
    raise ValueError(f"Non-finite JSON constant {value!r} is not supported.")


class FinanceRecordBatch(StrictModule, NonTrainableState):
    """Canonical finite-JSON table independent of any dataframe implementation."""

    record_kind: str = eqx.field(static=True)
    columns: tuple[str, ...] = eqx.field(static=True)
    primary_key: tuple[str, ...] = eqx.field(static=True)
    encoded_rows: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    context_json: str = eqx.field(static=True)
    batch_id: str = eqx.field(static=True)

    def __init__(
        self,
        record_kind: str,
        records: Sequence[Mapping[str, Any]],
        /,
        *,
        primary_key: Sequence[str],
        context: Mapping[str, Any] | None = None,
    ):
        kind = _text(record_kind, "record_kind")
        if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
            raise TypeError("records must be a sequence of mappings.")
        values = tuple(records)
        if not values or any(not isinstance(record, Mapping) for record in values):
            raise TypeError("records must contain at least one mapping.")
        columns = tuple(sorted(_text(str(name), "column name") for name in values[0]))
        if not columns or any(set(record) != set(columns) for record in values):
            raise ValueError(
                "Every finance record must contain the same non-empty columns."
            )
        key = _names(primary_key, "primary_key")
        if not set(key).issubset(columns):
            raise ValueError("Every primary-key field must be a record column.")
        encoded = tuple(
            tuple(canonical_json(record[column]) for column in columns)
            for record in values
        )
        indices = tuple(columns.index(name) for name in key)
        encoded = tuple(
            sorted(
                encoded,
                key=lambda row: tuple(row[index] for index in indices),
            )
        )
        keys = tuple(tuple(row[index] for index in indices) for row in encoded)
        if len(set(keys)) != len(keys):
            raise ValueError("Finance record primary keys must be unique.")
        if context is not None and not isinstance(context, Mapping):
            raise TypeError("context must be a mapping or None.")
        context_json = canonical_json({} if context is None else dict(context))
        content = {
            "kind": "finance-record-batch",
            "record_kind": kind,
            "columns": list(columns),
            "primary_key": list(key),
            "rows": [list(row) for row in encoded],
            "context": json.loads(context_json),
        }
        self.record_kind = kind
        self.columns = columns
        self.primary_key = key
        self.encoded_rows = encoded
        self.context_json = context_json
        self.batch_id = canonical_fingerprint(content)

    @property
    def row_count(self) -> int:
        return len(self.encoded_rows)

    def context(self) -> Mapping[str, object]:
        """Return a read-only decoded batch context."""
        value = _decode_cell(self.context_json)
        if not isinstance(value, dict):
            raise RuntimeError("Finance record batch context invariant was violated.")
        return MappingProxyType(value)

    def to_records(self) -> tuple[Mapping[str, object], ...]:
        """Decode rows into immutable mappings in canonical primary-key order."""
        return tuple(
            MappingProxyType(
                {
                    column: _decode_cell(cell)
                    for column, cell in zip(self.columns, row, strict=True)
                }
            )
            for row in self.encoded_rows
        )

    def to_record(self) -> dict[str, object]:
        """Return a deterministic JSON-ready table representation."""
        return {
            "kind": "finance-record-batch",
            "record_kind": self.record_kind,
            "columns": list(self.columns),
            "primary_key": list(self.primary_key),
            "rows": [list(row) for row in self.encoded_rows],
            "context": _decode_cell(self.context_json),
            "batch_id": self.batch_id,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, object], /) -> FinanceRecordBatch:
        """Reconstruct and content-verify a serialized record batch."""
        if not isinstance(record, Mapping):
            raise TypeError("Finance record batch record must be a mapping.")
        expected = {
            "kind",
            "record_kind",
            "columns",
            "primary_key",
            "rows",
            "context",
            "batch_id",
        }
        if set(record) != expected or record["kind"] != "finance-record-batch":
            raise ValueError("Finance record batch fields are not canonical.")
        columns = record["columns"]
        primary_key = record["primary_key"]
        rows = record["rows"]
        context = record["context"]
        if not isinstance(columns, Sequence) or isinstance(columns, str):
            raise TypeError("Serialized columns must be a sequence.")
        if not isinstance(primary_key, Sequence) or isinstance(primary_key, str):
            raise TypeError("Serialized primary_key must be a sequence.")
        if not isinstance(rows, Sequence) or isinstance(rows, str):
            raise TypeError("Serialized rows must be a sequence.")
        column_names = tuple(str(item) for item in columns)
        decoded: list[dict[str, object]] = []
        for row in rows:
            if not isinstance(row, Sequence) or isinstance(row, str):
                raise TypeError("Serialized rows must contain cell sequences.")
            if len(row) != len(column_names) or any(
                not isinstance(cell, str) for cell in row
            ):
                raise ValueError(
                    "Serialized rows must align with columns as JSON strings."
                )
            decoded.append(
                {
                    name: _decode_cell(cell)
                    for name, cell in zip(column_names, row, strict=True)
                }
            )
        if not isinstance(context, Mapping):
            raise TypeError("Serialized context must be a mapping.")
        value = cls(
            str(record["record_kind"]),
            decoded,
            primary_key=tuple(str(item) for item in primary_key),
            context=context,
        )
        if value.columns != column_names or record["batch_id"] != value.batch_id:
            raise ValueError("Serialized finance record batch identity is invalid.")
        return value


def polars_to_market_records(
    frame: pl.DataFrame,
    /,
    *,
    primary_key: Sequence[str],
    context: Mapping[str, Any] | None = None,
) -> FinanceRecordBatch:
    """Detach a Polars table into canonical market records."""
    if not isinstance(frame, pl.DataFrame):
        raise TypeError("frame must be a polars.DataFrame.")
    return FinanceRecordBatch(
        "market-observation",
        frame.to_dicts(),
        primary_key=primary_key,
        context=context,
    )


def market_records_to_polars(batch: FinanceRecordBatch, /) -> pl.DataFrame:
    """Materialize canonical market records as a new Polars dataframe."""
    if not isinstance(batch, FinanceRecordBatch):
        raise TypeError("batch must be a FinanceRecordBatch.")
    if batch.record_kind != "market-observation":
        raise ValueError("Only market-observation batches can become market tables.")
    return pl.DataFrame([dict(record) for record in batch.to_records()])


__all__ = [
    "FinanceRecordBatch",
    "market_records_to_polars",
    "polars_to_market_records",
]
