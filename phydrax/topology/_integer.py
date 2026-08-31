#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._coefficients import PrimeField
from ._complex import CompactBoundary
from ._resources import TopologyResourceError, TopologyResourcePolicy


class ExactIntegerCOO(StrictModule, NonTrainableState):
    """Canonical sparse integer matrix with arbitrary-precision host coefficients."""

    row_indices: Array
    column_indices: Array
    coefficients: tuple[int, ...] = eqx.field(static=True)
    row_count: int = eqx.field(static=True)
    column_count: int = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    target_id: str = eqx.field(static=True)
    matrix_id: str = eqx.field(static=True)

    def __init__(
        self,
        row_count: int,
        column_count: int,
        row_indices: ArrayLike,
        column_indices: ArrayLike,
        coefficients: Sequence[int],
        /,
        *,
        source_id: str,
        target_id: str,
    ):
        rows_count = int(row_count)
        columns_count = int(column_count)
        if rows_count < 0 or columns_count < 0:
            raise ValueError("Exact integer matrix dimensions must be non-negative.")
        rows = np.asarray(row_indices)
        columns = np.asarray(column_indices)
        values = tuple(int(value) for value in coefficients)
        if rows.ndim != 1 or columns.ndim != 1:
            raise ValueError("Exact integer matrix indices must be rank-1.")
        if rows.shape != columns.shape or rows.size != len(values):
            raise ValueError("Exact integer matrix coordinate arrays must align.")
        if not np.issubdtype(rows.dtype, np.integer) or not np.issubdtype(
            columns.dtype, np.integer
        ):
            raise TypeError("Exact integer matrix indices require integer dtypes.")
        if np.any(rows < 0) or np.any(rows >= rows_count):
            raise ValueError("Exact integer matrix row index is out of bounds.")
        if np.any(columns < 0) or np.any(columns >= columns_count):
            raise ValueError("Exact integer matrix column index is out of bounds.")
        accumulator: dict[tuple[int, int], int] = {}
        for row, column, value in zip(rows, columns, values, strict=True):
            key = (int(row), int(column))
            accumulator[key] = accumulator.get(key, 0) + value
        entries = tuple(
            (row, column, value)
            for (row, column), value in sorted(accumulator.items())
            if value
        )
        normalized_rows = np.asarray([value[0] for value in entries], dtype=np.int32)
        normalized_columns = np.asarray([value[1] for value in entries], dtype=np.int32)
        normalized_values = tuple(value[2] for value in entries)
        source = str(source_id)
        target = str(target_id)
        if not source or not target:
            raise ValueError("Exact integer matrix source and target IDs are required.")
        self.row_indices = jnp.asarray(normalized_rows)
        self.column_indices = jnp.asarray(normalized_columns)
        self.coefficients = normalized_values
        self.row_count = rows_count
        self.column_count = columns_count
        self.source_id = source
        self.target_id = target
        self.matrix_id = canonical_fingerprint(
            {
                "kind": "exact-integer-coo",
                "shape": [rows_count, columns_count],
                "source": source,
                "target": target,
                "rows": array_tree_fingerprint(normalized_rows),
                "columns": array_tree_fingerprint(normalized_columns),
                "coefficients": list(normalized_values),
            }
        )

    @classmethod
    def zero(
        cls,
        row_count: int,
        column_count: int,
        /,
        *,
        source_id: str,
        target_id: str,
    ) -> "ExactIntegerCOO":
        return cls(
            row_count,
            column_count,
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.int32),
            (),
            source_id=source_id,
            target_id=target_id,
        )

    @classmethod
    def identity(cls, size: int, /, *, coordinate_id: str) -> "ExactIntegerCOO":
        count = int(size)
        indices = np.arange(count, dtype=np.int32)
        return cls(
            count,
            count,
            indices,
            indices,
            (1,) * count,
            source_id=coordinate_id,
            target_id=coordinate_id,
        )

    @classmethod
    def from_boundary(cls, boundary: CompactBoundary, /) -> "ExactIntegerCOO":
        return cls(
            boundary.row_count,
            boundary.column_count,
            boundary.row_indices,
            boundary.column_indices,
            tuple(int(value) for value in np.asarray(boundary.coefficients)),
            source_id=f"{boundary.source_id}:degree:{boundary.degree}",
            target_id=f"{boundary.source_id}:degree:{boundary.degree - 1}",
        )

    @property
    def nonzero_count(self) -> int:
        return len(self.coefficients)

    @property
    def maximum_bit_length(self) -> int:
        return max((abs(value).bit_length() for value in self.coefficients), default=0)

    def entries(self, /) -> tuple[tuple[int, int, int], ...]:
        return tuple(
            (int(row), int(column), value)
            for row, column, value in zip(
                np.asarray(self.row_indices),
                np.asarray(self.column_indices),
                self.coefficients,
                strict=True,
            )
        )

    def columns(self, /) -> tuple[dict[int, int], ...]:
        values = [dict() for _ in range(self.column_count)]
        for row, column, coefficient in self.entries():
            values[column][row] = coefficient
        return tuple(values)

    def transpose(self, /) -> "ExactIntegerCOO":
        return ExactIntegerCOO(
            self.column_count,
            self.row_count,
            self.column_indices,
            self.row_indices,
            self.coefficients,
            source_id=self.target_id,
            target_id=self.source_id,
        )

    def compose(self, right: "ExactIntegerCOO", /) -> "ExactIntegerCOO":
        """Return ``self ∘ right`` under column-vector conventions."""
        if not isinstance(right, ExactIntegerCOO):
            raise TypeError("Exact matrix composition requires ExactIntegerCOO.")
        if right.row_count != self.column_count:
            raise ValueError("Exact matrix composition dimensions do not align.")
        if right.target_id != self.source_id:
            raise ValueError("Exact matrix composition coordinate IDs do not align.")
        left_columns = self.columns()
        accumulator: dict[tuple[int, int], int] = {}
        for target_column, right_column in enumerate(right.columns()):
            for middle, right_value in right_column.items():
                for row, left_value in left_columns[middle].items():
                    key = (row, target_column)
                    accumulator[key] = accumulator.get(key, 0) + left_value * right_value
        entries = tuple(
            (row, column, value)
            for (row, column), value in sorted(accumulator.items())
            if value
        )
        return ExactIntegerCOO(
            self.row_count,
            right.column_count,
            np.asarray([value[0] for value in entries], dtype=np.int32),
            np.asarray([value[1] for value in entries], dtype=np.int32),
            tuple(value[2] for value in entries),
            source_id=right.source_id,
            target_id=self.target_id,
        )

    def add(
        self,
        other: "ExactIntegerCOO",
        /,
        *,
        scale: int = 1,
    ) -> "ExactIntegerCOO":
        if not isinstance(other, ExactIntegerCOO):
            raise TypeError("Exact matrix addition requires ExactIntegerCOO.")
        if (
            self.row_count != other.row_count
            or self.column_count != other.column_count
            or self.source_id != other.source_id
            or self.target_id != other.target_id
        ):
            raise ValueError("Exact matrix addition requires identical coordinates.")
        rows = np.concatenate(
            (np.asarray(self.row_indices), np.asarray(other.row_indices))
        )
        columns = np.concatenate(
            (np.asarray(self.column_indices), np.asarray(other.column_indices))
        )
        values = self.coefficients + tuple(
            int(scale) * value for value in other.coefficients
        )
        return ExactIntegerCOO(
            self.row_count,
            self.column_count,
            rows,
            columns,
            values,
            source_id=self.source_id,
            target_id=self.target_id,
        )

    def scale(self, value: int, /) -> "ExactIntegerCOO":
        return ExactIntegerCOO(
            self.row_count,
            self.column_count,
            self.row_indices,
            self.column_indices,
            tuple(int(value) * coefficient for coefficient in self.coefficients),
            source_id=self.source_id,
            target_id=self.target_id,
        )

    def apply_integer(self, vector: Sequence[int], /) -> tuple[int, ...]:
        values = tuple(int(value) for value in vector)
        if len(values) != self.column_count:
            raise ValueError("Exact matrix vector has the wrong coordinate count.")
        output = [0] * self.row_count
        for row, column, coefficient in self.entries():
            output[row] += coefficient * values[column]
        return tuple(output)

    def apply_field(
        self,
        vector: Sequence[int],
        field: PrimeField,
        /,
    ) -> tuple[int, ...]:
        if not isinstance(field, PrimeField):
            raise TypeError("Field application requires a PrimeField.")
        return tuple(field.normalize(value) for value in self.apply_integer(vector))

    def dense(
        self,
        /,
        *,
        resources: TopologyResourcePolicy | None = None,
    ) -> np.ndarray:
        policy = TopologyResourcePolicy() if resources is None else resources
        entries = self.row_count * self.column_count
        if entries > policy.max_reduction_entries:
            raise TopologyResourceError(
                "Exact integer dense materialization exceeds max_reduction_entries."
            )
        matrix = np.zeros((self.row_count, self.column_count), dtype=object)
        for row, column, coefficient in self.entries():
            matrix[row, column] = coefficient
        return matrix

    def equals(self, other: "ExactIntegerCOO", /) -> bool:
        return (
            isinstance(other, ExactIntegerCOO)
            and self.row_count == other.row_count
            and self.column_count == other.column_count
            and self.source_id == other.source_id
            and self.target_id == other.target_id
            and self.entries() == other.entries()
        )


class ExactChainComplex(StrictModule, NonTrainableState):
    """Derived algebraic chain complex with exact integer differentials."""

    boundaries: tuple[ExactIntegerCOO, ...]
    counts: tuple[int, ...] = eqx.field(static=True)
    complex_id: str = eqx.field(static=True)

    def __init__(
        self,
        boundaries: Sequence[ExactIntegerCOO],
        /,
        *,
        complex_id: str,
    ):
        values = tuple(boundaries)
        if not values:
            raise ValueError("Exact chain complexes require at least degree zero.")
        if values[0].row_count != 0:
            raise ValueError("Degree-zero exact boundary must have zero rows.")
        counts = tuple(value.column_count for value in values)
        for degree, boundary in enumerate(values):
            expected_rows = 0 if degree == 0 else counts[degree - 1]
            if boundary.row_count != expected_rows:
                raise ValueError("Exact chain boundary dimensions do not align.")
            if degree and boundary.target_id != values[degree - 1].source_id:
                raise ValueError("Exact chain boundary coordinate IDs do not align.")
        for lower, upper in zip(values[:-1], values[1:], strict=True):
            composition = lower.compose(upper)
            if composition.nonzero_count:
                raise ValueError(
                    "Exact chain complex violates boundary-of-boundary zero."
                )
        identifier = str(complex_id)
        if not identifier:
            raise ValueError("Exact chain complex ID must be non-empty.")
        self.boundaries = values
        self.counts = counts
        self.complex_id = canonical_fingerprint(
            {
                "kind": "exact-chain-complex",
                "declared": identifier,
                "boundaries": [value.matrix_id for value in values],
            }
        )


def block_matrix(
    blocks: Mapping[tuple[int, int], ExactIntegerCOO],
    row_sizes: Sequence[int],
    column_sizes: Sequence[int],
    /,
    *,
    source_id: str,
    target_id: str,
) -> ExactIntegerCOO:
    """Assemble one exact block matrix from coordinate-compatible blocks."""
    rows_ = tuple(int(value) for value in row_sizes)
    columns_ = tuple(int(value) for value in column_sizes)
    row_offsets = np.cumsum((0,) + rows_[:-1], dtype=np.int64)
    column_offsets = np.cumsum((0,) + columns_[:-1], dtype=np.int64)
    rows = []
    columns = []
    coefficients = []
    for (block_row, block_column), matrix in blocks.items():
        if (
            matrix.row_count != rows_[block_row]
            or matrix.column_count != columns_[block_column]
        ):
            raise ValueError("Exact block matrix dimensions do not align.")
        for row, column, value in matrix.entries():
            rows.append(int(row_offsets[block_row]) + row)
            columns.append(int(column_offsets[block_column]) + column)
            coefficients.append(value)
    return ExactIntegerCOO(
        sum(rows_),
        sum(columns_),
        np.asarray(rows, dtype=np.int32),
        np.asarray(columns, dtype=np.int32),
        coefficients,
        source_id=source_id,
        target_id=target_id,
    )


__all__ = ["ExactChainComplex", "ExactIntegerCOO", "block_matrix"]
