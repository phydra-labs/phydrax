#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Integer-dyadic exact coordinate bank with rows materialized on demand."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from fractions import Fraction
from typing import Any, final, overload, TYPE_CHECKING

import numpy as np
from numpy.typing import DTypeLike, NDArray

from ..geometry._rational import binary64_dyadic_entries, dyadic_entry
from ._exact_algebraic_geometry import coefficient_payload, RadicalElement


if TYPE_CHECKING:
    from ._coordinate_enclosure import ExactScalar


# Exponent marking an entry held in the sparse table (not an int16 dyadic exponent).
_SPARSE = -32768
_EXPONENT_LIMIT = 32767
_INT64_LOWER = -(1 << 63)
_INT64_UPPER = (1 << 63) - 1
# |significand| <= 2**53 converts to binary64 exactly; larger ones round once.
_BINARY64_EXACT = 1 << 53
_SMALLEST_NORMAL = np.finfo(np.float64).tiny


def _dyadic_value(significand: int, exponent: int, /) -> Fraction:
    if exponent >= 0:
        return Fraction(significand << exponent)
    return Fraction(significand, 1 << -exponent)


@final
class ExactCoordinateBank(Sequence[tuple["ExactScalar", ...]]):
    """Immutable exact coordinate rows on the canonical integer-dyadic substrate.

    Each dyadic rational entry ``s * 2**e`` keeps its odd int64 significand
    ``s`` (``0`` for zero) and int16 exponent ``e``. Rationals outside that range
    and every ``RadicalElement`` live in a sparse table keyed by flat entry
    index. Rows are built as ``Fraction``/``RadicalElement`` tuples only when
    indexed. Equality is equality of the per-entry ``coefficient_payload`` (the
    canonical parts are compared directly) and the hash is cached.
    """

    __slots__ = (
        "_significands",
        "_exponents",
        "_sparse_index",
        "_sparse_values",
        "_shape",
        "_hash",
    )

    _significands: NDArray[np.int64]
    _exponents: NDArray[np.int16]
    _sparse_index: NDArray[np.int64]
    _sparse_values: tuple[ExactScalar, ...]
    _shape: tuple[int, int]
    _hash: int

    def __init__(
        self,
        significands: NDArray[np.int64],
        exponents: NDArray[np.int16],
        sparse_index: NDArray[np.int64],
        sparse_values: tuple[ExactScalar, ...],
        /,
    ) -> None:
        """Bind canonical parts; use ``from_rows`` or ``from_binary64`` to build."""
        sig = np.array(significands, dtype=np.int64, copy=True)
        exp = np.array(exponents, dtype=np.int16, copy=True)
        index = np.array(sparse_index, dtype=np.int64, copy=True).reshape(-1)
        values = tuple(sparse_values)
        if sig.ndim != 2 or exp.shape != sig.shape:
            raise ValueError("Exact coordinate bank parts must be one rank-2 shape.")
        flat = exp.reshape(-1)
        marked = np.flatnonzero(flat == _SPARSE)
        if (
            index.shape != marked.shape
            or not np.array_equal(index, marked)
            or len(values) != index.size
            or np.any(sig.reshape(-1)[marked] != 0)
        ):
            raise ValueError("Exact coordinate bank sparse entries must match their marks.")
        dense = flat != _SPARSE
        dense_sig = sig.reshape(-1)[dense]
        dense_exp = flat[dense]
        if np.any((dense_sig == 0) & (dense_exp != 0)) or np.any(
            (dense_sig != 0) & (dense_sig % 2 == 0)
        ):
            raise ValueError("Exact coordinate bank significands must be canonical.")
        for value in values:
            if isinstance(value, Fraction):
                entry = dyadic_entry(value)
                if entry is not None and _dense(*entry):
                    raise ValueError(
                        "A dense dyadic entry cannot be held in the sparse table."
                    )
            elif not isinstance(value, RadicalElement):
                raise TypeError("Exact coordinate bank entries must be ExactScalar.")
        for array in (sig, exp, index):
            array.setflags(write=False)
        shape = (int(sig.shape[0]), int(sig.shape[1]))
        for name, part in (
            ("_significands", sig),
            ("_exponents", exp),
            ("_sparse_index", index),
            ("_sparse_values", values),
            ("_shape", shape),
            (
                "_hash",
                hash((shape, sig.tobytes(), exp.tobytes(), index.tobytes(), values)),
            ),
        ):
            object.__setattr__(self, name, part)

    def __setattr__(self, name: str, value: object, /) -> None:
        raise AttributeError("ExactCoordinateBank is immutable.")

    def __delattr__(self, name: str, /) -> None:
        raise AttributeError("ExactCoordinateBank is immutable.")

    @classmethod
    def from_rows(
        cls, rows: Sequence[Sequence[ExactScalar]], /
    ) -> ExactCoordinateBank:
        """Pack regular rows of ``Fraction``/``RadicalElement`` values exactly."""
        if isinstance(rows, ExactCoordinateBank):
            return rows
        materialized = tuple(rows)
        width = len(materialized[0]) if materialized else 0
        significands: list[int] = []
        exponents: list[int] = []
        sparse_index: list[int] = []
        sparse_values: list[ExactScalar] = []
        for row in materialized:
            if len(row) != width:
                raise ValueError("Exact coordinate rows must share one width.")
            for value in row:
                if isinstance(value, Fraction):
                    entry = dyadic_entry(value)
                    if entry is not None and _dense(*entry):
                        significands.append(entry[0])
                        exponents.append(entry[1])
                        continue
                elif not isinstance(value, RadicalElement):
                    raise TypeError(
                        "Exact coordinate rows must contain Fraction or RadicalElement values."
                    )
                sparse_index.append(len(significands))
                sparse_values.append(value)
                significands.append(0)
                exponents.append(_SPARSE)
        shape = (len(materialized), width)
        return cls(
            np.asarray(significands, dtype=np.int64).reshape(shape),
            np.asarray(exponents, dtype=np.int16).reshape(shape),
            np.asarray(sparse_index, dtype=np.int64),
            tuple(sparse_values),
        )

    @classmethod
    def from_binary64(cls, values: NDArray[np.float64], /) -> ExactCoordinateBank:
        """Exact rows of finite binary64 coordinates (each entry is dyadic)."""
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2:
            raise ValueError("Binary64 coordinate banks must be rank-2 arrays.")
        significands, exponents = binary64_dyadic_entries(array)
        return cls(
            significands,
            exponents.astype(np.int16),
            np.empty((0,), dtype=np.int64),
            (),
        )

    @property
    def shape(self) -> tuple[int, int]:
        """``(rows, ambient dimension)``."""
        return self._shape

    @property
    def rational(self) -> bool:
        """Whether every entry is a ``Fraction``."""
        return all(isinstance(value, Fraction) for value in self._sparse_values)

    def __len__(self) -> int:
        return self._shape[0]

    def _sparse(self, flat: int, /) -> ExactScalar:
        return self._sparse_values[int(np.searchsorted(self._sparse_index, flat))]

    def _entry(self, flat: int, significand: int, exponent: int, /) -> ExactScalar:
        if exponent == _SPARSE:
            return self._sparse(flat)
        return _dyadic_value(significand, exponent)

    def _row(self, row: int, /) -> tuple[ExactScalar, ...]:
        width = self._shape[1]
        base = row * width
        return tuple(
            self._entry(base + axis, significand, exponent)
            for axis, (significand, exponent) in enumerate(
                zip(
                    self._significands[row].tolist(),
                    self._exponents[row].tolist(),
                    strict=True,
                )
            )
        )

    @overload
    def __getitem__(self, index: int, /) -> tuple[ExactScalar, ...]: ...

    @overload
    def __getitem__(self, index: slice, /) -> tuple[tuple[ExactScalar, ...], ...]: ...

    def __getitem__(
        self, index: int | slice, /
    ) -> tuple[ExactScalar, ...] | tuple[tuple[ExactScalar, ...], ...]:
        if isinstance(index, slice):
            return tuple(self._row(row) for row in range(*index.indices(len(self))))
        row = int(index)
        if row != index or isinstance(index, bool):
            raise TypeError("Exact coordinate bank rows are indexed by integers.")
        if row < 0:
            row += len(self)
        if not 0 <= row < len(self):
            raise IndexError("Exact coordinate bank row is out of range.")
        return self._row(row)

    def __iter__(self) -> Iterator[tuple[ExactScalar, ...]]:
        for row in range(len(self)):
            yield self._row(row)

    def rows(self, indices: NDArray[np.integer[Any]], /) -> tuple[tuple[ExactScalar, ...], ...]:
        """Materialize the selected rows (bounds-checked int vector)."""
        selected = np.asarray(indices)
        if selected.ndim != 1 or selected.dtype.kind not in "iu":
            raise TypeError("Exact coordinate bank rows must be an integer vector.")
        if np.any(selected < 0) or np.any(selected >= len(self)):
            raise IndexError("Exact coordinate bank row is out of range.")
        return tuple(self._row(int(row)) for row in selected.tolist())

    def take(self, indices: NDArray[np.integer[Any]], /) -> ExactCoordinateBank:
        """The bank of the selected rows, without materializing them."""
        selected = np.asarray(indices)
        if selected.ndim != 1 or selected.dtype.kind not in "iu":
            raise TypeError("Exact coordinate bank rows must be an integer vector.")
        if np.any(selected < 0) or np.any(selected >= len(self)):
            raise IndexError("Exact coordinate bank row is out of range.")
        rows = selected.astype(np.int64)
        significands = self._significands[rows]
        exponents = self._exponents[rows]
        marked = np.flatnonzero(exponents.reshape(-1) == _SPARSE)
        width = self._shape[1]
        values = tuple(
            self._sparse(int(rows[flat // width]) * width + flat % width)
            for flat in marked.tolist()
        )
        return ExactCoordinateBank(significands, exponents, marked, values)

    def replace_rows(
        self, indices: NDArray[np.integer[Any]], replacement: ExactCoordinateBank, /
    ) -> ExactCoordinateBank:
        """A bank equal to this one with the indexed rows taken from ``replacement``."""
        selected = np.asarray(indices)
        if (
            selected.ndim != 1
            or selected.dtype.kind not in "iu"
            or replacement.shape != (selected.size, self._shape[1])
        ):
            raise ValueError("Replacement rows must align with their row indices.")
        rows = selected.astype(np.int64)
        if np.any(rows < 0) or np.any(rows >= len(self)) or np.unique(rows).size != rows.size:
            raise IndexError("Replacement rows must be unique rows of the bank.")
        significands = np.array(self._significands, copy=True)
        exponents = np.array(self._exponents, copy=True)
        significands[rows] = replacement._significands
        exponents[rows] = replacement._exponents
        width = self._shape[1]
        source_row = np.full((len(self),), -1, dtype=np.int64)
        source_row[rows] = np.arange(rows.size, dtype=np.int64)
        marked = np.flatnonzero(exponents.reshape(-1) == _SPARSE)
        values: list[ExactScalar] = []
        for flat in marked.tolist():
            row, axis = divmod(flat, width)
            if source_row[row] >= 0:
                origin, bank = int(source_row[row]) * width + axis, replacement
            else:
                origin, bank = flat, self
            values.append(bank._sparse(origin))
        return ExactCoordinateBank(significands, exponents, marked, tuple(values))

    def binary64(self) -> NDArray[np.float64]:
        """Correctly rounded binary64 value of every entry (``float`` of each value)."""
        significands = self._significands
        exponents = self._exponents.astype(np.int64)
        dense = exponents != _SPARSE
        with np.errstate(over="ignore"):
            result = np.ldexp(
                significands.astype(np.float64), np.where(dense, exponents, 0)
            )
        # Wide significands round once on conversion; a subnormal or overflowing
        # result would round twice, so those entries use exact division.
        wide = np.abs(significands) > _BINARY64_EXACT
        recheck = ~dense | (
            wide & ((np.abs(result) < _SMALLEST_NORMAL) | ~np.isfinite(result))
        )
        width = self._shape[1]
        for flat in np.flatnonzero(recheck.reshape(-1)).tolist():
            row, axis = divmod(flat, width)
            result[row, axis] = float(
                self._entry(
                    flat, int(significands[row, axis]), int(self._exponents[row, axis])
                )
            )
        if not np.all(np.isfinite(result)):
            raise OverflowError("Exact coordinate is too large for binary64.")
        return result

    def coefficient_payloads(
        self,
    ) -> tuple[tuple[tuple[str, int, int] | tuple[str, tuple[tuple[int, int, int], ...]], ...], ...]:
        """``coefficient_payload`` of every entry, row by row, without Fractions."""
        width = self._shape[1]
        result = []
        for row, (significand_row, exponent_row) in enumerate(
            zip(self._significands.tolist(), self._exponents.tolist(), strict=True)
        ):
            payload: list[
                tuple[str, int, int] | tuple[str, tuple[tuple[int, int, int], ...]]
            ] = []
            for axis, (significand, exponent) in enumerate(
                zip(significand_row, exponent_row, strict=True)
            ):
                if exponent == _SPARSE:
                    payload.append(
                        coefficient_payload(self._entry(row * width + axis, 0, exponent))
                    )
                elif exponent >= 0:
                    payload.append(("rational", significand << exponent, 1))
                else:
                    payload.append(("rational", significand, 1 << -exponent))
            result.append(tuple(payload))
        return tuple(result)

    def __array__(
        self, dtype: DTypeLike | None = None, copy: bool | None = None
    ) -> np.ndarray:
        del copy
        if dtype is None or np.dtype(dtype) == np.dtype(object):
            values = np.empty(self._shape if len(self) else (0,), dtype=object)
            for row in range(len(self)):
                values[row] = self._row(row)
            return values
        if not len(self):
            return np.empty((0,), dtype=dtype)
        return self.binary64().astype(dtype, copy=False)

    def __eq__(self, other: object, /) -> bool:
        if not isinstance(other, ExactCoordinateBank):
            return NotImplemented
        if other is self:
            return True
        return (
            other._shape == self._shape
            and np.array_equal(self._significands, other._significands)
            and np.array_equal(self._exponents, other._exponents)
            and np.array_equal(self._sparse_index, other._sparse_index)
            and self._sparse_values == other._sparse_values
        )

    def __hash__(self) -> int:
        return self._hash

    def __repr__(self) -> str:
        return (
            f"ExactCoordinateBank(shape={self._shape}, sparse={len(self._sparse_values)})"
        )

    def __reduce__(self) -> tuple[Any, ...]:
        return (
            ExactCoordinateBank,
            (
                self._significands,
                self._exponents,
                self._sparse_index,
                self._sparse_values,
            ),
        )


def _dense(significand: int, exponent: int, /) -> bool:
    return (
        _INT64_LOWER <= significand <= _INT64_UPPER
        and -_EXPONENT_LIMIT <= exponent <= _EXPONENT_LIMIT
    )


__all__ = ["ExactCoordinateBank"]
