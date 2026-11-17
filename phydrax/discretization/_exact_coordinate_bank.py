#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Integer-dyadic exact coordinate bank with rows materialized on demand."""

from __future__ import annotations

import weakref
from collections.abc import Iterator, Sequence
from fractions import Fraction
from typing import Any, final, overload, TYPE_CHECKING, TypeVar

import numpy as np
from jax.typing import ArrayLike
from numpy.typing import DTypeLike, NDArray

from .._fingerprint import canonical_fingerprint, numeric_array_fingerprint
from .._live_proofs import _ImmutableFingerprintLeaf, LiveProofLedger
from ..geometry._rational import binary64_dyadic_entries, dyadic_entry
from ._exact_algebraic_geometry import (
    canonical_exact_scalar,
    coefficient_payload,
    RadicalElement,
)


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

_Integer = TypeVar("_Integer", bound=np.integer[Any])
type _CoefficientPayload = (
    tuple[str, int, int] | tuple[str, tuple[tuple[int, int, int], ...]]
)
type _BankFingerprint = tuple[
    str,
    tuple[weakref.ReferenceType[NDArray[Any]], ...],
    tuple[_CoefficientPayload, ...],
    int,
]
_BANK_FINGERPRINTS: LiveProofLedger[_BankFingerprint] = LiveProofLedger()


def _sealed_array(array: NDArray[_Integer], /) -> NDArray[_Integer]:
    """Own immutable bytes directly, without a mutable array in the base chain."""
    sealed = np.ndarray(array.shape, dtype=array.dtype, buffer=array.tobytes(order="C"))
    if not isinstance(sealed.base, bytes) or sealed.flags.writeable:
        raise ValueError(
            "Exact coordinate bank arrays must be backed by immutable bytes."
        )
    return sealed


def _sealed_parts(
    arrays: tuple[NDArray[Any], ...], shape: tuple[int, int], sparse_count: int, /
) -> bool:
    if (
        type(shape) is not tuple
        or len(shape) != 2
        or any(type(size) is not int or size < 0 for size in shape)
    ):
        return False
    expected = (
        (np.dtype(np.int64), shape),
        (np.dtype(np.int16), shape),
        (np.dtype(np.int64), (sparse_count,)),
        (np.dtype(np.int64), (shape[0] + 1,)),
    )
    return all(
        type(array) is np.ndarray
        and isinstance(array.base, bytes)
        and not array.flags.writeable
        and array.flags.c_contiguous
        and array.dtype == dtype
        and array.shape == expected_shape
        for array, (dtype, expected_shape) in zip(arrays, expected, strict=True)
    )


def _immutable_sparse_values(values: tuple[ExactScalar, ...], /) -> bool:
    if type(values) is not tuple:
        return False
    for value in values:
        if type(value) is Fraction:
            continue
        if type(value) is not RadicalElement or type(value.terms) is not tuple:
            return False
        if any(
            type(term) is not tuple
            or len(term) != 2
            or type(term[0]) is not int
            or type(term[1]) is not Fraction
            for term in value.terms
        ):
            return False
    return True


def _dyadic_value(significand: int, exponent: int, /) -> Fraction:
    if exponent >= 0:
        return Fraction(significand << exponent)
    return Fraction(significand, 1 << -exponent)


@final
class ExactCoordinateBank(_ImmutableFingerprintLeaf, Sequence[tuple["ExactScalar", ...]]):
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
        "_sparse_row_starts",
        "_shape",
        "_hash",
        "__weakref__",
    )

    _significands: NDArray[np.int64]
    _exponents: NDArray[np.int16]
    _sparse_index: NDArray[np.int64]
    _sparse_values: tuple[ExactScalar, ...]
    _sparse_row_starts: NDArray[np.int64]
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
            raise ValueError(
                "Exact coordinate bank sparse entries must match their marks."
            )
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
            elif isinstance(value, RadicalElement):
                canonical = canonical_exact_scalar(
                    value, role="Exact coordinate bank entry"
                )
                if isinstance(canonical, Fraction):
                    raise ValueError(
                        "A rational radical cannot be held in the sparse table."
                    )
            else:
                raise TypeError("Exact coordinate bank entries must be ExactScalar.")
        shape = (int(sig.shape[0]), int(sig.shape[1]))
        # Sparse entries are stored in flat order, so one offset per row lets a
        # row read its sparse values in sequence without a search per entry.
        row_starts = np.searchsorted(
            index, np.arange(shape[0] + 1, dtype=np.int64) * shape[1]
        ).astype(np.int64)
        sig = _sealed_array(sig)
        exp = _sealed_array(exp)
        index = _sealed_array(index)
        row_starts = _sealed_array(row_starts)
        if not _sealed_parts((sig, exp, index, row_starts), shape, len(values)):
            raise ValueError(
                "Exact coordinate bank arrays must be backed by immutable bytes."
            )
        for name, part in (
            ("_significands", sig),
            ("_exponents", exp),
            ("_sparse_index", index),
            ("_sparse_values", values),
            ("_sparse_row_starts", row_starts),
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
    def from_rows(cls, rows: Sequence[Sequence[ExactScalar]], /) -> ExactCoordinateBank:
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
                if isinstance(value, RadicalElement):
                    value = canonical_exact_scalar(value, role="Exact coordinate row")
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
    def from_binary64(cls, values: ArrayLike, /) -> ExactCoordinateBank:
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

    def canonical_parts(
        self, /
    ) -> tuple[
        NDArray[np.int64],
        NDArray[np.int16],
        NDArray[np.int64],
        NDArray[np.int64],
        tuple[ExactScalar, ...],
    ]:
        """Read-only dyadic arrays, sparse indices, row offsets, and exact values."""
        return (
            self._significands.view(),
            self._exponents.view(),
            self._sparse_index.view(),
            self._sparse_row_starts.view(),
            self._sparse_values,
        )

    def __len__(self) -> int:
        return self._shape[0]

    def _sparse(self, flat: int, /) -> ExactScalar:
        return self._sparse_values[int(np.searchsorted(self._sparse_index, flat))]

    def _entry(self, flat: int, significand: int, exponent: int, /) -> ExactScalar:
        if exponent == _SPARSE:
            return self._sparse(flat)
        return _dyadic_value(significand, exponent)

    def _row(self, row: int, /) -> tuple[ExactScalar, ...]:
        position = self._sparse_row_starts[row].item()
        values: list[ExactScalar] = []
        for significand, exponent in zip(
            self._significands[row].tolist(),
            self._exponents[row].tolist(),
            strict=True,
        ):
            if exponent == _SPARSE:
                values.append(self._sparse_values[position])
                position += 1
            else:
                values.append(_dyadic_value(significand, exponent))
        return tuple(values)

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

    def rows(
        self, indices: NDArray[np.integer[Any]], /
    ) -> tuple[tuple[ExactScalar, ...], ...]:
        """Materialize the selected rows (bounds-checked int vector)."""
        selected = np.asarray(indices)
        if selected.ndim != 1 or selected.dtype.kind not in "iu":
            raise TypeError("Exact coordinate bank rows must be an integer vector.")
        if np.any(selected < 0) or np.any(selected >= len(self)):
            raise IndexError("Exact coordinate bank row is out of range.")
        return tuple(self._row(int(row)) for row in selected.tolist())

    def dyadic_rows(
        self, indices: NDArray[np.integer[Any]], /
    ) -> tuple[NDArray[np.int64], NDArray[np.int16], NDArray[np.bool_]]:
        """Gather packed dyadics and identify entries requiring sparse handling."""
        selected = np.asarray(indices)
        if selected.ndim != 2 or selected.dtype.kind not in "iu":
            raise TypeError("Exact coordinate bank row routes must be an integer matrix.")
        if np.any(selected < 0) or np.any(selected >= len(self)):
            raise IndexError("Exact coordinate bank row is out of range.")
        significands = self._significands[selected]
        exponents = self._exponents[selected]
        return significands, exponents, exponents != _SPARSE

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
        if (
            np.any(rows < 0)
            or np.any(rows >= len(self))
            or np.unique(rows).size != rows.size
        ):
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

    def _interval_proposals(self) -> NDArray[np.float64]:
        """Propose dense dyadics without refusing nonfinite or sparse entries."""
        dense = self._exponents != _SPARSE
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            proposals = np.ldexp(
                self._significands.astype(np.float64),
                np.where(dense, self._exponents.astype(np.int64), 0),
            )
        return np.where(dense, proposals, np.nan)

    def interval_endpoints(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Outward binary64 endpoints, lifting exceptional exact entries only."""
        return self._verified_interval_endpoints(self._interval_proposals())

    def _verified_interval_endpoints(
        self, lower: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Verify proposals in place, converting rejected entries exactly."""
        from ._exact_algebraic_geometry import coefficient_endpoint

        upper = lower.copy()
        # The binary64 conversion is only a proposal. Verify its bits against
        # the exact odd significand and exponent using integer arithmetic; a
        # rounded, underflowed, or sparse entry takes directed exact conversion.
        bits = lower.view(np.uint64)
        fraction = bits & np.uint64((1 << 52) - 1)
        exponent = (bits >> np.uint64(52)) & np.uint64(0x7FF)
        mantissa = np.where(exponent == 0, fraction, fraction | np.uint64(1 << 52))
        binary_exponent = np.where(exponent == 0, -1074, exponent.astype(np.int64) - 1075)
        shift = self._exponents.astype(np.int64) - binary_exponent
        magnitude = np.abs(self._significands).astype(np.uint64)
        safe_shift = np.clip(shift, 0, 52).astype(np.uint64)
        shifted_mantissa = np.right_shift(mantissa, safe_shift)
        discarded = mantissa & (np.left_shift(np.uint64(1), safe_shift) - np.uint64(1))
        same_sign = (bits >> np.uint64(63)) == (self._significands < 0)
        exact_nonzero = (
            (self._significands != 0)
            & (exponent != np.uint64(0x7FF))
            & (shift >= 0)
            & (shift <= 52)
            & (shifted_mantissa == magnitude)
            & (discarded == 0)
            & same_sign
        )
        exact_zero = (self._significands == 0) & (bits == 0)
        exceptional = (self._exponents == _SPARSE) | ~(exact_nonzero | exact_zero)
        width = self._shape[1]
        for flat in np.flatnonzero(exceptional.reshape(-1)).tolist():
            row, axis = divmod(flat, width)
            value = self._entry(
                flat, int(self._significands[row, axis]), int(self._exponents[row, axis])
            )
            lower[row, axis] = coefficient_endpoint(value, upper=False, saturate=True)
            upper[row, axis] = coefficient_endpoint(value, upper=True, saturate=True)
        return lower, upper

    def coefficient_payloads(
        self,
    ) -> tuple[
        tuple[tuple[str, int, int] | tuple[str, tuple[tuple[int, int, int], ...]], ...],
        ...,
    ]:
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

    def _validate_derived_parts(self) -> None:
        """Refuse inconsistent un-hashed indexing metadata and cached hash."""
        sig, exp, index, offsets = (
            self._significands,
            self._exponents,
            self._sparse_index,
            self._sparse_row_starts,
        )
        shape = self._shape
        if (
            type(shape) is not tuple
            or len(shape) != 2
            or any(type(size) is not int or size < 0 for size in shape)
            or type(sig) is not np.ndarray
            or type(exp) is not np.ndarray
            or type(index) is not np.ndarray
            or type(offsets) is not np.ndarray
            or sig.dtype != np.dtype(np.int64)
            or exp.dtype != np.dtype(np.int16)
            or index.dtype != np.dtype(np.int64)
            or offsets.dtype != np.dtype(np.int64)
            or sig.shape != shape
            or exp.shape != shape
            or index.shape != (len(self._sparse_values),)
            or offsets.shape != (shape[0] + 1,)
        ):
            raise ValueError(
                "Exact coordinate bank derived shapes or dtypes are invalid."
            )
        expected_offsets = np.searchsorted(
            index, np.arange(shape[0] + 1, dtype=np.int64) * shape[1]
        )
        if not np.array_equal(offsets, expected_offsets):
            raise ValueError(
                "Exact coordinate bank row offsets differ from sparse indices."
            )

        def contents(array: np.ndarray, /) -> bytes:
            if (
                isinstance(array.base, bytes)
                and array.flags.c_contiguous
                and len(array.base) == array.nbytes
            ):
                return array.base
            return array.tobytes(order="C")

        expected_hash = hash(
            (shape, contents(sig), contents(exp), contents(index), self._sparse_values)
        )
        if type(self._hash) is not int or self._hash != expected_hash:
            raise ValueError("Exact coordinate bank derived hash differs from its parts.")

    def content_fingerprint(self) -> str:
        """Bind canonical dense dyadics and exact sparse entries with framed arrays."""
        arrays = (
            self._significands,
            self._exponents,
            self._sparse_index,
            self._sparse_row_starts,
        )
        sparse = tuple(coefficient_payload(value) for value in self._sparse_values)
        sealed = _immutable_sparse_values(self._sparse_values) and _sealed_parts(
            arrays, self._shape, len(sparse)
        )
        # The bytes-backed arrays cannot change; their identities and the sparse
        # payload are checked below before accepting this identity-only hit.
        recorded = _BANK_FINGERPRINTS._find_identity_bound(self) if sealed else None
        if recorded is not None:
            digest, references, previous_sparse, previous_hash = recorded
            if (
                type(self._hash) is int
                and previous_hash == self._hash
                and previous_sparse == sparse
                and all(
                    reference() is array
                    for reference, array in zip(references, arrays, strict=True)
                )
            ):
                return digest
        self._validate_derived_parts()
        digest = canonical_fingerprint(
            {
                "kind": "exact-coordinate-bank",
                "significands": numeric_array_fingerprint(self._significands),
                "exponents": numeric_array_fingerprint(self._exponents),
                "sparse_indices": numeric_array_fingerprint(self._sparse_index),
                "sparse_values": sparse,
            }
        )
        if sealed:
            _BANK_FINGERPRINTS._record_identity_bound(
                self,
                (
                    digest,
                    tuple(weakref.ref(array) for array in arrays),
                    sparse,
                    self._hash,
                ),
            )
        return digest

    def _live_leaf_fingerprint(self, /) -> str:
        return self.content_fingerprint()

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
