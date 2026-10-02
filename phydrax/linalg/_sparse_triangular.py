#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum
from hashlib import sha256
from math import isfinite
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._strict import StrictModule
from ..typing import checked, parse
from ._sparse_contract import AbstractSparseLinearOperator, SparseStorage


SparseTriangle: TypeAlias = Literal["lower", "upper"]


class SparseTriangularStatus(IntEnum):
    SUCCESS = 0
    ZERO_PIVOT = 1
    NONFINITE = 2


class SparseTriangularAnalysis(StrictModule):
    """Host symbolic analysis and fixed-shape level schedules for one CSR pattern."""

    indices: Array
    indptr: Array
    row_indices: Array
    diagonal_positions: Array
    row_levels: Array
    transpose_indices: Array
    transpose_indptr: Array
    transpose_row_indices: Array
    transpose_value_positions: Array
    transpose_diagonal_positions: Array
    transpose_row_levels: Array
    shape: tuple[int, int] = eqx.field(static=True)
    triangle: SparseTriangle = eqx.field(static=True)
    unit_diagonal: bool = eqx.field(static=True)
    number_levels: int = eqx.field(static=True)
    transpose_number_levels: int = eqx.field(static=True)
    row_width: int = eqx.field(static=True)
    transpose_row_width: int = eqx.field(static=True)
    pattern_id: str = eqx.field(static=True)


class SparseTriangularSolveDiagnostics(StrictModule):
    """Pivot and schedule evidence for one staged triangular solve."""

    minimum_pivot: Array
    finite: Array
    level_count: Array
    right_hand_sides: Array


class SparseTriangularSolveResult(StrictModule):
    """Coordinate solution with explicit triangular failure status."""

    value: Array
    status: Array
    diagnostics: SparseTriangularSolveDiagnostics

    @property
    def success(self) -> Array:
        return self.status == int(SparseTriangularStatus.SUCCESS)


class SparseTriangularFactor(StrictModule):
    """Reusable symbolic triangular schedule paired with refreshable values."""

    analysis: SparseTriangularAnalysis
    values: Array
    pivot_tolerance: float = eqx.field(static=True)
    factor_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        analysis: SparseTriangularAnalysis,
        values: ArrayLike,
        /,
        *,
        pivot_tolerance: float = 0.0,
        factor_id: str | None = None,
    ) -> None:
        values_ = jnp.asarray(values)
        if values_.shape != analysis.indices.shape:
            raise ValueError("Triangular values must match the analyzed CSR pattern.")
        if not jnp.issubdtype(values_.dtype, jnp.inexact):
            raise TypeError("Triangular values must use an inexact dtype.")
        tolerance = float(pivot_tolerance)
        if not isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("pivot_tolerance must be finite and non-negative.")
        self.analysis = analysis
        self.values = values_
        self.pivot_tolerance = tolerance
        self.factor_id = (
            f"triangular/{analysis.pattern_id}" if factor_id is None else str(factor_id)
        )
        if not self.factor_id:
            raise ValueError("factor_id must be non-empty.")

    def solve(
        self,
        right_hand_side: ArrayLike,
        /,
        *,
        transpose: bool = False,
        adjoint: bool = False,
    ) -> SparseTriangularSolveResult:
        return solve_sparse_triangular(
            self.analysis,
            self.values,
            right_hand_side,
            pivot_tolerance=self.pivot_tolerance,
            transpose=transpose,
            adjoint=adjoint,
        )


def _storage(value: AbstractSparseLinearOperator | SparseStorage, /) -> SparseStorage:
    if isinstance(value, AbstractSparseLinearOperator):
        return value.sparse_storage()
    if isinstance(value, SparseStorage):
        return value
    raise TypeError("Expected AbstractSparseLinearOperator or SparseStorage.")


def _validated_host_pattern(storage: SparseStorage, /) -> tuple[np.ndarray, np.ndarray]:
    if storage.shape[0] != storage.shape[1]:
        raise ValueError("Sparse triangular analysis requires a square pattern.")
    indices = np.asarray(storage.indices, dtype=np.int64)
    indptr = np.asarray(storage.indptr, dtype=np.int64)
    if indptr[0] != 0 or indptr[-1] != indices.size:
        raise ValueError("CSR indptr endpoints are inconsistent with the index vector.")
    if np.any(indptr[1:] < indptr[:-1]):
        raise ValueError("CSR indptr must be nondecreasing.")
    if np.any(indices < 0) or np.any(indices >= storage.shape[1]):
        raise ValueError("CSR column index is out of range.")
    for row in range(storage.shape[0]):
        columns = indices[indptr[row] : indptr[row + 1]]
        if columns.size > 1 and np.any(columns[1:] <= columns[:-1]):
            raise ValueError(
                "Sparse triangular analysis requires sorted, duplicate-free rows."
            )
    return indices, indptr


def _levels(
    indices: np.ndarray,
    indptr: np.ndarray,
    triangle: SparseTriangle,
    /,
) -> np.ndarray:
    size = indptr.size - 1
    levels = np.zeros(size, dtype=np.int32)
    order = range(size) if triangle == "lower" else range(size - 1, -1, -1)
    for row in order:
        columns = indices[indptr[row] : indptr[row + 1]]
        dependencies = (
            columns[columns < row] if triangle == "lower" else columns[columns > row]
        )
        levels[row] = (
            0 if dependencies.size == 0 else 1 + int(np.max(levels[dependencies]))
        )
    return levels


def _orientation_analysis(
    indices: np.ndarray,
    indptr: np.ndarray,
    triangle: SparseTriangle,
    unit_diagonal: bool,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    size = indptr.size - 1
    diagonal = np.full(size, -1, dtype=np.int64)
    for row in range(size):
        start, stop = indptr[row], indptr[row + 1]
        columns = indices[start:stop]
        outside = columns > row if triangle == "lower" else columns < row
        if np.any(outside):
            raise ValueError(
                f"CSR pattern contains entries outside its {triangle} triangle."
            )
        hits = np.flatnonzero(columns == row)
        if hits.size:
            diagonal[row] = start + int(hits[0])
        elif not unit_diagonal:
            raise ValueError(f"Non-unit triangular row {row} has no diagonal entry.")
    return diagonal, _levels(indices, indptr, triangle)


def analyze_sparse_triangular(
    operator_or_storage: AbstractSparseLinearOperator | SparseStorage,
    /,
    *,
    triangle: SparseTriangle,
    unit_diagonal: bool = False,
) -> SparseTriangularAnalysis:
    """Analyze one immutable CSR triangular pattern on the host."""
    triangle = parse(triangle, SparseTriangle, "triangle")
    storage = _storage(operator_or_storage)
    indices, indptr = _validated_host_pattern(storage)
    diagonal, levels = _orientation_analysis(
        indices, indptr, triangle, bool(unit_diagonal)
    )
    size = storage.shape[0]
    rows = np.repeat(np.arange(size, dtype=np.int64), np.diff(indptr))
    transpose_rows = indices
    order = np.lexsort((rows, transpose_rows))
    transpose_indices = rows[order]
    transpose_positions = np.arange(indices.size, dtype=np.int64)[order]
    transpose_counts = np.bincount(transpose_rows, minlength=size)
    transpose_indptr = np.concatenate(([0], np.cumsum(transpose_counts))).astype(np.int64)
    transpose_triangle: SparseTriangle = "upper" if triangle == "lower" else "lower"
    transpose_diagonal, transpose_levels = _orientation_analysis(
        transpose_indices,
        transpose_indptr,
        transpose_triangle,
        bool(unit_diagonal),
    )
    index_dtype = storage.indices.dtype
    pattern_bytes = b"|".join(
        (
            np.asarray(storage.shape, dtype=np.int64).tobytes(),
            indices.tobytes(),
            indptr.tobytes(),
            triangle.encode(),
            str(bool(unit_diagonal)).encode(),
        )
    )
    return SparseTriangularAnalysis(
        indices=jnp.asarray(indices, dtype=index_dtype),
        indptr=jnp.asarray(indptr, dtype=index_dtype),
        row_indices=jnp.asarray(rows, dtype=index_dtype),
        diagonal_positions=jnp.asarray(diagonal, dtype=index_dtype),
        row_levels=jnp.asarray(levels, dtype=jnp.int32),
        transpose_indices=jnp.asarray(transpose_indices, dtype=index_dtype),
        transpose_indptr=jnp.asarray(transpose_indptr, dtype=index_dtype),
        transpose_row_indices=jnp.asarray(
            np.repeat(np.arange(size, dtype=np.int64), transpose_counts),
            dtype=index_dtype,
        ),
        transpose_value_positions=jnp.asarray(transpose_positions, dtype=index_dtype),
        transpose_diagonal_positions=jnp.asarray(transpose_diagonal, dtype=index_dtype),
        transpose_row_levels=jnp.asarray(transpose_levels, dtype=jnp.int32),
        shape=storage.shape,
        triangle=triangle,
        unit_diagonal=bool(unit_diagonal),
        number_levels=int(levels.max(initial=-1)) + 1,
        transpose_number_levels=int(transpose_levels.max(initial=-1)) + 1,
        row_width=int(np.max(np.diff(indptr), initial=0)),
        transpose_row_width=int(np.max(transpose_counts, initial=0)),
        pattern_id=sha256(pattern_bytes).hexdigest(),
    )


def solve_sparse_triangular(
    analysis: SparseTriangularAnalysis,
    values: ArrayLike,
    right_hand_side: ArrayLike,
    /,
    *,
    pivot_tolerance: float = 0.0,
    transpose: bool = False,
    adjoint: bool = False,
) -> SparseTriangularSolveResult:
    """Execute one fixed-capacity CSR row substitution per triangular row."""
    if not isinstance(analysis, SparseTriangularAnalysis):
        raise TypeError("analysis must be SparseTriangularAnalysis.")
    tolerance = float(pivot_tolerance)
    if not isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("pivot_tolerance must be finite and non-negative.")
    values_ = jnp.asarray(values)
    if values_.shape != analysis.indices.shape:
        raise ValueError("values must match the analyzed CSR nonzero pattern.")
    rhs = jnp.asarray(right_hand_side)
    vector_input = rhs.ndim == 1
    if vector_input:
        rhs = rhs[:, None]
    if rhs.ndim != 2 or rhs.shape[0] != analysis.shape[0]:
        raise ValueError("right_hand_side must have shape (n,) or (n, k).")
    if not jnp.issubdtype(rhs.dtype, jnp.inexact):
        raise TypeError("right_hand_side must use an inexact dtype.")
    dtype = jnp.result_type(values_.dtype, rhs.dtype)
    rhs = rhs.astype(dtype)
    values_ = values_.astype(dtype)
    use_transpose = bool(transpose or adjoint)
    if use_transpose:
        indices = analysis.transpose_indices
        indptr = analysis.transpose_indptr
        rows = analysis.transpose_row_indices
        values_ = values_[analysis.transpose_value_positions]
        diagonal_positions = analysis.transpose_diagonal_positions
        number_levels = analysis.transpose_number_levels
        row_width = analysis.transpose_row_width
        lower = analysis.triangle == "upper"
    else:
        indices = analysis.indices
        indptr = analysis.indptr
        rows = analysis.row_indices
        diagonal_positions = analysis.diagonal_positions
        number_levels = analysis.number_levels
        row_width = analysis.row_width
        lower = analysis.triangle == "lower"
    if adjoint:
        values_ = jnp.conj(values_)
    safe_diagonal_positions = jnp.maximum(diagonal_positions, 0)
    diagonal = (
        jnp.ones((analysis.shape[0],), dtype=dtype)
        if analysis.unit_diagonal
        else values_[safe_diagonal_positions]
    )
    valid_pivot = jnp.isfinite(diagonal) & (jnp.abs(diagonal) > tolerance)
    safe_diagonal = jnp.where(valid_pivot, diagonal, jnp.ones_like(diagonal))
    entry_positions = jnp.arange(values_.size, dtype=diagonal_positions.dtype)
    off_diagonal = (
        jnp.ones(values_.shape, dtype=jnp.bool_)
        if analysis.unit_diagonal
        else entry_positions != safe_diagonal_positions[rows]
    )
    off_values = jnp.where(off_diagonal, values_, jnp.zeros((), dtype=dtype))
    initial = jnp.zeros_like(rhs)
    offsets = jnp.arange(row_width, dtype=indptr.dtype)

    def solve_row(position: Array, solution: Array) -> Array:
        row = jnp.where(lower, position, analysis.shape[0] - 1 - position)
        entry_positions = indptr[row] + offsets
        valid = entry_positions < indptr[row + 1]
        safe_positions = jnp.where(valid, entry_positions, 0)
        columns = indices[safe_positions]
        products = off_values[safe_positions, None] * solution[columns]
        row_sum = jnp.sum(jnp.where(valid[:, None], products, 0.0), axis=0)
        candidate = (rhs[row] - row_sum) / safe_diagonal[row]
        return solution.at[row].set(candidate)

    solution = jax.lax.fori_loop(0, analysis.shape[0], solve_row, initial)
    finite = (
        jnp.all(jnp.isfinite(solution))
        & jnp.all(jnp.isfinite(values_))
        & jnp.all(jnp.isfinite(rhs))
    )
    zero_pivot = jnp.all(jnp.isfinite(diagonal)) & jnp.any(jnp.abs(diagonal) <= tolerance)
    status = jnp.where(
        ~finite,
        int(SparseTriangularStatus.NONFINITE),
        jnp.where(
            zero_pivot,
            int(SparseTriangularStatus.ZERO_PIVOT),
            int(SparseTriangularStatus.SUCCESS),
        ),
    ).astype(jnp.int32)
    result_value = solution[:, 0] if vector_input else solution
    return SparseTriangularSolveResult(
        value=result_value,
        status=status,
        diagnostics=SparseTriangularSolveDiagnostics(
            minimum_pivot=jnp.min(jnp.abs(diagonal)),
            finite=finite,
            level_count=jnp.asarray(number_levels, dtype=jnp.int32),
            right_hand_sides=jnp.asarray(rhs.shape[1], dtype=jnp.int32),
        ),
    )


__all__ = [
    "SparseTriangle",
    "SparseTriangularAnalysis",
    "SparseTriangularFactor",
    "SparseTriangularSolveDiagnostics",
    "SparseTriangularSolveResult",
    "SparseTriangularStatus",
    "analyze_sparse_triangular",
    "solve_sparse_triangular",
]
