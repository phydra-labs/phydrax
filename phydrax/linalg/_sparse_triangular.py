#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
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
from .._trainable import NonTrainableState
from ..typing import checked, parse
from ._sparse_contract import AbstractSparseLinearOperator, SparseStorage


SparseTriangle: TypeAlias = Literal["lower", "upper"]

# Wavefront blocking model, in units of one gathered CSR entry per right-hand
# side. On CPU one sequential fori step (loop, slicing, and scatter) costs
# about 1.4 us and one gathered entry about 5 ns, measured on 13-neighbor 2-D
# clouds with n = 64k: a one-row step versus 86-row level blocks. Blocks trade
# that overhead against padded rows. The working-set cap bounds one step's
# gather to ``block_rows * row_width`` entries per right-hand side, and the
# padding cap bounds resident schedule rows to twice the matrix size per
# orientation.
_LEVEL_STEP_OVERHEAD_ENTRIES = 256
_MAX_LEVEL_STEP_ENTRIES = 1 << 16
_MAX_LEVEL_PADDING = 2


class SparseTriangularStatus(IntEnum):
    SUCCESS = 0
    ZERO_PIVOT = 1
    NONFINITE = 2


class SparseTriangularAnalysis(StrictModule):
    """Host symbolic analysis and fixed-shape level schedules for one CSR pattern.

    Each orientation's ``level_schedule`` lists its rows by dependency level,
    then row index, packed into fixed-width blocks whose rows share one level;
    padding slots hold ``shape[0]``. Prepared position/column/validity grids
    retain every original CSR reduction slot. A solve runs one step per block.
    """

    indices: Array
    indptr: Array
    row_indices: Array
    diagonal_positions: Array
    row_levels: Array
    level_schedule: Array
    level_schedule_widths: Array
    transpose_indices: Array
    schedule_positions: Array
    schedule_columns: Array
    schedule_valid: Array
    transpose_indptr: Array
    transpose_row_indices: Array
    transpose_value_positions: Array
    transpose_diagonal_positions: Array
    transpose_row_levels: Array
    transpose_level_schedule: Array
    transpose_level_schedule_widths: Array
    transpose_schedule_positions: Array
    transpose_schedule_columns: Array
    transpose_schedule_valid: Array
    shape: tuple[int, int] = eqx.field(static=True)
    triangle: SparseTriangle = eqx.field(static=True)
    unit_diagonal: bool = eqx.field(static=True)
    number_levels: int = eqx.field(static=True)
    transpose_number_levels: int = eqx.field(static=True)
    row_width: int = eqx.field(static=True)
    transpose_row_width: int = eqx.field(static=True)
    solve_work_units_upper: int = eqx.field(static=True)
    transpose_solve_work_units_upper: int = eqx.field(static=True)
    pattern_id: str = eqx.field(static=True)

    @property
    def numeric_preparation_work_units_upper(self) -> int:
        """One orientation's coefficient/pivot preparation, once per refresh."""
        slots = max(self.schedule_positions.size, self.transpose_schedule_positions.size)
        return 8 * self.indices.size + 2 * slots + 16 * self.shape[0] + 32

    @property
    def promoted_rhs_work_units_upper(self) -> int:
        """Additional casts and promoted pivot evidence for one vector RHS."""
        slots = max(self.schedule_positions.size, self.transpose_schedule_positions.size)
        return slots + 16 * self.shape[0] + 32


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


class _PreparedTriangularSubstitution(StrictModule, NonTrainableState):
    """One orientation's refresh-owned numerical values and pivot evidence."""

    scheduled_values: Array
    diagonal: Array
    safe_diagonal: Array
    finite_values: Array
    finite_diagonal: Array
    zero_pivot: Array
    minimum_pivot: Array
    pivot_tolerance: float = eqx.field(static=True)


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


def _analysis_storage_bytes(
    size: int,
    nnz: int,
    index_itemsize: int,
    /,
    *,
    row_width: int,
    transpose_row_width: int,
) -> int:
    """Upper bound on one analysis' resident arrays for ``size`` rows, ``nnz`` entries.

    Five index arrays per entry, then per orientation the row pointers,
    diagonal positions, int32 levels and block-width selectors, and at most
    ``_MAX_LEVEL_PADDING * size`` scheduled slots.
    """
    per_orientation = (size + 1) + size + _MAX_LEVEL_PADDING * size
    return (
        index_itemsize * (5 * nnz + 2 * per_orientation)
        + _MAX_LEVEL_PADDING
        * size
        * (row_width + transpose_row_width)
        * (2 * index_itemsize + np.dtype(np.bool_).itemsize)
        + 2 * (1 + _MAX_LEVEL_PADDING) * size * np.dtype(np.int32).itemsize
    )


def _level_schedule(levels: np.ndarray, row_width: int, /) -> np.ndarray:
    """Pack same-level rows into the cheapest admissible fixed block width.

    Rows of one level are mutually independent, so a block substitutes them
    together. Single-row blocks are always admissible and reproduce the
    row-sequential step count when every level holds one row.
    """
    size = levels.size
    if size == 0:
        return np.zeros((0, 1), dtype=np.int64)
    counts = np.bincount(levels)
    largest = int(counts.max())
    width = max(row_width, 1)
    candidates = sorted({1 << power for power in range(largest.bit_length())} | {largest})
    best_cost, block, steps = size * (_LEVEL_STEP_OVERHEAD_ENTRIES + width), 1, size
    for candidate in candidates[1:]:
        candidate_steps = int(np.sum(-(-counts // candidate)))
        if (
            candidate * width > _MAX_LEVEL_STEP_ENTRIES
            or candidate_steps * candidate > _MAX_LEVEL_PADDING * size
        ):
            continue
        cost = candidate_steps * (_LEVEL_STEP_OVERHEAD_ENTRIES + candidate * width)
        if cost < best_cost:
            best_cost, block, steps = cost, candidate, candidate_steps
    order = np.argsort(levels, kind="stable")
    ordered_levels = levels[order]
    level_starts = np.concatenate(([0], np.cumsum(counts)))
    block_starts = np.concatenate(([0], np.cumsum(-(-counts // block))))
    rank = np.arange(size) - level_starts[ordered_levels]
    schedule = np.full((steps, block), size, dtype=np.int64)
    schedule[block_starts[ordered_levels] + rank // block, rank % block] = order
    return schedule


def _entry_width_choices(row_width: int, /) -> tuple[int, ...]:
    """Logarithmically many kernels, none wider than the admitted row window."""
    return tuple(
        sorted({1 << power for power in range(row_width.bit_length())} | {row_width})
    )


def _schedule_entry_widths(
    schedule: np.ndarray,
    indptr: np.ndarray,
    row_width: int,
    /,
) -> tuple[np.ndarray, int]:
    choices = np.asarray(_entry_width_choices(row_width), dtype=np.int64)
    lengths = np.concatenate((np.diff(indptr), np.zeros(1, dtype=np.int64)))
    required = np.max(lengths[schedule], axis=1)
    selectors = np.searchsorted(choices, required).astype(np.int32)
    prefix_slots = int(np.sum(choices[selectors])) * schedule.shape[1]
    reduction_slots = schedule.size * row_width
    # Prepared substitution owns coefficient/pivot evidence once per refresh.
    # Each RHS still owns its finite/solution checks and the complete original
    # padding/reduction tree. This bound is for the coefficient dtype; promoted
    # RHS work is accounted separately by the consuming factor plan.
    work_upper = (
        3 * prefix_slots
        + 2 * reduction_slots
        + 8 * schedule.size
        + 6 * (indptr.size - 1)
        + 32
    )
    return selectors, work_upper


def _scheduled_entries(
    schedule: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    row_width: int,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prepare exact CSR slot order, including the original zero padding."""
    padded_indptr = np.concatenate((indptr, indptr[-1:]))
    positions = padded_indptr[schedule][:, None, :] + np.arange(row_width)[None, :, None]
    valid = positions < padded_indptr[schedule + 1][:, None, :]
    positions = np.where(valid, positions, 0)
    columns = indices[positions]
    return positions, columns, valid


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
    row_width = int(np.max(np.diff(indptr), initial=0))
    transpose_row_width = int(np.max(transpose_counts, initial=0))
    schedule = _level_schedule(levels, row_width)
    transpose_schedule = _level_schedule(transpose_levels, transpose_row_width)
    schedule_widths, solve_work_upper = _schedule_entry_widths(
        schedule, indptr, row_width
    )
    transpose_schedule_widths, transpose_solve_work_upper = _schedule_entry_widths(
        transpose_schedule,
        transpose_indptr,
        transpose_row_width,
    )
    schedule_positions, schedule_columns, schedule_valid = _scheduled_entries(
        schedule,
        indices,
        indptr,
        row_width,
    )
    transpose_positions_grid, transpose_columns_grid, transpose_valid = (
        _scheduled_entries(
            transpose_schedule,
            transpose_indices,
            transpose_indptr,
            transpose_row_width,
        )
    )
    index_dtype = storage.indices.dtype
    pattern_bytes = b"|".join(
        (
            np.asarray(storage.shape, dtype=np.int64).tobytes(),
            indices.tobytes(),
            indptr.tobytes(),
            triangle.encode(),
            str(bool(unit_diagonal)).encode(),
            b"prepared-scheduled-gather-original-reduction-tree:numeric-block-cache",
        )
    )
    return SparseTriangularAnalysis(
        indices=jnp.asarray(indices, dtype=index_dtype),
        indptr=jnp.asarray(indptr, dtype=index_dtype),
        row_indices=jnp.asarray(rows, dtype=index_dtype),
        diagonal_positions=jnp.asarray(diagonal, dtype=index_dtype),
        row_levels=jnp.asarray(levels, dtype=jnp.int32),
        level_schedule=jnp.asarray(schedule, dtype=index_dtype),
        level_schedule_widths=jnp.asarray(schedule_widths, dtype=jnp.int32),
        schedule_positions=jnp.asarray(schedule_positions, dtype=index_dtype),
        schedule_columns=jnp.asarray(schedule_columns, dtype=index_dtype),
        schedule_valid=jnp.asarray(schedule_valid),
        transpose_indices=jnp.asarray(transpose_indices, dtype=index_dtype),
        transpose_indptr=jnp.asarray(transpose_indptr, dtype=index_dtype),
        transpose_row_indices=jnp.asarray(
            np.repeat(np.arange(size, dtype=np.int64), transpose_counts),
            dtype=index_dtype,
        ),
        transpose_value_positions=jnp.asarray(transpose_positions, dtype=index_dtype),
        transpose_diagonal_positions=jnp.asarray(transpose_diagonal, dtype=index_dtype),
        transpose_row_levels=jnp.asarray(transpose_levels, dtype=jnp.int32),
        transpose_level_schedule=jnp.asarray(transpose_schedule, dtype=index_dtype),
        transpose_level_schedule_widths=jnp.asarray(
            transpose_schedule_widths, dtype=jnp.int32
        ),
        transpose_schedule_positions=jnp.asarray(
            transpose_positions_grid, dtype=index_dtype
        ),
        transpose_schedule_columns=jnp.asarray(transpose_columns_grid, dtype=index_dtype),
        transpose_schedule_valid=jnp.asarray(transpose_valid),
        shape=storage.shape,
        triangle=triangle,
        unit_diagonal=bool(unit_diagonal),
        number_levels=int(levels.max(initial=-1)) + 1,
        transpose_number_levels=int(transpose_levels.max(initial=-1)) + 1,
        row_width=row_width,
        transpose_row_width=transpose_row_width,
        solve_work_units_upper=solve_work_upper,
        transpose_solve_work_units_upper=transpose_solve_work_upper,
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
    """Prepare current values, then use the canonical scheduled substitution."""
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
    prepared = _prepare_sparse_triangular_substitution(
        analysis,
        values_.astype(dtype),
        pivot_tolerance=tolerance,
        transpose=transpose,
        adjoint=adjoint,
    )
    return _solve_prepared_sparse_triangular(
        analysis,
        prepared,
        rhs,
        vector_input=vector_input,
        transpose=bool(transpose or adjoint),
    )


def _prepare_sparse_triangular_substitution(
    analysis: SparseTriangularAnalysis,
    values: ArrayLike,
    /,
    *,
    pivot_tolerance: float = 0.0,
    transpose: bool = False,
    adjoint: bool = False,
) -> _PreparedTriangularSubstitution:
    """Refresh one numeric orientation; no RHS or numerical static cache."""
    if not isinstance(analysis, SparseTriangularAnalysis):
        raise TypeError("analysis must be SparseTriangularAnalysis.")
    tolerance = float(pivot_tolerance)
    if not isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("pivot_tolerance must be finite and non-negative.")
    values_ = jnp.asarray(values)
    if values_.shape != analysis.indices.shape:
        raise ValueError("values must match the analyzed CSR nonzero pattern.")
    if transpose or adjoint:
        values_ = values_[analysis.transpose_value_positions]
        rows = analysis.transpose_row_indices
        diagonal_positions = analysis.transpose_diagonal_positions
        positions, valid = (
            analysis.transpose_schedule_positions,
            analysis.transpose_schedule_valid,
        )
    else:
        rows = analysis.row_indices
        diagonal_positions = analysis.diagonal_positions
        positions, valid = analysis.schedule_positions, analysis.schedule_valid
    if adjoint:
        values_ = jnp.conj(values_)
    safe_diagonal_positions = jnp.maximum(diagonal_positions, 0)
    diagonal = (
        jnp.ones((analysis.shape[0],), dtype=values_.dtype)
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
    finite_diagonal = jnp.all(jnp.isfinite(diagonal))
    off_values = jnp.where(off_diagonal, values_, jnp.zeros((), dtype=values_.dtype))
    return _PreparedTriangularSubstitution(
        scheduled_values=jnp.where(
            valid, off_values[positions], jnp.zeros((), dtype=values_.dtype)
        ),
        diagonal=diagonal,
        safe_diagonal=jnp.concatenate(
            (safe_diagonal, jnp.ones((1,), dtype=values_.dtype))
        ),
        finite_values=jnp.all(jnp.isfinite(values_)),
        finite_diagonal=finite_diagonal,
        zero_pivot=finite_diagonal & jnp.any(jnp.abs(diagonal) <= tolerance),
        minimum_pivot=jnp.min(jnp.abs(diagonal)),
        pivot_tolerance=tolerance,
    )


def _solve_prepared_sparse_triangular(
    analysis: SparseTriangularAnalysis,
    prepared: _PreparedTriangularSubstitution,
    right_hand_side: Array,
    /,
    *,
    vector_input: bool = False,
    transpose: bool = False,
) -> SparseTriangularSolveResult:
    """Reuse immutable numeric evidence; inspect every actual RHS and solution."""
    dtype = jnp.result_type(prepared.scheduled_values.dtype, right_hand_side.dtype)
    rhs = right_hand_side.astype(dtype)
    scheduled_values = prepared.scheduled_values.astype(dtype)
    if dtype == prepared.scheduled_values.dtype:
        padded_diagonal = prepared.safe_diagonal
        zero_pivot = prepared.zero_pivot
        minimum_pivot = prepared.minimum_pivot
    else:
        # Abs/pivot thresholds must be evaluated in the actual promoted dtype,
        # not inherited from lower-precision coefficient evidence.
        diagonal = prepared.diagonal.astype(dtype)
        valid_pivot = jnp.isfinite(diagonal) & (
            jnp.abs(diagonal) > prepared.pivot_tolerance
        )
        safe_diagonal = jnp.where(valid_pivot, diagonal, jnp.ones_like(diagonal))
        padded_diagonal = jnp.concatenate((safe_diagonal, jnp.ones((1,), dtype=dtype)))
        zero_pivot = prepared.finite_diagonal & jnp.any(
            jnp.abs(diagonal) <= prepared.pivot_tolerance,
        )
        minimum_pivot = jnp.min(jnp.abs(diagonal))
    if transpose:
        schedule = analysis.transpose_level_schedule
        schedule_widths = analysis.transpose_level_schedule_widths
        number_levels, row_width = (
            analysis.transpose_number_levels,
            analysis.transpose_row_width,
        )
        schedule_columns, schedule_valid = (
            analysis.transpose_schedule_columns,
            analysis.transpose_schedule_valid,
        )
    else:
        schedule = analysis.level_schedule
        schedule_widths = analysis.level_schedule_widths
        number_levels, row_width = analysis.number_levels, analysis.row_width
        schedule_columns, schedule_valid = (
            analysis.schedule_columns,
            analysis.schedule_valid,
        )
    # Padding retains the exact original reduction positions and absorbs the
    # writes of empty scheduled rows; cached safe pivots include its unit entry.
    padded_rhs = jnp.concatenate((rhs, jnp.zeros((1, rhs.shape[1]), dtype=dtype)))
    initial = jnp.zeros_like(padded_rhs)

    def width_kernel(width: int) -> Callable[[tuple[Array, Array]], Array]:

        def substitute(operand: tuple[Array, Array]) -> Array:
            step, solution = operand
            block = schedule[step]
            columns = schedule_columns[step, :width]
            valid = schedule_valid[step, :width]
            products = scheduled_values[step, :width, ..., None] * solution[columns]
            products = jnp.where(valid[..., None], products, 0.0)
            # Do not shorten/reassociate the declared floating reduction:
            # omitted CSR padding was exactly zero in these original slots.
            products = jnp.pad(products, ((0, row_width - width), (0, 0), (0, 0)))
            row_sum = jnp.sum(products, axis=0)
            candidate = (padded_rhs[block] - row_sum) / padded_diagonal[block][:, None]
            return solution.at[block].set(candidate)

        return substitute

    kernels = tuple(width_kernel(width) for width in _entry_width_choices(row_width))

    def solve_block(step: Array, solution: Array) -> Array:
        return jax.lax.switch(schedule_widths[step], kernels, (step, solution))

    solution = jax.lax.fori_loop(0, schedule.shape[0], solve_block, initial)[:-1]
    finite = (
        jnp.all(jnp.isfinite(solution))
        & prepared.finite_values
        & jnp.all(jnp.isfinite(rhs))
    )
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
            minimum_pivot=minimum_pivot,
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
