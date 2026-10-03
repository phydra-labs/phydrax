#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from hashlib import sha256
from math import isfinite, prod
from numbers import Integral
from typing import Any, Literal, NoReturn, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, core as jax_core
from jax.typing import ArrayLike, DTypeLike

from .._strict import StrictModule
from ..typing import parse
from ._properties import LinearCapabilityError
from ._sparse_contract import AbstractSparseLinearOperator, SparseStorage
from ._sparse_ordering import (
    _order_pattern,
    _pattern_identifier,
    _validated_pattern,
    _WorkMeter,
    PreparedSparseOrdering,
    SparseOrdering,
    SparseOrderingPolicy,
)
from ._sparse_triangular import (
    analyze_sparse_triangular,
    solve_sparse_triangular,
    SparseTriangularAnalysis,
    SparseTriangularStatus,
)


SparseFactorizationKind: TypeAlias = Literal["auto", "lu", "cholesky"]


class SparseFactorizationStatus(IntEnum):
    SUCCESS = 0
    ZERO_PIVOT = 1
    NONPOSITIVE_PIVOT = 2
    NONFINITE = 3


class SparseFactorizationPolicy(StrictModule):
    """Symbolic fill, ordering, dropping, and explicit pivot-replacement policy.

    ``ordering`` selects the symmetric fill-reducing ordering owned by
    `prepare_sparse_ordering`; its work is charged to ``max_symbolic_work``.
    """

    kind: SparseFactorizationKind = eqx.field(static=True)
    ordering: SparseOrdering = eqx.field(static=True)
    fill_level: int | None = eqx.field(static=True)
    drop_tolerance: float = eqx.field(static=True)
    maximum_fill_per_row: int | None = eqx.field(static=True)
    pivot_tolerance: float = eqx.field(static=True)
    diagonal_shift: float = eqx.field(static=True)
    allow_pivot_replacement: bool = eqx.field(static=True)
    replacement_value: float = eqx.field(static=True)
    max_factor_nnz: int = eqx.field(static=True)
    max_factor_bytes: int = eqx.field(static=True)
    max_symbolic_work: int = eqx.field(static=True)

    def __init__(
        self,
        kind: SparseFactorizationKind = "auto",
        /,
        *,
        ordering: SparseOrdering = "natural",
        fill_level: int | None = None,
        drop_tolerance: float = 0.0,
        maximum_fill_per_row: int | None = None,
        pivot_tolerance: float = 0.0,
        diagonal_shift: float = 0.0,
        allow_pivot_replacement: bool = False,
        replacement_value: float = 1e-12,
        max_factor_nnz: int = 2_000_000,
        max_factor_bytes: int = 256 * 1024 * 1024,
        max_symbolic_work: int = 512_000_000,
    ) -> None:
        kind = parse(kind, SparseFactorizationKind, "kind")
        ordering = parse(ordering, SparseOrdering, "ordering")
        fill = None if fill_level is None else int(fill_level)
        maximum_fill = None if maximum_fill_per_row is None else int(maximum_fill_per_row)
        resource_limits = {
            "max_factor_nnz": max_factor_nnz,
            "max_factor_bytes": max_factor_bytes,
            "max_symbolic_work": max_symbolic_work,
        }
        for name, value in resource_limits.items():
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be a host integer.")
            if value < 1:
                raise ValueError(f"{name} must be positive.")
        if fill is not None and fill < 0:
            raise ValueError("fill_level must be non-negative or None.")
        if maximum_fill is not None and maximum_fill < 0:
            raise ValueError("maximum_fill_per_row must be non-negative or None.")
        numeric = tuple(
            float(value)
            for value in (
                drop_tolerance,
                pivot_tolerance,
                diagonal_shift,
                replacement_value,
            )
        )
        if any(not isfinite(value) for value in numeric):
            raise ValueError("Sparse factorization numeric policies must be finite.")
        if numeric[0] < 0.0 or numeric[1] < 0.0 or numeric[2] < 0.0:
            raise ValueError("Drop, pivot, and shift tolerances must be non-negative.")
        if numeric[3] <= 0.0:
            raise ValueError("replacement_value must be positive.")
        self.kind = kind
        self.ordering = ordering
        self.fill_level = fill
        self.drop_tolerance = numeric[0]
        self.maximum_fill_per_row = maximum_fill
        self.pivot_tolerance = numeric[1]
        self.diagonal_shift = numeric[2]
        self.allow_pivot_replacement = bool(allow_pivot_replacement)
        self.replacement_value = numeric[3]
        self.max_factor_nnz = int(max_factor_nnz)
        self.max_factor_bytes = int(max_factor_bytes)
        self.max_symbolic_work = int(max_symbolic_work)


class SparseFactorizationPlan(StrictModule):
    """Immutable host symbolic plan for refreshable sparse LU or Cholesky values.

    The plan stores only the factor pattern: CSR rows of the combined factor
    and a column-major index of its strictly lower entries. The numeric
    kernel derives every elimination target at runtime from that pattern, so
    stored bytes scale with the factor nonzeros, not with the number of
    elimination updates. ``row_width``, ``column_width`` and ``upper_width``
    bound the fixed per-pivot windows: the longest factor row, the longest
    strictly lower column, and the longest strictly upper row.
    ``ordering_id`` identifies the symmetric ordering ``permutation`` and
    ``ordering_work`` is its share of ``symbolic_work``.
    """

    permutation: Array
    inverse_permutation: Array
    factor_indices: Array
    factor_indptr: Array
    factor_rows: Array
    input_positions: Array
    input_conjugate: Array
    diagonal_positions: Array
    column_positions: Array
    column_offsets: Array
    lower_positions: Array
    upper_positions: Array | None
    lower_analysis: SparseTriangularAnalysis
    upper_analysis: SparseTriangularAnalysis | None
    shape: tuple[int, int] = eqx.field(static=True)
    batch_shape: tuple[int, ...] = eqx.field(static=True)
    kind: Literal["lu", "cholesky"] = eqx.field(static=True)
    policy: SparseFactorizationPolicy = eqx.field(static=True)
    input_pattern_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    input_nnz: int = eqx.field(static=True)
    factor_nnz: int = eqx.field(static=True)
    factor_bytes: int = eqx.field(static=True)
    symbolic_work: int = eqx.field(static=True)
    row_width: int = eqx.field(static=True)
    column_width: int = eqx.field(static=True)
    upper_width: int = eqx.field(static=True)
    ordering_id: str = eqx.field(static=True)
    ordering_work: int = eqx.field(static=True)
    storage_plan: Any = None


class SparseFactorizationDiagnostics(StrictModule):
    """Numerical refresh evidence for one fixed symbolic factor pattern."""

    minimum_pivot: Array
    replaced_pivots: Array
    dropped_entries: Array
    input_nonzeros: Array
    factor_nonzeros: Array
    fill_ratio: Array
    finite: Array


class SparseFactorizationSolveResult(StrictModule):
    """Sparse-factor solve with factor and triangular status evidence."""

    value: Array
    status: Array
    factorization_status: Array
    lower_status: Array
    upper_status: Array

    @property
    def success(self) -> Array:
        return self.status == int(SparseFactorizationStatus.SUCCESS)


class PreparedSparseFactorization(StrictModule):
    """Refreshable sparse factor values paired with immutable symbolic analysis."""

    plan: SparseFactorizationPlan
    factor_values: Array
    status: Array
    diagnostics: SparseFactorizationDiagnostics
    factorization_id: str = eqx.field(static=True)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self.plan.batch_shape

    def solve(
        self,
        right_hand_side: ArrayLike,
        /,
    ) -> SparseFactorizationSolveResult:
        rhs = jnp.asarray(right_hand_side)
        size = self.plan.shape[0]
        shared_vector = rhs.shape == (size,)
        batched_vector = rhs.shape == self.batch_shape + (size,)
        shared_matrix = rhs.ndim == 2 and rhs.shape[0] == size
        batched_matrix = (
            rhs.ndim == len(self.batch_shape) + 2
            and rhs.shape[: len(self.batch_shape)] == self.batch_shape
            and rhs.shape[-2] == size
        )
        if shared_vector:
            rhs = jnp.broadcast_to(rhs, self.batch_shape + (size,))[..., None]
            vector_input = True
        elif batched_vector:
            rhs = rhs[..., None]
            vector_input = True
        elif shared_matrix:
            rhs = jnp.broadcast_to(rhs, self.batch_shape + rhs.shape)
            vector_input = False
        elif batched_matrix:
            vector_input = False
        else:
            raise ValueError(
                "right_hand_side must have shape (n,), (n, k), batch_shape + (n,), or batch_shape + (n, k)."
            )
        batch_count = int(np.prod(self.batch_shape)) if self.batch_shape else 1
        factors = self.factor_values.reshape((batch_count, -1))
        statuses = self.status.reshape((batch_count,))
        right_hand_sides = rhs.reshape((batch_count, size, rhs.shape[-1]))

        def solve_one(
            factor_values: Array, factor_status: Array, value: Array
        ) -> tuple[Array, Array, Array, Array]:
            permuted_rhs = value[self.plan.permutation]
            lower_values = factor_values[self.plan.lower_positions]
            if self.plan.kind == "lu":
                lower_values = jnp.where(
                    self.plan.lower_analysis.row_indices
                    == self.plan.lower_analysis.indices,
                    jnp.ones((), dtype=lower_values.dtype),
                    lower_values,
                )
            lower = solve_sparse_triangular(
                self.plan.lower_analysis,
                lower_values,
                permuted_rhs,
                pivot_tolerance=self.plan.policy.pivot_tolerance,
            )
            if self.plan.kind == "lu":
                if self.plan.upper_analysis is None or self.plan.upper_positions is None:
                    raise ValueError("LU plan is missing upper triangular analysis.")
                upper_values = factor_values[self.plan.upper_positions]
                upper = solve_sparse_triangular(
                    self.plan.upper_analysis,
                    upper_values,
                    lower.value,
                    pivot_tolerance=self.plan.policy.pivot_tolerance,
                )
            else:
                upper = solve_sparse_triangular(
                    self.plan.lower_analysis,
                    lower_values,
                    lower.value,
                    pivot_tolerance=self.plan.policy.pivot_tolerance,
                    adjoint=True,
                )
            solution = (
                jnp.zeros_like(upper.value).at[self.plan.permutation].set(upper.value)
            )
            triangular_success = (lower.status == int(SparseTriangularStatus.SUCCESS)) & (
                upper.status == int(SparseTriangularStatus.SUCCESS)
            )
            triangular_zero_pivot = (
                lower.status == int(SparseTriangularStatus.ZERO_PIVOT)
            ) | (upper.status == int(SparseTriangularStatus.ZERO_PIVOT))
            triangular_nonfinite = (
                lower.status == int(SparseTriangularStatus.NONFINITE)
            ) | (upper.status == int(SparseTriangularStatus.NONFINITE))
            triangular_status = jnp.where(
                triangular_nonfinite,
                int(SparseFactorizationStatus.NONFINITE),
                jnp.where(
                    triangular_zero_pivot,
                    int(SparseFactorizationStatus.ZERO_PIVOT),
                    jnp.where(
                        triangular_success,
                        int(SparseFactorizationStatus.SUCCESS),
                        int(SparseFactorizationStatus.ZERO_PIVOT),
                    ),
                ),
            )
            any_nonfinite = (
                factor_status == int(SparseFactorizationStatus.NONFINITE)
            ) | (triangular_status == int(SparseFactorizationStatus.NONFINITE))
            result_status = jnp.where(
                any_nonfinite,
                int(SparseFactorizationStatus.NONFINITE),
                jnp.where(
                    factor_status != int(SparseFactorizationStatus.SUCCESS),
                    factor_status,
                    triangular_status,
                ),
            ).astype(jnp.int32)
            return solution, result_status, lower.status, upper.status

        value, status, lower_status, upper_status = jax.vmap(solve_one)(
            factors,
            statuses,
            right_hand_sides,
        )
        value = value.reshape(self.batch_shape + (size, rhs.shape[-1]))
        if vector_input:
            value = value[..., 0]
        return SparseFactorizationSolveResult(
            value=value,
            status=status.reshape(self.batch_shape),
            factorization_status=self.status,
            lower_status=lower_status.reshape(self.batch_shape),
            upper_status=upper_status.reshape(self.batch_shape),
        )


@dataclass
class _SymbolicResourceTracker:
    size: int
    kind: Literal["lu", "cholesky"]
    batch_count: int
    value_itemsize: int
    index_itemsize: int
    max_factor_nnz: int
    max_factor_bytes: int
    max_symbolic_work: int
    base_bytes: int
    factor_nnz: int = 0
    lower_nnz: int = 0
    strictly_lower_nnz: int = 0
    upper_nnz: int = 0
    factor_bytes: int = 0
    symbolic_work: int = 0

    def _refuse(
        self,
        metric: str,
        required: int,
        limit: int,
        /,
        *,
        factor_nnz: int | None = None,
        factor_bytes: int | None = None,
        symbolic_work: int | None = None,
    ) -> NoReturn:
        observed_nnz = self.factor_nnz if factor_nnz is None else factor_nnz
        observed_bytes = self.factor_bytes if factor_bytes is None else factor_bytes
        observed_work = self.symbolic_work if symbolic_work is None else symbolic_work
        raise LinearCapabilityError(
            "Sparse symbolic factorization refused before allocation: "
            f"{metric} requires {required}, exceeding limit {limit}; "
            f"factor_nnz={observed_nnz}/{self.max_factor_nnz}, "
            f"factor_bytes={observed_bytes}/{self.max_factor_bytes}, "
            f"symbolic_work={observed_work}/{self.max_symbolic_work}."
        )

    def _retained_bytes(
        self,
        factor_nnz: int,
        lower_nnz: int,
        strictly_lower_nnz: int,
        upper_nnz: int,
        /,
    ) -> int:
        index = self.index_itemsize
        triangular_fixed = index * (4 * self.size + 2) + 8 * self.size
        fixed = self.base_bytes + index * (5 * self.size + 2) + triangular_fixed
        if self.kind == "lu":
            fixed += triangular_fixed
        return (
            fixed
            + self.batch_count * factor_nnz * self.value_itemsize
            + factor_nnz * (3 * index + 1)
            + strictly_lower_nnz * index
            + lower_nnz * 6 * index
            + upper_nnz * 6 * index
        )

    def reserve_fixed_bytes(self, /) -> None:
        required = self._retained_bytes(0, 0, 0, 0)
        if required > self.max_factor_bytes:
            self._refuse(
                "factor_bytes",
                required,
                self.max_factor_bytes,
                factor_bytes=required,
            )
        self.factor_bytes = required

    def add_fixed_bytes(self, count: int, /) -> None:
        projected = self.factor_bytes + count
        if projected > self.max_factor_bytes:
            self._refuse(
                "factor_bytes",
                projected,
                self.max_factor_bytes,
                factor_bytes=projected,
            )
        self.factor_bytes = projected

    def add_work(self, count: int = 1, /) -> None:
        projected = self.symbolic_work + count
        if projected > self.max_symbolic_work:
            self._refuse(
                "symbolic_work",
                projected,
                self.max_symbolic_work,
                symbolic_work=projected,
            )
        self.symbolic_work = projected

    def add_factor_entry(self, row: int, column: int, /) -> None:
        projected_nnz = self.factor_nnz + 1
        projected_lower = self.lower_nnz + (column <= row)
        projected_strictly_lower = self.strictly_lower_nnz + (column < row)
        projected_upper = self.upper_nnz + (self.kind == "lu" and column >= row)
        projected_bytes = self._retained_bytes(
            projected_nnz,
            projected_lower,
            projected_strictly_lower,
            projected_upper,
        )
        if projected_nnz > self.max_factor_nnz:
            self._refuse(
                "factor_nnz",
                projected_nnz,
                self.max_factor_nnz,
                factor_nnz=projected_nnz,
                factor_bytes=projected_bytes,
            )
        if projected_bytes > self.max_factor_bytes:
            self._refuse(
                "factor_bytes",
                projected_bytes,
                self.max_factor_bytes,
                factor_nnz=projected_nnz,
                factor_bytes=projected_bytes,
            )
        self.factor_nnz = projected_nnz
        self.lower_nnz = projected_lower
        self.strictly_lower_nnz = projected_strictly_lower
        self.upper_nnz = projected_upper
        self.factor_bytes = projected_bytes


def _symbolic_resource_tracker(
    storage: SparseStorage,
    policy: SparseFactorizationPolicy,
    kind: Literal["lu", "cholesky"],
    base_bytes: int,
    /,
) -> _SymbolicResourceTracker:
    tracker = _SymbolicResourceTracker(
        size=storage.shape[0],
        kind=kind,
        batch_count=prod(storage.batch_shape or (1,)),
        value_itemsize=storage.values.dtype.itemsize,
        index_itemsize=storage.indices.dtype.itemsize,
        max_factor_nnz=policy.max_factor_nnz,
        max_factor_bytes=policy.max_factor_bytes,
        max_symbolic_work=policy.max_symbolic_work,
        base_bytes=base_bytes,
    )
    tracker.reserve_fixed_bytes()
    return tracker


def _resolved_ordering(
    storage: SparseStorage,
    indices: np.ndarray,
    indptr: np.ndarray,
    pattern_id: str,
    policy: SparseFactorizationPolicy,
    ordering: PreparedSparseOrdering | None,
    tracker: _SymbolicResourceTracker,
    /,
) -> PreparedSparseOrdering:
    """Charge a prepared ordering, or prepare the policy's, before any fill.

    Ordering work is symbolic work: it is charged to ``max_symbolic_work``
    before diagonal or fill entries are reserved, whether it was prepared here
    or earlier, so equal orderings give equal plans.
    """
    if ordering is None:
        return _order_pattern(
            storage.shape[0],
            indices,
            indptr,
            SparseOrderingPolicy(policy.ordering),
            None,
            _WorkMeter(None, tracker.add_work),
        )
    if not isinstance(ordering, PreparedSparseOrdering):
        raise TypeError("ordering must be a PreparedSparseOrdering or None.")
    if ordering.method != policy.ordering:
        raise ValueError(
            f"Prepared ordering method {ordering.method!r} differs from the "
            f"policy ordering {policy.ordering!r}."
        )
    if ordering.pattern_id != pattern_id:
        raise ValueError("Prepared ordering belongs to a different sparse pattern.")
    tracker.add_work(ordering.work)
    return ordering


def _permuted_entries(
    indices: np.ndarray,
    indptr: np.ndarray,
    permutation: np.ndarray,
    tracker: _SymbolicResourceTracker,
    /,
) -> dict[tuple[int, int], tuple[int, bool]]:
    inverse = np.empty_like(permutation)
    inverse[permutation] = np.arange(permutation.size)
    entries: dict[tuple[int, int], tuple[int, bool]] = {}
    for old_row in range(permutation.size):
        new_row = int(inverse[old_row])
        for position in range(indptr[old_row], indptr[old_row + 1]):
            tracker.add_work()
            new_column = int(inverse[indices[position]])
            if tracker.kind == "lu":
                coordinate = (new_row, new_column)
                conjugate = False
            else:
                coordinate = (
                    max(new_row, new_column),
                    min(new_row, new_column),
                )
                conjugate = new_row < new_column
            previous = entries.get(coordinate)
            if previous is None:
                if coordinate[0] != coordinate[1]:
                    tracker.add_factor_entry(*coordinate)
                entries[coordinate] = (position, conjugate)
            elif previous[1] and not conjugate:
                entries[coordinate] = (position, False)
    return entries


def _insert_symbolic_entry(
    rows: list[dict[int, int]],
    column_rows: list[set[int]],
    row: int,
    column: int,
    level: int,
    tracker: _SymbolicResourceTracker,
    /,
) -> None:
    previous = rows[row].get(column)
    if previous is None:
        tracker.add_factor_entry(row, column)
        rows[row][column] = level
        column_rows[column].add(row)
    elif level < previous:
        rows[row][column] = level


def _lu_symbolic_rows(
    size: int,
    entries: dict[tuple[int, int], tuple[int, bool]],
    fill_level: int | None,
    tracker: _SymbolicResourceTracker,
    /,
) -> list[dict[int, int]]:
    rows = [dict() for _ in range(size)]
    column_rows = [set() for _ in range(size)]
    for row, column in entries:
        rows[row][column] = 0
        column_rows[column].add(row)
    for row in range(size):
        rows[row].setdefault(row, 0)
        column_rows[row].add(row)
    if fill_level == 0:
        return rows
    for pivot in range(size):
        upper = [
            (column, level)
            for column, level in sorted(rows[pivot].items())
            if column > pivot
        ]
        below = [row for row in sorted(column_rows[pivot]) if row > pivot]
        for row in below:
            lower_level = rows[row][pivot]
            for column, upper_level in upper:
                tracker.add_work()
                level = lower_level + upper_level + 1
                if fill_level is None or level <= fill_level:
                    _insert_symbolic_entry(
                        rows,
                        column_rows,
                        row,
                        column,
                        level,
                        tracker,
                    )
    return rows


def _cholesky_symbolic_rows(
    size: int,
    entries: dict[tuple[int, int], tuple[int, bool]],
    fill_level: int | None,
    tracker: _SymbolicResourceTracker,
    /,
) -> list[dict[int, int]]:
    rows = [dict() for _ in range(size)]
    column_rows = [set() for _ in range(size)]
    for row, column in entries:
        rows[row][column] = 0
        column_rows[column].add(row)
    for row in range(size):
        rows[row].setdefault(row, 0)
        column_rows[row].add(row)
    if fill_level == 0:
        return rows
    for pivot in range(size):
        neighbors = [row for row in sorted(column_rows[pivot]) if row > pivot]
        for left_index, row in enumerate(neighbors):
            left_level = rows[row][pivot]
            for column in neighbors[: left_index + 1]:
                tracker.add_work()
                right_level = rows[column][pivot]
                level = left_level + right_level + 1
                if fill_level is None or level <= fill_level:
                    _insert_symbolic_entry(
                        rows,
                        column_rows,
                        row,
                        column,
                        level,
                        tracker,
                    )
    return rows


def _csr_from_rows(
    rows: list[dict[int, int]],
    tracker: _SymbolicResourceTracker,
    /,
) -> tuple[np.ndarray, np.ndarray, dict[tuple[int, int], int]]:
    indices: list[int] = []
    indptr = [0]
    positions: dict[tuple[int, int], int] = {}
    for row, columns in enumerate(rows):
        for column in sorted(columns):
            tracker.add_work()
            positions[(row, column)] = len(indices)
            indices.append(column)
        indptr.append(len(indices))
    return (
        np.asarray(indices, dtype=np.int64),
        np.asarray(indptr, dtype=np.int64),
        positions,
    )


def _column_index(
    indices: np.ndarray,
    indptr: np.ndarray,
    diagonal: np.ndarray,
    tracker: _SymbolicResourceTracker,
    /,
) -> tuple[np.ndarray, np.ndarray, int, int, int]:
    """Return the column-major index of strictly lower factor entries and widths.

    ``column_positions[column_offsets[k] : column_offsets[k + 1]]`` are the CSR
    positions of the entries ``(i, k)`` with ``i > k`` in ascending row order.
    The widths are the longest factor row, strictly lower column and strictly
    upper row.
    """
    tracker.add_work(indices.size)
    size = indptr.size - 1
    rows = np.repeat(np.arange(size, dtype=np.int64), np.diff(indptr))
    strictly_lower = np.flatnonzero(indices < rows)
    columns = indices[strictly_lower]
    # A stable sort keeps CSR (ascending-row) order inside every column.
    column_positions = strictly_lower[np.argsort(columns, kind="stable")]
    counts = np.bincount(columns, minlength=size)
    column_offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
    return (
        column_positions,
        column_offsets,
        int(np.diff(indptr).max(initial=0)),
        int(counts.max(initial=0)),
        int((indptr[1:] - diagonal - 1).max(initial=0)),
    )


def _triangular_pattern(
    rows: list[dict[int, int]],
    combined_positions: dict[tuple[int, int], int],
    triangle: Literal["lower", "upper"],
    index_dtype: DTypeLike,
    tracker: _SymbolicResourceTracker,
    /,
    *,
    unit_diagonal: bool,
) -> tuple[SparseTriangularAnalysis, np.ndarray]:
    selected: list[list[int]] = []
    factor_positions: list[int] = []
    for row, columns in enumerate(rows):
        kept: list[int] = []
        for column in sorted(columns):
            tracker.add_work()
            if column <= row if triangle == "lower" else column >= row:
                kept.append(column)
        selected.append(kept)
        factor_positions.extend(combined_positions[(row, column)] for column in kept)
    triangular_nnz = len(factor_positions)
    tracker.add_work(2 * triangular_nnz)
    indices = np.asarray([column for row in selected for column in row], dtype=np.int64)
    indptr = np.concatenate(
        ([0], np.cumsum([len(row) for row in selected], dtype=np.int64))
    )
    storage = SparseStorage(
        jnp.ones((indices.size,), dtype=jnp.float64),
        jnp.asarray(indices, dtype=index_dtype),
        jnp.asarray(indptr, dtype=index_dtype),
        shape=(len(rows), len(rows)),
    )
    return (
        analyze_sparse_triangular(
            storage,
            triangle=triangle,
            unit_diagonal=unit_diagonal,
        ),
        np.asarray(factor_positions, dtype=np.int64),
    )


# Symbolic planning reads concrete host patterns. Evaluate it eagerly even
# when a caller prepares inside an ambient trace such as a guarded cond.
@jax.ensure_compile_time_eval()
def prepare_sparse_factorization(
    operator: AbstractSparseLinearOperator,
    policy: SparseFactorizationPolicy | None = None,
    /,
    *,
    ordering: PreparedSparseOrdering | None = None,
) -> SparseFactorizationPlan:
    """Build a bounded host symbolic factorization plan without reading values.

    Factor fill is bounded by the policy's own ``max_factor_nnz``,
    ``max_factor_bytes`` and ``max_symbolic_work``. Factoring stored sparse
    values is not a dense materialization, so ``MaterializationPolicy`` does not
    apply; solve plans charge the retained factor to
    ``SolveResourcePolicy.preconditioner_bytes``.

    ``ordering`` reuses a `PreparedSparseOrdering` of this exact pattern whose
    method matches ``policy.ordering``; otherwise the policy's ordering is
    prepared with default capacities. Ordering work is charged first, so a
    work cap below it refuses before any factor entry is reserved.
    """
    policy_ = SparseFactorizationPolicy() if policy is None else policy
    if not isinstance(policy_, SparseFactorizationPolicy):
        raise TypeError("policy must be SparseFactorizationPolicy or None.")
    storage, input_indices, input_indptr = _validated_pattern(operator)
    # Canonicalization is symbolic work. Retain its route-to-CSR scatter for
    # numeric Jacobians carried through cond/while_loop, where even invariant
    # relation indices are traced arrays and cannot be converted to NumPy.
    from ..sparse._linear import (
        _SparseStoragePlan,
        SparseCoordinateOperator,
        SparseLinearMap,
    )

    if (
        isinstance(operator, SparseCoordinateOperator)
        and operator._storage_plan is not None
    ):
        storage_plan = operator._storage_plan
    elif isinstance(operator, (SparseCoordinateOperator, SparseLinearMap)):
        storage_plan = _SparseStoragePlan(
            operator.relation,
            block_shape=operator.block_shape
            if isinstance(operator, SparseCoordinateOperator)
            else None,
        )
    else:
        storage_plan = None
    storage_plan_arrays = (
        {}
        if storage_plan is None
        else {
            id(leaf): leaf for leaf in jax.tree.leaves(storage_plan) if eqx.is_array(leaf)
        }
    )
    base_bytes = sum(
        array.size * array.dtype.itemsize for array in storage_plan_arrays.values()
    )
    kind: Literal["lu", "cholesky"]
    if policy_.kind == "auto":
        kind = "cholesky" if operator.properties.certifies("positive_definite") else "lu"
    else:
        kind = policy_.kind
    if kind == "cholesky" and not operator.properties.certifies("self_adjoint"):
        raise ValueError("Sparse Cholesky requires a certified self-adjoint operator.")
    tracker = _symbolic_resource_tracker(storage, policy_, kind, base_bytes)
    input_pattern_id = _pattern_identifier(storage.shape, input_indices, input_indptr)
    prepared_ordering = _resolved_ordering(
        storage,
        input_indices,
        input_indptr,
        input_pattern_id,
        policy_,
        ordering,
        tracker,
    )
    ordering_work = tracker.symbolic_work
    for row in range(storage.shape[0]):
        tracker.add_work()
        tracker.add_factor_entry(row, row)
    permutation = prepared_ordering.permutation
    inverse = prepared_ordering.inverse_permutation
    entries = _permuted_entries(input_indices, input_indptr, permutation, tracker)
    rows = (
        _lu_symbolic_rows(storage.shape[0], entries, policy_.fill_level, tracker)
        if kind == "lu"
        else _cholesky_symbolic_rows(
            storage.shape[0],
            entries,
            policy_.fill_level,
            tracker,
        )
    )
    factor_indices, factor_indptr, positions = _csr_from_rows(rows, tracker)
    input_positions = np.full(factor_indices.size, -1, dtype=np.int64)
    input_conjugate = np.zeros(factor_indices.size, dtype=np.bool_)
    for coordinate, factor_position in positions.items():
        tracker.add_work()
        route = entries.get(coordinate)
        if route is not None:
            input_positions[factor_position] = route[0]
            input_conjugate[factor_position] = route[1]
    diagonal = np.asarray(
        [positions[(row, row)] for row in range(storage.shape[0])], dtype=np.int64
    )
    (
        column_positions,
        column_offsets,
        row_width,
        column_width,
        upper_width,
    ) = _column_index(factor_indices, factor_indptr, diagonal, tracker)
    lower_analysis, lower_positions = _triangular_pattern(
        rows,
        positions,
        "lower",
        storage.indices.dtype,
        tracker,
        unit_diagonal=kind == "lu",
    )
    if kind == "lu":
        upper_analysis, upper_positions = _triangular_pattern(
            rows,
            positions,
            "upper",
            storage.indices.dtype,
            tracker,
            unit_diagonal=False,
        )
    else:
        upper_analysis = None
        upper_positions = None
    plan_payload = b"|".join(
        (
            input_pattern_id.encode(),
            kind.encode(),
            prepared_ordering.ordering_id.encode(),
            str(policy_.fill_level).encode(),
            str(storage.batch_shape).encode(),
            str(storage.index_width).encode(),
            str(policy_.max_factor_nnz).encode(),
            str(policy_.max_factor_bytes).encode(),
            str(policy_.max_symbolic_work).encode(),
            str(tracker.factor_bytes).encode(),
            str(tracker.symbolic_work).encode(),
            factor_indices.tobytes(),
            factor_indptr.tobytes(),
        )
    )
    # Level schedules are sized by the analyzed level histogram, so they are
    # charged once both analyses exist; the plan identity keeps the
    # pattern-determined bytes counted during symbolic construction.
    tracker.add_fixed_bytes(
        sum(
            analysis.level_schedule.nbytes + analysis.transpose_level_schedule.nbytes
            for analysis in (lower_analysis, upper_analysis)
            if analysis is not None
        )
    )
    index_dtype = storage.indices.dtype
    return SparseFactorizationPlan(
        permutation=jnp.asarray(permutation, dtype=index_dtype),
        inverse_permutation=jnp.asarray(inverse, dtype=index_dtype),
        factor_indices=jnp.asarray(factor_indices, dtype=index_dtype),
        factor_indptr=jnp.asarray(factor_indptr, dtype=index_dtype),
        factor_rows=jnp.asarray(
            np.repeat(
                np.arange(storage.shape[0], dtype=np.int64),
                np.diff(factor_indptr),
            ),
            dtype=index_dtype,
        ),
        input_positions=jnp.asarray(input_positions, dtype=index_dtype),
        input_conjugate=jnp.asarray(input_conjugate),
        diagonal_positions=jnp.asarray(diagonal, dtype=index_dtype),
        column_positions=jnp.asarray(column_positions, dtype=index_dtype),
        column_offsets=jnp.asarray(column_offsets, dtype=index_dtype),
        lower_positions=jnp.asarray(lower_positions, dtype=index_dtype),
        upper_positions=(
            None
            if upper_positions is None
            else jnp.asarray(upper_positions, dtype=index_dtype)
        ),
        lower_analysis=lower_analysis,
        upper_analysis=upper_analysis,
        shape=storage.shape,
        batch_shape=storage.batch_shape,
        kind=kind,
        policy=policy_,
        input_pattern_id=input_pattern_id,
        plan_id=sha256(plan_payload).hexdigest(),
        input_nnz=input_indices.size,
        factor_nnz=tracker.factor_nnz,
        factor_bytes=tracker.factor_bytes,
        symbolic_work=tracker.symbolic_work,
        row_width=row_width,
        column_width=column_width,
        upper_width=upper_width,
        ordering_id=prepared_ordering.ordering_id,
        ordering_work=ordering_work,
        storage_plan=storage_plan,
    )


def _window(start: Array, stop: Array, width: int, /) -> tuple[Array, Array]:
    """Return ``width`` consecutive positions from ``start`` masked below ``stop``.

    Masked positions are replaced by zero so gathers stay in bounds; scatters
    route them out of bounds and drop them.
    """
    positions = start + jnp.arange(width, dtype=start.dtype)
    valid = positions < stop
    return jnp.where(valid, positions, 0), valid


def _prune_row(
    values: Array,
    plan: SparseFactorizationPlan,
    row: Array,
    /,
) -> tuple[Array, Array]:
    safe_positions, valid = _window(
        plan.factor_indptr[row], plan.factor_indptr[row + 1], plan.row_width
    )
    row_values = values[safe_positions]
    diagonal = safe_positions == plan.diagonal_positions[row]
    row_scale = jnp.max(jnp.where(valid, jnp.abs(row_values), 0.0))
    threshold_keep = jnp.abs(row_values) >= (plan.policy.drop_tolerance * row_scale)
    candidate = valid & ~diagonal & threshold_keep
    if plan.policy.maximum_fill_per_row is None:
        selected = candidate
    elif plan.policy.maximum_fill_per_row == 0:
        selected = jnp.zeros_like(candidate)
    else:
        count = min(plan.policy.maximum_fill_per_row, plan.row_width)
        scores = jnp.where(candidate, jnp.abs(row_values), -jnp.inf)
        _, selected_indices = jax.lax.top_k(scores, count)
        selected = (
            jnp.any(
                jnp.arange(scores.size)[:, None] == selected_indices[None, :],
                axis=1,
            )
            & candidate
        )
    removed = valid & ~(diagonal | selected)
    pruned = values.at[jnp.where(removed, safe_positions, values.size)].set(
        jnp.zeros((), values.dtype), mode="drop"
    )
    return pruned, jnp.sum(removed, dtype=jnp.int32)


def _eliminate_pivot(
    values: Array,
    marker: Array,
    plan: SparseFactorizationPlan,
    pivot: Array,
    denominator: Array,
    /,
) -> tuple[Array, Array]:
    """Divide pivot column ``k`` and apply its rank-one update on the pattern.

    Targets come from the stored factor rows at runtime: ``marker`` maps every
    column of the pivot's update row (``U[k, j > k]`` for LU, the conjugate
    column ``L[j > k, k]`` for Cholesky) to its window slot. Each target row's
    entries right of its pivot-column entry look up the marker, so an update
    is applied exactly where the symbolic pattern holds ``(i, j)``. Targets are
    unique within one pivot, which keeps the scatter deterministic.
    """
    size = values.size
    below, below_valid = _window(
        plan.column_offsets[pivot], plan.column_offsets[pivot + 1], plan.column_width
    )
    below = plan.column_positions[below]
    multipliers = values[below] / denominator
    values = values.at[jnp.where(below_valid, below, size)].set(multipliers, mode="drop")
    if plan.kind == "lu":
        right, right_valid = _window(
            plan.diagonal_positions[pivot] + 1,
            plan.factor_indptr[pivot + 1],
            plan.upper_width,
        )
        keys = plan.factor_indices[right]
        right_values = values[right]
    else:
        right_valid = below_valid
        keys = plan.factor_rows[below]
        right_values = jnp.conj(multipliers)
    marked = jnp.where(right_valid, keys, marker.size)
    marker = marker.at[marked].set(jnp.arange(keys.size, dtype=marker.dtype), mode="drop")
    rows = plan.factor_rows[below]
    tail = below[:, None] + 1 + jnp.arange(plan.row_width, dtype=below.dtype)[None, :]
    tail_valid = below_valid[:, None] & (tail < plan.factor_indptr[rows + 1][:, None])
    tail = jnp.where(tail_valid, tail, 0)
    slot = marker[plan.factor_indices[tail]]
    hit = tail_valid & (slot >= 0)
    updates = multipliers[:, None] * right_values[jnp.where(hit, slot, 0)]
    values = values.at[jnp.where(hit, tail, size)].add(-updates, mode="drop")
    return values, marker.at[marked].set(-1, mode="drop")


def _refresh_workspace_bytes(plan: SparseFactorizationPlan, itemsize: int, /) -> int:
    """Transient bytes of one numeric refresh besides the factor values.

    One pivot holds the dense column marker and a ``column_width x row_width``
    target window (positions, columns, slots, update values and masks).
    """
    index = plan.factor_indices.dtype.itemsize
    window = plan.column_width * plan.row_width * (3 * index + itemsize + 2)
    return plan.shape[0] * index + window


def refresh_sparse_factorization(
    plan: SparseFactorizationPlan,
    operator: AbstractSparseLinearOperator,
    /,
) -> PreparedSparseFactorization:
    """Refresh independent numeric factors under one shared CSR pattern."""
    if not isinstance(plan, SparseFactorizationPlan):
        raise TypeError("plan must be a SparseFactorizationPlan.")
    from ..sparse._linear import SparseCoordinateOperator, SparseLinearMap

    routed = isinstance(operator, (SparseCoordinateOperator, SparseLinearMap))
    traced_routes = routed and any(
        isinstance(leaf, jax_core.Tracer)
        for leaf in jax.tree_util.tree_leaves(operator.relation)
    )
    if (
        isinstance(operator, (SparseCoordinateOperator, SparseLinearMap))
        and traced_routes
        and plan.storage_plan is not None
    ):
        storage = plan.storage_plan.apply(
            operator.coefficients, relation=operator.relation
        )
    else:
        storage, indices, indptr = _validated_pattern(operator)
        if _pattern_identifier(storage.shape, indices, indptr) != plan.input_pattern_id:
            raise ValueError(
                "Sparse factorization refresh requires an unchanged CSR pattern."
            )
    if storage.batch_shape != plan.batch_shape:
        raise ValueError(
            "Sparse factorization refresh requires an unchanged value batch shape."
        )
    return refresh_sparse_factorization_values(plan, storage.values)


def refresh_sparse_factorization_values(
    plan: SparseFactorizationPlan,
    values: ArrayLike,
    /,
) -> PreparedSparseFactorization:
    """Refresh traced numeric values in the plan's original CSR entry order.

    Symbolic analysis and pattern validation belong to
    :func:`prepare_sparse_factorization`. This numeric-only boundary accepts
    exactly ``plan.batch_shape + (plan.input_nnz,)`` values without inspecting
    traced CSR indices on the host.
    """
    if not isinstance(plan, SparseFactorizationPlan):
        raise TypeError("plan must be a SparseFactorizationPlan.")
    numeric_values = jnp.asarray(values)
    if numeric_values.shape != plan.batch_shape + (plan.input_nnz,):
        raise ValueError("Sparse factorization values must match the plan's input shape.")
    if not jnp.issubdtype(numeric_values.dtype, jnp.inexact):
        raise TypeError("Sparse factorization values must have an inexact dtype.")

    def factor_one(
        input_values: Array,
    ) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
        safe_input = jnp.maximum(plan.input_positions, 0)
        gathered = input_values[safe_input]
        gathered = jnp.where(plan.input_conjugate, jnp.conj(gathered), gathered)
        values = jnp.where(
            plan.input_positions >= 0,
            gathered,
            jnp.zeros((), dtype=input_values.dtype),
        )
        values = values.at[plan.diagonal_positions].add(plan.policy.diagonal_shift)
        initial = (
            values,
            jnp.full((plan.shape[0],), -1, dtype=plan.factor_indices.dtype),
            jnp.asarray(int(SparseFactorizationStatus.SUCCESS), dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(jnp.inf, dtype=values.real.dtype),
        )

        def factor_step(
            pivot_index: Array, carry: tuple[Array, Array, Array, Array, Array, Array]
        ) -> tuple[Array, Array, Array, Array, Array, Array]:
            current, marker, status, replaced, dropped, minimum_pivot = carry
            if (
                plan.policy.drop_tolerance > 0.0
                or plan.policy.maximum_fill_per_row is not None
            ):
                current, row_dropped = _prune_row(current, plan, pivot_index)
                dropped = dropped + row_dropped
            pivot_position = plan.diagonal_positions[pivot_index]
            pivot = current[pivot_position]
            finite = jnp.isfinite(pivot)
            if plan.kind == "cholesky":
                real_pivot = jnp.real(pivot)
                real_dtype = real_pivot.dtype
                hermitian_roundoff = (
                    64.0
                    * jnp.finfo(real_dtype).eps
                    * jnp.maximum(
                        jnp.ones((), dtype=real_dtype),
                        jnp.abs(real_pivot),
                    )
                )
                imaginary_tolerance = jnp.maximum(
                    jnp.asarray(plan.policy.pivot_tolerance, dtype=real_dtype),
                    hermitian_roundoff,
                )
                acceptable = (
                    finite
                    & (real_pivot > plan.policy.pivot_tolerance)
                    & (jnp.abs(jnp.imag(pivot)) <= imaginary_tolerance)
                )
                failure_status = int(SparseFactorizationStatus.NONPOSITIVE_PIVOT)
                replacement = jnp.asarray(
                    max(
                        plan.policy.replacement_value,
                        plan.policy.pivot_tolerance,
                    ),
                    dtype=current.dtype,
                )
            else:
                acceptable = finite & (jnp.abs(pivot) > plan.policy.pivot_tolerance)
                failure_status = int(SparseFactorizationStatus.ZERO_PIVOT)
                phase = jnp.where(
                    jnp.abs(pivot) > 0.0,
                    pivot / jnp.abs(pivot),
                    jnp.ones((), dtype=current.dtype),
                )
                replacement = phase * jnp.asarray(
                    max(
                        plan.policy.replacement_value,
                        plan.policy.pivot_tolerance,
                    ),
                    dtype=current.dtype,
                )
            bad = ~acceptable
            status = jnp.where(
                ~finite,
                int(SparseFactorizationStatus.NONFINITE),
                jnp.where(
                    (status == int(SparseFactorizationStatus.SUCCESS)) & bad,
                    failure_status,
                    status,
                ),
            ).astype(jnp.int32)
            use_replacement = bad & plan.policy.allow_pivot_replacement
            effective_pivot = jnp.where(
                acceptable,
                pivot,
                jnp.where(
                    use_replacement,
                    replacement,
                    jnp.ones((), dtype=current.dtype),
                ),
            )
            replaced = replaced + use_replacement.astype(jnp.int32)
            status = jnp.where(
                use_replacement & (status == failure_status),
                int(SparseFactorizationStatus.SUCCESS),
                status,
            ).astype(jnp.int32)
            minimum_pivot = jnp.minimum(minimum_pivot, jnp.abs(effective_pivot))
            if plan.kind == "cholesky":
                factor_pivot = jnp.sqrt(jnp.real(effective_pivot)).astype(current.dtype)
                current = current.at[pivot_position].set(factor_pivot)
                denominator = factor_pivot
            else:
                current = current.at[pivot_position].set(effective_pivot)
                denominator = effective_pivot
            current, marker = _eliminate_pivot(
                current, marker, plan, pivot_index, denominator
            )
            return current, marker, status, replaced, dropped, minimum_pivot

        values, _, status, replaced, dropped, minimum_pivot = jax.lax.fori_loop(
            0,
            plan.shape[0],
            factor_step,
            initial,
        )
        finite = jnp.all(jnp.isfinite(values))
        status = jnp.where(
            ~finite,
            int(SparseFactorizationStatus.NONFINITE),
            status,
        ).astype(jnp.int32)
        factor_nonzeros = jnp.count_nonzero(values).astype(jnp.int32)
        return (
            values,
            status,
            minimum_pivot,
            replaced,
            dropped,
            factor_nonzeros,
            finite,
        )

    batch_count = int(np.prod(plan.batch_shape)) if plan.batch_shape else 1
    flattened_input = numeric_values.reshape((batch_count, plan.input_nnz))
    (
        values,
        status,
        minimum_pivot,
        replaced,
        dropped,
        factor_nonzeros,
        finite,
    ) = jax.vmap(factor_one)(flattened_input)
    factor_size = plan.factor_indices.size
    values = values.reshape(plan.batch_shape + (factor_size,))
    status = status.reshape(plan.batch_shape)
    minimum_pivot = minimum_pivot.reshape(plan.batch_shape)
    replaced = replaced.reshape(plan.batch_shape)
    dropped = dropped.reshape(plan.batch_shape)
    factor_nonzeros = factor_nonzeros.reshape(plan.batch_shape)
    finite = finite.reshape(plan.batch_shape)
    diagnostics = SparseFactorizationDiagnostics(
        minimum_pivot=minimum_pivot,
        replaced_pivots=replaced,
        dropped_entries=dropped,
        input_nonzeros=jnp.full(
            plan.batch_shape,
            plan.input_nnz,
            dtype=jnp.int32,
        ),
        factor_nonzeros=factor_nonzeros,
        fill_ratio=factor_nonzeros / max(plan.input_nnz, 1),
        finite=finite,
    )
    return PreparedSparseFactorization(
        plan=plan,
        factor_values=values,
        status=status,
        diagnostics=diagnostics,
        factorization_id=f"{plan.plan_id}/numeric",
    )


def factorize_sparse(
    operator: AbstractSparseLinearOperator,
    policy: SparseFactorizationPolicy | None = None,
    /,
) -> PreparedSparseFactorization:
    """Symbolically plan and numerically factor one sparse operator."""
    plan = prepare_sparse_factorization(operator, policy)
    return refresh_sparse_factorization(plan, operator)


__all__ = [
    "PreparedSparseFactorization",
    "SparseFactorizationDiagnostics",
    "SparseFactorizationKind",
    "SparseFactorizationPlan",
    "SparseFactorizationPolicy",
    "SparseFactorizationSolveResult",
    "SparseFactorizationStatus",
    "factorize_sparse",
    "prepare_sparse_factorization",
    "refresh_sparse_factorization",
    "refresh_sparse_factorization_values",
]
