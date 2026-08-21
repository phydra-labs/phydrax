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
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ._sparse_contract import AbstractSparseLinearOperator, SparseStorage
from ._sparse_triangular import (
    analyze_sparse_triangular,
    solve_sparse_triangular,
    SparseTriangularAnalysis,
    SparseTriangularStatus,
)


SparseFactorizationKind: TypeAlias = Literal["auto", "lu", "cholesky"]
SparseOrdering: TypeAlias = Literal["natural", "reverse-cuthill-mckee"]


class SparseFactorizationStatus(IntEnum):
    SUCCESS = 0
    ZERO_PIVOT = 1
    NONPOSITIVE_PIVOT = 2
    NONFINITE = 3


class SparseFactorizationPolicy(StrictModule):
    """Symbolic fill, ordering, dropping, and explicit pivot-replacement policy."""

    kind: SparseFactorizationKind = eqx.field(static=True)
    ordering: SparseOrdering = eqx.field(static=True)
    fill_level: int | None = eqx.field(static=True)
    drop_tolerance: float = eqx.field(static=True)
    maximum_fill_per_row: int | None = eqx.field(static=True)
    pivot_tolerance: float = eqx.field(static=True)
    diagonal_shift: float = eqx.field(static=True)
    allow_pivot_replacement: bool = eqx.field(static=True)
    replacement_value: float = eqx.field(static=True)

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
    ):
        if kind not in ("auto", "lu", "cholesky"):
            raise ValueError(f"Unknown sparse factorization kind {kind!r}.")
        if ordering not in ("natural", "reverse-cuthill-mckee"):
            raise ValueError(f"Unknown sparse ordering {ordering!r}.")
        fill = None if fill_level is None else int(fill_level)
        maximum_fill = None if maximum_fill_per_row is None else int(maximum_fill_per_row)
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


class SparseFactorizationPlan(StrictModule):
    """Immutable host symbolic plan for refreshable sparse LU or Cholesky values."""

    permutation: Array
    inverse_permutation: Array
    factor_indices: Array
    factor_indptr: Array
    factor_rows: Array
    input_positions: Array
    input_conjugate: Array
    diagonal_positions: Array
    multiplier_positions: Array
    multiplier_valid: Array
    update_targets: Array
    update_left: Array
    update_right: Array
    update_valid: Array
    row_positions: Array
    row_valid: Array
    lower_positions: Array
    upper_positions: Array | None
    lower_analysis: SparseTriangularAnalysis
    upper_analysis: SparseTriangularAnalysis | None
    shape: tuple[int, int] = eqx.field(static=True)
    kind: Literal["lu", "cholesky"] = eqx.field(static=True)
    policy: SparseFactorizationPolicy = eqx.field(static=True)
    input_pattern_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    input_nnz: int = eqx.field(static=True)


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

    def solve(
        self,
        right_hand_side: ArrayLike,
        /,
    ) -> SparseFactorizationSolveResult:
        rhs = jnp.asarray(right_hand_side)
        vector_input = rhs.ndim == 1
        if vector_input:
            rhs = rhs[:, None]
        if rhs.ndim != 2 or rhs.shape[0] != self.plan.shape[0]:
            raise ValueError("right_hand_side must have shape (n,) or (n, k).")
        permuted_rhs = rhs[self.plan.permutation]
        lower_values = self.factor_values[self.plan.lower_positions]
        if self.plan.kind == "lu":
            lower_values = jnp.where(
                self.plan.lower_analysis.row_indices == self.plan.lower_analysis.indices,
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
                raise ValueError("LU plan is missing its upper triangular analysis.")
            upper_values = self.factor_values[self.plan.upper_positions]
            upper = solve_sparse_triangular(
                self.plan.upper_analysis,
                upper_values,
                lower.value,
                pivot_tolerance=self.plan.policy.pivot_tolerance,
            )
            permuted_solution = upper.value
            upper_status = upper.status
        else:
            upper = solve_sparse_triangular(
                self.plan.lower_analysis,
                lower_values,
                lower.value,
                pivot_tolerance=self.plan.policy.pivot_tolerance,
                adjoint=True,
            )
            permuted_solution = upper.value
            upper_status = upper.status
        solution = (
            jnp.zeros_like(permuted_solution)
            .at[self.plan.permutation]
            .set(permuted_solution)
        )
        triangular_success = (lower.status == int(SparseTriangularStatus.SUCCESS)) & (
            upper_status == int(SparseTriangularStatus.SUCCESS)
        )
        status = jnp.where(
            self.status != int(SparseFactorizationStatus.SUCCESS),
            self.status,
            jnp.where(
                triangular_success,
                int(SparseFactorizationStatus.SUCCESS),
                int(SparseFactorizationStatus.ZERO_PIVOT),
            ),
        ).astype(jnp.int32)
        output = solution[:, 0] if vector_input else solution
        return SparseFactorizationSolveResult(
            value=output,
            status=status,
            factorization_status=self.status,
            lower_status=lower.status,
            upper_status=upper_status,
        )


def _validated_pattern(
    operator: AbstractSparseLinearOperator,
    /,
) -> tuple[SparseStorage, np.ndarray, np.ndarray]:
    if not isinstance(operator, AbstractSparseLinearOperator):
        raise TypeError("operator must be an AbstractSparseLinearOperator.")
    storage = operator.sparse_storage()
    if storage.shape[0] != storage.shape[1]:
        raise ValueError("Sparse factorization requires a square operator.")
    indices = np.asarray(storage.indices, dtype=np.int64)
    indptr = np.asarray(storage.indptr, dtype=np.int64)
    if indptr[0] != 0 or indptr[-1] != indices.size:
        raise ValueError("CSR indptr endpoints are inconsistent with its indices.")
    if np.any(indptr[1:] < indptr[:-1]):
        raise ValueError("CSR indptr must be nondecreasing.")
    if np.any(indices < 0) or np.any(indices >= storage.shape[1]):
        raise ValueError("CSR column index is out of range.")
    for row in range(storage.shape[0]):
        columns = indices[indptr[row] : indptr[row + 1]]
        if columns.size > 1 and np.any(columns[1:] <= columns[:-1]):
            raise ValueError("Sparse factorization requires canonical sorted CSR rows.")
    return storage, indices, indptr


def _pattern_identifier(
    shape: tuple[int, int], indices: np.ndarray, indptr: np.ndarray, /
) -> str:
    payload = b"|".join(
        (
            np.asarray(shape, dtype=np.int64).tobytes(),
            indices.tobytes(),
            indptr.tobytes(),
        )
    )
    return sha256(payload).hexdigest()


def _permutation(
    shape: tuple[int, int],
    indices: np.ndarray,
    indptr: np.ndarray,
    ordering: SparseOrdering,
    /,
) -> np.ndarray:
    if ordering == "natural":
        return np.arange(shape[0], dtype=np.int64)
    import scipy.sparse as sp
    from scipy.sparse.csgraph import reverse_cuthill_mckee

    graph = sp.csr_matrix((np.ones(indices.size), indices, indptr), shape=shape)
    symmetric = graph + graph.T
    return np.asarray(
        reverse_cuthill_mckee(symmetric, symmetric_mode=True), dtype=np.int64
    )


def _permuted_entries(
    indices: np.ndarray,
    indptr: np.ndarray,
    permutation: np.ndarray,
    /,
) -> dict[tuple[int, int], int]:
    inverse = np.empty_like(permutation)
    inverse[permutation] = np.arange(permutation.size)
    entries: dict[tuple[int, int], int] = {}
    for old_row in range(permutation.size):
        new_row = int(inverse[old_row])
        for position in range(indptr[old_row], indptr[old_row + 1]):
            new_column = int(inverse[indices[position]])
            entries[(new_row, new_column)] = position
    return entries


def _lu_symbolic_rows(
    size: int,
    entries: dict[tuple[int, int], int],
    fill_level: int | None,
    /,
) -> list[dict[int, int]]:
    rows = [dict() for _ in range(size)]
    for row, column in entries:
        rows[row][column] = 0
    for row in range(size):
        rows[row].setdefault(row, 0)
    for pivot in range(size):
        upper = tuple(
            (column, level)
            for column, level in sorted(rows[pivot].items())
            if column > pivot
        )
        for row in range(pivot + 1, size):
            if pivot not in rows[row]:
                continue
            lower_level = rows[row][pivot]
            for column, upper_level in upper:
                level = lower_level + upper_level + 1
                if fill_level is None or level <= fill_level:
                    previous = rows[row].get(column)
                    if previous is None or level < previous:
                        rows[row][column] = level
    return rows


def _cholesky_symbolic_rows(
    size: int,
    entries: dict[tuple[int, int], int],
    fill_level: int | None,
    /,
) -> list[dict[int, int]]:
    rows = [dict() for _ in range(size)]
    for row, column in entries:
        lower_row, lower_column = max(row, column), min(row, column)
        rows[lower_row][lower_column] = 0
    for row in range(size):
        rows[row].setdefault(row, 0)
    for pivot in range(size):
        neighbors = [row for row in range(pivot + 1, size) if pivot in rows[row]]
        for left_index, row in enumerate(neighbors):
            left_level = rows[row][pivot]
            for column in neighbors[: left_index + 1]:
                right_level = rows[column][pivot]
                level = left_level + right_level + 1
                if fill_level is None or level <= fill_level:
                    previous = rows[row].get(column)
                    if previous is None or level < previous:
                        rows[row][column] = level
    return rows


def _csr_from_rows(
    rows: list[dict[int, int]], /
) -> tuple[np.ndarray, np.ndarray, dict[tuple[int, int], int]]:
    indices: list[int] = []
    indptr = [0]
    positions: dict[tuple[int, int], int] = {}
    for row, columns in enumerate(rows):
        for column in sorted(columns):
            positions[(row, column)] = len(indices)
            indices.append(column)
        indptr.append(len(indices))
    return (
        np.asarray(indices, dtype=np.int64),
        np.asarray(indptr, dtype=np.int64),
        positions,
    )


def _padded(rows: list[list[int]], /, *, fill: int = 0) -> tuple[np.ndarray, np.ndarray]:
    width = max((len(row) for row in rows), default=0)
    width = max(width, 1)
    values = np.full((len(rows), width), fill, dtype=np.int64)
    valid = np.zeros((len(rows), width), dtype=bool)
    for index, row in enumerate(rows):
        values[index, : len(row)] = row
        valid[index, : len(row)] = True
    return values, valid


def _operation_tables(
    kind: Literal["lu", "cholesky"],
    rows: list[dict[int, int]],
    positions: dict[tuple[int, int], int],
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    size = len(rows)
    multipliers: list[list[int]] = []
    targets: list[list[int]] = []
    lefts: list[list[int]] = []
    rights: list[list[int]] = []
    for pivot in range(size):
        below = [row for row in range(pivot + 1, size) if pivot in rows[row]]
        multipliers.append([positions[(row, pivot)] for row in below])
        pivot_targets: list[int] = []
        pivot_lefts: list[int] = []
        pivot_rights: list[int] = []
        if kind == "lu":
            upper = [column for column in rows[pivot] if column > pivot]
            for row in below:
                for column in upper:
                    target = positions.get((row, column))
                    if target is not None:
                        pivot_targets.append(target)
                        pivot_lefts.append(positions[(row, pivot)])
                        pivot_rights.append(positions[(pivot, column)])
        else:
            for left_index, row in enumerate(below):
                for column in below[: left_index + 1]:
                    target = positions.get((row, column))
                    if target is not None:
                        pivot_targets.append(target)
                        pivot_lefts.append(positions[(row, pivot)])
                        pivot_rights.append(positions[(column, pivot)])
        targets.append(pivot_targets)
        lefts.append(pivot_lefts)
        rights.append(pivot_rights)
    multiplier_values, multiplier_valid = _padded(multipliers)
    target_values, update_valid = _padded(targets)
    left_values, _ = _padded(lefts)
    right_values, _ = _padded(rights)
    return (
        multiplier_values,
        multiplier_valid,
        target_values,
        left_values,
        right_values,
        update_valid,
    )


def _triangular_pattern(
    rows: list[dict[int, int]],
    combined_positions: dict[tuple[int, int], int],
    triangle: Literal["lower", "upper"],
    index_dtype,
    /,
    *,
    unit_diagonal: bool,
) -> tuple[SparseTriangularAnalysis, np.ndarray]:
    selected: list[list[int]] = []
    factor_positions: list[int] = []
    for row, columns in enumerate(rows):
        kept = [
            column
            for column in sorted(columns)
            if (column <= row if triangle == "lower" else column >= row)
        ]
        selected.append(kept)
        factor_positions.extend(combined_positions[(row, column)] for column in kept)
    indices = np.asarray([column for row in selected for column in row], dtype=np.int64)
    indptr = np.concatenate(
        ([0], np.cumsum([len(row) for row in selected], dtype=np.int64))
    )
    storage = SparseStorage(
        jnp.ones((indices.size,), dtype=float),
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


def prepare_sparse_factorization(
    operator: AbstractSparseLinearOperator,
    policy: SparseFactorizationPolicy | None = None,
    /,
) -> SparseFactorizationPlan:
    """Build a host symbolic sparse factorization plan without reading values."""
    policy_ = SparseFactorizationPolicy() if policy is None else policy
    if not isinstance(policy_, SparseFactorizationPolicy):
        raise TypeError("policy must be SparseFactorizationPolicy or None.")
    storage, input_indices, input_indptr = _validated_pattern(operator)
    kind: Literal["lu", "cholesky"]
    if policy_.kind == "auto":
        kind = "cholesky" if operator.properties.certifies("positive_definite") else "lu"
    else:
        kind = policy_.kind
    if kind == "cholesky" and not operator.properties.certifies("self_adjoint"):
        raise ValueError("Sparse Cholesky requires a certified self-adjoint operator.")
    permutation = _permutation(
        storage.shape, input_indices, input_indptr, policy_.ordering
    )
    inverse = np.empty_like(permutation)
    inverse[permutation] = np.arange(permutation.size)
    entries = _permuted_entries(input_indices, input_indptr, permutation)
    rows = (
        _lu_symbolic_rows(storage.shape[0], entries, policy_.fill_level)
        if kind == "lu"
        else _cholesky_symbolic_rows(storage.shape[0], entries, policy_.fill_level)
    )
    factor_indices, factor_indptr, positions = _csr_from_rows(rows)
    input_positions = np.full(factor_indices.size, -1, dtype=np.int64)
    input_conjugate = np.zeros(factor_indices.size, dtype=bool)
    if kind == "lu":
        for coordinate, factor_position in positions.items():
            input_positions[factor_position] = entries.get(coordinate, -1)
    else:
        for (row, column), factor_position in positions.items():
            direct = entries.get((row, column))
            reflected = entries.get((column, row))
            if direct is not None:
                input_positions[factor_position] = direct
            elif reflected is not None:
                input_positions[factor_position] = reflected
                input_conjugate[factor_position] = row != column
    diagonal = np.asarray(
        [positions[(row, row)] for row in range(storage.shape[0])], dtype=np.int64
    )
    (
        multipliers,
        multiplier_valid,
        update_targets,
        update_left,
        update_right,
        update_valid,
    ) = _operation_tables(kind, rows, positions)
    row_positions, row_valid = _padded(
        [
            list(range(factor_indptr[row], factor_indptr[row + 1]))
            for row in range(storage.shape[0])
        ]
    )
    lower_analysis, lower_positions = _triangular_pattern(
        rows,
        positions,
        "lower",
        storage.indices.dtype,
        unit_diagonal=kind == "lu",
    )
    if kind == "lu":
        upper_analysis, upper_positions = _triangular_pattern(
            rows,
            positions,
            "upper",
            storage.indices.dtype,
            unit_diagonal=False,
        )
    else:
        upper_analysis = None
        upper_positions = None
    input_pattern_id = _pattern_identifier(storage.shape, input_indices, input_indptr)
    plan_payload = b"|".join(
        (
            input_pattern_id.encode(),
            kind.encode(),
            policy_.ordering.encode(),
            str(policy_.fill_level).encode(),
            factor_indices.tobytes(),
            factor_indptr.tobytes(),
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
        multiplier_positions=jnp.asarray(multipliers, dtype=index_dtype),
        multiplier_valid=jnp.asarray(multiplier_valid),
        update_targets=jnp.asarray(update_targets, dtype=index_dtype),
        update_left=jnp.asarray(update_left, dtype=index_dtype),
        update_right=jnp.asarray(update_right, dtype=index_dtype),
        update_valid=jnp.asarray(update_valid),
        row_positions=jnp.asarray(row_positions, dtype=index_dtype),
        row_valid=jnp.asarray(row_valid),
        lower_positions=jnp.asarray(lower_positions, dtype=index_dtype),
        upper_positions=(
            None
            if upper_positions is None
            else jnp.asarray(upper_positions, dtype=index_dtype)
        ),
        lower_analysis=lower_analysis,
        upper_analysis=upper_analysis,
        shape=storage.shape,
        kind=kind,
        policy=policy_,
        input_pattern_id=input_pattern_id,
        plan_id=sha256(plan_payload).hexdigest(),
        input_nnz=input_indices.size,
    )


def _prune_row(
    values: Array,
    plan: SparseFactorizationPlan,
    row: Array,
    /,
) -> tuple[Array, Array]:
    positions = plan.row_positions[row]
    valid = plan.row_valid[row]
    safe_positions = jnp.where(valid, positions, 0)
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
        count = min(
            plan.policy.maximum_fill_per_row,
            plan.row_positions.shape[1],
        )
        scores = jnp.where(candidate, jnp.abs(row_values), -jnp.inf)
        _, selected_indices = jax.lax.top_k(scores, count)
        selected = (
            jnp.any(
                jnp.arange(scores.size)[:, None] == selected_indices[None, :],
                axis=1,
            )
            & candidate
        )
    keep = valid & (diagonal | selected)
    row_marker = jnp.zeros(values.shape, dtype=bool).at[safe_positions].max(valid)
    keep_marker = jnp.zeros(values.shape, dtype=bool).at[safe_positions].max(keep)
    pruned = jnp.where(row_marker & ~keep_marker, jnp.zeros((), values.dtype), values)
    dropped = jnp.sum((row_marker & ~keep_marker).astype(jnp.int32))
    return pruned, dropped


def refresh_sparse_factorization(
    plan: SparseFactorizationPlan,
    operator: AbstractSparseLinearOperator,
    /,
) -> PreparedSparseFactorization:
    """Refresh numerical sparse factors under one unchanged symbolic pattern."""
    if not isinstance(plan, SparseFactorizationPlan):
        raise TypeError("plan must be a SparseFactorizationPlan.")
    storage, indices, indptr = _validated_pattern(operator)
    if _pattern_identifier(storage.shape, indices, indptr) != plan.input_pattern_id:
        raise ValueError(
            "Sparse factorization refresh requires an unchanged CSR pattern."
        )
    safe_input = jnp.maximum(plan.input_positions, 0)
    gathered = storage.values[safe_input]
    gathered = jnp.where(plan.input_conjugate, jnp.conj(gathered), gathered)
    values = jnp.where(
        plan.input_positions >= 0,
        gathered,
        jnp.zeros((), dtype=storage.values.dtype),
    )
    values = values.at[plan.diagonal_positions].add(plan.policy.diagonal_shift)
    initial = (
        values,
        jnp.asarray(int(SparseFactorizationStatus.SUCCESS), dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(jnp.inf, dtype=values.real.dtype),
    )

    def factor_step(pivot_index, carry):
        current, status, replaced, dropped, minimum_pivot = carry
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
                * jnp.maximum(jnp.ones((), dtype=real_dtype), jnp.abs(real_pivot))
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
                max(plan.policy.replacement_value, plan.policy.pivot_tolerance),
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
                max(plan.policy.replacement_value, plan.policy.pivot_tolerance),
                dtype=current.dtype,
            )
        bad = ~acceptable
        status = jnp.where(
            (status == int(SparseFactorizationStatus.SUCCESS)) & bad,
            jnp.where(
                finite,
                failure_status,
                int(SparseFactorizationStatus.NONFINITE),
            ),
            status,
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
        pivot_magnitude = jnp.abs(effective_pivot)
        minimum_pivot = jnp.minimum(minimum_pivot, pivot_magnitude)
        if plan.kind == "cholesky":
            factor_pivot = jnp.sqrt(jnp.real(effective_pivot)).astype(current.dtype)
            current = current.at[pivot_position].set(factor_pivot)
            denominator = factor_pivot
        else:
            current = current.at[pivot_position].set(effective_pivot)
            denominator = effective_pivot
        multiplier_positions = plan.multiplier_positions[pivot_index]
        multiplier_valid = plan.multiplier_valid[pivot_index]
        safe_multipliers = jnp.where(multiplier_valid, multiplier_positions, 0)
        previous = current[safe_multipliers]
        divided = previous / denominator
        current = current.at[safe_multipliers].add(
            jnp.where(multiplier_valid, divided - previous, 0.0)
        )
        targets = plan.update_targets[pivot_index]
        left = plan.update_left[pivot_index]
        right = plan.update_right[pivot_index]
        update_valid = plan.update_valid[pivot_index]
        safe_targets = jnp.where(update_valid, targets, 0)
        safe_left = jnp.where(update_valid, left, 0)
        safe_right = jnp.where(update_valid, right, 0)
        right_values = current[safe_right]
        if plan.kind == "cholesky":
            right_values = jnp.conj(right_values)
        updates = current[safe_left] * right_values
        current = current.at[safe_targets].add(jnp.where(update_valid, -updates, 0.0))
        return current, status, replaced, dropped, minimum_pivot

    values, status, replaced, dropped, minimum_pivot = jax.lax.fori_loop(
        0, plan.shape[0], factor_step, initial
    )
    finite = jnp.all(jnp.isfinite(values))
    status = jnp.where(
        (status == int(SparseFactorizationStatus.SUCCESS)) & ~finite,
        int(SparseFactorizationStatus.NONFINITE),
        status,
    ).astype(jnp.int32)
    factor_nonzeros = jnp.count_nonzero(values)
    diagnostics = SparseFactorizationDiagnostics(
        minimum_pivot=minimum_pivot,
        replaced_pivots=replaced,
        dropped_entries=dropped,
        input_nonzeros=jnp.asarray(plan.input_nnz, dtype=jnp.int32),
        factor_nonzeros=factor_nonzeros.astype(jnp.int32),
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
    "SparseOrdering",
    "factorize_sparse",
    "prepare_sparse_factorization",
    "refresh_sparse_factorization",
]
