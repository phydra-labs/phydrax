#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Host elimination over exact fields with zero and rank decisions."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from numbers import Integral
from typing import cast, Protocol, Self

import numpy as np


class ExactField(Protocol):
    """Arithmetic whose truth value decides mathematical zero exactly.

    Division by a nonzero value must return the exact field quotient. Elimination
    deliberately uses neither ordered pivots nor approximate magnitude tests.
    """

    def __bool__(self) -> bool: ...

    def __neg__(self) -> Self: ...

    def __sub__(self, other: Self, /) -> Self: ...

    def __mul__(self, other: Self | Fraction | int, /) -> Self: ...

    def __rmul__(self, other: Fraction | int, /) -> Self: ...

    def __truediv__(self, other: Self, /) -> Self: ...


def _rectangular_width(matrix: Sequence[Sequence[object]], name: str, /) -> int:
    if not matrix:
        return 0
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise ValueError(f"{name} must be rectangular.")
    return width


def _exact_sparse_scalar[T: ExactField](value: T, name: str, /) -> T:
    if isinstance(value, (bool, np.bool_, Integral, np.integer)):
        raise TypeError(
            f"{name} must use exact field scalars, not {type(value).__name__}."
        )
    if isinstance(
        value, (float, complex, Decimal, np.floating, np.complexfloating)
    ):
        raise TypeError(
            f"{name} must use exact field scalars, not {type(value).__name__}."
        )
    return value


def _is_exact_nonzero[T: ExactField](value: T, /) -> bool:
    return bool(value)


def _exact_product[T: ExactField](left: T, right: T, /) -> T:
    return left * right


def _exact_negative_product[T: ExactField](left: T, right: T, /) -> T:
    product = _exact_product(left, right)
    return (product - product) - product


def _exact_subtract_product[T: ExactField](
    minuend: T, left: T, right: T, /
) -> T:
    return minuend - _exact_product(left, right)


def solve_exact[T: ExactField](
    matrix: Sequence[Sequence[T]],
    right: Sequence[Sequence[T]],
    *,
    singular_error: type[ValueError] = ValueError,
    singular_message: str = "Exact linear system is singular.",
) -> list[list[T]]:
    """Solve using one deterministic exact factorization and triangular actions.

    Singular systems are refused instead of selecting a numerical pivot or
    returning a least-squares result.
    """
    size = len(matrix)
    matrix_width = _rectangular_width(matrix, "matrix")
    _rectangular_width(right, "right")
    if matrix_width != size:
        raise ValueError(f"matrix must be square; got shape ({size}, {matrix_width}).")
    if len(right) != size:
        raise ValueError(
            f"right must have {size} rows to match matrix; got {len(right)}."
        )
    return factor_exact(
        matrix,
        singular_error=singular_error,
        singular_message=singular_message,
    ).solve(right)


@dataclass(frozen=True, slots=True)
class ExactLinearSolveEvidence:
    """Deterministic exact rank and pivot record for one row system."""

    row_count: int
    variable_count: int
    right_count: int
    rank: int
    pivot_rows: tuple[int, ...]
    pivot_columns: tuple[int, ...]
    row_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ExactSparseRankFactorization[T: ExactField]:
    """Canonical sparse row-echelon factorization with exact nullspace evidence."""

    evidence: ExactLinearSolveEvidence
    reduced_pivot_rows: tuple[tuple[tuple[int, T], ...], ...]

    def __post_init__(self) -> None:
        if type(self.evidence) is not ExactLinearSolveEvidence:
            raise TypeError("Sparse factor evidence must be ExactLinearSolveEvidence.")
        if self.evidence.right_count != 0:
            raise ValueError("A sparse rank factorization has no right-hand sides.")
        if len(self.reduced_pivot_rows) != self.evidence.rank:
            raise ValueError("Reduced sparse pivot rows must match the exact rank.")
        if len(self.evidence.pivot_columns) != self.evidence.rank:
            raise ValueError("Sparse pivot columns must match the exact rank.")

    def nullspace_basis(self) -> tuple[tuple[T | Fraction, ...], ...]:
        """Return the canonical exact basis ordered by increasing free column."""
        evidence = self.evidence
        pivot_columns = evidence.pivot_columns
        pivot_set = frozenset(pivot_columns)
        free_columns = tuple(
            column
            for column in range(evidence.variable_count)
            if column not in pivot_set
        )
        pivot_rows = tuple(dict(row) for row in self.reduced_pivot_rows)
        basis: list[tuple[T | Fraction, ...]] = []
        for free_column in free_columns:
            vector: list[T | Fraction] = [
                Fraction(0) for _ in range(evidence.variable_count)
            ]
            vector[free_column] = Fraction(1)
            for row_index in range(evidence.rank - 1, -1, -1):
                pivot_column = pivot_columns[row_index]
                products = [
                    coefficient * vector[column]
                    for column, coefficient in pivot_rows[row_index].items()
                    if column != pivot_column
                ]
                if products:
                    value = -products[0]
                    for product in products[1:]:
                        value = value - product
                    vector[pivot_column] = cast(T | Fraction, value)
                else:
                    vector[pivot_column] = Fraction(0)
            basis.append(tuple(vector))
        return tuple(basis)



@dataclass(frozen=True, slots=True)
class ExactLinearFactorization[T: ExactField]:
    """Immutable exact LU factors with deterministic row pivots and rank evidence."""

    factors: tuple[tuple[T, ...], ...]
    row_permutation: tuple[int, ...]
    determinant: T | Fraction
    evidence: ExactLinearSolveEvidence

    def __post_init__(self) -> None:
        size = len(self.factors)
        if any(len(row) != size for row in self.factors):
            raise ValueError("Exact factor data must be square.")
        if tuple(sorted(self.row_permutation)) != tuple(range(size)):
            raise ValueError("Exact factor row permutation must be a permutation.")
        if (
            self.evidence.row_count != size
            or self.evidence.variable_count != size
            or self.evidence.rank != size
            or self.evidence.pivot_columns != tuple(range(size))
        ):
            raise ValueError("Exact factor evidence must prove full square rank.")

    def solve(self, right: Sequence[Sequence[T]], /) -> list[list[T]]:
        """Apply the retained LU factors to one or more exact right-hand sides."""
        size = len(self.factors)
        width = _rectangular_width(right, "right")
        if len(right) != size:
            raise ValueError(
                f"right must have {size} rows to match matrix; got {len(right)}."
            )
        solution = [
            list(right[self.row_permutation[row]])
            for row in range(size)
        ]
        for row in range(size):
            for column in range(width):
                value = solution[row][column]
                for prior in range(row):
                    value -= self.factors[row][prior] * solution[prior][column]
                solution[row][column] = value
        for row in range(size - 1, -1, -1):
            diagonal = self.factors[row][row]
            for column in range(width):
                value = solution[row][column]
                for following in range(row + 1, size):
                    value -= self.factors[row][following] * solution[following][column]
                solution[row][column] = value / diagonal
        return solution


def factor_exact[T: ExactField](
    matrix: Sequence[Sequence[T]],
    /,
    *,
    singular_error: type[ValueError] = ValueError,
    singular_message: str = "Exact linear system is singular.",
) -> ExactLinearFactorization[T]:
    """Prepare one exact pivoted LU factorization for repeated right-hand sides."""
    size = len(matrix)
    width = _rectangular_width(matrix, "matrix")
    if width != size:
        raise ValueError(f"matrix must be square; got shape ({size}, {width}).")
    factors = [list(row) for row in matrix]
    row_permutation = list(range(size))
    pivot_rows = []
    determinant: T | Fraction = Fraction(1)
    for column in range(size):
        pivot = next(
            (row for row in range(column, size) if factors[row][column]), None
        )
        if pivot is None:
            raise singular_error(singular_message)
        if pivot != column:
            factors[column], factors[pivot] = factors[pivot], factors[column]
            row_permutation[column], row_permutation[pivot] = (
                row_permutation[pivot],
                row_permutation[column],
            )
            determinant = -determinant
        diagonal = factors[column][column]
        determinant *= diagonal
        for row in range(column + 1, size):
            multiplier = factors[row][column] / diagonal
            factors[row][column] = multiplier
            for trailing in range(column + 1, size):
                factors[row][trailing] -= multiplier * factors[column][trailing]
        pivot_rows.append(row_permutation[column])
    evidence = ExactLinearSolveEvidence(
        size,
        size,
        0,
        size,
        tuple(pivot_rows),
        tuple(range(size)),
        tuple(range(size)),
    )
    return ExactLinearFactorization(
        tuple(tuple(row) for row in factors),
        tuple(row_permutation),
        determinant,
        evidence,
    )


@dataclass(frozen=True, slots=True)
class ExactRationalAlgebraicFactorization[T: ExactField]:
    """Prepared rational-matrix LU solve action over an exact field extension."""

    rational_factorization: ExactLinearFactorization[Fraction]
    algebraic_factorization: ExactLinearFactorization[T]
    rational_embedding: Callable[[Fraction], T]

    @property
    def evidence(self) -> ExactLinearSolveEvidence:
        return self.rational_factorization.evidence

    def _lift_right(self, value: Fraction | T, /) -> T:
        if isinstance(value, Fraction):
            return self.rational_embedding(value)
        return cast(T, value)

    def solve(
        self, right: Sequence[Sequence[Fraction | T]], /
    ) -> list[list[Fraction | T]]:
        size = len(self.rational_factorization.factors)
        width = _rectangular_width(right, "right")
        if len(right) != size:
            raise ValueError(
                f"right must have {size} rows to match matrix; got {len(right)}."
            )
        rational_columns = tuple(
            column
            for column in range(width)
            if all(isinstance(row[column], Fraction) for row in right)
        )
        algebraic_columns = tuple(
            column
            for column in range(width)
            if column not in rational_columns
        )
        solution: list[list[Fraction | T]] = [
            [Fraction(0) for _ in range(width)] for _ in range(size)
        ]
        if rational_columns:
            rational_right = [
                [
                    cast(Fraction, right[row][column])
                    for column in rational_columns
                ]
                for row in range(size)
            ]
            rational_solution = self.rational_factorization.solve(rational_right)
            for row in range(size):
                for index, column in enumerate(rational_columns):
                    solution[row][column] = rational_solution[row][index]
        if algebraic_columns:
            algebraic_right = [
                [
                    self._lift_right(right[row][column])
                    for column in algebraic_columns
                ]
                for row in range(size)
            ]
            algebraic_solution = self.algebraic_factorization.solve(algebraic_right)
            for row in range(size):
                for index, column in enumerate(algebraic_columns):
                    solution[row][column] = algebraic_solution[row][index]
        return solution


def factor_exact_rational_algebraic[T: ExactField](
    matrix: Sequence[Sequence[Fraction]],
    rational_embedding: Callable[[Fraction], T],
    /,
    *,
    singular_error: type[ValueError] = ValueError,
    singular_message: str = "Exact linear system is singular.",
) -> ExactRationalAlgebraicFactorization[T]:
    """Prepare rational LU factors and their exact field-extension action."""
    rational_factorization = factor_exact(
        matrix,
        singular_error=singular_error,
        singular_message=singular_message,
    )
    algebraic_factorization = ExactLinearFactorization(
        tuple(
            tuple(rational_embedding(value) for value in row)
            for row in rational_factorization.factors
        ),
        rational_factorization.row_permutation,
        rational_embedding(rational_factorization.determinant),
        rational_factorization.evidence,
    )
    return ExactRationalAlgebraicFactorization(
        rational_factorization,
        algebraic_factorization,
        rational_embedding,
    )




class ExactInconsistentSystemError(ValueError):
    """An exact rectangular system has nonzero residual rows after elimination."""

    def __init__(
        self,
        evidence: ExactLinearSolveEvidence,
        transformed_row_indices: tuple[int, ...],
        transformed_row_ids: tuple[int, ...],
        residuals: tuple[tuple[object, ...], ...],
        /,
    ) -> None:
        self.evidence = evidence
        self.transformed_row_indices = transformed_row_indices
        self.transformed_row_ids = transformed_row_ids
        self.residuals = residuals
        super().__init__(
            "Exact rectangular system is inconsistent at transformed rows "
            f"{transformed_row_indices} with row IDs {transformed_row_ids}."
        )


class ExactRankDeficientSystemError(ValueError):
    """An exact rectangular system cannot determine every requested variable."""

    def __init__(self, evidence: ExactLinearSolveEvidence, /) -> None:
        self.evidence = evidence
        super().__init__(
            "Exact rectangular system has rank "
            f"{evidence.rank} for {evidence.variable_count} variables."
        )


def _sparse_column_count(column_count: int, /) -> int:
    if isinstance(column_count, bool) or not isinstance(column_count, Integral):
        raise TypeError("column_count must be an integer.")
    columns = int(column_count)
    if columns < 0:
        raise ValueError("column_count must be nonnegative.")
    return columns


def _sparse_row_identifiers(
    row_count: int, row_ids: Sequence[int] | None, /
) -> tuple[int, ...]:
    identifiers = tuple(range(row_count)) if row_ids is None else tuple(row_ids)
    if len(identifiers) != row_count or any(
        isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0
        for value in identifiers
    ):
        raise ValueError("row_ids must align with matrix rows as nonnegative integers.")
    identifiers_ = tuple(int(value) for value in identifiers)
    if len(set(identifiers_)) != len(identifiers_):
        raise ValueError("row_ids must be unique.")
    return identifiers_


def _exact_sparse_elimination[T: ExactField](
    matrix: Sequence[Mapping[int, T]],
    columns: int,
    identifiers: tuple[int, ...],
    right: Sequence[Sequence[T]] | None,
    /,
    *,
    eliminate_above: bool,
    markowitz: bool,
) -> tuple[
    list[tuple[dict[int, T], list[T]]],
    list[int],
    list[int],
    list[int],
]:
    row_count = len(matrix)
    rows: list[tuple[dict[int, T], list[T]]] = []
    for row_index, coefficients in enumerate(matrix):
        sparse_row: dict[int, T] = {}
        for column_raw, coefficient in coefficients.items():
            if (
                isinstance(column_raw, bool)
                or not isinstance(column_raw, Integral)
                or not 0 <= int(column_raw) < columns
            ):
                raise ValueError(
                    f"matrix row {row_index} names a column outside [0, {columns})."
                )
            column = int(column_raw)
            exact_coefficient = _exact_sparse_scalar(
                coefficient, f"matrix row {row_index} coefficient"
            )
            if exact_coefficient:
                sparse_row[column] = exact_coefficient
        values = () if right is None else right[row_index]
        exact_values = [
            _exact_sparse_scalar(value, f"right row {row_index} value")
            for value in values
        ]
        rows.append((sparse_row, exact_values))
    current_row_ids = list(identifiers)
    pivot_row = 0
    pivot_rows: list[int] = []
    pivot_columns: list[int] = []
    ordered_column = 0
    remaining_columns = set(range(columns)) if markowitz else set()
    while pivot_row < row_count:
        if markowitz:
            column_rows = {
                column: tuple(
                    row_index
                    for row_index in range(pivot_row, row_count)
                    if rows[row_index][0].get(column, Fraction(0))
                )
                for column in sorted(remaining_columns)
            }
            best: tuple[tuple[int, int, int, int, int], int, int] | None = None
            for column, incident_rows in column_rows.items():
                if not incident_rows:
                    continue
                column_degree = len(incident_rows)
                for row_index in incident_rows:
                    row_degree = sum(
                        index in remaining_columns
                        for index in rows[row_index][0]
                    )
                    key = (
                        (row_degree - 1) * (column_degree - 1),
                        row_degree,
                        column_degree,
                        column,
                        current_row_ids[row_index],
                    )
                    candidate = (key, column, row_index)
                    if best is None or candidate < best:
                        best = candidate
            if best is None:
                break
            _, column, selected = best
            remaining_columns.remove(column)
        else:
            selected = None
            while ordered_column < columns:
                column = ordered_column
                ordered_column += 1
                selected = next(
                    (
                        row_index
                        for row_index in range(pivot_row, row_count)
                        if rows[row_index][0].get(column, Fraction(0))
                    ),
                    None,
                )
                if selected is not None:
                    break
            if selected is None:
                break
        rows[pivot_row], rows[selected] = rows[selected], rows[pivot_row]
        current_row_ids[pivot_row], current_row_ids[selected] = (
            current_row_ids[selected],
            current_row_ids[pivot_row],
        )
        coefficients, values = rows[pivot_row]
        divisor = coefficients[column]
        normalized = {
            index: value / divisor for index, value in coefficients.items()
        }
        normalized_right = [value / divisor for value in values]
        for row_index, (other, other_right) in enumerate(rows):
            if row_index == pivot_row or (
                not eliminate_above and row_index < pivot_row
            ):
                continue
            factor = other.get(column)
            if factor is None:
                continue
            reduced: dict[int, T] = {}
            for index in sorted(set(other) | set(normalized)):
                existing = other.get(index)
                pivot_value = normalized.get(index)
                if existing is None:
                    if pivot_value is None:
                        continue
                    value = _exact_negative_product(factor, pivot_value)
                elif pivot_value is None:
                    value = existing
                else:
                    value = _exact_subtract_product(existing, factor, pivot_value)
                if _is_exact_nonzero(value):
                    reduced[index] = value
            rows[row_index] = (
                reduced,
                [
                    left - factor * right_value
                    for left, right_value in zip(
                        other_right, normalized_right, strict=True
                    )
                ],
            )
        rows[pivot_row] = (normalized, normalized_right)
        pivot_rows.append(current_row_ids[pivot_row])
        pivot_columns.append(column)
        pivot_row += 1
        if pivot_row == row_count:
            break
    return rows, current_row_ids, pivot_rows, pivot_columns


def factor_exact_sparse[T: ExactField](
    matrix: Sequence[Mapping[int, T]],
    column_count: int,
    /,
    *,
    row_ids: Sequence[int] | None = None,
) -> ExactSparseRankFactorization[T]:
    """Factor a sparse exact matrix into canonical pivot rows and a nullspace basis.

    Deterministic Markowitz scores, row/column degrees, canonical column
    indices, and stable row IDs select pivots. Forward elimination preserves
    sparse structure; `nullspace_basis` back-substitutes from pivot rows.
    """
    columns = _sparse_column_count(column_count)
    row_count = len(matrix)
    identifiers = _sparse_row_identifiers(row_count, row_ids)
    rows, _, pivot_rows, pivot_columns = _exact_sparse_elimination(
        matrix, columns, identifiers, None, eliminate_above=False, markowitz=True
    )
    evidence = ExactLinearSolveEvidence(
        row_count,
        columns,
        0,
        len(pivot_columns),
        tuple(pivot_rows),
        tuple(pivot_columns),
        identifiers,
    )
    echelon_pivot_rows = tuple(
        tuple(sorted(rows[row_index][0].items()))
        for row_index in range(len(pivot_columns))
    )
    return ExactSparseRankFactorization(evidence, echelon_pivot_rows)


def solve_exact_sparse[T: ExactField](
    matrix: Sequence[Mapping[int, T]],
    right: Sequence[Sequence[T]],
    column_count: int,
    /,
    *,
    row_ids: Sequence[int] | None = None,
) -> tuple[list[list[T]], ExactLinearSolveEvidence]:
    """Solve a consistent, full-column-rank sparse exact system for many RHS.

    Matrix rows are sparse mappings from canonical column indices to exact
    coefficients. One deterministic Gauss–Jordan factorization is shared by all
    right-hand sides. Rank deficiency and inconsistency retain exact pivot and
    residual evidence instead of selecting a numerical or least-squares route.
    """
    columns = _sparse_column_count(column_count)
    row_count = len(matrix)
    if len(right) != row_count:
        raise ValueError(
            f"right must have {row_count} rows to match matrix; got {len(right)}."
        )
    right_count = _rectangular_width(right, "right")
    identifiers = _sparse_row_identifiers(row_count, row_ids)
    rows, current_row_ids, pivot_rows, pivot_columns = _exact_sparse_elimination(
        matrix, columns, identifiers, right, eliminate_above=True, markowitz=False
    )
    evidence = ExactLinearSolveEvidence(
        row_count,
        columns,
        right_count,
        len(pivot_columns),
        tuple(pivot_rows),
        tuple(pivot_columns),
        identifiers,
    )
    inconsistent = tuple(
        row_index
        for row_index, (coefficients, values) in enumerate(rows)
        if not coefficients and any(values)
    )
    if inconsistent:
        raise ExactInconsistentSystemError(
            evidence,
            inconsistent,
            tuple(current_row_ids[index] for index in inconsistent),
            tuple(tuple(rows[index][1]) for index in inconsistent),
        )
    if len(pivot_columns) != columns:
        raise ExactRankDeficientSystemError(evidence)
    solution = cast(
        list[list[T]],
        [[Fraction(0) for _ in range(right_count)] for _ in range(columns)],
    )
    for row_index, column in enumerate(pivot_columns):
        solution[column] = rows[row_index][1]
    return solution, evidence


def determinant_exact[T: ExactField](matrix: Sequence[Sequence[T]], /) -> T | Fraction:
    """Return the exact determinant; a singular matrix returns rational zero."""
    dimension = len(matrix)
    width = _rectangular_width(matrix, "matrix")
    if width != dimension:
        raise ValueError(f"matrix must be square; got shape ({dimension}, {width}).")
    rows = [list(row) for row in matrix]
    determinant: T | Fraction = Fraction(1)
    for column in range(dimension):
        pivot = next((row for row in range(column, dimension) if rows[row][column]), None)
        if pivot is None:
            return Fraction(0)
        if pivot != column:
            rows[column], rows[pivot] = rows[pivot], rows[column]
            determinant = -determinant
        divisor = rows[column][column]
        determinant *= divisor
        for row in range(column + 1, dimension):
            factor = rows[row][column] / divisor
            for trailing in range(column + 1, dimension):
                rows[row][trailing] -= factor * rows[column][trailing]
    return determinant


def determinant_small_algebra[T](
    matrix: Sequence[Sequence[T]],
    /,
    *,
    add: Callable[[T, T], T],
    subtract: Callable[[T, T], T],
    multiply: Callable[[T, T], T],
) -> T:
    """Form a one-, two-, or three-dimensional determinant without division.

    Host symbolic preparation supplies its scalar algebra explicitly. No pivot
    decisions or approximate scalar conversion are needed for polynomial rings.
    """
    dimension = len(matrix)
    if dimension not in (1, 2, 3) or any(len(row) != dimension for row in matrix):
        raise ValueError("Small algebraic determinants require a square matrix of order one through three.")
    if dimension == 1:
        return matrix[0][0]
    if dimension == 2:
        return subtract(
            multiply(matrix[0][0], matrix[1][1]),
            multiply(matrix[0][1], matrix[1][0]),
        )
    first = multiply(
        matrix[0][0],
        subtract(
            multiply(matrix[1][1], matrix[2][2]),
            multiply(matrix[1][2], matrix[2][1]),
        ),
    )
    second = multiply(
        matrix[0][1],
        subtract(
            multiply(matrix[1][0], matrix[2][2]),
            multiply(matrix[1][2], matrix[2][0]),
        ),
    )
    third = multiply(
        matrix[0][2],
        subtract(
            multiply(matrix[1][0], matrix[2][1]),
            multiply(matrix[1][1], matrix[2][0]),
        ),
    )
    return add(subtract(first, second), third)
