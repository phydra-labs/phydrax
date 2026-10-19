#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Host dense elimination over fields with exact zero decisions."""

from collections.abc import Sequence
from fractions import Fraction
from typing import Protocol, Self


class ExactField(Protocol):
    """Arithmetic whose Boolean conversion decides mathematical zero exactly.

    Division by a Boolean-nonzero value must produce the field quotient without rounding.
    Elimination uses no ordered or approximate pivot heuristic.
    """

    def __bool__(self) -> bool: ...

    def __neg__(self) -> Self: ...

    def __sub__(self, other: Self, /) -> Self: ...

    def __mul__(self, other: Self | Fraction | int, /) -> Self: ...

    def __rmul__(self, other: Fraction | int, /) -> Self: ...

    def __truediv__(self, other: Self, /) -> Self: ...


def solve_exact[T: ExactField](
    matrix: list[list[T]],
    right: list[list[T]],
    *,
    singular_error: type[ValueError] = ValueError,
    singular_message: str = "Canonical coordinate lattice is not unisolvent.",
) -> list[list[T]]:
    """Solve a field-valued system by Gauss–Jordan elimination; refuse singular input."""
    size = len(matrix)
    rows = [list(a) + list(b) for a, b in zip(matrix, right, strict=True)]
    for column in range(size):
        pivot = next((row for row in range(column, size) if rows[row][column]), None)
        if pivot is None:
            raise singular_error(singular_message)
        rows[column], rows[pivot] = rows[pivot], rows[column]
        divisor = rows[column][column]
        rows[column] = [value / divisor for value in rows[column]]
        for row in range(size):
            if row == column or not rows[row][column]:
                continue
            factor = rows[row][column]
            rows[row] = [
                a - factor * b for a, b in zip(rows[row], rows[column], strict=True)
            ]
    return [row[size:] for row in rows]


def determinant_exact[T: ExactField](matrix: Sequence[Sequence[T]], /) -> T | Fraction:
    """Return the field-valued determinant, with rational zero for singular input."""
    dimension = len(matrix)
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
