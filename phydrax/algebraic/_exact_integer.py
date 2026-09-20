#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

from fractions import Fraction
from typing import Any

import numpy as np


def _integer_matrix(values: Any, /) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 2:
        raise ValueError("Exact integer matrices must be rank two.")
    result = np.empty(array.shape, dtype=object)
    for index in np.ndindex(array.shape):
        value = array[index]
        integer = int(value)
        if value != integer:
            raise ValueError("Exact integer matrices must contain integer values.")
        result[index] = integer
    return result


def exact_integer_rank(values: Any, /) -> int:
    """Return matrix rank using fraction-free exact row elimination."""

    matrix = _integer_matrix(values)
    rows, columns = matrix.shape
    rank = 0
    previous_pivot = 1
    for column in range(columns):
        pivot_row = next(
            (row for row in range(rank, rows) if matrix[row, column] != 0),
            None,
        )
        if pivot_row is None:
            continue
        if pivot_row != rank:
            matrix[[rank, pivot_row]] = matrix[[pivot_row, rank]]
        pivot = matrix[rank, column]
        for row in range(rank + 1, rows):
            for target_column in range(column + 1, columns):
                numerator = (
                    pivot * matrix[row, target_column]
                    - matrix[row, column] * matrix[rank, target_column]
                )
                matrix[row, target_column] = numerator // previous_pivot
            matrix[row, column] = 0
        previous_pivot = pivot
        rank += 1
        if rank == rows:
            break
    return rank


def _swap_rows(matrix: np.ndarray, left: int, right: int) -> None:
    if left != right:
        matrix[[left, right]] = matrix[[right, left]]


def _swap_columns(matrix: np.ndarray, left: int, right: int) -> None:
    if left != right:
        matrix[:, [left, right]] = matrix[:, [right, left]]


def smith_normal_decomposition(
    values: Any,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return D, U, V with U @ values @ V == D over the integers."""

    matrix = _integer_matrix(values)
    rows, columns = matrix.shape
    left = np.eye(rows, dtype=object)
    right = np.eye(columns, dtype=object)
    pivot_index = 0
    while pivot_index < min(rows, columns):
        positions = [
            (row, column)
            for row in range(pivot_index, rows)
            for column in range(pivot_index, columns)
            if matrix[row, column] != 0
        ]
        if not positions:
            break
        row, column = min(positions, key=lambda item: abs(matrix[item]))
        _swap_rows(matrix, pivot_index, row)
        _swap_rows(left, pivot_index, row)
        _swap_columns(matrix, pivot_index, column)
        _swap_columns(right, pivot_index, column)

        while True:
            changed = False
            for row in range(pivot_index + 1, rows):
                if matrix[row, pivot_index] == 0:
                    continue
                quotient = matrix[row, pivot_index] // matrix[pivot_index, pivot_index]
                matrix[row] -= quotient * matrix[pivot_index]
                left[row] -= quotient * left[pivot_index]
                if matrix[row, pivot_index] != 0 and abs(matrix[row, pivot_index]) < abs(
                    matrix[pivot_index, pivot_index]
                ):
                    _swap_rows(matrix, row, pivot_index)
                    _swap_rows(left, row, pivot_index)
                changed = True
                break
            if changed:
                continue
            for column in range(pivot_index + 1, columns):
                if matrix[pivot_index, column] == 0:
                    continue
                quotient = matrix[pivot_index, column] // matrix[pivot_index, pivot_index]
                matrix[:, column] -= quotient * matrix[:, pivot_index]
                right[:, column] -= quotient * right[:, pivot_index]
                if matrix[pivot_index, column] != 0 and abs(
                    matrix[pivot_index, column]
                ) < abs(matrix[pivot_index, pivot_index]):
                    _swap_columns(matrix, column, pivot_index)
                    _swap_columns(right, column, pivot_index)
                changed = True
                break
            if changed:
                continue
            pivot = matrix[pivot_index, pivot_index]
            offender = next(
                (
                    (row, column)
                    for row in range(pivot_index + 1, rows)
                    for column in range(pivot_index + 1, columns)
                    if matrix[row, column] % pivot != 0
                ),
                None,
            )
            if offender is None:
                break
            row, _ = offender
            matrix[pivot_index] += matrix[row]
            left[pivot_index] += left[row]

        pivot = matrix[pivot_index, pivot_index]
        for row in range(pivot_index + 1, rows):
            quotient = matrix[row, pivot_index] // pivot
            matrix[row] -= quotient * matrix[pivot_index]
            left[row] -= quotient * left[pivot_index]
        for column in range(pivot_index + 1, columns):
            quotient = matrix[pivot_index, column] // pivot
            matrix[:, column] -= quotient * matrix[:, pivot_index]
            right[:, column] -= quotient * right[:, pivot_index]
        if matrix[pivot_index, pivot_index] < 0:
            matrix[pivot_index] *= -1
            left[pivot_index] *= -1
        pivot_index += 1

    diagonal = np.asarray(matrix, dtype=object)
    original = _integer_matrix(values)
    if not np.array_equal(left @ original @ right, diagonal):
        raise RuntimeError("Exact Smith decomposition identity failed.")
    return diagonal, left, right


def invert_unimodular(values: Any, /) -> np.ndarray:
    """Invert a square unimodular integer matrix exactly."""

    matrix = _integer_matrix(values)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Unimodular inversion requires a square matrix.")
    size = matrix.shape[0]
    augmented = [
        [Fraction(matrix[row, column]) for column in range(size)]
        + [Fraction(int(row == column)) for column in range(size)]
        for row in range(size)
    ]
    for column in range(size):
        pivot = next(
            (row for row in range(column, size) if augmented[row][column]),
            None,
        )
        if pivot is None:
            raise ValueError("Matrix is singular and not unimodular.")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(
                    augmented[row], augmented[column], strict=True
                )
            ]
    inverse = np.empty((size, size), dtype=object)
    for row in range(size):
        for column in range(size):
            value = augmented[row][size + column]
            if value.denominator != 1:
                raise ValueError(
                    "Matrix inverse is not integral; matrix is not unimodular."
                )
            inverse[row, column] = value.numerator
    return inverse


def smith_rank(diagonal: Any, /) -> int:
    matrix = np.asarray(diagonal, dtype=object)
    return sum(matrix[index, index] != 0 for index in range(min(matrix.shape)))


__all__ = [
    "exact_integer_rank",
    "invert_unimodular",
    "smith_normal_decomposition",
    "smith_rank",
]
