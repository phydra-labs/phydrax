#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Integration-piece records shared by nested geometry and field transfer."""

from __future__ import annotations

from fractions import Fraction
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from ._coordinate_enclosure import Polynomial


type RationalAffine = tuple[tuple[tuple[Fraction, ...], ...], tuple[Fraction, ...]]


class NestedReferencePair(NamedTuple):
    """Map the finer reference cell affinely into its paired coarser cell.

    ``exact_matrix``/``exact_offset`` retain a source-owned rational map whose
    binary64 round-to-nearest views are ``matrix``/``offset``; rational consumers
    read `rational_pair_affine`, never the rounded execution view. Without them the
    binary64 arrays exactly represent the rational map (dyadic subdivision).
    """

    source_cell: int
    target_cell: int
    fine_is_source: bool
    matrix: NDArray[np.float64]
    offset: NDArray[np.float64]
    exact_matrix: tuple[tuple[Fraction, ...], ...] | None = None
    exact_offset: tuple[Fraction, ...] | None = None


def exact_reference_pair(
    source_cell: int,
    target_cell: int,
    fine_is_source: bool,
    matrix: tuple[tuple[Fraction, ...], ...],
    offset: tuple[Fraction, ...],
    /,
) -> NestedReferencePair:
    """Bind one rational affine map to its round-to-nearest execution arrays.

    The rational rows are retained only when some entry is not a binary64 value, so
    dyadic maps keep one deterministic representation.
    """
    rounded_matrix = np.asarray(
        [[float(value) for value in row] for row in matrix], dtype=np.float64
    )
    rounded_offset = np.asarray([float(value) for value in offset], dtype=np.float64)
    if not np.all(np.isfinite(rounded_matrix)) or not np.all(np.isfinite(rounded_offset)):
        raise ValueError(
            "nested_reference.exact_pair must have finite binary64 views; got overflow"
        )
    dyadic = all(
        Fraction(float(value)) == value for row in (*matrix, offset) for value in row
    )
    return NestedReferencePair(
        int(source_cell),
        int(target_cell),
        bool(fine_is_source),
        rounded_matrix,
        rounded_offset,
        None if dyadic else tuple(tuple(row) for row in matrix),
        None if dyadic else tuple(offset),
    )


def rational_pair_affine(pair: NestedReferencePair, /) -> RationalAffine:
    """The exact affine map of one pair, validating its rounded execution view."""
    matrix = np.asarray(pair.matrix, dtype=np.float64)
    offset = np.asarray(pair.offset, dtype=np.float64)
    if (pair.exact_matrix is None) != (pair.exact_offset is None):
        raise ValueError(
            "nested_reference.exact_pair must retain both rational matrix and offset; got one"
        )
    if pair.exact_matrix is None or pair.exact_offset is None:
        return (
            tuple(tuple(Fraction(float(value)) for value in row) for row in matrix),
            tuple(Fraction(float(value)) for value in offset),
        )
    exact_matrix, exact_offset = pair.exact_matrix, pair.exact_offset
    if (
        len(exact_matrix) != matrix.shape[0]
        or any(len(row) != matrix.shape[1] for row in exact_matrix)
        or len(exact_offset) != offset.shape[0]
        or any(
            not isinstance(value, Fraction)
            for row in (*exact_matrix, exact_offset)
            for value in row
        )
        or [[float(value) for value in row] for row in exact_matrix] != matrix.tolist()
        or [float(value) for value in exact_offset] != offset.tolist()
    ):
        raise ValueError(
            "nested_reference.exact_pair must round to its binary64 execution map; got unequal rows"
        )
    return exact_matrix, exact_offset


def rational_pair_arrays(
    pair: NestedReferencePair, /
) -> tuple[NDArray[np.object_], NDArray[np.object_]]:
    """Host Fraction ``(matrix, offset)`` arrays for rational composition helpers."""
    matrix, offset = rational_pair_affine(pair)
    exact_matrix = np.empty(np.shape(pair.matrix), dtype=object)
    for row, values in enumerate(matrix):
        for column, value in enumerate(values):
            exact_matrix[row, column] = value
    exact_offset = np.empty(len(offset), dtype=object)
    for row, value in enumerate(offset):
        exact_offset[row] = value
    return exact_matrix, exact_offset


def rational_affine_bits(affine: RationalAffine, /) -> int:
    """Largest numerator-plus-denominator bit length of one exact affine map."""
    matrix, offset = affine
    return max(
        (
            abs(value.numerator).bit_length() + value.denominator.bit_length()
            for row in (*matrix, offset)
            for value in row
        ),
        default=1,
    )


def reference_pair_record(
    pair: NestedReferencePair, /
) -> tuple[int, int, bool, tuple[tuple[str, ...], ...], tuple[str, ...]]:
    """Canonical fingerprint record binding the rational affine map of one pair."""
    matrix, offset = rational_pair_affine(pair)
    return (
        int(pair.source_cell),
        int(pair.target_cell),
        bool(pair.fine_is_source),
        tuple(tuple(str(value) for value in row) for row in matrix),
        tuple(str(value) for value in offset),
    )


class PolynomialReferencePair(NamedTuple):
    """Map the finer reference cell polynomially with enclosed Jacobian bounds."""

    source_cell: int
    target_cell: int
    fine_is_source: bool
    arguments: tuple[Polynomial, ...]
    source_kind: str
    target_kind: str
    jacobian_bounds: tuple[Fraction, Fraction]
