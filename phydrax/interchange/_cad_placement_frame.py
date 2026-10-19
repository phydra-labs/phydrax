#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""STEP direction intent and directed affine-pose enclosures."""

from __future__ import annotations

from fractions import Fraction

import numpy as np

from ..discretization._coordinate_budget import CoordinateEnclosureBudget
from ..geometry._rational import (
    fraction_down,
    fraction_up,
    rational_cross,
    rational_dot,
    rational_vector,
    RationalVector,
)
from ..geometry.brep._exact import square_root_interval
from ._cad import refuse


def _product_interval(
    first: tuple[Fraction, Fraction], second: tuple[Fraction, Fraction], /
) -> tuple[Fraction, Fraction]:
    products = tuple(a * b for a in first for b in second)
    return min(products), max(products)


def _normalized(
    vector: RationalVector, label: str, /
) -> tuple[tuple[Fraction, Fraction], ...]:
    """Enclose normalization after rational rescaling prevents overflow."""
    scale = max(abs(value) for value in vector)
    if scale == 0:
        raise refuse(
            "unsupported-entity",
            label,
            "Placement directions must be nonzero; got a degenerate frame.",
        )
    scaled = tuple(value / scale for value in vector)
    lo, hi = square_root_interval(rational_dot(scaled, scaled))
    if lo <= 0.0 or not np.isfinite(hi):
        raise refuse(
            "unsupported-entity",
            label,
            "Placement direction normalization must be certified; got unresolved bounds.",
        )
    denominator = (Fraction(lo), Fraction(hi))
    return tuple(
        (
            min(value / denominator[0], value / denominator[1]),
            max(value / denominator[0], value / denominator[1]),
        )
        for value in scaled
    )


def source_frame(
    first_axis: np.ndarray,
    third_axis: np.ndarray,
    translation: np.ndarray,
    label: str,
    budget: CoordinateEnclosureBudget,
    /,
    *,
    second_axis: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Enclose the proper frame defined by source direction projection."""
    budget.reserve(65536, 65536)
    values = (first_axis, third_axis, translation)
    if any(value.shape != (3,) or not np.all(np.isfinite(value)) for value in values):
        raise refuse(
            "malformed",
            label,
            "Placement directions and origin must be finite three-vectors; got invalid values.",
        )
    first, third = rational_vector(first_axis), rational_vector(third_axis)
    transverse = rational_cross(third, first)
    if second_axis is not None:
        if second_axis.shape != (3,) or not np.all(np.isfinite(second_axis)):
            raise refuse(
                "malformed",
                label,
                "Placement second direction must be a finite three-vector; got invalid values.",
            )
        if rational_dot(rational_vector(second_axis), transverse) <= 0:
            raise refuse(
                "unsupported-entity",
                label,
                "Placement directions must define a proper frame; got reflection or degeneracy.",
            )
    projected = rational_cross(transverse, third)
    x, z = _normalized(projected, label), _normalized(third, label)
    y = []
    for axis in range(3):
        following, preceding = (axis + 1) % 3, (axis + 2) % 3
        positive = _product_interval(z[following], x[preceding])
        negative = _product_interval(z[preceding], x[following])
        y.append((positive[0] - negative[1], positive[1] - negative[0]))
    envelope = np.empty((2, 3, 4), dtype=np.float64)
    matrix = np.empty((3, 4), dtype=np.float64)
    for row in range(3):
        for column, interval in enumerate((x[row], y[row], z[row])):
            envelope[0, row, column] = fraction_down(interval[0])
            envelope[1, row, column] = fraction_up(interval[1])
            matrix[row, column] = float(sum(interval, Fraction(0)) / 2)
        matrix[row, 3] = translation[row]
        envelope[:, row, 3] = translation[row]
    return matrix, envelope


def compose_enclosures(
    outer: np.ndarray, inner: np.ndarray, budget: CoordinateEnclosureBudget, /
) -> np.ndarray:
    """Enclose affine composition without rounding any rational intermediate."""
    budget.reserve(65536, 65536)
    result = np.empty((2, 3, 4), dtype=np.float64)
    for row in range(3):
        for column in range(4):
            low, high = Fraction(0), Fraction(0)
            for axis in range(3):
                term = _product_interval(
                    (
                        Fraction(float(outer[0, row, axis])),
                        Fraction(float(outer[1, row, axis])),
                    ),
                    (
                        Fraction(float(inner[0, axis, column])),
                        Fraction(float(inner[1, axis, column])),
                    ),
                )
                low += term[0]
                high += term[1]
            if column == 3:
                low += Fraction(float(outer[0, row, 3]))
                high += Fraction(float(outer[1, row, 3]))
            result[0, row, column], result[1, row, column] = (
                fraction_down(low),
                fraction_up(high),
            )
    return result


def inverse_enclosure(
    envelope: np.ndarray, budget: CoordinateEnclosureBudget, /
) -> np.ndarray:
    """Enclose inverse source pose using the source rotation transpose."""
    transpose = np.zeros((2, 3, 4), dtype=np.float64)
    transpose[:, :, :3] = envelope[:, :, :3].transpose((0, 2, 1))
    shift = np.zeros((2, 3, 4), dtype=np.float64)
    shift[:, :, :3] = np.eye(3, dtype=np.float64)
    shift[0, :, 3], shift[1, :, 3] = -envelope[1, :, 3], -envelope[0, :, 3]
    return compose_enclosures(transpose, shift, budget)


def pose_deviation_bounds(
    matrix: np.ndarray, envelope: np.ndarray, /
) -> tuple[float, float]:
    """Bound rotation operator and translation errors by enclosed Euclidean norms."""
    entries = pose_entry_error_bounds(matrix, envelope)
    rotation_squared = sum(
        (Fraction(float(value)) ** 2 for value in entries[:, :3].flat),
        Fraction(0),
    )
    translation_squared = sum(
        (Fraction(float(value)) ** 2 for value in entries[:, 3]),
        Fraction(0),
    )
    return square_root_interval(rotation_squared)[1], square_root_interval(
        translation_squared
    )[1]


def pose_entry_error_bounds(matrix: np.ndarray, envelope: np.ndarray, /) -> np.ndarray:
    """Enclose each source affine coefficient's absolute represented error."""
    result = np.empty((3, 4), dtype=np.float64)
    for row in range(3):
        for column in range(4):
            value = Fraction(float(matrix[row, column]))
            error = max(
                abs(value - Fraction(float(envelope[0, row, column]))),
                abs(value - Fraction(float(envelope[1, row, column]))),
            )
            result[row, column] = fraction_up(error)
    return result
