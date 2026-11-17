#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Exact integer-dyadic affine Jacobians and quality decisions."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
from numpy.typing import NDArray

from .._exact_kernels import float64_interval_product
from .._meshcore import meshcore_available
from ..geometry._rational import fraction_down, fraction_up
from ._coordinate_enclosure import Expression
from ._exact_coordinate_bank import ExactCoordinateBank


def _basis_gradients(
    basis: tuple[Expression, ...], dimension: int
) -> tuple[NDArray[np.object_], int] | None:
    """Return integer gradients sharing one dyadic exponent, or refuse the basis."""
    gradients: list[list[Fraction]] = []
    for expression in basis:
        if not isinstance(expression, dict):
            return None
        row = [Fraction(0) for _ in range(dimension)]
        for powers, coefficient in expression.items():
            if (
                len(powers) != dimension
                or any(not isinstance(power, int) or power < 0 for power in powers)
                or not isinstance(coefficient, Fraction)
                or coefficient.denominator & (coefficient.denominator - 1)
            ):
                return None
            degree = sum(powers)
            if degree > 1 and coefficient:
                return None
            if degree == 1:
                row[powers.index(1)] = coefficient
        gradients.append(row)
    exponent = min(
        (
            -(value.denominator.bit_length() - 1)
            for row in gradients
            for value in row
            if value
        ),
        default=0,
    )
    integers = np.asarray(
        [
            [
                value.numerator << (-(value.denominator.bit_length() - 1) - exponent)
                for value in row
            ]
            for row in gradients
        ],
        dtype=np.object_,
    ).reshape((len(basis), dimension))
    return integers, exponent


def _determinants(
    jacobian: NDArray[np.object_], dimension: int, ambient: int
) -> NDArray[np.object_]:
    """Evaluate signed square determinants or the embedded Gram determinant."""
    if dimension == 1:
        return jacobian[:, 0, 0]
    if dimension == 2:
        minor = (
            jacobian[:, 0, 0] * jacobian[:, 1, 1] - jacobian[:, 0, 1] * jacobian[:, 1, 0]
        )
        if ambient == 2:
            return minor
        second = (
            jacobian[:, 0, 0] * jacobian[:, 2, 1] - jacobian[:, 0, 1] * jacobian[:, 2, 0]
        )
        third = (
            jacobian[:, 1, 0] * jacobian[:, 2, 1] - jacobian[:, 1, 1] * jacobian[:, 2, 0]
        )
        return minor * minor + second * second + third * third
    return (
        jacobian[:, 0, 0]
        * (jacobian[:, 1, 1] * jacobian[:, 2, 2] - jacobian[:, 1, 2] * jacobian[:, 2, 1])
        - jacobian[:, 0, 1]
        * (jacobian[:, 1, 0] * jacobian[:, 2, 2] - jacobian[:, 1, 2] * jacobian[:, 2, 0])
        + jacobian[:, 0, 2]
        * (jacobian[:, 1, 0] * jacobian[:, 2, 1] - jacobian[:, 1, 1] * jacobian[:, 2, 0])
    )


def _quality_screen(
    determinants: NDArray[np.object_],
    squared_scale: NDArray[np.object_],
    floor: Fraction,
    embedded: bool,
) -> tuple[bool | None, ...]:
    """Compare normalized bounds; inconclusive intervals require integer comparison."""
    count = determinants.size
    if not meshcore_available():
        return (None,) * count
    floor_lower, floor_upper = fraction_down(floor), fraction_up(floor)
    if not np.isfinite(floor_upper):
        return (None,) * count
    det_lower = np.empty(count, dtype=np.float64)
    det_upper = np.empty(count, dtype=np.float64)
    scale_lower = np.empty(count, dtype=np.float64)
    scale_upper = np.empty(count, dtype=np.float64)
    for index, (determinant, scale) in enumerate(
        zip(determinants.tolist(), squared_scale.tolist(), strict=True)
    ):
        if not isinstance(determinant, int) or not isinstance(scale, int):
            raise TypeError("Affine dyadic determinants and scales must be integers.")
        magnitude = abs(determinant)
        if embedded:
            exponent = max(magnitude.bit_length(), scale.bit_length())
            scale_exponent = exponent
        else:
            exponent = max(magnitude.bit_length(), (scale.bit_length() + 1) // 2)
            scale_exponent = 2 * exponent
        det_fraction = Fraction(magnitude, 1 << exponent)
        scale_fraction = Fraction(scale, 1 << scale_exponent)
        det_lower[index], det_upper[index] = (
            fraction_down(det_fraction),
            fraction_up(det_fraction),
        )
        scale_lower[index], scale_upper[index] = (
            fraction_down(scale_fraction),
            fraction_up(scale_fraction),
        )
    floor_low = np.full(count, floor_lower, dtype=np.float64)
    floor_high = np.full(count, floor_upper, dtype=np.float64)
    if embedded:
        left_low, left_high = det_lower, det_upper
    else:
        left_low, left_high = float64_interval_product(
            det_lower, det_upper, det_lower, det_upper
        )
        floor_low, floor_high = float64_interval_product(
            floor_low, floor_high, floor_low, floor_high
        )
    right_low, right_high = float64_interval_product(
        scale_lower, scale_upper, floor_low, floor_high
    )
    return tuple(
        True
        if left_low[i] > right_high[i]
        else False
        if left_high[i] < right_low[i]
        else None
        for i in range(count)
    )


def affine_dyadic_quality(
    basis: tuple[Expression, ...],
    coordinates: ExactCoordinateBank,
    routes: NDArray[np.int64],
    dimension: int,
    floor: Fraction,
) -> tuple[tuple[bool, Fraction] | None, ...]:
    """Certify supported rows exactly; leave unsupported rows to the scalar path.

    Each row has its own coordinate scale, so neither values nor resource usage
    of a cell depend on other cells' exponent ranges.
    """
    count = routes.shape[0]
    unsupported: tuple[tuple[bool, Fraction] | None, ...] = (None,) * count
    ambient = coordinates.shape[1]
    if (
        routes.ndim != 2
        or routes.shape[1] != len(basis)
        or (dimension, ambient) not in ((1, 1), (2, 2), (3, 3), (2, 3))
    ):
        return unsupported
    prepared = _basis_gradients(basis, dimension)
    if prepared is None:
        return unsupported
    gradients, basis_exponent = prepared
    valid_routes = np.all((routes >= 0) & (routes < coordinates.shape[0]), axis=1)
    selected = np.flatnonzero(valid_routes)
    if not selected.size or not basis:
        return unsupported
    significands, exponents, dense_entries = coordinates.dyadic_rows(routes[selected])
    dense = np.all(dense_entries, axis=(1, 2))
    selected = selected[dense]
    if not selected.size:
        return unsupported
    exponents = exponents[dense]
    significands = significands[dense]
    # Ignore canonical zero exponents when choosing the row scale.
    minimum = np.min(np.where(significands != 0, exponents, 32767), axis=(1, 2)).astype(
        np.int64
    )
    minimum = np.where(minimum == 32767, 0, minimum)
    shifts = np.where(
        significands != 0, exponents.astype(np.int64) - minimum[:, None, None], 0
    )
    scaled = np.left_shift(significands.astype(np.object_), shifts.astype(np.object_))
    jacobian = np.matmul(scaled.transpose(0, 2, 1), gradients)
    determinants = _determinants(jacobian, dimension, ambient)
    squared_scale = np.ones(selected.size, dtype=np.object_)
    for column in range(dimension):
        squared_scale *= np.sum(jacobian[:, :, column] * jacobian[:, :, column], axis=1)
    result = list(unsupported)
    embedded = ambient != dimension
    screened = _quality_screen(determinants, squared_scale, floor, embedded)
    for position, row in enumerate(selected):
        determinant = determinants[position]
        scale = squared_scale[position]
        quality = screened[position]
        if quality is None:
            if embedded:
                quality = determinant * floor.denominator > floor.numerator * scale
            else:
                quality = (
                    determinant * determinant * floor.denominator**2
                    > floor.numerator**2 * scale
                )
        power = 2 * dimension if embedded else dimension
        exponent = power * (int(minimum[position]) + basis_exponent)
        exact = (
            Fraction(determinant << exponent)
            if exponent >= 0
            else Fraction(determinant, 1 << -exponent)
        )
        result[int(row)] = (bool(determinant > 0 and quality), exact)
    return tuple(result)
