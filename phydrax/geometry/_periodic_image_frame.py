#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact reusable image frames for periodic translations and isometries."""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from typing import assert_never

import numpy as np
from numpy.typing import ArrayLike

import phydrax.discretization._periodic_cell as periodic_cell_owner

from ..discretization._exact_algebraic_geometry import (
    coefficient_endpoint,
    ExactScalar,
    RadicalElement,
    rational_coefficient,
)
from ..discretization._periodic_exact import (
    exact_periodic_element,
    ExactIsometry,
    PeriodicIsometryGroup,
    PeriodicMeshIdentification,
)
from ..linalg._exact import solve_exact
from ..typing import checked
from ._rational import dyadic_integers, fraction_endpoint


type PeriodicImage = int | tuple[int, ...]


def _exact_scalar(value: object, /) -> ExactScalar:
    if isinstance(value, RadicalElement):
        return RadicalElement.from_stored_fields(value)
    if isinstance(value, Fraction):
        return value
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return Fraction(int(value))
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("Periodic image coordinates must be finite.")
        return Fraction(number)
    raise TypeError(
        f"Periodic image coordinates must be exact real scalars; got {type(value).__name__}."
    )


def _integer_array(
    value: ArrayLike,
    name: str,
    shape: tuple[int | None, ...],
    /,
) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != len(shape) or any(
        expected is not None and actual != expected
        for actual, expected in zip(array.shape, shape, strict=True)
    ):
        raise ValueError(f"{name} must have shape {shape}; got {array.shape}.")
    if array.dtype.kind not in "iuO":
        raise ValueError(f"{name} must contain integers; got dtype {array.dtype}.")
    lower, upper = np.iinfo(np.int64).min, np.iinfo(np.int64).max
    result = np.empty(array.shape, dtype=np.int64)
    for index, item in enumerate(array.reshape(-1)):
        if isinstance(item, (bool, np.bool_)) or not isinstance(item, (int, np.integer)):
            raise ValueError(f"{name} must contain integer scalars.")
        integer = int(item)
        if integer < lower or integer > upper:
            raise ValueError(
                f"{name} contains an exponent that cannot be represented by int64."
            )
        result.reshape(-1)[index] = integer
    return result


def _immutable_array(value: np.ndarray, /) -> np.ndarray:
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


def _identity(dimension: int, /) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(
        tuple(Fraction(int(row == column)) for column in range(dimension))
        for row in range(dimension)
    )


def _matrix_point(
    matrix: ExactIsometry,
    point: tuple[ExactScalar, ...],
    /,
) -> tuple[ExactScalar, ...]:
    dimension = len(matrix) - 1
    if len(point) != dimension:
        raise ValueError("Exact periodic action and source point dimensions differ.")
    return tuple(
        sum(
            (matrix[row][column] * point[column] for column in range(dimension)),
            matrix[row][dimension],
        )
        for row in range(dimension)
    )


def _matrix_bank(
    cell: PeriodicMeshIdentification, exponents: np.ndarray, /
) -> np.ndarray:
    return np.asarray(
        tuple(exact_periodic_element(cell, row) for row in exponents),
        dtype=object,
    ).reshape((len(exponents), cell.ambient_dimension + 1, cell.ambient_dimension + 1))


def _vectors_exact(cell: periodic_cell_owner.PeriodicCell, /) -> np.ndarray:
    vectors = np.asarray(cell.vectors, dtype=np.float64)
    return np.asarray(
        [[Fraction(float(value)) for value in row] for row in vectors], dtype=object
    )


def _translation_dual(
    vectors: tuple[tuple[Fraction, ...], ...], /
) -> tuple[tuple[Fraction, ...], ...]:
    if not vectors:
        return ()
    rank, dimension = len(vectors), len(vectors[0])
    if any(len(row) != dimension for row in vectors):
        raise ValueError("Periodic translation vectors must share one ambient dimension.")
    gram = tuple(
        tuple(
            sum(
                (left[axis] * right[axis] for axis in range(dimension)),
                Fraction(0),
            )
            for right in vectors
        )
        for left in vectors
    )
    inverse = solve_exact(
        gram,
        tuple(
            tuple(Fraction(int(row == column)) for column in range(rank))
            for row in range(rank)
        ),
        singular_message="Periodic translation vectors must be linearly independent.",
    )
    return tuple(
        tuple(
            sum(
                (
                    inverse[axis][index] * vectors[index][coordinate]
                    for index in range(rank)
                ),
                Fraction(0),
            )
            for axis in range(rank)
        )
        for coordinate in range(dimension)
    )


def _project_point(
    point: tuple[ExactScalar, ...], dual: tuple[tuple[Fraction, ...], ...], axis: int, /
) -> ExactScalar:
    return sum(
        (point[coordinate] * dual[coordinate][axis] for coordinate in range(len(point))),
        Fraction(0),
    )


def lattice_overlap_images(
    points: np.ndarray,
    vectors: np.ndarray,
    /,
) -> tuple[range, ...]:
    """Return a complete symmetric lattice shift range for source overlap."""
    point_bank = np.asarray(points, dtype=object)
    vector_bank = np.asarray(vectors, dtype=object)
    if (
        point_bank.ndim != 2
        or point_bank.shape[0] == 0
        or vector_bank.ndim != 2
        or vector_bank.shape[0] == 0
        or point_bank.shape[1] != vector_bank.shape[1]
    ):
        raise ValueError(
            "lattice_overlap_images requires nonempty point and vector banks in one ambient dimension."
        )
    dimension, rank = point_bank.shape[1], vector_bank.shape[0]
    exact_vectors = tuple(
        tuple(_exact_scalar(value) for value in row) for row in vector_bank
    )
    dual = _translation_dual(
        tuple(
            tuple(
                rational_coefficient(value, role="periodic lattice overlap vector")
                for value in row
            )
            for row in exact_vectors
        )
    )
    exact_points = tuple(
        tuple(_exact_scalar(value) for value in row) for row in point_bank
    )
    result: list[range] = []
    for axis in range(rank):
        projections = tuple(_project_point(point, dual, axis) for point in exact_points)
        limit = math.floor(max(projections) - min(projections))
        result.append(range(-limit, limit + 1))
    if len(result) != rank or dimension < rank:
        raise ValueError("lattice_overlap_images received an invalid lattice rank.")
    return tuple(result)


def _lattice_ranges_from_actions(
    actions: np.ndarray, rank: int, periodic_axes: tuple[bool, ...], /
) -> tuple[range, ...]:
    if len(actions) == 0:
        raise ValueError("PeriodicImageFrame.image_exponents must be nonempty.")
    if len(periodic_axes) != rank:
        raise ValueError("PeriodicCell periodic axes do not match its lattice rank.")
    lower = [min(int(row[axis]) for row in actions) for axis in range(rank)]
    upper = [max(int(row[axis]) for row in actions) for axis in range(rank)]
    ranges = tuple(range(lo, hi + 1) for lo, hi in zip(lower, upper, strict=True))
    count = math.prod(len(interval) for interval in ranges)
    if (
        count != len(actions)
        or len({tuple(int(value) for value in row) for row in actions}) != count
    ):
        raise ValueError(
            "PeriodicImageFrame.image_exponents must form a rectangular lattice image bank."
        )
    if any(
        not periodic and any(int(row[axis]) != 0 for row in actions)
        for axis, periodic in enumerate(periodic_axes)
    ):
        raise ValueError("Nonperiodic lattice axes must have zero image exponents.")
    return ranges


def _normal_lattice_coordinates(
    coordinates: np.ndarray,
    roots: np.ndarray,
    vertex_exponents: np.ndarray,
    cell: periodic_cell_owner.PeriodicCell,
    image_exponents: np.ndarray | None,
    /,
) -> _PeriodicImageCoordinates:
    vectors = np.asarray(cell.vectors, dtype=np.float64)
    packed, exponent = dyadic_integers(np.concatenate((coordinates, vectors), axis=0))
    actual = packed[: len(coordinates)]
    lattice = packed[len(coordinates) :]
    points = actual[roots] + vertex_exponents.astype(object) @ lattice
    difference = actual - points
    residual_integer = max((abs(int(value)) for value in difference.flat), default=0)
    residual = Fraction(residual_integer) * Fraction(2) ** exponent
    scale = max(1.0, float(np.max(np.abs(coordinates), initial=0.0)))
    residual_bound = 64.0 * np.finfo(np.float64).eps * scale
    if image_exponents is None:
        ranges = lattice_overlap_images(points, lattice)
        ranges = tuple(
            interval if periodic else range(0, 1)
            for interval, periodic in zip(ranges, cell.periodic_axes, strict=True)
        )
    else:
        ranges = _lattice_ranges_from_actions(
            image_exponents, cell.rank, cell.periodic_axes
        )
    image_count = math.prod(len(interval) for interval in ranges)
    return _PeriodicImageCoordinates(
        points,
        points,
        lattice,
        None,
        ranges,
        exponent,
        image_count,
        residual,
        residual_bound,
        None,
    )


def _exact_coordinates(
    coordinates: np.ndarray,
    exact_coordinates: tuple[tuple[ExactScalar, ...], ...] | None,
    /,
) -> np.ndarray:
    if exact_coordinates is None:
        return np.asarray(
            [[Fraction(float(value)) for value in row] for row in coordinates],
            dtype=object,
        )
    values = np.asarray(exact_coordinates, dtype=object)
    if values.shape != coordinates.shape:
        raise ValueError(
            "PeriodicImageFrame.exact_coordinates must match the coordinate bank shape."
        )
    result = np.empty(values.shape, dtype=object)
    for index, value in enumerate(values.reshape(-1)):
        result.reshape(-1)[index] = _exact_scalar(value)
    return result


def _exact_lift_points(
    exact_coordinates: np.ndarray,
    roots: np.ndarray,
    vertex_exponents: np.ndarray,
    cell: PeriodicMeshIdentification,
    /,
) -> np.ndarray:
    matrices = _matrix_bank(cell, vertex_exponents)
    points = np.empty(exact_coordinates.shape, dtype=object)
    for row, (root, matrix) in enumerate(zip(roots, matrices, strict=True)):
        source = tuple(exact_coordinates[int(root)])
        mapped = _matrix_point(
            tuple(tuple(value for value in line) for line in matrix), source
        )
        points[row] = mapped
    return points


def _residual(
    points: np.ndarray,
    coordinates: np.ndarray,
    /,
) -> tuple[Fraction | RadicalElement, float, float]:
    errors = tuple(
        abs(_exact_scalar(value) - Fraction(float(numeric)))
        for value, numeric in zip(points.flat, coordinates.flat, strict=True)
    )
    residual = max(errors, default=Fraction(0))
    scale = max(1.0, float(np.max(np.abs(coordinates), initial=0.0)))
    bound = 64.0 * np.finfo(np.float64).eps * scale
    if residual > Fraction(bound):
        raise ValueError(
            "PeriodicImageFrame.residual exceeds the coordinate construction bound."
        )
    try:
        residual_float = coefficient_endpoint(residual, upper=True)
    except (OverflowError, ValueError) as failure:
        raise ValueError(
            "PeriodicImageFrame.residual is not representable as a finite binary64 value."
        ) from failure
    if not math.isfinite(residual_float):
        raise ValueError(
            "PeriodicImageFrame.residual is not representable as a finite binary64 value."
        )
    return residual, residual_float, bound


def _exact_lattice_coordinates(
    coordinates: np.ndarray,
    roots: np.ndarray,
    vertex_exponents: np.ndarray,
    cell: periodic_cell_owner.PeriodicCell,
    image_exponents: np.ndarray | None,
    exact_coordinates: tuple[tuple[ExactScalar, ...], ...] | None,
    /,
) -> _PeriodicImageCoordinates:
    exact = _exact_coordinates(coordinates, exact_coordinates)
    points = _exact_lift_points(exact, roots, vertex_exponents, cell)
    if exact_coordinates is not None and np.any(points != exact):
        raise ValueError(
            "PeriodicImageFrame.exact_coordinates must obey the declared periodic source lifts."
        )
    residual, _, residual_bound = _residual(points, coordinates)
    lattice = _vectors_exact(cell)
    if image_exponents is None:
        ranges = lattice_overlap_images(points, lattice)
        ranges = tuple(
            interval if periodic else range(0, 1)
            for interval, periodic in zip(ranges, cell.periodic_axes, strict=True)
        )
    else:
        ranges = _lattice_ranges_from_actions(
            image_exponents, cell.rank, cell.periodic_axes
        )
    return _PeriodicImageCoordinates(
        points,
        points,
        lattice,
        None,
        ranges,
        0,
        math.prod(len(interval) for interval in ranges),
        residual,
        residual_bound,
        None,
    )


def _dyadic_integer_bank(values: np.ndarray, /) -> tuple[np.ndarray, int]:
    fractions: list[Fraction] = []
    for value in values.reshape(-1):
        exact = _exact_scalar(value)
        if isinstance(exact, RadicalElement):
            if not exact.terms:
                exact = Fraction(0)
            elif len(exact.terms) == 1 and exact.terms[0][0] == 1:
                exact = exact.terms[0][1]
            else:
                raise ValueError(
                    "Periodic isometry matrices must have exact dyadic rational entries."
                )
        if exact.denominator & (exact.denominator - 1):
            raise ValueError(
                "Periodic isometry matrices must have exact dyadic rational entries."
            )
        fractions.append(exact)
    exponent = min(
        (-(value.denominator.bit_length() - 1) for value in fractions), default=0
    )
    exponent = min(exponent, 0)
    multiplier = 1 << -exponent
    integers = np.asarray(
        [value.numerator * (multiplier // value.denominator) for value in fractions],
        dtype=object,
    ).reshape(values.shape)
    return integers, exponent


def _isometry_actions(
    cell: PeriodicIsometryGroup, image_exponents: np.ndarray | None, /
) -> np.ndarray:
    if image_exponents is None:
        if any(order == 0 for order in cell.orders):
            raise ValueError(
                "PeriodicImageFrame.image_exponents or source_support_boxes must bound infinite isometry actions."
            )
        count = math.prod(cell.orders)
        if count > np.iinfo(np.intp).max:
            raise ValueError(
                "PeriodicImageFrame finite action bank is not representable by a host array."
            )
        return np.asarray(
            tuple(product(*(range(order) for order in cell.orders))), dtype=np.int64
        ).reshape((count, cell.rank))
    actions = _integer_array(
        image_exponents,
        "PeriodicImageFrame.image_exponents",
        (None, cell.rank),
    )
    if len(actions) == 0:
        raise ValueError("PeriodicImageFrame.image_exponents must be nonempty.")
    return actions


def _normal_isometry_coordinates(
    coordinates: np.ndarray,
    roots: np.ndarray,
    vertex_exponents: np.ndarray,
    cell: PeriodicIsometryGroup,
    image_exponents: np.ndarray,
    /,
) -> _PeriodicImageCoordinates:
    matrices = _matrix_bank(cell, image_exponents)
    vertex_matrices = _matrix_bank(cell, vertex_exponents)
    packed, exponent = _dyadic_integer_bank(
        np.concatenate(
            (
                coordinates.reshape(-1).astype(object),
                matrices.reshape(-1),
                vertex_matrices.reshape(-1),
            )
        )
    )
    dimension = coordinates.shape[1]
    coordinate_count = coordinates.size
    matrix_count = len(matrices) * (dimension + 1) ** 2
    actual = packed[:coordinate_count].reshape(coordinates.shape)
    group_matrices = packed[coordinate_count : coordinate_count + matrix_count].reshape(
        (len(matrices), dimension + 1, dimension + 1)
    )
    vertex_integer_matrices = packed[coordinate_count + matrix_count :].reshape(
        vertex_matrices.shape
    )
    base_points = np.empty(coordinates.shape, dtype=object)
    translation_scale = 1 << -exponent
    for row, (root, matrix) in enumerate(
        zip(roots, vertex_integer_matrices, strict=True)
    ):
        for axis in range(dimension):
            base_points[row, axis] = sum(
                (
                    int(actual[int(root), column]) * int(matrix[axis, column])
                    for column in range(dimension)
                ),
                int(matrix[axis, dimension]) * translation_scale,
            )
    points = base_points * translation_scale
    difference = actual * (1 << (-2 * exponent)) - points
    residual_integer = max((abs(int(value)) for value in difference.flat), default=0)
    residual = Fraction(residual_integer) * Fraction(2) ** (3 * exponent)
    scale = max(1.0, float(np.max(np.abs(coordinates), initial=0.0)))
    residual_bound = 64.0 * np.finfo(np.float64).eps * scale
    if residual > Fraction(residual_bound):
        raise ValueError(
            "PeriodicImageFrame.residual exceeds the coordinate construction bound."
        )
    action_rows = np.array(image_exponents, dtype=np.int64, copy=True)
    return _PeriodicImageCoordinates(
        points,
        base_points,
        None,
        group_matrices,
        (),
        exponent,
        len(action_rows),
        residual,
        residual_bound,
        action_rows,
    )


def _exact_isometry_coordinates(
    coordinates: np.ndarray,
    roots: np.ndarray,
    vertex_exponents: np.ndarray,
    cell: PeriodicIsometryGroup,
    image_exponents: np.ndarray,
    exact_coordinates: tuple[tuple[ExactScalar, ...], ...] | None,
    /,
) -> _PeriodicImageCoordinates:
    exact = _exact_coordinates(coordinates, exact_coordinates)
    points = _exact_lift_points(exact, roots, vertex_exponents, cell)
    if exact_coordinates is not None and np.any(points != exact):
        raise ValueError(
            "PeriodicImageFrame.exact_coordinates must obey the declared periodic source lifts."
        )
    residual, _, residual_bound = _residual(points, coordinates)
    group_matrices = _matrix_bank(cell, image_exponents)
    actions = np.array(image_exponents, dtype=np.int64, copy=True)
    return _PeriodicImageCoordinates(
        points,
        points,
        None,
        group_matrices,
        (),
        0,
        len(actions),
        residual,
        residual_bound,
        actions,
    )


def _support_boxes(
    source_boxes: tuple[tuple[np.ndarray, np.ndarray], ...], dimension: int, /
) -> tuple[tuple[tuple[Fraction, ...], tuple[Fraction, ...]], ...]:
    if not source_boxes:
        raise ValueError("PeriodicImageFrame.source_support_boxes must be nonempty.")
    result = []
    for lower, upper in source_boxes:
        low = np.asarray(lower, dtype=np.float64)
        high = np.asarray(upper, dtype=np.float64)
        if (
            low.shape != (dimension,)
            or high.shape != (dimension,)
            or not np.all(np.isfinite(low))
            or not np.all(np.isfinite(high))
            or np.any(low > high)
        ):
            raise ValueError(
                "PeriodicImageFrame.source_support_boxes must have ordered finite ambient bounds."
            )
        result.append(
            (
                tuple(Fraction(float(value)) for value in low),
                tuple(Fraction(float(value)) for value in high),
            )
        )
    return tuple(result)


def _transform_exact_box(
    lower: tuple[Fraction, ...],
    upper: tuple[Fraction, ...],
    matrix: ExactIsometry,
    /,
) -> tuple[tuple[Fraction, ...], tuple[Fraction, ...]]:
    dimension = len(matrix) - 1
    low, high = [], []
    for row in matrix[:-1]:
        minimum = maximum = row[-1]
        for weight, lo, hi in zip(row[:-1], lower, upper, strict=True):
            minimum += weight * (lo if weight >= 0 else hi)
            maximum += weight * (hi if weight >= 0 else lo)
        low.append(minimum)
        high.append(maximum)
    if len(lower) != dimension or len(upper) != dimension:
        raise ValueError(
            "PeriodicImageFrame.source_support_boxes have an invalid dimension."
        )
    return tuple(low), tuple(high)


@checked
def transformed_source_box(
    lower: np.ndarray,
    upper: np.ndarray,
    matrix: ExactIsometry,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Return directed binary64 bounds for an exact affine source-box image."""
    low = np.asarray(lower, dtype=np.float64)
    high = np.asarray(upper, dtype=np.float64)
    if (
        low.ndim != 1
        or high.shape != low.shape
        or not np.all(np.isfinite(low))
        or not np.all(np.isfinite(high))
        or np.any(low > high)
    ):
        raise ValueError("transformed_source_box requires ordered finite bounds.")
    exact = _transform_exact_box(
        tuple(Fraction(float(value)) for value in low),
        tuple(Fraction(float(value)) for value in high),
        matrix,
    )
    lower_float = np.asarray(
        [fraction_endpoint(value, upper=False) for value in exact[0]],
        dtype=np.float64,
    )
    upper_float = np.asarray(
        [fraction_endpoint(value, upper=True) for value in exact[1]],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(lower_float)) or not np.all(np.isfinite(upper_float)):
        raise ValueError(
            "transformed_source_box bounds are not representable in binary64."
        )
    return lower_float, upper_float


def _project_box(
    lower: tuple[Fraction, ...],
    upper: tuple[Fraction, ...],
    dual: tuple[tuple[Fraction, ...], ...],
    axis: int,
    /,
) -> tuple[Fraction, Fraction]:
    minimum = maximum = Fraction(0)
    for coordinate in range(len(lower)):
        coefficient = dual[coordinate][axis]
        minimum += coefficient * (
            lower[coordinate] if coefficient >= 0 else upper[coordinate]
        )
        maximum += coefficient * (
            upper[coordinate] if coefficient >= 0 else lower[coordinate]
        )
    return minimum, maximum


def periodic_overlap_image_exponents(
    cell: PeriodicIsometryGroup,
    source_boxes: tuple[tuple[np.ndarray, np.ndarray], ...],
    /,
) -> np.ndarray:
    """Bound infinite isometry actions whose transformed source boxes can overlap."""
    dimension = cell.ambient_dimension
    boxes = _support_boxes(source_boxes, dimension)
    if any(order <= 0 for order in cell.linear_orders):
        raise ValueError(
            "Periodic overlap enumeration requires finite linear generator orders."
        )
    finite_actions = tuple(product(*(range(order) for order in cell.linear_orders)))
    transformed = tuple(
        _transform_exact_box(
            lower,
            upper,
            exact_periodic_element(cell, np.asarray(action, dtype=np.int64)),
        )
        for action in finite_actions
        for lower, upper in boxes
    )
    infinite_axes = tuple(axis for axis, order in enumerate(cell.orders) if order == 0)
    translations = []
    for axis in infinite_axes:
        exponent = [0] * cell.rank
        exponent[axis] = cell.linear_orders[axis]
        matrix = exact_periodic_element(cell, np.asarray(exponent, dtype=np.int64))
        linear_identity = _identity(dimension)
        if tuple(tuple(row[:dimension]) for row in matrix[:dimension]) != linear_identity:
            raise ValueError(
                "An infinite periodic generator does not expose a pure translation power."
            )
        translations.append(tuple(matrix[row][dimension] for row in range(dimension)))
    dual = _translation_dual(tuple(translations))
    winding_ranges: dict[int, tuple[int, int]] = {}
    for column, axis in enumerate(infinite_axes):
        lower_winding: Fraction | None = None
        upper_winding: Fraction | None = None
        for candidate in transformed:
            candidate_low, candidate_high = _project_box(
                candidate[0], candidate[1], dual, column
            )
            for target in boxes:
                target_low, target_high = _project_box(target[0], target[1], dual, column)
                low = target_low - candidate_high
                high = target_high - candidate_low
                lower_winding = low if lower_winding is None else min(lower_winding, low)
                upper_winding = (
                    high if upper_winding is None else max(upper_winding, high)
                )
        if lower_winding is None or upper_winding is None:
            raise ValueError("Periodic overlap support bounds produced no image pairs.")
        winding_ranges[axis] = (math.ceil(lower_winding), math.floor(upper_winding))
    ranges = []
    for axis, order in enumerate(cell.orders):
        if order > 0:
            ranges.append(range(order))
        else:
            lower_image_winding, upper_image_winding = winding_ranges[axis]
            period = cell.linear_orders[axis]
            ranges.append(
                range(
                    period * lower_image_winding,
                    period * (upper_image_winding + 1),
                )
            )
    image_count = math.prod(len(interval) for interval in ranges)
    if image_count > np.iinfo(np.intp).max:
        raise ValueError(
            "Periodic overlap action bank is not representable by a host array."
        )
    actions = tuple(product(*ranges))
    return _integer_array(
        actions,
        "PeriodicImageFrame.overlap_image_exponents",
        (None, cell.rank),
    )


@dataclass(frozen=True, slots=True)
class _PeriodicImageCoordinates:
    points: np.ndarray
    base_points: np.ndarray
    lattice: np.ndarray | None
    group_matrices: np.ndarray | None
    ranges: tuple[range, ...]
    exponent: int
    image_count: int
    residual: Fraction | RadicalElement
    residual_bound: float
    actions: np.ndarray | None


@dataclass(frozen=True, slots=True)
class PeriodicImageFrame:
    """Exact image coordinates with retained source lifts and action exponents.

    Lattice coordinate banks use integer numerators scaled by ``2**exponent``.
    Isometry image coordinates use the same exponent in their retained dyadic
    matrix bank; :meth:`image_points` returns integer numerators scaled by
    ``2**exponent`` for translations and ``2**(3*exponent)`` for isometries.
    ``source_roots`` and ``source_exponents`` retain each authored lift.
    """

    points: np.ndarray
    base_points: np.ndarray
    lattice: np.ndarray | None
    group_matrices: np.ndarray | None
    ranges: tuple[range, ...]
    exponent: int
    image_count: int
    residual: Fraction | RadicalElement
    residual_float: float
    residual_bound: float
    image_exponents: np.ndarray | None
    source_roots: np.ndarray
    source_exponents: np.ndarray

    def images(self) -> Iterator[PeriodicImage]:
        if self.lattice is not None:
            return product(*self.ranges)
        return iter(range(self.image_count))

    @checked
    def image_points(
        self,
        image: PeriodicImage,
        rows: slice = slice(None),
        /,
    ) -> np.ndarray:
        if self.lattice is not None:
            if not isinstance(image, tuple) or len(image) != len(self.ranges):
                raise TypeError(
                    "PeriodicImageFrame.image must be an integer exponent tuple for a lattice frame."
                )
            exponents = []
            for value, interval in zip(image, self.ranges, strict=True):
                if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                    raise TypeError("Periodic lattice image exponents must be integers.")
                exponent = int(value)
                if exponent not in interval:
                    raise IndexError(
                        "Periodic lattice image exponent is outside this frame."
                    )
                exponents.append(exponent)
            shift = np.asarray(exponents, dtype=object)
            return self.points[rows] + shift @ self.lattice
        if isinstance(image, bool) or not isinstance(image, (int, np.integer)):
            raise TypeError(
                "PeriodicImageFrame.image must be an isometry action-bank index."
            )
        index = int(image)
        if index < 0 or index >= self.image_count:
            raise IndexError("Periodic isometry image index is outside this frame.")
        matrices = self.group_matrices
        if matrices is None:
            raise TypeError("PeriodicImageFrame has no isometry action bank.")
        matrix = matrices[index]
        translation_scale = 1 << (-2 * self.exponent)
        return (
            self.base_points[rows] @ matrix[:-1, :-1].T
            + matrix[:-1, -1] * translation_scale
        )


def _finish_periodic_image_frame(
    prepared: _PeriodicImageCoordinates,
    roots: np.ndarray,
    vertex_exponents: np.ndarray,
    /,
) -> PeriodicImageFrame:
    if prepared.residual > Fraction(prepared.residual_bound):
        raise ValueError(
            "PeriodicImageFrame.residual exceeds the coordinate construction bound."
        )
    residual_float = coefficient_endpoint(prepared.residual, upper=True)
    if not math.isfinite(residual_float):
        raise ValueError("PeriodicImageFrame residual is not finite in binary64.")
    arrays = [prepared.points, prepared.base_points]
    if prepared.lattice is not None:
        arrays.append(prepared.lattice)
    if prepared.group_matrices is not None:
        arrays.append(prepared.group_matrices)
    if prepared.actions is not None:
        arrays.append(prepared.actions)
    arrays = [_immutable_array(value) for value in arrays]
    position = 0
    points, base_points = arrays[position], arrays[position + 1]
    position += 2
    lattice = None
    if prepared.lattice is not None:
        lattice = arrays[position]
        position += 1
    group_matrices = None
    if prepared.group_matrices is not None:
        group_matrices = arrays[position]
        position += 1
    actions = None if prepared.actions is None else arrays[position]
    return PeriodicImageFrame(
        points,
        base_points,
        lattice,
        group_matrices,
        prepared.ranges,
        prepared.exponent,
        prepared.image_count,
        prepared.residual,
        residual_float,
        prepared.residual_bound,
        actions,
        _immutable_array(roots),
        _immutable_array(vertex_exponents),
    )


@checked
def prepare_periodic_image_frame(
    coordinates: np.ndarray,
    roots: np.ndarray,
    vertex_shifts: np.ndarray,
    cell: periodic_cell_owner.PeriodicCell | PeriodicIsometryGroup,
    /,
    *,
    image_exponents: np.ndarray | None = None,
    source_support_boxes: tuple[tuple[np.ndarray, np.ndarray], ...] | None = None,
    exact_coordinates: tuple[tuple[ExactScalar, ...], ...] | None = None,
) -> PeriodicImageFrame:
    """Prepare exact source lifts and a reusable periodic image bank."""
    coordinate_bank = np.asarray(coordinates, dtype=np.float64)
    if (
        coordinate_bank.ndim != 2
        or coordinate_bank.shape[0] == 0
        or coordinate_bank.shape[1] != cell.ambient_dimension
        or not np.all(np.isfinite(coordinate_bank))
    ):
        raise ValueError(
            "PeriodicImageFrame.coordinates must be a nonempty finite ambient point bank."
        )
    root_rows = _integer_array(roots, "PeriodicImageFrame.roots", (len(coordinate_bank),))
    if np.any(root_rows < 0) or np.any(root_rows >= len(coordinate_bank)):
        raise ValueError("PeriodicImageFrame.roots must index the coordinate bank.")
    vertex_rows = _integer_array(
        vertex_shifts,
        "PeriodicImageFrame.vertex_shifts",
        (len(coordinate_bank), cell.rank),
    )
    if (
        isinstance(cell, PeriodicIsometryGroup)
        and any(order == 0 for order in cell.orders)
        and image_exponents is None
        and source_support_boxes is not None
    ):
        image_exponents = periodic_overlap_image_exponents(cell, source_support_boxes)
    match cell:
        case periodic_cell_owner.PeriodicCell():
            if any(
                not periodic and np.any(vertex_rows[:, axis] != 0)
                for axis, periodic in enumerate(cell.periodic_axes)
            ):
                raise ValueError(
                    "PeriodicImageFrame.vertex_shifts must be zero on nonperiodic lattice axes."
                )
            if image_exponents is not None:
                image_exponents = _integer_array(
                    image_exponents,
                    "PeriodicImageFrame.image_exponents",
                    (None, cell.rank),
                )
                if len(image_exponents) == 0:
                    raise ValueError(
                        "PeriodicImageFrame.image_exponents must be nonempty."
                    )
            prepared = (
                _exact_lattice_coordinates(
                    coordinate_bank,
                    root_rows,
                    vertex_rows,
                    cell,
                    image_exponents,
                    exact_coordinates,
                )
                if exact_coordinates is not None
                else _normal_lattice_coordinates(
                    coordinate_bank,
                    root_rows,
                    vertex_rows,
                    cell,
                    image_exponents,
                )
            )
        case PeriodicIsometryGroup():
            actions = _isometry_actions(cell, image_exponents)
            prepared = (
                _exact_isometry_coordinates(
                    coordinate_bank,
                    root_rows,
                    vertex_rows,
                    cell,
                    actions,
                    exact_coordinates,
                )
                if exact_coordinates is not None
                else _normal_isometry_coordinates(
                    coordinate_bank,
                    root_rows,
                    vertex_rows,
                    cell,
                    actions,
                )
            )
        case unreachable:
            assert_never(unreachable)
    return _finish_periodic_image_frame(prepared, root_rows, vertex_rows)
