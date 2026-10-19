#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact rational/algebraic periodic image arithmetic and bounded enumeration."""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from typing import assert_never

import equinox as eqx
import numpy as np

import phydrax.discretization._periodic_cell as periodic_cell_owner
from phydrax.geometry._rational import (
    dyadic_integers,
    fraction_endpoint,
    translation_ranges,
)
from phydrax.linalg._small_batched import cofactor_determinant_2, cofactor_determinant_3

from .._strict import StrictModule
from ..discretization._coordinate_budget import CoordinateEnclosureBudget
from ..discretization._exact_algebraic_geometry import (
    RadicalElement,
    reserve_algebraic_work,
)
from ..discretization._periodic_exact import (
    exact_isometry_power,
    exact_periodic_element,
    exact_periodic_generators,
    PeriodicIsometryGroup,
)
from ..typing import checked


def lattice_overlap_images(
    points: np.ndarray,
    vectors: np.ndarray,
    /,
    *,
    budget: CoordinateEnclosureBudget | None = None,
) -> tuple[range, ...]:
    """Bound potentially intersecting images without rounding source coordinates."""
    if budget is not None:
        reserve_algebraic_work(
            tuple(points.flat) + tuple(vectors.flat),
            32 * points.size,
            8 * points.size,
            4,
            budget=budget,
        )

    dimension = vectors.shape[0]
    determinant = (
        cofactor_determinant_2(vectors[0], vectors[1])
        if dimension == 2
        else cofactor_determinant_3(vectors[0], vectors[1], vectors[2])
    )
    fractions = []
    for point in points:
        row = []
        for axis in range(dimension):
            replaced = vectors.copy()
            replaced[axis] = point
            numerator = (
                cofactor_determinant_2(replaced[0], replaced[1])
                if dimension == 2
                else cofactor_determinant_3(replaced[0], replaced[1], replaced[2])
            )
            row.append(
                numerator / determinant
                if isinstance(numerator, (Fraction, RadicalElement))
                else Fraction(int(numerator), int(determinant))
            )
        fractions.append(row)
    ranges = []
    for axis in range(dimension):
        values = [row[axis] for row in fractions]
        diameter = max(values) - min(values)
        limit = math.floor(diameter)
        ranges.append(range(-limit, limit + 1))
    return tuple(ranges)


type PeriodicImage = int | tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PeriodicImageFrame:
    """Retain coefficient-field source lifts and a bounded group image bank."""

    points: np.ndarray
    base_points: np.ndarray
    lattice: np.ndarray | None
    group_matrices: np.ndarray | None
    ranges: tuple[range, ...]
    exponent: int
    image_count: int
    residual: Fraction
    residual_float: float
    residual_bound: float
    image_exponents: np.ndarray | None = None

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
        *,
        budget: CoordinateEnclosureBudget,
    ) -> np.ndarray:
        if self.lattice is not None:
            match image:
                case tuple():
                    reserve_periodic_exact_terms(self.points[rows].size, budget=budget)
                    return self.points[rows] + periodic_exact_product(
                        np.asarray(image, dtype=object), self.lattice, budget=budget
                    )
                case _:
                    raise TypeError(
                        "PeriodicImageFrame.image must be an integer exponent tuple for a lattice; got a different selector type."
                    )
        matrices = self.group_matrices
        match image:
            case int() if matrices is not None:
                matrix = matrices[image]
                reserve_periodic_exact_terms(
                    self.base_points[rows].size + self.base_points.shape[1], budget=budget
                )
                return periodic_exact_product(
                    self.base_points[rows], matrix[:-1, :-1].T, budget=budget
                ) + matrix[:-1, -1] * (1 << (-2 * self.exponent))
            case _:
                raise TypeError(
                    "PeriodicImageFrame.image must be a group-bank index for an isometry frame; got an incompatible selector or frame."
                )


@checked
def reserve_periodic_exact_terms(
    work: int, storage_upper: int = 0, /, *, budget: CoordinateEnclosureBudget
) -> None:
    """Charge scalar and term visits to the request coefficient ledger."""
    budget.admit_work_bound(work)
    budget.reserve(work, storage_upper)


@checked
def periodic_exact_product(
    left: np.ndarray, right: np.ndarray, /, *, budget: CoordinateEnclosureBudget
) -> np.ndarray:
    """Multiply field-valued banks after admitting their dense dot-product terms."""
    rows = left.shape[0] if left.ndim == 2 else 1
    columns = right.shape[1] if right.ndim == 2 else 1
    work = rows * left.shape[-1] * columns
    if any(isinstance(value, RadicalElement) for value in (*left.flat, *right.flat)):
        reserve_algebraic_work(
            tuple(left.flat) + tuple(right.flat),
            2 * work,
            rows * columns,
            2,
            budget=budget,
        )
    else:
        reserve_periodic_exact_terms(work, budget=budget)
    return left @ right


@checked
def exact_periodic_group_element(
    cell: PeriodicIsometryGroup,
    exponents: np.ndarray,
    /,
    *,
    prepared_generators: tuple[
        tuple[tuple[tuple[Fraction, ...], ...], ...], tuple[int, ...]
    ]
    | None = None,
    budget: CoordinateEnclosureBudget,
) -> np.ndarray:
    """Compose authored dyadic isometries over Q, without matrix roundoff."""

    matrices, orders = (
        exact_periodic_generators(cell, budget=budget)
        if prepared_generators is None
        else prepared_generators
    )
    return np.asarray(
        exact_periodic_element(
            matrices, orders, tuple(int(value) for value in exponents), budget=budget
        ),
        dtype=object,
    )


@checked
def exact_periodic_dyadic_bank(
    values: np.ndarray, /, *, budget: CoordinateEnclosureBudget
) -> tuple[np.ndarray, int]:
    """Pack composed rational dyadics without rounding into an immutable integer frame."""
    reserve_periodic_exact_terms(3 * values.size, budget=budget)
    fractions = [
        value if isinstance(value, Fraction) else Fraction(float(value))
        for value in values.reshape(-1)
    ]
    exponent = min(
        (-value.denominator.bit_length() + 1 for value in fractions), default=0
    )
    integers = np.asarray(
        [
            value.numerator * ((1 << -exponent) // value.denominator)
            for value in fractions
        ],
        dtype=object,
    ).reshape(values.shape)
    return integers, exponent


@checked
def transformed_source_box(
    lower: np.ndarray,
    upper: np.ndarray,
    matrix: tuple[tuple[Fraction, ...], ...],
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[np.ndarray, np.ndarray]:
    """Enclose the affine image of a whole-source box with directed endpoints.

    The caller admits the returned array owners. Signed row extrema use the
    full input enclosure, independently of the source map's degree.
    """
    dimension = len(matrix) - 1
    if len(lower) != dimension or len(upper) != dimension:
        raise ValueError(
            "PeriodicImageFrame.source_boxes must have one bound per ambient axis; got incompatible ranks."
        )
    if any(Fraction(lo) > Fraction(hi) for lo, hi in zip(lower, upper, strict=True)):
        raise ValueError(
            "PeriodicImageFrame.source_boxes must have ordered bounds; got a reversed extent."
        )
    lows, highs = [], []
    for row in matrix[:-1]:
        reserve_periodic_exact_terms(2 * dimension + 2, budget=budget)
        low = row[-1] + sum(
            (
                weight * Fraction(lo if weight >= 0 else hi)
                for weight, lo, hi in zip(row[:-1], lower, upper, strict=True)
            ),
            Fraction(0),
        )
        high = row[-1] + sum(
            (
                weight * Fraction(hi if weight >= 0 else lo)
                for weight, lo, hi in zip(row[:-1], lower, upper, strict=True)
            ),
            Fraction(0),
        )
        lows.append(fraction_endpoint(low, upper=False, saturate=True))
        highs.append(fraction_endpoint(high, upper=True, saturate=True))
    if not np.all(np.isfinite(lows)) or not np.all(np.isfinite(highs)):
        raise ValueError(
            "PeriodicImageFrame.source_boxes must fit finite enclosure range; got nonfinite transformed bounds."
        )
    return np.asarray(lows, dtype=np.float64), np.asarray(highs, dtype=np.float64)


@checked
def periodic_overlap_image_exponents(
    cell: PeriodicIsometryGroup,
    source_boxes: tuple[tuple[np.ndarray, np.ndarray], ...],
    maximum_images: int,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> np.ndarray:
    """Enumerate SCI group actions from whole-source support bounds."""

    matrices, orders = exact_periodic_generators(cell, budget=budget)
    finite_count = int(np.prod(cell.linear_orders, dtype=object))
    if finite_count > maximum_images:
        raise PeriodicEmbeddingResourceError(
            f"PeriodicImageFrame.finite_images must fit maximum_images; got {finite_count} images against {maximum_images}."
        )
    if not source_boxes:
        raise ValueError(
            "PeriodicImageFrame.source_boxes must contain nonempty source support; got an empty bank."
        )
    # Conservative CPython tuple/list and array bounds keep the image banks live
    # until the caller's source scope
    # releases the resulting bank.
    reserve_periodic_exact_terms(
        finite_count * len(orders),
        128
        + finite_count * (40 + 48 * len(orders))
        + finite_count * len(source_boxes) * (512 + 96 * cell.ambient_dimension),
        budget=budget,
    )
    finite_actions = tuple(product(*(range(order) for order in cell.linear_orders)))
    transformed_boxes = []
    for action in finite_actions:
        matrix = exact_periodic_element(matrices, orders, action, budget=budget)
        for lower, upper in source_boxes:
            transformed_boxes.append(
                transformed_source_box(lower, upper, matrix, budget=budget)
            )
    translations = []
    for matrix, period in zip(matrices, cell.translation_periods, strict=True):
        if period:
            translation = exact_isometry_power(matrix, period, budget=budget)
            translations.append(tuple(row[-1] for row in translation[:-1]))
    ranges = translation_ranges(transformed_boxes, tuple(translations))
    image_count = finite_count * int(
        np.prod([len(interval) for interval in ranges], dtype=object)
    )
    if image_count > maximum_images:
        raise PeriodicEmbeddingResourceError(
            f"PeriodicImageFrame.overlap_images must fit maximum_images; got {image_count} images against {maximum_images}."
        )
    reserve_periodic_exact_terms(
        image_count * len(orders),
        128 + image_count * (64 + 56 * len(orders)),
        budget=budget,
    )
    actions = []
    for finite in finite_actions:
        for translated in product(*ranges):
            iterator = iter(translated)
            actions.append(
                tuple(
                    value if order else value + period * next(iterator)
                    for value, order, period in zip(
                        finite, orders, cell.translation_periods, strict=True
                    )
                )
            )
    result = np.asarray(actions, dtype=np.int64).reshape((image_count, len(orders)))
    result.setflags(write=False)
    return result


class _PeriodicImageCoordinates(StrictModule):
    points: np.ndarray
    base_points: np.ndarray
    lattice: np.ndarray | None
    group_matrices: np.ndarray | None
    ranges: tuple[range, ...] = eqx.field(static=True)
    exponent: int = eqx.field(static=True)
    image_count: int = eqx.field(static=True)
    residual: Fraction
    residual_bound: float = eqx.field(static=True)
    actions: np.ndarray | None


def _lattice_image_coordinates(
    coordinates: np.ndarray,
    roots: np.ndarray,
    vertex_shifts: np.ndarray,
    cell: periodic_cell_owner.PeriodicCell,
    maximum_images: int,
    image_exponents: np.ndarray | None,
    *,
    budget: CoordinateEnclosureBudget,
) -> _PeriodicImageCoordinates:
    reserve_periodic_exact_terms(
        3 * (coordinates.size + cell.vectors.size), budget=budget
    )
    values, exponent = dyadic_integers(
        np.concatenate((coordinates, np.asarray(cell.vectors)))
    )
    actual, lattice = values[: coordinates.shape[0]], values[coordinates.shape[0] :]
    reserve_periodic_exact_terms(actual.size, budget=budget)
    points = actual[roots] + periodic_exact_product(
        vertex_shifts.astype(object), lattice, budget=budget
    )
    base_points = points
    reserve_periodic_exact_terms(3 * actual.size + 2, budget=budget)
    residual_integer = max(np.abs(actual - points).reshape(-1), default=0)
    residual = Fraction(int(residual_integer)) * Fraction(2) ** exponent
    scale = max(
        1.0,
        float(np.max(np.abs(coordinates))),
        float(np.max(np.abs(vertex_shifts @ np.asarray(cell.vectors)), initial=0.0)),
    )
    residual_bound = 64.0 * np.finfo(np.float64).eps * scale
    if image_exponents is None:
        ranges = lattice_overlap_images(points, lattice)
        ranges = tuple(
            value if periodic else range(1)
            for value, periodic in zip(ranges, cell.periodic_axes, strict=True)
        )
        image_count = int(np.prod([len(value) for value in ranges], dtype=object))
    else:
        actions = np.asarray(image_exponents)
        if (
            actions.ndim != 2
            or actions.shape[1] != cell.rank
            or not np.issubdtype(actions.dtype, np.integer)
            or len(actions) == 0
        ):
            raise ValueError(
                "PeriodicImageFrame.image_exponents must contain nonempty integer lattice action rows; got an incompatible action bank."
            )
        reserve_periodic_exact_terms(3 * actions.size, budget=budget)
        lower = [int(value) for value in np.min(actions, axis=0)]
        upper = [int(value) for value in np.max(actions, axis=0)]
        image_count = int(
            np.prod(
                [high - low + 1 for low, high in zip(lower, upper, strict=True)],
                dtype=object,
            )
        )
        if image_count > maximum_images:
            raise PeriodicEmbeddingResourceError(
                f"PeriodicImageFrame.image_count must fit maximum_images; got {image_count} images against {maximum_images}."
            )
        if image_count != len(actions) or len(np.unique(actions, axis=0)) != image_count:
            raise ValueError(
                "PeriodicImageFrame.image_exponents must form a rectangular lattice image bank; got missing or repeated actions."
            )
        ranges = tuple(
            range(low, high + 1) for low, high in zip(lower, upper, strict=True)
        )
    return _PeriodicImageCoordinates(
        points,
        base_points,
        lattice,
        None,
        ranges,
        exponent,
        image_count,
        residual,
        residual_bound,
        None,
    )


def _isometry_image_coordinates(
    coordinates: np.ndarray,
    roots: np.ndarray,
    vertex_shifts: np.ndarray,
    cell: PeriodicIsometryGroup,
    maximum_images: int,
    image_exponents: np.ndarray | None,
    *,
    budget: CoordinateEnclosureBudget,
) -> _PeriodicImageCoordinates:
    dimension = coordinates.shape[1]
    if image_exponents is None:
        if any(order == 0 for order in cell.orders):
            raise ValueError(
                "PeriodicImageFrame.image_exponents must bound mixed rotational/translational images; got no fundamental-domain action bank."
            )
        image_count = int(np.prod(cell.orders, dtype=object))
        if image_count > maximum_images:
            raise PeriodicEmbeddingResourceError(
                f"PeriodicImageFrame.image_count must fit maximum_images; got {image_count} images against {maximum_images}."
            )
        actions = np.asarray(
            tuple(product(*(range(order) for order in cell.orders))),
            dtype=np.int64,
        ).reshape((image_count, cell.rank))
    else:
        actions = np.asarray(image_exponents)
        if (
            actions.ndim != 2
            or actions.shape[1] != cell.rank
            or not np.issubdtype(actions.dtype, np.integer)
        ):
            raise ValueError(
                "PeriodicImageFrame.image_exponents must have one integer exponent per generator; got an incompatible action bank."
            )
        image_count = len(actions)
        if image_count > maximum_images:
            raise PeriodicEmbeddingResourceError(
                f"PeriodicImageFrame.image_count must fit maximum_images; got {image_count} images against {maximum_images}."
            )
        if image_count == 0:
            raise ValueError(
                "PeriodicImageFrame.image_exponents must contain at least one action; got an empty bank."
            )

    prepared_generators = exact_periodic_generators(cell, budget=budget)
    matrices = [
        exact_periodic_group_element(
            cell, action, prepared_generators=prepared_generators, budget=budget
        )
        for action in actions
    ]
    vertex_matrices = np.asarray(
        [
            exact_periodic_group_element(
                cell, shift, prepared_generators=prepared_generators, budget=budget
            )
            for shift in vertex_shifts
        ],
        dtype=object,
    )
    actions = np.asarray(actions, dtype=np.int64)
    actions.setflags(write=False)
    packed, exponent = exact_periodic_dyadic_bank(
        np.concatenate(
            (
                coordinates.reshape(-1),
                np.asarray(matrices).reshape(-1),
                vertex_matrices.reshape(-1),
            )
        ),
        budget=budget,
    )
    coordinate_count = coordinates.size
    matrix_count = image_count * (dimension + 1) ** 2
    actual = packed[:coordinate_count].reshape(coordinates.shape)
    group_matrices = packed[coordinate_count : coordinate_count + matrix_count].reshape(
        (image_count, dimension + 1, dimension + 1)
    )
    vertex_matrices = packed[coordinate_count + matrix_count :].reshape(
        vertex_matrices.shape
    )
    base_points = np.empty_like(actual)
    for vertex, matrix in enumerate(vertex_matrices):
        reserve_periodic_exact_terms(2 * dimension, budget=budget)
        base_points[vertex] = periodic_exact_product(
            actual[roots[vertex]], matrix[:-1, :-1].T, budget=budget
        ) + matrix[:-1, -1] * (1 << -exponent)
    reserve_periodic_exact_terms(5 * actual.size, budget=budget)
    points = base_points * (1 << -exponent)
    difference = actual * (1 << (-2 * exponent)) - points
    residual_integer = max(np.abs(difference).reshape(-1), default=0)
    residual = Fraction(int(residual_integer)) * Fraction(2) ** (3 * exponent)
    residual_bound = cell.tolerance
    squared = max(
        (sum(int(value) ** 2 for value in row) for row in difference), default=0
    )
    if Fraction(squared) * Fraction(2) ** (6 * exponent) > Fraction(cell.tolerance) ** 2:
        raise ValueError(
            "PeriodicImageFrame.residual must fit the declared Euclidean construction tolerance; got an excessive isometry lift residual."
        )
    return _PeriodicImageCoordinates(
        points,
        base_points,
        None,
        group_matrices,
        (),
        exponent,
        image_count,
        residual,
        residual_bound,
        actions,
    )


def _exact_image_coordinates(
    coordinates: np.ndarray,
    exact_coordinates: tuple[tuple[Fraction | RadicalElement, ...], ...],
    roots: np.ndarray,
    vertex_shifts: np.ndarray,
    cell: periodic_cell_owner.PeriodicCell | PeriodicIsometryGroup,
    maximum_images: int,
    image_exponents: np.ndarray | None,
    *,
    budget: CoordinateEnclosureBudget,
) -> _PeriodicImageCoordinates:
    """Apply rational SCI actions to authenticated, unrounded source vertices."""
    actual = np.asarray(exact_coordinates, dtype=object)
    if actual.shape != coordinates.shape:
        raise ValueError(
            "PeriodicImageFrame.exact_coordinates must match the coordinate bank shape; got incompatible banks"
        )
    reserve_algebraic_work(
        tuple(actual.flat), 4 * actual.size, 4 * actual.size, 2, budget=budget
    )
    matrices, orders = exact_periodic_generators(cell, budget=budget)
    points = np.empty_like(actual)
    for vertex, root in enumerate(roots):
        matrix = np.asarray(
            exact_periodic_element(
                matrices,
                orders,
                tuple(int(v) for v in vertex_shifts[vertex]),
                budget=budget,
            ),
            dtype=object,
        )
        points[vertex] = (
            periodic_exact_product(matrix[:-1, :-1], actual[int(root)], budget=budget)
            + matrix[:-1, -1]
        )
    if np.any(points != actual):
        raise ValueError(
            "PeriodicImageFrame.exact_coordinates must obey the declared periodic lifts; got inconsistent source images"
        )
    errors = tuple(
        abs(Fraction(float(representative)) - value)
        for value, representative in zip(actual.flat, coordinates.flat, strict=True)
    )
    if isinstance(cell, PeriodicIsometryGroup):
        reserve_algebraic_work(errors, 4 * actual.size, 2 * actual.size, 2, budget=budget)
        squared_error = max(
            (
                sum(
                    (
                        value * value
                        for value in errors[start : start + coordinates.shape[1]]
                    ),
                    Fraction(0),
                )
                for start in range(0, len(errors), coordinates.shape[1])
            ),
            default=Fraction(0),
        )
        if squared_error > Fraction(cell.tolerance) ** 2:
            raise ValueError(
                "PeriodicImageFrame.residual must fit the declared Euclidean construction tolerance; got an excessive residual"
            )
    residual = max(
        (
            value.interval(budget=budget)[1]
            if isinstance(value, RadicalElement)
            else value
            for value in errors
        ),
        default=Fraction(0),
    )
    residual_bound = (
        64.0
        * np.finfo(np.float64).eps
        * max(
            1.0,
            float(np.max(np.abs(coordinates))),
            float(np.max(np.abs(vertex_shifts @ np.asarray(cell.vectors)), initial=0.0)),
        )
        if isinstance(cell, periodic_cell_owner.PeriodicCell)
        else cell.tolerance
    )
    match cell:
        case periodic_cell_owner.PeriodicCell():
            lattice = np.asarray(
                [[Fraction(float(v)) for v in row] for row in np.asarray(cell.vectors)],
                dtype=object,
            )
            if image_exponents is None:
                ranges = lattice_overlap_images(points, lattice, budget=budget)
                ranges = tuple(
                    value if periodic else range(1)
                    for value, periodic in zip(ranges, cell.periodic_axes, strict=True)
                )
            else:
                actions = np.asarray(image_exponents)
                if (
                    actions.ndim != 2
                    or actions.shape[1] != cell.rank
                    or not np.issubdtype(actions.dtype, np.integer)
                    or len(actions) == 0
                ):
                    raise ValueError(
                        "PeriodicImageFrame.image_exponents must contain nonempty integer lattice actions; got an incompatible bank"
                    )
                ranges = tuple(
                    range(int(lo), int(hi) + 1)
                    for lo, hi in zip(
                        np.min(actions, axis=0), np.max(actions, axis=0), strict=True
                    )
                )
                count = math.prod(len(value) for value in ranges)
                if count > maximum_images:
                    raise PeriodicEmbeddingResourceError(
                        f"PeriodicImageFrame.image_count must fit maximum_images; got {count} images against {maximum_images}."
                    )
                if count != len(actions) or len(np.unique(actions, axis=0)) != count:
                    raise ValueError(
                        "PeriodicImageFrame.image_exponents must form a rectangular lattice image bank; got missing or repeated actions"
                    )
            return _PeriodicImageCoordinates(
                points,
                points,
                lattice,
                None,
                ranges,
                0,
                math.prod(len(value) for value in ranges),
                residual,
                residual_bound,
                None,
            )
        case PeriodicIsometryGroup():
            if image_exponents is None:
                if any(order == 0 for order in cell.orders):
                    raise ValueError(
                        "PeriodicImageFrame.image_exponents must bound mixed group images; got no declared images"
                    )
                count = math.prod(cell.orders)
                if count > maximum_images:
                    raise PeriodicEmbeddingResourceError(
                        f"PeriodicImageFrame.image_count must fit maximum_images; got {count} images against {maximum_images}."
                    )
                actions = np.asarray(
                    tuple(product(*(range(order) for order in cell.orders))),
                    dtype=np.int64,
                ).reshape((count, cell.rank))
            else:
                actions = np.asarray(image_exponents)
                if (
                    actions.ndim != 2
                    or actions.shape[1] != cell.rank
                    or not np.issubdtype(actions.dtype, np.integer)
                    or len(actions) == 0
                ):
                    raise ValueError(
                        "PeriodicImageFrame.image_exponents must contain nonempty integer group actions; got an incompatible bank"
                    )
            if len(actions) > maximum_images:
                raise PeriodicEmbeddingResourceError(
                    f"PeriodicImageFrame.image_count must fit maximum_images; got {len(actions)} images against {maximum_images}."
                )
            group_matrices = np.asarray(
                [
                    exact_periodic_element(
                        matrices, orders, tuple(int(v) for v in action), budget=budget
                    )
                    for action in actions
                ],
                dtype=object,
            )
            actions = np.asarray(actions, dtype=np.int64)
            actions.setflags(write=False)
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
        case unreachable:
            assert_never(unreachable)


@checked
def prepare_periodic_image_frame(
    coordinates: np.ndarray,
    roots: np.ndarray,
    vertex_shifts: np.ndarray,
    cell: periodic_cell_owner.PeriodicCell | PeriodicIsometryGroup,
    maximum_images: int,
    /,
    *,
    image_exponents: np.ndarray | None = None,
    source_support_boxes: tuple[tuple[np.ndarray, np.ndarray], ...] | None = None,
    exact_coordinates: tuple[tuple[Fraction | RadicalElement, ...], ...] | None = None,
    budget: CoordinateEnclosureBudget,
) -> PeriodicImageFrame:
    """Prepare coefficient-field group arithmetic for embedding and source support."""
    if (
        isinstance(cell, PeriodicIsometryGroup)
        and any(order == 0 for order in cell.orders)
        and image_exponents is None
        and source_support_boxes is not None
    ):
        image_exponents = periodic_overlap_image_exponents(
            cell,
            source_support_boxes,
            maximum_images,
            budget=budget,
        )
    if exact_coordinates is not None:
        prepared = _exact_image_coordinates(
            coordinates,
            exact_coordinates,
            roots,
            vertex_shifts,
            cell,
            maximum_images,
            image_exponents,
            budget=budget,
        )
        return _finish_periodic_image_frame(prepared, maximum_images)
    match cell:
        case periodic_cell_owner.PeriodicCell():
            prepared = _lattice_image_coordinates(
                coordinates,
                roots,
                vertex_shifts,
                cell,
                maximum_images,
                image_exponents,
                budget=budget,
            )
        case PeriodicIsometryGroup():
            prepared = _isometry_image_coordinates(
                coordinates,
                roots,
                vertex_shifts,
                cell,
                maximum_images,
                image_exponents,
                budget=budget,
            )
        case unreachable:
            assert_never(unreachable)
    return _finish_periodic_image_frame(prepared, maximum_images)


def _finish_periodic_image_frame(
    prepared: _PeriodicImageCoordinates, maximum_images: int, /
) -> PeriodicImageFrame:
    points, base_points, lattice, group_matrices = (
        prepared.points,
        prepared.base_points,
        prepared.lattice,
        prepared.group_matrices,
    )
    ranges, exponent, image_count, residual, residual_bound = (
        prepared.ranges,
        prepared.exponent,
        prepared.image_count,
        prepared.residual,
        prepared.residual_bound,
    )
    if image_count > maximum_images:
        raise PeriodicEmbeddingResourceError(
            f"PeriodicImageFrame.image_count must fit maximum_images; got {image_count} images against {maximum_images}."
        )
    residual_float = float(residual)
    if Fraction(residual_float) < residual:
        residual_float = float(np.nextafter(residual_float, np.inf))
    if residual > Fraction(float(residual_bound)):
        raise ValueError(
            "PeriodicImageFrame.residual must fit the declared coordinate construction bound; got an excessive group lift residual."
        )
    for values in (points, base_points, lattice, group_matrices):
        if values is not None:
            values.setflags(write=False)
    return PeriodicImageFrame(
        points,
        base_points,
        lattice,
        group_matrices,
        ranges,
        exponent,
        image_count,
        residual,
        residual_float,
        residual_bound,
        prepared.actions,
    )


class PeriodicEmbeddingResourceError(ValueError):
    """Report image or candidate allowance refusal before publication."""
