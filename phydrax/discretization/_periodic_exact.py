#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from fractions import Fraction
from functools import cache
from itertools import product
from math import lcm
from typing import final, TypeAlias, Union

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..typing import checked
from ._periodic_cell import PeriodicCell


PeriodicMeshIdentification: TypeAlias = Union[PeriodicCell, "PeriodicIsometryGroup"]
ExactIsometry: TypeAlias = tuple[tuple[Fraction, ...], ...]


class PeriodicIsometryIdentityError(ValueError):
    """An affine generator does not define the exact declared periodic group."""


def _fraction(value: object, /) -> Fraction:
    if isinstance(value, (int, np.integer)):
        return Fraction(int(value))
    if isinstance(value, (float, np.floating)):
        return Fraction.from_float(float(value))
    raise TypeError(
        "Exact periodic matrix entries must be real floating-point or integer scalars."
    )


def _exact_matrix(value: ArrayLike, /) -> ExactIsometry:
    matrix = np.asarray(value, dtype=np.float64)
    return tuple(tuple(_fraction(item) for item in row) for row in matrix)


def _identity(dimension: int, /) -> ExactIsometry:
    return tuple(
        tuple(Fraction(int(row == column)) for column in range(dimension))
        for row in range(dimension)
    )


def _multiply(first: ExactIsometry, second: ExactIsometry, /) -> ExactIsometry:
    return tuple(
        tuple(
            sum(
                (
                    first[row][inner] * second[inner][column]
                    for inner in range(len(second))
                ),
                Fraction(),
            )
            for column in range(len(second[0]))
        )
        for row in range(len(first))
    )


def _power(matrix: ExactIsometry, exponent: int, /) -> ExactIsometry:
    value = int(exponent)
    factor = matrix
    if value < 0:
        factor = _inverse_isometry(matrix)
        value = -value
    result = _identity(len(matrix))
    while value:
        if value & 1:
            result = _multiply(result, factor)
        value >>= 1
        if value:
            factor = _multiply(factor, factor)
    return result


def _inverse_isometry(matrix: ExactIsometry, /) -> ExactIsometry:
    dimension = len(matrix) - 1
    rotation = tuple(
        tuple(matrix[row][column] for column in range(dimension))
        for row in range(dimension)
    )
    inverse_rotation = tuple(
        tuple(rotation[column][row] for column in range(dimension))
        for row in range(dimension)
    )
    offset = tuple(matrix[row][dimension] for row in range(dimension))
    inverse_offset = tuple(
        -sum(
            (
                inverse_rotation[row][column] * offset[column]
                for column in range(dimension)
            ),
            Fraction(),
        )
        for row in range(dimension)
    )
    return tuple(
        tuple(inverse_rotation[row]) + (inverse_offset[row],) for row in range(dimension)
    ) + (tuple(Fraction() for _ in range(dimension)) + (Fraction(1),),)


def _determinant(matrix: tuple[tuple[Fraction, ...], ...], /) -> Fraction:
    rows = [list(row) for row in matrix]
    determinant = Fraction(1)
    for column in range(len(rows)):
        pivot = next((row for row in range(column, len(rows)) if rows[row][column]), None)
        if pivot is None:
            return Fraction()
        if pivot != column:
            rows[column], rows[pivot] = rows[pivot], rows[column]
            determinant = -determinant
        value = rows[column][column]
        determinant *= value
        for row in range(column + 1, len(rows)):
            if not rows[row][column]:
                continue
            scale = rows[row][column] / value
            for inner in range(column + 1, len(rows)):
                rows[row][inner] -= scale * rows[column][inner]
    return determinant


def _totient(value: int, /) -> int:
    result = value
    factor = 2
    remainder = value
    while factor * factor <= remainder:
        if remainder % factor == 0:
            while remainder % factor == 0:
                remainder //= factor
            result -= result // factor
        factor += 1
    if remainder > 1:
        result -= result // remainder
    return result


@cache
def _possible_rational_rotation_orders(dimension: int, /) -> tuple[int, ...]:
    """Finite-order candidates from cyclotomic-degree bounds for rational matrices."""
    # If a rational matrix has finite order m, each cyclotomic factor Phi_n in
    # its characteristic polynomial consumes phi(n) dimensions. The elementary
    # bound phi(n) >= sqrt(n / 2) makes this finite without a policy cutoff.
    maximum_n = max(1, 2 * dimension * dimension)
    states: list[set[int]] = [set() for _ in range(dimension + 1)]
    states[0].add(1)
    for order in range(1, maximum_n + 1):
        degree = _totient(order)
        if degree > dimension:
            continue
        for used in range(dimension - degree, -1, -1):
            for prior in tuple(states[used]):
                states[used + degree].add(lcm(prior, order))
    return tuple(sorted(set.union(*states)))


def _linear_order(matrix: ExactIsometry, /) -> int:
    dimension = len(matrix) - 1
    linear = tuple(
        tuple(matrix[row][column] for column in range(dimension))
        for row in range(dimension)
    )
    target = tuple(
        tuple(linear[row][column] for column in range(dimension))
        for row in range(dimension)
    )
    identity = _identity(dimension)
    for order in _possible_rational_rotation_orders(dimension):
        power = _identity(dimension)
        for _ in range(order):
            power = tuple(
                tuple(
                    sum(
                        (
                            power[row][inner] * target[inner][column]
                            for inner in range(dimension)
                        ),
                        Fraction(),
                    )
                    for column in range(dimension)
                )
                for row in range(dimension)
            )
        if power == identity:
            return order
    return 0


def _validate_faithful_generators(
    generators: tuple[ExactIsometry, ...],
    linear_orders: tuple[int, ...],
    orders: tuple[int, ...],
    /,
) -> None:
    dimension = len(generators[0]) - 1
    linear_identity = _identity(dimension)
    finite = []
    translations = []
    for matrix, linear_order, order in zip(
        generators, linear_orders, orders, strict=True
    ):
        if matrix == _identity(dimension + 1):
            raise PeriodicIsometryIdentityError(
                "Periodic generators must not be identity actions."
            )
        if order == 0:
            if linear_order <= 0:
                raise PeriodicIsometryIdentityError(
                    "Infinite isometry generators require a finite linear order."
                )
            pure_translation = _power(matrix, linear_order)
            pure_linear = tuple(
                tuple(pure_translation[row][column] for column in range(dimension))
                for row in range(dimension)
            )
            if pure_linear != linear_identity:
                raise PeriodicIsometryIdentityError(
                    "An infinite isometry generator must expose an exact pure-translation power."
                )
            translation = tuple(
                pure_translation[row][dimension] for row in range(dimension)
            )
            if not any(translation):
                raise PeriodicIsometryIdentityError(
                    "Infinite isometry generators must translate after their linear order."
                )
            translations.append(translation)
            if linear_order > 1:
                finite.append((matrix, linear_order))
        else:
            if linear_order == 0:
                raise PeriodicIsometryIdentityError(
                    "Finite affine actions require a proven finite linear order."
                )
            finite.append((matrix, order))
    if translations:
        rows = [list(row) for row in zip(*translations, strict=True)]
        rank = 0
        column = 0
        while rank < len(rows) and column < len(translations):
            pivot = next(
                (index for index in range(rank, len(rows)) if rows[index][column]),
                None,
            )
            if pivot is None:
                column += 1
                continue
            rows[rank], rows[pivot] = rows[pivot], rows[rank]
            divisor = rows[rank][column]
            rows[rank] = [value / divisor for value in rows[rank]]
            for index in range(len(rows)):
                if index == rank or not rows[index][column]:
                    continue
                factor = rows[index][column]
                rows[index] = [
                    value - factor * pivot_value
                    for value, pivot_value in zip(rows[index], rows[rank], strict=True)
                ]
            rank += 1
            column += 1
        if rank != len(translations):
            raise PeriodicIsometryIdentityError(
                "Infinite translational generators must be rationally independent."
            )
    seen = set()
    finite_ranges = tuple(range(order) for _, order in finite)
    for exponents in product(*finite_ranges):
        action = _identity(dimension + 1)
        for (matrix, _), exponent in zip(finite, exponents, strict=True):
            action = _multiply(action, _power(matrix, exponent))
        if action in seen:
            raise PeriodicIsometryIdentityError(
                "Periodic finite generators must have faithful independent exponent rows."
            )
        seen.add(action)


def _proper_isometry(matrix: ExactIsometry, /) -> None:
    if len(matrix) < 3 or any(len(row) != len(matrix) for row in matrix):
        raise PeriodicIsometryIdentityError(
            "Periodic generators must be square homogeneous maps with ambient dimension at least two."
        )
    dimension = len(matrix) - 1
    if matrix[-1] != tuple(Fraction() for _ in range(dimension)) + (Fraction(1),):
        raise PeriodicIsometryIdentityError(
            "Periodic generators must have the exact homogeneous bottom row."
        )
    linear = tuple(
        tuple(matrix[row][column] for column in range(dimension))
        for row in range(dimension)
    )
    gram = tuple(
        tuple(
            sum(
                (
                    linear[inner][row] * linear[inner][column]
                    for inner in range(dimension)
                ),
                Fraction(),
            )
            for column in range(dimension)
        )
        for row in range(dimension)
    )
    if gram != _identity(dimension):
        raise PeriodicIsometryIdentityError(
            "Periodic generators must be exact Euclidean isometries of represented coordinates."
        )
    if _determinant(linear) != 1:
        raise PeriodicIsometryIdentityError(
            "Periodic generators must preserve orientation exactly."
        )


def _affine_order(matrix: ExactIsometry, linear_order: int, /) -> int:
    if linear_order == 0:
        return 0
    identity = _identity(len(matrix))
    return linear_order if _power(matrix, linear_order) == identity else 0


def _integer_exponents(value: ArrayLike, count: int, /) -> np.ndarray:
    exponents = np.asarray(value)
    if exponents.shape != (count,) or exponents.dtype.kind not in "iu":
        raise ValueError("Periodic exponents must be one integer per generator.")
    if (
        exponents.dtype.kind == "u"
        and exponents.size
        and np.any(exponents > np.iinfo(np.int64).max)
    ):
        raise ValueError("Periodic exponents must fit int64.")
    return exponents.astype(np.int64, copy=False)


@final
class PeriodicIsometryGroup(StrictModule, NonTrainableState):
    """Commuting exact affine isometries identifying one periodic quotient."""

    generators: Array
    orders: tuple[int, ...] = eqx.field(static=True)
    linear_orders: tuple[int, ...] = eqx.field(static=True)
    translation_periods: tuple[int, ...] = eqx.field(static=True)
    group_id: str = eqx.field(static=True)
    _exact_generators: tuple[ExactIsometry, ...] = eqx.field(static=True)

    @checked
    def __init__(
        self,
        generators: ArrayLike,
        /,
    ) -> None:
        values = np.asarray(generators, dtype=np.float64)
        if (
            values.ndim != 3
            or values.shape[0] == 0
            or values.shape[1] != values.shape[2]
            or values.shape[1] < 3
        ):
            raise ValueError(
                "generators must have shape (rank > 0, dimension + 1, dimension + 1), with dimension at least two."
            )
        if np.any(~np.isfinite(values)):
            raise ValueError("Periodic generators must be finite.")
        exact = tuple(_exact_matrix(item) for item in values)
        for matrix in exact:
            _proper_isometry(matrix)
        for first in range(len(exact)):
            for second in range(first):
                if _multiply(exact[first], exact[second]) != _multiply(
                    exact[second], exact[first]
                ):
                    raise PeriodicIsometryIdentityError(
                        "Periodic generator actions must commute exactly."
                    )
        linear_orders = tuple(_linear_order(matrix) for matrix in exact)
        orders = tuple(
            _affine_order(matrix, order)
            for matrix, order in zip(exact, linear_orders, strict=True)
        )
        _validate_faithful_generators(exact, linear_orders, orders)
        self.generators = jnp.asarray(values)
        self.orders = orders
        self.linear_orders = linear_orders
        self.translation_periods = tuple(order for order in orders)
        self._exact_generators = exact
        self.group_id = canonical_fingerprint(
            {
                "kind": "periodic-isometry-group",
                "generators": array_tree_fingerprint(values),
                "orders": list(orders),
                "linear_orders": list(linear_orders),
            }
        )

    @property
    def rank(self) -> int:
        return len(self.orders)

    @property
    def ambient_dimension(self) -> int:
        return self.generators.shape[1] - 1

    def validate_restored(self, /) -> None:
        rebuilt = PeriodicIsometryGroup(self.generators)
        if rebuilt.group_id != self.group_id or rebuilt.orders != self.orders:
            raise ValueError(
                "Stored periodic group metadata does not match its generators."
            )

    def apply(self, points: ArrayLike, exponents: ArrayLike, /) -> np.ndarray:
        values = np.asarray(points, dtype=np.float64)
        if values.ndim < 1 or values.shape[-1] != self.ambient_dimension:
            raise ValueError("points must end in the periodic ambient dimension.")
        exponent = _integer_exponents(exponents, self.rank)
        matrix = exact_periodic_element(self, exponent)
        flat = values.reshape((-1, self.ambient_dimension))
        result = np.empty_like(flat)
        for row, point in enumerate(flat):
            exact_point = tuple(_fraction(item) for item in point) + (Fraction(1),)
            mapped = tuple(
                sum(
                    (
                        matrix[i][j] * exact_point[j]
                        for j in range(self.ambient_dimension + 1)
                    ),
                    Fraction(),
                )
                for i in range(self.ambient_dimension)
            )
            result[row] = np.asarray([float(item) for item in mapped], dtype=np.float64)
        return result.reshape(values.shape)

    def fixed(self, points: ArrayLike, /) -> np.ndarray:
        values = np.asarray(points, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.ambient_dimension:
            raise ValueError("points must have shape (point_count, ambient_dimension).")
        fixed = np.zeros((values.shape[0], self.rank), dtype=np.bool_)
        for generator, matrix in enumerate(self._exact_generators):
            for row in range(values.shape[0]):
                point = values[row, :]
                exact_point = tuple(_fraction(item) for item in point) + (Fraction(1),)
                mapped = tuple(
                    sum(
                        (
                            matrix[i][j] * exact_point[j]
                            for j in range(self.ambient_dimension + 1)
                        ),
                        Fraction(),
                    )
                    for i in range(self.ambient_dimension)
                )
                fixed[row, generator] = mapped == exact_point[:-1]
        return fixed


def reduce_periodic_exponents(
    values: ArrayLike,
    orders: ArrayLike | tuple[int, ...],
    /,
) -> np.ndarray:
    """Reduce finite actions while retaining distinct exponents for infinite windings."""
    exponents = np.asarray(values)
    periods = np.asarray(orders, dtype=np.int64)
    if (
        exponents.dtype.kind not in "iu"
        or exponents.ndim < 1
        or exponents.shape[-1] != periods.size
    ):
        raise ValueError(
            "Exponent rows must be integer-valued and end in the generator rank."
        )
    if periods.ndim != 1 or np.any(periods < 0):
        raise ValueError("orders must be a nonnegative integer vector.")
    return np.where(periods > 0, np.mod(exponents, np.maximum(periods, 1)), exponents)


def identification_id(cell: PeriodicMeshIdentification, /) -> str:
    if isinstance(cell, PeriodicCell):
        return cell.cell_id
    if isinstance(cell, PeriodicIsometryGroup):
        return cell.group_id
    raise TypeError("cell must be PeriodicCell or PeriodicIsometryGroup.")


def identification_orders(cell: PeriodicMeshIdentification, /) -> np.ndarray:
    if isinstance(cell, PeriodicCell):
        return np.asarray(
            [0 if periodic else 1 for periodic in cell.periodic_axes], dtype=np.int64
        )
    if isinstance(cell, PeriodicIsometryGroup):
        return np.asarray(cell.orders, dtype=np.int64)
    raise TypeError("cell must be PeriodicCell or PeriodicIsometryGroup.")


def exact_periodic_generators(
    cell: PeriodicMeshIdentification, /
) -> tuple[tuple[ExactIsometry, ...], tuple[int, ...]]:
    if isinstance(cell, PeriodicIsometryGroup):
        return cell._exact_generators, cell.orders
    if isinstance(cell, PeriodicCell):
        rank = cell.rank
        dimension = cell.ambient_dimension
        vectors = np.asarray(cell.vectors, dtype=np.float64)
        generators = []
        for axis in range(rank):
            matrix = [list(row) for row in _identity(dimension + 1)]
            if cell.periodic_axes[axis]:
                for row in range(dimension):
                    matrix[row][dimension] = _fraction(vectors[axis, row])
            generators.append(tuple(tuple(row) for row in matrix))
        return tuple(generators), tuple(
            0 if periodic else 1 for periodic in cell.periodic_axes
        )
    raise TypeError("cell must be PeriodicCell or PeriodicIsometryGroup.")


def exact_periodic_element(
    cell: PeriodicMeshIdentification, exponents: ArrayLike, /
) -> ExactIsometry:
    generators, orders = exact_periodic_generators(cell)
    powers = reduce_periodic_exponents(
        _integer_exponents(exponents, len(generators)), orders
    )
    result = _identity(cell.ambient_dimension + 1)
    for matrix, exponent in zip(generators, powers, strict=True):
        result = _multiply(result, _power(matrix, int(exponent)))
    return result


def apply_periodic(
    cell: PeriodicMeshIdentification, points: ArrayLike, exponents: ArrayLike, /
) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] != cell.ambient_dimension:
        raise ValueError("points must end in the periodic ambient dimension.")
    shifts = np.asarray(exponents)
    rank = len(identification_orders(cell))
    if shifts.shape == (rank,):
        shifts = np.broadcast_to(shifts, values.shape[:-1] + (rank,))
    if shifts.shape != values.shape[:-1] + (rank,) or shifts.dtype.kind not in "iu":
        raise ValueError(
            "exponents must be one integer row per point, aligned with the periodic rank."
        )
    flat_points = values.reshape((-1, values.shape[-1]))
    flat_shifts = shifts.reshape((-1, rank))
    output = np.empty_like(flat_points)
    for row, (point, exponent) in enumerate(zip(flat_points, flat_shifts, strict=True)):
        matrix = exact_periodic_element(cell, exponent)
        vector = tuple(_fraction(item) for item in point) + (Fraction(1),)
        output[row] = np.asarray(
            [
                float(
                    sum(
                        (matrix[i][j] * vector[j] for j in range(len(vector))),
                        Fraction(),
                    )
                )
                for i in range(cell.ambient_dimension)
            ],
            dtype=np.float64,
        )
    return output.reshape(values.shape)


__all__ = [
    "ExactIsometry",
    "PeriodicIsometryGroup",
    "PeriodicIsometryIdentityError",
    "PeriodicMeshIdentification",
    "apply_periodic",
    "exact_periodic_element",
    "exact_periodic_generators",
    "identification_id",
    "identification_orders",
    "reduce_periodic_exponents",
]
