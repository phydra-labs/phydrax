#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
# Genz--Malik orbit and moment construction includes adaptations from quadax
# commit 4b9e6de792bdbfe410c723374dc17ec53386a667 under LICENSES/QUADAX-MIT.txt.

from __future__ import annotations

import functools
import itertools
import math
from fractions import Fraction
from numbers import Integral
from typing import Literal, NamedTuple, TypeAlias, TypeVar

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._quadrature_rules import QuadratureRuleData


_Scalar = TypeVar("_Scalar", float, Fraction)


AdaptiveCubatureFamily: TypeAlias = Literal[
    "genz-malik",
    "tensor-gauss-kronrod",
]

GENZ_MALIK_DEGREES = (7, 9, 11, 13)
_DEFAULT_MAXIMUM_POINTS = 65_536
_DEFAULT_MAXIMUM_BYTES = 64 * 1024**2
_FS_DELTA2 = {
    7: Fraction(9, 19),
    9: Fraction(9, 19),
    11: Fraction(9, 19),
    13: Fraction(4707, 10000),
}


class AdaptiveCubatureRuleData(StrictModule, NonTrainableState):
    """One embedded cubature rule and its directional split operators."""

    points: Array
    weights: Array
    embedded_weights: Array
    split_weights: Array
    dimension: int = eqx.field(static=True)
    exact_degree: int | None = eqx.field(static=True)
    embedded_degree: int | None = eqx.field(static=True)
    family: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    storage_bytes: int = eqx.field(static=True)
    rule_id: str = eqx.field(static=True)
    weight_l1_norm: float = eqx.field(static=True)
    negative_weight_mass: float = eqx.field(static=True)

    def __init__(
        self,
        points: ArrayLike,
        weights: ArrayLike,
        embedded_weights: ArrayLike,
        split_weights: ArrayLike,
        /,
        *,
        exact_degree: int | None,
        embedded_degree: int | None,
        family: AdaptiveCubatureFamily,
        source_id: str,
        maximum_rule_bytes: int = _DEFAULT_MAXIMUM_BYTES,
    ):
        points_host = np.asarray(points, dtype=float)
        weights_host = np.asarray(weights, dtype=float).reshape((-1,))
        embedded_host = np.asarray(embedded_weights, dtype=float).reshape((-1,))
        split_host = np.asarray(split_weights, dtype=float)
        if points_host.ndim != 2 or points_host.shape[0] < 1 or points_host.shape[1] < 1:
            raise ValueError(
                "Adaptive cubature points must have shape (num_points, dimension)."
            )
        num_points, dimension = points_host.shape
        if weights_host.shape != (num_points,) or embedded_host.shape != (num_points,):
            raise ValueError("Adaptive cubature weights must align with the point axis.")
        if split_host.shape != (dimension, num_points):
            raise ValueError(
                "Adaptive cubature split weights must have shape (dimension, num_points)."
            )
        if family not in ("genz-malik", "tensor-gauss-kronrod"):
            raise ValueError(f"Unsupported adaptive cubature family: {family!r}.")
        if not source_id:
            raise ValueError("Adaptive cubature source_id must be nonempty.")
        if any(
            np.any(~np.isfinite(value))
            for value in (points_host, weights_host, embedded_host, split_host)
        ):
            raise ValueError("Adaptive cubature data must be finite.")
        if np.any(np.abs(points_host) >= 1.0):
            raise ValueError(
                "Adaptive cubature points must lie strictly inside the cube."
            )

        order = np.lexsort(
            tuple(points_host[:, axis] for axis in range(dimension - 1, -1, -1))
        )
        points_host = np.asarray(points_host[order], dtype=float)
        weights_host = np.asarray(weights_host[order], dtype=float)
        embedded_host = np.asarray(embedded_host[order], dtype=float)
        split_host = np.asarray(split_host[:, order], dtype=float)
        if num_points > 1 and np.any(np.all(points_host[1:] == points_host[:-1], axis=1)):
            raise ValueError("Adaptive cubature points must be unique.")

        tolerance = 2048.0 * np.finfo(float).eps * max(1, num_points)
        mass = float(2**dimension)
        if not np.isclose(np.sum(weights_host), mass, rtol=tolerance, atol=tolerance):
            raise ValueError("High cubature weights do not integrate the cube mass.")
        if not np.isclose(np.sum(embedded_host), mass, rtol=tolerance, atol=tolerance):
            raise ValueError("Embedded cubature weights do not integrate the cube mass.")
        if not np.allclose(
            np.sum(split_host, axis=1), 0.0, rtol=tolerance, atol=tolerance
        ):
            raise ValueError("Cubature split operators must annihilate constants.")

        degree = _optional_degree(exact_degree, "exact_degree")
        embedded = _optional_degree(embedded_degree, "embedded_degree")
        if degree is not None and embedded is not None and embedded >= degree:
            raise ValueError("embedded_degree must be lower than exact_degree.")
        maximum_bytes = _positive_integer(maximum_rule_bytes, "maximum_rule_bytes")
        storage_bytes = int(
            points_host.nbytes
            + weights_host.nbytes
            + embedded_host.nbytes
            + split_host.nbytes
        )
        if storage_bytes > maximum_bytes:
            raise ValueError(
                "Adaptive cubature rule requires "
                f"{storage_bytes} bytes, exceeding maximum_rule_bytes={maximum_bytes}."
            )

        self.points = jnp.asarray(points_host)
        self.weights = jnp.asarray(weights_host)
        self.embedded_weights = jnp.asarray(embedded_host)
        self.split_weights = jnp.asarray(split_host)
        self.dimension = int(dimension)
        self.exact_degree = degree
        self.embedded_degree = embedded
        self.family = str(family)
        self.source_id = str(source_id)
        self.storage_bytes = storage_bytes
        self.weight_l1_norm = float(np.sum(np.abs(weights_host)))
        self.negative_weight_mass = float(-np.sum(weights_host[weights_host < 0.0]))
        self.rule_id = canonical_fingerprint(
            {
                "kind": "adaptive-cubature-rule",
                "family": self.family,
                "dimension": self.dimension,
                "exact_degree": self.exact_degree,
                "embedded_degree": self.embedded_degree,
                "source_id": self.source_id,
                "storage_bytes": self.storage_bytes,
                "weight_l1_norm": self.weight_l1_norm,
                "negative_weight_mass": self.negative_weight_mass,
                "data": array_tree_fingerprint(
                    (points_host, weights_host, embedded_host, split_host)
                ),
            }
        )

    @property
    def num_points(self) -> int:
        return int(self.points.shape[0])


class _GenzMalikHostTable(NamedTuple):
    points: np.ndarray
    weights: np.ndarray
    embedded_weights: np.ndarray
    split_weights: np.ndarray


def _positive_integer(value: int, name: str, /) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive.")
    return result


def _optional_degree(value: int | None, name: str, /) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer or None.")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative.")
    return result


def _dimension(value: int, /) -> int:
    return _positive_integer(value, "Adaptive cubature dimension")


def _degree(value: int, /) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("Genz--Malik degree must be an integer.")
    degree = int(value)
    if degree not in GENZ_MALIK_DEGREES:
        raise ValueError(f"Genz--Malik degree must be one of {GENZ_MALIK_DEGREES}.")
    return degree


def _even_classes(degree: int, dimension: int) -> list[tuple[int, ...]]:
    out: list[tuple[int, ...]] = []

    def rec(remaining: int, largest: int, current: tuple[int, ...]) -> None:
        out.append(current)
        for part in range(min(remaining, largest), 1, -2):
            if len(current) < dimension:
                rec(remaining - part, part, current + (part,))

    rec(degree - degree % 2, degree, ())
    return sorted(set(out))


def _distinct_permutations(
    values: tuple[_Scalar, ...],
) -> list[tuple[_Scalar, ...]]:
    distinct = sorted(set(values), key=repr)
    counts = [values.count(value) for value in distinct]
    out: list[tuple[_Scalar, ...]] = []

    def rec(current: tuple[_Scalar, ...]) -> None:
        if len(current) == len(values):
            out.append(current)
            return
        for index, value in enumerate(distinct):
            if counts[index] > 0:
                counts[index] -= 1
                rec(current + (value,))
                counts[index] += 1

    rec(())
    return out


def _moment_system(
    orbits: list[dict[tuple[Fraction, ...], int]],
    dimension: int,
    degree: int,
) -> tuple[list[list[Fraction]], list[Fraction]]:
    matrix: list[list[Fraction]] = []
    rhs: list[Fraction] = []
    for alpha in _even_classes(degree, dimension):
        exponents = alpha + (0,) * (dimension - len(alpha))
        value = Fraction(1)
        for exponent in exponents:
            value *= Fraction(2, exponent + 1)
        rhs.append(value)
        row: list[Fraction] = []
        for orbit in orbits:
            total = Fraction(0)
            for point, count in orbit.items():
                term = Fraction(count)
                for coordinate, exponent in zip(point, exponents, strict=True):
                    term *= coordinate ** (exponent // 2)
                total += term
            row.append(total)
        matrix.append(row)
    return matrix, rhs


def _orbit_weights(
    orbits: list[dict[tuple[Fraction, ...], int]],
    dimension: int,
    degree: int,
) -> tuple[list[Fraction], Fraction]:
    matrix, rhs = _moment_system(orbits, dimension, degree)
    num_rows = len(matrix)
    num_columns = len(orbits)
    augmented = [list(row) + [value] for row, value in zip(matrix, rhs, strict=True)]
    pivots: list[int] = []
    for column in range(num_columns):
        selected = next(
            (row for row in range(len(pivots), num_rows) if augmented[row][column]),
            None,
        )
        if selected is None:
            continue
        pivot_row = len(pivots)
        augmented[pivot_row], augmented[selected] = (
            augmented[selected],
            augmented[pivot_row],
        )
        pivot = augmented[pivot_row][column]
        augmented[pivot_row] = [value / pivot for value in augmented[pivot_row]]
        for row in range(num_rows):
            if row != pivot_row and augmented[row][column]:
                factor = augmented[row][column]
                augmented[row] = [
                    value - factor * pivot_value
                    for value, pivot_value in zip(
                        augmented[row], augmented[pivot_row], strict=True
                    )
                ]
        pivots.append(column)
    if len(pivots) < num_columns:
        raise RuntimeError(
            "The Genz--Malik moment equations do not determine every orbit weight."
        )
    weights = [Fraction(0)] * num_columns
    for row, column in enumerate(pivots):
        weights[column] = augmented[row][num_columns]
    residual = max(
        (abs(augmented[row][num_columns]) for row in range(len(pivots), num_rows)),
        default=Fraction(0),
    )
    return weights, residual


def _fs_partitions(maximum: int, dimension: int) -> list[tuple[int, ...]]:
    out: list[tuple[int, ...]] = []

    def rec(remaining: int, largest: int, current: tuple[int, ...]) -> None:
        out.append(current)
        for part in range(min(remaining, largest), 0, -1):
            if len(current) < dimension:
                rec(remaining - part, part, current + (part,))

    rec(maximum, maximum, ())
    return sorted(set(out))


def _elementary_symmetric(values: list[Fraction], degree: int) -> Fraction:
    if degree < 0 or degree > len(values):
        return Fraction(0)
    return sum(
        (
            functools.reduce(lambda left, right: left * right, choice, Fraction(1))
            for choice in itertools.combinations(values, degree)
        ),
        Fraction(0),
    )


def _fs_generators(count: int, delta_squared: Fraction) -> list[Fraction]:
    generators: list[Fraction] = []
    for order in range(2, count + 2):
        rhs = Fraction(1, 3**order)
        for previous in generators:
            rhs *= 1 - previous / delta_squared
        constant = sum(
            (
                (-1) ** index
                * _elementary_symmetric(generators, index)
                / (2 * (order - index) + 1)
                for index in range(order - 1)
            ),
            Fraction(0),
        )
        coefficient = sum(
            (
                (-1) ** index
                * _elementary_symmetric(generators, index - 1)
                / (2 * (order - index) + 1)
                for index in range(1, order)
            ),
            Fraction(0),
        )
        generators.append((rhs - constant) / (coefficient + rhs / delta_squared))
    return generators


def _orbit_size(pattern: tuple[int, ...], dimension: int) -> int:
    active = len(pattern)
    multiplicities = [pattern.count(value) for value in set(pattern)]
    arrangements = math.factorial(dimension) // math.factorial(dimension - active)
    for multiplicity in multiplicities:
        arrangements //= math.factorial(multiplicity)
    return arrangements * 2**active


def _genz_malik_point_count(dimension: int, degree: int) -> int:
    maximum = (degree - 1) // 2 - 1
    return (
        sum(
            _orbit_size(pattern, dimension)
            for pattern in _fs_partitions(maximum, dimension)
        )
        + 2**dimension
    )


def _fs_orbit(generators: list[float], dimension: int) -> np.ndarray:
    padded = tuple(generators) + (0.0,) * (dimension - len(generators))
    points: set[tuple[float, ...]] = set()
    for permutation in _distinct_permutations(padded):
        signs = itertools.product(
            *((1.0, -1.0) if value else (1.0,) for value in permutation)
        )
        points.update(
            tuple(
                sign * float(value)
                for sign, value in zip(choice, permutation, strict=True)
            )
            for choice in signs
        )
    return np.asarray(sorted(points), dtype=float).reshape((-1, dimension))


def _fs_orbit_squares(
    generators: list[Fraction], dimension: int
) -> dict[tuple[Fraction, ...], int]:
    padded = tuple(generators) + (Fraction(0),) * (dimension - len(generators))
    return {
        tuple(permutation): 2 ** sum(1 for coordinate in permutation if coordinate)
        for permutation in _distinct_permutations(padded)
    }


@functools.lru_cache(maxsize=32)
def _genz_malik_host_table(dimension: int, degree: int) -> _GenzMalikHostTable:
    order = (degree - 1) // 2
    delta_squared = _FS_DELTA2[degree]
    squares = [Fraction(0)] + _fs_generators(order - 1, delta_squared)
    generators = [math.sqrt(float(value)) for value in squares]
    patterns = _fs_partitions(order - 1, dimension)
    orbits = [
        _fs_orbit([generators[index] for index in pattern], dimension)
        for pattern in patterns
    ]
    exact_orbits = [
        _fs_orbit_squares([squares[index] for index in pattern], dimension)
        for pattern in patterns
    ]
    orbits.append(_fs_orbit([math.sqrt(float(delta_squared))] * dimension, dimension))
    exact_orbits.append(_fs_orbit_squares([delta_squared] * dimension, dimension))

    high_orbit_weights, high_residual = _orbit_weights(exact_orbits, dimension, degree)
    embedded_orbit_weights, embedded_residual = _orbit_weights(
        exact_orbits[:-1], dimension, degree - 2
    )
    if max(high_residual, embedded_residual) != 0:
        raise RuntimeError(
            "The Genz--Malik generators do not satisfy their exact moment equations."
        )

    sizes = np.asarray([len(orbit) for orbit in orbits], dtype=int)
    points = np.concatenate(orbits, axis=0)
    weights = np.repeat(
        np.asarray([float(weight) for weight in high_orbit_weights]), sizes
    )
    embedded_weights = np.repeat(
        np.asarray([float(weight) for weight in embedded_orbit_weights] + [0.0]),
        sizes,
    )

    index = {tuple(map(float, point)): position for position, point in enumerate(points)}
    ratio = float(squares[2] / squares[1])
    split_weights = np.zeros((dimension, len(points)), dtype=float)
    center = np.zeros((dimension,), dtype=float)
    for axis in range(dimension):
        split_weights[axis, index[tuple(center)]] = -2.0 * (1.0 - ratio)
        for sign in (1.0, -1.0):
            for generator_index, weight in ((2, 1.0), (1, -ratio)):
                node = np.zeros((dimension,), dtype=float)
                node[axis] = sign * generators[generator_index]
                split_weights[axis, index[tuple(node)]] = weight

    return _GenzMalikHostTable(
        points,
        weights,
        embedded_weights,
        split_weights,
    )


def genz_malik_rule_data(
    dimension: int,
    degree: int = 9,
    *,
    maximum_points: int = _DEFAULT_MAXIMUM_POINTS,
    maximum_rule_bytes: int = _DEFAULT_MAXIMUM_BYTES,
) -> AdaptiveCubatureRuleData:
    """Prepare one published embedded Genz--Malik rule on ``[-1, 1]^d``."""
    dimension_ = _dimension(dimension)
    degree_ = _degree(degree)
    maximum_points_ = _positive_integer(maximum_points, "maximum_points")
    maximum_bytes = _positive_integer(maximum_rule_bytes, "maximum_rule_bytes")
    count = _genz_malik_point_count(dimension_, degree_)
    if count > maximum_points_:
        raise ValueError(
            f"Genz--Malik degree {degree_} in dimension {dimension_} requires "
            f"{count} points, exceeding maximum_points={maximum_points_}."
        )
    required_bytes = count * (2 * dimension_ + 2) * np.dtype(float).itemsize
    if required_bytes > maximum_bytes:
        raise ValueError(
            f"Genz--Malik degree {degree_} in dimension {dimension_} requires "
            f"{required_bytes} bytes, exceeding maximum_rule_bytes={maximum_bytes}."
        )
    table = _genz_malik_host_table(dimension_, degree_)
    if table.points.shape[0] != count:
        raise RuntimeError("Genz--Malik orbit count disagrees with its expansion.")
    return AdaptiveCubatureRuleData(
        table.points,
        table.weights,
        table.embedded_weights,
        table.split_weights,
        exact_degree=degree_,
        embedded_degree=degree_ - 2,
        family="genz-malik",
        source_id="doi:10.1137/0720038",
        maximum_rule_bytes=maximum_bytes,
    )


def _outer(values: tuple[np.ndarray, ...]) -> np.ndarray:
    result = np.asarray(1.0)
    for value in values:
        result = np.multiply.outer(result, value)
    return np.asarray(result).reshape((-1,))


def tensor_product_cubature_rule_data(
    rules: tuple[QuadratureRuleData, ...],
    *,
    source_ids: tuple[str, ...],
    maximum_points: int = _DEFAULT_MAXIMUM_POINTS,
    maximum_rule_bytes: int = _DEFAULT_MAXIMUM_BYTES,
) -> AdaptiveCubatureRuleData:
    """Prepare one tensor product of embedded one-dimensional rules."""
    if len(rules) < 1 or len(source_ids) != len(rules):
        raise ValueError("Tensor cubature requires one source ID per nonempty axis rule.")
    maximum_points_ = _positive_integer(maximum_points, "maximum_points")
    maximum_bytes = _positive_integer(maximum_rule_bytes, "maximum_rule_bytes")
    for axis, rule in enumerate(rules):
        if rule.embedded_weights is None:
            raise ValueError(
                f"Tensor cubature axis {axis} does not carry embedded weights."
            )
    dimension = len(rules)
    counts = tuple(int(rule.nodes.shape[0]) for rule in rules)
    count = math.prod(counts)
    if count > maximum_points_:
        raise ValueError(
            f"Tensor cubature requires {count} points, exceeding "
            f"maximum_points={maximum_points_}."
        )
    required_bytes = count * (2 * dimension + 2) * np.dtype(float).itemsize
    if required_bytes > maximum_bytes:
        raise ValueError(
            f"Tensor cubature requires {required_bytes} bytes, exceeding "
            f"maximum_rule_bytes={maximum_bytes}."
        )

    nodes = tuple(np.asarray(rule.nodes, dtype=float) for rule in rules)
    high = tuple(np.asarray(rule.weights, dtype=float) for rule in rules)
    embedded = tuple(np.asarray(rule.embedded_weights, dtype=float) for rule in rules)
    points = np.stack(np.meshgrid(*nodes, indexing="ij"), axis=-1).reshape(
        (-1, dimension)
    )
    split_weights = np.stack(
        [
            _outer(
                tuple(
                    high[other] if other != axis else high[axis] - embedded[axis]
                    for other in range(dimension)
                )
            )
            for axis in range(dimension)
        ],
        axis=0,
    )
    exact_degrees = tuple(rule.degree for rule in rules)
    available_exact_degrees = tuple(
        degree for degree in exact_degrees if degree is not None
    )
    embedded_degrees = tuple(
        2 * int(np.count_nonzero(axis_weights)) - 1 for axis_weights in embedded
    )
    return AdaptiveCubatureRuleData(
        points,
        _outer(high),
        _outer(embedded),
        split_weights,
        exact_degree=(
            min(available_exact_degrees)
            if len(available_exact_degrees) == len(rules)
            else None
        ),
        embedded_degree=(min(embedded_degrees) if embedded_degrees else None),
        family="tensor-gauss-kronrod",
        source_id="tensor-product:" + ":".join(source_ids),
        maximum_rule_bytes=maximum_bytes,
    )


__all__ = [
    "AdaptiveCubatureFamily",
    "AdaptiveCubatureRuleData",
    "GENZ_MALIK_DEGREES",
    "genz_malik_rule_data",
    "tensor_product_cubature_rule_data",
]
