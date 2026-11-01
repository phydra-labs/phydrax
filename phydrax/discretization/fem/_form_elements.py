# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Polynomial differential-form elements with entity moment functionals.

Simplex cells use ``simplex:N`` and tensor cells use ``tensor:N``. Named
interval, triangle, tetrahedron, quadrilateral and hexahedron cells are accepted.
All component axes follow the exterior owner's lexicographic blade convention.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from fractions import Fraction
from functools import lru_cache
from hashlib import sha256
from itertools import combinations, product
from math import comb, factorial, lcm, prod
from typing import final, Literal, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._host_table_cache import host_table_cache
from ..._polynomial._cubature import simplex_rule_data, tensor_product_rule_data
from ..._polynomial._orthogonal import (
    standard_series_derivative_coefficients,
    standard_vandermonde,
)
from ..._strict import StrictModule
from ...ein import contract
from ...exterior._form_type import FormProxy, FormTwist, FormType, FormValueSpec
from ...linalg._exact import determinant_exact, matmul_exact, solve_exact
from ...typing import parse
from .._reference_cell import reference_cell_topology
from ._reference import FiniteElementSpec


type FormElementFamily = Literal["trimmed", "full", "tensor-trimmed"]
type DofLabel = tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]
type _Polynomial = dict[tuple[int, ...], float]
type _ExactPolynomial = dict[tuple[int, ...], Fraction]
type _HostArray = npt.NDArray[np.float64]
type _ExactCoefficientBank = tuple[tuple[tuple[Fraction, ...], ...], ...]
type _ExactMomentBank = tuple[tuple[tuple[Fraction, ...], ...], ...]
type _ExactMatrix = tuple[tuple[Fraction, ...], ...]
# One shared exact zero keeps large sparse banks to one pointer per zero entry.
_EXACT_ZERO = Fraction(0)


@lru_cache(maxsize=128)
def _exact_data_digest(value: object, /) -> str:
    """SHA-256 of a tagged, length-prefixed encoding of nested Fraction tuples."""
    digest = sha256()
    update = digest.update
    # Banks share Fraction objects and repeat few values: encode each value once
    # and remember it per object (every object stays alive inside ``value``).
    by_value: dict[tuple[int, int], bytes] = {}
    by_object: dict[int, bytes] = {}

    def encode(item: Fraction, /) -> bytes:
        key = (item.numerator, item.denominator)
        cached = by_value.get(key)
        if cached is None:
            parts = [b"F"]
            for integer in key:
                magnitude = abs(integer)
                length = max(1, (magnitude.bit_length() + 7) // 8)
                parts.append(b"-" if integer < 0 else b"+")
                parts.append(length.to_bytes(8, "big"))
                parts.append(magnitude.to_bytes(length, "big"))
            cached = by_value[key] = b"".join(parts)
        by_object[id(item)] = cached
        return cached

    def feed(item: object, /) -> None:
        if isinstance(item, Fraction):
            update(by_object.get(id(item)) or encode(item))
            return
        if not isinstance(item, tuple):
            raise TypeError("Exact basis fingerprints require tuples of Fractions.")
        update(b"T")
        update(len(item).to_bytes(8, "big"))
        for child in item:
            if type(child) is Fraction:
                update(by_object.get(id(child)) or encode(child))
            else:
                feed(child)

    feed(value)
    return digest.hexdigest()


def _compositions(total: int, length: int) -> tuple[tuple[int, ...], ...]:
    if length == 0:
        return ((),) if total == 0 else ()
    if length == 1:
        return ((total,),)
    return tuple(
        (first, *rest)
        for first in range(total + 1)
        for rest in _compositions(total - first, length - 1)
    )


def _exponents(n: int, r: int) -> tuple[tuple[int, ...], ...]:
    return tuple(alpha for degree in range(r + 1) for alpha in _compositions(degree, n))


def _multiply(left: _Polynomial, right: _Polynomial) -> _Polynomial:
    result: _Polynomial = {}
    for alpha, a in left.items():
        for beta, b in right.items():
            gamma = tuple(x + y for x, y in zip(alpha, beta, strict=True))
            result[gamma] = result.get(gamma, 0.0) + a * b
    return result


def _power(poly: _Polynomial, degree: int, n: int) -> _Polynomial:
    result: _Polynomial = {(0,) * n: 1.0}
    for _ in range(degree):
        result = _multiply(result, poly)
    return result


def _minor(matrix: _HostArray, rows: tuple[int, ...], columns: tuple[int, ...]) -> float:
    if not rows:
        return 1.0
    return float(np.linalg.det(matrix[np.ix_(rows, columns)]))


type _GeneratorLabel = tuple[tuple[int, ...], tuple[int, ...]]


def _generators(
    n: int, k: int, r: int, trimmed: bool
) -> tuple[tuple[tuple[int, ...], ...], _HostArray, tuple[_GeneratorLabel, ...]]:
    """Project the canonical exact simplex generators for execution."""
    exponents, exact_coefficients, labels = _exact_generators(n, k, r, trimmed)
    return exponents, np.asarray(exact_coefficients, dtype=np.float64), labels


def _exact_multiply(
    left: _ExactPolynomial, right: _ExactPolynomial, /
) -> _ExactPolynomial:
    result: _ExactPolynomial = {}
    for alpha, first in left.items():
        for beta, second in right.items():
            exponent = tuple(a + b for a, b in zip(alpha, beta, strict=True))
            result[exponent] = result.get(exponent, Fraction(0)) + first * second
    return {exponent: value for exponent, value in result.items() if value}


def _exact_power(
    polynomial: _ExactPolynomial, degree: int, variables: int, /
) -> _ExactPolynomial:
    result: _ExactPolynomial = {(0,) * variables: Fraction(1)}
    for _ in range(degree):
        result = _exact_multiply(result, polynomial)
    return result


def _exact_add_scaled(
    destination: _ExactPolynomial,
    source: _ExactPolynomial,
    scale: Fraction,
    /,
) -> None:
    if not scale:
        return
    for exponent, value in source.items():
        result = destination.get(exponent, Fraction(0)) + scale * value
        if result:
            destination[exponent] = result
        else:
            destination.pop(exponent, None)


def _exact_barycentric(n: int, /) -> tuple[_ExactPolynomial, ...]:
    zero = (0,) * n
    axes = tuple(tuple(int(i == j) for i in range(n)) for j in range(n))
    return (
        {
            zero: Fraction(1),
            **{axis: Fraction(-1) for axis in axes},
        },
        *({axis: Fraction(1)} for axis in axes),
    )


def _exact_barycentric_product(alpha: tuple[int, ...], n: int, /) -> _ExactPolynomial:
    result: _ExactPolynomial = {(0,) * n: Fraction(1)}
    for exponent, coordinate in zip(alpha, _exact_barycentric(n), strict=True):
        result = _exact_multiply(result, _exact_power(coordinate, exponent, n))
    return result


def _exact_minor(
    matrix: tuple[tuple[Fraction, ...], ...],
    rows: tuple[int, ...],
    columns: tuple[int, ...],
    /,
) -> Fraction:
    if not rows:
        return Fraction(1)
    return Fraction(
        determinant_exact(
            tuple(tuple(matrix[row][column] for column in columns) for row in rows)
        )
    )


@lru_cache(maxsize=128)
def _exact_generators(
    n: int, k: int, r: int, trimmed: bool, /
) -> tuple[
    tuple[tuple[int, ...], ...],
    _ExactCoefficientBank,
    tuple[_GeneratorLabel, ...],
]:
    """Exact Bernstein--Whitney generators, or full polynomial generators."""
    exponents = _exponents(n, max(r, 0))
    blades = tuple(combinations(range(n), k))
    generators: list[tuple[tuple[_ExactPolynomial, ...], _GeneratorLabel]] = []
    if not trimmed or k == 0:
        for alpha in _compositions(r, n + 1):
            scalar = _exact_barycentric_product(alpha, n)
            for component, blade in enumerate(blades):
                components = tuple(
                    scalar if index == component else {} for index in range(len(blades))
                )
                generators.append(
                    (components, (alpha, tuple(axis + 1 for axis in blade)))
                )
    else:
        gradients = tuple(
            tuple(Fraction(-1) for _ in range(n))
            if vertex == 0
            else tuple(Fraction(int(axis == vertex - 1)) for axis in range(n))
            for vertex in range(n + 1)
        )
        barycentric = _exact_barycentric(n)
        for sigma in combinations(range(n + 1), k + 1):
            for alpha in _compositions(r - 1, n + 1):
                if any(alpha[index] for index in range(sigma[0])):
                    continue
                scalar = _exact_barycentric_product(alpha, n)
                components: list[_ExactPolynomial] = [{} for _ in range(len(blades))]
                for position, vertex in enumerate(sigma):
                    other = tuple(item for item in sigma if item != vertex)
                    polynomial = _exact_multiply(scalar, barycentric[vertex])
                    for component, blade in enumerate(blades):
                        factor = Fraction(
                            factorial(k) * (-1 if position % 2 else 1)
                        ) * _exact_minor(gradients, other, blade)
                        _exact_add_scaled(components[component], polynomial, factor)
                generators.append((tuple(components), (alpha, sigma)))
    coefficients: _ExactCoefficientBank = tuple(
        tuple(
            tuple(
                polynomial[component].get(exponent, Fraction(0))
                for polynomial, _ in generators
            )
            for component in range(len(blades))
        )
        for exponent in exponents
    )
    return (
        exponents,
        coefficients,
        tuple(label for _, label in generators),
    )


@lru_cache(maxsize=128)
def _exact_dual_coefficients(
    generators: _ExactCoefficientBank,
    moments: _ExactMomentBank,
    /,
) -> _ExactCoefficientBank:
    dof_count = len(moments)
    generator_count = len(generators[0][0]) if generators and generators[0] else 0
    if (
        dof_count != generator_count
        or any(len(row) != len(generators) for row in moments)
        or any(len(row) != len(generators[0]) for row in generators)
        or any(
            len(moments[dof][monomial]) != len(generators[monomial])
            for dof in range(dof_count)
            for monomial in range(len(generators))
        )
    ):
        raise ValueError("Exact polynomial generators and moment functionals disagree.")
    component_count = len(generators[0])
    # pairing[dof][generator] = moments[dof] . generators[:, :, generator]
    pairing = _exact_moment_product(
        _integer_moment_rows(moments),
        _integer_term_columns(
            tuple(row for components in generators for row in components),
            generator_count,
        ),
    )
    inverse = solve_exact(
        pairing,
        [
            [Fraction(int(row == column)) for column in range(dof_count)]
            for row in range(dof_count)
        ],
    )
    # coefficients[monomial][component] = generators[monomial][component] @ inverse
    coefficients = _exact_moment_product(
        _integer_rows(tuple(row for components in generators for row in components)),
        _integer_term_columns(inverse, dof_count),
    )
    return tuple(
        coefficients[start : start + component_count]
        for start in range(0, len(coefficients), component_count)
    )


def _exact_shifted_legendre(degree: int, /) -> _ExactPolynomial:
    return {
        (power,): Fraction(
            (-1 if (degree + power) % 2 else 1)
            * comb(degree, power)
            * comb(degree + power, power)
        )
        for power in range(degree + 1)
    }


def _exact_integral_interval(polynomial: _ExactPolynomial, exponent: int, /) -> Fraction:
    return sum(
        (
            coefficient / (power[0] + exponent + 1)
            for power, coefficient in polynomial.items()
        ),
        Fraction(0),
    )


@lru_cache(maxsize=128)
def _exact_tensor_interval_data(
    order: int, /
) -> tuple[
    tuple[tuple[Fraction, ...], ...],
    tuple[tuple[Fraction, ...], ...],
    tuple[_ExactPolynomial, ...],
    tuple[_ExactPolynomial, ...],
    tuple[_ExactPolynomial, ...],
    tuple[_ExactPolynomial, ...],
]:
    zero_candidates: list[_ExactPolynomial] = [
        {(0,): Fraction(1), (1,): Fraction(-1)},
        {(1,): Fraction(1)},
    ]
    bubble: _ExactPolynomial = {(1,): Fraction(1), (2,): Fraction(-1)}
    zero_candidates.extend(
        _exact_multiply(bubble, _exact_shifted_legendre(mode))
        for mode in range(order - 1)
    )
    one_candidates = tuple(_exact_shifted_legendre(mode) for mode in range(order))
    interior = order - 1
    zero_coefficients = [
        [Fraction(0) for _ in range(order + 1)] for _ in range(order + 1)
    ]
    zero_coefficients[0][0] = Fraction(1)
    zero_coefficients[1][order] = Fraction(1)
    if interior:
        matrix = [
            [
                _exact_integral_interval(zero_candidates[mode + 2], exponent)
                for mode in range(interior)
            ]
            for exponent in range(interior)
        ]
        right = [
            [
                -_exact_integral_interval(zero_candidates[0], exponent),
                *(Fraction(int(column == exponent)) for column in range(interior)),
                -_exact_integral_interval(zero_candidates[1], exponent),
            ]
            for exponent in range(interior)
        ]
        solved = solve_exact(matrix, right)
        for mode, row in enumerate(solved, start=2):
            zero_coefficients[mode] = row
    zero = []
    for dof in range(order + 1):
        polynomial: _ExactPolynomial = {}
        for candidate, candidate_polynomial in enumerate(zero_candidates):
            _exact_add_scaled(
                polynomial,
                candidate_polynomial,
                zero_coefficients[candidate][dof],
            )
        zero.append(polynomial)
    one_matrix = [
        [_exact_integral_interval(candidate, exponent) for candidate in one_candidates]
        for exponent in range(order)
    ]
    one_inverse = solve_exact(
        one_matrix,
        [
            [Fraction(int(row == column)) for column in range(order)]
            for row in range(order)
        ],
    )
    one = []
    for dof in range(order):
        polynomial = {}
        for candidate, candidate_polynomial in enumerate(one_candidates):
            _exact_add_scaled(
                polynomial, candidate_polynomial, one_inverse[candidate][dof]
            )
        one.append(polynomial)
    return (
        tuple(tuple(row) for row in zero_coefficients),
        tuple(tuple(row) for row in one_inverse),
        tuple(zero_candidates),
        tuple(one_candidates),
        tuple(zero),
        tuple(one),
    )


@lru_cache(maxsize=128)
def _exact_tensor_coefficients(
    n: int,
    order: int,
    exponents: tuple[tuple[int, ...], ...],
    labels: tuple[DofLabel, ...],
    blades: tuple[tuple[int, ...], ...],
    /,
) -> _ExactCoefficientBank:
    _, _, _, _, zero, one = _exact_tensor_interval_data(order)
    return tuple(
        tuple(
            tuple(
                (
                    prod(
                        (
                            (one if axis in component_blade else zero)[index[axis]].get(
                                (exponent[axis],), _EXACT_ZERO
                            )
                            for axis in range(n)
                        ),
                        start=Fraction(1),
                    )
                    or _EXACT_ZERO
                    if component_blade == label_blade
                    else _EXACT_ZERO
                )
                for _, index, label_blade in labels
            )
            for component_blade in blades
        )
        for exponent in exponents
    )


@lru_cache(maxsize=128)
def _exact_tensor_generators(
    n: int,
    order: int,
    exponents: tuple[tuple[int, ...], ...],
    labels: tuple[DofLabel, ...],
    blades: tuple[tuple[int, ...], ...],
    /,
) -> tuple[_ExactCoefficientBank, tuple[_GeneratorLabel, ...]]:
    _, _, zero_candidates, one_candidates, _, _ = _exact_tensor_interval_data(order)
    coefficients = tuple(
        tuple(
            tuple(
                (
                    prod(
                        (
                            (one_candidates if axis in blade else zero_candidates)[
                                index[axis]
                            ].get((exponent[axis],), _EXACT_ZERO)
                            for axis in range(n)
                        ),
                        start=Fraction(1),
                    )
                    or _EXACT_ZERO
                    if blade == label_blade
                    else _EXACT_ZERO
                )
                for _, index, label_blade in labels
            )
            for blade in blades
        )
        for exponent in exponents
    )
    generator_labels = tuple((index, blade) for _, index, blade in labels)
    return coefficients, generator_labels


def _monomials(points: _HostArray, exponents: tuple[tuple[int, ...], ...]) -> _HostArray:
    powers = np.asarray(exponents, dtype=np.int64).reshape(
        len(exponents), points.shape[1]
    )
    return np.prod(points[:, None, :] ** powers[None, :, :], axis=-1)


def _simplex_quadrature(n: int, degree: int) -> tuple[_HostArray, _HostArray]:
    if n == 0:
        return np.zeros((1, 0), dtype=np.float64), np.ones(1, dtype=np.float64)
    rule = simplex_rule_data(n, degree)
    return np.asarray(rule.points, dtype=np.float64), np.asarray(
        rule.weights, dtype=np.float64
    )


def _wedge_sign(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    return (-1) ** sum(a > b for a in left for b in right)


def _substitute(
    alpha: tuple[int, ...], origin: _HostArray, jacobian: _HostArray
) -> _Polynomial:
    m = jacobian.shape[1]
    result: _Polynomial = {(0,) * m: 1.0}
    for axis, degree in enumerate(alpha):
        affine: _Polynomial = {(0,) * m: float(origin[axis])}
        for coordinate in range(m):
            key = tuple(int(i == coordinate) for i in range(m))
            affine[key] = float(jacobian[axis, coordinate])
        result = _multiply(result, _power(affine, degree, m))
    return result


class _Prepared(NamedTuple):
    exponents: tuple[tuple[int, ...], ...]
    labels: tuple[DofLabel, ...]
    entities: tuple[tuple[tuple[int, ...], ...], ...]
    points: _HostArray
    weights: _HostArray
    nodes: _HostArray


class _EntityMoments(NamedTuple):
    labels: tuple[DofLabel, ...]
    points: _HostArray
    weights: _HostArray
    nodes: _HostArray


class _SimplexTests(NamedTuple):
    exponents: tuple[tuple[int, ...], ...]
    coefficients: _HostArray
    labels: tuple[_GeneratorLabel, ...]
    points: _HostArray
    quadrature: _HostArray
    values: _HostArray


def _finish_prepared(
    exponents: tuple[tuple[int, ...], ...],
    entities: tuple[tuple[tuple[int, ...], ...], ...],
    blocks: list[_EntityMoments],
    component_count: int,
) -> _Prepared:
    labels = tuple(label for block in blocks for label in block.labels)
    total = sum(block.points.shape[0] for block in blocks)
    weights = np.zeros((len(labels), total, component_count), dtype=np.float64)
    row, column = 0, 0
    for block in blocks:
        rows, columns = block.weights.shape[:2]
        weights[row : row + rows, column : column + columns] = block.weights
        row += rows
        column += columns
    return _Prepared(
        exponents,
        labels,
        entities,
        np.concatenate([block.points for block in blocks]),
        weights,
        np.concatenate([block.nodes for block in blocks]),
    )


def _constant_simplex_prepared(
    n: int,
    blades: tuple[tuple[int, ...], ...],
    exponents: tuple[tuple[int, ...], ...],
    entities: tuple[tuple[tuple[int, ...], ...], ...],
) -> _Prepared:
    q, quadrature = _simplex_quadrature(n, 0)
    count = len(blades)
    weights = np.zeros((count, q.shape[0], count), dtype=np.float64)
    for component in range(count):
        weights[component, :, component] = quadrature
    labels = tuple(
        (tuple(range(n + 1)), (0,) * (n + 1), tuple(axis + 1 for axis in blade))
        for blade in blades
    )
    return _Prepared(
        exponents,
        labels,
        entities,
        q,
        weights,
        np.full((count, n), 1.0 / (n + 1), dtype=np.float64),
    )


def _simplex_tests(
    m: int, k: int, r: int, family: FormElementFamily
) -> _SimplexTests | None:
    degree = r + k - m - (1 if family == "trimmed" else 0)
    if degree < 0 or (family == "full" and degree == 0 and m != k):
        return None
    exponents, coefficients, labels = _generators(m, m - k, degree, family == "full")
    if coefficients.shape[-1] == 0:
        return None
    points, quadrature = _simplex_quadrature(m, r + degree)
    values = np.einsum("qm,mcd->qdc", _monomials(points, exponents), coefficients)
    return _SimplexTests(exponents, coefficients, labels, points, quadrature, values)


def _simplex_pairing(
    jacobian: _HostArray, n: int, k: int
) -> tuple[_HostArray, _HostArray]:
    m = jacobian.shape[1]
    blades = tuple(combinations(range(n), k))
    local_blades = tuple(combinations(range(m), k))
    dual_blades = tuple(combinations(range(m), m - k))
    pullback = np.asarray(
        [[_minor(jacobian, blade, local) for blade in blades] for local in local_blades],
        dtype=np.float64,
    )
    wedge = np.asarray(
        [
            [
                _wedge_sign(local, dual) if set(local).isdisjoint(dual) else 0
                for dual in dual_blades
            ]
            for local in local_blades
        ],
        dtype=np.float64,
    )
    return pullback, wedge


def _simplex_entity_moments(
    n: int,
    k: int,
    face: tuple[int, ...],
    vertices: _HostArray,
    tests: _SimplexTests,
) -> _EntityMoments:
    origin = vertices[face[0]]
    jacobian = (vertices[list(face[1:])] - origin).T
    pullback, wedge = _simplex_pairing(jacobian, n, k)
    weights = np.einsum(
        "qdt,lt,lc,q->dqc", tests.values, wedge, pullback, tests.quadrature
    )
    labels = tuple(
        (face, alpha, tuple(face[i] for i in sigma)) for alpha, sigma in tests.labels
    )
    nodes = np.stack([vertices[list(face)].mean(axis=0) for _ in tests.labels])
    return _EntityMoments(
        labels, origin[None, :] + tests.points @ jacobian.T, weights, nodes
    )


def _exact_integral_simplex(polynomial: _ExactPolynomial, dimension: int, /) -> Fraction:
    return sum(
        (
            coefficient
            * prod(factorial(value) for value in exponent)
            / factorial(dimension + sum(exponent))
            for exponent, coefficient in polynomial.items()
        ),
        Fraction(0),
    )


def _exact_substitute_monomial(
    exponent: tuple[int, ...],
    origin: tuple[Fraction, ...],
    jacobian: tuple[tuple[Fraction, ...], ...],
    /,
) -> _ExactPolynomial:
    variables = len(jacobian[0]) if jacobian else 0
    zero = (0,) * variables
    result: _ExactPolynomial = {zero: Fraction(1)}
    for row, degree in enumerate(exponent):
        affine: _ExactPolynomial = {}
        if origin[row]:
            affine[zero] = origin[row]
        for column in range(variables):
            value = jacobian[row][column]
            if value:
                axis = tuple(int(index == column) for index in range(variables))
                affine[axis] = value
        result = _exact_multiply(result, _exact_power(affine, degree, variables))
    return result


def _exact_simplex_pairing(
    jacobian: tuple[tuple[Fraction, ...], ...], n: int, k: int, /
) -> tuple[
    tuple[tuple[Fraction, ...], ...],
    tuple[tuple[Fraction, ...], ...],
]:
    m = len(jacobian[0]) if jacobian else 0
    blades = tuple(combinations(range(n), k))
    local_blades = tuple(combinations(range(m), k))
    dual_blades = tuple(combinations(range(m), m - k))
    pullback = tuple(
        tuple(_exact_minor(jacobian, blade, local) for blade in blades)
        for local in local_blades
    )
    wedge = tuple(
        tuple(
            Fraction(_wedge_sign(local, dual) if set(local).isdisjoint(dual) else 0)
            for dual in dual_blades
        )
        for local in local_blades
    )
    return pullback, wedge


@lru_cache(maxsize=128)
def _exact_simplex_functional_moments(
    n: int,
    k: int,
    order: int,
    family: FormElementFamily,
    exponents: tuple[tuple[int, ...], ...],
    labels: tuple[DofLabel, ...],
    /,
) -> _ExactMomentBank:
    blades = tuple(combinations(range(n), k))
    if family == "full" and order == 0:
        rows = []
        for label in labels:
            component = blades.index(tuple(axis - 1 for axis in label[2]))
            row = [[Fraction(0) for _ in blades] for _ in exponents]
            row[0][component] = Fraction(1, factorial(n))
            rows.append(tuple(tuple(values) for values in row))
        return tuple(rows)
    vertices = (
        (0,) * n,
        *(tuple(int(axis == coordinate) for axis in range(n)) for coordinate in range(n)),
    )
    result = []
    test_cache: dict[
        int,
        tuple[
            tuple[tuple[int, ...], ...],
            _ExactCoefficientBank,
            dict[_GeneratorLabel, int],
        ],
    ] = {}
    for face, alpha, sigma in labels:
        entity_dimension = len(face) - 1
        degree = order + k - entity_dimension - int(family == "trimmed")
        if degree < 0:
            raise ValueError(
                "A declared simplex moment has negative test polynomial degree."
            )
        tests = test_cache.get(entity_dimension)
        if tests is None:
            test_exponents, test_coefficients, test_labels = _exact_generators(
                entity_dimension,
                entity_dimension - k,
                degree,
                family == "full",
            )
            tests = (
                test_exponents,
                test_coefficients,
                dict(zip(test_labels, range(len(test_labels)), strict=True)),
            )
            test_cache[entity_dimension] = tests
        test_exponents, test_coefficients, test_positions = tests
        local_sigma = tuple(face.index(vertex) for vertex in sigma)
        test_position = test_positions.get((alpha, local_sigma))
        if test_position is None:
            raise ValueError("A simplex DOF has no exact complementary test generator.")
        test_polynomials = tuple(
            {
                exponent: test_coefficients[monomial][component][test_position]
                for monomial, exponent in enumerate(test_exponents)
                if test_coefficients[monomial][component][test_position]
            }
            for component in range(len(test_coefficients[0]))
        )
        origin = tuple(Fraction(value) for value in vertices[face[0]])
        jacobian = tuple(
            tuple(
                Fraction(vertices[face[column + 1]][row] - vertices[face[0]][row])
                for column in range(entity_dimension)
            )
            for row in range(n)
        )
        pullback, wedge = _exact_simplex_pairing(jacobian, n, k)
        row: list[list[Fraction]] = [[Fraction(0) for _ in blades] for _ in exponents]
        for monomial, exponent in enumerate(exponents):
            pulled = _exact_substitute_monomial(exponent, origin, jacobian)
            for dual_component, test_polynomial in enumerate(test_polynomials):
                integral = _exact_integral_simplex(
                    _exact_multiply(pulled, test_polynomial), entity_dimension
                )
                if not integral:
                    continue
                for local_component in range(len(pullback)):
                    sign = wedge[local_component][dual_component]
                    if not sign:
                        continue
                    for component, factor in enumerate(pullback[local_component]):
                        row[monomial][component] += integral * sign * factor
        result.append(tuple(tuple(values) for values in row))
    return tuple(result)


@lru_cache(maxsize=128)
def _exact_tensor_functional_moments(
    n: int,
    k: int,
    exponents: tuple[tuple[int, ...], ...],
    labels: tuple[DofLabel, ...],
    /,
) -> _ExactMomentBank:
    blades = tuple(combinations(range(n), k))
    blade_positions = {blade: position for position, blade in enumerate(blades)}
    vertices = _tensor_vertices(n)
    result = []
    for face, index, blade in labels:
        free = tuple(
            axis
            for axis in range(n)
            if len({vertices[vertex][axis] for vertex in face}) > 1
        )
        origin = tuple(
            Fraction(0) if axis in free else Fraction(vertices[face[0]][axis])
            for axis in range(n)
        )
        alpha = tuple(index[axis] - int(axis not in blade) for axis in free)
        component = blade_positions[blade]
        row = []
        for exponent in exponents:
            integral = Fraction(1)
            for axis in range(n):
                integral *= (
                    Fraction(1, exponent[axis] + alpha[free.index(axis)] + 1)
                    if axis in free
                    else origin[axis] ** exponent[axis]
                )
            row.append(
                tuple(
                    (integral or _EXACT_ZERO) if column == component else _EXACT_ZERO
                    for column in range(len(blades))
                )
            )
        result.append(tuple(row))
    return tuple(result)


def _prepare_simplex(n: int, k: int, r: int, family: FormElementFamily) -> _Prepared:
    exponents = _exponents(n, max(r, 0))
    blades = tuple(combinations(range(n), k))
    entities = tuple(tuple(combinations(range(n + 1), m + 1)) for m in range(n + 1))
    vertices = np.concatenate((np.zeros((1, n)), np.eye(n)), axis=0)
    if family == "full" and r == 0:
        return _constant_simplex_prepared(n, blades, exponents, entities)
    blocks: list[_EntityMoments] = []
    for m in range(k, n + 1):
        tests = _simplex_tests(m, k, r, family)
        if tests is None:
            continue
        for face in entities[m]:
            blocks.append(_simplex_entity_moments(n, k, face, vertices, tests))
    return _finish_prepared(exponents, entities, blocks, len(blades))


def _tensor_vertices(n: int) -> tuple[tuple[int, ...], ...]:
    if n == 1:
        return ((0,), (1,))
    # Named quadrilateral/hexahedron ordering agrees with the mesh substrate.
    if n == 2:
        return ((0, 0), (1, 0), (1, 1), (0, 1))
    if n == 3:
        return tuple((*v, z) for z in range(2) for v in _tensor_vertices(2))
    return tuple(product(range(2), repeat=n))


def _form_orientation_generators(
    dimension: int, family: FormElementFamily, /
) -> tuple[tuple[int, ...], ...]:
    if family != "tensor-trimmed":
        result = []
        for axis in range(dimension):
            permutation = list(range(dimension + 1))
            permutation[axis], permutation[axis + 1] = (
                permutation[axis + 1],
                permutation[axis],
            )
            result.append(tuple(permutation))
        return tuple(result)
    vertices = _tensor_vertices(dimension)
    positions = {vertex: index for index, vertex in enumerate(vertices)}
    result = []
    for axis in range(dimension):
        result.append(
            tuple(
                positions[
                    tuple(
                        1 - value if coordinate == axis else value
                        for coordinate, value in enumerate(vertex)
                    )
                ]
                for vertex in vertices
            )
        )
    for axis in range(dimension - 1):
        result.append(
            tuple(
                positions[
                    tuple(
                        vertex[axis + 1]
                        if coordinate == axis
                        else vertex[axis]
                        if coordinate == axis + 1
                        else value
                        for coordinate, value in enumerate(vertex)
                    )
                ]
                for vertex in vertices
            )
        )
    return tuple(result)


def _cube_face(
    vertices: tuple[tuple[int, ...], ...], states: tuple[int, ...]
) -> tuple[int, ...]:
    return tuple(
        index
        for index, vertex in enumerate(vertices)
        if all(
            state == -1 or state == coordinate
            for state, coordinate in zip(states, vertex, strict=True)
        )
    )


def _tensor_entities(n: int) -> tuple[tuple[tuple[int, ...], ...], ...]:
    vertices = _tensor_vertices(n)
    faces: list[list[tuple[int, ...]]] = [[] for _ in range(n + 1)]
    for states in product((-1, 0, 1), repeat=n):
        faces[states.count(-1)].append(_cube_face(vertices, states))
    return tuple(tuple(sorted(level)) for level in faces)


class _TensorInterval(NamedTuple):
    zero: _HostArray
    one: _HostArray


@host_table_cache(max_entries=256, max_bytes=1048576)
def _tensor_interval(r: int) -> _TensorInterval:
    """Project the exact one-dimensional interval duals for execution."""
    zero_exact, one_exact, _, _, _, _ = _exact_tensor_interval_data(r)
    zero = np.asarray(zero_exact, dtype=np.float64)
    one = np.asarray(one_exact, dtype=np.float64)
    zero.setflags(write=False)
    one.setflags(write=False)
    return _TensorInterval(zero, one)


@host_table_cache(max_entries=256, max_bytes=16777216)
def _tensor_derivative(
    r: int, source: tuple[DofLabel, ...], target: tuple[DofLabel, ...]
) -> _HostArray:
    """Tensor Stokes map: endpoint traces minus monomial test derivatives."""
    positions = {(blade, index): dof for dof, (_, index, blade) in enumerate(target)}
    result = np.zeros((len(target), len(source)), dtype=np.float64)
    for column, (_, index, blade) in enumerate(source):
        for axis, mode in enumerate(index):
            if axis in blade:
                continue
            output_blade = tuple(sorted((*blade, axis)))
            sign = (-1) ** sum(component < axis for component in blade)
            rows = (
                ((0, -1),)
                if mode == 0
                else tuple((alpha, 1) for alpha in range(r))
                if mode == r
                else ((mode, -mode),)
            )
            for alpha, coefficient in rows:
                output_index = (*index[:axis], alpha, *index[axis + 1 :])
                row = positions[(output_blade, output_index)]
                result[row, column] = sign * coefficient
    result.setflags(write=False)
    return result


def _tensor_quadrature(m: int, r: int) -> tuple[_HostArray, _HostArray]:
    if m == 0:
        return np.zeros((1, 0), dtype=np.float64), np.ones(1, dtype=np.float64)
    rule = tensor_product_rule_data(m, r, family="gauss")
    return np.asarray(rule.points, dtype=np.float64), np.asarray(
        rule.weights, dtype=np.float64
    )


def _tensor_entity_chart(
    n: int, face: tuple[int, ...]
) -> tuple[tuple[int, ...], _HostArray, _HostArray]:
    vertices = _tensor_vertices(n)
    free = tuple(axis for axis in range(n) if len({vertices[v][axis] for v in face}) > 1)
    origin = np.asarray(vertices[face[0]], dtype=np.float64)
    origin[list(free)] = 0.0
    return free, origin, np.eye(n, dtype=np.float64)[:, list(free)]


def _tensor_test_index(
    n: int,
    r: int,
    alpha: tuple[int, ...],
    free: tuple[int, ...],
    blade: tuple[int, ...],
    origin: _HostArray,
) -> tuple[int, ...]:
    return tuple(
        alpha[free.index(axis)] + int(axis not in blade)
        if axis in free
        else r * int(origin[axis])
        for axis in range(n)
    )


def _tensor_entity_moments(
    n: int,
    r: int,
    face: tuple[int, ...],
    blades: tuple[tuple[int, ...], ...],
    q: _HostArray,
    quadrature: _HostArray,
) -> _EntityMoments | None:
    free, origin, _ = _tensor_entity_chart(n, face)
    points = np.broadcast_to(origin, (q.shape[0], n)).copy()
    points[:, list(free)] = q
    labels: list[DofLabel] = []
    nodes: list[_HostArray] = []
    blocks: list[_HostArray] = []
    for component, blade in enumerate(blades):
        if not set(blade).issubset(free):
            continue
        modes = tuple(range(r) if axis in blade else range(r - 1) for axis in free)
        for alpha in product(*modes):
            index = _tensor_test_index(n, r, alpha, free, blade, origin)
            labels.append((face, index, blade))
            nodes.append(points.mean(axis=0))
            density = np.prod(q ** np.asarray(alpha, dtype=np.int64)[None, :], axis=-1)
            block = np.zeros((q.shape[0], len(blades)), dtype=np.float64)
            block[:, component] = quadrature * density
            blocks.append(block)
    if not blocks:
        return None
    return _EntityMoments(tuple(labels), points, np.stack(blocks), np.stack(nodes))


def _prepare_tensor(n: int, k: int, r: int) -> _Prepared:
    blades = tuple(combinations(range(n), k))
    exponents = tuple(product(range(r + 1), repeat=n))
    entities = _tensor_entities(n)
    blocks: list[_EntityMoments] = []
    for m, level in enumerate(entities):
        if m < k:
            continue
        q, quadrature = _tensor_quadrature(m, r)
        for face in level:
            block = _tensor_entity_moments(n, r, face, blades, q, quadrature)
            if block is not None:
                blocks.append(block)
    return _finish_prepared(exponents, entities, blocks, len(blades))


def _entity_dof_indices(
    labels: tuple[DofLabel, ...], face: tuple[int, ...]
) -> tuple[int, ...]:
    return tuple(i for i, label in enumerate(labels) if label[0] == face)


def _interior_dof_indices(
    labels: tuple[DofLabel, ...], vertex_count: int
) -> tuple[int, ...]:
    return tuple(i for i, label in enumerate(labels) if len(label[0]) == vertex_count)


def _canonical_entity_ids(
    family: FormElementFamily,
    n: int,
    m: int,
    face: tuple[int, ...],
    global_vertices: tuple[int, ...],
) -> tuple[int, ...]:
    ids = tuple(global_vertices[i] for i in face)
    if family != "tensor-trimmed":
        return ids
    coordinates = _tensor_vertices(n)
    free, _, _ = _tensor_entity_chart(n, face)
    local_coordinates = tuple(tuple(coordinates[v][axis] for axis in free) for v in face)
    return tuple(ids[local_coordinates.index(vertex)] for vertex in _tensor_vertices(m))


def _form_reference_vertices(n: int, family: FormElementFamily) -> _HostArray:
    if family == "tensor-trimmed":
        return np.asarray(_tensor_vertices(n), dtype=np.float64)
    return np.concatenate((np.zeros((1, n)), np.eye(n)), axis=0)


def _vertex_permutation_affine(
    reference: _HostArray,
    vertices: tuple[int, ...],
) -> tuple[_HostArray, _HostArray]:
    if tuple(sorted(vertices)) != tuple(range(reference.shape[0])):
        raise ValueError("vertices must be a permutation of reference vertex indices.")
    n = reference.shape[1]
    images = reference[list(vertices)]
    origin = images[0]
    coordinates = tuple(tuple(vertex) for vertex in reference)
    neighbors = tuple(
        coordinates.index(tuple(int(axis == i) for i in range(n))) for axis in range(n)
    )
    jacobian = (images[list(neighbors)] - origin).T
    if not np.array_equal(reference @ jacobian.T + origin, images):
        raise ValueError("Tensor vertex permutation must be an affine cube symmetry.")
    return origin, jacobian


def _form_pullback_matrix(
    n: int, k: int, jacobian: _HostArray, twist: FormTwist
) -> _HostArray:
    blades = tuple(combinations(range(n), k))
    matrix = np.asarray(
        [[_minor(jacobian, source, target) for source in blades] for target in blades],
        dtype=np.float64,
    )
    if twist == "twisted":
        matrix *= np.sign(np.linalg.det(jacobian))
    return matrix


def _exact_vertex_permutation_affine(
    n: int,
    family: FormElementFamily,
    vertices: tuple[int, ...],
    /,
) -> tuple[tuple[Fraction, ...], _ExactMatrix]:
    reference = (
        _tensor_vertices(n)
        if family == "tensor-trimmed"
        else (
            (0,) * n,
            *(
                tuple(int(axis == coordinate) for axis in range(n))
                for coordinate in range(n)
            ),
        )
    )
    if tuple(sorted(vertices)) != tuple(range(len(reference))):
        raise ValueError("vertices must be a permutation of reference vertex indices.")
    images = tuple(reference[index] for index in vertices)
    origin = tuple(Fraction(value) for value in images[0])
    coordinate_positions = {vertex: index for index, vertex in enumerate(reference)}
    neighbors = tuple(
        coordinate_positions[tuple(int(axis == column) for axis in range(n))]
        for column in range(n)
    )
    jacobian: _ExactMatrix = tuple(
        tuple(
            Fraction(images[neighbors[column]][row] - images[0][row])
            for column in range(n)
        )
        for row in range(n)
    )
    if any(
        tuple(
            origin[row]
            + sum(
                (
                    jacobian[row][column] * reference[vertex][column]
                    for column in range(n)
                ),
                Fraction(0),
            )
            for row in range(n)
        )
        != tuple(Fraction(value) for value in images[vertex])
        for vertex in range(len(reference))
    ):
        raise ValueError("Tensor vertex permutation must be an affine cube symmetry.")
    return origin, jacobian


def _exact_form_pullback_matrix(
    n: int,
    k: int,
    jacobian: _ExactMatrix,
    twist: FormTwist,
    /,
) -> _ExactMatrix:
    blades = tuple(combinations(range(n), k))
    matrix = tuple(
        tuple(_exact_minor(jacobian, source, target) for source in blades)
        for target in blades
    )
    if twist != "twisted":
        return matrix
    determinant = Fraction(determinant_exact(jacobian))
    if not determinant:
        raise ValueError("Form permutation chart must be nonsingular.")
    sign = Fraction(1 if determinant > 0 else -1)
    return tuple(tuple(sign * value for value in row) for row in matrix)


type _IntegerMomentRows = tuple[tuple[int, tuple[tuple[int, int], ...]], ...]
# Per-DOF scales and, per ``component * dof_count + dof``, nonzero integer
# coefficients ``(monomial, numerator)`` over that DOF's scale.
type _IntegerCoefficientColumns = tuple[
    tuple[int, ...], tuple[tuple[tuple[int, int], ...], ...]
]
# Per-column scales and, per ``monomial * component_count + component``,
# nonzero integer entries ``(column, numerator)`` over that column's scale.
type _IntegerTermColumns = tuple[tuple[int, ...], tuple[tuple[tuple[int, int], ...], ...]]


def _integer_coefficient_columns(
    coefficients: _ExactCoefficientBank, /
) -> _IntegerCoefficientColumns:
    """Each DOF's basis coefficients as integers over their denominator LCM."""
    component_count = len(coefficients[0])
    dof_count = len(coefficients[0][0])
    terms: list[list[tuple[int, Fraction]]] = [
        [] for _ in range(component_count * dof_count)
    ]
    scales = [1] * dof_count
    for monomial, components in enumerate(coefficients):
        for component, dofs in enumerate(components):
            offset = component * dof_count
            for dof, value in enumerate(dofs):
                if value:
                    terms[offset + dof].append((monomial, value))
                    scales[dof] = lcm(scales[dof], value.denominator)
    return tuple(scales), tuple(
        tuple(
            (monomial, value.numerator * (scales[index % dof_count] // value.denominator))
            for monomial, value in entries
        )
        for index, entries in enumerate(terms)
    )


def _integer_rows(rows: Iterable[Sequence[Fraction]], /) -> _IntegerMomentRows:
    """Nonzero entries of each row as integers over the row's denominator LCM."""
    result = []
    for row in rows:
        terms = tuple((index, value) for index, value in enumerate(row) if value)
        scale = lcm(1, *(value.denominator for _, value in terms))
        result.append(
            (
                scale,
                tuple(
                    (index, value.numerator * (scale // value.denominator))
                    for index, value in terms
                ),
            )
        )
    return tuple(result)


def _integer_moment_rows(functional_moments: _ExactMomentBank, /) -> _IntegerMomentRows:
    """Moment rows with terms flattened as ``monomial * component_count + component``."""
    return _integer_rows(
        tuple(value for components in moments for value in components)
        for moments in functional_moments
    )


def _integer_term_columns(
    rows: Sequence[Sequence[Fraction]], column_count: int, /
) -> _IntegerTermColumns:
    """Rows' nonzero entries as integers over each column's denominator LCM."""
    scales = [1] * column_count
    nonzero = []
    for row in rows:
        entries = tuple((column, value) for column, value in enumerate(row) if value)
        for column, value in entries:
            scales[column] = lcm(scales[column], value.denominator)
        nonzero.append(entries)
    return tuple(scales), tuple(
        tuple(
            (column, value.numerator * (scales[column] // value.denominator))
            for column, value in entries
        )
        for entries in nonzero
    )


def _substituted_term_columns(
    exponents: tuple[tuple[int, ...], ...],
    columns: _IntegerCoefficientColumns,
    origin: tuple[Fraction, ...],
    jacobian: _ExactMatrix,
    pullback: _ExactMatrix,
    /,
) -> _IntegerTermColumns:
    """Exact pullback of every basis polynomial under ``x -> origin + J x``.

    Each canonical monomial is substituted once and kept as an integer row over
    its denominator LCM. Each DOF then accumulates integers over one common
    scale: its coefficient scale times the pullback and substitution LCMs.
    """
    monomial_count = len(exponents)
    component_count = len(pullback)
    dof_scales, source = columns
    dof_count = len(dof_scales)
    positions = {exponent: index for index, exponent in enumerate(exponents)}
    images: list[tuple[int, tuple[tuple[int, int], ...]]] = []
    for exponent in exponents:
        image = _exact_substitute_monomial(exponent, origin, jacobian)
        scale = lcm(1, *(value.denominator for value in image.values()))
        # Images leaving the canonical space get indices past it; they must cancel.
        images.append(
            (
                scale,
                tuple(
                    (
                        positions.setdefault(target, len(positions)),
                        value.numerator * (scale // value.denominator),
                    )
                    for target, value in image.items()
                ),
            )
        )
    width = len(positions)
    image_scale = lcm(1, *(scale for scale, _ in images))
    image_factors = tuple((image_scale // scale, entries) for scale, entries in images)
    pullback_scale = lcm(1, *(value.denominator for row in pullback for value in row))
    mixing = tuple(
        tuple(
            (
                source_component * dof_count,
                value.numerator * (pullback_scale // value.denominator),
            )
            for source_component, value in enumerate(pullback[target])
            if value
        )
        for target in range(component_count)
    )
    term_columns: list[list[tuple[int, int]]] = [
        [] for _ in range(monomial_count * component_count)
    ]
    for dof in range(dof_count):
        for target, sources in enumerate(mixing):
            accumulated = [0] * width
            for offset, weight in sources:
                for monomial, coefficient in source[offset + dof]:
                    factor, entries = image_factors[monomial]
                    factor *= weight * coefficient
                    for index, entry in entries:
                        accumulated[index] += factor * entry
            for index, total in enumerate(accumulated):
                if not total:
                    continue
                if index >= monomial_count:
                    raise ValueError(
                        "Exact form permutation left the canonical monomial space."
                    )
                term_columns[index * component_count + target].append((dof, total))
    common = pullback_scale * image_scale
    return tuple(common * scale for scale in dof_scales), tuple(
        tuple(entries) for entries in term_columns
    )


def _exact_moment_product(
    moment_rows: _IntegerMomentRows, columns: _IntegerTermColumns, /
) -> _ExactMatrix:
    """Exact ``moments[row] . transformed[:, :, column]`` over nonzero terms.

    Rows and columns are integers over their scales, so each entry is one
    integer accumulation whose normalized quotient equals the rational sum.
    """
    column_scales, term_columns = columns
    dof_count = len(column_scales)
    rows: list[tuple[Fraction, ...]] = []
    for row_scale, row_terms in moment_rows:
        accumulated = [0] * dof_count
        for index, weight in row_terms:
            for column, entry in term_columns[index]:
                accumulated[column] += weight * entry
        rows.append(
            tuple(
                Fraction(total, row_scale * column_scales[column])
                if total
                else _EXACT_ZERO
                for column, total in enumerate(accumulated)
            )
        )
    return tuple(rows)


def _exact_moment_permutation(
    dimension: int,
    form_degree: int,
    family: FormElementFamily,
    twist: FormTwist,
    exponents: tuple[tuple[int, ...], ...],
    columns: _IntegerCoefficientColumns,
    moment_rows: _IntegerMomentRows,
    vertices: tuple[int, ...],
    /,
) -> _ExactMatrix:
    origin, jacobian = _exact_vertex_permutation_affine(dimension, family, vertices)
    pullback = _exact_form_pullback_matrix(dimension, form_degree, jacobian, twist)
    return _exact_moment_product(
        moment_rows,
        _substituted_term_columns(exponents, columns, origin, jacobian, pullback),
    )


@lru_cache(maxsize=128)
def _exact_form_permutation_matrix(
    dimension: int,
    form_degree: int,
    family: FormElementFamily,
    twist: FormTwist,
    exponents: tuple[tuple[int, ...], ...],
    coefficients: _ExactCoefficientBank,
    functional_moments: _ExactMomentBank,
    vertices: tuple[int, ...],
    /,
) -> _ExactMatrix:
    return _exact_moment_permutation(
        dimension,
        form_degree,
        family,
        twist,
        exponents,
        _integer_coefficient_columns(coefficients),
        _integer_moment_rows(functional_moments),
        vertices,
    )


@lru_cache(maxsize=128)
def _exact_form_orientation_actions(
    dimension: int,
    form_degree: int,
    family: FormElementFamily,
    twist: FormTwist,
    exponents: tuple[tuple[int, ...], ...],
    coefficients: _ExactCoefficientBank,
    functional_moments: _ExactMomentBank,
    vertices: tuple[tuple[int, ...], ...],
    /,
) -> tuple[_ExactMatrix, ...]:
    columns = _integer_coefficient_columns(coefficients)
    moment_rows = _integer_moment_rows(functional_moments)
    return tuple(
        _exact_moment_permutation(
            dimension,
            form_degree,
            family,
            twist,
            exponents,
            columns,
            moment_rows,
            permutation,
        )
        for permutation in vertices
    )


def _substitute_coefficients(
    exponents: tuple[tuple[int, ...], ...],
    coefficients: _HostArray,
    origin: _HostArray,
    jacobian: _HostArray,
) -> _HostArray:
    transformed = np.zeros_like(coefficients)
    lookup = {alpha: index for index, alpha in enumerate(exponents)}
    for monomial, alpha in enumerate(exponents):
        for exponent, factor in _substitute(alpha, origin, jacobian).items():
            if factor != 0.0:
                transformed[lookup[exponent]] += factor * coefficients[monomial]
    return transformed


def _blade_derivative_terms(
    n: int, k: int, axis: int
) -> tuple[tuple[int, int, int], ...]:
    target_blades = tuple(combinations(range(n), k + 1))
    terms: list[tuple[int, int, int]] = []
    for component, blade in enumerate(combinations(range(n), k)):
        if axis not in blade:
            out = tuple(sorted((axis, *blade)))
            sign = (-1) ** sum(i < axis for i in blade)
            terms.append((component, target_blades.index(out), sign))
    return tuple(terms)


def _polynomial_derivative_terms(
    exponents: tuple[tuple[int, ...], ...],
    target_exponents: tuple[tuple[int, ...], ...],
    n: int,
    k: int,
) -> tuple[tuple[int, int, int, int, int], ...]:
    lookup = {alpha: index for index, alpha in enumerate(target_exponents)}
    terms: list[tuple[int, int, int, int, int]] = []
    for index, alpha in enumerate(exponents):
        for axis, degree in enumerate(alpha):
            if degree == 0:
                continue
            lower = tuple(value - int(i == axis) for i, value in enumerate(alpha))
            if lower not in lookup:
                raise ValueError(
                    "Target polynomial degree cannot represent the exterior derivative."
                )
            for component, output, sign in _blade_derivative_terms(n, k, axis):
                terms.append((index, component, lookup[lower], output, sign * degree))
    return tuple(terms)


def _form_entity_chart(
    n: int,
    family: FormElementFamily,
    face: tuple[int, ...],
) -> tuple[tuple[int, ...], _HostArray, _HostArray]:
    if family == "tensor-trimmed":
        return _tensor_entity_chart(n, face)
    vertices = _form_reference_vertices(n, family)
    return (), vertices[face[0]], (vertices[list(face[1:])] - vertices[face[0]]).T


def _tensor_density_weights(
    labels: tuple[DofLabel, ...],
    indices: tuple[int, ...],
    free: tuple[int, ...],
    blades: tuple[tuple[int, ...], ...],
    points: Array,
    quadrature: Array,
) -> Array:
    weights = jnp.zeros((len(indices), points.shape[0], len(blades)), dtype=points.dtype)
    for output, row in enumerate(indices):
        _, index, blade = labels[row]
        alpha = tuple(index[axis] - int(axis not in blade) for axis in free)
        density = jnp.prod(
            points ** jnp.asarray(alpha, dtype=jnp.int32)[None, :], axis=-1
        )
        weights = weights.at[output, :, blades.index(blade)].set(quadrature * density)
    return weights


def _constant_density_weights(count: int, points: Array, quadrature: Array) -> Array:
    weights = jnp.zeros((count, points.shape[0], count), dtype=points.dtype)
    for component in range(count):
        weights = weights.at[component, :, component].set(quadrature)
    return weights


def _simplex_density_weights(
    n: int,
    k: int,
    r: int,
    family: FormElementFamily,
    points: Array,
    quadrature: Array,
    jacobian: _HostArray,
) -> Array:
    m = points.shape[1]
    degree = r + k - m - int(family == "trimmed")
    exponents, tests, _ = _generators(m, m - k, degree, family == "full")
    powers = jnp.asarray(exponents, dtype=jnp.int32).reshape(len(exponents), m)
    monomials = jnp.prod(points[:, None, :] ** powers[None, :, :], axis=-1)
    test_values = contract("qm,mct->qtc", monomials, jnp.asarray(tests))
    pullback, wedge = _simplex_pairing(jacobian, n, k)
    return contract(
        "qdt,lt,lc,q->dqc",
        test_values,
        jnp.asarray(wedge),
        jnp.asarray(pullback),
        quadrature,
    )


@final
class _TensorFactors(StrictModule):
    """Interval modal duals and canonical product/component routes."""

    order: int = eqx.field(static=True)
    zero_projection_errors: _ExactMatrix = eqx.field(static=True)
    one_projection_errors: _ExactMatrix = eqx.field(static=True)
    zero: Array
    one: Array
    zero_derivatives: Array
    one_derivatives: Array
    indices: Array
    differential_axes: Array
    components: Array

    def __init__(self, n: int, k: int, r: int, labels: tuple[DofLabel, ...]) -> None:
        interval = _tensor_interval(r)
        zero_exact, one_exact, _, _, _, _ = _exact_tensor_interval_data(r)
        blades = tuple(combinations(range(n), k))
        self.order = r
        self.zero_projection_errors = tuple(
            tuple(abs(Fraction(float(value)) - value) for value in row)
            for row in zero_exact
        )
        self.one_projection_errors = tuple(
            tuple(abs(Fraction(float(value)) - value) for value in row)
            for row in one_exact
        )
        self.zero = jnp.asarray(interval.zero)
        self.one = jnp.asarray(interval.one)
        self.zero_derivatives = standard_series_derivative_coefficients(
            "legendre", self.zero[2:], scale=2.0
        )
        self.one_derivatives = standard_series_derivative_coefficients(
            "legendre", self.one, scale=2.0
        )
        self.indices = jnp.asarray(
            tuple(tuple(label[1][axis] for label in labels) for axis in range(n)),
            dtype=jnp.int32,
        )
        self.differential_axes = jnp.asarray(
            tuple(tuple(axis in label[2] for label in labels) for axis in range(n)),
            dtype=jnp.bool_,
        )
        self.components = jnp.asarray(
            tuple(
                tuple(float(label[2] == blade) for blade in blades) for label in labels
            ),
            dtype=jnp.float64,
        )

    def tabulate(self, points: Array, /) -> tuple[Array, Array]:
        r = self.order
        modal = standard_vandermonde(
            "legendre", 2.0 * points.reshape(-1) - 1.0, r - 1
        ).reshape((*points.shape, r))
        bubble = points * (1.0 - points)
        zero = (
            (1.0 - points)[..., None] * self.zero[0]
            + points[..., None] * self.zero[1]
            + bubble[..., None] * (modal[..., : r - 1] @ self.zero[2:])
        )
        zero_derivative = (
            self.zero[1]
            - self.zero[0]
            + (1.0 - 2.0 * points)[..., None] * (modal[..., : r - 1] @ self.zero[2:])
            + bubble[..., None] * (modal[..., : r - 1] @ self.zero_derivatives)
        )
        one = jnp.pad(modal @ self.one, ((0, 0), (0, 0), (0, 1)))
        one_derivative = jnp.pad(modal @ self.one_derivatives, ((0, 0), (0, 0), (0, 1)))
        indices = self.indices[None, :, :]
        factors = jnp.where(
            self.differential_axes[None, :, :],
            jnp.take_along_axis(one, indices, axis=-1),
            jnp.take_along_axis(zero, indices, axis=-1),
        )
        differentiated = jnp.where(
            self.differential_axes[None, :, :],
            jnp.take_along_axis(one_derivative, indices, axis=-1),
            jnp.take_along_axis(zero_derivative, indices, axis=-1),
        )
        values = contract("qb,bc->qbc", jnp.prod(factors, axis=1), self.components)
        derivatives = tuple(
            contract(
                "qb,bc->qbc",
                jnp.prod(
                    jnp.concatenate(
                        (
                            factors[:, :axis],
                            differentiated[:, axis : axis + 1],
                            factors[:, axis + 1 :],
                        ),
                        axis=1,
                    ),
                    axis=1,
                ),
                self.components,
            )
            for axis in range(points.shape[1])
        )
        return values, jnp.stack(derivatives, axis=-1)


@final
class FormBasis(StrictModule):
    """A moment-dual polynomial basis and its complete interpolation functionals.

    Exact Fraction banks define the source basis; coefficients and functional
    moments are binary64 execution projections with their conversion errors
    retained separately.
    """

    dimension: int = eqx.field(static=True)
    form_degree: int = eqx.field(static=True)
    order: int = eqx.field(static=True)
    family: FormElementFamily = eqx.field(static=True)
    twist: FormTwist = eqx.field(static=True)
    exponents: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    dof_labels: tuple[DofLabel, ...] = eqx.field(static=True)
    entity_vertices: tuple[tuple[tuple[int, ...], ...], ...] = eqx.field(static=True)
    exact_generators: _ExactCoefficientBank = eqx.field(static=True)
    exact_generator_labels: tuple[_GeneratorLabel, ...] = eqx.field(static=True)
    exact_coefficients: _ExactCoefficientBank = eqx.field(static=True)
    exact_functional_moments: _ExactMomentBank = eqx.field(static=True)
    coefficient_projection_errors: _ExactCoefficientBank = eqx.field(static=True)
    functional_moment_projection_errors: _ExactMomentBank = eqx.field(static=True)
    coefficients: Array
    tensor_factors: _TensorFactors | None
    functional_points: Array
    functional_weights: Array
    functional_moments: Array
    reference_nodes: Array
    basis_id: str = eqx.field(static=True)

    def __init__(
        self,
        dimension: int,
        form_degree: int,
        order: int,
        family: FormElementFamily,
        twist: FormTwist,
        /,
    ) -> None:
        family_ = parse(family, FormElementFamily, "family")
        twist_ = parse(twist, FormTwist, "twist")
        if dimension < 1 or not 0 <= form_degree <= dimension:
            raise ValueError("Require dimension >= 1 and 0 <= form_degree <= dimension.")
        if order < (0 if family_ == "full" else 1):
            raise ValueError(
                "Full forms require order >= 0; trimmed forms require order >= 1."
            )
        prepared = (
            _prepare_tensor(dimension, form_degree, order)
            if family_ == "tensor-trimmed"
            else _prepare_simplex(dimension, form_degree, order, family_)
        )
        blades = tuple(combinations(range(dimension), form_degree))
        if family_ == "tensor-trimmed":
            exact_coefficients = _exact_tensor_coefficients(
                dimension,
                order,
                prepared.exponents,
                prepared.labels,
                blades,
            )
            exact_generators, exact_generator_labels = _exact_tensor_generators(
                dimension,
                order,
                prepared.exponents,
                prepared.labels,
                blades,
            )
            exact_functional_moments = _exact_tensor_functional_moments(
                dimension,
                form_degree,
                prepared.exponents,
                prepared.labels,
            )
            tensor_factors = _TensorFactors(
                dimension, form_degree, order, prepared.labels
            )
        else:
            generator_exponents, exact_generators, exact_generator_labels = (
                _exact_generators(
                    dimension,
                    form_degree,
                    order,
                    family_ == "trimmed",
                )
            )
            if generator_exponents != prepared.exponents:
                raise ValueError(
                    "Exact generator monomials differ from execution preparation."
                )
            exact_functional_moments = _exact_simplex_functional_moments(
                dimension,
                form_degree,
                order,
                family_,
                prepared.exponents,
                prepared.labels,
            )
            exact_coefficients = _exact_dual_coefficients(
                exact_generators, exact_functional_moments
            )
            tensor_factors = None
        if (
            len(exact_generators) != len(prepared.exponents)
            or len(exact_generator_labels) != len(prepared.labels)
            or any(len(row) != len(blades) for row in exact_generators)
            or any(
                len(row) != len(exact_generator_labels)
                for monomial in exact_generators
                for row in monomial
            )
        ):
            raise ValueError("Exact form generator bank has inconsistent extents.")
        if (
            len(exact_functional_moments) != len(prepared.labels)
            or any(
                len(row) != len(prepared.exponents)
                or any(len(components) != len(blades) for components in row)
                for row in exact_functional_moments
            )
            or len(exact_coefficients) != len(prepared.exponents)
            or any(len(row) != len(blades) for row in exact_coefficients)
            or any(
                len(row) != len(prepared.labels)
                for monomial in exact_coefficients
                for row in monomial
            )
        ):
            raise ValueError("Exact form basis banks have inconsistent extents.")
        coefficients = np.asarray(exact_coefficients, dtype=np.float64)
        functional_moments = np.asarray(exact_functional_moments, dtype=np.float64)
        if not np.all(np.isfinite(coefficients)) or not np.all(
            np.isfinite(functional_moments)
        ):
            raise ValueError("Exact form basis has no finite binary64 projection.")
        # Banks share Fraction objects and repeat few values; project each value
        # once and remember it per object (the banks keep every object alive).
        projection_errors: dict[tuple[int, int], Fraction] = {}
        object_errors: dict[int, Fraction] = {}

        def projection_error(value: Fraction, /) -> Fraction:
            error = object_errors.get(id(value))
            if error is None:
                key = (value.numerator, value.denominator)
                error = projection_errors.get(key)
                if error is None:
                    error = projection_errors[key] = abs(Fraction(float(value)) - value)
                object_errors[id(value)] = error
            return error

        coefficient_projection_errors = tuple(
            tuple(tuple(projection_error(value) for value in dofs) for dofs in components)
            for components in exact_coefficients
        )
        functional_moment_projection_errors = tuple(
            tuple(
                tuple(projection_error(value) for value in components)
                for components in monomials
            )
            for monomials in exact_functional_moments
        )
        self.dimension = dimension
        self.form_degree = form_degree
        self.order = order
        self.family = family_
        self.twist = twist_
        self.exponents = prepared.exponents
        self.dof_labels = prepared.labels
        self.entity_vertices = prepared.entities
        self.exact_generators = exact_generators
        self.exact_generator_labels = exact_generator_labels
        self.exact_coefficients = exact_coefficients
        self.exact_functional_moments = exact_functional_moments
        self.coefficient_projection_errors = coefficient_projection_errors
        self.functional_moment_projection_errors = functional_moment_projection_errors
        self.coefficients = jnp.asarray(coefficients)
        self.tensor_factors = tensor_factors
        self.functional_points = jnp.asarray(prepared.points)
        self.functional_weights = jnp.asarray(prepared.weights)
        self.functional_moments = jnp.asarray(functional_moments)
        self.reference_nodes = jnp.asarray(prepared.nodes)
        orientation_generators = _form_orientation_generators(dimension, family_)
        orientation_actions = _exact_form_orientation_actions(
            dimension,
            form_degree,
            family_,
            twist_,
            prepared.exponents,
            exact_coefficients,
            exact_functional_moments,
            orientation_generators,
        )
        self.basis_id = canonical_fingerprint(
            {
                "kind": "form-basis",
                "dimension": dimension,
                "degree": form_degree,
                "order": order,
                "family": family_,
                "twist": twist_,
                "exponents": prepared.exponents,
                "labels": prepared.labels,
                "exact_generator_labels": exact_generator_labels,
                "exact_generators": _exact_data_digest(exact_generators),
                "exact_coefficients": _exact_data_digest(exact_coefficients),
                "exact_functional_moments": _exact_data_digest(exact_functional_moments),
                "orientation_generators": orientation_generators,
                "orientation_actions": _exact_data_digest(orientation_actions),
            }
        )

    @property
    def local_dof_count(self) -> int:
        return self.coefficients.shape[-1]

    def tabulate_components(self, points: ArrayLike, /) -> tuple[Array, Array]:
        points_ = jnp.asarray(points, dtype=self.coefficients.dtype)
        if points_.ndim != 2 or points_.shape[1] != self.dimension:
            raise ValueError("Points must have shape (point_count, dimension).")
        if self.tensor_factors is not None:
            return self.tensor_factors.tabulate(points_)
        powers = jnp.asarray(self.exponents, dtype=jnp.int32)
        # Static integer powers preserve polynomial derivatives at coordinate zeros.
        axis_powers = tuple(
            jnp.stack(
                tuple(
                    jax.lax.integer_pow(points_[:, axis], exponent)
                    for exponent in range(max(row[axis] for row in self.exponents) + 1)
                ),
                axis=-1,
            )
            for axis in range(self.dimension)
        )
        monomials = jnp.prod(
            jnp.stack(
                tuple(
                    axis_powers[axis][:, powers[:, axis]]
                    for axis in range(self.dimension)
                ),
                axis=-1,
            ),
            axis=-1,
        )
        values = contract("qm,mcb->qbc", monomials, self.coefficients)
        derivatives = []
        for axis in range(self.dimension):
            lower = powers.at[:, axis].add(-1)
            lower = jnp.maximum(lower, 0)
            derivative = jnp.prod(
                jnp.stack(
                    tuple(
                        axis_powers[column][:, lower[:, column]]
                        for column in range(self.dimension)
                    ),
                    axis=-1,
                ),
                axis=-1,
            ) * powers[None, :, axis].astype(points_.dtype)
            derivatives.append(contract("qm,mcb->qbc", derivative, self.coefficients))
        return values, jnp.stack(derivatives, axis=-1)

    def interpolate(self, values: ArrayLike, /) -> Array:
        values_ = jnp.asarray(values)
        expected = (
            self.functional_points.shape[0],
            comb(self.dimension, self.form_degree),
        )
        if values_.shape[:2] != expected:
            raise ValueError(
                "Interpolation samples must have shape (functional_points, components, ...)."
            )
        return contract("dqc,qc...->d...", self.functional_weights, values_)

    def functional_weights_at(
        self,
        entity_vertices: tuple[int, ...],
        entity_points: ArrayLike,
        quadrature_weights: ArrayLike,
        /,
    ) -> tuple[Array, Array]:
        """Evaluate entity moment densities on caller-provided cubature.

        The vertex tuple must use ``entity_vertices``' declared ordering.
        Simplex coordinates are the ordered barycentric affine chart; tensor
        coordinates are the free reference axes in increasing order. Cubature
        weights integrate this chart, including any caller's subcell Jacobian.
        Rows belonging to other entities are identically zero.
        """
        if not any(entity_vertices in level for level in self.entity_vertices):
            raise ValueError("Use a declared reference entity and its vertex ordering.")
        points = jnp.asarray(entity_points, dtype=self.coefficients.dtype)
        quadrature = jnp.asarray(quadrature_weights, dtype=self.coefficients.dtype)
        free, origin, jacobian = _form_entity_chart(
            self.dimension, self.family, entity_vertices
        )
        if (
            points.ndim != 2
            or points.shape[1] != jacobian.shape[1]
            or quadrature.shape != (points.shape[0],)
        ):
            raise ValueError(
                "Entity cubature requires points(q,entity_dimension) and weights(q)."
            )
        reference_points = jnp.asarray(origin)[None, :] + points @ jnp.asarray(jacobian.T)
        blades = tuple(combinations(range(self.dimension), self.form_degree))
        indices = _entity_dof_indices(self.dof_labels, entity_vertices)
        result = jnp.zeros(
            (self.local_dof_count, points.shape[0], len(blades)), dtype=points.dtype
        )
        if not indices:
            return reference_points, result
        if self.family == "tensor-trimmed":
            block = _tensor_density_weights(
                self.dof_labels, indices, free, blades, points, quadrature
            )
        elif self.family == "full" and self.order == 0:
            block = _constant_density_weights(len(indices), points, quadrature)
        else:
            block = _simplex_density_weights(
                self.dimension,
                self.form_degree,
                self.order,
                self.family,
                points,
                quadrature,
                jacobian,
            )
        return reference_points, result.at[jnp.asarray(indices)].set(block)

    def _require_exterior_derivative_target(self, target: FormBasis, /) -> None:
        if (target.dimension, target.form_degree, target.twist) != (
            self.dimension,
            self.form_degree + 1,
            self.twist,
        ):
            raise ValueError(
                "Exterior derivative target must have next degree, same dimension and twist."
            )
        minimum_order = self.order - int(target.family == "full")
        if (target.family == "tensor-trimmed") != (self.family == "tensor-trimmed"):
            raise ValueError("Tensor and simplicial derivative spaces cannot be mixed.")
        if target.order < minimum_order:
            raise ValueError(
                "Target family/order does not contain the exterior derivative."
            )

    def exterior_derivative_matrix(self, target: FormBasis, /) -> Array:
        self._require_exterior_derivative_target(target)
        if self.tensor_factors is not None:
            if target.order == self.order:
                return jnp.asarray(
                    _tensor_derivative(self.order, self.dof_labels, target.dof_labels),
                    dtype=self.coefficients.dtype,
                )
            _, gradient = self.tabulate_components(target.functional_points)
            exterior = jnp.zeros(
                (
                    gradient.shape[0],
                    self.local_dof_count,
                    comb(self.dimension, self.form_degree + 1),
                ),
                dtype=gradient.dtype,
            )
            for axis in range(self.dimension):
                for component, output, sign in _blade_derivative_terms(
                    self.dimension, self.form_degree, axis
                ):
                    exterior = exterior.at[:, :, output].add(
                        sign * gradient[:, :, component, axis]
                    )
            return contract("dqc,qbc->db", target.functional_weights, exterior)
        derivatives = jnp.zeros(
            (
                len(target.exponents),
                comb(self.dimension, self.form_degree + 1),
                self.local_dof_count,
            ),
            dtype=self.coefficients.dtype,
        )
        terms = _polynomial_derivative_terms(
            self.exponents, target.exponents, self.dimension, self.form_degree
        )
        for index, component, lower, output, factor in terms:
            derivatives = derivatives.at[lower, output].add(
                factor * self.coefficients[index, component]
            )
        result = contract("dmc,mcb->db", target.functional_moments, derivatives)
        return result

    def exact_exterior_derivative_matrix(self, target: FormBasis, /) -> _ExactMatrix:
        """Return the exact local moment action of the exterior derivative."""
        self._require_exterior_derivative_target(target)
        target_component_count = comb(self.dimension, self.form_degree + 1)
        dof_count = self.local_dof_count
        coefficients = self.exact_coefficients
        derivatives = [
            [
                [Fraction(0) for _ in range(dof_count)]
                for _ in range(target_component_count)
            ]
            for _ in target.exponents
        ]
        for index, component, lower, output, factor in _polynomial_derivative_terms(
            self.exponents, target.exponents, self.dimension, self.form_degree
        ):
            row = derivatives[lower][output]
            source = coefficients[index][component]
            for dof in range(dof_count):
                row[dof] += factor * source[dof]
        return matmul_exact(
            [
                [value for components in moments for value in components]
                for moments in target.exact_functional_moments
            ],
            [row for components in derivatives for row in components],
        )

    def permutation_matrix(self, vertices: tuple[int, ...], /) -> Array:
        """Return local moments of pullback under v_i -> v_vertices[i].

        Local coefficients = matrix @ canonical coefficients. Twisted forms
        include the orientation-character multiplier. Tensor permutations must
        describe an affine cube symmetry, not an arbitrary vertex relabeling.
        """
        reference = _form_reference_vertices(self.dimension, self.family)
        origin, jacobian = _vertex_permutation_affine(reference, vertices)
        pullback = _form_pullback_matrix(
            self.dimension, self.form_degree, jacobian, self.twist
        )
        if self.tensor_factors is not None:
            points = self.functional_points @ jnp.asarray(jacobian.T) + jnp.asarray(
                origin
            )
            values, _ = self.tabulate_components(points)
            transformed_values = contract("ce,qbe->qbc", jnp.asarray(pullback), values)
            return contract("dqc,qbc->db", self.functional_weights, transformed_values)
        original = np.einsum("ce,meb->mcb", pullback, np.asarray(self.coefficients))
        transformed = _substitute_coefficients(self.exponents, original, origin, jacobian)
        return jnp.asarray(
            np.einsum("dmc,mcb->db", np.asarray(self.functional_moments), transformed)
        )

    def exact_permutation_matrix(self, vertices: tuple[int, ...], /) -> _ExactMatrix:
        """Return the exact pullback action on local entity moments."""
        return _exact_form_permutation_matrix(
            self.dimension,
            self.form_degree,
            self.family,
            self.twist,
            self.exponents,
            self.exact_coefficients,
            self.exact_functional_moments,
            vertices,
        )

    def canonical_permutation(
        self, global_vertices: tuple[int, ...], /
    ) -> tuple[int, ...]:
        """Map local vertices to a deterministic, geometrically legal chart."""
        if self.family != "tensor-trimmed":
            ordered = tuple(sorted(global_vertices))
            return tuple(ordered.index(vertex) for vertex in global_vertices)
        return _canonical_cube_permutation(
            _tensor_vertices(self.dimension), global_vertices
        )

    def entity_permutation_matrix(self, global_vertices: tuple[int, ...], /) -> Array:
        """Map independently canonical entity moments to local element moments.

        Columns retain local entity-block positions; each block's columns are
        interpreted in its globally canonical entity chart. Unlike a whole-cell
        permutation, this assembly transform does not move entity blocks.
        """
        expected = (
            2**self.dimension if self.family == "tensor-trimmed" else self.dimension + 1
        )
        if len(global_vertices) != expected or len(set(global_vertices)) != expected:
            raise ValueError("Provide one distinct global identifier per local vertex.")
        result = jnp.zeros(
            (self.local_dof_count, self.local_dof_count), dtype=self.coefficients.dtype
        )
        for m, level in enumerate(self.entity_vertices):
            for face in level:
                indices = _entity_dof_indices(self.dof_labels, face)
                if indices:
                    block = self._entity_moment_transform(
                        m, face, len(indices), global_vertices
                    )
                    result = result.at[
                        jnp.asarray(indices)[:, None], jnp.asarray(indices)[None, :]
                    ].set(block)
        return result

    def exact_entity_permutation_matrix(
        self, global_vertices: tuple[int, ...], /
    ) -> _ExactMatrix:
        """Return the exact local-to-global entity moment action."""
        expected = (
            2**self.dimension if self.family == "tensor-trimmed" else self.dimension + 1
        )
        if len(global_vertices) != expected or len(set(global_vertices)) != expected:
            raise ValueError("Provide one distinct global identifier per local vertex.")
        result = [
            [Fraction(0) for _ in range(self.local_dof_count)]
            for _ in range(self.local_dof_count)
        ]
        for m, level in enumerate(self.entity_vertices):
            for face in level:
                indices = _entity_dof_indices(self.dof_labels, face)
                if not indices:
                    continue
                block = self._exact_entity_moment_transform(
                    m, face, len(indices), global_vertices
                )
                for row, index in enumerate(indices):
                    for column, source in enumerate(indices):
                        result[index][source] = block[row][column]
        return tuple(tuple(row) for row in result)

    def _entity_moment_transform(
        self,
        m: int,
        face: tuple[int, ...],
        count: int,
        global_vertices: tuple[int, ...],
    ) -> Array:
        if m == 0 or (self.family == "full" and self.order == 0):
            return jnp.eye(count, dtype=self.coefficients.dtype)
        entity = FormBasis(m, self.form_degree, self.order, self.family, "untwisted")
        vertex_count = 2**m if self.family == "tensor-trimmed" else m + 1
        interior = _interior_dof_indices(entity.dof_labels, vertex_count)
        if len(interior) != count:
            raise ValueError("Entity moment block dimensions disagree.")
        ids = _canonical_entity_ids(self.family, self.dimension, m, face, global_vertices)
        full = entity.permutation_matrix(entity.canonical_permutation(ids))
        return full[jnp.asarray(interior)[:, None], jnp.asarray(interior)[None, :]]

    def _exact_entity_moment_transform(
        self,
        m: int,
        face: tuple[int, ...],
        count: int,
        global_vertices: tuple[int, ...],
    ) -> _ExactMatrix:
        if m == 0 or (self.family == "full" and self.order == 0):
            return tuple(
                tuple(Fraction(int(row == column)) for column in range(count))
                for row in range(count)
            )
        entity = FormBasis(m, self.form_degree, self.order, self.family, "untwisted")
        vertex_count = 2**m if self.family == "tensor-trimmed" else m + 1
        interior = _interior_dof_indices(entity.dof_labels, vertex_count)
        if len(interior) != count:
            raise ValueError("Entity moment block dimensions disagree.")
        ids = _canonical_entity_ids(self.family, self.dimension, m, face, global_vertices)
        full = entity.exact_permutation_matrix(entity.canonical_permutation(ids))
        return tuple(tuple(full[row][column] for column in interior) for row in interior)


def _canonical_cube_permutation(
    coordinates: tuple[tuple[int, ...], ...], global_vertices: tuple[int, ...]
) -> tuple[int, ...]:
    n = len(coordinates[0])
    if len(coordinates) != len(global_vertices) or len(set(global_vertices)) != len(
        global_vertices
    ):
        raise ValueError("Cube chart requires distinct global vertices.")
    origin_index = min(range(len(global_vertices)), key=global_vertices.__getitem__)
    origin = coordinates[origin_index]
    neighbors = tuple(
        index
        for index, vertex in enumerate(coordinates)
        if sum(a != b for a, b in zip(origin, vertex, strict=True)) == 1
    )
    ordered = tuple(sorted(neighbors, key=global_vertices.__getitem__))
    axes = tuple(
        next(axis for axis in range(n) if coordinates[index][axis] != origin[axis])
        for index in ordered
    )
    reference = _tensor_vertices(n)
    return tuple(
        reference.index(
            tuple(
                vertex[axis] if origin[axis] == 0 else 1 - vertex[axis] for axis in axes
            )
        )
        for vertex in coordinates
    )


@final
class _ProxyTabulator(StrictModule):
    basis: FormBasis
    value_spec: FormValueSpec = eqx.field(static=True)

    def __init__(self, basis: FormBasis, value_spec: FormValueSpec, /) -> None:
        self.basis = basis
        self.value_spec = value_spec

    def __call__(self, points: Array, /) -> tuple[Array, Array]:
        from ...exterior._algebra import form_to_vector

        values, gradients = self.basis.tabulate_components(points)
        converted = form_to_vector(values, self.value_spec)
        gradient_values = tuple(
            form_to_vector(gradients[..., axis], self.value_spec)
            for axis in range(self.basis.dimension)
        )
        return converted, jnp.stack(gradient_values, axis=-1)


def _admit_form_cell(
    cell_kind: str, family: FormElementFamily
) -> tuple[int, FormElementFamily]:
    dimension = reference_cell_topology(cell_kind).dimension
    family_ = parse(family, FormElementFamily, "family")
    tensor = cell_kind in ("quadrilateral", "hexahedron") or cell_kind.startswith(
        "tensor:"
    )
    simplex = cell_kind in (
        "interval",
        "triangle",
        "tetrahedron",
    ) or cell_kind.startswith("simplex:")
    if not simplex and not tensor:
        raise ValueError(
            "Form elements require a simplex or tensor-product reference cell."
        )
    if tensor and family_ != "tensor-trimmed":
        raise ValueError("Tensor cells require family='tensor-trimmed'.")
    if simplex and cell_kind != "interval" and family_ == "tensor-trimmed":
        raise ValueError("Tensor-trimmed forms require a tensor cell or interval.")
    return dimension, family_


def _default_form_proxy(n: int, k: int) -> FormProxy:
    if n == 1 or (n == 2 and k == 1):
        raise ValueError("This form degree requires an explicit proxy.")
    if k == 0:
        return "scalar"
    if k == n:
        return "density"
    if k == 1:
        return "circulation"
    if k == n - 1:
        return "flux"
    return "components"


def _form_value_spec(
    n: int, k: int, twist: FormTwist | None, proxy: FormProxy | None
) -> FormValueSpec:
    proxy_ = parse(
        _default_form_proxy(n, k) if proxy is None else proxy, FormProxy, "proxy"
    )
    if twist is None and proxy_ in ("flux", "density"):
        raise ValueError("Physical flux and density elements require explicit twist.")
    twist_ = "untwisted" if twist is None else parse(twist, FormTwist, "twist")
    return FormValueSpec(FormType(n, k, twist=twist_), proxy=proxy_)


def _form_entity_dofs(basis: FormBasis) -> tuple[tuple[tuple[int, ...], ...], ...]:
    return tuple(
        tuple(_entity_dof_indices(basis.dof_labels, face) for face in level)
        for level in basis.entity_vertices
    )


def form_element(
    cell_kind: str,
    form_degree: int,
    order: int,
    /,
    *,
    family: FormElementFamily = "trimmed",
    twist: FormTwist | None = None,
    proxy: FormProxy | None = None,
) -> FiniteElementSpec:
    """Construct all-order P_r^- Lambda^k, P_r Lambda^k or Q_r^- Lambda^k.

    ``order`` is the FEEC polynomial index (lowest trimmed order is one).
    Flux and density proxies require explicit twist. In ambiguous degrees
    (one-dimensional cells and degree one in two dimensions), proxy is required.
    """
    n, family_ = _admit_form_cell(cell_kind, family)
    value_spec = _form_value_spec(n, form_degree, twist, proxy)
    basis = FormBasis(n, form_degree, order, family_, value_spec.form_type.twist)
    return FiniteElementSpec(
        family_,
        cell_kind,
        order,
        basis.reference_nodes,
        _form_entity_dofs(basis),
        value_spec=value_spec,
        continuity="discontinuous" if form_degree == n or order == 0 else "conforming",
        representation="polynomial_moment",
        tabulator=_ProxyTabulator(basis, value_spec),
        tabulator_id=basis.basis_id,
        form_basis=basis,
    )


__all__ = ["DofLabel", "FormBasis", "FormElementFamily", "form_element"]
