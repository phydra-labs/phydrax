# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Compatible polynomial, product, and rational differential-form elements.

Simplex/tensor cells admit their canonical n-D references; named prism and
pyramid cells own product and rational complexes with exact entity moments.
All component axes follow the exterior owner's lexicographic blade convention.
"""

from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
from itertools import combinations, product
from math import comb, factorial, prod
from typing import final, Literal, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._polynomial._cubature import simplex_rule_data, tensor_product_rule_data
from ..._polynomial._orthogonal import (
    standard_series_derivative_coefficients,
    standard_vandermonde,
)
from ..._strict import StrictModule
from ...ein import contract
from ...exterior._form_type import FormProxy, FormTwist, FormType, FormValueSpec
from ...typing import parse
from .._coordinate_enclosure import Expression
from .._reference_cell import reference_cell_topology
from ._reference import FiniteElementSpec


type FormElementFamily = Literal[
    "trimmed", "full", "tensor-trimmed", "prism-trimmed", "pyramid-trimmed"
]
type DofLabel = tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]
type _Polynomial = dict[tuple[int, ...], float]
type _HostArray = npt.NDArray[np.float64]


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


def _barycentric(n: int) -> tuple[_Polynomial, ...]:
    zero = (0,) * n
    axes = tuple(tuple(int(i == j) for i in range(n)) for j in range(n))
    return ({zero: 1.0, **{axis: -1.0 for axis in axes}}, *({axis: 1.0} for axis in axes))


def _barycentric_product(alpha: tuple[int, ...], n: int) -> _Polynomial:
    result: _Polynomial = {(0,) * n: 1.0}
    for exponent, coordinate in zip(alpha, _barycentric(n), strict=True):
        result = _multiply(result, _power(coordinate, exponent, n))
    return result


def _minor(matrix: _HostArray, rows: tuple[int, ...], columns: tuple[int, ...]) -> float:
    if not rows:
        return 1.0
    return float(np.linalg.det(matrix[np.ix_(rows, columns)]))


type _GeneratorLabel = tuple[tuple[int, ...], tuple[int, ...]]


def _full_generators(
    n: int,
    r: int,
    exponents: tuple[tuple[int, ...], ...],
    blades: tuple[tuple[int, ...], ...],
) -> tuple[list[_HostArray], list[_GeneratorLabel]]:
    positions = {alpha: index for index, alpha in enumerate(exponents)}
    generators: list[_HostArray] = []
    labels: list[_GeneratorLabel] = []
    for alpha in _compositions(r, n + 1):
        scalar = _barycentric_product(alpha, n)
        for component, blade in enumerate(blades):
            coeff = np.zeros((len(exponents), len(blades)), dtype=np.float64)
            for exponent, value in scalar.items():
                coeff[positions[exponent], component] = value
            generators.append(coeff)
            labels.append((alpha, tuple(axis + 1 for axis in blade)))
    return generators, labels


def _whitney_generator(
    n: int,
    k: int,
    alpha: tuple[int, ...],
    sigma: tuple[int, ...],
    exponents: tuple[tuple[int, ...], ...],
    blades: tuple[tuple[int, ...], ...],
    gradients: _HostArray,
) -> _HostArray:
    positions = {exponent: index for index, exponent in enumerate(exponents)}
    coeff = np.zeros((len(exponents), len(blades)), dtype=np.float64)
    scalar = _barycentric_product(alpha, n)
    for position, vertex in enumerate(sigma):
        other = tuple(v for v in sigma if v != vertex)
        polynomial = _multiply(scalar, _barycentric(n)[vertex])
        for component, blade in enumerate(blades):
            factor = factorial(k) * (-1.0) ** position * _minor(gradients, other, blade)
            for exponent, value in polynomial.items():
                coeff[positions[exponent], component] += factor * value
    return coeff


def _trimmed_generators(
    n: int,
    k: int,
    r: int,
    exponents: tuple[tuple[int, ...], ...],
    blades: tuple[tuple[int, ...], ...],
) -> tuple[list[_HostArray], list[_GeneratorLabel]]:
    gradients = np.concatenate((-np.ones((1, n)), np.eye(n)), axis=0)
    generators: list[_HostArray] = []
    labels: list[_GeneratorLabel] = []
    for sigma in combinations(range(n + 1), k + 1):
        for alpha in _compositions(r - 1, n + 1):
            if any(alpha[i] for i in range(sigma[0])):
                continue
            generators.append(
                _whitney_generator(n, k, alpha, sigma, exponents, blades, gradients)
            )
            labels.append((alpha, sigma))
    return generators, labels


def _generators(
    n: int, k: int, r: int, trimmed: bool
) -> tuple[tuple[tuple[int, ...], ...], _HostArray, tuple[_GeneratorLabel, ...]]:
    """Independent Bernstein--Whitney generators, or full polynomial generators."""
    exponents = _exponents(n, max(r, 0))
    blades = tuple(combinations(range(n), k))
    if not trimmed or k == 0:
        generators, labels = _full_generators(n, r, exponents, blades)
    else:
        generators, labels = _trimmed_generators(n, k, r, exponents, blades)
    coefficients = (
        np.stack(generators, axis=-1)
        if generators
        else np.zeros((len(exponents), len(blades), 0), dtype=np.float64)
    )
    return exponents, coefficients, tuple(labels)


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


def _integral(poly: _Polynomial, n: int) -> float:
    return sum(
        value * prod(factorial(a) for a in alpha) / factorial(n + sum(alpha))
        for alpha, value in poly.items()
    )


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
    generators: _HostArray
    labels: tuple[DofLabel, ...]
    entities: tuple[tuple[tuple[int, ...], ...], ...]
    points: _HostArray
    weights: _HostArray
    moments: _HostArray
    nodes: _HostArray


class _EntityMoments(NamedTuple):
    labels: tuple[DofLabel, ...]
    points: _HostArray
    weights: _HostArray
    moments: _HostArray
    nodes: _HostArray


class _SimplexTests(NamedTuple):
    exponents: tuple[tuple[int, ...], ...]
    coefficients: _HostArray
    labels: tuple[_GeneratorLabel, ...]
    points: _HostArray
    quadrature: _HostArray
    values: _HostArray


def _immutable_prepared(prepared: _Prepared, /) -> _Prepared:
    """Freeze cached host reference arrays before publishing the signature."""
    for value in prepared:
        if isinstance(value, np.ndarray):
            value.setflags(write=False)
    return prepared


def _finish_prepared(
    exponents: tuple[tuple[int, ...], ...],
    generators: _HostArray,
    entities: tuple[tuple[tuple[int, ...], ...], ...],
    blocks: list[_EntityMoments],
) -> _Prepared:
    labels = tuple(label for block in blocks for label in block.labels)
    total = sum(block.points.shape[0] for block in blocks)
    weights = np.zeros((len(labels), total, generators.shape[1]), dtype=np.float64)
    row, column = 0, 0
    for block in blocks:
        rows, columns = block.weights.shape[:2]
        weights[row : row + rows, column : column + columns] = block.weights
        row += rows
        column += columns
    return _immutable_prepared(
        _Prepared(
            exponents,
            generators,
            labels,
            entities,
            np.concatenate([block.points for block in blocks]),
            weights,
            np.concatenate([block.moments for block in blocks]),
            np.concatenate([block.nodes for block in blocks]),
        )
    )


def _constant_simplex_prepared(
    n: int,
    blades: tuple[tuple[int, ...], ...],
    exponents: tuple[tuple[int, ...], ...],
    generators: _HostArray,
    entities: tuple[tuple[tuple[int, ...], ...], ...],
) -> _Prepared:
    q, quadrature = _simplex_quadrature(n, 0)
    count = len(blades)
    weights = np.zeros((count, q.shape[0], count), dtype=np.float64)
    moments = np.zeros((count, 1, count), dtype=np.float64)
    for component in range(count):
        weights[component, :, component] = quadrature
        moments[component, 0, component] = 1.0 / factorial(n)
    labels = tuple(
        (tuple(range(n + 1)), (0,) * (n + 1), tuple(axis + 1 for axis in blade))
        for blade in blades
    )
    return _immutable_prepared(
        _Prepared(
            exponents,
            generators,
            labels,
            entities,
            q,
            weights,
            moments,
            np.full((count, n), 1.0 / (n + 1), dtype=np.float64),
        )
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


def _simplex_test_polynomials(tests: _SimplexTests, test: int) -> tuple[_Polynomial, ...]:
    return tuple(
        {
            a: float(tests.coefficients[index, component, test])
            for index, a in enumerate(tests.exponents)
            if tests.coefficients[index, component, test] != 0.0
        }
        for component in range(tests.coefficients.shape[1])
    )


def _simplex_moment_row(
    exponents: tuple[tuple[int, ...], ...],
    test_polynomials: tuple[_Polynomial, ...],
    origin: _HostArray,
    jacobian: _HostArray,
    pullback: _HostArray,
    wedge: _HostArray,
) -> _HostArray:
    row = np.zeros((len(exponents), pullback.shape[1]), dtype=np.float64)
    for monomial, exponent in enumerate(exponents):
        pulled = _substitute(exponent, origin, jacobian)
        for dual_component, test_poly in enumerate(test_polynomials):
            integral = _integral(_multiply(pulled, test_poly), jacobian.shape[1])
            for local_component in range(pullback.shape[0]):
                sign = wedge[local_component, dual_component]
                if sign != 0:
                    row[monomial] += integral * sign * pullback[local_component]
    return row


def _simplex_entity_moments(
    n: int,
    k: int,
    face: tuple[int, ...],
    vertices: _HostArray,
    exponents: tuple[tuple[int, ...], ...],
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
    moments = np.stack(
        [
            _simplex_moment_row(
                exponents,
                _simplex_test_polynomials(tests, test),
                origin,
                jacobian,
                pullback,
                wedge,
            )
            for test in range(len(tests.labels))
        ]
    )
    nodes = np.stack([vertices[list(face)].mean(axis=0) for _ in tests.labels])
    return _EntityMoments(
        labels, origin[None, :] + tests.points @ jacobian.T, weights, moments, nodes
    )


@lru_cache(maxsize=64)
def _prepare_simplex(n: int, k: int, r: int, family: FormElementFamily) -> _Prepared:
    exponents, generators, _ = _generators(n, k, r, family == "trimmed")
    blades = tuple(combinations(range(n), k))
    entities = tuple(tuple(combinations(range(n + 1), m + 1)) for m in range(n + 1))
    vertices = np.concatenate((np.zeros((1, n)), np.eye(n)), axis=0)
    if family == "full" and r == 0:
        return _constant_simplex_prepared(n, blades, exponents, generators, entities)
    blocks: list[_EntityMoments] = []
    for m in range(k, n + 1):
        tests = _simplex_tests(m, k, r, family)
        if tests is None:
            continue
        for face in entities[m]:
            blocks.append(_simplex_entity_moments(n, k, face, vertices, exponents, tests))
    return _finish_prepared(exponents, generators, entities, blocks)


def _tensor_vertices(n: int) -> tuple[tuple[int, ...], ...]:
    if n == 1:
        return ((0,), (1,))
    # Named quadrilateral/hexahedron ordering agrees with the mesh substrate.
    if n == 2:
        return ((0, 0), (1, 0), (1, 1), (0, 1))
    if n == 3:
        return tuple((*v, z) for z in range(2) for v in _tensor_vertices(2))
    return tuple(product(range(2), repeat=n))


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
    zero_monomials: _HostArray
    one_monomials: _HostArray


def _solve_reference_moments(moments: _HostArray, rhs: _HostArray, /) -> _HostArray:
    """Solve one immutable host reference dual without compiler startup."""
    if (
        moments.ndim != 2
        or moments.shape[0] != moments.shape[1]
        or rhs.ndim != 2
        or rhs.shape[0] != moments.shape[0]
    ):
        raise ValueError(
            "Reference moment solve requires a square matrix and aligned RHS."
        )
    spectrum = np.linalg.svd(moments, compute_uv=False)
    if (
        not spectrum.size
        or not np.all(np.isfinite(spectrum))
        or spectrum[-1] <= np.finfo(np.float64).eps * spectrum[0] * moments.shape[0]
    ):
        raise ValueError("Reference moments are not numerically unisolvent.")
    result = np.linalg.solve(moments, rhs)
    if not np.all(np.isfinite(result)):
        raise ValueError("Reference moment solve produced non-finite coefficients.")
    return result


@lru_cache(maxsize=32)
def _tensor_interval(r: int) -> _TensorInterval:
    """Admit reusable interval duals, never a global tensor Vandermonde."""
    q, quadrature = _tensor_quadrature(1, r + 1)
    x = q[:, 0]
    legendre = np.polynomial.legendre.legvander(2.0 * x - 1.0, r - 1)
    zero_values = np.column_stack(
        (1.0 - x, x, x[:, None] * (1.0 - x[:, None]) * legendre[:, : r - 1])
    )
    zero = np.zeros((r + 1, r + 1), dtype=np.float64)
    zero[0, 0] = zero[1, r] = 1.0
    if r > 1:
        interior_moments = np.stack(
            tuple((quadrature * x**alpha) @ zero_values for alpha in range(r - 1))
        )
        rhs = np.zeros((r - 1, r + 1), dtype=np.float64)
        rhs[:, 0] = -interior_moments[:, 0]
        rhs[:, r] = -interior_moments[:, 1]
        rhs[:, 1:r] = np.eye(r - 1, dtype=np.float64)
        # Endpoint traces stay exact: only the bubble block needs a solve.
        zero[2:] = _solve_reference_moments(interior_moments[:, 2:], rhs)
    one_moments = np.stack(
        tuple((quadrature * x**alpha) @ legendre for alpha in range(r))
    )
    one = _solve_reference_moments(one_moments, np.eye(r, dtype=np.float64))
    zero.setflags(write=False)
    one.setflags(write=False)
    zero_generators = np.zeros((r + 1, r + 1), dtype=np.float64)
    zero_generators[:2, :2] = np.asarray(((1.0, 0.0), (-1.0, 1.0)))
    one_generators = np.zeros((r + 1, r), dtype=np.float64)
    bubble = np.polynomial.Polynomial((0.0, 1.0, -1.0))
    for mode in range(r):
        polynomial = np.polynomial.Legendre.basis(mode, domain=(0.0, 1.0)).convert(
            kind=np.polynomial.Polynomial
        )
        one_generators[: polynomial.coef.size, mode] = polynomial.coef
        if mode < r - 1:
            bubble_polynomial = bubble * polynomial
            zero_generators[: bubble_polynomial.coef.size, mode + 2] = (
                bubble_polynomial.coef
            )
    zero_monomials, one_monomials = (
        zero_generators @ zero,
        one_generators @ one,
    )
    zero_monomials.setflags(write=False)
    one_monomials.setflags(write=False)
    return _TensorInterval(zero, one, zero_monomials, one_monomials)


@lru_cache(maxsize=32)
def _tensor_coefficients(
    n: int,
    r: int,
    exponents: tuple[tuple[int, ...], ...],
    labels: tuple[DofLabel, ...],
    blades: tuple[tuple[int, ...], ...],
) -> _HostArray:
    interval = _tensor_interval(r)
    coefficients = np.zeros((len(exponents), len(blades), len(labels)), dtype=np.float64)
    powers = np.asarray(exponents, dtype=np.int64)
    for dof, (_, index, blade) in enumerate(labels):
        factors = tuple(
            (interval.one_monomials if axis in blade else interval.zero_monomials)[
                powers[:, axis], index[axis]
            ]
            for axis in range(n)
        )
        coefficients[:, blades.index(blade), dof] = np.prod(np.stack(factors), axis=0)
    coefficients.setflags(write=False)
    return coefficients


@lru_cache(maxsize=32)
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


def _tensor_moment_row(
    exponents: tuple[tuple[int, ...], ...],
    alpha: tuple[int, ...],
    free: tuple[int, ...],
    origin: _HostArray,
    component: int,
    component_count: int,
) -> _HostArray:
    row = np.zeros((len(exponents), component_count), dtype=np.float64)
    for monomial, exponent in enumerate(exponents):
        integral = 1.0
        for axis, degree in enumerate(exponent):
            integral *= (
                1.0 / (degree + alpha[free.index(axis)] + 1)
                if axis in free
                else origin[axis] ** degree
            )
        row[monomial, component] = integral
    return row


def _tensor_entity_moments(
    n: int,
    r: int,
    face: tuple[int, ...],
    exponents: tuple[tuple[int, ...], ...],
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
    rows: list[_HostArray] = []
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
            rows.append(
                _tensor_moment_row(exponents, alpha, free, origin, component, len(blades))
            )
    if not blocks:
        return None
    return _EntityMoments(
        tuple(labels), points, np.stack(blocks), np.stack(rows), np.stack(nodes)
    )


@lru_cache(maxsize=32)
def _prepare_tensor(n: int, k: int, r: int) -> _Prepared:
    blades = tuple(combinations(range(n), k))
    exponents = tuple(product(range(r + 1), repeat=n))
    entities = _tensor_entities(n)
    # Only the component extent is used while assembling the moment functionals.
    generators = np.zeros((len(exponents), len(blades), 0), dtype=np.float64)
    blocks: list[_EntityMoments] = []
    for m, level in enumerate(entities):
        if m < k:
            continue
        q, quadrature = _tensor_quadrature(m, r)
        for face in level:
            block = _tensor_entity_moments(n, r, face, exponents, blades, q, quadrature)
            if block is not None:
                blocks.append(block)
    return _finish_prepared(exponents, generators, entities, blocks)


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
    if family in ("prism-trimmed", "pyramid-trimmed"):
        from ._hybrid_forms import hybrid_kind

        return np.asarray(
            reference_cell_topology(hybrid_kind(family)).vertices, dtype=np.float64
        )
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
    axes = tuple(tuple(int(axis == i) for i in range(n)) for axis in range(n))
    if all(axis in coordinates for axis in axes):
        neighbors = tuple(coordinates.index(axis) for axis in axes)
        jacobian = (images[list(neighbors)] - origin).T
    else:
        from .._coordinate_enclosure import _solve_exact

        candidates = tuple(combinations(range(1, len(reference)), n))
        neighbors = next(
            (
                indices
                for indices in candidates
                if _minor(
                    (reference[list(indices)] - reference[0]).T,
                    tuple(range(n)),
                    tuple(range(n)),
                )
                != 0
            ),
            None,
        )
        if neighbors is None:
            raise ValueError("Reference vertices do not span an affine cell chart.")
        source = [
            [
                Fraction(float(reference[index, axis] - reference[0, axis]))
                for axis in range(n)
            ]
            for index in neighbors
        ]
        rhs = [
            [Fraction(float(images[index, axis] - images[0, axis])) for axis in range(n)]
            for index in neighbors
        ]
        jacobian = np.asarray(_solve_exact(source, rhs), dtype=np.float64).T
        origin = images[0] - jacobian @ reference[0]
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
    if family in ("prism-trimmed", "pyramid-trimmed"):
        from ._hybrid_forms import entity_chart

        origin, jacobian = entity_chart(family, face)
        return (), origin, jacobian
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
    zero: Array
    one: Array
    zero_derivatives: Array
    one_derivatives: Array
    indices: Array
    differential_axes: Array
    components: Array

    def __init__(self, n: int, k: int, r: int, labels: tuple[DofLabel, ...]) -> None:
        interval = _tensor_interval(r)
        blades = tuple(combinations(range(n), k))
        self.order = r
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
class _HybridFactors(StrictModule):
    """Exact immutable generator source with deliberate numerical runtime leaves."""

    source_bank: tuple[tuple[Fraction, ...], ...] = eqx.field(static=True)
    generators: Array
    dual_rank: Array
    dual_condition: Array
    dual_solve_error: Array
    dual_status: Array
    body_test_exponents: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    body_test_source_bank: tuple[tuple[Fraction, ...], ...] = eqx.field(static=True)
    body_test_coefficients: Array

    def __init__(
        self,
        source_bank: tuple[tuple[Fraction, ...], ...],
        generators: ArrayLike,
        condition: float,
        solve_error: float,
        solve_status: ArrayLike,
        body_test_exponents: tuple[tuple[int, ...], ...],
        body_test_source_bank: tuple[tuple[Fraction, ...], ...],
        body_test_coefficients: ArrayLike,
    ) -> None:
        from ...linalg import LinearSolveStatus

        generators_ = jnp.asarray(generators)
        if (
            generators_.ndim != 3
            or not source_bank
            or len(source_bank) != generators_.shape[0] * generators_.shape[1]
        ):
            raise ValueError(
                "Hybrid generator source and numerical component axes disagree."
            )
        if (
            any(len(row) != generators_.shape[-1] for row in source_bank)
            or not np.isfinite(condition)
            or condition > 1e12
        ):
            raise ValueError("Hybrid generator source is incomplete or ill conditioned.")
        expected = np.asarray(source_bank, dtype=np.float64).reshape(generators_.shape)
        if not np.array_equal(np.asarray(generators_), expected):
            raise ValueError(
                "Hybrid numerical generators must be the rounding of their exact source."
            )
        status = jnp.asarray(solve_status, dtype=jnp.int32)
        if (
            not np.isfinite(solve_error)
            or solve_error < 0
            or np.any(np.asarray(status) != int(LinearSolveStatus.SUCCESS))
        ):
            raise ValueError(
                "Hybrid native entity dual solve did not succeed with finite error evidence."
            )
        tests = jnp.asarray(body_test_coefficients, dtype=generators_.dtype)
        if (
            tests.ndim != 3
            or tests.shape[0] != len(body_test_exponents)
            or tests.shape[1] != generators_.shape[1]
            or len(body_test_source_bank) != tests.shape[0] * tests.shape[1]
            or any(len(row) != tests.shape[2] for row in body_test_source_bank)
        ):
            raise ValueError(
                "Hybrid moment tests require complete polynomial and form-component identities."
            )
        expected_tests = np.asarray(body_test_source_bank, dtype=np.float64).reshape(
            tests.shape
        )
        if not np.array_equal(np.asarray(tests), expected_tests):
            raise ValueError(
                "Hybrid numerical moment tests no longer match their exact source."
            )
        self.source_bank = source_bank
        self.generators = generators_
        self.dual_rank = jnp.asarray(generators_.shape[-1], dtype=jnp.int32)
        self.dual_condition = jnp.asarray(condition, dtype=generators_.dtype)
        self.dual_solve_error = jnp.asarray(solve_error, dtype=generators_.dtype)
        self.dual_status = status
        self.body_test_exponents = body_test_exponents
        self.body_test_source_bank = body_test_source_bank
        self.body_test_coefficients = tests


@lru_cache(maxsize=96)
def _polynomial_form_preparation(
    dimension: int,
    form_degree: int,
    order: int,
    family: FormElementFamily,
    /,
) -> tuple[_Prepared, _HostArray]:
    prepared = (
        _prepare_tensor(dimension, form_degree, order)
        if family == "tensor-trimmed"
        else _prepare_simplex(dimension, form_degree, order, family)
    )
    if family == "tensor-trimmed":
        coefficients = _tensor_coefficients(
            dimension,
            order,
            prepared.exponents,
            prepared.labels,
            tuple(combinations(range(dimension), form_degree)),
        )
    else:
        matrix = np.einsum("dmc,mcb->db", prepared.moments, prepared.generators)
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError(
                "Polynomial generators and entity functionals disagree in dimension."
            )
        flat = prepared.generators.reshape(-1, matrix.shape[0])
        coefficients = _solve_reference_moments(matrix.T, flat.T).T.reshape(
            prepared.generators.shape
        )
        coefficients.setflags(write=False)
    return prepared, coefficients


def _entity_form_basis(
    dimension: int,
    form_degree: int,
    order: int,
    family: FormElementFamily,
    /,
) -> FormBasis:
    return FormBasis(dimension, form_degree, order, family, "untwisted")


@final
class FormBasis(StrictModule):
    """A complete canonical moment-dual polynomial, product, or rational form basis."""

    dimension: int = eqx.field(static=True)
    form_degree: int = eqx.field(static=True)
    order: int = eqx.field(static=True)
    family: FormElementFamily = eqx.field(static=True)
    twist: FormTwist = eqx.field(static=True)
    exponents: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    dof_labels: tuple[DofLabel, ...] = eqx.field(static=True)
    entity_vertices: tuple[tuple[tuple[int, ...], ...], ...] = eqx.field(static=True)
    coefficients: Array
    tensor_factors: _TensorFactors | None
    hybrid_factors: _HybridFactors | None
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
        if family_ in ("prism-trimmed", "pyramid-trimmed"):
            from ._hybrid_forms import prepare_hybrid

            if dimension != 3:
                raise ValueError("Hybrid compatible forms require dimension three.")
            hybrid = prepare_hybrid(family_, form_degree, order)
            prepared, coefficients = hybrid.prepared, hybrid.coefficients
            tensor_factors = None
            hybrid_factors = _HybridFactors(
                hybrid.source_bank,
                prepared.generators,
                hybrid.condition,
                hybrid.solve_error,
                hybrid.solve_status,
                hybrid.body_test_exponents,
                hybrid.body_test_source_bank,
                hybrid.body_test_coefficients,
            )
        else:
            hybrid_factors = None
            prepared, coefficients = _polynomial_form_preparation(
                dimension, form_degree, order, family_
            )
            tensor_factors = (
                _TensorFactors(dimension, form_degree, order, prepared.labels)
                if family_ == "tensor-trimmed"
                else None
            )
        self.dimension = dimension
        self.form_degree = form_degree
        self.order = order
        self.family = family_
        self.twist = twist_
        self.exponents = prepared.exponents
        self.dof_labels = prepared.labels
        self.entity_vertices = prepared.entities
        self.coefficients = jnp.asarray(coefficients)
        self.tensor_factors = tensor_factors
        self.hybrid_factors = hybrid_factors
        self.functional_points = jnp.asarray(prepared.points)
        self.functional_weights = jnp.asarray(prepared.weights)
        self.functional_moments = jnp.asarray(prepared.moments)
        self.reference_nodes = jnp.asarray(prepared.nodes)
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
                **(
                    {}
                    if hybrid_factors is None
                    else {
                        "generator_source": tuple(
                            tuple((value.numerator, value.denominator) for value in row)
                            for row in hybrid_factors.source_bank
                        ),
                        "moment_exponents": hybrid_factors.body_test_exponents,
                        "moment_source": tuple(
                            tuple((value.numerator, value.denominator) for value in row)
                            for row in hybrid_factors.body_test_source_bank
                        ),
                    }
                ),
            }
        )

    @property
    def local_dof_count(self) -> int:
        return self.coefficients.shape[-1]

    def tabulate_components(self, points: ArrayLike, /) -> tuple[Array, Array]:
        points_ = jnp.asarray(points, dtype=self.coefficients.dtype)
        if points_.ndim != 2 or points_.shape[1] != self.dimension:
            raise ValueError("Points must have shape (point_count, dimension).")
        if self.hybrid_factors is not None:
            from ._hybrid_forms import tabulate_hybrid

            values, gradients = tabulate_hybrid(
                points_,
                self.hybrid_factors.generators,
                self.exponents,
                self.family,
                self.form_degree,
            )
            return (
                contract("qgc,gb->qbc", values, self.coefficients),
                contract("qgca,gb->qbca", gradients, self.coefficients),
            )
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

    def entity_kind(self, face: tuple[int, ...], /) -> str:
        """Return the scientific reference topology of a declared entity."""
        if not any(face in level for level in self.entity_vertices):
            raise ValueError("Use a declared reference entity and its vertex ordering.")
        if self.family in ("prism-trimmed", "pyramid-trimmed"):
            from ._hybrid_forms import entity_kind

            return entity_kind(self.family, face)
        dimension = next(
            index for index, level in enumerate(self.entity_vertices) if face in level
        )
        return (
            f"tensor:{dimension}"
            if self.family == "tensor-trimmed"
            else f"simplex:{dimension}"
        )

    def entity_basis(self, face: tuple[int, ...], /) -> FormBasis:
        """Own the oriented trace induced by the cell/entity incidence.

        An ambient twisted codimension-one form receives its coorientation from
        the oriented cell boundary. Its intrinsic entity moment is therefore an
        untwisted form in that induced chart; applying a second twist here would
        erase the shared-facet incidence sign already carried by the face order.
        """
        if self.family in ("prism-trimmed", "pyramid-trimmed"):
            from ._hybrid_forms import trace_basis

            return trace_basis(self.family, self.form_degree, self.order, face)
        dimension = reference_cell_topology(self.entity_kind(face)).dimension
        return _entity_form_basis(
            dimension,
            self.form_degree,
            self.order,
            self.family,
        )

    def component_expressions(self, /) -> tuple[tuple[Expression, ...], ...]:
        """Host source expressions in physical reference component axes.

        Rows are local basis DOFs, columns are increasing differential-form
        blades. Rational sources retain their actual collapsed denominators.
        """
        from ._form_expressions import component_expressions

        return component_expressions(self)

    def functional_density_expressions(
        self, face: tuple[int, ...], /
    ) -> tuple[tuple[Expression, ...], ...]:
        """Host moment densities in the declared entity chart and ambient blades."""
        from ._form_expressions import functional_density_expressions

        return functional_density_expressions(self, face)

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
        if self.family in ("prism-trimmed", "pyramid-trimmed"):
            from ._hybrid_forms import functional_weights_at

            return functional_weights_at(self, entity_vertices, points, quadrature)
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

    def exterior_derivative_matrix(self, target: FormBasis, /) -> Array:
        if (target.dimension, target.form_degree, target.twist) != (
            self.dimension,
            self.form_degree + 1,
            self.twist,
        ):
            raise ValueError(
                "Exterior derivative target must have next degree, same dimension and twist."
            )
        if self.hybrid_factors is None and target.hybrid_factors is not None:
            raise ValueError(
                "Exterior differentiation cannot replace a source reference topology by a hybrid topology."
            )
        if self.family in ("prism-trimmed", "pyramid-trimmed"):
            if target.family != self.family or target.order < self.order:
                raise ValueError(
                    "Hybrid exterior derivatives require the same family and containing order."
                )
            factors = self.hybrid_factors
            if factors is None:
                raise ValueError(
                    "Hybrid exterior differentiation requires its source generator owner."
                )
            terms = _polynomial_derivative_terms(
                self.exponents, target.exponents, self.dimension, self.form_degree
            )
            indices, components, lowers, outputs, weights = (
                jnp.asarray(column, dtype=jnp.int32)
                for column in zip(*terms, strict=True)
            )
            derivatives = jnp.zeros(
                (
                    len(target.exponents),
                    comb(self.dimension, self.form_degree + 1),
                    factors.generators.shape[-1],
                ),
                dtype=self.coefficients.dtype,
            )
            derivatives = derivatives.at[lowers, outputs].add(
                weights[:, None] * factors.generators[indices, components]
            )
            action = contract("dmc,mcg->dg", target.functional_moments, derivatives)
            return action @ self.coefficients
        minimum_order = self.order - int(target.family == "full")
        if (target.family == "tensor-trimmed") != (self.family == "tensor-trimmed"):
            raise ValueError("Tensor and simplicial derivative spaces cannot be mixed.")
        if target.order < minimum_order:
            raise ValueError(
                "Target family/order does not contain the exterior derivative."
            )
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
        if self.tensor_factors is not None or self.family in (
            "prism-trimmed",
            "pyramid-trimmed",
        ):
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

    def canonical_permutation(
        self, global_vertices: tuple[int, ...], /
    ) -> tuple[int, ...]:
        """Map local vertices to a deterministic, geometrically legal chart."""
        if self.family in ("prism-trimmed", "pyramid-trimmed"):
            from ._hybrid_forms import canonical_permutation

            return canonical_permutation(self.family, global_vertices)
        if self.family != "tensor-trimmed":
            ordered = tuple(sorted(global_vertices))
            return tuple(ordered.index(vertex) for vertex in global_vertices)
        return _canonical_cube_permutation(
            _tensor_vertices(self.dimension), global_vertices
        )

    def entity_permutation_matrix(
        self,
        global_vertices: tuple[int, ...],
        /,
        *,
        entity_bases: tuple[tuple[FormBasis | None, ...], ...] | None = None,
    ) -> Array:
        """Map independently canonical entity moments to local element moments.

        Columns retain local entity-block positions; each block's columns are
        interpreted in its globally canonical entity chart. Unlike a whole-cell
        permutation, this assembly transform does not move entity blocks.
        """
        expected = len(_form_reference_vertices(self.dimension, self.family))
        if len(global_vertices) != expected or len(set(global_vertices)) != expected:
            raise ValueError("Provide one distinct global identifier per local vertex.")
        if entity_bases is not None and (
            len(entity_bases) != len(self.entity_vertices)
            or any(
                len(bases) != len(entities)
                for bases, entities in zip(
                    entity_bases, self.entity_vertices, strict=True
                )
            )
        ):
            raise ValueError("Canonical moment owners must identify every local entity.")
        result = jnp.zeros(
            (self.local_dof_count, self.local_dof_count), dtype=self.coefficients.dtype
        )
        for m, level in enumerate(self.entity_vertices):
            for entity_index, face in enumerate(level):
                indices = _entity_dof_indices(self.dof_labels, face)
                if indices:
                    block = self._entity_moment_transform(
                        m,
                        face,
                        len(indices),
                        global_vertices,
                        None if entity_bases is None else entity_bases[m][entity_index],
                    )
                    result = result.at[
                        jnp.asarray(indices)[:, None], jnp.asarray(indices)[None, :]
                    ].set(block)
        return result

    def _entity_moment_transform(
        self,
        m: int,
        face: tuple[int, ...],
        count: int,
        global_vertices: tuple[int, ...],
        canonical: FormBasis | None,
    ) -> Array:
        if m == 0 or (self.family == "full" and self.order == 0):
            return jnp.eye(count, dtype=self.coefficients.dtype)
        if m == self.dimension and self.hybrid_factors is not None:
            # Interior moments belong to this cell's declared source chart; no
            # shared entity may replace that chart by coincident array width.
            return jnp.eye(count, dtype=self.coefficients.dtype)
        entity = self.entity_basis(face)
        reference = _form_reference_vertices(entity.dimension, entity.family)
        interior = _interior_dof_indices(entity.dof_labels, len(reference))
        if len(interior) != count:
            raise ValueError("Entity moment block dimensions disagree.")
        if self.family in ("prism-trimmed", "pyramid-trimmed"):
            ids = tuple(global_vertices[index] for index in face)
        else:
            ids = _canonical_entity_ids(
                self.family, self.dimension, m, face, global_vertices
            )
        if canonical is None or canonical.basis_id == entity.basis_id:
            full = entity.permutation_matrix(entity.canonical_permutation(ids))
            return full[jnp.asarray(interior)[:, None], jnp.asarray(interior)[None, :]]
        return _entity_basis_change(entity, canonical, ids, interior)


def _entity_basis_change(
    local: FormBasis,
    canonical: FormBasis,
    ids: tuple[int, ...],
    interior: tuple[int, ...],
) -> Array:
    """Full functional change of coordinates, not equal-width moment matching."""
    if (local.dimension, local.form_degree, local.twist) != (
        canonical.dimension,
        canonical.form_degree,
        canonical.twist,
    ):
        raise ValueError(
            "Shared form entities require the same scientific form identity."
        )
    reference = _form_reference_vertices(local.dimension, local.family)
    canonical_reference = _form_reference_vertices(canonical.dimension, canonical.family)
    if not np.array_equal(reference, canonical_reference):
        raise ValueError(
            "Shared form entity moments require the same reference topology."
        )
    canonical_interior = _interior_dof_indices(canonical.dof_labels, len(reference))
    if len(canonical_interior) != len(interior):
        raise ValueError("Shared form entities require equal interior moment spaces.")
    origin, jacobian = _vertex_permutation_affine(
        reference, canonical.canonical_permutation(ids)
    )
    pullback = _form_pullback_matrix(
        local.dimension, local.form_degree, jacobian, local.twist
    )
    original = np.asarray(
        contract("ce,meb->mcb", jnp.asarray(pullback), canonical.coefficients)
    )
    substituted = _substitute_coefficients(
        canonical.exponents, original, origin, jacobian
    )
    coefficients = np.zeros(
        (len(local.exponents), original.shape[1], canonical.local_dof_count),
        dtype=np.float64,
    )
    positions = {index: row for row, index in enumerate(local.exponents)}
    for index, coefficient in zip(canonical.exponents, substituted, strict=True):
        if index not in positions:
            if np.any(coefficient != 0):
                raise ValueError(
                    "Shared entity form spaces have different polynomial support."
                )
        else:
            coefficients[positions[index]] = coefficient
    full = np.asarray(local.functional_moments).reshape(
        local.local_dof_count, -1
    ) @ coefficients.reshape(-1, canonical.local_dof_count)
    block = full[np.ix_(interior, canonical_interior)]
    spectrum = np.linalg.svd(block, compute_uv=False)
    if (
        spectrum.size != block.shape[0]
        or not np.all(np.isfinite(spectrum))
        or spectrum[-1] <= 0
        or spectrum[0] / spectrum[-1] > 1e12
    ):
        raise ValueError(
            "Shared entity moment transformation is singular or ill conditioned."
        )
    boundary = [
        index
        for index in range(canonical.local_dof_count)
        if index not in canonical_interior
    ]
    if boundary and np.max(np.abs(full[np.ix_(interior, boundary)]), initial=0) > (
        256 * np.finfo(np.float64).eps * max(float(np.max(np.abs(block), initial=0)), 1.0)
    ):
        raise ValueError(
            "Entity moment conversion mixes independently owned boundary coordinates."
        )
    return jnp.asarray(block)


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

        def convert_gradient(components: Array) -> Array:
            return form_to_vector(components, self.value_spec)

        return converted, jax.vmap(convert_gradient, in_axes=-1, out_axes=-1)(gradients)


def _admit_form_cell(
    cell_kind: str, family: FormElementFamily
) -> tuple[int, FormElementFamily]:
    dimension = reference_cell_topology(cell_kind).dimension
    family_ = parse(family, FormElementFamily, "family")
    if cell_kind in ("prism", "pyramid"):
        expected = "prism-trimmed" if cell_kind == "prism" else "pyramid-trimmed"
        if family_ == "trimmed":
            family_ = "prism-trimmed" if cell_kind == "prism" else "pyramid-trimmed"
        if family_ != expected:
            raise ValueError(
                f"{cell_kind.capitalize()} cells require family={expected!r}."
            )
        return dimension, family_
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
    if family_ in ("prism-trimmed", "pyramid-trimmed"):
        raise ValueError(
            "Hybrid form families require their declared prism or pyramid reference cell."
        )
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
    """Construct simplex, tensor, prism-product, or rational-pyramid FEEC forms.

    ``order`` is the FEEC approximation index, not the total degree of a
    pyramid's rational components. Lowest trimmed order is one. Hybrids own
    scalar polynomials through order r and k>0 polynomials through r-1.
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
        representation="rational_moment"
        if family_ == "pyramid-trimmed"
        else "polynomial_moment",
        tabulator=_ProxyTabulator(basis, value_spec),
        tabulator_id=basis.basis_id,
        form_basis=basis,
    )


__all__ = ["DofLabel", "FormBasis", "FormElementFamily", "form_element"]
