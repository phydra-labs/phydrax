# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Immutable preparation of product and rational compatible form spaces.

The pyramid complex is the trace-constrained generalized Whitney complex of
its five rational barycentric coordinates. Its quadrilateral trace is Q_r^-
and its triangular traces are P_r^-. Coefficients are polynomial pullbacks to
the collapsed cube; no polynomial claim is made about physical components.
"""

from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
from itertools import combinations, permutations, product
from math import comb, factorial
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import Array

from ...ein import contract
from .._reference_cell import reference_cell_topology
from . import _form_elements as forms


class HybridPreparation(NamedTuple):
    prepared: forms._Prepared
    coefficients: forms._HostArray
    condition: float
    source_bank: tuple[tuple[Fraction, ...], ...]
    solve_error: float
    solve_status: np.ndarray
    body_test_exponents: tuple[tuple[int, ...], ...]
    body_test_source_bank: tuple[tuple[Fraction, ...], ...]
    body_test_coefficients: forms._HostArray


def hybrid_kind(family: forms.FormElementFamily) -> str:
    if family == "prism-trimmed":
        return "prism"
    if family == "pyramid-trimmed":
        return "pyramid"
    raise ValueError("A hybrid form family is required.")


def entity_kind(family: forms.FormElementFamily, face: tuple[int, ...]) -> str:
    if len(face) == 1:
        return "simplex:0"
    if len(face) == 2:
        return "interval"
    if len(face) == 3:
        return "triangle"
    if len(face) == 4:
        return "quadrilateral"
    return hybrid_kind(family)


def entity_chart(
    family: forms.FormElementFamily, face: tuple[int, ...]
) -> tuple[forms._HostArray, forms._HostArray]:
    topology = reference_cell_topology(hybrid_kind(family))
    vertices = np.asarray(topology.vertices, dtype=np.float64)
    if len(face) == len(vertices):
        return np.zeros(3, dtype=np.float64), np.eye(3, dtype=np.float64)
    origin = vertices[face[0]]
    if len(face) == 4:
        return origin, np.stack(
            (vertices[face[1]] - origin, vertices[face[3]] - origin), axis=1
        )
    return origin, (vertices[list(face[1:])] - origin).T


def trace_basis(
    family: forms.FormElementFamily, k: int, r: int, face: tuple[int, ...]
) -> forms.FormBasis:
    kind = entity_kind(family, face)
    n = reference_cell_topology(kind).dimension
    if n == 3:
        return forms.FormBasis(3, k, r, family, "untwisted")
    return forms.FormBasis(
        n,
        k,
        r,
        "tensor-trimmed" if kind == "quadrilateral" else "trimmed",
        "untwisted",
    )


def _derivative(poly: forms._Polynomial, axis: int) -> forms._Polynomial:
    result: forms._Polynomial = {}
    for alpha, value in poly.items():
        if alpha[axis]:
            lower = list(alpha)
            lower[axis] -= 1
            result[tuple(lower)] = value * alpha[axis]
    return result


def _sum_polynomials(
    polys: tuple[forms._Polynomial, ...], factors: tuple[float, ...]
) -> forms._Polynomial:
    result: forms._Polynomial = {}
    for poly, factor in zip(polys, factors, strict=True):
        for alpha, value in poly.items():
            result[alpha] = result.get(alpha, 0.0) + factor * value
    return {alpha: value for alpha, value in result.items() if value != 0}


def _pyramid_coordinates() -> tuple[forms._Polynomial, ...]:
    a: forms._Polynomial = {(1, 0, 0): 1.0}
    b: forms._Polynomial = {(0, 1, 0): 1.0}
    z: forms._Polynomial = {(0, 0, 1): 1.0}
    one: forms._Polynomial = {(0, 0, 0): 1.0}
    aa, bb, zz = (_sum_polynomials((one, p), (1.0, -1.0)) for p in (a, b, z))
    return tuple(
        forms._multiply(forms._multiply(x, y), zz)
        for x, y in ((aa, bb), (a, bb), (a, b), (aa, b))
    ) + (z,)


def _whitney_polynomials(
    coordinates: tuple[forms._Polynomial, ...],
    k: int,
    alpha: tuple[int, ...],
    sigma: tuple[int, ...],
    n: int,
) -> tuple[forms._Polynomial, ...]:
    scalar = {(0,) * n: 1.0}
    for coordinate, power in zip(coordinates, alpha, strict=True):
        scalar = forms._multiply(scalar, forms._power(coordinate, power, n))
    components: list[forms._Polynomial] = []
    for blade in combinations(range(n), k):
        summands, factors = [], []
        for position, vertex in enumerate(sigma):
            others = tuple(v for v in sigma if v != vertex)
            for order in permutations(range(k)):
                term = forms._multiply(scalar, coordinates[vertex])
                for i, axis in enumerate(blade):
                    term = forms._multiply(
                        term, _derivative(coordinates[others[order[i]]], axis)
                    )
                sign = (-1) ** sum(
                    order[i] > order[j] for i in range(k) for j in range(i + 1, k)
                )
                summands.append(term)
                factors.append(float(factorial(k) * (-1) ** position * sign))
        components.append(_sum_polynomials(tuple(summands), tuple(factors)))
    return tuple(components)


def _coefficient_array(
    polys: tuple[tuple[forms._Polynomial, ...], ...],
) -> tuple[tuple[tuple[int, ...], ...], forms._HostArray]:
    exponents = tuple(
        sorted(
            {alpha for field in polys for component in field for alpha in component},
            key=lambda alpha: (sum(alpha), alpha),
        )
    )
    positions = {alpha: i for i, alpha in enumerate(exponents)}
    coefficients = np.zeros((len(exponents), len(polys[0]), len(polys)), dtype=np.float64)
    for column, field in enumerate(polys):
        for component, poly in enumerate(field):
            for alpha, value in poly.items():
                coefficients[positions[alpha], component, column] = value
    return exponents, coefficients


def _independent_columns(matrix: forms._HostArray) -> tuple[int, ...]:
    # Modified Gram--Schmidt is host-only symbolic-space admission. Stable
    # lexicographic pivots preserve the declared monomial scientific identity.
    scale = np.max(np.linalg.norm(matrix, axis=0), initial=0.0)
    tolerance = 2048 * np.finfo(np.float64).eps * max(scale, 1.0)
    orthogonal: list[forms._HostArray] = []
    indices: list[int] = []
    for index in range(matrix.shape[1]):
        residual = matrix[:, index].copy()
        for _ in range(2):
            for column in orthogonal:
                residual -= column * (column @ residual)
        norm = np.linalg.norm(residual)
        if norm > tolerance:
            indices.append(index)
            orthogonal.append(residual / norm)
    return tuple(indices)


@lru_cache(maxsize=32)
def _prism_generators(
    k: int, r: int
) -> tuple[tuple[tuple[int, ...], ...], forms._HostArray]:
    exponents = tuple(product(range(r + 1), repeat=3))
    positions = {alpha: i for i, alpha in enumerate(exponents)}
    blades = tuple(combinations(range(3), k))
    fields: list[forms._HostArray] = []
    for axial_degree in (0, 1):
        base_degree = k - axial_degree
        if not 0 <= base_degree <= 2:
            continue
        base_exponents, base, _ = forms._generators(2, base_degree, r, True)
        base_blades = tuple(combinations(range(2), base_degree))
        for column in range(base.shape[-1]):
            for power in range(r + 1 - axial_degree):
                field = np.zeros((len(exponents), len(blades)), dtype=np.float64)
                for component, blade in enumerate(base_blades):
                    output = blades.index(blade + ((2,) if axial_degree else ()))
                    for alpha, coefficient in zip(
                        base_exponents, base[:, component, column], strict=True
                    ):
                        field[positions[alpha + (power,)], output] = coefficient
                fields.append(field)
    coefficients = np.stack(fields, axis=-1)
    coefficients.setflags(write=False)
    return exponents, coefficients


def _exact_nullspace(matrix: forms._HostArray) -> tuple[tuple[Fraction, ...], ...]:
    rows = [[Fraction(float(value)) for value in row] for row in matrix if np.any(row)]
    width = matrix.shape[1]
    pivots: list[int] = []
    active = 0
    for column in range(width):
        candidates = [index for index in range(active, len(rows)) if rows[index][column]]
        if not candidates:
            continue
        pivot = candidates[0]
        rows[active], rows[pivot] = rows[pivot], rows[active]
        divisor = rows[active][column]
        rows[active] = [value / divisor for value in rows[active]]
        for index in range(len(rows)):
            if index != active and rows[index][column]:
                factor = rows[index][column]
                rows[index] = [
                    value - factor * basis
                    for value, basis in zip(rows[index], rows[active], strict=True)
                ]
        pivots.append(column)
        active += 1
        if active == len(rows):
            break
    free = tuple(column for column in range(width) if column not in pivots)
    vectors = []
    for column in free:
        vector = [Fraction(int(index == column)) for index in range(width)]
        for row, pivot in zip(rows, pivots, strict=False):
            vector[pivot] = -row[column]
        vectors.append(tuple(vector))
    return tuple(tuple(vector[row] for vector in vectors) for row in range(width))


@lru_cache(maxsize=32)
def _pyramid_generators(
    k: int, r: int
) -> tuple[
    tuple[tuple[int, ...], ...], forms._HostArray, tuple[tuple[Fraction, ...], ...]
]:
    if k == 3:
        # The terminal space is the exact exterior image of the trace-admitted
        # flux space. Unrestricted four-vertex Whitney products on a five-
        # coordinate rational cone include a spurious top cohomology class.
        source_exponents, source_coefficients, source_bank = _pyramid_generators(2, r)
        source_blades = tuple(combinations(range(3), 2))
        terms: dict[tuple[int, ...], list[Fraction]] = {}
        width = source_coefficients.shape[-1]
        for row, alpha in enumerate(source_exponents):
            for axis in range(3):
                if not alpha[axis]:
                    continue
                lower = tuple(
                    power - int(column == axis) for column, power in enumerate(alpha)
                )
                target = terms.setdefault(lower, [Fraction(0)] * width)
                for component, _, sign in forms._blade_derivative_terms(3, 2, axis):
                    for column, value in enumerate(
                        source_bank[row * len(source_blades) + component]
                    ):
                        target[column] += sign * alpha[axis] * value
        exponents = tuple(sorted(terms, key=lambda alpha: (sum(alpha), alpha)))
        bank = tuple(tuple(terms[alpha]) for alpha in exponents)
        coefficients = np.asarray(bank, dtype=np.float64).reshape(
            len(exponents), 1, width
        )
        coefficients.setflags(write=False)
        return exponents, coefficients, bank
    coordinates = _pyramid_coordinates()
    labels = tuple(
        (alpha, sigma)
        for sigma in combinations(range(5), k + 1)
        for alpha in forms._compositions(r - 1, 5)
        if not (alpha[0] and alpha[2])
    )
    polys = tuple(
        _whitney_polynomials(coordinates, k, alpha, sigma, 3) for alpha, sigma in labels
    )
    exponents, coefficients = _coefficient_array(polys)
    independent = _independent_columns(coefficients.reshape(-1, len(labels)))
    coefficients = coefficients[:, :, independent]
    source_bank = tuple(
        tuple(Fraction(float(value)) for value in row)
        for row in coefficients.reshape(-1, coefficients.shape[-1])
    )
    # Triangular restrictions are Whitney polynomial forms automatically. The
    # base restriction must be explicitly intersected with Q_r^- Lambda^k.
    if k <= 2:
        base = reference_cell_topology("pyramid").entities[2][0]
        origin, jacobian = entity_chart("pyramid-trimmed", base)
        # The base chart has reversed x/y axes; substitute that chart in the
        # collapsed coordinates before comparing polynomial support.
        restricted = tuple(
            forms._substitute(alpha, origin, jacobian) for alpha in exponents
        )
        powers = tuple(
            product(
                *(
                    range(
                        max(
                            (alpha[axis] for poly in restricted for alpha in poly),
                            default=0,
                        )
                        + 1
                    )
                    for axis in range(2)
                )
            )
        )
        positions = {alpha: i for i, alpha in enumerate(powers)}
        pullback = (
            forms._form_pullback_matrix(3, k, jacobian, "untwisted")
            if k == 0
            else np.asarray(
                [
                    [
                        forms._minor(jacobian, ambient, local)
                        for ambient in combinations(range(3), k)
                    ]
                    for local in combinations(range(2), k)
                ]
            )
        )
        restriction = np.zeros(
            (len(powers), len(tuple(combinations(range(2), k))), coefficients.shape[-1]),
            dtype=np.float64,
        )
        for row, poly in enumerate(restricted):
            for alpha, value in poly.items():
                restriction[positions[alpha]] += value * (pullback @ coefficients[row])
        disallowed: list[forms._HostArray] = []
        for row, alpha in enumerate(powers):
            for component, blade in enumerate(combinations(range(2), k)):
                if any(alpha[axis] > r - int(axis in blade) for axis in range(2)):
                    disallowed.append(restriction[row, component])
        constraints = (
            np.stack(disallowed)
            if disallowed
            else np.zeros((0, coefficients.shape[-1]), dtype=np.float64)
        )
        if constraints.shape[0]:
            kernel = _exact_nullspace(constraints)
            source_bank = tuple(
                tuple(
                    sum(
                        (
                            value * kernel[index][column]
                            for index, value in enumerate(row)
                        ),
                        Fraction(0),
                    )
                    for column in range(len(kernel[0]))
                )
                for row in source_bank
            )
            coefficients = np.asarray(source_bank, dtype=np.float64).reshape(
                coefficients.shape[:2] + (len(kernel[0]),)
            )
    coefficients.setflags(write=False)
    return exponents, coefficients, source_bank


def _expand(points: forms._HostArray) -> forms._HostArray:
    return np.stack(
        (
            (1 - points[:, 2]) * points[:, 0] + 0.5 * points[:, 2],
            (1 - points[:, 2]) * points[:, 1] + 0.5 * points[:, 2],
            points[:, 2],
        ),
        axis=-1,
    )


def _cube_jacobian(points: forms._HostArray) -> forms._HostArray:
    s = 1 - points[:, 2]
    result = np.zeros((points.shape[0], 3, 3), dtype=np.float64)
    result[:, 0, 0] = s
    result[:, 1, 1] = s
    result[:, 2, 2] = 1
    result[:, 0, 2] = 0.5 - points[:, 0]
    result[:, 1, 2] = 0.5 - points[:, 1]
    return result


def _body_rule(
    family: forms.FormElementFamily, degree: int
) -> tuple[forms._HostArray, forms._HostArray, forms._HostArray]:
    if family == "prism-trimmed":
        base, bw = forms._simplex_quadrature(2, degree)
        interval, iw = forms._tensor_quadrature(1, degree)
        points = np.asarray(
            [(x, y, z) for x, y in base for (z,) in interval], dtype=np.float64
        )
        return points, (bw[:, None] * iw[None, :]).reshape(-1), points
    chart, weights = forms._tensor_quadrature(3, degree)
    return _expand(chart), weights * (1 - chart[:, 2]) ** 2, chart


def _solve_hybrid_dual(
    matrix: forms._HostArray,
) -> tuple[forms._HostArray, float, np.ndarray]:
    """Solve one immutable host reference dual without compiler startup."""
    from ...linalg import LinearSolveStatus

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Hybrid entity dual requires one square functional matrix.")
    identity = np.eye(matrix.shape[0], dtype=np.float64)
    coefficients = np.linalg.solve(matrix, identity)
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("Hybrid entity dual solve produced non-finite coefficients.")
    residual = matrix @ coefficients - identity
    scale = max(
        float(np.max(np.abs(matrix) @ np.abs(coefficients), initial=0.0)),
        1.0,
    )
    solve_error = float(np.max(np.abs(residual), initial=0.0)) / scale
    status = np.asarray(int(LinearSolveStatus.SUCCESS), dtype=np.int32)
    status.setflags(write=False)
    return coefficients, solve_error, status


@lru_cache(maxsize=32)
def _hybrid_exponent_cover(
    family: forms.FormElementFamily, k: int, r: int
) -> tuple[tuple[int, ...], ...]:
    raw = (
        _prism_generators(k, r)[0]
        if family == "prism-trimmed"
        else _pyramid_generators(k, r)[0]
    )
    cover = set(raw)
    if k:
        for alpha in _hybrid_exponent_cover(family, k - 1, r):
            for axis, power in enumerate(alpha):
                if power:
                    cover.add(
                        tuple(
                            value - int(column == axis)
                            for column, value in enumerate(alpha)
                        )
                    )
    return tuple(sorted(cover, key=lambda alpha: (sum(alpha), alpha)))


@lru_cache(maxsize=48)
def prepare_hybrid(family: forms.FormElementFamily, k: int, r: int) -> HybridPreparation:
    kind = hybrid_kind(family)
    topology = reference_cell_topology(kind)
    # Bound the largest generator, cubature, or functional coefficient array
    # before allocating any of them. This is an entry-work allowance, not a
    # nominal order selector and not a claim about elapsed time or factor flops.
    generators_bound = 10 * (comb(r + 3, 4) - (comb(r + 1, 4) if r >= 3 else 0))
    axis_points = 2 * r + 12
    points_bound = axis_points**3 + 10 * axis_points**2 + 20 * axis_points + 6
    entry_work = max(
        (r + 2) ** 3 * 10 * (r + 4) ** 3,
        points_bound * 3 * generators_bound,
    )
    if entry_work > 24_000_000:
        raise ValueError(
            "Hybrid form preparation exceeds its 24000000 coefficient-entry work bound."
        )
    if kind == "prism":
        exponents, generators = _prism_generators(k, r)
        source_bank = tuple(
            tuple(Fraction(float(value)) for value in row)
            for row in generators.reshape(-1, generators.shape[-1])
        )
    else:
        exponents, generators, source_bank = _pyramid_generators(k, r)
    independent = _independent_columns(generators.reshape(-1, generators.shape[-1]))
    generators = generators[:, :, independent]
    source_bank = tuple(tuple(row[index] for index in independent) for row in source_bank)
    if k:
        closed_exponents = _hybrid_exponent_cover(family, k, r)
        if closed_exponents != exponents:
            positions = {alpha: row for row, alpha in enumerate(exponents)}
            expanded = np.zeros(
                (len(closed_exponents), generators.shape[1], generators.shape[-1]),
                dtype=np.float64,
            )
            bank = []
            for row, alpha in enumerate(closed_exponents):
                if alpha in positions:
                    original = positions[alpha]
                    expanded[row] = generators[original]
                    bank.extend(
                        source_bank[
                            original * generators.shape[1] : (original + 1)
                            * generators.shape[1]
                        ]
                    )
                else:
                    bank.extend(
                        (Fraction(0),) * generators.shape[-1]
                        for _ in range(generators.shape[1])
                    )
            exponents, generators, source_bank = closed_exponents, expanded, tuple(bank)
    dimension = generators.shape[-1]
    blades = tuple(combinations(range(3), k))
    blocks: list[forms._EntityMoments] = []
    trace_bases: dict[str, forms.FormBasis] = {}
    for m, entities in enumerate(topology.entities[:-1]):
        if m < k:
            continue
        for face in entities:
            origin, jacobian = entity_chart(family, face)
            if m == 0:
                points = origin[None, :]
                weights = np.ones((1, 1, 1), dtype=np.float64)
                labels = ((face, (), ()),)
            else:
                trace_kind = entity_kind(family, face)
                trace = trace_bases.get(trace_kind)
                if trace is None:
                    trace = trace_basis(family, k, r, face)
                    trace_bases[trace_kind] = trace
                interior = forms._interior_dof_indices(trace.dof_labels, len(face))
                if not interior:
                    continue
                points = (
                    origin[None, :] + np.asarray(trace.functional_points) @ jacobian.T
                )
                pullback = np.asarray(
                    [
                        [forms._minor(jacobian, ambient, local) for ambient in blades]
                        for local in combinations(range(m), k)
                    ]
                )
                weights = np.einsum(
                    "dql,lc->dqc",
                    np.asarray(trace.functional_weights)[list(interior)],
                    pullback,
                )
                active = np.any(weights != 0, axis=(0, 2))
                points, weights = points[active], weights[:, active]
                labels = tuple(
                    (face, trace.dof_labels[index][1], trace.dof_labels[index][2])
                    for index in interior
                )
            blocks.append(
                forms._EntityMoments(
                    labels,
                    points,
                    weights,
                    np.zeros(
                        (len(labels), len(exponents), len(blades)), dtype=np.float64
                    ),
                    np.broadcast_to(origin, (len(labels), 3)).copy(),
                )
            )
    from ._hybrid_functionals import exact_functional_moments, generator_functional_action
    from ._hybrid_moments import body_moments

    boundary_labels = tuple(label for block in blocks for label in block.labels)
    boundary_moments = exact_functional_moments(
        family,
        k,
        r,
        exponents,
        boundary_labels,
        (),
        (),
    )
    boundary = generator_functional_action(boundary_moments, source_bank, len(blades))
    test_exponents, test_coefficients = body_moments(
        family, k, r, exponents, generators, boundary, source_bank
    )
    body_count = test_coefficients.shape[-1]
    if body_count:
        degree = (
            max(sum(alpha) for alpha in exponents)
            + max(sum(alpha) for alpha in test_exponents)
            + 2
        )
        entries = (
            ((degree + 3) ** 3 + sum(block.points.shape[0] for block in blocks))
            * dimension
            * len(blades)
        )
        if entries > 24_000_000:
            raise ValueError(
                "Hybrid moment cubature exceeds its 24000000 coefficient-entry work bound before expansion."
            )
        points, measure, chart = _body_rule(family, degree)
        density = np.einsum(
            "qm,mcb->qbc", forms._monomials(chart, test_exponents), test_coefficients
        )
        if kind == "pyramid":
            jacobians = _cube_jacobian(chart)
            action = np.asarray(
                [
                    [
                        [forms._minor(matrix, ambient, local) for ambient in blades]
                        for local in blades
                    ]
                    for matrix in jacobians
                ]
            )
            chart_measure = measure / (1 - chart[:, 2]) ** 2
        else:
            action = np.broadcast_to(
                np.eye(len(blades), dtype=np.float64),
                (points.shape[0], len(blades), len(blades)),
            )
            chart_measure = measure
        body_weights = np.einsum("qbc,qce,q->bqe", density, action, chart_measure)
        face = topology.entities[3][0]
        labels = tuple((face, (index,), ()) for index in range(body_count))
        blocks.append(
            forms._EntityMoments(
                labels,
                points,
                body_weights,
                np.zeros((body_count, len(exponents), len(blades)), dtype=np.float64),
                np.broadcast_to(np.mean(points, axis=0), (body_count, 3)).copy(),
            )
        )
    prepared = forms._finish_prepared(exponents, generators, topology.entities, blocks)
    body_source_bank = tuple(
        tuple(Fraction(float(value)) for value in row)
        for row in test_coefficients.reshape(
            len(test_exponents) * len(blades), body_count
        )
    )
    exact_moments = exact_functional_moments(
        family,
        k,
        r,
        exponents,
        prepared.labels,
        test_exponents,
        body_source_bank,
    )
    prepared = prepared._replace(moments=np.asarray(exact_moments, dtype=np.float64))
    matrix = generator_functional_action(exact_moments, source_bank, len(blades))
    spectrum = np.linalg.svd(matrix, compute_uv=False)
    condition = float(spectrum[0] / spectrum[-1])
    if not np.isfinite(condition) or condition > 1e12:
        raise ValueError(
            "Hybrid entity dual is singular or exceeds condition bound 1e12."
        )
    coefficients, solve_error, solve_status = _solve_hybrid_dual(matrix)
    for array in prepared[1:]:
        if isinstance(array, np.ndarray):
            array.setflags(write=False)
    coefficients.setflags(write=False)
    test_coefficients.setflags(write=False)
    return HybridPreparation(
        prepared,
        coefficients,
        condition,
        source_bank,
        solve_error,
        solve_status,
        test_exponents,
        body_source_bank,
        test_coefficients,
    )


def tabulate_hybrid(
    points: Array,
    coefficients: Array,
    exponents: tuple[tuple[int, ...], ...],
    family: forms.FormElementFamily,
    k: int,
) -> tuple[Array, Array]:
    """Vectorized local polynomial/rational evaluation and analytic gradients."""
    powers = jnp.asarray(exponents, dtype=jnp.int32)
    identity = jnp.eye(3, dtype=jnp.int32)
    if family == "pyramid-trimmed":
        s = 1 - points[:, 2]
        chart = jnp.stack(
            (
                (points[:, 0] - 0.5 * points[:, 2]) / s,
                (points[:, 1] - 0.5 * points[:, 2]) / s,
                points[:, 2],
            ),
            axis=-1,
        )
        # The scalar apex limit is unique; derivative limits need not be.
        chart = jnp.where(
            (s == 0)[:, None], jnp.asarray((0.5, 0.5, 1.0), dtype=points.dtype), chart
        )
    else:
        chart = points
    degree = max(max(alpha) for alpha in exponents)
    table = jnp.cumprod(
        jnp.broadcast_to(chart[:, :, None], (*chart.shape, degree + 1))
        .at[:, :, 0]
        .set(1),
        axis=-1,
    )
    axes = jnp.arange(3, dtype=jnp.int32)
    monomials = jnp.prod(table[:, axes[None, :], powers], axis=-1)
    lower = jnp.maximum(powers[:, None, :] - identity[None, :, :], 0)
    monomial_derivatives = (
        jnp.prod(table[:, axes[None, None, :], lower], axis=-1) * powers[None, :, :]
    )
    values = contract("qm,mcb->qbc", monomials, coefficients)
    derivatives = contract("qma,mcb->qbca", monomial_derivatives, coefficients)
    if family != "pyramid-trimmed":
        return values, derivatives
    count = len(tuple(combinations(range(3), k)))
    a, b = chart[:, 0] - 0.5, chart[:, 1] - 0.5
    inv = jnp.zeros((points.shape[0], 3, 3), dtype=points.dtype)
    inv = inv.at[:, 0, 0].set(1 / s).at[:, 1, 1].set(1 / s).at[:, 2, 2].set(1)
    inv = inv.at[:, 0, 2].set(a / s).at[:, 1, 2].set(b / s)
    action = jnp.zeros((points.shape[0], count, count), dtype=points.dtype)
    gradient = jnp.zeros((points.shape[0], count, count, 3), dtype=points.dtype)
    entries = {
        0: ((0, 0, 0, 0, 0, 1),),
        1: (
            (0, 0, 0, 0, 1, 1),
            (1, 1, 0, 0, 1, 1),
            (2, 0, 1, 0, 1, 1),
            (2, 1, 0, 1, 1, 1),
            (2, 2, 0, 0, 0, 1),
        ),
        2: (
            (0, 0, 0, 0, 2, 1),
            (1, 0, 0, 1, 2, 1),
            (1, 1, 0, 0, 1, 1),
            (2, 0, 1, 0, 2, -1),
            (2, 2, 0, 0, 1, 1),
        ),
        3: ((0, 0, 0, 0, 2, 1),),
    }[k]
    rows, columns, pa, pb, denominator, signs = (
        jnp.asarray(column, dtype=jnp.int32) for column in zip(*entries, strict=True)
    )
    aa = jnp.where(pa[None, :] == 0, 1, a[:, None])
    bb = jnp.where(pb[None, :] == 0, 1, b[:, None])
    terms = signs[None, :] * aa * bb / s[:, None] ** denominator[None, :]
    action = action.at[:, rows, columns].set(terms)
    da = signs[None, :] * pa[None, :] * bb / s[:, None] ** denominator[None, :]
    db = signs[None, :] * pb[None, :] * aa / s[:, None] ** denominator[None, :]
    dz = jnp.where(
        denominator[None, :] == 0, 0, terms * denominator[None, :] / s[:, None]
    )
    gradient = gradient.at[:, rows, columns, :].set(jnp.stack((da, db, dz), axis=-1))
    physical = contract("qce,qbe->qbc", action, values)
    chart_gradient = contract("qcea,qbe->qbca", gradient, values) + contract(
        "qce,qbea->qbca", action, derivatives
    )
    return physical, contract("qbca,qad->qbcd", chart_gradient, inv)


@lru_cache(maxsize=2)
def _hybrid_symmetries(family: forms.FormElementFamily) -> tuple[tuple[int, ...], ...]:
    reference = np.asarray(
        reference_cell_topology(hybrid_kind(family)).vertices, dtype=np.float64
    )
    charts = np.c_[reference, np.ones((len(reference),), dtype=np.float64)]
    result = []
    for permutation in permutations(range(len(reference))):
        affine = np.linalg.lstsq(charts, reference[list(permutation)], rcond=None)[0]
        if np.max(np.abs(charts @ affine - reference[list(permutation)])) <= 1e-13:
            result.append(permutation)
    return tuple(result)


def canonical_permutation(
    family: forms.FormElementFamily, global_vertices: tuple[int, ...]
) -> tuple[int, ...]:
    if len(global_vertices) != len(
        reference_cell_topology(hybrid_kind(family)).vertices
    ) or len(set(global_vertices)) != len(global_vertices):
        raise ValueError(
            "Provide one distinct scientific vertex identifier per hybrid vertex."
        )
    canonical = min(
        _hybrid_symmetries(family),
        key=lambda order: tuple(global_vertices[index] for index in order),
    )
    return tuple(canonical.index(index) for index in range(len(canonical)))


def functional_weights_at(
    basis: forms.FormBasis, face: tuple[int, ...], points: Array, quadrature: Array
) -> tuple[Array, Array]:
    origin, jacobian = entity_chart(basis.family, face)
    if (
        points.ndim != 2
        or points.shape[1] != jacobian.shape[1]
        or quadrature.shape != (points.shape[0],)
    ):
        raise ValueError(
            "Entity cubature requires points(q,entity_dimension) and weights(q)."
        )
    reference = jnp.asarray(origin)[None, :] + points @ jnp.asarray(jacobian.T)
    blades = tuple(combinations(range(3), basis.form_degree))
    indices = forms._entity_dof_indices(basis.dof_labels, face)
    result = jnp.zeros(
        (basis.local_dof_count, points.shape[0], len(blades)), dtype=points.dtype
    )
    if not indices:
        return reference, result
    if len(face) == len(reference_cell_topology(hybrid_kind(basis.family)).vertices):
        chart = reference
        if basis.family == "pyramid-trimmed":
            s = 1 - reference[:, 2]
            chart = jnp.stack(
                (
                    (reference[:, 0] - 0.5 * reference[:, 2]) / s,
                    (reference[:, 1] - 0.5 * reference[:, 2]) / s,
                    reference[:, 2],
                ),
                axis=-1,
            )
            jacobians = jnp.zeros((points.shape[0], 3, 3), dtype=points.dtype)
            jacobians = jacobians.at[:, 0, 0].set(s).at[:, 1, 1].set(s).at[:, 2, 2].set(1)
            jacobians = (
                jacobians.at[:, 0, 2]
                .set(0.5 - chart[:, 0])
                .at[:, 1, 2]
                .set(0.5 - chart[:, 1])
            )
            if basis.form_degree == 0:
                action = jnp.ones((points.shape[0], 1, 1), dtype=points.dtype)
            else:
                rows = jnp.asarray(blades, dtype=jnp.int32)
                minors = jacobians[:, rows[:, None, :, None], rows[None, :, None, :]]
                action = jnp.linalg.det(minors).transpose((0, 2, 1))
            chart_weights = quadrature / s**2
        else:
            action = jnp.broadcast_to(
                jnp.eye(len(blades), dtype=points.dtype),
                (points.shape[0], len(blades), len(blades)),
            )
            chart_weights = quadrature
        factors = basis.hybrid_factors
        if factors is None:
            raise ValueError(
                "Hybrid moment densities require their complete source owner."
            )
        powers = jnp.asarray(factors.body_test_exponents, dtype=jnp.int32)
        degree = max(max(alpha) for alpha in factors.body_test_exponents)
        table = jnp.cumprod(
            jnp.broadcast_to(chart[:, :, None], (*chart.shape, degree + 1))
            .at[:, :, 0]
            .set(1),
            axis=-1,
        )
        monomials = jnp.prod(
            table[:, jnp.arange(3, dtype=jnp.int32)[None, :], powers], axis=-1
        )
        density = contract("qm,mcb->qbc", monomials, factors.body_test_coefficients)
        block = contract("qbe,qec,q->bqc", density, action, chart_weights)
    elif len(face) == 1:
        block = (
            jnp.ones((1, points.shape[0], 1), dtype=points.dtype)
            * quadrature[None, :, None]
        )
    else:
        trace = trace_basis(basis.family, basis.form_degree, basis.order, face)
        interior = forms._interior_dof_indices(trace.dof_labels, len(face))
        entity = trace.entity_vertices[-1][0]
        _, weights = trace.functional_weights_at(entity, points, quadrature)
        pullback = np.asarray(
            [
                [forms._minor(jacobian, ambient, local) for ambient in blades]
                for local in combinations(range(points.shape[1]), basis.form_degree)
            ]
        )
        block = contract(
            "dql,lc->dqc", weights[jnp.asarray(interior)], jnp.asarray(pullback)
        )
    return reference, result.at[jnp.asarray(indices)].set(block)
