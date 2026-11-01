#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Sufficient exact monotonicity proofs for polynomial mapped cells."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations, product

import numpy as np

from ..discretization._cell_geometry import _coordinate_coefficient_identity
from ..discretization._coordinate_enclosure import (
    add,
    axes,
    bernstein_coefficients,
    bernstein_degree,
    BernsteinDomain,
    compose_expression,
    constant,
    derivative,
    evaluate_expression,
    Expression,
    linear_combinations,
    multiply,
    normalize_pyramid_reference_derivative,
    pyramid_reference_to_box_arguments,
    RationalPolynomial,
    scale,
    subtract,
)
from ..discretization._exact_algebraic_geometry import ExactScalar, RadicalElement
from ..linalg._exact import determinant_small_algebra, solve_exact
from ._embedding_cells import EmbeddingCell


class _SingularProjection(ValueError):
    """Selected output-coordinate rows are singular at the exact cell centroid."""


@dataclass(frozen=True, slots=True)
class MappedInjectivityWitness:
    """Exact monotonicity proof for one source-bound coordinate projection.

    ``projection_rows`` selects ambient output coordinates. The exact square
    ``projection_matrix`` left-preconditions those coordinates; Bernstein
    principal-minor controls prove strict monotonicity over the reference cell.
    """

    projection_rows: tuple[int, ...]
    projection_matrix: tuple[tuple[ExactScalar, ...], ...]
    principal_minor_controls: tuple[tuple[str, ...], ...]
    denominator_controls: tuple[tuple[str, ...], ...] = ()


def _sign(value: ExactScalar, /) -> int:
    if isinstance(value, RadicalElement):
        return value.sign()
    return (value > 0) - (value < 0)


def _determinant(matrix: tuple[tuple[Expression, ...], ...]) -> Expression:
    if len(matrix) == 1:
        return matrix[0][0]
    if len(matrix) == 2:
        return subtract(
            multiply(matrix[0][0], matrix[1][1]),
            multiply(matrix[0][1], matrix[1][0]),
        )
    if len(matrix) == 3:
        a, b, c = matrix[0]
        d, e, f = matrix[1]
        g, h, i = matrix[2]
        return add(
            subtract(
                multiply(a, subtract(multiply(e, i), multiply(f, h))),
                multiply(b, subtract(multiply(d, i), multiply(f, g))),
            ),
            multiply(c, subtract(multiply(d, h), multiply(e, g))),
        )
    raise ValueError(
        "Mapped injectivity proof supports reference dimensions one through three."
    )


def _strict_positive_controls(
    polynomial: Expression, domain: BernsteinDomain, dimension: int
) -> tuple[str, ...] | None:
    if not isinstance(polynomial, dict):
        return None
    controls = bernstein_coefficients(polynomial, domain, dimension)
    if not controls or any(_sign(value) <= 0 for value in controls):
        return None
    return tuple(repr(_coordinate_coefficient_identity(value)) for value in controls)


def _positive_definite_controls(
    jacobian: tuple[tuple[Expression, ...], ...],
    domain: BernsteinDomain,
    dimension: int,
    /,
) -> tuple[tuple[str, ...], ...] | None:
    symmetric = tuple(
        tuple(
            scale(
                add(jacobian[first][second], jacobian[second][first]),
                Fraction(1, 2),
            )
            for second in range(dimension)
        )
        for first in range(dimension)
    )
    controls: list[tuple[str, ...]] = []
    for size in range(1, dimension + 1):
        principal = tuple(
            tuple(symmetric[row][column] for column in range(size)) for row in range(size)
        )
        positive = _strict_positive_controls(_determinant(principal), domain, dimension)
        if positive is None:
            return None
        controls.append(positive)
    return tuple(controls)


def _positive_denominator(
    denominator: Mapping[tuple[int, ...], ExactScalar],
    domain: BernsteinDomain,
    dimension: int,
    /,
) -> tuple[dict, tuple[str, ...]] | None:
    """Prove one rational denominator has one strict sign on the whole cell."""
    polynomial = dict(denominator)
    controls = _strict_positive_controls(polynomial, domain, dimension)
    if controls is not None:
        return polynomial, controls
    negative = scale(polynomial, -1)
    controls = _strict_positive_controls(negative, domain, dimension)
    if controls is None:
        return None
    return negative, controls


def _polynomial_scaled_jacobian(
    jacobian: tuple[tuple[Expression, ...], ...],
    domain: BernsteinDomain,
    dimension: int,
    /,
) -> tuple[tuple[tuple[Expression, ...], ...], tuple[tuple[str, ...], ...]] | None:
    """Clear denominators with one proved-positive common factor for all entries."""
    denominators: list[dict] = []
    evidence: list[tuple[str, ...]] = []
    normalized: list[list[tuple[dict, dict | None]]] = []
    for row in jacobian:
        normalized_row = []
        for entry in row:
            if not isinstance(entry, RationalPolynomial):
                normalized_row.append((entry, None))
                continue
            result = _positive_denominator(entry.denominator, domain, dimension)
            if result is None:
                return None
            denominator, controls = result
            numerator = dict(entry.numerator)
            if denominator != dict(entry.denominator):
                numerator = scale(numerator, -1)
            normalized_row.append((numerator, denominator))
            if denominator not in denominators:
                denominators.append(denominator)
                evidence.append(controls)
        normalized.append(normalized_row)
    output = []
    for row in normalized:
        values = []
        for numerator, denominator in row:
            value = numerator
            for factor in denominators:
                if denominator is None or factor != denominator:
                    value = multiply(value, factor)
            values.append(value)
        output.append(tuple(values))
    return tuple(output), tuple(evidence)


def prove_mapped_cell_injective(
    cell: EmbeddingCell, /
) -> MappedInjectivityWitness | None:
    """Prove injectivity by exact monotonicity after a constant linear projection.

    On a convex reference cell, positive definiteness of the projected
    Jacobian's symmetric part implies strict monotonicity. The projection
    preconditioner is solved exactly at the reference-cell centroid, then its
    Bernstein principal-minor controls prove positivity across the whole cell.
    """
    if cell.affine:
        return MappedInjectivityWitness((), (), ())
    coordinates = cell.proof_coordinates
    dimension = cell.dimension
    ambient = len(coordinates)
    if dimension < 1 or dimension > 3 or ambient < dimension:
        return None
    domain = cell.proof_domain
    denominator_evidence: list[tuple[str, ...]] = []
    for expression in coordinates:
        if isinstance(expression, RationalPolynomial):
            proof = _positive_denominator(expression.denominator, domain, dimension)
            if proof is None:
                return None
            denominator_evidence.append(proof[1])
    if cell.pyramid_normalizations is None:
        if cell.kind == "pyramid":
            return None
        jacobian = tuple(
            tuple(derivative(expression, axis) for axis in range(dimension))
            for expression in coordinates
        )
        center_jacobian_expressions = jacobian
    else:
        if (
            cell.pyramid_reference_arguments is None
            or cell.pyramid_reference_arguments != pyramid_reference_to_box_arguments()
            or len(cell.pyramid_normalizations) != len(cell.coordinates)
            or any(
                normalization.apex_trace is None or normalization.apex_value is None
                for normalization in cell.pyramid_normalizations
            )
        ):
            return None
        center_jacobian_expressions = tuple(
            tuple(derivative(expression, axis) for axis in range(dimension))
            for expression in cell.coordinates
        )
        # Normalized Jacobian entries are polynomials on the closed collapsed
        # cube. A direction-dependent apex value (non-parallelogram base) is
        # admissible: monotonicity integrates the Jacobian along segments whose
        # interiors avoid the apex, and coordinates keep a unique apex value.
        normalized_jacobian_rows = []
        for expression in cell.coordinates:
            row = []
            for axis in range(dimension):
                normalization = normalize_pyramid_reference_derivative(expression, axis)
                if not isinstance(normalization.expression, dict):
                    return None
                row.append(normalization.expression)
            normalized_jacobian_rows.append(tuple(row))
        jacobian = tuple(normalized_jacobian_rows)
    cleared = _polynomial_scaled_jacobian(jacobian, domain, dimension)
    if cleared is None:
        return None
    jacobian, denominator_controls = cleared
    denominator_evidence.extend(denominator_controls)
    denominator_proof = tuple(denominator_evidence)
    identity: tuple[tuple[ExactScalar, ...], ...] = tuple(
        tuple(Fraction(int(row == column)) for column in range(dimension))
        for row in range(dimension)
    )

    for rows in combinations(range(ambient), dimension):
        direct = tuple(
            tuple(jacobian[row][axis] for axis in range(dimension)) for row in rows
        )
        controls = _positive_definite_controls(direct, domain, dimension)
        if controls is not None:
            return MappedInjectivityWitness(rows, identity, controls, denominator_proof)

    center = tuple(
        sum(
            (reference[axis] for reference in cell.reference_vertices),
            Fraction(0),
        )
        / len(cell.reference_vertices)
        for axis in range(dimension)
    )
    for rows in combinations(range(ambient), dimension):
        center_jacobian: tuple[tuple[ExactScalar, ...], ...] = tuple(
            tuple(
                evaluate_expression(center_jacobian_expressions[row][axis], center)
                for axis in range(dimension)
            )
            for row in rows
        )
        try:
            solved_projection = solve_exact(
                center_jacobian,
                identity,
                singular_error=_SingularProjection,
            )
        except _SingularProjection:
            continue
        projection_matrix: tuple[tuple[ExactScalar, ...], ...] = tuple(
            tuple(value for value in row) for row in solved_projection
        )
        projected_columns = tuple(
            linear_combinations(
                tuple(jacobian[row][axis] for row in rows),
                projection_matrix,
            )
            for axis in range(dimension)
        )
        projected_jacobian = tuple(
            tuple(projected_columns[axis][row] for axis in range(dimension))
            for row in range(dimension)
        )
        controls = _positive_definite_controls(projected_jacobian, domain, dimension)
        if controls is not None:
            return MappedInjectivityWitness(
                rows, projection_matrix, controls, denominator_proof
            )
    if ambient > dimension:
        # A curved embedded map can need a projection combining more than
        # ``dimension`` ambient axes. Its exact centroid left inverse supplies
        # the candidate; the same whole-cell Bernstein test proves it.
        center_rows = tuple(
            tuple(
                evaluate_expression(center_jacobian_expressions[row][axis], center)
                for axis in range(dimension)
            )
            for row in range(ambient)
        )
        gram = tuple(
            tuple(
                sum(
                    (center_rows[row][i] * center_rows[row][j] for row in range(ambient)),
                    Fraction(0),
                )
                for j in range(dimension)
            )
            for i in range(dimension)
        )
        transpose = tuple(
            tuple(center_rows[row][axis] for row in range(ambient))
            for axis in range(dimension)
        )
        try:
            projection = solve_exact(gram, transpose, singular_error=_SingularProjection)
        except _SingularProjection:
            projection = None
        if projection is not None:
            projection_matrix = tuple(tuple(row) for row in projection)
            projected_columns = tuple(
                linear_combinations(
                    tuple(jacobian[row][axis] for row in range(ambient)),
                    projection_matrix,
                )
                for axis in range(dimension)
            )
            projected_jacobian = tuple(
                tuple(projected_columns[axis][row] for axis in range(dimension))
                for row in range(dimension)
            )
            controls = _positive_definite_controls(projected_jacobian, domain, dimension)
            if controls is not None:
                return MappedInjectivityWitness(
                    tuple(range(ambient)), projection_matrix, controls, denominator_proof
                )
    # The centroid preconditioner fails on strongly curved cells; search a constant
    # preconditioner against the Jacobian's Bernstein control matrices, on uniform
    # subdivisions of the reference domain when the whole-cell controls are too
    # loose (their hull holds no common monotone direction).
    for level in range(SUBDIVISION_LEVELS + 1):
        if level and _child_maps(domain, dimension) is None:
            break
        for rows in combinations(range(ambient), dimension):
            matrices = subdivided_jacobian_controls(
                tuple(
                    tuple(jacobian[row][axis] for axis in range(dimension))
                    for row in rows
                ),
                domain,
                dimension,
                level,
            )
            if matrices is None:
                return None
            found = monotone_preconditioner(matrices)
            if found is not None:
                return MappedInjectivityWitness(
                    rows, found[0], found[1], denominator_proof
                )
    return None


SUBDIVISION_LEVELS = 2
_HALF = Fraction(1, 2)
_AffineMap = tuple[tuple[Fraction, ...], tuple[tuple[Fraction, ...], ...]]


def subdivided_jacobian_controls(
    jacobian: tuple[tuple[Expression, ...], ...],
    domain: BernsteinDomain,
    dimension: int,
    level: int,
    /,
) -> tuple[_ExactMatrix, ...] | None:
    """Jacobian control matrices over ``level`` uniform refinements of the domain.

    Each piece's controls enclose the Jacobian values on that piece, so their
    union encloses every value on the cell, more tightly as the level grows. A
    domain without a uniform refinement keeps its whole-cell controls.
    """
    maps = _subdivision_maps(domain, dimension, level)
    if maps is None:
        return jacobian_control_matrices(jacobian, domain, dimension)
    matrices: list[_ExactMatrix] = []
    for offset, linear in maps:
        arguments = _affine_arguments(offset, linear)
        piece = jacobian_control_matrices(
            tuple(
                tuple(compose_expression(entry, arguments) for entry in row)
                for row in jacobian
            )
            if level
            else jacobian,
            domain,
            dimension,
        )
        if piece is None:
            return None
        matrices.extend(piece)
    return tuple(matrices)


def _scaled_identity(
    dimension: int, value: Fraction, /
) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(
        tuple(value if row == column else Fraction(0) for column in range(dimension))
        for row in range(dimension)
    )


def _child_maps(domain: str, dimension: int, /) -> tuple[_AffineMap, ...] | None:
    """One uniform refinement of the standard domain into affine children."""
    zero = Fraction(0)
    if domain == "box":
        return tuple(
            (offsets, _scaled_identity(dimension, _HALF))
            for offsets in product((zero, _HALF), repeat=dimension)
        )
    triangle = (
        ((zero, zero), _scaled_identity(2, _HALF)),
        ((_HALF, zero), _scaled_identity(2, _HALF)),
        ((zero, _HALF), _scaled_identity(2, _HALF)),
        ((_HALF, _HALF), _scaled_identity(2, -_HALF)),
    )
    if domain == "simplex" and dimension == 1:
        return (((zero,), ((_HALF,),)), ((_HALF,), ((_HALF,),)))
    if domain == "simplex" and dimension == 2:
        return triangle
    if domain == "prism" and dimension == 3:
        return tuple(
            (
                (*offset, height),
                (
                    (*linear[0], zero),
                    (*linear[1], zero),
                    (zero, zero, _HALF),
                ),
            )
            for offset, linear in triangle
            for height in (zero, _HALF)
        )
    return None


def _subdivision_maps(
    domain: str, dimension: int, level: int, /
) -> tuple[_AffineMap, ...] | None:
    maps: tuple[_AffineMap, ...] = (
        (
            tuple(Fraction(0) for _ in range(dimension)),
            _scaled_identity(dimension, Fraction(1)),
        ),
    )
    if not level:
        return maps
    children = _child_maps(domain, dimension)
    if children is None:
        return None
    for _ in range(level):
        maps = tuple(
            (
                tuple(
                    offset[row]
                    + sum(
                        (
                            linear[row][inner] * child[0][inner]
                            for inner in range(dimension)
                        ),
                        Fraction(0),
                    )
                    for row in range(dimension)
                ),
                tuple(
                    tuple(
                        sum(
                            (
                                linear[row][inner] * child[1][inner][column]
                                for inner in range(dimension)
                            ),
                            Fraction(0),
                        )
                        for column in range(dimension)
                    )
                    for row in range(dimension)
                ),
            )
            for offset, linear in maps
            for child in children
        )
    return maps


def _affine_arguments(
    offset: tuple[Fraction, ...], linear: tuple[tuple[Fraction, ...], ...], /
) -> tuple[Expression, ...]:
    dimension = len(offset)
    variables = axes(dimension)
    arguments: list[Expression] = []
    for row in range(dimension):
        expression: Expression = constant(offset[row], dimension)
        for column in range(dimension):
            if linear[row][column]:
                expression = add(
                    expression, scale(variables[column], linear[row][column])
                )
        arguments.append(expression)
    return tuple(arguments)


_ExactMatrix = tuple[tuple[ExactScalar, ...], ...]


def jacobian_control_matrices(
    jacobian: tuple[tuple[Expression, ...], ...],
    domain: BernsteinDomain,
    dimension: int,
    /,
) -> tuple[_ExactMatrix, ...] | None:
    """Bernstein control matrices of a square polynomial Jacobian at one degree.

    Every value of the Jacobian on the domain is a convex combination of these
    matrices (all entries share the elevated degree and control index).
    """
    entries = [entry for row in jacobian for entry in row if isinstance(entry, dict)]
    if len(entries) != sum(len(row) for row in jacobian):
        return None
    degrees = [bernstein_degree(entry, domain, dimension) for entry in entries]
    degree = tuple(max(values) for values in zip(*degrees, strict=True))
    columns = [
        bernstein_coefficients(entry, domain, dimension, degree=degree)
        for entry in entries
    ]
    size = len(jacobian)
    return tuple(
        tuple(
            tuple(columns[row * size + column][index] for column in range(size))
            for row in range(size)
        )
        for index in range(len(columns[0]))
    )


def _exact_determinant(matrix: _ExactMatrix, /) -> ExactScalar:
    if len(matrix) == 1:
        return matrix[0][0]
    total: ExactScalar = Fraction(0)
    for column, value in enumerate(matrix[0]):
        minor = tuple(
            tuple(entry for index, entry in enumerate(row) if index != column)
            for row in matrix[1:]
        )
        term = value * _exact_determinant(minor)
        total = total + term if column % 2 == 0 else total - term
    return total


# Primal feasibility tolerance of the Kelley cutting-plane programs (controls are
# normalized to unit magnitude and ``M`` to the unit box, so it is absolute).
_KELLEY_FEASIBILITY_TOLERANCE = 1.0e-7


def _least_eigenpairs(
    matrix: np.ndarray, controls: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    """Least eigenvalue and eigenvector of every ``sym(M C_k)``."""
    products = matrix @ controls
    values, vectors = np.linalg.eigh(0.5 * (products + np.swapaxes(products, 1, 2)))
    return values[:, 0], vectors[:, :, 0]


def _seed_indices(count: int, /) -> np.ndarray:
    return np.unique(np.linspace(0, count - 1, 8).astype(int))


def _search_starts(controls: np.ndarray, /) -> list[np.ndarray]:
    """Unit-box inverses of the mean and of the seed controls."""
    # Seeds only start the cuts (the program is complete from any seed).
    starts = []
    for matrix in (controls.mean(axis=0), *controls[_seed_indices(len(controls))]):
        try:
            inverse = np.linalg.inv(matrix)
        except np.linalg.LinAlgError:
            continue
        if np.all(np.isfinite(inverse)):
            starts.append(inverse / np.max(np.abs(inverse)))
    return starts


def _singular_combination(
    first: _ExactMatrix, second: _ExactMatrix, weight: Fraction, /
) -> bool:
    """Exactly decide a singular matrix on the segment from ``second`` to ``first``.

    A zero of ``det(first)``, or a determinant at ``weight first + (1 - weight)
    second`` that is zero or of the opposite sign (a root on ``[weight, 1]``).
    """
    sign = _sign(_exact_determinant(first))
    if sign == 0:
        return True
    complement = 1 - weight
    combination = tuple(
        tuple(
            weight * left + complement * right
            for left, right in zip(first_row, second_row, strict=True)
        )
        for first_row, second_row in zip(first, second, strict=True)
    )
    return _sign(_exact_determinant(combination)) != sign


def _singular_control_hull(
    controls: tuple[_ExactMatrix, ...], floats: np.ndarray, /
) -> bool:
    """Exactly prove that no constant monotone preconditioner exists.

    ``sym(M C)`` positive definite makes ``M C`` nonsingular, and the positive
    definite cone is convex, so a preconditioner keeps every convex combination
    of the controls nonsingular. A negative real eigenvalue ``lambda`` of
    ``B^-1 A`` puts the singular matrix ``A - lambda B`` on the segment, at
    ``t A + (1 - t) B`` with ``t = 1 / (1 - lambda)``. Floats propose the pairs
    and ``t``; `_singular_combination` decides each proposal exactly. Partners
    are the search seeds, then every control on which the best seeded
    preconditioner is not positive definite, worst first (the controls that
    refute it). Only an exact singular combination returns ``True``.
    """
    if floats.size == 0 or not np.all(np.isfinite(floats)):
        return False
    magnitude = float(np.max(np.abs(floats)))
    if magnitude == 0.0:
        return False
    floats = floats / magnitude
    seeds = _seed_indices(len(floats))
    partners = list(seeds)
    starts = _search_starts(floats)
    if starts:
        margins = max(
            (_least_eigenpairs(start, floats)[0] for start in starts),
            key=lambda values: float(np.min(values)),
        )
        refuting = np.flatnonzero(margins <= 0.0)
        tried = set(partners)
        partners.extend(
            int(index)
            for index in refuting[np.argsort(margins[refuting], kind="stable")]
            if int(index) not in tried
        )
    for partner in partners:
        try:
            inverse = np.linalg.inv(floats[partner])
        except np.linalg.LinAlgError:
            continue
        if not np.all(np.isfinite(inverse)):
            continue
        eigenvalues = np.linalg.eigvals(inverse @ floats)
        hits = np.argwhere((eigenvalues.imag == 0.0) & (eigenvalues.real < 0.0))
        order = np.argsort(eigenvalues.real[hits[:, 0], hits[:, 1]], kind="stable")
        for control, column in hits[order]:
            weight = Fraction(1.0 / (1.0 - float(eigenvalues.real[control, column])))
            if 0 < weight < 1 and _singular_combination(
                controls[int(control)], controls[partner], weight
            ):
                return True
    return False


def _float_search_preconditioner(controls: np.ndarray, /) -> np.ndarray | None:
    """A box-bounded ``M`` maximizing ``min_k lambda_min(sym(M C_k))``, if positive.

    The objective is concave in ``M`` (a minimum of least eigenvalues of linear
    symmetric matrices). Kelley's cutting planes ``t <= v^T M C_k v`` at
    least-eigenvalue vectors ``v`` give linear programs whose optima bound it
    from above: an optimum ``t <= 0`` proves that no preconditioner exists;
    otherwise every iterate cuts off the previous program point, and the search
    stops once the attained margin is positive and within half the bound, the
    bound and the attained margin agree to binary64 resolution, or the new cuts
    are violated by no more than the program's primal feasibility tolerance (the
    program point cannot move). Inverses of the mean and of each control seed
    the cuts. Floats only propose ``M``; the caller decides positivity exactly,
    and ``None`` only means that no proposal was found.
    """
    from scipy.optimize import linprog

    if controls.size == 0 or not np.all(np.isfinite(controls)):
        return None
    magnitude = float(np.max(np.abs(controls)))
    if magnitude == 0.0:
        return None
    controls = controls / magnitude
    # sym(M C) positive definite forces sign(det C) = sign(det M) for every C.
    determinants = np.linalg.det(controls)
    if not (np.all(determinants > 0.0) or np.all(determinants < 0.0)):
        return None
    size = controls.shape[1]

    def margins(matrix: np.ndarray, /) -> tuple[np.ndarray, np.ndarray]:
        return _least_eigenpairs(matrix, controls)

    def cuts(vectors: np.ndarray, selected: np.ndarray, /) -> list[np.ndarray]:
        # Row of t - v^T M (C v) <= 0 over the unknowns (M row-major, t).
        applied = np.einsum("kij,kj->ki", controls[selected], vectors[selected])
        coefficients = np.einsum("ki,kj->kij", vectors[selected], applied)
        return [np.concatenate((-row.ravel(), (1.0,))) for row in coefficients]

    starts = _search_starts(controls)
    if not starts:
        return None
    best = max(starts, key=lambda matrix: float(np.min(margins(matrix)[0])))
    values, vectors = margins(best)
    best_margin = float(np.min(values))
    rows = cuts(vectors, np.arange(values.size))
    objective = np.concatenate((np.zeros(size * size), (-1.0,)))
    bounds = [(-1.0, 1.0)] * (size * size) + [(None, None)]
    while True:
        solution = linprog(
            objective,
            A_ub=np.asarray(rows),
            b_ub=np.zeros(len(rows)),
            bounds=bounds,
            method="highs",
            options={"primal_feasibility_tolerance": _KELLEY_FEASIBILITY_TOLERANCE},
        )
        point = solution.x
        if not solution.success or point is None:
            return best if best_margin > 0.0 else None
        bound = float(point[-1])
        if bound <= 0.0:
            return None
        candidate = np.asarray(point[:-1]).reshape(size, size)
        values, vectors = margins(candidate)
        if float(np.min(values)) > best_margin:
            best, best_margin = candidate, float(np.min(values))
        if best_margin > 0.0 and best_margin >= 0.5 * bound:
            return best
        if bound - best_margin <= 1.0e-12 * max(1.0, bound):
            return best if best_margin > 0.0 else None
        # Cuts violated by no more than the feasibility tolerance leave the
        # program point in place: further rounds cannot tighten the bound.
        if bound - float(np.min(values)) <= _KELLEY_FEASIBILITY_TOLERANCE:
            return best if best_margin > 0.0 else None
        rows.extend(cuts(vectors, np.flatnonzero(values < bound)))


def monotone_preconditioner(
    controls: tuple[_ExactMatrix, ...], /
) -> tuple[_ExactMatrix, tuple[tuple[str, ...], ...]] | None:
    """An exact ``M`` with ``sym(M C)`` positive definite at every control ``C``.

    The positive definite cone is convex, so ``sym(M J(x))`` is then positive
    definite for every convex combination ``J(x)`` of the controls. Returns
    ``M`` with the leading principal minors of every ``sym(M C)`` (per size).
    """
    if not controls:
        return None
    size = len(controls[0])
    floats = np.asarray(
        [[[float(value) for value in row] for row in matrix] for matrix in controls],
        dtype=np.float64,
    )
    # Cheap exact refutation first: the cutting-plane search spends nearly all of
    # its programs proving that no preconditioner exists.
    if _singular_control_hull(controls, floats):
        return None
    proposal = _float_search_preconditioner(floats)
    if proposal is None:
        return None
    scale = float(np.max(np.abs(proposal)))
    matrix: _ExactMatrix = tuple(
        tuple(Fraction(round(float(value) / scale * 2.0**30)) for value in row)
        for row in proposal
    )
    integer_matrix = tuple(tuple(int(value) for value in row) for row in matrix)
    minors: list[list[str]] = [[] for _ in range(size)]
    for control in controls:
        values = _rational_leading_minors(integer_matrix, control)
        if values is None:
            values = _leading_minors(matrix, control)
        for order, minor in enumerate(values):
            if _sign(minor) <= 0:
                return None
            minors[order].append(repr(_coordinate_coefficient_identity(minor)))
    return matrix, tuple(tuple(values) for values in minors)


def _leading_minors(
    matrix: _ExactMatrix, control: _ExactMatrix, /
) -> tuple[ExactScalar, ...]:
    """Leading principal minors of ``sym(M C)`` in exact scalars."""
    size = len(control)
    product = tuple(
        tuple(
            sum(
                (matrix[row][inner] * control[inner][column] for inner in range(size)),
                Fraction(0),
            )
            for column in range(size)
        )
        for row in range(size)
    )
    symmetric = tuple(
        tuple((product[row][column] + product[column][row]) / 2 for column in range(size))
        for row in range(size)
    )
    return tuple(
        _exact_determinant(tuple(row[:order] for row in symmetric[:order]))
        for order in range(1, size + 1)
    )


def _rational_leading_minors(
    matrix: tuple[tuple[int, ...], ...], control: _ExactMatrix, /
) -> tuple[Fraction, ...] | None:
    """`_leading_minors` of a rational control over one shared denominator.

    With ``C = N / d`` (integer ``N``), ``2 d sym(M C) = M N + (M N)^T`` is an
    integer matrix, and its order-``k`` minor over ``(2 d)^k`` is the exact
    minor. ``None`` when an entry is not rational.
    """
    rational_rows: list[tuple[Fraction, ...]] = []
    for row in control:
        rational_row: list[Fraction] = []
        for value in row:
            if not isinstance(value, Fraction | int):
                return None
            rational_row.append(Fraction(value))
        rational_rows.append(tuple(rational_row))
    denominator = math.lcm(*(value.denominator for row in rational_rows for value in row))
    numerators = tuple(
        tuple(value.numerator * (denominator // value.denominator) for value in row)
        for row in rational_rows
    )
    size = len(control)
    product = tuple(
        tuple(
            sum(matrix[row][inner] * numerators[inner][column] for inner in range(size))
            for column in range(size)
        )
        for row in range(size)
    )
    doubled = tuple(
        tuple(product[row][column] + product[column][row] for column in range(size))
        for row in range(size)
    )
    return tuple(
        Fraction(
            determinant_small_algebra(
                tuple(row[:order] for row in doubled[:order]),
                add=int.__add__,
                subtract=int.__sub__,
                multiply=int.__mul__,
            ),
            (2 * denominator) ** order,
        )
        for order in range(1, size + 1)
    )


__all__ = [
    "jacobian_control_matrices",
    "MappedInjectivityWitness",
    "monotone_preconditioner",
    "prove_mapped_cell_injective",
    "SUBDIVISION_LEVELS",
    "subdivided_jacobian_controls",
]
