#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Sufficient exact monotonicity proofs for polynomial mapped cells."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations

import numpy as np

from ..discretization._cell_geometry import _coordinate_coefficient_identity
from ..discretization._coordinate_enclosure import (
    add,
    bernstein_coefficients,
    bernstein_degree,
    BernsteinDomain,
    derivative,
    evaluate_expression,
    Expression,
    linear_combinations,
    multiply,
    normalize_pyramid_reference_derivative,
    pyramid_reference_to_box_arguments,
    scale,
    subtract,
)
from ..discretization._exact_algebraic_geometry import ExactScalar, RadicalElement
from ..linalg._exact import solve_exact
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
    coordinates = tuple(
        expression
        for expression in cell.proof_coordinates
        if isinstance(expression, dict)
    )
    if len(coordinates) != len(cell.proof_coordinates):
        return None
    dimension = cell.dimension
    ambient = len(coordinates)
    if dimension < 1 or dimension > 3 or ambient < dimension:
        return None
    domain = cell.proof_domain
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
            return MappedInjectivityWitness(rows, identity, controls)

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
            return MappedInjectivityWitness(rows, projection_matrix, controls)
    # The centroid preconditioner fails on strongly curved cells; search a constant
    # preconditioner against the Jacobian's own Bernstein control matrices.
    for rows in combinations(range(ambient), dimension):
        matrices = jacobian_control_matrices(
            tuple(
                tuple(jacobian[row][axis] for axis in range(dimension)) for row in rows
            ),
            domain,
            dimension,
        )
        found = None if matrices is None else monotone_preconditioner(matrices)
        if found is not None:
            return MappedInjectivityWitness(rows, found[0], found[1])
    return None


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


def _float_search_preconditioner(controls: np.ndarray, /) -> np.ndarray | None:
    """Unit-norm ``M`` maximizing the least eigenvalue of every ``sym(M C_k)``.

    The objective is concave in ``M``; candidates are the inverses of the mean
    and of each control, refined by soft-min supergradient ascent. Floats only
    propose ``M``; the caller decides positivity exactly.
    """
    if not np.all(np.isfinite(controls)):
        return None

    def margins(matrix: np.ndarray, /) -> tuple[np.ndarray, np.ndarray]:
        products = matrix @ controls
        values, vectors = np.linalg.eigh(0.5 * (products + np.swapaxes(products, 1, 2)))
        return values[:, 0], vectors[:, :, 0]

    starts = []
    for matrix in (controls.mean(axis=0), *controls):
        try:
            inverse = np.linalg.inv(matrix)
        except np.linalg.LinAlgError:
            continue
        if np.all(np.isfinite(inverse)):
            starts.append(inverse / np.linalg.norm(inverse))
    if not starts:
        return None
    current = max(starts, key=lambda matrix: float(np.min(margins(matrix)[0])))
    best, best_margin = current, float(np.min(margins(current)[0]))
    for step in range(400):
        if best_margin > 0.0 and step >= 50:
            break
        values, vectors = margins(current)
        spread = max(float(np.ptp(values)), 1.0e-12)
        weights = np.exp(-(values - values.min()) / (0.05 * spread))
        weights /= weights.sum()
        gradient = np.einsum(
            "k,ki,kj->ij", weights, vectors, np.einsum("kij,kj->ki", controls, vectors)
        )
        current = current + 0.2 / np.sqrt(1.0 + step) * gradient
        current = current / np.linalg.norm(current)
        margin = float(np.min(margins(current)[0]))
        if margin > best_margin:
            best, best_margin = current, margin
    return best if best_margin > 0.0 else None


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
    proposal = _float_search_preconditioner(
        np.asarray(
            [[[float(value) for value in row] for row in matrix] for matrix in controls],
            dtype=np.float64,
        )
    )
    if proposal is None:
        return None
    scale = float(np.max(np.abs(proposal)))
    matrix: _ExactMatrix = tuple(
        tuple(Fraction(round(float(value) / scale * 2.0**30)) for value in row)
        for row in proposal
    )
    minors: list[list[str]] = [[] for _ in range(size)]
    for control in controls:
        product = tuple(
            tuple(
                sum(
                    (
                        matrix[row][inner] * control[inner][column]
                        for inner in range(size)
                    ),
                    Fraction(0),
                )
                for column in range(size)
            )
            for row in range(size)
        )
        symmetric = tuple(
            tuple(
                (product[row][column] + product[column][row]) / 2
                for column in range(size)
            )
            for row in range(size)
        )
        for order in range(1, size + 1):
            minor = _exact_determinant(tuple(row[:order] for row in symmetric[:order]))
            if _sign(minor) <= 0:
                return None
            minors[order - 1].append(repr(_coordinate_coefficient_identity(minor)))
    return matrix, tuple(tuple(values) for values in minors)


__all__ = [
    "jacobian_control_matrices",
    "MappedInjectivityWitness",
    "monotone_preconditioner",
    "prove_mapped_cell_injective",
]
