#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Sufficient exact monotonicity proofs for polynomial mapped cells."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations

from ..discretization._cell_geometry import _coordinate_coefficient_identity
from ..discretization._coordinate_enclosure import (
    add,
    bernstein_coefficients,
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
    raise ValueError("Mapped injectivity proof supports reference dimensions one through three.")


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
            tuple(symmetric[row][column] for column in range(size))
            for row in range(size)
        )
        positive = _strict_positive_controls(
            _determinant(principal), domain, dimension
        )
        if positive is None:
            return None
        controls.append(positive)
    return tuple(controls)


def prove_mapped_cell_injective(cell: EmbeddingCell, /) -> MappedInjectivityWitness | None:
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
            or cell.pyramid_reference_arguments
            != pyramid_reference_to_box_arguments()
            or len(cell.pyramid_normalizations) != len(cell.coordinates)
            or any(
                normalization.apex_trace is None
                or normalization.apex_value is None
                for normalization in cell.pyramid_normalizations
            )
        ):
            return None
        center_jacobian_expressions = tuple(
            tuple(derivative(expression, axis) for axis in range(dimension))
            for expression in cell.coordinates
        )
        normalized_jacobian_rows = []
        for expression in cell.coordinates:
            row = []
            for axis in range(dimension):
                normalization = normalize_pyramid_reference_derivative(
                    expression, axis
                )
                if (
                    normalization.apex_trace is None
                    or normalization.apex_value is None
                    or not isinstance(normalization.expression, dict)
                ):
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
            tuple(jacobian[row][axis] for axis in range(dimension))
            for row in rows
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
        controls = _positive_definite_controls(
            projected_jacobian, domain, dimension
        )
        if controls is not None:
            return MappedInjectivityWitness(
                rows, projection_matrix, controls
            )
    return None


__all__ = ["MappedInjectivityWitness", "prove_mapped_cell_injective"]
