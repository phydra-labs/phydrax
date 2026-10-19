#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Mapped measure and planar facet-containment calculus in rational arithmetic.

Source polynomials determine the measure. A zero plane polynomial establishes
planarity over the entire face.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from typing import assert_never

import numpy as np

import phydrax.discretization._coordinate_enclosure as algebra
import phydrax.discretization._reference_subdivision as reference_subdivision_owner
from phydrax.discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    reserve_polynomial,
)
from phydrax.discretization._exact_algebraic_geometry import rational_coefficient
from phydrax.discretization._reference_cell_types import CellGeometryElement

from ..discretization._reference_cell import reference_cell_topology
from ..typing import checked, parse


@dataclass
class SubdivisionLedger:
    """Track charged subdivision work for one certificate request.

    ``pieces`` counts every examined piece (including a refused over-budget
    piece), ``candidate_pairs`` every evaluated piece-source pair, and
    ``maximum_depth`` the deepest subdivision level examined.
    """

    pieces: int = 0
    candidate_pairs: int = 0
    maximum_depth: int = 0

    def charge(self, depth: int, /) -> None:
        self.pieces += 1
        self.maximum_depth = max(self.maximum_depth, depth)


@checked
def integrate(
    polynomial: algebra.Polynomial,
    domain: algebra.BernsteinDomain,
    dimension: int,
    *,
    budget: CoordinateEnclosureBudget,
) -> Fraction:
    """Integrate power monomials exactly, including the collapsed pyramid cube."""

    numerator, denominator = algebra.coefficient_profile((polynomial,), budget=budget)
    budget.reserve(dimension * len(polynomial))
    # Every integration weight divides by products bounded by
    # (sum(index) + dimension)!. Its log bound follows n! <= n**n;
    # summing these bit bounds also bounds the accumulated denominator.
    weight_bits = sum(
        (sum(index) + dimension) * max(sum(index) + dimension, 1).bit_length()
        for index in polynomial
    )
    reserve_polynomial(
        2 * len(polynomial),
        1,
        0,
        numerator + denominator + weight_bits + max(len(polynomial), 1).bit_length(),
        budget=budget,
    )
    result = Fraction(0)
    if polynomial:
        domain = parse(domain, algebra.BernsteinDomain, "integrate.domain")
    for index, coefficient in polynomial.items():
        match domain:
            case "simplex":
                weight = Fraction(
                    math.prod(math.factorial(i) for i in index),
                    math.factorial(sum(index) + dimension),
                )
            case "prism":
                weight = Fraction(
                    math.factorial(index[0]) * math.factorial(index[1]),
                    math.factorial(index[0] + index[1] + 2) * (index[2] + 1),
                )
            case "box":
                weight = Fraction(1, math.prod(i + 1 for i in index))
            case unreachable:
                assert_never(unreachable)
        result += (
            rational_coefficient(coefficient, role="mapped_coverage.integrand") * weight
        )
    return result


@checked
def enclosure(value: Fraction) -> tuple[float, float]:
    rounded = float(value)
    exact = Fraction(rounded)
    return (
        float(np.nextafter(rounded, -math.inf)) if exact > value else rounded,
        float(np.nextafter(rounded, math.inf)) if exact < value else rounded,
    )


def map_domain(kind: str) -> algebra.BernsteinDomain:
    return (
        "simplex"
        if kind in ("triangle", "tetrahedron")
        else "prism"
        if kind == "prism"
        else "box"
    )


@checked
def face_charts(
    coordinates: tuple[algebra.Expression, ...],
    kind: str,
    face: tuple[int, ...],
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[tuple[tuple[algebra.Expression, ...], algebra.BernsteinDomain], ...]:
    topology = reference_cell_topology(kind)
    corners = np.asarray(topology.vertices, dtype=np.float64)[
        np.asarray(face, dtype=np.int64)
    ]
    if kind == "pyramid" and 4 in face:
        # Cube-side parameterizations preserve the physical outward orientation.
        sides: dict[
            tuple[int, ...],
            tuple[
                tuple[int, int, int],
                tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
            ],
        ] = {
            (0, 1, 4): ((0, 0, 0), ((1, 0), (0, 0), (0, 1))),
            (1, 2, 4): ((1, 0, 0), ((0, 0), (1, 0), (0, 1))),
            (2, 3, 4): ((1, 1, 0), ((-1, 0), (0, 0), (0, 1))),
            (3, 0, 4): ((0, 1, 0), ((0, 0), (-1, 0), (0, 1))),
        }
        side = sides[face]
        arguments = algebra.affine_arguments(
            np.asarray(side[0], dtype=np.float64),
            np.asarray(side[1], dtype=np.float64),
            budget=budget,
        )
        composition = algebra.ExpressionComposition(arguments, budget=budget)
        return (
            (tuple(composition(value, budget=budget) for value in coordinates), "box"),
        )
    pieces = (corners,) if len(face) <= 3 else (corners[[0, 1, 2]], corners[[0, 2, 3]])
    result: list[tuple[tuple[algebra.Expression, ...], algebra.BernsteinDomain]] = []
    for piece in pieces:
        arguments = algebra.affine_arguments(
            piece[0], (piece[1:] - piece[0]).T, budget=budget
        )
        composition = algebra.ExpressionComposition(arguments, budget=budget)
        result.append(
            (tuple(composition(value, budget=budget) for value in coordinates), "simplex")
        )
    return tuple(result)


@checked
def prepared_face_charts(
    element: CellGeometryElement,
    local: algebra.CoordinateCoefficients,
    kind: str,
    face: tuple[int, ...],
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[tuple[tuple[algebra.Expression, ...], algebra.BernsteinDomain], ...] | None:
    """Reuse oriented source-face charts in the request's coefficient ledger."""

    with budget.temporary_scope():
        coefficients, key = algebra.coordinate_preparation_key(
            element, local, budget=budget
        )
        budget.reserve(len(face))
        face_key = (
            (key[0], key[1], key[2], kind, face)
            if len(key) == 3
            else (key[0], key[1], key[2], key[3], kind, face)
        )
        cached = budget.face_chart_cache.get(face_key)
        if cached is not None:
            return cached
        coordinates = budget.coordinate_cache.get(key)
        if coordinates is None:
            coordinates = algebra.coordinate_expressions(
                element, coefficients, budget=budget
            )
        if coordinates is None:
            return None
        charts = face_charts(coordinates, kind, face, budget=budget)
        budget.retain_basis((face_key, charts))
        budget.face_chart_cache[face_key] = charts
        return charts


@checked
def face_flux(
    coordinates: tuple[algebra.Expression, ...],
    kind: str,
    face: tuple[int, ...],
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> Fraction | None:
    """Integrate the oriented face before containment triangulation."""
    if len(face) == 4:
        corners = np.asarray(reference_cell_topology(kind).vertices, dtype=np.float64)[
            np.asarray(face)
        ]
        # Reference-cell vertices establish the parallelogram parameterization.
        # Prove the reference parallelogram before using its full box chart.
        if np.array_equal(corners[2], corners[1] + corners[3] - corners[0]):
            arguments = algebra.affine_arguments(
                corners[0],
                np.stack((corners[1] - corners[0], corners[3] - corners[0]), axis=1),
                budget=budget,
            )
            composition = algebra.ExpressionComposition(arguments, budget=budget)
            return chart_flux(
                tuple(composition(value, budget=budget) for value in coordinates),
                "box",
                budget=budget,
            )
    result = Fraction(0)
    for chart, domain in face_charts(coordinates, kind, face, budget=budget):
        contribution = chart_flux(chart, domain, budget=budget)
        if contribution is None:
            return None
        result += contribution
    return result


@checked
def prepared_face_flux(
    element: CellGeometryElement,
    local: algebra.CoordinateCoefficients,
    kind: str,
    face: tuple[int, ...],
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> Fraction | None:
    """Reuse oriented face flux in the request's coefficient ledger."""

    with budget.temporary_scope():
        coefficients, key = algebra.coordinate_preparation_key(
            element, local, budget=budget
        )
        budget.reserve(len(face))
        face_key = (
            (key[0], key[1], key[2], kind, face)
            if len(key) == 3
            else (key[0], key[1], key[2], key[3], kind, face)
        )
        if face_key in budget.face_flux_cache:
            cached = budget.face_flux_cache[face_key]
            return (
                None
                if cached is None
                else rational_coefficient(cached, role="mapped_coverage.face_flux")
            )
        coordinates = budget.coordinate_cache.get(key)
        if coordinates is None:
            coordinates = algebra.coordinate_expressions(
                element, coefficients, budget=budget
            )
        flux = (
            None
            if coordinates is None
            else face_flux(coordinates, kind, face, budget=budget)
        )
        budget.retain_basis((face_key, flux))
        budget.face_flux_cache[face_key] = flux
        return flux


@checked
def subdivision_children(
    domain: algebra.BernsteinDomain, dimension: int, *, budget: CoordinateEnclosureBudget
) -> tuple[tuple[algebra.Polynomial, ...], ...]:
    if dimension == 1:
        return tuple(
            algebra.affine_arguments(
                np.asarray((start,), dtype=np.float64),
                np.asarray(((0.5,),), dtype=np.float64),
                budget=budget,
            )
            for start in (0.0, 0.5)
        )
    if domain == "prism":
        variables = algebra.axes(3, budget=budget)
        return tuple(
            (
                *tuple(
                    algebra.compose(value, variables[:2], budget=budget)
                    for value in triangle
                ),
                algebra.compose(height[0], (variables[2],), budget=budget),
            )
            for triangle in subdivision_children("simplex", 2, budget=budget)
            for height in subdivision_children("box", 1, budget=budget)
        )
    if domain == "simplex":
        if dimension == 3:
            origins, matrices = reference_subdivision_owner.simplex_child_maps(dimension)
            return tuple(
                algebra.affine_arguments(origin, matrix, budget=budget)
                for origin, matrix in zip(origins, matrices, strict=True)
            )
        triangles = (
            ((0, 0), (0.5, 0), (0, 0.5)),
            ((0.5, 0), (1, 0), (0.5, 0.5)),
            ((0, 0.5), (0.5, 0.5), (0, 1)),
            ((0.5, 0), (0.5, 0.5), (0, 0.5)),
        )
        return tuple(
            algebra.affine_arguments(
                np.asarray(row[0], dtype=np.float64),
                (np.asarray(row[1:], dtype=np.float64) - row[0]).T,
                budget=budget,
            )
            for row in triangles
        )
    return tuple(
        algebra.affine_arguments(
            np.asarray(origin, dtype=np.float64),
            0.5 * np.eye(dimension, dtype=np.float64),
            budget=budget,
        )
        for origin in product((0.0, 0.5), repeat=dimension)
    )


@checked
def nonnegative(
    polynomial: algebra.Expression,
    domain: algebra.BernsteinDomain,
    dimension: int,
    maximum_nodes: int,
    maximum_depth: int,
    maximum_pieces: int,
    work: SubdivisionLedger,
    *,
    budget: CoordinateEnclosureBudget,
) -> bool:
    pending = [(polynomial, 0)]
    children: tuple[tuple[algebra.Polynomial, ...], ...] | None = None
    while pending:
        current, depth = pending.pop()
        work.charge(depth)
        if (
            work.pieces > maximum_pieces
            or algebra.expression_node_count(current, domain, dimension) > maximum_nodes
        ):
            return False
        coefficients = algebra.expression_bernstein_coefficients(
            current, domain, dimension, budget=budget
        )
        if min(coefficients) >= 0:
            continue
        if max(coefficients) < 0 or depth >= maximum_depth:
            return False
        if children is None:
            children = subdivision_children(domain, dimension, budget=budget)
        pending.extend(
            (algebra.expression_compose(current, child, budget=budget), depth + 1)
            for child in children
        )
    return True


@checked
def projected_jacobian(
    coordinates: tuple[algebra.Expression, ...],
    axes: tuple[int, ...],
    *,
    budget: CoordinateEnclosureBudget,
) -> algebra.Expression:
    return algebra.expression_determinant(
        tuple(
            tuple(
                algebra.expression_derivative(coordinates[axis], parameter, budget=budget)
                for parameter in range(len(axes))
            )
            for axis in axes
        ),
        budget=budget,
    )


def _rational_chart_measure(
    coordinates: tuple[algebra.Expression, ...],
    domain: algebra.BernsteinDomain,
    axes: tuple[int, ...],
    *,
    budget: CoordinateEnclosureBudget,
) -> Fraction | None:
    if len(axes) != 2 or domain not in ("simplex", "box"):
        return None
    vertices = (
        ((0, 0), (1, 0), (0, 1))
        if domain == "simplex"
        else ((0, 0), (1, 0), (1, 1), (0, 1))
    )
    projected = tuple(coordinates[axis] for axis in axes)
    result = Fraction(0)
    for first, second in zip(vertices, (*vertices[1:], vertices[0]), strict=True):
        origin = np.asarray(first, dtype=np.float64)
        direction = (np.asarray(second, dtype=np.float64) - origin)[:, None]
        arguments = algebra.affine_arguments(origin, direction, budget=budget)
        chart = tuple(
            algebra.expression_compose(value, arguments, budget=budget)
            for value in projected
        )
        start = tuple(
            algebra.expression_reference_evaluate(
                value, (Fraction(0),), "box", budget=budget
            )
            for value in chart
        )
        end = tuple(
            algebra.expression_reference_evaluate(
                value, (Fraction(1),), "box", budget=budget
            )
            for value in chart
        )
        displacement = tuple(b - a for a, b in zip(start, end, strict=True))
        cross = algebra.expression_add(
            algebra.expression_scale(
                algebra.expression_add(
                    chart[0], algebra.constant(-start[0], 1, budget=budget), budget=budget
                ),
                displacement[1],
                budget=budget,
            ),
            algebra.expression_scale(
                algebra.expression_add(
                    chart[1], algebra.constant(-start[1], 1, budget=budget), budget=budget
                ),
                -displacement[0],
                budget=budget,
            ),
            budget=budget,
        )
        if not cross and any(displacement):
            result += rational_coefficient(
                (start[0] * end[1] - start[1] * end[0]) / 2,
                role="mapped_coverage.chart_measure",
            )
            continue
        green = algebra.expression_add(
            algebra.expression_multiply(
                chart[0],
                algebra.expression_derivative(chart[1], 0, budget=budget),
                budget=budget,
            ),
            algebra.expression_scale(
                algebra.expression_multiply(
                    chart[1],
                    algebra.expression_derivative(chart[0], 0, budget=budget),
                    budget=budget,
                ),
                -1,
                budget=budget,
            ),
            budget=budget,
        )
        match green:
            case algebra.RationalPolynomial():
                return None
            case dict():
                result += integrate(green, "box", 1, budget=budget) / 2
            case unreachable:
                assert_never(unreachable)
    return result


@checked
def projected_chart_measure(
    coordinates: tuple[algebra.Expression, ...],
    domain: algebra.BernsteinDomain,
    axes: tuple[int, ...],
    *,
    jacobian: algebra.Expression,
    budget: CoordinateEnclosureBudget,
) -> Fraction | None:
    """Return the oriented planar area in rational arithmetic using the prepared Jacobian.

    Polynomial Jacobians integrate directly. A rational chart must
    prove each full boundary curve is straight, or have polynomial Green data;
    only then can endpoint cross-products replace that boundary integral.
    """
    match jacobian:
        case dict():
            return integrate(jacobian, domain, len(axes), budget=budget)
        case algebra.RationalPolynomial():
            return _rational_chart_measure(coordinates, domain, axes, budget=budget)
        case unreachable:
            assert_never(unreachable)


def _constant_coordinate_flux(
    polynomials: tuple[algebra.Polynomial, ...],
    domain: algebra.BernsteinDomain,
    axis: int,
    height: Fraction,
    *,
    budget: CoordinateEnclosureBudget,
) -> Fraction:
    """Integrate the sole normal component surviving a constant coordinate."""
    tangent = tuple(
        tuple(algebra.derivative(row, parameter, budget=budget) for parameter in range(2))
        for index, row in enumerate(polynomials)
        if index != axis
    )
    jacobian = algebra.add(
        algebra.multiply(tangent[0][0], tangent[1][1], budget=budget),
        algebra.scale(
            algebra.multiply(tangent[0][1], tangent[1][0], budget=budget),
            -1,
            budget=budget,
        ),
        budget=budget,
    )
    measure = integrate(jacobian, domain, 2, budget=budget)
    budget.reserve(2)
    bits = (
        abs(height.numerator).bit_length()
        + height.denominator.bit_length()
        + abs(measure.numerator).bit_length()
        + measure.denominator.bit_length()
        + 4
    )
    reserve_polynomial(2, 1, 0, bits, budget=budget)
    return (-1 if axis == 1 else 1) * height * measure / 3


def _polynomial_chart_flux(
    polynomials: tuple[algebra.Polynomial, ...],
    domain: algebra.BernsteinDomain,
    *,
    budget: CoordinateEnclosureBudget,
) -> Fraction:
    if any(not value for value in polynomials):
        # An identically zero coordinate has zero derivatives. Every
        # divergence term therefore contains that coordinate or one of
        # its derivatives: the polynomial flux is exactly zero.
        return Fraction(0)
    for axis, value in enumerate(polynomials):
        if len(value) != 1 or (0, 0) not in value:
            continue
        # A constant physical coordinate annihilates the other two normal
        # components. Integrate the remaining projected Jacobian before the
        # scalar height multiplication instead of expanding a density bank.
        return _constant_coordinate_flux(
            polynomials,
            domain,
            axis,
            rational_coefficient(value[(0, 0)], role="mapped_coverage.constant_height"),
            budget=budget,
        )
    tangent = tuple(
        tuple(algebra.derivative(value, axis, budget=budget) for value in polynomials)
        for axis in range(2)
    )
    density_terms = []
    for axis, value in enumerate(polynomials):
        if not (
            (tangent[0][(axis + 1) % 3] and tangent[1][(axis + 2) % 3])
            or (tangent[0][(axis + 2) % 3] and tangent[1][(axis + 1) % 3])
        ):
            # Both rational cross products are zero. Do not construct or
            # repeatedly reduce an empty divergence-density summand.
            continue
        weight = algebra.add(
            algebra.multiply(
                tangent[0][(axis + 1) % 3], tangent[1][(axis + 2) % 3], budget=budget
            ),
            algebra.scale(
                algebra.multiply(
                    tangent[0][(axis + 2) % 3],
                    tangent[1][(axis + 1) % 3],
                    budget=budget,
                ),
                -1,
                budget=budget,
            ),
            budget=budget,
        )
        if weight:
            density_terms.append(algebra.multiply(value, weight, budget=budget))
    return (
        integrate(
            algebra.sum_polynomials(tuple(density_terms), budget=budget),
            domain,
            2,
            budget=budget,
        )
        / 3
    )


def _planar_rational_chart_flux(
    coordinates: tuple[algebra.Expression, ...],
    domain: algebra.BernsteinDomain,
    *,
    budget: CoordinateEnclosureBudget,
) -> Fraction | None:
    reference = ((0, 0), (1, 0), (0, 1))
    points = tuple(
        tuple(
            algebra.expression_reference_evaluate(
                value, tuple(Fraction(x) for x in point), domain, budget=budget
            )
            for value in coordinates
        )
        for point in reference
    )
    first = tuple(b - a for a, b in zip(points[0], points[1], strict=True))
    second = tuple(b - a for a, b in zip(points[0], points[2], strict=True))
    normal = (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )
    axis = next((index for index, value in enumerate(normal) if value), None)
    if axis is None:
        return None
    height = sum(
        (value * weight for value, weight in zip(points[0], normal, strict=True)),
        Fraction(0),
    )
    plane = algebra.expression_add(
        algebra.expression_sum(
            tuple(
                algebra.expression_scale(value, weight, budget=budget)
                for value, weight in zip(coordinates, normal, strict=True)
            ),
            budget=budget,
        ),
        algebra.constant(-height, 2, budget=budget),
        budget=budget,
    )
    if plane:
        return None
    axes = tuple(index for index in range(3) if index != axis)
    jacobian = projected_jacobian(coordinates, axes, budget=budget)
    measure = projected_chart_measure(
        coordinates, domain, axes, jacobian=jacobian, budget=budget
    )
    if measure is None:
        return None
    return rational_coefficient(
        height * (-1 if axis == 1 else 1) * measure / (3 * normal[axis]),
        role="mapped_coverage.planar_flux",
    )


@checked
def chart_flux(
    coordinates: tuple[algebra.Expression, ...],
    domain: algebra.BernsteinDomain,
    *,
    budget: CoordinateEnclosureBudget,
) -> Fraction | None:
    """Integrate outward divergence flux in rational arithmetic over a polynomial or proved planar chart."""
    if len(coordinates) != 3 or domain not in ("simplex", "box"):
        return None
    match coordinates:
        case (dict() as first, dict() as second, dict() as third):
            return _polynomial_chart_flux((first, second, third), domain, budget=budget)
        case _:
            return _planar_rational_chart_flux(coordinates, domain, budget=budget)
