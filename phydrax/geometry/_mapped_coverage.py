#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Exact mapped measure and planar facet-containment calculus.

Only canonical source polynomials participate. A zero plane polynomial proves
planarity over the entire face; corner positions do not establish that premise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from itertools import product

import numpy as np

from ..discretization import _coordinate_enclosure as algebra
from ..discretization._cell_geometry import CellGeometryElement
from ..discretization._reference_cell import reference_cell_topology


@dataclass
class SubdivisionLedger:
    """Actually charged subdivision work of one certificate owner.

    ``pieces`` counts every examined piece (including a refused over-budget
    piece), ``candidate_pairs`` every evaluated piece-source pair, and
    ``maximum_depth`` the deepest subdivision level actually examined.
    """

    pieces: int = 0
    candidate_pairs: int = 0
    maximum_depth: int = 0

    def charge(self, depth: int, /) -> None:
        self.pieces += 1
        self.maximum_depth = max(self.maximum_depth, depth)


def integrate(polynomial: algebra.Polynomial, domain: str, dimension: int) -> Fraction:
    """Integrate power monomials exactly, including the collapsed pyramid cube."""
    budget = algebra._COORDINATE_BUDGET.get()
    if budget is not None:
        numerator, denominator = algebra._coefficient_profile((polynomial,))
        budget.reserve(dimension * len(polynomial))
        # Every integration weight divides by products bounded by
        # (sum(index) + dimension)!. Its log bound follows n! <= n**n;
        # summing these bit bounds also bounds the accumulated denominator.
        weight_bits = sum(
            (sum(index) + dimension) * max(sum(index) + dimension, 1).bit_length()
            for index in polynomial
        )
        algebra._reserve_polynomial(
            2 * len(polynomial),
            1,
            0,
            numerator + denominator + weight_bits + max(len(polynomial), 1).bit_length(),
        )
    result = Fraction(0)
    for index, coefficient in polynomial.items():
        if domain == "simplex":
            weight = Fraction(
                math.prod(math.factorial(i) for i in index),
                math.factorial(sum(index) + dimension),
            )
        elif domain == "prism":
            weight = Fraction(
                math.factorial(index[0]) * math.factorial(index[1]),
                math.factorial(index[0] + index[1] + 2) * (index[2] + 1),
            )
        elif domain == "box":
            weight = Fraction(1, math.prod(i + 1 for i in index))
        else:
            raise ValueError("Unsupported exact integration domain.")
        result += coefficient * weight
    return result


def enclosure(value: Fraction) -> tuple[float, float]:
    rounded = float(value)
    exact = Fraction(rounded)
    return (
        float(np.nextafter(rounded, -math.inf)) if exact > value else rounded,
        float(np.nextafter(rounded, math.inf)) if exact < value else rounded,
    )


def map_domain(kind: str) -> str:
    return (
        "simplex"
        if kind in ("triangle", "tetrahedron")
        else "prism"
        if kind == "prism"
        else "box"
    )


def face_charts(
    coordinates: tuple[algebra.Expression, ...], kind: str, face: tuple[int, ...]
) -> tuple[tuple[tuple[algebra.Expression, ...], str], ...]:
    topology = reference_cell_topology(kind)
    corners = np.asarray(topology.vertices, dtype=np.float64)[np.asarray(face)]
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
            np.asarray(side[0], dtype=np.float64), np.asarray(side[1], dtype=np.float64)
        )
        composition = algebra.ExpressionComposition(arguments)
        return ((tuple(composition(value) for value in coordinates), "box"),)
    pieces = (corners,) if len(face) <= 3 else (corners[[0, 1, 2]], corners[[0, 2, 3]])
    result = []
    for piece in pieces:
        arguments = algebra.affine_arguments(piece[0], (piece[1:] - piece[0]).T)
        composition = algebra.ExpressionComposition(arguments)
        result.append((tuple(composition(value) for value in coordinates), "simplex"))
    return tuple(result)


def prepared_face_charts(
    element: CellGeometryElement,
    local: algebra.CoordinateCoefficients,
    kind: str,
    face: tuple[int, ...],
    /,
) -> tuple[tuple[tuple[algebra.Expression, ...], str], ...] | None:
    """Share full oriented source-face preparation in its original ledger."""
    budget = algebra._COORDINATE_BUDGET.get()
    if budget is None:
        coordinates = algebra.coordinate_expressions(element, local)
        return None if coordinates is None else face_charts(coordinates, kind, face)
    with budget.temporary_scope():
        coefficients, key = algebra._coordinate_preparation_key(element, local)
        budget.reserve(len(face))
        face_key = (*key, kind, face)
        cached = budget.face_chart_cache.get(face_key)
        if cached is not None:
            return cached
        coordinates = budget.coordinate_cache.get(key)
        if coordinates is None:
            coordinates = algebra.coordinate_expressions(element, coefficients)
        if coordinates is None:
            return None
        charts = face_charts(coordinates, kind, face)
        budget.retain_basis((face_key, charts))
        budget.face_chart_cache[face_key] = charts
        return charts


def face_flux(
    coordinates: tuple[algebra.Expression, ...], kind: str, face: tuple[int, ...], /
) -> Fraction | None:
    """Integrate the complete oriented face without containment triangulation."""
    if len(face) == 4:
        corners = np.asarray(reference_cell_topology(kind).vertices, dtype=np.float64)[
            np.asarray(face)
        ]
        # These are canonical reference-cell vertices, not physical samples.
        # Prove the reference parallelogram before using its full box chart.
        if np.array_equal(corners[2], corners[1] + corners[3] - corners[0]):
            arguments = algebra.affine_arguments(
                corners[0],
                np.stack((corners[1] - corners[0], corners[3] - corners[0]), axis=1),
            )
            composition = algebra.ExpressionComposition(arguments)
            return chart_flux(tuple(composition(value) for value in coordinates), "box")
    result = Fraction(0)
    for chart, domain in face_charts(coordinates, kind, face):
        contribution = chart_flux(chart, domain)
        if contribution is None:
            return None
        result += contribution
    return result


def prepared_face_flux(
    element: CellGeometryElement,
    local: algebra.CoordinateCoefficients,
    kind: str,
    face: tuple[int, ...],
    /,
) -> Fraction | None:
    """Retain actual exact oriented flux in the original complete source owner."""
    budget = algebra._COORDINATE_BUDGET.get()
    if budget is None:
        coordinates = algebra.coordinate_expressions(element, local)
        return None if coordinates is None else face_flux(coordinates, kind, face)
    with budget.temporary_scope():
        coefficients, key = algebra._coordinate_preparation_key(element, local)
        budget.reserve(len(face))
        face_key = (*key, kind, face)
        if face_key in budget.face_flux_cache:
            return budget.face_flux_cache[face_key]
        coordinates = budget.coordinate_cache.get(key)
        if coordinates is None:
            coordinates = algebra.coordinate_expressions(element, coefficients)
        flux = None if coordinates is None else face_flux(coordinates, kind, face)
        budget.retain_basis((face_key, flux))
        budget.face_flux_cache[face_key] = flux
        return flux


def _node_count(polynomial: algebra.Expression, domain: str, dimension: int) -> int:
    return algebra.expression_node_count(polynomial, domain, dimension)


def _children(domain: str, dimension: int) -> tuple[tuple[algebra.Polynomial, ...], ...]:
    if dimension == 1:
        return tuple(
            algebra.affine_arguments(
                np.asarray((start,), dtype=np.float64),
                np.asarray(((0.5,),), dtype=np.float64),
            )
            for start in (0.0, 0.5)
        )
    if domain == "prism":
        variables = algebra.axes(3)
        return tuple(
            (
                *tuple(algebra.compose(value, variables[:2]) for value in triangle),
                algebra.compose(height[0], (variables[2],)),
            )
            for triangle in _children("simplex", 2)
            for height in _children("box", 1)
        )
    if domain == "simplex":
        if dimension == 3:
            from ..discretization._cell_geometry_validity import _simplex_child_maps

            origins, matrices = _simplex_child_maps(dimension)
            return tuple(
                algebra.affine_arguments(origin, matrix)
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
            )
            for row in triangles
        )
    return tuple(
        algebra.affine_arguments(
            np.asarray(origin, dtype=np.float64),
            0.5 * np.eye(dimension, dtype=np.float64),
        )
        for origin in product((0.0, 0.5), repeat=dimension)
    )


def nonnegative(
    polynomial: algebra.Expression,
    domain: str,
    dimension: int,
    maximum_nodes: int,
    maximum_depth: int,
    maximum_pieces: int,
    work: SubdivisionLedger,
) -> bool:
    pending = [(polynomial, 0)]
    children: tuple[tuple[algebra.Polynomial, ...], ...] | None = None
    while pending:
        current, depth = pending.pop()
        work.charge(depth)
        if (
            work.pieces > maximum_pieces
            or _node_count(current, domain, dimension) > maximum_nodes
        ):
            return False
        coefficients = algebra.expression_bernstein_coefficients(
            current, domain, dimension
        )
        if min(coefficients) >= 0:
            continue
        if max(coefficients) < 0 or depth >= maximum_depth:
            return False
        if children is None:
            children = _children(domain, dimension)
        pending.extend(
            (algebra.expression_compose(current, child), depth + 1) for child in children
        )
    return True


def projected_jacobian(
    coordinates: tuple[algebra.Expression, ...], axes: tuple[int, ...]
) -> algebra.Expression:
    return algebra.expression_determinant(
        tuple(
            tuple(
                algebra.expression_derivative(coordinates[axis], parameter)
                for parameter in range(len(axes))
            )
            for axis in axes
        )
    )


def projected_chart_measure(
    coordinates: tuple[algebra.Expression, ...],
    domain: str,
    axes: tuple[int, ...],
    *,
    jacobian: algebra.Expression,
) -> Fraction | None:
    """Exact oriented planar area using this chart's already prepared Jacobian.

    Polynomial Jacobians integrate directly. A genuinely rational chart must
    prove each full boundary curve is straight, or have polynomial Green data;
    only then can endpoint cross-products replace that boundary integral.
    """
    if not isinstance(jacobian, algebra.RationalPolynomial):
        return integrate(jacobian, domain, len(axes))
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
        arguments = algebra.affine_arguments(origin, direction)
        chart = tuple(algebra.expression_compose(value, arguments) for value in projected)
        start = tuple(
            algebra.expression_reference_evaluate(value, (Fraction(0),), "box")
            for value in chart
        )
        end = tuple(
            algebra.expression_reference_evaluate(value, (Fraction(1),), "box")
            for value in chart
        )
        displacement = tuple(b - a for a, b in zip(start, end, strict=True))
        cross = algebra.expression_add(
            algebra.expression_scale(
                algebra.expression_add(chart[0], algebra.constant(-start[0], 1)),
                displacement[1],
            ),
            algebra.expression_scale(
                algebra.expression_add(chart[1], algebra.constant(-start[1], 1)),
                -displacement[0],
            ),
        )
        if not cross and any(displacement):
            result += (start[0] * end[1] - start[1] * end[0]) / 2
            continue
        green = algebra.expression_add(
            algebra.expression_multiply(
                chart[0], algebra.expression_derivative(chart[1], 0)
            ),
            algebra.expression_scale(
                algebra.expression_multiply(
                    chart[1], algebra.expression_derivative(chart[0], 0)
                ),
                -1,
            ),
        )
        if isinstance(green, algebra.RationalPolynomial):
            return None
        result += integrate(green, "box", 1) / 2
    return result


def chart_flux(
    coordinates: tuple[algebra.Expression, ...], domain: str
) -> Fraction | None:
    """Exact outward divergence flux of a full polynomial or proved planar chart."""
    if len(coordinates) != 3 or domain not in ("simplex", "box"):
        return None
    if not any(isinstance(value, algebra.RationalPolynomial) for value in coordinates):
        polynomials = tuple(
            value
            for value in coordinates
            if not isinstance(value, algebra.RationalPolynomial)
        )
        if any(not value for value in polynomials):
            # An identically zero coordinate has zero derivatives. Every
            # divergence term therefore contains that coordinate or one of
            # its derivatives: the complete polynomial flux is exactly zero.
            return Fraction(0)
        tangent = tuple(
            tuple(algebra.derivative(value, axis) for value in polynomials)
            for axis in range(2)
        )
        density_terms = []
        for axis, value in enumerate(polynomials):
            if not (
                (tangent[0][(axis + 1) % 3] and tangent[1][(axis + 2) % 3])
                or (tangent[0][(axis + 2) % 3] and tangent[1][(axis + 1) % 3])
            ):
                # Both exact cross products are zero. Do not construct or
                # repeatedly reduce an empty divergence-density summand.
                continue
            weight = algebra.add(
                algebra.multiply(tangent[0][(axis + 1) % 3], tangent[1][(axis + 2) % 3]),
                algebra.scale(
                    algebra.multiply(
                        tangent[0][(axis + 2) % 3], tangent[1][(axis + 1) % 3]
                    ),
                    -1,
                ),
            )
            if weight:
                density_terms.append(algebra.multiply(value, weight))
        return integrate(algebra.sum_polynomials(tuple(density_terms)), domain, 2) / 3
    reference = ((0, 0), (1, 0), (0, 1))
    points = tuple(
        tuple(
            algebra.expression_reference_evaluate(
                value, tuple(Fraction(x) for x in point), domain
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
                algebra.expression_scale(value, weight)
                for value, weight in zip(coordinates, normal, strict=True)
            )
        ),
        algebra.constant(-height, 2),
    )
    if plane:
        return None
    axes = tuple(index for index in range(3) if index != axis)
    jacobian = projected_jacobian(coordinates, axes)
    measure = projected_chart_measure(coordinates, domain, axes, jacobian=jacobian)
    if measure is None:
        return None
    return height * (-1 if axis == 1 else 1) * measure / (3 * normal[axis])
