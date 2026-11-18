from __future__ import annotations

from fractions import Fraction
from itertools import combinations
from random import Random

import pytest

from phydrax.discretization._coordinate_enclosure import (
    evaluate_expression,
    Expression,
    Polynomial,
    polynomial_bounds,
)
from phydrax.discretization._reference_cell import reference_cell_topology
from phydrax.geometry._batch_boundary_traces import (
    batch_simplex_face_equal,
    quadratic_face_class,
    quadratic_tetrahedron_bank,
)
from phydrax.geometry._embedding_cells import EmbeddingCell
from phydrax.geometry._restricted_embedding import (
    _rational_simplex_map,
    restriction_trace_equal,
)


def _cell(coordinates: tuple[Expression, ...], ids: tuple[int, ...]) -> EmbeddingCell:
    topology = reference_cell_topology("tetrahedron")
    references = tuple(
        tuple(Fraction(float(value)) for value in point) for point in topology.vertices
    )
    corners = tuple(
        tuple(evaluate_expression(expression, point) for expression in coordinates)
        for point in references
    )
    bounds = tuple(
        polynomial_bounds(expression, "simplex", 3) for expression in coordinates
    )
    return EmbeddingCell(
        1,
        "mapped",
        0,
        "tetrahedron",
        3,
        "simplex",
        tuple(range(4)),
        ids,
        references,
        coordinates,
        corners,
        None,
        None,
        tuple(lower for lower, _ in bounds),
        tuple(upper for _, upper in bounds),
        tuple(tuple(face) for face in topology.entities[2]),
    )


def test_integer_midpoints_equal_fraction_definitions() -> None:
    random = Random(742)
    exponents = (
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (2, 0, 0),
        (0, 2, 0),
        (0, 0, 2),
        (1, 1, 0),
        (1, 0, 1),
        (0, 1, 1),
    )
    for _ in range(25):
        coordinates: tuple[Expression, ...] = tuple(
            {
                exponent: Fraction(
                    random.randrange(-100, 101) * (1 << 100), random.randrange(1, 101)
                )
                for exponent in exponents
            }
            for _ in range(3)
        )
        cell = _cell(coordinates, (10, 11, 12, 13))
        bank = quadratic_tetrahedron_bank(cell, _rational_simplex_map(cell), {})
        assert bank is not None
        denominator, values = bank
        for row, (left, right) in zip(values, combinations(range(4), 2), strict=True):
            point = tuple(
                (a + b) / 2
                for a, b in zip(
                    cell.reference_vertices[left],
                    cell.reference_vertices[right],
                    strict=True,
                )
            )
            expected = tuple(
                evaluate_expression(expression, point) for expression in coordinates
            )
            assert tuple(Fraction(value, denominator) for value in row) == expected


@pytest.mark.parametrize(
    "coefficient", (Fraction(0), Fraction(1, 13), Fraction(-1, 2**200))
)
def test_quadratic_trace_mismatch_is_exact(coefficient: Fraction) -> None:
    x: Polynomial = {(1, 0, 0): Fraction(1)}
    y: Polynomial = {(0, 1, 0): Fraction(1)}
    z: Polynomial = {(0, 0, 1): Fraction(1)}
    bump: Polynomial = {(0, 0, 1): Fraction(1), (1, 1, 0): coefficient}
    first = _cell((x, y, z), (10, 11, 12, 13))
    second = _cell((x, y, bump), (10, 11, 12, 14))
    requests = ((first, second, (10, 11, 12)), (second, second, (10, 11, 12)))
    assert batch_simplex_face_equal(requests, {}, {}, {}, {}) == (coefficient == 0, True)
    assert batch_simplex_face_equal(tuple(reversed(requests)), {}, {}, {}, {}) == (
        True,
        coefficient == 0,
    )
    split = tuple(
        batch_simplex_face_equal((request,), {}, {}, {}, {})[0] for request in requests
    )
    assert split == (coefficient == 0, True)


def test_quadratic_face_classifies_planarity_exactly() -> None:
    x: Polynomial = {(1, 0, 0): Fraction(1)}
    y: Polynomial = {(0, 1, 0): Fraction(1)}
    for coefficient in (Fraction(1, 7), Fraction(-1, 2**200)):
        bumped: Polynomial = {(0, 0, 1): Fraction(1), (1, 1, 0): coefficient}
        cell = _cell((x, y, bumped), (10, 11, 12, 13))
        points = tuple(
            (cell.corner_images[i][0], cell.corner_images[i][1], cell.corner_images[i][2])
            for i in (0, 1, 2)
        )
        assert quadratic_face_class(cell, (10, 11, 12), points, {}, {}) == "nonplanar"
        points = tuple(
            (cell.corner_images[i][0], cell.corner_images[i][1], cell.corner_images[i][2])
            for i in (0, 1, 3)
        )
        assert quadratic_face_class(cell, (10, 11, 13), points, {}, {}) == "affine"


def test_higher_order_trace_uses_exact_general_route() -> None:
    x: Polynomial = {(1, 0, 0): Fraction(1)}
    y: Polynomial = {(0, 1, 0): Fraction(1)}
    z: Polynomial = {(0, 0, 1): Fraction(1)}
    cubic: Polynomial = {(0, 0, 1): Fraction(1), (2, 1, 0): Fraction(1, 7)}
    first = _cell((x, y, z), (10, 11, 12, 13))
    second = _cell((x, y, cubic), (10, 11, 12, 14))
    face = (10, 11, 12)
    expected = restriction_trace_equal(first, second, face)
    assert expected is False
    assert batch_simplex_face_equal(((first, second, face),), {}, {}, {}, {}) == (
        expected,
    )


def test_planar_nonlinear_and_degenerate_faces_remain_distinct() -> None:
    x: Polynomial = {(1, 0, 0): Fraction(1), (1, 1, 0): Fraction(1, 8)}
    y: Polynomial = {(0, 1, 0): Fraction(1)}
    z: Polynomial = {(0, 0, 1): Fraction(1)}
    planar = _cell((x, y, z), (10, 11, 12, 13))
    points = tuple(
        (
            planar.corner_images[i][0],
            planar.corner_images[i][1],
            planar.corner_images[i][2],
        )
        for i in (0, 1, 2)
    )
    assert quadratic_face_class(planar, (10, 11, 12), points, {}, {}) == "undecided"
    degenerate = _cell((x, {}, z), (10, 11, 12, 13))
    points = tuple(
        (
            degenerate.corner_images[i][0],
            degenerate.corner_images[i][1],
            degenerate.corner_images[i][2],
        )
        for i in (0, 1, 2)
    )
    assert quadratic_face_class(degenerate, (10, 11, 12), points, {}, {}) == "undecided"
    collapsed = _cell((x, {}, {}), (10, 11, 12, 13))
    points = tuple(
        (
            collapsed.corner_images[i][0],
            collapsed.corner_images[i][1],
            collapsed.corner_images[i][2],
        )
        for i in (0, 1, 3)
    )
    assert quadratic_face_class(collapsed, (10, 11, 13), points, {}, {}) == "degenerate"
