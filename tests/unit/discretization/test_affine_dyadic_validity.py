#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import random
from fractions import Fraction

import numpy as np
import pytest

from phydrax.discretization._affine_dyadic_validity import affine_dyadic_quality
from phydrax.discretization._cell_geometry_validity import (
    _affine_quality,
    CellValidityPolicy,
    CellValidityStatus,
)
from phydrax.discretization._coordinate_enclosure import Polynomial
from phydrax.discretization._exact_coordinate_bank import ExactCoordinateBank


def _simplex_basis(dimension: int) -> tuple[Polynomial, ...]:
    zero = (0,) * dimension
    units = tuple(
        tuple(int(axis == column) for axis in range(dimension))
        for column in range(dimension)
    )
    first: Polynomial = {zero: Fraction(1), **{unit: Fraction(-1) for unit in units}}
    remaining: tuple[Polynomial, ...] = tuple({unit: Fraction(1)} for unit in units)
    return (first, *remaining)


@pytest.mark.parametrize(
    ("dimension", "rows", "expected"),
    [
        (1, ((0,), (3,), (-2,), (0,)), ((True, 3), (False, -2), (False, 0))),
        (
            2,
            ((0, 0), (2, 0), (0, 3), (0, -3), (4, 0)),
            ((True, 6), (False, -6), (False, 0)),
        ),
        (
            3,
            ((0, 0, 0), (2, 0, 0), (0, 3, 0), (0, 0, 4), (0, 0, -4), (0, 0, 0)),
            ((True, 24), (False, -24), (False, 0)),
        ),
        (
            2,
            ((0, 0, 0), (2, 0, 0), (0, 3, 4), (0, -3, -4), (4, 0, 0)),
            ((True, 100), (True, 100), (False, 0)),
        ),
    ],
)
def test_exact_determinants_and_orientation(
    dimension: int, rows: tuple, expected: tuple
) -> None:
    bank = ExactCoordinateBank.from_rows(
        tuple(tuple(Fraction(x) for x in row) for row in rows)
    )
    routes = np.asarray(
        [(0, *range(1, dimension), last) for last in range(dimension, dimension + 3)],
        dtype=np.int64,
    )
    result = affine_dyadic_quality(
        _simplex_basis(dimension), bank, routes, dimension, Fraction(0)
    )
    assert result == tuple(
        (valid, Fraction(determinant)) for valid, determinant in expected
    )


@pytest.mark.parametrize("embedded", [False, True])
def test_strict_threshold_and_scalar_parity_across_groupings(embedded: bool) -> None:
    floor = Fraction(0.36 if embedded else 0.6)
    policy = CellValidityPolicy(relative_determinant_floor=float(floor))
    epsilon = Fraction(1, 1 << 55)
    heights = (Fraction(3) - epsilon, Fraction(3), Fraction(3) + epsilon)
    rows = [(Fraction(0), Fraction(0)), (Fraction(1), Fraction(0))]
    rows.extend((Fraction(4), height) for height in heights)
    if embedded:
        rows = [(*row, Fraction(0)) for row in rows]
    bank = ExactCoordinateBank.from_rows(rows)
    routes = np.asarray(((0, 1, 2), (0, 1, 3), (0, 1, 4)), dtype=np.int64)
    basis = _simplex_basis(2)
    actual = affine_dyadic_quality(basis, bank, routes, 2, floor)
    expected = []
    for height in heights:
        jacobian = ((Fraction(1), Fraction(4)), (Fraction(0), height))
        if embedded:
            jacobian += ((Fraction(0), Fraction(0)),)
        status, determinant = _affine_quality(jacobian, 2, policy)
        expected.append((status == CellValidityStatus.CERTIFIED_VALID, determinant))
    assert actual == tuple(expected)
    assert actual == tuple(
        affine_dyadic_quality(basis, bank, routes[i : i + 1], 2, floor)[0]
        for i in range(len(routes))
    )
    exact_floor = Fraction(9, 25) if embedded else Fraction(3, 5)
    exact_results = affine_dyadic_quality(basis, bank, routes, 2, exact_floor)
    assert all(item is not None for item in exact_results)
    assert [item[0] for item in exact_results if item is not None] == [False, False, True]


def test_unsupported_rows_and_bases_are_refused_independently() -> None:
    bank = ExactCoordinateBank.from_rows(
        ((Fraction(0),), (Fraction(1),), (Fraction(1, 3),), (Fraction(1 << 80),))
    )
    routes = np.asarray(((0, 1), (0, 2), (0, 3), (1, 0)), dtype=np.int64)
    basis = _simplex_basis(1)
    assert affine_dyadic_quality(basis, bank, routes, 1, Fraction(0)) == (
        (True, Fraction(1)),
        None,
        (True, Fraction(1 << 80)),
        (False, Fraction(-1)),
    )
    assert (
        affine_dyadic_quality(
            ({(2,): Fraction(1)}, basis[1]), bank, routes, 1, Fraction(0)
        )
        == (None,) * 4
    )
    assert (
        affine_dyadic_quality(
            ({(1,): Fraction(1, 3)}, basis[1]), bank, routes, 1, Fraction(0)
        )
        == (None,) * 4
    )


@pytest.mark.parametrize(("dimension", "ambient"), ((1, 1), (2, 2), (3, 3), (2, 3)))
def test_dyadic_block_matches_exact_affine_quality(dimension: int, ambient: int) -> None:
    generator = random.Random(8050 + 10 * dimension + ambient)
    rows: list[tuple[Fraction, ...]] = []
    jacobians = []
    for _ in range(24):
        points = tuple(
            tuple(
                Fraction(generator.randint(-32, 32), 2 ** generator.randint(0, 100))
                for _ in range(ambient)
            )
            for _ in range(dimension + 1)
        )
        rows.extend(points)
        jacobians.append(
            tuple(
                tuple(
                    points[column + 1][axis] - points[0][axis]
                    for column in range(dimension)
                )
                for axis in range(ambient)
            )
        )
    bank = ExactCoordinateBank.from_rows(rows)
    routes = np.arange(len(rows), dtype=np.int64).reshape((-1, dimension + 1))
    policy = CellValidityPolicy(relative_determinant_floor=0.25)
    actual = affine_dyadic_quality(
        _simplex_basis(dimension),
        bank,
        routes,
        dimension,
        Fraction(policy.determinant_floor()),
    )
    expected = tuple(
        (status == CellValidityStatus.CERTIFIED_VALID, determinant)
        for status, determinant in (
            _affine_quality(jacobian, dimension, policy) for jacobian in jacobians
        )
    )
    assert actual == expected
    assert actual == tuple(
        result
        for start in range(0, len(routes), 5)
        for result in affine_dyadic_quality(
            _simplex_basis(dimension),
            bank,
            routes[start : start + 5],
            dimension,
            Fraction(policy.determinant_floor()),
        )
    )
