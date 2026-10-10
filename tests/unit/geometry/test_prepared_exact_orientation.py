#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from fractions import Fraction
from itertools import permutations

import numpy as np
import pytest

from phydrax._geometry_predicates import _exact_integer, _ORIENT2D, _ORIENT3D
from phydrax.discretization._coordinate_enclosure import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResourceError,
)


@pytest.mark.parametrize("rational", [False, True])
@pytest.mark.parametrize("dimension", [2, 3])
def test_prepared_exact_orientation_preserves_kernel_permutations_and_degeneracy(
    dimension: int, rational: bool
) -> None:
    spec = _ORIENT2D if dimension == 2 else _ORIENT3D
    vertices = (
        ((0, 0), (3, 1), (-1, 4))
        if dimension == 2
        else ((0, 0, 0), (3, 1, 0), (-1, 4, 1), (1, -2, 5))
    )
    if rational:
        vertices = tuple(
            tuple(Fraction(value + 1, 3 + axis * 2) for axis, value in enumerate(row))
            for row in vertices
        )
    cases = [
        tuple(vertices[index] for index in order)
        for order in permutations(range(dimension + 1))
    ]
    cases.extend((vertices, (*vertices[:-1], vertices[0])))
    points = tuple(
        np.asarray([case[index] for case in cases], dtype=object)
        for index in range(dimension + 1)
    )
    expected = np.asarray(
        [int(value > 0) - int(value < 0) for value in spec.kernel(np, *points)[0]],
        dtype=np.int8,
    )
    with CoordinateEnclosureBudget(1000000, 16_000_000).activate() as ledger:
        actual = _exact_integer(spec, points)
        np.testing.assert_array_equal(actual, expected)
        prepared = tuple(ledger.exact_orientation_preparation_cache.values())
        count = len(prepared)
        first_work = ledger.work_units
        np.testing.assert_array_equal(_exact_integer(spec, points), expected)
        assert len(ledger.exact_orientation_preparation_cache) == count
        assert tuple(ledger.exact_orientation_preparation_cache.values()) == prepared
        assert 0 < ledger.work_units - first_work < first_work


def test_prepared_exact_orientation_keys_complete_ordered_values_and_predicate_kind() -> (
    None
):
    points = tuple(
        np.asarray([row], dtype=object)
        for row in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))
    )
    with CoordinateEnclosureBudget(100000, 4_000_000).activate() as ledger:
        assert _exact_integer(_ORIENT3D, points)[0] == 1
        changed = (points[0], points[2], points[1], points[3])
        assert _exact_integer(_ORIENT3D, changed)[0] == -1
        shifted = tuple(
            np.asarray([[value + 11 for value in row[0]]], dtype=object) for row in points
        )
        assert _exact_integer(_ORIENT3D, shifted)[0] == 1
        assert len(ledger.exact_orientation_preparation_cache) == 3
        planar = tuple(row[:, :2] for row in points[:3])
        assert _exact_integer(_ORIENT2D, planar)[0] == 1
        assert {key[0] for key in ledger.exact_orientation_preparation_cache} == {
            "orient2d",
            "orient3d",
        }


def test_prepared_exact_orientation_preserves_huge_fraction_bits_and_refusal_prefixes() -> (
    None
):
    points = tuple(
        np.asarray([row], dtype=object)
        for row in (
            (Fraction(1, 2**1074), Fraction(0), Fraction(0)),
            (Fraction(2**900), Fraction(1, 3), Fraction(0)),
            (Fraction(0), Fraction(2**800), Fraction(1, 7)),
            (Fraction(1, 5), Fraction(0), Fraction(2**700)),
        )
    )
    expected = np.asarray(
        [int(value > 0) - int(value < 0) for value in _ORIENT3D.kernel(np, *points)[0]],
        dtype=np.int8,
    )
    with CoordinateEnclosureBudget(100000, 16_000_000).activate():
        np.testing.assert_array_equal(_exact_integer(_ORIENT3D, points), expected)
    with (
        CoordinateEnclosureBudget(1, 16_000_000).activate() as ledger,
        pytest.raises(CoordinateEnclosureResourceError) as refused,
    ):
        _exact_integer(_ORIENT3D, points)
    assert refused.value.completed == ledger.work_units
    assert not ledger.exact_orientation_preparation_cache
    with (
        CoordinateEnclosureBudget(100000, 1).activate() as ledger,
        pytest.raises(CoordinateEnclosureResourceError),
    ):
        _exact_integer(_ORIENT3D, points)
    assert not ledger.exact_orientation_preparation_cache
