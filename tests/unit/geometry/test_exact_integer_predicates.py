#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Exact integer predicate banks: exact signs, one-owner input kinds, ledger charges.

References are cofactor expansions in Python integers, written independently
of the predicate kernels.
"""

from fractions import Fraction
from typing import Any

import numpy as np
import pytest

from phydrax.discretization._coordinate_enclosure import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResourceError,
)
from phydrax.geometry import (
    incircle,
    orient2d,
    orient3d,
    PredicateMode,
    segment_intersections_2d,
    SegmentIntersectionStatus,
)


_SCALE = 2**300


def _bank(rows: Any) -> np.ndarray:
    return np.asarray([[int(value) for value in row] for row in rows], dtype=object)


def _reference_orient3d(
    a: tuple[int, ...], b: tuple[int, ...], c: tuple[int, ...], d: tuple[int, ...]
) -> int:
    u, v, w = ([q[axis] - a[axis] for axis in range(3)] for q in (b, c, d))
    value = (
        u[0] * (v[1] * w[2] - v[2] * w[1])
        - u[1] * (v[0] * w[2] - v[2] * w[0])
        + u[2] * (v[0] * w[1] - v[1] * w[0])
    )
    return (value > 0) - (value < 0)


@pytest.mark.parametrize("offset", [-1, 0, 1], ids=["below", "on", "above"])
def test_orient3d_integer_bank_is_exact_beyond_binary64(offset: int) -> None:
    # A plane point 2**300 away is perturbed by one unit: binary64 rounding
    # would erase the offset entirely.
    a, b, c = (0, 0, 0), (_SCALE, 0, 0), (0, _SCALE, 0)
    d = (_SCALE // 3, _SCALE // 5, offset)
    result = orient3d(
        _bank([a]), _bank([b]), _bank([c]), _bank([d]), mode=PredicateMode.EXACT
    )

    assert np.asarray(result.certain).tolist() == [True]
    assert np.asarray(result.signs).tolist() == [_reference_orient3d(a, b, c, d)]


def test_orient2d_integer_bank_follows_the_package_convention() -> None:
    a, b = _bank([(0, 0)] * 3), _bank([(_SCALE, 1)] * 3)
    c = _bank([(2 * _SCALE, 3), (2 * _SCALE, 2), (2 * _SCALE, 1)])
    result = orient2d(a, b, c, mode=PredicateMode.FILTERED)

    # det[b - a, c - a] = SCALE * (c_y - 2): counterclockwise, collinear, clockwise.
    assert np.asarray(result.signs).tolist() == [1, 0, -1]
    assert bool(np.all(np.asarray(result.certain)))


def test_segment_classification_of_an_integer_bank_is_exact() -> None:
    a, b = _bank([(0, 0)]), _bank([(2 * _SCALE, 2)])
    c, d = _bank([(_SCALE, 1)]), _bank([(3 * _SCALE, 3)])
    status = segment_intersections_2d(a, b, c, d, mode=PredicateMode.EXACT).status

    assert np.asarray(status).tolist() == [SegmentIntersectionStatus.COLLINEAR_OVERLAP]


@pytest.mark.parametrize(
    "entry",
    [True, Fraction(1, 2), 0.5, np.int64(1)],
    ids=["boolean", "fraction", "float", "fixed-width"],
)
def test_integer_banks_refuse_other_entry_kinds(entry: object) -> None:
    bank = np.asarray([[0, entry, 0]], dtype=object)
    with pytest.raises(TypeError, match="Python integer object arrays"):
        orient3d(
            bank,
            _bank([(1, 0, 0)]),
            _bank([(0, 1, 0)]),
            _bank([(0, 0, 1)]),
            mode=PredicateMode.EXACT,
        )


@pytest.mark.parametrize(
    "other",
    [np.asarray([[1, 0, 0]], dtype=np.int64), np.asarray([[1.0, 0.0, 0.0]])],
    ids=["int64", "float64"],
)
def test_integer_banks_refuse_mixed_numeric_owners(other: np.ndarray) -> None:
    with pytest.raises(TypeError, match="Python integer object arrays"):
        orient3d(
            _bank([(0, 0, 0)]),
            other,
            _bank([(0, 1, 0)]),
            _bank([(0, 0, 1)]),
            mode=PredicateMode.EXACT,
        )


def test_integer_banks_support_orientation_only() -> None:
    with pytest.raises(TypeError, match="orient2d and orient3d"):
        incircle(
            _bank([(0, 0)]),
            _bank([(1, 0)]),
            _bank([(0, 1)]),
            _bank([(1, 1)]),
            mode=PredicateMode.EXACT,
        )


def test_integer_bank_charges_the_active_ledger_before_evaluation() -> None:
    rows = 16
    banks = tuple(
        _bank([point] * rows)
        for point in ((0, 0, 0), (_SCALE, 0, 0), (0, _SCALE, 0), (0, 0, _SCALE))
    )
    budget = CoordinateEnclosureBudget(10**9, 10**9)
    with budget.activate():
        orient3d(*banks, mode=PredicateMode.EXACT)

    # Every input entry and the prepared common plane/query actions are charged;
    # repeated rows reuse that exact preparation instead of replaying a 68-visit
    # determinant kernel for each row.
    assert budget.work_units > sum(bank.size for bank in banks)
    assert budget.peak_bytes_upper > 0
    assert budget.temporary_bytes_upper == 0

    starved = CoordinateEnclosureBudget(budget.work_units - 1, 10**9)
    with starved.activate(), pytest.raises(CoordinateEnclosureResourceError):
        orient3d(*banks, mode=PredicateMode.EXACT)
