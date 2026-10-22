#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Exact power-source group actions composed on a bounded lattice."""

from __future__ import annotations

from fractions import Fraction
from itertools import product

import numpy as np
import pytest

from phydrax.discretization import ExactPowerCellGeometryLinearActionSource
from phydrax.discretization._exact_power_geometry import (
    _ExactPowerBudget,
    _prepare_power_images,
)
from phydrax.discretization._periodic_exact import (
    exact_periodic_element,
    ExactPeriodicActionLattice,
)
from phydrax.geometry._rational import rational_vector
from tests._support.coordinate_references import coordinate_reference_budget
from tests._support.exact_power_geometry import periodic_power_source

type _Matrix = tuple[tuple[Fraction, ...], ...]


def _homogeneous(linear: tuple[tuple[int, ...], ...], shift: tuple[int, ...]) -> _Matrix:
    return (
        *(
            (*(Fraction(value) for value in row), Fraction(offset))
            for row, offset in zip(linear, shift, strict=True)
        ),
        (Fraction(0), Fraction(0), Fraction(0), Fraction(1)),
    )


_IDENTITY = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
_QUARTER_TURN = ((0, -1, 0), (1, 0, 0), (0, 0, 1))
_GROUPS: dict[str, tuple[tuple[_Matrix, ...], tuple[int, ...]]] = {
    "translations": (
        tuple(
            _homogeneous(_IDENTITY, shift)
            for shift in ((4, 0, 0), (0, 4, 0), (0, 0, 4))
        ),
        (0, 0, 0),
    ),
    "screw": (
        (
            _homogeneous(_QUARTER_TURN, (0, 0, 0)),
            _homogeneous(_IDENTITY, (0, 0, 3)),
        ),
        (4, 0),
    ),
    "noncommuting-ordered-word": (
        (
            _homogeneous(_QUARTER_TURN, (0, 0, 0)),
            _homogeneous(_IDENTITY, (3, 0, 0)),
        ),
        (4, 0),
    ),
}


def _keys(rank: int, dense: bool) -> tuple[tuple[int, ...], ...]:
    if dense:
        return tuple(product(range(-2, 3), repeat=rank))
    # Isolated exponents have no retained neighbour and use direct evaluation.
    return ((3, -2, 0), (0, 0, -3), (1, 1, 1)) if rank == 3 else ((5, -3), (2, 2))


@pytest.mark.parametrize("dense", (True, False), ids=("box", "isolated"))
@pytest.mark.parametrize("group", tuple(_GROUPS))
def test_action_lattice_equals_direct_elements_without_extra_work(
    group: str, dense: bool
) -> None:
    matrices, orders = _GROUPS[group]
    keys = _keys(len(orders), dense)
    with coordinate_reference_budget() as budget:
        lattice = ExactPeriodicActionLattice(matrices, orders)
        start = budget.work_units
        planned = lattice._planned_work(
            sorted(
                {lattice.key(key) for key in keys},
                key=lambda value: (sum(map(abs, value)), value),
            )
        )
        elements = lattice.prepare(keys, budget=budget)
        lattice_work = budget.work_units - start
        start = budget.work_units
        direct = tuple(
            exact_periodic_element(matrices, orders, key, budget=budget) for key in keys
        )
        direct_work = budget.work_units - start
        # Retained elements are returned again without further work.
        assert lattice.prepare(keys, budget=budget) == elements
        assert budget.work_units - start == direct_work
    assert elements == direct
    assert planned == lattice_work
    if dense:
        # One multiplication per retained neighbour replaces repeated powers.
        assert lattice_work < direct_work // 4
    else:
        assert lattice_work <= direct_work


def test_power_images_equal_direct_group_action_images() -> None:
    source = periodic_power_source()
    preparation = source.periodic_preparation
    assert preparation is not None
    points = tuple(rational_vector(row) for row in np.asarray(source.site_points))
    weights = rational_vector(np.asarray(source.site_weights))
    matrices = tuple(
        tuple(rational_vector(row) for row in matrix) for matrix in preparation.generators
    )
    with coordinate_reference_budget(
        queries_upper=len(preparation.image_sites)
    ) as budget:
        sites = _prepare_power_images(
            preparation,
            points,
            weights,
            _ExactPowerBudget(10_000_000, 16384, budget=budget),
            budget=budget,
        )
        expected = []
        for owner, exponents in zip(
            preparation.image_sites, preparation.image_exponents, strict=True
        ):
            action = exact_periodic_element(
                matrices,
                preparation.orders,
                tuple(int(value) for value in exponents),
                budget=budget,
            )
            point = points[int(owner)]
            expected.append(
                tuple(
                    sum(
                        (action[axis][column] * point[column] for column in range(3)),
                        action[axis][3],
                    )
                    for axis in range(3)
                )
            )
    assert sites.points == tuple(expected)
    assert sites.weights == tuple(weights[int(owner)] for owner in preparation.image_sites)


def test_linear_action_negative_translation_exponents_use_exact_inverse() -> None:
    source = periodic_power_source()
    owner = ExactPowerCellGeometryLinearActionSource(
        source,
        np.asarray([[0, -1], [0, 1], [1, -1]]),
        (
            (Fraction(1), Fraction(0)),
            (Fraction(1, 3), Fraction(2, 3)),
            (Fraction(1), Fraction(0)),
        ),
        np.asarray([[-1, 0, 0], [0, -2, 1], [-1, -1, -1]]),
        periodic_preparation=source.periodic_preparation,
    )
    with coordinate_reference_budget(
        queries_upper=owner.preparation_query_upper()
    ) as budget:
        parent = source.prepare(budget=budget)
        prepared = owner.prepare(budget=budget)
    origin, first = parent.vertices[0], parent.vertices[1]
    blended = tuple(
        Fraction(1, 3) * a + Fraction(2, 3) * b for a, b in zip(origin, first, strict=True)
    )
    assert prepared.vertices == (
        (origin[0] - 4, origin[1], origin[2]),
        (blended[0], blended[1] - 8, blended[2] + 4),
        (first[0] - 4, first[1] - 4, first[2] - 4),
    )
