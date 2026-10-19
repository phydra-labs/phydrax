# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Rational child correspondence reuses source evaluations at shared corners."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from phydrax._resource_ticket import HostBatchTicket
from phydrax.discretization._cell_geometry import coordinate_lagrange_element
from phydrax.discretization._cell_geometry_transfer import (
    _plc_nested_bound,
    _plc_parent_polynomials,
)
from phydrax.discretization._cell_geometry_transition_types import (
    NestedReferenceWitnesses,
)
from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.discretization._coordinate_enclosure import evaluate, Polynomial


@pytest.mark.parametrize(
    "offset", (Fraction(0), Fraction(1, 8)), ids=("exact", "changed-shared-corner")
)
def test_child_correspondence_reuses_exact_source_corners_under_original_work(
    offset: Fraction,
) -> None:
    root = (
        (Fraction(0), Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(1)),
    )
    midpoint = (Fraction(1, 2), Fraction(0), Fraction(0))
    first = (root[0], midpoint, root[2], root[3])
    second = (midpoint, root[1], (offset, Fraction(1), Fraction(0)), root[3])
    references = np.asarray(
        ((root[0], midpoint, root[2], root[3]), (midpoint, root[1], root[2], root[3])),
        dtype=np.float64,
    )
    witnesses = NestedReferenceWitnesses(
        np.asarray((11, 12), dtype=np.int64),
        np.asarray((7, 7), dtype=np.int64),
        references,
    )
    # Four root corners cost 9 each and the midpoint costs 12. Re-evaluating
    # shared corners costs 78, which cannot pass this unchanged arithmetic cap.
    with HostBatchTicket(48, bytes_upper=1 << 20, queries_upper=0) as ticket:
        with CoordinateEnclosureBudget(
            ticket, maximum_work_units=48, maximum_memory_bytes=1 << 20
        ).activate() as budget:
            bound = _plc_nested_bound(
                witnesses, {11: first, 12: second}, {7: root}, budget=budget
            )
            assert budget.work_units == 48
            assert budget.temporary_bytes_upper == 0
    assert bound == offset


def test_children_reuse_parent_polynomials_without_recharging_preparation() -> None:
    element = coordinate_lagrange_element("tetrahedron", 1)
    roots = tuple(
        tuple(Fraction(float(value)) for value in point)
        for point in np.asarray(element.reference_nodes)
    )
    shifted = tuple((point[0] + 2, *point[1:]) for point in roots)
    bank = (*roots, *shifted)
    route = np.arange(4, dtype=np.int64)
    memo: dict[int, tuple[Polynomial, ...]] = {}
    with HostBatchTicket(100_000, bytes_upper=1 << 20, queries_upper=0) as ticket:
        with CoordinateEnclosureBudget(
            ticket, maximum_work_units=100_000, maximum_memory_bytes=1 << 20
        ).activate() as budget:
            first = _plc_parent_polynomials(0, element, bank, route, memo, budget=budget)
            prepared_work = budget.work_units
            repeated = _plc_parent_polynomials(
                0, element, bank, route, memo, budget=budget
            )
            assert budget.work_units == prepared_work
            other = _plc_parent_polynomials(
                1, element, bank, route + 4, memo, budget=budget
            )
            point = (Fraction(1, 4),) * 3
            assert evaluate(first[0], point, budget=budget) == Fraction(1, 4)
            assert evaluate(repeated[0], point, budget=budget) == Fraction(1, 4)
            assert evaluate(other[0], point, budget=budget) == Fraction(9, 4)
