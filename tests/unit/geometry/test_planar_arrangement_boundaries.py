#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from fractions import Fraction

import pytest

from phydrax.geometry.brep._planar_arrangement import (
    arrange,
    Cell,
    selected_cell_boundaries,
)


def points(*coordinates: tuple[int, int]) -> tuple[tuple[Fraction, Fraction], ...]:
    return tuple((Fraction(x), Fraction(y)) for x, y in coordinates)


def rectangle(
    x0: int, y0: int, x1: int, y1: int
) -> tuple[tuple[Fraction, Fraction], ...]:
    return points((x0, y0), (x1, y0), (x1, y1), (x0, y1))


def test_empty_and_disjoint_regions() -> None:
    first = rectangle(0, 0, 1, 1)
    second = rectangle(3, 0, 4, 1)
    cells = arrange(((first,), (second,)))
    assert selected_cell_boundaries(()) == ()
    assert selected_cell_boundaries(cells) == ((first,), (second,))


def test_shared_halfedges_cancel_and_collinear_breakpoints_survive() -> None:
    left = rectangle(0, 0, 1, 1)
    right = rectangle(1, 0, 2, 1)
    cells = arrange(((left,), (right,)))
    assert selected_cell_boundaries(cells) == (
        (points((0, 0), (1, 0), (2, 0), (2, 1), (1, 1), (0, 1)),),
    )


def test_hole_assignment_and_filling() -> None:
    outer = rectangle(0, 0, 4, 4)
    inner = rectangle(1, 1, 3, 3)
    cells = arrange(((outer,), (inner,)))
    ring = tuple(cell for cell in cells if cell.members == (0,))
    assert selected_cell_boundaries(ring) == (
        (outer, points((1, 1), (1, 3), (3, 3), (3, 1))),
    )
    assert selected_cell_boundaries(cells) == ((outer,),)


def test_hole_with_separate_island() -> None:
    outer = rectangle(0, 0, 6, 6)
    hole = rectangle(1, 1, 5, 5)
    island = rectangle(2, 2, 3, 3)
    cells = arrange(((outer, hole), (island,)))
    assert selected_cell_boundaries(cells) == (
        (outer, points((1, 1), (1, 5), (5, 5), (5, 1))),
        (island,),
    )


def test_point_touch_is_rejected() -> None:
    cells = (
        Cell((rectangle(0, 0, 1, 1),), (0,)),
        Cell((rectangle(1, 1, 2, 2),), (1,)),
    )
    with pytest.raises(ValueError, match="nonmanifold or self-touching"):
        selected_cell_boundaries(cells)


def test_non_atomic_contact_and_duplicate_cell_are_rejected() -> None:
    cell = Cell((rectangle(0, 0, 2, 1),), (0,))
    touching = Cell((rectangle(1, 1, 2, 2),), (1,))
    with pytest.raises(ValueError, match="atomic halfedges"):
        selected_cell_boundaries((cell, touching))
    with pytest.raises(ValueError, match="repeat a directed halfedge"):
        selected_cell_boundaries((cell, cell))
