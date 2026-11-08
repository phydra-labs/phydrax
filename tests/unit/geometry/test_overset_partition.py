#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from dataclasses import FrozenInstanceError
from fractions import Fraction
from itertools import combinations

import numpy as np
import pytest

from phydrax.geometry._overset_partition import (
    OversetPartition,
    partition_overset_triangles,
)
from phydrax.geometry._planar_primitives import area2, cross, intersection


type Part = tuple[str, np.ndarray, np.ndarray]
type Points = tuple[tuple[float, float], ...]


def _part(
    name: str, points: Points, cells: tuple[tuple[int, int, int], ...] = ((0, 1, 2),)
) -> Part:
    return name, np.asarray(points, dtype=np.float64), np.asarray(cells, dtype=np.int64)


def _triangle(name: str, x: float) -> Part:
    return _part(name, ((x, 0), (x + 4, 0), (x, 4)))


def _wall(points: Points) -> np.ndarray:
    return np.asarray(points, dtype=np.float64)


def _exact_wall(wall: np.ndarray) -> tuple[tuple[Fraction, Fraction], ...]:
    return tuple(
        (Fraction.from_float(float(x)), Fraction.from_float(float(y))) for x, y in wall
    )


def _assert_exact_partition(
    result: OversetPartition,
    parts: tuple[Part, ...],
    cavities: tuple[np.ndarray, ...] = (),
) -> None:
    assert result.total_area == sum((piece.area for piece in result.pieces), Fraction(0))
    assert result.total_area == sum(
        (cell.owned_area for cell in result.cells), Fraction(0)
    )
    assert len(result.cells) == sum(len(part[2]) for part in parts)
    for cell in result.cells:
        owned = sum(
            (
                piece.area
                for piece in result.pieces
                if (piece.part_index, piece.cell_index)
                == (cell.part_index, cell.cell_index)
            ),
            Fraction(0),
        )
        assert owned == cell.owned_area
        assert cell.source_area == cell.owned_area + cell.excluded_area
        assert 0 <= cell.cavity_area <= cell.excluded_area
        assert cell.excluded_area >= 0
    for piece in result.pieces:
        name, coordinates, cells = parts[piece.part_index]
        assert piece.part_name == name
        source = tuple(
            (
                Fraction.from_float(float(coordinates[i, 0])),
                Fraction.from_float(float(coordinates[i, 1])),
            )
            for i in cells[piece.cell_index]
        )
        assert piece.area == area2(piece.vertices) / 2 > 0
        assert piece.area == area2(intersection(piece.vertices, source)) / 2
        assert piece.edges == tuple(
            zip(piece.vertices, piece.vertices[1:] + piece.vertices[:1], strict=True)
        )
        assert all(
            cross(a, b, point) >= 0 for a, b in piece.edges for point in piece.vertices
        )
        for wall in cavities:
            assert area2(intersection(piece.vertices, _exact_wall(wall))) == 0
    for first, second in combinations(result.pieces, 2):
        assert area2(intersection(first.vertices, second.vertices)) == 0


def test_three_overlapping_parts_are_owned_once_in_priority_order() -> None:
    parts = (_triangle("first", 0), _triangle("second", 1), _triangle("third", 2))
    result = partition_overset_triangles(parts, ())
    _assert_exact_partition(result, parts)
    assert result.total_area == 15
    assert [cell.source_area for cell in result.cells] == [8, 8, 8]
    assert [cell.owned_area for cell in result.cells] == [
        8,
        Fraction(7, 2),
        Fraction(7, 2),
    ]
    assert [cell.excluded_area for cell in result.cells] == [
        0,
        Fraction(9, 2),
        Fraction(9, 2),
    ]


def test_priority_order_changes_ownership_not_union() -> None:
    first, second = _triangle("left", 0), _triangle("right", 1)
    forward = partition_overset_triangles((first, second), ())
    backward = partition_overset_triangles((second, first), ())
    _assert_exact_partition(forward, (first, second))
    _assert_exact_partition(backward, (second, first))
    assert forward.total_area == backward.total_area == Fraction(23, 2)
    assert {c.part_name: c.owned_area for c in forward.cells} == {
        "left": 8,
        "right": Fraction(7, 2),
    }
    assert {c.part_name: c.owned_area for c in backward.cells} == {
        "left": Fraction(7, 2),
        "right": 8,
    }


def test_overlapping_cavities_are_subtracted_once_from_the_union() -> None:
    parts = (_triangle("first", 0), _triangle("second", 1), _triangle("third", 2))
    cavities = (
        _wall(((1, 0), (2, 0), (2, 1), (1, 1))),
        _wall(((1.5, 1), (2.5, 1), (2.5, 0), (1.5, 0))),
    )
    result = partition_overset_triangles(parts, cavities)
    _assert_exact_partition(result, parts, cavities)
    assert result.total_area == Fraction(27, 2)
    assert [cell.owned_area for cell in result.cells] == [
        Fraction(13, 2),
        Fraction(7, 2),
        Fraction(7, 2),
    ]
    assert [cell.cavity_area for cell in result.cells] == [
        Fraction(3, 2),
        Fraction(3, 2),
        Fraction(1, 2),
    ]


def test_original_cells_and_zero_owned_cells_are_recorded() -> None:
    square = _part("square", ((0, 0), (2, 0), (2, 2), (0, 2)), ((0, 1, 2), (0, 2, 3)))
    duplicate = _part("lower", ((0, 0), (2, 0), (2, 2), (0, 2)), ((0, 1, 2), (0, 2, 3)))
    result = partition_overset_triangles((square, duplicate), ())
    _assert_exact_partition(result, (square, duplicate))
    assert result.total_area == 4
    assert [(c.part_index, c.cell_index, c.owned_area) for c in result.cells] == [
        (0, 0, 2),
        (0, 1, 2),
        (1, 0, 0),
        (1, 1, 0),
    ]
    assert {piece.part_name for piece in result.pieces} == {"square"}


def test_non_binary_intersections_and_immutable_evidence() -> None:
    parts = (_triangle("base", 0), _part("cross", ((1, -1), (3, 2), (1, 2))))
    result = partition_overset_triangles(parts, ())
    _assert_exact_partition(result, parts)
    assert (Fraction(13, 5), Fraction(7, 5)) in {
        vertex for piece in result.pieces for vertex in piece.vertices
    }
    with pytest.raises(FrozenInstanceError):
        result.total_area = Fraction(0)  # ty: ignore[invalid-assignment]
    with pytest.raises(FrozenInstanceError):
        result.pieces[0].area = Fraction(0)  # ty: ignore[invalid-assignment]


def test_empty_meshes_reversed_cells_and_cavity_cover() -> None:
    empty = _part("empty", (), ())
    empty = (empty[0], empty[1].reshape(0, 2), empty[2].reshape(0, 3))
    reverse = _part("reverse", ((0, 0), (0, 2), (2, 0)))
    cavity = _wall(((-1, -1), (3, -1), (3, 3), (-1, 3), (-1, -1)))
    result = partition_overset_triangles((empty, reverse), (cavity,))
    _assert_exact_partition(result, (empty, reverse), (cavity,))
    assert result.pieces == ()
    assert result.total_area == 0
    assert result.cells[0].source_area == result.cells[0].excluded_area == 2
    assert partition_overset_triangles((), ()).total_area == 0


@pytest.mark.parametrize(
    "wall",
    [
        ((0, 0), (2, 0), (1, 1), (2, 2), (0, 2)),
        ((0, 0), (2, 2), (0, 2), (2, 0)),
        ((0, 0), (1, 0), (2, 0)),
        ((0, 0), (2, 0), (1, 0), (2, 2), (0, 2)),
    ],
)
def test_rejects_nonconvex_degenerate_and_backtracking_cavities(wall: Points) -> None:
    with pytest.raises(ValueError):
        partition_overset_triangles((_triangle("part", 0),), (_wall(wall),))


def test_rejects_overlapping_cells_within_one_part() -> None:
    part = _part(
        "bad", ((0, 0), (4, 0), (0, 4), (1, 0), (5, 0), (1, 4)), ((0, 1, 2), (3, 4, 5))
    )
    with pytest.raises(ValueError, match="disjoint interiors"):
        partition_overset_triangles((part,), ())


def test_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="positive area"):
        partition_overset_triangles((_part("bad", ((0, 0), (1, 0), (2, 0))),), ())
    with pytest.raises(ValueError, match="out of range"):
        partition_overset_triangles(
            (_part("bad", ((0, 0), (1, 0), (0, 1)), ((0, 1, 3),)),), ()
        )
    with pytest.raises(ValueError, match="finite"):
        partition_overset_triangles((_part("bad", ((0, 0), (1, 0), (0, np.nan))),), ())
    with pytest.raises(ValueError, match="strings"):
        partition_overset_triangles((_triangle(1, 0),), ())  # ty: ignore[invalid-argument-type]


def test_same_part_names_still_preserve_distinct_source_indices() -> None:
    parts = (_triangle("same", 0), _triangle("same", 1))
    result = partition_overset_triangles(parts, ())
    _assert_exact_partition(result, parts)
    assert [cell.owned_area for cell in result.cells] == [8, Fraction(7, 2)]
