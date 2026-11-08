#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from fractions import Fraction as F

import numpy as np
import pytest

from phydrax.geometry._overset_interfaces import partition_overset_interfaces
from phydrax.geometry._overset_partition import (
    OversetPartition,
    OwnedPolygon,
    partition_overset_triangles,
)
from phydrax.geometry._planar_primitives import signed_area2


def _piece(
    name: str, part: int, cell: int, points: list[tuple[int | F, int | F]]
) -> OwnedPolygon:
    vertices = tuple((F(x), F(y)) for x, y in points)
    return OwnedPolygon(name, part, cell, vertices, signed_area2(vertices) / 2)


def _partition(*pieces: OwnedPolygon) -> OversetPartition:
    return OversetPartition(pieces, sum((p.area for p in pieces), F(0)), ())


def test_t_junction_and_multiple_parts_have_exact_atomic_ownership() -> None:
    left = _piece("left", 0, 7, [(0, 0), (1, 0), (1, 2), (0, 2)])
    lower = _piece("lower", 1, 11, [(1, 0), (2, 0), (2, 1), (1, 1)])
    upper = _piece("upper", 2, 13, [(1, 1), (2, 1), (2, 2), (1, 2)])
    edges = partition_overset_interfaces(_partition(left, lower, upper))
    by_endpoints = {(e.start, e.end): e for e in edges}
    assert len(edges) == 10
    for lo, hi, name, part, cell in [(0, 1, "lower", 1, 11), (1, 2, "upper", 2, 13)]:
        edge = by_endpoints[((F(1), F(lo)), (F(1), F(hi)))]
        assert edge.kind == "inter_part"
        assert [
            (o.part_name, o.part_index, o.cell_index, o.orientation) for o in edge.owners
        ] == [("left", 0, 7, 1), (name, part, cell, -1)]
        assert edge.normal == (F(1), F(0))
        assert edge.length_squared == 1
    assert sum(e.kind == "inter_part" for e in edges) == 3
    assert sum(e.kind == "physical_boundary" for e in edges) == 7


def test_same_part_edges_and_rational_geometry() -> None:
    a = _piece("part", 0, 3, [(0, 0), (F(1, 3), 0), (0, 1)])
    b = _piece("part", 0, 4, [(F(1, 3), 0), (F(1, 3), 1), (0, 1)])
    edges = partition_overset_interfaces(_partition(a, b))
    shared = next(e for e in edges if e.kind == "same_part")
    assert (shared.start, shared.end) == ((F(0), F(1)), (F(1, 3), F(0)))
    assert shared.normal == (F(-1), F(-1, 3))
    assert shared.length_squared == F(10, 9)
    assert {owner.cell_index for owner in shared.owners} == {3, 4}
    assert sum(e.kind == "physical_boundary" for e in edges) == 4


def test_disconnected_cavity_boundaries_are_exposed() -> None:
    pieces = (
        _piece("p", 0, 0, [(0, 0), (3, 0), (3, 1), (0, 1)]),
        _piece("p", 0, 1, [(0, 2), (3, 2), (3, 3), (0, 3)]),
        _piece("p", 0, 2, [(0, 1), (1, 1), (1, 2), (0, 2)]),
        _piece("p", 0, 3, [(2, 1), (3, 1), (3, 2), (2, 2)]),
    )
    edges = partition_overset_interfaces(_partition(*pieces))
    boundary = {(e.start, e.end) for e in edges if e.kind == "physical_boundary"}
    assert ((F(1), F(1)), (F(2), F(1))) in boundary
    assert ((F(1), F(2)), (F(2), F(2))) in boundary
    assert ((F(1), F(1)), (F(1), F(2))) in boundary
    assert ((F(2), F(1)), (F(2), F(2))) in boundary
    assert sum(e.kind == "same_part" for e in edges) == 4


def test_rejects_overlaps_including_containment_and_crossings() -> None:
    outer = _piece("a", 0, 0, [(0, 0), (4, 0), (4, 4), (0, 4)])
    inside = _piece("b", 1, 0, [(1, 1), (2, 1), (2, 2), (1, 2)])
    crossing = _piece("c", 2, 0, [(3, -1), (5, -1), (5, 1), (3, 1)])
    for other in (inside, crossing, outer):
        with pytest.raises(ValueError, match="overlap"):
            partition_overset_interfaces(_partition(outer, other))


def test_rejects_invalid_orientation_and_inconsistent_identity() -> None:
    clockwise = _piece("a", 0, 0, [(0, 0), (0, 1), (1, 0)])
    with pytest.raises(ValueError, match="invalid"):
        partition_overset_interfaces(_partition(clockwise))
    a = _piece("a", 0, 0, [(0, 0), (1, 0), (1, 1), (0, 1)])
    b = _piece("b", 0, 1, [(1, 0), (2, 0), (2, 1), (1, 1)])
    with pytest.raises(ValueError, match="identity"):
        partition_overset_interfaces(_partition(a, b))


def test_point_touching_components_are_nonmanifold() -> None:
    a = _piece("a", 0, 0, [(0, 0), (1, 0), (1, 1), (0, 1)])
    b = _piece("b", 1, 0, [(1, 1), (2, 1), (2, 2), (1, 2)])
    with pytest.raises(ValueError, match="nonmanifold"):
        partition_overset_interfaces(_partition(a, b))


def test_priority_partition_with_multiple_bodies_and_cavities_has_exact_coverage() -> (
    None
):
    def square(name: str, x: int) -> tuple[str, np.ndarray, np.ndarray]:
        points = np.array([[x, 0], [x + 2, 0], [x + 2, 2], [x, 2]], dtype=np.float64)
        cells = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        return name, points, cells

    cavities = (
        np.array([[0.25, 0.25], [0.5, 0.25], [0.5, 0.5], [0.25, 0.5]], dtype=np.float64),
        np.array([[3.25, 1.25], [3.5, 1.25], [3.5, 1.5], [3.25, 1.5]], dtype=np.float64),
    )
    partition = partition_overset_triangles(
        (square("a", 0), square("b", 1), square("c", 2)), cavities
    )
    edges = partition_overset_interfaces(partition)
    assert {o.part_name for e in edges for o in e.owners} == {"a", "b", "c"}
    assert {e.kind for e in edges} == {"physical_boundary", "same_part", "inter_part"}
    for index, piece in enumerate(partition.pieces):
        expected = tuple(
            sum((abs(b[axis] - a[axis]) for a, b in piece.edges), F(0))
            for axis in range(2)
        )
        actual = tuple(
            sum(
                (
                    abs(e.end[axis] - e.start[axis])
                    for e in edges
                    for owner in e.owners
                    if owner.piece_index == index
                ),
                F(0),
            )
            for axis in range(2)
        )
        assert actual == expected
    for cavity in cavities:
        for x, y in cavity:
            point = (F.from_float(float(x)), F.from_float(float(y)))
            assert any(
                point in (e.start, e.end) and e.kind == "physical_boundary" for e in edges
            )
