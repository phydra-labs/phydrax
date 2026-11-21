#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native declared-contact proofs are rational verdicts on random and
adversarial polyhedral meshes, and decide every pair of a conforming grid."""

from __future__ import annotations

import random

import numpy as np
import pytest

from phydrax._meshcore import meshcore_available
from phydrax.discretization._cell_mesh import CellMesh
from tests._support.declared_contacts import declared_contact_differential, non_dyadic


pytestmark = pytest.mark.skipif(
    not meshcore_available(), reason="the native meshcore library is absent"
)

# Outward loops of the unit cube over corner offsets (x, y, z).
_CUBE_FACES = (
    ((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)),
    ((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)),
    ((0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)),
    ((1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)),
    ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)),
    ((0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)),
)

type Cells = list[list[np.ndarray]]


def _grid(size: int, generator: random.Random, /) -> tuple[np.ndarray, Cells]:
    """A size^3 grid of unit cubes, every face loop from a random start."""
    count = size + 1
    points = np.asarray(
        [(x, y, z) for x in range(count) for y in range(count) for z in range(count)],
        dtype=np.float64,
    )

    def vertex(x: int, y: int, z: int, /) -> int:
        return (x * count + y) * count + z

    cells: Cells = []
    for x in range(size):
        for y in range(size):
            for z in range(size):
                loops = []
                for face in _CUBE_FACES:
                    loop = [vertex(x + a, y + b, z + c) for a, b, c in face]
                    start = generator.randrange(len(loop))
                    loops.append(np.asarray(loop[start:] + loop[:start], dtype=np.int32))
                cells.append(loops)
    return points, cells


def _all_pairs(cells: Cells, /) -> set[tuple[int, int]]:
    return {
        (left, right)
        for left in range(len(cells))
        for right in range(left + 1, len(cells))
    }


@pytest.mark.parametrize("seed", range(3))
def test_native_contacts_decide_a_mapped_conforming_grid(seed: int) -> None:
    generator = random.Random(seed)
    points, cells = _grid(2, generator)
    # A dyadic shear keeps every face planar and convex and exact in binary64.
    matrix = np.eye(3) + np.asarray(
        [[generator.randint(-2, 2) / 8 for _ in range(3)] for _ in range(3)]
    )
    mesh = CellMesh.from_polyhedra(points @ matrix.T, cells)
    for corner_map in (None, non_dyadic):
        native, rational = declared_contact_differential(mesh, corner_map)
        assert native == rational == _all_pairs(cells)


@pytest.mark.parametrize("seed", range(3))
def test_native_contacts_decide_folded_faces_by_their_fans(seed: int) -> None:
    generator = random.Random(seed)
    points, cells = _grid(2, generator)
    # The interior vertex moves off every plane through it: its faces fold.
    points[13] += [generator.randint(-3, 3) / 16 for _ in range(3)]
    mesh = CellMesh.from_polyhedra(points, cells)
    for corner_map in (None, non_dyadic):
        native, rational = declared_contact_differential(mesh, corner_map)
        assert native == rational == _all_pairs(cells)


def _two_cubes(offset: tuple[float, float, float], /) -> CellMesh:
    """Unit cubes at the origin and at ``offset`` with distinct vertex identities."""
    corners = [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)]
    points = np.asarray(
        [*corners, *(np.add(corner, offset) for corner in corners)], dtype=np.float64
    )
    cells = [
        [
            np.asarray([shift + corners.index(corner) for corner in face], dtype=np.int32)
            for face in _CUBE_FACES
        ]
        for shift in (0, 8)
    ]
    return CellMesh.from_polyhedra(points, cells)


@pytest.mark.parametrize(
    "offset",
    ((0.5, 0.0, 0.0), (1.0, 0.5, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 1.0), (2.0, 0.0, 0.0)),
    ids=("overlap", "offset-face", "duplicate-face", "duplicate-corner", "apart"),
)
def test_native_contacts_refuse_undeclared_contacts(
    offset: tuple[float, float, float],
) -> None:
    mesh = _two_cubes(offset)
    for corner_map in (None, non_dyadic):
        native, rational = declared_contact_differential(mesh, corner_map)
        assert native == rational
        assert native == ({(0, 1)} if offset == (2.0, 0.0, 0.0) else set())


def test_native_contacts_refuse_a_hanging_vertex_interface() -> None:
    points, cells = _grid(1, random.Random(3))
    # A second cube on x = 1 whose shared face and bottom carry a hanging
    # vertex m on the edge a-b that the first cube does not declare.
    far = np.asarray(
        [(2, y, z) for y in (0, 1) for z in (0, 1)] + [(1, 0.5, 0)], dtype=np.float64
    )
    points = np.vstack((points, far))
    # First-cube vertex (x, y, z) is 4 x + 2 y + z.
    a, b, c, d = 4, 6, 7, 5
    e, h, f, g, m = 8, 9, 10, 11, 12
    second = [
        np.asarray(loop, dtype=np.int32)
        for loop in (
            (a, d, c, b, m),
            (a, m, b, f, e),
            (d, h, g, c),
            (e, f, g, h),
            (a, e, h, d),
            (b, c, g, f),
        )
    ]
    mesh = CellMesh.from_polyhedra(points, [cells[0], second])
    native, rational = declared_contact_differential(mesh)
    assert native <= rational
    assert not native
