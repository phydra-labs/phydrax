#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native declared-contact proofs are rational verdicts on random and
adversarial polyhedral meshes, and decide every pair of a conforming grid."""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import replace
from fractions import Fraction
from typing import cast

import numpy as np
import pytest

from phydrax._meshcore import meshcore_available
from phydrax.discretization._cell_geometry import CellGeometrySpec
from phydrax.discretization._cell_mesh import CellMesh
from phydrax.discretization._exact_algebraic_geometry import ExactScalar
from phydrax.geometry._embedding_cells import prepare_embedding_cells
from phydrax.geometry._global_embedding import _cell_simplices, _VolumePieceFilter
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


def test_native_contacts_refuse_a_flat_simplex_piece() -> None:
    mesh = _two_cubes((3.0, 0.0, 0.0))
    cube = prepare_embedding_cells(mesh, CellGeometrySpec.affine(mesh))[1]
    # Far from the cube, yet the rational pass refuses its zero-volume piece.
    flat = replace(
        cube,
        kind="tetrahedron",
        vertex_ids=(100, 101, 102, 103),
        corner_images=tuple(
            (Fraction(x), Fraction(y), Fraction(0))
            for x, y in ((9, 0), (10, 0), (9, 1), (10, 1))
        ),
        facet_routes=(),
    )
    cells = (flat, cube)
    piece_filter = _VolumePieceFilter.prepare(
        mesh, cells, [_cell_simplices(cell) for cell in cells]
    )
    if piece_filter is None:
        raise AssertionError("Both cells are straight cells in R3.")
    _, _, native = piece_filter.certify(
        np.asarray([0], dtype=np.intp), np.asarray([1], dtype=np.intp)
    )
    assert not native


def _native_pairs(
    corners: tuple[tuple[tuple[ExactScalar, ...], ...], ...],
    ids: tuple[tuple[int, ...], ...],
    routes: tuple[tuple[tuple[int, ...], ...], ...],
    pieces: tuple[tuple[tuple[ExactScalar, ...], ...], ...],
) -> set[tuple[int, int]]:
    mesh = _two_cubes((3.0, 0.0, 0.0))
    template = prepare_embedding_cells(mesh, CellGeometrySpec.affine(mesh))[0]
    cells = tuple(
        replace(
            template,
            global_id=index,
            vertex_ids=ids[index],
            corner_images=corners[index],
            facet_routes=routes[index],
        )
        for index in range(2)
    )
    piece_filter = _VolumePieceFilter.prepare(mesh, cells, [(pieces[0],), (pieces[1],)])
    if piece_filter is None:
        raise AssertionError("Both cells are straight cells in R3.")
    return piece_filter.certify(
        np.asarray([0], dtype=np.intp), np.asarray([1], dtype=np.intp)
    )[2]


def _shared_triangle_pair(
    number: Callable[[int], ExactScalar], /
) -> set[tuple[int, int]]:
    shared = tuple((number(x), number(y), number(0)) for x, y in ((0, 0), (3, 0), (0, 3)))
    apex = (number(1), number(1), number(-3))
    far = ((number(4), number(5), number(0)), (number(0), number(0), number(3)))
    far = (*far, (number(0), number(3), number(3)))
    return _native_pairs(
        ((*shared, apex), (*shared, *far)),
        ((0, 1, 2, 10), (0, 1, 2, 20, 21, 22)),
        (((0, 1, 2), (0, 2, 1)), ((0, 2, 1), (3, 4, 5))),
        ((*shared, apex), (shared[0], *far)),
    )


def test_native_contacts_need_fraction_corners() -> None:
    # Integer corners make the rational pass refuse what Fraction corners prove.
    assert not _shared_triangle_pair(cast(Callable[[int], ExactScalar], int))


def test_native_contacts_refuse_a_zero_length_declared_edge() -> None:
    third = Fraction(1, 3)
    p = (third, third, third)
    q = (third + Fraction(1, 2**70), third, third)
    below = ((Fraction(1), third, Fraction(-1)), (third, Fraction(1), Fraction(-1)))
    below = (*below, (third - 1, third - 1, Fraction(-1)))
    above = tuple((x, y, -z) for x, y, z in below)
    assert not _native_pairs(
        ((p, p, q, *below), (p, p, q, *above)),
        ((1, 2, 50, 51, 52, 53), (1, 2, 60, 61, 62, 63)),
        (((0, 1, 3), (0, 1, 4)), ((1, 0, 3), (0, 1, 5))),
        ((q, *below), (q, *above)),
    )
