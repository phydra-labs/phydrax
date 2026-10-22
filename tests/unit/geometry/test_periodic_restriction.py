# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Exact boundary-subdivision and parent-partition premises of the restriction theorem."""

from collections.abc import Iterator
from contextlib import contextmanager
from fractions import Fraction

import numpy as np
import pytest

from phydrax._resource_ticket import HostBatchTicket
from phydrax.discretization import CellMesh, PolyhedralConnectivity
from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.geometry import _periodic_restriction as restriction


@contextmanager
def _budget() -> Iterator[CoordinateEnclosureBudget]:
    work, memory = 10_000_000, 256 * 1024**2
    with HostBatchTicket(work, bytes_upper=memory) as ticket:
        with CoordinateEnclosureBudget(
            ticket, maximum_work_units=work, maximum_memory_bytes=memory
        ).activate() as budget:
            yield budget


def _box(
    lower: tuple[int | Fraction, int | Fraction, int | Fraction],
    upper: tuple[int | Fraction, int | Fraction, int | Fraction],
    *,
    split_top: bool = False,
    conforming: bool = True,
) -> tuple[dict[int, tuple[Fraction, ...]], tuple[tuple[int, ...], ...]]:
    """Return a box's exact vertex bank and outward counter-clockwise face loops.

    ``split_top`` splits the top face at ``x = mid`` into two rectangles; with
    ``conforming=False`` the two side faces keep their unsplit top edges.
    """
    (x0, y0, z0), (x1, y1, z1) = (
        tuple(Fraction(value) for value in lower),
        tuple(Fraction(value) for value in upper),
    )
    points: dict[int, tuple[Fraction, ...]] = {}

    def vertex(x: Fraction, y: Fraction, z: Fraction) -> int:
        key = (x, y, z)
        for index, point in points.items():
            if point == key:
                return index
        points[len(points)] = key
        return len(points) - 1

    a, b, c, d = (vertex(x0, y0, z0), vertex(x1, y0, z0), vertex(x1, y1, z0), vertex(x0, y1, z0))
    e, f, g, h = (vertex(x0, y0, z1), vertex(x1, y0, z1), vertex(x1, y1, z1), vertex(x0, y1, z1))
    faces: list[tuple[int, ...]] = [(a, d, c, b), (a, e, h, d), (b, c, g, f)]
    if split_top:
        mid = (x0 + x1) / 2
        m, n = vertex(mid, y0, z1), vertex(mid, y1, z1)
        faces += [(e, m, n, h), (m, f, g, n)]
        faces += (
            [(a, b, f, m, e), (d, h, n, g, c)]
            if conforming
            else [(a, b, f, e), (d, h, g, c)]
        )
    else:
        faces += [(e, f, g, h), (a, b, f, e), (d, h, g, c)]
    return points, tuple(faces)


def _cell(
    cell_id: int,
    points: dict[int, tuple[Fraction, ...]],
    faces: tuple[tuple[int, ...], ...],
    budget: CoordinateEnclosureBudget,
) -> restriction._ConvexCell | None:
    return restriction._convex_cell(
        cell_id, faces, tuple(points[row] for row in range(len(points))), budget=budget
    )


def test_conforming_box_subdivisions_are_proved_with_exact_volume() -> None:
    with _budget() as budget:
        cube = _cell(0, *_box((0, 0, 0), (1, 1, 1)), budget)
        split = _cell(1, *_box((0, 0, 0), (1, 1, 1), split_top=True), budget)
        inward = _cell(
            2,
            _box((0, 0, 0), (1, 1, 1))[0],
            tuple(face[::-1] for face in _box((0, 0, 0), (1, 1, 1))[1]),
            budget,
        )
    assert cube is not None and cube.volume == 1 and len(cube.planes) == 6
    assert split is not None and split.volume == 1 and len(split.planes) == 6
    assert inward is not None and inward.volume == 1


def test_overlapping_coplanar_faces_are_refused() -> None:
    points, faces = _box((0, 0, 0), (2, 1, 1), split_top=True)
    # Replace the split top by two rectangles that overlap on [1/2, 3/2].
    top = {tuple(sorted(face)) for face in faces[3:5]}
    rest = tuple(face for face in faces if tuple(sorted(face)) not in top)
    index = {point: key for key, point in points.items()}
    for x in (Fraction(1, 2), Fraction(3, 2)):
        for y in (Fraction(0), Fraction(1)):
            index.setdefault((x, y, Fraction(1)), len(points))
            points[index[(x, y, Fraction(1))]] = (x, y, Fraction(1))

    def v(x: int | Fraction, y: int | Fraction) -> int:
        return index[(Fraction(x), Fraction(y), Fraction(1))]

    overlapping = (
        (v(0, 0), v(Fraction(3, 2), 0), v(Fraction(3, 2), 1), v(0, 1)),
        (v(Fraction(1, 2), 0), v(2, 0), v(2, 1), v(Fraction(1, 2), 1)),
    )
    with _budget() as budget:
        assert _cell(0, points, rest + overlapping, budget) is None
        # A duplicated sheet (the same top face twice) is refused as well.
        cube_points, cube_faces = _box((0, 0, 0), (1, 1, 1))
        assert _cell(1, cube_points, cube_faces + (cube_faces[3],), budget) is None


def test_t_junction_and_pinched_vertex_link_are_refused() -> None:
    with _budget() as budget:
        assert (
            _cell(0, *_box((0, 0, 0), (1, 1, 1), split_top=True, conforming=False), budget)
            is None
        )
        # Two unit cubes sharing one corner: every edge is paired, but the
        # common vertex has two link cycles and the union is not convex.
        first_points, first_faces = _box((0, 0, 0), (1, 1, 1))
        second_points, second_faces = _box((1, 1, 1), (2, 2, 2))
        points = dict(first_points)
        remap = {}
        for key, point in second_points.items():
            existing = next((k for k, p in points.items() if p == point), None)
            remap[key] = existing if existing is not None else len(points)
            points[remap[key]] = point
        faces = first_faces + tuple(tuple(remap[v] for v in face) for face in second_faces)
        assert _cell(1, points, faces, budget) is None


def test_inconsistent_orientation_and_wrong_supporting_plane_are_refused() -> None:
    points, faces = _box((0, 0, 0), (1, 1, 1))
    flipped = (faces[0][::-1],) + faces[1:]
    lifted = dict(points)
    # Lift one top vertex off its supporting plane: the face is no longer planar
    # on an outward support of the hull.
    top = faces[3][0]
    lifted[top] = (lifted[top][0], lifted[top][1], Fraction(5, 4))
    with _budget() as budget:
        assert _cell(0, points, flipped, budget) is None
        assert _cell(1, lifted, faces, budget) is None


type _Corner = tuple[int | Fraction, int | Fraction, int | Fraction]
type _Boxes = tuple[tuple[_Corner, _Corner], ...]


def _partition(
    children: _Boxes, planes: tuple[tuple[int | Fraction, ...], ...]
) -> tuple[tuple[int, int, tuple[int, ...], Fraction], ...] | None:
    with _budget() as budget:
        parent = _cell(0, *_box((0, 0, 0), (1, 1, 1)), budget)
        assert parent is not None
        proved: list[tuple[int, restriction._ConvexCell]] = []
        for index, (lower, upper) in enumerate(children):
            child = _cell(index + 1, *_box(lower, upper), budget)
            assert child is not None
            proved.append((0, child))
        plane_rows = tuple(tuple(Fraction(value) for value in row) for row in planes)
        return restriction._parent_partition(
            {0: parent}, tuple(proved), plane_rows, budget=budget
        )


_HALF = ((1, 0, 0, Fraction(-1, 2)),)


def test_exact_parent_partition_is_proved() -> None:
    rows = _partition(
        (((0, 0, 0), (Fraction(1, 2), 1, 1)), ((Fraction(1, 2), 0, 0), (1, 1, 1))), _HALF
    )
    assert rows is not None
    assert {row[2] for row in rows} == {(-1,), (1,)}
    assert sum(row[3] for row in rows) == 1


@pytest.mark.parametrize(
    ("children", "planes"),
    [
        pytest.param((((0, 0, 0), (Fraction(1, 2), 1, 1)),), _HALF, id="missing-child"),
        pytest.param(
            (((0, 0, 0), (Fraction(1, 2), 1, 1)), ((Fraction(3, 5), 0, 0), (1, 1, 1))),
            _HALF,
            id="sliver-gap",
        ),
        pytest.param(
            (((0, 0, 0), (Fraction(1, 2), 1, 1)), ((Fraction(1, 2), 0, 0), (1, 1, 1))),
            ((1, 0, 0, Fraction(-1, 3)),),
            id="wrong-plane",
        ),
        pytest.param(
            (((0, 0, 0), (Fraction(1, 2), 1, 1)), ((Fraction(3, 2), 0, 0), (2, 1, 1))),
            _HALF,
            id="wrong-lift",
        ),
        pytest.param(
            (((0, 0, 0), (Fraction(1, 2), 1, 1)), ((0, 0, 0), (Fraction(1, 2), 1, 1))),
            _HALF,
            id="duplicated-sign-cell",
        ),
    ],
)
def test_parent_partition_counterexamples_are_refused(
    children: _Boxes, planes: tuple[tuple[int | Fraction, ...], ...]
) -> None:
    assert _partition(children, planes) is None


def test_altered_seam_pairing_is_refused() -> None:
    with _budget() as budget:
        assert restriction._paired_quotient_facets(
            np.asarray([0, 0, 1, 1, 2, 2]), budget=budget
        )
        # An altered seam leaves one orbit unmatched and another over-shared.
        assert not restriction._paired_quotient_facets(
            np.asarray([0, 0, 1, 2, 2, 2]), budget=budget
        )
        assert not restriction._paired_quotient_facets(
            np.asarray([], dtype=np.int64), budget=budget
        )


def test_inverse_requires_complete_signed_source_boundary_subcomplex() -> None:
    points: dict[int, tuple[Fraction, ...]] = {}
    cells: list[tuple[tuple[int, ...], ...]] = []
    for lower, upper in (
        ((0, 0, 0), (Fraction(1, 2), 1, 1)),
        ((Fraction(1, 2), 0, 0), (1, 1, 1)),
    ):
        local, faces = _box(lower, upper)
        remap: dict[int, int] = {}
        for row, point in local.items():
            existing = next((index for index, value in points.items() if value == point), len(points))
            points[existing] = point
            remap[row] = existing
        cells.append(tuple(tuple(remap[row] for row in face) for face in faces))
    coordinates = np.asarray(tuple(points.values()), dtype=np.float64)
    source = CellMesh.from_polyhedra(coordinates, tuple(cells))
    assert isinstance(source.connectivity, PolyhedralConnectivity)
    loops = restriction._cell_faces(source.connectivity)
    counts: dict[tuple[int, ...], int] = {}
    for _, faces in loops:
        for face in faces:
            key = tuple(sorted(face))
            counts[key] = counts.get(key, 0) + 1
    boundary = tuple(
        face for _, faces in loops for face in faces if counts[tuple(sorted(face))] == 1
    )
    target = CellMesh.from_polyhedra(coordinates, (boundary,))
    assert isinstance(target.connectivity, PolyhedralConnectivity)
    bank = tuple(points[row] for row in range(len(points)))
    with _budget() as budget:
        assert restriction._inverse_boundary_subcomplex(
            source.connectivity, target.connectivity, bank, bank, {0: 0, 1: 0},
            budget=budget,
        )
        assert not restriction._inverse_boundary_subcomplex(
            source.connectivity, target.connectivity, bank, bank, {0: 0},
            budget=budget,
        )
        assert not restriction._inverse_boundary_subcomplex(
            source.connectivity, target.connectivity, bank, bank, {0: 0, 1: 1},
            budget=budget,
        )
        shifted = tuple((point[0] + 1, point[1], point[2]) for point in bank)
        assert not restriction._inverse_boundary_subcomplex(
            source.connectivity, target.connectivity, bank, shifted, {0: 0, 1: 0},
            budget=budget,
        )
