#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Two prisms over one notched plane: conforming notch fills certify, mismatched
fills are refused, whatever the declared shared stratum looks like."""

from itertools import product

import numpy as np
import pytest

from phydrax.discretization._cell_geometry import CellGeometrySpec
from phydrax.discretization._cell_geometry_validity import (
    certify_cell_geometry_validity,
)
from phydrax.discretization._cell_mesh import CellMesh
from phydrax.geometry._global_embedding import certify_global_embedding


# Plane z = 0: a, b, c, d the square corners, u on edge d-a; the notch under
# the chord a-b has either one apex r with a hanging vertex s on the chord
# ("hanging") or two apexes r1, r2 ("two").
_A, _B, _C, _D, _U, _R1, _R2, _M, _E, _N = range(10)
_PLANE = {
    _A: (0.0, 0.0),
    _B: (4.0, 0.0),
    _C: (4.0, 4.0),
    _D: (0.0, 4.0),
    _U: (0.0, 2.0),
    _M: (-2.0, 2.0),
    _E: (-2.0, 0.0),
    _N: (-2.0, 4.0),
}
_NOTCHES = {
    "hanging": {_R1: (2.0, 0.5), _R2: (2.0, 0.0)},
    "two": {_R1: (1.0, 0.5), _R2: (3.0, 0.5)},
}
# Notch boundary from a to b (part of the face P) and the chord from a to b.
_NOTCH_PATH = {"hanging": (_R1,), "two": (_R1, _R2)}
_CHORD = {"hanging": (_R2,), "two": ()}
_FILLS = {
    "hanging": {
        "quad": ((_A, _R2, _B, _R1),),
        "split": ((_A, _R2, _R1), (_R2, _B, _R1)),
    },
    "two": {
        "quad": ((_A, _B, _R2, _R1),),
        "diagonal-a-r2": ((_A, _B, _R2), (_A, _R2, _R1)),
        "diagonal-b-r1": ((_A, _B, _R1), (_B, _R2, _R1)),
    },
}
_LEVEL = {-1: 100, 1: 200}


def _prism(
    side: int,
    notch: str,
    fill: str,
    footprint: tuple[int, ...],
    top: tuple[tuple[int, ...], ...],
    /,
) -> tuple[tuple[int, ...], ...]:
    """Outward faces of the prism over ``footprint`` (CCW from above, z = 0)
    towards z = ``side``.

    The face at z = 0 is ``top`` plus the notch fill (all CCW from above); the
    chord a-b keeps its hanging vertex on the side face through it.
    """
    lift = _LEVEL[side]
    plane = [*top, *_FILLS[notch][fill]]
    # Loops CCW from above point up: outward for the lower cell at z = 0 and
    # at its far side for the upper cell.
    far = tuple(lift + vertex for vertex in footprint)
    sides = []
    for first, second in zip(footprint, (*footprint[1:], footprint[0]), strict=True):
        if (first, second) == (_E, _A):
            continue
        inner = _CHORD[notch] if (first, second) == (_A, _B) else ()
        if (first, second) == (_A, _B) and _E in footprint:
            # One planar front face through e, a and b, bottom corner a kept.
            sides.append((_E, _A, *inner, _B, lift + _B, lift + _A, lift + _E))
            continue
        sides.append((first, *inner, second, lift + second, lift + first))
    if side < 0:
        return (*plane, far[::-1], *(loop[::-1] for loop in sides))
    return (*(loop[::-1] for loop in plane), far, *sides)


def _mesh_case(
    notch: str, lower_fill: str, upper_fill: str, extra_edge: bool, nested: bool, /
) -> tuple[np.ndarray, tuple[tuple[tuple[int, ...], ...], ...]]:
    plane = {**_PLANE, **_NOTCHES[notch]}
    face = (_A, *_NOTCH_PATH[notch], _B, _C, _D, _U)
    tops = ((_A, *_NOTCH_PATH[notch], _B, _C, _U), (_U, _C, _D)) if nested else (face,)
    lower_footprint = (_E, _A, _B, _C, _D, _U, _M) if extra_edge else (_A, _B, _C, _D, _U)
    upper_footprint = (_A, _B, _C, _D, _N, _M, _U) if extra_edge else (_A, _B, _C, _D, _U)
    lower_tops = (*tops, (_E, _A, _U, _M)) if extra_edge else tops
    upper_tops = (*tops, (_U, _D, _N, _M)) if extra_edge else tops
    coordinates = np.zeros((300, 3), dtype=np.float64)
    for vertex, (x, y) in plane.items():
        for offset, z in ((0, 0.0), (100, -1.0), (200, 1.0)):
            coordinates[offset + vertex] = (x, y, z)
    cells = []
    for side, footprint, top, fill in (
        (-1, lower_footprint, lower_tops, lower_fill),
        (1, upper_footprint, upper_tops, upper_fill),
    ):
        cells.append(_prism(side, notch, fill, footprint, top))
    used = sorted({vertex for cell in cells for loop in cell for vertex in loop})
    row = {vertex: index for index, vertex in enumerate(used)}
    return coordinates[used], tuple(
        tuple(tuple(row[vertex] for vertex in loop) for loop in cell) for cell in cells
    )


_MATRIX = [
    pytest.param(
        notch,
        lower,
        upper,
        extra_edge,
        nested,
        id=f"{notch}-{lower}-{upper}-edge{int(extra_edge)}-nested{int(nested)}",
    )
    for notch, fills in _FILLS.items()
    for lower, upper in product(fills, repeat=2)
    for extra_edge, nested in product((False, True), repeat=2)
]


@pytest.mark.parametrize(("notch", "lower", "upper", "extra_edge", "nested"), _MATRIX)
def test_notch_fills_certify_exactly_when_they_conform(
    notch: str, lower: str, upper: str, extra_edge: bool, nested: bool
) -> None:
    coordinates, cells = _mesh_case(notch, lower, upper, extra_edge, nested)
    mesh = CellMesh.from_polyhedra(
        coordinates,
        [[np.asarray(loop, dtype=np.int32) for loop in cell] for cell in cells],
    )
    geometry = CellGeometrySpec.affine(mesh)
    validity = certify_cell_geometry_validity(geometry, mesh=mesh)
    assert validity.all_certified
    certificate = certify_global_embedding(mesh, geometry, validity)
    if lower == upper:
        assert certificate.status == "certified", certificate.findings
    else:
        assert certificate.status == "violated", certificate.findings
        assert any(finding.check == "cell_contact" for finding in certificate.findings)
