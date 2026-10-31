from fractions import Fraction
from itertools import product

import numpy as np
import pytest

from phydrax.discretization._exact_algebraic_geometry import RadicalElement
from phydrax.geometry._exact_polyhedral_geometry import (
    determinant3,
    outward_tetrahedron_enclosures,
    Point3,
    point_in_triangle3,
    point_on_segment3,
    tetrahedron_contact_points,
)


def _point(x: int | Fraction, y: int | Fraction, z: int | Fraction) -> Point3:
    return Fraction(x), Fraction(y), Fraction(z)


_ORIGIN = _point(0, 0, 0)
_X = _point(1, 0, 0)
_Y = _point(0, 1, 0)
_Z = _point(0, 0, 1)
_BASE = (_ORIGIN, _X, _Y, _Z)
_QUARTER = Fraction(1, 4)
_HALF = Fraction(1, 2)
_THREE_QUARTERS = Fraction(3, 4)
_SMALL_FACE = (
    _point(_QUARTER, _QUARTER, 0),
    _point(_THREE_QUARTERS, _QUARTER, 0),
    _point(_QUARTER, _THREE_QUARTERS, 0),
)
_SMALL_EDGE = (_point(_QUARTER, _QUARTER, 0), _point(_HALF, _QUARTER, 0))


def _affine_image(points: tuple[Point3, ...], /) -> tuple[Point3, ...]:
    # This invertible exact affine map preserves every closed intersection.
    root_two = RadicalElement.sqrt(Fraction(2))
    return tuple(
        (
            x + y / 3 + Fraction(1, 7),
            root_two * y + z / 5 - Fraction(2, 7),
            z + Fraction(3, 7),
        )
        for x, y, z in points
    )


@pytest.mark.parametrize("algebraic", (False, True), ids=("rational", "algebraic-affine"))
@pytest.mark.parametrize(
    ("first", "second", "expected"),
    (
        (
            _BASE,
            (_ORIGIN, _point(-1, 0, 0), _point(0, -1, 0), _point(0, 0, -1)),
            (_ORIGIN,),
        ),
        (_BASE, (_ORIGIN, _X, _point(0, -1, 0), _point(0, 0, -1)), (_ORIGIN, _X)),
        (_BASE, (_ORIGIN, _X, _Y, _point(0, 0, -1)), (_ORIGIN, _X, _Y)),
        (_BASE, (_Z, _Y, _X, _ORIGIN), _BASE),
        (
            _BASE,
            tuple(
                (x + _THREE_QUARTERS, y + _THREE_QUARTERS, z + _THREE_QUARTERS)
                for x, y, z in _BASE
            ),
            (),
        ),
        (
            _BASE,
            (_ORIGIN, _X, _Y, _point(_QUARTER, _QUARTER, _QUARTER)),
            (_ORIGIN, _X, _Y, _point(_QUARTER, _QUARTER, _QUARTER)),
        ),
        (
            _BASE,
            (_ORIGIN, _X, _point(0, _HALF, 0), _point(0, 0, _HALF)),
            (_ORIGIN, _X, _point(0, _HALF, 0), _point(0, 0, _HALF)),
        ),
        (
            _BASE,
            (_ORIGIN, _point(_HALF, 0, 0), _point(0, _HALF, 0), _point(0, 0, _HALF)),
            (_ORIGIN, _point(_HALF, 0, 0), _point(0, _HALF, 0), _point(0, 0, _HALF)),
        ),
        (
            _BASE,
            tuple((x + _QUARTER, y, z) for x, y, z in _BASE),
            (
                _point(_QUARTER, 0, 0),
                _X,
                _point(_QUARTER, _THREE_QUARTERS, 0),
                _point(_QUARTER, 0, _THREE_QUARTERS),
            ),
        ),
        (_BASE, (*_SMALL_FACE, _point(_QUARTER, _QUARTER, -1)), _SMALL_FACE),
        (
            _BASE,
            (
                *_SMALL_EDGE,
                _point(_QUARTER, _QUARTER, -1),
                _point(_QUARTER, -_QUARTER, -1),
            ),
            _SMALL_EDGE,
        ),
        (
            _BASE,
            (
                _point(_QUARTER, _QUARTER, 0),
                _point(_QUARTER, _QUARTER, -1),
                _point(_THREE_QUARTERS, _QUARTER, -1),
                _point(_QUARTER, _THREE_QUARTERS, -1),
            ),
            (_point(_QUARTER, _QUARTER, 0),),
        ),
        (
            (_point(-1, 0, 0), _X, _Z, _point(0, 1, 1)),
            (_point(0, -1, 0), _Y, _point(0, 0, -1), _point(1, 0, -1)),
            (_ORIGIN,),
        ),
    ),
    ids=(
        "shared-vertex",
        "shared-edge",
        "shared-face",
        "identical",
        "disjoint-overlapping-boxes",
        "contained-shared-face",
        "contained-shared-edge",
        "contained-shared-vertex",
        "volume-overlap",
        "coplanar-face-overlap",
        "edge-on-face",
        "vertex-on-face",
        "crossing-support-edges",
    ),
)
def test_exact_tetrahedron_contacts_preserve_closed_intersection(
    first: tuple[Point3, ...],
    second: tuple[Point3, ...],
    expected: tuple[Point3, ...],
    algebraic: bool,
) -> None:
    if algebraic:
        first, second, expected = map(_affine_image, (first, second, expected))
    assert set(tetrahedron_contact_points(first, second)) == set(expected)
    assert set(tetrahedron_contact_points(second, first)) == set(expected)


def test_identical_degenerate_tetrahedra_do_not_bypass_admission() -> None:
    flat = (_ORIGIN, _X, _Y, _point(1, 1, 0))
    with pytest.raises(ValueError, match="nondegenerate"):
        tetrahedron_contact_points(flat, flat)


def _in_common_hull(point: Point3, common: tuple[Point3, ...], /) -> bool:
    match len(common):
        case 1:
            return point == common[0]
        case 2:
            return point_on_segment3(point, common[0], common[1])
        case 3:
            return point_in_triangle3(point, (common[0], common[1], common[2]))
        case _:
            return False


def _assert_certified_contacts_in_common_hull(
    tetrahedra: tuple[tuple[Point3, ...], ...], pairs: tuple[tuple[int, int], ...], /
) -> np.ndarray:
    enclosures = outward_tetrahedron_enclosures(tetrahedra)
    first = np.asarray([pair[0] for pair in pairs], dtype=np.intp)
    second = np.asarray([pair[1] for pair in pairs], dtype=np.intp)
    certified = enclosures.common_hull_contacts(first, second)
    for (left, right), proven in zip(pairs, certified.tolist(), strict=True):
        if not proven:
            continue
        common = tuple(point for point in tetrahedra[left] if point in tetrahedra[right])
        contacts = tetrahedron_contact_points(tetrahedra[left], tetrahedra[right])
        assert all(_in_common_hull(point, common) for point in contacts)
    return certified


def test_binary64_filter_certifies_neighbors_only_inside_their_common_hull() -> None:
    apex = _point(_QUARTER, _QUARTER, -1)
    tetrahedra = (
        _BASE,
        (_ORIGIN, _X, _Y, apex),  # shares the bottom face, opposite side
        (_X, _point(2, 0, 0), _point(1, 1, 0), _point(1, 0, 1)),  # shares a corner
        (_point(3, 3, 3), _point(4, 3, 3), _point(3, 4, 3), _point(3, 3, 4)),
        (*_SMALL_FACE, _point(_QUARTER, _QUARTER, _QUARTER)),  # overlaps the base
        (_ORIGIN, _X, _Y, _point(1, 1, 0)),  # degenerate
    )
    certified = _assert_certified_contacts_in_common_hull(
        tetrahedra, ((0, 1), (0, 2), (0, 3), (0, 4), (0, 5))
    )
    assert certified.tolist() == [True, True, True, False, False]


def test_binary64_filter_is_sound_on_touching_lattice_tetrahedra() -> None:
    # Small lattice corners make coplanar, collinear and touching pairs common.
    lattice = tuple(_point(*corner) for corner in product(range(3), repeat=3))
    generator = np.random.default_rng(7)
    tetrahedra: list[tuple[Point3, ...]] = []
    while len(tetrahedra) < 120:
        corners = tuple(lattice[index] for index in generator.choice(27, 4, False))
        if determinant3(*corners):
            tetrahedra.append(corners)
    pairs = tuple(
        (int(left), int(right))
        for left, right in generator.integers(0, len(tetrahedra), (1500, 2))
        if left != right
    )
    certified = _assert_certified_contacts_in_common_hull(tuple(tetrahedra), pairs)
    assert certified.any() and not certified.all()
