from __future__ import annotations

from fractions import Fraction
from random import Random

import pytest

from phydrax.geometry._boundary_degree_embedding import (
    _exact_triangle_contact_finding,
    _integer_triangle_contact_finding,
    BoundaryFace,
    BoundaryTriangle,
)
from phydrax.geometry._exact_polyhedral_geometry import Point3, triangle_contact_points3
from phydrax.geometry._integer_triangle_contact import (
    contact_at_point,
    contact_on_segment,
    integer_triangle,
    integer_triangle_contacts,
)


def _point(x: int, y: int, z: int, denominator: int = 1) -> Point3:
    return Fraction(x, denominator), Fraction(y, denominator), Fraction(z, denominator)


def _check_pair(first: tuple[Point3, Point3, Point3], second: tuple[Point3, Point3, Point3], shared: int) -> None:
    ids_a = (0, 1, 2)
    ids_b = tuple(range(shared)) + tuple(range(3, 6 - shared))
    triangle_a = BoundaryTriangle(0, 10, ids_a, first)
    triangle_b = BoundaryTriangle(1, 11, (ids_b[0], ids_b[1], ids_b[2]), second)
    faces = (
        BoundaryFace(10, ids_a, first, (triangle_a,)),
        BoundaryFace(11, ids_b, second, (triangle_b,)),
    )
    images = dict(zip(ids_a, first, strict=True))
    images.update(zip(ids_b, second, strict=True))
    integer_a, integer_b = integer_triangle(first), integer_triangle(second)
    assert integer_a is not None and integer_b is not None
    actual = _integer_triangle_contact_finding(faces, triangle_a, triangle_b, images, integer_a, integer_b)
    expected = _exact_triangle_contact_finding(faces, triangle_a, triangle_b, images)
    assert actual == expected
    separated = any(
        max(point[axis] for point in first) < min(point[axis] for point in second)
        or max(point[axis] for point in second) < min(point[axis] for point in first)
        for axis in range(3)
    )
    degenerate, contacts, scale = integer_triangle_contacts(integer_a, integer_b)
    if separated or degenerate:
        return
    reference = set(triangle_contact_points3(first, second))
    actual_points = {
        tuple(Fraction(value, denominator * scale) for value in numerator)
        for numerator, denominator in contacts
    }
    assert actual_points == reference


def test_integer_contacts_match_fraction_reference_randomized() -> None:
    rng = Random(81919)
    for index in range(600):
        points = tuple(
            _point(
                rng.randint(-4, 4),
                rng.randint(-4, 4),
                0 if index % 2 else rng.randint(-4, 4),
                rng.randint(1, 4),
            )
            for _ in range(6)
        )
        first = (points[0], points[1], points[2])
        for shared in (0, 1, 2):
            second = tuple(first[:shared]) + points[3:6-shared]
            _check_pair(first, (second[0], second[1], second[2]), shared)


@pytest.mark.parametrize("second", [
    (_point(0, 0, 0), _point(2, 0, 0), _point(0, 2, 0)),
    (_point(1, 0, 0), _point(3, 0, 0), _point(1, 2, 0)),
    (_point(2, 0, 0), _point(3, 0, 0), _point(2, 1, 0)),
    (_point(3, 0, 0), _point(4, 0, 0), _point(3, 1, 0)),
    (_point(1, -1, -1), _point(1, 2, 1), _point(1, 2, -1)),
])
def test_integer_contacts_cover_coplanar_touching_and_crossing(second: tuple[Point3, Point3, Point3]) -> None:
    first = (_point(0, 0, 0), _point(2, 0, 0), _point(0, 2, 0))
    _check_pair(first, second, 0)


def test_homogeneous_contact_detects_bad_scale_and_displacement() -> None:
    point = ((2, 4, 0), 2)
    expected = _point(1, 2, 0)
    assert contact_at_point(point, 1, expected)
    assert not contact_at_point(point, 2, expected)
    assert contact_on_segment(point, 1, _point(0, 0, 0), _point(2, 4, 0))
    assert not contact_on_segment(((2, 5, 0), 2), 1, _point(0, 0, 0), _point(2, 4, 0))
