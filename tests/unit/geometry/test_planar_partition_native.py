import random
from fractions import Fraction
from itertools import combinations

import numpy as np
import pytest

from phydrax._meshcore import planar_partition
from phydrax.geometry._planar_primitives import partition_verdict


type Point = tuple[Fraction, Fraction]
type Triangle = tuple[Point, Point, Point]


def _cross(a: Point, b: Point, c: Point) -> Fraction:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _area(points: tuple[Point, ...]) -> Fraction:
    return abs(
        sum(
            (
                a[0] * b[1] - a[1] * b[0]
                for a, b in zip(points, points[1:] + points[:1], strict=True)
            ),
            Fraction(0),
        )
    )


def _intersection(a: Triangle, b: Triangle) -> Fraction:
    sign = _cross(*b)
    if not sign:
        return Fraction(0)
    polygon = list(a)
    for start, end in zip(b, b[1:] + b[:1], strict=True):
        if not polygon:
            break
        output: list[Point] = []
        previous = polygon[-1]
        old = _cross(start, end, previous) * sign
        for point in polygon:
            value = _cross(start, end, point) * sign
            if (old >= 0) != (value >= 0):
                ratio = old / (old - value)
                output.append(
                    (
                        previous[0] + ratio * (point[0] - previous[0]),
                        previous[1] + ratio * (point[1] - previous[1]),
                    )
                )
            if value >= 0:
                output.append(point)
            previous, old = point, value
        polygon = output
    return _area(tuple(polygon))


def _reference(
    rows: tuple[Triangle, ...], declared: tuple[Triangle, ...]
) -> tuple[int, int, int]:
    if not declared or sum((_area(t) for t in rows), Fraction(0)) != sum(
        (_area(t) for t in declared), Fraction(0)
    ):
        return (1, -1, -1)
    for row, triangle in enumerate(rows):
        if sum(
            (_intersection(triangle, other) for other in declared), Fraction(0)
        ) != _area(triangle):
            return (2, row, -1)
    for first, second in combinations(range(len(rows)), 2):
        if _intersection(rows[first], rows[second]):
            return (3, first, second)
    return (0, -1, -1)


def test_generated_exact_partition_decisions() -> None:
    rng = random.Random(59483)

    def point() -> Point:
        return (
            Fraction(rng.randrange(-7, 8), rng.randrange(1, 5)),
            Fraction(rng.randrange(-7, 8), rng.randrange(1, 5)),
        )

    def triangle() -> Triangle:
        return (point(), point(), point())

    for iteration in range(600):
        rows = tuple(triangle() for _ in range(1 + iteration % 4))
        declared = rows if iteration % 2 else tuple(triangle() for _ in rows)
        assert partition_verdict(rows, declared) == _reference(rows, declared)


def test_ordered_decisions_and_large_exact_coordinates() -> None:
    z, t = Fraction(0), Fraction(2)
    a = ((z, z), (t, z), (z, t))
    b = ((t, t), (z, t), (t, z))
    overlap = ((z, z), (t, z), (t, t))
    assert partition_verdict((a, b), (a, b)) == (0, -1, -1)
    assert partition_verdict((a,), ()) == (1, -1, -1)
    assert partition_verdict((a, overlap), (a, b)) == (3, 0, 1)
    shifted = tuple((x + 3, y) for x, y in a)
    shifted_triangle = (shifted[0], shifted[1], shifted[2])
    assert partition_verdict((shifted_triangle,), (a,)) == (2, 0, -1)
    scale = Fraction(2**1100 + 1, 2**1101 + 1)

    def scaled(triangle: Triangle) -> Triangle:
        points = tuple((x * scale, y * scale) for x, y in triangle)
        return (points[0], points[1], points[2])

    assert partition_verdict((scaled(a), scaled(overlap)), (scaled(a), scaled(b))) == (
        3,
        0,
        1,
    )


def test_partition_contacts_and_finding_order() -> None:
    z, o, t, q = (Fraction(value) for value in (0, 1, 2, 4))
    large = ((z, z), (q, z), (z, q))
    nested = ((z, z), (o, z), (z, o))
    touching = ((q, z), (q, q), (z, q))
    collinear = ((z, z), (t, z), (q, z))
    choices = (large, nested, touching, collinear)
    for first in choices:
        for second in choices:
            rows = (first, second)
            assert partition_verdict(rows, rows) == _reference(rows, rows)


def test_native_partition_rejects_misaligned_arrays() -> None:
    with pytest.raises(ValueError, match="incompatible shapes"):
        planar_partition(
            1,
            0,
            np.empty((0,), dtype=np.uint32),
            np.zeros((2,), dtype=np.int64),
            np.empty((0,), dtype=np.int8),
            np.empty((0, 2), dtype=np.float64),
            np.empty((0, 2), dtype=np.float64),
        )
