"""Screened source-facet candidates keep every positive-area overlap."""

from __future__ import annotations

import random
from fractions import Fraction
from itertools import product
from math import gcd, lcm

import pytest

from phydrax.geometry._planar_coverage import _overlap, OrientedFacet, SourceFacetIndex
from phydrax.geometry._planar_primitives import (
    area2,
    intersection,
    overlap_area2,
    plane_key,
    plane_orientation,
    project_polygon,
    projection_axes,
)


type Point3 = tuple[Fraction, Fraction, Fraction]


def _point(x: float, y: float, z: float = 0.0) -> Point3:
    return Fraction(x), Fraction(y), Fraction(z)


def _sources() -> tuple[OrientedFacet, ...]:
    # A 4 x 4 grid of unit squares split into triangles on z = 0, plus a
    # tilted triangle on another plane.
    facets: list[OrientedFacet] = []
    for x, y in product(range(4), range(4)):
        a, b = _point(x, y), _point(x + 1, y)
        c, d = _point(x + 1, y + 1), _point(x, y + 1)
        facets.append(OrientedFacet(len(facets), (0, -1), (a, b, c)))
        facets.append(OrientedFacet(len(facets), (0, -1), (a, c, d)))
    facets.append(
        OrientedFacet(
            len(facets), (0, -1), (_point(0, 0, 1), _point(1, 0, 2), _point(0, 1, 1))
        )
    )
    return tuple(facets)


_QUERIES: dict[str, tuple[Point3, Point3, Point3]] = {
    "inside": (_point(0.25, 0.125), _point(0.75, 0.125), _point(0.75, 0.5)),
    "shared-vertex": (_point(1, 1), _point(1.5, 1.25), _point(1.25, 1.5)),
    "touching-edge": (_point(0, 0), _point(1, 0), _point(0.5, -1)),
    "collinear-edge": (_point(0.5, 0), _point(1.5, 0), _point(1, 1)),
    "touching-vertex": (_point(4, 4), _point(5, 4), _point(5, 5)),
    "large": (_point(-1, -1), _point(5, -1), _point(-1, 5)),
    "sliver": (_point(1, 1), _point(3, 3), _point(3, 3.0000001)),
    "disjoint": (_point(10, 10), _point(11, 10), _point(10, 11)),
    "other-plane": (_point(0, 0, 1), _point(1, 0, 2), _point(0, 1, 1)),
}


@pytest.mark.parametrize("name", tuple(_QUERIES))
def test_screened_candidates_keep_every_positive_overlap(name: str) -> None:
    sources = _sources()
    index = SourceFacetIndex(sources)
    query = OrientedFacet(-1, (0, -1), _QUERIES[name])
    candidates = index.candidates(query.vertices, 3)
    unscreened = tuple(
        row for row in range(len(sources)) if _overlap(sources[row], query, 3) > 0
    )

    assert tuple(row for row in candidates if _overlap(sources[row], query, 3) > 0) == (
        unscreened
    )
    assert list(candidates) == sorted(candidates)
    if name == "disjoint":
        assert candidates == ()


@pytest.mark.parametrize("name", tuple(_QUERIES))
def test_overlap_measures_equal_the_clipped_intersection(name: str) -> None:
    sources = _sources()
    query = OrientedFacet(-1, (0, -1), _QUERIES[name])
    for source in sources:
        for first, second in ((source, query), (query, source)):
            try:
                plane = plane_key(first.vertices)
                coplanar = plane == plane_key(second.vertices)
            except ValueError:
                coplanar = False
            clipped = (
                area2(
                    intersection(
                        project_polygon(first.vertices, projection_axes(plane)),
                        project_polygon(second.vertices, projection_axes(plane)),
                    )
                )
                / 2
                if coplanar
                else Fraction(0)
            )
            assert _overlap(first, second, 3) == clipped


def test_batch_candidates_equal_single_queries() -> None:
    index = SourceFacetIndex(_sources())
    queries = tuple(_QUERIES.values())
    assert index.batch_candidates(queries) == tuple(
        index.candidates(query, 3) for query in queries
    )


def _rational_plane(
    points: tuple[tuple[Fraction, ...], ...],
) -> tuple[tuple[int, ...], int]:
    # The rational-arithmetic plane and orientation the integer forms replace.
    origin, first, second = points[0], points[1], points[2]
    u = tuple(first[axis] - origin[axis] for axis in range(3))
    v = tuple(second[axis] - origin[axis] for axis in range(3))
    normal = (
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    )
    if not any(normal):
        raise ValueError("collinear")
    row = (*normal, -sum(normal[axis] * origin[axis] for axis in range(3)))
    denominator = lcm(*(value.denominator for value in row))
    integers = tuple(int(value * denominator) for value in row)
    divisor = gcd(*integers)
    key = tuple(value // divisor for value in integers)
    if next(value for value in key if value) < 0:
        key = tuple(-value for value in key)
    return key, 1 if sum(normal[i] * key[i] for i in range(3)) > 0 else -1


def test_integer_planes_and_lattice_overlaps_equal_rational_ones() -> None:
    generator = random.Random(7)

    def value() -> Fraction:
        return Fraction(generator.randint(-9, 9), generator.choice((1, 2, 3, 7)))

    for _ in range(2000):
        points = tuple((value(), value(), value()) for _ in range(3))
        if generator.random() < 0.2:
            points = (
                points[0],
                points[1],
                tuple(2 * b - a for a, b in zip(points[0], points[1], strict=True)),
            )
        try:
            expected = _rational_plane(points)
        except ValueError:
            with pytest.raises(ValueError):
                plane_key(points)
            with pytest.raises(ValueError):
                plane_orientation(points)
        else:
            assert (plane_key(points), plane_orientation(points)) == expected
        first = tuple((value(), value()) for _ in range(3))
        second = tuple((value(), value()) for _ in range(3))
        if generator.random() < 0.3:
            second = (first[0], first[1], second[2])
        assert overlap_area2(first, second) == area2(intersection(first, second))
        assert overlap_area2(second, first) == area2(intersection(first, second))
