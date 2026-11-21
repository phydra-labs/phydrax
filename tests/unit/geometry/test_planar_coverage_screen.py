"""Screened source-facet candidates keep every positive-area overlap."""

from __future__ import annotations

from fractions import Fraction
from itertools import product

import pytest

from phydrax.geometry._planar_coverage import _overlap, OrientedFacet, SourceFacetIndex
from phydrax.geometry._planar_primitives import (
    area2,
    intersection,
    plane_key,
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
