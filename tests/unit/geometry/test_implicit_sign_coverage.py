#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Sign-component region coverage of targets over crease-bearing implicit sources.

The overlapping two-sphere union has a concave seam circle where its field is
not smooth, so a target face there carries no root chart. The region argument
needs only the discovery's enclosure-complete cover. Its claims are checked
against exact sphere geometry: boundary distances, enclosed volume between the
exact parallel bodies, and the exact location of every disagreement point.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from fractions import Fraction

import numpy as np
import pytest
from scipy.spatial import Delaunay

import phydrax as phx
from phydrax._bvh import prepare_bvh
from phydrax.geometry._contracts import CompiledGeometry
from phydrax.geometry.implicit import (
    AdaptiveImplicitSurface,
    AdaptiveImplicitSurfacePolicy,
    discover_adaptive_implicit_surface,
)
from phydrax.geometry.implicit._source_charts import (
    _box_witness_distance_upper,
    _SIGN_COVER_WHOLE_PIECE,
    _split_triangle_pieces,
    _triangle_piece_boxes,
    certify_implicit_sign_coverage,
)


_TOLERANCE = 0.04
_DOMAIN = np.asarray(((-0.8, -0.6, -0.6), (0.8, 0.6, 0.6)), dtype=np.float64)
_RADIUS = 0.3
_CENTERS = np.asarray(((-0.2, 0.0, 0.0), (0.2, 0.0, 0.0)), dtype=np.float64)


def _discover(geometry: CompiledGeometry) -> AdaptiveImplicitSurface:
    diagonal = float(np.linalg.norm(_DOMAIN[1] - _DOMAIN[0]))
    return discover_adaptive_implicit_surface(
        geometry,
        domain=_DOMAIN,
        policy=AdaptiveImplicitSurfacePolicy(
            relative_cell_diameter=0.5 * _TOLERANCE / diagonal,
            root_tolerance=1.0e-10,
            allow_nonsmooth_field=True,
        ),
        source_id="sign-coverage",
    )


def _tetrahedra(
    surface: AdaptiveImplicitSurface,
    inside: Callable[[np.ndarray, float], np.ndarray],
    longest_edge: float = 0.06,
) -> tuple[np.ndarray, Delaunay, np.ndarray]:
    """A Delaunay fill of the discovered roots and its kept tetrahedra.

    ``inside(points, margin)`` marks points deeper than ``margin`` in the
    material; tetrahedra with an inside centroid and no edge longer than
    ``longest_edge`` (by default twice the interior grid spacing) are kept.
    """
    roots = np.asarray(surface.scaffold.root_points, dtype=np.float64)
    axis = np.arange(-0.5, 0.5001, 0.03)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(
        (-1, 3)
    )
    points = np.concatenate((roots, grid[inside(grid, 0.02)]))
    triangulation = Delaunay(points)
    corners = points[triangulation.simplices]
    edges = corners[:, (0, 0, 0, 1, 1, 2)] - corners[:, (1, 2, 3, 2, 3, 3)]
    kept = inside(np.mean(corners, axis=1), 0.0) & (
        np.max(np.linalg.norm(edges, axis=2), axis=1) <= longest_edge
    )
    return points, triangulation, kept


def _boundary(tetrahedra: np.ndarray) -> np.ndarray:
    faces = np.sort(
        tetrahedra[:, ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))].reshape((-1, 3)),
        axis=1,
    )
    unique, counts = np.unique(faces, axis=0, return_counts=True)
    return unique[counts == 1]


def _piece_boxes(triangles: np.ndarray, width: float) -> tuple[np.ndarray, np.ndarray]:
    """Outward boxes of exact dyadic facet pieces no wider than ``width``."""
    rows = np.arange(triangles.shape[0])
    parameters = np.repeat(_SIGN_COVER_WHOLE_PIECE[None], triangles.shape[0], axis=0)
    depths = np.zeros(triangles.shape[0], dtype=np.int64)
    while True:
        lower, upper = _triangle_piece_boxes(triangles, rows, parameters)
        wide = np.linalg.norm(upper - lower, axis=1) > width
        if not np.any(wide):
            return lower, upper
        split = _split_triangle_pieces(rows[wide], parameters[wide], depths[wide])
        rows, parameters, depths = (
            np.concatenate((kept[~wide], halves))
            for kept, halves in zip((rows, parameters, depths), split, strict=True)
        )


def _mesh_to_source(surface: AdaptiveImplicitSurface, triangles: np.ndarray) -> float:
    """Certified bound: every facet piece box to the farthest point of a root box."""
    lower, upper = _piece_boxes(triangles, _TOLERANCE * 0.25)
    witnesses = np.asarray(surface.boundary_witnesses, dtype=np.float64).reshape((-1, 3))
    errors = np.asarray(surface.boundary_errors, dtype=np.float64).reshape((-1,))
    witness_lower = np.nextafter(witnesses - errors[:, None], -np.inf)
    witness_upper = np.nextafter(witnesses + errors[:, None], np.inf)
    return float(
        np.max(
            _box_witness_distance_upper(
                prepare_bvh(witness_lower, witness_upper, dtype=np.float64),
                witness_lower,
                witness_upper,
                lower,
                upper,
                _TOLERANCE * 2.0**-6,
            )
        )
    )


def _union_distance(points: np.ndarray) -> np.ndarray:
    """Exact distance to the boundary of the two overlapping balls.

    The boundary is each sphere's cap outside the other ball; a cap's closest
    point is the radial projection when it lies on the cap, else a point of
    the seam circle (plane x = 0, radius sqrt(0.3^2 - 0.2^2)).
    """
    seam_radius = math.sqrt(_RADIUS**2 - 0.2**2)
    candidates = [
        np.hypot(points[:, 0], np.hypot(points[:, 1], points[:, 2]) - seam_radius)
    ]
    for center, other in ((_CENTERS[0], _CENTERS[1]), (_CENTERS[1], _CENTERS[0])):
        offset = points - center
        length = np.linalg.norm(offset, axis=1)
        projection = center + _RADIUS * offset / length[:, None]
        on_cap = np.linalg.norm(projection - other, axis=1) >= _RADIUS
        candidates.append(np.where(on_cap, np.abs(length - _RADIUS), np.inf))
    return np.min(np.stack(candidates), axis=0)


def _union_inside(points: np.ndarray, margin: float) -> np.ndarray:
    return np.min(
        np.stack([np.linalg.norm(points - center, axis=1) for center in _CENTERS]),
        axis=0,
    ) < (_RADIUS - margin)


def _union_volume(radius: float) -> float:
    """Exact volume of two balls of one radius whose centers are 0.4 apart."""
    separation = 0.4
    lens = (
        math.pi * (4.0 * radius + separation) * (2.0 * radius - separation) ** 2 / 12.0
        if separation < 2.0 * radius
        else 0.0
    )
    return 2.0 * 4.0 / 3.0 * math.pi * radius**3 - lens


@pytest.fixture(scope="module")
def overlapping_union() -> AdaptiveImplicitSurface:
    geometry = (
        phx.geometry.Sphere(tuple(float(value) for value in _CENTERS[0]), _RADIUS)
        | phx.geometry.Sphere(tuple(float(value) for value in _CENTERS[1]), _RADIUS)
    ).compile()
    return _discover(geometry)


def test_overlapping_union_regions_agree_beyond_the_certified_band(
    overlapping_union: AdaptiveImplicitSurface,
) -> None:
    surface = overlapping_union
    cover = surface.cover
    assert cover is not None and cover.complete and cover.certified
    points, triangulation, kept = _tetrahedra(surface, _union_inside)
    tetrahedra = triangulation.simplices[kept]
    faces = _boundary(tetrahedra)
    triangles = points[faces]
    epsilon = _mesh_to_source(surface, triangles)
    assert epsilon < _TOLERANCE
    result = certify_implicit_sign_coverage(
        surface,
        triangles,
        points[tetrahedra],
        mesh_to_source=epsilon,
        tolerance=_TOLERANCE,
    )
    assert result.status == "certified", result.detail
    delta = result.distance_upper
    assert epsilon <= delta <= _TOLERANCE
    # Exact distance of every target boundary point used: corners and centroids.
    assert np.all(_union_distance(points[np.unique(faces)]) <= delta)
    assert np.all(_union_distance(np.mean(triangles, axis=1)) <= delta)
    # The enclosed volume lies between the exact inner and outer parallel
    # bodies of the union (unions of the balls of radius 0.3 -+ delta).
    corners = points[tetrahedra]
    volume = float(np.sum(np.abs(np.linalg.det(corners[:, 1:] - corners[:, :1])))) / 6.0
    assert _union_volume(_RADIUS - delta) <= volume <= _union_volume(_RADIUS + delta)
    # Every point where the target and the exact source regions disagree is
    # within delta of the exact union surface.
    samples = np.random.default_rng(7).uniform(
        (-0.55, -0.35, -0.35), (0.55, 0.35, 0.35), (200_000, 3)
    )
    located = triangulation.find_simplex(samples)
    in_target = (located >= 0) & kept[np.maximum(located, 0)]
    differ = in_target != _union_inside(samples, 0.0)
    assert np.any(differ)
    assert np.all(_union_distance(samples[differ]) <= delta)


def test_missing_component_is_a_violated_region() -> None:
    first = (np.asarray((-0.37, 0.013, -0.017)), 0.19)
    second = (np.asarray((0.36, -0.019, 0.021)), 0.14)
    geometry = (
        phx.geometry.Sphere(tuple(float(value) for value in first[0]), first[1])
        | phx.geometry.Sphere(tuple(float(value) for value in second[0]), second[1])
    ).compile()
    surface = _discover(geometry)

    def left_ball(points: np.ndarray, margin: float) -> np.ndarray:
        return np.linalg.norm(points - first[0], axis=1) < first[1] - margin

    # The left ball is convex and 0.4 from the right one: longer edges than the
    # union's leave no hole between the root shell and the interior grid.
    points, triangulation, kept = _tetrahedra(surface, left_ball, 0.1)
    tetrahedra = triangulation.simplices[kept]
    faces = _boundary(tetrahedra)
    # Every target boundary corner is a discovered root (within the 1e-10
    # root tolerance) of the left sphere.
    root_count = np.asarray(surface.scaffold.root_points).shape[0]
    assert np.all(faces < root_count)
    assert np.all(
        np.abs(np.linalg.norm(points[faces] - first[0], axis=2) - first[1]) <= 1e-9
    )
    triangles = points[faces]
    epsilon = _mesh_to_source(surface, triangles)
    assert epsilon < _TOLERANCE
    result = certify_implicit_sign_coverage(
        surface,
        triangles,
        points[tetrahedra],
        mesh_to_source=epsilon,
        tolerance=_TOLERANCE,
    )
    assert result.status == "violated"
    assert "misses a zero-free leaf inside the source region" in result.detail


def test_tight_target_bound_resolves_the_band_at_the_source_resolution(
    overlapping_union: AdaptiveImplicitSurface,
) -> None:
    # A valid but nearly exhausted target bound leaves almost no tolerance; the
    # band is then resolved at the source's target-level cell diagonal, and its
    # reported distance (beyond the tolerance) is still exact-checked.
    surface = overlapping_union
    points, triangulation, kept = _tetrahedra(surface, _union_inside)
    tetrahedra = triangulation.simplices[kept]
    triangles = points[_boundary(tetrahedra)]
    epsilon = max(_mesh_to_source(surface, triangles), _TOLERANCE * (1.0 - 2.0**-20))
    assert epsilon < _TOLERANCE
    result = certify_implicit_sign_coverage(
        surface,
        triangles,
        points[tetrahedra],
        mesh_to_source=epsilon,
        tolerance=_TOLERANCE,
    )
    assert result.status == "certified", result.detail
    assert math.isfinite(result.band_extent)
    assert result.distance_upper == math.nextafter(epsilon + result.band_extent, math.inf)
    assert result.distance_upper > _TOLERANCE
    samples = np.random.default_rng(11).uniform(
        (-0.55, -0.35, -0.35), (0.55, 0.35, 0.35), (100_000, 3)
    )
    located = triangulation.find_simplex(samples)
    in_target = (located >= 0) & kept[np.maximum(located, 0)]
    differ = in_target != _union_inside(samples, 0.0)
    assert np.any(differ)
    assert np.all(_union_distance(samples[differ]) <= result.distance_upper)


def test_facet_pieces_halve_exactly_and_enclose_their_exact_corners() -> None:
    triangles = np.asarray(
        [[[0.1, 0.2, 0.3], [0.7, -0.1, 0.35], [0.2, 0.9, -0.4]]], dtype=np.float64
    )
    rows = np.zeros(1, dtype=np.int64)
    parameters = _SIGN_COVER_WHOLE_PIECE[None].copy()
    depths = np.zeros(1, dtype=np.int64)
    for _ in range(3):
        rows, parameters, depths = _split_triangle_pieces(rows, parameters, depths)
    assert rows.shape == (64,) and np.all(depths == 3)
    # The pieces tile the parameter triangle exactly: their exact signed
    # areas are equal, positive, and sum to the whole.
    areas = [
        (Fraction(p[1, 0]) - Fraction(p[0, 0])) * (Fraction(p[2, 1]) - Fraction(p[0, 1]))
        - (Fraction(p[2, 0]) - Fraction(p[0, 0]))
        * (Fraction(p[1, 1]) - Fraction(p[0, 1]))
        for p in parameters
    ]
    assert sum(abs(area) for area in areas) == 1 and len(set(map(abs, areas))) == 1
    lower, upper = _triangle_piece_boxes(triangles, rows, parameters)
    a, b, c = ([Fraction(float(v)) for v in corner] for corner in triangles[0])
    for piece, low, high in zip(parameters, lower, upper, strict=True):
        for s, t in piece:
            exact = [
                a[axis]
                + Fraction(float(s)) * (b[axis] - a[axis])
                + Fraction(float(t)) * (c[axis] - a[axis])
                for axis in range(3)
            ]
            assert all(
                Fraction(float(low[axis])) <= exact[axis] <= Fraction(float(high[axis]))
                for axis in range(3)
            )
