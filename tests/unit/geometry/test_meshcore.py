#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


import gc
import importlib.metadata
import threading
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from typing import Any

import numpy as np
import pytest
from jax.typing import DTypeLike
from scipy.spatial import ConvexHull

from phydrax._meshcore import (
    delaunay_2d,
    delaunay_3d,
    IncrementalDelaunay3D,
    IntersectionClass,
    meshcore_available,
    meshcore_identity,
    MeshcoreError,
    MeshcoreStatus,
    NativeExecutionBudget,
    NativeResourceFailure,
    PlanarExecutionEvidence,
    PlcRecoveryFailure,
    polygon_intersection_moments,
    polygon_intersection_simplices,
    polyhedron_clip_moments,
    recover_plc_3d,
    segment_triangle_intersections,
    TetMesh3D,
    tetrahedron_intersection_moments,
    tetrahedron_intersection_simplices,
    triangle_intersection_classes,
    triangle_intersections,
    TRIANGULATION_3D_STATISTICS,
)
from phydrax.geometry import (
    ConstrainedDelaunayTriangulation,
    DelaunayTriangulation,
    incircle,
    insphere,
    orient2d,
    orient3d,
    PowerDiagram,
    PredicateMode,
    VoronoiDiagram,
)


pytestmark = [
    pytest.mark.meshcore,
    pytest.mark.skipif(
        not meshcore_available(),
        reason="native phydrax-meshcore library is not available",
    ),
]

EXACT = PredicateMode.EXACT
UNIT_TET = np.asarray(
    ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
)


def _signed_areas(corners: Any) -> Any:
    first = corners[:, 1] - corners[:, 0]
    second = corners[:, 2] - corners[:, 0]
    return 0.5 * (first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0])


def _tet_moments(vertices: Any) -> Any:
    volume = abs(np.linalg.det(vertices[1:] - vertices[0])) / 6.0
    return volume, volume * np.mean(vertices, axis=0)


def test_meshcore_scenario_1() -> None:
    name, version, digest = meshcore_identity().split(" ")
    assert name == "phydrax-meshcore"
    assert version == importlib.metadata.version("phydrax")
    assert len(digest) == 64 and all(char in "0123456789abcdef" for char in digest)
    shifted_overlap = np.asarray(
        ((0.25, 0.25, 0.25), (0.5, 0.25, 0.25), (0.25, 0.5, 0.25), (0.25, 0.25, 0.5))
    )
    shared_face = np.asarray(
        ((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, -1.0))
    )
    first = np.stack([UNIT_TET, UNIT_TET, UNIT_TET, UNIT_TET, UNIT_TET[[0, 2, 1, 3]]])
    second = np.stack(
        [UNIT_TET, 0.5 * UNIT_TET, UNIT_TET + 0.25, shared_face, UNIT_TET + 2.0]
    )
    volume, moment, status = tetrahedron_intersection_moments(first, second)

    np.testing.assert_array_equal(status, MeshcoreStatus.OK)
    expected = [
        _tet_moments(UNIT_TET),
        _tet_moments(0.5 * UNIT_TET),
        _tet_moments(shifted_overlap),
    ]
    for index, (expected_volume, expected_moment) in enumerate(expected):
        assert volume[index] == pytest.approx(expected_volume, rel=1e-14, abs=1e-16)
        np.testing.assert_allclose(moment[index], expected_moment, rtol=1e-13, atol=1e-16)
    # Coincident face contact and disjoint cells have exactly zero measure.
    assert volume[3] == 0.0 and volume[4] == 0.0
    np.testing.assert_array_equal(moment[3:], 0.0)
    flat = UNIT_TET.copy()
    flat[3] = (0.25, 0.25, 0.0)
    volume, _, status = tetrahedron_intersection_moments(flat[None], UNIT_TET[None])
    assert status[0] == MeshcoreStatus.DEGENERATE_INPUT
    assert volume[0] == 0.0
    square = np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))
    shifted = square + 0.5
    collinear = np.asarray(((0.0, 0.0), (0.5, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))
    nonconvex = np.asarray(((0.0, 0.0), (2.0, 0.0), (1.0, 0.5), (2.0, 2.0), (0.0, 2.0)))
    first = np.zeros((4, 5, 2))
    second = np.zeros((4, 5, 2))
    first[0, :4], second[0, :4] = square, shifted
    first[1, :4], second[1, :4] = square[::-1], shifted
    first[2, :5], second[2, :4] = collinear, square + np.asarray((1.0, 0.0))
    first[3, :5], second[3, :4] = nonconvex, square
    area, moment, status = polygon_intersection_moments(
        first, np.asarray((4, 4, 5, 5)), second, np.asarray((4, 4, 4, 4))
    )

    np.testing.assert_array_equal(
        status,
        (
            MeshcoreStatus.OK,
            MeshcoreStatus.OK,
            MeshcoreStatus.OK,
            MeshcoreStatus.INVALID_INPUT,
        ),
    )
    np.testing.assert_allclose(area[:2], 0.25, rtol=1e-15)
    np.testing.assert_allclose(moment[:2], 0.25 * 0.75, rtol=1e-15)
    assert area[2] == 0.0


def test_meshcore_scenario_2() -> None:
    corner = np.asarray(
        ((0.5, 0.0, 0.0), (1.0, 0.0, 0.0), (0.5, 0.5, 0.0), (0.5, 0.0, 0.5))
    )
    full_volume, full_moment = _tet_moments(UNIT_TET)
    corner_volume, corner_moment = _tet_moments(corner)
    normals = np.asarray(((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)))[:, None, :]
    offsets = np.asarray((0.5, 0.0, -1.0))[:, None]
    volume, moment, status = polyhedron_clip_moments(
        normals, offsets, np.ones(3, dtype=np.int32), np.stack([UNIT_TET] * 3)
    )

    np.testing.assert_array_equal(status, MeshcoreStatus.OK)
    assert volume[0] == pytest.approx(full_volume - corner_volume, rel=1e-14)
    np.testing.assert_allclose(moment[0], full_moment - corner_moment, rtol=1e-13)
    # A halfspace through a face keeps the whole cell; a separating one keeps nothing.
    assert volume[1] == pytest.approx(full_volume, rel=1e-15)
    assert volume[2] == 0.0
    rng = np.random.default_rng(4)
    first = rng.random((40, 4, 3))
    second = 0.3 + rng.random((40, 4, 3))
    simplices, counts, volume, moment, status = tetrahedron_intersection_simplices(
        first, second
    )
    reference_volume, reference_moment, _ = tetrahedron_intersection_moments(
        first, second
    )
    np.testing.assert_array_equal(status, MeshcoreStatus.OK)
    np.testing.assert_array_equal(volume, reference_volume)
    np.testing.assert_array_equal(moment, reference_moment)
    present = np.arange(simplices.shape[1])[None, :] < counts[:, None]
    edges = simplices[..., 1:, :] - simplices[..., :1, :]
    signed = np.sum(edges[..., 0, :] * np.cross(edges[..., 1, :], edges[..., 2, :]), -1)
    measures = np.where(present, signed / 6.0, 0.0)
    np.testing.assert_allclose(np.sum(measures, axis=1), volume, atol=1e-15)
    np.testing.assert_allclose(
        np.sum(measures[..., None] * np.mean(simplices, axis=2), axis=1),
        moment,
        atol=1e-15,
    )
    assert np.all(counts[volume == 0.0] == 0)

    square = np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))
    triangles, triangle_counts, area, _, polygon_status = polygon_intersection_simplices(
        square[None], (4,), (square + 0.5)[None], (4,)
    )
    assert polygon_status[0] == MeshcoreStatus.OK and triangle_counts[0] == 4
    corner_edges = triangles[0, :4, 1:] - triangles[0, :4, :1]
    corner_areas = 0.5 * (
        corner_edges[:, 0, 0] * corner_edges[:, 1, 1]
        - corner_edges[:, 0, 1] * corner_edges[:, 1, 0]
    )
    assert np.all(corner_areas > 0.0)
    assert np.sum(corner_areas) == pytest.approx(area[0], rel=1e-15)
    *_, refused = polygon_intersection_simplices(
        square[None], (4,), (square + 0.5)[None], (4,), simplex_capacity=3
    )
    assert refused[0] == MeshcoreStatus.CAPACITY_EXCEEDED
    rng = np.random.default_rng(0)
    random_points = rng.random((200, 2))
    lattice = np.stack(np.meshgrid(np.arange(6.0), np.arange(6.0)), axis=-1).reshape(
        (-1, 2)
    )
    for points in (random_points, lattice):
        triangulation = DelaunayTriangulation(points)
        _assert_empty_circles(points, triangulation.simplices)
        corners = points[triangulation.simplices]
        area = np.sum(_signed_areas(corners))
        assert area == pytest.approx(ConvexHull(points).volume, rel=1e-12)
        assert triangulation.evidence.predicate_mode is PredicateMode.EXACT
    repeated = DelaunayTriangulation(lattice)
    np.testing.assert_array_equal(
        repeated.simplices, DelaunayTriangulation(lattice).simplices
    )
    points = np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)))
    triangulation = DelaunayTriangulation(points)
    np.testing.assert_array_equal(triangulation.vertex_map, (0, 1, 2, 1, 4))
    assert triangulation.evidence.duplicate_count == 1
    assert 3 not in triangulation.simplices
    with pytest.raises(ValueError, match="span"):
        DelaunayTriangulation(np.asarray(((0.0, 0.0), (1.0, 1.0), (2.0, 2.0))))
    rng = np.random.default_rng(1)
    random_points = rng.random((60, 3))
    lattice = np.stack(np.meshgrid(*(np.arange(4.0),) * 3), axis=-1).reshape((-1, 3))
    for points in (random_points, lattice):
        tetrahedra = DelaunayTriangulation(points).simplices
        a, b, c, d = (points[tetrahedra[:, index]] for index in range(4))
        assert np.all(orient3d(a, b, c, d, mode=EXACT).signs == 1)
        for query in points:
            signs = insphere(
                a, b, c, d, np.broadcast_to(query, a.shape), mode=EXACT
            ).signs
            assert np.all(signs <= 0)
        volume = np.sum(np.linalg.det(np.stack((b - a, c - a, d - a), axis=1))) / 6.0
        assert volume == pytest.approx(ConvexHull(points).volume, rel=1e-12)


def _assert_empty_circles(points: Any, triangles: Any) -> None:
    a, b, c = (points[triangles[:, index]] for index in range(3))
    assert np.all(orient2d(a, b, c, mode=EXACT).signs == 1)
    for query in points:
        signs = incircle(a, b, c, np.broadcast_to(query, a.shape), mode=EXACT).signs
        assert np.all(signs <= 0)


def _segment_coverage(triangulation: Any, segments: Any) -> None:
    points = triangulation.points
    triangles = triangulation.triangles
    for segment_id, (start, stop) in enumerate(segments):
        cells, corners = np.nonzero(triangulation.segment_ids == segment_id)
        edges = np.stack(
            (triangles[cells, (corners + 1) % 3], triangles[cells, (corners + 2) % 3]),
            axis=1,
        )
        edges = np.unique(np.sort(edges, axis=1), axis=0)
        a, b = points[start], points[stop]
        ends = points[edges.reshape((-1,))]
        on_line = orient2d(a, b, ends, mode=EXACT).signs
        assert np.all(on_line == 0)
        length = np.sum(np.linalg.norm(points[edges[:, 1]] - points[edges[:, 0]], axis=1))
        assert length == pytest.approx(np.linalg.norm(b - a), rel=1e-12)


def test_meshcore_scenario_3() -> None:
    l_shape = np.asarray(
        ((0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (1.0, 1.0), (1.0, 2.0), (0.0, 2.0))
    )
    boundary = np.asarray([(index, (index + 1) % 6) for index in range(6)])
    segments = np.concatenate((boundary, np.asarray(((0, 3),))))
    triangulation = ConstrainedDelaunayTriangulation(
        l_shape, segments, min_angle=25.0, max_area=0.05, max_steiner=5000
    )

    assert triangulation.evidence.status == "ok"
    assert triangulation.evidence.minimum_angle_degrees >= 25.0
    corners = triangulation.points[triangulation.triangles]
    areas = _signed_areas(corners)
    assert np.all(areas > 0.0)
    assert np.max(areas) <= 0.05
    assert np.sum(areas) == pytest.approx(3.0, rel=1e-12)
    np.testing.assert_array_equal(triangulation.points[:6], l_shape)
    _segment_coverage(triangulation, segments)
    outer = np.asarray(((0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)))
    inner = np.asarray(((1.0, 1.0), (3.0, 1.0), (3.0, 3.0), (1.0, 3.0)))
    points = np.concatenate((outer, inner))
    segments = np.asarray(
        [(index, (index + 1) % 4) for index in range(4)]
        + [(4 + index, 4 + (index + 1) % 4) for index in range(4)]
    )
    carved = ConstrainedDelaunayTriangulation(
        points, segments, holes=np.asarray(((2.0, 2.0),))
    )
    corners = carved.points[carved.triangles]
    area = np.sum(_signed_areas(corners))
    assert area == pytest.approx(12.0, rel=1e-14)
    _segment_coverage(carved, segments)

    limited = ConstrainedDelaunayTriangulation(
        points, segments, holes=np.asarray(((2.0, 2.0),)), min_angle=30.0, max_steiner=1
    )
    assert limited.evidence.status == "refinement_limit"
    assert limited.evidence.steiner_count <= 1
    with pytest.raises(ValueError, match="cross"):
        ConstrainedDelaunayTriangulation(outer, np.asarray(((0, 2), (1, 3))))
    rng = np.random.default_rng(2)
    planar = VoronoiDiagram(
        rng.random((80, 2)), box_lower=(0.0, 0.0), box_upper=(1.0, 1.0)
    )
    assert np.sum(planar.cells.measures) == pytest.approx(1.0, rel=1e-12)
    assert np.all(planar.cells.measures > 0.0)
    assert _reciprocal_faces(planar.cells)

    spatial = VoronoiDiagram(
        rng.random((60, 3)), box_lower=(0.0, 0.0, 0.0), box_upper=(1.0, 2.0, 1.0)
    )
    assert np.sum(spatial.cells.measures) == pytest.approx(2.0, rel=1e-12)
    assert _reciprocal_faces(spatial.cells)
    centroids = spatial.cells.centroids
    assert np.all((centroids >= 0.0) & (centroids <= (1.0, 2.0, 1.0)))

    # Right triangle x >= 0, y >= 0, x + y <= 1 as a convex domain in the unit box.
    triangle = VoronoiDiagram(
        rng.random((40, 2)) * 0.5,
        box_lower=(0.0, 0.0),
        box_upper=(1.0, 1.0),
        domain_normals=np.asarray(((1.0, 1.0),)),
        domain_offsets=np.asarray((1.0,)),
    )
    assert np.sum(triangle.cells.measures) == pytest.approx(0.5, rel=1e-12)
    assert np.any(triangle.cells.face_labels == -(1 + 2 * 2))

    collinear = VoronoiDiagram(
        np.stack((np.linspace(0.1, 0.9, 5), np.full(5, 0.5)), axis=1),
        box_lower=(0.0, 0.0),
        box_upper=(1.0, 1.0),
    )
    assert collinear.dual is None
    assert np.sum(collinear.cells.measures) == pytest.approx(1.0, rel=1e-14)
    rng = np.random.default_rng(4)
    points = rng.random((50, 3))
    unweighted = PowerDiagram(
        points, np.zeros(50), box_lower=(0.0, 0.0, 0.0), box_upper=(1.0, 1.0, 1.0)
    )
    voronoi = VoronoiDiagram(points, box_lower=(0.0, 0.0, 0.0), box_upper=(1.0, 1.0, 1.0))
    np.testing.assert_allclose(
        unweighted.cells.measures, voronoi.cells.measures, rtol=1e-10
    )

    weights = rng.random(50) * 0.02
    weighted = PowerDiagram(
        points, weights, box_lower=(0.0, 0.0, 0.0), box_upper=(1.0, 1.0, 1.0)
    )
    assert np.sum(weighted.cells.measures) == pytest.approx(1.0, rel=1e-12)
    assert _reciprocal_faces(weighted.cells)

    planar = np.asarray(((0.2, 0.2), (0.8, 0.2), (0.5, 0.8), (0.5, 0.4)))
    hidden = PowerDiagram(
        planar,
        np.asarray((1.0, 1.0, 1.0, 0.0)),
        box_lower=(0.0, 0.0),
        box_upper=(1.0, 1.0),
    )
    # ty: ignore[not-subscriptable]
    assert hidden.dual_vertex_map[3] == -1
    assert hidden.cells.measures[3] == 0.0
    assert hidden.evidence.redundant_count == 1
    assert np.sum(hidden.cells.measures) == pytest.approx(1.0, rel=1e-12)


def _reciprocal_faces(cells: Any) -> Any:
    offsets = cells.cell_face_offsets
    owners = np.repeat(np.arange(cells.cell_count), np.diff(offsets))
    labels = cells.face_labels
    interior = labels >= 0
    forward = set(zip(owners[interior].tolist(), labels[interior].tolist(), strict=True))
    return all((other, owner) in forward for owner, other in forward)


# Rational contact reference, independent of the native kernel: the
# intersection of two closed triangles is the convex hull of every
# segment/triangle intersection of an edge of one with the other, each
# computed exactly in Fractions.


def _vector(point: Any) -> Any:
    return tuple(Fraction(float(value)) for value in point)


def _sub(p: Any, q: Any) -> Any:
    return tuple(a - b for a, b in zip(p, q, strict=True))


def _cross(p: Any, q: Any) -> Any:
    return (
        p[1] * q[2] - p[2] * q[1],
        p[2] * q[0] - p[0] * q[2],
        p[0] * q[1] - p[1] * q[0],
    )


def _dot(p: Any, q: Any) -> Any:
    return sum(a * b for a, b in zip(p, q, strict=True))


def _normal(triangle: Any) -> Any:
    return _cross(_sub(triangle[1], triangle[0]), _sub(triangle[2], triangle[0]))


def _edge_side(triangle: Any, normal: Any, k: int, x: Any) -> Any:
    u, v = triangle[(k + 1) % 3], triangle[(k + 2) % 3]
    return _dot(_cross(_sub(v, u), _sub(x, u)), normal)


def _containment(x: Any, triangle: Any) -> int:
    """-1 outside, 0 on the boundary, 1 inside a triangle whose plane holds x."""

    sides = [_edge_side(triangle, _normal(triangle), k, x) for k in range(3)]
    if min(sides) < 0:
        return -1
    return 0 if min(sides) == 0 else 1


def _segment_hits(p: Any, q: Any, triangle: Any) -> Any:
    normal = _normal(triangle)
    sp = _dot(normal, _sub(p, triangle[0]))
    sq = _dot(normal, _sub(q, triangle[0]))
    direction = _sub(q, p)
    if sp == 0 and sq == 0:
        low, high = Fraction(0), Fraction(1)
        for k in range(3):
            g0 = _edge_side(triangle, normal, k, p)
            g1 = _edge_side(triangle, normal, k, q)
            if g0 == g1:
                if g0 < 0:
                    return set()
                continue
            crossing = g0 / (g0 - g1)
            if g1 < g0:
                high = min(high, crossing)
            else:
                low = max(low, crossing)
        if low > high:
            return set()
        return {
            tuple(a + t * d for a, d in zip(p, direction, strict=True))
            for t in (low, high)
        }
    if sp * sq > 0:
        return set()
    t = sp / (sp - sq)
    x = tuple(a + t * d for a, d in zip(p, direction, strict=True))
    return {x} if _containment(x, triangle) >= 0 else set()


def _hull_count(points: Any, normal: Any) -> int:
    if len(points) <= 1:
        return len(points)
    origin = points[0]
    spans = [_sub(point, origin) for point in points[1:]]
    if all(_cross(spans[0], span) == (0, 0, 0) for span in spans):
        return 2
    axis = next(k for k in range(3) if normal[k] != 0)
    flat = sorted({(point[(axis + 1) % 3], point[(axis + 2) % 3]) for point in points})

    def turn(o: Any, a: Any, b: Any) -> Any:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    hull: list[Any] = []
    for sequence in (flat, flat[::-1]):
        chain: list[Any] = []
        for point in sequence:
            while len(chain) >= 2 and turn(chain[-2], chain[-1], point) <= 0:
                chain.pop()
            chain.append(point)
        hull.extend(chain[:-1])
    return len(hull)


def _extremes(points: Any) -> Any:
    ordered = sorted(points)
    return ordered[0], ordered[-1]


def _triangle_reference(first: Any, second: Any) -> tuple[int, int, Any]:
    a = [_vector(point) for point in first]
    b = [_vector(point) for point in second]
    points: set[Any] = set()
    for k in range(3):
        points |= _segment_hits(a[k], a[(k + 1) % 3], b)
        points |= _segment_hits(b[k], b[(k + 1) % 3], a)
    ordered = sorted(points)
    coplanar = all(_dot(_normal(b), _sub(vertex, b[0])) == 0 for vertex in a)
    count = _hull_count(ordered, _normal(a))
    shared = set(a) & set(b)
    if count == 0:
        kind = IntersectionClass.DISJOINT
    elif count >= 3:
        coincident = set(a) == set(b)
        kind = (
            IntersectionClass.COINCIDENT
            if coincident
            else IntersectionClass.COPLANAR_OVERLAP
        )
    elif count == 1:
        kind = (
            IntersectionClass.SHARED_VERTEX
            if ordered[0] in shared
            else IntersectionClass.TOUCHING
        )
    else:
        p, q = _extremes(ordered)
        middle = tuple((x + y) / 2 for x, y in zip(p, q, strict=True))
        if not coplanar and _containment(middle, a) == 1 and _containment(middle, b) == 1:
            kind = IntersectionClass.CROSSING
        elif p in shared and q in shared:
            kind = IntersectionClass.SHARED_EDGE
        else:
            kind = IntersectionClass.TOUCHING
    return kind, count, ordered


def _segment_reference(segment: Any, triangle: Any) -> tuple[int, int, Any]:
    p, q = (_vector(point) for point in segment)
    t = [_vector(point) for point in triangle]
    ordered = sorted(_segment_hits(p, q, t))
    coplanar = all(_dot(_normal(t), _sub(x, t[0])) == 0 for x in (p, q))
    shared = {p, q} & set(t)
    if not ordered:
        kind = IntersectionClass.DISJOINT
    elif len(ordered) == 1:
        x = ordered[0]
        if x in shared:
            kind = IntersectionClass.SHARED_VERTEX
        elif not coplanar and x not in (p, q) and _containment(x, t) == 1:
            kind = IntersectionClass.CROSSING
        else:
            kind = IntersectionClass.TOUCHING
    else:
        middle = tuple((x + y) / 2 for x, y in zip(*ordered, strict=True))
        if set(ordered) == {p, q} and shared == {p, q}:
            kind = IntersectionClass.SHARED_EDGE
        elif _containment(middle, t) == 1:
            kind = IntersectionClass.COPLANAR_OVERLAP
        else:
            kind = IntersectionClass.TOUCHING
    return kind, len(ordered), ordered


def _lattice_pairs(seed: int, count: int, corners: int) -> tuple[Any, Any]:
    rng = np.random.default_rng(seed)
    first = rng.integers(-1, 2, size=(count, corners, 3)).astype(np.float64)
    second = rng.integers(-1, 2, size=(count, 3, 3)).astype(np.float64)
    # Every third pair lies in one plane, where coplanar contacts are frequent.
    first[::3, :, 2] = 0.0
    second[::3, :, 2] = 0.0
    return first, second


def _assert_points_match(points: Any, bounds: Any, count: int, reference: Any) -> None:
    exact = np.asarray([[float(value) for value in point] for point in reference])
    for k in range(count):
        distance = np.max(np.abs(exact - points[k]), axis=1)
        assert np.min(distance) <= bounds[k] + 2.0**-50


def test_triangle_contacts_match_rational_reference() -> None:
    first, second = _lattice_pairs(5, 900, 3)
    classes, counts, points, bounds, features, status = triangle_intersections(
        first, second
    )
    only_classes, only_status = triangle_intersection_classes(first, second)
    np.testing.assert_array_equal(only_classes, classes)
    np.testing.assert_array_equal(only_status, status)
    seen = set()
    for index in range(first.shape[0]):
        a = [_vector(point) for point in first[index]]
        b = [_vector(point) for point in second[index]]
        if _normal(a) == (0, 0, 0) or _normal(b) == (0, 0, 0):
            assert status[index] == MeshcoreStatus.DEGENERATE_INPUT, index
            assert classes[index] == -1, index
            continue
        kind, count, reference = _triangle_reference(first[index], second[index])
        assert status[index] == MeshcoreStatus.OK, index
        assert (classes[index], counts[index]) == (kind, count), index
        _assert_points_match(points[index], bounds[index], count, reference)
        assert np.all(features[index, count:] == -1), index
        seen.add(int(kind))
    assert seen == {int(kind) for kind in IntersectionClass}


def test_segment_contacts_match_rational_reference() -> None:
    segments, triangles = _lattice_pairs(6, 900, 2)
    classes, counts, points, bounds, _, status = segment_triangle_intersections(
        segments, triangles
    )
    seen = set()
    for index in range(segments.shape[0]):
        t = [_vector(point) for point in triangles[index]]
        if _normal(t) == (0, 0, 0) or np.array_equal(
            segments[index, 0], segments[index, 1]
        ):
            assert status[index] == MeshcoreStatus.DEGENERATE_INPUT, index
            continue
        kind, count, reference = _segment_reference(segments[index], triangles[index])
        assert status[index] == MeshcoreStatus.OK, index
        assert (classes[index], counts[index]) == (kind, count), index
        _assert_points_match(points[index], bounds[index], count, reference)
        seen.add(int(kind))
    assert seen == {int(kind) for kind in IntersectionClass} - {
        IntersectionClass.COINCIDENT
    }


def test_contact_adjacency_follows_vertex_identity() -> None:
    first = UNIT_TET[None, :3]
    # Shares the origin with the base triangle and touches it nowhere else.
    second = np.asarray((((0.0, 0.0, 0.0), (-1.0, 0.0, 1.0), (0.0, -1.0, 1.0)),))
    classes, status = triangle_intersection_classes(
        first, second, first_ids=[[10, 11, 12]], second_ids=[[10, 20, 21]]
    )
    assert (classes[0], status[0]) == (IntersectionClass.SHARED_VERTEX, MeshcoreStatus.OK)
    # Coincident positions without shared identity are an illegal contact.
    classes, _ = triangle_intersection_classes(
        first, second, first_ids=[[10, 11, 12]], second_ids=[[30, 20, 21]]
    )
    assert classes[0] == IntersectionClass.TOUCHING
    # One id naming two positions is refused per item.
    classes, status = triangle_intersection_classes(
        first, second, first_ids=[[10, 11, 12]], second_ids=[[11, 20, 21]]
    )
    assert (classes[0], status[0]) == (-1, MeshcoreStatus.INVALID_INPUT)
    nonfinite = second.copy()
    nonfinite[0, 1, 2] = np.inf
    classes, status = triangle_intersection_classes(first, nonfinite)
    assert (classes[0], status[0]) == (-1, MeshcoreStatus.NONFINITE_INPUT)
    with pytest.raises(ValueError, match="both inputs"):
        triangle_intersection_classes(first, second, first_ids=[[0, 1, 2]])
    with pytest.raises(ValueError, match="shape"):
        triangle_intersections(first, second[:, :2])
    with pytest.raises(ValueError, match="equal length"):
        segment_triangle_intersections(np.zeros((2, 2, 3)), first)


def _cube_lattice() -> Any:
    axis = np.arange(3.0)
    return np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3)


def test_incremental_delaunay_batches_match_the_point_set() -> None:
    rng = np.random.default_rng(12)
    points = rng.random((1200, 3))
    points[900] = points[4]  # an earlier vertex
    points[1001] = points[1000]  # a duplicate within one batch
    triangulation = IncrementalDelaunay3D(points[:300], max_vertices=2000)
    first, first_status = triangulation.insert(points[300:1000])
    second, second_status = triangulation.insert(points[1000:])
    mesh_points, cells, vertex_map, constraints, regions = triangulation.finalize()
    expected_cells, expected_map = delaunay_3d(points)

    np.testing.assert_array_equal(first_status, MeshcoreStatus.OK)
    np.testing.assert_array_equal(second_status, MeshcoreStatus.OK)
    assert first[600] == 4 and second[1] == 1000
    np.testing.assert_array_equal(cells, expected_cells)
    np.testing.assert_array_equal(vertex_map, expected_map)
    np.testing.assert_array_equal(mesh_points, points)
    np.testing.assert_array_equal(constraints, -1)
    np.testing.assert_array_equal(regions, -1)
    statistics = dict(zip(TRIANGULATION_3D_STATISTICS, triangulation.statistics()))
    assert statistics["vertex_ids"] == 1200
    assert statistics["live_vertices"] == 1198
    assert statistics["finite_cells"] == cells.shape[0]
    assert statistics["refused_insertions"] == 0
    assert statistics["peak_retained_bytes"] >= statistics["retained_bytes"] > 0


def test_incremental_delaunay_preserves_constraints_and_regions() -> None:
    lattice = _cube_lattice()
    triangulation = IncrementalDelaunay3D(lattice, max_vertices=500)
    _, cells, _, _, _ = triangulation.finalize()
    facets = np.asarray(
        [
            np.delete(cell, slot)
            for cell in cells
            for slot in range(4)
            if np.all(lattice[np.delete(cell, slot), 2] == 1.0)
        ]
    )
    np.testing.assert_array_equal(
        triangulation.constrain_facets(facets, np.full(len(facets), 4)), MeshcoreStatus.OK
    )
    assert (
        triangulation.constrain_facets([[0, 1, 26]], [5])[0]
        == MeshcoreStatus.INVALID_INPUT
    )
    seeds = np.asarray(((0.31, 0.57, 0.23), (0.31, 0.57, 1.77)))
    np.testing.assert_array_equal(triangulation.label_regions(seeds, [0, 1]), 0)

    before = triangulation.finalize()[1]
    vertex, status = triangulation.insert([[0.3, 0.6, 1.0]])
    assert (vertex[0], status[0]) == (-1, MeshcoreStatus.CONSTRAINT_INTERSECTION)
    np.testing.assert_array_equal(triangulation.finalize()[1], before)

    extra = 2.0 * np.random.default_rng(13).random((300, 3))
    _, status = triangulation.insert(extra)
    np.testing.assert_array_equal(status, MeshcoreStatus.OK)
    mesh_points, cells, _, constraints, regions = triangulation.finalize()
    centroid_heights = mesh_points[cells, 2].mean(axis=1)
    np.testing.assert_array_equal(regions, np.where(centroid_heights < 1.0, 0, 1))
    constrained = [
        np.delete(cell, slot)
        for cell, labels in zip(cells, constraints, strict=True)
        for slot in range(4)
        if labels[slot] == 4
    ]
    corners = mesh_points[np.asarray(constrained)]
    np.testing.assert_array_equal(corners[..., 2], 1.0)
    edges = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    # Both sides of the constrained 2 x 2 square survive every insertion.
    assert 0.5 * np.sum(np.abs(edges[:, 2])) == pytest.approx(8.0, rel=1e-14)

    located, locations, status = triangulation.locate(
        [[0.3, 0.6, 1.0], [1.0, 1.0, 1.0], [5.0, 5.0, 5.0]]
    )
    np.testing.assert_array_equal(status, MeshcoreStatus.OK)
    np.testing.assert_array_equal(locations, (1, 3, -1))
    assert located[2, 0] == -1 and 13 in located[1]


def test_incremental_delaunay_limits_refuse_without_change() -> None:
    points = np.random.default_rng(14).random((400, 3))
    triangulation = IncrementalDelaunay3D(points[:100], max_vertices=400)
    before = triangulation.finalize()[1]
    vertices, status = triangulation.insert(points[100:300], work_limit=1)
    np.testing.assert_array_equal(status, MeshcoreStatus.CAPACITY_EXCEEDED)
    np.testing.assert_array_equal(vertices, -1)
    np.testing.assert_array_equal(triangulation.finalize()[1], before)
    with pytest.raises(MeshcoreError, match="capacity"):
        triangulation.insert(points[:201])
    with pytest.raises(ValueError, match="exponent|magnitude"):
        triangulation.insert([[2.0**130, 0.0, 0.0]])
    assert triangulation.statistics()[0] == 300
    triangulation.close()
    with pytest.raises(ValueError, match="closed"):
        triangulation.locate(points[:1])
    with pytest.raises(ValueError, match="span"):
        IncrementalDelaunay3D(points[:3])


@pytest.mark.parametrize("limit", ["max_work", "max_cavity_cells", "max_scratch_bytes"])
def test_planar_zero_limit_retains_actual_refusal(limit: str) -> None:
    points = np.asarray(
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)), dtype=np.float64
    )
    observed: list[PlanarExecutionEvidence] = []
    with pytest.raises(NativeResourceFailure) as caught:
        delaunay_2d(points, **{limit: 0}, record_native_resources=observed.append)
    evidence = observed[0]
    assert evidence.status is MeshcoreStatus.CAPACITY_EXCEEDED
    assert caught.value.work_evidence is evidence.work_evidence
    assert caught.value.memory_evidence is evidence.memory_evidence
    assert not evidence.work_evidence.flags.writeable
    assert not evidence.memory_evidence.flags.writeable
    if limit == "max_work":
        assert evidence.work_evidence[0] == 0
        assert evidence.work_evidence[7] > 0
    elif limit == "max_cavity_cells":
        assert evidence.work_evidence[6] > 0
        assert evidence.work_evidence[8] > 0
    else:
        assert evidence.memory_evidence[0] == 0
        assert evidence.memory_evidence[2] == 0
        assert evidence.memory_evidence[3] > 0
        assert evidence.memory_evidence[5] > 0


def test_cdt_observed_work_budget_is_reproducible() -> None:
    points = np.asarray(
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)), dtype=np.float64
    )
    segments = np.asarray(((0, 1), (1, 2), (2, 3), (3, 0)), dtype=np.int32)
    baseline = ConstrainedDelaunayTriangulation(points, segments)
    bounded = ConstrainedDelaunayTriangulation(
        points,
        segments,
        maximum_work=int(baseline.work_evidence[0]),
    )
    np.testing.assert_array_equal(bounded.points, baseline.points)
    np.testing.assert_array_equal(bounded.triangles, baseline.triangles)
    np.testing.assert_array_equal(bounded.segment_ids, baseline.segment_ids)
    np.testing.assert_array_equal(bounded.work_evidence, baseline.work_evidence)
    with pytest.raises(NativeResourceFailure) as caught:
        ConstrainedDelaunayTriangulation(
            points,
            segments,
            maximum_work=int(baseline.work_evidence[0]) - 1,
        )
    work = caught.value.work_evidence
    assert work is not None
    assert work[0] == baseline.work_evidence[0] - 1
    assert work[7] > 0


def test_plc_zero_scratch_cap_retains_actual_refusal() -> None:
    points = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    faces = np.asarray(((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)), dtype=np.int32)
    facet_regions = np.tile(np.asarray((-1, 0), dtype=np.int32), (4, 1))
    with pytest.raises(PlcRecoveryFailure) as caught:
        recover_plc_3d(
            points,
            np.arange(0, 13, 3, dtype=np.int64),
            faces.reshape(-1),
            np.arange(4, dtype=np.int32),
            facet_regions,
            boundary_policy="fixed",
            max_vertices=100,
            max_tetrahedra=100,
            work_limit=100000,
            max_scratch_bytes=0,
        )
    assert caught.value.status is MeshcoreStatus.CAPACITY_EXCEEDED
    assert caught.value.reason == "scratch_byte_budget"
    memory = caught.value.memory_evidence
    assert memory is not None
    assert memory[0] == 0
    assert memory[1] == 0
    assert memory[2] == 0
    assert memory[3] > 0
    assert memory[4] == 0
    assert memory[5] > 0
    assert not memory.flags.writeable


def test_source_only_plc_preparation_enforces_measured_work_boundary() -> None:
    from phydrax._meshcore import PLC_3D_COUNTERS, plc_source_constraints

    points = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    faces = np.asarray(((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)), dtype=np.int32)
    offsets = np.arange(0, 13, 3, dtype=np.int64)
    polygons = np.arange(4, dtype=np.int32)
    regions = np.tile(np.asarray((-1, 0), dtype=np.int32), (4, 1))
    source = plc_source_constraints(
        points,
        offsets,
        faces.reshape(-1),
        polygons,
        regions,
        work_limit=100000,
        max_scratch_bytes=1000000,
    )
    work_index = PLC_3D_COUNTERS.index("work_units")
    measured_work = int(source.counters[work_index])
    bounded = plc_source_constraints(
        points,
        offsets,
        faces.reshape(-1),
        polygons,
        regions,
        work_limit=measured_work,
        max_scratch_bytes=1000000,
    )
    expected_edges = {(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)}
    assert {tuple(sorted(edge)) for edge in bounded.plc_edges.tolist()} == expected_edges
    assert (
        dict(zip(PLC_3D_COUNTERS, bounded.counters.tolist(), strict=True))["cavities"]
        == 0
    )
    boundary = np.zeros((4, 4), dtype=np.int32)
    for triangle in bounded.input_triangles.tolist():
        for first, second in zip(triangle, triangle[1:] + triangle[:1], strict=True):
            boundary[first, second] += 1
            boundary[second, first] -= 1
    np.testing.assert_array_equal(boundary, np.zeros((4, 4), dtype=np.int32))
    with pytest.raises(PlcRecoveryFailure) as caught:
        plc_source_constraints(
            points,
            offsets,
            faces.reshape(-1),
            polygons,
            regions,
            work_limit=measured_work - 1,
            max_scratch_bytes=1000000,
        )
    assert caught.value.reason == "work_budget"
    assert caught.value.counters[work_index] == measured_work - 1


def _host_budget(
    byte_limit: int, *, mesh: TetMesh3D | None = None
) -> NativeExecutionBudget:
    return NativeExecutionBudget(
        max_work=1 << 40,
        max_geometry_queries=1 << 40,
        max_cavity_cells=1 << 20,
        max_scratch_bytes=byte_limit,
        max_wall_seconds=120.0,
        mesh=mesh,
    )


def test_deferred_worker_failure_flushes_accounting_before_propagation() -> None:
    budget = _host_budget(1 << 16)

    def fail_after_charge() -> None:
        with budget.deferred_worker():
            budget.charge(work=7, geometry_queries=3)
            raise ValueError("deferred preparation failed")

    with budget:
        with pytest.raises(ValueError, match="deferred preparation failed"):
            with budget.deferred_workers():
                with ThreadPoolExecutor(max_workers=1) as pool:
                    pool.submit(fail_after_charge).result()
    assert budget.evidence is not None
    assert budget.evidence.externally_charged_work == 7
    assert budget.evidence.externally_charged_geometry_queries == 3


def test_native_host_storage_bound_and_managed_payload_share_original_cap() -> None:
    import sys

    from phydrax.meshing._measurements import NativeExecutionRecord

    budget = _host_budget(4096)
    bound = sys.getsizeof(bytearray()) + 2048 + 1
    with pytest.raises(MeshcoreError) as caught, budget:
        with budget.host_workspace() as workspace:
            workspace.set_bound(bound)
            payload = bytearray(2048)
            assert sys.getsizeof(payload) <= bound
            payload[0] = 31
            try:
                budget.allocate_host_array((2048,), np.uint8)
            finally:
                assert payload[0] == 31
                del payload
                # Shrinking an existing token must work during refusal cleanup.
                workspace.set_bound(0)
    assert caught.value.status is MeshcoreStatus.CAPACITY_EXCEEDED
    assert budget.evidence is not None
    assert budget.evidence.memory_evidence.shape == (6,)
    assert budget.evidence.host_storage_live_bytes_upper == 0
    assert budget.evidence.host_storage_peak_bytes_upper == bound
    assert budget.evidence.memory_evidence[2] < bound
    record = NativeExecutionRecord(budget.evidence)
    assert int(np.asarray(record.host_storage_peak_bytes_upper)) == bound
    assert int(np.asarray(record.host_storage_live_bytes_upper)) == 0


def test_native_host_storage_atomic_growth_refusal_preserves_old_bound() -> None:
    budget = _host_budget(4096)
    with pytest.raises(MeshcoreError) as ended, budget:
        with budget.host_workspace() as workspace:
            workspace.set_bound(1024)
            payload = bytearray(512)
            with pytest.raises(MeshcoreError) as caught:
                workspace.set_bound(4096)
            assert caught.value.status is MeshcoreStatus.CAPACITY_EXCEEDED
            assert workspace.bound == 1024
            payload[-1] = 17
            workspace.set_bound(768)
            assert workspace.bound == 768 and payload[-1] == 17
            del payload
            workspace.set_bound(0)
    assert ended.value.status is MeshcoreStatus.CAPACITY_EXCEEDED
    assert budget.evidence is not None
    assert budget.evidence.host_storage_live_bytes_upper == 0
    assert budget.evidence.host_storage_peak_bytes_upper == 1024


def test_native_zero_host_workspace_does_not_allocate_a_token() -> None:
    budget = _host_budget(0)
    with budget:
        with budget.host_workspace() as workspace:
            workspace.set_bound(0)
            assert workspace.bound == 0
    assert budget.evidence is not None
    assert budget.evidence.memory_evidence[4] == 0
    assert budget.evidence.host_storage_peak_bytes_upper == 0


def test_native_retained_source_views_share_actual_base_owner_and_release_temporary_indices() -> (
    None
):
    source = np.arange(1024, dtype=np.float64)
    owners = (source, source[1:], source[2:])
    budget = _host_budget(1 << 16)
    with budget:
        with budget.host_workspace() as workspace:
            workspace.retain_owner(owners)
            retained = workspace.bound
            assert source.nbytes <= retained < 2 * source.nbytes
            workspace.retain_owner(source)
            assert workspace.bound == retained
            workspace.retain_owner(owners)
            assert workspace.bound == retained
    assert budget.evidence is not None
    assert budget.evidence.externally_charged_work > 0
    assert budget.evidence.host_storage_live_bytes_upper == 0
    np.testing.assert_array_equal(source, np.arange(1024, dtype=np.float64))


def test_native_retained_source_does_not_double_charge_managed_array_payload() -> None:
    budget = _host_budget(1 << 16)
    with budget:
        array = budget.allocate_host_array((1024,), np.float64)
        array[:] = np.arange(1024, dtype=np.float64)
        with budget.host_workspace() as workspace:
            workspace.retain_owner((array, array[1:]))
            assert workspace.bound < array.nbytes
    assert budget.evidence is not None
    assert budget.evidence.memory_evidence[2] >= array.nbytes
    np.testing.assert_array_equal(array, np.arange(1024, dtype=np.float64))


def test_native_retained_source_refuses_cycles_callbacks_and_resource_growth_atomically() -> (
    None
):
    source = np.arange(1024, dtype=np.float64)
    cyclic: dict[str, Any] = {}
    cyclic["self"] = cyclic
    with _host_budget(1 << 16) as budget:
        with budget.host_workspace() as workspace:
            with pytest.raises(ValueError, match="cyclic"):
                workspace.retain_owner(cyclic)
            assert workspace.bound == 0
            with pytest.raises(TypeError, match="unsupported owner"):
                workspace.retain_owner(lambda: source)
            assert workspace.bound == 0
            workspace.retain_owner(source)
    refused = _host_budget(4096)
    with pytest.raises(MeshcoreError) as caught, refused:
        with refused.host_workspace() as workspace:
            workspace.retain_owner(source)
    assert caught.value.status is MeshcoreStatus.CAPACITY_EXCEEDED
    assert refused.evidence is not None
    assert refused.evidence.host_storage_live_bytes_upper == 0
    np.testing.assert_array_equal(source, np.arange(1024, dtype=np.float64))


def test_retained_source_in_parent_workspace_respects_child_work_allowance() -> None:
    source = np.arange(4, dtype=np.float64)
    parent = _host_budget(1 << 16)
    child = NativeExecutionBudget(
        max_work=1,
        max_geometry_queries=1000,
        max_cavity_cells=1000,
        max_scratch_bytes=1 << 16,
        max_wall_seconds=120.0,
    )
    with parent:
        with parent.host_workspace() as workspace:
            with pytest.raises(MeshcoreError) as refused, child:
                workspace.retain_owner((source, source[1:]))
            assert refused.value.status is MeshcoreStatus.CAPACITY_EXCEEDED
            assert workspace.bound == 0
    assert child.evidence is not None and parent.evidence is not None
    assert child.evidence.externally_charged_work == 1
    assert parent.evidence.externally_charged_work == 1
    assert child.evidence.work_evidence[0] == 1
    assert parent.evidence.work_evidence[0] == 1
    assert parent.evidence.host_storage_live_bytes_upper == 0
    np.testing.assert_array_equal(source, np.arange(4, dtype=np.float64))


def test_native_retained_alias_keeps_actual_source_base_alive_until_close() -> None:
    import weakref

    source = np.arange(1024, dtype=np.float64)
    view = source[1:]
    source_reference = weakref.ref(source)
    view_reference = weakref.ref(view)
    with _host_budget(1 << 16) as budget:
        with budget.host_workspace() as workspace:
            workspace.retain_owner(view)
            del source, view
            gc.collect()
            retained_source = source_reference()
            retained_view = view_reference()
            assert retained_source is not None and retained_view is not None
            assert retained_view.base is retained_source
            np.testing.assert_array_equal(
                retained_view, np.arange(1, 1024, dtype=np.float64)
            )
            del retained_source, retained_view
        gc.collect()
        assert source_reference() is None and view_reference() is None


def test_native_retained_jax_logical_unmanaged_label_deduplicates_and_rolls_back() -> (
    None
):
    import jax.numpy as jnp

    source = jnp.asarray(np.arange(16, dtype=np.float64))
    other = jnp.asarray(np.arange(17, dtype=np.float64))
    with _host_budget(1 << 16) as budget:
        with budget.host_workspace() as workspace:
            assert workspace.logical_unmanaged_bytes_upper == 0
            workspace.retain_owner((source, source))
            assert workspace.logical_unmanaged_bytes_upper == source.nbytes
            admitted = workspace.bound
            workspace.retain_owner(source)
            assert workspace.bound == admitted
            assert workspace.logical_unmanaged_bytes_upper == source.nbytes
            with pytest.raises(TypeError, match="unsupported owner"):
                workspace.retain_owner((lambda: None, other))
            assert workspace.bound == admitted
            assert workspace.logical_unmanaged_bytes_upper == source.nbytes
        assert workspace.logical_unmanaged_bytes_upper == 0
    assert budget.evidence is not None
    assert budget.evidence.host_storage_live_bytes_upper == 0


def test_native_retains_real_form_geometry_owner_and_one_under_peak_rolls_back() -> None:
    from phydrax.discretization._cell_geometry import CellGeometrySpec
    from phydrax.discretization._cell_geometry_validity import cell_geometry_id
    from phydrax.discretization._cell_mesh import CellMesh
    from phydrax.discretization.fem._form_elements import form_element

    mesh = CellMesh.from_tetrahedra(
        np.asarray(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            dtype=np.float64,
        ),
        np.asarray(((0, 1, 2, 3),), dtype=np.int64),
    )
    element = form_element("tetrahedron", 0, 1, proxy="scalar")
    name = mesh.blocks[0].name
    geometry = CellGeometrySpec(
        {name: element},
        {name: np.arange(element.local_dof_count, dtype=np.int64)[None, :]},
        mesh.coordinates,
    )
    geometry.resolve(mesh)
    identity = cell_geometry_id(geometry)
    owners = (mesh, geometry)
    calibration = _host_budget(1 << 24)
    with calibration:
        with calibration.host_workspace() as workspace:
            workspace.retain_owner(owners)
            retained = workspace.bound
            logical = workspace.logical_unmanaged_bytes_upper
            assert logical >= geometry.coordinates.nbytes
            workspace.retain_owner(geometry)
            workspace.retain_owner(owners)
            assert workspace.bound == retained
            assert workspace.logical_unmanaged_bytes_upper == logical
    assert calibration.evidence is not None
    original_peak = (
        calibration.evidence.host_storage_peak_bytes_upper
        + calibration.evidence.memory_evidence[2]
    )
    refused = _host_budget(original_peak - 1)
    with pytest.raises(MeshcoreError) as final_refusal, refused:
        with refused.host_workspace() as workspace:
            with pytest.raises(MeshcoreError) as refusal:
                workspace.retain_owner(owners)
            assert refusal.value.status == MeshcoreStatus.CAPACITY_EXCEEDED
            assert workspace.bound == 0
            assert workspace.logical_unmanaged_bytes_upper == 0
    assert final_refusal.value.status == MeshcoreStatus.CAPACITY_EXCEEDED
    assert refused.evidence is not None
    assert refused.evidence.host_storage_live_bytes_upper == 0
    assert cell_geometry_id(geometry) == identity


def test_native_host_array_views_survive_scope_and_release_on_another_thread() -> None:
    points = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    mesh = TetMesh3D(
        points,
        np.asarray(((0, 1, 2, 3),), dtype=np.int32),
        np.zeros((1,), dtype=np.int32),
        np.asarray(((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)), dtype=np.int32),
        np.arange(4, dtype=np.int32),
        np.asarray(((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)), dtype=np.int32),
        np.arange(6, dtype=np.int32),
        boundary_policy="conforming",
    )
    try:
        baseline = int(mesh.memory_evidence()[1])
        source = np.arange(12, dtype=np.float64).reshape((3, 4))
        budget = _host_budget(4096, mesh=mesh)
        with budget:
            array = budget.allocate_host_array(source.shape, source.dtype)
            array[:] = source
            assert array.flags.c_contiguous and array.flags.writeable
            assert not array.flags.owndata
            assert array.dtype == source.dtype and array.shape == source.shape
            np.testing.assert_array_equal(array, source)
            views = [array[:, ::2]]
        retained = int(mesh.memory_evidence()[1])
        assert retained > baseline + array.nbytes
        del array
        gc.collect()
        assert int(mesh.memory_evidence()[1]) == retained
        np.testing.assert_array_equal(views[0], source[:, ::2])
        views[0][1, 1] = -7.0
        assert views[0][1, 1] == -7.0 and source[1, 2] == 6.0
        release = threading.Thread(target=views.clear)
        release.start()
        release.join()
        assert int(mesh.memory_evidence()[1]) == baseline
    finally:
        mesh.close()


@pytest.mark.parametrize("capacity_delta", [0, -1], ids=["exact", "one-byte-under"])
def test_native_host_and_geometry_allocations_share_the_original_cap(
    capacity_delta: int,
) -> None:
    points = np.asarray(
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)), dtype=np.float64
    )
    source = points.copy()
    baseline = _host_budget(1 << 20)
    with baseline:
        held = baseline.allocate_host_array((128,), np.int64)
        held.fill(37)
        expected_cells, expected_map = delaunay_2d(points)
    assert baseline.evidence is not None
    exact_peak = int(baseline.evidence.memory_evidence[2])
    del held
    gc.collect()
    budget = _host_budget(exact_peak + capacity_delta)
    if capacity_delta == 0:
        with budget:
            held = budget.allocate_host_array((128,), np.int64)
            held.fill(37)
            actual_cells, actual_map = delaunay_2d(points)
            np.testing.assert_array_equal(held, np.full((128,), 37, dtype=np.int64))
        np.testing.assert_array_equal(actual_cells, expected_cells)
        np.testing.assert_array_equal(actual_map, expected_map)
    else:
        with pytest.raises(MeshcoreError) as caught, budget:
            held = budget.allocate_host_array((128,), np.int64)
            held.fill(37)
            delaunay_2d(points)
        assert caught.value.status is MeshcoreStatus.CAPACITY_EXCEEDED
        np.testing.assert_array_equal(held, np.full((128,), 37, dtype=np.int64))
    assert budget.evidence is not None
    assert budget.evidence.memory_evidence[2] <= exact_peak + capacity_delta
    np.testing.assert_array_equal(points, source)


def test_native_host_refused_payload_does_not_inflate_an_earlier_peak() -> None:
    budget = _host_budget(4096)
    with pytest.raises(MeshcoreError) as caught, budget:
        array = budget.allocate_host_array((128,), np.uint8)
        array.fill(19)
        np.testing.assert_array_equal(array, np.full((128,), 19, dtype=np.uint8))
        remaining = budget.remaining().remaining_scratch_bytes
        del array
        gc.collect()
        budget.allocate_host_array((4096,), np.uint8)
    assert caught.value.status is MeshcoreStatus.CAPACITY_EXCEEDED
    assert budget.evidence is not None
    memory = budget.evidence.memory_evidence
    assert memory[1] == 0
    assert memory[2] == 4096 - remaining
    assert memory[3] == 4096
    # The earlier owner and payload succeeded, then the refused array's owner
    # metadata succeeded. Its refused payload is not a successful allocation.
    assert memory[4] == 3


@pytest.mark.parametrize("limit", ["zero", "one-byte-under"])
def test_native_host_nested_scope_cannot_renew_root_scratch_allowance(limit: str) -> None:
    baseline = _host_budget(4096)
    with baseline:
        source = baseline.allocate_host_array((6,), np.float64)
        source[:] = np.arange(6, dtype=np.float64)
    assert baseline.evidence is not None
    byte_limit = 0 if limit == "zero" else int(baseline.evidence.memory_evidence[2]) - 1
    root = _host_budget(byte_limit)
    nested = _host_budget(4096)
    with pytest.raises(MeshcoreError) as caught, root, nested:
        nested.allocate_host_array(source.shape, source.dtype)
    assert caught.value.status is MeshcoreStatus.CAPACITY_EXCEEDED
    np.testing.assert_array_equal(source, np.arange(6, dtype=np.float64))
    assert root.evidence is not None and nested.evidence is not None
    assert (
        root.evidence.status is nested.evidence.status is MeshcoreStatus.CAPACITY_EXCEEDED
    )
    assert root.evidence.memory_evidence[1] == 0
    assert root.evidence.memory_evidence[2] <= byte_limit


@pytest.mark.parametrize(
    ("shape", "dtype", "error_type"),
    [
        ((True,), np.float64, TypeError),
        ((-1,), np.float64, ValueError),
        ((1.5,), np.float64, TypeError),
        ((2**64,), np.uint8, ValueError),
        ((2**32, 2**32), np.float64, ValueError),
        ((0, 2**62), np.float64, ValueError),
        ((1,) * 65, np.float64, ValueError),
        ((1,), None, TypeError),
        ((1,), "O", TypeError),
        ((1,), np.dtype([("value", "O")]), TypeError),
        ((1,), "not-a-dtype", TypeError),
        ((1,), np.dtype((np.float64, (2,))), ValueError),
    ],
    ids=[
        "bool-extent",
        "negative-extent",
        "noninteger-extent",
        "dimension-overflow",
        "storage-overflow",
        "empty-stride-overflow",
        "rank-overflow",
        "implicit-dtype",
        "object-dtype",
        "nested-object-dtype",
        "invalid-dtype",
        "shape-changing-dtype",
    ],
)
def test_native_host_invalid_layout_refuses_before_native_allocation(
    shape: tuple[int, ...],
    dtype: DTypeLike,
    error_type: type[Exception],
) -> None:
    budget = _host_budget(4096)
    with budget:
        before = budget.remaining()
        with pytest.raises(error_type):
            budget.allocate_host_array(shape, dtype)
        assert (
            budget.remaining().remaining_scratch_bytes == before.remaining_scratch_bytes
        )
        accepted = budget.allocate_host_array((1,), np.int32)
        accepted[0] = 23
    assert accepted[0] == 23
    assert budget.evidence is not None
    assert budget.evidence.status is MeshcoreStatus.OK
    assert budget.evidence.memory_evidence[4] == 2


def test_native_host_allocation_requires_the_active_creating_thread() -> None:
    budget = _host_budget(4096)
    with pytest.raises(RuntimeError, match="active creating thread"):
        budget.allocate_host_array((1,), np.float64)
    with budget:
        with ThreadPoolExecutor(max_workers=1) as pool:
            rejected = pool.submit(budget.allocate_host_array, (1,), np.float64)
            with pytest.raises(RuntimeError, match="active creating thread"):
                rejected.result()
        with _host_budget(4096):
            with pytest.raises(RuntimeError, match="active creating thread"):
                budget.allocate_host_array((1,), np.float64)
        accepted = budget.allocate_host_array((), np.float64)
        accepted[()] = 3.5
    assert accepted.shape == () and accepted[()] == 3.5
    with pytest.raises(RuntimeError, match="active creating thread"):
        budget.allocate_host_array((1,), np.float64)


def test_native_host_empty_array_retains_its_shape_and_explicit_bool_dtype() -> None:
    budget = _host_budget(4096)
    with budget:
        empty = budget.allocate_host_array((0, 3), np.bool_)
        assert empty.shape == (0, 3) and empty.dtype == np.dtype(np.bool_)
        assert empty.nbytes == 0 and empty.flags.c_contiguous
        assert empty.base is not None
    assert empty.reshape((3, 0)).shape == (3, 0)
