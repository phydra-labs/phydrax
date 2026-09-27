#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


import importlib.metadata
from typing import Any

import numpy as np
import pytest
from scipy.spatial import ConvexHull

from phydrax._meshcore import (
    meshcore_available,
    meshcore_identity,
    MeshcoreStatus,
    polygon_intersection_moments,
    polygon_intersection_simplices,
    polyhedron_clip_moments,
    tetrahedron_intersection_moments,
    tetrahedron_intersection_simplices,
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


def test_identity_reports_the_phydrax_release_and_source_hash() -> None:
    name, version, digest = meshcore_identity().split(" ")
    assert name == "phydrax-meshcore"
    assert version == importlib.metadata.version("phydrax")
    assert len(digest) == 64 and all(char in "0123456789abcdef" for char in digest)


def test_tetrahedron_intersection_moments_match_analytic_cases() -> None:
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


def test_tetrahedron_intersection_rejects_degenerate_cells() -> None:
    flat = UNIT_TET.copy()
    flat[3] = (0.25, 0.25, 0.0)
    volume, _, status = tetrahedron_intersection_moments(flat[None], UNIT_TET[None])
    assert status[0] == MeshcoreStatus.DEGENERATE_INPUT
    assert volume[0] == 0.0


def test_polygon_intersection_moments_handle_orientation_and_contact() -> None:
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


def test_polyhedron_clip_moments_match_analytic_halves() -> None:
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


def test_intersection_simplices_partition_the_clipped_moments() -> None:
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


def _assert_empty_circles(points: Any, triangles: Any) -> None:
    a, b, c = (points[triangles[:, index]] for index in range(3))
    assert np.all(orient2d(a, b, c, mode=EXACT).signs == 1)
    for query in points:
        signs = incircle(a, b, c, np.broadcast_to(query, a.shape), mode=EXACT).signs
        assert np.all(signs <= 0)


def test_delaunay_2d_is_exactly_empty_circle_on_random_and_lattice_points() -> None:
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


def test_delaunay_2d_reports_duplicates_and_rejects_collinear_points() -> None:
    points = np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)))
    triangulation = DelaunayTriangulation(points)
    np.testing.assert_array_equal(triangulation.vertex_map, (0, 1, 2, 1, 4))
    assert triangulation.evidence.duplicate_count == 1
    assert 3 not in triangulation.simplices
    with pytest.raises(ValueError, match="span"):
        DelaunayTriangulation(np.asarray(((0.0, 0.0), (1.0, 1.0), (2.0, 2.0))))


def test_delaunay_3d_is_exactly_empty_sphere_and_fills_the_hull() -> None:
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


def test_constrained_delaunay_keeps_segments_and_reaches_the_angle_bound() -> None:
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


def test_constrained_delaunay_carves_holes_and_reports_refinement_limit() -> None:
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


def _reciprocal_faces(cells: Any) -> Any:
    offsets = cells.cell_face_offsets
    owners = np.repeat(np.arange(cells.cell_count), np.diff(offsets))
    labels = cells.face_labels
    interior = labels >= 0
    forward = set(zip(owners[interior].tolist(), labels[interior].tolist(), strict=True))
    return all((other, owner) in forward for owner, other in forward)


def test_voronoi_cells_partition_the_box_and_convex_domain() -> None:
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


def test_power_cells_partition_the_box_and_hide_redundant_generators() -> None:
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
