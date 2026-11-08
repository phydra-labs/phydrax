#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


import importlib.metadata
import os
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.spatial import ConvexHull

from phydrax._meshcore import (
    arrange_tetrahedra_exact,
    clip_reference_tetrahedron_exact,
    IntervalBoxProgram,
    meshcore_available,
    meshcore_identity,
    MeshcoreError,
    MeshcoreStatus,
    mixed_template,
    mixed_template_counts,
    MixedElementKind,
    parallel_for,
    PLC_SEARCH_BUDGET,
    PlcBoundaryPolicy,
    PlcEntityKind,
    PlcFailureReason,
    PlcRecovery3D,
    PlcSearchBudget,
    PlcSourceFailure,
    polygon_intersection_moments,
    polygon_intersection_simplices,
    polyhedron_clip_moments,
    prepare_plc_source,
    recover_plc,
    recover_plc_3d,
    surface_envelope,
    TetMesh3D,
    TetMeshArrays,
    TetMeshEdit,
    TetMeshEditKind,
    TetMeshEditStatus,
    TetMeshImplicitSource,
    TetMeshPlaneRefusal,
    TetMeshSourceComplex,
    TetMeshSourceEvidence,
    tetrahedron_dual_hexes,
    tetrahedron_intersection_moments,
    tetrahedron_intersection_simplices,
    triangle_dual_quads,
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
        l_shape, segments, min_angle=25.0, max_area=0.05
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


def _pack_integer_rows(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(rows, dtype=np.int64)
    words: list[int] = []
    offsets = [0]
    signs = []
    for value in values.flat:
        scalar = int(value)
        signs.append((scalar > 0) - (scalar < 0))
        magnitude = abs(scalar)
        while magnitude:
            words.append(magnitude & 0xFFFFFFFF)
            magnitude >>= 32
        offsets.append(len(words))
    return (
        np.asarray(words, dtype=np.uint32),
        np.asarray(offsets, dtype=np.int64),
        np.asarray(signs, dtype=np.int8).reshape(values.shape),
    )


def test_exact_reference_clips_export_source_witnesses() -> None:
    planes = np.asarray(
        (
            (-1, 0, 0, 0),
            (0, -1, 0, 0),
            (0, 0, -1, 0),
            (1, 1, 1, -1),
            (2, 0, 0, -1),
        ),
        dtype=np.int64,
    )
    plane_words, plane_offsets, plane_signs = _pack_integer_rows(planes)
    clipped = clip_reference_tetrahedron_exact(plane_words, plane_offsets, plane_signs)
    assert clipped.supporting_planes.shape[1] == 3
    assert clipped.vertex_incident_offsets.shape == (
        clipped.supporting_planes.shape[0] + 1,
    )
    assert clipped.face_offsets.shape == (clipped.face_planes.size + 1,)
    assert clipped.face_offsets[-1] == clipped.face_vertices.size
    assert np.all((clipped.face_planes >= 0) & (clipped.face_planes < planes.shape[0]))

    points = np.asarray(
        ((0, 0, 0, 1), (1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)),
        dtype=np.int64,
    )
    point_words, point_offsets, point_signs = _pack_integer_rows(points)
    arrangement_planes = np.asarray(
        ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (1, 1, 1, -1), (2, 0, 0, -1)),
        dtype=np.int64,
    )
    arrangement_plane_words, arrangement_plane_offsets, arrangement_plane_signs = (
        _pack_integer_rows(arrangement_planes)
    )
    arrangement = arrange_tetrahedra_exact(
        point_words,
        point_offsets,
        point_signs,
        arrangement_plane_words,
        arrangement_plane_offsets,
        arrangement_plane_signs,
        np.asarray(((0, 1, 2, 3),), dtype=np.int32),
        np.asarray(((4, -1, -2, -3),), dtype=np.int32),
        np.asarray((-1,), dtype=np.int32),
        np.asarray((0, 0), dtype=np.int64),
        np.empty((0,), dtype=np.int32),
    )
    assert arrangement.vertex_signs.shape == (4, 4)
    assert arrangement.vertex_word_offsets.shape == (17,)
    assert arrangement.job_face_offsets.shape == (2,)
    assert arrangement.job_face_offsets[-1] == arrangement.face_planes.size
    assert arrangement.face_offsets[-1] == arrangement.face_vertices.size


def test_plc_source_python_wrapper_preserves_topology_and_witnesses() -> None:
    points = np.asarray(
        (
            (0, 0, 0),
            (1, 0, 0),
            (0, 1, 0),
            (1, 1, 0),
            (0, 0, 1),
            (1, 0, 1),
            (0, 1, 1),
            (1, 1, 1),
        ),
        dtype=np.float64,
    )
    result = prepare_plc_source(
        points,
        np.asarray((0, 4, 8, 12, 16, 20, 24), dtype=np.int64),
        np.asarray(
            (0, 2, 3, 1, 4, 5, 7, 6, 0, 1, 5, 4, 2, 6, 7, 3, 0, 4, 6, 2, 1, 3, 7, 5),
            dtype=np.int32,
        ),
        np.arange(6, dtype=np.int32),
        np.tile(np.asarray((-1, 0), dtype=np.int32), (6, 1)),
        np.empty((0, 2), dtype=np.int32),
        boundary_policy=PlcBoundaryPolicy.CONFORMING,
    )
    assert result.status is MeshcoreStatus.OK
    assert result.points.shape == (8, 3)
    assert result.tetrahedra.shape == (0, 4)
    assert result.plc_edges.shape == (12, 2)
    assert result.input_triangles.shape == (12, 3)
    assert result.input_polygons.shape == (12,)
    assert result.source_strata.shape == (8,)
    assert np.all(result.source_strata == 0)
    assert np.all(result.source_entities == -1)
    recovered = recover_plc(
        points,
        np.asarray((0, 4, 8, 12, 16, 20, 24), dtype=np.int64),
        np.asarray(
            (0, 2, 3, 1, 4, 5, 7, 6, 0, 1, 5, 4, 2, 6, 7, 3, 0, 4, 6, 2, 1, 3, 7, 5),
            dtype=np.int32,
        ),
        np.arange(6, dtype=np.int32),
        np.tile(np.asarray((-1, 0), dtype=np.int32), (6, 1)),
        np.empty((0, 2), dtype=np.int32),
        boundary_policy=PlcBoundaryPolicy.CONFORMING,
    )
    assert recovered.status is MeshcoreStatus.OK
    assert recovered.tetrahedra.size > 0
    assert recovered.tetrahedron_regions.shape == (recovered.tetrahedra.shape[0],)
    assert recovered.face_orientations.shape == (recovered.faces.shape[0],)
    recovered_3d = recover_plc_3d(
        points,
        np.asarray((0, 4, 8, 12, 16, 20, 24), dtype=np.int64),
        np.asarray(
            (0, 2, 3, 1, 4, 5, 7, 6, 0, 1, 5, 4, 2, 6, 7, 3, 0, 4, 6, 2, 1, 3, 7, 5),
            dtype=np.int32,
        ),
        np.arange(6, dtype=np.int32),
        np.tile(np.asarray((-1, 0), dtype=np.int32), (6, 1)),
        boundary_policy=PlcBoundaryPolicy.CONFORMING,
    )
    assert recovered_3d.status is MeshcoreStatus.OK
    assert isinstance(recovered_3d, PlcRecovery3D)
    assert recovered_3d.tetrahedra.shape[1] == 4
    assert recovered_3d.witness_parameters.shape == (recovered_3d.points.shape[0], 2)


def test_mixed_template_python_wrappers() -> None:
    counts = mixed_template_counts(MixedElementKind.TETRAHEDRON, False, 0)
    assert counts.template_count == 11
    assert counts.child_count == 1
    assert counts.corner_count == 4
    parent = mixed_template(MixedElementKind.TETRAHEDRON, False, 0)
    assert parent.shape == (1, 4, 3)
    np.testing.assert_array_equal(
        parent[0],
        np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
    )


def test_parallel_for_wrapper_preserves_status_and_exception() -> None:
    calls = np.zeros((7,), dtype=np.int32)

    def fill(begin: int, end: int, _chunk_index: int) -> MeshcoreStatus:
        calls[begin:end] += 1
        return MeshcoreStatus.OK

    assert parallel_for(7, fill, chunk_size=2, parallelism=3) is MeshcoreStatus.OK
    np.testing.assert_array_equal(calls, 1)
    invoked: list[int] = []
    assert (
        parallel_for(
            4,
            lambda begin, end, chunk: invoked.append(chunk) or MeshcoreStatus.UNRESOLVED,
            chunk_size=1,
            parallelism=2,
        )
        is MeshcoreStatus.UNRESOLVED
    )
    assert sorted(invoked) == [0, 1, 2, 3]

    def fail(_begin: int, _end: int, _chunk: int) -> None:
        raise RuntimeError("callback failed")

    with pytest.raises(RuntimeError, match="callback failed"):
        parallel_for(2, fail, chunk_size=1, parallelism=2)


def _reciprocal_faces(cells: Any) -> Any:
    offsets = cells.cell_face_offsets
    owners = np.repeat(np.arange(cells.cell_count), np.diff(offsets))
    labels = cells.face_labels
    interior = labels >= 0
    forward = set(zip(owners[interior].tolist(), labels[interior].tolist(), strict=True))
    return all((other, owner) in forward for owner, other in forward)


def test_tetrahedron_dual_hexes_validates_and_preserves_shared_node_ids() -> None:
    nodes = np.arange(15, dtype=np.int64).reshape((1, 15))
    hexes = tetrahedron_dual_hexes(nodes, 15)
    np.testing.assert_array_equal(
        hexes,
        np.asarray(
            (
                (0, 4, 10, 5, 6, 11, 14, 12),
                (1, 4, 11, 8, 7, 10, 14, 13),
                (2, 5, 10, 7, 9, 12, 14, 13),
                (3, 6, 12, 9, 8, 11, 14, 13),
            ),
            dtype=np.int64,
        ),
    )
    with pytest.raises(ValueError, match="outside vertex_count"):
        tetrahedron_dual_hexes(nodes, 14)
    duplicate = nodes.copy()
    duplicate[0, 14] = duplicate[0, 13]
    with pytest.raises(ValueError, match="invalid input"):
        tetrahedron_dual_hexes(duplicate, 15)


def test_triangle_dual_quads_validates_and_preserves_shared_node_ids() -> None:
    nodes = np.arange(7, dtype=np.int64).reshape((1, 7))
    quads = triangle_dual_quads(nodes, 7)
    np.testing.assert_array_equal(
        quads,
        np.asarray(((0, 3, 6, 5), (1, 4, 6, 3), (2, 5, 6, 4)), dtype=np.int64),
    )
    with pytest.raises(ValueError, match="outside vertex_count"):
        triangle_dual_quads(nodes, 6)
    duplicate = nodes.copy()
    duplicate[0, 6] = duplicate[0, 5]
    with pytest.raises(ValueError, match="invalid input"):
        triangle_dual_quads(duplicate, 7)


def _declared_unit_tet(
    plane: np.ndarray, tolerance: float, initial_vertex_sizes: object | None = None
) -> TetMesh3D:
    # Face row 3 (z = 0) declares `plane`; the other rows declare none.
    faces = np.asarray(((1, 3, 2), (0, 2, 3), (0, 3, 1), (0, 1, 2)), dtype=np.int32)
    face_ids = np.asarray((101, 102, 103, 104), dtype=np.int64)
    planes = np.full((4, 3, 3), np.nan, dtype=np.float64)
    planes[3] = plane
    source = TetMeshSourceComplex(
        UNIT_TET.copy(),
        faces.copy(),
        np.arange(4, dtype=np.int32),
        np.zeros((4,), dtype=np.float64),
        np.empty((0, 2), dtype=np.int32),
        np.empty((0,), dtype=np.int32),
        np.empty((0,), dtype=np.float64),
        np.zeros((4,), dtype=np.int8),
        np.full((4,), -1, dtype=np.int32),
        np.zeros((4, 2), dtype=np.float64),
        face_ids,
        np.empty((0,), dtype=np.int64),
        planes,
        np.asarray((0.0, 0.0, 0.0, tolerance), dtype=np.float64),
    )
    return TetMesh3D(
        UNIT_TET.copy(),
        np.asarray(((0, 1, 2, 3),), dtype=np.int32),
        None,
        faces,
        face_ids,
        np.empty((0, 2), dtype=np.int32),
        np.empty((0,), dtype=np.int64),
        boundary_policy=PlcBoundaryPolicy.CONFORMING,
        initial_vertex_sizes=initial_vertex_sizes,
        source=source,
    )


def test_tet_candidate_quality_rechecks_geometry_and_retained_sizes() -> None:
    plane = np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
    with _declared_unit_tet(plane, 0.0) as state:
        state.refine(radius_edge_bound=10.0, sizes=np.full((4,), 2.0))
        snapshot = state.arrays()
        points, vertex_ids = snapshot.points, snapshot.vertex_ids
        current = state.evaluate_quality(radius_edge_bound=10.0, min_dihedral_degrees=0.0)
        candidate = state.evaluate_candidate_quality(
            points,
            vertex_ids=vertex_ids,
            radius_edge_bound=10.0,
            min_dihedral_degrees=0.0,
        )
        np.testing.assert_array_equal(candidate.criteria, current.criteria)
        np.testing.assert_array_equal(candidate.reasons, current.reasons)
        np.testing.assert_array_equal(candidate.values, current.values)
        stretched = points.copy()
        stretched[3, 2] = 10.0
        larger = state.evaluate_candidate_quality(
            stretched,
            vertex_ids=vertex_ids,
            radius_edge_bound=10.0,
            min_dihedral_degrees=0.0,
        )
        assert 1 in larger.criteria  # Retained SIZE target.
        inverted = points.copy()
        inverted[3, 2] = -1.0
        invalid = state.evaluate_candidate_quality(
            inverted,
            vertex_ids=vertex_ids,
            radius_edge_bound=10.0,
            min_dihedral_degrees=0.0,
        )
        assert np.any((invalid.criteria == 4) & (invalid.reasons == 2))
        assert (
            len({candidate.candidate_id, larger.candidate_id, invalid.candidate_id}) == 3
        )
        assert candidate.historical_identity == state.identity
        np.testing.assert_array_equal(state.arrays().points, points)
        with pytest.raises(ValueError, match="vertex IDs"):
            state.evaluate_candidate_quality(
                points,
                vertex_ids=vertex_ids[::-1],
                radius_edge_bound=10.0,
                min_dihedral_degrees=0.0,
            )
        with pytest.raises(ValueError, match="align"):
            state.evaluate_candidate_quality(
                points[:-1],
                vertex_ids=vertex_ids,
                radius_edge_bound=10.0,
                min_dihedral_degrees=0.0,
            )
        nonfinite = points.copy()
        nonfinite[0, 0] = np.nan
        with pytest.raises(ValueError, match="finite"):
            state.evaluate_candidate_quality(
                nonfinite,
                vertex_ids=vertex_ids,
                radius_edge_bound=10.0,
                min_dihedral_degrees=0.0,
            )


def test_tet_quality_can_omit_radius_without_omitting_validity() -> None:
    plane = np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
    with _declared_unit_tet(plane, 0.0) as state:
        points = state.arrays().points
        identifiers = state.arrays().vertex_ids
        quality = state.evaluate_quality(radius_edge_bound=None, min_dihedral_degrees=0.0)
        assert not np.any(quality.criteria == 0)
        candidate = state.evaluate_candidate_quality(
            points,
            vertex_ids=identifiers,
            radius_edge_bound=None,
            min_dihedral_degrees=0.0,
        )
        np.testing.assert_array_equal(candidate.criteria, quality.criteria)
        bounded = state.evaluate_candidate_quality(
            points,
            vertex_ids=identifiers,
            radius_edge_bound=2.0,
            min_dihedral_degrees=0.0,
        )
        assert bounded.candidate_id != candidate.candidate_id
        sliver = points.copy()
        sliver[3, 2] = 0.01
        angle_quality = state.evaluate_candidate_quality(
            sliver,
            vertex_ids=identifiers,
            radius_edge_bound=None,
            min_dihedral_degrees=10.0,
        )
        assert 2 in angle_quality.criteria
        assert not np.any(angle_quality.criteria == 0)
        inverted = points.copy()
        inverted[3, 2] = -1.0
        invalid = state.evaluate_candidate_quality(
            inverted,
            vertex_ids=identifiers,
            radius_edge_bound=None,
            min_dihedral_degrees=0.0,
        )
        assert np.any((invalid.criteria == 4) & (invalid.reasons == 2))
        assert candidate.candidate_id != invalid.candidate_id
        with pytest.raises(ValueError, match="radius_edge_bound"):
            state.evaluate_quality(radius_edge_bound=0.0, min_dihedral_degrees=0.0)
        with pytest.raises(ValueError, match="radius_edge_bound"):
            state.evaluate_candidate_quality(
                points,
                vertex_ids=identifiers,
                radius_edge_bound=0.0,
                min_dihedral_degrees=0.0,
            )
        with pytest.raises(ValueError, match="radius_edge_bound"):
            state.refine(radius_edge_bound=0.0)
        with pytest.raises(ValueError, match="radius_edge_bound"):
            state.improve(radius_edge_bound=0.0, min_dihedral_degrees=0.0)
        run = state.refine(radius_edge_bound=None, sizes=np.full((4,), 2.0))
        assert run.edit_status is TetMeshEditStatus.APPLIED
        updated = state.arrays()
        stretched = updated.points.copy()
        stretched[3, 2] = 10.0
        sized = state.evaluate_candidate_quality(
            stretched,
            vertex_ids=updated.vertex_ids,
            radius_edge_bound=None,
            min_dihedral_degrees=0.0,
        )
        assert 1 in sized.criteria
        improvement = state.improve(radius_edge_bound=None, min_dihedral_degrees=0.0)
        assert improvement.edit_status is TetMeshEditStatus.APPLIED


def test_tet_mesh_retains_and_restores_initial_vertex_sizes() -> None:
    plane = np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
    sizes = np.asarray((0.0, 0.25, 0.5, 1.0))
    with _declared_unit_tet(plane, 0.0, sizes) as state:
        arrays = state.arrays()
        np.testing.assert_array_equal(arrays.vertex_size_targets, sizes)
        with _restored_tet_mesh(arrays, arrays.witness_parameters) as restored:
            np.testing.assert_array_equal(restored.arrays().vertex_size_targets, sizes)
        with pytest.raises(ValueError, match="original vertex"):
            state.improve(
                radius_edge_bound=2.0,
                min_dihedral_degrees=10.0,
                boundary_collapse_quality=True,
            )
    for bad in ([-1.0, 0.0, 0.0, 0.0], [float("nan")] * 4, [1.0]):
        with pytest.raises(ValueError, match="initial_vertex_sizes"):
            _declared_unit_tet(plane, 0.0, bad)


def test_plc_selective_facet_recovery_validates_and_preserves_protected_face() -> None:
    offsets = np.arange(0, 13, 3, dtype=np.int64)
    polygons = np.asarray(((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)))
    facets = np.arange(4, dtype=np.int64)
    regions = np.tile(np.asarray((-1, 1)), (4, 1))
    with pytest.raises(ValueError, match="one boolean flag per facet"):
        recover_plc_3d(
            UNIT_TET,
            offsets,
            polygons.reshape(-1),
            facets,
            regions,
            protected_facets=np.asarray((True, False)),
        )
    result = recover_plc_3d(
        UNIT_TET,
        offsets,
        polygons.reshape(-1),
        facets,
        regions,
        protected_facets=np.asarray((False, False, False, True)),
        affine_metric=np.eye(3),
    )
    assert result.status is MeshcoreStatus.OK
    source_rows = result.input_polygons[result.face_sources]
    selected = result.faces[source_rows == 3]
    assert selected.shape == (1, 3)
    np.testing.assert_array_equal(np.sort(selected[0]), np.sort(polygons[3]))


def test_tet_mesh_selectively_protects_source_facet_groups() -> None:
    plane = np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
    with _declared_unit_tet(plane, 0.0) as state:
        before = state.arrays()
        with pytest.raises(ValueError):
            state.protect_facet_group_rows(np.asarray((99,), dtype=np.int32))
        assert state.arrays().generation == before.generation
        state.protect_facet_group_rows(np.asarray((3,), dtype=np.int32))
        protected = state.arrays()
        np.testing.assert_array_equal(
            protected.protected_facets, protected.face_sources == 104
        )
        assert np.count_nonzero(protected.protected_vertices) == 3
        state.refine(radius_edge_bound=2.0, sizes=np.full((4,), 0.2))
        refined = state.arrays()
        selected = refined.face_sources == 104
        assert np.count_nonzero(selected) == 1
        assert refined.protected_facets[selected].all()
        np.testing.assert_array_equal(
            np.sort(refined.vertex_ids[refined.faces[selected][0]]),
            np.sort(before.vertex_ids[before.faces[before.face_sources == 104][0]]),
        )


def test_declared_facet_plane_admits_records_and_refuses_boundary_vertices() -> None:
    lifted = np.asarray(((0.0, 0.0, 1e-3), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
    # A plane 1e-3 above the face misses every corner's rounding box: the
    # first corner of the face refuses it with its record as evidence.
    with pytest.raises(TetMeshPlaneRefusal) as refusal:
        _declared_unit_tet(lifted, 0.0)
    assert (refusal.value.vertex, refusal.value.face_row) == (0, 3)
    assert refusal.value.parameters == (0.0, 0.0)
    assert 1e-3 <= refusal.value.deviation <= 1e-3 * (1.0 + 1e-12)
    assert refusal.value.tolerance == 0.0
    # The declared tolerance admits the corners instead.
    state = _declared_unit_tet(lifted, 2e-3)
    try:
        records = state.plane_records()
        np.testing.assert_array_equal(records.vertex_ids, (0, 1, 2))
        np.testing.assert_array_equal(records.face_rows, (3, 3, 3))
        np.testing.assert_array_equal(
            records.parameters, ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
        )
        assert np.all(records.deviations >= 1e-3)
        arrays = state.arrays()
        np.testing.assert_array_equal(arrays.source_face_planes[3], lifted)
        assert np.all(np.isnan(arrays.source_face_planes[:3]))
        np.testing.assert_array_equal(
            arrays.source_face_plane_tolerances, (0, 0, 0, 2e-3)
        )
    finally:
        state.close()
    # Refinement splits the declared face on its exact plane (stratum 3).
    exact = np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
    state = _declared_unit_tet(exact, 0.0)
    try:
        run = state.refine(radius_edge_bound=2.0, sizes=np.full((4,), 0.2))
        assert run.status is MeshcoreStatus.OK
        arrays = state.arrays()
        on_plane = arrays.witness_strata == 3
        assert np.count_nonzero(on_plane) > 0
        np.testing.assert_array_equal(arrays.witness_entities[on_plane], 3)
        np.testing.assert_array_equal(arrays.points[on_plane, 2], 0.0)
        np.testing.assert_array_equal(
            arrays.points[on_plane, :2], arrays.witness_parameters[on_plane]
        )
        # Undeclared faces keep facet witnesses on their source rows.
        facet_points = arrays.witness_strata == 2
        assert np.count_nonzero(facet_points) > 0
        assert not np.any(arrays.witness_entities[facet_points] == 3)
        records = state.plane_records()
        assert np.all(records.deviations == 0.0)
    finally:
        state.close()


def _restored_tet_mesh(
    arrays: TetMeshArrays, witness_parameters: np.ndarray
) -> TetMesh3D:
    return TetMesh3D(
        arrays.points,
        arrays.tetrahedra,
        arrays.tetrahedron_regions,
        arrays.faces,
        arrays.face_sources,
        arrays.segments,
        arrays.segment_sources,
        boundary_policy=PlcBoundaryPolicy.CONFORMING,
        protection_radii=arrays.protection_radii,
        protected_vertices=arrays.protected_vertices,
        initial_vertex_sizes=arrays.vertex_size_targets,
        vertex_ids=arrays.vertex_ids,
        cell_ids=arrays.cell_ids,
        facet_ids=arrays.facet_ids,
        face_orientations=arrays.facet_orientations,
        segment_ids=arrays.segment_ids,
        source=TetMeshSourceComplex(
            arrays.source_points,
            arrays.source_faces,
            arrays.source_face_group_rows,
            arrays.source_face_tolerances,
            arrays.source_segments,
            arrays.source_segment_group_rows,
            arrays.source_segment_tolerances,
            arrays.witness_strata,
            arrays.witness_entities,
            witness_parameters,
            arrays.source_face_ids,
            arrays.source_segment_ids,
            arrays.source_face_planes,
            arrays.source_face_plane_tolerances,
        ),
        implicit_source=None
        if arrays.implicit_source is None
        else arrays.implicit_source.source,
        source_id=arrays.source_id,
        source_revision=arrays.source_revision,
        source_map_id=arrays.source_map_id,
        frame_id=arrays.frame_id,
    )


def test_implicit_tet_source_round_trips_owned_program_and_refuses_tamper() -> None:
    value = IntervalBoxProgram(
        np.asarray([[2, 4, 1, 2, -1, 3, -1, 0]], dtype=np.int64),
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.float64),
        np.zeros(5),
        np.zeros(5),
        np.asarray([4]),
        np.empty(0, dtype=np.int64),
    )
    gradient = IntervalBoxProgram(
        np.empty((0, 8), dtype=np.int64),
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.float64),
        np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        np.asarray([3, 4, 5]),
        np.empty(0, dtype=np.int64),
    )
    source = TetMeshImplicitSource(
        value,
        gradient,
        np.asarray([3], dtype=np.int32),
        0.01,
        0,
        "implicit",
        "revision",
        "binding",
        "world",
    )
    faces = np.asarray(((1, 3, 2), (0, 2, 3), (0, 3, 1), (0, 1, 2)), dtype=np.int32)
    with TetMesh3D(
        UNIT_TET,
        np.asarray([[0, 1, 2, 3]]),
        None,
        faces,
        np.arange(4),
        np.empty((0, 2), dtype=np.int32),
        np.empty(0, dtype=np.int32),
        boundary_policy=PlcBoundaryPolicy.CONFORMING,
        source_id="implicit",
        source_revision="revision",
        source_map_id="binding",
        frame_id="world",
        implicit_source=source,
    ) as state:
        value.instructions[0, 0] = 99
        arrays = state.arrays()
        assert arrays.implicit_source is not None
        assert arrays.implicit_source.source.value.instructions[0, 0] == 2
        assert arrays.implicit_source.source.orientation == 1
    with _restored_tet_mesh(arrays, arrays.witness_parameters) as restored:
        restored_source = restored.arrays().implicit_source
        assert restored_source is not None
        assert restored_source.source.binding_id == "binding"
        np.testing.assert_array_equal(restored_source.source.group_rows, [3])
    bad = arrays.implicit_source.source._replace(orientation=-1)
    tampered = arrays._replace(
        implicit_source=arrays.implicit_source._replace(source=bad)
    )
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        _restored_tet_mesh(tampered, tampered.witness_parameters)
    bad_frame = arrays.implicit_source.source._replace(frame_id="wrong")
    tampered = arrays._replace(
        implicit_source=arrays.implicit_source._replace(source=bad_frame)
    )
    with pytest.raises(ValueError, match="identity and frame"):
        _restored_tet_mesh(tampered, tampered.witness_parameters)
    program = arrays.implicit_source.source.value
    low, high = program.workspace_lower.copy(), program.workspace_upper.copy()
    low[3] = high[3] = 1.0
    off_band = arrays.implicit_source.source._replace(
        value=program._replace(workspace_lower=low, workspace_upper=high)
    )
    tampered = arrays._replace(
        implicit_source=arrays.implicit_source._replace(source=off_band)
    )
    with pytest.raises(MeshcoreError, match="UNRESOLVED"):
        _restored_tet_mesh(tampered, tampered.witness_parameters)


def test_plane_witnesses_and_records_round_trip_through_tet_mesh_arrays() -> None:
    exact = np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
    state = _declared_unit_tet(exact, 0.0)
    try:
        state.refine(radius_edge_bound=2.0, sizes=np.full((4,), 0.2))
        arrays = state.arrays()
        records = state.plane_records()
    finally:
        state.close()
    on_plane = arrays.witness_strata == 3
    assert np.count_nonzero(on_plane) > 0
    # A new mesh from the copied banks re-derives every plane witness
    # deviation and the plane records from the declared plane alone.
    restored = _restored_tet_mesh(arrays, arrays.witness_parameters)
    try:
        again = restored.arrays()
        for name in (
            "points",
            "witness_strata",
            "witness_entities",
            "witness_parameters",
            "witness_deviations",
        ):
            np.testing.assert_array_equal(getattr(again, name), getattr(arrays, name))
        np.testing.assert_array_equal(again.source_face_planes, arrays.source_face_planes)
        restored_records = restored.plane_records()
        for name in records._fields:
            np.testing.assert_array_equal(
                getattr(restored_records, name), getattr(records, name)
            )
    finally:
        restored.close()
    # A plane witness moved off its carrier's plane point is refused.
    tampered = arrays.witness_parameters.copy()
    tampered[np.flatnonzero(on_plane)[0], 0] += 1e-3
    with pytest.raises(ValueError, match="tet_mesh_create: invalid input"):
        _restored_tet_mesh(arrays, tampered)


def test_persistent_tet_mesh_owns_source_and_exposes_runs() -> None:
    mesh_points = UNIT_TET.copy()
    source_points = UNIT_TET.copy()
    faces = np.asarray(((1, 3, 2), (0, 2, 3), (0, 3, 1), (0, 1, 2)), dtype=np.int32)
    source_faces = faces.copy()
    segments = np.asarray(((0, 1),), dtype=np.int32)
    source_segments = segments.copy()
    face_ids = np.asarray((101, 102, 103, 104), dtype=np.int64)
    segment_ids = np.asarray((201,), dtype=np.int64)
    source = TetMeshSourceComplex(
        source_points,
        source_faces,
        np.arange(4, dtype=np.int32),
        np.asarray((0.1, 0.2, 0.3, 0.4), dtype=np.float64),
        source_segments,
        np.asarray((0,), dtype=np.int32),
        np.asarray((0.125,), dtype=np.float64),
        np.zeros((4,), dtype=np.int8),
        np.full((4,), -1, dtype=np.int32),
        np.zeros((4, 2), dtype=np.float64),
        face_ids,
        segment_ids,
        np.full((4, 3, 3), np.nan, dtype=np.float64),
        np.zeros((4,), dtype=np.float64),
    )
    state = TetMesh3D(
        mesh_points,
        np.asarray(((0, 1, 2, 3),), dtype=np.int32),
        None,
        faces,
        face_ids,
        segments,
        segment_ids,
        boundary_policy=PlcBoundaryPolicy.CONFORMING,
        source=source,
        source_id="persistent-test-source",
        source_revision="revision-7",
        source_map_id="persistent-test-source-map",
        frame_id="frame-11",
    )
    try:
        mesh_points[1, 0] = 8.0
        source_points[1, 0] = 9.0
        source_faces[0, 0] = 0
        source_segments[0, 1] = 2

        arrays = state.arrays()
        np.testing.assert_array_equal(arrays.points, UNIT_TET)
        assert isinstance(arrays, TetMeshArrays)
        np.testing.assert_array_equal(arrays.source_points, UNIT_TET)
        np.testing.assert_array_equal(
            arrays.source_faces,
            np.asarray(((1, 3, 2), (0, 2, 3), (0, 3, 1), (0, 1, 2)), dtype=np.int32),
        )
        np.testing.assert_array_equal(arrays.source_segment_ids, segment_ids)
        np.testing.assert_array_equal(arrays.source_segment_tolerances, (0.125,))

        assert arrays.vertex_dimensions.shape == arrays.protection_radii.shape == (4,)
        assert arrays.protected_vertices.shape == (4,)
        assert arrays.facet_orientations.shape == arrays.protected_facets.shape == (4,)
        assert arrays.protected_segments.shape == (1,)
        assert (
            arrays.source_id,
            arrays.source_revision,
            arrays.source_map_id,
            arrays.frame_id,
            arrays.generation,
        ) == (
            "persistent-test-source",
            "revision-7",
            "persistent-test-source-map",
            "frame-11",
            0,
        )

        evidence = state.source_evidence()
        assert evidence.requested_bound == 0.4
        assert evidence.achieved_bound == 0.0
        assert isinstance(evidence, TetMeshSourceEvidence)

        identity = state.identity
        before_unmet = state.unmet()
        quality = state.evaluate_quality(
            radius_edge_bound=0.5,
            min_dihedral_degrees=0.0,
            minimum_relative_determinant=0.0,
        )
        assert quality.identity == identity
        assert quality.cell_ids.size > 0
        assert state.identity == identity
        np.testing.assert_array_equal(state.unmet().cell_ids, before_unmet.cell_ids)
        np.testing.assert_array_equal(state.arrays().tetrahedra, arrays.tetrahedra)

        stale = state.apply_edit(
            TetMeshEdit(
                TetMeshEditKind.RECONNECT, expected_generation=arrays.generation + 1
            )
        )
        assert stale is TetMeshEditStatus.STALE_GENERATION
        assert state.generation == arrays.generation

        rejected = state.apply_edit(
            TetMeshEdit(
                TetMeshEditKind.RECONNECT,
                expected_generation=arrays.generation,
                removed_cell_ids=arrays.cell_ids[:1],
                replacement_cell_ids=np.asarray((1,), dtype=np.uint64),
                replacement_cell_vertices=np.asarray(((0, 0, 1, 2),), dtype=np.uint64),
                replacement_facet_ids=np.asarray((4,), dtype=np.uint64),
                replacement_facet_vertices=np.asarray(((0, 1, 2),), dtype=np.uint64),
                replacement_facet_sources=np.asarray((0,), dtype=np.int32),
            )
        )
        assert rejected is not TetMeshEditStatus.APPLIED
        assert state.generation == arrays.generation

        refinement = state.refine(radius_edge_bound=10.0, sizes=np.full((4,), 0.1))
        assert refinement.edit_status is TetMeshEditStatus.APPLIED
        # The size field is met with every cell within bound: facet and segment
        # splits refine the conforming boundary despite the acute input angles.
        assert refinement.status is MeshcoreStatus.OK
        assert refinement.inserted_vertices > 0
        assert refinement.unmet_cells == 0
        assert refinement.score_quantum is None
        assert state.unmet().cell_ids.size == 0
        with pytest.raises(ValueError, match="one value per current mesh vertex"):
            state.improve(
                radius_edge_bound=10.0,
                min_dihedral_degrees=0.0,
                sizes=np.full((state.arrays().points.shape[0] + 1,), 0.1),
            )
        # No `sizes`: refinement's retained size field constrains every edit.
        improvement = state.improve(radius_edge_bound=10.0, min_dihedral_degrees=0.0)
        assert improvement.edit_status is TetMeshEditStatus.APPLIED
        # The acceptance grid of the improvement scores is reported as evidence.
        assert improvement.score_quantum == 2.0**-20
        # Success exactly when no cell remains unmet; records describe each residual.
        assert (improvement.status is MeshcoreStatus.OK) == (improvement.unmet_cells == 0)
        if improvement.status is not MeshcoreStatus.OK:
            assert improvement.status is MeshcoreStatus.UNRESOLVED

        unmet = state.unmet()
        assert (
            unmet.cell_ids.shape
            == unmet.criteria.shape
            == unmet.reasons.shape
            == unmet.values.shape
        )
        assert (unmet.cell_ids.size == 0) == (improvement.unmet_cells == 0)
        # Improvement records its own targets only; the size field constrains its
        # edits and its residuals are refinement's records.
        assert np.all(unmet.criteria == 0)  # PHX_MC_TET_UNMET_RADIUS_EDGE
        assert np.all(unmet.values > 10.0)
        assert np.all(np.isfinite(unmet.values))

    finally:
        state.close()
    with pytest.raises(RuntimeError, match="closed"):
        state.arrays()


def test_tet_mesh_relocation_distinguishes_omitted_and_cleared_edge_witness() -> None:
    root = Path(__file__).resolve().parents[3]
    environment = os.environ.copy()

    script = """\
import numpy as np
from phydrax._meshcore import (
    PlcBoundaryPolicy,
    TetMesh3D,
    TetMeshEdit,
    TetMeshEditKind,
    TetMeshEditStatus,
    TetMeshSourceComplex,
)

SOURCE_POINTS = np.asarray(
    (
        (0., 0., 0.),
        (0., 0., 1.),
        (1., 0., 0.5),
        (-0.5, np.sqrt(3.) / 2., 0.5),
        (-0.5, -np.sqrt(3.) / 2., 0.5),
    )
)
SOURCE_TETRAHEDRA = np.asarray(
    ((0, 1, 2, 3), (0, 1, 3, 4), (0, 1, 4, 2)), dtype=np.int32
)
LOCAL_FACE_CORNERS = ((1, 3, 2), (0, 2, 3), (0, 3, 1), (0, 1, 2))
SOURCE_VERTEX_IDS = np.asarray((10, 20, 30, 40, 50), dtype=np.uint64)


def source_boundary_faces():
    face_uses = {}
    for tetrahedron in SOURCE_TETRAHEDRA.tolist():
        for local_face in LOCAL_FACE_CORNERS:
            face = tuple(int(tetrahedron[index]) for index in local_face)
            face_uses.setdefault(tuple(sorted(face)), []).append(face)
    return np.asarray(
        [uses[0] for uses in face_uses.values() if len(uses) == 1], dtype=np.int32
    )


SOURCE_FACES = source_boundary_faces()
SOURCE_FACE_IDS = np.arange(101, 101 + SOURCE_FACES.shape[0], dtype=np.int64)


def make_state():
    source = TetMeshSourceComplex(
        SOURCE_POINTS.copy(),
        SOURCE_FACES.copy(),
        np.arange(SOURCE_FACES.shape[0], dtype=np.int32),
        np.zeros((SOURCE_FACES.shape[0],), dtype=np.float64),
        np.asarray(((0, 1),), dtype=np.int32),
        np.asarray((0,), dtype=np.int32),
        np.asarray((0.,), dtype=np.float64),
        np.zeros((SOURCE_POINTS.shape[0],), dtype=np.int8),
        np.full((SOURCE_POINTS.shape[0],), -1, dtype=np.int32),
        np.zeros((SOURCE_POINTS.shape[0], 2), dtype=np.float64),
        SOURCE_FACE_IDS,
        np.asarray((201,), dtype=np.int64),
        np.full((SOURCE_FACES.shape[0], 3, 3), np.nan, dtype=np.float64),
        np.zeros((SOURCE_FACES.shape[0],), dtype=np.float64),
    )
    return TetMesh3D(
        SOURCE_POINTS,
        SOURCE_TETRAHEDRA,
        None,
        SOURCE_FACES,
        SOURCE_FACE_IDS,
        np.asarray(((0, 1),), dtype=np.int32),
        np.asarray((201,), dtype=np.int64),
        boundary_policy=PlcBoundaryPolicy.CONFORMING,
        vertex_ids=SOURCE_VERTEX_IDS,
        cell_ids=np.asarray((100, 101, 102), dtype=np.uint64),
        facet_ids=np.arange(400, 400 + SOURCE_FACES.shape[0], dtype=np.uint64),
        segment_ids=np.asarray((300,), dtype=np.uint64),
        source=source,
        source_id="edge-witness-source",
        source_revision="revision-1",
        source_map_id="edge-witness-map",
        frame_id="edge-witness-frame",
    )


def split_source_edge(state):
    source_vertex_ids = np.asarray((10, 20, 30, 40, 50, 60), dtype=np.uint64)
    replacement_cells = np.asarray(
        (
            (0, 5, 2, 3),
            (5, 1, 2, 3),
            (0, 5, 3, 4),
            (5, 1, 3, 4),
            (0, 5, 4, 2),
            (5, 1, 4, 2),
        ),
        dtype=np.int32,
    )
    local_faces = LOCAL_FACE_CORNERS
    boundary_sources = {
        tuple(sorted(int(vertex) for vertex in face)): row
        for row, face in enumerate(SOURCE_FACES.tolist())
    }
    replacement_cell_facet_sources = np.asarray(
        [
            [
                boundary_sources.get(
                    tuple(sorted(int(tetrahedron[index]) for index in face)), -1
                )
                for face in local_faces
            ]
            for tetrahedron in replacement_cells.tolist()
        ],
        dtype=np.int32,
    )
    replacement_cell_vertices = source_vertex_ids[replacement_cells]
    status = state.apply_edit(
        TetMeshEdit(
            TetMeshEditKind.SPLIT,
            expected_generation=0,
            split_edges=np.asarray(((10, 20),), dtype=np.uint64),
            removed_cell_ids=np.asarray((100, 101, 102), dtype=np.uint64),
            changed_vertex_ids=np.asarray((60,), dtype=np.uint64),
            changed_vertex_points=np.asarray(((0., 0., 0.5),)),
            changed_witness_strata=np.asarray((1,), dtype=np.int8),
            changed_witness_entities=np.asarray((0,), dtype=np.int32),
            changed_witness_parameters=np.asarray(((0.5, 0.),)),
            changed_witness_deviations=np.asarray((0.,)),
            changed_protection_radii=np.asarray((0.,)),
            changed_protected_vertices=np.asarray((0,), dtype=np.int8),
            changed_edge_witness_present=np.asarray((1,), dtype=np.int8),
            changed_edge_witness_endpoints=np.asarray(((10, 20),), dtype=np.uint64),
            changed_edge_parameters=np.asarray((0.5,)),
            replacement_cell_ids=np.arange(200, 206, dtype=np.uint64),
            replacement_cell_vertices=replacement_cell_vertices,
            replacement_cell_facet_sources=replacement_cell_facet_sources,
            replacement_cell_ancestry_offsets=np.arange(7, dtype=np.int64),
            replacement_cell_ancestor_ids=np.repeat(
                np.asarray((100, 101, 102), dtype=np.uint64), 2
            ),
            removed_segment_ids=np.asarray((300,), dtype=np.uint64),
            replacement_segment_ids=np.asarray((301, 302), dtype=np.uint64),
            replacement_segment_vertices=np.asarray(
                ((10, 60), (60, 20)), dtype=np.uint64
            ),
            replacement_segment_sources=np.asarray((0, 0), dtype=np.int32),
        )
    )
    assert status is TetMeshEditStatus.APPLIED, status
    arrays = state.arrays()
    row = int(np.flatnonzero(arrays.vertex_ids == 60)[0])
    assert arrays.vertex_edge_witness_present[row] == 1
    np.testing.assert_array_equal(arrays.vertex_edge_witness_endpoints[row], (10, 20))
    assert arrays.vertex_edge_parameters[row] == 0.5

def relocate(state, /, *, clear_edge_witness):
    before = state.arrays()
    edge_witness = {}
    if clear_edge_witness:
        edge_witness = {
            "changed_edge_witness_present": np.asarray((0,), dtype=np.int8),
            "changed_edge_witness_endpoints": np.zeros((1, 2), dtype=np.uint64),
            "changed_edge_parameters": np.zeros((1,), dtype=np.float64),
        }
    status = state.apply_edit(
        TetMeshEdit(
            TetMeshEditKind.RELOCATE,
            expected_generation=before.generation,
            removed_cell_ids=np.arange(200, 206, dtype=np.uint64),
            changed_vertex_ids=np.asarray((60,), dtype=np.uint64),
            changed_vertex_points=np.asarray(((0., 0., 0.25),)),
            changed_witness_strata=np.asarray((1,), dtype=np.int8),
            changed_witness_entities=np.asarray((0,), dtype=np.int32),
            changed_witness_parameters=np.asarray(((0.25, 0.),)),
            changed_witness_deviations=np.asarray((0.,)),
            changed_protection_radii=np.asarray((0.,)),
            changed_protected_vertices=np.asarray((0,), dtype=np.int8),
            **edge_witness,
        )
    )
    after = state.arrays()
    row = int(np.flatnonzero(after.vertex_ids == 60)[0])
    if clear_edge_witness:
        assert status is TetMeshEditStatus.APPLIED
        assert after.vertex_edge_witness_present[row] == 0
        np.testing.assert_array_equal(after.vertex_edge_witness_endpoints[row], (0, 0))
        assert after.vertex_edge_parameters[row] == 0.0
        np.testing.assert_allclose(after.points[row], (0., 0., 0.25))
        assert after.witness_strata[row] == 1
        assert after.witness_entities[row] == 0
        assert after.witness_parameters[row, 0] == 0.25
    else:
        assert after.vertex_edge_witness_present[row] == 1
        np.testing.assert_array_equal(after.vertex_edge_witness_endpoints[row], (10, 20))
        assert after.vertex_edge_parameters[row] == 0.5
        if status is not TetMeshEditStatus.APPLIED:
            assert after.generation == before.generation
            np.testing.assert_allclose(after.points[row], before.points[row])

for clear_edge_witness in (False, True):
    state = make_state()
    try:
        split_source_edge(state)
        relocate(state, clear_edge_witness=clear_edge_witness)
    finally:
        state.close()
print("edge-witness-relocation:ok")
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "edge-witness-relocation:ok" in completed.stdout


def test_tet_mesh_refine_binding_runs_in_isolated_process() -> None:
    root = Path(__file__).resolve().parents[3]
    environment = os.environ.copy()

    script = """\
import numpy as np
from phydrax._meshcore import (
    MeshcoreStatus,
    PlcBoundaryPolicy,
    TetMesh3D,
    TetMeshEditStatus,
    TetMeshSourceComplex,
)

points = np.asarray(((0., 0., 0.), (1., 0., 0.), (0., 1., 0.), (0., 0., 1.)))
faces = np.asarray(((1, 3, 2), (0, 2, 3), (0, 3, 1), (0, 1, 2)), dtype=np.int32)
segments = np.asarray(((0, 1),), dtype=np.int32)
face_ids = np.asarray((101, 102, 103, 104), dtype=np.int64)
segment_ids = np.asarray((201,), dtype=np.int64)
source = TetMeshSourceComplex(
    points.copy(),
    faces.copy(),
    np.arange(4, dtype=np.int32),
    np.asarray((0.1, 0.2, 0.3, 0.4), dtype=np.float64),
    segments.copy(),
    np.asarray((0,), dtype=np.int32),
    np.asarray((0.125,), dtype=np.float64),
    np.zeros((4,), dtype=np.int8),
    np.full((4,), -1, dtype=np.int32),
    np.zeros((4, 2), dtype=np.float64),
    face_ids,
    segment_ids,
    np.full((4, 3, 3), np.nan, dtype=np.float64),
    np.zeros((4,), dtype=np.float64),
)
state = TetMesh3D(
    points,
    np.asarray(((0, 1, 2, 3),), dtype=np.int32),
    None,
    faces,
    face_ids,
    segments,
    segment_ids,
    boundary_policy=PlcBoundaryPolicy.CONFORMING,
    source=source,
    source_id="isolated-refinement-source",
    source_revision="revision-7",
    source_map_id="isolated-refinement-source-map",
    frame_id="frame-11",
)
try:
    run = state.refine(radius_edge_bound=10.0, sizes=np.full((4,), 0.1))
    assert run.edit_status is TetMeshEditStatus.APPLIED
    assert run.status is MeshcoreStatus.OK
    assert run.inserted_vertices > 0
    assert run.unmet_cells == 0
    print(f"refine-result:{run.status.name}:{run.inserted_vertices}:{run.unmet_cells}")
finally:
    state.close()
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "refine-result:OK:" in completed.stdout


def test_fixed_refinement_next_to_thin_pole_fans_is_bounded() -> None:
    # A recovered FIXED unit-sphere PLC whose poles are fans of thin
    # triangles (ring edges ~2.6e-3 fanned to vertices 0.1-0.3 away) and whose
    # recovery left vertices 1.5e-4 apart. A spacing guard of the shortest
    # initial edge let refinement insert past 240k vertices (>3 GB); the local
    # feature-size guard bounds the insertions and records what stays unmet.
    data = np.load(
        Path(__file__).resolve().parents[2]
        / "data"
        / "meshing"
        / "thin_pole_fan_sphere.npz"
    )
    points = data["points"]
    tetrahedra = data["tetrahedra"]
    faces = data["faces"]
    segments = data["segments"]
    with TetMesh3D(
        points,
        tetrahedra,
        np.zeros((tetrahedra.shape[0],), dtype=np.int32),
        faces,
        np.zeros((faces.shape[0],), dtype=np.int32),
        segments,
        np.arange(segments.shape[0], dtype=np.int32),
        boundary_policy=PlcBoundaryPolicy.FIXED,
        face_orientations=data["face_orientations"],
    ) as state:
        run = state.refine(radius_edge_bound=2.0)
        assert run.edit_status is TetMeshEditStatus.APPLIED
        assert run.status is MeshcoreStatus.UNRESOLVED
        assert 0 < run.inserted_vertices <= 4 * points.shape[0]
        assert state.arrays().points.shape[0] == points.shape[0] + run.inserted_vertices
        assert run.unmet_cells == np.unique(np.asarray(state.unmet().cell_ids)).size


def test_fixed_cad_sphere_recovery_suppresses_its_splits_with_bounded_wraps() -> None:
    # A FIXED CAD unit-sphere PLC (640 points, 1276 triangles) whose recovery
    # leaves three split points to suppress through the volume. Each one's
    # star spans much of the ball (476 to 718 link walls) and has no
    # constrained Delaunay fill. Each wrap stops at its first stuck side
    # (34, 52 and 15 cells) before a later fill option succeeds; a wrap that
    # went on past its stuck sides laid 17,405 overlapping cells in the first
    # star alone and did not finish in 20 minutes.
    data = np.load(
        Path(__file__).resolve().parents[2] / "data" / "meshing" / "cad_sphere_plc.npz"
    )
    recovered = recover_plc_3d(
        data["points"],
        data["polygon_offsets"],
        data["polygon_vertices"],
        data["polygon_facets"],
        data["facet_regions"],
        segments=data["segments"],
        facet_tolerances=data["facet_tolerances"],
        segment_tolerances=data["segment_tolerances"],
        boundary_policy=PlcBoundaryPolicy.FIXED,
    )
    np.testing.assert_array_equal(recovered.points, data["points"])
    assert recovered.faces.shape[0] == data["polygon_facets"].shape[0]
    assert recovered.tetrahedra.shape[0] == 1329
    evidence = recovered.suppression_search
    assert (evidence.wraps, evidence.wrap_cells, evidence.max_wrap_cells) == (3, 101, 52)
    assert evidence.searches == 0
    assert evidence.max_wrap_cells < recovered.tetrahedra.shape[0]


def test_plc_recovery_declares_its_suppression_search_budget() -> None:
    # The exhaustive suppression search budget is part of the recovery
    # request: the documented default unless declared, echoed in the
    # evidence, positive int64 only, and refused for source preparation.
    points = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    )
    banks = (
        points,
        np.arange(0, 13, 3, dtype=np.int64),
        np.asarray((1, 2, 3, 0, 3, 2, 0, 1, 3, 0, 2, 1), dtype=np.int32),
        np.arange(4, dtype=np.int32),
        np.tile(np.asarray((-1, 1), dtype=np.int32), (4, 1)),
    )
    default = recover_plc_3d(*banks).suppression_search
    assert (default.node_budget, default.byte_budget) == PLC_SEARCH_BUDGET
    assert (default.searches, default.exceeded) == (0, 0)
    declared = recover_plc_3d(*banks, search_budget=PlcSearchBudget(5, 1 << 20))
    assert (
        declared.suppression_search.node_budget,
        declared.suppression_search.byte_budget,
    ) == (
        5,
        1 << 20,
    )
    for invalid in (
        PlcSearchBudget(0, 1),
        PlcSearchBudget(1, -1),
        PlcSearchBudget(True, 1),
    ):
        with pytest.raises(ValueError, match="search_budget"):
            recover_plc_3d(*banks, search_budget=invalid)
    with pytest.raises(ValueError, match="search_budget"):
        recover_plc_3d(*banks, search_budget=(1, 1))  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    "policy", [PlcBoundaryPolicy.FIXED, PlcBoundaryPolicy.CONFORMING]
)
def test_cylinder_crease_split_cavity_around_a_recovered_subface_recovers(
    policy: PlcBoundaryPolicy,
) -> None:
    # A CAD cylinder PLC (745 points, caps declared on their exact planes).
    # The split point of a lateral crease edge has a conflict cavity that
    # reaches both sides of a recovered subface on that edge; inserting its
    # cone erased the subface and failed after the topology write
    # (NONREPRESENTABLE POINT:764, INTERNAL_ERROR). The insertion is now
    # refused unchanged and the point goes into a star-shaped cavity.
    data = np.load(
        Path(__file__).resolve().parents[2]
        / "data"
        / "meshing"
        / "cylinder_declared_caps_plc.npz"
    )
    source_count = data["points"].shape[0]
    recovered = recover_plc_3d(
        data["points"],
        data["polygon_offsets"],
        data["polygon_vertices"],
        data["polygon_facets"],
        data["facet_regions"],
        segments=data["segments"],
        facet_tolerances=data["facet_tolerances"],
        segment_tolerances=data["segment_tolerances"],
        boundary_policy=policy,
        facet_planes=data["facet_planes"],
        facet_plane_tolerances=data["facet_plane_tolerances"],
    )
    assert recovered.status is MeshcoreStatus.OK
    np.testing.assert_array_equal(recovered.points[:source_count], data["points"])
    assert recovered.tetrahedra.shape[0] > 0
    if policy is PlcBoundaryPolicy.FIXED:
        assert recovered.faces.shape[0] == data["polygon_facets"].shape[0]
        assert not np.any(recovered.witness_strata[source_count:])


def _tilted_prism() -> tuple[np.ndarray, ...]:
    # A unit-square prism whose top facet lies on z = 1 + 0.1 x + 0.3 y, with
    # six correctly rounded top vertices fanned into six triangles that are
    # not exactly coplanar.
    def top(x: float, y: float) -> float:
        return float(1 + Fraction(0.1) * Fraction(x) + Fraction(0.3) * Fraction(y))

    corners = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    tops = (*corners, (0.5, 0.5), (0.3, 0.7))
    points = np.asarray(
        [(x, y, 0.0) for x, y in corners] + [(x, y, top(x, y)) for x, y in tops],
        dtype=np.float64,
    )
    polygons = (
        (0, 3, 2, 1),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
        (4, 5, 8),
        (5, 6, 8),
        (6, 7, 9),
        (7, 4, 9),
        (6, 9, 8),
        (4, 8, 9),
    )
    offsets = np.cumsum([0, *(len(polygon) for polygon in polygons)])
    vertices = np.concatenate([np.asarray(polygon) for polygon in polygons])
    facets = np.asarray((0, 1, 2, 3, 4, 5, 5, 5, 5, 5, 5))
    regions = np.tile(np.asarray((-1, 0)), (6, 1))
    planes = np.full((6, 3, 3), np.nan)
    planes[5] = ((0.0, 0.0, 1.0), (1.0, 0.0, 0.1), (0.0, 1.0, 0.3))
    return points, offsets, vertices, facets, regions, planes


def test_declared_facet_plane_groups_rounded_triangles_and_refuses_off_plane_vertices() -> (
    None
):
    points, offsets, vertices, facets, regions, planes = _tilted_prism()
    no_segments = np.empty((0, 2), dtype=np.int64)
    undeclared = prepare_plc_source(
        points, offsets, vertices, facets, regions, no_segments
    )
    declared = prepare_plc_source(
        points, offsets, vertices, facets, regions, no_segments, facet_planes=planes
    )
    # Rounding bends the top triangles, so only the declaration groups them:
    # its five internal diagonals stop being PLC edges.
    assert undeclared.plc_edges.shape[0] == 17
    assert declared.plc_edges.shape[0] == 12
    for policy in (PlcBoundaryPolicy.FIXED, PlcBoundaryPolicy.CONFORMING):
        recovered = recover_plc_3d(
            points,
            offsets,
            vertices,
            facets,
            regions,
            facet_planes=planes,
            boundary_policy=policy,
        )
        np.testing.assert_array_equal(recovered.points, points)
        assert recovered.faces.shape[0] == 16
        assert recovered.segments.shape[0] == 12

    lifted = points.copy()
    lifted[9, 2] += 1e-6
    with pytest.raises(PlcSourceFailure) as refusal:
        recover_plc_3d(lifted, offsets, vertices, facets, regions, facet_planes=planes)
    failure = refusal.value
    assert failure.reason is PlcFailureReason.SOURCE_DEVIATION
    assert (failure.first_kind, failure.first) == (PlcEntityKind.FACET, 5)
    assert (failure.second_kind, failure.second) == (PlcEntityKind.POINT, 9)
    normal = np.cross(planes[5, 1], planes[5, 2])
    distance = abs(np.dot(normal, lifted[9] - planes[5, 0])) / np.linalg.norm(normal)
    assert distance <= failure.certified_deviation <= distance * (1 + 1e-9)
    assert failure.declared_bound == 0.0
    tolerances = np.zeros((6,))
    tolerances[5] = 1e-5
    tolerated = recover_plc_3d(
        lifted,
        offsets,
        vertices,
        facets,
        regions,
        facet_planes=planes,
        facet_plane_tolerances=tolerances,
    )
    assert tolerated.segments.shape[0] == 12

    for row in (
        ((0.0, 0.0, 1.0), (1.0, 0.0, 0.1), (0.0, np.nan, 0.3)),
        ((0.0, 0.0, 1.0), (1.0, 0.0, 0.1), (2.0, 0.0, 0.2)),
    ):
        invalid = planes.copy()
        invalid[5] = row
        with pytest.raises(PlcSourceFailure) as refusal:
            prepare_plc_source(
                points,
                offsets,
                vertices,
                facets,
                regions,
                no_segments,
                facet_planes=invalid,
            )
        assert refusal.value.reason is PlcFailureReason.SOURCE_DEVIATION
        assert (refusal.value.first_kind, refusal.value.first) == (PlcEntityKind.FACET, 5)


def test_declared_planes_recover_a_rotated_cube_missing_its_face_diagonals() -> None:
    # Corners are the correct roundings of the exact corners of a rotated
    # cube of half side 1/8, and each face declares its exact plane (origin
    # +-h c_k, axes c_{k+1}, c_{k+2}; tolerance 0). The two triangles of a
    # face are not exactly coplanar and the Delaunay complex lacks some face
    # diagonals. Recovery used to split faces endlessly around such a
    # diagonal; it now recovers the diagonals as hidden edges, suppresses
    # every split point on them, and publishes neither them nor any point.
    c, s = np.cos(0.3), np.sin(0.3)
    columns = np.asarray(((c, -s, 0.0), (s * c, c * c, -s), (s * s, s * c, c))).T
    signs = np.asarray(
        [(x, y, z) for z in (-1, 1) for y in (-1, 1) for x in (-1, 1)], dtype=np.float64
    )
    points = np.asarray(
        [
            [
                float(
                    sum(Fraction(sign[k] / 8) * Fraction(columns[k][a]) for k in range(3))
                )
                for a in range(3)
            ]
            for sign in signs
        ]
    )
    sides = ((2, -1), (2, 1), (1, -1), (1, 1), (0, -1), (0, 1))
    planes = np.asarray(
        [
            (side / 8 * columns[k], columns[(k + 1) % 3], columns[(k + 2) % 3])
            for k, side in sides
        ]
    )
    triangles = np.asarray(
        (
            (0, 2, 3),
            (0, 3, 1),
            (4, 5, 7),
            (4, 7, 6),
            (0, 1, 5),
            (0, 5, 4),
            (2, 6, 7),
            (2, 7, 3),
            (0, 4, 6),
            (0, 6, 2),
            (1, 3, 7),
            (1, 7, 5),
        )
    )
    arguments = (
        points,
        np.arange(0, 37, 3),
        triangles.reshape(-1),
        np.repeat(np.arange(6), 2),
        np.tile(np.asarray((-1, 0)), (6, 1)),
    )
    for policy in (PlcBoundaryPolicy.FIXED, PlcBoundaryPolicy.CONFORMING):
        loose = recover_plc_3d(*arguments, boundary_policy=policy)
        declared = recover_plc_3d(*arguments, facet_planes=planes, boundary_policy=policy)
        assert loose.plc_edges.shape[0] == 18
        assert declared.plc_edges.shape[0] == 12
        assert declared.segments.shape[0] == 12
        np.testing.assert_array_equal(declared.points, points)
        assert declared.faces.shape[0] == 12


def test_surface_envelope_exposes_unsigned_distance_error_evidence() -> None:
    triangles = np.asarray(
        ((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)),
        dtype=np.int64,
    )
    result = surface_envelope(
        UNIT_TET,
        triangles,
        0.8,
        0.1,
        0.1,
    )

    assert result.surface_vertices.ndim == 2
    assert result.surface_vertices.shape[1] == 3
    assert result.surface_triangles.ndim == 2
    assert result.surface_triangles.shape[1] == 3
    assert result.volume_vertices.ndim == 2
    assert result.volume_vertices.shape[1] == 3
    assert result.tetrahedra.ndim == 2
    assert result.tetrahedra.shape[1] == 4
    assert result.bounds.shape == (4,)
    assert result.bounds[2] > 0.0
    assert result.bounds[3] < 0.8
    assert np.all(result.surface_triangles < result.surface_vertices.shape[0])
    assert np.all(result.tetrahedra < result.volume_vertices.shape[0])
    assert not result.bounds.flags.writeable


def _minimum_triangle_angles(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    corners = [vertices[triangles[:, k]] for k in range(3)]
    angles = []
    for k in range(3):
        first = corners[(k + 1) % 3] - corners[k]
        second = corners[(k + 2) % 3] - corners[k]
        cosine = np.sum(first * second, axis=1) / (
            np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
        )
        angles.append(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
    return np.min(np.stack(angles), axis=0)


def test_surface_envelope_remesh_keeps_the_band_and_raises_angles() -> None:
    triangles = np.asarray(
        ((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)),
        dtype=np.int64,
    )
    extracted = surface_envelope(UNIT_TET, triangles, 0.8, 0.1, 0.1)
    remeshed = surface_envelope(UNIT_TET, triangles, 0.8, 0.1, 0.1, remesh=True)

    assert extracted.remesh_counters is None
    assert remeshed.remesh_counters is not None
    assert remeshed.remesh_counters.shape == (5,)
    assert int(np.sum(remeshed.remesh_counters[:4])) > 0
    # The band and interpolation error are unchanged; only the source-to-surface
    # bound is recomputed on the remeshed vertices.
    np.testing.assert_array_equal(remeshed.bounds[1:], extracted.bounds[1:])
    assert 0.0 < remeshed.bounds[0] < remeshed.bounds[1]
    assert remeshed.tetrahedra.shape == (0, 4)
    np.testing.assert_array_equal(remeshed.volume_vertices, remeshed.surface_vertices)
    assert np.min(
        _minimum_triangle_angles(remeshed.surface_vertices, remeshed.surface_triangles)
    ) > np.min(
        _minimum_triangle_angles(extracted.surface_vertices, extracted.surface_triangles)
    )
    directed = {
        (int(face[k]), int(face[(k + 1) % 3]))
        for face in remeshed.surface_triangles
        for k in range(3)
    }
    assert len(directed) == 3 * remeshed.surface_triangles.shape[0]
    assert all((second, first) in directed for first, second in directed)
    again = surface_envelope(UNIT_TET, triangles, 0.8, 0.1, 0.1, remesh=True)
    np.testing.assert_array_equal(again.surface_vertices, remeshed.surface_vertices)
    np.testing.assert_array_equal(again.surface_triangles, remeshed.surface_triangles)


def test_surface_envelope_names_the_offset_domination_contract() -> None:
    triangles = np.asarray(
        ((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)),
        dtype=np.int64,
    )
    # sqrt(3) * 0.1 + 0.1 exceeds the offset 0.2: the kernel refuses with
    # RANGE_ERROR, and the message names the certified-error contract rather
    # than the coordinate domain.
    with pytest.raises(
        ValueError, match="does not dominate the certified envelope"
    ) as error:
        surface_envelope(UNIT_TET, triangles, 0.2, 0.1, 0.1)
    assert "2**-120" not in str(error.value)
