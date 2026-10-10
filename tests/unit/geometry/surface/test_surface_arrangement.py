#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import math
from fractions import Fraction

import numpy as np
import pytest

from phydrax._meshcore import (
    arrange_triangles,
    ARRANGEMENT_UNCLASSIFIED,
    ArrangementClassification,
    ArrangementFailure,
    IntersectionClass,
    meshcore_available,
    MeshcoreStatus,
    NativeExecutionBudget,
    triangle_intersection_classes,
)
from phydrax.geometry.surface import (
    arrange_triangle_surfaces,
    SurfaceArrangement,
    SurfaceArrangementError,
    SurfaceArrangementLimits,
)


pytestmark = [
    pytest.mark.meshcore,
    pytest.mark.skipif(
        not meshcore_available(),
        reason="native phydrax-meshcore library is not available",
    ),
]

_LEGAL = (
    IntersectionClass.DISJOINT,
    IntersectionClass.SHARED_VERTEX,
    IntersectionClass.SHARED_EDGE,
)
_CUBE_VERTICES = np.asarray(
    (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 1.0),
        (1.0, 1.0, 1.0),
        (0.0, 1.0, 1.0),
    )
)
_CUBE_FACES = np.asarray(
    (
        (0, 2, 1),
        (0, 3, 2),
        (4, 5, 6),
        (4, 6, 7),
        (0, 1, 5),
        (0, 5, 4),
        (3, 7, 6),
        (3, 6, 2),
        (0, 4, 7),
        (0, 7, 3),
        (1, 2, 6),
        (1, 6, 5),
    ),
    dtype=np.int64,
)


def _areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    corners = vertices[faces]
    return (
        np.linalg.norm(
            np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]),
            axis=1,
        )
        / 2.0
    )


def _assert_area_partition(
    arrangement: SurfaceArrangement,
    surfaces: tuple[tuple[np.ndarray, np.ndarray], ...],
) -> None:
    """Fragments of each source face tile it: equal area, same orientation."""

    fragment_area = _areas(arrangement.vertices, arrangement.triangles)
    corners = arrangement.vertices[arrangement.triangles]
    fragment_normal = np.cross(
        corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]
    )
    for index, (vertices, faces) in enumerate(surfaces):
        rows = arrangement.source_surface == index
        tiled = np.bincount(
            arrangement.source_face[rows],
            weights=fragment_area[rows],
            minlength=faces.shape[0],
        )
        np.testing.assert_allclose(tiled, _areas(vertices, faces), rtol=0, atol=1e-14)
        source = vertices[faces[arrangement.source_face[rows]]]
        source_normal = np.cross(source[:, 1] - source[:, 0], source[:, 2] - source[:, 0])
        assert np.all(np.sum(fragment_normal[rows] * source_normal, axis=1) > 0.0)


def _assert_conforming(arrangement: SurfaceArrangement, index: int) -> None:
    """Fragments of one surface meet only in shared welded vertices and edges."""

    faces = arrangement.triangles[arrangement.source_surface == index]
    first, second = np.triu_indices(faces.shape[0], k=1)
    classes, status = triangle_intersection_classes(
        arrangement.vertices[faces[first]],
        arrangement.vertices[faces[second]],
        first_ids=faces[first],
        second_ids=faces[second],
    )
    assert np.all(status == 0)
    assert np.all(np.isin(classes, _LEGAL))


def test_crossing_triangles_split_along_one_welded_segment() -> None:
    first = (
        np.asarray(((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 2.0, 0.0))),
        np.asarray(((0, 1, 2),)),
    )
    second = (
        np.asarray(((0.5, 0.5, -1.0), (0.7, 0.4, 1.0), (0.4, 0.9, 1.0))),
        np.asarray(((0, 1, 2),)),
    )

    arrangement = arrange_triangle_surfaces(*first, *second)

    assert arrangement.evidence.contact_counts[IntersectionClass.CROSSING] == 1
    assert arrangement.intersection_edges.shape == (1, 2)
    ends = arrangement.intersection_edges[0]
    # Both ends are constructions on an edge of the second triangle interior to
    # the first: first-surface face (0, 1, 2) and a second-surface edge.
    features = arrangement.vertex_features[ends]
    np.testing.assert_array_equal(features[:, 0], [[0, 1, 2], [0, 1, 2]])
    assert np.all(features[:, 1, :2] >= 0) and np.all(features[:, 1, 2] == -1)
    np.testing.assert_allclose(arrangement.vertices[ends][:, 2], 0.0, atol=1e-15)
    assert np.all(arrangement.vertex_bounds[ends] <= 1e-15)
    assert arrangement.evidence.split_faces == (1, 1)
    _assert_area_partition(arrangement, (first, second))
    _assert_conforming(arrangement, 0)
    _assert_conforming(arrangement, 1)


def test_open_sheet_cuts_closed_surface_in_one_loop() -> None:
    sheet = (
        np.asarray(
            ((-1.0, -1.0, 0.4), (2.0, -1.0, 0.45), (2.0, 2.0, 0.6), (-1.0, 2.0, 0.55))
        ),
        np.asarray(((0, 1, 2), (0, 2, 3))),
    )
    cube = (_CUBE_VERTICES, _CUBE_FACES)

    arrangement = arrange_triangle_surfaces(*sheet, *cube)

    edges = arrangement.intersection_edges
    degree = np.bincount(edges.reshape(-1), minlength=arrangement.vertices.shape[0])
    assert np.all(degree[np.unique(edges)] == 2)
    assert np.unique(edges).size == edges.shape[0]
    _assert_area_partition(arrangement, (sheet, cube))
    _assert_conforming(arrangement, 0)
    _assert_conforming(arrangement, 1)
    # Cut loop length: the sheet plane crosses the four lateral cube faces.
    lengths = np.linalg.norm(
        arrangement.vertices[edges[:, 0]] - arrangement.vertices[edges[:, 1]], axis=1
    )
    plane_normal = np.cross(sheet[0][1] - sheet[0][0], sheet[0][2] - sheet[0][0])
    heights = (
        sheet[0][0] @ plane_normal
        - np.asarray(((0, 0), (1, 0), (1, 1), (0, 1))) @ plane_normal[:2]
    ) / plane_normal[2]
    corners = np.column_stack(((0.0, 1.0, 1.0, 0.0), (0.0, 0.0, 1.0, 1.0), heights))
    expected = np.sum(np.linalg.norm(corners - np.roll(corners, -1, axis=0), axis=1))
    assert np.sum(lengths) == pytest.approx(expected, abs=1e-14)


@pytest.mark.parametrize(
    ("second_order", "orientation"),
    [((0, 1, 2), 1), ((0, 2, 1), -1)],
    ids=["equal-normals", "opposite-normals"],
)
def test_coplanar_overlap_records_coincident_fragments(
    second_order: tuple[int, int, int], orientation: int
) -> None:
    first = (
        np.asarray(((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 2.0, 0.0))),
        np.asarray(((0, 1, 2),)),
    )
    second = (
        np.asarray(((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 2.0, 0.0))),
        np.asarray((second_order,)),
    )

    arrangement = arrange_triangle_surfaces(*first, *second)

    coincident = arrangement.coincident_face >= 0
    area = _areas(arrangement.vertices, arrangement.triangles)
    # The exact overlap is the triangle (0, 0), (2, 0), (1, 1) of area 1.
    for index in (0, 1):
        rows = coincident[:, 1 - index] & (arrangement.source_surface == index)
        assert np.sum(area[rows]) == pytest.approx(1.0, abs=1e-15)
    assert set(arrangement.coincident_orientation[coincident].tolist()) == {orientation}
    assert arrangement.evidence.coincident_vertices == 2
    _assert_area_partition(arrangement, (first, second))


def test_touching_cubes_weld_coincident_vertices_by_identity() -> None:
    arrangement = arrange_triangle_surfaces(
        _CUBE_VERTICES, _CUBE_FACES, _CUBE_VERTICES + (1.0, 0.0, 0.0), _CUBE_FACES
    )

    assert arrangement.evidence.coincident_vertices == 4
    assert arrangement.evidence.constructed_vertices == 0
    assert arrangement.vertices.shape[0] == 12
    shared = (arrangement.vertex_features[:, 0, 0] >= 0) & (
        arrangement.vertex_features[:, 1, 0] >= 0
    )
    np.testing.assert_array_equal(arrangement.vertex_features[shared, 0, 0], [1, 2, 5, 6])
    np.testing.assert_array_equal(arrangement.vertex_features[shared, 1, 0], [0, 3, 4, 7])
    assert set(arrangement.coincident_orientation.reshape(-1).tolist()) == {-1, 0}


def test_arrangement_is_deterministic() -> None:
    rotated = (_CUBE_VERTICES - 0.5) @ np.asarray(
        ((0.8, -0.6, 0.0), (0.6, 0.8, 0.0), (0.0, 0.0, 1.0))
    ).T + (0.8, 0.7, 0.3)

    runs = [
        arrange_triangle_surfaces(_CUBE_VERTICES, _CUBE_FACES, rotated, _CUBE_FACES)
        for _ in range(2)
    ]

    assert runs[0].arrangement_id == runs[1].arrangement_id
    np.testing.assert_array_equal(runs[0].triangles, runs[1].triangles)


def test_self_intersecting_surface_is_refused() -> None:
    crossing = (
        np.asarray(
            (
                (0.0, 0.0, 0.0),
                (2.0, 0.0, 0.0),
                (0.0, 2.0, 0.0),
                (0.5, 0.5, -1.0),
                (0.7, 0.4, 1.0),
                (0.4, 0.9, 1.0),
            )
        ),
        np.asarray(((0, 1, 2), (3, 4, 5))),
    )

    with pytest.raises(SurfaceArrangementError) as refused:
        arrange_triangle_surfaces(*crossing, _CUBE_VERTICES + 5.0, _CUBE_FACES)

    assert refused.value.status == "self_intersecting_surface"
    assert refused.value.surface == 0
    assert refused.value.faces == (0, 1)


def test_degenerate_triangle_is_refused() -> None:
    collinear = np.asarray(((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (3.0, 3.0, 3.0)))

    with pytest.raises(SurfaceArrangementError) as refused:
        arrange_triangle_surfaces(
            _CUBE_VERTICES, _CUBE_FACES, collinear, np.asarray(((0, 1, 2),))
        )

    assert refused.value.status == "degenerate_triangle"
    assert refused.value.surface == 1


def test_coordinates_outside_exact_domain_are_refused_with_status() -> None:
    scale = np.ldexp(np.float64(1.0), 200)

    with pytest.raises(SurfaceArrangementError) as refused:
        arrange_triangle_surfaces(
            _CUBE_VERTICES, _CUBE_FACES, _CUBE_VERTICES * scale, _CUBE_FACES
        )

    # Every cube face has a corner off the origin, so all of them are named.
    assert refused.value.status == "invalid_coordinates"
    assert refused.value.surface == 1
    assert refused.value.faces == tuple(range(12))


def test_candidate_budget_is_refused() -> None:
    with pytest.raises(SurfaceArrangementError) as refused:
        arrange_triangle_surfaces(
            _CUBE_VERTICES,
            _CUBE_FACES,
            _CUBE_VERTICES + 0.5,
            _CUBE_FACES,
            limits=SurfaceArrangementLimits(maximum_candidate_pairs=40),
        )

    assert refused.value.status == "limit_exceeded"


def test_three_sheets_share_one_original_triple_plane_vertex() -> None:
    faces = np.asarray(((0, 1, 2),), dtype=np.int64)
    sheets = (
        (
            np.asarray(
                ((-2.0, -2.0, 0.0), (2.0, -2.0, 0.0), (0.0, 2.0, 0.0)), dtype=np.float64
            ),
            faces,
        ),
        (
            np.asarray(
                ((0.0, -2.0, -2.0), (0.0, 2.0, -2.0), (0.0, 0.0, 2.0)), dtype=np.float64
            ),
            faces,
        ),
        (
            np.asarray(
                ((-2.0, 0.0, -2.0), (2.0, 0.0, -2.0), (0.0, 0.0, 2.0)), dtype=np.float64
            ),
            faces,
        ),
    )

    arrangement = arrange_triangle_surfaces(*sheets[0], *sheets[1], operands=(sheets[2],))

    center = np.flatnonzero(np.all(arrangement.vertices == 0.0, axis=1))
    np.testing.assert_array_equal(arrangement.construction_types[center], [3])
    vertex = center.item()
    np.testing.assert_array_equal(arrangement.vertex_features[vertex], [[0, 1, 2]] * 3)
    np.testing.assert_array_equal(
        arrangement.vertex_locations[arrangement.vertex_locations[:, 0] == vertex],
        [[vertex, 0, 6], [vertex, 1, 6], [vertex, 2, 6]],
    )
    np.testing.assert_array_equal(arrangement.input_face_offsets, [0, 1, 2, 3])
    # Each sheet's contact graph contains both crossing lines, split at the
    # one welded triple point rather than crossing without a shared vertex.
    edges = arrangement.intersection_edges
    assert np.count_nonzero(np.any(edges == vertex, axis=1)) == 6
    assert all(
        vertex in arrangement.triangles[arrangement.source_surface == operand]
        for operand in range(3)
    )
    _assert_area_partition(arrangement, sheets)


def test_coplanar_three_operand_coverage_retains_every_original_face() -> None:
    faces = np.asarray(((0, 1, 2),), dtype=np.int64)
    outer = np.asarray(
        ((0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (0.0, 4.0, 0.0)), dtype=np.float64
    )
    inner = np.asarray(
        ((1.0, 1.0, 0.0), (2.0, 1.0, 0.0), (1.0, 2.0, 0.0)), dtype=np.float64
    )
    sheets = ((outer, faces), (outer, faces[:, ::-1]), (inner, faces))

    arrangement = arrange_triangle_surfaces(*sheets[0], *sheets[1], operands=(sheets[2],))

    owner = arrangement.source_surface
    index = np.arange(owner.size)
    np.testing.assert_array_equal(arrangement.coincident_face[index, owner], -1)
    np.testing.assert_array_equal(arrangement.coincident_orientation[index, owner], 0)
    area = _areas(arrangement.vertices, arrangement.triangles)
    for operand in range(3):
        rows = owner == operand
        for other in range(3):
            if other == operand:
                continue
            covered = rows & (arrangement.coincident_face[:, other] == 0)
            expected = 0.5 if operand == 2 or other == 2 else 8.0
            assert np.sum(area[covered]) == pytest.approx(expected, abs=1e-14)
            sign = -1 if (operand == 1) != (other == 1) else 1
            assert np.all(arrangement.coincident_orientation[covered, other] == sign)
    _assert_area_partition(arrangement, sheets)


@pytest.mark.parametrize("exponent", [-450, 0, 450], ids=["tiny", "unit", "large"])
def test_native_line_plane_construction_has_outward_exact_bounds(exponent: int) -> None:
    scale = float(np.ldexp(np.float64(1.0), exponent))
    vertices = (
        np.asarray(
            (
                (0.0, 0.0, 0.0),
                (4.0, 0.0, 0.0),
                (0.0, 4.0, 0.0),
                (1.0, 1.0, -1.0),
                (2.0, 1.0, 2.0),
                (1.0, 2.0, 2.0),
            ),
            dtype=np.float64,
        )
        * scale
    )
    native = arrange_triangles(
        vertices,
        np.asarray(((0, 1, 2), (3, 4, 5)), dtype=np.int64),
        np.asarray((0, 1), dtype=np.int32),
        np.asarray(((0, 1),), dtype=np.int64),
        maximum_vertices=64,
        maximum_fragments=64,
    )

    endpoints = np.unique(native.contact_edges)
    exact_scale = Fraction.from_float(scale)
    expected = (
        (4 * exact_scale / 3, exact_scale, Fraction(0)),
        (exact_scale, 4 * exact_scale / 3, Fraction(0)),
    )
    actual = endpoints[np.argsort(native.vertices[endpoints, 0])]
    # A rational independent reference compares the actual binary64 bound,
    # not an allclose tolerance or a rounded version of the expected point.
    for vertex, exact_point in zip(actual, expected[::-1], strict=True):
        error = max(
            abs(Fraction.from_float(float(coordinate)) - exact_coordinate)
            for coordinate, exact_coordinate in zip(
                native.vertices[vertex], exact_point, strict=True
            )
        )
        assert error <= Fraction.from_float(float(native.bounds[vertex]))
        np.testing.assert_array_equal(
            native.vertices[vertex],
            np.asarray(tuple(float(value) for value in exact_point), dtype=np.float64),
        )
    np.testing.assert_array_equal(native.constructions[endpoints], [1, 1])
    for vertex in endpoints:
        location = native.locations[native.locations[:, 0] == vertex]
        np.testing.assert_array_equal(location[:, 1], [0, 1])
        assert location[0, 2] == 6
        assert location[1, 2] in (4, 5)


def test_native_frame_accepts_exact_normal_after_overflow_cancellation() -> None:
    scale = np.ldexp(np.float64(1.0), 600)
    vertices = (
        np.asarray(
            ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (2.0, 3.0, 4.0)),
            dtype=np.float64,
        )
        * scale
    )

    native = arrange_triangles(
        vertices,
        np.asarray(((0, 1, 2),), dtype=np.int64),
        np.asarray((0,), dtype=np.int32),
        np.empty((0, 2), dtype=np.int64),
        maximum_vertices=3,
        maximum_fragments=1,
    )

    np.testing.assert_array_equal(native.vertices, vertices)
    np.testing.assert_array_equal(native.fragments, [[0, 1, 2]])
    np.testing.assert_array_equal(native.bounds, [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(native.locations, [[0, 0, 0], [1, 0, 1], [2, 0, 2]])


def test_nearly_coplanar_nary_triple_retains_original_vertex_identity() -> None:
    epsilon = np.ldexp(np.float64(1.0), -40)
    faces = np.asarray(((1, 2, 4),), dtype=np.int64)
    # The referenced vertices have noncompact original ids. The three exact
    # planes are z=0, 3x-z=1 and z=epsilon*(3y-1), meeting at (1/3,1/3,0).
    first = np.asarray(
        (
            (9.0, 9.0, 9.0),
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (8.0, 8.0, 8.0),
            (0.0, 2.0, 0.0),
        ),
        dtype=np.float64,
    )
    second = np.asarray(
        (
            (9.0, 9.0, 9.0),
            (0.0, -1.0, -1.0),
            (1.0, -1.0, 2.0),
            (8.0, 8.0, 8.0),
            (0.0, 2.0, -1.0),
        ),
        dtype=np.float64,
    )
    third = np.asarray(
        (
            (9.0, 9.0, 9.0),
            (-1.0, 0.0, -epsilon),
            (-1.0, 1.0, 2 * epsilon),
            (8.0, 8.0, 8.0),
            (2.0, 0.0, -epsilon),
        ),
        dtype=np.float64,
    )

    arrangement = arrange_triangle_surfaces(
        first,
        faces,
        second,
        faces,
        operands=((third, faces),),
    )

    triple = np.flatnonzero(arrangement.construction_types == 3).item()
    expected = (Fraction(1, 3), Fraction(1, 3), Fraction(0))
    error = max(
        abs(Fraction.from_float(float(coordinate)) - exact)
        for coordinate, exact in zip(arrangement.vertices[triple], expected, strict=True)
    )
    assert error <= Fraction.from_float(float(arrangement.vertex_bounds[triple]))
    np.testing.assert_array_equal(
        arrangement.vertices[triple],
        np.asarray(tuple(float(value) for value in expected), dtype=np.float64),
    )
    np.testing.assert_array_equal(arrangement.vertex_features[triple], [[1, 2, 4]] * 3)
    np.testing.assert_array_equal(
        arrangement.vertex_locations[arrangement.vertex_locations[:, 0] == triple],
        [[triple, 0, 6], [triple, 1, 6], [triple, 2, 6]],
    )
    assert arrangement.evidence.triple_plane_vertices == 1


def test_exact_distinct_contact_points_cannot_merge_in_rounded_publication() -> None:
    faces = np.asarray(((0, 1, 2),), dtype=np.int64)
    first = np.asarray(
        ((1.0, 1.0, 0.0), (2.0, 1.0, 0.0), (1.0, 2.0, 0.0)),
        dtype=np.float64,
    )
    delta = np.ldexp(np.float64(1.0), -60)
    second = np.asarray(
        ((1.0, 1.0, -delta), (2.0, 1.0, 1.0), (1.0, 2.0, 1.0)),
        dtype=np.float64,
    )
    # The two exact cut endpoints differ from (1,1,0) by delta/(1+delta)
    # on separate axes. All three positions round to the same binary64 point.
    with pytest.raises(SurfaceArrangementError) as refused:
        arrange_triangle_surfaces(first, faces, second, faces)

    assert refused.value.status == "unrepresentable_publication"
    assert refused.value.surface == 0
    assert refused.value.faces == (0,)


def test_nary_exact_incidence_budget_is_refused() -> None:
    vertices = np.asarray(
        ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 2.0, 0.0)),
        dtype=np.float64,
    )
    faces = np.asarray(((0, 1, 2),), dtype=np.int64)

    with pytest.raises(SurfaceArrangementError) as refused:
        arrange_triangle_surfaces(
            vertices,
            faces,
            vertices,
            faces,
            operands=((vertices, faces),),
            limits=SurfaceArrangementLimits(maximum_incidence_records=1),
        )

    assert refused.value.status == "limit_exceeded"


def test_rotated_nary_cubes_publish_without_collapsed_contact_ears() -> None:
    def rotated_cube(
        axis: tuple[float, float, float],
        angle: float,
        offset: tuple[float, float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        unit = np.asarray(axis, dtype=np.float64)
        unit = unit / np.linalg.norm(unit)
        cross = np.asarray(
            (
                (0.0, -unit[2], unit[1]),
                (unit[2], 0.0, -unit[0]),
                (-unit[1], unit[0], 0.0),
            ),
            dtype=np.float64,
        )
        rotation = (
            np.eye(3, dtype=np.float64)
            + math.sin(angle) * cross
            + (1.0 - math.cos(angle)) * cross @ cross
        )
        vertices = (
            (_CUBE_VERTICES - 0.5) @ rotation.T
            + 0.5
            + np.asarray(offset, dtype=np.float64)
        )
        return vertices, _CUBE_FACES

    # Independently rounded square corners are not necessarily exactly
    # coplanar. Their tiny represented kinks must retain every contact vertex,
    # but a free triangulation chord must not force a rounded zero-area ear.
    cubes = (
        rotated_cube((1.0, 2.0, 3.0), 0.19, (0.0, 0.0, 0.0)),
        rotated_cube((3.0, -1.0, 2.0), -0.23, (0.37, 0.17, -0.09)),
        rotated_cube((2.0, 1.0, -1.0), 0.31, (0.19, -0.11, 0.33)),
    )

    arrangement = arrange_triangle_surfaces(*cubes[0], *cubes[1], operands=(cubes[2],))

    _assert_area_partition(arrangement, cubes)
    _assert_conforming(arrangement, 0)
    _assert_conforming(arrangement, 1)
    _assert_conforming(arrangement, 2)
    assert arrangement.evidence.triple_plane_vertices > 0


def _crossing_cubes(
    surfaces: tuple[int, int] = (0, 1),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Two crossing cubes as native arrangement input with all cross pairs."""

    vertices = np.concatenate((_CUBE_VERTICES, _CUBE_VERTICES + (0.5, 0.25, 0.25)))
    faces = np.concatenate((_CUBE_FACES, _CUBE_FACES + 8))
    labels = np.repeat(np.asarray(surfaces, dtype=np.int32), 12)
    first, second = np.meshgrid(np.arange(12), np.arange(12, 24), indexing="ij")
    pairs = np.stack((first.reshape(-1), second.reshape(-1)), axis=1)
    return vertices, faces, labels, pairs


def test_native_classification_matches_box_membership() -> None:
    vertices, faces, labels, pairs = _crossing_cubes()

    native = arrange_triangles(
        vertices,
        faces,
        labels,
        pairs,
        maximum_vertices=256,
        maximum_fragments=256,
        classification_operands=2,
        classification_work_limit=1 << 20,
    )

    classification = native.classification
    assert classification is not None
    assert classification.status == MeshcoreStatus.OK
    windings = classification.component_windings[classification.fragment_components]
    owner = labels[native.fragment_faces]
    rows = np.arange(owner.size)
    np.testing.assert_array_equal(windings[rows, owner], ARRANGEMENT_UNCLASSIFIED)
    other = 1 - owner
    # Independent oracle: open-box membership of each fragment centroid. No
    # face planes of the two cubes coincide, so every pair is classified.
    centroid = native.vertices[native.fragments].mean(axis=1)
    lower = np.where(other[:, None] == 0, 0.0, np.asarray((0.5, 0.25, 0.25)))
    inside = np.all((centroid > lower) & (centroid < lower + 1.0), axis=1)
    np.testing.assert_array_equal(windings[rows, other], inside.astype(np.int32))
    assert classification.components < owner.size


def test_native_classification_admits_exact_work_and_refuses_one_under() -> None:
    vertices, faces, labels, pairs = _crossing_cubes()

    def classify(limit: int) -> ArrangementClassification:
        native = arrange_triangles(
            vertices,
            faces,
            labels,
            pairs,
            maximum_vertices=256,
            maximum_fragments=256,
            classification_operands=2,
            classification_work_limit=limit,
        )
        assert native.classification is not None
        return native.classification

    generous = classify(1 << 20)
    required = generous.pair_scans + generous.matrix_entries
    exact = classify(required)
    under = classify(required - 1)
    zero = classify(0)

    assert exact.status == MeshcoreStatus.OK
    np.testing.assert_array_equal(exact.component_windings, generous.component_windings)
    assert (exact.pair_scans, exact.matrix_entries) == (
        generous.pair_scans,
        generous.matrix_entries,
    )
    assert generous.matrix_entries == 2 * generous.components
    for refused in (under, zero):
        assert refused.status == MeshcoreStatus.CAPACITY_EXCEEDED
        assert refused.fragment_components.size == 0
        assert refused.component_windings.shape == (0, 2)
    assert zero.pair_scans == 0


@pytest.mark.parametrize(
    ("surfaces", "declared"),
    [((0, 2147483647), 2), ((0, 1), 1 << 40), ((1, 1), 2)],
    ids=["sparse-int32-max-label", "count-beyond-triangles", "missing-label"],
)
def test_native_classification_refuses_undeclared_operand_namespace(
    surfaces: tuple[int, int], declared: int
) -> None:
    vertices, faces, labels, pairs = _crossing_cubes(surfaces)
    if surfaces[0] == surfaces[1]:
        pairs = np.empty((0, 2), dtype=np.int64)

    with pytest.raises(ArrangementFailure) as refused:
        arrange_triangles(
            vertices,
            faces,
            labels,
            pairs,
            maximum_vertices=256,
            maximum_fragments=256,
            classification_operands=declared,
            classification_work_limit=1 << 40,
        )

    assert refused.value.status == MeshcoreStatus.INVALID_INPUT


def test_native_classification_respects_ambient_byte_cap() -> None:
    vertices, faces, labels, pairs = _crossing_cubes()
    budget = NativeExecutionBudget(
        max_work=1 << 40,
        max_geometry_queries=1 << 40,
        max_cavity_cells=1 << 20,
        max_scratch_bytes=64,
        max_wall_seconds=float("inf"),
    )

    # A scratch cap below the exact construction and classification state is
    # refused with evidence, never exceeded.
    with pytest.raises(ArrangementFailure) as refused, budget:
        arrange_triangles(
            vertices,
            faces,
            labels,
            pairs,
            maximum_vertices=256,
            maximum_fragments=256,
            classification_operands=2,
            classification_work_limit=1 << 20,
        )

    assert refused.value.status == MeshcoreStatus.CAPACITY_EXCEEDED
    assert budget.evidence is not None
    assert budget.evidence.status == MeshcoreStatus.CAPACITY_EXCEEDED
