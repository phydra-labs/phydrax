#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import numpy as np
import pytest

from phydrax._meshcore import meshcore_available
from phydrax.geometry.surface import (
    arrange_triangle_surfaces,
    SurfaceArrangement,
    SurfaceArrangementError,
)


pytestmark = [
    pytest.mark.meshcore,
    pytest.mark.skipif(
        not meshcore_available(), reason="native phydrax-meshcore library is unavailable"
    ),
]


def test_crossing_open_triangles_publish_welded_split_features() -> None:
    first_vertices = np.asarray(
        ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 2.0, 0.0))
    )
    first_faces = np.asarray(((0, 1, 2),), dtype=np.int64)
    second_vertices = np.asarray(
        ((0.5, 0.5, -1.0), (0.7, 0.4, 1.0), (0.4, 0.9, 1.0))
    )
    second_faces = np.asarray(((0, 1, 2),), dtype=np.int64)

    arrangement = arrange_triangle_surfaces(
        first_vertices, first_faces, second_vertices, second_faces
    )

    assert isinstance(arrangement, SurfaceArrangement)
    assert arrangement.intersection_edges.shape == (1, 2)
    assert arrangement.evidence.fragments[0] > 1
    assert arrangement.evidence.fragments[1] > 1
    assert arrangement.evidence.split_faces == (1, 1)
    assert set(arrangement.source_surface.tolist()) == {0, 1}
    endpoints = arrangement.intersection_edges[0]
    np.testing.assert_array_equal(
        arrangement.vertex_features[endpoints, 0], ((0, 1, 2), (0, 1, 2))
    )
    assert np.all(np.sum(arrangement.vertex_features[endpoints, 1] >= 0, axis=1) == 2)
    np.testing.assert_allclose(arrangement.vertices[endpoints, 2], 0.0, atol=1e-15)


def test_nary_arrangement_retains_triple_operand_feature_incidence() -> None:
    first_vertices = np.asarray(
        ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 2.0, 0.0))
    )
    second_vertices = np.asarray(
        ((0.5, 0.5, -1.0), (0.7, 0.4, 1.0), (0.4, 0.9, 1.0))
    )
    third_vertices = np.asarray(
        ((0.5, -1.0, -1.0), (0.5, 2.0, -1.0), (0.5, 0.0, 1.0))
    )
    faces = np.asarray(((0, 1, 2),), dtype=np.int64)

    arrangement = arrange_triangle_surfaces(
        first_vertices,
        faces,
        second_vertices,
        faces,
        operands=((third_vertices, faces),),
    )

    assert arrangement.input_face_offsets == (0, 1, 2, 3)
    assert set(arrangement.source_surface.tolist()) == {0, 1, 2}
    assert arrangement.vertex_features.shape[1] == 3
    incident = np.any(arrangement.vertex_features >= 0, axis=-1)
    assert np.any(np.all(incident, axis=1))


def test_self_intersecting_input_is_refused_without_repair() -> None:
    crossing_vertices = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (0.0, 2.0, 0.0),
            (0.5, 0.5, -1.0),
            (0.7, 0.4, 1.0),
            (0.4, 0.9, 1.0),
        )
    )
    crossing_faces = np.asarray(((0, 1, 2), (3, 4, 5)), dtype=np.int64)
    remote_vertices = np.asarray(((5.0, 0.0, 0.0), (6.0, 0.0, 0.0), (5.0, 1.0, 0.0)))
    remote_faces = np.asarray(((0, 1, 2),), dtype=np.int64)

    with pytest.raises(SurfaceArrangementError) as refused:
        arrange_triangle_surfaces(
            crossing_vertices, crossing_faces, remote_vertices, remote_faces
        )

    assert refused.value.status == "self_intersecting_surface"
    assert refused.value.surface == 0
    assert refused.value.faces == (0, 1)


def test_exactly_degenerate_triangle_is_refused() -> None:
    collinear = np.asarray(((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (3.0, 3.0, 3.0)))
    faces = np.asarray(((0, 1, 2),), dtype=np.int64)

    with pytest.raises(SurfaceArrangementError) as refused:
        arrange_triangle_surfaces(
            collinear,
            faces,
            np.asarray(((0.0, 0.0, 2.0), (1.0, 0.0, 2.0), (0.0, 1.0, 2.0))),
            faces,
        )

    assert refused.value.status == "degenerate_triangle"
    assert refused.value.surface == 0
    assert refused.value.faces == (0,)


def test_inexact_integer_coordinates_are_not_rounded_to_float64() -> None:
    base = 2**53
    vertices = np.asarray(
        ((base + 1, 0, 0), (base + 3, 0, 0), (base + 1, 2, 0)), dtype=np.int64
    )
    faces = np.asarray(((0, 1, 2),), dtype=np.int64)

    with pytest.raises(ValueError, match="exactly representable"):
        arrange_triangle_surfaces(vertices, faces, vertices, faces)
