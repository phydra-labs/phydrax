#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import numpy as np
import pytest

from phydrax.geometry import BVHBuildKind, BVHBuildPolicy
from phydrax.geometry.simplicial import (
    TriangleBVH,
    TriangleMesh,
    WindingNumberRoute,
)


def _sphere(longitudes: int = 16, latitudes: int = 10):
    vertices = [(0.0, 0.0, 1.0)]
    for row in range(1, latitudes):
        polar = np.pi * row / latitudes
        for column in range(longitudes):
            azimuth = 2.0 * np.pi * column / longitudes
            vertices.append(
                (
                    np.sin(polar) * np.cos(azimuth),
                    np.sin(polar) * np.sin(azimuth),
                    np.cos(polar),
                )
            )
    vertices.append((0.0, 0.0, -1.0))
    faces = [
        (0, 1 + column, 1 + (column + 1) % longitudes) for column in range(longitudes)
    ]
    for row in range(latitudes - 2):
        for column in range(longitudes):
            a = 1 + row * longitudes + column
            b = 1 + row * longitudes + (column + 1) % longitudes
            faces += [(a, a + longitudes, b + longitudes), (a, b + longitudes, b)]
    south = len(vertices) - 1
    base = 1 + (latitudes - 2) * longitudes
    faces += [
        (south, base + (column + 1) % longitudes, base + column)
        for column in range(longitudes)
    ]
    return np.asarray(vertices), np.asarray(faces)


def _solid_angle_winding(vertices, faces, points):
    a, b, c = (vertices[faces[:, corner]][None] - points[:, None] for corner in range(3))
    length_a, length_b, length_c = (np.linalg.norm(value, axis=-1) for value in (a, b, c))
    numerator = np.sum(a * np.cross(b, c), axis=-1)
    denominator = (
        length_a * length_b * length_c
        + np.sum(a * b, axis=-1) * length_c
        + np.sum(b * c, axis=-1) * length_a
        + np.sum(c * a, axis=-1) * length_b
    )
    return np.sum(2.0 * np.arctan2(numerator, denominator), axis=-1) / (4.0 * np.pi)


def _queries(seed: int, count: int = 60):
    return np.random.default_rng(seed).uniform(-1.4, 1.4, (count, 3))


@pytest.mark.parametrize("kind", tuple(BVHBuildKind))
def test_exact_winding_matches_brute_force_on_closed_and_open_meshes(kind) -> None:
    vertices, faces = _sphere()
    points = _queries(1)
    policy = BVHBuildPolicy(kind, leaf_size=4)

    for surface in (faces, faces[: faces.shape[0] // 2 + 3]):
        index = TriangleBVH(TriangleMesh(vertices, surface), policy=policy)
        result = index.winding_number(points)

        assert result.route is WindingNumberRoute.EXACT
        assert not result.approximate and result.opening_angle is None
        np.testing.assert_allclose(
            np.asarray(result.values),
            _solid_angle_winding(vertices, surface, points),
            atol=1.0e-10,
        )


def test_fast_winding_reports_its_approximation_and_classifies_far_points() -> None:
    vertices, faces = _sphere()
    index = TriangleBVH(TriangleMesh(vertices, faces), policy=BVHBuildPolicy(leaf_size=4))
    radius = np.linalg.norm(_queries(2), axis=-1, keepdims=True)
    points = _queries(2) / radius * np.where(radius > 1.0, 1.5, 0.5)

    fast = index.fast_winding_number(points, opening_angle=2.0)
    exact = index.winding_number(points)

    assert fast.route is WindingNumberRoute.FAST_DIPOLE
    assert fast.approximate and fast.opening_angle == 2.0
    assert np.array_equal(
        np.abs(np.asarray(fast.values)) > 0.5, np.abs(np.asarray(exact.values)) > 0.5
    )
    np.testing.assert_allclose(
        np.asarray(fast.values), np.asarray(exact.values), atol=5.0e-2
    )
    with pytest.raises(ValueError, match="opening_angle"):
        index.fast_winding_number(points, opening_angle=0.0)


def test_refit_matches_rebuilt_hierarchy_queries() -> None:
    vertices, faces = _sphere()
    rng = np.random.default_rng(3)
    moved = vertices * np.asarray((1.4, 0.7, 1.1)) + 0.03 * rng.standard_normal(
        vertices.shape
    )
    points = _queries(4)
    policy = BVHBuildPolicy(BVHBuildKind.SAH, leaf_size=4)

    refitted = TriangleBVH(TriangleMesh(vertices, faces), policy=policy).refit(moved)
    rebuilt = TriangleBVH(TriangleMesh(moved, faces), policy=policy)

    first = refitted.query(points)
    second = rebuilt.query(points)
    assert np.array_equal(np.asarray(first.face_index), np.asarray(second.face_index))
    np.testing.assert_allclose(np.asarray(first.distance), np.asarray(second.distance))
    np.testing.assert_allclose(
        np.asarray(refitted.winding_number(points).values),
        _solid_angle_winding(moved, faces, points),
        atol=1.0e-10,
    )


def test_nearest_faces_match_single_leaf_exhaustive_reference() -> None:
    vertices, faces = _sphere()
    points = _queries(5, 30)

    nearest = TriangleBVH(
        TriangleMesh(vertices, faces), policy=BVHBuildPolicy(leaf_size=4)
    ).nearest_faces(points, k=2)
    # One leaf holding every face evaluates all of them: the exhaustive route.
    reference = TriangleBVH(
        TriangleMesh(vertices, faces), policy=BVHBuildPolicy(leaf_size=faces.shape[0])
    ).nearest_faces(points, k=2)

    assert np.array_equal(np.asarray(nearest.items), np.asarray(reference.items))
    np.testing.assert_allclose(
        np.asarray(nearest.distance_squared), np.asarray(reference.distance_squared)
    )
