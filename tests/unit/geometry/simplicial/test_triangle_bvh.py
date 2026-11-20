#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import jax
import numpy as np
import pytest

from phydrax._meshcore import meshcore_available
from phydrax.geometry import BVHBuildKind, BVHBuildPolicy
from phydrax.geometry.simplicial import (
    TriangleBVH,
    TriangleMesh,
    WindingNumberRoute,
)


_requires_meshcore = pytest.mark.skipif(
    not meshcore_available(), reason="native phydrax-meshcore library is not available"
)


def _sphere(longitudes: int = 16, latitudes: int = 10) -> Any:
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


def _solid_angle_winding(vertices: Any, faces: Any, points: Any) -> Any:
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


def _queries(seed: int, count: int = 60) -> Any:
    return np.random.default_rng(seed).uniform(-1.4, 1.4, (count, 3))


def test_triangle_bvh_scenario_1() -> None:
    for kind in tuple(BVHBuildKind):
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


def _triangle_squared_distance(point: np.ndarray, triangle: np.ndarray) -> float:
    """Independent scalar projection via a NumPy Gram solve and edge clipping."""
    directions = np.stack((triangle[1] - triangle[0], triangle[2] - triangle[0]))
    coordinates = np.linalg.solve(
        directions @ directions.T, directions @ (point - triangle[0])
    )
    candidates = []
    if np.all(coordinates >= 0.0) and np.sum(coordinates) <= 1.0:
        candidates.append(triangle[0] + coordinates @ directions)
    for first, second in ((0, 1), (1, 2), (2, 0)):
        edge = triangle[second] - triangle[first]
        fraction = np.clip(
            np.dot(point - triangle[first], edge) / np.dot(edge, edge), 0.0, 1.0
        )
        candidates.append(triangle[first] + fraction * edge)
    return min(float(np.dot(point - closest, point - closest)) for closest in candidates)


def test_nearest_faces_match_independent_triangle_reference() -> None:
    vertices, faces = _sphere()
    points = _queries(5, 30)
    nearest = TriangleBVH(
        TriangleMesh(vertices, faces), policy=BVHBuildPolicy(leaf_size=4)
    ).nearest_faces(points, k=2)
    reference = np.asarray(
        [
            [_triangle_squared_distance(point, vertices[face]) for face in faces]
            for point in points
        ],
        dtype=np.float64,
    )
    ranked = np.sort(reference, axis=-1)[:, :2]
    selected = np.take_along_axis(reference, np.asarray(nearest.items), axis=-1)
    # Shared-edge projections are geometric ties; vector width can perturb their
    # last bit. Compare independent distances, not a second BVH's near-tie order.
    np.testing.assert_allclose(selected, ranked, atol=1.0e-12, rtol=1.0e-12)
    np.testing.assert_allclose(
        nearest.distance_squared, ranked, atol=1.0e-12, rtol=1.0e-12
    )


def test_nearest_faces_orders_exact_geometric_ties_by_original_index() -> None:
    vertices = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
        ),
        dtype=np.float64,
    )
    faces = np.asarray(((0, 1, 2), (3, 4, 5)), dtype=np.int32)
    nearest = TriangleBVH(
        TriangleMesh(vertices, faces), policy=BVHBuildPolicy(leaf_size=1)
    ).nearest_faces(np.asarray(((0.25, 0.25, 2.0),), dtype=np.float64), k=2)
    np.testing.assert_array_equal(nearest.items, ((0, 1),))
    np.testing.assert_array_equal(nearest.distance_squared, ((4.0, 4.0),))


def _native_case() -> tuple[TriangleBVH, Any, Any]:
    vertices, faces = _sphere(32, 20)
    index = TriangleBVH(TriangleMesh(vertices, faces), policy=BVHBuildPolicy(leaf_size=4))
    points = np.random.default_rng(8).uniform(-1.3, 1.3, (400, 3))
    # Corners and edge midpoints raised off the surface reach vertex and edge features.
    edge_points = 0.5 * (vertices[faces[:, 0]] + vertices[faces[:, 1]])
    points = np.concatenate((points, 1.05 * vertices[:40], 0.97 * edge_points[:40]))
    return index, vertices, points


@pytest.mark.meshcore
@_requires_meshcore
def test_native_nearest_and_query_match_traced_route() -> None:
    index, _, points = _native_case()
    native = index.nearest_faces(points, k=2)
    traced = jax.jit(lambda values: index.nearest_faces(values, k=2))(points)
    np.testing.assert_allclose(
        native.distance_squared, traced.distance_squared, atol=1e-15
    )
    # Faces agree wherever the two nearest are not tied to rounding.
    gap = np.asarray(traced.distance_squared[:, 1] - traced.distance_squared[:, 0])
    separated = gap > 1.0e-12
    np.testing.assert_array_equal(
        np.asarray(native.items[separated, 0]), np.asarray(traced.items[separated, 0])
    )
    query = index.query(points)
    traced_query = jax.jit(lambda values: index.query(values))(points)
    np.testing.assert_array_equal(
        np.asarray(query.face_index)[separated],
        np.asarray(traced_query.face_index)[separated],
    )
    np.testing.assert_allclose(query.distance, traced_query.distance, atol=1e-15)
    np.testing.assert_allclose(
        np.asarray(query.closest_point)[separated],
        np.asarray(traced_query.closest_point)[separated],
        atol=1e-14,
    )


@pytest.mark.meshcore
@_requires_meshcore
def test_native_winding_numbers_match_traced_route() -> None:
    index, _, points = _native_case()
    exact = index.winding_number(points)
    traced = jax.jit(lambda values: index.winding_number(values).values)(points)
    assert exact.route is WindingNumberRoute.EXACT
    np.testing.assert_allclose(exact.values, traced, atol=1e-12)
    np.testing.assert_array_equal(np.abs(exact.values) > 0.5, np.abs(traced) > 0.5)
    fast = index.fast_winding_number(points, opening_angle=2.0)
    traced_fast = jax.jit(lambda values: index.fast_winding_number(values).values)(points)
    assert fast.route is WindingNumberRoute.FAST_DIPOLE and fast.opening_angle == 2.0
    np.testing.assert_allclose(fast.values, traced_fast, atol=1e-12)


@pytest.mark.meshcore
@_requires_meshcore
def test_native_signed_distance_matches_traced_route_and_sphere() -> None:
    index, _, points = _native_case()
    radius = np.linalg.norm(points, axis=-1)
    outside_band = np.abs(radius - 1.0) > 0.02
    for sign in ("pseudonormal", "winding_number"):
        native = np.asarray(index.signed_distance(points, sign=sign))
        traced = np.asarray(
            jax.jit(lambda values, sign=sign: index.signed_distance(values, sign=sign))(
                points
            )
        )
        np.testing.assert_allclose(native, traced, atol=1e-15)
        np.testing.assert_array_equal(
            (native < 0.0)[outside_band], (radius < 1.0)[outside_band]
        )
        np.testing.assert_allclose(
            np.abs(native), np.asarray(index.query(points).distance), atol=1e-15
        )
    with pytest.raises(ValueError, match="sign"):
        index.signed_distance(points, sign="nearest")  # ty: ignore[invalid-argument-type]


def test_pseudonormal_sign_holds_beside_corners() -> None:
    # Radial points just outside and inside each corner of a coarse sphere have
    # the corner (or an incident edge) as closest feature.
    vertices, faces = _sphere(6, 4)
    index = TriangleBVH(TriangleMesh(vertices, faces), policy=BVHBuildPolicy(leaf_size=2))
    points = np.concatenate((1.01 * vertices, 0.99 * vertices))
    distance = np.asarray(jax.jit(lambda values: index.signed_distance(values))(points))
    np.testing.assert_array_equal(distance < 0.0, np.repeat((False, True), len(vertices)))
