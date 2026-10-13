#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

import io

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.typing import NDArray

from phydrax._fingerprint import array_tree_fingerprint
from phydrax.geometry import (
    BVHBuildKind,
    BVHBuildPolicy,
    ParameterId,
    PlanarMeshRegion,
    Polygon,
    SegmentBVH,
    SegmentQueryExecution,
)
from phydrax.geometry.simplicial._bvh import _segment_ray_parity


def _reference(
    vertices: NDArray[np.float64],
    edges: NDArray[np.int32],
    points: NDArray[np.float64],
    k: int,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    distances = np.empty((points.shape[0], edges.shape[0]), dtype=np.float64)
    for row, point in enumerate(points):
        for column in range(edges.shape[0]):
            first, second = edges[column, 0], edges[column, 1]
            direction = vertices[second] - vertices[first]
            fraction = np.clip(
                np.dot(point - vertices[first], direction) / np.dot(direction, direction),
                0.0,
                1.0,
            )
            projected = vertices[first] + fraction * direction
            distances[row, column] = np.dot(point - projected, point - projected)
    order = np.argsort(distances, axis=-1, kind="stable")[:, :k]
    return order, np.take_along_axis(distances, order, axis=-1)


@pytest.mark.parametrize("kind", tuple(BVHBuildKind))
@pytest.mark.parametrize("k", (1, 2))
@pytest.mark.parametrize("execution", ("bvh", "dense"))
def test_nearest_matches_independent_segments(
    kind: BVHBuildKind, k: int, execution: SegmentQueryExecution
) -> None:
    rng = np.random.default_rng(903)
    vertices = rng.normal(size=(42, 2)).astype(np.float64)
    edges = np.arange(42, dtype=np.int32).reshape((-1, 2))
    points = rng.normal(size=(19, 2)).astype(np.float64)
    index = SegmentBVH(
        vertices, edges, policy=BVHBuildPolicy(kind, leaf_size=3), execution=execution
    )
    actual = index.nearest_segments(points, k=k)
    expected, squared = _reference(vertices, edges, points, k)
    np.testing.assert_array_equal(actual.items, expected)
    np.testing.assert_allclose(actual.distance_squared, squared, atol=1.0e-13)


def test_ties_keep_original_segment_order() -> None:
    vertices = np.asarray(
        ((-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0), (-1.0, -1.0), (1.0, -1.0)),
        dtype=np.float64,
    )
    edges = np.asarray(((0, 1), (2, 3), (4, 5)), dtype=np.int32)
    index = SegmentBVH(
        vertices, edges, policy=BVHBuildPolicy(leaf_size=1), execution="bvh"
    )
    actual = index.nearest_segments(np.asarray(((0.0, 0.0), (0.0, -2.0))), k=2)
    np.testing.assert_array_equal(actual.items, ((0, 1), (0, 2)))
    np.testing.assert_allclose(actual.distance_squared, 1.0)
    np.testing.assert_array_equal(
        index.query(jnp.zeros((2,), dtype=jnp.float64)).segment_index, 0
    )


def test_refit_changes_embedding_without_changing_order() -> None:
    vertices = np.asarray(
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)), dtype=np.float64
    )
    edges = np.asarray(((0, 1), (1, 2), (2, 3), (3, 0)), dtype=np.int32)
    moved = vertices * np.asarray((1.7, 0.3)) + np.asarray((2.0, -3.0))
    points = np.asarray(((2.3, -2.4), (0.0, 0.0), (3.0, -3.1)), dtype=np.float64)
    index = SegmentBVH(vertices, edges).refit(moved)
    actual = index.nearest_segments(points, k=2)
    expected, distances = _reference(moved, edges, points, 2)
    np.testing.assert_array_equal(actual.items, expected)
    np.testing.assert_allclose(actual.distance_squared, distances)


def test_nearest_projection_supports_reverse_mode() -> None:
    vertices = jnp.asarray(
        ((0.0, 0.0), (2.0, 0.0), (0.0, 3.0), (2.0, 3.0)), dtype=jnp.float64
    )
    edges = jnp.asarray(((0, 1), (2, 3)), dtype=jnp.int32)
    index = SegmentBVH(vertices, edges, execution="bvh")
    points = jnp.asarray(((0.7, 0.4),), dtype=jnp.float64)
    gradient = jax.grad(
        lambda moved: jnp.sum(
            index.refit(moved).nearest_segments(points, k=2).distance_squared
        )
    )(vertices)
    np.testing.assert_allclose(
        np.asarray(gradient)[:, 1], (-0.52, -0.28, 3.38, 1.82), atol=1.0e-12
    )


def test_single_segment_missing_second_rank() -> None:
    index = SegmentBVH(
        jnp.asarray(((0.0, 0.0), (1.0, 0.0)), dtype=jnp.float64),
        jnp.asarray(((0, 1),), dtype=jnp.int32),
    )
    nearest = index.nearest_segments(jnp.asarray((0.3, 2.0), dtype=jnp.float64), k=2)
    np.testing.assert_array_equal(nearest.items, (0, -1))
    np.testing.assert_array_equal(nearest.distance_squared, (4.0, np.inf))


@pytest.mark.parametrize("count", (4, 67, 130))
def test_tiled_ray_parity_matches_scalar_reference(count: int) -> None:
    angles = np.arange(count, dtype=np.float64) * 2.0 * np.pi / count
    vertices = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    segments = np.stack((vertices, np.roll(vertices, -1, axis=0)), axis=1)
    points = np.asarray(
        ((0.0, 0.0), (2.0, 0.0), (-2.0, 0.0), (0.5, 0.2), (0.0, 1.2)), dtype=np.float64
    )
    expected = []
    for x, y in points:
        hits = 0
        for first, second in segments:
            if (first[1] > y) != (second[1] > y):
                intercept = first[0] + (second[0] - first[0]) * (y - first[1]) / (
                    second[1] - first[1]
                )
                hits += x < intercept
        expected.append(hits % 2 == 1)
    edge_vertices = jnp.arange(count, dtype=jnp.int32)
    edges = jnp.stack((edge_vertices, (edge_vertices + 1) % count), axis=1)
    np.testing.assert_array_equal(
        _segment_ray_parity(jnp.asarray(points), jnp.asarray(vertices), edges), expected
    )


def test_polygon_pair_tiles_catch_crossing_after_first_tile() -> None:
    count = 130
    angles = np.arange(count, dtype=np.float64) * 2.0 * np.pi / count
    vertices = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    geometry = Polygon(vertices, feature_id="streamed-polygon").compile()
    assert bool(geometry.validity().accepted)
    changed = vertices.copy()
    changed[[90, 100]] = changed[[100, 90]]
    parameter = geometry.schema.index(ParameterId("streamed-polygon", "vertices"))
    state = geometry.state.replace_at(parameter, jnp.asarray(changed))
    assert not bool(geometry.validity(state).accepted)


def test_planar_closest_point_keeps_two_segment_evidence_after_refit() -> None:
    vertices = jnp.asarray(
        ((0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)), dtype=jnp.float64
    )
    geometry = PlanarMeshRegion(
        vertices, ((0, 1, 2, 3),), feature_id="evidence-square"
    ).compile()
    parameter = geometry.schema.index(ParameterId("evidence-square", "vertices"))
    offset = jnp.asarray((3.0, -2.0), dtype=jnp.float64)
    state = geometry.state.replace_at(parameter, vertices + offset)
    points = (
        jnp.asarray(
            ((0.7, -0.4), (0.7, 0.2), (1.0, 1.0), (0.0, 0.0), (3.0, 1.0)),
            dtype=jnp.float64,
        )
        + offset
    )
    result = geometry.with_state(state).closest_point(points)
    expected = np.asarray(
        ((0.7, 0.0), (0.7, 0.0), (1.0, 0.0), (0.0, 0.0), (2.0, 1.0)),
        dtype=np.float64,
    ) + np.asarray(offset)
    np.testing.assert_allclose(result.closest_point, expected, atol=1.0e-13)
    np.testing.assert_array_equal(result.source_entity_id, (0, 0, 0, 0, 1))
    np.testing.assert_array_equal(result.unique, (True, True, False, False, True))
    np.testing.assert_array_equal(result.regular, (True, True, False, False, True))
    np.testing.assert_allclose(
        result.margin,
        (np.sqrt(0.65) - 0.4, np.sqrt(0.49) - 0.2, 0.0, 0.0, np.sqrt(2.0) - 1.0),
        atol=1.0e-13,
    )


def test_dense_admission_refuses_and_auto_uses_bounded_queries() -> None:
    vertices = jnp.asarray(((0.0, 0.0), (1.0, 0.0)), dtype=jnp.float64)
    edges = jnp.asarray(((0, 1),), dtype=jnp.int32)
    points = jnp.asarray(((0.3, 0.2),), dtype=jnp.float64)
    dense = SegmentBVH(vertices, edges, execution="dense", dense_workspace_bytes=1)
    with pytest.raises(ValueError, match="needs 64 workspace bytes"):
        dense.nearest_segments(points)
    automatic = SegmentBVH(vertices, edges, dense_workspace_bytes=1)
    result = automatic.nearest_segments(points, k=2)
    np.testing.assert_array_equal(result.items, ((0, -1),))
    np.testing.assert_allclose(result.distance_squared[:, 0], 0.04)


def test_segment_bvh_refitted_leaves_round_trip() -> None:
    vertices = np.asarray(
        ((0.0, 0.0), (2.0, 0.0), (0.0, 3.0), (2.0, 3.0)), dtype=np.float64
    )
    edges = np.asarray(((0, 1), (2, 3)), dtype=np.int32)
    template = SegmentBVH(vertices, edges, execution="bvh", dense_workspace_bytes=128)
    moved = vertices * np.asarray((1.4, 0.8)) + np.asarray((3.0, -2.0))
    index = template.refit(moved)
    payload = io.BytesIO()
    eqx.tree_serialise_leaves(payload, index)
    payload.seek(0)
    restored = eqx.tree_deserialise_leaves(payload, template)
    assert array_tree_fingerprint(restored) == array_tree_fingerprint(index)
    assert array_tree_fingerprint(restored) != array_tree_fingerprint(template)
    assert restored.execution == "bvh"
    assert restored.dense_workspace_bytes == 128
    points = np.asarray(((3.7, -1.5), (4.1, 1.0)), dtype=np.float64)
    expected_items, expected_squared = _reference(moved, edges, points, 2)
    actual = restored.nearest_segments(points, k=2)
    np.testing.assert_array_equal(actual.items, expected_items)
    np.testing.assert_allclose(actual.distance_squared, expected_squared, atol=1e-13)


def test_compiled_polygon_current_leaves_round_trip() -> None:
    vertices = jnp.asarray(
        ((0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)), dtype=jnp.float64
    )
    template = Polygon(vertices, feature_id="persisted-polygon").compile()
    parameter = template.schema.index(ParameterId("persisted-polygon", "vertices"))
    moved = vertices + jnp.asarray((3.0, -2.0), dtype=jnp.float64)
    geometry = template.with_state(template.state.replace_at(parameter, moved))
    payload = io.BytesIO()
    eqx.tree_serialise_leaves(payload, geometry)
    payload.seek(0)
    restored = eqx.tree_deserialise_leaves(payload, template)
    fingerprint = array_tree_fingerprint(geometry)
    assert array_tree_fingerprint(restored) == fingerprint
    assert array_tree_fingerprint(restored) != array_tree_fingerprint(template)
    paths = tuple(record["path"] for record in fingerprint["signature"])
    assert not any(path.endswith(".intersection_pairs") for path in paths)
    assert any(path.startswith(".kernel.index.bvh.") for path in paths)
    points = jnp.asarray(((3.5, -1.5), (5.5, -1.0)), dtype=jnp.float64)
    np.testing.assert_array_equal(restored.contains(points), (True, False))
    np.testing.assert_allclose(restored.boundary_field(points), (-0.5, 0.5), atol=1e-13)
    assert bool(restored.validity().accepted)
    assert restored.schema == template.schema
