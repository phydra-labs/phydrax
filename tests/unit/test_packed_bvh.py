#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._bvh import (
    bvh_nearest_items,
    bvh_overlap_pairs,
    bvh_overlap_pairs_host,
    BVHBuildKind,
    BVHBuildPolicy,
    point_select_leaf_items,
    prepare_bvh,
    ray_select_leaf_items,
    refit_packed_bvh_bounds,
)


_KINDS = tuple(BVHBuildKind)


def _candidate_sets(indices, valid):
    return tuple(
        frozenset(int(value) for value in row[row_valid])
        for row, row_valid in zip(indices, valid, strict=True)
    )


def _random_boxes(seed: int, count: int, dimension: int = 3, scale: float = 0.08):
    rng = np.random.default_rng(seed)
    lower = rng.random((count, dimension))
    return lower, lower + scale * rng.random((count, dimension))


def _brute_pairs(first_lower, first_upper, second_lower, second_upper, touching):
    extent = np.minimum(first_upper[:, None], second_upper[None]) - np.maximum(
        first_lower[:, None], second_lower[None]
    )
    hit = np.all(extent >= 0.0 if touching else extent > 0.0, axis=-1)
    return np.argwhere(hit)


def test_packed_bvh_query_chunks_preserve_candidates_and_completeness() -> None:
    lower = jnp.stack(
        (jnp.arange(8, dtype="float64"), jnp.zeros(8), jnp.zeros(8)), axis=-1
    )
    upper = lower + jnp.asarray((0.75, 1.0, 1.0))
    bvh = prepare_bvh(lower, upper, policy=BVHBuildPolicy(leaf_size=2))
    points = jnp.asarray(((0.25, 0.5, 0.5), (3.25, 0.5, 0.5), (7.25, 0.5, 0.5)))

    chunked = point_select_leaf_items(
        points,
        bvh=bvh,
        maximum_candidates=4,
        query_batch_capacity=1,
    )
    vectorized = point_select_leaf_items(
        points,
        bvh=bvh,
        maximum_candidates=4,
        query_batch_capacity=8,
    )

    assert _candidate_sets(*chunked[:2]) == _candidate_sets(*vectorized[:2])
    assert jnp.all(chunked[2])
    assert jnp.array_equal(chunked[2], vectorized[2])


def test_packed_bvh_ray_chunks_preserve_candidates() -> None:
    lower = jnp.asarray(((0.0, -1.0, -1.0), (2.0, -1.0, -1.0)))
    upper = jnp.asarray(((1.0, 1.0, 1.0), (3.0, 1.0, 1.0)))
    bvh = prepare_bvh(lower, upper, policy=BVHBuildPolicy(leaf_size=1))
    origins = jnp.asarray(((-1.0, 0.0, 0.0), (1.5, 0.0, 0.0)))
    directions = jnp.asarray(((1.0, 0.0, 0.0), (1.0, 0.0, 0.0)))

    indices, valid, complete = ray_select_leaf_items(
        origins,
        directions,
        bvh=bvh,
        maximum_candidates=2,
        query_batch_capacity=1,
    )

    assert _candidate_sets(indices, valid) == (frozenset((0, 1)), frozenset((1,)))
    assert jnp.all(complete)


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("count", (1, 7, 300))
def test_every_build_partitions_items_into_bounded_nested_leaves(kind, count) -> None:
    lower, upper = _random_boxes(count, count)
    bvh = prepare_bvh(
        lower, upper, policy=BVHBuildPolicy(kind, leaf_size=4), dtype=jnp.float64
    )

    leaf_items = np.asarray(bvh.leaf_items)
    assert leaf_items.shape[1] == 4
    assert np.array_equal(np.sort(leaf_items[leaf_items >= 0]), np.arange(count))
    left = np.asarray(bvh.left)
    right = np.asarray(bvh.right)
    node_min = np.asarray(bvh.bbox_min)
    node_max = np.asarray(bvh.bbox_max)
    for level in range(bvh.max_depth):
        nodes = np.arange(bvh.level_offsets[level], bvh.level_offsets[level + 1])
        parents = nodes[left[nodes] >= 0]
        children = np.concatenate((left[parents], right[parents]))
        assert np.all(children >= bvh.level_offsets[level + 1])
        assert np.all(children < bvh.level_offsets[level + 2])
        for child in (left[parents], right[parents]):
            assert np.all(node_min[parents] <= node_min[child])
            assert np.all(node_max[parents] >= node_max[child])
    leaf_nodes = np.asarray(bvh.leaf_node)
    for leaf, node in enumerate(leaf_nodes):
        items = leaf_items[leaf][leaf_items[leaf] >= 0]
        assert np.all(node_min[node] <= lower[items])
        assert np.all(node_max[node] >= upper[items])


def test_morton_build_resolves_identical_centers_by_item_index() -> None:
    lower = np.zeros((40, 2))
    upper = np.ones((40, 2))
    bvh = prepare_bvh(
        lower, upper, policy=BVHBuildPolicy(BVHBuildKind.MORTON, leaf_size=3)
    )

    leaf_items = np.asarray(bvh.leaf_items)
    assert np.array_equal(np.sort(leaf_items[leaf_items >= 0]), np.arange(40))
    # Identical codes split by sorted position, so the tree stays balanced.
    assert bvh.max_depth <= 6


def test_narrow_storage_rounds_bounds_outward() -> None:
    lower = np.asarray(((0.1, 1.0 / 3.0), (0.7, 2.0 / 3.0)))
    upper = lower + np.asarray((1.0e-9, 0.2))
    bvh = prepare_bvh(lower, upper, dtype=jnp.float32)

    assert np.all(np.asarray(bvh.item_bbox_min, dtype=np.float64) <= lower)
    assert np.all(np.asarray(bvh.item_bbox_max, dtype=np.float64) >= upper)


@pytest.mark.parametrize("kind", _KINDS)
def test_nearest_items_match_brute_force_with_index_ordered_ties(kind) -> None:
    rng = np.random.default_rng(3)
    points = rng.random((200, 3))
    points[:20] = points[100:120]
    queries = np.concatenate((rng.random((30, 3)), points[105:107]))
    bvh = prepare_bvh(
        points, points, policy=BVHBuildPolicy(kind, leaf_size=8), dtype=jnp.float64
    )

    result = bvh_nearest_items(bvh, queries, k=3, query_batch_capacity=8)

    distance = np.sum((queries[:, None] - points[None]) ** 2, axis=-1)
    expected = np.stack(
        [np.lexsort((np.arange(points.shape[0]), row))[:3] for row in distance]
    )
    assert np.array_equal(np.asarray(result.items), expected)
    np.testing.assert_allclose(
        np.asarray(result.distance_squared), np.take_along_axis(distance, expected, 1)
    )
    assert np.asarray(result.items)[-2:, :2].tolist() == [[5, 105], [6, 106]]


def test_nearest_items_mark_missing_neighbors() -> None:
    points = np.asarray(((0.0, 0.0), (1.0, 0.0)))
    bvh = prepare_bvh(points, points, dtype=jnp.float64)

    result = bvh_nearest_items(bvh, np.asarray((0.9, 0.0)), k=3)

    assert np.asarray(result.items).tolist() == [1, 0, -1]
    assert np.isinf(np.asarray(result.distance_squared)[2])


@pytest.mark.parametrize("kind", _KINDS)
def test_refit_matches_rebuild_queries_and_differentiates_bounds(kind) -> None:
    lower, upper = _random_boxes(5, 150, scale=0.02)
    policy = BVHBuildPolicy(kind, leaf_size=4)
    bvh = prepare_bvh(lower, upper, policy=policy, dtype=jnp.float64)
    rng = np.random.default_rng(6)
    moved_lower = lower * np.asarray((1.5, 0.5, 1.0)) + 0.1 * rng.standard_normal(
        lower.shape
    )
    moved_upper = moved_lower + (upper - lower)
    queries = rng.random((25, 3))

    refitted = refit_packed_bvh_bounds(bvh, moved_lower, moved_upper)
    rebuilt = prepare_bvh(moved_lower, moved_upper, policy=policy, dtype=jnp.float64)

    first = bvh_nearest_items(refitted, queries, k=2)
    second = bvh_nearest_items(rebuilt, queries, k=2)
    assert np.array_equal(np.asarray(first.items), np.asarray(second.items))
    np.testing.assert_allclose(
        np.asarray(first.distance_squared), np.asarray(second.distance_squared)
    )
    np.testing.assert_allclose(np.asarray(refitted.bbox_min[0]), moved_lower.min(0))
    gradient = jax.grad(
        lambda values: refit_packed_bvh_bounds(bvh, values, values + 0.02).bbox_min[0, 0]
    )(jnp.asarray(moved_lower))
    expected = np.zeros_like(moved_lower)
    expected[np.argmin(moved_lower[:, 0]), 0] = 1.0
    np.testing.assert_allclose(np.asarray(gradient), expected)


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("touching", (False, True))
def test_overlap_pairs_return_every_brute_force_pair(kind, touching) -> None:
    first_lower, first_upper = _random_boxes(11, 120)
    second_lower, second_upper = _random_boxes(12, 90)
    # Exactly touching boxes separate the strict and touching predicates.
    second_lower[:10] = (
        first_upper[:10] * np.asarray((1.0, 0.0, 0.0))
        + np.asarray((0.0, 1.0, 1.0)) * first_lower[:10]
    )
    second_upper[:10] = second_lower[:10] + 0.05
    expected = _brute_pairs(
        first_lower, first_upper, second_lower, second_upper, touching
    )
    first = prepare_bvh(
        first_lower,
        first_upper,
        policy=BVHBuildPolicy(kind, leaf_size=4),
        dtype=jnp.float64,
    )
    second = prepare_bvh(
        second_lower,
        second_upper,
        policy=BVHBuildPolicy(kind, leaf_size=5),
        dtype=jnp.float64,
    )

    host = np.stack(
        bvh_overlap_pairs_host(first, second, include_touching=touching), axis=1
    )
    device = bvh_overlap_pairs(
        first,
        second,
        capacity=expected.shape[0] + 3,
        include_touching=touching,
        traversal_width=8,
    )

    assert np.array_equal(host, expected)
    count = int(device.count)
    assert count == expected.shape[0] and not bool(device.overflow)
    assert int(np.sum(np.asarray(device.valid))) == count
    assert np.array_equal(
        np.stack(
            (
                np.asarray(device.first_items)[:count],
                np.asarray(device.second_items)[:count],
            ),
            axis=1,
        ),
        expected,
    )


def test_overlap_pair_overflow_is_flagged_with_the_true_count() -> None:
    lower, upper = _random_boxes(21, 60, scale=0.3)
    bvh = prepare_bvh(lower, upper, policy=BVHBuildPolicy(leaf_size=4))
    expected = _brute_pairs(lower, upper, lower, upper, False)

    result = bvh_overlap_pairs(bvh, bvh, capacity=5)

    assert bool(result.overflow)
    assert int(result.count) == expected.shape[0]
    retained = np.stack(
        (np.asarray(result.first_items)[:5], np.asarray(result.second_items)[:5]),
        axis=1,
    )
    assert {tuple(row) for row in retained} <= {tuple(row) for row in expected}


def test_host_self_overlap_returns_ordered_pairs_and_tolerance_gaps() -> None:
    lower = np.asarray(((0.0, 0.0), (1.0 + 1.0e-3, 0.0), (5.0, 5.0)))
    upper = lower + 1.0
    bvh = prepare_bvh(lower, upper, dtype=jnp.float64)

    strict = bvh_overlap_pairs_host(bvh, bvh)
    padded = bvh_overlap_pairs_host(
        bvh, bvh, include_touching=True, absolute_tolerance=2.0e-3
    )

    assert np.stack(strict, axis=1).tolist() == [[0, 0], [1, 1], [2, 2]]
    assert np.stack(padded, axis=1).tolist() == [[0, 0], [0, 1], [1, 0], [1, 1], [2, 2]]
    with pytest.raises(ValueError, match="include_touching"):
        bvh_overlap_pairs_host(bvh, bvh, absolute_tolerance=1.0e-3)
