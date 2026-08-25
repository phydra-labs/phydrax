#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import numpy as np

from phydrax.geometry._bvh_overlap import (
    build_host_aabb_overlap_bvh,
    OverlapSearchStatus,
    query_host_aabb_overlaps,
)


def _query(index, lower, upper, **kwargs):
    return query_host_aabb_overlaps(
        index,
        np.asarray(lower, dtype=np.float64),
        np.asarray(upper, dtype=np.float64),
        **kwargs,
    )


def test_identity_and_containment_emit_source_and_target_ids():
    index = build_host_aabb_overlap_bvh(
        [[0.0, 0.0], [3.0, 3.0]],
        [[2.0, 2.0], [4.0, 4.0]],
        global_ids=[20, 10],
        local_ids=[7, 3],
    )
    result = _query(
        index,
        [[0.5, 0.5], [10.0, 10.0]],
        [[1.5, 1.5], [11.0, 11.0]],
        source_global_ids=[101, 102],
        source_local_ids=[5, 2],
    )
    assert result.status is OverlapSearchStatus.SUCCESS
    assert result.candidate_count == 1
    assert result.source_global_ids.tolist() == [101]
    assert result.target_global_ids.tolist() == [20]
    assert result.source_local_ids.tolist() == [5]
    assert result.target_local_ids.tolist() == [7]
    assert result.content_identity


def test_touching_boxes_are_excluded_unless_zero_measure_is_requested():
    index = build_host_aabb_overlap_bvh([[0.0, 0.0]], [[1.0, 1.0]], global_ids=[4])
    lower = [[1.0, 0.0]]
    upper = [[2.0, 1.0]]
    assert _query(index, lower, upper).candidate_count == 0
    result = _query(index, lower, upper, include_zero_measure=True)
    assert result.candidate_count == 1


def test_candidates_are_stably_sorted_and_permutation_invariant():
    index_a = build_host_aabb_overlap_bvh(
        [[0.0, 0.0], [0.0, 0.0]],
        [[3.0, 3.0], [2.0, 2.0]],
        global_ids=[30, 10],
        local_ids=[1, 8],
    )
    index_b = build_host_aabb_overlap_bvh(
        [[0.0, 0.0], [0.0, 0.0]],
        [[2.0, 2.0], [3.0, 3.0]],
        global_ids=[10, 30],
        local_ids=[8, 1],
    )
    lower = [[0.0, 0.0], [1.0, 1.0]]
    upper = [[1.5, 1.5], [2.5, 2.5]]
    kwargs = {"source_global_ids": [20, 5], "source_local_ids": [9, 2]}
    first = _query(index_a, lower, upper, **kwargs)
    second = _query(index_b, lower, upper, **kwargs)
    assert first.status is second.status is OverlapSearchStatus.SUCCESS
    expected = [(5, 10), (5, 30), (20, 10), (20, 30)]
    assert list(zip(first.source_global_ids, first.target_global_ids)) == expected
    assert np.array_equal(first.source_global_ids, second.source_global_ids)
    assert np.array_equal(first.target_global_ids, second.target_global_ids)
    assert first.content_identity == second.content_identity


def test_tolerance_covers_positive_measure_boundary_roundoff():
    index = build_host_aabb_overlap_bvh(
        [[0.0, 0.0]], [[1.0, 1.0]], global_ids=[1], absolute_tolerance=1e-10
    )
    result = _query(index, [[1.0 - 5e-12, 0.0]], [[2.0, 1.0]], source_global_ids=[2])
    assert result.candidate_count == 1


def test_candidate_and_memory_limits_fail_closed():
    index = build_host_aabb_overlap_bvh([[0.0, 0.0]], [[2.0, 2.0]], global_ids=[1])
    kwargs = {"source_global_ids": [2, 3], "source_local_ids": [0, 1]}
    candidate_limited = _query(
        index,
        [[0.0, 0.0], [0.0, 0.0]],
        [[1.0, 1.0], [1.0, 1.0]],
        max_candidates=1,
        **kwargs,
    )
    memory_limited = _query(
        index,
        [[0.0, 0.0]],
        [[1.0, 1.0]],
        max_memory_bytes=31,
        source_global_ids=[2],
        source_local_ids=[0],
    )
    assert candidate_limited.status is OverlapSearchStatus.CANDIDATE_LIMIT
    assert candidate_limited.candidate_count == 0
    assert memory_limited.status is OverlapSearchStatus.MEMORY_LIMIT
    assert memory_limited.candidate_count == 0


def test_invalid_bounds_fail_closed():
    index = build_host_aabb_overlap_bvh([[0.0, np.nan]], [[1.0, 1.0]], global_ids=[1])
    assert index.status is OverlapSearchStatus.INVALID_BOUNDS
    result = _query(index, [[0.0, 0.0]], [[1.0, 1.0]])
    assert result.status is OverlapSearchStatus.INVALID_BOUNDS
    assert result.candidate_count == 0


def test_repeated_queries_are_deterministic():
    index = build_host_aabb_overlap_bvh(
        [[-1.0, -1.0], [0.0, 0.0]],
        [[2.0, 2.0], [3.0, 3.0]],
        global_ids=[9, 3],
    )
    first = _query(index, [[0.5, 0.5]], [[2.5, 2.5]], source_global_ids=[77])
    second = _query(index, [[0.5, 0.5]], [[2.5, 2.5]], source_global_ids=[77])
    assert first.status is second.status is OverlapSearchStatus.SUCCESS
    assert first.content_identity == second.content_identity
    assert np.array_equal(first.source_global_ids, second.source_global_ids)
    assert np.array_equal(first.target_global_ids, second.target_global_ids)
