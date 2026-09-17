#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp

from phydrax._bvh import (
    build_packed_bvh,
    point_select_leaf_items,
    ray_select_leaf_items,
)


def _candidate_sets(indices, valid):
    return tuple(
        frozenset(int(value) for value in row[row_valid])
        for row, row_valid in zip(indices, valid, strict=True)
    )


def test_packed_bvh_query_chunks_preserve_candidates_and_completeness() -> None:
    lower = jnp.stack((jnp.arange(8, dtype=float), jnp.zeros(8), jnp.zeros(8)), axis=-1)
    upper = lower + jnp.asarray((0.75, 1.0, 1.0))
    bvh = build_packed_bvh(lower, upper, 0.5 * (lower + upper), leaf_size=2)
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
    bvh = build_packed_bvh(lower, upper, 0.5 * (lower + upper), leaf_size=1)
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
