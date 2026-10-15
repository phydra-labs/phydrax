#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from phydrax import config as pxconfig
from phydrax.discretization.spatial._bvh_neighbor_query import (
    BVHNeighborQueryPlan,
    BVHNeighborQueryStatus,
    BVHSourcePlan,
)


def _oracle(
    points: np.ndarray,
    targets: np.ndarray,
    ids: np.ndarray,
    active: np.ndarray,
    k: int,
    *,
    periods: tuple[float | None, ...] | None = None,
    radii: np.ndarray | None = None,
    target_ids: np.ndarray | None = None,
) -> np.ndarray:
    result = np.full((targets.shape[0], k), -1, dtype=np.int32)
    for row, target in enumerate(targets):
        delta = target - points
        if periods is not None:
            for axis, period in enumerate(periods):
                if period is not None:
                    # Graph/Morton minimum-image arithmetic rounds to the
                    # nearest image; remainder-based wrapping has different
                    # float32 seam rounding and is not the declared formula.
                    delta[:, axis] -= np.round(delta[:, axis] / period) * period
        # Independent host calculation, intentionally no tree/bound logic.
        squared = np.sum(delta * delta, axis=1)
        allowed = active.copy()
        if radii is not None:
            allowed &= squared <= radii[row] ** 2
        if target_ids is not None:
            allowed &= ids != target_ids[row]
        indices = np.flatnonzero(allowed)
        order = np.lexsort((ids[indices], squared[indices]))
        chosen = indices[order[:k]]
        result[row, : chosen.size] = chosen
    return result


@pytest.mark.parametrize("dimension", (1, 2, 3, 6))
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_exact_against_numpy(dimension: int, dtype: type[np.floating]) -> None:
    rng = np.random.default_rng(19)
    points = rng.normal(size=(37, dimension)).astype(dtype)
    targets = rng.normal(size=(9, dimension)).astype(dtype)
    points[0:4] = 0
    targets[0] = 0
    ids = np.arange(37, dtype=np.int64)[::-1].copy()
    ids[:4] = [np.iinfo(np.int64).max, np.iinfo(np.int64).min, 1 << 50, -(1 << 50)]
    active = np.ones(37, dtype=np.bool_)
    active[4::5] = False
    source_plan = BVHSourcePlan(dimension, 37, leaf_size=4)
    sources = source_plan.prepare(points, active_mask=active, stable_ids=ids)
    query = BVHNeighborQueryPlan(source_plan, 9, 8, target_chunk_size=2)
    result = query.query(sources, targets)
    np.testing.assert_array_equal(
        result.source_indices, _oracle(points, targets, ids, active, 8)
    )
    assert bool(result.evidence.successful)
    assert bool(result.evidence.complete)
    assert sources.points.dtype == dtype
    # Selection stops gradients, but the artifact's logical geometry stays live.
    derivative = jax.grad(lambda x: jnp.sum(source_plan.prepare(x).points))(
        jnp.asarray(points)
    )
    np.testing.assert_array_equal(derivative, np.ones_like(points))


def test_radius_self_masks_and_maximum_id() -> None:
    points = np.asarray([[0.0], [1.0], [-1.0], [2.0], [9.0]], dtype=np.float64)
    ids = np.asarray([np.iinfo(np.int64).max, -7, 2, 10, 99], dtype=np.int64)
    source_plan = BVHSourcePlan(1, 5, leaf_size=1)
    sources = source_plan.prepare(points, stable_ids=ids)
    query = BVHNeighborQueryPlan(source_plan, 3, 6, target_chunk_size=1)
    targets = np.asarray([[0.0], [2.0], [9.0]], dtype=np.float64)
    target_ids = np.asarray([-100, 10, 99], dtype=np.int64)
    radii = np.asarray([1.0, np.inf, 0.0], dtype=np.float64)
    result = query.query(
        sources,
        targets,
        radius=2.0,
        target_radii=radii,
        target_stable_ids=target_ids,
        exclude_self=True,
        target_mask=np.asarray([True, True, False], dtype=np.bool_),
    )
    expected = _oracle(
        points,
        targets,
        ids,
        np.ones(5, dtype=np.bool_),
        6,
        radii=np.minimum(radii, 2.0),
        target_ids=target_ids,
    )
    expected[2] = -1
    np.testing.assert_array_equal(result.source_indices, expected)
    np.testing.assert_array_equal(result.counts, [3, 2, 0])
    assert result.status[2] == BVHNeighborQueryStatus.INACTIVE_TARGET
    # A maximum-valued ID alone is not an empty node or empty result.
    singleton = BVHSourcePlan(1, 1)
    one = BVHNeighborQueryPlan(singleton, 1, 1).query(
        singleton.prepare(points[:1], stable_ids=ids[:1]), targets[:1]
    )
    np.testing.assert_array_equal(one.source_indices, [[0]])


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_periodic_seams_and_out_of_box(dtype: type[np.floating]) -> None:
    points = np.asarray(
        [[0.99, 2.0], [0.01, 2.0], [12.02, 3.0], [-4.95, 2.0], [0.5, 4.0]], dtype=dtype
    )
    targets = np.asarray([[0.0, 2.0], [3.01, 3.0]], dtype=dtype)
    ids = np.asarray([5, 4, 3, 2, 1], dtype=np.uint64)
    periods = (1.0, None)
    source_plan = BVHSourcePlan(2, 5, leaf_size=1, periodic_lengths=periods)
    result = BVHNeighborQueryPlan(source_plan, 2, 5).query(
        source_plan.prepare(points, stable_ids=ids), targets
    )
    np.testing.assert_array_equal(
        result.source_indices,
        _oracle(points, targets, ids, np.ones(5, dtype=np.bool_), 5, periods=periods),
    )
    assert bool(result.evidence.successful)


def test_coincident_cloud_exact_identity() -> None:
    points = jnp.zeros((2048, 3), dtype=jnp.float64)
    source_plan = BVHSourcePlan(3, 2048)
    result = BVHNeighborQueryPlan(source_plan, 2, 8, target_chunk_size=1).query(
        source_plan.prepare(points), points[:2]
    )
    np.testing.assert_array_equal(result.source_indices, np.tile(np.arange(8), (2, 1)))
    assert bool(result.evidence.successful)


def test_invalid_sources_targets_and_duplicate_ids() -> None:
    points = jnp.asarray([[0.0], [jnp.nan], [2.0]], dtype=jnp.float64)
    targets = jnp.asarray([[0.0], [jnp.inf], [jnp.nan]], dtype=jnp.float64)
    source_plan = BVHSourcePlan(1, 3)
    query = BVHNeighborQueryPlan(source_plan, 3, 2)
    result = query.query(
        source_plan.prepare(points),
        targets,
        target_mask=jnp.asarray([True, True, False], dtype=jnp.bool_),
    )
    np.testing.assert_array_equal(
        result.status,
        [
            BVHNeighborQueryStatus.INVALID_SOURCES,
            BVHNeighborQueryStatus.INVALID_TARGET,
            BVHNeighborQueryStatus.INACTIVE_TARGET,
        ],
    )
    assert not bool(result.evidence.successful)
    assert not bool(result.evidence.finite)
    assert not bool(result.evidence.sources_valid)
    assert not np.any(result.valid)
    assert result.evidence.invalid_sources == 1
    # Invalid inactive coordinates do not invalidate a cloud.
    sources = source_plan.prepare(
        points, active_mask=jnp.asarray([True, False, True], dtype=jnp.bool_)
    )
    result = query.query(sources, jnp.zeros((3, 1), dtype=jnp.float64))
    assert bool(result.evidence.successful)
    duplicate = source_plan.prepare(
        jnp.zeros((3, 1), dtype=jnp.float64),
        stable_ids=jnp.asarray([1, 1, 2], dtype=jnp.int64),
    )
    result = query.query(duplicate, jnp.zeros((3, 1), dtype=jnp.float64))
    assert not bool(result.evidence.sources_valid)
    assert not np.any(result.valid)
    np.testing.assert_array_equal(
        result.status, np.full(3, BVHNeighborQueryStatus.INVALID_SOURCES)
    )
    empty = source_plan.prepare(points, active_mask=jnp.zeros(3, dtype=jnp.bool_))
    result = query.query(empty, jnp.zeros((3, 1), dtype=jnp.float64))
    assert bool(result.evidence.successful)
    np.testing.assert_array_equal(result.counts, [0, 0, 0])
    # Finite geometry whose squared distances cannot be represented is refused.
    huge = source_plan.prepare(jnp.full((3, 1), 1e200, dtype=jnp.float64))
    result = query.query(huge, jnp.zeros((3, 1), dtype=jnp.float64))
    assert not bool(result.evidence.sources_valid)
    assert bool(result.evidence.finite)


def test_self_exclusion_and_metadata_refusals() -> None:
    points = jnp.zeros((3, 1), dtype=jnp.float64)
    source_plan = BVHSourcePlan(1, 3)
    sources = source_plan.prepare(points)
    query = BVHNeighborQueryPlan(source_plan, 3, 2)
    result = query.query(sources, points, exclude_self=True)
    np.testing.assert_array_equal(result.source_indices, [[1, 2], [0, 2], [0, 1]])
    with pytest.raises(ValueError, match="unequal capacities"):
        BVHNeighborQueryPlan(source_plan, 1, 1).query(
            sources, points[:1], exclude_self=True
        )
    declared = source_plan.prepare(points, stable_ids=jnp.arange(3, dtype=jnp.int32))
    np.testing.assert_array_equal(
        query.query(declared, points, exclude_self=True).source_indices,
        [[1, 2], [0, 2], [0, 1]],
    )
    with pytest.raises(ValueError, match="source plan"):
        query.query(BVHSourcePlan(1, 3, leaf_size=1).prepare(points), points)
    with pytest.raises(TypeError, match="PreparedBVHSources"):
        query.query(points, points)  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError, match="precision"):
        query.query(sources, points.astype(jnp.float32))
    invalid_radius = query.query(
        sources, points, target_radii=jnp.asarray([0.0, -1.0, jnp.nan], dtype=jnp.float64)
    )
    np.testing.assert_array_equal(invalid_radius.status, [0, 2, 2])
    assert not bool(invalid_radius.evidence.successful)
    # Checked traversal failure must erase all potentially partial neighbors.
    narrow = BVHSourcePlan(1, 3, leaf_size=1)
    broken = narrow.prepare(points)
    broken = eqx.tree_at(
        lambda value: value.tree.left, broken, broken.tree.left.at[0].set(999)
    )
    refused = BVHNeighborQueryPlan(narrow, 3, 2).query(broken, points)
    assert not bool(refused.evidence.complete)
    assert not np.any(refused.valid)
    np.testing.assert_array_equal(
        refused.status, np.full(3, BVHNeighborQueryStatus.TRAVERSAL_FAILURE)
    )


def test_float64_one_ulp_and_conservative_bounds() -> None:
    near = np.nextafter(np.float64(1.0), np.float64(0.0))
    points = np.asarray([[1.0], [near], [-1.0], [0.0]], dtype=np.float64)
    ids = np.asarray([0, 100, -1, np.iinfo(np.int64).max], dtype=np.int64)
    source_plan = BVHSourcePlan(1, 4, leaf_size=1)
    targets = np.asarray([[0.0], [np.nextafter(0.0, 1.0)]], dtype=np.float64)
    result = BVHNeighborQueryPlan(source_plan, 2, 3).query(
        source_plan.prepare(points, stable_ids=ids), targets
    )
    np.testing.assert_array_equal(
        result.source_indices,
        _oracle(points, targets, ids, np.ones(4, dtype=np.bool_), 3),
    )


def test_gpu_warmed_build_query_transfer_guard() -> None:
    if jax.default_backend() != "gpu":
        pytest.skip("CUDA smoke requires a GPU runtime")
    source_plan = BVHSourcePlan(3, 33, leaf_size=4)
    query_plan = BVHNeighborQueryPlan(source_plan, 3, 5, target_chunk_size=1)

    @jax.jit
    def execute(points: Array, targets: Array) -> Array:
        return query_plan.query(source_plan.prepare(points), targets).source_indices

    points = jnp.arange(99, dtype=jnp.float64).reshape((33, 3)) / 99
    targets = points[:3]
    expected = execute(points, targets)
    expected.block_until_ready()
    with pxconfig.context(
        pxconfig.NumericsConfig(transfer_guard_device_to_host="disallow")
    ):
        actual = execute(points, targets)
        actual.block_until_ready()
    np.testing.assert_array_equal(actual, expected)


def test_unsigned_maximum_ids_and_mixed_sign_self_exclusion() -> None:
    points = jnp.zeros((4, 1), dtype=jnp.float64)
    ids = jnp.asarray([np.iinfo(np.uint64).max, 1 << 63, 0, 1], dtype=jnp.uint64)
    source_plan = BVHSourcePlan(1, 4, leaf_size=1)
    sources = source_plan.prepare(points, stable_ids=ids)
    query = BVHNeighborQueryPlan(source_plan, 2, 4)
    targets = points[:2]
    result = query.query(
        sources,
        targets,
        target_stable_ids=jnp.asarray([-1, 0], dtype=jnp.int64),
        exclude_self=True,
    )
    np.testing.assert_array_equal(result.source_indices, [[2, 3, 1, 0], [3, 1, 0, -1]])
    assert bool(result.evidence.successful)
    # Signed -1 must not equal uint64 maximum via wrapping conversion.
    signed = source_plan.prepare(
        points, stable_ids=jnp.asarray([-1, 0, 1, 2], dtype=jnp.int64)
    )
    result = query.query(
        signed,
        targets,
        target_stable_ids=jnp.asarray([np.iinfo(np.uint64).max, 0], dtype=jnp.uint64),
        exclude_self=True,
    )
    np.testing.assert_array_equal(result.source_indices, [[0, 1, 2, 3], [0, 2, 3, -1]])
