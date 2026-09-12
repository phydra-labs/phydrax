from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.discretization.spatial import (
    DistributedMortonNeighborQueryPlan,
    MortonAddressPlan,
    MortonNeighborQueryPlan,
)


def _address() -> MortonAddressPlan:
    return MortonAddressPlan(
        (0.0, 0.0),
        (1.0, 1.0),
        12,
        periodic_axes=(True, False),
    )


def test_distributed_morton_query_matches_single_device_authority() -> None:
    source = jnp.asarray(
        [[0.98, 0.5], [0.05, 0.5], [0.25, 0.2], [0.45, 0.4], [0.7, 0.8], [0.9, 0.1]]
    )
    target = jnp.asarray([[0.02, 0.5], [0.5, 0.5], [0.8, 0.2]])
    stable_ids = jnp.asarray([60, 10, 50, 20, 40, 30])
    distributed = DistributedMortonNeighborQueryPlan(
        _address(),
        6,
        3,
        3,
        1,
        maximum_leaf_occupancy=2,
        target_top_nodes=1,
    ).query(
        source,
        target,
        source_stable_ids=stable_ids,
        devices=(jax.devices()[0],),
    )
    authority = MortonNeighborQueryPlan(
        _address(),
        6,
        3,
        3,
        maximum_leaf_occupancy=2,
        target_top_nodes=1,
    ).query(source, target, source_stable_ids=stable_ids)

    assert bool(distributed.evidence.successful)
    np.testing.assert_array_equal(distributed.source_indices, authority.source_indices)
    np.testing.assert_array_equal(distributed.valid, authority.valid)
    np.testing.assert_array_equal(
        distributed.source_stable_ids,
        stable_ids[authority.source_indices],
    )


def test_distributed_morton_query_rejects_duplicate_global_ids() -> None:
    source = jnp.asarray([[0.1, 0.1], [0.2, 0.2], [0.8, 0.8], [0.9, 0.9]])
    target = jnp.asarray([[0.15, 0.15], [0.85, 0.85]])
    result = DistributedMortonNeighborQueryPlan(
        _address(),
        4,
        2,
        1,
        1,
        maximum_leaf_occupancy=2,
        target_top_nodes=1,
    ).query(
        source,
        target,
        source_stable_ids=jnp.asarray([1, 1, 2, 3]),
        devices=(jax.devices()[0],),
    )
    assert not bool(result.evidence.stable_ids_unique)
    assert not bool(result.evidence.successful)
    np.testing.assert_array_equal(result.valid, False)


def test_distributed_morton_query_merges_source_shards_when_available() -> None:
    devices = tuple(jax.devices()[:2])
    if len(devices) < 2:
        pytest.skip("requires two real or explicitly configured JAX devices")
    source = jnp.asarray(
        [[0.05, 0.2], [0.2, 0.2], [0.4, 0.2], [0.6, 0.2], [0.8, 0.2], [0.95, 0.2]]
    )
    target = jnp.asarray([[0.1, 0.2], [0.3, 0.2], [0.7, 0.2], [0.9, 0.2]])
    result = DistributedMortonNeighborQueryPlan(
        _address(),
        6,
        4,
        2,
        2,
        maximum_leaf_occupancy=2,
        target_top_nodes=1,
    ).query(source, target, devices=devices)
    authority = MortonNeighborQueryPlan(
        _address(),
        6,
        4,
        2,
        maximum_leaf_occupancy=2,
        target_top_nodes=1,
    ).query(source, target)
    assert bool(result.evidence.successful)
    np.testing.assert_array_equal(result.source_indices, authority.source_indices)
    np.testing.assert_array_equal(result.valid, authority.valid)
