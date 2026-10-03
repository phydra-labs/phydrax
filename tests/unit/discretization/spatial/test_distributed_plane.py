from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._execution_runtime import ExecutionGroup, ExecutionRuntime
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


def _group(device_count: int) -> ExecutionGroup:
    devices = jax.devices()
    if len(devices) < device_count or len(devices) % device_count:
        pytest.skip(f"requires {device_count} real or explicitly configured JAX devices")
    return ExecutionRuntime.current().child_groups(len(devices) // device_count)[0]


def test_distributed_morton_contracts() -> None:
    source = jnp.asarray(
        [[0.98, 0.5], [0.05, 0.5], [0.25, 0.2], [0.45, 0.4], [0.7, 0.8], [0.9, 0.1]]
    )
    target = jnp.asarray([[0.02, 0.5], [0.5, 0.5], [0.8, 0.2]])
    stable_ids = jnp.asarray([60, 10, 50, 20, 40, 30])
    single = _group(1)
    distributed = DistributedMortonNeighborQueryPlan(
        _address(),
        6,
        3,
        3,
        single,
    ).query(
        source,
        target,
        source_stable_ids=stable_ids,
    )
    authority = MortonNeighborQueryPlan(
        _address(),
        6,
        3,
        3,
    ).query(source, target, source_stable_ids=stable_ids)

    assert bool(distributed.evidence.successful)
    np.testing.assert_array_equal(distributed.source_indices, authority.source_indices)
    np.testing.assert_array_equal(distributed.valid, authority.valid)
    np.testing.assert_array_equal(
        distributed.source_stable_ids,
        stable_ids[authority.source_indices],
    )
    source = jnp.asarray([[0.1, 0.1], [0.2, 0.2], [0.8, 0.8], [0.9, 0.9]])
    target = jnp.asarray([[0.15, 0.15], [0.85, 0.85]])
    result = DistributedMortonNeighborQueryPlan(
        _address(),
        4,
        2,
        1,
        single,
    ).query(
        source,
        target,
        source_stable_ids=jnp.asarray([1, 1, 2, 3]),
    )
    assert not bool(result.evidence.stable_ids_unique)
    assert not bool(result.evidence.successful)
    np.testing.assert_array_equal(result.valid, False)


@pytest.mark.parametrize("device_count", (2, 4), ids=("two-owners", "four-owners"))
def test_distributed_morton_matches_single_device_authority(device_count: int) -> None:
    group = _group(device_count)
    rng = np.random.default_rng(device_count)
    source = jnp.asarray(rng.uniform(0.0, 1.0, (24, 2)))
    target = jnp.asarray(rng.uniform(0.0, 1.0, (8, 2)))
    result = DistributedMortonNeighborQueryPlan(
        _address(),
        24,
        8,
        3,
        group,
    ).query(source, target)
    authority = MortonNeighborQueryPlan(
        _address(),
        24,
        8,
        3,
    ).query(source, target)
    assert bool(result.evidence.successful)
    np.testing.assert_array_equal(result.source_indices, authority.source_indices)
    np.testing.assert_array_equal(result.valid, authority.valid)
    # Target rows are sent only to owners reached by their certified shells.
    assert int(result.evidence.communicated_targets) <= 8 * (device_count - 1)
