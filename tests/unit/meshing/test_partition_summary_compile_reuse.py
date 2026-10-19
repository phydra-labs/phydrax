# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Partition numerical summaries and retained exterior flags reuse capacity buckets."""

import jax
import jax.numpy as jnp
import numpy as np

from phydrax._execution_runtime import ExecutionRuntime
from phydrax.discretization._adaptive_simplex import AdaptiveSimplexParts
from phydrax.meshing._device_adaptation import (
    _adaptive_execution_parts,
    _retained_exterior_values,
)
from phydrax.meshing._distribution import _partition_summary_values
from tools.compile_counter import CompileCounter


def test_partition_summary_retains_exact_owner_cut_and_halo_counts() -> None:
    owners = jnp.asarray([0, 1, 1, -1], dtype=jnp.int32)
    vertices = jnp.asarray(
        [[0, 1, 2], [1, 3, 2], [4, 5, 6], [-1, -1, -1]], dtype=jnp.int64
    )
    closure_owners = jnp.asarray([[0, 1, 1, -1], [1, 0, -1, -1]], dtype=jnp.int32)
    closure_valid = closure_owners >= 0
    actual = _partition_summary_values(owners, vertices, closure_owners, closure_valid, 3)
    jax.block_until_ready(actual)
    np.testing.assert_array_equal(actual[0], [1.0, 2.0])
    assert int(actual[1]) == 1
    assert int(actual[2]) == 3
    changed_owners = jnp.asarray([1, 0, 0, -1], dtype=jnp.int32)
    with CompileCounter() as counter:
        changed = _partition_summary_values(
            changed_owners, vertices, closure_owners, closure_valid, 3
        )
        smaller = _partition_summary_values(
            owners, vertices, closure_owners, closure_valid, 2
        )
        jax.block_until_ready((changed, smaller))
    assert counter.count == 0
    np.testing.assert_array_equal(changed[0], [2.0, 1.0])
    np.testing.assert_array_equal(smaller[0], [1.0, 1.0])
    assert int(changed[1]) == int(smaller[1]) == 1
    assert int(changed[2]) == int(smaller[2]) == 3


def test_retained_exterior_keeps_packet_values_and_ids_dynamic() -> None:
    ids = jnp.asarray([[0, 5, -1], [3, 7, -1]], dtype=jnp.int64)
    active = ids >= 0
    valid = active
    flags = jnp.asarray(
        [
            [[True, False, False], [False, True, False], [False, False, True]],
            [[False, False, True], [True, False, True], [True, True, True]],
        ]
    )
    actual = _retained_exterior_values(ids, active, ids, valid, flags)
    jax.block_until_ready(actual)
    np.testing.assert_array_equal(
        actual, np.asarray(flags) & np.asarray(active)[..., None]
    )
    changed_ids = ids[:, jnp.asarray([1, 0, 2])]
    changed_flags = ~flags
    with CompileCounter() as counter:
        changed = _retained_exterior_values(
            changed_ids, active, ids, valid, changed_flags
        )
        jax.block_until_ready(changed)
    assert counter.count == 0
    np.testing.assert_array_equal(
        changed, np.asarray(changed_flags)[:, [1, 0, 2]] & np.asarray(active)[..., None]
    )


def test_internal_adaptive_parts_reuse_only_compatible_root_execution_mesh() -> None:
    devices = tuple(jax.devices())
    routes = tuple(
        (source, target)
        for source in range(len(devices))
        for target in range(len(devices))
        if source != target
    )
    actual = _adaptive_execution_parts(devices, neighbor_pairs=routes)
    root = ExecutionRuntime.current().root_group
    assert actual.mesh == root.mesh
    assert actual.axis_name == root.mesh.axis_names[0]
    assert actual.neighbor_pairs == routes
    if len(devices) > 1:
        subset = _adaptive_execution_parts(devices[:1], neighbor_pairs=())
        declared = AdaptiveSimplexParts(devices[:1], neighbor_pairs=())
        assert subset.mesh == declared.mesh
        assert subset.axis_name == declared.axis_name
    explicit = AdaptiveSimplexParts(
        devices, axis_name="explicit-owner", neighbor_pairs=routes
    )
    assert explicit.axis_name == "explicit-owner"
