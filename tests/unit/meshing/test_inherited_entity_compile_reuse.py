# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Stable numerical ownership and inherited scientific entity joins."""

import jax
import jax.numpy as jnp
import numpy as np

from phydrax.meshing._device_adaptation import (
    _inherited_entity_values,
    _source_solver_owner_values,
)
from tools.compile_counter import CompileCounter


def test_source_owner_join_keeps_ids_and_ownership_dynamic() -> None:
    identifiers = jnp.asarray([30, 10, 20], dtype=jnp.int64)
    ownership = jnp.asarray([1, 0, 1], dtype=jnp.int32)
    cell_ids = jnp.asarray([[10, 30, -1], [20, 10, -1]], dtype=jnp.int64)
    active = cell_ids >= 0
    actual, accepted = _source_solver_owner_values(
        identifiers, ownership, cell_ids, active
    )
    jax.block_until_ready((actual, accepted))
    np.testing.assert_array_equal(actual, [[0, 1, -1], [1, 0, -1]])
    np.testing.assert_array_equal(accepted, np.ones(cell_ids.shape, dtype=np.bool_))
    changed_ownership = 1 - ownership
    missing_ids = cell_ids.at[0, 0].set(99)
    with CompileCounter() as counter:
        changed, accepted = _source_solver_owner_values(
            identifiers, changed_ownership, cell_ids, active
        )
        _, refused = _source_solver_owner_values(
            identifiers, ownership, missing_ids, active
        )
        jax.block_until_ready((changed, accepted, refused))
    assert counter.count == 0
    np.testing.assert_array_equal(changed, [[1, 0, -1], [0, 1, -1]])
    np.testing.assert_array_equal(accepted, np.ones(cell_ids.shape, dtype=np.bool_))
    np.testing.assert_array_equal(refused, [[False, True, True], [True, True, True]])


def test_inherited_entity_join_preserves_predecessor_ids_and_warm_zero() -> None:
    table = jnp.asarray([[1, 2], [2, 3], [5, 6], [-1, -1]], dtype=jnp.int64)
    keys = (
        jnp.asarray([[1, 2], [4, 5]], dtype=jnp.int64),
        jnp.asarray([[1, 2], [2, 3], [3, 4]], dtype=jnp.int64),
    )
    identifiers = (
        jnp.asarray([10, 40], dtype=jnp.int64),
        jnp.asarray([10, 20, 30], dtype=jnp.int64),
    )
    actual = _inherited_entity_values(table, 3, keys, identifiers, partition_count=2)
    jax.block_until_ready(actual)
    np.testing.assert_array_equal(actual[0], [10, 20, 41, -1])
    np.testing.assert_array_equal(actual[1][:2], [[3, 4], [4, 5]])
    np.testing.assert_array_equal(actual[2][:2], [30, 40])
    np.testing.assert_array_equal(actual[1][2:], -np.ones((4, 2), dtype=np.int64))
    np.testing.assert_array_equal(actual[2][2:], -np.ones(4, dtype=np.int64))
    changed_identifiers = tuple(value + 100 for value in identifiers)
    with CompileCounter() as counter:
        repeated = _inherited_entity_values(
            table, 3, keys, identifiers, partition_count=2
        )
        changed = _inherited_entity_values(
            table, 2, keys, changed_identifiers, partition_count=2
        )
        jax.block_until_ready((repeated, changed))
    assert counter.count == 0
    for before, after in zip(actual, repeated, strict=True):
        np.testing.assert_array_equal(before, after)
    np.testing.assert_array_equal(changed[0], [110, 120, -1, -1])
    np.testing.assert_array_equal(changed[2][:2], [130, 140])
