#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import numpy as np
from jax import Array

from phydrax.meshing._collective_organization import (
    _bind_vertex_source_rows,
    _entity_membership_values,
    _initial_membership_values,
    _select_vertex_occurrence_values,
    _vertex_membership_values,
    _vertex_occurrence_values,
)
from tools.compile_counter import CompileCounter


def _selection_inputs(offset: int) -> tuple[Array, Array, Array, Array, Array]:
    identifiers, active, sources, weights, targets = (
        jax.device_put(value)
        for value in (
            np.asarray(((40, 10, 30), (10, 20, 99)), dtype=np.int64) + offset,
            np.asarray(((True, True, True), (True, True, False)), dtype=np.bool_),
            np.asarray(
                (
                    ((7, 9, -1), (7, -1, -1), (9, 12, -1)),
                    ((12, -1, -1), (7, 12, -1), (-1, -1, -1)),
                ),
                dtype=np.int64,
            ),
            np.asarray(
                (
                    ((0.5, 0.5, -0.0), (1.0, -0.0, 0.0), (0.25, 0.75, -0.0)),
                    ((1.0, 0.0, -0.0), (0.5, 0.5, -0.0), (0.0, 0.0, 0.0)),
                ),
                dtype=np.float64,
            ),
            np.asarray((10, 20, 30, 40), dtype=np.int64) + offset,
        )
    )
    return identifiers, active, sources, weights, targets


def test_vertex_selection_exact_values_and_dynamic_ids_compile_reuse() -> None:
    initial = _selection_inputs(0)
    changed = _selection_inputs(100)
    first = jax.block_until_ready(_select_vertex_occurrence_values(*initial))
    with CompileCounter() as compiler:
        repeated = jax.block_until_ready(_select_vertex_occurrence_values(*initial))
        dynamic = jax.block_until_ready(_select_vertex_occurrence_values(*changed))
    assert compiler.count == 0
    expected_ids = np.asarray(
        ((7, -1, -1), (7, 12, -1), (9, 12, -1), (7, 9, -1)), dtype=np.int64
    )
    expected_weights = np.asarray(
        ((1.0, -0.0, 0.0), (0.5, 0.5, -0.0), (0.25, 0.75, -0.0), (0.5, 0.5, -0.0)),
        dtype=np.float64,
    )
    for identifiers, weights in (first, repeated, dynamic):
        np.testing.assert_array_equal(identifiers, expected_ids)
        np.testing.assert_array_equal(
            np.asarray(weights).view(np.uint64), expected_weights.view(np.uint64)
        )


def test_vertex_source_rows_exact_binding_and_dynamic_count_compile_reuse() -> None:
    source_ids = jax.device_put(np.asarray((12, 7, 9), dtype=np.int64))
    sources = jax.device_put(np.asarray(((7, 9, -1), (12, -1, -1)), dtype=np.int64))
    absent = jax.device_put(np.asarray(((7, 9, -1), (99, -1, -1)), dtype=np.int64))
    changed_ids = jax.device_put(np.asarray((112, 107, 109), dtype=np.int64))
    changed_sources = jax.device_put(
        np.asarray(((107, 109, -1), (112, -1, -1)), dtype=np.int64)
    )
    full_count = jax.device_put(np.asarray(2, dtype=np.int64))
    partial_count = jax.device_put(np.asarray(1, dtype=np.int64))
    first = jax.block_until_ready(
        _bind_vertex_source_rows(source_ids, sources, full_count)
    )
    with CompileCounter() as compiler:
        repeated = jax.block_until_ready(
            _bind_vertex_source_rows(source_ids, sources, full_count)
        )
        dynamic = jax.block_until_ready(
            _bind_vertex_source_rows(changed_ids, changed_sources, partial_count)
        )
        _, rejected = jax.block_until_ready(
            _bind_vertex_source_rows(source_ids, absent, full_count)
        )
        _, masked = jax.block_until_ready(
            _bind_vertex_source_rows(source_ids, absent, partial_count)
        )
    assert compiler.count == 0
    for rows, passed in (first, repeated, dynamic):
        np.testing.assert_array_equal(rows, ((1, 2, -1), (0, -1, -1)))
        assert bool(passed)
    assert not bool(rejected)
    assert bool(masked)


def test_vertex_definition_membership_keeps_mask_and_count_dynamic() -> None:
    membership = jax.device_put(np.asarray([True, False, True]))
    sources = jax.device_put(
        np.asarray(
            [[10, 20, -1], [10, -1, -1], [20, 30, -1], [30, -1, -1]], dtype=np.int64
        )
    )
    rows = jax.device_put(
        np.asarray([[0, 1, -1], [0, -1, -1], [1, 2, -1], [2, -1, -1]], dtype=np.int64)
    )
    first = jax.block_until_ready(_vertex_membership_values(membership, sources, rows, 3))
    np.testing.assert_array_equal(first, [False, True, False, False])
    changed_membership = jax.device_put(np.asarray([True, True, False]))
    with CompileCounter() as compiler:
        repeated = _vertex_membership_values(membership, sources, rows, 3)
        changed = _vertex_membership_values(changed_membership, sources, rows, 4)
        jax.block_until_ready((repeated, changed))
    assert compiler.count == 0
    np.testing.assert_array_equal(repeated, first)
    np.testing.assert_array_equal(changed, [True, True, False, False])


def test_entity_definition_membership_keeps_rows_and_count_dynamic() -> None:
    membership = jax.device_put(np.asarray([True, False, True]))
    rows = jax.device_put(np.asarray([2, 0, -1, 1], dtype=np.int64))
    first = jax.block_until_ready(_entity_membership_values(membership, rows, 3))
    np.testing.assert_array_equal(first, [True, True, False, False])
    changed_membership = jax.device_put(np.asarray([False, True, True]))
    with CompileCounter() as compiler:
        repeated = _entity_membership_values(membership, rows, 3)
        changed = _entity_membership_values(changed_membership, rows, 4)
        jax.block_until_ready((repeated, changed))
    assert compiler.count == 0
    np.testing.assert_array_equal(repeated, first)
    np.testing.assert_array_equal(changed, [True, False, False, True])


def test_initial_memberships_fuse_dynamic_declarations_and_active_prefixes() -> None:
    sources = (
        jax.device_put(np.asarray((7, 9, 7, -1), dtype=np.int64)),
        jax.device_put(np.asarray((2, 3, 5, -1), dtype=np.int64)),
    )
    selections = (np.asarray((7,), dtype=np.int64), np.asarray((3, 5), dtype=np.int64))
    jax.clear_caches()
    with CompileCounter() as cold:
        first = jax.block_until_ready(
            _initial_membership_values(sources, selections, (3, 3))
        )
    with CompileCounter() as warm:
        changed = jax.block_until_ready(
            _initial_membership_values(
                sources,
                (np.asarray((9,), dtype=np.int64), np.asarray((2, 5), dtype=np.int64)),
                (2, 2),
            )
        )
    assert cold.count == 1
    assert warm.count == 0
    np.testing.assert_array_equal(first[0], (True, False, True, False))
    np.testing.assert_array_equal(first[1], (False, True, True, False))
    np.testing.assert_array_equal(changed[0], (False, True, False, False))
    np.testing.assert_array_equal(changed[1], (True, False, False, False))


def test_vertex_ancestry_selection_is_one_kernel_with_exact_dynamic_supports() -> None:
    ids = jax.device_put(np.asarray(((11, 23, 47, 59),), dtype=np.int64))
    parents = jax.device_put(
        np.asarray((((-1, -1), (-1, -1), (-1, -1), (0, 1)),), dtype=np.int32)
    )
    cursors = jax.device_put(np.asarray(((4, 0, 0, 0),), dtype=np.int64))
    active = jax.device_put(np.asarray(((True, True, True, True),), dtype=np.bool_))
    targets = jax.device_put(np.asarray((11, 23, 47, 59), dtype=np.int64))
    changed_ids = jax.device_put(np.asarray(((101, 103, 107, 109),), dtype=np.int64))
    changed_targets = jax.device_put(np.asarray((101, 103, 107, 109), dtype=np.int64))
    jax.clear_caches()
    with CompileCounter() as cold:
        first = jax.block_until_ready(
            _vertex_occurrence_values(ids, parents, cursors, active, targets, 3, None)
        )
    with CompileCounter() as warm:
        changed = jax.block_until_ready(
            _vertex_occurrence_values(
                changed_ids, parents, cursors, active, changed_targets, 3, None
            )
        )
    assert cold.count == 1
    assert warm.count == 0
    assert bool(first[2]) and bool(changed[2])
    np.testing.assert_array_equal(first[0][3], (11, 23, -1))
    np.testing.assert_array_equal(changed[0][3], (101, 103, -1))
    np.testing.assert_array_equal(first[1][3], (0.5, 0.5, 0.0))
    assert np.asarray(first[1]).tobytes() == np.asarray(changed[1]).tobytes()
