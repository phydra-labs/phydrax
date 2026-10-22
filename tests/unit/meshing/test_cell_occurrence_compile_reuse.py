#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import numpy as np
from jax import Array

from phydrax.meshing._collective_organization import (
    _bind_cell_source_rows,
    _bind_entity_support_rows,
    _cell_occurrence_values,
    _select_cell_occurrence_values,
    _translate_uniform_cell_roots,
)
from tools.compile_counter import CompileCounter


def _ids(values: tuple[int, ...]) -> Array:
    return jax.device_put(np.asarray(values, dtype=np.int64))


def test_cell_translation_selection_exact_order_and_warm_reuse() -> None:
    roots = _ids((40, 10, 30, 10, 20, 99))
    children = _ids((30, 10, 40, 20)).reshape((2, 2))
    parents = _ids((7, 9))
    active = jax.device_put(np.asarray((True, True, True, True, True, False)))
    targets = _ids((10, 10, 20, 30))
    translated = jax.block_until_ready(
        _translate_uniform_cell_roots(roots, children, parents)
    )
    first = jax.block_until_ready(
        _select_cell_occurrence_values(translated, active, roots, targets)
    )
    changed_roots = _ids((140, 110, 130, 110, 120, 199))
    changed_children = _ids((130, 110, 140, 120)).reshape((2, 2))
    changed_parents = _ids((107, 109))
    changed_targets = _ids((110, 110, 120, 130))
    with CompileCounter() as compiler:
        repeated_roots = _translate_uniform_cell_roots(roots, children, parents)
        repeated = jax.block_until_ready(
            _select_cell_occurrence_values(repeated_roots, active, roots, targets)
        )
        dynamic_roots = _translate_uniform_cell_roots(
            changed_roots, changed_children, changed_parents
        )
        dynamic = jax.block_until_ready(
            _select_cell_occurrence_values(
                dynamic_roots, active, changed_roots, changed_targets
            )
        )
    assert compiler.count == 0
    np.testing.assert_array_equal(translated, (9, 7, 7, 7, 9, 99))
    for selected in (first, repeated):
        np.testing.assert_array_equal(selected[0], (7, 7, 9, 7))
        np.testing.assert_array_equal(selected[1], active)
        np.testing.assert_array_equal(selected[2], roots)
        np.testing.assert_array_equal(selected[3], (1, 3, 4, 2, 0, 5))
    np.testing.assert_array_equal(dynamic[0], (107, 107, 109, 107))
    np.testing.assert_array_equal(dynamic[3], first[3])


def test_cell_binding_exact_rows_missing_predicate_and_dynamic_count() -> None:
    source = _ids((30, 10, 20))
    occurrences = _ids((10, 20, 99))
    partial = jax.device_put(np.asarray(2, dtype=np.int64))
    full = jax.device_put(np.asarray(3, dtype=np.int64))
    first = jax.block_until_ready(_bind_cell_source_rows(source, occurrences, partial))
    changed_source = _ids((130, 110, 120))
    changed_occurrences = _ids((110, 120, 199))
    with CompileCounter() as compiler:
        repeated = jax.block_until_ready(
            _bind_cell_source_rows(source, occurrences, partial)
        )
        dynamic = jax.block_until_ready(
            _bind_cell_source_rows(changed_source, changed_occurrences, full)
        )
    assert compiler.count == 0
    for rows, passed in (first, repeated):
        np.testing.assert_array_equal(rows, (1, 2, 0))
        assert bool(passed)
    np.testing.assert_array_equal(dynamic[0], (1, 2, 0))
    assert not bool(dynamic[1])


def test_entity_support_binding_real_support_width_and_dynamic_prefix() -> None:
    vertices = _ids((10, 20, 30, 99))
    keys = _ids((10, 20, 20, 30, 10, 30)).reshape((3, 2))
    supports = _ids((7, -1, 9, -1, 12, -1, -1, -1)).reshape((4, 2))
    scientific = _ids((9, 12, 7, 9)).reshape((2, 2))
    count = jax.device_put(np.asarray(3, dtype=np.int64))
    changed_count = jax.device_put(np.asarray(4, dtype=np.int64))
    first = jax.block_until_ready(
        _bind_entity_support_rows(vertices, count, keys, supports, scientific)
    )
    changed_vertices = _ids((110, 120, 130, 199))
    changed_keys = _ids((110, 120, 120, 130, 110, 130)).reshape((3, 2))
    changed_supports = _ids((107, -1, 109, -1, 112, -1, -1, -1)).reshape((4, 2))
    changed_scientific = _ids((109, 112, 107, 109)).reshape((2, 2))
    with CompileCounter() as compiler:
        repeated = jax.block_until_ready(
            _bind_entity_support_rows(vertices, count, keys, supports, scientific)
        )
        dynamic = jax.block_until_ready(
            _bind_entity_support_rows(
                changed_vertices,
                changed_count,
                changed_keys,
                changed_supports,
                changed_scientific,
            )
        )
    assert compiler.count == 0
    for rows in (first, repeated, dynamic):
        np.testing.assert_array_equal(rows, (1, 0, -1))


def test_cell_ancestry_selection_fuses_cursor_projection_and_root_translation() -> None:
    ids = _ids((40, 10, 30, 20)).reshape((1, 4))
    parents = jax.device_put(np.asarray(((-1, -1, 0, 1),), dtype=np.int32))
    cursors = _ids((0, 4, 0, 0)).reshape((1, 4))
    active = jax.device_put(np.asarray(((False, True, True, True),), dtype=np.bool_))
    targets = _ids((10, 20, 30))
    uniform = (_ids((40, 10)).reshape((1, 2)), _ids((7,)))
    changed_uniform = (_ids((40, 10)).reshape((1, 2)), _ids((107,)))
    jax.clear_caches()
    with CompileCounter() as cold:
        first = jax.block_until_ready(
            _cell_occurrence_values(ids, parents, cursors, active, targets, uniform)
        )
    with CompileCounter() as warm:
        changed = jax.block_until_ready(
            _cell_occurrence_values(
                ids, parents, cursors, active, targets, changed_uniform
            )
        )
    assert cold.count == 1
    assert warm.count == 0
    np.testing.assert_array_equal(first[0], (7, 7, 7))
    np.testing.assert_array_equal(changed[0], (107, 107, 107))
    np.testing.assert_array_equal(first[1], (False, True, True, True))
    np.testing.assert_array_equal(first[2], (40, 10, 30, 20))
    np.testing.assert_array_equal(first[3], (1, 3, 2, 0))
