# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Root-bank and boundary predicates reuse numerical array signatures."""

import jax
import jax.numpy as jnp
import numpy as np

from phydrax.geometry._collective_domain_coverage import (
    _boundary_contact_values,
    _root_bank_masked_values_match,
    _root_bank_rows_match,
    _root_bank_values_match,
    _root_parent_rows_match,
)
from tools.compile_counter import CompileCounter


def test_root_bank_predicates_keep_values_and_rows_dynamic() -> None:
    values = jnp.asarray([[1, 2], [3, 4], [-1, -1]], dtype=jnp.int64)
    expected = jnp.asarray([[3, 4], [1, 2]], dtype=jnp.int64)
    rows = jnp.asarray([1, 0, 0], dtype=jnp.int32)
    valid = jnp.asarray([True, True, False])
    padded = jnp.asarray([[1, 2], [3, 4], [9, 9]], dtype=jnp.int64)
    first = (
        _root_bank_values_match(values, values),
        _root_bank_masked_values_match(valid, values, padded),
        _root_bank_rows_match(valid, values, expected, rows),
    )
    jax.block_until_ready(first)
    assert all(bool(value) for value in first)
    changed = values.at[0, 0].set(99)
    changed_rows = rows.at[0].set(0)
    with CompileCounter() as counter:
        refused = (
            _root_bank_values_match(values, changed),
            _root_bank_masked_values_match(valid, changed, padded),
            _root_bank_rows_match(valid, values, expected, changed_rows),
        )
        jax.block_until_ready(refused)
    assert counter.count == 0
    assert not any(bool(value) for value in refused)


def test_root_parent_predicate_keeps_lookup_findings_dynamic() -> None:
    valid = jnp.asarray([True, True, False])
    present = jnp.asarray([True, True, False])
    parents = jnp.asarray([10, 20, -1], dtype=jnp.int64)
    target = jnp.asarray([20, 10], dtype=jnp.int64)
    rows = jnp.asarray([1, 0, 0], dtype=jnp.int32)
    assert bool(_root_parent_rows_match(valid, present, present, target, parents, rows))
    absent = present.at[1].set(False)
    changed_parents = parents.at[1].set(99)
    with CompileCounter() as counter:
        missing = _root_parent_rows_match(valid, present, absent, target, parents, rows)
        mismatch = _root_parent_rows_match(
            valid, present, present, target, changed_parents, rows
        )
        jax.block_until_ready((missing, mismatch))
    assert counter.count == 0
    assert not bool(missing)
    assert not bool(mismatch)


def test_boundary_contact_predicate_keeps_faces_and_barycentrics_dynamic() -> None:
    barycentric = jnp.asarray(np.eye(3)[None], dtype=jnp.float64)
    boundary = jnp.asarray([[True, False, True]])
    single = boundary
    accepted = _boundary_contact_values(boundary, single, barycentric)
    jax.block_until_ready(accepted)
    assert bool(accepted)
    exterior_single = jnp.asarray([[True, True, True]])
    no_boundary = jnp.zeros_like(boundary)
    with CompileCounter() as counter:
        missing = _boundary_contact_values(boundary, exterior_single, barycentric)
        absent = _boundary_contact_values(no_boundary, single, barycentric)
        jax.block_until_ready((missing, absent))
    assert counter.count == 0
    assert not bool(missing)
    assert not bool(absent)
