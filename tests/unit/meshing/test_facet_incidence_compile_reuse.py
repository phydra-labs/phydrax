# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Facet joins retain incidence and multiplicity with warm numerical reuse."""

import jax
import jax.numpy as jnp
import numpy as np

from phydrax.geometry._collective_domain_coverage import (
    _facet_incidence_values,
    _facet_multiplicity_values,
    _lookup,
    _root_barycentric_values,
)
from tools.compile_counter import CompileCounter


def test_facet_incidence_preserves_exact_join_and_warm_zero() -> None:
    cells = jnp.asarray([[0, 1, 2], [1, 3, 2]], dtype=jnp.int64)
    keys = jnp.asarray([[0, 1], [0, 2], [1, 2], [1, 3], [2, 3]], dtype=jnp.int64)
    columns = jnp.asarray([[1, 2], [0, 2], [0, 1]], dtype=jnp.int32)
    incidence, accepted = _facet_incidence_values(cells, keys, columns)
    counts, multiplicity = _facet_multiplicity_values(incidence, keys)
    jax.block_until_ready((incidence, accepted, counts, multiplicity))
    np.testing.assert_array_equal(incidence, [2, 1, 0, 4, 2, 3])
    np.testing.assert_array_equal(counts, [1, 1, 2, 1, 1])
    assert bool(accepted) and bool(multiplicity)
    changed_cells, changed_keys = cells + 10, keys + 10
    absent_keys = keys.at[2].set(jnp.asarray([5, 6], dtype=jnp.int64))
    invalid_incidence = jnp.zeros_like(incidence)
    with CompileCounter() as counter:
        renamed, renamed_accepted = _facet_incidence_values(
            changed_cells, changed_keys, columns
        )
        _, absent_accepted = _facet_incidence_values(cells, absent_keys, columns)
        _, invalid_multiplicity = _facet_multiplicity_values(invalid_incidence, keys)
        jax.block_until_ready(
            (renamed, renamed_accepted, absent_accepted, invalid_multiplicity)
        )
    assert counter.count == 0
    np.testing.assert_array_equal(renamed, incidence)
    assert bool(renamed_accepted)
    assert not bool(absent_accepted)
    assert not bool(invalid_multiplicity)


def test_root_barycentric_join_keeps_support_values_dynamic() -> None:
    corners = jnp.asarray([[10, 20, 30], [20, 40, 30]], dtype=jnp.int64)
    rows = jnp.asarray([1, 0], dtype=jnp.int32)
    support = jnp.broadcast_to(corners[rows, None, :], (2, 3, 3))
    weights = jnp.asarray(
        [[[0.5, 0.5, 0], [0, 1, 0], [0, 0, 1]], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]],
        dtype=jnp.float64,
    )
    vertices, actual, accepted = _root_barycentric_values(support, weights, corners, rows)
    jax.block_until_ready((vertices, actual, accepted))
    np.testing.assert_array_equal(vertices, corners[rows])
    np.testing.assert_array_equal(actual, weights)
    assert bool(accepted)
    changed_support = support + 100
    changed_corners = corners + 100
    absent_support = support.at[0, 0].set(99)
    with CompileCounter() as counter:
        changed = _root_barycentric_values(
            changed_support, weights, changed_corners, rows
        )
        _, _, refused = _root_barycentric_values(absent_support, weights, corners, rows)
        jax.block_until_ready((changed, refused))
    assert counter.count == 0
    np.testing.assert_array_equal(changed[1], weights)
    assert bool(changed[2])
    assert not bool(refused)


def test_coverage_lookup_keeps_scientific_identifiers_dynamic() -> None:
    keys = jnp.asarray([30, 10, 20], dtype=jnp.int64)
    query = jnp.asarray([20, 10, 99], dtype=jnp.int64)
    actual = _lookup(keys, query)
    jax.block_until_ready(actual)
    np.testing.assert_array_equal(actual[0], [2, 1, 0])
    np.testing.assert_array_equal(actual[1], [True, True, False])
    changed_keys, changed_query = keys + 100, query + 100
    with CompileCounter() as counter:
        changed = _lookup(changed_keys, changed_query)
        jax.block_until_ready(changed)
    assert counter.count == 0
    for before, after in zip(actual, changed, strict=True):
        np.testing.assert_array_equal(before, after)
