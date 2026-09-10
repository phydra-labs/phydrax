#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _index_space():
    plan = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformAxisSpec(8, periodic=True, endpoint=False),
            phx.discretization.UniformAxisSpec(8, periodic=True, endpoint=False),
        ),
        axis_names=("x", "y"),
    )
    bounds = jnp.asarray([[0.0, 0.0], [1.0, 1.0]])
    return plan.prepare(bounds), plan.prepare_index_space(bounds)


def test_tensor_index_space_matches_dense_indexed_geometry_without_dense_points():
    dense, index = _index_space()
    logical = jnp.asarray([0, 9, 63], dtype=jnp.int32)
    coordinates, supported = index.vertices().coordinates_at(logical)
    measures, measure_supported = index.vertices().measure_at(logical)

    np.testing.assert_allclose(coordinates, dense.points[logical])
    np.testing.assert_allclose(measures, dense.measure.weights[logical])
    assert jnp.all(supported & measure_supported)
    assert index.stored_axis_values == 32
    assert index.stored_axis_values < index.size


def test_sparse_blocks_wrap_periodic_closure_and_align_transitions():
    _, index = _index_space()
    plan = phx.discretization.SparseBlockTopologyPlan(
        index,
        (2, 2),
        8,
        layout=index.vertices(),
        closure_offsets=((0, 0), (-1, 0), (1, 0)),
    )
    first = jax.jit(lambda ids: plan.build(ids))(jnp.asarray([0, 1, 8]))
    transition = jax.jit(lambda previous, ids: plan.refresh(previous, ids))(
        first, jnp.asarray([54, 55, 62])
    )
    second = transition.candidate

    assert bool(first.evidence.successful)
    assert bool(second.evidence.successful)
    assert int(first.evidence.required_blocks) == 3
    assert int(second.generation) == 1
    assert bool(transition.key_transition.topology_changed)
    assert int(first.materialize_support().sum()) == 12


def test_sparse_block_overflow_returns_no_usable_node_support():
    _, index = _index_space()
    plan = phx.discretization.SparseBlockTopologyPlan(
        index,
        (2, 2),
        1,
        layout=index.vertices(),
    )
    state = plan.build(jnp.asarray([0, 18]))

    assert bool(state.evidence.overflow)
    assert not bool(state.evidence.successful)
    assert not jnp.any(state.node_valid)
    assert not jnp.any(state.lookup(jnp.asarray([0, 18])).supported)
