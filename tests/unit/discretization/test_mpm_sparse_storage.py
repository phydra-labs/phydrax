#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _routes(position):
    grid_plan = phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformAxisSpec(16, periodic=True, endpoint=False)
            for _ in range(2)
        ),
        axis_names=("x", "y"),
    )
    bounds = jnp.asarray([[0.0, 0.0], [1.0, 1.0]])
    grid = grid_plan.prepare(bounds)
    index_space = grid_plan.prepare_index_space(bounds)
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(position.shape[0]),
        jnp.ones((position.shape[0],)),
        ambient_dimension=2,
    ).prepare()
    splat = phx.discretization.ParticleGridSplatPlan(
        grid, assignment=phx.discretization.TensorBSplineSplatAssignment(2)
    ).prepare(particles)
    return splat.build(position), index_space


def _storage(index_space, capacity):
    topology = phx.discretization.SparseBlockTopologyPlan(
        index_space,
        (4, 4),
        capacity,
        layout=index_space.vertices(),
    )
    return phx.discretization.BlockSparseMPMNodalStoragePlan(topology)


def test_sparse_blocks_follow_routes_and_transition_by_logical_key():
    first_routes, index_space = _routes(jnp.asarray([[0.15, 0.15], [0.2, 0.2]]))
    second_routes, _ = _routes(jnp.asarray([[0.75, 0.75], [0.8, 0.8]]))
    storage = _storage(index_space, 16)
    first = storage.build(first_routes)
    second = storage.build(second_routes, first)

    assert bool(first.evidence.successful)
    assert bool(second.evidence.successful)
    assert int(first.evidence.required_blocks) > 0
    assert int(second.evidence.required_blocks) > 0
    assert int(second.generation) == 1
    assert not jnp.array_equal(first.groups.group_keys, second.groups.group_keys)


def test_compact_pack_unpack_and_route_mapping_match_supported_dense_values():
    routes, index_space = _routes(jnp.asarray([[0.15, 0.15], [0.2, 0.2]]))
    storage = _storage(index_space, 8)
    topology = storage.build(routes)
    dense = jnp.arange(16 * 16 * 2, dtype=jnp.float64).reshape((16, 16, 2))
    compact = storage.pack(dense, topology)
    restored = storage.unpack(compact, topology)
    mapped = storage.mapped_stencil(routes, topology)
    support = topology.materialize_support()

    np.testing.assert_array_equal(
        jnp.where(support[..., None], restored, 0.0),
        jnp.where(support[..., None], dense, 0.0),
    )
    assert mapped.indices.shape == routes.stencil.indices.shape
    assert jnp.all(mapped.valid == routes.stencil.valid)
    assert compact.shape == (8 * 16, 2)


def test_sparse_block_overflow_rejects_before_storage_use():
    routes, index_space = _routes(
        jnp.asarray([[0.1, 0.1], [0.4, 0.4], [0.7, 0.7], [0.9, 0.9]])
    )
    topology = _storage(index_space, 1).build(routes)

    assert bool(topology.evidence.overflow)
    assert not bool(topology.evidence.successful)
    assert not jnp.any(topology.node_valid)


def test_dense_storage_adapter_is_identity_for_field_payloads():
    dense = jnp.ones((16, 16, 2, 3))
    storage = phx.discretization.DenseMPMNodalStoragePlan((16, 16))
    np.testing.assert_array_equal(storage.unpack(storage.pack(dense, None), None), dense)
