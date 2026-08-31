#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _routes(position):
    grid = phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformAxisSpec(16, periodic=True, endpoint=False)
            for _ in range(2)
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(position.shape[0]), jnp.ones((position.shape[0],)), ambient_dimension=2
    ).prepare()
    splat = phx.discretization.ParticleGridSplatPlan(
        grid, assignment=phx.discretization.TensorBSplineSplatAssignment(2)
    ).prepare(particles)
    return splat.build(position)


def test_active_blocks_follow_routes_halo_and_previous_union():
    first_routes = _routes(jnp.asarray([[0.15, 0.15], [0.2, 0.2]]))
    second_routes = _routes(jnp.asarray([[0.75, 0.75], [0.8, 0.8]]))
    plan = phx.discretization.MPMActiveBlockPlan((16, 16), (4, 4), 16, halo_blocks=1)
    first = plan.build(first_routes)
    second = plan.build(second_routes, first)

    assert bool(first.successful)
    assert bool(second.successful)
    assert int(first.active_block_count) > 0
    assert int(second.active_block_count) > 0
    assert jnp.all(second.current_previous_union >= first.active_block_mask)
    assert jnp.any(first.active_node_mask)


def test_compact_block_pack_unpack_and_route_mapping_match_dense_values():
    routes = _routes(jnp.asarray([[0.15, 0.15], [0.2, 0.2]]))
    blocks = phx.discretization.MPMActiveBlockPlan((16, 16), (4, 4), 16)
    active = blocks.build(routes)
    storage = phx.discretization.BlockSparseMPMNodalStoragePlan(blocks)
    dense = jnp.arange(16 * 16 * 2, dtype=jnp.float64).reshape((16, 16, 2))
    compact = storage.pack(dense, active)
    restored = storage.unpack(compact, active)
    indices, valid = storage.map_route_indices(routes, active)

    np.testing.assert_array_equal(
        jnp.where(active.active_node_mask[..., None], restored, 0.0),
        jnp.where(active.active_node_mask[..., None], dense, 0.0),
    )
    assert indices.shape == routes.stencil.indices.shape
    assert jnp.all(valid == routes.stencil.valid)
    assert compact.shape == (16, 16, 2)


def test_active_block_overflow_rejects_before_storage_use():
    routes = _routes(jnp.asarray([[0.1, 0.1], [0.4, 0.4], [0.7, 0.7], [0.9, 0.9]]))
    plan = phx.discretization.MPMActiveBlockPlan((16, 16), (4, 4), 1, halo_blocks=0)
    active = plan.build(routes)

    assert bool(active.overflow)
    assert not bool(active.successful)


def test_dense_storage_adapter_is_identity_for_field_payloads():
    routes = _routes(jnp.asarray([[0.2, 0.2]]))
    active = phx.discretization.MPMActiveBlockPlan((16, 16), (4, 4), 16).build(routes)
    dense = jnp.ones((16, 16, 2, 3))
    storage = phx.discretization.DenseMPMNodalStoragePlan((16, 16))
    np.testing.assert_array_equal(
        storage.unpack(storage.pack(dense, active), active), dense
    )
