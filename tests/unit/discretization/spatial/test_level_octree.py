from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp

from phydrax.discretization.spatial import MortonAddressPlan, SparseLevelOctreePlan


def _points():
    return jnp.asarray(
        [
            [0.0625, 0.0625],
            [0.1875, 0.8125],
            [0.3125, 0.3125],
            [0.4375, 0.9375],
            [0.5625, 0.0625],
            [0.6875, 0.6875],
            [0.8125, 0.3125],
            [0.9375, 0.9375],
        ]
    )


def test_sparse_level_octree_builds_bounded_far_and_near_relations() -> None:
    address = MortonAddressPlan((0.0, 0.0), (1.0, 1.0), 3)
    tree = SparseLevelOctreePlan(
        address,
        8,
        far_interaction_capacity=2048,
        near_interaction_capacity=512,
    ).prepare(_points())
    assert bool(tree.evidence.successful)
    assert int(tree.evidence.active_leaves) == 8
    assert int(tree.evidence.required_far_interactions) > 0
    assert int(tree.evidence.required_near_interactions) >= 8
    far_target = tree.far_targets[tree.far_active]
    far_source = tree.far_sources[tree.far_active]
    assert bool(
        jnp.all(
            tree.hierarchy.node_levels[far_target]
            == tree.hierarchy.node_levels[far_source]
        )
    )
    assert bool(jnp.all(far_target != far_source))
    near_target = tree.near_targets[tree.near_active]
    near_source = tree.near_sources[tree.near_active]
    assert bool(jnp.all(tree.hierarchy.node_is_leaf[near_target]))
    assert bool(jnp.all(tree.hierarchy.node_is_leaf[near_source]))


def test_sparse_level_octree_fails_closed_on_relation_capacity() -> None:
    address = MortonAddressPlan((0.0, 0.0), (1.0, 1.0), 3)
    plan = SparseLevelOctreePlan(
        address,
        8,
        far_interaction_capacity=1,
        near_interaction_capacity=1,
    )
    tree = eqx.filter_jit(plan.prepare)(_points())
    assert not bool(tree.evidence.successful)
    assert int(tree.evidence.required_far_interactions) > 1
    assert int(tree.evidence.required_near_interactions) > 1
