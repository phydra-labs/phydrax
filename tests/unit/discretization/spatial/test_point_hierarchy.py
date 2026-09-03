from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from phydrax.discretization.spatial import MortonAddressPlan, MortonPointHierarchyPlan


def _plan(*, nodes: int | None = None) -> MortonPointHierarchyPlan:
    return MortonPointHierarchyPlan(
        MortonAddressPlan((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 4),
        6,
        node_capacity=nodes,
        target_leaf_occupancy=2,
    )


def test_point_hierarchy_partitions_active_points() -> None:
    points = jnp.asarray(
        [
            [0.1, 0.1, 0.1],
            [0.12, 0.1, 0.1],
            [0.8, 0.8, 0.8],
            [0.82, 0.8, 0.8],
            [0.5, 0.2, 0.7],
            [0.4, 0.6, 0.3],
        ]
    )
    hierarchy = _plan().build(points, stable_ids=jnp.asarray([4, 1, 5, 0, 3, 2]))
    assert bool(hierarchy.evidence.successful)
    leaf_count = int(hierarchy.evidence.active_leaves)
    leaf_items = int(
        jnp.sum(jnp.where(hierarchy.node_is_leaf, hierarchy.node_item_counts, 0))
    )
    assert leaf_count >= 3
    assert leaf_items == points.shape[0]
    active_children = hierarchy.node_children[hierarchy.node_active]
    assert bool(
        jnp.all((active_children < 0) | (active_children < hierarchy.node_active.size))
    )


def test_point_hierarchy_is_permutation_invariant_by_stable_id() -> None:
    points = jnp.asarray(
        [
            [0.1, 0.2, 0.3],
            [0.7, 0.2, 0.4],
            [0.1, 0.8, 0.6],
            [0.8, 0.7, 0.9],
            [0.3, 0.4, 0.5],
            [0.6, 0.5, 0.4],
        ]
    )
    stable_ids = jnp.asarray([10, 11, 12, 13, 14, 15])
    permutation = jnp.asarray([4, 2, 5, 0, 3, 1])
    first = _plan().build(points, stable_ids=stable_ids)
    second = _plan().build(points[permutation], stable_ids=stable_ids[permutation])
    np.testing.assert_array_equal(first.sorted_stable_ids, second.sorted_stable_ids)
    np.testing.assert_array_equal(first.node_prefixes, second.node_prefixes)
    np.testing.assert_array_equal(first.node_levels, second.node_levels)
    np.testing.assert_array_equal(first.node_active, second.node_active)


def test_point_hierarchy_handles_empty_and_coincident_points() -> None:
    points = jnp.full((6, 3), 0.25)
    empty = _plan().build(points, active_mask=jnp.zeros((6,), dtype=bool))
    assert bool(empty.evidence.successful)
    assert int(empty.evidence.active_nodes) == 0
    assert int(empty.root_slot) == -1

    coincident = _plan().build(points)
    assert bool(coincident.evidence.successful)
    assert int(coincident.evidence.active_leaves) == 1
    assert int(coincident.evidence.maximum_leaf_occupancy) == 6


def test_point_hierarchy_rejects_invalid_and_capacity_exhaustion() -> None:
    points = jnp.asarray(
        [
            [0.1, 0.1, 0.1],
            [0.9, 0.9, 0.9],
            [0.2, 0.8, 0.4],
            [0.7, 0.2, 0.6],
            [0.4, 0.7, 0.2],
            [1.1, 0.5, 0.5],
        ]
    )
    invalid = _plan().build(points)
    assert not bool(invalid.evidence.successful)
    assert int(invalid.evidence.invalid_points) == 1

    exhausted = _plan(nodes=1).build(points.at[-1].set(jnp.asarray([0.6, 0.4, 0.8])))
    assert not bool(exhausted.evidence.successful)
    assert int(exhausted.evidence.required_nodes) > 1


def test_point_hierarchy_rejects_duplicate_ids_across_distinct_cells() -> None:
    points = jnp.asarray(
        [
            [0.1, 0.1, 0.1],
            [0.9, 0.9, 0.9],
            [0.2, 0.8, 0.4],
            [0.7, 0.2, 0.6],
            [0.4, 0.7, 0.2],
            [0.6, 0.4, 0.8],
        ]
    )
    hierarchy = _plan().build(points, stable_ids=jnp.asarray([4, 7, 1, 7, 2, 3]))
    assert not bool(hierarchy.evidence.stable_ids_unique)
    assert not bool(hierarchy.evidence.successful)


def test_point_hierarchy_build_and_refresh_jit() -> None:
    plan = _plan()
    points = jnp.asarray(
        [
            [0.1, 0.1, 0.1],
            [0.11, 0.1, 0.1],
            [0.8, 0.8, 0.8],
            [0.81, 0.8, 0.8],
            [0.3, 0.6, 0.2],
            [0.6, 0.3, 0.7],
        ]
    )
    build = eqx.filter_jit(plan.build)
    hierarchy = build(points)
    assert bool(hierarchy.evidence.successful)
    refresh = eqx.filter_jit(plan.refresh)
    refitted = refresh(hierarchy, points + 1.0e-4)
    assert bool(refitted.accepted_candidate)
    assert bool(refitted.refitted)
