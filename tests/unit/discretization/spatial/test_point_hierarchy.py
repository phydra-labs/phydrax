from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from phydrax.discretization.spatial import (
    MortonAddressPlan,
    MortonPointHierarchyPlan,
    MortonPointHierarchyState,
)


def _plan(*, nodes: int | None = None) -> MortonPointHierarchyPlan:
    return MortonPointHierarchyPlan(
        MortonAddressPlan((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 4),
        6,
        node_capacity=nodes,
        target_leaf_occupancy=2,
    )


def test_point_hierarchy_contracts() -> None:
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
    assert bool(jnp.all(hierarchy.logical_point_leaf_slots >= 0))
    np.testing.assert_array_equal(
        hierarchy.sorted_point_leaf_slots,
        hierarchy.logical_point_leaf_slots[hierarchy.storage_to_logical],
    )
    active_children = hierarchy.node_children[hierarchy.node_active]
    assert bool(
        jnp.all((active_children < 0) | (active_children < hierarchy.node_active.size))
    )
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
    points = jnp.full((6, 3), 0.25)
    empty = _plan().build(points, active_mask=jnp.zeros((6,), dtype="bool"))
    assert bool(empty.evidence.successful)
    assert int(empty.evidence.active_nodes) == 0
    assert int(empty.root_slot) == -1

    coincident = _plan().build(points)
    assert bool(coincident.evidence.successful)
    assert int(coincident.evidence.active_leaves) == 1
    assert int(coincident.evidence.maximum_leaf_occupancy) == 6
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


def _assert_same_hierarchy(
    actual: MortonPointHierarchyState, expected: MortonPointHierarchyState
) -> None:
    for actual_leaf, expected_leaf in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        np.testing.assert_array_equal(actual_leaf, expected_leaf)


@pytest.mark.parametrize(
    ("coordinates", "active", "ids", "expected_refit"),
    [
        ([0.1, 0.2, 0.6, 0.7], None, None, True),
        ([0.15, 0.3, 0.65, 0.75], None, None, True),
        ([0.2, 0.1, 0.6, 0.7], None, None, False),
        ([0.1, 0.55, 0.6, 0.7], None, None, False),
        ([0.1, 0.2, 0.6, 0.7], [True, False, True, True], None, False),
        ([0.1, 0.2, 0.6, 0.7], None, [24, 21, 23, 22], False),
    ],
    ids=[
        "unchanged",
        "codes-change-within-leaves",
        "order-change",
        "leaf-crossing-with-same-order",
        "mask-change",
        "id-change-with-same-order",
    ],
)
def test_hierarchy_refresh_matches_canonical_build(
    coordinates: list[float],
    active: list[bool] | None,
    ids: list[int] | None,
    expected_refit: bool,
) -> None:
    plan = MortonPointHierarchyPlan(
        MortonAddressPlan((0.0,), (1.0,), 4),
        4,
        target_leaf_occupancy=2,
    )
    initial_points = jnp.asarray([[0.1], [0.2], [0.6], [0.7]], dtype=jnp.float64)
    initial_ids = jnp.asarray([14, 11, 13, 12], dtype=jnp.int64)
    previous = plan.build(initial_points, stable_ids=initial_ids, epoch=5)
    points = jnp.asarray(coordinates, dtype=jnp.float64)[:, None]
    active_mask = None if active is None else jnp.asarray(active, dtype=jnp.bool_)
    stable_ids = None if ids is None else jnp.asarray(ids, dtype=jnp.int64)
    transition = eqx.filter_jit(plan.refresh)(
        previous, points, active_mask=active_mask, stable_ids=stable_ids
    )
    assert bool(transition.accepted_candidate)
    assert bool(transition.refitted) is expected_refit
    assert bool(transition.rebuilt) is not expected_refit
    assert int(transition.accepted.epoch) == 6
    fresh = plan.build(
        points,
        active_mask=previous.point_order.active if active_mask is None else active_mask,
        stable_ids=initial_ids if stable_ids is None else stable_ids,
        epoch=6,
    )
    _assert_same_hierarchy(transition.candidate, fresh)
    _assert_same_hierarchy(transition.accepted, fresh)
    np.testing.assert_array_equal(transition.accepted.point_order.points, points)


@pytest.mark.parametrize("failure", ["domain", "duplicate-ids", "capacity"])
def test_hierarchy_refresh_rolls_back_rejected_candidate(failure: str) -> None:
    plan = MortonPointHierarchyPlan(
        MortonAddressPlan((0.0,), (1.0,), 4),
        4,
        node_capacity=3,
        target_leaf_occupancy=2,
    )
    points = jnp.asarray([[0.1], [0.2], [0.6], [0.7]], dtype=jnp.float64)
    previous = plan.build(points, epoch=7)
    assert bool(previous.evidence.successful)
    stable_ids = None
    match failure:
        case "domain":
            points = points.at[1, 0].set(1.1)
        case "duplicate-ids":
            stable_ids = jnp.asarray([0, 1, 1, 3], dtype=jnp.int64)
        case "capacity":
            points = points.at[1, 0].set(0.55)
        case _:
            raise AssertionError(failure)
    transition = eqx.filter_jit(plan.refresh)(previous, points, stable_ids=stable_ids)
    assert not bool(transition.accepted_candidate)
    assert not bool(transition.refitted)
    assert not bool(transition.rebuilt)
    assert not bool(transition.candidate.evidence.successful)
    assert int(transition.candidate.epoch) == 8
    np.testing.assert_array_equal(transition.candidate.point_order.points, points)
    _assert_same_hierarchy(transition.accepted, previous)


def test_hierarchy_refresh_preserves_live_geometry_gradients() -> None:
    plan = MortonPointHierarchyPlan(
        MortonAddressPlan((0.0,), (1.0,), 4),
        4,
        target_leaf_occupancy=2,
    )
    initial = jnp.asarray([[0.1], [0.2], [0.6], [0.7]], dtype=jnp.float64)
    previous = plan.build(initial)
    points = initial + 1.0e-3
    direction = jnp.asarray([[0.2], [-0.3], [0.4], [-0.1]], dtype=jnp.float64)

    def objective(coordinates: Array) -> Array:
        hierarchy = plan.refresh(previous, coordinates).accepted
        return jnp.sum(hierarchy.point_order.sorted_coordinates**2) + jnp.sum(
            hierarchy.node_centers
        )

    gradient = eqx.filter_jit(jax.grad(objective))(points)
    np.testing.assert_allclose(gradient, 2 * points, rtol=1.0e-13, atol=1.0e-13)
    _, tangent = jax.jvp(objective, (points,), (direction,))
    np.testing.assert_allclose(tangent, jnp.sum(2 * points * direction), rtol=1.0e-13)


def test_hierarchy_refresh_retains_omitted_mask_and_ids() -> None:
    plan = MortonPointHierarchyPlan(
        MortonAddressPlan((0.0,), (1.0,), 4),
        4,
        target_leaf_occupancy=2,
    )
    points = jnp.asarray([[0.1], [0.2], [0.6], [0.7]], dtype=jnp.float64)
    active = jnp.asarray([True, False, True, True], dtype=jnp.bool_)
    identifiers = jnp.asarray([41, 43, 40, 42], dtype=jnp.int64)
    previous = plan.build(points, active_mask=active, stable_ids=identifiers)
    moved = points + 1.0e-3
    transition = eqx.filter_jit(plan.refresh)(previous, moved)
    assert bool(transition.refitted)
    np.testing.assert_array_equal(transition.accepted.point_order.active, active)
    np.testing.assert_array_equal(transition.accepted.point_order.stable_ids, identifiers)
    _assert_same_hierarchy(
        transition.accepted,
        plan.build(moved, active_mask=active, stable_ids=identifiers, epoch=1),
    )
