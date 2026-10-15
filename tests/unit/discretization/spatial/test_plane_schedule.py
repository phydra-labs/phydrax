from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from phydrax.discretization.spatial import MortonAddressPlan
from phydrax.discretization.spatial._plane_schedule import (
    MortonPlaneSchedulePlan,
    MortonPlaneScheduleState,
)


def _plan(*, nodes: int | None = None) -> MortonPlaneSchedulePlan:
    return MortonPlaneSchedulePlan(
        MortonAddressPlan((0.0, 0.0), (1.0, 1.0), 8),
        8,
        node_capacity=nodes,
        maximum_leaf_occupancy=2,
        coarsening_factor=2,
        target_top_nodes=1,
    )


def _points() -> jnp.ndarray:
    return jnp.asarray(
        [
            [0.05, 0.10],
            [0.15, 0.20],
            [0.25, 0.30],
            [0.35, 0.40],
            [0.55, 0.60],
            [0.65, 0.70],
            [0.75, 0.80],
            [0.85, 0.90],
        ]
    )


def test_plane_schedule_contracts() -> None:
    points = _points()
    schedule = _plan().build(points, stable_ids=jnp.asarray([7, 1, 6, 2, 5, 3, 4, 0]))
    assert bool(schedule.evidence.successful)
    assert int(schedule.evidence.active_points) == points.shape[0]
    assert int(schedule.evidence.active_leaves) == 4
    np.testing.assert_array_equal(schedule.plane_active_counts, [4, 2, 1])
    np.testing.assert_array_equal(schedule.plane_offsets, [0, 4, 6, 7])
    assert bool(jnp.all(schedule.logical_point_leaf_slots >= 0))
    np.testing.assert_array_equal(
        schedule.sorted_point_leaf_slots,
        schedule.logical_point_leaf_slots[schedule.point_order.storage_to_logical],
    )

    for node in np.flatnonzero(np.asarray(schedule.node_active)):
        start = int(schedule.node_item_starts[node])
        count = int(schedule.node_item_counts[node])
        center = np.asarray(schedule.node_centers[node])
        half_width = np.asarray(schedule.node_half_widths[node])
        logical = np.asarray(
            schedule.point_order.storage_to_logical[start : start + count]
        )
        contained = np.asarray(points)[logical]
        assert np.all(contained >= center - half_width - 1.0e-12)
        assert np.all(contained <= center + half_width + 1.0e-12)
        child_count = int(schedule.node_child_counts[node])
        if child_count:
            child_start = int(schedule.node_child_starts[node])
            children = np.arange(child_start, child_start + child_count)
            np.testing.assert_array_equal(schedule.node_parents[children], node)
            assert int(jnp.sum(schedule.node_item_counts[children])) == count
    points = _points()
    stable_ids = jnp.asarray([12, 17, 11, 16, 10, 15, 13, 14])
    permutation = jnp.asarray([6, 2, 7, 0, 5, 1, 4, 3])
    first = _plan().build(points, stable_ids=stable_ids)
    second = _plan().build(points[permutation], stable_ids=stable_ids[permutation])
    np.testing.assert_array_equal(
        first.point_order.sorted_stable_ids,
        second.point_order.sorted_stable_ids,
    )
    np.testing.assert_array_equal(first.node_prefixes, second.node_prefixes)
    np.testing.assert_array_equal(first.node_bit_levels, second.node_bit_levels)
    np.testing.assert_array_equal(first.node_item_counts, second.node_item_counts)
    coincident = jnp.full((8, 2), 0.25)
    terminal = _plan().build(coincident)
    assert bool(terminal.evidence.successful)
    assert int(terminal.evidence.oversized_terminal_buckets) == 1
    assert int(terminal.evidence.maximum_terminal_bucket_occupancy) == 8
    assert int(terminal.evidence.maximum_leaf_occupancy) == 2

    duplicate_ids = _plan().build(
        coincident, stable_ids=jnp.asarray([0, 1, 2, 3, 4, 5, 6, 6])
    )
    assert not bool(duplicate_ids.evidence.successful)
    assert not bool(duplicate_ids.evidence.stable_ids_unique)

    invalid = _plan().build(coincident.at[-1].set(jnp.asarray([1.1, 0.5])))
    assert not bool(invalid.evidence.successful)
    assert int(invalid.evidence.invalid_points) == 1

    exhausted = _plan(nodes=2).build(_points())
    assert not bool(exhausted.evidence.successful)
    assert int(exhausted.evidence.required_nodes) == 7
    plan = _plan()
    points = _points()
    build = eqx.filter_jit(plan.build)
    empty = build(points, active_mask=jnp.zeros((8,), dtype="bool"))
    assert bool(empty.evidence.successful)
    assert int(empty.evidence.active_nodes) == 0
    np.testing.assert_array_equal(empty.logical_point_leaf_slots, -1)

    initial = build(points)
    refresh = eqx.filter_jit(plan.refresh)
    refitted = refresh(initial, points + 1.0e-5)
    assert bool(refitted.accepted_candidate)
    assert bool(refitted.refitted)
    assert int(refitted.accepted.epoch) == 1

    rejected = refresh(initial, points.at[-1].set(jnp.asarray([1.5, 0.5])))
    assert not bool(rejected.accepted_candidate)
    assert int(rejected.accepted.epoch) == 0


def _assert_same_schedule(
    actual: MortonPlaneScheduleState, expected: MortonPlaneScheduleState
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
        ([0.1, 0.55, 0.6, 0.7], None, None, True),
        ([0.25, 0.25, 0.25, 0.25], None, None, True),
        ([0.2, 0.1, 0.6, 0.7], None, None, False),
        ([0.1, 0.2, 0.6, 0.7], [True, False, True, True], None, False),
        ([0.1, 0.2, 0.6, 0.7], None, [200, 201, 202, 203], False),
    ],
    ids=[
        "unchanged",
        "codes-change",
        "common-prefix-change-with-fixed-groups",
        "terminal-bucket-evidence-change",
        "order-change",
        "mask-change",
        "id-change-with-same-order",
    ],
)
def test_plane_refresh_matches_canonical_build(
    coordinates: list[float],
    active: list[bool] | None,
    ids: list[int] | None,
    expected_refit: bool,
) -> None:
    plan = MortonPlaneSchedulePlan(
        MortonAddressPlan((0.0,), (1.0,), 4),
        4,
        maximum_leaf_occupancy=2,
        coarsening_factor=2,
        target_top_nodes=1,
    )
    initial_points = jnp.asarray([[0.1], [0.2], [0.6], [0.7]], dtype=jnp.float64)
    initial_ids = jnp.asarray([100, 101, 102, 103], dtype=jnp.int64)
    previous = plan.build(initial_points, stable_ids=initial_ids, epoch=5)
    points = jnp.asarray(coordinates, dtype=jnp.float64)[:, None]
    active_mask = None if active is None else jnp.asarray(active, dtype=jnp.bool_)
    stable_ids = None if ids is None else jnp.asarray(ids, dtype=jnp.int64)
    padding = jnp.asarray([0.015], dtype=jnp.float64)
    transition = eqx.filter_jit(plan.refresh)(
        previous,
        points,
        active_mask=active_mask,
        stable_ids=stable_ids,
        bounding_padding=padding,
    )
    assert bool(transition.accepted_candidate)
    assert bool(transition.refitted) is expected_refit
    assert bool(transition.rebuilt) is not expected_refit
    assert int(transition.accepted.epoch) == 6
    fresh = plan.build(
        points,
        active_mask=previous.point_order.active if active_mask is None else active_mask,
        stable_ids=initial_ids if stable_ids is None else stable_ids,
        bounding_padding=padding,
        epoch=6,
    )
    _assert_same_schedule(transition.candidate, fresh)
    _assert_same_schedule(transition.accepted, fresh)
    np.testing.assert_array_equal(transition.accepted.point_order.points, points)
    np.testing.assert_array_equal(transition.accepted.bounding_padding, padding)
    if coordinates == [0.1, 0.55, 0.6, 0.7]:
        assert not np.array_equal(
            transition.accepted.node_bit_levels, previous.node_bit_levels
        )
        np.testing.assert_array_equal(
            transition.accepted.logical_point_leaf_slots,
            previous.logical_point_leaf_slots,
        )
    if coordinates == [0.25, 0.25, 0.25, 0.25]:
        assert int(transition.accepted.evidence.oversized_terminal_buckets) == 1
        assert int(transition.accepted.evidence.maximum_terminal_bucket_occupancy) == 4


@pytest.mark.parametrize("failure", ["domain", "duplicate-ids", "padding"])
def test_plane_refresh_rolls_back_rejected_candidate(failure: str) -> None:
    plan = _plan()
    points = _points()
    previous = plan.build(points, bounding_padding=0.01, epoch=7)
    stable_ids = None
    padding = 0.01
    match failure:
        case "domain":
            points = points.at[1, 0].set(1.1)
        case "duplicate-ids":
            stable_ids = jnp.asarray([0, 1, 1, 3, 4, 5, 6, 7], dtype=jnp.int64)
        case "padding":
            padding = -0.01
        case _:
            raise AssertionError(failure)
    transition = eqx.filter_jit(plan.refresh)(
        previous, points, stable_ids=stable_ids, bounding_padding=padding
    )
    assert not bool(transition.accepted_candidate)
    assert not bool(transition.refitted)
    assert not bool(transition.rebuilt)
    assert not bool(transition.candidate.evidence.successful)
    assert int(transition.candidate.epoch) == 8
    np.testing.assert_array_equal(transition.candidate.point_order.points, points)
    _assert_same_schedule(transition.accepted, previous)


def test_plane_refresh_preserves_geometry_gradients() -> None:
    plan = MortonPlaneSchedulePlan(
        MortonAddressPlan((0.0,), (1.0,), 4),
        4,
        maximum_leaf_occupancy=2,
        coarsening_factor=2,
        target_top_nodes=1,
    )
    initial = jnp.asarray([[0.1], [0.2], [0.6], [0.7]], dtype=jnp.float64)
    previous = plan.build(initial)
    points = initial + 1.0e-3
    padding = 0.01
    direction = jnp.asarray([[0.2], [-0.3], [0.4], [-0.1]], dtype=jnp.float64)

    def objective(coordinates: Array) -> Array:
        schedule = plan.refresh(previous, coordinates, bounding_padding=padding).accepted
        return jnp.sum(schedule.point_order.sorted_coordinates**2) + jnp.sum(
            schedule.node_centers**2 + schedule.node_half_widths**2
        )

    def oracle(coordinates: Array) -> Array:
        lower = coordinates[jnp.asarray([0, 2, 0], dtype=jnp.int32), 0]
        upper = coordinates[jnp.asarray([1, 3, 3], dtype=jnp.int32), 0]
        centers = 0.5 * (lower + upper)
        half_widths = 0.5 * (upper - lower) + padding
        return jnp.sum(coordinates**2) + jnp.sum(centers**2 + half_widths**2)

    value, gradient = eqx.filter_jit(jax.value_and_grad(objective))(points)
    oracle_value, oracle_gradient = jax.value_and_grad(oracle)(points)
    np.testing.assert_allclose(value, oracle_value, rtol=1.0e-13, atol=1.0e-13)
    np.testing.assert_allclose(gradient, oracle_gradient, rtol=1.0e-13, atol=1.0e-13)
    _, tangent = jax.jvp(objective, (points,), (direction,))
    np.testing.assert_allclose(
        tangent, jnp.sum(oracle_gradient * direction), rtol=1.0e-13
    )


def test_plane_refresh_retains_omitted_mask_and_ids() -> None:
    plan = _plan()
    points = _points()
    active = jnp.asarray(
        [True, False, True, True, False, True, True, False], dtype=jnp.bool_
    )
    identifiers = jnp.asarray([41, 43, 40, 42, 45, 44, 47, 46], dtype=jnp.int64)
    previous = plan.build(points, active_mask=active, stable_ids=identifiers)
    moved = points + 1.0e-5
    transition = eqx.filter_jit(plan.refresh)(previous, moved)
    assert bool(transition.refitted)
    np.testing.assert_array_equal(transition.accepted.point_order.active, active)
    np.testing.assert_array_equal(transition.accepted.point_order.stable_ids, identifiers)
    _assert_same_schedule(
        transition.accepted,
        plan.build(moved, active_mask=active, stable_ids=identifiers, epoch=1),
    )


def test_plane_refresh_rolls_back_node_capacity_failure() -> None:
    plan = _plan(nodes=3)
    points = _points()
    initial_active = jnp.asarray(
        [True, True, False, False, False, False, False, False], dtype=jnp.bool_
    )
    previous = plan.build(points, active_mask=initial_active)
    assert bool(previous.evidence.successful)
    transition = eqx.filter_jit(plan.refresh)(
        previous, points, active_mask=jnp.ones((8,), dtype=jnp.bool_)
    )
    assert not bool(transition.accepted_candidate)
    assert not bool(transition.refitted)
    assert not bool(transition.rebuilt)
    assert int(transition.candidate.evidence.required_nodes) > plan.node_capacity
    _assert_same_schedule(transition.accepted, previous)
