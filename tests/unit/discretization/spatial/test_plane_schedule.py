from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from phydrax.discretization.spatial import MortonAddressPlan
from phydrax.discretization.spatial._plane_schedule import MortonPlaneSchedulePlan


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


def test_plane_schedule_partitions_points_and_children() -> None:
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


def test_plane_schedule_is_permutation_invariant_by_stable_id() -> None:
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


def test_plane_schedule_exposes_terminal_buckets_and_failures() -> None:
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


def test_plane_schedule_empty_build_and_atomic_refresh_jit() -> None:
    plan = _plan()
    points = _points()
    build = eqx.filter_jit(plan.build)
    empty = build(points, active_mask=jnp.zeros((8,), dtype=bool))
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
