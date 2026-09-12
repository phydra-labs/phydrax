from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from phydrax.discretization.spatial import MortonAddressPlan
from phydrax.discretization.spatial._plane_interactions import (
    MortonPlaneInteractionPlan,
)
from phydrax.discretization.spatial._plane_schedule import MortonPlaneSchedulePlan


def _schedule():
    points = jnp.asarray(
        [
            [0.05, 0.05, 0.05],
            [0.08, 0.05, 0.05],
            [0.22, 0.20, 0.20],
            [0.25, 0.20, 0.20],
            [0.70, 0.70, 0.70],
            [0.73, 0.70, 0.70],
            [0.92, 0.90, 0.90],
            [0.95, 0.90, 0.90],
        ]
    )
    plan = MortonPlaneSchedulePlan(
        MortonAddressPlan((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 12),
        8,
        maximum_leaf_occupancy=2,
        coarsening_factor=2,
        target_top_nodes=1,
    )
    return plan, plan.build(points)


def _coverage(schedule, interactions):
    count = schedule.point_order.sorted_active.size
    coverage = np.zeros((count, count), dtype=np.int32)
    for relation in (interactions.far, interactions.near):
        for route in np.flatnonzero(np.asarray(relation.valid)):
            source = int(relation.source_indices[route])
            target = int(relation.target_indices[route])
            source_start = int(schedule.node_item_starts[source])
            source_count = int(schedule.node_item_counts[source])
            target_start = int(schedule.node_item_starts[target])
            target_count = int(schedule.node_item_counts[target])
            coverage[
                target_start : target_start + target_count,
                source_start : source_start + source_count,
            ] += 1
    return coverage


def test_dual_tree_routes_cover_each_ordered_pair_once() -> None:
    schedule_plan, schedule = _schedule()
    interactions = MortonPlaneInteractionPlan(
        schedule_plan,
        opening_angle=0.5,
        queue_capacity=64,
        far_capacity=128,
        near_capacity=64,
    ).build(schedule)
    assert bool(interactions.evidence.successful)
    np.testing.assert_array_equal(_coverage(schedule, interactions), 1)
    assert float(interactions.evidence.maximum_accepted_ratio) < 0.5
    near_nodes = interactions.near.source_indices[interactions.near.valid]
    assert bool(jnp.all(schedule.node_planes[near_nodes] == 0))


def test_plane_scales_are_finite_powers_of_two_and_enclose_nodes() -> None:
    _, schedule = _schedule()
    active = schedule.node_active
    scales = schedule.node_scales[active]
    radii = jnp.sqrt(jnp.sum(schedule.node_half_widths[active] ** 2, axis=-1))
    assert bool(schedule.evidence.successful)
    assert int(schedule.evidence.invalid_scales) == 0
    assert bool(jnp.all(scales >= radii))
    np.testing.assert_allclose(jnp.log2(scales), jnp.round(jnp.log2(scales)))


def test_dual_tree_capacity_failure_is_fail_closed() -> None:
    schedule_plan, schedule = _schedule()
    interactions = MortonPlaneInteractionPlan(
        schedule_plan,
        opening_angle=0.5,
        queue_capacity=1,
        far_capacity=1,
        near_capacity=1,
    ).build(schedule)
    assert not bool(interactions.evidence.successful)
    assert bool(
        interactions.evidence.queue_overflow
        | interactions.evidence.far_overflow
        | interactions.evidence.near_overflow
    )
    np.testing.assert_array_equal(interactions.far.valid, False)
    np.testing.assert_array_equal(interactions.near.valid, False)


def test_dual_tree_build_is_filter_jittable() -> None:
    schedule_plan, schedule = _schedule()
    plan = MortonPlaneInteractionPlan(
        schedule_plan,
        opening_angle=0.5,
        queue_capacity=64,
        far_capacity=128,
        near_capacity=64,
    )
    eager = plan.build(schedule)
    compiled = eqx.filter_jit(plan.build)(schedule)
    assert bool(compiled.evidence.successful)
    np.testing.assert_array_equal(compiled.far.valid, eager.far.valid)
    np.testing.assert_array_equal(compiled.near.valid, eager.near.valid)
