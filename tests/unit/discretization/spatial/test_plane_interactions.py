from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.discretization.spatial import MortonAddressPlan
from phydrax.discretization.spatial._plane_interactions import (
    MortonPlaneInteractionPlan,
)
from phydrax.discretization.spatial._plane_schedule import MortonPlaneSchedulePlan


def _schedule() -> Any:
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


def _coverage(schedule: Any, interactions: Any) -> Any:
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


def test_plane_interactions_scenario_1() -> None:
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
    _, schedule = _schedule()
    active = schedule.node_active
    scales = schedule.node_scales[active]
    radii = jnp.sqrt(jnp.sum(schedule.node_half_widths[active] ** 2, axis=-1))
    assert bool(schedule.evidence.successful)
    assert int(schedule.evidence.invalid_scales) == 0
    assert bool(jnp.all(scales >= radii))
    np.testing.assert_allclose(jnp.log2(scales), jnp.round(jnp.log2(scales)))
    schedule_plan, _ = _schedule()
    invalid = schedule_plan.build(
        jnp.full((8, 3), 0.5),
        bounding_padding=jnp.asarray([0.0, -0.1, 0.0]),
    )
    assert not bool(invalid.evidence.successful)
    assert int(invalid.evidence.invalid_padding) == 1
    address = MortonAddressPlan((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 12)
    source_points = jnp.asarray(
        [
            [0.05, 0.05, 0.05],
            [0.10, 0.05, 0.05],
            [0.30, 0.25, 0.20],
            [0.65, 0.70, 0.75],
            [0.85, 0.85, 0.85],
            [0.92, 0.90, 0.88],
        ]
    )
    target_points = jnp.asarray(
        [
            [0.12, 0.08, 0.07],
            [0.40, 0.45, 0.40],
            [0.72, 0.68, 0.74],
            [0.95, 0.92, 0.90],
        ]
    )
    source_plan = MortonPlaneSchedulePlan(
        address,
        6,
        maximum_leaf_occupancy=2,
        coarsening_factor=2,
        target_top_nodes=1,
    )
    target_plan = MortonPlaneSchedulePlan(
        address,
        4,
        maximum_leaf_occupancy=1,
        coarsening_factor=2,
        target_top_nodes=1,
    )
    source = source_plan.build(source_points, bounding_padding=0.01)
    target = target_plan.build(
        target_points,
        bounding_padding=jnp.asarray([0.02, 0.01, 0.03]),
    )
    interactions = MortonPlaneInteractionPlan(
        source_plan,
        target_plan,
        opening_angle=0.5,
        queue_capacity=64,
        far_capacity=64,
        near_capacity=64,
    ).build(source, target, same_support=False)
    assert bool(interactions.evidence.successful)
    coverage = np.zeros((4, 6), dtype=np.int32)
    for relation in (interactions.far, interactions.near):
        for route in np.flatnonzero(np.asarray(relation.valid)):
            source_node = int(relation.source_indices[route])
            target_node = int(relation.target_indices[route])
            source_start = int(source.node_item_starts[source_node])
            source_count = int(source.node_item_counts[source_node])
            target_start = int(target.node_item_starts[target_node])
            target_count = int(target.node_item_counts[target_node])
            coverage[
                target_start : target_start + target_count,
                source_start : source_start + source_count,
            ] += 1
    np.testing.assert_array_equal(coverage, 1)
    np.testing.assert_allclose(source.bounding_padding, 0.01)
    np.testing.assert_allclose(target.bounding_padding, [0.02, 0.01, 0.03])
    assert int(source.evidence.invalid_padding) == 0
    assert int(target.evidence.invalid_padding) == 0


def _reference_routes(schedule_plan: Any, schedule: Any) -> Any:
    """Independent host traversal in consumer-visible step/frontier/child order."""
    centers = np.asarray(schedule.node_centers)
    half_widths = np.asarray(schedule.node_half_widths)
    planes = np.asarray(schedule.node_planes)
    child_starts = np.asarray(schedule.node_child_starts)
    child_counts = np.asarray(schedule.node_child_counts)
    top_plane = schedule_plan.plane_count - 1
    top_start = int(schedule.plane_offsets[top_plane])
    top_count = int(schedule.plane_active_counts[top_plane])
    top = range(top_start, top_start + top_count)
    frontier = [(source, target) for target in top for source in top]
    required_queue = len(frontier)
    far: list[tuple[int, int]] = []
    maximum_ratio = 0.0

    def classify(pairs: Any) -> Any:
        nonlocal maximum_ratio
        opened = []
        for source, target in pairs:
            distance = np.linalg.norm(centers[target] - centers[source])
            extent = np.linalg.norm(half_widths[source] + half_widths[target])
            if source != target and extent < 0.5 * distance:
                far.append((source, target))
                maximum_ratio = max(maximum_ratio, extent / distance)
            else:
                opened.append((source, target))
        return opened

    frontier = classify(frontier)
    required_queue = max(required_queue, len(frontier))
    for target_side in (True, False):
        for _ in range(schedule_plan.plane_count - 1):
            expanded = []
            for source, target in frontier:
                node = target if target_side else source
                children = (
                    range(child_starts[node], child_starts[node] + child_counts[node])
                    if planes[node] > 0
                    else (node,)
                )
                expanded.extend(
                    (source, child) if target_side else (child, target)
                    for child in children
                )
            frontier = classify(expanded)
            required_queue = max(required_queue, len(frontier))
    return far, frontier, required_queue, maximum_ratio


def _valid_routes(relation: Any) -> Any:
    valid = np.asarray(relation.valid)
    return list(
        zip(
            np.asarray(relation.source_indices)[valid].tolist(),
            np.asarray(relation.target_indices)[valid].tolist(),
            strict=True,
        )
    )


@pytest.mark.parametrize("widths", [None, (64,), (1, 2, 4, 8, 16, 32, 64)])
def test_stable_route_order_counts_and_ladder(widths: Any) -> None:
    schedule_plan, schedule = _schedule()
    plan = MortonPlaneInteractionPlan(
        schedule_plan,
        opening_angle=0.5,
        queue_capacity=64,
        far_capacity=128,
        near_capacity=64,
        frontier_widths=widths,
    )
    expected_far, expected_near, required_queue, maximum_ratio = _reference_routes(
        schedule_plan, schedule
    )
    for interactions in (plan.build(schedule), eqx.filter_jit(plan.build)(schedule)):
        assert bool(interactions.evidence.successful)
        assert _valid_routes(interactions.far) == expected_far
        assert _valid_routes(interactions.near) == expected_near
        assert int(interactions.evidence.required_far) == len(expected_far)
        assert int(interactions.evidence.required_near) == len(expected_near)
        assert int(interactions.evidence.required_queue) == required_queue
        np.testing.assert_allclose(
            interactions.evidence.maximum_accepted_ratio, maximum_ratio, rtol=1e-6
        )
        np.testing.assert_array_equal(_coverage(schedule, interactions), 1)


@pytest.mark.parametrize("resource", ["far", "near"])
def test_capacity_refusal_preserves_required_counts(resource: str) -> None:
    schedule_plan, schedule = _schedule()
    far, near, required_queue, maximum_ratio = _reference_routes(schedule_plan, schedule)
    assert len(far) > 1 and len(near) > 1
    result = MortonPlaneInteractionPlan(
        schedule_plan,
        opening_angle=0.5,
        queue_capacity=64,
        far_capacity=1 if resource == "far" else 128,
        near_capacity=1 if resource == "near" else 64,
        frontier_widths=(1, 4, 16, 64),
    ).build(schedule)
    assert not bool(result.evidence.successful)
    assert bool(result.evidence.far_overflow) == (resource == "far")
    assert bool(result.evidence.near_overflow) == (resource == "near")
    assert int(result.evidence.required_far) == len(far)
    assert int(result.evidence.required_near) == len(near)
    assert int(result.evidence.required_queue) == required_queue
    np.testing.assert_allclose(
        result.evidence.maximum_accepted_ratio, maximum_ratio, rtol=1e-6
    )
    np.testing.assert_array_equal(result.far.valid, False)
    np.testing.assert_array_equal(result.near.valid, False)
    np.testing.assert_array_equal(result.far.source_indices, 0)
    np.testing.assert_array_equal(result.near.target_indices, 0)


@pytest.mark.parametrize(
    "widths", [(), (0, 64), (8, 4, 64), (4, 4, 64), (4, 32), (4.0, 64)]
)
def test_invalid_frontier_ladder_is_refused(widths: Any) -> None:
    schedule_plan, _ = _schedule()
    with pytest.raises(ValueError, match="frontier_widths"):
        MortonPlaneInteractionPlan(
            schedule_plan,
            opening_angle=0.5,
            queue_capacity=64,
            far_capacity=128,
            near_capacity=64,
            frontier_widths=widths,
        )


def test_ladder_queue_overflow_matches_full_width() -> None:
    schedule_plan, schedule = _schedule()
    results = [
        MortonPlaneInteractionPlan(
            schedule_plan,
            opening_angle=1.0e-6,
            queue_capacity=4,
            far_capacity=128,
            near_capacity=64,
            frontier_widths=widths,
        ).build(schedule)
        for widths in (None, (1, 2, 4))
    ]
    assert bool(results[0].evidence.queue_overflow)
    for first, second in zip(
        jax.tree.leaves(results[0]), jax.tree.leaves(results[1]), strict=True
    ):
        np.testing.assert_array_equal(first, second)


def test_recheck_refreshes_current_geometry_evidence() -> None:
    schedule_plan, schedule = _schedule()
    plan = MortonPlaneInteractionPlan(
        schedule_plan,
        opening_angle=0.5,
        queue_capacity=64,
        far_capacity=128,
        near_capacity=64,
    )
    interactions = plan.build(schedule)
    rechecked = eqx.filter_jit(plan.recheck)(schedule, interactions=interactions)
    assert bool(rechecked.successful)
    np.testing.assert_allclose(
        rechecked.maximum_accepted_ratio, interactions.evidence.maximum_accepted_ratio
    )
    # Enlarging each center separation preserves topology but changes the MAC ratio.
    expanded = eqx.tree_at(
        lambda state: state.node_centers, schedule, schedule.node_centers * 2.0
    )
    refreshed = plan.recheck(expanded, interactions=interactions)
    assert bool(refreshed.successful)
    np.testing.assert_allclose(
        refreshed.maximum_accepted_ratio,
        interactions.evidence.maximum_accepted_ratio / 2.0,
    )
    assert int(refreshed.required_far) == int(interactions.evidence.required_far)
    assert int(refreshed.required_near) == int(interactions.evidence.required_near)
    collapsed = eqx.tree_at(
        lambda state: state.node_centers, schedule, jnp.zeros_like(schedule.node_centers)
    )
    assert not bool(plan.recheck(collapsed, interactions=interactions).successful)
    invalid = eqx.tree_at(
        lambda state: state.node_centers,
        schedule,
        jnp.full_like(schedule.node_centers, jnp.nan),
    )
    evidence = plan.recheck(invalid, interactions=interactions)
    assert not bool(evidence.successful)
    assert not bool(evidence.finite)
    invalid_result = plan.build(invalid)
    assert not bool(invalid_result.evidence.finite)
    np.testing.assert_array_equal(invalid_result.far.valid, False)
    np.testing.assert_array_equal(invalid_result.near.valid, False)


def test_recheck_enforces_node_radius_and_cutoff_rebuild() -> None:
    schedule_plan, schedule = _schedule()
    plan = MortonPlaneInteractionPlan(
        schedule_plan,
        opening_angle=0.5,
        queue_capacity=64,
        far_capacity=128,
        near_capacity=64,
    )
    interactions = plan.build(schedule)
    source, _ = _valid_routes(interactions.far)[0]
    radius = float(jnp.linalg.norm(schedule.node_half_widths[source]))
    bounded = MortonPlaneInteractionPlan(
        schedule_plan,
        opening_angle=0.5,
        queue_capacity=64,
        far_capacity=128,
        near_capacity=64,
        maximum_node_radius=radius / 2.0,
    )
    assert not bool(bounded.recheck(schedule, interactions=interactions).successful)
    cutoff = MortonPlaneInteractionPlan(
        schedule_plan,
        opening_angle=0.5,
        queue_capacity=64,
        far_capacity=128,
        near_capacity=64,
        interaction_cutoff=2.0,
    )
    cutoff_interactions = cutoff.build(schedule)
    assert bool(cutoff_interactions.evidence.successful)
    assert not bool(cutoff.recheck(schedule, interactions=cutoff_interactions).complete)


def test_recheck_rejects_same_node_far_routes_and_prior_failure() -> None:
    schedule_plan, schedule = _schedule()
    plan = MortonPlaneInteractionPlan(
        schedule_plan,
        opening_angle=0.5,
        queue_capacity=64,
        far_capacity=128,
        near_capacity=64,
    )
    interactions = plan.build(schedule)
    assert bool(jnp.any(interactions.far.valid))
    self_routes = eqx.tree_at(
        lambda state: state.far.target_indices,
        interactions,
        interactions.far.source_indices,
    )
    assert not bool(plan.recheck(schedule, interactions=self_routes).successful)
    failed = eqx.tree_at(
        lambda state: state.evidence.successful, interactions, jnp.asarray(False)
    )
    assert not bool(plan.recheck(schedule, interactions=failed).successful)


def test_ladder_preserves_asymmetric_support_routes() -> None:
    address = MortonAddressPlan((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 12)
    source_plan = MortonPlaneSchedulePlan(
        address, 8, maximum_leaf_occupancy=2, coarsening_factor=2, target_top_nodes=1
    )
    target_plan = MortonPlaneSchedulePlan(
        address, 4, maximum_leaf_occupancy=1, coarsening_factor=2, target_top_nodes=1
    )
    _, source = _schedule()
    target = target_plan.build(
        jnp.asarray(
            [[0.12, 0.08, 0.07], [0.4, 0.45, 0.4], [0.72, 0.68, 0.74], [0.95, 0.92, 0.9]]
        )
    )
    results = [
        MortonPlaneInteractionPlan(
            source_plan,
            target_plan,
            opening_angle=0.5,
            queue_capacity=64,
            far_capacity=128,
            near_capacity=64,
            frontier_widths=widths,
        ).build(source, target)
        for widths in (None, (1, 4, 16, 64))
    ]
    assert bool(results[0].evidence.successful)
    for first, second in zip(
        jax.tree.leaves(results[0]), jax.tree.leaves(results[1]), strict=True
    ):
        np.testing.assert_array_equal(first, second)
    plan = MortonPlaneInteractionPlan(
        source_plan,
        target_plan,
        opening_angle=0.5,
        queue_capacity=64,
        far_capacity=128,
        near_capacity=64,
    )
    assert bool(plan.recheck(source, target, interactions=results[0]).successful)
