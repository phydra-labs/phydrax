from __future__ import annotations

from itertools import combinations

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.discretization.spatial import AdaptiveOctreePlan, MortonAddressPlan


def _clustered_points(dimension: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    points = np.concatenate(
        (
            0.3 + 0.004 * rng.standard_normal((60, dimension)),
            0.7 + 0.03 * rng.standard_normal((40, dimension)),
            rng.uniform(0.0, 1.0, (20, dimension)),
        )
    )
    return np.clip(points, 0.0, 0.999)


def _address(dimension: int, depth: int = 8) -> MortonAddressPlan:
    return MortonAddressPlan((0.0,) * dimension, (1.0,) * dimension, depth)


def _leaf_coverage(tree) -> np.ndarray:
    """Count, for every leaf and every point, the routes whose source spans the point."""
    parents = np.asarray(tree.node_parents)
    starts = np.asarray(tree.node_point_starts)
    ends = np.asarray(tree.node_point_ends)
    routes = {}
    for name in ("u_list", "v_list", "w_list", "x_list"):
        relation = getattr(tree, name).routes
        valid = np.asarray(relation.valid)
        routes[name] = (
            np.asarray(relation.target_indices)[valid],
            np.asarray(relation.source_indices)[valid],
        )
    leaves = np.asarray(tree.leaf_nodes)
    coverage = np.zeros((leaves.size, tree.point_count), dtype=np.int64)
    for row, leaf in enumerate(leaves):
        ancestors = [leaf]
        while parents[ancestors[-1]] >= 0:
            ancestors.append(parents[ancestors[-1]])
        delta = np.zeros((tree.point_count + 1,), dtype=np.int64)
        for name, receivers in (
            ("u_list", [leaf]),
            ("w_list", [leaf]),
            ("v_list", ancestors),
            ("x_list", ancestors),
        ):
            targets, sources = routes[name]
            selected = sources[np.isin(targets, receivers)]
            np.add.at(delta, starts[selected], 1)
            np.add.at(delta, ends[selected], -1)
        coverage[row] = np.cumsum(delta)[:-1]
    return coverage


def _touching_leaf_level_gap(tree) -> int:
    leaves = np.asarray(tree.leaf_nodes)
    centers = np.asarray(tree.node_centers)[leaves]
    half_widths = np.asarray(tree.node_half_widths)[leaves]
    levels = np.asarray(tree.node_levels)[leaves]
    gap = 0
    for first, second in combinations(range(leaves.size), 2):
        touching = np.all(
            np.abs(centers[first] - centers[second])
            <= half_widths[first] + half_widths[second] + 1e-12
        )
        if touching:
            gap = max(gap, abs(int(levels[first]) - int(levels[second])))
    return gap


def test_adaptive_octree_is_compact_and_places_every_point_in_one_leaf() -> None:
    points = _clustered_points(3)
    address = _address(3, depth=10)
    tree = AdaptiveOctreePlan(address, leaf_capacity=8).prepare(points)

    codes = np.asarray(address.encode(jnp.asarray(points)).codes)
    full_depth_nodes = sum(
        np.unique(codes >> np.uint64(3 * (address.maximum_depth - level))).size
        for level in range(address.maximum_depth + 1)
    )
    assert 4 * tree.node_count < full_depth_nodes

    leaves = np.asarray(tree.leaf_nodes)
    starts = np.asarray(tree.node_point_starts)[leaves]
    ends = np.asarray(tree.node_point_ends)[leaves]
    membership = np.zeros((tree.point_count + 1,), dtype=np.int64)
    np.add.at(membership, starts, 1)
    np.add.at(membership, ends, -1)
    np.testing.assert_array_equal(np.cumsum(membership)[:-1], 1)
    assert int(np.max(ends - starts)) <= 8
    assert bool(tree.evidence.leaf_capacity_satisfied)

    order = np.asarray(tree.point_order)
    np.testing.assert_array_equal(np.sort(order), np.arange(points.shape[0]))
    point_leaves = np.asarray(tree.point_leaves)
    sorted_slot = np.empty_like(order)
    sorted_slot[order] = np.arange(order.size)
    node_starts = np.asarray(tree.node_point_starts)
    node_ends = np.asarray(tree.node_point_ends)
    assert np.all(node_starts[point_leaves] <= sorted_slot)
    assert np.all(sorted_slot < node_ends[point_leaves])
    np.testing.assert_array_equal(np.asarray(tree.locate(points)), point_leaves)
    lower = np.asarray(tree.node_centers - tree.node_half_widths)[point_leaves]
    upper = np.asarray(tree.node_centers + tree.node_half_widths)[point_leaves]
    assert np.all((points >= lower) & (points < upper))


def test_leaves_tile_the_address_box_for_arbitrary_query_points() -> None:
    tree = AdaptiveOctreePlan(_address(2), leaf_capacity=4).prepare(_clustered_points(2))
    queries = np.random.default_rng(5).uniform(0.0, 1.0, (400, 2))
    leaves = np.asarray(tree.locate(queries))
    assert np.all(np.asarray(tree.node_child_counts)[leaves] == 0)
    lower = np.asarray(tree.node_centers - tree.node_half_widths)[leaves]
    upper = np.asarray(tree.node_centers + tree.node_half_widths)[leaves]
    assert np.all((queries >= lower) & (queries < upper))
    outside = np.asarray(tree.locate(np.asarray([[1.0, 0.5], [-0.1, 0.2]])))
    np.testing.assert_array_equal(outside, [-1, -1])


def test_balanced_octree_keeps_touching_leaves_within_one_level() -> None:
    points = _clustered_points(2)
    unbalanced = AdaptiveOctreePlan(_address(2), leaf_capacity=4).prepare(points)
    balanced = AdaptiveOctreePlan(_address(2), leaf_capacity=4, balanced=True).prepare(
        points
    )
    assert _touching_leaf_level_gap(unbalanced) > 1
    assert _touching_leaf_level_gap(balanced) <= 1
    assert balanced.node_count > unbalanced.node_count


@pytest.mark.parametrize(
    ("dimension", "balanced", "padding"),
    [(2, False, 0.0), (2, True, 0.03), (3, False, 0.0), (3, True, 0.02)],
)
def test_interaction_lists_cover_every_source_point_exactly_once(
    dimension: int, balanced: bool, padding: float
) -> None:
    tree = AdaptiveOctreePlan(
        _address(dimension),
        leaf_capacity=4,
        balanced=balanced,
        separation_padding=padding,
    ).prepare(_clustered_points(dimension, seed=dimension))
    assert bool(tree.evidence.successful)
    np.testing.assert_array_equal(_leaf_coverage(tree), 1)
    for name in ("u_list", "v_list", "w_list", "x_list"):
        assert int(getattr(tree, name).required_routes) > 0

    levels = np.asarray(tree.node_levels)
    leaf = np.asarray(tree.node_child_counts) == 0
    occupied = np.asarray(tree.node_point_ends) > np.asarray(tree.node_point_starts)

    def routes(name):
        relation = getattr(tree, name).routes
        valid = np.asarray(relation.valid)
        return (
            np.asarray(relation.target_indices)[valid],
            np.asarray(relation.source_indices)[valid],
        )

    for name in ("u_list", "v_list", "w_list", "x_list"):
        assert np.all(occupied[routes(name)[1]])
    targets, sources = routes("u_list")
    assert np.all(leaf[targets] & leaf[sources])
    targets, sources = routes("v_list")
    assert np.all(levels[targets] == levels[sources])
    targets, sources = routes("w_list")
    assert np.all(leaf[targets] & (levels[sources] > levels[targets]))
    targets, sources = routes("x_list")
    assert np.all(leaf[sources] & (levels[sources] < levels[targets]))


def test_interaction_list_rows_gather_target_routes() -> None:
    tree = AdaptiveOctreePlan(_address(2), leaf_capacity=4).prepare(_clustered_points(2))
    interaction = tree.u_list
    relation = interaction.routes
    valid = np.asarray(relation.valid)
    targets = np.asarray(relation.target_indices)[valid]
    sources = np.asarray(relation.source_indices)[valid]
    leaves = np.asarray(tree.leaf_nodes)
    rows, row_valid = interaction.rows(leaves)
    rows = np.asarray(rows)
    row_valid = np.asarray(row_valid)
    for leaf, row, mask in zip(leaves, rows, row_valid):
        np.testing.assert_array_equal(row[mask], sources[targets == leaf])


def test_interaction_capacity_overflow_is_reported() -> None:
    points = _clustered_points(3)
    complete = AdaptiveOctreePlan(_address(3), leaf_capacity=4).prepare(points)
    bounded = AdaptiveOctreePlan(
        _address(3),
        leaf_capacity=4,
        u_capacity=3,
        v_capacity=2,
        w_capacity=1,
        x_capacity=1,
    ).prepare(points)
    assert bool(complete.evidence.successful)
    assert not bool(bounded.evidence.successful)
    for name, capacity in (
        ("u_list", 3),
        ("v_list", 2),
        ("w_list", 1),
        ("x_list", 1),
    ):
        interaction = getattr(bounded, name)
        assert bool(interaction.overflow)
        assert interaction.routes.capacity == capacity
        assert int(jnp.sum(interaction.routes.valid)) == capacity
        assert int(interaction.required_routes) == int(
            getattr(complete, name).required_routes
        )
        assert not bool(getattr(complete, name).overflow)


def test_plan_rejects_invalid_configuration_and_points() -> None:
    with pytest.raises(ValueError, match="periodic"):
        AdaptiveOctreePlan(
            MortonAddressPlan((0.0, 0.0), (1.0, 1.0), 4, periodic_axes=(True, False)),
            leaf_capacity=2,
        )
    with pytest.raises(ValueError, match="leaf_capacity"):
        AdaptiveOctreePlan(_address(2), leaf_capacity=0)
    with pytest.raises(ValueError, match="separation_padding"):
        AdaptiveOctreePlan(_address(2), leaf_capacity=2, separation_padding=-0.1)
    with pytest.raises(ValueError, match="capacities"):
        AdaptiveOctreePlan(_address(2), leaf_capacity=2, v_capacity=0)
    with pytest.raises(TypeError, match="balanced"):
        AdaptiveOctreePlan(_address(2), leaf_capacity=2, balanced=1)
    plan = AdaptiveOctreePlan(_address(2), leaf_capacity=2)
    with pytest.raises(ValueError, match="lie in"):
        plan.prepare(np.asarray([[0.5, 0.5], [1.0, 0.2]]))
    with pytest.raises(ValueError, match="shape"):
        plan.prepare(np.zeros((0, 2)))
