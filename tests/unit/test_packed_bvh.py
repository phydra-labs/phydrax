#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._bvh import (
    bvh_host_minima,
    bvh_nearest_items,
    bvh_overlap_pair_blocks,
    bvh_overlap_pairs,
    bvh_overlap_pairs_host,
    BVHBuildKind,
    BVHBuildPolicy,
    point_select_leaf_items,
    prepare_bvh,
    ray_select_leaf_items,
    refit_packed_bvh_bounds,
)


_KINDS = tuple(BVHBuildKind)


def _candidate_sets(indices: Any, valid: Any) -> Any:
    return tuple(
        frozenset(int(value) for value in row[row_valid])
        for row, row_valid in zip(indices, valid, strict=True)
    )


def _random_boxes(seed: int, count: int, dimension: int = 3, scale: float = 0.08) -> Any:
    rng = np.random.default_rng(seed)
    lower = rng.random((count, dimension))
    return lower, lower + scale * rng.random((count, dimension))


def _brute_pairs(
    first_lower: Any,
    first_upper: Any,
    second_lower: Any,
    second_upper: Any,
    touching: Any,
) -> Any:
    extent = np.minimum(first_upper[:, None], second_upper[None]) - np.maximum(
        first_lower[:, None], second_lower[None]
    )
    hit = np.all(extent >= 0.0 if touching else extent > 0.0, axis=-1)
    return np.argwhere(hit)


def test_packed_bvh_scenario_1() -> None:
    lower = jnp.stack(
        (jnp.arange(8, dtype="float64"), jnp.zeros(8), jnp.zeros(8)), axis=-1
    )
    upper = lower + jnp.asarray((0.75, 1.0, 1.0))
    bvh = prepare_bvh(lower, upper, policy=BVHBuildPolicy(leaf_size=2))
    points = jnp.asarray(((0.25, 0.5, 0.5), (3.25, 0.5, 0.5), (7.25, 0.5, 0.5)))

    chunked = point_select_leaf_items(
        points,
        bvh=bvh,
        maximum_candidates=4,
        query_batch_capacity=1,
    )
    vectorized = point_select_leaf_items(
        points,
        bvh=bvh,
        maximum_candidates=4,
        query_batch_capacity=8,
    )

    assert _candidate_sets(*chunked[:2]) == _candidate_sets(*vectorized[:2])
    assert jnp.all(chunked[2])
    assert jnp.array_equal(chunked[2], vectorized[2])
    lower = jnp.asarray(((0.0, -1.0, -1.0), (2.0, -1.0, -1.0)))
    upper = jnp.asarray(((1.0, 1.0, 1.0), (3.0, 1.0, 1.0)))
    bvh = prepare_bvh(lower, upper, policy=BVHBuildPolicy(leaf_size=1))
    origins = jnp.asarray(((-1.0, 0.0, 0.0), (1.5, 0.0, 0.0)))
    directions = jnp.asarray(((1.0, 0.0, 0.0), (1.0, 0.0, 0.0)))

    indices, valid, complete = ray_select_leaf_items(
        origins,
        directions,
        bvh=bvh,
        maximum_candidates=2,
        query_batch_capacity=1,
    )

    assert _candidate_sets(indices, valid) == (frozenset((0, 1)), frozenset((1,)))
    assert jnp.all(complete)
    for kind in _KINDS:
        for count in (1, 7, 300):
            lower, upper = _random_boxes(count, count)
            bvh = prepare_bvh(
                lower, upper, policy=BVHBuildPolicy(kind, leaf_size=4), dtype=jnp.float64
            )

            leaf_items = np.asarray(bvh.leaf_items)
            assert leaf_items.shape[1] == 4
            assert np.array_equal(np.sort(leaf_items[leaf_items >= 0]), np.arange(count))
            left = np.asarray(bvh.left)
            right = np.asarray(bvh.right)
            node_min = np.asarray(bvh.bbox_min)
            node_max = np.asarray(bvh.bbox_max)
            for level in range(bvh.max_depth):
                nodes = np.arange(bvh.level_offsets[level], bvh.level_offsets[level + 1])
                parents = nodes[left[nodes] >= 0]
                children = np.concatenate((left[parents], right[parents]))
                assert np.all(children >= bvh.level_offsets[level + 1])
                assert np.all(children < bvh.level_offsets[level + 2])
                for child in (left[parents], right[parents]):
                    assert np.all(node_min[parents] <= node_min[child])
                    assert np.all(node_max[parents] >= node_max[child])
            leaf_nodes = np.asarray(bvh.leaf_node)
            for leaf, node in enumerate(leaf_nodes):
                items = leaf_items[leaf][leaf_items[leaf] >= 0]
                assert np.all(node_min[node] <= lower[items])
                assert np.all(node_max[node] >= upper[items])
    lower = np.zeros((40, 2))
    upper = np.ones((40, 2))
    bvh = prepare_bvh(
        lower, upper, policy=BVHBuildPolicy(BVHBuildKind.MORTON, leaf_size=3)
    )

    leaf_items = np.asarray(bvh.leaf_items)
    assert np.array_equal(np.sort(leaf_items[leaf_items >= 0]), np.arange(40))
    # Identical codes split by sorted position, so the tree stays balanced.
    assert bvh.max_depth <= 6


def test_packed_bvh_scenario_2() -> None:
    lower = np.asarray(((0.1, 1.0 / 3.0), (0.7, 2.0 / 3.0)))
    upper = lower + np.asarray((1.0e-9, 0.2))
    bvh = prepare_bvh(lower, upper, dtype=jnp.float32)

    assert np.all(np.asarray(bvh.item_bbox_min, dtype=np.float64) <= lower)
    assert np.all(np.asarray(bvh.item_bbox_max, dtype=np.float64) >= upper)
    for kind in _KINDS:
        rng = np.random.default_rng(3)
        points = rng.random((200, 3))
        points[:20] = points[100:120]
        queries = np.concatenate((rng.random((30, 3)), points[105:107]))
        bvh = prepare_bvh(
            points, points, policy=BVHBuildPolicy(kind, leaf_size=8), dtype=jnp.float64
        )

        result = bvh_nearest_items(bvh, queries, k=3, query_batch_capacity=8)

        distance = np.sum((queries[:, None] - points[None]) ** 2, axis=-1)
        expected = np.stack(
            [np.lexsort((np.arange(points.shape[0]), row))[:3] for row in distance]
        )
        assert np.array_equal(np.asarray(result.items), expected)
        np.testing.assert_allclose(
            np.asarray(result.distance_squared), np.take_along_axis(distance, expected, 1)
        )
        assert np.asarray(result.items)[-2:, :2].tolist() == [[5, 105], [6, 106]]
    points = np.asarray(((0.0, 0.0), (1.0, 0.0)))
    bvh = prepare_bvh(points, points, dtype=jnp.float64)

    result = bvh_nearest_items(bvh, np.asarray((0.9, 0.0)), k=3)

    assert np.asarray(result.items).tolist() == [1, 0, -1]
    assert np.isinf(np.asarray(result.distance_squared)[2])
    for kind in _KINDS:
        lower, upper = _random_boxes(5, 150, scale=0.02)
        policy = BVHBuildPolicy(kind, leaf_size=4)
        bvh = prepare_bvh(lower, upper, policy=policy, dtype=jnp.float64)
        rng = np.random.default_rng(6)
        moved_lower = lower * np.asarray((1.5, 0.5, 1.0)) + 0.1 * rng.standard_normal(
            lower.shape
        )
        moved_upper = moved_lower + (upper - lower)
        queries = rng.random((25, 3))

        refitted = refit_packed_bvh_bounds(bvh, moved_lower, moved_upper)
        rebuilt = prepare_bvh(moved_lower, moved_upper, policy=policy, dtype=jnp.float64)

        first = bvh_nearest_items(refitted, queries, k=2)
        second = bvh_nearest_items(rebuilt, queries, k=2)
        assert np.array_equal(np.asarray(first.items), np.asarray(second.items))
        np.testing.assert_allclose(
            np.asarray(first.distance_squared), np.asarray(second.distance_squared)
        )
        np.testing.assert_allclose(np.asarray(refitted.bbox_min[0]), moved_lower.min(0))
        gradient = jax.grad(
            lambda values: refit_packed_bvh_bounds(bvh, values, values + 0.02).bbox_min[
                0, 0
            ]
        )(jnp.asarray(moved_lower))
        expected = np.zeros_like(moved_lower)
        expected[np.argmin(moved_lower[:, 0]), 0] = 1.0
        np.testing.assert_allclose(np.asarray(gradient), expected)


def test_packed_bvh_scenario_3() -> None:
    for kind in _KINDS:
        for touching in (False, True):
            first_lower, first_upper = _random_boxes(11, 120)
            second_lower, second_upper = _random_boxes(12, 90)
            # Exactly touching boxes separate the strict and touching predicates.
            second_lower[:10] = (
                first_upper[:10] * np.asarray((1.0, 0.0, 0.0))
                + np.asarray((0.0, 1.0, 1.0)) * first_lower[:10]
            )
            second_upper[:10] = second_lower[:10] + 0.05
            expected = _brute_pairs(
                first_lower, first_upper, second_lower, second_upper, touching
            )
            first = prepare_bvh(
                first_lower,
                first_upper,
                policy=BVHBuildPolicy(kind, leaf_size=4),
                dtype=jnp.float64,
            )
            second = prepare_bvh(
                second_lower,
                second_upper,
                policy=BVHBuildPolicy(kind, leaf_size=5),
                dtype=jnp.float64,
            )

            host = np.stack(
                bvh_overlap_pairs_host(first, second, include_touching=touching), axis=1
            )
            device = bvh_overlap_pairs(
                first,
                second,
                capacity=expected.shape[0] + 3,
                include_touching=touching,
                traversal_width=8,
            )

            assert np.array_equal(host, expected)
            count = int(device.count)
            assert count == expected.shape[0] and not bool(device.overflow)
            assert int(np.sum(np.asarray(device.valid))) == count
            assert np.array_equal(
                np.stack(
                    (
                        np.asarray(device.first_items)[:count],
                        np.asarray(device.second_items)[:count],
                    ),
                    axis=1,
                ),
                expected,
            )
    lower, upper = _random_boxes(21, 60, scale=0.3)
    bvh = prepare_bvh(lower, upper, policy=BVHBuildPolicy(leaf_size=4))
    expected = _brute_pairs(lower, upper, lower, upper, False)

    result = bvh_overlap_pairs(bvh, bvh, capacity=5)

    assert bool(result.overflow)
    # Pair counts beyond int32 must not wrap.
    assert result.count.dtype == jnp.int64
    assert int(result.count) == expected.shape[0]
    retained = np.stack(
        (np.asarray(result.first_items)[:5], np.asarray(result.second_items)[:5]),
        axis=1,
    )
    assert {tuple(row) for row in retained} <= {tuple(row) for row in expected}
    lower = np.asarray(((0.0, 0.0), (1.0 + 1.0e-3, 0.0), (5.0, 5.0)))
    upper = lower + 1.0
    bvh = prepare_bvh(lower, upper, dtype=jnp.float64)

    strict = bvh_overlap_pairs_host(bvh, bvh)
    padded = bvh_overlap_pairs_host(
        bvh, bvh, include_touching=True, absolute_tolerance=2.0e-3
    )

    assert np.stack(strict, axis=1).tolist() == [[0, 0], [1, 1], [2, 2]]
    assert np.stack(padded, axis=1).tolist() == [[0, 0], [0, 1], [1, 0], [1, 1], [2, 2]]
    with pytest.raises(ValueError, match="include_touching"):
        bvh_overlap_pairs_host(bvh, bvh, absolute_tolerance=1.0e-3)


@pytest.mark.parametrize("touching", (False, True))
def test_metered_bvh_blocks_preserve_all_pairs_and_admit_rejected_tests(
    touching: bool,
) -> None:
    lower = np.asarray(((0.0, 0.0), (1.0, 0.0), (4.0, 4.0), (0.25, 0.25)))
    upper = lower + 1.0
    tree = prepare_bvh(
        lower,
        upper,
        policy=BVHBuildPolicy(leaf_size=1),
        dtype=np.float64,
    )
    batches: list[int] = []
    blocks = list(
        bvh_overlap_pair_blocks(
            tree,
            tree,
            include_touching=touching,
            visit=batches.append,
            maximum_block_pairs=2,
        )
    )
    actual = {
        (int(first), int(second))
        for rows, columns in blocks
        for first, second in zip(rows, columns, strict=True)
    }
    expected = {tuple(row) for row in _brute_pairs(lower, upper, lower, upper, touching)}
    assert actual == expected
    assert sum(batches) >= len(actual)
    assert all(0 <= batch <= 4 for batch in batches)
    default = bvh_overlap_pairs_host(tree, tree, include_touching=touching)
    assert actual == set(zip(default[0].tolist(), default[1].tolist(), strict=True))


def test_bvh_visit_refusal_precedes_first_overlap_result() -> None:
    tree = prepare_bvh(
        np.zeros((2, 2), dtype=np.float64),
        np.ones((2, 2), dtype=np.float64),
        policy=BVHBuildPolicy(leaf_size=1),
        dtype=np.float64,
    )
    calls: list[int] = []

    def refuse(count: int) -> None:
        calls.append(count)
        raise ValueError("original work allowance exhausted")

    with pytest.raises(ValueError, match="original work allowance"):
        next(bvh_overlap_pair_blocks(tree, tree, visit=refuse))
    assert calls == [1]


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("seed", (4, 17))
def test_host_weighted_minima_match_exhaustive_objectives(
    kind: BVHBuildKind, seed: int
) -> None:
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(37, 3))
    centers[1] = centers[0]
    weights = rng.uniform(-0.2, 0.2, size=(37, 2))
    weights[1] = weights[0]
    points = np.concatenate((centers[:2], rng.normal(size=(11, 3))))
    tree = prepare_bvh(
        centers, centers, policy=BVHBuildPolicy(kind=kind, leaf_size=3), dtype=np.float64
    )
    visits: list[tuple[str, int]] = []
    owners: list[object] = []
    storage: list[int] = []

    def bound(
        point: np.ndarray,
        node: int,
        lower: np.ndarray,
        upper: np.ndarray,
        out: np.ndarray,
    ) -> None:
        del node
        out[:] = np.linalg.norm(
            np.maximum(np.maximum(lower - point, point - upper), 0.0)
        ) + weights.min(axis=0)

    def values(point: np.ndarray, items: np.ndarray, out: np.ndarray) -> None:
        out[:] = np.linalg.norm(centers[items] - point, axis=1)[:, None] + weights[items]

    actual, keys, nodes, items = bvh_host_minima(
        tree,
        points,
        objective_count=2,
        node_lower_bounds=bound,
        item_values=values,
        visit=lambda kind, count: visits.append((kind, count)),
        admit_storage=storage.append,
        retain_owner=owners.append,
    )
    exhaustive = (
        np.linalg.norm(points[:, None] - centers[None], axis=2)[:, :, None]
        + weights[None]
    )
    expected_keys = np.argmin(exhaustive, axis=1)
    expected = np.take_along_axis(exhaustive, expected_keys[:, None], axis=1)[:, 0]
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(keys, expected_keys)
    assert nodes == sum(count for kind, count in visits if kind == "node")
    assert items == sum(count for kind, count in visits if kind == "item")
    assert items <= points.shape[0] * centers.shape[0]
    assert owners and all(count > 0 for count in storage)


@pytest.mark.parametrize("refusal", ("storage", "node", "item"))
def test_host_minimum_refusal_publishes_no_partial_objective(refusal: str) -> None:
    centers = np.asarray(((0.0, 0.0), (1.0, 0.0), (2.0, 0.0)))
    tree = prepare_bvh(
        centers, centers, policy=BVHBuildPolicy(leaf_size=1), dtype=np.float64
    )
    events: list[str] = []

    def storage(count: int) -> None:
        assert count > 0
        events.append("storage")
        if refusal == "storage":
            raise ValueError("original scratch allowance exhausted")

    def visit(kind: str, count: int) -> None:
        assert count > 0
        events.append(kind)
        if kind == refusal:
            raise ValueError("original query allowance exhausted")

    def bound(
        point: np.ndarray,
        node: int,
        lower: np.ndarray,
        upper: np.ndarray,
        out: np.ndarray,
    ) -> None:
        del point, node, lower, upper
        events.append("bound")
        out[:] = -np.inf

    def values(point: np.ndarray, items: np.ndarray, out: np.ndarray) -> None:
        events.append("value")
        out[:, 0] = np.linalg.norm(centers[items] - point, axis=1)

    with pytest.raises(ValueError, match="original .* allowance"):
        bvh_host_minima(
            tree,
            centers[:1],
            objective_count=1,
            node_lower_bounds=bound,
            item_values=values,
            visit=visit,
            admit_storage=storage,
            retain_owner=lambda owner: None,
        )
    assert "value" not in events
    if refusal in ("storage", "node"):
        assert "bound" not in events


def _affine_source_cover_case(dimension: int) -> tuple[Any, np.ndarray, np.ndarray]:
    from phydrax.geometry._mesh_certificates import ParametricCurveBoundarySource
    from phydrax.geometry.brep._patches import LineCurve

    if dimension == 2:
        coordinates = np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 0.5), (0.0, 0.5)))
        items = np.asarray(((0, 1), (1, 2), (2, 3), (3, 0), (0, 1)))
        curve = LineCurve((0.2, 0.15), (0.5, 0.1))
    else:
        coordinates = np.asarray(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (1.0, 1.0, 0.0),
                (0.0, 0.0, 0.35),
                (1.0, 0.0, 0.35),
                (0.0, 1.0, 0.35),
                (1.0, 1.0, 0.35),
            )
        )
        items = np.asarray(((0, 1, 2), (2, 1, 3), (4, 5, 6), (6, 5, 7), (0, 1, 2)))
        curve = LineCurve((0.1, 0.2, 0.15), (0.7, 0.2, 0.1))
    source = ParametricCurveBoundarySource(
        (curve,),
        ((0.0, 1.0),),
        source_id="independent-line-source",
        source_revision="r1",
        covering_radius=0.07,
        maximum_samples=128,
    )
    return source, coordinates, items


@pytest.mark.parametrize("dimension", (2, 3), ids=("closed-segments", "closed-triangles"))
def test_affine_source_cover_matches_exhaustive_and_independent_physical_distance(
    dimension: int,
) -> None:
    from phydrax.geometry._mesh_certificates import (
        _EmbeddingState,
        _point_simplex_distances,
        _source_to_mesh,
        MeshCertificateLimits,
    )

    source, coordinates, items = _affine_source_cover_case(dimension)
    limits = MeshCertificateLimits()
    samples = source.boundary_samples(limits.maximum_source_samples)
    assert samples.complete and samples.semantics == "certified"
    exhaustive = np.min(
        _point_simplex_distances(samples.points, coordinates[items]), axis=1
    )
    coordinate = samples.points[:, dimension - 1]
    physical = np.minimum(coordinate, (0.5 if dimension == 2 else 0.35) - coordinate)
    np.testing.assert_allclose(
        exhaustive, physical, rtol=0.0, atol=128 * np.finfo(np.float64).eps
    )
    scale = max(float(np.max(np.abs(coordinates))), float(np.max(np.abs(samples.points))))
    epsilon = np.finfo(np.float64).eps
    slack = 32 * epsilon * scale
    upper_bounds = exhaustive * (1 + 16 * epsilon) + slack + samples.covering_radius
    lower_bounds = np.maximum(
        exhaustive * (1 - 16 * epsilon) - slack - samples.point_error, 0.0
    )
    expected_upper, expected_lower = (
        float(np.max(upper_bounds)),
        float(np.max(lower_bounds)),
    )
    state = _EmbeddingState([], [])
    actual = _source_to_mesh(state, source, coordinates, items, 0.3, limits)
    assert actual[0] == "certified" and actual[3] == samples.points.shape[0]
    np.testing.assert_allclose(
        actual[1:3], (expected_upper, expected_lower), rtol=0.0, atol=128 * epsilon
    )
    expected_findings = []
    violated = lower_bounds > 0.3
    unresolved = (upper_bounds > 0.3) & ~violated
    for status, mask in (("violated", violated), ("unresolved", unresolved)):
        if np.any(mask):
            expected_findings.append(
                ("source_uncovered", status, tuple(np.flatnonzero(mask)))
            )
    assert [
        (finding.check, finding.status, finding.entity_ids) for finding in state.findings
    ] == expected_findings


@pytest.mark.parametrize(
    "resource, override, finding",
    (
        ("queries", {"maximum_distance_evaluations": 1}, "distance_capacity"),
        ("work", {"maximum_work_units": 1}, "source_cover_expression_budget"),
        ("storage", {"maximum_scratch_bytes": 1}, "source_cover_expression_budget"),
    ),
)
def test_affine_source_cover_refuses_original_resources_without_changing_source(
    resource: str,
    override: dict[str, int],
    finding: str,
) -> None:
    from phydrax._fingerprint import array_tree_fingerprint
    from phydrax.geometry._mesh_certificates import (
        _EmbeddingState,
        _source_to_mesh,
        MeshCertificateLimits,
    )

    source, coordinates, items = _affine_source_cover_case(3)
    original = array_tree_fingerprint((source, coordinates, items))
    state = _EmbeddingState([], [])
    actual = _source_to_mesh(
        state, source, coordinates, items, 0.3, MeshCertificateLimits(**override)
    )
    assert np.isinf(actual[1]) and actual[2] == 0.0
    assert len(state.findings) == 1 and state.findings[0].check == finding
    assert state.findings[0].status == "unresolved"
    if resource == "queries":
        assert state.findings[0].resource == "distance_evaluations"
        requested, achieved = (
            dict(state.findings[0].requested),
            dict(state.findings[0].achieved),
        )
        assert requested["requested"] > requested["limit"]
        assert achieved["completed"] <= requested["limit"]
    else:
        assert state.findings[0].resource == (
            "coefficient_work" if resource == "work" else "polynomial_storage"
        )
    assert array_tree_fingerprint((source, coordinates, items)) == original
