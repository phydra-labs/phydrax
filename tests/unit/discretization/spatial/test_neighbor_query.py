from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from phydrax.discretization.spatial import (
    MortonAddressPlan,
    MortonNeighborQueryPlan,
    MortonNeighborQueryStatus,
    MortonRadiusRelationPlan,
)


def _address(
    dimension: int = 2, *, periodic: bool = False, depth: int = 10
) -> MortonAddressPlan:
    return MortonAddressPlan(
        (0.0,) * dimension,
        (1.0,) * dimension,
        depth,
        periodic_axes=(periodic,) * dimension,
    )


def _plan(
    sources: int,
    targets: int,
    neighbors: int,
    *,
    candidates: int | None = None,
    periodic: tuple[bool, bool] = (False, False),
) -> MortonNeighborQueryPlan:
    return MortonNeighborQueryPlan(
        MortonAddressPlan((0.0, 0.0), (1.0, 1.0), 10, periodic_axes=periodic),
        sources,
        targets,
        neighbors,
        maximum_candidates=candidates,
    )


def _clustered(
    rng: np.random.Generator, count: int, dimension: int, *, periodic: bool
) -> np.ndarray:
    centers = rng.uniform(0.1, 0.9, (5, dimension))
    labels = rng.integers(0, 5, count)
    points = centers[labels] + rng.normal(scale=0.04, size=(count, dimension))
    points = np.mod(points, 1.0) if periodic else np.clip(points, 0.0, 0.999)
    # Coincident copies create exact distance ties that must order by stable ID.
    points[count - count // 8 :] = points[: count // 8]
    return points


def _squared_distances(
    target: np.ndarray, source: np.ndarray, *, periodic: bool
) -> np.ndarray:
    relative = target[:, None, :] - source[None, :, :]
    if periodic:
        relative = relative - np.round(relative) * 1.0
    return np.sum(relative * relative, axis=-1)


def test_nearest_neighbors_match_brute_force_on_clustered_points() -> None:
    for dimension, periodic, exclude_self, radius in [
        (2, False, False, None),
        (2, True, True, None),
        (3, False, True, None),
        (3, True, False, 0.08),
    ]:
        rng = np.random.default_rng(7 + dimension)
        source = _clustered(rng, 400, dimension, periodic=periodic)
        source_ids = rng.permutation(400).astype(np.int64) * 3 + 11
        active = rng.uniform(size=400) > 0.1
        if exclude_self:
            target, target_ids = source, source_ids
        else:
            jittered = source[rng.integers(0, 400, 80)] + rng.normal(
                scale=0.01, size=(80, dimension)
            )
            jittered = (
                np.mod(jittered, 1.0) if periodic else np.clip(jittered, 0.0, 0.999)
            )
            target = np.concatenate((source[:40], jittered))
            target_ids = np.arange(120, dtype=np.int64)
        k = 6
        result = MortonNeighborQueryPlan(
            _address(dimension, periodic=periodic),
            source.shape[0],
            target.shape[0],
            k,
            maximum_candidates=256,
            target_chunk_size=32,
        ).query(
            jnp.asarray(source),
            jnp.asarray(target),
            source_mask=jnp.asarray(active),
            source_stable_ids=jnp.asarray(source_ids),
            target_stable_ids=jnp.asarray(target_ids),
            exclude_self=exclude_self,
            radius=radius,
        )

        assert bool(result.evidence.successful)
        np.testing.assert_array_equal(result.status, MortonNeighborQueryStatus.COMPLETE)
        distances = _squared_distances(target, source, periodic=periodic)
        eligible = active[None, :] & np.ones_like(distances, dtype=bool)
        if exclude_self:
            eligible &= source_ids[None, :] != target_ids[:, None]
        if radius is not None:
            eligible &= distances <= radius**2
        for row in range(target.shape[0]):
            candidates = np.flatnonzero(eligible[row])
            order = np.lexsort((source_ids[candidates], distances[row, candidates]))
            expected = candidates[order][:k]
            count = int(result.counts[row])
            assert count == expected.size
            np.testing.assert_array_equal(result.source_indices[row, :count], expected)
            np.testing.assert_allclose(
                distances[row, np.asarray(result.source_indices[row, :count])],
                distances[row, expected],
            )


def test_radius_relation_matches_brute_force_on_clustered_points() -> None:
    for dimension, periodic, pair_once in [
        (2, False, False),
        (2, True, True),
        (3, True, False),
        (3, False, True),
    ]:
        rng = np.random.default_rng(31 + dimension)
        points = _clustered(rng, 300, dimension, periodic=periodic)
        ids = rng.permutation(300).astype(np.int64) + 1000
        radius = 0.06
        result = MortonRadiusRelationPlan(
            _address(dimension, periodic=periodic),
            300,
            300,
            20_000,
            maximum_candidates=300,
            target_chunk_size=64,
        ).query(
            jnp.asarray(points),
            jnp.asarray(points),
            radius,
            source_stable_ids=jnp.asarray(ids),
            target_stable_ids=jnp.asarray(ids),
            exclude_self=True,
            pair_once=pair_once,
        )

        assert bool(result.evidence.successful)
        distances = _squared_distances(points, points, periodic=periodic)
        target, source = np.nonzero(distances <= radius**2)
        keep = ids[target] != ids[source]
        if pair_once:
            keep &= ids[source] > ids[target]
            first, second = target[keep], source[keep]
        else:
            first, second = source[keep], target[keep]
        major = ids[first] if pair_once else ids[second]
        minor = ids[second] if pair_once else ids[first]
        order = np.lexsort((minor, major))
        valid = np.asarray(result.relation.valid)
        assert int(result.evidence.required_pairs) == order.size
        np.testing.assert_array_equal(
            np.asarray(result.relation.source_indices)[valid], first[order]
        )
        np.testing.assert_array_equal(
            np.asarray(result.relation.target_indices)[valid], second[order]
        )


def test_duplicate_and_equidistant_ties_order_by_stable_id() -> None:
    source = jnp.asarray(
        [
            [0.5, 0.5],
            [0.25, 0.5],
            [0.5, 0.5],
            [0.75, 0.5],
            [0.5, 0.5],
            [0.5, 0.25],
            [0.5, 0.75],
        ]
    )
    stable_ids = jnp.asarray([50, 70, 20, 40, 30, 10, 60])
    result = _plan(7, 1, 7).query(
        source,
        jnp.asarray([[0.5, 0.5]]),
        source_stable_ids=stable_ids,
    )
    assert bool(result.evidence.successful)
    np.testing.assert_array_equal(
        stable_ids[result.source_indices[0]], [20, 30, 50, 10, 40, 60, 70]
    )


def test_nearest_neighbors_exclude_self_with_stable_ids() -> None:
    source = jnp.asarray([[0.25, 0.5], [0.75, 0.5], [0.5, 0.25], [0.5, 0.75]])
    stable_ids = jnp.asarray([40, 10, 30, 20])
    result = _plan(4, 4, 3).query(
        source,
        source,
        source_stable_ids=stable_ids,
        target_stable_ids=stable_ids,
        exclude_self=True,
    )
    assert bool(result.evidence.successful)
    gathered_ids = stable_ids[result.source_indices]
    assert bool(jnp.all(gathered_ids != stable_ids[:, None]))
    np.testing.assert_array_equal(result.counts, 3)


def test_nearest_neighbors_handle_masks_radius_and_periodicity() -> None:
    source = jnp.asarray([[0.98, 0.5], [0.1, 0.5], [0.5, 0.5], [0.75, 0.5]])
    target = jnp.asarray([[0.02, 0.5], [0.5, 0.5]])
    result = _plan(4, 2, 3, periodic=(True, False)).query(
        source,
        target,
        source_mask=jnp.asarray([True, True, False, True]),
        target_mask=jnp.asarray([True, False]),
        radius=0.2,
    )
    assert bool(result.evidence.successful)
    np.testing.assert_array_equal(result.source_indices[0, :2], [0, 1])
    np.testing.assert_array_equal(result.valid[0], [True, True, False])
    np.testing.assert_array_equal(result.valid[1], False)
    np.testing.assert_array_equal(result.counts, [2, 0])
    np.testing.assert_array_equal(
        result.status,
        [MortonNeighborQueryStatus.COMPLETE, MortonNeighborQueryStatus.INACTIVE_TARGET],
    )


def test_tiny_candidate_capacity_reports_overflow_per_row() -> None:
    crowded = [[0.3, 0.3]] * 8
    source = jnp.asarray(crowded + [[0.9, 0.9], [0.92, 0.9]])
    target = jnp.asarray([[0.3, 0.3], [0.9, 0.91]])
    result = _plan(10, 2, 2, candidates=4).query(source, target)

    assert not bool(result.evidence.successful)
    assert int(result.evidence.overflow_rows) == 1
    assert int(result.evidence.required_candidates) >= 8
    np.testing.assert_array_equal(
        result.status,
        [
            MortonNeighborQueryStatus.CANDIDATE_OVERFLOW,
            MortonNeighborQueryStatus.COMPLETE,
        ],
    )
    np.testing.assert_array_equal(result.valid[0], False)
    np.testing.assert_array_equal(result.source_indices[1], [8, 9])

    relation = MortonRadiusRelationPlan(
        _address(), 10, 10, 64, maximum_candidates=4
    ).query(source, source, 0.05, exclude_self=True)
    assert not bool(relation.evidence.successful)
    assert not bool(relation.evidence.complete)
    assert int(relation.evidence.overflow_rows) == 8
    np.testing.assert_array_equal(relation.relation.valid, False)


def test_out_of_domain_targets_and_sources_fail_closed() -> None:
    source = jnp.asarray([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4]])
    target = jnp.asarray([[0.15, 0.15], [1.5, 0.5]])
    result = _plan(4, 2, 2).query(source, target)
    np.testing.assert_array_equal(
        result.status,
        [MortonNeighborQueryStatus.COMPLETE, MortonNeighborQueryStatus.INVALID_TARGET],
    )
    assert int(result.evidence.invalid_targets) == 1
    assert not bool(result.evidence.successful)

    invalid_source = _plan(4, 2, 2).query(
        source.at[3, 0].set(jnp.nan), target[:1].repeat(2, 0)
    )
    assert not bool(invalid_source.evidence.sources_valid)
    assert not bool(invalid_source.evidence.finite)
    np.testing.assert_array_equal(
        invalid_source.status, MortonNeighborQueryStatus.INVALID_SOURCES
    )
    np.testing.assert_array_equal(invalid_source.valid, False)


def test_nearest_neighbor_query_jits() -> None:
    source = jnp.asarray([[0.1, 0.1], [0.3, 0.3], [0.6, 0.6], [0.9, 0.9]])
    target = jnp.asarray([[0.2, 0.2], [0.8, 0.8]])
    query = eqx.filter_jit(_plan(4, 2, 2).query)
    result = query(source, target)
    assert bool(result.evidence.successful)
    np.testing.assert_array_equal(result.counts, 2)


def test_radius_relation_is_stable_pair_once_and_periodic() -> None:
    plan = MortonRadiusRelationPlan(_address(1, periodic=True, depth=16), 4, 4, 6)
    points = jnp.asarray([[0.98], [0.03], [0.4], [0.65]])
    stable_ids = jnp.asarray([40, 10, 30, 20])
    result = plan.query(
        points,
        points,
        0.26,
        source_stable_ids=stable_ids,
        target_stable_ids=stable_ids,
        exclude_self=True,
        pair_once=True,
    )
    assert bool(result.evidence.successful)
    left_ids = stable_ids[result.relation.source_indices[result.relation.valid]]
    right_ids = stable_ids[result.relation.target_indices[result.relation.valid]]
    np.testing.assert_array_equal(left_ids, [10, 20])
    np.testing.assert_array_equal(right_ids, [40, 30])
    assert int(jnp.sum(result.cell_counts)) == 4


def test_radius_relation_boundary_and_pair_overflow_are_explicit() -> None:
    address = MortonAddressPlan((0.0,), (1.0,), 8)
    points = jnp.asarray([[0.0], [0.25], [0.5], [0.75]])
    closed = MortonRadiusRelationPlan(address, 4, 4, 3, inclusive=True).query(
        points, points, 0.25, exclude_self=True, pair_once=True
    )
    assert bool(closed.evidence.successful)
    assert int(closed.evidence.required_pairs) == 3

    open_result = MortonRadiusRelationPlan(address, 4, 4, 3, inclusive=False).query(
        points, points, 0.25, exclude_self=True, pair_once=True
    )
    assert bool(open_result.evidence.successful)
    assert int(open_result.evidence.required_pairs) == 0

    overflow = MortonRadiusRelationPlan(address, 4, 4, 2, inclusive=True).query(
        points, points, 0.25, exclude_self=True, pair_once=True
    )
    assert not bool(overflow.evidence.successful)
    assert bool(overflow.evidence.pair_overflow)
    assert int(overflow.evidence.required_pairs) == 3
    np.testing.assert_array_equal(overflow.relation.valid, False)
