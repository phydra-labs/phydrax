from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from phydrax.discretization.spatial import (
    MortonAddressPlan,
    MortonNeighborQueryPlan,
    MortonRadiusRelationPlan,
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
        MortonAddressPlan(
            (0.0, 0.0),
            (1.0, 1.0),
            10,
            periodic_axes=periodic,
        ),
        sources,
        targets,
        neighbors,
        maximum_candidates=candidates,
        maximum_leaf_occupancy=2,
        coarsening_factor=2,
        target_top_nodes=1,
    )


def test_morton_neighbor_query_matches_exact_dense_order() -> None:
    source = jnp.asarray(
        [[0.1, 0.1], [0.8, 0.9], [0.2, 0.7], [0.7, 0.2], [0.4, 0.4], [0.6, 0.6]]
    )
    target = jnp.asarray([[0.15, 0.2], [0.75, 0.75], [0.5, 0.3]])
    result = _plan(6, 3, 3).query(source, target)
    assert bool(result.evidence.successful)
    expected = jnp.argsort(
        jnp.sum((target[:, None, :] - source[None, :, :]) ** 2, axis=-1),
        axis=1,
        stable=True,
    )[:, :3]
    np.testing.assert_array_equal(result.source_indices, expected)
    np.testing.assert_array_equal(result.valid, True)
    np.testing.assert_array_equal(result.counts, 3)


def test_morton_neighbor_query_preserves_stable_ties_and_self_exclusion() -> None:
    source = jnp.asarray([[0.25, 0.5], [0.75, 0.5], [0.5, 0.25], [0.5, 0.75]])
    stable_ids = jnp.asarray([40, 10, 30, 20])
    tied = _plan(4, 1, 4).query(
        source,
        jnp.asarray([[0.5, 0.5]]),
        source_stable_ids=stable_ids,
    )
    np.testing.assert_array_equal(tied.source_indices, [[1, 3, 2, 0]])

    self_query = _plan(4, 4, 3).query(
        source,
        source,
        source_stable_ids=stable_ids,
        target_stable_ids=stable_ids,
        exclude_self=True,
    )
    assert bool(self_query.evidence.successful)
    gathered_ids = stable_ids[self_query.source_indices]
    assert bool(jnp.all(gathered_ids != stable_ids[:, None]))
    np.testing.assert_array_equal(self_query.counts, 3)


def test_morton_neighbor_query_handles_masks_radius_and_periodicity() -> None:
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


def test_morton_neighbor_query_fails_closed_on_candidate_overflow() -> None:
    source = jnp.asarray([[0.1, 0.5], [0.2, 0.5], [0.3, 0.5], [0.4, 0.5]])
    target = jnp.asarray([[0.15, 0.5]])
    result = _plan(4, 1, 1, candidates=1).query(source, target)
    assert not bool(result.evidence.successful)
    assert int(result.evidence.required_candidates) > 1
    np.testing.assert_array_equal(result.valid, False)
    np.testing.assert_array_equal(result.counts, 0)


def test_morton_neighbor_query_jits() -> None:
    source = jnp.asarray([[0.1, 0.1], [0.3, 0.3], [0.6, 0.6], [0.9, 0.9]])
    target = jnp.asarray([[0.2, 0.2], [0.8, 0.8]])
    query = eqx.filter_jit(_plan(4, 2, 2).query)
    result = query(source, target)
    assert bool(result.evidence.successful)
    np.testing.assert_array_equal(result.counts, 2)


def test_morton_radius_relation_is_stable_pair_once_and_periodic() -> None:
    address = MortonAddressPlan(
        (0.0,),
        (1.0,),
        16,
        periodic_axes=(True,),
    )
    plan = MortonRadiusRelationPlan(
        address,
        4,
        4,
        6,
        maximum_leaf_occupancy=2,
        coarsening_factor=2,
        target_top_nodes=1,
    )
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


def test_morton_radius_relation_boundary_and_overflow_are_explicit() -> None:
    address = MortonAddressPlan((0.0,), (1.0,), 8)
    points = jnp.asarray([[0.0], [0.25], [0.5], [0.75]])
    closed = MortonRadiusRelationPlan(
        address,
        4,
        4,
        3,
        inclusive=True,
        maximum_leaf_occupancy=2,
    ).query(points, points, 0.25, exclude_self=True, pair_once=True)
    assert bool(closed.evidence.successful)
    assert int(closed.evidence.required_pairs) == 3

    open_result = MortonRadiusRelationPlan(
        address,
        4,
        4,
        3,
        inclusive=False,
        maximum_leaf_occupancy=2,
    ).query(points, points, 0.25, exclude_self=True, pair_once=True)
    assert bool(open_result.evidence.successful)
    assert int(open_result.evidence.required_pairs) == 0

    overflow = MortonRadiusRelationPlan(
        address,
        4,
        4,
        2,
        inclusive=True,
        maximum_leaf_occupancy=2,
    ).query(points, points, 0.25, exclude_self=True, pair_once=True)
    assert not bool(overflow.evidence.successful)
    assert bool(overflow.evidence.pair_overflow)
    np.testing.assert_array_equal(overflow.relation.valid, False)
