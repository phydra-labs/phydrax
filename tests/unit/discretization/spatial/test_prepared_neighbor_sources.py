from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from phydrax.discretization.spatial import (
    MortonAddressPlan,
    MortonNeighborQueryPlan,
    MortonNeighborQueryStatus,
    MortonRadiusRelationPlan,
    MortonRadiusShellWitnessPlan,
    MortonSourcePlan,
    PreparedMortonSources,
)


def _source_plan(capacity: int = 4, *, periodic: bool = False) -> MortonSourcePlan:
    return MortonSourcePlan(
        MortonAddressPlan((0.0,), (1.0,), 12, periodic_axes=(periodic,)), capacity
    )


def _points() -> Array:
    return jnp.asarray([[0.125], [0.375], [0.625], [0.875]], dtype=jnp.float64)


def test_one_artifact_serves_nearest_radius_and_shell_target_batches() -> None:
    plan = _source_plan()
    points = _points()
    ids = jnp.asarray([40, 10, 30, 20], dtype=jnp.int64)
    sources = plan.prepare(points, stable_ids=ids)
    nearest = MortonNeighborQueryPlan(plan, 2, 2)
    radius = MortonRadiusRelationPlan(plan, 2, 4)
    shell = MortonRadiusShellWitnessPlan(plan, 2)
    for targets in (
        jnp.asarray([[0.25], [0.75]], dtype=jnp.float64),
        jnp.asarray([[0.75], [0.25]], dtype=jnp.float64),
    ):
        result = nearest.query(sources, targets, radius=0.125)
        assert bool(result.evidence.successful)
        squared = np.square(np.asarray(targets) - np.asarray(points).T)
        expected = np.stack([np.lexsort((np.asarray(ids), row))[:2] for row in squared])
        np.testing.assert_array_equal(result.source_indices, expected)
        relation = radius.query(sources, targets, 0.125)
        assert bool(relation.evidence.successful)
        assert int(relation.evidence.required_pairs) == 4
        witness = shell.certify(sources, targets, 0.25, 0.03)
        assert bool(witness.evidence.certified)
    np.testing.assert_array_equal(sources.points, points)
    np.testing.assert_array_equal(sources.stable_ids, ids)


@pytest.mark.parametrize("kind", ["nearest", "radius", "shell"])
def test_queries_refuse_raw_sources_and_foreign_source_plans(kind: str) -> None:
    plan = _source_plan()
    foreign = MortonSourcePlan(MortonAddressPlan((-1.0,), (1.0,), 12), 4)
    targets = jnp.asarray([[0.25], [0.75]], dtype=jnp.float64)

    def query(sources: Array | PreparedMortonSources) -> None:
        match kind:
            case "nearest":
                MortonNeighborQueryPlan(plan, 2, 2).query(sources, targets)  # ty: ignore[invalid-argument-type]
            case "radius":
                MortonRadiusRelationPlan(plan, 2, 8).query(sources, targets, 0.2)  # ty: ignore[invalid-argument-type]
            case "shell":
                MortonRadiusShellWitnessPlan(plan, 2).certify(sources, targets, 0.2, 0.01)  # ty: ignore[invalid-argument-type]
            case _:
                raise AssertionError(kind)

    with pytest.raises(TypeError, match="PreparedMortonSources"):
        query(_points())
    with pytest.raises(ValueError, match="MortonSourcePlan"):
        query(foreign.prepare(_points()))


def test_masks_invalid_geometry_and_duplicate_active_ids_reach_query_evidence() -> None:
    plan = _source_plan()
    query = MortonNeighborQueryPlan(plan, 2, 2)
    points = _points().at[3, 0].set(jnp.nan)
    ids = jnp.asarray([5, 8, 8, 5], dtype=jnp.uint64)
    mask = jnp.asarray([True, True, False, False], dtype=jnp.bool_)
    sources = plan.prepare(points, active_mask=mask, stable_ids=ids)
    result = query.query(sources, _points()[:2])
    assert bool(result.evidence.successful)
    assert sources.stable_ids.dtype == ids.dtype
    assert int(sources.active_count) == 2
    np.testing.assert_array_equal(sources.active, mask)
    assert not bool(sources.finite[3])
    duplicate = plan.prepare(points, active_mask=mask.at[2].set(True), stable_ids=ids)
    refused = query.query(duplicate, _points()[:2])
    assert not bool(refused.evidence.sources_valid)
    assert not bool(jnp.any(refused.valid))
    np.testing.assert_array_equal(
        refused.status, MortonNeighborQueryStatus.INVALID_SOURCES
    )
    invalid = plan.prepare(points, stable_ids=jnp.arange(4, dtype=jnp.int64))
    invalid_result = query.query(invalid, _points()[:2])
    assert not bool(invalid_result.evidence.finite)
    assert int(invalid_result.evidence.invalid_sources) == 1


def test_empty_active_set_is_a_complete_empty_query_and_witness() -> None:
    plan = _source_plan()
    points = jnp.full((4, 1), jnp.nan, dtype=jnp.float32)
    sources = plan.prepare(points, active_mask=jnp.zeros((4,), dtype=jnp.bool_))
    targets = jnp.asarray([[0.25], [0.75]], dtype=jnp.float32)
    result = MortonNeighborQueryPlan(plan, 2, 3).query(sources, targets)
    assert bool(result.evidence.successful)
    np.testing.assert_array_equal(result.counts, 0)
    assert not bool(jnp.any(result.valid))
    radius = MortonRadiusRelationPlan(plan, 2, 0).query(sources, targets, 0.125)
    assert bool(radius.evidence.successful)
    assert int(radius.evidence.required_pairs) == 0
    shell = MortonRadiusShellWitnessPlan(plan, 2).certify(sources, targets, 0.25, 0.03)
    assert bool(shell.evidence.certified)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_prepared_logical_and_wrapped_geometry_remain_differentiable(
    dtype: jnp.dtype,
) -> None:
    plan = _source_plan(periodic=True)
    points = _points().astype(dtype).at[0, 0].set(1.125)
    query = MortonNeighborQueryPlan(plan, 1, 1)
    target = jnp.asarray([[0.2]], dtype=dtype)

    def squared_distance(values: Array) -> Array:
        sources = plan.prepare(values)
        result = query.query(sources, target)
        relative = target[0] - sources.points[result.source_indices[0, 0]]
        relative = relative - jnp.round(relative)
        return jnp.sum(relative * relative)

    gradient = jax.grad(squared_distance)(points)
    np.testing.assert_allclose(
        gradient[:, 0], [-0.15, 0.0, 0.0, 0.0], rtol=2e-6, atol=2e-7
    )
    wrapped_gradient = jax.grad(
        lambda values: jnp.sum(plan.prepare(values).sorted_coordinates)
    )(points)
    np.testing.assert_array_equal(wrapped_gradient, jnp.ones_like(points))
    assert plan.prepare(points).points.dtype == dtype
    coincident = points.at[0, 0].set(target[0, 0] + 1)
    np.testing.assert_allclose(
        jax.grad(squared_distance)(coincident), 0.0, rtol=0, atol=2e-7
    )


def test_refresh_reuses_order_rebuilds_on_motion_and_rolls_back_rejection() -> None:
    plan = _source_plan()
    points = _points()
    mask = jnp.asarray([True, True, True, False], dtype=jnp.bool_)
    ids = jnp.asarray([40, 10, 30, 20], dtype=jnp.int64)
    previous = plan.prepare(points, active_mask=mask, stable_ids=ids)
    refresh = eqx.filter_jit(plan.refresh)
    moved = points + jnp.asarray(0.01, dtype=points.dtype)
    reused = refresh(previous, moved)
    assert bool(reused.successful)
    assert bool(reused.reused_order)
    np.testing.assert_array_equal(reused.accepted.active, mask)
    np.testing.assert_array_equal(reused.accepted.stable_ids, ids)
    np.testing.assert_array_equal(
        reused.accepted.storage_to_logical, previous.storage_to_logical
    )
    fresh = plan.prepare(moved, active_mask=mask, stable_ids=ids)
    for actual, expected in zip(
        jax.tree.leaves(reused.candidate), jax.tree.leaves(fresh), strict=True
    ):
        np.testing.assert_array_equal(actual, expected)
    exchanged = points.at[0, 0].set(0.65).at[2, 0].set(0.1)
    rebuilt = refresh(previous, exchanged)
    assert bool(rebuilt.successful)
    assert not bool(rebuilt.reused_order)
    assert not bool(
        jnp.all(rebuilt.accepted.storage_to_logical == previous.storage_to_logical)
    )
    activated = refresh(previous, points, active_mask=jnp.ones((4,), dtype=jnp.bool_))
    assert bool(activated.successful)
    assert not bool(activated.reused_order)
    changed_ids = refresh(previous, points, stable_ids=ids.at[0].set(90))
    assert bool(changed_ids.successful)
    assert not bool(changed_ids.reused_order)
    for invalid_points, invalid_ids in (
        (points.at[0, 0].set(1.1), ids),
        (points, ids.at[0].set(10)),
    ):
        rejected = refresh(previous, invalid_points, stable_ids=invalid_ids)
        assert not bool(rejected.successful)
        assert not bool(rejected.reused_order)
        np.testing.assert_array_equal(rejected.accepted.points, previous.points)
        np.testing.assert_array_equal(rejected.accepted.stable_ids, previous.stable_ids)


def test_refresh_keeps_live_geometry_on_sort_free_path() -> None:
    plan = _source_plan()
    previous = plan.prepare(_points())
    gradient = jax.grad(
        lambda values: jnp.sum(
            plan.refresh(previous, values).accepted.sorted_coordinates ** 2
        )
    )(_points())
    np.testing.assert_allclose(gradient, 2 * _points(), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [jnp.int64, jnp.uint64])
def test_exact_ties_use_full_width_ids_and_float64_distance(dtype: jnp.dtype) -> None:
    plan = _source_plan()
    limits = np.iinfo(np.dtype(dtype))
    ids = jnp.asarray([limits.max, limits.min, 13, 7], dtype=dtype)
    points = jnp.full((4, 1), 0.5, dtype=jnp.float64)
    sources = plan.prepare(points, stable_ids=ids)
    for neighbors in (1, 3, 4):
        result = MortonNeighborQueryPlan(plan, 1, neighbors).query(sources, points[:1])
        assert bool(result.evidence.successful)
        np.testing.assert_array_equal(
            result.source_indices[0],
            np.argsort(np.asarray(ids), kind="stable")[:neighbors],
        )
    precise = points.at[0, 0].set(jnp.nextafter(jnp.float64(0.5), jnp.float64(1.0)))
    result = MortonNeighborQueryPlan(plan, 1, 1).query(plan.prepare(precise), points[:1])
    np.testing.assert_array_equal(result.source_indices[0], [1])


def test_source_validation_and_shell_declared_identity_relationship() -> None:
    plan = _source_plan()
    with pytest.raises(TypeError, match="floating"):
        plan.prepare(jnp.ones((4, 1), dtype=jnp.int32))
    with pytest.raises(ValueError, match="shape"):
        plan.prepare(jnp.ones((3, 1), dtype=jnp.float64))
    with pytest.raises(TypeError, match="integer"):
        plan.prepare(_points(), stable_ids=jnp.ones((4,), dtype=jnp.float64))
    shell = MortonRadiusShellWitnessPlan(plan, 4)
    with pytest.raises(ValueError, match="both or neither"):
        shell.certify(
            plan.prepare(_points(), stable_ids=jnp.arange(4, dtype=jnp.int64)),
            _points(),
            0.2,
            0.01,
            exclude_self=True,
        )
    with pytest.raises(ValueError, match="both or neither"):
        shell.certify(
            plan.prepare(_points()),
            _points(),
            0.2,
            0.01,
            target_stable_ids=jnp.arange(4, dtype=jnp.int64),
            exclude_self=True,
        )


def test_prepared_queries_promote_mixed_target_precision_without_narrowing() -> None:
    plan = _source_plan()
    points = _points().astype(jnp.float32)
    # Target precision distinguishes the two exactly equidistant float32 sources.
    target = jnp.asarray([[np.nextafter(0.25, 1.0)]], dtype=jnp.float64)
    sources = plan.prepare(points)
    result = MortonNeighborQueryPlan(plan, 1, 1).query(sources, target)
    assert bool(result.evidence.successful)
    np.testing.assert_array_equal(result.source_indices, [[1]])
    assert sources.points.dtype == jnp.float32


def test_refresh_refuses_static_precision_changes_and_foreign_binding() -> None:
    plan = _source_plan()
    previous = plan.prepare(_points())
    with pytest.raises(ValueError, match="coordinate dtype"):
        plan.refresh(previous, _points().astype(jnp.float32))
    with pytest.raises(ValueError, match="stable-ID dtype"):
        plan.refresh(previous, _points(), stable_ids=jnp.arange(4, dtype=jnp.int32))
    with pytest.raises(ValueError, match="MortonSourcePlan"):
        _source_plan(3).refresh(previous, _points()[:3])
