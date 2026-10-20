from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.solver import (
    CartesianExpansionSpace,
    PreparedUniformFMMStructure,
    UniformFMMPlan,
)
from phydrax.solver._particle_gravity import _broadcast_owner_rows
from phydrax.sparse import RelationAccumulation


_SOFTENING = 0.01
_BOX = (1.0, 1.0, 1.0)
_DEPTH = 8


def _points(count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Eight separated Gaussian clusters, so the dual tree accepts far pairs."""
    generator = np.random.default_rng(seed)
    corners = np.asarray(
        [[a, b, c] for a in (0.25, 0.75) for b in (0.25, 0.75) for c in (0.25, 0.75)]
    )
    centers = corners[np.arange(count) % corners.shape[0]]
    positions = np.clip(centers + 0.05 * generator.normal(size=(count, 3)), 0.02, 0.98)
    strengths = generator.normal(size=count)
    return positions, strengths


def _direct(
    positions: np.ndarray,
    strengths: np.ndarray,
    active: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Host O(N^2) reference for G = 1 over active sources and targets."""
    displacement = positions[None, :, :] - positions[:, None, :]
    radius_squared = np.sum(displacement**2, axis=-1) + _SOFTENING**2
    pair = active[:, None] & active[None, :] & ~np.eye(positions.shape[0], dtype=np.bool_)
    potential = np.where(pair, 1.0 / np.sqrt(radius_squared), 0.0) @ strengths
    weight = np.where(pair, radius_squared**-1.5, 0.0) * strengths[None, :]
    gradient = np.einsum("ij,ijk->ik", weight, displacement)
    return potential, gradient


def _plan(
    order: int,
    *,
    maximum_far_interactions: int | None = None,
    maximum_nodes: int | None = None,
    accumulation: RelationAccumulation = "deterministic",
) -> UniformFMMPlan:
    # Small leaves and binary coarsening give several planes, so M2M/M2L/L2L
    # carry far interactions at these test sizes.
    return UniformFMMPlan(
        1.0,
        CartesianExpansionSpace(order),
        softening=_SOFTENING,
        opening_angle=0.5,
        maximum_nodes=maximum_nodes,
        maximum_far_interactions=maximum_far_interactions,
        maximum_leaf_occupancy=8,
        coarsening_factor=2,
        target_top_nodes=1,
        accumulation=accumulation,
    )


def _relative_errors(
    potential: jax.Array,
    gradient: jax.Array,
    reference_potential: np.ndarray,
    reference_gradient: np.ndarray,
) -> tuple[float, float]:
    potential_error = np.max(np.abs(np.asarray(potential) - reference_potential))
    gradient_error = np.max(
        np.linalg.norm(np.asarray(gradient) - reference_gradient, axis=-1)
    )
    return (
        float(potential_error / np.max(np.abs(reference_potential))),
        float(gradient_error / np.max(np.linalg.norm(reference_gradient, axis=-1))),
    )


def test_signed_monopole_field_matches_direct_sum_and_improves_with_order() -> None:
    positions, strengths = _points(256, 11)
    active = np.ones((positions.shape[0],), dtype=np.bool_)
    reference_potential, reference_gradient = _direct(positions, strengths, active)
    errors = {}
    for order in (3, 5, 6):
        plan = _plan(order)
        structure = plan.prepare_structure(positions, box_size=_BOX, depth=_DEPTH)
        result = plan.evaluate_monopole(structure, strengths)
        assert bool(result.successful)
        assert int(result.evidence.expansion_order) == order
        assert int(result.evidence.required_far) > 0
        assert int(result.evidence.p2p_count) > 0
        errors[order] = _relative_errors(
            result.potential,
            result.gradient,
            reference_potential,
            reference_gradient,
        )
    # Declared relative tolerances (max-norm error over max-norm field), about
    # twice the measured opening-angle-0.5 truncation error of each order.
    assert errors[3][0] < 3.0e-3 and errors[3][1] < 1.5e-3
    assert errors[5][0] < 5.0e-4 and errors[5][1] < 4.0e-4
    assert errors[5][0] < errors[3][0]
    assert errors[5][1] < errors[3][1]
    assert errors[6][0] < errors[5][0]
    assert errors[6][1] < errors[5][1]


def test_prepared_structure_refreshes_strengths_like_fresh_preparation() -> None:
    positions, first = _points(200, 3)
    second = np.random.default_rng(4).normal(size=first.shape)
    plan = _plan(2)
    structure = plan.prepare_structure(positions, box_size=_BOX, depth=_DEPTH)
    reused = [plan.evaluate_monopole(structure, value) for value in (first, second)]
    for value, result in zip((first, second), reused, strict=True):
        fresh = plan.evaluate_monopole(
            plan.prepare_structure(positions, box_size=_BOX, depth=_DEPTH),
            value,
        )
        np.testing.assert_array_equal(result.potential, fresh.potential)
        np.testing.assert_array_equal(result.gradient, fresh.gradient)
    combined = plan.evaluate_monopole(structure, 2.0 * first - second)
    np.testing.assert_allclose(
        combined.potential,
        2.0 * reused[0].potential - reused[1].potential,
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        combined.gradient,
        2.0 * reused[0].gradient - reused[1].gradient,
        rtol=1.0e-10,
        atol=1.0e-10,
    )


def test_prepared_structure_is_reused_inside_compiled_loop() -> None:
    positions, strengths = _points(64, 5)
    plan = _plan(2)
    structure = plan.prepare_structure(positions, box_size=_BOX, depth=_DEPTH)

    @jax.jit
    def iterate(prepared: PreparedUniformFMMStructure, initial: jax.Array) -> jax.Array:
        def body(state: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
            count, value = state
            field = plan.evaluate_monopole(prepared, value)
            return count + 1, 0.5 * value + 1.0e-3 * field.potential

        def unfinished(state: tuple[jax.Array, jax.Array]) -> jax.Array:
            return state[0] < 3

        start = (jnp.asarray(0, dtype=jnp.int32), initial)
        return jax.lax.while_loop(unfinished, body, start)[1]

    expected = jnp.asarray(strengths)
    for _ in range(3):
        field = plan.evaluate_monopole(structure, expected)
        expected = 0.5 * expected + 1.0e-3 * field.potential
    np.testing.assert_allclose(
        iterate(structure, jnp.asarray(strengths)), expected, rtol=1.0e-10
    )


def test_zero_net_dipole_sources_match_direct_sum() -> None:
    centers, charges = _points(96, 17)
    offset = np.asarray([0.004, -0.003, 0.002])
    positions = np.concatenate([centers, centers + offset])
    strengths = np.concatenate([charges, -charges])
    assert abs(float(np.sum(strengths))) < 1.0e-12
    plan = _plan(3)
    result = plan.evaluate_monopole(
        plan.prepare_structure(positions, box_size=_BOX, depth=_DEPTH),
        strengths,
    )
    active = np.ones((positions.shape[0],), dtype=np.bool_)
    reference_potential, reference_gradient = _direct(positions, strengths, active)
    assert bool(result.successful)
    assert int(result.evidence.required_far) > 0
    potential_error, gradient_error = _relative_errors(
        result.potential,
        result.gradient,
        reference_potential,
        reference_gradient,
    )
    assert potential_error < 3.0e-4
    assert gradient_error < 3.0e-4


def test_inactive_rows_are_zero_and_excluded_from_sources() -> None:
    positions, strengths = _points(160, 19)
    active = np.ones((positions.shape[0],), dtype=np.bool_)
    active[::7] = False
    positions[~active] = np.nan
    strengths[~active] = np.nan
    plan = _plan(3)
    structure = plan.prepare_structure(
        positions, box_size=_BOX, depth=_DEPTH, active_mask=active
    )
    result = plan.evaluate_monopole(structure, strengths)
    reference_potential, reference_gradient = _direct(
        np.where(active[:, None], positions, 0.0),
        np.where(active, strengths, 0.0),
        active,
    )
    assert bool(result.successful)
    assert int(result.evidence.p2m_count) == int(np.sum(active))
    np.testing.assert_array_equal(np.asarray(result.potential)[~active], 0.0)
    np.testing.assert_array_equal(np.asarray(result.gradient)[~active], 0.0)
    potential_error, gradient_error = _relative_errors(
        result.potential,
        result.gradient,
        reference_potential,
        reference_gradient,
    )
    assert potential_error < 1.5e-3
    assert gradient_error < 3.0e-3


def test_capacity_exhaustion_is_reported_not_raised() -> None:
    positions, strengths = _points(128, 23)
    far_limited = _plan(1, maximum_far_interactions=1)
    far = far_limited.evaluate_monopole(
        far_limited.prepare_structure(positions, box_size=_BOX, depth=_DEPTH),
        strengths,
    )
    assert not bool(far.successful)
    assert not bool(far.evidence.successful)
    assert int(far.evidence.required_far) > int(far.evidence.far_capacity)
    assert np.isnan(np.asarray(far.potential)).all()
    assert np.isnan(np.asarray(far.gradient)).all()

    node_limited = _plan(1, maximum_nodes=2)
    nodes = node_limited.evaluate_monopole(
        node_limited.prepare_structure(positions, box_size=_BOX, depth=_DEPTH),
        strengths,
    )
    assert not bool(nodes.successful)
    assert int(nodes.evidence.required_nodes) > int(nodes.evidence.node_capacity)
    assert np.isnan(np.asarray(nodes.potential)).all()
    assert np.isnan(np.asarray(nodes.gradient)).all()


def test_monopole_route_refuses_short_range_and_foreign_structures() -> None:
    positions, strengths = _points(16, 29)
    split = UniformFMMPlan(
        1.0,
        CartesianExpansionSpace(1),
        softening=_SOFTENING,
        short_range_scale=0.1,
        short_range_cutoff=0.5,
    )
    split_structure = split.prepare_structure(positions, box_size=_BOX, depth=_DEPTH)
    with pytest.raises(ValueError, match="short_range_scale"):
        split.evaluate_monopole(split_structure, strengths)

    structure = _plan(1).prepare_structure(positions, box_size=_BOX, depth=_DEPTH)
    with pytest.raises(ValueError, match="different UniformFMMPlan"):
        _plan(2).evaluate_monopole(structure, strengths)
    with pytest.raises(ValueError, match="one value per prepared position"):
        _plan(1).evaluate_monopole(structure, strengths[:-1])


def test_structure_refresh_reuses_geometry_and_rejects_atomically() -> None:
    positions, strengths = _points(16, 37)
    plan = _plan(1)
    previous = plan.prepare_structure(positions, box_size=_BOX, depth=_DEPTH)
    baseline = plan.evaluate_monopole(previous, strengths)

    @jax.jit
    def refresh(points: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        transition = plan.refresh_structure(previous, points, box_size=_BOX, depth=_DEPTH)
        field = plan.evaluate_monopole(transition.accepted, strengths)
        return transition.successful, transition.reused_topology, field.potential

    successful, reused, potential = refresh(jnp.asarray(positions))
    assert bool(successful)
    assert bool(reused)
    np.testing.assert_array_equal(potential, baseline.potential)
    moved = positions + 1e-9
    moved_successful, moved_reused, moved_potential = refresh(jnp.asarray(moved))
    assert bool(moved_successful)
    assert bool(moved_reused)
    fresh = plan.evaluate_monopole(
        plan.prepare_structure(moved, box_size=_BOX, depth=_DEPTH), strengths
    )
    np.testing.assert_allclose(moved_potential, fresh.potential, rtol=1e-12, atol=1e-12)
    changed = positions.copy()
    changed[0, 0] = 1.25
    rejected = plan.refresh_structure(previous, changed, box_size=_BOX, depth=_DEPTH)
    assert not bool(rejected.successful)
    assert not bool(rejected.candidate.interactions.evidence.successful)
    np.testing.assert_array_equal(rejected.accepted.positions, previous.positions)
    rolled_back = plan.evaluate_monopole(rejected.accepted, strengths)
    np.testing.assert_array_equal(rolled_back.potential, baseline.potential)
    assert not bool(plan.evaluate_monopole(rejected.candidate, strengths).successful)


def test_structure_refresh_rebuilds_when_retained_far_mac_fails() -> None:
    positions = jnp.asarray(
        [[0.05, 0.2, 0.2], [0.25, 0.2, 0.2], [0.75, 0.2, 0.2], [0.95, 0.2, 0.2]],
        dtype=jnp.float64,
    )
    plan = UniformFMMPlan(
        1.0,
        CartesianExpansionSpace(1),
        softening=_SOFTENING,
        maximum_leaf_occupancy=2,
        coarsening_factor=2,
        target_top_nodes=1,
    )
    previous = plan.prepare_structure(positions, box_size=_BOX, depth=_DEPTH)
    assert int(previous.interactions.evidence.required_far) > 0
    moved = positions.at[1, 0].set(0.45).at[2, 0].set(0.55)
    schedule_update = previous.schedule_plan.refresh(previous.schedule, moved)
    recheck = previous.interaction_plan.recheck(
        schedule_update.candidate, interactions=previous.interactions
    )
    assert not bool(recheck.successful)
    transition = plan.refresh_structure(previous, moved, box_size=_BOX, depth=_DEPTH)
    assert bool(transition.successful)
    assert bool(transition.rebuilt)
    assert not bool(transition.reused_topology)
    strengths = np.asarray([1.0, -2.0, 0.5, 3.0], dtype=np.float64)
    result = plan.evaluate_monopole(transition.accepted, strengths)
    expected = _direct(np.asarray(moved), strengths, np.ones(4, dtype=np.bool_))
    np.testing.assert_allclose(result.potential, expected[0], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result.gradient, expected[1], rtol=1e-12, atol=1e-12)


def test_cutoff_motion_and_activity_changes_rebuild_structure() -> None:
    positions, _ = _points(12, 41)
    plan = UniformFMMPlan(
        1.0,
        CartesianExpansionSpace(1),
        softening=_SOFTENING,
        short_range_scale=0.1,
        short_range_cutoff=0.4,
        maximum_leaf_occupancy=3,
        coarsening_factor=2,
        target_top_nodes=1,
    )
    previous = plan.prepare_structure(positions, box_size=_BOX, depth=_DEPTH)
    unchanged = plan.refresh_structure(previous, positions, box_size=_BOX, depth=_DEPTH)
    assert bool(unchanged.successful)
    assert bool(unchanged.reused_topology)
    moved = plan.refresh_structure(
        previous, positions + 1e-6, box_size=_BOX, depth=_DEPTH
    )
    assert bool(moved.successful)
    assert bool(moved.rebuilt)
    assert not bool(moved.reused_topology)
    active = np.ones(positions.shape[0], dtype=np.bool_)
    active[0] = False
    masked = plan.refresh_structure(
        previous, positions, box_size=_BOX, depth=_DEPTH, active_mask=active
    )
    assert bool(masked.successful)
    assert bool(masked.rebuilt)
    np.testing.assert_array_equal(masked.accepted.active_mask, active)


def test_nonfinite_active_strength_refuses_even_without_pair_events() -> None:
    plan = _plan(1)
    structure = plan.prepare_structure(
        np.asarray([[0.2, 0.3, 0.4]], dtype=np.float64),
        box_size=_BOX,
        depth=_DEPTH,
    )
    result = plan.evaluate_monopole(structure, jnp.asarray([jnp.nan]))
    assert not bool(result.successful)
    assert np.isnan(np.asarray(result.potential)).all()
    assert np.isnan(np.asarray(result.gradient)).all()


def _strength_derivatives(
    plan: UniformFMMPlan,
    structure: PreparedUniformFMMStructure,
    weights: jax.Array,
    direction: jax.Array,
) -> tuple[Callable[[jax.Array], Any], Callable[[jax.Array], jax.Array]]:
    def values(strength: jax.Array) -> tuple[jax.Array, jax.Array]:
        result = plan.evaluate_monopole(structure, strength)
        return result.potential, result.gradient

    def objective(strength: jax.Array) -> jax.Array:
        potential, gradient = values(strength)
        return jnp.sum(potential) + jnp.sum(gradient * weights)

    return (
        jax.jit(lambda strength: jax.jvp(values, (strength,), (direction,))[1]),
        jax.jit(jax.grad(objective)),
    )


@pytest.mark.parametrize("accumulation", ("deterministic", "compensated"))
def test_ordered_field_and_strength_derivatives_match_fast_route(
    accumulation: RelationAccumulation,
) -> None:
    positions, strengths = _points(128, 43)
    # Inactive points leave low packed slots to coarser nodes after the leaves.
    active = np.ones((positions.shape[0],), dtype=np.bool_)
    active[::7] = False
    generator = np.random.default_rng(44)
    weights = jnp.asarray(generator.normal(size=positions.shape), dtype=jnp.float64)
    direction = jnp.asarray(generator.normal(size=strengths.shape), dtype=jnp.float64)
    outputs: dict[RelationAccumulation, tuple[Any, ...]] = {}
    policies: tuple[RelationAccumulation, RelationAccumulation] = ("fast", accumulation)
    for policy in policies:
        plan = _plan(3, accumulation=policy)
        structure = plan.prepare_structure(
            positions, box_size=_BOX, depth=_DEPTH, active_mask=active
        )
        result = plan.evaluate_monopole(structure, strengths)
        assert bool(result.successful)
        assert int(result.evidence.required_far) > 0
        assert int(result.evidence.l2l_count) > 0
        assert result.evidence.accumulation == policy
        jvp, vjp = _strength_derivatives(plan, structure, weights, direction)
        outputs[policy] = (
            result.potential,
            result.gradient,
            *jvp(jnp.asarray(strengths, dtype=jnp.float64)),
            vjp(jnp.asarray(strengths, dtype=jnp.float64)),
        )
    for fast, ordered in zip(outputs["fast"], outputs[accumulation], strict=True):
        np.testing.assert_allclose(ordered, fast, rtol=1e-11, atol=1e-11)


def test_ordered_compiled_strength_derivatives_replay_bitwise() -> None:
    # On GPU the unordered duplicate-index transposes of the L2L parent and
    # L2P leaf gathers made this strength VJP differ between replays.
    positions, strengths = _points(64, 47)
    generator = np.random.default_rng(48)
    weights = jnp.asarray(generator.normal(size=positions.shape), dtype=jnp.float64)
    direction = jnp.asarray(generator.normal(size=strengths.shape), dtype=jnp.float64)
    plan = _plan(3)
    structure = plan.prepare_structure(positions, box_size=_BOX, depth=_DEPTH)
    jvp, vjp = _strength_derivatives(plan, structure, weights, direction)
    values = jnp.asarray(strengths, dtype=jnp.float64)
    expected = jax.block_until_ready((jvp(values), vjp(values)))
    for _ in range(8):
        actual = jax.block_until_ready((jvp(values), vjp(values)))
        for left, right in zip(
            jax.tree.leaves(expected), jax.tree.leaves(actual), strict=True
        ):
            assert np.asarray(left).tobytes() == np.asarray(right).tobytes()


def test_owner_row_broadcast_masks_unowned_values_and_cotangents() -> None:
    rows = jnp.asarray(np.random.default_rng(49).normal(size=(3, 2)), dtype=jnp.float64)
    members = jnp.asarray([[0, 1, 4], [2, 3, 4], [4, 4, 4]], dtype=jnp.int32)
    owned = jnp.asarray(
        [[True, True, False], [True, True, True], [False, False, False]],
        dtype=jnp.bool_,
    )
    rows = rows.at[2].set(jnp.nan)
    broadcast = _broadcast_owner_rows(rows, members, owned, 5)
    expected = np.zeros((5, 2), dtype=np.float64)
    expected[[0, 1]] = np.asarray(rows[0])
    expected[[2, 3, 4]] = np.asarray(rows[1])
    np.testing.assert_array_equal(broadcast, expected)

    cotangent = jnp.arange(10, dtype=jnp.float64).reshape(5, 2)
    _, pullback = jax.vjp(lambda row: _broadcast_owner_rows(row, members, owned, 5), rows)
    np.testing.assert_array_equal(
        pullback(cotangent)[0],
        [
            np.sum(np.asarray(cotangent)[[0, 1]], axis=0),
            np.sum(np.asarray(cotangent)[[2, 3, 4]], axis=0),
            [0.0, 0.0],
        ],
    )
