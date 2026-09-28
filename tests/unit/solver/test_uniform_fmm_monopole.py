from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.solver import (
    CartesianExpansionSpace,
    PreparedUniformFMMStructure,
    UniformFMMPlan,
)


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

    node_limited = _plan(1, maximum_nodes=2)
    nodes = node_limited.evaluate_monopole(
        node_limited.prepare_structure(positions, box_size=_BOX, depth=_DEPTH),
        strengths,
    )
    assert not bool(nodes.successful)
    assert int(nodes.evidence.required_nodes) > int(nodes.evidence.node_capacity)


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
