# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from examples.meshfree_moving_surface_reaction_diffusion import (
    perform_epoch_transition,
    prepare_workflow,
    reprepare_workflow,
)
from phydrax.discretization.meshfree._capacity import MeshfreeCapacityMap
from phydrax.discretization.meshfree._moving import (
    MovingGeometryRefresh,
    MovingSurfacePlan,
    MovingSurfaceState,
    MovingSurfaceStatus,
)


def test_growing_sphere_dilution_and_nonuniform_native_diffusion() -> None:
    plan, initial, _pairs = prepare_workflow(size=16)
    result = eqx.filter_jit(plan.step)(initial, 0.01)
    assert bool(result.evidence.successful)
    np.testing.assert_allclose(
        result.state.concentration, (1 - 0.1 * 0.01) / (1 + 0.2 * 0.01) ** 2, atol=1e-10
    )
    np.testing.assert_allclose(result.evidence.conservation_residual, 0.0, atol=1e-10)
    perturbation = initial.measures * (1 + 0.2 * initial.points[:, 2])
    perturbed = eqx.tree_at(
        lambda s: (s.content, s.history_content),
        initial,
        (perturbation, initial.history_content.at[0].set(perturbation)),
    )
    diffused = plan.step(perturbed, 0.1)
    assert bool(diffused.evidence.successful)
    scaled = diffused.state.concentration * (1 + 0.2 * 0.1) ** 2 / (1 - 0.1 * 0.1)
    assert float(jnp.var(scaled)) < float(jnp.var(perturbed.concentration))
    np.testing.assert_allclose(diffused.evidence.conservation_residual, 0.0, atol=1e-10)


def test_refused_geometry_and_history_capacity_roll_back_all_state() -> None:
    plan, state, _pairs = prepare_workflow(size=16)

    def refresh(
        current: MovingSurfaceState, time: Array, dt: Array, args: Any, /
    ) -> MovingGeometryRefresh:
        geometry = plan.geometry_refresh(current, time, dt, args)
        return eqx.tree_at(lambda g: g.trust_valid, geometry, jnp.asarray(False))

    refusing = MovingSurfacePlan(
        refresh, plan.reaction, epoch=state.epoch, history_capacity=12, plan_id="refusing"
    )
    failed = refusing.step(state, 0.01)
    assert int(failed.evidence.status) == int(MovingSurfaceStatus.TRUST_REFUSED)
    np.testing.assert_array_equal(failed.state.content, state.content)
    np.testing.assert_array_equal(failed.state.history_content, state.history_content)
    np.testing.assert_array_equal(failed.state.points, state.points)
    short = MovingSurfacePlan(
        plan.geometry_refresh,
        plan.reaction,
        epoch=state.epoch,
        history_capacity=2,
        plan_id="short",
    )
    first = short.initialize(
        state.points,
        state.measures,
        state.normals,
        state.concentration,
        state.capacity,
        state.epoch,
    )
    accepted = short.step(first, 0.01).state
    checkpoint = short.checkpoint(accepted)
    exhausted = short.step(accepted, 0.01)
    assert int(exhausted.evidence.status) == int(MovingSurfaceStatus.HISTORY_EXHAUSTED)
    np.testing.assert_array_equal(
        exhausted.state.history_content, accepted.history_content
    )
    np.testing.assert_array_equal(short.rollback(checkpoint).content, accepted.content)


def test_epoch_remaps_all_extensive_histories_and_preserves_positive_active_spaces() -> (
    None
):
    plan, state, _pairs = prepare_workflow(size=16)
    state = plan.step(state, 0.01).state
    before = np.sum(np.asarray(state.history_content[:2]), axis=1)
    transitioned = perform_epoch_transition(plan, state)
    assert transitioned.successful and not transitioned.differentiation_available
    assert transitioned.state.epoch.index == state.epoch.index + 1
    assert int(transitioned.state.history_count) == 2
    np.testing.assert_allclose(
        np.sum(np.asarray(transitioned.state.history_content[:2]), axis=1),
        before,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        np.sum(np.asarray(transitioned.state.content)),
        np.sum(np.asarray(state.content)),
        atol=1e-10,
    )
    with pytest.raises(ValueError, match="changed topology epoch"):
        plan.step(transitioned.state, 0.01)
    next_plan = reprepare_workflow(transitioned.state)
    continued = next_plan.step(transitioned.state, 0.01)
    assert bool(continued.evidence.successful)
    assert int(continued.state.history_count) == 3
    np.testing.assert_allclose(continued.evidence.conservation_residual, 0.0, atol=1e-10)
    mapping = MeshfreeCapacityMap(8, np.asarray([1, 4, 6], dtype=np.int32))
    positive = mapping.positive_space(np.asarray([1.0, 2.0, 3.0], dtype=np.float64))
    assert positive.size == 3
    np.testing.assert_allclose(positive.riesz(jnp.ones(3)), [1.0, 2.0, 3.0], atol=1e-12)
    np.testing.assert_allclose(
        mapping.expand(np.asarray([1.0, 2.0, 3.0], dtype=np.float64)),
        [0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 3.0, 0.0],
        atol=0,
    )
    np.testing.assert_allclose(mapping.compact(jnp.arange(8.0)), [1.0, 4.0, 6.0], atol=0)
    with pytest.raises(ValueError, match="strictly positive"):
        mapping.positive_space(np.asarray([1.0, 0.0, 3.0], dtype=np.float64))


def test_failed_historical_remap_does_not_partially_change_epoch_or_current_field() -> (
    None
):
    plan, state, _pairs = prepare_workflow(size=16)
    state = plan.step(state, 0.01).state
    damaged = eqx.tree_at(
        lambda s: s.history_content, state, state.history_content.at[0, 0].set(jnp.nan)
    )
    result = perform_epoch_transition(plan, damaged)
    assert not result.successful
    assert result.state.epoch.epoch_id == damaged.epoch.epoch_id
    np.testing.assert_array_equal(result.state.points, damaged.points)
    np.testing.assert_array_equal(result.state.content, damaged.content)
    np.testing.assert_array_equal(result.state.history_content, damaged.history_content)
