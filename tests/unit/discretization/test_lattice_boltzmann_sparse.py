#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp

import phydrax as phx


def _index_space():
    plan = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(4, periodic=True),
            phx.discretization.UniformCellAxisSpec(4, periodic=True),
        ),
        axis_names=("x", "y"),
    )
    return plan.prepare_index_space(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))


def test_periodic_sparse_lbm_conserves_mass_and_is_jit_safe():
    prepared = phx.discretization.SparseLatticeBoltzmannPlan(
        _index_space(),
        phx.discretization.D2Q9(),
        (2, 2),
        4,
    ).prepare(jnp.arange(16))
    state = prepared.initialize_state(1.0, jnp.asarray((0.03, 0.0)))
    result = jax.jit(lambda value: prepared.step(value, 1.0))(state)

    assert bool(result.successful)
    assert result.accepted_state.populations.shape == (16, 9)
    assert jnp.isclose(result.diagnostics.total_mass, 16.0)
    assert jnp.abs(result.diagnostics.mass_defect) < 1.0e-12
    assert jnp.max(jnp.abs(result.velocity - jnp.asarray((0.03, 0.0)))) < 1.0e-12


def test_sparse_lbm_wall_streaming_and_geometry_transition_are_conservative():
    plan = phx.discretization.SparseLatticeBoltzmannPlan(
        _index_space(),
        phx.discretization.D2Q9(),
        (2, 2),
        3,
        collision=phx.discretization.TRTCollisionPlan(),
    )
    prepared = plan.prepare(jnp.asarray([0, 1, 4, 5]))
    state = prepared.initialize_state(1.0, jnp.asarray((0.02, 0.01)))
    step = prepared.step(state, 1.0)
    transition = prepared.refresh_geometry(
        step.accepted_state,
        jnp.asarray([1, 2, 5, 6]),
        activated_density=2.0,
    )

    assert bool(step.successful)
    assert jnp.abs(step.diagnostics.mass_defect) < 1.0e-12
    assert int(transition.evidence.retained_cells) == 2
    assert int(transition.evidence.activated_cells) == 2
    assert int(transition.evidence.retired_cells) == 2
    assert jnp.isclose(transition.evidence.activated_mass, 4.0)
    assert int(transition.state.topology.generation) == 1
