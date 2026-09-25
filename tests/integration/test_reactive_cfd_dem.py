#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp

import phydrax as phx
from tests._reactive_systems import reactive_fluid_sample, reactive_problem


def test_reactive_macro_window_commits_heat_species_and_mechanics_atomically():
    plan, state, boundary, schedule = reactive_problem()

    def update(fluid, momentum, energy, species, step_size):
        del momentum, step_size
        return fluid[0] + energy, fluid[1] + species

    result = phx.solver.advance_reactive_cfd_dem_window(
        plan,
        schedule,
        state,
        reactive_fluid_sample,
        update,
        (boundary,),
        jnp.zeros((0,)),
        jnp.asarray([0.001]),
        jnp.asarray(0.0),
        jnp.asarray(1.0e-5),
    )
    assert result.successful
    assert result.accepted_state.accepted_windows == 1
    assert jnp.linalg.norm(result.evaluation.momentum_residual) < 1.0e-12
    assert jnp.abs(result.evaluation.energy_residual) < 1.0e-12
    assert jnp.max(jnp.abs(result.evaluation.species_residual)) < 1.0e-12
    assert result.evaluation.continuum_successful
    assert result.evaluation.conversion_successful
    assert result.evaluation.dem_successful

    def invalid_update(fluid, momentum, energy, species, step_size):
        del momentum, energy, species, step_size
        return jnp.full_like(fluid[0], jnp.nan), fluid[1]

    rejected = phx.solver.advance_reactive_cfd_dem_window(
        plan,
        schedule,
        state,
        reactive_fluid_sample,
        invalid_update,
        (boundary,),
        jnp.zeros((0,)),
        jnp.asarray([0.001]),
        jnp.asarray(0.0),
        jnp.asarray(1.0e-5),
    )
    assert not rejected.successful
    assert rejected.accepted_state.accepted_windows == 0
    assert jnp.allclose(
        rejected.accepted_state.conversion_state.batches[0].internal_energy,
        state.conversion_state.batches[0].internal_energy,
    )


def test_checkpointed_reactive_vjp_guards_cotangent_on_replay_mismatch(monkeypatch):
    from phydrax.solver import _reactive_replay

    plan, state, boundary, schedule = reactive_problem()

    def update(fluid, momentum, energy, species, step_size):
        del momentum, step_size
        return fluid[0] + energy, fluid[1] + species

    def step(coupling_state, index):
        return phx.solver.advance_reactive_cfd_dem_window(
            plan,
            schedule,
            coupling_state,
            reactive_fluid_sample,
            update,
            (boundary,),
            jnp.zeros((0,)),
            jnp.asarray([0.001]),
            index * jnp.asarray(1.0e-5),
            jnp.asarray(1.0e-5),
        )

    def vjp():
        return phx.solver.checkpointed_reactive_vjp(
            lambda final_state: jnp.sum(final_state.fluid_state[0]),
            step,
            state,
            jnp.asarray(1.0),
            step_count=1,
            checkpoint=phx.solver.ReactiveCheckpointPolicy(1),
        )

    matched = vjp()
    assert matched.replay_matched
    fluid_cotangent = matched.initial_state_cotangent.fluid_state[0]
    assert jnp.all(jnp.isfinite(fluid_cotangent))
    assert jnp.all(fluid_cotangent != 0.0)

    monkeypatch.setattr(
        _reactive_replay,
        "reactive_replay_matches",
        lambda left, right: jnp.asarray(False),
    )
    mismatched = vjp()
    assert not mismatched.replay_matched
    assert mismatched.primal == matched.primal
    assert jax.tree.all(jax.tree.map(jnp.array_equal, mismatched.replay, matched.replay))
    assert all(
        jnp.all(jnp.isnan(leaf))
        for leaf in jax.tree.leaves(mismatched.initial_state_cotangent)
        if eqx.is_inexact_array(leaf)
    )
