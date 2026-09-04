#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from phydrax.applications.skeletal_muscle.musculotendon import (
    de_groote_fregly_2016_active_force_length,
    de_groote_fregly_2016_force_velocity,
    de_groote_fregly_2016_inverse_force_velocity,
    de_groote_fregly_2016_inverse_tendon_force_length,
    de_groote_fregly_2016_passive_force_length,
    de_groote_fregly_2016_tendon_force_length,
    DeGrooteFregly2016ImplicitTendonForcePlan,
    DeGrooteFregly2016Parameters,
    DeGrooteFregly2016Plan,
    DeGrooteFregly2016State,
)


def _parameters(count=2):
    return DeGrooteFregly2016Parameters(
        jnp.linspace(1200.0, 1800.0, count),
        jnp.linspace(0.09, 0.11, count),
        jnp.linspace(0.20, 0.24, count),
        jnp.linspace(0.10, 0.16, count),
        jnp.linspace(0.9, 1.1, count),
    )


def _equilibrium(parameters, activation=None):
    activation = (
        jnp.linspace(0.35, 0.55, parameters.muscle_capacity)
        if activation is None
        else jnp.asarray(activation)
    )
    normalized_fiber_length = jnp.ones_like(activation)
    active_length = de_groote_fregly_2016_active_force_length(
        parameters, normalized_fiber_length
    )
    force_velocity = de_groote_fregly_2016_force_velocity(
        parameters, jnp.zeros_like(activation)
    )
    passive = de_groote_fregly_2016_passive_force_length(
        parameters, normalized_fiber_length
    )
    cosine = jnp.cos(parameters.pennation_angle_at_optimum_rad)
    normalized_tendon_force = (
        activation * active_length * force_velocity + passive
    ) * cosine
    tendon_length = (
        de_groote_fregly_2016_inverse_tendon_force_length(
            parameters, normalized_tendon_force
        )
        * parameters.tendon_slack_length_m
    )
    musculotendon_length = (
        tendon_length + parameters.optimal_fiber_length_m * cosine
    )
    state = DeGrooteFregly2016State(activation, normalized_tendon_force)
    return state, musculotendon_length, jnp.zeros_like(activation)


def test_supplement_table_1_curves_match_independent_equations_and_inverses():
    parameters = _parameters(2)
    tendon_length = jnp.asarray([1.0, 1.04])
    fiber_length = jnp.asarray([0.8, 1.2])
    velocity = jnp.asarray([-0.5, 0.75])

    expected_tendon = 0.2 * np.exp(35.0 * (np.asarray(tendon_length) - 0.995)) - 0.25
    expected_passive = (
        np.exp(4.0 * (np.asarray(fiber_length) - 1.0) / 0.6) - 1.0
    ) / np.expm1(4.0)
    b1 = np.asarray([0.815, 0.433, 0.100])
    b2 = np.asarray([1.055, 0.717, 1.000])
    b3 = np.asarray([0.162, -0.030, 0.354])
    b4 = np.asarray([0.063, 0.200, 0.000])
    width = b3 + np.asarray(fiber_length)[:, None] * b4
    expected_active = np.sum(
        b1 * np.exp(-0.5 * ((np.asarray(fiber_length)[:, None] - b2) / width) ** 2),
        axis=-1,
    )
    expected_velocity = -0.318 * np.arcsinh(-8.149 * np.asarray(velocity) - 0.374) + 0.886

    np.testing.assert_allclose(
        de_groote_fregly_2016_tendon_force_length(parameters, tendon_length),
        expected_tendon,
        rtol=1.0e-6,
    )
    np.testing.assert_allclose(
        de_groote_fregly_2016_active_force_length(parameters, fiber_length),
        expected_active,
        rtol=1.0e-6,
    )
    np.testing.assert_allclose(
        de_groote_fregly_2016_passive_force_length(parameters, fiber_length),
        expected_passive,
        rtol=1.0e-6,
    )
    observed_velocity = de_groote_fregly_2016_force_velocity(parameters, velocity)
    np.testing.assert_allclose(observed_velocity, expected_velocity, rtol=1.0e-6)
    np.testing.assert_allclose(
        de_groote_fregly_2016_inverse_force_velocity(parameters, observed_velocity),
        velocity,
        rtol=2.0e-6,
    )
    np.testing.assert_allclose(
        de_groote_fregly_2016_inverse_tendon_force_length(
            parameters,
            de_groote_fregly_2016_tendon_force_length(parameters, tendon_length),
        ),
        tendon_length,
        rtol=1.0e-6,
    )


def test_explicit_formulation_equilibrium_force_energy_and_power_evidence():
    parameters = _parameters(2)
    prepared = DeGrooteFregly2016Plan(parameters, ("soleus", "gastrocnemius")).prepare()
    state, length, velocity = _equilibrium(parameters)
    evaluation = prepared.evaluate(state, state.activation, length, velocity)

    assert jnp.all(evaluation.successful)
    np.testing.assert_allclose(
        evaluation.tendon_force_N,
        parameters.maximum_isometric_force_N * state.normalized_tendon_force,
    )
    np.testing.assert_allclose(
        evaluation.evidence.force_equilibrium_residual_normalized,
        0.0,
        atol=2.0e-6,
    )
    np.testing.assert_allclose(
        evaluation.evidence.tendon_constitutive_residual_normalized,
        0.0,
        atol=2.0e-6,
    )
    np.testing.assert_allclose(
        evaluation.evidence.force_velocity_inverse_residual_normalized,
        0.0,
        atol=2.0e-6,
    )
    np.testing.assert_allclose(
        evaluation.evidence.tendon_rate_residual_per_s,
        0.0,
        atol=2.0e-6,
    )
    np.testing.assert_allclose(
        evaluation.evidence.length_closure_residual_m, 0.0, atol=1.0e-10
    )
    np.testing.assert_allclose(
        evaluation.evidence.power_balance_residual_W, 0.0, atol=2.0e-5
    )
    assert evaluation.force_owner == "de-groote-fregly-2016"
    assert evaluation.force_sign == "positive-is-tensile"

    force_rate = evaluation.rates.normalized_tendon_force_per_s
    tendon_energy_rate = jax.jvp(
        lambda force: prepared.evaluate(
            DeGrooteFregly2016State(state.activation, force),
            state.activation,
            length,
            velocity,
        ).evidence.tendon_energy_J,
        (state.normalized_tendon_force,),
        (force_rate,),
    )[1]
    np.testing.assert_allclose(
        tendon_energy_rate,
        evaluation.evidence.tendon_energy_rate_W,
        rtol=2.0e-5,
        atol=2.0e-5,
    )


def test_explicit_model_jit_vmap_jvp_trainable_leaves_and_atomic_rollback():
    parameters = _parameters(2)
    prepared = DeGrooteFregly2016Plan(parameters, ("a", "b")).prepare()
    state, length, velocity = _equilibrium(parameters)
    compiled = eqx.filter_jit(prepared.evaluate)(
        state, state.activation, length, velocity
    )
    assert jnp.all(compiled.successful)

    excitations = jnp.stack((state.activation, 0.9 * state.activation))
    batched_force = jax.vmap(
        lambda excitation: prepared.evaluate(
            state, excitation, length, velocity
        ).tendon_force_N
    )(excitations)
    assert batched_force.shape == (2, 2)

    def force_for_strength(strength):
        changed_parameters = eqx.tree_at(
            lambda value: value.maximum_isometric_force_N,
            parameters,
            strength,
        )
        changed = eqx.tree_at(
            lambda value: value.plan.parameters,
            prepared,
            changed_parameters,
        )
        return jnp.sum(
            changed.evaluate(state, state.activation, length, velocity).tendon_force_N
        )

    tangent = jax.jvp(
        force_for_strength,
        (parameters.maximum_isometric_force_N,),
        (jnp.ones_like(parameters.maximum_isometric_force_N),),
    )[1]
    assert jnp.isfinite(tangent)
    assert tangent > 0.0

    failed = prepared.candidate(
        state,
        jnp.asarray([1.2, 0.5]),
        length,
        velocity,
        1.0e-4,
    )
    assert not bool(failed.successful)
    rolled_back = prepared.commit(failed)
    assert jnp.array_equal(rolled_back.activation, state.activation)
    assert jnp.array_equal(
        rolled_back.normalized_tendon_force, state.normalized_tendon_force
    )


def test_implicit_formulation_uses_root_owned_sensitivity_and_rolls_back_failures():
    parameters = _parameters(1)
    state, length, velocity = _equilibrium(parameters, jnp.asarray([0.45]))
    prepared = DeGrooteFregly2016ImplicitTendonForcePlan(
        parameters, ("soleus",)
    ).prepare(state)
    candidate = prepared.candidate(
        state, state.activation, length, velocity, jnp.asarray(1.0e-5)
    )

    assert bool(candidate.successful)
    assert candidate.evidence.sensitivity_owner == "phydrax.nonlinear.implicit_root_result"
    np.testing.assert_allclose(
        candidate.evidence.algebraic_residual, 0.0, atol=2.0e-6
    )
    derivative = jax.jvp(
        lambda muscle_length: prepared.candidate(
            state,
            state.activation,
            muscle_length,
            velocity,
            jnp.asarray(1.0e-5),
        ).evidence.scaled_force_rate_control,
        (length,),
        (jnp.ones_like(length),),
    )[1]
    assert jnp.all(jnp.isfinite(derivative))

    failed = prepared.candidate(
        state, jnp.asarray([-0.1]), length, velocity, jnp.asarray(1.0e-5)
    )
    assert not bool(failed.successful)
    rolled_back = prepared.commit(failed)
    assert jnp.array_equal(rolled_back.activation, state.activation)
    assert jnp.array_equal(
        rolled_back.normalized_tendon_force, state.normalized_tendon_force
    )
