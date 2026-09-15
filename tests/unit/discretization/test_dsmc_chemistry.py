#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _reactive_species():
    return phx.discretization.dsmc.DSMCSpeciesPlan(
        ("A", "B", "C", "D"),
        jnp.asarray((1.0, 1.0, 1.5, 0.5)),
        jnp.ones((4,)),
        jnp.full((4,), 0.5),
        jnp.full((4,), 2.0),
        jnp.zeros((4,)),
        jnp.zeros((4,)),
        element_names=("a", "b"),
        elemental_composition=jnp.asarray(
            ((1, 0), (0, 1), (1, 1), (0, 0)), dtype=jnp.int32
        ),
    )


def _state():
    return phx.discretization.dsmc.DSMCParticleState(
        jnp.asarray(((0.25,), (0.75,))),
        jnp.asarray(((1.0,), (-1.0,))),
        jnp.asarray((0, 1), dtype=jnp.int32),
        jnp.zeros((2,)),
        jnp.zeros((2,)),
        jnp.ones((2,)),
        jnp.zeros((2,), dtype=jnp.int32),
        jnp.ones((2,), dtype=bool),
        jnp.zeros((2,), dtype=jnp.int32),
    )


def test_internal_physics_is_identity_for_rejected_collision_and_conservative_when_accepted():
    species = _reactive_species()
    channel = phx.discretization.dsmc.DSMCReactionChannelPlan(
        (0, 1), (2, 3), threshold_energy=0.0, probability=1.0
    )
    plan = phx.discretization.dsmc.DSMCInternalReactionPlan(
        species,
        (channel,),
        rotational_relaxation_probability=jnp.zeros((4,)),
    )
    indices = jnp.asarray((0,), dtype=jnp.int32)
    keys = jax.random.split(jax.random.key(11), 1)

    rejected = plan.apply(_state(), indices, 1 - indices, jnp.asarray((False,)), keys)
    np.testing.assert_array_equal(rejected.state.species_index, _state().species_index)
    np.testing.assert_array_equal(rejected.state.velocity, _state().velocity)
    assert not bool(rejected.reacted[0])

    accepted = plan.apply(_state(), indices, 1 - indices, jnp.asarray((True,)), keys)
    assert bool(accepted.successful)
    assert bool(accepted.reacted[0])
    np.testing.assert_array_equal(accepted.state.species_index, (2, 3))
    np.testing.assert_allclose(accepted.energy_defect, 0.0, atol=1.0e-12)
    before_momentum = jnp.sum(
        species.molecular_masses[_state().species_index, None] * _state().velocity,
        axis=0,
    )
    after_momentum = jnp.sum(
        species.molecular_masses[accepted.state.species_index, None]
        * accepted.state.velocity,
        axis=0,
    )
    np.testing.assert_allclose(after_momentum, before_momentum, atol=1.0e-12)


def test_reaction_can_draw_threshold_energy_from_rotational_reservoir():
    species = _reactive_species()
    channel = phx.discretization.dsmc.DSMCReactionChannelPlan(
        (0, 1), (2, 3), threshold_energy=1.5, probability=1.0
    )
    plan = phx.discretization.dsmc.DSMCInternalReactionPlan(
        species,
        (channel,),
        rotational_relaxation_probability=jnp.zeros((4,)),
    )
    incoming = _state()
    incoming = phx.discretization.dsmc.DSMCParticleState(
        incoming.position,
        incoming.velocity,
        incoming.species_index,
        jnp.asarray((0.5, 0.5)),
        incoming.vibrational_energy,
        incoming.statistical_weight,
        incoming.cell_id,
        incoming.active,
        incoming.incarnation,
    )
    result = plan.apply(
        incoming,
        jnp.asarray((0,), dtype=jnp.int32),
        jnp.asarray((1,), dtype=jnp.int32),
        jnp.asarray((True,)),
        jax.random.split(jax.random.key(12), 1),
    )

    assert bool(result.successful)
    assert bool(result.reacted[0])
    np.testing.assert_allclose(result.energy_defect, 0.0, atol=1.0e-12)


def test_reactive_plan_requires_element_and_mass_conservation():
    species = _reactive_species()
    invalid = phx.discretization.dsmc.DSMCReactionChannelPlan(
        (0, 1), (0, 0), threshold_energy=0.0, probability=1.0
    )
    with pytest.raises(ValueError, match="elements"):
        phx.discretization.dsmc.DSMCInternalReactionPlan(
            species,
            (invalid,),
            rotational_relaxation_probability=jnp.zeros((4,)),
        )
