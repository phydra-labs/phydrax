import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _species():
    return phx.discretization.dsmc.DSMCSpeciesPlan(
        ("A",),
        jnp.asarray((4.65e-26,)),
        jnp.asarray((3.7e-10,)),
        jnp.asarray((0.75,)),
        jnp.asarray((2.0,)),
        jnp.asarray((3390.0,)),
        jnp.asarray((0.0,)),
    )


def _particles():
    return phx.discretization.dsmc.DSMCParticleState(
        jnp.asarray(((0.25,), (0.75,))),
        jnp.asarray(((100.0,), (-100.0,))),
        jnp.asarray((0, 0), dtype=jnp.int32),
        jnp.zeros((2,)),
        jnp.zeros((2,)),
        jnp.ones((2,)),
        jnp.asarray((0, 0), dtype=jnp.int32),
        jnp.asarray((True, True)),
        jnp.asarray((0, 0), dtype=jnp.int32),
    )


def test_dsmc_streaming_collision_and_internal_energy_are_conservative():
    species = _species()
    cells = phx.discretization.dsmc.DSMCStructuredCellPlan(
        jnp.asarray((0.0,)), jnp.asarray((1.0,)), (1,)
    )
    streaming = phx.discretization.dsmc.DSMCStreamingPlan(
        cells, (("periodic", "periodic"),)
    )
    streamed = streaming.advance(_particles(), 0.001)
    assert bool(streamed.successful)
    assert jnp.all(streamed.state.active)

    collision_plan = phx.discretization.dsmc.VSSVHSCollisionPlan(
        species, jnp.asarray(((1.0,),))
    )
    first = jnp.asarray((0,), dtype=jnp.int32)
    second = jnp.asarray((1,), dtype=jnp.int32)
    relative_speed = jnp.asarray((200.0,))
    majorant = (
        collision_plan.collision_cross_section(
            jnp.asarray((0,)), jnp.asarray((0,)), relative_speed
        )
        * relative_speed
    )
    collision = collision_plan.collide(
        streamed.state,
        first,
        second,
        jnp.asarray(((0.0, 0.2, 0.3),)),
        1.01 * majorant,
    )
    assert bool(collision.successful)
    assert bool(collision.accepted[0])
    np.testing.assert_allclose(collision.energy_defect, 0.0, atol=1.0e-30)
    np.testing.assert_allclose(collision.momentum_defect, 0.0, atol=1.0e-30)

    internal_plan = phx.discretization.dsmc.DSMCInternalReactionPlan(
        species,
        rotational_relaxation_probability=jnp.asarray((0.0,)),
        vibrational_relaxation_probability=jnp.asarray((0.0,)),
    )
    internal = internal_plan.apply(
        collision.state,
        first,
        second,
        jnp.asarray(((0.9, 0.9, 0.9, 0.5),)),
    )
    assert bool(internal.successful)
    np.testing.assert_allclose(internal.energy_defect, 0.0, atol=1.0e-30)


def test_dsmc_surface_and_hybrid_exchange_are_action_reaction_pairs():
    interaction = phx.discretization.dsmc.DSMCSurfaceInteractionPlan(
        _species(), "specular", wall_temperature=300.0
    )
    result = interaction.interact(
        _particles(),
        jnp.asarray((0,), dtype=jnp.int32),
        jnp.asarray(((1.0,),)),
        jnp.asarray(((0.9, 0.2, 0.3, 0.5),)),
    )
    assert bool(result.successful)
    np.testing.assert_allclose(result.state.velocity[0, 0], -100.0)
    assert result.total_force_impulse[0] > 0.0

    interface = phx.solver.FixedContinuumDSMCInterfacePlan(3)
    exchange = interface.exchange(
        jnp.asarray((1.0, 2.0, 3.0)),
        jnp.asarray((1.2, 1.8, 3.1)),
        jnp.asarray((0.1, 0.1, 0.1)),
        0.01,
        2.0,
    )
    assert bool(exchange.successful)
    np.testing.assert_allclose(exchange.conservation_defect, 0.0, atol=0.0)


def test_dynamic_hybrid_ownership_is_hysteretic_and_capacity_bounded():
    plan = phx.solver.DynamicHybridOwnershipPlan(
        enter_threshold=0.05,
        leave_threshold=0.02,
        minimum_dwell_steps=2,
        buffer_layers=1,
    )
    state = plan.initialize(jnp.asarray((False, False, False)))
    adjacency = jnp.asarray(
        ((False, True, False), (True, False, True), (False, True, False))
    )
    result = plan.update(
        state,
        jnp.asarray((0.0, 0.1, 0.0)),
        adjacency,
        4,
        20,
    )
    assert bool(result.successful)
    assert result.required_particles == 12
    assert result.accepted.kinetic_mask.tolist() == [True, True, True]

    rejected = plan.update(
        state,
        jnp.asarray((0.0, 0.1, 0.0)),
        adjacency,
        4,
        4,
    )
    assert not bool(rejected.successful)
    assert rejected.accepted.kinetic_mask.tolist() == [False, False, False]
