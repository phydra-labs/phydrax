import jax
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

    pair_parameters = phx.discretization.dsmc.DSMCPairCollisionParameters(
        jnp.asarray(((3.7e-10,),)),
        jnp.asarray(((0.75,),)),
    )
    collision_plan = phx.discretization.dsmc.DSMCVHSCollisionPlan(
        species, pair_parameters
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
        jnp.asarray((True,)),
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
    )
    internal = internal_plan.apply(
        collision.state,
        first,
        second,
        collision.accepted,
        jax.random.split(jax.random.key(7), 1),
    )
    assert bool(internal.successful)
    np.testing.assert_allclose(internal.energy_defect, 0.0, atol=1.0e-30)


def test_dsmc_surface_and_hybrid_exchange_are_action_reaction_pairs():
    interaction = phx.discretization.dsmc.DSMCSurfaceInteractionPlan(
        _species(), "specular", jnp.asarray((0.0,)), wall_temperature=300.0
    )
    result = interaction.interact(
        _particles(),
        jnp.asarray((0,), dtype=jnp.int32),
        jnp.asarray(((1.0,),)),
        jax.random.split(jax.random.key(9), 1),
    )
    assert bool(result.successful)
    np.testing.assert_allclose(result.state.velocity[0, 0], -100.0)
    assert result.total_force_impulse[0] > 0.0

    schema = phx.solver.ContinuumDSMCConservedSchema(
        ("mass", "momentum-0", "total-energy"),
        ("kg", "kg-m-s", "J"),
        frame="laboratory",
        energy_reference="absolute",
    )
    interface = phx.solver.ContinuumDSMCInterfacePlan(schema, jnp.asarray((2.0,)))
    exchange = interface.exchange(
        jnp.asarray(((1.2, 1.8, 3.1),)),
        jnp.asarray((jnp.diag(jnp.asarray((0.1, 0.1, 0.1))),)),
        0.01,
    )
    assert bool(exchange.header.globally_eligible)
    np.testing.assert_allclose(exchange.conservation_defect, 0.0, atol=0.0)
    assert bool(jnp.any(exchange.extensive_exchange != 0.0))


def test_dynamic_hybrid_ownership_requires_host_epoch_and_conversion_evidence():
    plan = phx.solver.HybridOwnershipEpochPlan(
        enter_threshold=0.05,
        leave_threshold=0.02,
        minimum_dwell_steps=2,
        buffer_layers=1,
    )
    state = plan.initialize(jnp.asarray((False, False, False)))
    adjacency = jnp.asarray(
        ((False, True, False), (True, False, True), (False, True, False))
    )
    breakdown = phx.AdmissibilityHeader(
        jnp.ones((3,)),
        jnp.zeros((3,), dtype=jnp.uint32),
        "breakdown",
        "breakdown-evidence",
    )
    request = plan.classify(
        state,
        jnp.asarray((0.0, 0.1, 0.0)),
        breakdown,
        adjacency,
        4,
        20,
    )
    assert bool(request.header.globally_eligible)
    assert request.required_particles == 12
    assert request.candidate_mask.tolist() == [True, True, True]
    assert state.kinetic_mask.tolist() == [False, False, False]

    admitted = phx.AdmissibilityHeader(
        jnp.asarray(1.0),
        jnp.asarray(0, dtype=jnp.uint32),
        "conversion",
        "conversion-evidence",
    )
    transitioned = plan.transition_epoch(state, request, admitted, admitted)
    assert transitioned.kinetic_mask.tolist() == [True, True, True]

    rejected = plan.classify(
        state,
        jnp.asarray((0.0, 0.1, 0.0)),
        breakdown,
        adjacency,
        4,
        4,
    )
    assert not bool(rejected.header.globally_eligible)
    assert state.kinetic_mask.tolist() == [False, False, False]


def test_vss_uses_its_angular_parameter_and_preserves_pair_invariants():
    species = _species()
    parameters = phx.discretization.dsmc.DSMCPairCollisionParameters(
        jnp.asarray(((3.7e-10,),)),
        jnp.asarray(((0.75,),)),
    )
    vhs = phx.discretization.dsmc.DSMCVHSCollisionPlan(species, parameters)
    vss = phx.discretization.dsmc.DSMCVSSCollisionPlan(
        species, parameters, jnp.asarray(((2.0,),))
    )
    state = phx.discretization.dsmc.DSMCParticleState(
        jnp.asarray(((0.25, 0.5, 0.5), (0.75, 0.5, 0.5))),
        jnp.asarray(((100.0, 0.0, 0.0), (-100.0, 0.0, 0.0))),
        jnp.asarray((0, 0), dtype=jnp.int32),
        jnp.zeros((2,)),
        jnp.zeros((2,)),
        jnp.ones((2,)),
        jnp.asarray((0, 0), dtype=jnp.int32),
        jnp.asarray((True, True)),
        jnp.zeros((2,), dtype=jnp.int32),
    )
    relative_speed = jnp.asarray(200.0)
    majorant = (
        1.01
        * vhs.collision_cross_section(jnp.asarray(0), jnp.asarray(0), relative_speed)
        * relative_speed
    )
    arguments = (
        state,
        jnp.asarray((0,), dtype=jnp.int32),
        jnp.asarray((1,), dtype=jnp.int32),
        jnp.asarray((True,)),
        jnp.asarray(((0.0, 0.25, 0.5),)),
        majorant,
    )
    hard = vhs.collide(*arguments)
    soft = vss.collide(*arguments)

    assert bool(hard.successful & soft.successful)
    assert not bool(jnp.allclose(hard.state.velocity, soft.state.velocity))
    np.testing.assert_allclose(hard.energy_defect, 0.0, atol=1.0e-30)
    np.testing.assert_allclose(soft.energy_defect, 0.0, atol=1.0e-30)
