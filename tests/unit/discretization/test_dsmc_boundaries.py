#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _species():
    return phx.discretization.dsmc.DSMCSpeciesPlan(
        ("A",),
        jnp.asarray((1.0,)),
        jnp.asarray((1.0,)),
        jnp.asarray((0.5,)),
        jnp.asarray((0.0,)),
        jnp.asarray((0.0,)),
        jnp.asarray((0.0,)),
    )


def _empty_particles(capacity=4):
    return phx.discretization.dsmc.DSMCParticleState(
        jnp.full((capacity, 1), 0.5),
        jnp.zeros((capacity, 1)),
        jnp.zeros((capacity,), dtype=jnp.int32),
        jnp.zeros((capacity,)),
        jnp.zeros((capacity,)),
        jnp.zeros((capacity,)),
        -jnp.ones((capacity,), dtype=jnp.int32),
        jnp.zeros((capacity,), dtype=bool),
        jnp.zeros((capacity,), dtype=jnp.int32),
    )


def _reservoir(*, density=3.0, injection_capacity=3):
    cells = phx.discretization.dsmc.DSMCStructuredCellPlan(
        jnp.asarray((0.0,)), jnp.asarray((1.0,)), (1,)
    )
    face = phx.discretization.dsmc.DSMCBoundaryFacePlan(cells, 0, "lower", "reservoir")
    return phx.discretization.dsmc.DSMCReservoirFacePlan(
        face,
        _species(),
        jnp.asarray((density,)),
        jnp.asarray((0.0,)),
        jnp.asarray((1.0,)),
        temperature=1.0,
        maximum_injections_per_species=injection_capacity,
        boltzmann_constant=1.0,
    )


def test_half_range_reservoir_injects_inward_and_tracks_extensive_content():
    reservoir = _reservoir()
    result = reservoir.inject(
        _empty_particles(),
        reservoir.initialize(),
        jnp.asarray(1.0),
        jax.random.key(5),
    )

    assert bool(result.header.globally_eligible)
    assert int(result.injected_count) == 1
    active = np.flatnonzero(np.asarray(result.state.active))
    assert active.size == 1
    index = int(active[0])
    assert result.state.velocity[index, 0] > 0.0
    assert result.state.statistical_weight[index] == 1.0
    assert result.injected_mass == 1.0
    assert result.injected_energy > 0.0
    assert int(result.state.incarnation[index]) == 1


def test_reservoir_refuses_request_or_particle_capacity_without_truncation():
    request_limited = _reservoir(density=100.0, injection_capacity=1)
    result = request_limited.inject(
        _empty_particles(),
        request_limited.initialize(),
        jnp.asarray(1.0),
        jax.random.key(6),
    )
    assert not bool(result.header.globally_eligible)
    assert int(result.injected_count) == 0
    assert not bool(jnp.any(result.state.active))

    slots_limited = _reservoir(density=10.0, injection_capacity=8)
    full = _empty_particles(capacity=1)
    result = slots_limited.inject(
        full,
        slots_limited.initialize(),
        jnp.asarray(1.0),
        jax.random.key(7),
    )
    assert not bool(result.header.globally_eligible)
    assert int(result.injected_count) == 0


def test_production_surface_event_uses_hit_time_and_surface_ledger():
    species = _species()
    cells = phx.discretization.dsmc.DSMCStructuredCellPlan(
        jnp.asarray((0.0,)), jnp.asarray((1.0,)), (1,)
    )
    streaming = phx.discretization.dsmc.DSMCStreamingPlan(
        cells, (("surface", "specular"),)
    )
    face = phx.discretization.dsmc.DSMCBoundaryFacePlan(cells, 0, "lower", "surface")
    interaction = phx.discretization.dsmc.DSMCSurfaceInteractionPlan(
        species,
        "diffuse",
        jnp.asarray((0.0,)),
        wall_temperature=1.0,
        boltzmann_constant=1.0,
    )
    surface = phx.discretization.dsmc.DSMCSurfaceBoundaryPlan(face, interaction)
    parameters = phx.discretization.dsmc.DSMCPairCollisionParameters(
        jnp.asarray(((1.0,),)),
        jnp.asarray(((0.5,),)),
        boltzmann_constant=1.0,
    )
    collision = phx.discretization.dsmc.DSMCVHSCollisionPlan(species, parameters)
    internal = phx.discretization.dsmc.DSMCInternalReactionPlan(
        species, rotational_relaxation_probability=jnp.asarray((0.0,))
    )
    ntc = phx.discretization.dsmc.DSMCNTCSchedulePlan(
        cells, maximum_particles_per_cell=3, maximum_events_per_cell=4
    )
    moments = phx.discretization.dsmc.DSMCMomentPlan(
        species, cells, minimum_particles_per_cell=1
    )
    plan = phx.solver.DSMCProductionPlan(
        streaming,
        collision,
        internal,
        ntc,
        moments,
        surface_boundaries=(surface,),
    )
    particles = phx.discretization.dsmc.DSMCParticleState(
        jnp.asarray(((0.01,), (0.5,), (0.5,))),
        jnp.asarray(((-1.0,), (0.0,), (0.0,))),
        jnp.zeros((3,), dtype=jnp.int32),
        jnp.zeros((3,)),
        jnp.zeros((3,)),
        jnp.asarray((1.0, 1.0, 0.0)),
        jnp.asarray((0, 0, -1), dtype=jnp.int32),
        jnp.asarray((True, True, False)),
        jnp.zeros((3,), dtype=jnp.int32),
    )
    state = plan.initialize(
        particles, jax.random.key(8), majorant_sigma_speed=jnp.asarray((100.0,))
    )
    result = plan.advance(state, jnp.asarray(0.02))

    assert bool(result.successful)
    assert result.accepted.particles.velocity[0, 0] > 0.0
    assert result.accepted.particles.position[0, 0] > 0.0
    assert result.boundary_exchange.surface_mass == 0.0
    assert result.boundary_exchange.surface_momentum[0] < 0.0


def test_specular_streaming_keeps_exact_upper_face_hit_inside_domain():
    cells = phx.discretization.dsmc.DSMCStructuredCellPlan(
        jnp.asarray((0.0,)), jnp.asarray((1.0,)), (1,)
    )
    state = phx.discretization.dsmc.DSMCParticleState(
        jnp.asarray(((0.5,),)),
        jnp.asarray(((0.5,),)),
        jnp.zeros((1,), dtype=jnp.int32),
        jnp.zeros((1,)),
        jnp.zeros((1,)),
        jnp.ones((1,)),
        jnp.zeros((1,), dtype=jnp.int32),
        jnp.ones((1,), dtype=bool),
        jnp.zeros((1,), dtype=jnp.int32),
    )
    result = phx.discretization.dsmc.DSMCStreamingPlan(
        cells, (("specular", "specular"),)
    ).advance(state, 1.0)

    assert bool(result.successful)
    assert bool(result.state.active[0])
    assert result.state.position[0, 0] < 1.0
    np.testing.assert_allclose(result.state.velocity[0, 0], -0.5)
