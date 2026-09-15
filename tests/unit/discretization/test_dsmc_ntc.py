#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _plans(*, maximum_events_per_cell=64):
    species = phx.discretization.dsmc.DSMCSpeciesPlan(
        ("A",),
        jnp.asarray((4.65e-26,)),
        jnp.asarray((3.7e-10,)),
        jnp.asarray((0.75,)),
        jnp.asarray((0.0,)),
        jnp.asarray((0.0,)),
        jnp.asarray((0.0,)),
    )
    cells = phx.discretization.dsmc.DSMCStructuredCellPlan(
        jnp.asarray((0.0,)), jnp.asarray((1.0,)), (2,)
    )
    streaming = phx.discretization.dsmc.DSMCStreamingPlan(
        cells, (("periodic", "periodic"),)
    )
    pair = phx.discretization.dsmc.DSMCPairCollisionParameters(
        jnp.asarray(((3.7e-10,),)), jnp.asarray(((0.75,),))
    )
    collisions = phx.discretization.dsmc.DSMCVHSCollisionPlan(species, pair)
    internal = phx.discretization.dsmc.DSMCInternalReactionPlan(
        species, rotational_relaxation_probability=jnp.asarray((0.0,))
    )
    ntc = phx.discretization.dsmc.DSMCNTCSchedulePlan(
        cells,
        maximum_particles_per_cell=3,
        maximum_events_per_cell=maximum_events_per_cell,
    )
    moments = phx.discretization.dsmc.DSMCMomentPlan(
        species,
        cells,
        minimum_particles_per_cell=2,
        block_size=1,
        minimum_blocks=2,
        maximum_relative_standard_error=1.0e12,
    )
    return species, cells, streaming, collisions, internal, ntc, moments


def _particles(*, weight=1.0e18):
    return phx.discretization.dsmc.DSMCParticleState(
        jnp.asarray(((0.15,), (0.35,), (0.65,), (0.85,), (0.5,), (0.5,))),
        jnp.asarray(((1.0,), (-1.0,), (1.5,), (-1.5,), (0.0,), (0.0,))),
        jnp.asarray((0, 0, 0, 0, 0, 0), dtype=jnp.int32),
        jnp.zeros((6,)),
        jnp.zeros((6,)),
        jnp.asarray((weight, weight, weight, weight, 0.0, 0.0)),
        jnp.asarray((0, 0, 1, 1, -1, -1), dtype=jnp.int32),
        jnp.asarray((True, True, True, True, False, False)),
        jnp.zeros((6,), dtype=jnp.int32),
    )


def test_ntc_schedule_is_cell_local_distinct_and_remainder_preserving():
    _, _, _, _, _, ntc, _ = _plans()
    state = ntc.initialize(jnp.asarray((1.0e-16, 1.0e-16)))
    schedule = ntc.schedule(_particles(), state, jnp.asarray(0.01), jax.random.key(1))

    assert bool(schedule.header.globally_eligible)
    np.testing.assert_array_equal(schedule.candidate_counts, (2, 2))
    active = np.asarray(schedule.valid_events)
    first = np.asarray(schedule.first_indices)[active]
    second = np.asarray(schedule.second_indices)[active]
    cells = np.asarray(schedule.event_cells)[active]
    assert np.all(first != second)
    particle_cells = np.asarray(_particles().cell_id)
    np.testing.assert_array_equal(particle_cells[first], cells)
    np.testing.assert_array_equal(particle_cells[second], cells)
    np.testing.assert_allclose(schedule.candidate_state.fractional_remainder, 0.0)


def test_ntc_refuses_occupancy_event_capacity_and_unequal_weights():
    _, _, _, _, _, ntc, _ = _plans(maximum_events_per_cell=1)
    particles = _particles(weight=1.0e19)
    unequal = phx.discretization.dsmc.DSMCParticleState(
        particles.position,
        particles.velocity,
        particles.species_index,
        particles.rotational_energy,
        particles.vibrational_energy,
        particles.statistical_weight.at[1].set(2.0e19),
        particles.cell_id,
        particles.active,
        particles.incarnation,
    )
    schedule = ntc.schedule(
        unequal,
        ntc.initialize(jnp.asarray((1.0e-16, 1.0e-16))),
        jnp.asarray(0.01),
        jax.random.key(2),
    )

    assert not bool(schedule.header.globally_eligible)
    assert not bool(schedule.occupancy.header.globally_eligible)


def test_dsmc_production_interleaves_cell_local_events_and_rolls_back_majorant():
    _, _, streaming, collisions, internal, ntc, moments = _plans()
    plan = phx.solver.DSMCProductionPlan(streaming, collisions, internal, ntc, moments)
    particles = _particles(weight=1.0e20)
    relative = jnp.asarray((3.0,))
    required = (
        collisions.collision_cross_section(
            jnp.asarray((0,)), jnp.asarray((0,)), relative
        )[0]
        * relative[0]
    )
    initial = plan.initialize(
        particles,
        jax.random.key(3),
        majorant_sigma_speed=jnp.asarray((1.1 * required, 1.1 * required)),
    )
    result = eqx.filter_jit(plan.advance)(initial, jnp.asarray(0.01))

    assert bool(result.successful)
    assert int(result.accepted.accepted_steps) == 1
    assert result.schedule.first_indices.shape == (ntc.event_capacity,)
    np.testing.assert_allclose(result.boundary_exchange.net_mass, 0.0)
    assert bool(jnp.all(result.moments.header.eligible))

    too_small = plan.initialize(
        particles,
        jax.random.key(3),
        majorant_sigma_speed=jnp.asarray((0.1 * required, 0.1 * required)),
    )
    rejected = eqx.filter_jit(plan.advance)(too_small, jnp.asarray(0.01))
    assert not bool(rejected.successful)
    assert int(rejected.accepted.accepted_steps) == 0
    np.testing.assert_array_equal(
        rejected.accepted.particles.velocity, too_small.particles.velocity
    )
    updated = plan.update_majorant(
        rejected.accepted, rejected.required_majorant_sigma_speed
    )
    assert int(updated.ntc.epoch) == 1
    assert bool(
        jnp.all(
            updated.ntc.majorant_sigma_speed >= rejected.required_majorant_sigma_speed
        )
    )


def test_dsmc_initialization_requires_inert_inactive_slots():
    _, _, streaming, collisions, internal, ntc, moments = _plans()
    plan = phx.solver.DSMCProductionPlan(streaming, collisions, internal, ntc, moments)
    particles = _particles()
    invalid = phx.discretization.dsmc.DSMCParticleState(
        particles.position,
        particles.velocity,
        particles.species_index,
        particles.rotational_energy,
        particles.vibrational_energy,
        particles.statistical_weight.at[-1].set(1.0),
        particles.cell_id,
        particles.active,
        particles.incarnation,
    )
    with pytest.raises(ValueError, match="active/inactive"):
        plan.initialize(
            invalid,
            jax.random.key(4),
            majorant_sigma_speed=jnp.asarray((1.0, 1.0)),
        )
