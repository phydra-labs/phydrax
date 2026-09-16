#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


def test_thermal_potential_and_trapped_particle_reflection():
    phase = phx.applications.cosmology.phase_transitions
    potential = phase.ThermalQuarticPotential(
        quadratic_coefficient=1.0,
        cubic_coefficient=0.2,
        quartic_coefficient=1.0,
        reference_temperature=1.0,
    )
    stationary = phase.thermal_stationary_points(potential, jnp.asarray(0.8))
    assert bool(stationary.finite)
    assert stationary.stationary_fields.shape == (3,)
    assert jnp.isfinite(phase.thin_wall_thermal_action(1.0, 0.5, 1.0))

    plan = phase.ThinWallBubblePlan(
        surface_tension=1000.0,
        vacuum_energy_difference=0.0,
        inside_mass=1.0,
        outside_mass=100.0,
        time_step=0.02,
        maximum_steps=2,
        maximum_energy_residual=100.0,
    )
    particles = phase.BubbleParticleEnsemble(
        jnp.asarray([[0.99, 0.0, 0.0]]),
        jnp.asarray([[10.0, 0.0, 0.0]]),
        jnp.asarray([1.0]),
        jnp.asarray([True]),
        jnp.asarray([True]),
    )
    state = phase.prepare_thin_wall_bubble(plan, 1.0, 0.0, particles)
    stepped = phase.step_thin_wall_bubble(plan, state)
    assert bool(stepped.reflected[0])
    assert not bool(stepped.transmitted[0])
    assert stepped.state.particles.momenta[0, 0] < 0.0
    assert not bool(stepped.derivative_valid[0])


def test_fixed_target_decay_volume_acceptance_is_physical():
    fixed_target = phx.applications.fixed_target
    plan = fixed_target.DecayVolumePlan(10.0, 20.0, 2.0, length_unit_id="m")
    accepted = fixed_target.long_lived_particle_acceptance(
        plan,
        jnp.asarray([[0.1, 0.0, 10.0], [5.0, 0.0, 1.0]]),
        jnp.asarray([1.0, 1.0]),
        jnp.asarray([5.0, 5.0]),
    )
    assert jnp.all(accepted.valid)
    assert bool(accepted.geometric_acceptance[0])
    assert not bool(accepted.geometric_acceptance[1])
    assert accepted.total_acceptance[0] > 0.0
    assert accepted.total_acceptance[1] == 0.0
