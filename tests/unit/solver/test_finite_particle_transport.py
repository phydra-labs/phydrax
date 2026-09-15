#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _velocity(time, position, args):
    del time, args
    return jnp.broadcast_to(jnp.asarray((1.0, 0.0)), position.shape)


def _plans(
    *, motion, wall_policy=phx.discretization.FiniteParticleWallPolicy.IMPERMEABLE_SLIDE
):
    particles = phx.discretization.ParticleSetPlan(
        jnp.asarray((10, 11)),
        jnp.asarray((1.0, 0.0)),
        ambient_dimension=2,
        active_mask=jnp.asarray((True, False)),
    ).prepare()
    population = phx.discretization.ParticlePopulationPlan(particles)
    properties = phx.discretization.FiniteParticleProperties(
        jnp.asarray((0.1, 0.1)),
        jnp.asarray((0.5, 0.5)),
        jnp.asarray((1.0, 1.0)),
        jnp.zeros((2, 2, 2)),
    )
    units = phx.discretization.FiniteParticleTransportUnits(
        length_unit_id="m",
        time_unit_id="s",
        mass_unit_id="kg",
        temperature_unit_id="K",
        frame="laboratory",
    )
    field = phx.discretization.FiniteParticleVelocityFieldPlan(
        _velocity,
        provider_id="uniform-x",
        velocity_unit_id="m/s",
        frame="laboratory",
    )
    geometry = phx.geometry.Circle((0.0, 0.0), 1.0, feature_id="obstacle").compile()
    wall = phx.geometry.FiniteRadiusErosionPlan(
        geometry, "outside", geometry_id="circular-obstacle"
    )
    plan = phx.solver.FiniteParticleTransportPlan(
        population,
        properties,
        units,
        field,
        wall,
        motion=motion,
        wall_policy=wall_policy,
    )
    return population, plan


def test_finite_radius_erosion_respects_fluid_side_and_particle_radius():
    circle = phx.geometry.Circle((0.0, 0.0), 1.0, feature_id="circle").compile()
    outside = phx.geometry.FiniteRadiusErosionPlan(
        circle, "outside", geometry_id="circle"
    ).evaluate(jnp.asarray(((1.5, 0.0),)), jnp.asarray((0.2,)))
    np.testing.assert_allclose(outside.clearance, 0.3)
    np.testing.assert_allclose(outside.inward_normal, ((1.0, 0.0),))
    assert bool(outside.header.globally_eligible)

    inside = phx.geometry.FiniteRadiusErosionPlan(
        circle, "inside", geometry_id="circle-interior"
    ).evaluate(jnp.asarray(((0.25, 0.0),)), jnp.asarray((0.2,)))
    np.testing.assert_allclose(inside.clearance, 0.55)
    assert bool(inside.header.globally_eligible)


def test_overdamped_particle_localizes_contact_and_slides_without_penetration():
    population, plan = _plans(
        motion=phx.discretization.FiniteParticleMotionKind.OVERDAMPED_STOKES
    )
    state = plan.initialize(
        population.initialize(),
        jnp.asarray(((-1.5, 0.0), (0.0, 0.0))),
        jnp.zeros((2, 2)),
        jax.random.key(2),
    )
    result = eqx.filter_jit(plan.advance)(state, jnp.asarray(1.0), None)

    assert bool(result.successful)
    assert bool(result.contacted[0])
    np.testing.assert_allclose(result.accepted.position[0, 0], -1.1, atol=2.0e-7)
    assert result.accepted.population.active[0]
    assert result.accepted.event_count[0] == 1


def test_inertial_drag_uses_declared_exact_local_substep_without_auto_switch():
    population, plan = _plans(
        motion=phx.discretization.FiniteParticleMotionKind.INERTIAL_STOKES
    )
    state = plan.initialize(
        population.initialize(),
        jnp.asarray(((-3.0, 0.0), (0.0, 0.0))),
        jnp.zeros((2, 2)),
        jax.random.key(3),
    )
    result = plan.advance(state, jnp.asarray(0.1))
    decay = np.exp(-0.1 / 0.5)

    assert bool(result.successful)
    np.testing.assert_allclose(result.accepted.velocity[0, 0], 1.0 - decay)
    np.testing.assert_allclose(
        result.accepted.position[0, 0],
        -3.0 + 0.1 - 0.5 * (1.0 - decay),
    )


def test_absorbing_wall_deactivates_particle_transactionally():
    population, plan = _plans(
        motion=phx.discretization.FiniteParticleMotionKind.OVERDAMPED_STOKES,
        wall_policy=phx.discretization.FiniteParticleWallPolicy.ABSORB,
    )
    state = plan.initialize(
        population.initialize(),
        jnp.asarray(((-1.5, 0.0), (0.0, 0.0))),
        jnp.zeros((2, 2)),
        jax.random.key(4),
    )
    result = plan.advance(state, jnp.asarray(1.0))

    assert bool(result.successful)
    assert bool(result.absorbed[0])
    assert not bool(result.accepted.population.active[0])
    assert result.accepted.population.mass[0] == 0.0
    assert result.accepted.terminal_code[0] == -2
