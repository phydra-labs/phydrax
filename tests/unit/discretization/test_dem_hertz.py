#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp

import phydrax as phx


def _hertz_problem():
    particles = phx.discretization.ParticleSetPlan(
        jnp.asarray([0, 1]),
        jnp.asarray([1.0, 1.0]),
        ambient_dimension=3,
    ).prepare()
    spheres = phx.discretization.RigidSphereSetPlan(
        jnp.asarray([0.5, 0.5]), jnp.asarray([0, 0])
    )
    materials = phx.equations.DEMMaterialTable(
        jnp.asarray([2.0e6]),
        jnp.asarray([0.25]),
        jnp.asarray([[0.9]]),
        jnp.asarray([[0.4]]),
    )
    method = phx.discretization.SoftSphereDEMMethodPlan(
        phx.discretization.DEMContactModelPlan(
            phx.discretization.HertzNormalContactPlan(),
            tangential=phx.discretization.MindlinTangentialContactPlan(),
        ),
        maximum_overlap_fraction=0.25,
    )
    problem = phx.equations.DiscreteElementProblemIR(
        "hertz-contact", materials, gravity=jnp.zeros((3,))
    )
    return phx.equations.compile_discrete_element_problem(
        problem,
        particles,
        spheres,
        method,
        neighborhood=phx.discretization.DenseParticleNeighborhoodPlan(1),
    )


def test_hertz_force_matches_negative_distance_derivative_of_elastic_energy():
    compiled = _hertz_problem()

    def energy(distance):
        state = compiled.initialize_state(
            0.0,
            jnp.asarray([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]]),
            jnp.zeros((2, 3)),
        )
        return compiled.diagnostics(0.0, state).elastic_energy

    distance = jnp.asarray(0.9)
    state = compiled.initialize_state(
        0.0,
        jnp.asarray([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]]),
        jnp.zeros((2, 3)),
    )
    force_on_right = state.loads.total.force[1, 0]
    energy_gradient = jax.grad(energy)(distance)

    assert force_on_right > 0.0
    assert jnp.allclose(force_on_right, -energy_gradient, rtol=2.0e-6)


def test_mindlin_tangential_force_respects_friction_cone():
    compiled = _hertz_problem()
    state = compiled.initialize_state(
        0.0,
        jnp.asarray([[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]]),
        jnp.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),
    )
    detail = compiled.dynamics.step_detailed(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        state,
        jnp.asarray(1.0e-5),
        None,
    )
    response = detail.evaluation.particle_contact

    assert detail.successful
    assert jnp.linalg.norm(response.tangential_force[0]) <= (
        0.4 * jnp.linalg.norm(response.normal_force[0]) + 1.0e-9
    )
