#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


def test_rigid_marker_velocity_and_load_are_adjoint():
    particles = phx.discretization.ParticleSetPlan(
        jnp.asarray([0]), jnp.asarray([2.0]), ambient_dimension=2
    ).prepare()
    bodies = phx.discretization.RigidBodySetPlan(
        jnp.asarray([0]), jnp.asarray([0.5])
    ).prepare(particles)
    markers = phx.discretization.LagrangianMarkerSetPlan(
        jnp.asarray([0, 1]),
        jnp.asarray([[0.2, 0.0], [-0.2, 0.0]]),
        jnp.asarray([0.5, 0.5]),
    ).prepare()
    rigid_map = phx.discretization.RigidMarkerMapPlan(
        markers, bodies, jnp.asarray([0, 0])
    ).prepare()
    kinematics = bodies.kinematics(
        jnp.asarray([[0.5, 0.5]]),
        jnp.asarray([[0.1, -0.2]]),
        jnp.asarray([[0.0]]),
        jnp.asarray([[0.3]]),
    )
    operator = rigid_map.velocity_operator(kinematics)
    generalized = rigid_map.generalized_velocity(kinematics)
    multiplier = jnp.asarray([[0.4, -0.1], [-0.2, 0.3]])
    load = operator.adjoint_mv(multiplier)

    assert jnp.isclose(
        markers.active_velocity_space.inner(operator.mv(generalized), multiplier),
        rigid_map.generalized_velocity_space.inner(generalized, load),
        atol=1.0e-10,
    )
    evaluated = rigid_map.evaluate(kinematics)
    assert evaluated.position.shape == (2, 2)
    assert evaluated.velocity.shape == (2, 2)


def test_finite_element_marker_map_uses_paired_H_and_H_adjoint():
    markers = phx.discretization.LagrangianMarkerSetPlan(
        jnp.asarray([0, 1]),
        jnp.zeros((2, 2)),
        jnp.asarray([0.25, 0.75]),
    ).prepare()
    configuration_space = phx.linalg.ArraySpace((4,))
    marker_map = phx.discretization.FiniteElementImmersedMarkerMapPlan(
        markers, configuration_space, jnp.eye(4)
    ).prepare()
    configuration = jnp.asarray([0.2, 0.3, 0.7, 0.8])
    velocity = jnp.asarray([0.1, -0.1, 0.2, -0.2])
    multiplier = jnp.asarray([[0.4, -0.2], [0.3, 0.1]])
    load = marker_map.structural_load(multiplier)

    assert jnp.isclose(
        markers.active_velocity_space.inner(
            marker_map.active_velocity(velocity), multiplier
        ),
        configuration_space.inner(velocity, load),
        atol=1.0e-10,
    )
    state = marker_map.kinematics(configuration, velocity)
    assert jnp.allclose(markers.active_values(state.position).reshape((-1,)), configuration)
