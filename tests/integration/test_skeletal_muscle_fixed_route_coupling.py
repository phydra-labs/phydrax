#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx
from phydrax.applications.robotics import FixedBodyRoutePlan
from phydrax.applications.skeletal_muscle.musculotendon import (
    de_groote_fregly_2016_active_force_length,
    de_groote_fregly_2016_force_velocity,
    de_groote_fregly_2016_inverse_tendon_force_length,
    DeGrooteFregly2016Parameters,
    DeGrooteFregly2016Plan,
    DeGrooteFregly2016State,
)
from phydrax.discretization.particle import ReducedArticulationPlan


def _one_hinge():
    body_ids = jnp.asarray([100, 101], dtype=jnp.int64)
    particles = phx.discretization.ParticleSetPlan(
        body_ids, jnp.ones((2,)), ambient_dimension=3
    ).prepare()
    bodies = phx.discretization.RigidBodySetPlan(
        jnp.zeros((2,), dtype=jnp.int32),
        jnp.broadcast_to(jnp.eye(3), (2, 3, 3)),
        fixed_mask=jnp.asarray([True, False]),
    ).prepare(particles)
    position = jnp.asarray([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0]])
    orientation = jnp.broadcast_to(jnp.asarray([1.0, 0.0, 0.0, 0.0]), (2, 4))
    reference = bodies.kinematics(
        position, jnp.zeros_like(position), orientation, jnp.zeros_like(position)
    )
    graph = phx.discretization.RigidJointGraphPlan(
        hinge=phx.discretization.HingeJointSetPlan(
            jnp.asarray([11]),
            body_ids[:1],
            body_ids[1:],
            jnp.asarray([[0.2, 0.0, 0.0]]),
            jnp.asarray([[0.0, 0.0, 1.0]]),
        )
    ).prepare(bodies, reference)
    return body_ids, ReducedArticulationPlan(
        100, jnp.asarray([11]), body_ids[:1], body_ids[1:]
    ).prepare(graph, reference)


def test_de_groote_force_is_the_single_owner_pulled_back_through_fixed_route():
    body_ids, articulation = _one_hinge()
    route = FixedBodyRoutePlan(
        ("flexor",), (0, 2), (int(body_ids[0]), int(body_ids[1]))
    ).prepare(
        articulation,
        jnp.asarray([[0.0, 0.18, 0.0], [0.22, -0.10, 0.0]]),
    )
    configuration = jnp.asarray([0.3])
    generalized_velocity = jnp.asarray([-0.4])
    geometry = route.evaluate(configuration, generalized_velocity)

    activation = jnp.asarray([0.45])
    optimal_length = jnp.asarray([0.10])
    pennation = jnp.asarray([0.12])
    provisional = DeGrooteFregly2016Parameters(
        jnp.asarray([1600.0]),
        optimal_length,
        jnp.asarray([0.20]),
        pennation,
        jnp.asarray([1.0]),
    )
    normalized_force = (
        activation
        * de_groote_fregly_2016_active_force_length(provisional, jnp.ones((1,)))
        * de_groote_fregly_2016_force_velocity(provisional, jnp.zeros((1,)))
        * jnp.cos(pennation)
    )
    normalized_tendon_length = de_groote_fregly_2016_inverse_tendon_force_length(
        provisional, normalized_force
    )
    slack_length = (
        geometry.route_lengths_m - optimal_length * jnp.cos(pennation)
    ) / normalized_tendon_length
    parameters = DeGrooteFregly2016Parameters(
        jnp.asarray([1600.0]),
        optimal_length,
        slack_length,
        pennation,
        jnp.asarray([1.0]),
    )
    state = DeGrooteFregly2016State(activation, normalized_force)
    muscle = DeGrooteFregly2016Plan(parameters, ("flexor",)).prepare(state)
    evaluation = muscle.evaluate(
        state,
        activation,
        geometry.route_lengths_m,
        geometry.route_length_rates_m_per_s,
    )
    generalized_load, power = route.tensile_force_pullback(
        configuration,
        generalized_velocity,
        evaluation.tendon_force_N,
    )

    assert jnp.all(geometry.successful)
    assert jnp.all(evaluation.successful)
    assert evaluation.force_owner == "de-groote-fregly-2016"
    assert generalized_load.shape == (articulation.nv,)
    assert bool(power.successful)
    assert jnp.allclose(power.power_residual_W, 0.0, atol=1.0e-8)
