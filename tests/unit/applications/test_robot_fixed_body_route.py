#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import phydrax as phx
from phydrax.applications.robotics import FixedBodyRoutePlan
from phydrax.discretization.particle import ReducedArticulationPlan


def _articulation():
    body_ids = jnp.asarray([100, 101, 102], dtype=jnp.int64)
    particles = phx.discretization.ParticleSetPlan(
        body_ids, jnp.ones((3,)), ambient_dimension=3
    ).prepare()
    bodies = phx.discretization.RigidBodySetPlan(
        jnp.zeros((3,), dtype=jnp.int32),
        jnp.broadcast_to(jnp.eye(3), (3, 3, 3)),
        fixed_mask=jnp.asarray([True, False, False]),
    ).prepare(particles)
    position = jnp.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    orientation = jnp.broadcast_to(jnp.asarray([1.0, 0.0, 0.0, 0.0]), (3, 4))
    reference = bodies.kinematics(
        position, jnp.zeros_like(position), orientation, jnp.zeros_like(position)
    )
    graph = phx.discretization.RigidJointGraphPlan(
        hinge=phx.discretization.HingeJointSetPlan(
            jnp.asarray([11, 12]),
            body_ids[:2],
            body_ids[1:],
            jnp.asarray([[0.5, 0.0, 0.0], [1.5, 0.0, 0.0]]),
            jnp.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
        )
    ).prepare(bodies, reference)
    articulation = ReducedArticulationPlan(
        100,
        jnp.asarray([11, 12]),
        body_ids[:2],
        body_ids[1:],
    ).prepare(graph, reference)
    return body_ids, articulation


def _route():
    body_ids, articulation = _articulation()
    plan = FixedBodyRoutePlan(
        ("flexor", "reserved"),
        (0, 3, 5),
        (int(body_ids[0]), int(body_ids[1]), int(body_ids[2]), int(body_ids[0]), int(body_ids[2])),
        route_mask=(True, False),
    )
    local = jnp.asarray(
        [[0.0, 0.2, 0.0], [0.0, 0.1, 0.0], [0.0, -0.2, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )
    return plan.prepare(articulation, local)


def test_csr_route_length_jvp_and_transpose_are_exact_virtual_power_duals():
    route = _route()
    configuration = jnp.asarray([0.35, -0.2])
    velocity = jnp.asarray([0.7, -0.4])
    evaluation = route.evaluate(configuration, velocity)
    operator = route.length_jacobian_operator(configuration)

    assert evaluation.world_points_m.shape == (5, 3)
    expected_world_points = []
    for body_id, local_position in zip(
        route.plan.body_ids, route.local_positions_m
    ):
        local_transform = jnp.eye(4).at[:3, 3].set(local_position)
        expected_world_points.append(
            route.articulation.frame_transform(
                configuration, body_id, local_transform
            )[:3, 3]
        )
    assert jnp.allclose(
        evaluation.world_points_m,
        jnp.stack(expected_world_points),
    )
    assert evaluation.segment_vectors_m.shape == (3, 3)
    assert evaluation.route_lengths_m.shape == (2,)
    assert evaluation.route_lengths_m[0] > 0.0
    assert evaluation.route_lengths_m[1] == 0.0
    assert jnp.allclose(evaluation.route_length_rates_m_per_s, operator.mv(velocity))
    assert evaluation.successful.tolist() == [True, True]

    load, evidence = route.tensile_force_pullback(
        configuration, velocity, jnp.asarray([125.0, 900.0])
    )
    assert load.shape == (2,)
    assert bool(evidence.successful)
    assert jnp.allclose(evidence.route_power_W, evidence.generalized_power_W)
    assert jnp.allclose(evidence.power_residual_W, 0.0, atol=1.0e-11)


def test_route_is_jittable_vmappable_and_differentiable_in_local_coordinates():
    route = _route()
    configurations = jnp.asarray([[0.0, 0.0], [0.2, -0.1], [-0.3, 0.25]])
    lengths = jax.jit(jax.vmap(route.lengths))(configurations)
    assert lengths.shape == (3, 2)

    point = configurations[1]
    direction = jnp.asarray([0.4, -0.6])
    _, jvp = jax.jvp(route.lengths, (point,), (direction,))
    assert jnp.allclose(jvp, route.length_jacobian_operator(point).mv(direction))

    def length_from_local(local):
        changed = eqx.tree_at(lambda value: value.local_positions_m, route, local)
        return changed.lengths(point)[0]

    gradient = jax.grad(length_from_local)(route.local_positions_m)
    assert gradient.shape == route.local_positions_m.shape
    assert jnp.all(jnp.isfinite(gradient))


def test_degenerate_active_segment_and_compressive_input_fail_closed():
    body_ids, articulation = _articulation()
    route = FixedBodyRoutePlan(
        ("degenerate",), (0, 2), (int(body_ids[0]), int(body_ids[0]))
    ).prepare(articulation, jnp.zeros((2, 3)))
    evaluation = route.evaluate(jnp.zeros((2,)), jnp.ones((2,)))
    assert not bool(evaluation.successful[0])

    load, evidence = route.tensile_force_pullback(
        jnp.zeros((2,)), jnp.ones((2,)), jnp.asarray([-1.0])
    )
    assert not bool(evidence.successful)
    assert jnp.array_equal(load, jnp.zeros_like(load))


def test_plan_rejects_dynamic_or_invalid_topology_at_preparation():
    with pytest.raises(ValueError, match="at least two points"):
        FixedBodyRoutePlan(("route",), (0, 1), (100,))
    with pytest.raises(ValueError, match="CSR offsets"):
        FixedBodyRoutePlan(("route",), (1, 3), (100, 101, 102))
