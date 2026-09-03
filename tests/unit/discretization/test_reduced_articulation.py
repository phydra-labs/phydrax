#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx
from phydrax.discretization.particle._reduced_articulation import (
    ArticulationDualityEvidence,
    ArticulationKinematics,
    PreparedReducedArticulation,
    ReducedArticulationPlan,
    ReducedArticulationState,
)


def _prepared_bodies(count, *, dimension=3):
    body_ids = jnp.arange(100, 100 + count, dtype=jnp.int64)
    particles = phx.discretization.ParticleSetPlan(
        body_ids,
        jnp.ones((count,)),
        ambient_dimension=dimension,
    ).prepare()
    inertia = (
        jnp.ones((count,))
        if dimension == 2
        else jnp.broadcast_to(jnp.eye(3), (count, 3, 3))
    )
    bodies = phx.discretization.RigidBodySetPlan(
        jnp.zeros((count,), dtype=jnp.int32),
        inertia,
        fixed_mask=jnp.arange(count) == 0,
    ).prepare(particles)
    positions = jnp.pad(
        jnp.arange(count, dtype=jnp.float64)[:, None],
        ((0, 0), (0, dimension - 1)),
    )
    orientation = (
        jnp.zeros((count, 1))
        if dimension == 2
        else jnp.broadcast_to(
            jnp.asarray([1.0, 0.0, 0.0, 0.0]), (count, 4)
        )
    )
    angular = jnp.zeros((count, 1 if dimension == 2 else 3))
    reference = bodies.kinematics(
        positions,
        jnp.zeros_like(positions),
        orientation,
        angular,
    )
    return body_ids, bodies, reference


def _chain():
    body_ids, bodies, reference = _prepared_bodies(4)
    graph = phx.discretization.RigidJointGraphPlan(
        fixed=phx.discretization.FixedJointSetPlan(
            jnp.asarray([11]), body_ids[:1], body_ids[1:2]
        ),
        hinge=phx.discretization.HingeJointSetPlan(
            jnp.asarray([12]),
            body_ids[1:2],
            body_ids[2:3],
            jnp.asarray([[1.5, 0.0, 0.0]]),
            jnp.asarray([[0.0, 0.0, 1.0]]),
        ),
        prismatic=phx.discretization.PrismaticJointSetPlan(
            jnp.asarray([13]),
            body_ids[2:3],
            body_ids[3:4],
            jnp.asarray([[2.5, 0.0, 0.0]]),
            jnp.asarray([[0.0, 1.0, 0.0]]),
        ),
    ).prepare(bodies, reference)
    plan = ReducedArticulationPlan(
        int(body_ids[0]),
        jnp.asarray([13, 11, 12]),
        jnp.asarray([body_ids[2], body_ids[0], body_ids[1]]),
        jnp.asarray([body_ids[3], body_ids[1], body_ids[2]]),
    )
    return body_ids, reference, plan.prepare(graph, reference)


def _quaternion_multiply(left, right):
    return jnp.concatenate(
        (
            left[:1] * right[:1] - jnp.sum(left[1:] * right[1:], keepdims=True),
            left[:1] * right[1:]
            + right[:1] * left[1:]
            + jnp.cross(left[1:], right[1:]),
        )
    )


def _quaternion_conjugate(value):
    return jnp.concatenate((value[:1], -value[1:]))


def test_preparation_derives_stable_topology_layouts_and_reference_pose():
    body_ids, reference, articulation = _chain()

    assert isinstance(articulation, PreparedReducedArticulation)
    assert articulation.nq == 2
    assert articulation.nv == 2
    assert articulation.state_size == 4
    assert articulation.configuration_slice == slice(0, 2)
    assert articulation.velocity_slice == slice(2, 4)
    assert articulation.joint_configuration_slices == (
        slice(0, 0),
        slice(0, 1),
        slice(1, 2),
    )
    assert articulation.joint_velocity_slices == (
        slice(0, 0),
        slice(0, 1),
        slice(1, 2),
    )
    assert jnp.array_equal(articulation.body_ids, body_ids)
    assert jnp.array_equal(articulation.joint_ids, jnp.asarray([11, 12, 13]))
    assert jnp.array_equal(articulation.dof_body_indices, jnp.asarray([2, 3]))
    assert articulation.state_layout.shape == (4,)
    assert articulation.input_layout.shape == (2,)
    assert articulation.parent_reference_transforms.shape == (3, 4, 4)
    assert articulation.parent_axes.shape == (3, 3)
    assert articulation.parent_anchors.shape == (3, 3)

    zero = articulation.reference_configuration()
    kinematics = articulation.forward_kinematics(zero)
    assert isinstance(kinematics, ArticulationKinematics)
    assert kinematics.finite
    assert jnp.allclose(kinematics.bodies.position, reference.position, atol=1.0e-12)
    assert jnp.allclose(
        kinematics.bodies.orientation, reference.orientation, atol=1.0e-12
    )
    assert jnp.allclose(kinematics.body_transforms[:, :3, 3], reference.position)
    assert jnp.allclose(
        kinematics.body_transforms[:, 3],
        jnp.asarray([0.0, 0.0, 0.0, 1.0]),
    )

    state = ReducedArticulationState(
        jnp.asarray([0.2, -0.1]), jnp.asarray([0.3, 0.4])
    )
    packed = articulation.pack_state(state)
    unpacked = articulation.unpack_state(packed)
    assert packed.shape == (4,)
    assert jnp.array_equal(unpacked.configuration, state.configuration)
    assert jnp.array_equal(unpacked.velocity, state.velocity)
    assert jnp.array_equal(
        articulation.pack_state(state.configuration, state.velocity), packed
    )
    assert articulation.state_layout.geometry.contains(packed)


def test_hinge_and_prismatic_forward_geometry_is_parent_frame_exact():
    body_ids, _, articulation = _chain()
    angle = 0.5 * jnp.pi
    extension = 0.25
    configuration = jnp.asarray([angle, extension])
    kinematics = articulation.forward_kinematics(configuration)

    expected_position = jnp.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.5, 0.5, 0.0],
            [1.25, 1.5, 0.0],
        ]
    )
    expected_orientation = jnp.asarray(
        [jnp.cos(0.5 * angle), 0.0, 0.0, jnp.sin(0.5 * angle)]
    )
    assert jnp.allclose(kinematics.bodies.position, expected_position, atol=1.0e-12)
    assert jnp.allclose(
        kinematics.bodies.orientation[2:], expected_orientation, atol=1.0e-12
    )
    assert jnp.allclose(
        articulation.body_transform(configuration, int(body_ids[3]))[:3, 3],
        expected_position[3],
        atol=1.0e-12,
    )

    local_transform = jnp.eye(4).at[:3, 3].set(jnp.asarray([0.2, 0.0, 0.0]))
    frame = articulation.frame_transform(
        configuration, int(body_ids[3]), local_transform
    )
    assert jnp.allclose(
        frame[:3, 3],
        expected_position[3] + jnp.asarray([0.0, 0.2, 0.0]),
    )


def test_configuration_retraction_difference_and_body_jvp_are_consistent():
    body_ids, _, articulation = _chain()
    configuration = jnp.asarray([0.3, 0.1])
    velocity = jnp.asarray([0.4, -0.2])
    step_size = jnp.asarray(1.0e-5)
    point = articulation.integrate_configuration(configuration, velocity, step_size)
    difference = articulation.configuration_difference(configuration, point)
    assert jnp.allclose(difference, step_size * velocity, atol=1.0e-13)

    body_velocity = articulation.body_velocity_action(configuration, velocity)
    _, position_jvp = jax.jvp(
        lambda value: articulation.forward_kinematics(value).bodies.position,
        (configuration,),
        (velocity,),
    )
    orientation, orientation_jvp = jax.jvp(
        lambda value: articulation.forward_kinematics(value).bodies.orientation,
        (configuration,),
        (velocity,),
    )
    angular_jvp = jax.vmap(
        lambda tangent, quaternion: 2.0
        * _quaternion_multiply(tangent, _quaternion_conjugate(quaternion))[1:]
    )(orientation_jvp, orientation)
    assert jnp.allclose(body_velocity[:, :3], position_jvp, atol=1.0e-12)
    assert jnp.allclose(body_velocity[:, 3:], angular_jvp, atol=1.0e-12)

    jacobian = articulation.body_jacobian_operator(configuration)
    assert isinstance(jacobian, phx.linalg.FunctionLinearOperator)
    assert jnp.allclose(jacobian.mv(velocity), body_velocity, atol=1.0e-12)

    local_position = jnp.asarray([0.2, -0.1, 0.3])
    frame_jacobian = articulation.frame_jacobian_operator(
        configuration, int(body_ids[3]), local_position
    )
    local_transform = jnp.eye(4).at[:3, 3].set(local_position)
    _, frame_position_jvp = jax.jvp(
        lambda value: articulation.frame_transform(
            value, int(body_ids[3]), local_transform
        )[:3, 3],
        (configuration,),
        (velocity,),
    )
    assert jnp.allclose(
        frame_jacobian.mv(velocity)[:3], frame_position_jvp, atol=1.0e-12
    )


def test_body_load_pullback_reports_finite_power_duality():
    _, _, articulation = _chain()
    configuration = jnp.asarray([0.4, -0.15])
    velocity = jnp.asarray([0.7, -0.3])
    force = jnp.asarray(
        [
            [0.2, -0.4, 0.1],
            [1.0, 0.3, -0.2],
            [-0.5, 0.8, 0.6],
            [0.4, -0.9, 0.7],
        ]
    )
    torque = jnp.asarray(
        [
            [0.1, 0.2, 0.3],
            [-0.2, 0.5, 0.4],
            [0.7, -0.1, 0.2],
            [0.3, 0.6, -0.8],
        ]
    )
    load = phx.discretization.RigidBodyLoad(force, torque)
    generalized_load, evidence = articulation.body_load_pullback(
        configuration, load, velocity
    )

    assert generalized_load.shape == (articulation.nv,)
    assert isinstance(evidence, ArticulationDualityEvidence)
    assert evidence.finite
    assert evidence.valid
    body_velocity = articulation.body_velocity_action(configuration, velocity)
    body_power = jnp.sum(body_velocity[:, :3] * force) + jnp.sum(
        body_velocity[:, 3:] * torque
    )
    assert jnp.allclose(evidence.body_power, body_power, atol=1.0e-12)
    assert jnp.allclose(evidence.generalized_power, velocity @ generalized_load)
    assert jnp.abs(evidence.residual) < 1.0e-12


def test_malformed_disconnected_reversed_and_missing_tree_inputs_reject():
    body_ids, reference, articulation = _chain()
    graph = articulation.graph

    with pytest.raises(ValueError, match="matching shapes"):
        ReducedArticulationPlan(
            int(body_ids[0]),
            jnp.asarray([11, 12]),
            jnp.asarray([body_ids[0]]),
            jnp.asarray([body_ids[1], body_ids[2]]),
        )
    with pytest.raises(ValueError, match="left-to-right orientation"):
        ReducedArticulationPlan(
            int(body_ids[0]),
            jnp.asarray([11, 12, 13]),
            jnp.asarray([body_ids[0], body_ids[2], body_ids[2]]),
            jnp.asarray([body_ids[1], body_ids[1], body_ids[3]]),
        ).prepare(graph, reference)
    with pytest.raises(ValueError, match="joint ID is absent"):
        ReducedArticulationPlan(
            int(body_ids[0]),
            jnp.asarray([11, 12, 999]),
            jnp.asarray([body_ids[0], body_ids[1], body_ids[2]]),
            jnp.asarray([body_ids[1], body_ids[2], body_ids[3]]),
        ).prepare(graph, reference)
    with pytest.raises(ValueError, match="body ID is absent"):
        ReducedArticulationPlan(
            int(body_ids[0]),
            jnp.asarray([11, 12, 13]),
            jnp.asarray([body_ids[0], body_ids[1], body_ids[2]]),
            jnp.asarray([body_ids[1], body_ids[2], 999]),
        ).prepare(graph, reference)

    disconnected_ids, bodies, disconnected_reference = _prepared_bodies(4)
    disconnected_graph = phx.discretization.RigidJointGraphPlan(
        prismatic=phx.discretization.PrismaticJointSetPlan(
            jnp.asarray([21, 22]),
            jnp.asarray([disconnected_ids[0], disconnected_ids[2]]),
            jnp.asarray([disconnected_ids[1], disconnected_ids[3]]),
            jnp.asarray([[0.5, 0.0, 0.0], [2.5, 0.0, 0.0]]),
            jnp.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        )
    ).prepare(bodies, disconnected_reference)
    with pytest.raises(ValueError, match="connected articulation tree"):
        ReducedArticulationPlan(
            int(disconnected_ids[0]),
            jnp.asarray([21, 22]),
            jnp.asarray([disconnected_ids[0], disconnected_ids[2]]),
            jnp.asarray([disconnected_ids[1], disconnected_ids[3]]),
        ).prepare(disconnected_graph, disconnected_reference)

    duplicate_ids, duplicate_bodies, duplicate_reference = _prepared_bodies(3)
    duplicate_graph = phx.discretization.RigidJointGraphPlan(
        prismatic=phx.discretization.PrismaticJointSetPlan(
            jnp.asarray([31, 32]),
            jnp.asarray([duplicate_ids[0], duplicate_ids[2]]),
            jnp.asarray([duplicate_ids[1], duplicate_ids[1]]),
            jnp.asarray([[0.5, 0.0, 0.0], [1.5, 0.0, 0.0]]),
            jnp.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        )
    ).prepare(duplicate_bodies, duplicate_reference)
    with pytest.raises(ValueError, match="exactly one parent"):
        ReducedArticulationPlan(
            int(duplicate_ids[0]),
            jnp.asarray([31, 32]),
            jnp.asarray([duplicate_ids[0], duplicate_ids[2]]),
            jnp.asarray([duplicate_ids[1], duplicate_ids[1]]),
        ).prepare(duplicate_graph, duplicate_reference)


def test_cyclic_ball_distance_and_non_3d_tree_inputs_reject():
    cycle_ids, cycle_bodies, cycle_reference = _prepared_bodies(6)
    cycle_graph = phx.discretization.RigidJointGraphPlan(
        prismatic=phx.discretization.PrismaticJointSetPlan(
            jnp.arange(40, 46),
            cycle_ids,
            jnp.roll(cycle_ids, -1),
            jnp.zeros((6, 3)),
            jnp.broadcast_to(jnp.asarray([0.0, 0.0, 1.0]), (6, 3)),
        )
    ).prepare(cycle_bodies, cycle_reference)
    with pytest.raises(ValueError):
        ReducedArticulationPlan(
            int(cycle_ids[0]),
            jnp.arange(40, 46),
            cycle_ids,
            jnp.roll(cycle_ids, -1),
        ).prepare(cycle_graph, cycle_reference)

    for unsupported in ("ball", "distance"):
        body_ids, bodies, reference = _prepared_bodies(2)
        if unsupported == "ball":
            graph_plan = phx.discretization.RigidJointGraphPlan(
                ball=phx.discretization.BallJointSetPlan(
                    jnp.asarray([51]),
                    body_ids[:1],
                    body_ids[1:],
                    jnp.asarray([[0.5, 0.0, 0.0]]),
                )
            )
        else:
            graph_plan = phx.discretization.RigidJointGraphPlan(
                distance=phx.discretization.DistanceJointSetPlan(
                    jnp.asarray([51]),
                    body_ids[:1],
                    body_ids[1:],
                    jnp.asarray([[0.0, 0.0, 0.0]]),
                    jnp.asarray([[1.0, 0.0, 0.0]]),
                )
            )
        graph = graph_plan.prepare(bodies, reference)
        with pytest.raises(ValueError, match="only fixed, hinge, and prismatic"):
            ReducedArticulationPlan(
                int(body_ids[0]),
                jnp.asarray([51]),
                body_ids[:1],
                body_ids[1:],
            ).prepare(graph, reference)

    body_ids_2d, bodies_2d, reference_2d = _prepared_bodies(2, dimension=2)
    graph_2d = phx.discretization.RigidJointGraphPlan(
        fixed=phx.discretization.FixedJointSetPlan(
            jnp.asarray([61]), body_ids_2d[:1], body_ids_2d[1:]
        )
    ).prepare(bodies_2d, reference_2d)
    with pytest.raises(ValueError, match="three dimensions"):
        ReducedArticulationPlan(
            int(body_ids_2d[0]),
            jnp.asarray([61]),
            body_ids_2d[:1],
            body_ids_2d[1:],
        ).prepare(graph_2d, reference_2d)
