#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

from phydrax.discretization import ParticleSetPlan, RigidBodySetPlan
from phydrax.discretization.particle._rigid_constraint_dynamics import (
    RigidConstraintDynamicsPlan,
)
from phydrax.discretization.particle._rigid_joint_coordinates import (
    prepare_rigid_joint_coordinates,
)
from phydrax.discretization.particle._rigid_joints import (
    BallJointSetPlan,
    DistanceJointSetPlan,
    PrismaticJointSetPlan,
    RigidJointGraphPlan,
)


def _planar_bodies():
    particles = ParticleSetPlan(
        jnp.asarray([10, 11]),
        jnp.ones((2,)),
        ambient_dimension=2,
    ).prepare()
    bodies = RigidBodySetPlan(
        jnp.zeros((2,), dtype=jnp.int32),
        jnp.ones((2,)),
        fixed_mask=jnp.asarray([True, False]),
    ).prepare(particles)
    position = jnp.asarray([[0.0, 0.0], [1.0, 0.0]])
    reference = bodies.kinematics(
        position,
        jnp.zeros_like(position),
        jnp.zeros((2, 1)),
        jnp.zeros((2, 1)),
    )
    return bodies, reference


def test_planar_ball_joint_uses_native_se2_projection():
    bodies, reference = _planar_bodies()
    graph = RigidJointGraphPlan(
        ball=BallJointSetPlan(
            jnp.asarray([20]),
            jnp.asarray([10]),
            jnp.asarray([11]),
            jnp.asarray([[0.5, 0.0]]),
        )
    )
    dynamics = RigidConstraintDynamicsPlan(graph).prepare(bodies, reference)
    state = dynamics.initialize_state(
        reference.position,
        reference.velocity,
        reference.orientation,
        reference.angular_velocity,
    )
    result = dynamics.step(state, jnp.asarray(0.0), jnp.asarray(1.0e-3))
    assert result.successful
    assert result.evaluation.diagnostics.maximum_position_residual < 1.0e-10
    assert result.evaluation.diagnostics.constraint_rank == 2
    assert dynamics.joints.row_layout.row_count == 2


def test_planar_prismatic_coordinate_and_forbidden_rows_are_objective():
    bodies, reference = _planar_bodies()
    graph = RigidJointGraphPlan(
        prismatic=PrismaticJointSetPlan(
            jnp.asarray([30]),
            jnp.asarray([10]),
            jnp.asarray([11]),
            jnp.asarray([[0.5, 0.0]]),
            jnp.asarray([[1.0, 0.0]]),
        )
    ).prepare(bodies, reference)
    coordinates = prepare_rigid_joint_coordinates(graph)
    translated = bodies.kinematics(
        reference.position.at[1, 0].add(0.25),
        reference.velocity,
        reference.orientation,
        reference.angular_velocity,
    )
    evaluation = coordinates.evaluate(translated)
    residuals = graph.residuals(translated)
    assert jnp.allclose(residuals.prismatic_translation, 0.0)
    assert jnp.allclose(residuals.prismatic_rotation, 0.0)
    assert jnp.allclose(evaluation.prismatic_position, 0.25)

    forbidden = bodies.kinematics(
        translated.position.at[1, 1].add(0.1),
        translated.velocity,
        translated.orientation.at[1, 0].add(0.2),
        translated.angular_velocity,
    )
    forbidden_residuals = graph.residuals(forbidden)
    assert jnp.max(jnp.abs(forbidden_residuals.prismatic_translation)) > 0.0
    assert jnp.max(jnp.abs(forbidden_residuals.prismatic_rotation)) > 0.0


def test_planar_distance_joint_has_scaled_nonzero_gradient():
    bodies, reference = _planar_bodies()
    graph = RigidJointGraphPlan(
        distance=DistanceJointSetPlan(
            jnp.asarray([40]),
            jnp.asarray([10]),
            jnp.asarray([11]),
            jnp.asarray([[0.0, 0.0]]),
            jnp.asarray([[1.0, 0.0]]),
        )
    ).prepare(bodies, reference)
    assert jnp.allclose(graph.residuals(reference).distance, 0.0)
    stretched = bodies.kinematics(
        reference.position.at[1, 0].set(1.2),
        reference.velocity,
        reference.orientation,
        reference.angular_velocity,
    )
    assert graph.residuals(stretched).distance[0] > 0.0
    packed = graph.pack_residuals(graph.residuals(stretched))
    assert packed.shape == (1,)
    assert graph.row_layout.row_count == 1
