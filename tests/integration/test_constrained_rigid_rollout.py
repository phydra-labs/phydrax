#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp

import phydrax as phx


def test_constrained_rigid_chain_rollout_is_transactional_and_drift_free():
    body_ids = jnp.asarray([10, 11, 12], dtype=jnp.int64)
    masses = jnp.ones((3,))
    particles = phx.discretization.ParticleSetPlan(
        body_ids, masses, ambient_dimension=3
    ).prepare()
    bodies = phx.discretization.RigidBodySetPlan(
        jnp.zeros((3,), dtype=jnp.int32),
        jnp.stack((jnp.eye(3), jnp.eye(3), jnp.eye(3))),
        fixed_mask=jnp.asarray([True, False, False]),
    ).prepare(particles)
    position = jnp.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    orientation = jnp.asarray([[1.0, 0.0, 0.0, 0.0]] * 3)
    reference = bodies.kinematics(
        position,
        jnp.zeros_like(position),
        orientation,
        jnp.zeros_like(position),
    )
    graph = phx.discretization.RigidJointGraphPlan(
        ball=phx.discretization.BallJointSetPlan(
            jnp.asarray([20]),
            jnp.asarray([10]),
            jnp.asarray([11]),
            jnp.asarray([[0.5, 0.0, 0.0]]),
        ),
        hinge=phx.discretization.HingeJointSetPlan(
            jnp.asarray([21]),
            jnp.asarray([11]),
            jnp.asarray([12]),
            jnp.asarray([[1.5, 0.0, 0.0]]),
            jnp.asarray([[0.0, 0.0, 1.0]]),
        ),
    )

    def gravity(time, kinematics, args):
        del time, args
        return phx.discretization.RigidBodyLoad(
            masses[:, None] * jnp.asarray([0.0, -9.81, 0.0]),
            jnp.zeros_like(kinematics.angular_velocity),
        )

    dynamics = phx.discretization.RigidConstraintDynamicsPlan(graph).prepare(
        bodies,
        reference,
        external_load=gravity,
        external_load_id="rollout-gravity",
    )
    initial = dynamics.initialize_state(
        reference.position,
        reference.velocity,
        reference.orientation,
        reference.angular_velocity,
    )
    step_size = jnp.asarray(1.0e-3)

    @eqx.filter_jit
    def rollout(state):
        def advance(current, index):
            result = dynamics.step(
                current,
                index * step_size,
                step_size,
            )
            metrics = jnp.asarray(
                (
                    result.successful,
                    result.evaluation.diagnostics.maximum_position_residual,
                    result.evaluation.diagnostics.maximum_velocity_residual,
                    result.evaluation.diagnostics.quaternion_defect,
                    result.evaluation.diagnostics.fixed_pose_defect,
                )
            )
            return result.accepted_state, metrics

        return jax.lax.scan(advance, state, jnp.arange(16))

    final, metrics = rollout(initial)
    assert jnp.all(metrics[:, 0])
    assert jnp.max(metrics[:, 1]) < 1.0e-8
    assert jnp.max(metrics[:, 2]) < 1.0e-8
    assert jnp.max(metrics[:, 3]) < 1.0e-10
    assert jnp.max(metrics[:, 4]) < 1.0e-12
    assert jnp.allclose(final.kinematics.position[0], initial.kinematics.position[0])
    assert final.kinematics.position[2, 1] < initial.kinematics.position[2, 1]
