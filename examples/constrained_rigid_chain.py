#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Advance a fixed-topology ball-and-hinge rigid-body chain."""

import equinox as eqx
import jax
import jax.numpy as jnp

import phydrax as phx


def main():
    body_ids = jnp.asarray([100, 101, 102], dtype=jnp.int64)
    masses = jnp.asarray([1.0, 1.0, 1.0])
    particles = phx.discretization.ParticleSetPlan(
        body_ids,
        masses,
        ambient_dimension=3,
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
            jnp.asarray([200], dtype=jnp.int64),
            jnp.asarray([100], dtype=jnp.int64),
            jnp.asarray([101], dtype=jnp.int64),
            jnp.asarray([[0.5, 0.0, 0.0]]),
        ),
        hinge=phx.discretization.HingeJointSetPlan(
            jnp.asarray([201], dtype=jnp.int64),
            jnp.asarray([101], dtype=jnp.int64),
            jnp.asarray([102], dtype=jnp.int64),
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
        external_load_id="uniform-gravity",
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
                step_size * index,
                step_size,
            )
            evidence = jnp.asarray(
                (
                    result.successful,
                    result.evaluation.diagnostics.maximum_position_residual,
                    result.evaluation.diagnostics.maximum_velocity_residual,
                    result.evaluation.diagnostics.quaternion_defect,
                )
            )
            return result.accepted_state, evidence

        return jax.lax.scan(advance, state, jnp.arange(8))

    final, evidence = rollout(initial)
    print("successful:", bool(jnp.all(evidence[:, 0])))
    print("maximum position residual:", float(jnp.max(evidence[:, 1])))
    print("maximum velocity residual:", float(jnp.max(evidence[:, 2])))
    print("maximum quaternion defect:", float(jnp.max(evidence[:, 3])))
    print("final positions:")
    print(final.kinematics.position)


if __name__ == "__main__":
    main()
