#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exercise planar joints, exact friction projection, and Cosserat rods."""

import jax.numpy as jnp

import phydrax as phx


def main():
    body_ids = jnp.asarray([10, 11], dtype=jnp.int64)
    particles = phx.discretization.ParticleSetPlan(
        body_ids,
        jnp.ones((2,)),
        ambient_dimension=2,
    ).prepare()
    bodies = phx.discretization.RigidBodySetPlan(
        jnp.zeros((2,), dtype=jnp.int32),
        jnp.ones((2,)),
        fixed_mask=jnp.asarray([True, False]),
    ).prepare(particles)
    positions = jnp.asarray([[0.0, 0.0], [1.0, 0.0]])
    reference = bodies.kinematics(
        positions,
        jnp.zeros_like(positions),
        jnp.zeros((2, 1)),
        jnp.zeros((2, 1)),
    )
    graph = phx.discretization.RigidJointGraphPlan(
        prismatic=phx.discretization.PrismaticJointSetPlan(
            jnp.asarray([20]),
            body_ids[:1],
            body_ids[1:],
            jnp.asarray([[0.5, 0.0]]),
            jnp.asarray([[1.0, 0.0]]),
        )
    )
    dynamics = phx.discretization.RigidConstraintDynamicsPlan(graph).prepare(
        bodies, reference
    )
    state = dynamics.initialize_state(
        reference.position,
        reference.velocity.at[1, 0].set(0.25),
        reference.orientation,
        reference.angular_velocity,
    )
    step = dynamics.step(state, jnp.asarray(0.0), jnp.asarray(1.0e-3))
    coordinates = phx.discretization.prepare_rigid_joint_coordinates(
        dynamics.joints
    ).evaluate(step.accepted_state.kinematics)

    friction = phx.discretization.project_isotropic_coulomb_impulse(
        jnp.asarray([1.0]),
        jnp.asarray([[0.8, -0.3]]),
        jnp.asarray([0.5]),
    )

    rod = phx.applications.solid_mechanics.prepare_rod(
        phx.applications.solid_mechanics.RodPlan(
            jnp.asarray(((0, 1), (1, 2)), dtype=jnp.int32),
            jnp.asarray(((0.0, 0.0), (1.0, 0.0), (2.0, 0.0))),
            jnp.broadcast_to(jnp.eye(2), (2, 2, 2)),
            jnp.asarray((1.0, 1.0, 1.0)),
            jnp.asarray((0.2, 0.2)),
            jnp.broadcast_to(jnp.diag(jnp.asarray((100.0, 30.0))), (2, 2, 2)),
            jnp.asarray((((5.0,),),)),
        )
    )
    rod_evaluation = phx.applications.solid_mechanics.evaluate_rod(
        rod, rod.initialize_state()
    )

    print("planar step successful:", bool(step.successful))
    print("prismatic coordinate:", float(coordinates.prismatic_position[0]))
    print("friction cone successful:", bool(friction.successful[0]))
    print("rod evaluation valid:", bool(rod_evaluation.valid))
    print("rod rest energy:", float(rod_evaluation.potential_energy))


if __name__ == "__main__":
    main()
