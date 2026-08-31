#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Repeatable globally constrained rigid-chain benchmark."""

import json
import time

import equinox as eqx
import jax
import jax.numpy as jnp

import phydrax as phx


def _tree_bytes(tree):
    return sum(
        leaf.size * leaf.dtype.itemsize
        for leaf in jax.tree.leaves(tree)
        if eqx.is_array(leaf)
    )


def _case(body_count):
    body_ids = jnp.arange(1000, 1000 + body_count, dtype=jnp.int64)
    masses = jnp.ones((body_count,))
    particles = phx.discretization.ParticleSetPlan(
        body_ids, masses, ambient_dimension=3
    ).prepare()
    bodies = phx.discretization.RigidBodySetPlan(
        jnp.zeros((body_count,), dtype=jnp.int32),
        jnp.stack(tuple(jnp.eye(3) for _ in range(body_count))),
        fixed_mask=jnp.arange(body_count) == 0,
    ).prepare(particles)
    position = jnp.stack(
        (
            jnp.arange(body_count, dtype=float),
            jnp.zeros((body_count,)),
            jnp.zeros((body_count,)),
        ),
        axis=-1,
    )
    orientation = jnp.asarray([[1.0, 0.0, 0.0, 0.0]] * body_count)
    reference = bodies.kinematics(
        position,
        jnp.zeros_like(position),
        orientation,
        jnp.zeros_like(position),
    )
    edge = jnp.arange(body_count - 1)
    ball_edge = edge[::2]
    hinge_edge = edge[1::2]
    ball = phx.discretization.BallJointSetPlan(
        2000 + ball_edge,
        body_ids[ball_edge],
        body_ids[ball_edge + 1],
        jnp.stack(
            (
                ball_edge.astype(float) + 0.5,
                jnp.zeros_like(ball_edge, dtype=float),
                jnp.zeros_like(ball_edge, dtype=float),
            ),
            axis=-1,
        ),
    )
    hinge = phx.discretization.HingeJointSetPlan(
        3000 + hinge_edge,
        body_ids[hinge_edge],
        body_ids[hinge_edge + 1],
        jnp.stack(
            (
                hinge_edge.astype(float) + 0.5,
                jnp.zeros_like(hinge_edge, dtype=float),
                jnp.zeros_like(hinge_edge, dtype=float),
            ),
            axis=-1,
        ),
        jnp.broadcast_to(jnp.asarray([0.0, 0.0, 1.0]), (hinge_edge.size, 3)),
    )
    graph = phx.discretization.RigidJointGraphPlan(ball=ball, hinge=hinge)

    def gravity(time_, kinematics, args):
        del time_, args
        return phx.discretization.RigidBodyLoad(
            masses[:, None] * jnp.asarray([0.0, -9.81, 0.0]),
            jnp.zeros_like(kinematics.angular_velocity),
        )

    dynamics = phx.discretization.RigidConstraintDynamicsPlan(graph).prepare(
        bodies,
        reference,
        external_load=gravity,
        external_load_id="benchmark-gravity",
    )
    state = dynamics.initialize_state(
        reference.position,
        reference.velocity,
        reference.orientation,
        reference.angular_velocity,
    )
    return dynamics, state, ball.count, hinge.count


def _measure(body_count, repeats=8):
    dynamics, state, ball_count, hinge_count = _case(body_count)
    step_size = jnp.asarray(1.0e-3)

    @eqx.filter_jit
    def step(current, index):
        return dynamics.step(
            current,
            step_size * index,
            step_size,
        )

    compile_start = time.perf_counter()
    result = step(state, jnp.asarray(0))
    jax.block_until_ready(result.accepted_state.kinematics.position)
    compile_seconds = time.perf_counter() - compile_start
    state = result.accepted_state

    start = time.perf_counter()
    for index in range(1, repeats + 1):
        result = step(state, jnp.asarray(index))
        state = result.accepted_state
    jax.block_until_ready(state.kinematics.position)
    elapsed = time.perf_counter() - start
    position = result.evaluation.position_result
    velocity = result.evaluation.velocity_result
    return {
        "body_count": body_count,
        "mobile_coordinate_count": 6 * (body_count - 1),
        "ball_joint_count": ball_count,
        "hinge_joint_count": hinge_count,
        "constraint_row_count": 3 * ball_count + 5 * hinge_count,
        "state_bytes": _tree_bytes(state),
        "prepared_bytes": _tree_bytes(dynamics),
        "compile_seconds": compile_seconds,
        "steady_step_seconds": elapsed / repeats,
        "steps_per_second": repeats / elapsed,
        "successful": bool(result.successful),
        "position_iterations": int(position.diagnostics.iterations),
        "position_evaluations": int(position.diagnostics.residual_evaluations),
        "position_jvp_evaluations": int(position.diagnostics.jvp_evaluations),
        "velocity_iterations": int(velocity.diagnostics.iterations),
        "velocity_matvecs": int(velocity.diagnostics.matvec_count),
        "maximum_position_residual": float(
            result.evaluation.diagnostics.maximum_position_residual
        ),
        "maximum_velocity_residual": float(
            result.evaluation.diagnostics.maximum_velocity_residual
        ),
        "quaternion_defect": float(result.evaluation.diagnostics.quaternion_defect),
        "velocity_projection_energy_increase": float(
            result.evaluation.diagnostics.velocity_projection_energy_increase
        ),
        "constraint_rank": int(result.evaluation.diagnostics.constraint_rank),
        "constraint_condition": float(result.evaluation.diagnostics.constraint_condition),
        "prepared_id": dynamics.prepared_id,
        "solver_id": dynamics.solver.plan_id,
    }


def main():
    results = [_measure(count) for count in (4, 8, 16)]
    print(
        json.dumps(
            {
                "benchmark": "rigid-constraint-dynamics",
                "device": str(jax.devices()[0]),
                "dtype": str(jnp.ones(()).dtype),
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
