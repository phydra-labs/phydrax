#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Measured rigid, contact, and rod mechanics microbenchmarks."""

import json
import time

import equinox as eqx
import jax
import jax.numpy as jnp

import phydrax as phx


def _timed(function, *arguments, repeats=16):
    compiled = eqx.filter_jit(function)
    start = time.perf_counter()
    value = compiled(*arguments)
    jax.block_until_ready(value)
    compile_seconds = time.perf_counter() - start
    start = time.perf_counter()
    for _ in range(repeats):
        value = compiled(*arguments)
    jax.block_until_ready(value)
    elapsed = time.perf_counter() - start
    return value, compile_seconds, elapsed / repeats


def _rigid_case(body_count=8):
    body_ids = jnp.arange(body_count, dtype=jnp.int64)
    particles = phx.discretization.ParticleSetPlan(
        body_ids,
        jnp.ones((body_count,)),
        ambient_dimension=2,
    ).prepare()
    bodies = phx.discretization.RigidBodySetPlan(
        jnp.zeros((body_count,), dtype=jnp.int32),
        jnp.ones((body_count,)),
        fixed_mask=jnp.arange(body_count) == 0,
    ).prepare(particles)
    positions = jnp.stack(
        (jnp.arange(body_count, dtype=float), jnp.zeros((body_count,))), axis=-1
    )
    reference = bodies.kinematics(
        positions,
        jnp.zeros_like(positions),
        jnp.zeros((body_count, 1)),
        jnp.zeros((body_count, 1)),
    )
    edges = jnp.arange(body_count - 1)
    graph = phx.discretization.RigidJointGraphPlan(
        ball=phx.discretization.BallJointSetPlan(
            100 + edges,
            body_ids[:-1],
            body_ids[1:],
            jnp.stack(
                (edges.astype(float) + 0.5, jnp.zeros_like(edges, dtype=float)),
                axis=-1,
            ),
        )
    )
    dynamics = phx.discretization.RigidConstraintDynamicsPlan(graph).prepare(
        bodies, reference
    )
    state = dynamics.initialize_state(
        reference.position,
        reference.velocity,
        reference.orientation,
        reference.angular_velocity,
    )
    return dynamics, state


def _rod_case(node_count=17):
    positions = jnp.stack(
        (jnp.arange(node_count, dtype=float), jnp.zeros((node_count,))), axis=-1
    )
    segments = jnp.stack((jnp.arange(node_count - 1), jnp.arange(1, node_count)), axis=-1)
    rod = phx.applications.solid_mechanics.prepare_rod(
        phx.applications.solid_mechanics.RodPlan(
            segments,
            positions,
            jnp.broadcast_to(jnp.eye(2), (node_count - 1, 2, 2)),
            jnp.ones((node_count,)),
            jnp.ones((node_count - 1,)) * 0.2,
            jnp.broadcast_to(
                jnp.diag(jnp.asarray((100.0, 30.0))),
                (node_count - 1, 2, 2),
            ),
            jnp.broadcast_to(jnp.asarray([[[5.0]]]), (node_count - 2, 1, 1)),
        )
    )
    return rod, rod.initialize_state()


def main():
    dynamics, rigid_state = _rigid_case()
    rigid_result, rigid_compile, rigid_step = _timed(
        lambda state: dynamics.step(state, jnp.asarray(0.0), jnp.asarray(1.0e-3)),
        rigid_state,
        repeats=8,
    )

    normal = jnp.ones((1024,))
    tangent = jnp.broadcast_to(jnp.asarray([0.8, -0.3]), (1024, 2))
    friction, friction_compile, friction_step = _timed(
        lambda n, t: phx.discretization.project_isotropic_coulomb_impulse(
            n, t, jnp.asarray(0.5)
        ),
        normal,
        tangent,
    )

    rod, rod_state = _rod_case()
    rod_evaluation, rod_compile, rod_step = _timed(
        lambda state: phx.applications.solid_mechanics.evaluate_rod(rod, state),
        rod_state,
    )

    print(
        json.dumps(
            {
                "benchmark": "mechanics-expansion",
                "device": str(jax.devices()[0]),
                "dtype": str(jnp.ones(()).dtype),
                "rigid": {
                    "body_count": 8,
                    "constraint_rows": dynamics.joints.constraint_count,
                    "compile_seconds": rigid_compile,
                    "steady_seconds": rigid_step,
                    "successful": bool(rigid_result.successful),
                    "position_residual": float(
                        rigid_result.evaluation.diagnostics.maximum_position_residual
                    ),
                    "velocity_residual": float(
                        rigid_result.evaluation.diagnostics.maximum_velocity_residual
                    ),
                },
                "friction": {
                    "contact_count": 1024,
                    "compile_seconds": friction_compile,
                    "steady_seconds": friction_step,
                    "successful": bool(jnp.all(friction.successful)),
                    "minimum_branch_margin": float(jnp.min(friction.branch_margin)),
                },
                "rod": {
                    "node_count": 17,
                    "compile_seconds": rod_compile,
                    "steady_seconds": rod_step,
                    "valid": bool(rod_evaluation.valid),
                    "potential_energy": float(rod_evaluation.potential_energy),
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
