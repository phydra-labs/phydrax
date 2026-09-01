#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Measured contact-closure lowering, cone solve, and rough-contact kernels."""

import json
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _timed(function, argument, repeats):
    compiled = eqx.filter_jit(function)
    started = time.perf_counter()
    result = compiled(argument)
    jax.block_until_ready(result)
    compile_seconds = time.perf_counter() - started
    started = time.perf_counter()
    for _ in range(repeats):
        result = compiled(argument)
    jax.block_until_ready(result)
    return result, compile_seconds, (time.perf_counter() - started) / repeats


def _closure_case(segment_count=128):
    left = jnp.arange(segment_count, dtype=jnp.float64)
    right = left + 0.2
    x = jnp.stack((left, right), axis=-1).reshape((-1,))
    moving_positions = jnp.stack((x, jnp.full_like(x, 0.05)), axis=-1)
    source = phx.linalg.ArraySpace(moving_positions.shape, dtype=np.float64)
    moving_edges = jnp.arange(2 * segment_count, dtype=jnp.int32).reshape((-1, 2))
    moving_plan = phx.discretization.CollisionSurfacePlan(
        jnp.arange(2 * segment_count, dtype=jnp.int64),
        ambient_dimension=2,
        edges=moving_edges,
    )
    moving = phx.discretization.PreparedCollisionSurface(
        moving_plan,
        moving_positions,
        phx.discretization.selection_collision_operator(
            source, jnp.arange(2 * segment_count, dtype=jnp.int32)
        ),
    )
    static_plan = phx.discretization.CollisionSurfacePlan(
        jnp.asarray((10_000, 10_001), dtype=jnp.int64),
        ambient_dimension=2,
        edges=jnp.asarray(((0, 1),), dtype=jnp.int32),
        pair_policy=phx.discretization.ContactPairPolicy(
            2,
            body_ids=jnp.ones((2,), dtype=jnp.int64),
            material_ids=jnp.zeros((2,), dtype=jnp.int64),
            static_mask=jnp.ones((2,), dtype=bool),
        ),
    )
    static = phx.discretization.PreparedCollisionSurface(
        static_plan,
        jnp.asarray(((-1.0, 0.0), (segment_count + 1.0, 0.0))),
        phx.discretization.static_collision_operator(source, 2, 2, dtype=np.float64),
    )
    scene = phx.discretization.PreparedCollisionScene((moving, static))
    positions = scene.positions(source.zeros())
    velocities = scene.map_values(
        jnp.broadcast_to(jnp.asarray((0.2, -0.1)), source.shape)
    )
    search = phx.discretization.SweepAndPruneContactSearchPlan(
        edge_vertex_capacity=512,
        edge_edge_capacity=0,
        face_vertex_capacity=0,
        activation_distance=0.1,
    )
    epoch = search.build(scene, positions)
    kinematics = phx.discretization.evaluate_contact_kinematics(
        scene,
        epoch,
        positions,
        velocities,
        0.01,
        activation_distance=0.1,
    )
    materials = phx.applications.contact.ContactMaterialPairTable.uniform(
        normal_stiffness=100.0,
        static_friction=0.5,
        dynamic_friction=0.4,
    )
    closure = phx.applications.contact.ContactClosurePlan(
        phx.applications.contact.BarrierNormalContactLaw(0.1),
        materials,
        tangential=phx.applications.contact.RegularizedCoulombContactLaw(1.0e-3),
    )
    state = phx.applications.contact.remap_contact_route_state(
        phx.applications.contact.ContactRouteState.empty(0, 1, closure.closure_id),
        kinematics,
    ).candidate
    return kinematics, materials, closure, state


def main():
    kinematics, materials, closure, state = _closure_case()
    closure_value, closure_compile, closure_seconds = _timed(
        lambda route_state: (
            phx.applications.contact.evaluate_contact_closure(
                closure, kinematics, route_state
            ).evidence.total_potential
        ),
        state,
        30,
    )
    count = state.capacity
    blocks = jnp.broadcast_to(jnp.eye(2), (count, 2, 2))
    cone_program = phx.applications.contact.build_contact_cone_program(
        kinematics, materials, blocks, compliance=1.0e-6
    )
    cone_result, cone_compile, cone_seconds = _timed(
        lambda free: (
            phx.applications.contact.solve_contact_cone(
                eqx.tree_at(
                    lambda program: program.free_velocity,
                    cone_program,
                    free,
                ),
                solver=phx.applications.contact.ContactConeSolverPlan(
                    maximum_iterations=100
                ),
            ).impulse
        ),
        cone_program.free_velocity,
        20,
    )
    sap_result, sap_compile, sap_seconds = _timed(
        lambda free: (
            phx.applications.contact.solve_contact_sap(
                eqx.tree_at(
                    lambda program: program.free_velocity,
                    cone_program,
                    free,
                ),
                solver=phx.applications.contact.SAPContactSolverPlan(
                    maximum_iterations=100,
                    tolerance=1.0e-8,
                ),
            ).impulse
        ),
        cone_program.free_velocity,
        10,
    )
    rough_plan = phx.applications.contact.PeriodicRoughContactPlan(
        jnp.ones((64, 64)), maximum_iterations=200, tolerance=1.0e-8
    )
    rough_result, rough_compile, rough_seconds = _timed(
        lambda gap: (
            phx.applications.contact.solve_periodic_rough_contact(
                rough_plan, gap
            ).pressure
        ),
        -0.01 * jnp.ones((64, 64)),
        20,
    )
    print(
        json.dumps(
            {
                "benchmark": "contact-closure",
                "device": str(jax.devices()[0]),
                "dtype": str(state.accumulated_slip.dtype),
                "route_capacity": count,
                "active_routes": int(kinematics.evidence.active_contacts),
                "closure_compile_seconds": closure_compile,
                "closure_seconds": closure_seconds,
                "closure_potential": float(closure_value),
                "cone_compile_seconds": cone_compile,
                "cone_seconds": cone_seconds,
                "cone_impulse_norm": float(jnp.sqrt(jnp.sum(cone_result * cone_result))),
                "sap_compile_seconds": sap_compile,
                "sap_seconds": sap_seconds,
                "sap_impulse_norm": float(jnp.sqrt(jnp.sum(sap_result * sap_result))),
                "rough_grid": list(rough_plan.shape),
                "rough_compile_seconds": rough_compile,
                "rough_seconds": rough_seconds,
                "rough_load": float(jnp.sum(rough_result)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
