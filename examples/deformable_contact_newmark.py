#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Implicit Newmark step for one elastic triangle approaching a static segment."""

import jax.numpy as jnp

import phydrax as phx


def main():
    coordinates = jnp.asarray(
        ((-0.25, 0.08), (0.25, 0.08), (0.0, 0.48)), dtype=jnp.float64
    )
    cells = jnp.asarray(((0, 1, 2),), dtype=jnp.int32)
    mesh = phx.discretization.CellMesh.from_triangles(coordinates, cells)
    discretization = phx.discretization.FiniteElementPlan(
        mesh,
        phx.discretization.FiniteElementFieldSpec(
            "u",
            phx.discretization.lagrange_element("triangle", 1),
            component_shape=(2,),
        ),
    ).prepare()
    elastic = phx.equations.CellEnergyAction(
        "u",
        lambda values, gradients, points, context: (
            20.0 * 0.5 * jnp.sum(gradients * gradients, axis=(-1, -2))
        ),
        action_id="elastic-triangle-energy",
    )
    compiled = phx.equations.compile_finite_element_problem(
        phx.equations.FiniteElementForm("contacting-triangle", "u", (elastic,)),
        discretization,
    )

    dynamic_surface = phx.discretization.prepare_cell_mesh_collision_surface(
        mesh,
        compiled.state_space,
        body_id=0,
    )
    static_plan = phx.discretization.CollisionSurfacePlan(
        jnp.asarray((10, 11), dtype=jnp.int64),
        ambient_dimension=2,
        edges=jnp.asarray(((0, 1),), dtype=jnp.int32),
        pair_policy=phx.discretization.ContactPairPolicy(
            2,
            body_ids=jnp.ones((2,), dtype=jnp.int64),
            static_mask=jnp.ones((2,), dtype=bool),
        ),
    )
    static_surface = phx.discretization.PreparedCollisionSurface(
        static_plan,
        jnp.asarray(((-1.0, 0.0), (1.0, 0.0)), dtype=jnp.float64),
        phx.discretization.static_collision_operator(
            compiled.state_space,
            2,
            2,
            dtype=jnp.float64,
        ),
    )
    scene = phx.discretization.PreparedCollisionScene((dynamic_surface, static_surface))
    search = phx.discretization.SweepAndPruneContactSearchPlan(
        edge_vertex_capacity=16,
        edge_edge_capacity=0,
        face_vertex_capacity=0,
        activation_distance=0.1,
    )
    contact = phx.applications.contact.ConvergentContactPotentialPlan(0.1, 50.0).prepare(
        scene
    )
    ccd = phx.discretization.InclusionCCDPlan()
    inversion = phx.discretization.SimplexInversionStepPlan(cells, coordinates)
    mechanics = phx.applications.solid_mechanics.FiniteElementDynamicsState(
        jnp.zeros_like(coordinates),
        jnp.broadcast_to(jnp.asarray((0.0, -0.2)), coordinates.shape),
        jnp.zeros_like(coordinates),
    )
    accepted = phx.applications.contact.ContactDynamicsState(mechanics)
    plan = phx.applications.contact.prepare_finite_element_contact_dynamics(
        compiled,
        accepted,
        scene,
        contact,
        search,
        ccd,
        inversion=inversion,
    )
    result = phx.applications.contact.solve_finite_element_contact_step(
        phx.applications.contact.prepare_finite_element_contact_step(plan, accepted, 0.05)
    )
    print(
        {
            "accepted": bool(result.accepted),
            "minimum_gap": float(result.contact.minimum_gap),
            "iterations": int(result.diagnostics.iterations),
            "rejection_reasons": int(result.rejection_reasons),
        }
    )


if __name__ == "__main__":
    main()
