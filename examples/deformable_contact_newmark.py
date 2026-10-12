#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Implicit Newmark step for one elastic triangle approaching a static segment."""

import jax.numpy as jnp

import phydrax as phx


def main() -> None:
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

    participant = phx.applications.contact.prepare_finite_element_contact_participant(
        compiled,
        inversion=phx.discretization.SimplexInversionStepPlan(cells, coordinates),
    )
    static_plan = phx.discretization.CollisionSurfacePlan(
        jnp.asarray((10, 11), dtype=jnp.int64),
        ambient_dimension=2,
        edges=jnp.asarray(((0, 1),), dtype=jnp.int32),
        body_ids=1,
        static_mask=True,
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
    scene = phx.discretization.ContactParticipantScene(
        (
            participant.geometry,
            phx.discretization.LinearContactParticipant(static_surface),
        )
    )
    search = phx.discretization.SweepAndPruneContactSearchPlan(
        edge_vertex_capacity=16,
        edge_edge_capacity=0,
        face_vertex_capacity=0,
        activation_distance=0.1,
        envelope_radius=0.5,
    )
    contact = phx.applications.contact.ConvergentContactPotentialPlan(0.1, 50.0).prepare(
        scene,
        rest_positions=jnp.concatenate((coordinates, static_surface.rest_positions)),
    )
    ccd = phx.discretization.InclusionCCDPlan()
    mechanics = phx.applications.solid_mechanics.FiniteElementDynamicsState(
        jnp.zeros_like(coordinates),
        jnp.broadcast_to(jnp.asarray((0.0, -0.2)), coordinates.shape),
        jnp.zeros_like(coordinates),
    )
    plan = phx.applications.contact.MonolithicContactPlan(
        (participant,),
        scene,
        contact,
        search,
        ccd,
        geometry_owner_indices=(0, None),
        fixed_geometry_states=(None, jnp.zeros_like(coordinates)),
    )
    accepted = phx.applications.contact.initialize_monolithic_contact_state(
        plan, (mechanics.displacement,), (mechanics,)
    )
    result = phx.applications.contact.solve_monolithic_contact_step(
        phx.applications.contact.prepare_monolithic_contact_step(plan, accepted, 0.05)
    )
    if not bool(result.accepted):
        raise RuntimeError("Deformable-contact Newmark step was rejected")
    print(
        {
            "accepted": True,
            "minimum_gap": float(result.contact.minimum_gap),
            "iterations": int(result.diagnostics.iterations),
            "rejection_reasons": int(result.rejection_reasons),
        }
    )


if __name__ == "__main__":
    main()
