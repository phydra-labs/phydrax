#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Measured end-to-end finite-element barrier-contact step."""

import json
import time

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _case():
    coordinates = jnp.asarray(((-0.25, 0.08), (0.25, 0.08), (0.0, 0.48)))
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
    functional = phx.variational.Functional(
        "benchmark-contact",
        (
            phx.variational.LocalIntegralTerm(
                "benchmark-elasticity",
                region="body",
                fields=(phx.variational.FieldJetSpec("u", gradient=True),),
                density=lambda fields, geometry, context: (
                    10.0 * jnp.sum(fields["u"].gradient ** 2, axis=(-1, -2))
                ),
                density_id="benchmark-elasticity",
            ),
        ),
        variable_fields=("u",),
    )
    compiled = phx.equations.compile_finite_element_functional(
        functional,
        discretization,
        fields={"u": "u"},
        regions={"body": None},
    )
    moving = phx.discretization.prepare_cell_mesh_collision_surface(
        mesh, compiled.state_space
    )
    static_plan = phx.discretization.CollisionSurfacePlan(
        jnp.asarray((10, 11)),
        ambient_dimension=2,
        edges=jnp.asarray(((0, 1),)),
        pair_policy=phx.discretization.ContactPairPolicy(
            2,
            body_ids=jnp.ones((2,), dtype=jnp.int64),
            static_mask=jnp.ones((2,), dtype=bool),
        ),
    )
    static = phx.discretization.PreparedCollisionSurface(
        static_plan,
        jnp.asarray(((-1.0, 0.0), (1.0, 0.0))),
        phx.discretization.static_collision_operator(
            compiled.state_space, 2, 2, dtype=np.float64
        ),
    )
    scene = phx.discretization.PreparedCollisionScene((moving, static))
    search = phx.discretization.SweepAndPruneContactSearchPlan(
        edge_vertex_capacity=24,
        edge_edge_capacity=0,
        face_vertex_capacity=0,
        activation_distance=0.1,
    )
    contact = phx.applications.contact.ConvergentContactPotentialPlan(0.1, 20.0).prepare(
        scene
    )
    mechanics = phx.applications.solid_mechanics.FiniteElementDynamicsState(
        jnp.zeros_like(coordinates),
        jnp.broadcast_to(jnp.asarray((0.0, -0.05)), coordinates.shape),
        jnp.zeros_like(coordinates),
    )
    accepted = phx.applications.contact.ContactDynamicsState(mechanics)
    plan = phx.applications.contact.prepare_finite_element_contact_dynamics(
        compiled,
        accepted,
        scene,
        contact,
        search,
        phx.discretization.InclusionCCDPlan(time_tolerance=1.0e-7),
        inversion=phx.discretization.SimplexInversionStepPlan(cells, coordinates),
    )
    return plan, accepted


def main():
    plan, accepted = _case()
    prepared = phx.applications.contact.prepare_finite_element_contact_step(
        plan, accepted, 0.02
    )
    started = time.perf_counter()
    result = phx.applications.contact.solve_finite_element_contact_step(prepared)
    elapsed = time.perf_counter() - started
    print(
        json.dumps(
            {
                "benchmark": "contact-dynamics",
                "device": str(jax.devices()[0]),
                "dtype": str(accepted.mechanics.displacement.dtype),
                "elapsed_seconds": elapsed,
                "accepted": bool(result.accepted),
                "iterations": int(result.diagnostics.iterations),
                "objective_evaluations": int(result.diagnostics.objective_evaluations),
                "linear_iterations": int(result.diagnostics.linear_iterations),
                "active_contacts": int(result.contact.active_contacts),
                "minimum_gap": float(result.contact.minimum_gap),
                "rejection_reasons": int(result.rejection_reasons),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
