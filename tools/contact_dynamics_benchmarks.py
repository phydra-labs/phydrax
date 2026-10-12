#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Measured end-to-end finite-element barrier-contact step."""

import json
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _case() -> Any:
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
    static = phx.discretization.PreparedCollisionSurface(
        static_plan,
        jnp.asarray(((-1.0, 0.0), (1.0, 0.0)), dtype=jnp.float64),
        phx.discretization.static_collision_operator(
            compiled.state_space, 2, 2, dtype=np.float64
        ),
    )
    scene = phx.discretization.ContactParticipantScene(
        (participant.geometry, phx.discretization.LinearContactParticipant(static))
    )
    search = phx.discretization.SweepAndPruneContactSearchPlan(
        edge_vertex_capacity=24,
        edge_edge_capacity=0,
        face_vertex_capacity=0,
        activation_distance=0.1,
        envelope_radius=0.5,
    )
    contact = phx.applications.contact.ConvergentContactPotentialPlan(0.1, 20.0).prepare(
        scene, rest_positions=jnp.concatenate((coordinates, static.rest_positions))
    )
    mechanics = phx.applications.solid_mechanics.FiniteElementDynamicsState(
        jnp.zeros_like(coordinates),
        jnp.broadcast_to(jnp.asarray((0.0, -0.05)), coordinates.shape),
        jnp.zeros_like(coordinates),
    )
    plan = phx.applications.contact.MonolithicContactPlan(
        (participant,),
        scene,
        contact,
        search,
        phx.discretization.InclusionCCDPlan(time_tolerance=1.0e-7),
        geometry_owner_indices=(0, None),
        fixed_geometry_states=(None, jnp.zeros_like(coordinates)),
    )
    accepted = phx.applications.contact.initialize_monolithic_contact_state(
        plan, (mechanics.displacement,), (mechanics,)
    )
    return plan, accepted


def main() -> None:
    plan, accepted = _case()
    prepared = phx.applications.contact.prepare_monolithic_contact_step(
        plan, accepted, 0.02
    )
    started = time.perf_counter()
    result = phx.applications.contact.solve_monolithic_contact_step(prepared)
    jax.block_until_ready(result.accepted)
    elapsed = time.perf_counter() - started
    print(
        json.dumps(
            {
                "benchmark": "contact-dynamics",
                "device": str(jax.devices()[0]),
                "dtype": str(accepted.coordinates[0].dtype),
                "elapsed_seconds": elapsed,
                "accepted": bool(result.accepted),
                "iterations": int(result.diagnostics.iterations),
                "residual_evaluations": int(result.diagnostics.residual_evaluations),
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
