#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Public-API bounded tetrahedral affine-body/fixed-wall contact example.

Run: JAX_ENABLE_X64=1 python -m examples.monolithic_contact
"""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np

from phydrax.applications.contact import (
    AffineBodyHistory,
    AffineBodyPlan,
    ContactSolvePolicy,
    ConvergentContactPotentialPlan,
    initialize_monolithic_contact_state,
    LaggedCoulombFrictionPlan,
    MonolithicContactPlan,
    MonolithicContactState,
    prepare_affine_body_participant,
    prepare_monolithic_contact_step,
    solve_monolithic_contact_step,
)
from phydrax.discretization.contact import (
    CollisionSurfacePlan,
    ContactParticipantScene,
    DenseContactSearchPlan,
    InclusionCCDPlan,
    LinearContactParticipant,
    PreparedCollisionSurface,
    static_collision_operator,
)
from phydrax.linalg import ArraySpace


def make_case(
    body_count: int = 1, capacity_scale: int = 1, *, friction: bool = False
) -> tuple[MonolithicContactPlan, MonolithicContactState]:
    """Prepare separated tetrahedra over a finite wall; capacities are explicit."""
    if not jax.config.x64_enabled:
        raise ValueError("This example requires JAX_ENABLE_X64=1.")
    if body_count < 1 or capacity_scale < 1:
        raise ValueError("Body count and capacity scale must be positive.")
    local = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    controls = np.stack(
        [
            0.1 * local + np.asarray((0.3 * b, 0.0, 0.025), dtype=np.float64)
            for b in range(body_count)
        ]
    )
    faces = np.asarray(((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)), dtype=np.int32)
    topology = CollisionSurfacePlan(
        np.arange(4 * body_count, dtype=np.int64),
        ambient_dimension=3,
        faces=np.concatenate([faces + 4 * b for b in range(body_count)]),
        body_ids=np.repeat(np.arange(body_count, dtype=np.int32), 4),
        solver_clearance=0.001,
    )
    # Symmetric degree-two tetrahedral quadrature, weights in reference volume.
    a = (5.0 + 3.0 * np.sqrt(5.0)) / 20.0
    b = (5.0 - np.sqrt(5.0)) / 20.0
    shape = np.full((4, 4), b, dtype=np.float64)
    np.fill_diagonal(shape, a)
    body = AffineBodyPlan(
        topology,
        controls,
        np.tile(local, (body_count, 1)),
        np.repeat(np.arange(body_count, dtype=np.int32), 4),
        np.broadcast_to(shape, (body_count, 4, 4)),
        np.full((body_count, 4), 0.1**3 / 24.0, dtype=np.float64),
        np.full((body_count,), 1000.0, dtype=np.float64),
        np.full((body_count,), 1.0e5, dtype=np.float64),
    )
    participant = prepare_affine_body_participant(body)
    wall_topology = CollisionSurfacePlan(
        np.arange(4 * body_count + 100, 4 * body_count + 104, dtype=np.int64),
        ambient_dimension=3,
        faces=np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int32),
        body_ids=body_count,
        static_mask=True,
    )
    wall_space = ArraySpace((4, 3), dtype=np.float64)
    wall = LinearContactParticipant(
        PreparedCollisionSurface(
            wall_topology,
            np.asarray(
                (
                    (-1.0, -1.0, 0.0),
                    (0.3 * body_count + 1.0, -1.0, 0.0),
                    (0.3 * body_count + 1.0, 1.0, 0.0),
                    (-1.0, 1.0, 0.0),
                ),
                dtype=np.float64,
            ),
            static_collision_operator(wall_space, 4, 3, dtype=np.float64),
        )
    )
    scene = ContactParticipantScene((participant.geometry, wall))
    coordinates = jnp.zeros((body_count, 4, 3), dtype=jnp.float64)
    fixed = jnp.zeros((4, 3), dtype=jnp.float64)
    reference = scene.positions((coordinates, fixed))
    contact = ConvergentContactPotentialPlan(
        0.03, jnp.asarray(1.0e4, dtype=jnp.float64)
    ).prepare(scene, rest_positions=reference)
    vertex_count = 4 * body_count + 4
    search = DenseContactSearchPlan(
        edge_vertex_capacity=capacity_scale * scene.edge_count * vertex_count,
        edge_edge_capacity=capacity_scale * scene.edge_count**2,
        face_vertex_capacity=capacity_scale * scene.face_count * vertex_count,
        activation_distance=0.03,
        envelope_radius=0.02,
    )
    friction_plan = (
        LaggedCoulombFrictionPlan(jnp.asarray(0.2, dtype=jnp.float64), 1.0e-4).prepare(
            scene, contact
        )
        if friction
        else None
    )
    plan = MonolithicContactPlan(
        (participant,),
        scene,
        contact,
        search,
        InclusionCCDPlan(maximum_iterations=256),
        geometry_owner_indices=(0, None),
        fixed_geometry_states=(None, fixed),
        friction=friction_plan,
        policy=ContactSolvePolicy(),
    )
    history = AffineBodyHistory(jnp.zeros_like(coordinates), jnp.zeros_like(coordinates))
    return plan, initialize_monolithic_contact_state(plan, (coordinates,), (history,))


def main() -> None:
    plan, accepted = make_case()
    result = solve_monolithic_contact_step(
        prepare_monolithic_contact_step(plan, accepted, 0.001)
    )
    committed = result.commit()
    print(
        json.dumps(
            {
                "accepted": bool(result.accepted),
                "rejection_reasons": int(result.rejection_reasons),
                "residual_norm": float(result.diagnostics.final_residual_norm),
                "minimum_gap": float(result.diagnostics.minimum_gap),
                "candidate_count": int(result.diagnostics.candidate_count),
                "ccd_successful": bool(result.safety.successful),
                "ccd_conservative": bool(result.safety.conservative),
                "ccd_step_size": float(result.safety.step_size),
                "overflow_count": int(result.diagnostics.overflow_count),
                "state_version": int(committed.state_version),
                "rollback_applied": bool(result.rollback_applied),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not bool(result.accepted):
        raise RuntimeError("Bounded contact step rejected; inspect printed evidence.")


if __name__ == "__main__":
    main()
