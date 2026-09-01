#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


angle = jnp.linspace(0.0, 2.0 * jnp.pi, 33)
vertices = jnp.stack((jnp.cos(angle), jnp.sin(angle)), axis=-1)
geometry = phx.operators.FlowPanelGeometry2D.from_vertices(vertices)
plan = phx.solver.VortexPanelFlowPlan2D(geometry, prescribed_circulation=0.0)
result = plan.solve(jnp.asarray((1.0, 0.0)))
motion = phx.operators.RigidPanelMotion2D(
    jnp.asarray(0.1),
    jnp.asarray((0.2, -0.1)),
    jnp.asarray((0.05, 0.0)),
    jnp.asarray(0.02),
)
moving_result = plan.solve(jnp.asarray((1.0, 0.0)), motion=motion)
wall = phx.discretization.BoundarySheetParticleTransferPlan2D(64, 0.08, 0.1)
wall_result = wall.transfer(wall.initialize(dtype=float), geometry, result.sheet_strength)

print("boundary residual", result.boundary_residual_norm)
print("constraint residual", result.constraint_residual)
print("force", result.total_force)
print("moving force", moving_result.total_force)
print("wall circulation residual", wall_result.circulation_residual)
print(
    "successful",
    bool(result.successful & moving_result.successful & wall_result.successful),
)
