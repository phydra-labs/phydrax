#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


span = jnp.linspace(-2.0, 2.0, 7)
leading = jnp.stack((jnp.zeros_like(span), span, jnp.zeros_like(span)), axis=-1)
trailing = leading + jnp.asarray((1.0, 0.0, 0.0))
surface = phx.discretization.LiftingSurfacePlan(leading, trailing).prepare()
plan = phx.solver.SteadyVortexLatticePlan(
    surface,
    jnp.asarray((1.0, 0.0, 0.0)),
    wake_length=40.0,
    core_radius=0.02,
)
angle = jnp.deg2rad(5.0)
freestream = jnp.asarray((jnp.cos(angle), 0.0, jnp.sin(angle)))
result = plan.solve(freestream)
wake = phx.discretization.VortexWakePlan(48, surface.panel_count, 0.03)
unsteady = phx.solver.UnsteadyVortexLatticePlan(plan, wake)
unsteady_result = unsteady.step(unsteady.initialize(), freestream, 0.01)

print("circulation", result.circulation)
print("residual", result.residual_norm)
print("total force", result.total_force)
print("wake remaining", unsteady_result.wake_capacity_remaining)
print("successful", bool(result.successful & unsteady_result.successful))
