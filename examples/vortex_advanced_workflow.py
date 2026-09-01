#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


position = jnp.asarray(((-0.5, -0.2), (-0.1, 0.3), (0.4, -0.25), (0.7, 0.2)))
circulation = jnp.asarray((0.6, -0.2, 0.4, -0.8))
core = jnp.full((4,), 0.1)
probe_position = jnp.asarray(((2.0, 1.5), (-2.0, -1.5)))
source = phx.discretization.VortexSourceState(
    position,
    circulation,
    core_radius=core,
)
targets = phx.discretization.VortexTargetState(probe_position)
accelerated = phx.operators.FixedClusterVortexPlan2D(
    position,
    leaf_size=2,
    opening_angle=0.6,
).evaluate(source, targets)

count = 8
coordinates = jnp.arange(count) / count
xx, yy = jnp.meshgrid(coordinates, coordinates, indexing="ij")
vorticity = jnp.sin(2.0 * jnp.pi * xx) * jnp.cos(2.0 * jnp.pi * yy)
reconstruction = phx.applications.vortex_flow.PeriodicVorticityReconstructionPlan(
    (count, count),
    (1.0, 1.0),
).reconstruct(vorticity)

print("accelerated velocity", accelerated.velocity)
print("tree error bound", accelerated.diagnostics.backend_diagnostics.truncation_bound)
print("reconstructed velocity norm", jnp.linalg.norm(reconstruction.velocity))
print("successful", bool(accelerated.successful & reconstruction.successful))
