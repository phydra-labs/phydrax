#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


position = jnp.asarray(((-0.5, 0.0, 0.0), (0.5, 0.0, 0.0)))
strength = jnp.asarray(((0.0, 1.0, 0.2), (0.0, -1.0, 0.2)))
particles = phx.discretization.ParticleSetPlan(
    jnp.asarray((11, 22)),
    jnp.ones((2,)),
    ambient_dimension=3,
).prepare()
properties = phx.discretization.VortexParticleProperties(
    jnp.full((2,), 0.2),
    jnp.ones((2,)),
)
velocity = phx.operators.GaussianErfDirectVortexPlan3D(
    maximum_sources=2,
    maximum_targets=2,
    source_chunk_size=2,
    target_chunk_size=2,
    maximum_interactions=4,
)
method = phx.discretization.VortexParticleMethodPlan(velocity)
compiled = phx.equations.compile_vortex_particle_flow(
    phx.equations.VortexParticleFlowProblem("three-dimensional-vortex-stretching", 3),
    particles,
    properties,
    method,
)
state = compiled.initialize_state(position, strength)
evaluation = compiled.dynamics.evaluate(0.0, state)
diagnostics = compiled.dynamics.diagnostics(0.0, state)

print("velocity", evaluation[2])
print("strength rate", evaluation[3])
print("total vector strength", diagnostics.total_strength)
print("finite", bool(diagnostics.finite))
