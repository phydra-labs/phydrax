#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


position = jnp.asarray(((-0.3, 0.0), (0.0, 0.0), (0.3, 0.0)))
circulation = jnp.asarray((0.2, 1.0, 0.2))
particles = phx.discretization.ParticleSetPlan(
    jnp.arange(3),
    jnp.ones((3,)),
    ambient_dimension=2,
).prepare()
properties = phx.discretization.VortexParticleProperties(
    jnp.full((3,), 0.15),
    jnp.full((3,), 0.3),
)
velocity = phx.operators.GaussianDirectVortexPlan2D(
    maximum_sources=3,
    source_chunk_size=3,
    target_chunk_size=3,
)
diffusion = phx.operators.GaussianParticleStrengthExchangePlan(
    2,
    0.4,
    active_mask=particles.active_mask,
)
method = phx.discretization.VortexParticleMethodPlan(
    velocity,
    diffusion=diffusion,
)
compiled = phx.equations.compile_vortex_particle_flow(
    phx.equations.VortexParticleFlowProblem(
        "viscous-vortex-particles",
        2,
        0.01,
    ),
    particles,
    properties,
    method,
)
state = compiled.initialize_state(position, circulation)
evaluation = compiled.dynamics.evaluate(0.0, state)
diffusion_evaluation = evaluation[5]
backend = diffusion_evaluation.diagnostics.backend_diagnostics

print("strength rate", diffusion_evaluation.rate)
print("total rate", diffusion_evaluation.diagnostics.total_rate)
print("stable step", backend.stable_step)
print("successful", bool(diffusion_evaluation.successful))
