#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp

import phydrax as phx


initial_position = jnp.asarray(((-0.5, 0.0), (0.5, 0.0)))
initial_circulation = jnp.asarray((1.0, 1.0))
particles = phx.discretization.ParticleSetPlan(
    jnp.asarray((10, 20)),
    jnp.ones((2,)),
    ambient_dimension=2,
    name="vortex-pair-particles",
).prepare()
properties = phx.discretization.VortexParticleProperties(
    jnp.full((2,), 0.1),
    jnp.ones((2,)),
)
velocity = phx.operators.GaussianDirectVortexPlan2D(
    maximum_sources=2,
    maximum_targets=2,
    source_chunk_size=2,
    target_chunk_size=2,
)
method = phx.discretization.VortexParticleMethodPlan(velocity)
compiled = phx.equations.compile_vortex_particle_flow(
    phx.equations.VortexParticleFlowProblem("corotating-vortex-pair", 2),
    particles,
    properties,
    method,
)
problem = compiled.as_differential_problem(
    initial_position,
    initial_circulation,
    t0=0.0,
    t1=0.01,
)
solution = phx.solver.solve_diffrax(
    problem,
    save_times=jnp.asarray((0.0, 0.01)),
    solver=phx.solver.SSPRK33(),
    dt0=1.0e-3,
    max_steps=32,
)
final = compiled.dynamics.state_layout.unpack(solution.states[-1])
diagnostics = compiled.dynamics.diagnostics(solution.times[-1], solution.states[-1])
loss = lambda positions: jnp.sum(
    compiled.dynamics.initialize_state(positions, initial_circulation) ** 2
)
gradient = jax.grad(loss)(initial_position)

print("solver", solution.resolved_method)
print("final positions", final.position)
print("total circulation", diagnostics.total_strength)
print("impulse", diagnostics.linear_impulse)
print("gradient norm", jnp.linalg.norm(gradient))
print("successful", bool(solution.backend_successful & diagnostics.finite))
