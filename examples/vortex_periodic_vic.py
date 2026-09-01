#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


count = 16
domain = (
    phx.discretization.AxisDomain.periodic(0.0, 1.0),
    phx.discretization.AxisDomain.periodic(0.0, 1.0),
)
grid = phx.discretization.TensorGridPlan(
    (
        phx.discretization.UniformAxisSpec(count, periodic=True, endpoint=False),
        phx.discretization.UniformAxisSpec(count, periodic=True, endpoint=False),
    ),
    axis_names=("x", "y"),
).prepare(jnp.asarray(((0.0, 0.0), (1.0, 1.0))))
spectral = phx.discretization.TensorSpectralPlan(
    (
        phx.discretization.FourierBasisPlan(count),
        phx.discretization.FourierBasisPlan(count),
    ),
    axis_names=("x", "y"),
).prepare(domain)
position = jnp.asarray(((0.25, 0.5), (0.75, 0.5)))
circulation = jnp.asarray((1.0, -1.0))
particles = phx.discretization.ParticleSetPlan(
    jnp.asarray((1, 2)),
    jnp.ones((2,)),
    ambient_dimension=2,
).prepare()
properties = phx.discretization.VortexParticleProperties(
    jnp.full((2,), 0.08),
    jnp.full((2,), 0.5),
)
velocity = phx.operators.PeriodicVortexInCellPlan(
    particles,
    grid,
    spectral,
    phx.discretization.TensorBSplineSplatAssignment(2),
)
method = phx.discretization.VortexParticleMethodPlan(velocity)
compiled = phx.equations.compile_vortex_particle_flow(
    phx.equations.VortexParticleFlowProblem("periodic-vortex-dipole", 2),
    particles,
    properties,
    method,
)
state = compiled.initialize_state(position, circulation)
evaluation = compiled.dynamics.evaluate(0.0, state)[5]
backend = evaluation.diagnostics.backend_diagnostics

print("velocity", evaluation.velocity)
print("circulation residual", backend.compatibility_residual)
print("deposit balance", backend.balance_defect)
print("divergence norm", backend.divergence_norm)
print("successful", bool(evaluation.successful))
