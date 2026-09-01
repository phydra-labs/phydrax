#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Heat one spherical particle through a conservative radial shell model."""

import jax.numpy as jnp

import phydrax as phx


particles = phx.discretization.ParticleSetPlan(
    jnp.asarray([0]), jnp.ones((1,)), ambient_dimension=3
).prepare()
schema = phx.equations.ChemicalSpeciesSchema(
    ("solid",),
    (phx.equations.ChemicalPhaseKind.SOLID,),
    jnp.asarray([0.01]),
    ("X",),
    jnp.asarray([[1]]),
    jnp.zeros_like(jnp.asarray([0.01]), dtype=jnp.int32),
)
thermodynamics = phx.equations.ParticleThermodynamicMaterialPlan(
    phx.equations.PolynomialSpeciesThermodynamicsPlan(
        schema, jnp.asarray([20.0]), jnp.asarray([0.0])
    )
)
material = phx.equations.ParticleThermochemicalMaterialBundle(
    thermodynamics,
    phx.equations.ParticleTransportMaterialPlan(
        schema, jnp.asarray([0.5]), jnp.asarray([0.0])
    ),
)
batch_plan = phx.discretization.ParticleInternalBatchPlan(
    jnp.asarray([0]),
    phx.discretization.RadialShellMeshPlan(
        phx.discretization.ParticleInternalGeometry.SPHERE, 6
    ),
    1,
)
batch = batch_plan.prepare(particles)
species = jnp.ones((1, 6, 1))
initial = phx.discretization.initialize_particle_internal_batch(
    batch,
    thermodynamics.energy_from_temperature(jnp.full((1, 6), 300.0), species),
    species,
    jnp.full((1, 6), 0.2),
    jnp.ones((1, 6)),
    jnp.asarray([0.01]),
)
compiled = phx.equations.compile_particle_conversion_problem(
    phx.equations.ParticleConversionProblemIR("internal-heating", (material,)),
    particles,
    (batch_plan,),
)
state = compiled.initialize_state((initial,))
boundary = phx.equations.ParticleTransportBoundary(
    jnp.asarray([700.0]),
    jnp.zeros((1, 1)),
    jnp.asarray([0.02]),
    jnp.zeros((1, 1)),
    jnp.zeros((1,)),
    jnp.zeros((1, 1)),
)
solver = phx.solver.ParticleConversionSolverPlan(
    phx.solver.ParticleConversionBackend.STRUCTURED_NATIVE,
    substeps=2,
)
result = None
for step in range(20):
    result = phx.solver.advance_particle_conversion(
        compiled.dynamics,
        solver,
        state,
        (boundary,),
        jnp.asarray(step * 1.0e-4),
        jnp.asarray(1.0e-4),
    )
    state = result.accepted_state

metrics = batch.mesh.metrics(state.batches[0].outer_scale)
final = thermodynamics.state(
    state.batches[0].internal_energy,
    state.batches[0].species_amount,
    metrics.cell_measures,
    state.batches[0].porosity,
)
print(f"successful={bool(result.successful)}")
print(f"surface_temperature={float(final.temperature[0, -1]):.6f}")
print(f"energy_residual={float(result.replay.internal_energy_residual):.6e}")
