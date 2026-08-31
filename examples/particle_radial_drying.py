#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Dry a porous spherical particle with explicit liquid-vapor conversion."""

import jax.numpy as jnp

import phydrax as phx


particles = phx.discretization.ParticleSetPlan(
    jnp.asarray([0]), jnp.ones((1,)), ambient_dimension=3
).prepare()
schema = phx.equations.ParticleSpeciesSchema(
    ("liquid", "vapor"),
    (phx.equations.ParticlePhase.LIQUID, phx.equations.ParticlePhase.GAS),
    jnp.asarray([0.018, 0.018]),
    ("H2O",),
    jnp.asarray([[1, 1]]),
)
thermodynamics = phx.equations.ParticleThermodynamicMaterialPlan(
    schema, jnp.asarray([75.0, 35.0]), jnp.asarray([0.0, 0.0])
)
material = phx.equations.ParticleThermochemicalMaterialBundle(
    thermodynamics,
    phx.equations.ParticleTransportMaterialPlan(
        schema, jnp.asarray([0.2, 0.05]), jnp.asarray([1.0e-10, 1.0e-6])
    ),
)
phase_change = phx.equations.EvaporationPhaseChangePlan(
    schema,
    0,
    1,
    1.0e-6,
    4.0e4,
    phx.equations.AntoineSaturationPressurePlan(8.07131, 1730.63, 233.426),
)
batch_plan = phx.discretization.ParticleInternalBatchPlan(
    jnp.asarray([0]),
    phx.discretization.RadialShellMeshPlan(
        phx.discretization.ParticleInternalGeometry.SPHERE, 5
    ),
    2,
)
batch = batch_plan.prepare(particles)
species = jnp.zeros((1, 5, 2)).at[..., 0].set(0.1)
initial = phx.discretization.initialize_particle_internal_batch(
    batch,
    thermodynamics.energy_from_temperature(jnp.full((1, 5), 360.0), species),
    species,
    jnp.full((1, 5), 0.5),
    jnp.full((1, 5), 0.1),
    jnp.asarray([0.01]),
)
compiled = phx.equations.compile_particle_conversion_problem(
    phx.equations.ParticleConversionProblemIR(
        "radial-drying", (material,), phase_changes=(phase_change,)
    ),
    particles,
    (batch_plan,),
)
state = compiled.initialize_state((initial,))
boundary = phx.equations.ParticleTransportBoundary(
    jnp.asarray([380.0]),
    jnp.zeros((1, 2)),
    jnp.asarray([0.01]),
    jnp.asarray([[0.0, 1.0e-4]]),
    jnp.zeros((1,)),
    jnp.zeros((1, 2)),
)
solver = phx.solver.ParticleConversionSolverPlan(
    phx.solver.ParticleConversionBackend.STRUCTURED_NATIVE,
    substeps=2,
)
result = None
for step in range(10):
    result = phx.solver.advance_particle_conversion(
        compiled.dynamics,
        solver,
        state,
        (boundary,),
        jnp.asarray(step * 1.0e-3),
        jnp.asarray(1.0e-3),
    )
    state = result.accepted_state

liquid_initial = jnp.sum(initial.species_amount[..., 0])
liquid_final = jnp.sum(state.batches[0].species_amount[..., 0])
print(f"successful={bool(result.successful)}")
print(f"liquid_conversion={float(1.0 - liquid_final / liquid_initial):.6e}")
print(f"energy_residual={float(result.replay.internal_energy_residual):.6e}")
print(
    f"element_residual={float(jnp.max(jnp.abs(result.replay.element_residual[0]))):.6e}"
)
