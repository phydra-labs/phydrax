#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Solve stiff fluid-particle momentum and heat exchange in one Newton root."""

import jax.numpy as jnp

import phydrax as phx


particles = phx.discretization.ParticleSetPlan(
    jnp.asarray([0]), jnp.ones((1,)), ambient_dimension=2
).prepare()
schema = phx.equations.ChemicalSpeciesSchema(
    ("solid",),
    (phx.equations.ChemicalPhaseKind.SOLID,),
    jnp.asarray([1.0]),
    ("X",),
    jnp.asarray([[1]]),
    jnp.zeros_like(jnp.asarray([1.0]), dtype=jnp.int32),
)
thermodynamics = phx.equations.ParticleThermodynamicMaterialPlan(
    phx.equations.PolynomialSpeciesThermodynamicsPlan(
        schema, jnp.asarray([10.0]), jnp.asarray([0.0])
    )
)
material = phx.equations.ParticleThermochemicalMaterialBundle(
    thermodynamics,
    phx.equations.ParticleTransportMaterialPlan(
        schema, jnp.asarray([0.0]), jnp.asarray([0.0])
    ),
)
batch_plan = phx.discretization.ParticleInternalBatchPlan(
    jnp.asarray([0]),
    phx.discretization.RadialShellMeshPlan(
        phx.discretization.ParticleInternalGeometry.SPHERE, 1
    ),
    1,
)
batch = batch_plan.prepare(particles)
species = jnp.ones((1, 1, 1))
internal = phx.discretization.initialize_particle_internal_batch(
    batch,
    thermodynamics.energy_from_temperature(jnp.asarray([[300.0]]), species),
    species,
    jnp.asarray([[0.2]]),
    jnp.ones((1, 1)),
    jnp.asarray([0.1]),
)
conversion = phx.equations.compile_particle_conversion_problem(
    phx.equations.ParticleConversionProblemIR("monolithic-example", (material,)),
    particles,
    (batch_plan,),
)
conversion_state = conversion.initialize_state((internal,))
transfer = phx.discretization.ConservativeParticleGridTransferPlan(
    jnp.asarray([[0.0, 0.0]]), jnp.asarray([1.0]), 1.0, 1
).prepare(particles)
exchange = phx.equations.ParticleContinuumExchangePlan(
    transfer,
    jnp.asarray([1.0]),
    jnp.zeros((1, 1)),
    schema_id=schema.schema_id,
)
coupling = phx.equations.ReactiveMonolithicCouplingPlan(
    phx.equations.CellwiseReactiveFluidImplicitPlan(
        jnp.asarray([1.0]), jnp.asarray([10.0]), jnp.ones((1, 1))
    ),
    conversion.dynamics,
    exchange,
    jnp.asarray([100.0]),
)
state = phx.solver.initialize_reactive_monolithic_state(
    coupling,
    phx.equations.ReactiveFluidImplicitState(
        jnp.asarray([[1.0, 0.0]]), jnp.asarray([500.0]), jnp.zeros((1, 1))
    ),
    conversion_state,
    jnp.zeros((1, 2)),
)
boundary = phx.equations.ParticleTransportBoundary(
    jnp.asarray([300.0]),
    jnp.zeros((1, 1)),
    jnp.zeros((1,)),
    jnp.zeros((1, 1)),
    jnp.zeros((1,)),
    jnp.zeros((1, 1)),
)
stage = phx.solver.make_reactive_monolithic_stage(
    coupling,
    state,
    jnp.asarray([[0.0, 0.0]]),
    jnp.asarray([1.0]),
    jnp.asarray([True]),
    (boundary,),
    jnp.asarray(0.0),
    jnp.asarray(0.01),
)
prepared = phx.solver.prepare_reactive_monolithic_step(
    coupling,
    phx.solver.ReactiveMonolithicSolverPlan(
        preconditioner_mode=phx.solver.ReactiveMonolithicPreconditionerMode.SCHUR_COMPLEMENT
    ),
    stage,
)
result = phx.solver.solve_reactive_monolithic_step(prepared, state)
print(f"successful={bool(result.successful)}")
print(f"fluid_temperature={float(result.accepted_state.fluid.temperature[0]):.6f}")
print(f"particle_velocity={float(result.accepted_state.particle_velocity[0, 0]):.6f}")
print(
    f"momentum_residual={float(jnp.linalg.norm(result.evaluation.momentum_residual)):.6e}"
)
print(f"energy_residual={float(result.evaluation.energy_residual):.6e}")
