#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Advance one atomic nondistributed reactive CFD-DEM macro window."""

import jax.numpy as jnp

import phydrax as phx


particles = phx.discretization.ParticleSetPlan(
    jnp.asarray([0]), jnp.ones((1,)), ambient_dimension=2
).prepare()
dem_material = phx.equations.DEMMaterialTable(
    jnp.asarray([1.0e5]),
    jnp.asarray([0.25]),
    jnp.asarray([[0.9]]),
    jnp.asarray([[0.0]]),
)
dem = phx.equations.compile_discrete_element_problem(
    phx.equations.DiscreteElementProblemIR(
        "reactive-example", dem_material, gravity=jnp.zeros((2,))
    ),
    particles,
    phx.discretization.RigidSphereSetPlan(jnp.asarray([0.1]), jnp.asarray([0])),
    phx.discretization.SoftSphereDEMMethodPlan(
        phx.discretization.DEMContactModelPlan(
            phx.discretization.LinearSpringDashpotNormalPlan(1.0e3)
        )
    ),
    neighborhood=phx.discretization.DenseParticleNeighborhoodPlan(0),
)
dem_state = dem.initialize_state(0.0, jnp.asarray([[0.0, 0.0]]), jnp.zeros((1, 2)))
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
        schema, jnp.asarray([10.0]), jnp.asarray([0.0])
    )
)
material = phx.equations.ParticleThermochemicalMaterialBundle(
    thermodynamics,
    phx.equations.ParticleTransportMaterialPlan(
        schema, jnp.asarray([1.0]), jnp.asarray([0.0])
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
    phx.equations.ParticleConversionProblemIR("conversion", (material,)),
    particles,
    (batch_plan,),
)
mesh = phx.discretization.CellMesh(
    jnp.asarray(((-0.5, -0.5), (0.5, -0.5), (0.0, 1.0))),
    (phx.discretization.CellBlock("cell", "triangle", jnp.asarray(((0, 1, 2),))),),
)
measure = phx.discretization.DiscreteMeasure(
    "cell_volume",
    mesh.support.support_id,
    mesh.topology.entities(2).entity_set_id,
    jnp.asarray((1.0,)),
)
transfer = phx.discretization.MeshCompactKernelSplatAssignment(0.5, 1).prepare(
    phx.discretization.MeshSplatTarget(mesh, entity_dimension=2, measure=measure),
    jnp.zeros((particles.capacity, 2)),
    particles.active_mask,
    particles.particle_ids,
)
plan = phx.equations.ReactiveCFDDEMCouplingPlan(
    dem.dynamics,
    conversion.dynamics,
    phx.equations.ParticleContinuumExchangePlan(
        transfer,
        jnp.asarray([1.0]),
        jnp.asarray([[0.0]]),
        schema_id=schema.schema_id,
    ),
)
state = phx.solver.initialize_reactive_cfd_dem(
    plan,
    dem_state,
    conversion.initialize_state((internal,)),
    (jnp.asarray([500.0]), jnp.asarray([[0.0]])),
)
boundary = phx.equations.ParticleTransportBoundary(
    jnp.asarray([300.0]),
    jnp.zeros((1, 1)),
    jnp.zeros((1,)),
    jnp.zeros((1, 1)),
    jnp.zeros((1,)),
    jnp.zeros((1, 1)),
)


def sample(fluid_state):
    return phx.solver.ReactiveFluidFields(
        jnp.zeros((1, 2)),
        jnp.ones((1,)),
        jnp.ones((1,)),
        jnp.zeros((1, 2)),
        fluid_state[0],
        fluid_state[1],
    )


def update(fluid, momentum, energy, species_source, step_size):
    del momentum, step_size
    return fluid[0] + energy, fluid[1] + species_source


result = phx.solver.advance_reactive_cfd_dem_window(
    plan,
    phx.solver.ReactiveParticleCouplingSchedulePlan(
        phx.solver.ParticleConversionSolverPlan(
            phx.solver.ParticleConversionBackend.STRUCTURED_NATIVE
        ),
        dem_substeps=1,
    ),
    state,
    sample,
    update,
    (boundary,),
    jnp.zeros((0,)),
    jnp.asarray([0.001]),
    jnp.asarray(0.0),
    jnp.asarray(1.0e-5),
)
print(f"successful={bool(result.successful)}")
print(
    f"momentum_residual={float(jnp.linalg.norm(result.evaluation.momentum_residual)):.6e}"
)
print(f"energy_residual={float(result.evaluation.energy_residual):.6e}")
print(
    f"species_residual={float(jnp.max(jnp.abs(result.evaluation.species_residual))):.6e}"
)
print(f"accepted_windows={int(result.accepted_state.accepted_windows)}")
