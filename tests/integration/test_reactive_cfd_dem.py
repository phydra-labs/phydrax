#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


def _reactive_problem():
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
            "reactive-dem", dem_material, gravity=jnp.zeros((2,))
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
    schema = phx.equations.ChemicalSpeciesSchema.from_unique_species(
        ("solid",),
        (phx.equations.ChemicalPhaseKind.SOLID,),
        jnp.asarray([0.01]),
        ("X",),
        jnp.asarray([[1]]),
        jnp.zeros_like(jnp.asarray([0.01]), dtype=jnp.int32),
        gas_standard_pressure=101325.0,
    )
    thermodynamics = phx.equations.ParticleThermodynamicMaterialPlan(
        phx.equations.PolynomialSpeciesThermodynamicsPlan(
            schema, jnp.asarray([10.0]), jnp.asarray([0.0])
        )
    )
    transport = phx.equations.ParticleTransportMaterialPlan(
        schema, jnp.asarray([1.0]), jnp.asarray([0.0])
    )
    material = phx.equations.ParticleThermochemicalMaterialBundle(
        thermodynamics, transport
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
    conversion_state = conversion.initialize_state((internal,))
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
    exchange = phx.equations.ParticleContinuumExchangePlan(
        transfer,
        jnp.asarray([1.0]),
        jnp.asarray([[0.0]]),
        schema_id=schema.schema_id,
    )
    plan = phx.equations.ReactiveCFDDEMCouplingPlan(
        dem.dynamics, conversion.dynamics, exchange
    )
    state = phx.solver.initialize_reactive_cfd_dem(
        plan,
        dem_state,
        conversion_state,
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
    schedule = phx.solver.ReactiveParticleCouplingSchedulePlan(
        phx.solver.ParticleConversionSolverPlan(
            phx.solver.ParticleConversionBackend.STRUCTURED_NATIVE
        ),
        dem_substeps=1,
    )
    return plan, state, boundary, schedule


def _sample(fluid_state):
    return phx.solver.ReactiveFluidFields(
        jnp.zeros((1, 2)),
        jnp.ones((1,)),
        jnp.ones((1,)),
        jnp.zeros((1, 2)),
        fluid_state[0],
        fluid_state[1],
    )


def test_reactive_macro_window_commits_heat_species_and_mechanics_atomically():
    plan, state, boundary, schedule = _reactive_problem()

    def update(fluid, momentum, energy, species, step_size):
        del momentum, step_size
        return fluid[0] + energy, fluid[1] + species

    result = phx.solver.advance_reactive_cfd_dem_window(
        plan,
        schedule,
        state,
        _sample,
        update,
        (boundary,),
        jnp.zeros((0,)),
        jnp.asarray([0.001]),
        jnp.asarray(0.0),
        jnp.asarray(1.0e-5),
    )
    assert result.successful
    assert result.accepted_state.accepted_windows == 1
    assert jnp.linalg.norm(result.evaluation.momentum_residual) < 1.0e-12
    assert jnp.abs(result.evaluation.energy_residual) < 1.0e-12
    assert jnp.max(jnp.abs(result.evaluation.species_residual)) < 1.0e-12
    assert result.evaluation.continuum_successful
    assert result.evaluation.conversion_successful
    assert result.evaluation.dem_successful

    def invalid_update(fluid, momentum, energy, species, step_size):
        del momentum, energy, species, step_size
        return jnp.full_like(fluid[0], jnp.nan), fluid[1]

    rejected = phx.solver.advance_reactive_cfd_dem_window(
        plan,
        schedule,
        state,
        _sample,
        invalid_update,
        (boundary,),
        jnp.zeros((0,)),
        jnp.asarray([0.001]),
        jnp.asarray(0.0),
        jnp.asarray(1.0e-5),
    )
    assert not rejected.successful
    assert rejected.accepted_state.accepted_windows == 0
    assert jnp.allclose(
        rejected.accepted_state.conversion_state.batches[0].internal_energy,
        state.conversion_state.batches[0].internal_energy,
    )
