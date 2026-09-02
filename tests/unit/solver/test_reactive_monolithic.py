#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp

import phydrax as phx


def _problem(
    *,
    drag=0.0,
    particle_temperature=300.0,
    fluid_temperature=500.0,
    fluid_species=0.0,
    mass_transfer=0.0,
):
    particles = phx.discretization.ParticleSetPlan(
        jnp.asarray([0]), jnp.ones((1,)), ambient_dimension=2
    ).prepare()
    schema = phx.equations.ChemicalSpeciesSchema(
        ("A",),
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
        thermodynamics.energy_from_temperature(
            jnp.asarray([[particle_temperature]]), species
        ),
        species,
        jnp.asarray([[0.2]]),
        jnp.ones((1, 1)),
        jnp.asarray([0.1]),
    )
    conversion = phx.equations.compile_particle_conversion_problem(
        phx.equations.ParticleConversionProblemIR("monolithic", (material,)),
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
    transfer = phx.discretization.MeshCompactKernelSplatAssignment(1.0, 1).prepare(
        phx.discretization.MeshSplatTarget(mesh, entity_dimension=2, measure=measure),
        jnp.zeros((particles.capacity, 2)),
        particles.active_mask,
        particles.particle_ids,
    )
    exchange = phx.equations.ParticleContinuumExchangePlan(
        transfer,
        jnp.asarray([1.0]),
        jnp.full((1, 1), mass_transfer),
        schema_id=schema.schema_id,
    )
    fluid_plan = phx.equations.CellwiseReactiveFluidImplicitPlan(
        jnp.asarray([1.0]), jnp.asarray([10.0]), jnp.ones((1, 1))
    )
    coupling = phx.equations.ReactiveMonolithicCouplingPlan(
        fluid_plan, conversion.dynamics, exchange, jnp.asarray([drag])
    )
    fluid = phx.equations.ReactiveFluidImplicitState(
        jnp.asarray([[1.0, 0.0]]),
        jnp.asarray([fluid_temperature]),
        jnp.full((1, 1), fluid_species),
    )
    state = phx.solver.initialize_reactive_monolithic_state(
        coupling, fluid, conversion_state, jnp.asarray([[0.0, 0.0]])
    )
    boundary = phx.equations.ParticleTransportBoundary(
        jnp.asarray([particle_temperature]),
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
    return coupling, state, stage


def test_monolithic_heat_exchange_is_conservative_and_implicit():
    coupling, state, stage = _problem()
    prepared = phx.solver.prepare_reactive_monolithic_step(
        coupling, phx.solver.ReactiveMonolithicSolverPlan(), stage
    )
    result = phx.solver.solve_reactive_monolithic_step(prepared, state)
    assert result.successful
    assert result.accepted_state.fluid.temperature[0] < state.fluid.temperature[0]
    assert jnp.abs(result.evaluation.energy_residual) < 1.0e-12


def test_monolithic_implicit_drag_closes_momentum():
    coupling, state, stage = _problem(drag=100.0)
    prepared = phx.solver.prepare_reactive_monolithic_step(
        coupling, phx.solver.ReactiveMonolithicSolverPlan(), stage
    )
    result = phx.solver.solve_reactive_monolithic_step(prepared, state)
    assert result.successful
    assert result.accepted_state.particle_velocity[0, 0] > 0.0
    assert result.accepted_state.fluid.velocity[0, 0] < 1.0
    assert jnp.linalg.norm(result.evaluation.momentum_residual) < 1.0e-12


def test_monolithic_preconditioner_modes_recover_same_root():
    coupling, state, stage = _problem(drag=10.0)
    values = []
    for mode in phx.solver.ReactiveMonolithicPreconditionerMode:
        solver = phx.solver.ReactiveMonolithicSolverPlan(preconditioner_mode=mode)
        result = phx.solver.solve_reactive_monolithic_step(
            phx.solver.prepare_reactive_monolithic_step(coupling, solver, stage),
            state,
        )
        assert result.successful
        values.append(result.accepted_state.fluid.temperature)
    assert jnp.allclose(jnp.stack(values), values[0], rtol=1.0e-8)


def test_monolithic_residual_has_finite_matrix_free_jvp():
    coupling, _, stage = _problem(drag=1.0)
    unknown = coupling.initial_unknown(stage)
    tangent = jax.tree.map(jnp.ones_like, unknown)
    primal, derivative = jax.jvp(
        lambda value: coupling.evaluate(value, stage).residual,
        (unknown,),
        (tangent,),
    )
    assert all(jnp.all(jnp.isfinite(value)) for value in jax.tree.leaves(primal))
    assert all(jnp.all(jnp.isfinite(value)) for value in jax.tree.leaves(derivative))


def test_monolithic_species_exchange_closes_inventory():
    coupling, state, stage = _problem(fluid_species=2.0, mass_transfer=1.0)
    result = phx.solver.solve_reactive_monolithic_step(
        phx.solver.prepare_reactive_monolithic_step(
            coupling, phx.solver.ReactiveMonolithicSolverPlan(), stage
        ),
        state,
    )
    assert result.successful
    assert jnp.max(jnp.abs(result.evaluation.species_residual)) < 1.0e-12
    assert not jnp.isclose(
        result.accepted_state.fluid.species_concentration[0, 0],
        state.fluid.species_concentration[0, 0],
    )


def test_stiff_monolithic_exchange_converges_without_staggered_iterations():
    coupling, state, stage = _problem(drag=1.0e4)
    result = phx.solver.solve_reactive_monolithic_step(
        phx.solver.prepare_reactive_monolithic_step(
            coupling,
            phx.solver.ReactiveMonolithicSolverPlan(
                preconditioner_mode=(
                    phx.solver.ReactiveMonolithicPreconditionerMode.SCHUR_COMPLEMENT
                )
            ),
            stage,
        ),
        state,
    )
    assert result.successful
    assert jnp.linalg.norm(result.evaluation.momentum_residual) < 1.0e-12


def test_monolithic_radiative_source_is_solved_inside_root():
    coupling, state, stage = _problem()
    stage = phx.solver.make_reactive_monolithic_stage(
        coupling,
        state,
        stage.particle_position,
        stage.particle_mass,
        stage.particle_active,
        stage.conversion_boundaries,
        stage.time,
        stage.step_size,
        radiative_energy_rate=(
            jnp.full_like(state.conversion.batches[0].internal_energy, 10.0),
        ),
    )
    result = phx.solver.solve_reactive_monolithic_step(
        phx.solver.prepare_reactive_monolithic_step(
            coupling, phx.solver.ReactiveMonolithicSolverPlan(), stage
        ),
        state,
    )
    assert result.successful
    assert jnp.sum(result.accepted_state.conversion.batches[0].internal_energy) > jnp.sum(
        state.conversion.batches[0].internal_energy
    )


def test_monolithic_event_margin_rejects_and_rolls_back():
    coupling, state, stage = _problem()
    solver = phx.solver.ReactiveMonolithicSolverPlan(event_margin=2.0)
    result = phx.solver.solve_reactive_monolithic_step(
        phx.solver.prepare_reactive_monolithic_step(coupling, solver, stage), state
    )
    assert not result.successful
    assert result.event_split_required
    assert jnp.allclose(result.accepted_state.fluid.temperature, state.fluid.temperature)
