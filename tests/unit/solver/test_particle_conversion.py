#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


def _problem():
    schema = phx.equations.ChemicalSpeciesSchema(
        ("A",),
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
    transport = phx.equations.ParticleTransportMaterialPlan(
        schema, jnp.asarray([1.0]), jnp.asarray([1.0e-6])
    )
    material = phx.equations.ParticleThermochemicalMaterialBundle(
        thermodynamics, transport
    )
    particles = phx.discretization.ParticleSetPlan(
        jnp.asarray([0]), jnp.ones((1,)), ambient_dimension=3
    ).prepare()
    batch_plan = phx.discretization.ParticleInternalBatchPlan(
        jnp.asarray([0]),
        phx.discretization.RadialShellMeshPlan(
            phx.discretization.ParticleInternalGeometry.SPHERE, 2
        ),
        1,
    )
    batch = batch_plan.prepare(particles)
    species = jnp.ones((1, 2, 1))
    energy = thermodynamics.energy_from_temperature(
        jnp.asarray([[300.0, 400.0]]), species
    )
    batch_state = phx.discretization.initialize_particle_internal_batch(
        batch,
        energy,
        species,
        jnp.full((1, 2), 0.2),
        jnp.ones((1, 2)),
        jnp.asarray([1.0]),
    )
    compiled = phx.equations.compile_particle_conversion_problem(
        phx.equations.ParticleConversionProblemIR("solver", (material,)),
        particles,
        (batch_plan,),
    )
    state = compiled.initialize_state((batch_state,))
    boundary = phx.equations.ParticleTransportBoundary(
        jnp.asarray([500.0]),
        jnp.zeros((1, 1)),
        jnp.asarray([1.0]),
        jnp.zeros((1, 1)),
        jnp.zeros((1,)),
        jnp.zeros((1, 1)),
    )
    return compiled, state, boundary


def test_reference_and_structured_conversion_backends_agree_and_replay_balance():
    compiled, state, boundary = _problem()
    reference = phx.solver.advance_particle_conversion(
        compiled.dynamics,
        phx.solver.ParticleConversionSolverPlan(
            phx.solver.ParticleConversionBackend.REFERENCE_ROSENBROCK,
            substeps=2,
        ),
        state,
        (boundary,),
        jnp.asarray(0.0),
        jnp.asarray(1.0e-4),
    )
    structured = phx.solver.advance_particle_conversion(
        compiled.dynamics,
        phx.solver.ParticleConversionSolverPlan(
            phx.solver.ParticleConversionBackend.STRUCTURED_NATIVE,
            substeps=2,
        ),
        state,
        (boundary,),
        jnp.asarray(0.0),
        jnp.asarray(1.0e-4),
    )
    assert reference.successful
    assert structured.successful
    assert jnp.abs(reference.replay.internal_energy_residual) < 1.0e-9
    assert jnp.abs(structured.replay.internal_energy_residual) < 1.0e-9
    assert jnp.allclose(
        reference.accepted_state.batches[0].internal_energy,
        structured.accepted_state.batches[0].internal_energy,
        rtol=2.0e-3,
        atol=1.0e-7,
    )


def test_conversion_validity_certificate_masks_branchwise_derivatives_at_events():
    compiled, state, boundary = _problem()
    evaluation = compiled.dynamics.evaluate(state, (boundary,))
    policy = phx.solver.ParticleConversionSensitivityPolicy(
        species_margin=1.0e-12,
        porosity_margin=1.0e-12,
        scale_margin=1.0e-12,
        temperature_margin=1.0e-12,
        phase_margin=1.0e-12,
        reaction_margin=1.0e-12,
    )
    result = phx.solver.sharp_particle_conversion_jvp(
        lambda value: value**2,
        jnp.asarray(2.0),
        jnp.asarray(1.0),
        state,
        evaluation,
        policy,
    )
    assert result.usable
    assert jnp.isclose(result.sensitivity, 4.0)

    exhausted = phx.discretization.ParticleConversionState(
        (
            phx.discretization.ParticleInternalBatchState(
                state.batches[0].internal_energy,
                jnp.zeros_like(state.batches[0].species_amount),
                state.batches[0].porosity,
                state.batches[0].internal_surface_area,
                state.batches[0].outer_scale,
                state.batches[0].reaction_front,
                state.batches[0].active,
                state.batches[0].batch_id,
            ),
        ),
        state.ledger,
        state.state_id,
    )
    exhausted_evaluation = compiled.dynamics.evaluate(exhausted, (boundary,))
    exhausted_result = phx.solver.sharp_particle_conversion_jvp(
        lambda value: value**2,
        jnp.asarray(2.0),
        jnp.asarray(1.0),
        exhausted,
        exhausted_evaluation,
        policy,
    )
    assert not exhausted_result.usable
    assert jnp.isnan(exhausted_result.sensitivity)


def test_generic_hybrid_event_localizes_transverse_phase_exhaustion():
    plan = phx.solver.HybridEventPlan(
        lambda state, args: state[0],
        lambda state, args: jnp.zeros_like(state),
        lambda state, args: jnp.asarray([-1.0]),
        lambda state, args: jnp.asarray([0.0]),
        event_kind="phase_exhaustion",
        plan_id="phase-exhaustion",
    )
    result = phx.solver.localize_hybrid_event(
        plan,
        lambda time, args: jnp.asarray([0.5 - time]),
        jnp.asarray(0.0),
        jnp.asarray(1.0),
    )
    assert result.successful
    assert jnp.isclose(result.event_time, 0.5, atol=1.0e-10)
    assert jnp.all(jnp.isfinite(result.saltation_matrix))
