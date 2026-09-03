#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax.applications.reacting_flow._advance import (
    ReactiveAdvanceState,
    ReactiveIMEXPlan,
    ReactiveStrangPlan,
)
from phydrax.discretization.finite_volume._dynamics import PreparedFiniteVolumeDynamics
from phydrax.discretization.finite_volume._positivity import FluxPositivityPlan
from phydrax.discretization.finite_volume._riemann import RusanovFluxPlan
from phydrax.equations._chemical_mechanism import (
    ChemicalMechanismIR,
    ChemicalReactionSpec,
)
from phydrax.equations._chemical_rates import ArrheniusRatePlan
from phydrax.equations._chemical_species import ChemicalPhaseKind, ChemicalSpeciesSchema
from phydrax.equations._chemical_thermodynamics import (
    PolynomialSpeciesThermodynamicsPlan,
)
from phydrax.equations._gas_dynamics import HomogeneousMixtureEulerSystem
from phydrax.equations._homogeneous_thermodynamics import (
    HomogeneousHelmholtzPlan,
    IdealGasReferenceHelmholtzTerm,
    ZeroResidualHelmholtzTerm,
)
from phydrax.solver._finite_volume_runtime import PreparedFiniteVolumeRuntime


def _problem(rate=1.0):
    schema = ChemicalSpeciesSchema.from_unique_species(
        ("A", "B"),
        (ChemicalPhaseKind.GAS, ChemicalPhaseKind.GAS),
        jnp.asarray((0.01, 0.01)),
        ("E",),
        jnp.asarray(((1, 1),), dtype=jnp.int32),
        jnp.asarray((0, 0), dtype=jnp.int32),
        gas_standard_pressure=1.0e5,
        provenance="reacting-advance-test",
    )
    species_thermodynamics = PolynomialSpeciesThermodynamicsPlan(
        schema,
        jnp.asarray((20.0, 20.0)),
        jnp.asarray((0.0, -5.0e4)),
        reference_temperature=300.0,
        minimum_temperature=200.0,
        maximum_temperature=3000.0,
    )
    thermodynamics = HomogeneousHelmholtzPlan(
        IdealGasReferenceHelmholtzTerm(schema, species_thermodynamics),
        ZeroResidualHelmholtzTerm(schema),
    )
    system = HomogeneousMixtureEulerSystem(
        thermodynamics, 1, maximum_thermal_iterations=48
    )
    mechanism = ChemicalMechanismIR(
        "A-to-B",
        schema,
        species_thermodynamics,
        (
            ChemicalReactionSpec(
                "A->B",
                {"A": 1.0},
                {"B": 1.0},
                ArrheniusRatePlan(rate),
            ),
        ),
    ).prepare()
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(2, periodic=True),),
        axis_names=("x",),
    ).prepare(jnp.asarray(((0.0,), (1.0e6,))))
    discretization = phx.discretization.FiniteVolumePlan(
        grid, component_names=system.component_names
    ).prepare()
    method = phx.discretization.FiniteVolumeMethodPlan(
        phx.discretization.PiecewiseConstantReconstruction(),
        phx.discretization.RusanovFluxPlan(),
    )
    boundaries = phx.discretization.FiniteVolumeBoundarySet.periodic(("x",))
    dynamics = PreparedFiniteVolumeDynamics(system, discretization, method, boundaries)
    runtime = PreparedFiniteVolumeRuntime(
        dynamics,
        FluxPositivityPlan(2, fallback_flux=RusanovFluxPlan()),
    )
    one_cell = system.primitive_to_conserved(jnp.asarray((0.7, 0.3, 0.0, 800.0)))
    conserved = jnp.broadcast_to(one_cell, discretization.state_shape)
    return system, mechanism, runtime, conserved


def _mass_fraction_a(system, state):
    species = state.conserved[0, : system.species_count]
    return species[0] / jnp.sum(species)


def _invariants(system, state):
    species = state[..., : system.species_count]
    amount = species / system.thermodynamics.schema.molar_masses
    schema = system.thermodynamics.schema
    return (
        jnp.sum(species, axis=-1),
        schema.element_amount(amount),
        schema.charge_amount(amount),
        state[..., -1],
    )


def test_strang_fixed_schedule_is_second_order_and_preserves_all_invariants():
    system, mechanism, runtime, conserved = _problem()
    plan = ReactiveStrangPlan(runtime, mechanism)
    coarse = plan.advance(plan.initial_state(conserved), jnp.asarray(0.2))
    half = plan.advance(plan.initial_state(conserved), jnp.asarray(0.1))
    fine = plan.advance(half.state, jnp.asarray(0.1))
    exact = 0.7 * np.exp(-0.2)
    coarse_error = abs(float(_mass_fraction_a(system, coarse.state)) - exact)
    fine_error = abs(float(_mass_fraction_a(system, fine.state)) - exact)
    before = _invariants(system, conserved)
    after = _invariants(system, coarse.state.conserved)

    assert coarse.evidence.accepted
    assert half.evidence.accepted
    assert fine.evidence.accepted
    assert coarse_error > 3.0 * fine_error
    assert int(coarse.state.schedule_index) == 1
    assert int(fine.state.schedule_index) == 2
    np.testing.assert_allclose(after[0], before[0], atol=1.0e-12)
    np.testing.assert_allclose(after[1], before[1], atol=1.0e-12)
    np.testing.assert_allclose(after[2], before[2], atol=1.0e-12)
    np.testing.assert_array_equal(after[3], before[3])
    np.testing.assert_allclose(coarse.evidence.maximum_mass_defect, 0.0, atol=1.0e-12)
    np.testing.assert_array_equal(coarse.evidence.maximum_energy_defect, 0.0)
    assert coarse.evidence.maximum_diagnostic_heat_release > 0.0


def test_strang_restart_is_deterministic_and_failed_macro_step_fully_rolls_back():
    _, mechanism, runtime, conserved = _problem()
    plan = ReactiveStrangPlan(runtime, mechanism, schedule_substeps=2)
    first = plan.advance(plan.initial_state(conserved), jnp.asarray(0.02))
    restarted = ReactiveAdvanceState(
        first.state.time,
        first.state.conserved,
        first.state.transport_runtime_state,
        accepted_macro_steps=first.state.accepted_macro_steps,
        schedule_index=first.state.schedule_index,
        state_id=first.state.state_id,
    )
    continuous = plan.advance(first.state, jnp.asarray(0.02))
    resumed = plan.advance(restarted, jnp.asarray(0.02))

    assert continuous.evidence.accepted
    assert resumed.evidence.accepted
    np.testing.assert_array_equal(continuous.state.time, resumed.state.time)
    np.testing.assert_array_equal(continuous.state.conserved, resumed.state.conserved)
    np.testing.assert_array_equal(
        continuous.state.schedule_index, resumed.state.schedule_index
    )
    np.testing.assert_array_equal(
        continuous.state.transport_runtime_state.accepted_step,
        resumed.state.transport_runtime_state.accepted_step,
    )
    np.testing.assert_array_equal(
        continuous.state.transport_runtime_state.content_state.conservative_content,
        resumed.state.transport_runtime_state.content_state.conservative_content,
    )

    _, fast_mechanism, fast_runtime, fast_conserved = _problem(rate=1.0e6)
    rejecting = ReactiveStrangPlan(fast_runtime, fast_mechanism)
    initial = rejecting.initial_state(fast_conserved)
    failed = rejecting.advance(initial, jnp.asarray(0.01))
    assert failed.evidence.rolled_back
    assert not failed.evidence.accepted
    np.testing.assert_array_equal(failed.state.time, initial.time)
    np.testing.assert_array_equal(failed.state.conserved, initial.conserved)
    np.testing.assert_array_equal(
        failed.state.accepted_macro_steps, initial.accepted_macro_steps
    )
    np.testing.assert_array_equal(failed.state.schedule_index, initial.schedule_index)
    np.testing.assert_array_equal(
        failed.state.transport_runtime_state.accepted_step,
        initial.transport_runtime_state.accepted_step,
    )
    np.testing.assert_array_equal(
        failed.state.transport_runtime_state.content_state.conservative_content,
        initial.transport_runtime_state.content_state.conservative_content,
    )


def test_coupled_imex_agrees_with_strang_and_keeps_chemical_energy_source_zero():
    system, mechanism, runtime, conserved = _problem()
    strang = ReactiveStrangPlan(runtime, mechanism)
    imex = ReactiveIMEXPlan(
        runtime,
        mechanism,
        nonlinear_iterations=16,
        nonlinear_tolerance=1.0e-11,
    )
    step = jnp.asarray(1.0e-4)
    split_result = strang.advance(strang.initial_state(conserved), step)
    imex_result = imex.advance(imex.initial_state(conserved), step)

    assert split_result.evidence.accepted
    assert imex_result.evidence.accepted
    np.testing.assert_allclose(
        _mass_fraction_a(system, imex_result.state),
        _mass_fraction_a(system, split_result.state),
        rtol=1.0e-8,
        atol=1.0e-11,
    )
    np.testing.assert_array_equal(
        imex_result.state.conserved[..., -1], conserved[..., -1]
    )
    np.testing.assert_array_equal(imex_result.evidence.maximum_energy_defect, 0.0)
