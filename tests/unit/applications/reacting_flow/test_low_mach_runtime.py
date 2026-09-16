#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _runtime(rate=0.1):
    schema = phx.equations.ChemicalSpeciesSchema.from_unique_species(
        ("A", "B"),
        (phx.equations.ChemicalPhaseKind.GAS,) * 2,
        jnp.asarray((0.01, 0.01)),
        ("E",),
        jnp.asarray(((1, 1),), dtype=jnp.int32),
        jnp.zeros((2,), dtype=jnp.int32),
        gas_standard_pressure=1.0e5,
    )
    species = phx.equations.PolynomialSpeciesThermodynamicsPlan(
        schema,
        jnp.asarray((20.0, 20.0)),
        jnp.asarray((0.0, -2.0e4)),
        reference_temperature=300.0,
        minimum_temperature=200.0,
        maximum_temperature=3000.0,
    )
    thermo = phx.equations.HomogeneousHelmholtzPlan(
        phx.equations.IdealGasReferenceHelmholtzTerm(schema, species),
        phx.equations.ZeroResidualHelmholtzTerm(schema),
    )
    mechanism = phx.equations.ChemicalMechanismIR(
        "A-to-B",
        schema,
        species,
        (
            phx.equations.ChemicalReactionSpec(
                "A->B",
                {"A": 1.0},
                {"B": 1.0},
                phx.equations.ArrheniusRatePlan(rate),
            ),
        ),
    ).prepare()
    formulation = phx.applications.reacting_flow.LowMachReactingFormulation(
        thermo, 1, mechanism=mechanism, constraint_tolerance=1.0e-8
    )
    properties = phx.equations.ReferencePowerLawGasTransportPlan(
        jnp.asarray(((0.0, 1.0e-5), (1.0e-5, 0.0))),
        jnp.asarray((1.0e-5, 1.0e-5)),
        jnp.asarray((0.02, 0.02)),
    )
    mixture = phx.equations.MixtureAveragedTransportPlan(thermo, properties)
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(4, periodic=True),),
        axis_names=("x",),
    ).prepare(jnp.asarray(((0.0,), (1.0,))))
    finite_volume = phx.discretization.FiniteVolumePlan(grid).prepare()
    operators = phx.discretization.MACOperatorPlan(finite_volume).prepare()
    scalar = phx.discretization.MACScalarProblem(
        tuple(
            phx.discretization.MACScalarTransport(name, 0.0, advection="upwind")
            for name in ("rho:A", "rho:B", "rhoh")
        )
    ).prepare(operators)
    projection = phx.solver.MACVariableDensityProjectionPlan(
        operators, tolerance=1.0e-8, maximum_iterations=200
    )
    plan = phx.applications.reacting_flow.LowMachReactingFlowPlan(
        formulation,
        mixture,
        scalar,
        projection,
        phx.applications.reacting_flow.LowMachReactingSDCPlan(3, 1.0e-6),
        pressure_mode="closed",
        conservation_tolerance=1.0e-7,
        eos_tolerance=1.0e-6,
    )
    velocity = tuple(
        jnp.zeros(layout.shape, dtype=jnp.float64)
        for layout in finite_volume.face_layouts
    )
    state = plan.initialize(
        velocity,
        jnp.full(finite_volume.cell_shape, 800.0),
        jnp.broadcast_to(jnp.asarray((0.8, 0.2)), finite_volume.cell_shape + (2,)),
        1.0e5,
    )
    return plan, state


def test_closed_uniform_low_mach_reaction_conserves_mass_enthalpy_and_eos():
    plan, state = _runtime()
    result = plan.advance(state, 0.01)

    assert bool(result.successful)
    assert float(jnp.mean(result.accepted.species_density[..., 1])) > float(
        jnp.mean(state.species_density[..., 1])
    )
    np.testing.assert_allclose(
        jnp.sum(result.accepted.species_density, axis=-1),
        jnp.sum(state.species_density, axis=-1),
        rtol=1.0e-8,
    )
    np.testing.assert_allclose(
        result.accepted.enthalpy_density, state.enthalpy_density, rtol=1.0e-8
    )
    assert jnp.max(jnp.abs(result.diagnostics.eos_pressure_defect)) < 0.1
    assert jnp.max(jnp.abs(result.projection.divergence_defect)) < 1.0e-7


def test_failed_low_mach_source_rolls_back_every_state_leaf():
    plan, state = _runtime()
    source = jnp.zeros_like(state.species_density).at[..., 0].set(-1.0e8)
    result = plan.advance(state, 0.1, external_species_rate=source)

    assert not bool(result.successful)
    np.testing.assert_array_equal(result.accepted.species_density, state.species_density)
    np.testing.assert_array_equal(
        result.accepted.enthalpy_density, state.enthalpy_density
    )
    np.testing.assert_array_equal(
        result.accepted.thermodynamic_pressure, state.thermodynamic_pressure
    )
    np.testing.assert_array_equal(result.accepted.time, state.time)
