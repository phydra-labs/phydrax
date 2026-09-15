#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _resolved_plan():
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(8, periodic=True),
            phx.discretization.UniformCellAxisSpec(8, periodic=True),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray(((0.0, 0.0), (1.0, 1.0))))
    finite_volume = phx.discretization.FiniteVolumePlan(grid).prepare()
    operators = phx.discretization.MACOperatorPlan(finite_volume).prepare()
    schema = phx.equations.ChemicalSpeciesSchema.from_unique_species(
        ("cation", "anion"),
        (
            phx.equations.ChemicalPhaseKind.LIQUID,
            phx.equations.ChemicalPhaseKind.LIQUID,
        ),
        jnp.asarray((0.023, 0.035)),
        ("M", "X"),
        jnp.asarray(((1, 0), (0, 1)), dtype=jnp.int32),
        jnp.asarray((1, -1), dtype=jnp.int32),
        gas_standard_pressure=101325.0,
    )
    parameters = phx.equations.ElectrolyteTransportParameters(
        schema,
        jnp.asarray((1.0e-3, 1.0e-3)),
        jnp.asarray(300.0),
        jnp.asarray(1.0e8),
    )
    electrostatic = phx.solver.MACElectrostaticPlan(
        operators,
        phx.solver.MACElectrostaticBoundaryPlan.periodic(operators),
        permittivity=parameters.permittivity,
    )
    pnp = phx.solver.MACPoissonNernstPlanckPlan(
        electrostatic,
        phx.equations.IdealDiluteElectrochemicalClosure(schema),
        parameters,
    )
    momentum = phx.discretization.MACMomentumPlan(operators).prepare()
    projection = phx.solver.MACPressureProjectionPlan(
        operators, boundaries=momentum.boundaries, density=1000.0
    )
    plan = phx.solver.ResolvedElectroosmoticStokesPlan(
        pnp,
        momentum,
        projection,
        density=1000.0,
        kinematic_viscosity=1.0e-6,
        hydrodynamic_relaxation=1.0e-3,
        maximum_iterations=3,
        tolerance=1.0e-10,
    )
    return operators, plan


def test_resolved_electroosmotic_equilibrium_commits_atomically():
    operators, plan = _resolved_plan()
    concentrations = jnp.ones((8, 8, 2))
    velocity = tuple(
        jnp.zeros(layout.shape) for layout in operators.discretization.face_layouts
    )
    state = plan.initialize(concentrations, velocity)
    result = plan.advance(state, jnp.asarray(1.0e-4))

    assert bool(result.successful)
    assert int(result.accepted.accepted_steps) == 1
    np.testing.assert_allclose(result.accepted.concentrations, concentrations)
    np.testing.assert_allclose(result.ledger.species_content_defect, 0.0)
    assert result.ledger.divergence_norm < 1.0e-12
    assert result.force.fluid_power == 0.0


def test_resolved_electroosmotic_failure_rolls_back_whole_state():
    operators, plan = _resolved_plan()
    concentrations = jnp.ones((8, 8, 2))
    velocity = tuple(
        jnp.zeros(layout.shape) for layout in operators.discretization.face_layouts
    )
    state = plan.initialize(concentrations, velocity)
    result = plan.advance(state, jnp.asarray(-1.0e-4))

    assert not bool(result.successful)
    assert int(result.accepted.accepted_steps) == 0
    np.testing.assert_array_equal(result.accepted.concentrations, state.concentrations)
    for accepted, incoming in zip(result.accepted.velocity, state.velocity, strict=True):
        np.testing.assert_array_equal(accepted, incoming)


def test_thin_edl_slip_admits_only_electroneutral_thin_low_dukhin_state():
    plan = phx.solver.ThinEDLElectroosmoticSlipPlan(
        permittivity=7.0e-10,
        dynamic_viscosity=1.0e-3,
        zeta_potential=-0.05,
        characteristic_length=1.0e-4,
        surface_conductivity=0.0,
        bulk_conductivity=1.0,
        maximum_debye_ratio=0.05,
        maximum_dukhin_number=0.1,
    )
    evaluation = plan.evaluate(
        jnp.asarray((100.0, 100.0)),
        jnp.asarray((1.0, -1.0)),
        jnp.asarray(300.0),
        jnp.asarray((1000.0, 200.0)),
        jnp.asarray((0.0, 1.0)),
    )

    assert bool(evaluation.header.globally_eligible)
    assert evaluation.slip_velocity[0] > 0.0
    np.testing.assert_allclose(evaluation.slip_velocity[1], 0.0, atol=1.0e-15)
    assert not evaluation.volumetric_force_permitted
    provider = evaluation.boundary_provider()
    np.testing.assert_allclose(provider.value, evaluation.slip_velocity)

    charged = plan.evaluate(
        jnp.asarray((120.0, 100.0)),
        jnp.asarray((1.0, -1.0)),
        jnp.asarray(300.0),
        jnp.asarray((1000.0, 0.0)),
        jnp.asarray((0.0, 1.0)),
    )
    assert not bool(charged.header.globally_eligible)
