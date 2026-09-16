#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _plan():
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(16, periodic=True),),
        axis_names=("x",),
    ).prepare(jnp.asarray(((0.0,), (1.0,))))
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
        energy_tolerance=1.0e-8,
    )
    return operators, pnp


def test_mac_pnp_preserves_uniform_equilibrium_and_exact_layouts():
    operators, plan = _plan()
    concentrations = jnp.ones((16, 2))
    evaluation = eqx.filter_jit(plan.evaluate)(concentrations)

    assert bool(evaluation.header.globally_eligible)
    np.testing.assert_allclose(evaluation.concentration_rate, 0.0, atol=1.0e-12)
    np.testing.assert_allclose(evaluation.electrostatic.potential, 0.0, atol=1.0e-12)
    assert evaluation.flux.total_face_flux[0].shape == (16, 2)
    assert evaluation.electrostatic.electric_field[0].shape == (16,)
    assert evaluation.electrostatic.operators_id == operators.prepared_id


def test_mac_ionic_advection_is_conservative_and_ehd_force_is_power_consistent():
    operators, plan = _plan()
    coordinate = (jnp.arange(16) + 0.5) / 16.0
    perturbation = 0.02 * jnp.sin(2.0 * jnp.pi * coordinate)
    concentrations = jnp.stack((1.0 + perturbation, 1.0 - perturbation), axis=-1)
    velocity = (jnp.full((16,), 0.1),)
    evaluation = plan.evaluate(concentrations, face_velocity=velocity)

    assert bool(evaluation.header.globally_eligible)
    np.testing.assert_allclose(evaluation.flux.species_content_defect, 0.0, atol=1.0e-10)
    force = phx.solver.MACElectrohydrodynamicForcePlan(operators).evaluate(
        evaluation, face_velocity=velocity
    )
    assert bool(force.header.globally_eligible)
    expected_power = jnp.sum(
        operators.face_dual_measures[0] * velocity[0] * force.total_force[0]
    )
    np.testing.assert_allclose(force.fluid_power, expected_power, atol=1.0e-12)
    np.testing.assert_allclose(
        force.total_force[0], force.electric_force[0] + force.osmotic_force[0]
    )


def test_mac_pnp_explicit_step_rolls_back_outside_positivity_restriction():
    _, plan = _plan()
    coordinate = (jnp.arange(16) + 0.5) / 16.0
    concentrations = jnp.stack(
        (
            1.0 + 0.05 * jnp.sin(2.0 * jnp.pi * coordinate),
            1.0 - 0.05 * jnp.sin(2.0 * jnp.pi * coordinate),
        ),
        axis=-1,
    )
    before = plan.evaluate(concentrations)
    accepted = plan.step(
        concentrations, jnp.minimum(1.0e-4, 0.25 * before.explicit_step_restriction)
    )
    rejected = plan.step(concentrations, 2.0 * before.explicit_step_restriction)

    assert bool(accepted.successful)
    assert not bool(rejected.successful)
    np.testing.assert_array_equal(rejected.accepted, concentrations)
