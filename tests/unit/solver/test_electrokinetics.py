#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _periodic_electrolyte():
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(16, periodic=True),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    bridge = phx.discretization.StructuredCochainBridge(grid)
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
    electrostatic = phx.solver.CochainElectrostaticPlan(
        bridge,
        phx.solver.CochainElectrostaticBoundaryPlan.periodic(bridge),
        permittivity=parameters.permittivity,
    )
    closure = phx.equations.IdealDiluteElectrochemicalClosure(schema)
    return phx.solver.PoissonNernstPlanckPlan(
        electrostatic,
        closure,
        parameters,
        energy_tolerance=1.0e-8,
    )


def test_periodic_pnp_preserves_uniform_boltzmann_equilibrium():
    plan = _periodic_electrolyte()
    concentrations = jnp.ones((16, 2))
    evaluation = plan.evaluate(concentrations)

    assert evaluation.successful
    np.testing.assert_allclose(evaluation.concentration_rate, 0.0, atol=1e-12)
    np.testing.assert_allclose(evaluation.flux.species_mass_defect, 0.0, atol=1e-12)
    np.testing.assert_allclose(evaluation.electrostatic.potential, 0.0, atol=1e-12)
    coupling = phx.solver.CochainMACTransferPlan(plan.electrostatic.bridge)
    coupled = coupling.evaluate(evaluation)
    assert coupled.successful
    np.testing.assert_allclose(coupled.power_defect, 0.0, atol=1e-14)


def test_pnp_step_is_conservative_and_energy_dissipative():
    plan = _periodic_electrolyte()
    coordinate = (jnp.arange(16) + 0.5) / 16.0
    perturbation = 0.05 * jnp.sin(2.0 * jnp.pi * coordinate)
    concentrations = jnp.stack((1.0 + perturbation, 1.0 - perturbation), axis=-1)
    before = plan.evaluate(concentrations)
    step_size = jnp.minimum(1.0e-4, 0.25 * before.explicit_step_restriction)
    result = plan.step(concentrations, step_size)

    assert result.successful
    np.testing.assert_allclose(
        jnp.sum(result.concentrations, axis=0),
        jnp.sum(concentrations, axis=0),
        atol=2e-10,
    )
    assert result.evaluation.total_free_energy <= before.total_free_energy + 1e-8


def test_nonzero_dirichlet_electrostatic_lift_is_exact():
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(8, periodic=False),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    bridge = phx.discretization.StructuredCochainBridge(grid)
    boundary = phx.solver.CochainElectrostaticBoundaryPlan.dirichlet(
        bridge,
        jnp.asarray(2.0),
    )
    plan = phx.solver.CochainElectrostaticPlan(bridge, boundary)
    result = plan.solve(jnp.zeros((bridge.cochain.cell_counts[0],)))

    assert result.successful
    np.testing.assert_allclose(result.potential, 2.0, atol=1e-9)
    np.testing.assert_allclose(result.electric, 0.0, atol=1e-9)
