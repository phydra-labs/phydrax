#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
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


def test_bernoulli_has_finite_forward_and_reverse_derivatives_at_extremes():
    bernoulli = phx.discretization.stable_bernoulli
    arguments = jnp.asarray([-1.0e20, -1000.0, -1.0, 0.0, 1.0, 1000.0, 1.0e20])
    values, forward = jax.jvp(bernoulli, (arguments,), (jnp.ones_like(arguments),))
    reverse = jax.jit(jax.grad(lambda x: jnp.sum(bernoulli(x))))(arguments)

    assert jnp.all(jnp.isfinite(values))
    assert jnp.all(jnp.isfinite(forward))
    assert jnp.all(jnp.isfinite(reverse))
    np.testing.assert_allclose(forward, reverse, atol=1e-14)
    np.testing.assert_allclose(values[::3], [1.0e20, 1.0, 0.0], atol=1e-14)
    np.testing.assert_allclose(reverse[::3], [-1.0, -0.5, 0.0], atol=1e-14)
    np.testing.assert_allclose(jax.grad(jax.grad(bernoulli))(0.0), 1.0 / 6.0)


def test_sg_flux_has_physical_diffusion_drift_and_boltzmann_directions():
    flux = phx.discretization.scharfetter_gummel_flux
    # At zero drift, material moves from high to low concentration.
    np.testing.assert_allclose(flux(3.0, 1.0, 0.0, 2.0), 4.0)
    # Uniform density moves down the drift potential, with the exact drift rate.
    np.testing.assert_allclose(flux(3.0, 3.0, 2.0, 2.0), -12.0)
    difference = jnp.asarray([-3.0, -1.0, 0.0, 1.0, 3.0])
    np.testing.assert_allclose(
        flux(1.0, jnp.exp(-difference), difference), 0.0, atol=1e-14
    )
    np.testing.assert_allclose(
        flux(3.0, 1.0, difference), -flux(1.0, 3.0, -difference), atol=1e-14
    )


def test_pnp_ideal_diffusion_is_not_counted_twice():
    plan = _periodic_electrolyte()
    coordinate = plan.electrostatic.bridge.cochain.coordinates[0][:, 0]
    density = 1.0 + 0.2 * jnp.cos(2.0 * jnp.pi * coordinate)
    concentrations = jnp.stack((density, density), axis=-1)
    evaluation = plan.evaluate(concentrations)
    expected_rate = 1.0e-3 * 16.0**2 * (
        jnp.roll(concentrations, 1, axis=0)
        - 2.0 * concentrations
        + jnp.roll(concentrations, -1, axis=0)
    )

    assert evaluation.successful
    assert evaluation.flux.free_energy_dissipation > 0.0
    np.testing.assert_allclose(evaluation.concentration_rate, expected_rate, atol=1e-12)
    np.testing.assert_allclose(evaluation.flux.species_mass_defect, 0.0, atol=1e-14)


def test_pnp_preserves_nonuniform_boltzmann_equilibrium_with_fixed_charge():
    plan = _periodic_electrolyte()
    coordinate = plan.electrostatic.bridge.cochain.coordinates[0][:, 0]
    thermal_voltage = (
        phx.equations.UNIVERSAL_GAS_CONSTANT
        * plan.parameters.temperature
        / phx.equations.FARADAY_CONSTANT
    )
    dimensionless = 0.4 * jnp.cos(2.0 * jnp.pi * coordinate)
    potential = thermal_voltage * dimensionless
    concentrations = jnp.stack(
        (jnp.exp(-dimensionless), jnp.exp(dimensionless)), axis=-1
    )
    # The periodic finite-volume Laplacian eigenvalue is known independently
    # of the Poisson action. Positive charge and potential have the same sign.
    eigenvalue = 4.0 * 16.0**2 * jnp.sin(jnp.pi / 16.0) ** 2
    charge = plan.parameters.permittivity * eigenvalue * potential
    ionic_charge = phx.equations.FARADAY_CONSTANT * (
        concentrations[:, 0] - concentrations[:, 1]
    )
    equilibrium_plan = phx.solver.PoissonNernstPlanckPlan(
        plan.electrostatic,
        plan.closure,
        plan.parameters,
        fixed_charge=charge - ionic_charge,
    )
    evaluation = equilibrium_plan.evaluate(concentrations)

    assert evaluation.successful
    np.testing.assert_allclose(evaluation.electrostatic.potential, potential, atol=1e-11)
    np.testing.assert_allclose(evaluation.flux.edge_flux, 0.0, atol=1e-12)
    np.testing.assert_allclose(evaluation.concentration_rate, 0.0, atol=1e-10)
    np.testing.assert_allclose(evaluation.flux.species_mass_defect, 0.0, atol=1e-14)


def test_positive_charge_and_nonzero_dirichlet_data_give_correct_poisson_field():
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(8, periodic=False),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    bridge = phx.discretization.StructuredCochainBridge(grid)
    coordinate = bridge.cochain.coordinates[0][:, 0]
    lift = 2.0 + 3.0 * coordinate
    boundary = phx.solver.CochainElectrostaticBoundaryPlan.dirichlet(bridge, lift)
    plan = phx.solver.CochainElectrostaticPlan(bridge, boundary, permittivity=2.0)
    result = plan.solve(jnp.full(coordinate.shape, 24.0))
    expected_potential = lift + 6.0 * coordinate * (1.0 - coordinate)
    edge_coordinate = bridge.cochain.coordinates[1][:, 0]
    expected_electric = -3.0 - 6.0 * (1.0 - 2.0 * edge_coordinate)

    assert result.successful
    np.testing.assert_allclose(result.potential, expected_potential, atol=1e-9)
    np.testing.assert_allclose(result.physical_electric[0], expected_electric, atol=1e-9)


def test_neumann_charge_balance_recovers_quadratic_potential_and_gauge():
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(8, periodic=False),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    bridge = phx.discretization.StructuredCochainBridge(grid)
    coordinate = bridge.cochain.coordinates[0][:, 0]
    volumes = bridge.cochain.hodge_stars[0]
    # phi=x^2: rho=-2; outward grad(phi)=2 on the right boundary.
    source = jnp.zeros_like(coordinate).at[-1].set(2.0 / volumes[-1])
    boundary = phx.solver.CochainElectrostaticBoundaryPlan.neumann(bridge, source)
    result = phx.solver.CochainElectrostaticPlan(bridge, boundary).solve(
        jnp.full_like(coordinate, -2.0)
    )
    expected = coordinate**2
    expected -= jnp.sum(volumes * expected) / jnp.sum(volumes)
    assert result.successful
    np.testing.assert_allclose(result.potential, expected, atol=1e-9)
    np.testing.assert_allclose(
        result.physical_electric[0], -2 * bridge.cochain.coordinates[1][:, 0],
        atol=1e-9,
    )
