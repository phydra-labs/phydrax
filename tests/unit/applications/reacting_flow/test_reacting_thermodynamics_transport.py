#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

from phydrax.applications.reacting_flow._transport import (
    MixtureAveragedTransportPlan,
    StefanMaxwellTransportPlan,
)
from phydrax.equations._chemical_species import ChemicalPhaseKind, ChemicalSpeciesSchema
from phydrax.equations._chemical_thermodynamics import (
    PolynomialSpeciesThermodynamicsPlan,
    UNIVERSAL_GAS_CONSTANT,
)
from phydrax.equations._gas_dynamics import HomogeneousMixtureEulerSystem
from phydrax.equations._homogeneous_thermodynamics import (
    HomogeneousHelmholtzPlan,
    IdealGasReferenceHelmholtzTerm,
    ZeroResidualHelmholtzTerm,
)


def _gas_model():
    schema = ChemicalSpeciesSchema.from_unique_species(
        ("light", "middle", "heavy"),
        (ChemicalPhaseKind.GAS,) * 3,
        jnp.asarray((0.002, 0.016, 0.032)),
        ("E",),
        jnp.asarray(((1, 1, 1),), dtype=jnp.int32),
        jnp.asarray((0, 0, 0), dtype=jnp.int32),
        gas_standard_pressure=1.0e5,
        provenance="reacting-flow-test",
    )
    species_thermodynamics = PolynomialSpeciesThermodynamicsPlan(
        schema,
        jnp.asarray((20.0, 24.0, 28.0)),
        jnp.asarray((1.0e3, 1.8e4, -9.7e4)),
        reference_temperature=300.0,
        minimum_temperature=200.0,
        maximum_temperature=3000.0,
    )
    return HomogeneousHelmholtzPlan(
        IdealGasReferenceHelmholtzTerm(schema, species_thermodynamics),
        ZeroResidualHelmholtzTerm(schema),
    )


def _pressure_state(model, temperature, pressure, mass):
    molar_mass = 1.0 / jnp.sum(mass / model.schema.molar_masses)
    mole = mass * molar_mass / model.schema.molar_masses
    molar_density = pressure / (UNIVERSAL_GAS_CONSTANT * temperature)
    return model.evaluate(temperature, molar_density, mole)


def _transport(plan_type):
    model = _gas_model()
    diffusion = jnp.asarray(
        (
            (0.0, 1.0e-5, 1.2e-5),
            (1.0e-5, 0.0, 0.8e-5),
            (1.2e-5, 0.8e-5, 0.0),
        )
    )
    return plan_type(
        model,
        diffusion,
        jnp.asarray((1.0e-5, 1.3e-5, 1.7e-5)),
        jnp.asarray((0.02, 0.025, 0.03)),
    )


def test_catalog_gas_phase_standard_pressure_and_homogeneous_energy_inversion():
    model = _gas_model()
    schema = model.schema
    species_density = jnp.asarray((0.24, 0.36, 0.60))
    temperature = jnp.asarray(1100.0)
    state = model.evaluate_density_temperature(species_density, temperature)
    internal_energy_density = state.molar_density * state.molar_internal_energy
    recovered = model.solve_density_energy(species_density, internal_energy_density)

    assert schema.catalog.component_names == schema.species_names
    assert schema.phase_count == 1
    assert schema.phase_specs[0].kind is ChemicalPhaseKind.GAS
    assert schema.phase_specs[0].standard_pressure == 1.0e5
    assert model.ideal.schema.schema_id == schema.schema_id
    assert model.residual.schema.schema_id == schema.schema_id
    assert recovered.successful
    np.testing.assert_allclose(recovered.state.temperature, temperature, rtol=1.0e-10)
    np.testing.assert_allclose(
        recovered.state.pressure,
        state.pressure,
        rtol=1.0e-10,
    )


def test_homogeneous_euler_state_uses_every_species_density_slot():
    model = _gas_model()
    system = HomogeneousMixtureEulerSystem(model, 2)
    species_density = jnp.asarray((0.24, 0.36, 0.60))
    primitive = jnp.concatenate((species_density, jnp.asarray((4.0, -2.0, 900.0))))
    conserved = system.primitive_to_conserved(primitive)
    recovered = system.conserved_to_primitive(conserved)

    assert conserved.shape == (model.schema.species_count + system.dimension + 1,)
    np.testing.assert_allclose(system.density(conserved), jnp.sum(species_density))
    np.testing.assert_allclose(recovered, primitive, rtol=1.0e-10)
    assert system.admissible(conserved)

    invalid = conserved.at[1].set(-0.1)
    assert not system.admissible(invalid)


def test_mixture_averaged_transport_conserves_mass_and_carries_full_enthalpy():
    plan = _transport(MixtureAveragedTransportPlan)
    temperature = jnp.asarray(1000.0)
    pressure = jnp.asarray(101325.0)
    mass = jnp.asarray((0.2, 0.3, 0.5))
    density = _pressure_state(
        plan.thermodynamics, temperature, pressure, mass
    ).mass_density
    gradient = jnp.asarray(((0.05, -0.03), (-0.02, 0.01), (-0.03, 0.02)))
    result = plan.evaluate(
        temperature,
        pressure,
        density,
        mass,
        gradient,
        temperature_gradient=jnp.asarray((10.0, -4.0)),
    )
    species_enthalpy = (
        plan.thermodynamics.thermodynamics.evaluate(temperature).molar_enthalpy
        / plan.thermodynamics.schema.molar_masses
    )
    expected_enthalpy_flux = jnp.sum(
        result.species_mass_flux * species_enthalpy[:, None], axis=0
    )
    inert = plan.evaluate(
        temperature,
        pressure,
        density,
        mass,
        jnp.zeros_like(gradient),
    )

    assert result.successful
    np.testing.assert_allclose(result.net_mass_flux, 0.0, atol=1.0e-18)
    np.testing.assert_allclose(result.density_residual, 0.0, atol=1.0e-15)
    np.testing.assert_allclose(result.species_enthalpy_flux, expected_enthalpy_flux)
    np.testing.assert_allclose(
        result.total_heat_flux,
        result.conductive_heat_flux + expected_enthalpy_flux,
    )
    np.testing.assert_allclose(inert.species_mass_flux, 0.0, atol=0.0)
    np.testing.assert_allclose(inert.total_heat_flux, 0.0, atol=0.0)


def test_stefan_maxwell_matches_reference_system_mass_and_enthalpy_constraints():
    plan = _transport(StefanMaxwellTransportPlan)
    temperature = jnp.asarray(1000.0)
    pressure = jnp.asarray(101325.0)
    mass = jnp.asarray((0.2, 0.3, 0.5))
    density = _pressure_state(
        plan.thermodynamics, temperature, pressure, mass
    ).mass_density
    gradient = jnp.asarray(((0.05,), (-0.02,), (-0.03,)))
    result = plan.evaluate(temperature, pressure, density, mass, gradient)
    recovered_rhs = result.evidence.system_matrix @ result.diffusion_velocities
    species_enthalpy = (
        plan.thermodynamics.thermodynamics.evaluate(temperature).molar_enthalpy
        / plan.thermodynamics.schema.molar_masses
    )

    assert plan.support_tier == "research"
    assert result.successful
    np.testing.assert_allclose(
        recovered_rhs, result.evidence.right_hand_side, rtol=1.0e-10, atol=1.0e-12
    )
    np.testing.assert_allclose(result.net_mass_flux, 0.0, atol=1.0e-18)
    np.testing.assert_allclose(
        jnp.sum(mass[:, None] * result.diffusion_velocities, axis=0),
        0.0,
        atol=1.0e-15,
    )
    np.testing.assert_allclose(
        result.species_enthalpy_flux,
        jnp.sum(result.species_mass_flux * species_enthalpy[:, None], axis=0),
    )
