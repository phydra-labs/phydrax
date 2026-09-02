#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _material_model():
    schema = phx.equations.ChemicalSpeciesSchema.from_unique_species(
        ("gas",),
        (phx.equations.ChemicalPhaseKind.GAS,),
        jnp.asarray((0.02,)),
        ("gas",),
        jnp.asarray(((1,),), dtype=jnp.int32),
        jnp.asarray((0,), dtype=jnp.int32),
        gas_standard_pressure=1.0e5,
    )
    species = phx.equations.PolynomialSpeciesThermodynamicsPlan(
        schema,
        jnp.asarray(((20.0,),)),
        jnp.asarray((0.0,)),
        reference_temperature=300.0,
        minimum_temperature=100.0,
        maximum_temperature=3000.0,
    )
    ideal = phx.equations.IdealGasReferenceHelmholtzTerm(schema, species)
    return phx.equations.HomogeneousHelmholtzPlan(
        ideal, phx.equations.ZeroResidualHelmholtzTerm(schema)
    )


def test_role_specific_coefficients_produce_constant_planck_and_rosseland_means():
    grid = phx.equations.SpectralFrequencyGrid(
        jnp.asarray((1.0e12, 2.0e12, 3.0e12)),
        jnp.asarray((0.5e12, 1.0e12, 0.5e12)),
    )
    temperature_axis = jnp.asarray((200.0, 1000.0))
    pressure_axis = jnp.asarray((1.0e4, 1.0e6))
    absorption_table = phx.equations.RadiationCoefficientTable(
        temperature_axis,
        pressure_axis,
        grid,
        2.0 * jnp.ones((2, 2, 3)),
        phx.equations.RadiationCoefficientRole.ABSORPTION,
        provenance="synthetic constant coefficient",
    )
    transport_table = phx.equations.RadiationCoefficientTable(
        temperature_axis,
        pressure_axis,
        grid,
        3.0 * jnp.ones((2, 2, 3)),
        phx.equations.RadiationCoefficientRole.TRANSPORT,
        provenance="synthetic constant coefficient",
    )
    absorption = absorption_table.evaluate(jnp.asarray(500.0), jnp.asarray(1.0e5))
    transport = transport_table.evaluate(jnp.asarray(500.0), jnp.asarray(1.0e5))
    means = phx.equations.radiation_means(jnp.asarray(500.0), absorption, transport, grid)

    np.testing.assert_allclose(means.planck_absorption, 2.0, rtol=1.0e-12)
    np.testing.assert_allclose(means.rosseland_transport, 3.0, rtol=1.0e-12)
    assert bool(means.successful)


def test_radiation_matter_exchange_is_exactly_conservative():
    model = _material_model()
    species_density = jnp.asarray((1.0,))
    thermal = model.evaluate_density_temperature(species_density, jnp.asarray(700.0))
    molar_density = jnp.sum(species_density / model.schema.molar_masses)
    material_energy = molar_density * thermal.molar_internal_energy
    plan = phx.equations.RadiationMatterExchangePlan(
        model,
        phx.equations.RadiationScaleContract(reduced_light_speed=1.0e6),
        absorption_coefficient=1.0e-3,
    )

    result = plan.advance(
        species_density,
        jnp.asarray(1.0e4),
        material_energy,
        jnp.asarray(1.0e-4),
    )

    np.testing.assert_allclose(
        result.radiation_energy_density + result.material_internal_energy_density,
        1.0e4 + material_energy,
        rtol=1.0e-12,
    )
    assert bool(result.successful)
