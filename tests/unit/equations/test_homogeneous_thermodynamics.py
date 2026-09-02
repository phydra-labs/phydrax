#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax.numpy as jnp
import numpy as np

import phydrax as phx


R = phx.equations.UNIVERSAL_GAS_CONSTANT


def _model():
    schema = phx.equations.ChemicalSpeciesSchema.from_unique_species(
        ("A", "B"),
        (
            phx.equations.ChemicalPhaseKind.GAS,
            phx.equations.ChemicalPhaseKind.GAS,
        ),
        jnp.asarray((0.002, 0.028)),
        ("A", "B"),
        jnp.asarray(((1, 0), (0, 1)), dtype=jnp.int32),
        jnp.asarray((0, 0), dtype=jnp.int32),
        gas_standard_pressure=1.0e5,
    )
    thermodynamics = phx.equations.PolynomialSpeciesThermodynamicsPlan(
        schema,
        jnp.asarray(((2.5 * R,), (3.0 * R,))),
        jnp.asarray((1.0e4, 2.0e4)),
        reference_molar_entropy=jnp.asarray((100.0, 120.0)),
        reference_temperature=300.0,
        minimum_temperature=200.0,
        maximum_temperature=2000.0,
    )
    ideal = phx.equations.IdealGasReferenceHelmholtzTerm(schema, thermodynamics)
    return phx.equations.HomogeneousHelmholtzPlan(
        ideal, phx.equations.ZeroResidualHelmholtzTerm(schema)
    )


def test_ideal_mixture_helmholtz_derives_consistent_caloric_state():
    model = _model()
    temperature = jnp.asarray(700.0)
    density = jnp.asarray(3.0)
    composition = jnp.asarray((0.25, 0.75))

    state = model.evaluate(temperature, density, composition)
    chemical = model.evaluate_chemical(temperature, density, composition)

    np.testing.assert_allclose(state.pressure, density * R * temperature, rtol=1e-11)
    np.testing.assert_allclose(
        state.molar_enthalpy,
        state.molar_internal_energy + state.pressure / density,
        rtol=1e-11,
    )
    np.testing.assert_allclose(
        state.molar_gibbs_energy,
        jnp.sum(composition * chemical.chemical_potential),
        rtol=1e-9,
    )
    np.testing.assert_allclose(chemical.log_fugacity_coefficient, 0.0, atol=1e-10)
    assert bool(state.evidence.successful)
    assert bool(chemical.successful)


def test_density_energy_round_trip_is_batched_and_jittable():
    model = _model()
    species_density = jnp.asarray(((0.2, 0.8), (0.1, 0.5)))
    temperature = jnp.asarray((500.0, 1200.0))
    direct = model.evaluate_density_temperature(species_density, temperature)
    molar_density = jnp.sum(species_density / model.schema.molar_masses, axis=-1)
    energy_density = molar_density * direct.molar_internal_energy

    solved = eqx.filter_jit(model.solve_density_energy)(species_density, energy_density)

    np.testing.assert_allclose(solved.state.temperature, temperature, rtol=1e-10)
    np.testing.assert_allclose(solved.energy_residual, 0.0, atol=1e-6)
    np.testing.assert_array_equal(solved.successful, (True, True))


def test_thermodynamic_validity_is_per_batch_entry():
    model = _model()
    state = model.evaluate(
        jnp.asarray((500.0, 5000.0)),
        jnp.asarray((2.0, 2.0)),
        jnp.asarray(((0.5, 0.5), (0.5, 0.5))),
    )

    np.testing.assert_array_equal(state.evidence.successful, (True, False))


def test_component_catalog_supports_repeated_phase_occurrences():
    catalog = phx.equations.ChemicalComponentCatalog(
        ("water",),
        jnp.asarray((0.01801528,)),
        ("H", "O"),
        jnp.asarray(((2,), (1,)), dtype=jnp.int32),
        charges=jnp.asarray((0,), dtype=jnp.int32),
    )
    gas = phx.equations.ChemicalPhaseSpec(
        "vapor",
        phx.equations.ChemicalPhaseKind.GAS,
        3,
        standard_pressure=1.0e5,
    )
    liquid_one = phx.equations.ChemicalPhaseSpec(
        "liquid-one", phx.equations.ChemicalPhaseKind.LIQUID, 3
    )
    liquid_two = phx.equations.ChemicalPhaseSpec(
        "liquid-two", phx.equations.ChemicalPhaseKind.LIQUID, 3
    )
    schema = phx.equations.ChemicalSpeciesSchema(
        catalog,
        ("H2O(g)", "H2O(l1)", "H2O(l2)"),
        jnp.asarray((0, 0, 0), dtype=jnp.int32),
        (gas, liquid_one, liquid_two),
        jnp.asarray((0, 1, 2), dtype=jnp.int32),
    )

    np.testing.assert_allclose(schema.component_amount(jnp.asarray((1.0, 2.0, 3.0))), 6.0)
    assert schema.phase_count == 3
    assert schema.phase_slot_species_indices(2) == (2,)


def test_homogeneous_mixture_euler_round_trip_flux_and_reflection():
    model = _model()
    system = phx.equations.HomogeneousMixtureEulerSystem(model, 2)
    primitive = jnp.asarray((0.2, 0.8, 3.0, -1.0, 900.0))

    state = system.primitive_to_conserved(primitive)
    recovered = system.conserved_to_primitive(state)
    flux = system.physical_flux(state, 0)
    reflected = system.reflect_normal_state(state, jnp.asarray((1.0, 0.0)))

    np.testing.assert_allclose(recovered, primitive, rtol=1e-10)
    assert flux.shape == state.shape
    assert bool(system.admissible(state))
    np.testing.assert_allclose(
        reflected[system.species_count : -1],
        jnp.asarray((-state[system.species_count], state[system.species_count + 1])),
    )
