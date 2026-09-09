import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _ionized_system():
    electron_mass = 5.48579909065e-7
    schema = phx.equations.ChemicalSpeciesSchema.from_unique_species(
        ("N", "N+", "e-"),
        (phx.equations.ChemicalPhaseKind.GAS,) * 3,
        jnp.asarray((0.014, 0.014 - electron_mass, electron_mass)),
        ("N",),
        jnp.asarray(((1, 1, 0),), dtype=jnp.int32),
        jnp.asarray((0, 1, -1), dtype=jnp.int32),
        gas_standard_pressure=1.0e5,
    )
    heavy = phx.equations.PolynomialSpeciesThermodynamicsPlan(
        schema,
        jnp.asarray((2.5, 2.5, 1.5)) * phx.equations.UNIVERSAL_GAS_CONSTANT,
        jnp.asarray((0.0, 1.0e6, 0.0)),
        reference_temperature=300.0,
        minimum_temperature=100.0,
        maximum_temperature=30000.0,
    )
    modes = phx.equations.ThermalModeSchema(
        schema,
        (
            phx.equations.ThermalModeSpec(
                "vibration",
                jnp.asarray((3390.0, 3390.0, 0.0)),
                minimum_temperature=100.0,
                maximum_temperature=30000.0,
            ),
            phx.equations.ThermalModeSpec(
                "electron-translation",
                jnp.asarray((0.0, 0.0, 1.0)),
                kind="ideal-degrees-of-freedom",
                degrees_of_freedom=3.0,
                pressure_bearing=True,
                translational_work=True,
                minimum_temperature=100.0,
                maximum_temperature=50000.0,
            ),
        ),
    )
    base = phx.equations.TwoTemperatureThermodynamicsPlan(heavy, modes)
    thermodynamics = phx.equations.IonizedMixtureThermodynamicsPlan(base, 2, 1)
    return phx.equations.IonizedMultitemperatureEulerSystem(thermodynamics, 1)


def _neutral_primitive(system):
    ion_density = 1.0e-4
    electron_density = (
        ion_density
        / system.thermodynamics.schema.molar_masses[1]
        * system.thermodynamics.schema.molar_masses[2]
    )
    return jnp.asarray(
        (0.7, ion_density, electron_density, 100.0, 6000.0, 4000.0, 10000.0)
    )


def test_ionized_state_recovers_electron_pressure_and_charge():
    system = _ionized_system()
    conserved = system.primitive_to_conserved(_neutral_primitive(system))
    recovered = system.recover_thermodynamics(conserved)

    assert bool(recovered.successful)
    assert recovered.electron_pressure > 0.0
    np.testing.assert_allclose(
        recovered.total_pressure,
        recovered.heavy_pressure + recovered.electron_pressure,
        rtol=2.0e-7,
    )
    assert recovered.neutrality.relative_defect < 1.0e-7
    assert bool(system.admissible(conserved))


def test_ambipolar_transport_enforces_mass_and_current_constraints():
    system = _ionized_system()
    schema = system.thermodynamics.schema
    transport = phx.equations.AmbipolarPlasmaTransportPlan(
        schema,
        phx.equations.ConstantTransport(2.0e-5, 0.03),
        jnp.asarray((1.0e-4, 1.5e-4, 0.1)),
        electron_thermal_conductivity=0.2,
    )
    density = jnp.asarray(0.8)
    mass = jnp.asarray((0.999, 0.00099, 0.00001))
    gradient = jnp.asarray(((0.1,), (-0.04,), (-0.06,)))
    result = transport.evaluate(
        density,
        mass,
        gradient,
        6000.0,
        10000.0,
        jnp.asarray((20.0,)),
        jnp.asarray((30.0,)),
        2.0e5,
    )

    assert bool(result.successful)
    np.testing.assert_allclose(result.mass_flux_defect, 0.0, atol=2.0e-8)
    np.testing.assert_allclose(result.current_density, 0.0, atol=1.0e-6)
    assert result.entropy_production >= 0.0


def test_nonlte_levels_conserve_species_population_and_emit_groups():
    population = phx.equations.NonLTELevelPopulationPlan(
        2,
        jnp.asarray((0, 0, 1, 1)),
        jnp.asarray((0.0, 1.0e5, 0.0, 2.0e5)),
        jnp.asarray((1.0, 3.0, 1.0, 2.0)),
    )
    coefficient = phx.equations.NonLTERadiationCoefficientPlan(
        population,
        2,
        jnp.asarray((0, 2)),
        jnp.asarray((1, 3)),
        jnp.asarray((0, 1)),
        jnp.asarray((1.0e14, 2.0e14)),
        jnp.asarray((1.0e7, 2.0e7)),
        jnp.asarray((1.0e-20, 2.0e-20)),
    )
    result = coefficient.evaluate(jnp.asarray((2.0, 1.0)), 8000.0)

    assert bool(result.successful)
    np.testing.assert_allclose(
        jnp.sum(result.level_populations.level_molar_densities[:2]),
        2.0,
        rtol=2.0e-7,
    )
    assert jnp.all(result.absorption >= 0.0)
    assert jnp.all(result.emission_power_density >= 0.0)


def test_fixed_work_plasma_source_preserves_zero_rate_state_and_ledgers():
    system = _ionized_system()
    heavy = system.thermodynamics.base.heavy_thermodynamics
    reaction = phx.equations.ChemicalReactionSpec(
        "ionization",
        {"N": 1.0},
        {"N+": 1.0, "e-": 1.0},
        phx.equations.ArrheniusRatePlan(0.0),
    )
    mechanism = phx.equations.ChemicalMechanismIR(
        "zero-ionization",
        system.thermodynamics.schema,
        heavy,
        (reaction,),
    ).prepare()
    plasma = phx.equations.PreparedPlasmaMechanism(
        mechanism,
        (phx.equations.ReactionTemperatureSpec("electron"),),
        mode_energy_per_progress=jnp.zeros((1, system.mode_count)),
    )
    source = phx.solver.FixedWorkThermochemicalSourcePlan(
        plasma, substeps=2, newton_iterations=3
    )
    incoming = system.primitive_to_conserved(_neutral_primitive(system))
    result = source.advance(system, incoming, 1.0e-5)

    assert bool(result.evidence.successful)
    np.testing.assert_allclose(result.accepted, incoming, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(result.evidence.element_defect, 0.0, atol=0.0)
    np.testing.assert_allclose(result.evidence.charge_defect, 0.0, atol=0.0)
    np.testing.assert_allclose(result.evidence.energy_defect, 0.0, atol=0.0)
