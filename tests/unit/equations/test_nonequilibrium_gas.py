import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _thermodynamics():
    schema = phx.equations.ChemicalSpeciesSchema.from_unique_species(
        ("A2", "A"),
        (phx.equations.ChemicalPhaseKind.GAS,) * 2,
        jnp.asarray((0.028, 0.014)),
        ("A",),
        jnp.asarray(((2, 1),), dtype=jnp.int32),
        jnp.zeros((2,), dtype=jnp.int32),
        gas_standard_pressure=1.0e5,
    )
    heavy = phx.equations.PolynomialSpeciesThermodynamicsPlan(
        schema,
        jnp.asarray(
            (
                2.5 * phx.equations.UNIVERSAL_GAS_CONSTANT,
                1.5 * phx.equations.UNIVERSAL_GAS_CONSTANT,
            )
        ),
        jnp.asarray((0.0, 2.0e4)),
        reference_temperature=300.0,
        minimum_temperature=100.0,
        maximum_temperature=10000.0,
    )
    modes = phx.equations.ThermalModeSchema(
        schema,
        (
            phx.equations.ThermalModeSpec(
                "vibrational-electronic",
                jnp.asarray((3390.0, 5000.0)),
                minimum_temperature=100.0,
                maximum_temperature=20000.0,
            ),
        ),
    )
    return phx.equations.TwoTemperatureThermodynamicsPlan(
        heavy, modes, maximum_iterations=80
    )


def test_thermal_mode_energy_temperature_roundtrip_and_equilibrium_projection():
    thermodynamics = _thermodynamics()
    species_density = jnp.asarray((0.6, 0.2))
    mode_temperature = jnp.asarray((1800.0,))
    mode = thermodynamics.modes.evaluate(species_density, mode_temperature)
    recovered = thermodynamics.modes.solve_temperatures(
        species_density, mode.energy_densities
    )

    assert bool(recovered.successful)
    np.testing.assert_allclose(
        recovered.temperatures, mode_temperature, rtol=2.0e-6, atol=2.0e-3
    )
    temperature = jnp.asarray(1400.0)
    equilibrium = thermodynamics.evaluate(species_density, temperature, temperature[None])
    np.testing.assert_allclose(
        thermodynamics.equilibrium_internal_energy_density(species_density, temperature),
        equilibrium.heavy_internal_energy_density
        + jnp.sum(equilibrium.mode_energy_densities),
        rtol=2.0e-7,
    )


def test_two_temperature_euler_roundtrip_flux_and_admissibility():
    system = phx.equations.TwoTemperatureMixtureEulerSystem(_thermodynamics(), 1)
    primitive = jnp.asarray((0.6, 0.2, 300.0, 1200.0, 1800.0))
    conserved = system.primitive_to_conserved(primitive)
    recovered = system.conserved_to_primitive(conserved)

    assert bool(system.admissible(conserved))
    np.testing.assert_allclose(recovered, primitive, rtol=3.0e-6, atol=3.0e-3)
    flux = system.physical_flux(conserved, 0)
    assert flux.shape == conserved.shape
    assert jnp.all(jnp.isfinite(flux))
    assert system.frozen_sound_speed(conserved) > 0.0


def test_two_temperature_navier_stokes_diffuses_modal_energy_consistently():
    thermodynamics = _thermodynamics()
    system = phx.equations.TwoTemperatureMixtureNavierStokesSystem(
        thermodynamics,
        phx.equations.ConstantTransport(2.0e-5, 0.03),
        1,
        mode_diffusivities=(4.0e-5,),
    )

    def state_at(coordinate):
        primitive = jnp.asarray(
            (
                0.6,
                0.2,
                20.0 + coordinate,
                1200.0 + 5.0 * coordinate,
                1800.0 + 8.0 * coordinate,
            )
        )
        return system.primitive_to_conserved(primitive)

    state = state_at(jnp.asarray(0.0))
    gradient = jax.jacfwd(state_at)(jnp.asarray(0.0))[..., None]
    flux = system.viscous_flux(state, gradient)

    assert flux.shape == (system.component_count, 1)
    assert jnp.all(jnp.isfinite(flux))
    assert flux[system.mode_slice, 0].item() != 0.0
    assert system.maximum_diffusivity(state) >= 4.0e-5
    np.testing.assert_allclose(
        flux[system.energy_index, 0] - 20.0 * flux[system.momentum_slice, 0].item(),
        0.03 * 5.0 + flux[system.mode_slice, 0].item(),
        rtol=4.0e-5,
        atol=4.0e-5,
    )
