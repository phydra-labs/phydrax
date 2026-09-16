#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def test_antoine_pressure_temperature_inverse_and_domain_evidence():
    plan = phx.equations.AntoineSaturationPressurePlan(
        8.07131,
        1730.63,
        233.426,
        temperature_interval=(273.15, 373.15),
    )
    temperature = jnp.asarray([280.0, 320.0, 370.0])
    pressure = plan.evaluate_pressure(temperature)
    recovered = plan.evaluate_temperature(pressure.value)

    assert jnp.all(pressure.successful)
    assert jnp.all(recovered.successful)
    np.testing.assert_allclose(recovered.value, temperature, rtol=2e-6)
    assert not bool(plan.evaluate_pressure(jnp.asarray(400.0)).successful)


def _solid_liquid_material(*, width=0.0):
    return phx.equations.SolidLiquidEnthalpyPlan(
        1000.0,
        273.15,
        300.0,
        300.0 + width,
        2000.0,
        2500.0,
        2.0e5,
        2.0,
        0.5,
        1.0e-6,
        thermal_expansion=2.0e-4,
    )


def test_solid_liquid_enthalpy_roundtrip_and_isothermal_latent_plateau():
    material = _solid_liquid_material()
    latent_fraction = jnp.asarray([0.0, 0.25, 0.75, 1.0])
    enthalpy = material.enthalpy_from_temperature(
        jnp.full_like(latent_fraction, 300.0),
        liquid_fraction=latent_fraction,
    )
    state = material.evaluate(enthalpy)

    np.testing.assert_allclose(state.temperature, 300.0)
    np.testing.assert_allclose(state.liquid_fraction, latent_fraction)
    assert jnp.all(state.temperature_enthalpy_derivative == 0.0)
    assert jnp.all(state.successful)
    assert jnp.all(state.mushy_resistance >= 0.0)

    regularized = _solid_liquid_material(width=4.0)
    temperature = jnp.asarray([295.0, 301.0, 303.0, 307.0])
    enthalpy = regularized.enthalpy_from_temperature(temperature)
    recovered = regularized.evaluate(enthalpy)
    np.testing.assert_allclose(recovered.temperature, temperature, rtol=1e-6)
    assert jnp.all(recovered.temperature_enthalpy_derivative > 0.0)
    gradient = jax.grad(lambda value: jnp.sum(regularized.evaluate(value).temperature))(
        enthalpy
    )
    assert jnp.all(jnp.isfinite(gradient))


def test_binary_alloy_closure_conserves_declared_enthalpy_and_partitions_solute():
    plan = phx.equations.BinaryAlloyPhaseDiagramPlan(
        1000.0,
        273.15,
        330.0,
        -50.0,
        0.3,
        1.5,
        2000.0,
        2.0e5,
        2.0,
        0.5,
        1.0e-6,
        1.0e-10,
        1.0e-8,
    )
    concentration = jnp.asarray([0.02, 0.05])
    target_temperature = jnp.asarray([320.0, 340.0])
    liquid_fraction = 0.5 * (
        1.0
        + jnp.tanh(
            (
                target_temperature
                - (plan.melting_temperature + plan.liquidus_slope * concentration)
            )
            / plan.smoothing_width
        )
    )
    enthalpy = (
        plan.reference_density
        * plan.heat_capacity
        * (target_temperature - plan.reference_temperature)
        + plan.reference_density * plan.latent_heat * liquid_fraction
    )
    state = plan.evaluate(enthalpy, concentration)

    np.testing.assert_allclose(state.temperature, target_temperature, rtol=2e-5)
    np.testing.assert_allclose(
        state.liquid_fraction * state.liquid_concentration
        + (1.0 - state.liquid_fraction) * state.solid_concentration,
        concentration,
        rtol=2e-6,
    )
    assert jnp.all(state.successful)


def test_homogeneous_equilibrium_cavitation_barotrope_is_hyperbolic_and_energetic():
    material = phx.equations.HomogeneousEquilibriumCavitationMaterial(
        1.0e5,
        1.0,
        1000.0,
        300.0,
        1400.0,
        mixture_model="wallis",
        pressure_floor=-1.0e6,
    )
    density = jnp.asarray([0.8, 1.0, 50.0, 500.0, 1000.0, 1050.0])
    state = material.evaluate(density)

    assert jnp.all(state.successful)
    assert jnp.all(state.sound_speed > 0.0)
    assert jnp.all((state.vapor_fraction >= 0.0) & (state.vapor_fraction <= 1.0))
    np.testing.assert_allclose(
        material.density_from_pressure(material.pressure(density)),
        density,
        rtol=2e-5,
    )
    for value in (jnp.asarray(0.8), jnp.asarray(50.0), jnp.asarray(1050.0)):
        derivative = jax.grad(material.specific_internal_energy)(value)
        np.testing.assert_allclose(
            derivative,
            material.pressure(value) / value**2,
            rtol=2e-4,
            atol=2e-4,
        )

    system = phx.equations.BarotropicEulerSystem(2, material=material)
    primitive = jnp.asarray([500.0, 1.0, -0.5])
    conserved = system.primitive_to_conserved(primitive)
    np.testing.assert_allclose(system.conserved_to_primitive(conserved), primitive)
    assert system.admissible(conserved)
    assert jnp.all(jnp.isfinite(system.physical_flux(conserved, 0)))


def _two_material_system():
    eos = phx.equations.TwoMaterialEOSClosure(
        phx.equations.StiffenedGasMaterial(4.4, 2.0e5, 1800.0),
        phx.equations.StiffenedGasMaterial(1.33, 0.0, 1400.0, reference_energy=2.0e6),
    )
    return phx.equations.TwoMaterialVOFSystem(1, eos=eos)


def test_vof_phase_transfer_has_one_mass_owner_and_preserves_total_energy():
    system = _two_material_system()
    primitive = jnp.asarray([[900.0, 2.0, 0.0, 8.0e4, 0.8]])
    state = system.primitive_to_conserved(primitive)
    law = phx.equations.MerkleCavitationPlan(1.0e5, 1.0e-10, 1.0e-10, 2.0e6)
    plan = phx.equations.TwoMaterialVOFPhaseChangePlan(system, law)
    result = plan.step(state, 1.0e-4)

    assert jnp.all(result.accepted)
    np.testing.assert_allclose(
        jnp.sum(result.state[..., :2], axis=-1),
        jnp.sum(state[..., :2], axis=-1),
    )
    np.testing.assert_allclose(
        result.state[..., system.layout.energy_index],
        state[..., system.layout.energy_index],
    )
    assert jnp.all((result.state[..., system.alpha_index] >= 0.0))
    assert jnp.all((result.state[..., system.alpha_index] <= 1.0))
    assert jnp.max(jnp.abs(result.transfer.mass_defect)) == 0.0

    thermal = phx.equations.InterfaceHeatResistancePhaseChangePlan(350.0, 10.0, 2.0e6)
    thermal_plan = phx.equations.TwoMaterialVOFPhaseChangePlan(system, thermal)
    source = thermal_plan.differential_source(
        state, interface_area_density=jnp.asarray([2.0])
    )
    expected_sign = jnp.sign(system.eos.temperature(state) - 350.0)
    assert jnp.all(jnp.sign(source.transfer.raw_mass_rate) == expected_sign)
    assert jnp.all(jnp.isfinite(source.state_rate))


def test_unstructured_thermal_boundaries_close_heat_content_balance():
    eos = phx.equations.TwoMaterialEOSClosure(
        phx.equations.IdealGasMaterial(1.4),
        phx.equations.StiffenedGasMaterial(4.4, 2.0, 1.0),
    )
    system = phx.equations.TwoMaterialVOFSystem(2, eos=eos)
    discretization = phx.discretization.UnstructuredFiniteVolumePlan(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))),
        quadrilaterals=np.asarray(((0, 1, 2, 3),), dtype=np.int32),
        component_names=system.component_names,
    ).prepare()
    primitive = jnp.asarray([[1.2, 0.7, 0.0, 0.0, 2.5, 0.6]])
    state = system.primitive_to_conserved(primitive)
    temperature = float(system.eos.temperature(state)[0])

    heated = phx.discretization.UnstructuredTwoMaterialThermalDiffusionPlan(
        discretization,
        2.0,
        0.5,
        boundaries={
            "boundary": phx.discretization.UnstructuredThermalBoundaryCondition(
                "temperature", temperature + 10.0
            )
        },
    ).evaluate(system, state)
    assert heated.successful
    assert heated.cell_energy_rate[0] > 0.0
    assert heated.boundary_energy_rate > 0.0
    assert jnp.abs(heated.conservation_defect) < 1.0e-12

    cooled = phx.discretization.UnstructuredTwoMaterialThermalDiffusionPlan(
        discretization,
        2.0,
        0.5,
        boundaries={
            "boundary": phx.discretization.UnstructuredThermalBoundaryCondition(
                "heat_flux", 5.0
            )
        },
    ).evaluate(system, state)
    assert cooled.successful
    assert cooled.cell_energy_rate[0] < 0.0
    assert cooled.boundary_energy_rate < 0.0
    assert jnp.abs(cooled.conservation_defect) < 1.0e-12
