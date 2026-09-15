#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._physical import RelativityScaleContract
from phydrax.equations._relativistic_eos import (
    GammaLawEOS,
    HybridColdThermalEOS,
    PiecewisePolytropicEOS,
    RELATIVISTIC_EOS_COLD_CONSTRAINT_MISMATCH,
    RELATIVISTIC_EOS_COMPOSITION_BELOW_DOMAIN,
    RELATIVISTIC_EOS_DENSITY_BELOW_DOMAIN,
    RELATIVISTIC_EOS_SUCCESS,
    RELATIVISTIC_EOS_THERMAL_ABOVE_DOMAIN,
    RELATIVISTIC_EOS_THERMAL_BELOW_DOMAIN,
    TabulatedFiniteTemperatureEOS,
)
from phydrax.units import KILOGRAM


def _geometric_scale():
    return RelativityScaleContract.geometric(KILOGRAM)


def _cold_eos():
    return PiecewisePolytropicEOS(
        _geometric_scale(),
        jnp.asarray((1.0,)),
        jnp.asarray((1.5, 2.0)),
        0.1,
        maximum_density=8.0,
    )


def _tabulated_eos(*, pressure_scale=1.0):
    density = np.asarray((1.0, 2.0, 4.0))
    temperature = np.asarray((1.0, 2.0, 4.0))
    composition = np.asarray((0.1, 0.4, 0.8))
    rho, thermal, fraction = np.meshgrid(density, temperature, composition, indexing="ij")
    specific_energy = 0.15 + 0.2 * thermal + 0.05 * fraction
    pressure = pressure_scale * 0.4 * rho * specific_energy
    return TabulatedFiniteTemperatureEOS(
        _geometric_scale(),
        density,
        temperature,
        composition,
        pressure,
        specific_energy,
        provenance="synthetic analytic gamma-law table",
        source_checksum="0" * 64,
        license_id="CC0-1.0",
    )


def test_gamma_law_state_is_thermodynamically_consistent_and_vector_differentiable():
    scale = _geometric_scale()
    eos = GammaLawEOS(
        scale,
        5.0 / 3.0,
        minimum_density=0.1,
        maximum_density=10.0,
        maximum_specific_internal_energy=5.0,
    )
    density = jnp.asarray((0.5, 1.0, 2.0))
    energy = jnp.asarray((0.1, 0.3, 0.7))
    state = eos.evaluate(density, energy)

    np.testing.assert_allclose(state.pressure, (2.0 / 3.0) * density * energy)
    np.testing.assert_allclose(state.total_energy_density, density * (1.0 + energy))
    np.testing.assert_allclose(
        state.specific_enthalpy,
        1.0 + energy + state.pressure / density,
    )
    np.testing.assert_allclose(
        state.sound_speed_squared,
        (5.0 / 3.0) * state.pressure / (density * state.specific_enthalpy),
    )
    np.testing.assert_allclose(
        eos.sound_speed(density, energy) ** 2, state.sound_speed_squared
    )
    assert bool(jnp.all(state.qualified))
    assert bool(jnp.all(state.derivative_valid))
    assert jnp.all(state.status == RELATIVISTIC_EOS_SUCCESS)
    assert eos.rest_mass_density_unit.unit_id == scale.mass_density_unit.unit_id
    assert eos.pressure_unit.unit_id == scale.energy_density_unit.unit_id
    assert eos.specific_energy_unit.unit_id == scale.specific_energy_unit.unit_id
    assert eos.sound_speed_unit.unit_id == scale.dimensional_scale.velocity_unit.unit_id

    pressure_energy_gradient = jax.vmap(
        jax.grad(lambda value, rho: eos.pressure(rho, value)), in_axes=(0, 0)
    )(energy, density)
    np.testing.assert_allclose(pressure_energy_gradient, (2.0 / 3.0) * density)

    pressure_state = eos.evaluate_pressure(density, state.pressure)
    np.testing.assert_allclose(pressure_state.specific_internal_energy, energy)


def test_gamma_law_reports_exact_domain_failures_without_repairing_inputs():
    eos = GammaLawEOS(
        _geometric_scale(),
        4.0 / 3.0,
        minimum_density=0.1,
        maximum_specific_internal_energy=1.0,
    )
    state = eos.evaluate(
        jnp.asarray((0.09, 1.0, 1.0)),
        jnp.asarray((0.2, 1.1, 0.2)),
    )

    assert int(state.status[0]) == RELATIVISTIC_EOS_DENSITY_BELOW_DOMAIN
    assert int(state.status[1]) == RELATIVISTIC_EOS_THERMAL_ABOVE_DOMAIN
    assert int(state.status[2]) == RELATIVISTIC_EOS_SUCCESS
    assert state.rest_mass_density[0] == pytest.approx(0.09)
    assert state.specific_internal_energy[1] == pytest.approx(1.1)


def test_piecewise_polytrope_is_continuous_and_obeys_cold_first_law():
    eos = _cold_eos()
    break_density = jnp.asarray(1.0)
    left = eos.evaluate(break_density - 1.0e-6)
    right = eos.evaluate(break_density + 1.0e-6)
    at_break = eos.evaluate(break_density)

    np.testing.assert_allclose(left.pressure, right.pressure, rtol=4.0e-6)
    np.testing.assert_allclose(
        left.specific_internal_energy,
        right.specific_internal_energy,
        rtol=4.0e-6,
    )
    assert bool(at_break.physically_valid)
    assert not bool(at_break.derivative_valid)

    densities = jnp.asarray((0.5, 2.0, 4.0))
    energy_gradient = jax.vmap(
        jax.grad(lambda density: eos.evaluate(density).specific_internal_energy)
    )(densities)
    state = eos.evaluate(densities)
    np.testing.assert_allclose(eos.pressure(densities), state.pressure)
    np.testing.assert_allclose(
        energy_gradient,
        state.pressure / densities**2,
        rtol=2.0e-6,
    )

    inconsistent = eos.evaluate(jnp.asarray(2.0), jnp.asarray(0.0))
    assert int(inconsistent.status) == RELATIVISTIC_EOS_COLD_CONSTRAINT_MISMATCH
    assert not bool(inconsistent.physically_valid)


def test_hybrid_eos_uses_cold_energy_as_an_explicit_thermal_boundary():
    cold = _cold_eos()
    eos = HybridColdThermalEOS(cold, 1.5)
    density = jnp.asarray((0.6, 2.0, 3.0))
    cold_state = cold.evaluate(density)
    energy = cold_state.specific_internal_energy + jnp.asarray((0.1, 0.2, 0.4))
    state = eos.evaluate(density, energy)

    expected_pressure = cold_state.pressure + 0.5 * density * (
        energy - cold_state.specific_internal_energy
    )
    np.testing.assert_allclose(state.pressure, expected_pressure)
    np.testing.assert_allclose(
        state.sound_speed_squared,
        (
            state.pressure_density_derivative
            + state.pressure
            * state.pressure_specific_internal_energy_derivative
            / density**2
        )
        / state.specific_enthalpy,
    )
    pressure_state = eos.evaluate_pressure(density, state.pressure)
    np.testing.assert_allclose(
        pressure_state.specific_internal_energy, energy, rtol=2.0e-6
    )

    below = eos.evaluate(
        jnp.asarray(2.0),
        cold.evaluate(jnp.asarray(2.0)).specific_internal_energy - 0.01,
    )
    assert int(below.status) == RELATIVISTIC_EOS_THERMAL_BELOW_DOMAIN
    assert below.pressure != cold.evaluate(jnp.asarray(2.0)).pressure


def test_tabulated_eos_has_bounded_fixed_interpolation_and_smooth_branch_gradients():
    eos = _tabulated_eos()
    density = jnp.asarray((1.25, 1.5, 3.0))
    temperature = jnp.asarray((1.25, 1.5, 3.0))
    composition = jnp.asarray((0.2, 0.3, 0.6))
    state = eqx.filter_jit(eos.evaluate_temperature)(density, temperature, composition)
    expected_energy = 0.15 + 0.2 * temperature + 0.05 * composition

    np.testing.assert_allclose(state.specific_internal_energy, expected_energy)
    np.testing.assert_allclose(state.pressure, 0.4 * density * expected_energy)
    np.testing.assert_allclose(
        state.sound_speed_squared,
        (
            state.pressure_density_derivative
            + state.pressure
            * state.pressure_specific_internal_energy_derivative
            / density**2
        )
        / state.specific_enthalpy,
    )
    assert bool(jnp.all(state.qualified))
    assert bool(jnp.all(state.derivative_valid))
    knot = eos.evaluate_temperature(jnp.asarray(2.0), jnp.asarray(2.0), jnp.asarray(0.4))
    assert bool(knot.qualified)
    assert not bool(knot.derivative_valid)

    pressure_density_gradient = jax.vmap(
        jax.grad(
            lambda rho, thermal, fraction: (
                eos.evaluate_temperature(rho, thermal, fraction).pressure
            )
        ),
        in_axes=(0, 0, 0),
    )(density, temperature, composition)
    np.testing.assert_allclose(pressure_density_gradient, 0.4 * expected_energy)

    inverse = eos.evaluate(density, expected_energy, composition)
    np.testing.assert_allclose(inverse.temperature, temperature, rtol=2.0e-6)
    np.testing.assert_allclose(
        inverse.specific_internal_energy, expected_energy, rtol=2.0e-6
    )
    pressure_inverse = eos.evaluate_pressure(density, state.pressure, composition)
    np.testing.assert_allclose(
        pressure_inverse.specific_internal_energy, expected_energy, rtol=2.0e-6
    )


def test_tabulated_eos_support_endpoints_are_not_derivative_valid():
    eos = _tabulated_eos()
    states = eos.evaluate_temperature(
        jnp.asarray((1.0, 4.0, 1.5, 1.5, 1.5, 1.5)),
        jnp.asarray((1.5, 1.5, 1.0, 4.0, 1.5, 1.5)),
        jnp.asarray((0.2, 0.2, 0.2, 0.2, 0.1, 0.8)),
    )

    assert bool(jnp.all(states.qualified))
    assert not bool(jnp.any(states.derivative_valid))


def test_tabulated_eos_reports_each_support_failure_and_never_returns_boundary_values():
    eos = _tabulated_eos()
    state = eos.evaluate_temperature(
        jnp.asarray((0.5, 2.0, 2.0, 2.0)),
        jnp.asarray((2.0, 0.5, 2.0, 2.0)),
        jnp.asarray((0.4, 0.4, 0.0, 0.4)),
    )

    assert int(state.status[0]) == RELATIVISTIC_EOS_DENSITY_BELOW_DOMAIN
    assert int(state.status[1]) == RELATIVISTIC_EOS_THERMAL_BELOW_DOMAIN
    assert int(state.status[2]) == RELATIVISTIC_EOS_COMPOSITION_BELOW_DOMAIN
    assert int(state.status[3]) == RELATIVISTIC_EOS_SUCCESS
    assert bool(jnp.all(jnp.isnan(state.pressure[:3])))
    assert not bool(jnp.any(state.qualified[:3]))

    below_energy = eos.evaluate(jnp.asarray(2.0), jnp.asarray(0.1), jnp.asarray(0.4))
    assert int(below_energy.status) == RELATIVISTIC_EOS_THERMAL_BELOW_DOMAIN
    assert bool(jnp.isnan(below_energy.pressure))
    assert bool(jnp.isnan(below_energy.temperature))


def test_tabulated_identity_binds_all_numeric_content_and_host_evidence():
    base = _tabulated_eos()
    changed = _tabulated_eos(pressure_scale=1.01)

    assert base.table_id != changed.table_id
    assert base.eos_id != changed.eos_id
    assert base.table_evidence.qualified
    assert base.table_evidence.minimum_heat_capacity > 0.0
    assert base.table_evidence.minimum_adiabatic_pressure_derivative > 0.0
    assert base.table_evidence.minimum_causality_margin >= 0.0


def test_tabulated_eos_rejects_nonmonotone_or_acausal_content_on_host():
    density = np.asarray((1.0, 2.0, 4.0))
    temperature = np.asarray((1.0, 2.0, 4.0))
    composition = np.asarray((0.1, 0.4, 0.8))
    rho, thermal, fraction = np.meshgrid(density, temperature, composition, indexing="ij")
    energy = 0.1 + 0.2 * thermal + 0.01 * fraction
    pressure = 0.4 * rho * energy
    pressure[1, 1, 1] = 0.5 * pressure[0, 1, 1]

    with pytest.raises(ValueError, match="nondecreasing with density"):
        TabulatedFiniteTemperatureEOS(
            _geometric_scale(),
            density,
            temperature,
            composition,
            pressure,
            energy,
            provenance="invalid synthetic table",
            source_checksum="1" * 64,
            license_id="CC0-1.0",
        )

    with pytest.raises(ValueError, match="stable and causal"):
        _tabulated_eos(pressure_scale=4.0)
