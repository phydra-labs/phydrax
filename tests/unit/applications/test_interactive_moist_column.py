#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.atmosphere._interactive_column import InteractiveMoistColumnPlan
from phydrax.applications.atmosphere._moist import MoistThermodynamicPlan
from phydrax.applications.atmosphere._radiation import (
    ColumnOpticalProperties,
    ColumnRadiationPlan,
)
from phydrax.applications.atmosphere._surface import BulkSurfaceExchangePlan
from phydrax.metrix import EuclideanStateGeometry
from phydrax.solver._fixed_step import FixedStepProblem, FixedStepRolloutPlan


def _quiet(**kwargs):
    return InteractiveMoistColumnPlan(
        mixing_length=0.0, rain_fall_speed=0.0, snow_fall_speed=0.0, **kwargs
    )


def _radiation():
    optics = ColumnOpticalProperties(
        shortwave_absorption=(1e-5, 0.002, 0.04, 0.03),
        shortwave_scattering=(0.0, 0.0, 60.0, 30.0),
        longwave_absorption=(1e-4, 0.08, 50.0, 25.0),
        shortwave_asymmetry=(0.0, 0.0, 0.85, 0.7),
        reference_id="illustrative-grey-test-not-observational-calibration",
    )
    return ColumnRadiationPlan(optics)


def _same_state(actual, expected):
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_array_equal(a, b)


def test_fixed_composition_diagnosis_and_finite_condensation():
    plan = _quiet()
    initial = plan.initialize(
        jnp.asarray([100.0]), 2.0, 285.0, 100.0, rain_mass=0.1, surface_water_mass=10.0
    )
    np.testing.assert_allclose(plan.diagnose(initial).temperature, 285.0, atol=1e-12)
    result = plan.step(initial, 0.1)
    assert result.successful
    assert result.state.cloud_liquid_mass[0] > 0
    assert result.state.vapor_mass[0] < initial.vapor_mass[0]
    assert result.state.rain_mass[0] > 0
    assert plan.diagnose(result.state).temperature[0] > 285.0
    assert plan.diagnose(result.state).relative_humidity[0] > 1.0
    np.testing.assert_allclose(result.state.total_water, initial.total_water, atol=2e-13)
    np.testing.assert_allclose(result.state.total_energy, initial.total_energy, atol=2e-8)


def test_rain_has_adjacent_cell_time_of_flight_not_instant_fallout():
    thermo = MoistThermodynamicPlan()
    plan = InteractiveMoistColumnPlan(
        thermo,
        mixing_length=0.0,
        condensation_timescale=1e12,
        rain_evaporation_timescale=1e12,
        phase_conversion_timescale=1e12,
        autoconversion_timescale=1e12,
        rain_fall_speed=5.0,
        snow_fall_speed=0.0,
    )
    vapor = (
        thermo.saturation_pressure(290.0) * 100.0 / (thermo.vapor_gas_constant * 290.0)
    )
    initial = plan.initialize(
        jnp.full(3, 100.0), vapor, 290.0, 100.0, rain_mass=jnp.asarray([0.03, 0.0, 0.0])
    )
    one = plan.step(initial, 20.0)
    two = plan.step(one.state, 20.0)
    three = plan.step(two.state, 20.0)
    assert one.successful & two.successful & three.successful
    np.testing.assert_allclose(one.state.rain_mass, [0.0, 0.03, 0.0], atol=1e-13)
    np.testing.assert_allclose(two.state.rain_mass, [0.0, 0.0, 0.03], atol=1e-13)
    assert float(two.state.precipitated_water) < 1e-20
    np.testing.assert_allclose(three.state.precipitated_water, 0.03, atol=1e-13)
    np.testing.assert_allclose(three.state.total_energy, initial.total_energy, atol=2e-8)
    rejected = plan.step(initial, 20.01)
    assert not rejected.successful
    _same_state(rejected.state, initial)


def test_below_cloud_rain_evaporation_cools_and_retains_latent_energy():
    plan = _quiet(condensation_timescale=1e12, rain_evaporation_timescale=20.0)
    initial = plan.initialize(jnp.asarray([100.0]), 0.3, 290.0, 100.0, rain_mass=0.2)
    result = plan.step(initial, 0.1)
    assert result.successful
    assert 0 < result.rain_evaporated_water < 0.2
    assert result.state.rain_mass[0] < initial.rain_mass[0]
    np.testing.assert_allclose(
        result.state.vapor_mass - initial.vapor_mass,
        result.rain_evaporated_water,
        atol=1e-14,
    )
    assert plan.diagnose(result.state).temperature[0] < 290.0
    np.testing.assert_array_equal(result.state.internal_energy, initial.internal_energy)
    slow = eqx.tree_at(lambda p: p.rain_evaporation_timescale, plan, jnp.asarray(1e12))
    almost_inert = slow.step(initial, 0.1)
    assert almost_inert.successful
    assert almost_inert.rain_evaporated_water < result.rain_evaporated_water * 1e-8


@pytest.mark.parametrize(
    "temperature,rain,snow,warms", [(280.0, 0.0, 1.0, False), (260.0, 1.0, 0.0, True)]
)
def test_falling_phase_conversion_retains_caloric_energy(temperature, rain, snow, warms):
    plan = _quiet(
        condensation_timescale=1e12,
        rain_evaporation_timescale=1e12,
        phase_conversion_timescale=10.0,
    )
    initial = plan.initialize(
        jnp.asarray([100.0]), 0.2, temperature, 100.0, rain_mass=rain, snow_mass=snow
    )
    result = plan.step(initial, 1.0)
    assert result.successful
    changed = plan.diagnose(result.state).temperature[0] - temperature
    assert bool(changed > 0) == warms
    assert (
        (result.state.snow_mass[0] > snow)
        if warms
        else (result.state.rain_mass[0] > rain)
    )
    np.testing.assert_array_equal(result.state.internal_energy, initial.internal_energy)
    np.testing.assert_allclose(result.state.total_water, initial.total_water, atol=1e-13)


def test_radiative_environment_matches_independent_interface_fluxes():
    plan = _quiet(radiation=_radiation())
    initial = plan.initialize(
        jnp.asarray([100.0, 120.0]),
        jnp.asarray([0.2, 0.5]),
        jnp.asarray([280.0, 290.0]),
        100.0,
        surface_temperature=295.0,
    )
    view = plan.diagnose(initial)
    radiation = plan.radiation.evaluate(
        view.temperature,
        initial.layer_mass,
        initial.vapor_mass,
        initial.cloud_liquid_mass,
        initial.cloud_ice_mass,
        view.surface_temperature,
        350.0,
    )
    result = plan.step(initial, 1.0, solar_down=350.0)
    assert result.successful
    material_change = (
        jnp.sum(result.state.internal_energy - initial.internal_energy)
        + result.state.slab.energy
        - initial.slab.energy
    )
    np.testing.assert_allclose(material_change, -radiation.space_heating, atol=2e-8)
    np.testing.assert_allclose(
        result.state.environment_energy, radiation.space_heating, atol=1e-12
    )
    np.testing.assert_allclose(result.state.total_energy, initial.total_energy, atol=2e-8)


def test_exhausted_wet_slab_rejects_entire_transaction_even_with_rainfall():
    exchange = BulkSurfaceExchangePlan(
        stability="neutral",
        heat_transfer_coefficient=0.0,
        moisture_transfer_coefficient=0.1,
    )
    plan = _quiet(surface_exchange=exchange)
    plan = eqx.tree_at(lambda p: p.rain_fall_speed, plan, jnp.asarray(100.0))
    initial = plan.initialize(
        jnp.asarray([100.0]),
        0.01,
        290.0,
        100.0,
        surface_temperature=300.0,
        surface_water_mass=1e-8,
        rain_mass=0.5,
    )
    result = plan.step(initial, 1.0, wind_speed=50.0, heating_rate=10.0)
    assert not result.successful
    _same_state(result.state, initial)
    assert result.precipitated_water == 0 and result.surface_water_flux == 0


def test_restart_exactness_and_numeric_parameter_binding(tmp_path):
    plan = _quiet(radiation=_radiation(), surface_exchange=BulkSurfaceExchangePlan())
    initial = plan.initialize(
        jnp.asarray([100.0, 110.0]),
        jnp.asarray([0.2, 0.4]),
        jnp.asarray([285.0, 290.0]),
        100.0,
        surface_temperature=295.0,
    )
    first = plan.step(initial, 1.0, solar_down=330.0)
    assert first.successful
    path = plan.save_checkpoint(tmp_path / "interactive.phx", first.state)
    restored = plan.load_checkpoint(path)
    resumed = plan.step(restored, 1.0, solar_down=280.0)
    direct = plan.step(first.state, 1.0, solar_down=280.0)
    assert resumed.successful & direct.successful
    _same_state(resumed.state, direct.state)
    different = eqx.tree_at(
        lambda p: p.condensation_timescale, plan, plan.condensation_timescale * 1.1
    )
    with pytest.raises(ValueError, match="numeric physics"):
        different.load_checkpoint(path)
    native = plan.fixed_step_method().step(
        first.state.step_count,
        first.state.time,
        restored,
        jnp.asarray(1.0),
        {"solar_down": 280.0},
    )
    assert native.successful
    _same_state(native.accepted_state, direct.state)


def test_interior_learned_flux_uses_same_closed_budget_and_donor_veto():
    plan = _quiet(condensation_timescale=1e12, rain_evaporation_timescale=1e12)
    initial = plan.initialize(
        jnp.asarray([100.0, 110.0]),
        jnp.asarray([0.2, 0.4]),
        jnp.asarray([285.0, 290.0]),
        100.0,
    )
    result = plan.step(initial, 1.0, water_flux=0.01, energy_flux=2e4)
    assert result.successful
    np.testing.assert_allclose(
        result.state.vapor_mass,
        initial.vapor_mass + jnp.asarray([-0.01, 0.01]),
        atol=1e-13,
    )
    np.testing.assert_allclose(
        result.state.internal_energy,
        initial.internal_energy + jnp.asarray([-2e4, 2e4]),
        atol=1e-8,
    )
    np.testing.assert_allclose(result.state.total_energy, initial.total_energy, atol=2e-8)
    rejected = plan.step(initial, 1.0, water_flux=0.21)
    assert not rejected.successful
    _same_state(rejected.state, initial)


def test_short_observable_derivative_epsilon_sweep_and_switch_invalidity():
    plan = _quiet(radiation=_radiation(), background_diffusivity=0.2)
    initial = plan.initialize(
        jnp.asarray([100.0, 110.0]),
        jnp.asarray([0.2, 0.4]),
        jnp.asarray([285.0, 290.0]),
        100.0,
    )

    def observable(heating):
        state = initial
        valid = jnp.asarray(True)
        for _ in range(3):
            result = plan.step(
                state, 1.0, solar_down=330.0, heating_rate=jnp.asarray([0.0, heating])
            )
            state, valid = result.state, valid & result.derivative_valid
        return plan.diagnose(state).temperature[-1], valid

    assert observable(10.0)[1]
    automatic = jax.grad(lambda x: observable(x)[0])(10.0)
    for epsilon in (0.1, 0.01, 0.001):
        numerical = (observable(10.0 + epsilon)[0] - observable(10.0 - epsilon)[0]) / (
            2 * epsilon
        )
        np.testing.assert_allclose(automatic, numerical, rtol=3e-5, atol=2e-10)
    saturation = (
        plan.thermodynamics.saturation_pressure(285.0)
        * 100
        / (plan.thermodynamics.vapor_gas_constant * 285.0)
    )
    at_switch = _quiet().initialize(jnp.asarray([100.0]), saturation, 285.0, 100.0)
    switched = _quiet().step(at_switch, 0.1)
    assert switched.successful and not switched.derivative_valid


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_state_precision_survives_mixed_parameter_precision_and_native_scan(dtype):
    with jax.enable_x64(True):
        # Radiation, slab and exchange constructors contain float64 numerical
        # leaves under x64. Explicit float32 inventory ownership must still be
        # coherent before and throughout a scan, not repaired at its endpoint.
        plan = _quiet(
            radiation=_radiation(),
            surface_exchange=BulkSurfaceExchangePlan(),
            background_diffusivity=0.2,
        )
        initial = plan.initialize(
            jnp.asarray([100.0, 110.0], dtype),
            jnp.asarray([0.2, 0.4], dtype),
            jnp.asarray([285.0, 290.0], dtype),
            jnp.asarray([100.0, 100.0], dtype),
            surface_temperature=jnp.asarray(295.0, dtype),
            surface_water_mass=jnp.asarray(1000.0, dtype),
        )
        for leaf in jax.tree.leaves(initial):
            if eqx.is_inexact_array(leaf):
                assert leaf.dtype == jnp.dtype(dtype)
        forcing = {
            "solar_down": jnp.asarray(340.0, jnp.float64),
            "wind_speed": jnp.asarray(5.0, jnp.float64),
        }
        final, successful = eqx.filter_jit(
            lambda state: plan.advance(state, jnp.asarray(1.0, jnp.float64), 4, **forcing)
        )(initial)
        assert jnp.all(successful)
        assert final.time == 4.0 and final.slab.water_mass < initial.slab.water_mass
        assert final.environment_energy != initial.environment_energy
        for leaf in jax.tree.leaves(final):
            if eqx.is_inexact_array(leaf):
                assert leaf.dtype == jnp.dtype(dtype)
        problem = FixedStepProblem(
            plan.fixed_step_method(),
            initial,
            t0=0.0,
            t1=4.0,
            step_size=1.0,
            args=forcing,
            state_geometry=EuclideanStateGeometry(),
        )
        native = FixedStepRolloutPlan(retention="final").rollout(problem)
        assert native.successful
        for leaf in jax.tree.leaves(native.final_state):
            if eqx.is_inexact_array(leaf):
                assert leaf.dtype == jnp.dtype(dtype)
        np.testing.assert_allclose(
            native.final_state.internal_energy,
            final.internal_energy,
            rtol=16 * jnp.finfo(dtype).eps,
            atol=0.0,
        )
        np.testing.assert_allclose(
            native.final_state.slab.energy,
            final.slab.energy,
            rtol=16 * jnp.finfo(dtype).eps,
            atol=0.0,
        )
        scale = jnp.maximum(jnp.abs(initial.total_energy), 1.0)
        assert (
            jnp.abs(final.total_energy - initial.total_energy)
            <= 256 * jnp.finfo(dtype).eps * scale
        )

        def observable(coefficient):
            varied = eqx.tree_at(
                lambda p: p.surface_exchange.heat_transfer_coefficient, plan, coefficient
            )
            state, _ = varied.advance(initial, 1.0, 4, **forcing)
            return varied.diagnose(state).temperature[-1]

        derivative = jax.grad(observable)(plan.surface_exchange.heat_transfer_coefficient)
        assert jnp.isfinite(derivative) and derivative > 0


def test_active_mixing_length_floor_has_unequal_one_sided_physical_derivatives():
    with jax.enable_x64(True):
        plan = InteractiveMoistColumnPlan(
            mixing_length=1.0, rain_fall_speed=0.0, snow_fall_speed=0.0
        )
        initial = plan.initialize(
            jnp.asarray([100.0, 110.0]),
            jnp.asarray([0.2, 0.4]),
            jnp.asarray([285.0, 290.0]),
            jnp.asarray([100.0, 100.0]),
        )

        def result(length, ventilation):
            varied = eqx.tree_at(lambda p: p.mixing_length, plan, jnp.asarray(length))
            step = varied.step(initial, 0.1, ventilation=ventilation, shear=0.01)
            return step, varied.diagnose(step.state).temperature[-1]

        center, value = result(1.0, 0.1)
        epsilon = 1e-3
        lower, left = result(1.0 - epsilon, 0.1)
        upper, right = result(1.0 + epsilon, 0.1)
        assert center.successful & lower.successful & upper.successful
        assert not center.derivative_valid
        assert lower.derivative_valid & upper.derivative_valid
        left_derivative, right_derivative = (
            (value - left) / epsilon,
            (right - value) / epsilon,
        )
        assert not np.isclose(left_derivative, right_derivative, rtol=0.01, atol=1e-10)
        inactive, _ = result(1.0, 0.0)
        assert inactive.successful & inactive.derivative_valid


@pytest.mark.parametrize("outward", [-1.0, 1.0])
def test_temperature_admission_boundary_is_not_a_regular_heating_response(outward):
    with jax.enable_x64(True):
        plan = _quiet()
        temperature = (
            plan.thermodynamics.minimum_temperature
            if outward < 0
            else plan.thermodynamics.maximum_temperature
        )
        initial = plan.initialize(jnp.asarray([100.0]), 0.0, temperature, 100.0)
        center = plan.step(initial, 0.1)
        rejected = plan.step(initial, 0.1, heating_rate=outward)
        inward = plan.step(initial, 0.1, heating_rate=-outward)
        assert center.successful and not center.derivative_valid
        assert not plan.diagnose(initial).derivative_valid[0]
        assert not rejected.successful and inward.successful
        _same_state(rejected.state, initial)
        center_temperature = plan.diagnose(center.state).temperature[0]
        outward_derivative = (
            plan.diagnose(rejected.state).temperature[0] - center_temperature
        ) / outward
        inward_derivative = (
            plan.diagnose(inward.state).temperature[0] - center_temperature
        ) / (-outward)
        assert outward_derivative == 0
        np.testing.assert_allclose(
            inward_derivative, 0.1 / (100 * plan.thermodynamics.dry_cv), rtol=1e-7
        )
