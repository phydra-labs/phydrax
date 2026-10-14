"""Sensor dynamics and ancestry-defined gas-transfer observations."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from examples.nuclear_fuel_calibration import prepare_synthetic_inference_rod
from phydrax import config as pxconfig
from phydrax.applications.nuclear_fuel._inventory import FuelGasLedger
from phydrax.applications.nuclear_fuel._observables import (
    FuelGasAncestry,
    FuelObservationPlan,
    FuelSensor,
    FuelSensorParameters,
    thermocouple_response,
)
from phydrax.applications.nuclear_fuel._problem import PreparedFuelRod
from phydrax.applications.nuclear_fuel._sensitivity import (
    FuelFixedGrid,
    FuelReplayResult,
    replay_fuel_grid,
)
from phydrax.applications.nuclear_fuel._simulation import FuelRodSimulation
from phydrax.applications.nuclear_fuel._specification import FuelRodSpecification
from phydrax.observation import CoordinateLayout


def _ledger(
    produced: float, released: float, retained: float, fill: float
) -> FuelGasLedger:
    # Canonical He, Ar, Kr, Xe, N2; Xe carries this independent accounting example.
    production = jnp.asarray([0.0, 0.0, 0.0, produced, 0.0], dtype=jnp.float64)
    release = jnp.asarray([0.0, 0.0, 0.0, released, 0.0], dtype=jnp.float64)
    retention = jnp.asarray([0.0, 0.0, 0.0, retained, 0.0], dtype=jnp.float64)
    filling = jnp.asarray([fill, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float64)
    zero = jnp.zeros(5, dtype=jnp.float64)
    return FuelGasLedger(filling, production, retention, release, zero, zero)


def test_refabricated_fgr_preserves_parent_transfers_not_current_stock() -> None:
    with pxconfig.context(pxconfig.NumericsConfig(enable_x64=True)):
        ancestor = _ledger(100.0, 20.0, 80.0, 5.0)
        segment_start = _ledger(0.0, 0.0, 80.0, 8.0)
        current = _ledger(50.0, 5.0, 125.0, 8.0)
        ancestry = FuelGasAncestry(
            ancestor,
            segment_start,
            parent_rod_id="parent",
            segment_rod_id="refabricated",
            refabrication_id="puncture-and-refill",
        )
        incremental, incremental_valid = ancestry.fraction(current, cumulative=False)
        cumulative, cumulative_valid = ancestry.fraction(current, cumulative=True)
        np.testing.assert_allclose(incremental, 0.1)
        np.testing.assert_allclose(cumulative, 1.0 / 6.0)
        assert bool(incremental_valid) and bool(cumulative_valid)


def test_thermocouple_lag_has_exact_ramp_response_on_nonuniform_grid() -> None:
    with pxconfig.context(pxconfig.NumericsConfig(enable_x64=True)):
        time = np.asarray([0.0, 0.2, 0.9, 2.0, 3.0])
        tau = 0.7
        slope = 12.0
        measured = thermocouple_response(300.0 + slope * time, time, tau)
        # Independent solution of tau*y' + y = T0 + a*t, y(0)=T0.
        reference = 300.0 + slope * (time - tau + tau * np.exp(-time / tau))
        np.testing.assert_allclose(measured, reference, rtol=2.0e-14, atol=2.0e-12)


@pytest.fixture(scope="module")
def cooled_rod() -> tuple[PreparedFuelRod, FuelFixedGrid, FuelReplayResult]:
    with pxconfig.context(pxconfig.NumericsConfig(enable_x64=True)):
        prepared = prepare_synthetic_inference_rod(
            include_cooldown=True, initial_oxide_thickness=1.0e-6
        )
        grid = FuelFixedGrid(
            tuple(float(time) for time in np.asarray(prepared.admission.history.times)),
            prepared.prepared_id,
        )
        replay = replay_fuel_grid(prepared, grid)
        assert bool(replay.successful), tuple(
            int(attempt.status) for attempt in replay.attempts
        )
        return prepared, grid, replay


def test_thermocouple_site_queries_actual_fe_temperature_at_physical_position(
    cooled_rod: tuple[PreparedFuelRod, FuelFixedGrid, FuelReplayResult],
) -> None:
    prepared, grid, replay = cooled_rod
    specification = prepared.admission.specification
    if not isinstance(specification, FuelRodSpecification):
        raise TypeError("Observation fixture requires scientific fabrication metadata.")
    radius = specification.pellet.radius * 0.37
    position = specification.rod.stack_length / 2.0
    sensor = FuelSensor(
        "off-axis-tc", "thermocouple-temperature", radius=radius, axial_position=position
    )
    plan = FuelObservationPlan(prepared, grid, (sensor,), ((sensor.sensor_id, 60.0),))
    observation = plan.evaluate_states(replay.states)
    state = replay.states[grid.times.index(60.0)]
    radial_coordinates = np.asarray(prepared.geometry.fuel.nodes)[:, 0]
    order = np.argsort(radial_coordinates)
    independent = np.interp(
        radius,
        radial_coordinates[order],
        np.asarray(state.fields.temperature_fuel)[order],
    )
    assert bool(observation.successful)
    np.testing.assert_allclose(
        observation.product.values[0], independent, atol=1.0e-11, rtol=1.0e-13
    )


def test_cold_pressure_requires_computed_cooldown_not_imposed_temperature(
    cooled_rod: tuple[PreparedFuelRod, FuelFixedGrid, FuelReplayResult],
) -> None:
    prepared, grid, replay = cooled_rod
    sensor = FuelSensor(
        "puncture", "cold-pressure", cold_temperature=300.0, cold_tolerance=1.0
    )
    plan = FuelObservationPlan(
        prepared, grid, (sensor,), ((sensor.sensor_id, 60.0), (sensor.sensor_id, 360.0))
    )
    observation = plan.evaluate_states(replay.states)
    assert not bool(observation.valid[0])
    assert bool(observation.valid[1])
    hot, cold = replay.states[grid.times.index(60.0)], replay.states[-1]
    assert np.max(np.asarray(hot.fields.temperature_fuel)) > 301.0
    assert np.max(np.abs(np.asarray(cold.fields.temperature_fuel) - 300.0)) <= 1.0
    np.testing.assert_allclose(
        observation.product.values[1], cold.fields.pressure, rtol=0.0, atol=0.0
    )
    assert float(cold.fields.pressure) < float(hot.fields.pressure)


def test_cold_diameter_accounts_once_for_metal_consumption_and_oxide(
    cooled_rod: tuple[PreparedFuelRod, FuelFixedGrid, FuelReplayResult],
) -> None:
    prepared, grid, replay = cooled_rod
    specification = prepared.admission.specification
    if not isinstance(specification, FuelRodSpecification):
        raise TypeError("Observation fixture requires scientific fabrication metadata.")
    position = specification.rod.stack_length / 2.0
    metal = FuelSensor(
        "metal-od", "diameter-metal", axial_position=position, cold_temperature=300.0
    )
    exterior = FuelSensor(
        "exterior-od",
        "diameter-with-oxide",
        axial_position=position,
        cold_temperature=300.0,
    )
    plan = FuelObservationPlan(
        prepared,
        grid,
        (metal, exterior),
        ((exterior.sensor_id, 360.0), (metal.sensor_id, 360.0)),
    )
    observation = plan.evaluate_states(replay.states)
    state = replay.states[-1]
    oxide = np.average(
        np.asarray(state.cladding_sites.oxide_thickness),
        weights=np.asarray(prepared.geometry.cladding.site_weight),
    )
    # CORROS stoichiometry, PNNL-19418 Vol.1 Rev.2 section2.6.1; source
    # oxygen volume coefficient0.6789 and WCAP-3269-41 density6550 kg/m^3.
    consumed = oxide * 1000.0 / 0.6789 * 91.224 / (2.0 * 15.999) / 6550.0
    outer = specification.cladding.outer_radius
    index = np.flatnonzero(np.asarray(prepared.geometry.cladding.nodes)[:, 0] == outer)[0]
    displacement = np.asarray(state.fields.displacement_cladding)[index, 0]
    independent_metal = 2.0 * (outer + displacement - consumed)
    assert bool(observation.successful)
    np.testing.assert_allclose(
        observation.product.values,
        [independent_metal + 2.0 * oxide, independent_metal],
        rtol=2.0e-13,
        atol=1.0e-15,
    )


def test_thermocouple_drift_window_is_observation_validity_not_temperature_clipping(
    cooled_rod: tuple[PreparedFuelRod, FuelFixedGrid, FuelReplayResult],
) -> None:
    prepared, grid, replay = cooled_rod
    sensor = FuelSensor(
        "windowed-tc",
        "thermocouple-temperature",
        axial_position=0.15,
        drift_valid_window=(0.0, 30.0),
    )
    plan = FuelObservationPlan(
        prepared, grid, (sensor,), ((sensor.sensor_id, 20.0), (sensor.sensor_id, 60.0))
    )
    nuisance = FuelSensorParameters(
        plan.sensor_layout,
        offset=jnp.asarray([2.0], dtype=jnp.float64),
        drift=jnp.asarray([0.1], dtype=jnp.float64),
        gain=jnp.asarray([1.0], dtype=jnp.float64),
    )
    measured = plan.evaluate_states(replay.states, sensor_nuisance=nuisance)
    baseline = plan.evaluate_states(replay.states)
    np.testing.assert_allclose(
        measured.product.values - baseline.product.values, [4.0, 8.0], atol=1.0e-12
    )
    np.testing.assert_array_equal(measured.valid, [True, False])


def test_valid_pressure_sample_does_not_hide_failed_production_history(
    tmp_path: Path,
) -> None:
    with pxconfig.context(pxconfig.NumericsConfig(enable_x64=True)):
        prepared = prepare_synthetic_inference_rod()
        # One 60 s maximum step could reach the history end, so the run is
        # admitted; the first 5 s step then exhausts the one-step budget.
        simulation = FuelRodSimulation(
            prepared,
            time_step=5.0,
            maximum_steps=1,
            maximum_time_step=60.0,
            checkpoint_directory=tmp_path,
        )
        try:
            result = simulation.run()
        finally:
            simulation.close()
        assert not bool(result.successful)
        np.testing.assert_array_equal(np.asarray(result.times), [0.0, 5.0])
        grid = FuelFixedGrid((0.0, 5.0), prepared.prepared_id)
        sensor = FuelSensor("pressure-before-exhaustion", "in-pile-pressure")
        plan = FuelObservationPlan(prepared, grid, (sensor,), ((sensor.sensor_id, 5.0),))
        observation = plan.evaluate(result)
        assert bool(observation.valid[0])
        assert not bool(observation.successful)


# Sensor index 0 is a lagged thermocouple and index 1 a pressure transducer;
# samples are (pressure, 60 s), (thermocouple, 5 s), (thermocouple, 60 s).
_NUISANCE_GAIN = (1.3, 0.7)
_NUISANCE_OFFSET = (2.0, -3.0)  # [K, Pa]
_NUISANCE_DRIFT = (0.02, -0.1)  # [K/s, Pa/s]


@pytest.fixture(scope="module")
def lagged_sensor_plan(
    cooled_rod: tuple[PreparedFuelRod, FuelFixedGrid, FuelReplayResult],
) -> FuelObservationPlan:
    prepared, grid, _ = cooled_rod
    with pxconfig.context(pxconfig.NumericsConfig(enable_x64=True)):
        return _lagged_sensor_plan(prepared, grid)


def _lagged_sensor_plan(
    prepared: PreparedFuelRod, grid: FuelFixedGrid, /
) -> FuelObservationPlan:
    position = float(prepared.admission.specification.rod.stack_length) / 2.0
    sensors = (
        FuelSensor(
            "lagged",
            "thermocouple-temperature",
            axial_position=position,
            response_time=7.0,
        ),
        FuelSensor("pressure", "in-pile-pressure", axial_position=position),
    )
    samples = (("pressure", 60.0), ("lagged", 5.0), ("lagged", 60.0))
    return FuelObservationPlan(prepared, grid, sensors, samples)


def _nuisance(
    plan: FuelObservationPlan,
    /,
    *,
    gain: tuple[float, float] = _NUISANCE_GAIN,
    offset: tuple[float, float] = _NUISANCE_OFFSET,
) -> FuelSensorParameters:
    return FuelSensorParameters(
        plan.sensor_layout,
        gain=jnp.asarray(gain, dtype=jnp.float64),
        offset=jnp.asarray(offset, dtype=jnp.float64),
        drift=jnp.asarray(_NUISANCE_DRIFT, dtype=jnp.float64),
    )


def test_sensor_nuisance_scales_offsets_and_drifts_the_lagged_signal(
    cooled_rod: tuple[PreparedFuelRod, FuelFixedGrid, FuelReplayResult],
    lagged_sensor_plan: FuelObservationPlan,
) -> None:
    _, grid, replay = cooled_rod
    plan = lagged_sensor_plan
    with pxconfig.context(pxconfig.NumericsConfig(enable_x64=True)):
        physical = plan.evaluate_states(replay.states)
        measured = plan.evaluate_states(replay.states, sensor_nuisance=_nuisance(plan))
    # Independent measurement model on the nuisance-free lagged signal:
    # gain * signal + offset + drift * (t - t0), per named sensor.
    reference = [
        _NUISANCE_GAIN[sensor] * float(physical.product.values[row])
        + _NUISANCE_OFFSET[sensor]
        + _NUISANCE_DRIFT[sensor] * (grid.times[time] - grid.times[0])
        for row, (sensor, time) in enumerate(plan.samples)
    ]
    assert bool(physical.successful) and bool(measured.successful)
    np.testing.assert_allclose(measured.product.values, reference, rtol=1.0e-14, atol=0.0)


def test_zero_sensor_gain_invalidates_only_that_sensors_samples(
    cooled_rod: tuple[PreparedFuelRod, FuelFixedGrid, FuelReplayResult],
    lagged_sensor_plan: FuelObservationPlan,
) -> None:
    _, _, replay = cooled_rod
    plan = lagged_sensor_plan
    with pxconfig.context(pxconfig.NumericsConfig(enable_x64=True)):
        rejected = plan.evaluate_states(
            replay.states, sensor_nuisance=_nuisance(plan, gain=(1.3, 0.0))
        )
    assert not bool(rejected.successful)
    np.testing.assert_array_equal(rejected.valid, [False, True, True])


def test_sensor_parameters_of_another_sensor_layout_are_refused(
    cooled_rod: tuple[PreparedFuelRod, FuelFixedGrid, FuelReplayResult],
    lagged_sensor_plan: FuelObservationPlan,
) -> None:
    _, _, replay = cooled_rod
    # The same sensor names in another order identify different detectors.
    reordered = FuelSensorParameters(
        CoordinateLayout(("pressure", "lagged")),
        gain=jnp.asarray(_NUISANCE_GAIN, dtype=jnp.float64),
        offset=jnp.asarray(_NUISANCE_OFFSET, dtype=jnp.float64),
        drift=jnp.asarray(_NUISANCE_DRIFT, dtype=jnp.float64),
    )
    with pxconfig.context(pxconfig.NumericsConfig(enable_x64=True)):
        with pytest.raises(ValueError, match="observation sensor layout"):
            lagged_sensor_plan.evaluate_states(replay.states, sensor_nuisance=reordered)


def test_nonfinite_sensor_parameters_invalidate_that_sensors_samples(
    cooled_rod: tuple[PreparedFuelRod, FuelFixedGrid, FuelReplayResult],
    lagged_sensor_plan: FuelObservationPlan,
) -> None:
    _, _, replay = cooled_rod
    plan = lagged_sensor_plan
    with pxconfig.context(pxconfig.NumericsConfig(enable_x64=True)):
        rejected = plan.evaluate_states(
            replay.states, sensor_nuisance=_nuisance(plan, offset=(np.inf, -3.0))
        )
    assert not bool(rejected.successful)
    np.testing.assert_array_equal(rejected.valid, [True, False, False])
