import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.climate import (
    ClimateDrivers,
    decay_average,
    GasBoxModel,
    MODEL_YEAR_SECONDS,
    MultilayerEnergyBalance,
    Myhre1998Forcing,
    read_reduced_climate_checkpoint,
    ReducedClimatePlan,
    write_reduced_climate_checkpoint,
)
from phydrax.applications.geophysics import GeophysicalTimeSpec
from phydrax.dynamics import TimeGrid
from phydrax.solver import FixedStepRolloutPlan


@pytest.fixture(autouse=True)
def _double_precision():
    with jax.enable_x64(True):
        yield


def _runtime(plan=None, steps=4, duration=1.0):
    plan = ReducedClimatePlan() if plan is None else plan
    return plan.prepare(
        TimeGrid(np.arange(steps + 1) * duration, time_id="climate-test-grid"),
        seconds_per_time_unit=MODEL_YEAR_SECONDS,
    )


def _drivers(runtime, emission=(10.0, 20.0, 3.0)):
    count = runtime.step_count
    return ClimateDrivers(
        jnp.broadcast_to(jnp.asarray(emission), (count, 3)),
        jnp.broadcast_to(runtime.plan.gases.background, (count, 3)),
        jnp.zeros((count, 3)),
        jnp.zeros((count, len(runtime.plan.forcing.external_names))),
    )


def test_reservoir_zero_near_zero_and_equilibrium_limits():
    model = GasBoxModel(((1.0,), (1.0,), (1.0,)), ((0.0,), (1.0e-14,), (0.2,)))
    boxes = jnp.asarray([[2.0], [3.0], [10.0]])
    rates = jnp.asarray((1.0, 2.0, 2.0))
    result = model.advance(
        boxes,
        jnp.zeros(3),
        jnp.asarray(0.0),
        jnp.asarray(5.0),
        rates,
        model.background,
        ("emissions",) * 3,
    )
    assert bool(result.successful)
    np.testing.assert_allclose(
        result.boxes[:, 0], (7.0, 13.0, 10.0), atol=2.0e-11, rtol=0.0
    )
    np.testing.assert_allclose(
        jnp.sum(result.boxes - boxes, axis=-1) + result.sink_increment,
        5.0 * rates,
        atol=2.0e-14,
    )
    assert np.isfinite(jax.grad(lambda x: decay_average(x))(jnp.asarray(0.0)))
    np.testing.assert_allclose(
        jax.grad(lambda x: decay_average(x))(jnp.asarray(0.0)), -0.5, atol=1.0e-14
    )


def test_inactive_nan_drivers_do_not_poison_reservoir_parameter_gradients():
    model = GasBoxModel(((1.0,), (1.0,), (1.0,)), ((0.2,), (0.3,), (0.4,)))
    boxes = jnp.zeros_like(model.fractions)
    unused = jnp.full((3,), jnp.nan)

    def forward(rate):
        varied = eqx.tree_at(
            lambda gas: gas.decay_rates, model, model.decay_rates.at[0, 0].set(rate)
        )
        return varied.advance(
            boxes,
            jnp.zeros(3),
            jnp.asarray(0.0),
            jnp.asarray(1.0),
            jnp.ones(3),
            unused,
            ("emissions",) * 3,
        ).boxes[0, 0]

    value, derivative = jax.value_and_grad(forward)(jnp.asarray(0.2))
    expected_value = -np.expm1(-0.2) / 0.2
    expected_derivative = (1.2 * np.exp(-0.2) - 1.0) / 0.2**2
    np.testing.assert_allclose(value, expected_value, atol=1.0e-13)
    np.testing.assert_allclose(derivative, expected_derivative, atol=1.0e-13)

    def inverse(rate):
        varied = eqx.tree_at(
            lambda gas: gas.decay_rates, model, model.decay_rates.at[0, 0].set(rate)
        )
        target = model.background + 1.0 / model.inventory_per_concentration
        return varied.advance(
            boxes,
            jnp.zeros(3),
            jnp.asarray(0.0),
            jnp.asarray(1.0),
            unused,
            target,
            ("concentration",) * 3,
        ).emissions[0]

    inferred, inverse_derivative = jax.value_and_grad(inverse)(jnp.asarray(0.2))
    np.testing.assert_allclose(inferred, 1.0 / expected_value, atol=1.0e-12)
    np.testing.assert_allclose(
        inverse_derivative, -expected_derivative / expected_value**2, atol=1.0e-12
    )


def test_concentration_inverse_roundtrip_uses_same_frozen_lifetime():
    model = GasBoxModel(
        response_coefficients=((0.02, 0.4, 0.01), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    )
    boxes = 5.0 * model.fractions
    sink = jnp.asarray((30.0, 0.0, 0.0))
    emission = jnp.asarray((11.0, -1.0, 3.0))
    forward = model.advance(
        boxes,
        sink,
        jnp.asarray(1.0),
        jnp.asarray(0.75),
        emission,
        model.background,
        ("emissions",) * 3,
    )
    inverse = model.advance(
        boxes,
        sink,
        jnp.asarray(1.0),
        jnp.asarray(0.75),
        jnp.zeros(3),
        model.concentration(forward.boxes),
        ("concentration",) * 3,
    )
    assert bool(forward.successful & inverse.successful)
    np.testing.assert_allclose(inverse.emissions, emission, atol=2.0e-12, rtol=2.0e-12)
    np.testing.assert_allclose(inverse.boxes, forward.boxes, atol=2.0e-12, rtol=2.0e-12)
    np.testing.assert_allclose(
        inverse.sink_increment, forward.sink_increment, atol=2.0e-12
    )


def test_lifetime_root_is_certified_and_has_implicit_sensitivity():
    model = GasBoxModel(
        response_coefficients=((0.0, 0.5, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        solve_tolerance=1.0e-11,
    )
    boxes = jnp.zeros_like(model.fractions)

    def response(temperature):
        return model.lifetime(boxes, jnp.zeros(3), temperature).multiplier[0]

    result = model.lifetime(boxes, jnp.zeros(3), jnp.asarray(2.0))
    assert bool(result.successful)
    assert float(result.residual[0]) < 1.0e-9
    assert float(result.multiplier[0]) > 1.0
    finite_difference = (response(2.0 + 1.0e-4) - response(2.0 - 1.0e-4)) / 2.0e-4
    np.testing.assert_allclose(
        jax.grad(response)(jnp.asarray(2.0)), finite_difference, rtol=2.0e-7, atol=1.0e-10
    )


def test_infeasible_lifetime_and_exhausted_solve_commit_nothing():
    for coefficients, iterations in (
        (((0.0, 1.0e5, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), 64),
        (((0.0, 0.5, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), 1),
    ):
        model = GasBoxModel(
            response_coefficients=coefficients,
            solve_iterations=iterations,
            solve_tolerance=1.0e-12,
        )
        runtime = _runtime(ReducedClimatePlan(gases=model))
        state = runtime.initial_state(temperature=(1.0, 0.0))
        drivers = jax.tree.map(lambda value: value[0], _drivers(runtime))
        rejected = runtime.step(state, drivers)
        assert not bool(rejected.successful)
        for old, accepted in zip(
            jax.tree.leaves(state), jax.tree.leaves(rejected.state), strict=True
        ):
            np.testing.assert_array_equal(accepted, old)


def test_multilayer_exchange_budget_equilibrium_and_singular_heating():
    model = MultilayerEnergyBalance((8.0, 40.0, 120.0), (0.8, 0.3), feedback=1.25)
    result = model.advance(
        jnp.asarray((0.1, -0.3, 0.7)), jnp.asarray(17.0), jnp.asarray(3.0)
    )
    assert bool(result.successful)
    assert abs(float(result.energy_residual)) < 2.0e-12
    equilibrium = model.advance(jnp.full((3,), 2.4), jnp.asarray(100.0), jnp.asarray(3.0))
    np.testing.assert_allclose(equilibrium.temperature, 2.4, atol=2.0e-13)
    singular = MultilayerEnergyBalance((8.0,), (), feedback=0.0).advance(
        jnp.asarray((2.0,)), jnp.asarray(5.0), jnp.asarray(4.0)
    )
    assert bool(singular.successful)
    np.testing.assert_allclose(singular.temperature, (4.5,), atol=1.0e-13)
    np.testing.assert_allclose(singular.surface_temperature_integral, 16.25, atol=1.0e-13)
    assert abs(float(singular.energy_residual)) < 1.0e-13


def test_named_forcing_overlap_background_and_driver_replacement():
    forcing = Myhre1998Forcing(("solar", "volcanic"))
    background = jnp.asarray((278.0, 730.0, 270.0))
    baseline = forcing.evaluate(
        background, background, jnp.zeros(2), jnp.zeros(3), ("emissions",) * 3
    )
    np.testing.assert_array_equal(baseline.components, jnp.zeros(7))
    doubled = forcing.evaluate(
        background * jnp.asarray((2.0, 1.0, 1.0)),
        background,
        jnp.asarray((0.2, -0.5)),
        jnp.zeros(3),
        ("emissions",) * 3,
    )
    np.testing.assert_allclose(doubled.total, 5.35 * np.log(2.0) - 0.3, atol=1.0e-13)
    raised = forcing.evaluate(
        background * jnp.asarray((1.0, 2.0, 2.0)),
        background,
        jnp.zeros(2),
        jnp.zeros(3),
        ("emissions",) * 3,
    )
    assert float(raised.components[2]) < 0.0 and float(raised.components[4]) < 0.0
    replaced = forcing.evaluate(
        background * 2.0,
        background,
        jnp.asarray((0.2, -0.5)),
        jnp.asarray((1.0, 2.0, 3.0)),
        ("forcing",) * 3,
    )
    np.testing.assert_allclose(replaced.total, 5.7, atol=1.0e-13)
    np.testing.assert_array_equal(replaced.components[jnp.asarray((2, 4))], 0.0)


def test_forcing_driven_gases_neither_decay_nor_solve_inactive_lifetimes():
    gas = GasBoxModel(
        response_coefficients=((0.0, 1.0e5, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    )
    runtime = _runtime(ReducedClimatePlan(gases=gas, roles=("forcing",) * 3))
    initial = runtime.initial_state(boxes=3.0 * gas.fractions, temperature=(1.0, 0.0))
    drivers = ClimateDrivers(
        jnp.full(3, jnp.nan),
        jnp.full(3, jnp.nan),
        jnp.asarray((1.0, 2.0, 3.0)),
        jnp.zeros(0),
    )
    result = runtime.step(initial, drivers)
    assert bool(result.successful)
    np.testing.assert_array_equal(result.state.boxes, initial.boxes)
    np.testing.assert_array_equal(result.state.cumulative_sink, initial.cumulative_sink)
    np.testing.assert_array_equal(
        result.state.cumulative_emissions, initial.cumulative_emissions
    )
    np.testing.assert_allclose(result.forcing.total, 6.0, atol=1.0e-13)
    np.testing.assert_allclose(result.state.cumulative_forcing_energy, 6.0, atol=1.0e-13)


def test_state_dependent_reservoir_timestep_refinement():
    model = GasBoxModel(
        response_coefficients=((0.02, 0.0, 0.04), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    )

    def integrate(count):
        duration = jnp.asarray(20.0 / count)

        def step(state, _):
            boxes, sink = state
            result = model.advance(
                boxes,
                sink,
                jnp.asarray(0.0),
                duration,
                jnp.asarray((10.0, 0.0, 0.0)),
                model.background,
                ("emissions",) * 3,
            )
            return (result.boxes, sink + result.sink_increment), result.successful

        final, success = jax.lax.scan(
            step, (jnp.zeros_like(model.fractions), jnp.zeros(3)), None, length=count
        )
        assert bool(jnp.all(success))
        return jnp.sum(final[0][0])

    reference = integrate(256)
    errors = np.asarray([abs(integrate(count) - reference) for count in (4, 8, 16)])
    assert errors[1] < 0.65 * errors[0]
    assert errors[2] < 0.65 * errors[1]


def test_rollout_budgets_and_native_restart_parity(tmp_path):
    runtime = _runtime(steps=6)
    drivers = _drivers(runtime)
    initial = runtime.initial_state()
    rollout = eqx.filter_jit(FixedStepRolloutPlan(retention="final").rollout)
    complete = rollout(runtime.problem(initial, drivers))
    first = rollout(runtime.problem(initial, drivers, stop_step=3))
    assert bool(complete.successful & first.successful)
    path = tmp_path / "climate.chk"
    write_reduced_climate_checkpoint(path, runtime, first.final_state, drivers)
    restored = read_reduced_climate_checkpoint(path, runtime, initial, drivers)
    restarted = rollout(runtime.problem(restored, drivers, start_step=3))
    assert bool(restarted.successful)
    for expected, actual in zip(
        jax.tree.leaves(complete.final_state),
        jax.tree.leaves(restarted.final_state),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)
    final = complete.final_state
    np.testing.assert_allclose(
        jnp.sum(final.boxes, axis=-1) + final.cumulative_sink,
        final.cumulative_emissions,
        atol=1.0e-11,
    )
    np.testing.assert_allclose(
        runtime.plan.energy.heat_content(final.temperature),
        final.cumulative_forcing_energy - final.cumulative_outgoing_energy,
        atol=1.0e-11,
    )
    changed_drivers = eqx.tree_at(
        lambda value: value.emissions, drivers, drivers.emissions.at[-1, 0].add(1.0)
    )
    with pytest.raises(ValueError):
        read_reduced_climate_checkpoint(path, runtime, initial, changed_drivers)
    changed_runtime = eqx.tree_at(
        lambda value: value.plan.energy.feedback, runtime, jnp.asarray(1.3)
    )
    with pytest.raises(ValueError):
        read_reduced_climate_checkpoint(path, changed_runtime, initial, drivers)


def test_failed_native_rollout_freezes_at_last_accepted_boundary():
    runtime = _runtime(steps=4)
    initial = runtime.initial_state()
    drivers = _drivers(runtime)
    drivers = eqx.tree_at(
        lambda value: value.emissions, drivers, drivers.emissions.at[1, 0].set(jnp.nan)
    )
    result = runtime.rollout(initial, drivers)
    assert not bool(result.successful)
    assert int(result.final_state.step_index) == 1
    for leaf in jax.tree.leaves(result.states):
        np.testing.assert_array_equal(leaf[-1], leaf[1])
    np.testing.assert_array_equal(result.valid, (True, True, False, False, False))


def test_geophysical_clock_and_explicit_duration_give_same_physics():
    plan = ReducedClimatePlan()
    years = _runtime(plan, steps=1)
    clock = GeophysicalTimeSpec(unit="d")
    days = plan.prepare(TimeGrid((0.0, 365.25), time_id=clock.time_id), time_spec=clock)
    year_result = years.step(
        years.initial_state(), jax.tree.map(lambda x: x[0], _drivers(years))
    )
    day_result = days.step(
        days.initial_state(), jax.tree.map(lambda x: x[0], _drivers(days))
    )
    assert bool(year_result.successful & day_result.successful)
    np.testing.assert_allclose(
        year_result.state.boxes, day_result.state.boxes, atol=1.0e-13
    )
    np.testing.assert_allclose(
        year_result.state.temperature, day_result.state.temperature, atol=1.0e-13
    )
