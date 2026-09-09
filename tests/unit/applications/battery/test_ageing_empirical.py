#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.battery._ageing_empirical import (
    advance_empirical_ageing,
    EmpiricalAgeingCoefficients,
    EmpiricalAgeingState,
    EmpiricalAgeingStatus,
    EmpiricalAgeingStressHistory,
    EmpiricalAgeingSupport,
    EmpiricalAgeingTopology,
    initial_empirical_ageing_state,
)


def _support(
    *,
    throughput_bounds=(0.0, 1.0e6),
    maximum_time_gap_s=4.0,
    maximum_macrostep_s=8.0,
):
    return EmpiricalAgeingSupport(
        (280.0, 330.0),
        (0.0, 1.0),
        (0.0, 10.0),
        throughput_bounds,
        maximum_time_gap_s=maximum_time_gap_s,
        maximum_macrostep_s=maximum_macrostep_s,
        source_id="test:empirical-ageing-support",
    )


def _topology(
    node_count=3,
    *,
    support=None,
    maximum_time_gap_s=4.0,
    maximum_macrostep_s=8.0,
):
    support_ = (
        _support(
            maximum_time_gap_s=maximum_time_gap_s,
            maximum_macrostep_s=maximum_macrostep_s,
        )
        if support is None
        else support
    )
    return EmpiricalAgeingTopology(support_, node_count)


def _coefficients(**overrides):
    values = {
        "calendar_rate_per_s": 0.02,
        "throughput_rate_per_c": 0.03,
        "calendar_activation_temperature_k": 0.0,
        "throughput_activation_temperature_k": 0.0,
        "calendar_soc_coefficient": 0.0,
        "throughput_soc_coefficient": 0.0,
        "throughput_current_coefficient_per_a": 0.0,
        "reference_temperature_k": 300.0,
        "reference_state_of_charge": 0.5,
        "reference_current_magnitude_a": 2.0,
        "calendar_exponent": 1.5,
        "throughput_exponent": 2.0,
        "capacity_loss_scale": 0.4,
        "resistance_growth_scale": 0.7,
    }
    values.update(overrides)
    return EmpiricalAgeingCoefficients(
        values["calendar_rate_per_s"],
        values["throughput_rate_per_c"],
        values["calendar_activation_temperature_k"],
        values["throughput_activation_temperature_k"],
        values["calendar_soc_coefficient"],
        values["throughput_soc_coefficient"],
        values["throughput_current_coefficient_per_a"],
        values["reference_temperature_k"],
        values["reference_state_of_charge"],
        values["reference_current_magnitude_a"],
        values["calendar_exponent"],
        values["throughput_exponent"],
        values["capacity_loss_scale"],
        values["resistance_growth_scale"],
    )


def _history(
    times=(0.0, 2.0, 4.0),
    temperature=(300.0, 300.0, 300.0),
    state_of_charge=(0.5, 0.5, 0.5),
    current=(2.0, 2.0, 2.0),
):
    return EmpiricalAgeingStressHistory(
        jnp.asarray(times),
        jnp.asarray(temperature),
        jnp.asarray(state_of_charge),
        jnp.asarray(current),
    )


def _assert_state_equal(actual, expected):
    np.testing.assert_array_equal(actual.calendar_exposure, expected.calendar_exposure)
    np.testing.assert_array_equal(
        actual.throughput_exposure, expected.throughput_exposure
    )
    np.testing.assert_array_equal(actual.time_s, expected.time_s)
    np.testing.assert_array_equal(
        actual.charge_throughput_c, expected.charge_throughput_c
    )


def test_exact_additive_law_and_stable_observables_are_recovered():
    coefficients = _coefficients()
    result = advance_empirical_ageing(
        _topology(), coefficients, initial_empirical_ageing_state(), _history()
    )

    expected_calendar_exposure = 0.02 * 4.0
    expected_throughput_exposure = 0.03 * 2.0 * 4.0
    expected_calendar_damage = expected_calendar_exposure**1.5
    expected_throughput_damage = expected_throughput_exposure**2.0
    expected_damage = expected_calendar_damage + expected_throughput_damage

    assert result.successful
    np.testing.assert_allclose(
        result.accepted_state.calendar_exposure, expected_calendar_exposure
    )
    np.testing.assert_allclose(
        result.accepted_state.throughput_exposure, expected_throughput_exposure
    )
    np.testing.assert_allclose(result.accepted_state.charge_throughput_c, 8.0)
    np.testing.assert_allclose(
        result.observables.calendar_damage, expected_calendar_damage
    )
    np.testing.assert_allclose(
        result.observables.throughput_damage, expected_throughput_damage
    )
    np.testing.assert_allclose(result.observables.total_damage, expected_damage)
    np.testing.assert_allclose(
        result.observables.capacity_loss_fraction,
        -np.expm1(-0.4 * expected_damage),
    )
    np.testing.assert_allclose(
        result.observables.resistance_growth_fraction,
        np.expm1(0.7 * expected_damage),
    )
    np.testing.assert_allclose(
        result.observables.state_of_health, np.exp(-0.4 * expected_damage)
    )
    np.testing.assert_allclose(
        result.observables.capacity_loss_fraction + result.observables.state_of_health,
        1.0,
    )


def test_time_weighted_quadrature_uses_every_nonlinear_stress_node():
    coefficients = _coefficients(
        calendar_soc_coefficient=1.1,
        throughput_soc_coefficient=-0.4,
        throughput_current_coefficient_per_a=0.2,
        calendar_exponent=1.0,
        throughput_exponent=1.0,
    )
    times = np.asarray((0.0, 1.0, 4.0))
    soc = np.asarray((0.3, 0.9, 0.4))
    current = np.asarray((1.0, 4.0, 2.0))
    result = advance_empirical_ageing(
        _topology(),
        coefficients,
        initial_empirical_ageing_state(),
        _history(times=times, state_of_charge=soc, current=current),
    )

    gaps = np.diff(times)
    calendar_rates = 0.02 * np.exp(1.1 * (soc - 0.5))
    throughput_rates = 0.03 * np.exp(-0.4 * (soc - 0.5) + 0.2 * (np.abs(current) - 2.0))
    expected_calendar = np.sum(0.5 * (calendar_rates[:-1] + calendar_rates[1:]) * gaps)
    expected_throughput = np.sum(
        0.5
        * (
            throughput_rates[:-1] * np.abs(current[:-1])
            + throughput_rates[1:] * np.abs(current[1:])
        )
        * gaps
    )
    endpoint_only_calendar = (
        0.5 * (calendar_rates[0] + calendar_rates[-1]) * (times[-1] - times[0])
    )

    np.testing.assert_allclose(
        result.stress_summary.calendar_exposure, expected_calendar, rtol=2.0e-6
    )
    np.testing.assert_allclose(
        result.stress_summary.throughput_exposure, expected_throughput, rtol=2.0e-6
    )
    assert not np.isclose(expected_calendar, endpoint_only_calendar)


def test_zero_rates_advance_physical_coordinates_without_damage():
    coefficients = _coefficients(
        calendar_rate_per_s=0.0,
        throughput_rate_per_c=0.0,
    )
    result = advance_empirical_ageing(
        _topology(), coefficients, initial_empirical_ageing_state(), _history()
    )

    assert result.successful
    np.testing.assert_array_equal(result.accepted_state.calendar_exposure, 0.0)
    np.testing.assert_array_equal(result.accepted_state.throughput_exposure, 0.0)
    np.testing.assert_allclose(result.accepted_state.time_s, 4.0)
    np.testing.assert_allclose(result.accepted_state.charge_throughput_c, 8.0)
    np.testing.assert_array_equal(result.observables.total_damage, 0.0)
    np.testing.assert_array_equal(result.observables.capacity_loss_fraction, 0.0)
    np.testing.assert_array_equal(result.observables.resistance_growth_fraction, 0.0)
    np.testing.assert_array_equal(result.observables.state_of_health, 1.0)


def test_rest_has_calendar_exposure_but_signed_current_adds_equal_throughput():
    topology = _topology()
    coefficients = _coefficients()
    state = initial_empirical_ageing_state()
    rest = advance_empirical_ageing(
        topology,
        coefficients,
        state,
        _history(current=(0.0, 0.0, 0.0)),
    )
    charge = advance_empirical_ageing(topology, coefficients, state, _history())
    discharge = advance_empirical_ageing(
        topology,
        coefficients,
        state,
        _history(current=(-2.0, -2.0, -2.0)),
    )

    assert rest.successful & charge.successful & discharge.successful
    assert rest.accepted_state.calendar_exposure > 0.0
    np.testing.assert_array_equal(rest.accepted_state.throughput_exposure, 0.0)
    np.testing.assert_allclose(
        charge.accepted_state.calendar_exposure,
        discharge.accepted_state.calendar_exposure,
    )
    np.testing.assert_allclose(
        charge.accepted_state.throughput_exposure,
        discharge.accepted_state.throughput_exposure,
    )
    np.testing.assert_allclose(
        charge.accepted_state.charge_throughput_c,
        discharge.accepted_state.charge_throughput_c,
    )
    assert charge.observables.total_damage > rest.observables.total_damage


def test_adjacent_history_subdivision_is_exactly_invariant():
    support = _support(maximum_time_gap_s=2.0, maximum_macrostep_s=4.0)
    coefficients = _coefficients(
        calendar_activation_temperature_k=750.0,
        throughput_activation_temperature_k=500.0,
        calendar_soc_coefficient=0.3,
        throughput_soc_coefficient=-0.2,
        throughput_current_coefficient_per_a=0.1,
    )
    times = (0.0, 1.0, 2.0, 3.0, 4.0)
    temperature = (295.0, 300.0, 305.0, 310.0, 300.0)
    soc = (0.4, 0.5, 0.65, 0.6, 0.45)
    current = (1.0, 2.0, -3.0, -1.5, 0.5)
    whole = advance_empirical_ageing(
        _topology(5, support=support),
        coefficients,
        initial_empirical_ageing_state(),
        _history(times, temperature, soc, current),
    )
    first = advance_empirical_ageing(
        _topology(3, support=support),
        coefficients,
        initial_empirical_ageing_state(),
        _history(times[:3], temperature[:3], soc[:3], current[:3]),
    )
    second = advance_empirical_ageing(
        _topology(3, support=support),
        coefficients,
        first.accepted_state,
        _history(times[2:], temperature[2:], soc[2:], current[2:]),
    )

    assert whole.successful & first.successful & second.successful
    np.testing.assert_allclose(
        second.accepted_state.calendar_exposure,
        whole.accepted_state.calendar_exposure,
        rtol=2.0e-6,
    )
    np.testing.assert_allclose(
        second.accepted_state.throughput_exposure,
        whole.accepted_state.throughput_exposure,
        rtol=2.0e-6,
    )
    np.testing.assert_allclose(
        second.accepted_state.charge_throughput_c,
        whole.accepted_state.charge_throughput_c,
    )
    np.testing.assert_allclose(
        second.observables.capacity_loss_fraction,
        whole.observables.capacity_loss_fraction,
        rtol=2.0e-6,
    )
    np.testing.assert_allclose(
        second.observables.resistance_growth_fraction,
        whole.observables.resistance_growth_fraction,
        rtol=2.0e-6,
    )


def test_positive_declared_coefficients_give_monotone_one_way_outputs():
    topology = _topology()
    coefficients = _coefficients(
        calendar_activation_temperature_k=600.0,
        throughput_activation_temperature_k=400.0,
        calendar_soc_coefficient=0.2,
        throughput_soc_coefficient=0.3,
        throughput_current_coefficient_per_a=0.1,
    )
    first = advance_empirical_ageing(
        topology, coefficients, initial_empirical_ageing_state(), _history()
    )
    second_history = _history(times=(4.0, 6.0, 8.0))
    second = advance_empirical_ageing(
        topology, coefficients, first.accepted_state, second_history
    )

    assert first.successful & second.successful
    assert (
        second.accepted_state.calendar_exposure > first.accepted_state.calendar_exposure
    )
    assert (
        second.accepted_state.throughput_exposure
        > first.accepted_state.throughput_exposure
    )
    assert second.observables.total_damage > first.observables.total_damage
    assert (
        second.observables.capacity_loss_fraction
        > first.observables.capacity_loss_fraction
    )
    assert (
        second.observables.resistance_growth_fraction
        > first.observables.resistance_growth_fraction
    )
    assert second.observables.state_of_health < first.observables.state_of_health


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("calendar_rate_per_s", -1.0),
        ("throughput_rate_per_c", -1.0),
        ("calendar_exponent", 0.0),
        ("throughput_exponent", 0.0),
        ("capacity_loss_scale", -1.0),
        ("resistance_growth_scale", -1.0),
    ),
)
def test_nonmonotone_physical_coefficients_are_rejected(name, value):
    with pytest.raises(Exception, match="outside their physical domain"):
        _coefficients(**{name: value})


@pytest.mark.parametrize(
    ("topology", "history", "expected_status"),
    (
        (
            _topology(),
            _history(times=(0.0, 2.0, 1.0)),
            EmpiricalAgeingStatus.INVALID_TIMES,
        ),
        (
            _topology(maximum_time_gap_s=2.0, maximum_macrostep_s=8.0),
            _history(times=(0.0, 3.0, 4.0)),
            EmpiricalAgeingStatus.TIME_GAP_EXCEEDED,
        ),
        (
            _topology(maximum_time_gap_s=3.0, maximum_macrostep_s=4.0),
            _history(times=(0.0, 2.5, 5.0)),
            EmpiricalAgeingStatus.MACROSTEP_EXCEEDED,
        ),
        (
            _topology(),
            _history(temperature=(300.0, 335.0, 300.0)),
            EmpiricalAgeingStatus.TEMPERATURE_OUT_OF_SUPPORT,
        ),
        (
            _topology(),
            _history(state_of_charge=(0.5, 1.1, 0.5)),
            EmpiricalAgeingStatus.STATE_OF_CHARGE_OUT_OF_SUPPORT,
        ),
        (
            _topology(),
            _history(current=(2.0, 11.0, 2.0)),
            EmpiricalAgeingStatus.CURRENT_OUT_OF_SUPPORT,
        ),
        (
            _topology(
                support=_support(
                    throughput_bounds=(0.0, 5.0),
                    maximum_time_gap_s=4.0,
                    maximum_macrostep_s=8.0,
                )
            ),
            _history(),
            EmpiricalAgeingStatus.THROUGHPUT_OUT_OF_SUPPORT,
        ),
    ),
)
def test_invalid_times_gaps_and_stress_support_fail_closed(
    topology, history, expected_status
):
    state = initial_empirical_ageing_state()
    result = advance_empirical_ageing(topology, _coefficients(), state, history)

    assert not result.successful
    assert result.status == int(expected_status)
    _assert_state_equal(result.accepted_state, state)
    np.testing.assert_array_equal(result.observables.total_damage, 0.0)


def test_invalid_latent_state_and_history_shape_fail_explicitly():
    invalid_state = EmpiricalAgeingState(-0.1, 0.0, 0.0, 0.0)
    result = advance_empirical_ageing(
        _topology(), _coefficients(), invalid_state, _history()
    )
    assert result.status == int(EmpiricalAgeingStatus.INVALID_STATE)
    _assert_state_equal(result.accepted_state, invalid_state)

    with pytest.raises(ValueError, match="does not match"):
        advance_empirical_ageing(
            _topology(4),
            _coefficients(),
            initial_empirical_ageing_state(),
            _history(),
        )


def test_jit_vmap_and_gradients_cover_coefficients_and_stress_inputs():
    topology = _topology()
    coefficients = _coefficients(
        calendar_activation_temperature_k=500.0,
        throughput_activation_temperature_k=300.0,
        calendar_soc_coefficient=0.2,
        throughput_soc_coefficient=0.3,
        throughput_current_coefficient_per_a=0.1,
    )
    state = initial_empirical_ageing_state()
    history = _history()

    compiled = jax.jit(advance_empirical_ageing)(topology, coefficients, state, history)
    assert compiled.successful
    assert compiled.observables.total_damage.shape == ()

    currents = jnp.asarray(
        (
            (1.0, 2.0, 3.0),
            (-1.0, -2.0, -3.0),
            (0.5, 1.0, 1.5),
        )
    )

    def evaluate_current(current):
        stress = EmpiricalAgeingStressHistory(
            history.times_s,
            history.temperature_k,
            history.state_of_charge,
            current,
        )
        return advance_empirical_ageing(
            topology, coefficients, state, stress
        ).observables.total_damage

    batched = jax.vmap(evaluate_current)(currents)
    assert batched.shape == (3,)
    np.testing.assert_allclose(batched[0], batched[1])
    assert batched[0] > batched[2]

    def objective(scales, current):
        varied = eqx.tree_at(
            lambda value: (
                value.capacity_loss_scale,
                value.resistance_growth_scale,
            ),
            coefficients,
            (scales[0], scales[1]),
        )
        stress = EmpiricalAgeingStressHistory(
            history.times_s,
            history.temperature_k,
            history.state_of_charge,
            current,
        )
        outputs = advance_empirical_ageing(topology, varied, state, stress).observables
        return outputs.capacity_loss_fraction + 0.1 * outputs.resistance_growth_fraction

    scale_gradient, current_gradient = jax.grad(objective, argnums=(0, 1))(
        jnp.asarray((0.4, 0.7)), jnp.asarray((1.0, 2.0, 3.0))
    )
    assert np.all(np.isfinite(scale_gradient))
    assert np.all(np.isfinite(current_gradient))
    assert scale_gradient[0] > 0.0
    assert scale_gradient[1] > 0.0
    assert np.all(current_gradient > 0.0)


def test_macrostep_is_pure_and_does_not_modify_supplied_inputs():
    topology = _topology()
    coefficients = _coefficients()
    state = initial_empirical_ageing_state()
    history = _history(
        temperature=(295.0, 305.0, 310.0),
        state_of_charge=(0.3, 0.6, 0.8),
        current=(-1.0, 2.0, -3.0),
    )
    before = tuple(
        np.asarray(value).copy()
        for value in (
            history.times_s,
            history.temperature_k,
            history.state_of_charge,
            history.current_a,
            state.calendar_exposure,
            state.throughput_exposure,
            state.time_s,
            state.charge_throughput_c,
        )
    )

    first = advance_empirical_ageing(topology, coefficients, state, history)
    replay = advance_empirical_ageing(topology, coefficients, state, history)

    after = (
        history.times_s,
        history.temperature_k,
        history.state_of_charge,
        history.current_a,
        state.calendar_exposure,
        state.throughput_exposure,
        state.time_s,
        state.charge_throughput_c,
    )
    for actual, expected in zip(after, before, strict=True):
        np.testing.assert_array_equal(actual, expected)
    _assert_state_equal(first.accepted_state, replay.accepted_state)
    np.testing.assert_array_equal(
        first.observables.capacity_loss_fraction,
        replay.observables.capacity_loss_fraction,
    )
