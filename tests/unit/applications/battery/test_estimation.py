#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.battery._ecm import (
    ThermalEquivalentCircuitParameters,
    ThermalEquivalentCircuitPlan,
)
from phydrax.applications.battery._estimation import (
    estimate_exact_affine_ecm,
    ExactAffineECMEstimationPlan,
    GloballyAffineSOCPropertyLaw,
)
from phydrax.applications.battery._properties import (
    ConstantPropertyLaw,
    TabulatedPropertyLaw,
)
from phydrax.stochastic import StateSpaceProblem
from phydrax.uq import kalman_filter, rts_smoother


def _constant_law(value, support, *, quantity, unit):
    return ConstantPropertyLaw(
        value,
        jnp.asarray(support),
        quantity=quantity,
        coordinate="state_of_charge",
        value_unit=unit,
        coordinate_unit="1",
        source_id=f"test:{quantity}",
    )


def _tabulated_law(nodes, values, *, quantity, unit, source_mask=None):
    return TabulatedPropertyLaw(
        jnp.asarray(nodes),
        jnp.asarray(values),
        source_mask=source_mask,
        quantity=quantity,
        coordinate="state_of_charge",
        value_unit=unit,
        coordinate_unit="1",
        source_id=f"test:{quantity}",
    )


def _global_law(intercept, slope, *, quantity, unit):
    return GloballyAffineSOCPropertyLaw(
        intercept,
        slope,
        quantity=quantity,
        value_unit=unit,
        source_id=f"test:global:{quantity}",
    )


def _global_laws():
    return {
        "open_circuit_voltage": _global_law(
            3.1,
            1.2,
            quantity="reference-open-circuit-voltage",
            unit="V",
        ),
        "entropic_coefficient": _global_law(
            1.0e-3,
            0.0,
            quantity="entropic-coefficient",
            unit="V/K",
        ),
    }


def _qualify(plan, parameters, **kwargs):
    return plan.prepare(parameters, **_global_laws(), **kwargs)


def _parameters(
    *,
    branches=2,
    capacity=100.0,
    ocv=None,
    entropic=None,
):
    resistances = jnp.asarray((0.2, 0.4)[:branches])
    capacitances = jnp.asarray((5.0, 10.0)[:branches])
    if ocv is None:
        nodes = jnp.asarray((0.1, 0.4, 0.7, 0.9))
        ocv = _tabulated_law(
            nodes,
            3.1 + 1.2 * nodes,
            quantity="reference-open-circuit-voltage",
            unit="V",
        )
    if entropic is None:
        entropic = _constant_law(
            1.0e-3,
            (0.1, 0.9),
            quantity="entropic-coefficient",
            unit="V/K",
        )
    return ThermalEquivalentCircuitParameters(
        0.05,
        resistances,
        capacitances,
        capacity,
        100.0,
        0.5,
        300.0,
        300.0,
        ocv,
        entropic,
    )


def _prepared(
    *,
    branches=2,
    capacity=100.0,
    process_covariance=None,
    temperature_mode="known",
):
    plan = ExactAffineECMEstimationPlan(
        ThermalEquivalentCircuitPlan(branches),
        jnp.asarray((0.2, 0.8)),
        temperature_mode=temperature_mode,
    )
    return _qualify(
        plan,
        _parameters(branches=branches, capacity=capacity),
        process_noise_covariance_rate=process_covariance,
        voltage_variance_v2=2.5e-3,
    )


def _truth(prepared, times, currents, temperatures, initial):
    values = [np.asarray(initial, dtype=float)]
    resistances = np.asarray(prepared.branch_resistances_ohm)
    capacitances = np.asarray(prepared.branch_capacitances_f)
    capacity = float(prepared.reference_capacity_c)
    for index in range(1, len(times)):
        duration = float(times[index] - times[index - 1])
        decay = np.exp(-duration / (resistances * capacitances))
        previous = values[-1]
        state = np.empty_like(previous)
        state[0] = previous[0] + currents[index - 1] * duration / capacity
        state[1:] = decay * previous[1:] + currents[index - 1] * resistances * (
            1.0 - decay
        )
        values.append(state)
    states = np.stack(values)
    temperature_delta = temperatures - float(prepared.reference_temperature_k)
    intercept = float(prepared.ocv_coefficients_v[0]) + temperature_delta * float(
        prepared.entropic_coefficients_v_per_k[0]
    )
    slope = float(prepared.ocv_coefficients_v[1]) + temperature_delta * float(
        prepared.entropic_coefficients_v_per_k[1]
    )
    voltage = (
        intercept
        + slope * states[:, 0]
        + np.sum(states[:, 1:], axis=-1)
        + float(prepared.series_resistance_ohm) * currents
    )
    return states, voltage


def _problem(*, mask=None, process_covariance=None):
    prepared = _prepared(process_covariance=process_covariance)
    times = np.asarray((0.0, 0.4, 1.1, 2.0, 3.2))
    currents = np.asarray((0.8, -0.3, 0.5, 0.0, -0.2))
    temperatures = np.asarray((300.0, 301.0, 302.0, 301.5, 300.5))
    _, exact_voltage = _truth(
        prepared,
        times,
        currents,
        temperatures,
        np.asarray((0.45, 0.01, -0.02)),
    )
    measured_voltage = exact_voltage + np.asarray((0.01, -0.015, 0.005, 0.012, -0.008))
    problem = prepared.problem(
        times,
        currents,
        measured_voltage,
        temperatures,
        prior_mean=jnp.asarray((0.43, 0.0, 0.0)),
        prior_covariance=jnp.diag(jnp.asarray((2.0e-2, 1.0e-2, 1.5e-2))),
        voltage_mask=mask,
        case_ids=("synthetic-cell",),
        identity_id="synthetic-exact-affine-trace",
    )
    return prepared, problem


def _manual_filter(problem):
    prior = problem.model.prior
    observations = problem.observations
    mean = np.asarray(prior.mean)
    covariance = np.asarray(prior.covariance)
    predicted_means = []
    predicted_covariances = []
    filtered_means = []
    filtered_covariances = []
    transitions = []
    innovations = []
    previous_time = float(problem.initial_time)
    for index, time in enumerate(np.asarray(observations.times)):
        context = problem.step_context(0, index)
        transition = problem.model.transition.parameters(previous_time, time, context)
        matrix = np.asarray(transition.transition)
        offset = np.asarray(transition.offset)
        process_covariance = np.asarray(transition.covariance)
        predicted_mean = matrix @ mean + offset
        predicted_covariance = matrix @ covariance @ matrix.T + process_covariance
        observation_matrix, observation_offset, observation_covariance = (
            problem.model.observation.parameters(time, context)
        )
        observation_matrix = np.asarray(observation_matrix)
        observation_offset = np.asarray(observation_offset)
        observation_covariance = np.asarray(observation_covariance)
        innovation = np.asarray(observations.values[index]) - (
            observation_matrix @ predicted_mean + observation_offset
        )
        if bool(observations.observation_mask[index, 0]):
            innovation_covariance = (
                observation_matrix @ predicted_covariance @ observation_matrix.T
                + observation_covariance
            )
            gain = np.linalg.solve(
                innovation_covariance,
                observation_matrix @ predicted_covariance,
            ).T
            mean = predicted_mean + gain @ innovation
            update = np.eye(mean.size) - gain @ observation_matrix
            covariance = (
                update @ predicted_covariance @ update.T
                + gain @ observation_covariance @ gain.T
            )
        else:
            innovation = np.zeros_like(innovation)
            mean = predicted_mean
            covariance = predicted_covariance
        predicted_means.append(predicted_mean)
        predicted_covariances.append(predicted_covariance)
        filtered_means.append(mean)
        filtered_covariances.append(covariance)
        transitions.append(matrix)
        innovations.append(innovation)
        previous_time = float(time)
    return tuple(
        np.stack(value)
        for value in (
            predicted_means,
            predicted_covariances,
            filtered_means,
            filtered_covariances,
            transitions,
            innovations,
        )
    )


def _manual_rts(
    filtered_means,
    filtered_covariances,
    predicted_means,
    predicted_covariances,
    transitions,
):
    means = filtered_means.copy()
    covariances = filtered_covariances.copy()
    gains = np.zeros((means.shape[0] - 1, means.shape[1], means.shape[1]))
    for index in range(means.shape[0] - 2, -1, -1):
        cross = filtered_covariances[index] @ transitions[index + 1].T
        gain = cross @ np.linalg.pinv(predicted_covariances[index + 1])
        means[index] = filtered_means[index] + gain @ (
            means[index + 1] - predicted_means[index + 1]
        )
        covariances[index] = (
            filtered_covariances[index]
            + gain @ (covariances[index + 1] - predicted_covariances[index + 1]) @ gain.T
        )
        gains[index] = gain
    return means, covariances, gains


def test_analytic_exact_transition_observation_and_passive_signs():
    factor = np.asarray(
        (
            (1.0e-3, 0.0),
            (2.0e-4, 2.0e-3),
            (-1.0e-4, 4.0e-4),
        )
    )
    process_rate = factor @ factor.T
    prepared, problem = _problem(process_covariance=process_rate)
    context = problem.step_context(0, 1)
    start, end = np.asarray(problem.observations.times)[0:2]
    transition = problem.model.transition.parameters(start, end, context)
    duration = end - start
    rates = np.asarray((0.0, -1.0, -0.25))
    expected_matrix = np.diag(np.exp(rates * duration))
    expected_offset = np.asarray(
        (
            0.8 * duration / 100.0,
            0.8 * 0.2 * (1.0 - np.exp(-duration)),
            0.8 * 0.4 * (1.0 - np.exp(-0.25 * duration)),
        )
    )
    pair_rates = rates[:, None] + rates[None, :]
    expected_integral = np.empty_like(pair_rates)
    zero = pair_rates == 0.0
    expected_integral[zero] = duration
    expected_integral[~zero] = np.expm1(pair_rates[~zero] * duration) / pair_rates[~zero]
    expected_covariance = process_rate * expected_integral

    np.testing.assert_allclose(transition.transition, expected_matrix, rtol=2e-6)
    np.testing.assert_allclose(transition.offset, expected_offset, rtol=2e-6)
    np.testing.assert_allclose(transition.covariance, expected_covariance, rtol=2e-6)

    observation_matrix, observation_offset, observation_covariance = (
        problem.model.observation.parameters(end, context)
    )
    np.testing.assert_allclose(observation_matrix, np.asarray(((1.2, 1.0, 1.0),)))
    expected_observation_offset = 3.1 + 1.0e-3 - 0.3 * 0.05
    np.testing.assert_allclose(observation_offset, (expected_observation_offset,))
    np.testing.assert_allclose(observation_covariance, ((2.5e-3,),))

    first_context = problem.step_context(0, 0)
    identity = problem.model.transition.parameters(start, start, first_context)
    np.testing.assert_allclose(identity.transition, np.eye(3))
    np.testing.assert_allclose(identity.offset, np.zeros(3), atol=1e-12)
    np.testing.assert_allclose(identity.covariance, np.zeros((3, 3)), atol=1e-12)

    negative_problem = prepared.problem(
        jnp.asarray((0.0, 1.0)),
        jnp.asarray((-0.8, -0.8)),
        jnp.asarray((3.0, 3.0)),
        300.0,
        prior_mean=jnp.asarray((0.5, 0.0, 0.0)),
        prior_covariance=jnp.eye(3),
    )
    negative = negative_problem.model.transition.parameters(
        0.0, 1.0, negative_problem.step_context(0, 1)
    )
    assert float(transition.offset[0]) > 0.0
    assert np.all(np.asarray(transition.offset[1:]) > 0.0)
    assert float(negative.offset[0]) < 0.0
    assert np.all(np.asarray(negative.offset[1:]) < 0.0)
    assert problem.model.metadata["terminal_current_sign"] == (
        "positive-enters-positive-terminal"
    )


def test_filter_and_rts_match_independent_exact_recursions_and_native_results():
    _, problem = _problem()
    application = estimate_exact_affine_ecm(
        problem,
        filter_method="sequential",
        smoother_method="sequential",
    )
    manual = _manual_filter(problem)
    (
        predicted_mean,
        predicted_covariance,
        filtered_mean,
        filtered_covariance,
        matrices,
        innovation,
    ) = manual
    manual_smooth = _manual_rts(
        filtered_mean,
        filtered_covariance,
        predicted_mean,
        predicted_covariance,
        matrices,
    )

    np.testing.assert_allclose(
        application.filter_result.predicted_means, predicted_mean, rtol=2e-5
    )
    np.testing.assert_allclose(
        application.filter_result.predicted_covariances,
        predicted_covariance,
        rtol=2e-5,
        atol=2e-7,
    )
    np.testing.assert_allclose(
        application.filter_result.filtered_means, filtered_mean, rtol=2e-5
    )
    np.testing.assert_allclose(
        application.filter_result.filtered_covariances,
        filtered_covariance,
        rtol=2e-5,
        atol=2e-7,
    )
    np.testing.assert_allclose(application.innovations_v, innovation[:, 0], rtol=2e-5)
    np.testing.assert_allclose(
        application.smoother_result.means, manual_smooth[0], rtol=3e-5
    )
    np.testing.assert_allclose(
        application.smoother_result.covariances,
        manual_smooth[1],
        rtol=3e-5,
        atol=3e-7,
    )
    np.testing.assert_allclose(
        application.smoother_result.gains,
        manual_smooth[2],
        rtol=3e-5,
        atol=1e-12,
    )

    direct_filter = kalman_filter(problem, method="sequential")
    direct_smoother = rts_smoother(direct_filter, method="sequential")
    np.testing.assert_array_equal(
        application.filter_result.filtered_means, direct_filter.filtered_means
    )
    np.testing.assert_array_equal(
        application.smoother_result.means, direct_smoother.means
    )
    assert isinstance(problem, StateSpaceProblem)
    assert bool(application.successful)
    assert application.innovation_diagnostics.passed


def test_missing_voltage_is_a_native_masked_prediction_with_zero_diagnostics():
    mask = jnp.asarray((True, False, True, False, True))
    _, problem = _problem(mask=mask)
    result = estimate_exact_affine_ecm(
        problem,
        filter_method="sequential",
        smoother_method="sequential",
    )

    np.testing.assert_allclose(
        result.filter_result.filtered_means[~np.asarray(mask)],
        result.filter_result.predicted_means[~np.asarray(mask)],
    )
    np.testing.assert_allclose(
        result.filter_result.filtered_covariances[~np.asarray(mask)],
        result.filter_result.predicted_covariances[~np.asarray(mask)],
    )
    np.testing.assert_array_equal(result.innovations_v[~mask], 0.0)
    np.testing.assert_array_equal(result.filtered_voltage_residuals_v[~mask], 0.0)
    np.testing.assert_array_equal(
        result.filter_result.observed_counts, mask.astype(jnp.int32)
    )


def test_soc_support_exit_is_fail_closed_and_retains_native_history():
    prepared = _prepared(branches=1, capacity=10.0)
    problem = prepared.problem(
        jnp.asarray((0.0, 1.0, 2.0, 3.0)),
        jnp.asarray((1.0, 1.0, -1.0, -1.0)),
        jnp.asarray((4.0, 4.0, 4.0, 4.0)),
        300.0,
        prior_mean=jnp.asarray((0.75, 0.0)),
        prior_covariance=jnp.diag(jnp.asarray((1.0e-12, 1.0e-12))),
        voltage_mask=jnp.zeros((4,), dtype=bool),
    )
    result = estimate_exact_affine_ecm(
        problem,
        filter_method="sequential",
        smoother_method="sequential",
    )

    np.testing.assert_array_equal(
        result.predicted_mean_in_support,
        jnp.asarray((True, False, False, False)),
    )
    np.testing.assert_array_equal(result.valid, jnp.asarray((True, False, False, False)))
    assert not bool(result.successful)
    assert bool(jnp.all(result.filter_result.valid))
    assert bool(result.likelihood_successful)
    assert bool(jnp.all(result.likelihood_valid))
    np.testing.assert_array_equal(
        result.physical_mean_support_valid,
        jnp.asarray((True, False, False, False)),
    )
    with pytest.raises(RuntimeError, match="declared SOC support"):
        estimate_exact_affine_ecm(
            problem,
            filter_method="sequential",
            smoother_method="sequential",
            raise_on_failure=True,
        )


def test_case_axes_jit_and_vmap_preserve_exact_native_parameterization():
    prepared = _prepared(branches=1)
    times = jnp.asarray(((0.0, 0.5, 1.0), (0.0, 0.5, 1.0)))
    problem = prepared.problem(
        times,
        jnp.asarray(((0.2, 0.4, 0.0), (-0.2, -0.4, 0.0))),
        jnp.asarray(((3.7, 3.7, 3.7), (3.7, 3.7, 3.7))),
        jnp.asarray(((300.0, 301.0, 302.0), (300.0, 299.0, 298.0))),
        prior_mean=jnp.asarray(((0.5, 0.0), (0.5, 0.0))),
        prior_covariance=jnp.eye(2) * 1.0e-2,
        case_axes=("cell",),
        case_ids=("charge", "discharge"),
    )

    indices = jnp.asarray(((0, 1), (1, 1)), dtype=jnp.int32)

    def inspect(index):
        context = problem.step_context(index[0], index[1])
        transition = problem.model.transition.parameters(0.0, 0.5, context)
        observation, offset, covariance = problem.model.observation.parameters(
            0.5, context
        )
        return transition.offset, observation, offset, covariance

    offsets, matrices, voltage_offsets, covariances = jax.jit(jax.vmap(inspect))(indices)
    assert offsets.shape == (2, 2)
    assert float(offsets[0, 0]) > 0.0
    assert float(offsets[1, 0]) < 0.0
    np.testing.assert_allclose(matrices[..., 0, 0], 1.2)
    np.testing.assert_allclose(covariances[..., 0, 0], 2.5e-3)
    assert float(voltage_offsets[0, 0]) > float(voltage_offsets[1, 0])

    compiled = eqx.filter_jit(
        lambda native_problem: estimate_exact_affine_ecm(
            native_problem,
            filter_method="sequential",
            smoother_method="sequential",
        )
    )(problem)
    assert compiled.filter_result.filtered_means.shape == (2, 3, 2)
    assert bool(jnp.all(compiled.successful))


def test_deterministic_identities_and_dynamic_parameter_leaves():
    prepared_a, problem_a = _problem()
    prepared_b, problem_b = _problem()
    result_a = estimate_exact_affine_ecm(
        problem_a,
        filter_method="sequential",
        smoother_method="sequential",
    )
    result_b = estimate_exact_affine_ecm(
        problem_b,
        filter_method="sequential",
        smoother_method="sequential",
    )
    assert prepared_a.plan.plan_id == prepared_b.plan.plan_id
    assert prepared_a.prepared_id == prepared_b.prepared_id
    assert problem_a.problem_id == problem_b.problem_id
    assert result_a.estimation_id == result_b.estimation_id

    leaves = jax.tree.leaves(prepared_a)
    assert any(leaf is prepared_a.series_resistance_ohm for leaf in leaves)
    assert any(leaf is prepared_a.branch_resistances_ohm for leaf in leaves)
    changed = eqx.tree_at(
        lambda value: value.series_resistance_ohm,
        prepared_a,
        prepared_a.series_resistance_ohm * 1.1,
    )
    assert float(changed.series_resistance_ohm) != float(prepared_a.series_resistance_ohm)


def test_covariances_are_validated_at_the_qualified_boundary():
    plan = ExactAffineECMEstimationPlan(
        ThermalEquivalentCircuitPlan(1), jnp.asarray((0.2, 0.8))
    )
    parameters = _parameters(branches=1)
    with pytest.raises(ValueError, match="positive semidefinite"):
        _qualify(
            plan,
            parameters,
            process_noise_covariance_rate=jnp.asarray(((-1.0, 0.0), (0.0, 1.0))),
            voltage_variance_v2=1.0e-3,
        )
    with pytest.raises(ValueError, match="symmetric"):
        _qualify(
            plan,
            parameters,
            process_noise_covariance_rate=jnp.asarray(((1.0, 0.2), (0.0, 1.0))),
            voltage_variance_v2=1.0e-3,
        )
    with pytest.raises(ValueError, match="positive"):
        _qualify(plan, parameters, voltage_variance_v2=0.0)

    prepared = _qualify(plan, parameters, voltage_variance_v2=1.0e-3)
    with pytest.raises(ValueError, match="positive semidefinite"):
        prepared.problem(
            jnp.asarray((0.0, 1.0)),
            jnp.zeros((2,)),
            jnp.ones((2,)) * 3.7,
            300.0,
            prior_mean=jnp.asarray((0.5, 0.0)),
            prior_covariance=jnp.asarray(((-1.0, 0.0), (0.0, 1.0))),
        )


@pytest.mark.parametrize(
    ("keyword", "message"),
    (
        ({"temperature_mode": "dynamic"}, "temperature_mode"),
        ({"hysteresis": True}, "hysteresis"),
        ({"capacity_fade": True}, "capacity_fade"),
        ({"parameter_estimation": True}, "parameter_estimation"),
    ),
)
def test_general_thermal_hysteretic_fading_and_parameter_routes_are_refused(
    keyword, message
):
    with pytest.raises(ValueError, match=message):
        ExactAffineECMEstimationPlan(
            ThermalEquivalentCircuitPlan(1),
            jnp.asarray((0.2, 0.8)),
            **keyword,
        )


def test_finite_nonlinear_and_gapped_laws_cannot_stand_in_for_global_affine_contract():
    plan = ExactAffineECMEstimationPlan(
        ThermalEquivalentCircuitPlan(1), jnp.asarray((0.2, 0.8))
    )
    parameters = _parameters(branches=1)
    finite_affine = parameters.open_circuit_voltage
    with pytest.raises(TypeError, match="GloballyAffineSOCPropertyLaw"):
        plan.prepare(
            parameters,
            open_circuit_voltage=finite_affine,
            entropic_coefficient=_global_laws()["entropic_coefficient"],
            voltage_variance_v2=1.0e-3,
        )

    nonlinear_ocv = _tabulated_law(
        (0.1, 0.4, 0.7, 0.9),
        (3.2, 3.5, 4.2, 4.3),
        quantity="reference-open-circuit-voltage",
        unit="V",
    )
    nonlinear_parameters = _parameters(branches=1, ocv=nonlinear_ocv)
    with pytest.raises(ValueError, match="does not agree"):
        _qualify(plan, nonlinear_parameters, voltage_variance_v2=1.0e-3)

    gapped_ocv = _tabulated_law(
        (0.1, 0.4, 0.7, 0.9),
        (3.2, 3.5, 3.8, 4.0),
        source_mask=jnp.asarray((True, True, False, True)),
        quantity="reference-open-circuit-voltage",
        unit="V",
    )
    gapped_entropic = _constant_law(
        0.0,
        (0.1, 0.9),
        quantity="entropic-coefficient",
        unit="V/K",
    )
    gapped_parameters = ThermalEquivalentCircuitParameters(
        0.05,
        jnp.asarray((0.2,)),
        jnp.asarray((5.0,)),
        100.0,
        100.0,
        0.5,
        300.0,
        300.0,
        gapped_ocv,
        gapped_entropic,
    )
    with pytest.raises(ValueError, match="physical support gap"):
        _qualify(plan, gapped_parameters, voltage_variance_v2=1.0e-3)

    disagreeing = _global_laws()
    disagreeing["open_circuit_voltage"] = _global_law(
        3.1,
        1.3,
        quantity="reference-open-circuit-voltage",
        unit="V",
    )
    with pytest.raises(ValueError, match="does not agree"):
        plan.prepare(parameters, **disagreeing, voltage_variance_v2=1.0e-3)

    global_ocv = _global_laws()["open_circuit_voltage"]
    far_values = eqx.filter_jit(global_ocv.evaluate)(jnp.asarray((-100.0, 100.0)))
    np.testing.assert_array_equal(far_values.support, jnp.asarray((True, True)))
    np.testing.assert_allclose(far_values.values, 3.1 + 1.2 * np.asarray((-100.0, 100.0)))


def test_isothermal_route_requires_one_externally_known_temperature():
    isothermal_plan = ExactAffineECMEstimationPlan(
        ThermalEquivalentCircuitPlan(1),
        jnp.asarray((0.2, 0.8)),
        temperature_mode="isothermal",
    )
    isothermal = _qualify(
        isothermal_plan,
        _parameters(branches=1),
        voltage_variance_v2=1.0e-3,
    )
    with pytest.raises(ValueError, match="one scalar known temperature"):
        isothermal.problem(
            jnp.asarray((0.0, 1.0)),
            jnp.zeros((2,)),
            jnp.ones((2,)) * 3.7,
            jnp.asarray((300.0, 301.0)),
            prior_mean=jnp.asarray((0.5, 0.0)),
            prior_covariance=jnp.eye(2),
        )
