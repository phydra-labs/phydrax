#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from phydrax.applications.electrophysiology._neurons import (
    AdaptiveExponentialIntegrateAndFire,
    advance_point_neuron,
    advance_threshold_detector,
    initialize_point_neuron,
    initialize_threshold_detector,
    LeakyIntegrateAndFire,
    PointNeuronState,
    reset_point_neuron,
    ThresholdDetector,
)


jax.config.update("jax_enable_x64", True)


def test_lif_charge_includes_outward_synapse_affinity_and_inward_injection():
    model = LeakyIntegrateAndFire(0.2, 0.01, -65.0, -45.0, -68.0)
    initial = np.asarray([-70.0, -62.0, -55.0])
    injected = np.asarray([0.1, -0.03, 0.2])
    synaptic_g = np.asarray([0.02, 0.005, 0.0])
    synaptic_offset = np.asarray([1.2, 0.35, 0.04])
    state = initialize_point_neuron(model, initial)
    elapsed = 2.3
    result = jax.jit(advance_point_neuron)(
        model, state, elapsed, injected, synaptic_g, synaptic_offset
    )
    total_g = 0.01 + synaptic_g
    equilibrium = (-0.65 + injected - synaptic_offset) / total_g
    expected = equilibrium + (initial - equilibrium) * np.exp(-total_g * elapsed / 0.2)
    np.testing.assert_allclose(result.voltage_mV, expected, rtol=2.0e-13, atol=2.0e-13)


def test_zero_leak_charge_has_finite_parameter_and_current_sensitivities():
    model = LeakyIntegrateAndFire(0.25, 0.0, -65.0, -45.0, -68.0)
    state = initialize_point_neuron(model, -60.0)
    elapsed, current = 3.0, 0.1

    def voltage(leak, injected):
        varied = eqx.tree_at(lambda value: value.leak_conductance_uS, model, leak)
        return advance_point_neuron(varied, state, elapsed, injected).voltage_mV

    value, derivatives = jax.value_and_grad(voltage, argnums=(0, 1))(
        jnp.asarray(0.0), jnp.asarray(current)
    )
    expected_leak = (-65.0 + 60.0) * elapsed / 0.25 - current * elapsed**2 / (2 * 0.25**2)
    np.testing.assert_allclose(value, -60.0 + current * elapsed / 0.25, atol=1.0e-13)
    np.testing.assert_allclose(derivatives, [expected_leak, elapsed / 0.25], atol=1.0e-12)


def test_refractory_release_inside_vector_segment_preserves_exact_free_charge():
    model = LeakyIntegrateAndFire(0.2, 0.01, -65.0, -45.0, -70.0, refractory_ms=2.0)
    before = initialize_point_neuron(model, [-45.0, -45.0])
    reset = reset_point_neuron(model, before, jnp.asarray([1.0, 3.0]))
    result = advance_point_neuron(model, reset, 3.0, 0.2, time_ms=2.0)
    equilibrium = -65.0 + 0.2 / 0.01
    expected_first = equilibrium + (-70.0 - equilibrium) * np.exp(-0.01 * 2.0 / 0.2)
    np.testing.assert_allclose(result.voltage_mV, [expected_first, -70.0], atol=1.0e-12)
    released = advance_point_neuron(model, result, 1.0, 0.2, time_ms=5.0)
    expected_second = equilibrium + (-70.0 - equilibrium) * np.exp(-0.01 / 0.2)
    np.testing.assert_allclose(released.voltage_mV[1], expected_second, atol=1.0e-12)


def test_adex_reset_increment_and_refractory_adaptation_remain_physical():
    model = AdaptiveExponentialIntegrateAndFire(
        0.2,
        0.01,
        -65.0,
        -30.0,
        -60.0,
        2.0,
        0.004,
        100.0,
        0.03,
        refractory_ms=5.0,
    )
    state = PointNeuronState(jnp.asarray(-30.0), jnp.asarray(0.08), jnp.asarray(0.0))
    reset = reset_point_neuron(model, state, 4.0)
    held = advance_point_neuron(model, reset, 3.0, injected_current_nA=100.0, time_ms=4.0)
    equilibrium_w = 0.004 * (-60.0 + 65.0)
    expected_w = equilibrium_w + (0.11 - equilibrium_w) * np.exp(-3.0 / 100.0)
    np.testing.assert_allclose(held.voltage_mV, -60.0, atol=0.0)
    np.testing.assert_allclose(held.adaptation_nA, expected_w, atol=1.0e-14)
    split = advance_point_neuron(model, held, 3.0, 0.1, time_ms=7.0)
    at_release = advance_point_neuron(model, held, 2.0, 0.1, time_ms=7.0)
    after_release = advance_point_neuron(model, at_release, 1.0, 0.1, time_ms=9.0)
    np.testing.assert_allclose(split.voltage_mV, after_release.voltage_mV, atol=1.0e-12)
    np.testing.assert_allclose(
        split.adaptation_nA, after_release.adaptation_nA, atol=1.0e-14
    )


def test_adex_subthreshold_segment_matches_independent_ode_and_differentiates():
    model = AdaptiveExponentialIntegrateAndFire(
        0.2,
        0.01,
        -65.0,
        -30.0,
        -60.0,
        2.0,
        0.002,
        100.0,
        0.03,
        exponential_threshold_mV=-50.0,
    )
    state = PointNeuronState(jnp.asarray(-54.0), jnp.asarray(0.02), jnp.asarray(0.0))
    elapsed, injected = 0.5, 0.18

    def rhs(_, value):
        voltage, adaptation = value
        return [
            (
                -0.01 * (voltage + 65.0)
                + 0.02 * np.exp((voltage + 50.0) / 2.0)
                - adaptation
                + injected
            )
            / 0.2,
            (0.002 * (voltage + 65.0) - adaptation) / 100.0,
        ]

    reference = solve_ivp(rhs, (0.0, elapsed), [-54.0, 0.02], rtol=1.0e-12, atol=1.0e-13)
    result = jax.jit(advance_point_neuron)(model, state, elapsed, injected)
    np.testing.assert_allclose(
        [result.voltage_mV, result.adaptation_nA],
        reference.y[:, -1],
        rtol=1.0e-9,
        atol=1.0e-9,
    )

    def voltage_for_onset(onset):
        varied = eqx.tree_at(lambda value: value.exponential_threshold_mV, model, onset)
        return advance_point_neuron(varied, state, elapsed, injected).voltage_mV

    epsilon = 1.0e-4
    derivative = jax.grad(voltage_for_onset)(jnp.asarray(-50.0))
    finite_difference = (
        voltage_for_onset(-50.0 + epsilon) - voltage_for_onset(-50.0 - epsilon)
    ) / (2 * epsilon)
    assert float(derivative) < 0.0
    np.testing.assert_allclose(derivative, finite_difference, rtol=2.0e-6, atol=1.0e-9)


def test_adex_distinguishes_exponential_onset_from_spike_cutoff():
    with pytest.raises(ValueError):
        AdaptiveExponentialIntegrateAndFire(
            0.2,
            0.01,
            -65.0,
            -50.0,
            -60.0,
            2.0,
            0.002,
            100.0,
            0.03,
            exponential_threshold_mV=-45.0,
        )


def test_threshold_hysteresis_does_not_refire_a_plateau_or_subthreshold_chatter():
    detector = ThresholdDetector(-50.0, -55.0)
    armed = initialize_threshold_detector(detector, -60.0)
    firings = []
    for voltage in [-48.0, -40.0, -50.0, -51.0, -48.0, -56.0, -49.0]:
        armed, fired = advance_threshold_detector(detector, armed, voltage)
        firings.append(bool(fired))
    assert firings == [True, False, False, False, False, False, True]
    equal = ThresholdDetector(-50.0)
    armed = initialize_threshold_detector(equal, -60.0)
    armed, first = advance_threshold_detector(equal, armed, -50.0)
    _, second = advance_threshold_detector(equal, armed, -50.0)
    assert bool(first) and not bool(second)
