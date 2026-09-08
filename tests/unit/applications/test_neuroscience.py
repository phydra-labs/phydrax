#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.applications import neuroscience as ns


def _connectivity():
    return ns.RegionalConnectivity(
        ("stimulated", "downstream"),
        [[0.0, 0.2], [1.0, 0.0]],
        [[0.0, 0.137], [0.211, 0.0]],
    )


def _history(time, args):
    del time, args
    return jnp.zeros((2, 2))


def _pulse(time, neural, args):
    del args
    return (
        jnp.zeros_like(neural).at[0, 0].set(0.3 * jnp.exp(-(((time - 0.5) / 0.2) ** 2)))
    )


def _bold_problem(connectivity, coupling=0.6, *, initial_balloon=None):
    return ns.regional_bold_problem(
        connectivity,
        ns.Hopf(a_per_s=-0.4, frequency_hz=0.1, coupling_per_s=coupling),
        _history,
        ns.BalloonWindkessel(),
        ns.NeuralBOLDDrive([1.0, 0.0], [0.0, 0.0], gain=0.5),
        t0=0.0,
        t1=3.0,
        drive=_pulse,
        initial_balloon=initial_balloon,
    )


def _solve(problem, times, **kwargs):
    return ns.solve_regional(
        problem,
        save_times=times,
        solver=dfx.Heun(),
        stepsize_controller=dfx.ConstantStepSize(),
        dt0=0.025,
        **kwargs,
    )


def test_directed_mixed_delays_use_exact_nongrid_prehistory_and_declared_units():
    weights = np.asarray([[0.0, 2.0, 0.5], [0.3, 0.0, 0.0], [0.0, 0.0, 0.0]])
    delay_ms = np.asarray([[0.0, 137.0, 0.0], [211.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    connectivity = ns.RegionalConnectivity(
        ("a", "b", "isolated"),
        weights,
        delay_ms,
        delay_unit="ms",
        normalization="incoming_abs",
    )
    initial = jnp.asarray([[0.2, 0.1], [0.3, 0.15], [0.4, 0.2]])
    slope = jnp.asarray([[0.02, 0.01], [0.03, -0.01], [0.04, 0.01]])
    model = ns.WilsonCowan(tau_e_s=0.2, tau_i_s=0.1, coupling_gain=0.7)
    problem = ns.regional_problem(
        connectivity, model, lambda time, args: initial + time * slope, t0=0.0, t1=0.4
    )
    # The receiver a sees delayed b but current-stage isolated; b sees delayed a.
    incoming_e = jnp.asarray(
        [
            (2.0 * (0.3 - 0.137 * 0.03) + 0.5 * 0.4) / 2.5,
            0.2 - 0.211 * 0.02,
            0.0,
        ]
    )
    e, i = initial[:, 0], initial[:, 1]
    expected_e = (
        -e + (1.0 - e) * jax.nn.sigmoid(12.0 * e - 10.0 * i - 2.0 + 0.7 * incoming_e)
    ) / 0.2
    expected_i = (-i + (1.0 - i) * jax.nn.sigmoid(10.0 * e - 3.0)) / 0.1
    np.testing.assert_allclose(
        problem.initial_right_derivative,
        jnp.stack((expected_e, expected_i), axis=-1),
        rtol=2e-13,
        atol=2e-13,
    )
    seconds = ns.RegionalConnectivity(
        connectivity.region_ids, weights, delay_ms * 0.001, normalization="incoming_abs"
    )
    comparison = ns.regional_problem(
        seconds, model, lambda time, args: initial + time * slope, t0=0.0, t1=0.4
    )
    np.testing.assert_array_equal(
        comparison.initial_right_derivative, problem.initial_right_derivative
    )


def test_instantaneous_hopf_self_coupling_cancels_and_hertz_sets_rotation():
    connectivity = ns.RegionalConnectivity(("oscillator",), [[7.0]], [[0.0]])
    problem = ns.regional_problem(
        connectivity,
        ns.Hopf(a_per_s=1.0, cubic_per_s=1.0, frequency_hz=0.5, coupling_per_s=3.0),
        lambda time, args: jnp.asarray([[1.0, 0.0]]),
        t0=0.0,
        t1=1.0,
    )
    times = jnp.asarray([0.0, 0.5, 1.0])
    solved = ns.solve_regional(problem, save_times=times, rtol=1e-9, atol=1e-11)
    expected = jnp.stack((jnp.cos(jnp.pi * times), jnp.sin(jnp.pi * times)), axis=-1)
    np.testing.assert_allclose(solved.neural.values[:, 0], expected, rtol=2e-8, atol=2e-8)


def test_balloon_equilibrium_and_pulse_recover_without_losing_positive_ratios():
    model = ns.BalloonWindkessel()
    resting = ns.balloon_equilibrium(1)
    np.testing.assert_allclose(model(resting, jnp.zeros(1)), 0.0, atol=1e-14)
    np.testing.assert_array_equal(model.bold(resting), jnp.zeros(1))
    problem = phx.solver.DifferentialProblem(
        lambda time, state, args: model(
            state, 0.1 * jnp.exp(-(((time - 1.0) / 0.2) ** 2))
        ),
        resting,
        t0=0.0,
        t1=25.0,
    )
    solved = phx.solver.solve_diffrax(
        problem, save_times=jnp.linspace(0.0, 25.0, 101), rtol=1e-8, atol=1e-10
    )
    bold = model.bold(solved.states)[:, 0]
    peak = int(jnp.argmax(bold))
    assert 1.0 < float(solved.times[peak]) < 10.0
    assert float(bold[peak]) > 1e-5
    assert abs(float(bold[-1])) < 0.02 * float(bold[peak])
    np.testing.assert_allclose(jnp.exp(solved.states[-1, :, 1:]), 1.0, atol=2e-4)
    assert bool(jnp.all(solved.valid & jnp.all(jnp.isfinite(solved.states), axis=(1, 2))))


def test_segmented_continuation_preserves_neural_history_and_all_balloon_state():
    initial = jnp.asarray(
        [
            [0.02, jnp.log(1.1), jnp.log(1.02), jnp.log(0.98)],
            [0.01, jnp.log(1.05), 0.0, 0.0],
        ]
    )
    problem = _bold_problem(_connectivity(), initial_balloon=initial)
    full = _solve(problem, jnp.asarray([3.0]), segmented=True, max_steps_per_segment=8)
    partial = _solve(
        problem,
        jnp.asarray([0.0]),
        segmented=True,
        max_steps_per_segment=8,
        max_segments=1,
    )
    restarted = _solve(
        problem,
        jnp.asarray([partial.continuation.time, 3.0]),
        segmented=True,
        max_steps_per_segment=8,
        continuation=partial.continuation,
    )
    np.testing.assert_allclose(
        restarted.native.states[-1], full.native.states[-1], rtol=2e-12, atol=2e-12
    )
    np.testing.assert_allclose(
        restarted.bold.values[-1], full.bold.values[-1], rtol=2e-12, atol=2e-12
    )
    assert bool(jnp.all(restarted.bold.sample_valid))
    assert bool(jnp.all(restarted.neural.sample_valid))


def test_delayed_bold_parameter_gradient_and_masked_fit_failure_boundary():
    connectivity = _connectivity()
    times = jnp.linspace(0.0, 3.0, 13)

    def predict(coupling, args=None):
        del args
        return _solve(
            _bold_problem(connectivity, coupling),
            times,
            adjoint=dfx.DirectAdjoint(),
            max_steps=256,
        )

    true = jnp.asarray(0.6)
    truth = predict(true)
    active = jnp.ones(truth.bold.values.shape, dtype=bool).at[2, 0].set(False)
    observed = ns.BOLDObservation.from_samples(
        truth.region_ids,
        times,
        jnp.where(active, truth.bold.values, jnp.nan),
        valid=active,
        standard_deviation=1e-4,
    )
    residual = observed.least_squares_problem(predict).residual
    _, tangent = jax.jvp(
        lambda gain: residual(gain, None), (true,), (jnp.ones_like(true),)
    )
    step = 1e-4
    finite_difference = (residual(true + step, None) - residual(true - step, None)) / (
        2.0 * step
    )
    np.testing.assert_allclose(tangent, finite_difference, rtol=2e-4, atol=2e-5)
    assert float(jnp.sum(tangent * tangent)) > 1e-4
    np.testing.assert_allclose(residual(true, None), 0.0, atol=1e-9)
    assert float(tangent[4]) == 0.0
    # A failed active prediction cannot improve the loss by erasing its datum.
    invalid_bold = eqx.tree_at(
        lambda series: series.value_valid,
        truth.bold,
        truth.bold.value_valid.at[3, 1].set(False),
    )
    invalid = eqx.tree_at(lambda result: result.bold, truth, invalid_bold)
    assert bool(jnp.isnan(observed.residual(invalid)[7]))
    reordered = ns.RegionalSolution(
        truth.native, truth.neural, truth.bold, truth.region_ids[::-1]
    )
    with pytest.raises(ValueError, match="region order"):
        observed.residual(reordered)
    percent = ns.BOLDObservation.from_samples(
        truth.region_ids,
        times * 1000.0,
        truth.bold.values * 100.0,
        time_unit="ms",
        signal_unit="percent",
        standard_deviation=0.01,
    )
    np.testing.assert_allclose(
        percent.target,
        observed.target + jnp.where(active, 0.0, truth.bold.values),
        rtol=1e-14,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        percent.standard_deviation,
        observed.standard_deviation.at[2, 0].set(1e-4),
        rtol=1e-14,
    )


def test_rejects_invalid_delays_parameters_and_nonphysical_initial_hemodynamics():
    with pytest.raises(ValueError, match="nonnegative"):
        ns.RegionalConnectivity(("a",), [[1.0]], [[-0.001]])
    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="positive"):
        ns.WilsonCowan(tau_e_s=0.0)
    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="strictly between"):
        ns.BalloonWindkessel(extraction=1.0)
    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="strictly positive"):
        _bold_problem(
            _connectivity(), initial_balloon=jnp.zeros((2, 4)).at[0, 1].set(-1000.0)
        )
    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="active datum"):
        ns.BOLDObservation.from_samples(
            ("a",), [0.0, 1.0], [[jnp.nan], [jnp.nan]], valid=[[False], [False]]
        )
