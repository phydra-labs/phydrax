import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from phydrax._trainable import partition_trainable
from phydrax.applications import electrophysiology as ep
from phydrax.domain import HyperRectangle
from phydrax.nn import population as pc


def _neuron():
    return ep.LeakyIntegrateAndFire(0.2, 0.01, -65.0, -50.0, -62.0, refractory_ms=2.0)


def test_physical_rate_period_matches_reset_to_threshold_and_refractory():
    neuron = _neuron()
    current = jnp.asarray([0.16, 0.3, 0.8])
    rate = pc.lif_rate_response(neuron, current)
    charge_ms = 1000.0 / rate - neuron.refractory_ms
    initial = ep.initialize_point_neuron(neuron, neuron.reset_mV)
    at_threshold = ep.advance_point_neuron(
        neuron, initial, charge_ms, injected_current_nA=current
    )
    np.testing.assert_allclose(at_threshold.voltage_mV, neuron.threshold_mV, atol=1e-10)
    assert jnp.all(rate < 1000.0 / neuron.refractory_ms)
    rheobase = neuron.leak_conductance_uS * (neuron.threshold_mV - neuron.resting_mV)
    np.testing.assert_array_equal(
        pc.lif_rate_response(neuron, jnp.asarray([-1.0, rheobase])), 0.0
    )
    reset = ep.reset_point_neuron(neuron, at_threshold, time_ms=4.0)
    still_refractory = ep.advance_point_neuron(
        neuron, reset, 1.0, injected_current_nA=current, time_ms=4.0
    )
    np.testing.assert_allclose(still_refractory.voltage_mV, neuron.reset_mV)


def test_zero_leak_population_uses_the_exact_perfect_integrator_limit():
    neuron = ep.LeakyIntegrateAndFire(0.2, 0.0, -65.0, -50.0, -62.0, refractory_ms=2.0)
    currents = jnp.asarray([0.1, 0.4])
    expected = 1000.0 / (2.0 + 0.2 * 12.0 / currents)
    np.testing.assert_allclose(
        pc.lif_rate_response(neuron, currents), expected, rtol=1e-12
    )
    population = pc.prepare_lif_population(
        HyperRectangle([-1.0], [1.0]),
        neuron,
        2,
        key=jr.key(7),
        encoders=[[1.0], [-1.0]],
        intercepts=0.0,
        maximum_rates_hz=[60.0, 90.0],
    )
    np.testing.assert_allclose(
        jnp.diag(population.rates(jnp.asarray([[1.0], [-1.0]]))), [60.0, 90.0]
    )
    np.testing.assert_array_equal(population.rates(jnp.asarray([0.0])), [0.0, 0.0])


def test_encoder_support_and_rate_inversion_hold_at_multidimensional_box_corners():
    domain = HyperRectangle([-2.0, 1.0], [4.0, 5.0])
    maximum = jnp.asarray([60.0, 90.0, 130.0])
    intercepts = jnp.asarray([-0.5, 0.0, 0.7])
    population = pc.prepare_lif_population(
        domain,
        _neuron(),
        3,
        key=jr.key(17),
        encoders=[[2.0, 1.0], [-1.0, 3.0], [-1.0, -2.0]],
        intercepts=intercepts,
        maximum_rates_hz=maximum,
    )
    center = 0.5 * (domain.lower + domain.upper)
    radius = 0.5 * (domain.upper - domain.lower)
    directions = jnp.sign(population.encoders)
    maxima = center + radius * directions
    onsets = center + radius * directions * intercepts[:, None]
    np.testing.assert_allclose(jnp.diag(population.rates(maxima)), maximum, rtol=1e-12)
    threshold = _neuron().leak_conductance_uS * (
        _neuron().threshold_mV - _neuron().resting_mV
    )
    np.testing.assert_allclose(
        jnp.diag(population.currents(onsets)), threshold, atol=1e-14
    )
    all_corners = jnp.asarray([[-2.0, 1.0], [-2.0, 5.0], [4.0, 1.0], [4.0, 5.0]])
    assert jnp.all(population.rates(all_corners) <= maximum[None, :] + 1e-10)
    with pytest.raises(ValueError):
        pc.prepare_lif_population(
            domain, _neuron(), 3, key=jr.key(0), maximum_rates_hz=500.0
        )


def test_rank_diagnoses_duplicate_and_silent_neurons_without_hiding_masked_rank():
    population = pc.LIFPopulation(
        HyperRectangle([-1.0], [1.0]),
        _neuron(),
        [[1.0], [1.0], [-1.0], [1.0]],
        [0.2, 0.2, 0.2, 0.0],
        [0.15, 0.15, 0.15, 0.0],
    )
    points = jnp.linspace(-1.0, 1.0, 40)[:, None]
    target = (
        population.rates(points)[:, 0] * 0.002 - population.rates(points)[:, 2] * 0.001
    )
    deficient = pc.fit_population_decoder(population, points, target, ridge=0.0)
    assert int(deficient.least_squares.rank) == 2
    assert not bool(deficient.least_squares.valid)
    assert jnp.isinf(deficient.least_squares.condition_number)
    np.testing.assert_array_equal(deficient.silent_neurons, [False, False, False, True])
    selected = pc.fit_population_decoder(
        population, points, target, ridge=0.0, neuron_mask=[True, False, True, False]
    )
    assert bool(selected.least_squares.valid)
    assert int(selected.least_squares.rank) == 2
    np.testing.assert_allclose(selected(points), target, atol=1e-12)
    rates = population.rates(points).at[:, 1].set(jnp.nan).at[:, 3].set(jnp.inf)
    np.testing.assert_allclose(selected.decode_rates(rates), target, atol=1e-12)
    regularized = pc.fit_population_decoder(population, points, target, ridge=1e-3)
    assert bool(regularized.least_squares.valid)
    assert int(regularized.least_squares.rank) == 2


def test_weighted_samples_are_scale_replication_and_padding_invariant_with_ridge():
    population = pc.prepare_lif_population(
        HyperRectangle([-1.0], [1.0]), _neuron(), 8, key=jr.key(3)
    )
    points = jnp.linspace(-1.0, 1.0, 21)[:, None]
    target = points[:, 0] ** 2
    weights = jnp.linspace(0.001, 0.003, 21)
    reference = pc.fit_population_decoder(
        population, points, target, weights=weights, ridge=0.2
    )
    scaled = pc.fit_population_decoder(
        population, points, target, weights=10000.0 * weights, ridge=0.2
    )
    replicated = pc.fit_population_decoder(
        population,
        jnp.repeat(points, 2, axis=0),
        jnp.repeat(target, 2),
        weights=jnp.repeat(weights / 2.0, 2),
        ridge=0.2,
    )
    padded_points = jnp.concatenate((points, jnp.asarray([[jnp.nan], [0.3], [0.7]])))
    padded_target = jnp.concatenate((target, jnp.asarray([jnp.nan, 1e12, 1e12])))
    padded = pc.fit_population_decoder(
        population,
        padded_points,
        padded_target,
        mask=jnp.concatenate(
            (jnp.ones((21,), dtype=bool), jnp.asarray([False, True, False]))
        ),
        weights=jnp.concatenate((weights, jnp.asarray([jnp.inf, 0.0, 1.0]))),
        ridge=0.2,
    )
    for code in (scaled, replicated, padded):
        np.testing.assert_allclose(
            code(points), reference(points), atol=1e-11, rtol=1e-10
        )
    assert int(padded.least_squares.sample_count) == 21
    assessment = pc.assess_population_code(reference, points, target, weights=weights)
    scaled_assessment = pc.assess_population_code(
        reference, points, target, weights=10000 * weights
    )
    np.testing.assert_allclose(assessment.rmse, scaled_assessment.rmse, atol=1e-12)
    empty = pc.fit_population_decoder(
        population, points, target, mask=jnp.zeros((21,), dtype=bool)
    )
    assert not bool(empty.least_squares.valid)
    empty_assessment = pc.assess_population_code(
        reference, points, target, weights=jnp.zeros((21,))
    )
    assert not bool(empty_assessment.valid)
    assert jnp.isnan(empty_assessment.rmse)


def test_held_out_nonlinear_approximation_is_frozen_and_callable_under_jit():
    keys = jr.split(jr.key(12), 3)
    population = pc.prepare_lif_population(
        HyperRectangle([-1.0], [1.0]), _neuron(), 64, key=keys[0]
    )
    training = pc.sample_population_points(population, 512, key=keys[1])
    held_out = pc.sample_population_points(population, 128, key=keys[2])

    def target(point):
        return jnp.asarray([point[0] ** 2, jnp.sin(2.0 * point[0])])

    code = pc.fit_population_decoder(population, training, target)
    report = pc.assess_population_code(code, held_out, target)
    assert bool(report.valid)
    assert jnp.all(report.rmse < 0.08)
    np.testing.assert_allclose(
        jax.jit(lambda points: code(points))(held_out), report.prediction, atol=1e-10
    )
    trainable, _ = partition_trainable({"code": code, "coefficient": jnp.asarray(0.5)})
    assert trainable["code"] is None
    assert trainable["coefficient"] is not None


def test_filtered_spikes_preserve_streaming_and_separate_temporal_errors():
    population = pc.LIFPopulation(
        HyperRectangle([-1.0], [1.0]), _neuron(), [[1.0]], [0.2], [0.15]
    )
    training = jnp.linspace(0.1, 1.0, 20)[:, None]
    code = pc.fit_population_decoder(
        population, training, population.rates(training)[:, 0] * 0.01, ridge=0.0
    )
    points = jnp.concatenate((jnp.full((20, 1), 0.2), jnp.full((40, 1), 0.9)))
    rates = population.rates(points)
    expected_counts = rates * 2.0 / 1000.0
    filtered = pc.filter_population_spikes(
        expected_counts, dt_ms=2.0, time_constant_ms=20.0
    )
    first = pc.filter_population_spikes(
        expected_counts[:23], dt_ms=2.0, time_constant_ms=20.0
    )
    second = pc.filter_population_spikes(
        expected_counts[23:],
        dt_ms=2.0,
        time_constant_ms=20.0,
        initial_rate_hz=first.final_rate_hz,
    )
    np.testing.assert_allclose(
        jnp.concatenate((first.rates_hz, second.rates_hz)), filtered.rates_hz, atol=1e-12
    )
    target = rates[:, 0] * 0.01
    report = pc.assess_population_spikes(code, points, filtered, target)
    assert report.approximation.rmse < 1e-12
    assert report.filtering.rmse > 0.02
    assert report.spike_variability.rmse < 1e-12
    np.testing.assert_allclose(
        report.total.residual,
        report.approximation.residual
        + report.filtering.residual
        + report.spike_variability.residual,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        pc.decode_filtered_spikes(code, filtered), report.total.prediction
    )
    stationary = pc.filter_population_spikes(
        jnp.ones((5, 1)),
        dt_ms=10.0,
        time_constant_ms=20.0,
        initial_rate_hz=jnp.asarray([100.0]),
    )
    np.testing.assert_allclose(stationary.rates_hz, 100.0, atol=1e-12)
    with pytest.raises(TypeError):
        pc.decode_filtered_spikes(code, expected_counts)
