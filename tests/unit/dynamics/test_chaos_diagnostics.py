#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _scalar_data(values, *, source_id, case_axes=(), case_axis_roles=()):
    values = jnp.asarray(values)
    case_shape = values.shape[:-1]
    coordinates = jnp.broadcast_to(
        jnp.arange(values.shape[-1], dtype=float), case_shape + (values.shape[-1],)
    )
    return phx.dynamics.TrajectoryData(
        coordinates,
        values[..., None],
        state_layout=phx.dynamics.StateLayout((1,), component_names=("observable",)),
        case_axes=case_axes,
        case_axis_roles=case_axis_roles,
        source_id=source_id,
    )


def test_finite_size_growth_recovers_finite_amplitude_linear_rate():
    matrix = jnp.diag(jnp.asarray([2.0, 0.5]))
    system = phx.dynamics.DiscreteSystem(
        lambda coordinate, state, args: matrix @ state,
        state_layout=phx.dynamics.StateLayout((2,)),
        system_id="finite-size-linear-map",
    )
    evolution = phx.dynamics.DiscreteEvolution(system)
    grid = phx.dynamics.IterationGrid.from_steps(6, iteration_id="finite-size-grid")

    result = phx.dynamics.analysis.finite_size_growth(
        evolution,
        jnp.asarray([0.2, -0.3]),
        grid,
        directions=jnp.asarray([[1.0, 0.0], [0.0, 1.0]]),
        perturbation_distance=1e-5,
        rescale_interval=1,
    )

    assert bool(result.valid)
    np.testing.assert_allclose(
        np.asarray(result.growth_rates),
        np.broadcast_to(np.log([[2.0], [0.5]]), (2, 6)),
        rtol=2e-10,
        atol=2e-10,
    )
    np.testing.assert_allclose(
        np.asarray(result.average_growth_rates), np.log([2.0, 0.5]), atol=2e-10
    )


def test_recurrence_rqa_preserves_theiler_mask_and_line_statistics():
    periodic = np.tile(np.arange(8, dtype=float), 25)
    data = _scalar_data(periodic, source_id="periodic-rqa")

    result = phx.dynamics.analysis.recurrence_quantification(
        data,
        1e-12,
        theiler_window=1,
        minimum_diagonal_length=2,
        minimum_vertical_length=2,
    )

    assert bool(result.valid)
    assert not bool(result.eligible[20, 20])
    assert not bool(result.eligible[20, 21])
    assert float(result.determinism) > 0.95
    assert int(result.longest_diagonal) > 100
    assert int(jnp.sum(result.diagonal_length_histogram)) > 0


def test_zero_one_test_separates_periodic_and_logistic_observables():
    count = 1400
    periodic = np.sin(2.0 * np.pi * np.arange(count) / 37.0)
    logistic = np.empty((count,))
    logistic[0] = 0.1234567
    for index in range(count - 1):
        logistic[index + 1] = 4.0 * logistic[index] * (1.0 - logistic[index])
    data = _scalar_data(
        np.stack((periodic, logistic)),
        source_id="zero-one-comparison",
        case_axes=("process",),
        case_axis_roles=("process",),
    )

    result = phx.dynamics.analysis.zero_one_test(
        data,
        burn_in=100,
        num_frequencies=32,
        seed=17,
        minimum_samples=1000,
    )

    assert bool(jnp.all(result.valid))
    assert float(result.statistic[0]) < 0.2
    assert float(result.statistic[1]) > 0.7
    assert float(result.statistic[1] - result.statistic[0]) > 0.6
    assert bool(jnp.all(result.used_sample_mask[:, :100] == 0))


def test_correlation_dimension_records_fit_window_and_theiler_pairs():
    rng = np.random.default_rng(9)
    values = rng.uniform(0.0, 1.0, 1200)
    data = _scalar_data(values, source_id="uniform-line-dimension")
    radii = np.logspace(-2.0, -0.45, 18)

    result = phx.dynamics.analysis.correlation_dimension(
        data,
        radii,
        theiler_window=8,
        fit_indices=(2, 13),
    )

    assert bool(result.valid)
    assert 0.85 < float(result.dimension) < 1.15
    assert float(result.r_squared) > 0.995
    assert int(jnp.sum(result.fit_mask)) == 11
    assert int(result.eligible_pair_count) > 500_000


def test_surrogate_protocol_and_uncertainty_summary_preserve_rng_and_sources():
    time = jnp.arange(512, dtype=float)
    series = jnp.sin(2.0 * jnp.pi * time / 32.0)
    significance = phx.dynamics.analysis.surrogate_significance(
        series,
        lambda values: jnp.mean(values[:-1] * values[1:]),
        statistic_id="lag-one-product",
        method="shuffle",
        alternative="greater",
        num_surrogates=99,
        seed=23,
    )

    assert bool(significance.valid)
    assert float(significance.p_value) <= 0.02
    assert significance.seed == 23
    assert significance.num_surrogates == 99

    samples = jnp.asarray(
        [
            [[0.8], [1.0], [1.2]],
            [[1.8], [2.0], [2.2]],
        ]
    )
    uncertainty = phx.dynamics.analysis.summarize_chaos_uncertainty(
        samples,
        metric_names=("largest-exponent",),
        case_axes=("initial-condition", "tolerance"),
        source_kinds=("initial_condition", "numerics"),
        bootstrap_samples=64,
        seed=29,
    )

    assert bool(uncertainty.valid)
    np.testing.assert_allclose(np.asarray(uncertainty.mean), [1.5], atol=1e-12)
    assert uncertainty.source_kinds == ("initial_condition", "numerics")
    assert uncertainty.bootstrap_means.shape == (64, 1)
    assert bool(jnp.all(uncertainty.source_variance >= 0.0))
