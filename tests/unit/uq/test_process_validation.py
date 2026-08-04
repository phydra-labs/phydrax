import coordax as cx
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


def _ar1(key, shape, *, correlation=0.85):
    noise = jr.normal(key, shape)
    states = [noise[..., 0]]
    innovation_scale = jnp.sqrt(1.0 - correlation**2)
    for index in range(1, shape[-1]):
        states.append(correlation * states[-1] + innovation_scale * noise[..., index])
    return jnp.stack(states, axis=-1)


def _split(*, calibration=19, test=64):
    return phx.uq.ProcessValidationSplit(
        ("train-0", "train-1"),
        tuple(f"calibration-{index}" for index in range(calibration)),
        tuple(f"test-{index}" for index in range(test)),
    )


def _synthetic_events(key, *, num_paths=4096, rate=3.0, capacity=20):
    count_key, time_key, channel_key, mark_key = jr.split(key, 4)
    counts = jnp.minimum(jr.poisson(count_key, rate, (num_paths,)), capacity)
    valid = jnp.arange(capacity)[None, :] < counts[:, None]
    times = jnp.sort(jr.uniform(time_key, (num_paths, capacity)), axis=-1)
    channels = jr.bernoulli(channel_key, 1.0 / 3.0, (num_paths, capacity)).astype(
        jnp.int32
    )
    marks = channels + 0.1 * jr.normal(mark_key, (num_paths, capacity))
    return phx.stochastic.JumpEventBatch(
        jnp.where(valid, times, 0.0),
        jnp.where(valid, channels, 0),
        jnp.where(valid, marks, 0.0),
        valid,
        jnp.zeros((num_paths,), dtype=jnp.int32),
    )


def _monte_carlo_zero():
    return phx.uq.MonteCarloEstimate(
        replicates=jnp.zeros((3,)),
        mean=jnp.asarray(0.0),
        standard_error=jnp.asarray(0.0),
        lower=jnp.asarray(0.0),
        upper=jnp.asarray(0.0),
        confidence=0.95,
    )


def _temporal_diagnostics(*, mean_error=0.01, covariance_error=0.02):
    return phx.uq.TemporalMomentDiagnostics(
        times=jnp.arange(2.0),
        mean=jnp.zeros((2,)),
        covariance=jnp.eye(2),
        cross_covariance=jnp.eye(2),
        correlation=jnp.eye(2),
        lag_autocorrelation=jnp.ones((2,)),
        mean_relative_error=jnp.asarray(mean_error),
        covariance_relative_error=jnp.asarray(covariance_error),
        event_shape=(),
        num_samples=1024,
    )


def test_complete_trajectory_scores_reject_independent_time_marginals():
    targets = _ar1(jr.key(0), (96, 6))
    path_forecast = _ar1(jr.key(1), (256, 96, 6))
    independent_time_forecast = jr.normal(jr.key(2), (256, 96, 6))

    path_scores = phx.uq.trajectory_score_diagnostics(path_forecast, targets)
    independent_scores = phx.uq.trajectory_score_diagnostics(
        independent_time_forecast, targets
    )

    assert path_scores.event_shape == (6,)
    assert path_scores.valid_energy_cases == 96
    assert (
        path_scores.trajectory_energy_score < independent_scores.trajectory_energy_score
    )
    assert path_scores.variogram_score < 0.75 * independent_scores.variogram_score


def test_jump_event_diagnostics_recover_counts_channels_and_marks():
    candidate = _synthetic_events(jr.key(3))
    reference = _synthetic_events(jr.key(4))

    diagnostics = phx.uq.jump_event_diagnostics(
        candidate,
        t0=0.0,
        t1=1.0,
        num_channels=2,
        reference=reference,
    )

    assert diagnostics.reference is not None
    assert jnp.abs(diagnostics.candidate.count_mean - 3.0) < 0.1
    assert jnp.allclose(
        diagnostics.candidate.channel_probabilities,
        jnp.asarray([2.0 / 3.0, 1.0 / 3.0]),
        atol=0.02,
    )
    assert jnp.allclose(
        diagnostics.candidate.mark_mean,
        jnp.asarray([0.0, 1.0]),
        atol=0.02,
    )
    assert diagnostics.count_wasserstein_distance < 0.1
    assert diagnostics.channel_frequency_l1 < 0.03


def test_first_passage_diagnostics_require_the_correct_analytic_law():
    hitting_times = jr.exponential(jr.key(5), (4096,))
    observed = hitting_times <= 2.0
    censored = jnp.where(observed, hitting_times, jnp.inf)
    evaluation_times = jnp.asarray([0.25, 0.5, 1.0, 1.5, 2.0])

    correct = phx.uq.first_passage_diagnostics(
        censored,
        observed,
        evaluation_times,
        lambda time: 1.0 - jnp.exp(-time),
        horizon=2.0,
    )
    wrong = phx.uq.first_passage_diagnostics(
        censored,
        observed,
        evaluation_times,
        lambda time: 1.0 - jnp.exp(-2.0 * time),
        horizon=2.0,
    )

    assert correct.passed
    assert not wrong.passed


def test_paired_refinement_is_the_only_numerical_variance_evidence():
    exact = jnp.asarray([1.0, -2.0])
    fine_errors = jnp.asarray([[0.1, -0.2], [0.2, -0.1], [-0.1, 0.3]])
    fine = exact + fine_errors
    coarse = exact + 4.0 * fine_errors
    numerical = phx.uq.paired_refinement_uncertainty(
        coarse,
        fine,
        refinement_ratio=2.0,
        convergence_order=2.0,
    )
    process = jnp.asarray([-1.0, 0.0, 1.0])[:, None]
    prediction = phx.uq.PredictiveField(
        cx.Field(process + exact, dims=("process", "x")),
        (phx.uq.SampleAxis("process", "process"),),
    )

    decomposition = phx.uq.predictive_variance_decomposition(
        prediction,
        numerical_uncertainty=numerical,
    )

    assert decomposition.order == ("process", "numerical")
    assert jnp.allclose(numerical.mean_squared_error, jnp.mean(fine_errors**2, axis=0))
    assert jnp.allclose(
        decomposition.components["numerical"], numerical.mean_squared_error
    )
    assert jnp.allclose(decomposition.reconstructed, decomposition.total)

    mislabeled = phx.uq.PredictiveField(
        cx.Field(jnp.zeros((3, 2)), dims=("refinement", "x")),
        (phx.uq.SampleAxis("refinement", "numerical"),),
    )
    with pytest.raises(ValueError, match="not refinement evidence"):
        phx.uq.predictive_variance_decomposition(mislabeled)


def test_process_calibration_uses_disjoint_cases_and_retains_raw_scores():
    with pytest.raises(ValueError, match="must be disjoint"):
        phx.uq.ProcessValidationSplit(("shared",), ("shared",), ("test",))

    split = _split(calibration=19, test=128)
    location = jnp.zeros((19, 3))
    scale = jnp.ones((19, 3))
    alternating = jnp.where(jnp.arange(19)[:, None] % 2 == 0, 2.0, -2.0)
    target = jnp.broadcast_to(alternating, (19, 3))
    scale_calibrator = phx.uq.HorizonScaleCalibrator.fit(
        location,
        scale,
        target,
        jnp.asarray([0.25, 0.5, 1.0]),
        split,
    )
    trajectory_calibrator = phx.uq.ProcessConformalCalibrator.calibrate_trajectory(
        location,
        target,
        split,
        alpha=0.1,
    )

    targets = jr.normal(jr.key(6), (128, 3))
    raw_samples = 0.3 * jr.normal(jr.key(7), (256, 128, 3))
    calibrated_samples = jr.normal(jr.key(8), (256, 128, 3))
    report = phx.uq.process_calibration_report(
        raw_samples,
        calibrated_samples,
        targets,
        jnp.asarray([0.25, 0.5, 1.0]),
        split,
    )

    assert jnp.allclose(scale_calibrator.scale_multiplier, 2.0)
    assert trajectory_calibrator.kind == "trajectory"
    assert report.raw_horizon.marginal_crps.shape == (3,)
    assert report.calibrated_horizon.marginal_crps.shape == (3,)
    assert jnp.mean(report.calibrated_pointwise_coverage_error) < jnp.mean(
        report.raw_pointwise_coverage_error
    )


def test_shift_matrix_requires_every_promoted_stochastic_shift():
    names = ("baseline", "horizon", "covariance", "initial", "regime")
    kinds = (
        "in_distribution",
        "rollout_horizon",
        "covariance",
        "initial_condition",
        "parameter_regime",
    )
    raw_scores = jnp.asarray(
        [
            [1.00, 1.10, 1.05, 1.10, 1.15],
            [1.02, 1.12, 1.06, 1.12, 1.16],
            [0.98, 1.08, 1.04, 1.08, 1.14],
        ]
    )
    calibrated_scores = raw_scores * jnp.asarray([1.0, 0.95, 0.96, 0.96, 0.95])
    coverages = jnp.full_like(raw_scores, 0.9)

    matrix = phx.uq.process_shift_evaluation_matrix(
        raw_scores,
        calibrated_scores,
        coverages,
        coverages,
        scenario_names=names,
        shift_kinds=kinds,
        seeds=(11, 12, 13),
    )

    assert matrix.paired_reference_excess is None
    assert matrix.calibrated_score.mean.shape == (5,)
    assert matrix.worst_calibrated_score_degradation_upper < 0.15

    with pytest.raises(ValueError, match="missing required scenarios"):
        phx.uq.process_shift_evaluation_matrix(
            raw_scores[:, :-1],
            calibrated_scores[:, :-1],
            coverages[:, :-1],
            coverages[:, :-1],
            scenario_names=names[:-1],
            shift_kinds=kinds[:-1],
            seeds=(11, 12, 13),
        )


def test_retention_report_rejects_broken_statistics_and_provenance():
    split = _split(calibration=19, test=128)
    targets = jr.normal(jr.key(9), (128, 3))
    samples = jr.normal(jr.key(10), (128, 128, 3))
    calibration = phx.uq.process_calibration_report(
        samples,
        samples,
        targets,
        jnp.asarray([0.25, 0.5, 1.0]),
        split,
    )
    names = ("baseline", "horizon", "covariance", "initial", "regime")
    kinds = (
        "in_distribution",
        "rollout_horizon",
        "covariance",
        "initial_condition",
        "parameter_regime",
    )
    scores = jnp.asarray(
        [
            [1.0, 1.02, 1.01, 1.02, 1.03],
            [1.0, 1.01, 1.02, 1.01, 1.02],
            [1.0, 1.02, 1.01, 1.02, 1.01],
        ]
    )
    coverages = jnp.full_like(scores, 0.9)
    shifts = phx.uq.process_shift_evaluation_matrix(
        scores,
        scores,
        coverages,
        coverages,
        scenario_names=names,
        shift_kinds=kinds,
        seeds=(20, 21, 22),
    )
    prediction = phx.uq.PredictiveField(
        cx.Field(jnp.asarray([[-1.0], [0.0], [1.0]]), dims=("process", "x")),
        (phx.uq.SampleAxis("process", "process"),),
    )
    decomposition = phx.uq.predictive_variance_decomposition(prediction)
    zero = _monte_carlo_zero()
    semigroup = phx.uq.SemigroupMonteCarloDiagnostics(
        candidate=zero,
        reference=zero,
        excess=zero,
        num_samples=128,
        num_replicates=3,
    )
    thresholds = phx.uq.ProcessRetentionThresholds(
        max_calibrated_coverage_error_upper=0.5,
        max_shift_score_degradation_upper=0.2,
    )

    passed = phx.uq.process_retention_report(
        temporal=_temporal_diagnostics(),
        semigroup=semigroup,
        calibration=calibration,
        shifts=shifts,
        decomposition=decomposition,
        deterministic_replay=True,
        stable_realization_ids=True,
        rough_path_replay=True,
        broken_reference_rejected=True,
        raw_results_retained=True,
        calibrated_results_retained=True,
        uncertainty_sources=("process",),
        thresholds=thresholds,
    )
    rejected = phx.uq.process_retention_report(
        temporal=_temporal_diagnostics(mean_error=1.0),
        semigroup=semigroup,
        calibration=calibration,
        shifts=shifts,
        decomposition=decomposition,
        deterministic_replay=False,
        stable_realization_ids=True,
        rough_path_replay=True,
        broken_reference_rejected=False,
        raw_results_retained=True,
        calibrated_results_retained=True,
        uncertainty_sources=("process",),
        thresholds=thresholds,
    )

    assert passed.passed
    assert not rejected.passed
    assert rejected.failures == (
        "deterministic_replay",
        "broken_reference_rejected",
        "mean_relative_error",
    )
    with pytest.raises(RuntimeError, match="retention gates failed"):
        rejected.raise_for_failure()


def test_trajectory_diagnostics_reject_unidentifiable_or_invalid_score_inputs():
    targets = jnp.zeros((4, 3))
    with pytest.raises(ValueError, match="at least two forecast realizations"):
        phx.uq.trajectory_score_diagnostics(jnp.zeros((1, 4, 3)), targets)

    weights = jnp.ones((4, 3)).at[0, 0].set(-1.0)
    with pytest.raises(ValueError, match="finite and non-negative"):
        phx.uq.trajectory_score_diagnostics(
            jnp.zeros((2, 4, 3)),
            targets,
            weights=weights,
        )


def test_event_and_first_passage_diagnostics_reject_failed_path_evidence():
    events = _synthetic_events(jr.key(11), num_paths=8, capacity=4)
    failed = phx.stochastic.JumpEventBatch(
        events.times,
        events.channels,
        events.marks,
        events.valid,
        jnp.full((8,), phx.stochastic.JUMP_MAX_EVENTS),
    )
    with pytest.raises(ValueError, match="at least one successful path"):
        phx.uq.jump_event_diagnostics(
            failed,
            t0=0.0,
            t1=1.0,
            num_channels=2,
        )

    with pytest.raises(ValueError, match="finite and no later than horizon"):
        phx.uq.first_passage_diagnostics(
            jnp.asarray([0.25, jnp.inf]),
            jnp.asarray([True, True]),
            jnp.asarray([0.5, 1.0]),
            jnp.asarray([0.2, 0.5]),
            horizon=1.0,
        )


def test_numerical_diagnostics_reject_uncoupled_or_insufficient_refinements():
    with pytest.raises(ValueError, match="equal non-scalar shapes"):
        phx.uq.paired_refinement_uncertainty(
            jnp.zeros((3, 2)),
            jnp.zeros((4, 2)),
            refinement_ratio=2.0,
            convergence_order=1.0,
        )

    mask = jnp.asarray([[True, False], [True, False], [True, True]])
    with pytest.raises(ValueError, match="at least two valid refinement pairs"):
        phx.uq.paired_refinement_uncertainty(
            jnp.zeros((3, 2)),
            jnp.ones((3, 2)),
            refinement_ratio=2.0,
            convergence_order=1.0,
            mask=mask,
        )
