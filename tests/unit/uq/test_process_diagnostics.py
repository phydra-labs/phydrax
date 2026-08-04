import coordax as cx
import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


def _gaussian_process():
    return phx.stochastic.LatentGaussianCoefficientProcess(
        jnp.asarray([0.2, -0.1]),
        jnp.asarray([[0.4, 0.1], [0.0, 0.3]]),
        label="diagnostic-process",
    )


def test_horizon_scores_preserve_horizons_and_reject_biased_forecasts():
    target_key, sample_key = jr.split(jr.key(0))
    targets = jr.normal(target_key, (128, 3, 2))
    calibrated = jr.normal(sample_key, (256, 128, 3, 2))
    biased = calibrated + 2.0
    horizons = jnp.asarray([0.1, 0.5, 1.0])

    reference = phx.uq.horizon_score_diagnostics(
        calibrated,
        targets,
        horizons,
        lower_quantile=0.05,
        upper_quantile=0.95,
    )
    shifted = phx.uq.horizon_score_diagnostics(biased, targets, horizons)

    assert reference.marginal_crps.shape == (3,)
    assert reference.energy_score.shape == (3,)
    assert jnp.all(reference.valid_cases == 128)
    assert jnp.all(reference.pointwise_coverage > 0.82)
    assert jnp.all(reference.pointwise_coverage < 0.98)
    assert jnp.mean(reference.marginal_crps) < jnp.mean(shifted.marginal_crps)
    assert jnp.mean(reference.energy_score) < jnp.mean(shifted.energy_score)


def test_uniform_pit_and_exchangeable_observable_ranks_pass_dkw_gate():
    pit = (jnp.arange(400, dtype=float) + 0.5) / 400.0
    analytic = phx.uq.pit_diagnostics(pit, bins=20)

    draws = jr.normal(jr.key(1), (32, 512))
    ensemble = phx.uq.observable_rank_diagnostics(
        draws[:-1],
        draws[-1],
        key=jr.key(2),
    )

    assert analytic.passed
    assert analytic.valid_count == 400
    assert ensemble.passed
    assert ensemble.histogram.shape == (32,)
    assert ensemble.valid_count == 512


def test_semigroup_diagnostics_report_reference_monte_carlo_floor():
    process = _gaussian_process()
    diagnostics = phx.uq.semigroup_mc_diagnostics(
        process,
        jnp.asarray([1.0, -1.0]),
        t0=0.0,
        tmid=0.4,
        t1=1.0,
        key=jr.key(3),
        num_samples=512,
        num_replicates=6,
        reference_law=process,
    )

    assert diagnostics.reference is not None
    assert diagnostics.excess is not None
    assert diagnostics.candidate.replicates.shape == (6,)
    assert diagnostics.candidate.lower <= diagnostics.candidate.mean
    assert diagnostics.candidate.mean <= diagnostics.candidate.upper
    assert jnp.abs(diagnostics.excess.mean) < 0.02


def test_temporal_moments_detect_ar1_dependence():
    rho = 0.75
    innovations = jr.normal(jr.key(4), (4096, 6))
    states = [innovations[:, 0] / jnp.sqrt(1.0 - rho**2)]
    for index in range(1, innovations.shape[1]):
        states.append(rho * states[-1] + innovations[:, index])
    trajectories = jnp.stack(states, axis=1)

    diagnostics = phx.uq.temporal_moment_diagnostics(
        trajectories,
        jnp.arange(6, dtype=float),
    )

    assert diagnostics.mean.shape == (6,)
    assert diagnostics.covariance.shape == (6, 6)
    assert diagnostics.cross_covariance.shape == (6, 6)
    assert diagnostics.lag_autocorrelation.shape == (6,)
    assert jnp.abs(diagnostics.lag_autocorrelation[1] - rho) < 0.04


def test_predictive_variance_decomposition_obeys_total_variance_identity():
    epistemic = jnp.asarray([-1.0, 0.0, 1.0])[:, None, None]
    process = jnp.asarray([-2.0, -0.5, 0.5, 2.0])[None, :, None]
    values = epistemic + process + jnp.asarray([0.0, 1.0])[None, None, :]
    prediction = phx.uq.PredictiveField(
        cx.Field(values, dims=("epistemic", "process", "x")),
        (
            phx.uq.SampleAxis("epistemic", "epistemic"),
            phx.uq.SampleAxis("process", "process"),
        ),
        conditional_variance=cx.Field(jnp.asarray([0.25, 0.5]), dims=("x",)),
    )

    diagnostics = phx.uq.predictive_variance_decomposition(
        prediction,
        order=("process", "epistemic"),
    )

    assert diagnostics.order == ("observation", "process", "epistemic")
    assert jnp.allclose(
        diagnostics.components["observation"],
        jnp.asarray([0.25, 0.5]),
    )
    assert jnp.allclose(diagnostics.reconstructed, diagnostics.total)
    assert jnp.max(jnp.abs(diagnostics.remainder)) < 1e-6
