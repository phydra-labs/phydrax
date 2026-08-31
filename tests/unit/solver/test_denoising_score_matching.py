import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


class _ConditionalGaussianScore(eqx.Module):
    process: phx.stochastic.AbstractGaussianDiffusion
    clean_state: jnp.ndarray

    def __call__(self, state, time):
        mean = self.process.transition_mean_scale(0.0, time) * self.clean_state
        scale = self.process.transition_scale(0.0, time)
        return -(state - mean) / scale**2


class _ZeroScore(eqx.Module):
    def __call__(self, state, time):
        del time
        return jnp.zeros_like(state)


def _score_function(model, process):
    state = phx.domain.HyperRectangle(
        jnp.full(process.state_shape, -100.0),
        jnp.full(process.state_shape, 100.0),
        label="x",
    )
    domain = state @ phx.domain.TimeInterval(0.0, process.terminal_time)
    return domain.Function("x", "t")(model)


def _data(clean, *, mask=None, independent=True):
    count = clean.shape[0]
    return phx.integration.weighted(
        clean,
        jnp.zeros((count,)),
        normalized=True,
        independent=independent,
        mask=mask,
        provenance="denoising-score-test",
    )


@pytest.mark.parametrize(
    "process",
    [
        phx.stochastic.VariancePreservingDiffusion(
            2, beta_minimum=0.2, beta_maximum=2.0
        ),
        phx.stochastic.VarianceExplodingDiffusion(
            2, initial_scale=0.02, terminal_scale=2.0
        ),
    ],
)
def test_exact_conditional_score_has_zero_denoising_objective(process):
    clean_state = jnp.asarray([0.4, -0.7])
    clean = jnp.broadcast_to(clean_state, (64, 2))
    score = _score_function(_ConditionalGaussianScore(process, clean_state), process)
    term = phx.terms.DenoisingScoreMatchingTerm(
        "score",
        _data(clean),
        process,
        phx.terms.UniformTimeSamplingPolicy(0.02, 0.9),
    )
    batch = term.sample(key=jr.key(1))
    diagnostics = term.diagnostics({"score": score}, key=jr.key(2), batch=batch)
    compiled = eqx.filter_jit(
        lambda function: term.loss({"score": function}, key=jr.key(2), batch=batch)
    )(score)

    assert diagnostics.passed
    assert diagnostics.valid_fraction == 1.0
    assert diagnostics.objective < 1e-24
    assert compiled < 1e-24
    assert jnp.isfinite(diagnostics.objective_standard_error)


def test_denoising_weighting_changes_only_declared_node_weights():
    process = phx.stochastic.VariancePreservingDiffusion(1)
    clean = jnp.zeros((32, 1))
    score = _score_function(_ZeroScore(), process)
    policy = phx.terms.UniformTimeSamplingPolicy(0.05, 0.8)
    objectives = []
    for weighting in ("unit", "conditional-variance", "diffusion-rate"):
        term = phx.terms.DenoisingScoreMatchingTerm(
            "score",
            _data(clean),
            process,
            policy,
            weighting=weighting,
        )
        batch = term.sample(key=jr.key(3))
        diagnostics = term.diagnostics({"score": score}, key=jr.key(4), batch=batch)
        objectives.append(diagnostics.objective)
        assert diagnostics.weighting == weighting
        assert diagnostics.minimum_objective_weight > 0.0
    assert not jnp.allclose(jnp.asarray(objectives), objectives[0])


def test_denoising_masks_invalid_samples_and_rejects_empty_mass():
    process = phx.stochastic.VariancePreservingDiffusion(1)
    clean = jnp.asarray([[0.0], [jnp.nan], [1.0]])
    score = _score_function(_ZeroScore(), process)
    term = phx.terms.DenoisingScoreMatchingTerm(
        "score",
        _data(clean, mask=jnp.asarray([True, False, True])),
        process,
        phx.terms.UniformTimeSamplingPolicy(0.05, 0.8),
    )
    diagnostics = term.diagnostics({"score": score}, key=jr.key(5))
    assert diagnostics.valid_fraction == pytest.approx(2.0 / 3.0)
    assert jnp.isfinite(diagnostics.objective)

    empty = phx.terms.DenoisingScoreMatchingTerm(
        "score",
        _data(jnp.zeros((3, 1)), mask=jnp.zeros((3,), dtype=bool)),
        process,
        phx.terms.UniformTimeSamplingPolicy(0.05, 0.8),
    )
    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="no valid"):
        empty.loss({"score": score}, key=jr.key(6))


def test_resampled_denoising_provider_runs_once_per_materialized_batch():
    process = phx.stochastic.VariancePreservingDiffusion(1)
    calls = []

    def provider(key):
        calls.append(key)
        return _data(jnp.zeros((8, 1)))

    term = phx.terms.DenoisingScoreMatchingTerm(
        "score",
        provider,
        process,
        phx.terms.UniformTimeSamplingPolicy(0.05, 0.8),
        sampling_mode="resample",
    )
    batch = term.sample(key=jr.key(7))
    score = _score_function(_ZeroScore(), process)
    term.loss({"score": score}, key=jr.key(8), batch=batch)

    assert len(calls) == 1


def test_denoising_rejects_zero_noise_endpoint_and_wrong_score_shape():
    process = phx.stochastic.VariancePreservingDiffusion(1)
    with pytest.raises(ValueError, match="strictly positive"):
        phx.terms.DenoisingScoreMatchingTerm(
            "score",
            _data(jnp.zeros((4, 1))),
            process,
            phx.terms.UniformTimeSamplingPolicy(0.0, 0.8),
        )

    wrong = _score_function(lambda state, time: jnp.zeros((2,)), process)
    term = phx.terms.DenoisingScoreMatchingTerm(
        "score",
        _data(jnp.zeros((4, 1))),
        process,
        phx.terms.UniformTimeSamplingPolicy(0.05, 0.8),
    )
    with pytest.raises(ValueError, match="same shape"):
        term.loss({"score": wrong}, key=jr.key(9))
