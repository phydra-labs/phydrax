import jax.random as jr
import jax.scipy as jsp
import numpy as np

import phydrax as phx


def test_targeted_continuous_priors_are_normalized_and_invert_reference_probabilities():
    cases = (
        (phx.uq.PowerLaw(-1.0, 0.2, 4.0), 0.2, 4.0),
        (phx.uq.TruncatedNormal(0.0, 1.0, -1.5, 2.0), -1.5, 2.0),
        (phx.uq.HalfNormal(1.2), 0.0, 8.0),
        (phx.uq.SineAngle(), 0.0, np.pi),
        (phx.uq.CosineAngle(), -0.5 * np.pi, 0.5 * np.pi),
    )
    for prior, lower, upper in cases:
        nodes = np.linspace(lower, upper, 20_001)
        density = np.exp(np.asarray(prior.log_prob(nodes)))
        np.testing.assert_allclose(np.trapezoid(density, nodes), 1.0, atol=2e-4)
        probabilities = np.asarray([0.1, 0.5, 0.9])
        values = prior.icdf(probabilities)
        assert np.all(np.asarray(prior.contains(values)))


def test_heavy_tail_priors_have_finite_samples_and_correct_center():
    cauchy = phx.uq.Cauchy(1.0, 2.0)
    student = phx.uq.StudentT(5.0, location=-0.5, scale=1.5)
    np.testing.assert_allclose(cauchy.icdf(0.5), 1.0, atol=1e-12)
    np.testing.assert_allclose(student.icdf(0.5), -0.5, atol=1e-10)
    assert np.all(np.isfinite(np.asarray(cauchy.sample(jr.key(1), (32,)))))
    assert np.all(np.isfinite(np.asarray(student.sample(jr.key(2), (32,)))))
    np.testing.assert_allclose(
        student.log_prob(-0.5),
        jsp.stats.t.logpdf(0.0, 5.0) - np.log(1.5),
        atol=1e-12,
    )


def test_truncated_normal_infinite_boundaries_have_finite_moments():
    standard = phx.uq.TruncatedNormal(0.0, 1.0, -np.inf, np.inf)
    positive = phx.uq.TruncatedNormal(0.0, 1.0, 0.0, np.inf)

    np.testing.assert_allclose((standard.mean, standard.variance), (0.0, 1.0))
    np.testing.assert_allclose(positive.mean, np.sqrt(2.0 / np.pi))
    np.testing.assert_allclose(positive.variance, 1.0 - 2.0 / np.pi)
    assert np.isneginf(phx.uq.Cauchy(0.0, 1.0).log_prob(np.nan))
    assert np.isneginf(phx.uq.StudentT(3.0).log_prob(np.inf))
