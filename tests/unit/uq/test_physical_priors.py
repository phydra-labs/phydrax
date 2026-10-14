import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import numpy as np
import pytest

import phydrax as phx
from phydrax import config as pxconfig


def test_physical_priors_scenario_1() -> None:
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
    standard = phx.uq.TruncatedNormal(0.0, 1.0, -np.inf, np.inf)
    positive = phx.uq.TruncatedNormal(0.0, 1.0, 0.0, np.inf)

    np.testing.assert_allclose((standard.mean, standard.variance), (0.0, 1.0))
    np.testing.assert_allclose(positive.mean, np.sqrt(2.0 / np.pi))
    np.testing.assert_allclose(positive.variance, 1.0 - 2.0 / np.pi)
    assert np.isneginf(phx.uq.Cauchy(0.0, 1.0).log_prob(np.nan))
    assert np.isneginf(phx.uq.StudentT(3.0).log_prob(np.inf))


def test_batched_truncated_normal_preserves_each_normalized_law() -> None:
    with pxconfig.context(pxconfig.NumericsConfig(enable_x64=True)):
        law = phx.uq.TruncatedNormal(
            0.0,
            np.asarray([1.0, 2.0], dtype=np.float64),
            np.asarray([-np.inf, 0.0], dtype=np.float64),
            np.inf,
        )
        # An ordinary unit normal and a scale-two half-normal are independent
        # closed-form oracles for distinct support/scale members of one batch.
        points = np.asarray([[0.0, 0.0], [1.0, 2.0]], dtype=np.float64)
        reference = -0.5 * points**2 / np.asarray([1.0, 4.0])
        reference -= 0.5 * np.log(2.0 * np.pi)
        np.testing.assert_allclose(law.log_prob(points), reference, rtol=1e-12)
        np.testing.assert_allclose(
            law.icdf(0.5), [0.0, 1.3489795003921634], rtol=1e-12, atol=1e-15
        )
        np.testing.assert_array_equal(
            law.contains(np.asarray([0.0, -0.01], dtype=np.float64)), [True, False]
        )
        assert law.batch_shape == (2,)
        assert law.event_shape == ()


@pytest.mark.parametrize(
    ("scale", "low", "high"),
    (
        ((1.0, -1.0), (0.0, 0.0), (1.0, 1.0)),
        ((1.0, 1.0), (0.0, 2.0), (1.0, 1.0)),
        ((1.0, 1.0), (0.0, 100.0), (1.0, 101.0)),
    ),
    ids=("invalid-scale-member", "empty-support-member", "zero-mass-member"),
)
def test_batched_truncated_normal_refuses_one_invalid_member(
    scale: tuple[float, float],
    low: tuple[float, float],
    high: tuple[float, float],
) -> None:
    with (
        pxconfig.context(pxconfig.NumericsConfig(enable_x64=True)),
        pytest.raises(ValueError),
    ):
        phx.uq.TruncatedNormal(
            jnp.asarray(0.0, dtype=jnp.float64),
            jnp.asarray(scale, dtype=jnp.float64),
            jnp.asarray(low, dtype=jnp.float64),
            jnp.asarray(high, dtype=jnp.float64),
        )
