import jax.numpy as jnp
import numpy as np

from phydrax.signal import WelchSpectrumPlan


def test_welch_recovers_sinusoid_frequency_and_variance():
    sample_interval = 0.01
    time = jnp.arange(1000) * sample_interval
    signal = jnp.sin(2.0 * jnp.pi * 10.0 * time)
    result = WelchSpectrumPlan(sample_interval, 200).evaluate(signal)
    peak = result.frequencies[jnp.argmax(result.power_spectral_density)]
    frequency_step = result.frequencies[1] - result.frequencies[0]
    variance = jnp.sum(result.power_spectral_density) * frequency_step

    np.testing.assert_allclose(peak, 10.0, atol=0.5)
    np.testing.assert_allclose(variance, 0.5, rtol=2.0e-3)
    assert result.segment_count == 9


def test_welch_preserves_leading_signal_axes():
    sample_interval = 0.02
    time = jnp.arange(256) * sample_interval
    signals = jnp.stack(
        (
            jnp.sin(2.0 * jnp.pi * 5.0 * time),
            jnp.sin(2.0 * jnp.pi * 10.0 * time),
        )
    )
    result = WelchSpectrumPlan(sample_interval, 128, overlap=64).evaluate(signals)
    peaks = result.frequencies[jnp.argmax(result.power_spectral_density, axis=-1)]

    np.testing.assert_allclose(peaks, (5.078125, 10.15625), atol=0.4)
    assert result.power_spectral_density.shape == (2, 65)


def test_welch_tukey_median_policy_retains_one_sided_power():
    sample_interval = 0.01
    time = jnp.arange(1200) * sample_interval
    signal = jnp.sin(2.0 * jnp.pi * 8.0 * time)
    result = WelchSpectrumPlan(
        sample_interval,
        200,
        overlap=100,
        window="tukey",
        tukey_alpha=0.2,
        average="median",
    ).evaluate(signal)
    frequency_step = result.frequencies[1] - result.frequencies[0]
    pairs = 2.0 * np.arange(1, (result.segment_count - 1) // 2 + 1)
    median_bias = 1.0 + np.sum(1.0 / (pairs + 1.0) - 1.0 / pairs)
    np.testing.assert_allclose(
        jnp.sum(result.power_spectral_density) * frequency_step,
        0.5 / median_bias,
        rtol=5.0e-2,
    )
    assert result.segment_count == 11
