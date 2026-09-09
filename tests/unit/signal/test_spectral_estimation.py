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
