#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def test_sos_filter_matches_scipy_design_response_and_streaming_state() -> None:
    plan = phx.signal.design_iir_sos("butterworth", 4, 0.2)
    impulse = jnp.zeros((128,)).at[0].set(1.0)

    full = plan.apply(impulse)
    first = plan.apply(impulse[:64])
    second = plan.apply(impulse[64:], state=first.state)

    assert bool(full.finite)
    np.testing.assert_allclose(
        jnp.concatenate((first.values, second.values)), full.values, atol=1.0e-12
    )
    assert jnp.max(jnp.abs(full.values[80:])) < 1.0e-4


def test_stft_inverse_uses_overlap_normalization() -> None:
    signal = jnp.sin(0.1 * jnp.arange(128))
    plan = phx.signal.STFTPlan(jnp.hanning(32), 8, fft_size=32)

    spectrum = plan.transform(signal)
    restored = plan.inverse(spectrum, length=signal.size)

    np.testing.assert_allclose(restored[1:-1], signal[1:-1], atol=2.0e-6)


def test_streaming_fft_convolution_matches_direct_causal_filtering() -> None:
    kernel = jnp.asarray((0.25, 0.5, 0.25))
    values = jnp.arange(16.0)
    plan = phx.signal.StreamingFFTConvolutionPlan(kernel, 8)

    first, state = plan.apply(values[:8], plan.initial_state())
    second, _ = plan.apply(values[8:], state)
    expected = jnp.convolve(values, kernel, mode="full")[: values.size]

    np.testing.assert_allclose(jnp.concatenate((first, second)), expected, atol=1.0e-12)


def test_multitaper_cross_spectrum_and_coherence_detect_shared_tone() -> None:
    samples = jnp.arange(256)
    left = jnp.sin(2.0 * jnp.pi * 0.125 * samples)
    right = 2.0 * left

    result = phx.signal.multitaper_spectrum(left)
    frequencies, cross, coherence = phx.signal.cross_spectrum_and_coherence(left, right)
    peak = int(jnp.argmax(result.spectrum))

    assert jnp.isclose(result.frequencies[peak], 0.125)
    assert jnp.isclose(frequencies[peak], 0.125)
    assert jnp.abs(cross[peak]) > 0.0
    assert coherence[peak] > 0.999


def test_nonuniform_resampling_preserves_affine_vector_fields() -> None:
    source = jnp.asarray((0.0, 0.1, 0.4, 1.0))
    target = jnp.linspace(0.0, 1.0, 11)
    values = jnp.stack((2.0 * source + 1.0, -3.0 * source + 2.0), axis=-1)

    result = phx.signal.resample_nonuniform(source, values, target)

    expected = jnp.stack((2.0 * target + 1.0, -3.0 * target + 2.0), axis=-1)
    np.testing.assert_allclose(result, expected, atol=1.0e-12)
