#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
import pytest

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


def test_length_one_streaming_kernel_keeps_empty_history_across_blocks() -> None:
    plan = phx.signal.StreamingFFTConvolutionPlan(jnp.asarray([2.0]), 4)
    first_values = jnp.arange(4.0)
    second_values = jnp.arange(4.0, 8.0)

    first, state = plan.apply(first_values, plan.initial_state())
    second, state = plan.apply(second_values, state)

    assert state.history.shape == (0,)
    np.testing.assert_allclose(first, 2.0 * first_values)
    np.testing.assert_allclose(second, 2.0 * second_values)


def test_streaming_filter_states_reject_same_shaped_foreign_plans() -> None:
    convolution = phx.signal.StreamingFFTConvolutionPlan(jnp.asarray([1.0, 0.0]), 4)
    other_convolution = phx.signal.StreamingFFTConvolutionPlan(jnp.asarray([0.0, 1.0]), 4)
    with pytest.raises(ValueError, match="different plan"):
        convolution.apply(jnp.ones((4,)), other_convolution.initial_state())

    sos = phx.signal.design_iir_sos("butterworth", 2, 0.2)
    other_sos = phx.signal.design_iir_sos("butterworth", 2, 0.3)
    with pytest.raises(ValueError, match="different filter plan"):
        sos.apply(jnp.ones((4,)), state=other_sos.initial_state())


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


def test_nonuniform_resampling_is_jittable_and_differentiable_in_source_times() -> None:
    source = jnp.asarray((0.0, 0.4, 1.0))
    values = jnp.asarray((1.0, 2.0, -1.0))
    targets = jnp.asarray((0.2, 0.7))

    def total(source_times):
        return jnp.sum(phx.signal.resample_nonuniform(source_times, values, targets))

    assert jnp.isfinite(jax.jit(total)(source))
    assert jnp.all(jnp.isfinite(jax.jit(jax.grad(total))(source)))


@pytest.mark.parametrize(
    "call",
    (
        lambda: phx.signal.design_iir_sos("butterworth", 2.5, 0.2),
        lambda: phx.signal.design_fir(4.5, 0.2),
        lambda: phx.signal.STFTPlan(jnp.ones((4,)), 1.5),
        lambda: phx.signal.StreamingFFTConvolutionPlan(jnp.ones((2,)), 4.5),
    ),
)
def test_signal_topology_sizes_require_exact_integers(call) -> None:
    with pytest.raises(TypeError, match="integer"):
        call()


@pytest.mark.parametrize(
    "keywords",
    (
        {"sample_spacing": 0.0},
        {"time_bandwidth": 4.0},
        {"time_bandwidth": float("nan")},
        {"taper_count": 0},
        {"taper_count": 9},
    ),
)
def test_multitaper_rejects_invalid_parameter_domains(keywords) -> None:
    with pytest.raises(ValueError):
        phx.signal.multitaper_spectrum(jnp.ones((8,)), **keywords)
