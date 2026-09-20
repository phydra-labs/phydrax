#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""IIR, time-frequency, streaming, multitaper, and irregular-sample DSP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
import scipy.signal as scipy_signal
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._interpolation import linear_interpolate


IIRDesignKind: TypeAlias = Literal["butterworth", "chebyshev1", "chebyshev2", "elliptic"]


@dataclass(frozen=True, slots=True)
class SOSFilterState:
    delays: Array


@dataclass(frozen=True, slots=True)
class SOSFilterResult:
    values: Array
    state: SOSFilterState
    finite: Array


@dataclass(frozen=True, slots=True)
class SOSFilterPlan:
    """Direct-form-II-transposed second-order-section cascade."""

    sections: Array
    plan_id: str

    def __init__(self, sections: ArrayLike, /):
        sections_ = np.asarray(sections, dtype=np.float64)
        if sections_.ndim != 2 or sections_.shape[1] != 6 or sections_.shape[0] == 0:
            raise ValueError("sections must have shape (section, 6).")
        if not np.all(np.isfinite(sections_)) or np.any(sections_[:, 3] == 0.0):
            raise ValueError("SOS coefficients must be finite with nonzero a0.")
        sections_ = sections_ / sections_[:, 3:4]
        payload = {"kind": "sos-filter-plan", "sections": sections_.tolist()}
        object.__setattr__(self, "sections", jnp.asarray(sections_))
        object.__setattr__(self, "plan_id", canonical_fingerprint(payload))

    def initial_state(self, sample_shape: tuple[int, ...] = ()) -> SOSFilterState:
        return SOSFilterState(
            jnp.zeros(
                (self.sections.shape[0], 2, *sample_shape), dtype=self.sections.dtype
            )
        )

    def apply(
        self,
        values: ArrayLike,
        /,
        *,
        state: SOSFilterState | None = None,
    ) -> SOSFilterResult:
        signal = jnp.asarray(values)
        if signal.ndim < 1:
            raise ValueError("SOS input must begin with a time axis.")
        state_ = self.initial_state(signal.shape[1:]) if state is None else state
        if state_.delays.shape != (self.sections.shape[0], 2, *signal.shape[1:]):
            raise ValueError("SOS state does not match sections and sample shape.")

        def sample_step(delays, sample):
            output = sample
            updated = []
            for index in range(self.sections.shape[0]):
                b0, b1, b2, _, a1, a2 = self.sections[index]
                first, second = delays[index]
                filtered = b0 * output + first
                updated.append(
                    jnp.stack(
                        (
                            b1 * output - a1 * filtered + second,
                            b2 * output - a2 * filtered,
                        )
                    )
                )
                output = filtered
            return jnp.stack(updated), output

        final, filtered = jax.lax.scan(sample_step, state_.delays, signal)
        return SOSFilterResult(
            filtered, SOSFilterState(final), jnp.all(jnp.isfinite(filtered))
        )


def design_iir_sos(
    kind: IIRDesignKind,
    order: int,
    cutoff: float | tuple[float, float],
    /,
    *,
    btype: Literal["lowpass", "highpass", "bandpass", "bandstop"] = "lowpass",
    ripple_db: float = 1.0,
    attenuation_db: float = 40.0,
) -> SOSFilterPlan:
    """Design a normalized-digital IIR filter and return one stable SOS plan."""

    order_ = int(order)
    if order_ <= 0:
        raise ValueError("IIR order must be positive.")
    if kind == "butterworth":
        sections = scipy_signal.butter(order_, cutoff, btype=btype, output="sos")
    elif kind == "chebyshev1":
        sections = scipy_signal.cheby1(
            order_, ripple_db, cutoff, btype=btype, output="sos"
        )
    elif kind == "chebyshev2":
        sections = scipy_signal.cheby2(
            order_, attenuation_db, cutoff, btype=btype, output="sos"
        )
    elif kind == "elliptic":
        sections = scipy_signal.ellip(
            order_, ripple_db, attenuation_db, cutoff, btype=btype, output="sos"
        )
    else:
        raise ValueError(f"Unknown IIR design kind {kind!r}.")
    return SOSFilterPlan(sections)


def design_fir(
    taps: int,
    cutoff: float | tuple[float, float],
    /,
    *,
    pass_zero: bool | Literal["bandpass", "lowpass", "highpass", "bandstop"] = True,
    window: str = "hann",
) -> Array:
    """Design one normalized-digital linear-phase FIR response."""

    count = int(taps)
    if count <= 0:
        raise ValueError("FIR tap count must be positive.")
    return jnp.asarray(
        scipy_signal.firwin(count, cutoff, pass_zero=pass_zero, window=window)
    )


@dataclass(frozen=True, slots=True)
class STFTPlan:
    window: Array
    hop_size: int
    fft_size: int
    plan_id: str

    def __init__(
        self, window: ArrayLike, hop_size: int, /, *, fft_size: int | None = None
    ):
        window_ = np.asarray(window, dtype=np.float64)
        hop = int(hop_size)
        fft = int(window_.size if fft_size is None else fft_size)
        if window_.ndim != 1 or window_.size == 0 or not np.all(np.isfinite(window_)):
            raise ValueError("STFT window must be a finite non-empty vector.")
        if hop <= 0 or hop > window_.size or fft < window_.size:
            raise ValueError("STFT hop/FFT sizes are incompatible with the window.")
        payload = {
            "kind": "stft-plan",
            "window": window_.tolist(),
            "hop_size": hop,
            "fft_size": fft,
        }
        object.__setattr__(self, "window", jnp.asarray(window_))
        object.__setattr__(self, "hop_size", hop)
        object.__setattr__(self, "fft_size", fft)
        object.__setattr__(self, "plan_id", canonical_fingerprint(payload))

    def transform(self, values: ArrayLike, /) -> Array:
        signal = jnp.asarray(values)
        if signal.ndim != 1 or signal.size < self.window.size:
            raise ValueError("STFT input must be a long-enough one-dimensional signal.")
        frame_count = 1 + (signal.size - self.window.size) // self.hop_size
        starts = jnp.arange(frame_count) * self.hop_size
        frames = jax.vmap(
            lambda start: jax.lax.dynamic_slice(signal, (start,), (self.window.size,))
        )(starts)
        return jnp.fft.rfft(frames * self.window, n=self.fft_size, axis=-1)

    def inverse(self, spectrum: ArrayLike, /, *, length: int | None = None) -> Array:
        coefficients = jnp.asarray(spectrum)
        if coefficients.ndim != 2 or coefficients.shape[1] != self.fft_size // 2 + 1:
            raise ValueError("STFT spectrum shape does not match the plan.")
        frames = (
            jnp.fft.irfft(coefficients, n=self.fft_size, axis=-1)[:, : self.window.size]
            * self.window
        )
        output_length = self.hop_size * (frames.shape[0] - 1) + self.window.size
        output = jnp.zeros((output_length,), dtype=frames.dtype)
        normalization = jnp.zeros_like(output)
        for index in range(frames.shape[0]):
            start = index * self.hop_size
            output = output.at[start : start + self.window.size].add(frames[index])
            normalization = normalization.at[start : start + self.window.size].add(
                self.window * self.window
            )
        safe = jnp.where(normalization > jnp.finfo(output.dtype).eps, normalization, 1.0)
        reconstructed = output / safe
        return reconstructed if length is None else reconstructed[: int(length)]


@dataclass(frozen=True, slots=True)
class FFTConvolutionState:
    history: Array


@dataclass(frozen=True, slots=True)
class StreamingFFTConvolutionPlan:
    kernel: Array
    block_size: int
    fft_size: int
    plan_id: str

    def __init__(self, kernel: ArrayLike, block_size: int, /):
        kernel_ = np.asarray(kernel, dtype=np.float64)
        block = int(block_size)
        if kernel_.ndim != 1 or kernel_.size == 0 or not np.all(np.isfinite(kernel_)):
            raise ValueError("Streaming convolution kernel must be a finite vector.")
        if block <= 0:
            raise ValueError("block_size must be positive.")
        fft = 1
        while fft < block + kernel_.size - 1:
            fft *= 2
        payload = {
            "kind": "streaming-fft-convolution",
            "kernel": kernel_.tolist(),
            "block": block,
        }
        object.__setattr__(self, "kernel", jnp.asarray(kernel_))
        object.__setattr__(self, "block_size", block)
        object.__setattr__(self, "fft_size", fft)
        object.__setattr__(self, "plan_id", canonical_fingerprint(payload))

    def initial_state(self) -> FFTConvolutionState:
        return FFTConvolutionState(
            jnp.zeros((self.kernel.size - 1,), dtype=self.kernel.dtype)
        )

    def apply(self, block: ArrayLike, state: FFTConvolutionState, /):
        values = jnp.asarray(block)
        if values.shape != (self.block_size,):
            raise ValueError("Streaming block shape does not match the plan.")
        extended = jnp.concatenate((state.history, values))
        transformed = jnp.fft.rfft(extended, self.fft_size) * jnp.fft.rfft(
            self.kernel, self.fft_size
        )
        full = jnp.fft.irfft(transformed, self.fft_size)
        start = self.kernel.size - 1
        output = full[start : start + self.block_size]
        next_state = FFTConvolutionState(extended[-(self.kernel.size - 1) :])
        return output, next_state


@dataclass(frozen=True, slots=True)
class MultitaperSpectrumResult:
    frequencies: Array
    spectrum: Array
    individual_spectra: Array
    degrees_of_freedom: float


def multitaper_spectrum(
    values: ArrayLike,
    /,
    *,
    time_bandwidth: float = 3.5,
    taper_count: int | None = None,
    sample_spacing: float = 1.0,
) -> MultitaperSpectrumResult:
    signal = jnp.asarray(values)
    if signal.ndim != 1 or signal.size < 2:
        raise ValueError("Multitaper input must be a one-dimensional signal.")
    count = (
        max(1, int(2 * time_bandwidth) - 1) if taper_count is None else int(taper_count)
    )
    tapers = jnp.asarray(
        scipy_signal.windows.dpss(signal.size, time_bandwidth, Kmax=count)
    )
    transformed = jnp.fft.rfft(tapers * signal[None, :], axis=-1)
    individual = jnp.abs(transformed) ** 2
    spectrum = jnp.mean(individual, axis=0)
    frequencies = jnp.fft.rfftfreq(signal.size, d=sample_spacing)
    return MultitaperSpectrumResult(frequencies, spectrum, individual, float(2 * count))


def cross_spectrum_and_coherence(
    left: ArrayLike,
    right: ArrayLike,
    /,
    *,
    time_bandwidth: float = 3.5,
    taper_count: int | None = None,
    sample_spacing: float = 1.0,
):
    left_ = jnp.asarray(left)
    right_ = jnp.asarray(right)
    if left_.shape != right_.shape or left_.ndim != 1:
        raise ValueError("Cross-spectrum signals must be aligned vectors.")
    count = (
        max(1, int(2 * time_bandwidth) - 1) if taper_count is None else int(taper_count)
    )
    tapers = jnp.asarray(
        scipy_signal.windows.dpss(left_.size, time_bandwidth, Kmax=count)
    )
    left_fft = jnp.fft.rfft(tapers * left_[None, :], axis=-1)
    right_fft = jnp.fft.rfft(tapers * right_[None, :], axis=-1)
    cross = jnp.mean(left_fft * jnp.conj(right_fft), axis=0)
    left_power = jnp.mean(jnp.abs(left_fft) ** 2, axis=0)
    right_power = jnp.mean(jnp.abs(right_fft) ** 2, axis=0)
    denominator = jnp.maximum(left_power * right_power, jnp.finfo(left_power.dtype).tiny)
    coherence = jnp.clip(jnp.abs(cross) ** 2 / denominator, 0.0, 1.0)
    frequencies = jnp.fft.rfftfreq(left_.size, d=sample_spacing)
    return frequencies, cross, coherence


def resample_nonuniform(
    source_times: ArrayLike,
    values: ArrayLike,
    target_times: ArrayLike,
    /,
) -> Array:
    """Piecewise-linear resampling from strictly increasing irregular samples."""

    source = jnp.asarray(source_times)
    field = jnp.asarray(values)
    target = jnp.asarray(target_times)
    if source.ndim != 1 or target.ndim != 1 or field.shape[0] != source.size:
        raise ValueError("Nonuniform resampling requires aligned leading sample axes.")
    host = np.asarray(source)
    if not np.all(np.isfinite(host)) or np.any(np.diff(host) <= 0.0):
        raise ValueError("source_times must be finite and strictly increasing.")
    return linear_interpolate(
        source,
        field,
        target,
        axis=0,
        bounds="clip",
    ).values


__all__ = [
    "FFTConvolutionState",
    "IIRDesignKind",
    "MultitaperSpectrumResult",
    "SOSFilterPlan",
    "SOSFilterResult",
    "SOSFilterState",
    "STFTPlan",
    "StreamingFFTConvolutionPlan",
    "cross_spectrum_and_coherence",
    "design_fir",
    "design_iir_sos",
    "multitaper_spectrum",
    "resample_nonuniform",
]
