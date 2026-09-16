#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax._fingerprint import array_tree_fingerprint, canonical_fingerprint
from phydrax._spectral._fourier import fourier_resample as _fourier_resample
from phydrax._strict import StrictModule
from phydrax._trainable import NonTrainableState

from ._windows import blackman_window, hamming_window, hann_window, tukey_window


def fourier_resample(
    values: ArrayLike,
    output_shape: Sequence[int],
    /,
    *,
    axes: Sequence[int] | None = None,
    phase_offsets: Sequence[ArrayLike] | None = None,
) -> Array:
    """Band-limited periodic resampling over explicit or trailing signal axes.

    When ``axes`` is omitted, the trailing ``len(output_shape)`` axes are
    transformed. Unselected axes are independent payload or batch axes.
    """
    array = jnp.asarray(values)
    shape = tuple(int(size) for size in output_shape)
    if not shape or any(size <= 0 for size in shape):
        raise ValueError("output_shape must contain positive signal sizes.")
    if axes is None:
        if array.ndim < len(shape):
            raise ValueError(
                "The input rank must be at least the number of output dimensions."
            )
        resolved_axes = tuple(range(array.ndim - len(shape), array.ndim))
    else:
        resolved_axes = tuple(axes)
    return _fourier_resample(
        array,
        shape,
        axes=resolved_axes,
        phase_offsets=phase_offsets,
    )


class FourierWindow(StrEnum):
    """Deterministic acquisition windows for a complex Fourier spectrum."""

    RECTANGULAR = "rectangular"
    HANN = "hann"
    HAMMING = "hamming"
    BLACKMAN = "blackman"
    TUKEY = "tukey"


class FourierSpectrumResult(StrictModule, NonTrainableState):
    """Complex Riemann-sum spectrum and its Parseval evidence."""

    frequencies: Array
    spectrum: Array
    windowed_samples: Array
    parseval_residual: Array
    successful: Array
    plan_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        frequencies: ArrayLike,
        spectrum: ArrayLike,
        windowed_samples: ArrayLike,
        parseval_residual: ArrayLike,
        successful: ArrayLike,
        plan_id: str,
        /,
    ):
        frequency = jnp.asarray(frequencies)
        transformed = jnp.asarray(spectrum)
        samples = jnp.asarray(windowed_samples)
        residual = jnp.asarray(parseval_residual, dtype=frequency.dtype).reshape(())
        if frequency.ndim != 1 or transformed.shape[-1] != frequency.size:
            raise ValueError("Fourier spectrum and frequency axis must align.")
        self.frequencies = frequency
        self.spectrum = transformed
        self.windowed_samples = samples
        self.parseval_residual = residual
        self.successful = jnp.asarray(successful, dtype=bool).reshape(())
        self.plan_id = str(plan_id)
        self.result_id = canonical_fingerprint(
            {
                "kind": "fourier-spectrum-result",
                "plan": self.plan_id,
                "successful": bool(self.successful),
                "arrays": array_tree_fingerprint(
                    {
                        "frequencies": np.asarray(frequency),
                        "spectrum": np.asarray(transformed),
                        "parseval_residual": np.asarray(residual),
                    }
                ),
            }
        )


class FourierSpectrumPlan(StrictModule, NonTrainableState):
    """Fixed-shape complex Fourier transform with an explicit exponent sign.

    ``exponent_sign=+1`` evaluates ``dt Σₙ xₙ exp(+i 2π ν tₙ)``.
    ``exponent_sign=-1`` evaluates the corresponding negative-sign transform.
    Frequencies are cycles per unit time; no implicit angular-frequency factor
    is introduced.
    """

    sample_count: int = eqx.field(static=True)
    sample_interval: float = eqx.field(static=True)
    padding_count: int = eqx.field(static=True)
    exponent_sign: int = eqx.field(static=True)
    shifted: bool = eqx.field(static=True)
    window: FourierWindow = eqx.field(static=True)
    tukey_alpha: float = eqx.field(static=True)
    parseval_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        sample_count: int,
        sample_interval: float,
        /,
        *,
        padding_count: int = 0,
        exponent_sign: int = 1,
        shifted: bool = True,
        window: FourierWindow = FourierWindow.RECTANGULAR,
        tukey_alpha: float = 0.5,
        parseval_tolerance: float = 1.0e-10,
    ):
        count = int(sample_count)
        interval = float(sample_interval)
        padding = int(padding_count)
        alpha = float(tukey_alpha)
        tolerance = float(parseval_tolerance)
        if (
            count <= 0
            or padding < 0
            or exponent_sign not in (-1, 1)
            or not isinstance(shifted, bool)
            or not isinstance(window, FourierWindow)
            or not isfinite(interval)
            or interval <= 0.0
            or not isfinite(alpha)
            or not 0.0 <= alpha <= 1.0
            or not isfinite(tolerance)
            or tolerance <= 0.0
        ):
            raise ValueError("Fourier spectrum plan parameters are invalid.")
        self.sample_count = count
        self.sample_interval = interval
        self.padding_count = padding
        self.exponent_sign = int(exponent_sign)
        self.shifted = shifted
        self.window = window
        self.tukey_alpha = alpha
        self.parseval_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "fourier-spectrum-plan",
                "sample_count": count,
                "sample_interval": interval,
                "padding_count": padding,
                "exponent_sign": exponent_sign,
                "shifted": shifted,
                "window": window.value,
                "tukey_alpha": alpha,
                "parseval_tolerance": tolerance,
            }
        )

    def evaluate(self, samples: ArrayLike, /) -> FourierSpectrumResult:
        values = jnp.asarray(samples)
        if values.ndim < 1 or values.shape[-1] != self.sample_count:
            raise ValueError("Fourier input must end in the planned sample count.")
        if self.window is FourierWindow.RECTANGULAR:
            taper = jnp.ones((self.sample_count,), dtype=values.real.dtype)
        elif self.window is FourierWindow.HANN:
            taper = hann_window(
                self.sample_count, periodic=False, dtype=values.real.dtype
            )
        elif self.window is FourierWindow.HAMMING:
            taper = hamming_window(
                self.sample_count, periodic=False, dtype=values.real.dtype
            )
        elif self.window is FourierWindow.BLACKMAN:
            taper = blackman_window(
                self.sample_count, periodic=False, dtype=values.real.dtype
            )
        else:
            taper = tukey_window(
                self.sample_count,
                self.tukey_alpha,
                periodic=False,
                dtype=values.real.dtype,
            )
        windowed = values * taper
        transform_count = self.sample_count + self.padding_count
        if self.exponent_sign == 1:
            spectrum = (
                jnp.fft.ifft(windowed, n=transform_count, axis=-1)
                * transform_count
                * self.sample_interval
            )
        else:
            spectrum = (
                jnp.fft.fft(windowed, n=transform_count, axis=-1) * self.sample_interval
            )
        frequencies = jnp.fft.fftfreq(transform_count, d=self.sample_interval)
        if self.shifted:
            spectrum = jnp.fft.fftshift(spectrum, axes=-1)
            frequencies = jnp.fft.fftshift(frequencies)
        time_energy = jnp.sum(jnp.abs(windowed) ** 2, axis=-1) * self.sample_interval
        frequency_spacing = 1.0 / (transform_count * self.sample_interval)
        frequency_energy = jnp.sum(jnp.abs(spectrum) ** 2, axis=-1) * frequency_spacing
        scale = jnp.maximum(
            time_energy,
            jnp.asarray(jnp.finfo(values.real.dtype).tiny, dtype=values.real.dtype),
        )
        residual = jnp.max(jnp.abs(frequency_energy - time_energy) / scale)
        successful = (
            jnp.all(jnp.isfinite(windowed))
            & jnp.all(jnp.isfinite(spectrum))
            & (residual <= self.parseval_tolerance)
        )
        return FourierSpectrumResult(
            frequencies,
            spectrum,
            windowed,
            residual,
            successful,
            self.plan_id,
        )


__all__ = [
    "FourierSpectrumPlan",
    "FourierSpectrumResult",
    "FourierWindow",
    "fourier_resample",
]
