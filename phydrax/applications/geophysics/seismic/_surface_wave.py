#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....series import SampledSeries


class SurfaceWaveModes(StrictModule):
    frequencies_Hz: Array
    phase_velocities_m_s: Array
    mode_valid: Array
    root_margins: Array
    derivative_available: Array


class LayeredLoveWavePlan(StrictModule, NonTrainableState):
    """SH propagator dispersion for isotropic layers over a halfspace."""

    thickness_m: Array
    density_kg_m3: Array
    shear_velocity_m_s: Array
    mode_count: int = eqx.field(static=True)
    scan_count: int = eqx.field(static=True)

    def __init__(
        self,
        thickness_m: ArrayLike,
        density_kg_m3: ArrayLike,
        shear_velocity_m_s: ArrayLike,
        /,
        *,
        mode_count: int = 4,
        scan_count: int = 512,
    ):
        thickness = np.asarray(thickness_m, dtype=float)
        density = np.asarray(density_kg_m3, dtype=float)
        velocity = np.asarray(shear_velocity_m_s, dtype=float)
        modes, scans = int(mode_count), int(scan_count)
        if (
            thickness.ndim != 1
            or density.shape != (thickness.size + 1,)
            or velocity.shape != density.shape
            or thickness.size == 0
            or np.any(~np.isfinite(thickness))
            or np.any(thickness <= 0)
            or np.any(~np.isfinite(density))
            or np.any(density <= 0)
            or np.any(~np.isfinite(velocity))
            or np.any(velocity <= 0)
            or modes <= 0
            or scans < 32
            or np.min(velocity[:-1]) >= velocity[-1]
        ):
            raise ValueError(
                "Love-wave layers require slower finite layers over a faster halfspace."
            )
        self.thickness_m = jnp.asarray(thickness)
        self.density_kg_m3 = jnp.asarray(density)
        self.shear_velocity_m_s = jnp.asarray(velocity)
        self.mode_count, self.scan_count = modes, scans

    def residual(self, frequency_Hz: float, phase_velocity_m_s: float) -> float:
        frequency, phase = float(frequency_Hz), float(phase_velocity_m_s)
        if frequency <= 0 or phase <= 0:
            raise ValueError("Love-wave frequency and phase velocity must be positive.")
        omega, wavenumber = 2 * np.pi * frequency, 2 * np.pi * frequency / phase
        state = np.asarray((1.0 + 0j, 0.0 + 0j))
        for thickness, density, velocity in zip(
            np.asarray(self.thickness_m),
            np.asarray(self.density_kg_m3[:-1]),
            np.asarray(self.shear_velocity_m_s[:-1]),
            strict=True,
        ):
            modulus = density * velocity**2
            vertical = np.sqrt((omega / velocity) ** 2 - wavenumber**2 + 0j)
            argument = vertical * thickness
            cosine, sine = np.cos(argument), np.sin(argument)
            matrix = np.asarray(
                (
                    (cosine, sine / (modulus * vertical)),
                    (-modulus * vertical * sine, cosine),
                )
            )
            state = matrix @ state
        halfspace_modulus = float(
            self.density_kg_m3[-1] * self.shear_velocity_m_s[-1] ** 2
        )
        decay = np.sqrt(
            wavenumber**2 - (omega / float(self.shear_velocity_m_s[-1])) ** 2 + 0j
        )
        value = state[1] + halfspace_modulus * decay * state[0]
        if abs(value.imag) > 1e-7 * max(abs(value.real), 1.0):
            raise ValueError("Love-wave trial lies outside the real trapped-mode branch.")
        return float(value.real)

    def solve(self, frequencies_Hz: ArrayLike, /) -> SurfaceWaveModes:
        frequencies = np.asarray(frequencies_Hz, dtype=float)
        if (
            frequencies.ndim != 1
            or frequencies.size == 0
            or np.any(~np.isfinite(frequencies))
            or np.any(frequencies <= 0)
        ):
            raise ValueError("Love-wave frequencies must be a positive finite vector.")
        lower = float(np.min(np.asarray(self.shear_velocity_m_s[:-1]))) * (1 + 1e-8)
        upper = float(self.shear_velocity_m_s[-1]) * (1 - 1e-8)
        roots = np.full((frequencies.size, self.mode_count), np.nan)
        margins = np.zeros_like(roots)
        valid = np.zeros_like(roots, dtype=bool)
        for frequency_index, frequency in enumerate(frequencies):
            candidates = np.linspace(lower, upper, self.scan_count)
            residuals = np.asarray(
                [self.residual(frequency, value) for value in candidates]
            )
            brackets = np.flatnonzero(residuals[:-1] * residuals[1:] < 0)
            for mode, bracket in enumerate(brackets[: self.mode_count]):
                left, right = candidates[bracket], candidates[bracket + 1]
                left_value = residuals[bracket]
                for _ in range(64):
                    middle = 0.5 * (left + right)
                    value = self.residual(frequency, middle)
                    if left_value * value <= 0:
                        right = middle
                    else:
                        left, left_value = middle, value
                roots[frequency_index, mode] = 0.5 * (left + right)
                margins[frequency_index, mode] = (
                    candidates[bracket + 1] - candidates[bracket]
                )
                valid[frequency_index, mode] = True
        # Sort by phase velocity and keep explicit invalid entries; no silent mode relabeling.
        derivative = np.all(valid, axis=0) & np.all(margins > 0, axis=0)
        return SurfaceWaveModes(
            jnp.asarray(frequencies),
            jnp.asarray(roots),
            jnp.asarray(valid),
            jnp.asarray(margins),
            jnp.asarray(derivative),
        )


class HomogeneousRayleighWavePlan(StrictModule, NonTrainableState):
    p_velocity_m_s: float = eqx.field(static=True)
    s_velocity_m_s: float = eqx.field(static=True)
    phase_velocity_m_s: float = eqx.field(static=True)

    def __init__(self, p_velocity_m_s: float, s_velocity_m_s: float, /):
        p, s = float(p_velocity_m_s), float(s_velocity_m_s)
        if not np.isfinite(p) or not np.isfinite(s) or p <= s or s <= 0:
            raise ValueError("Rayleigh halfspace requires finite vp > vs > 0.")
        ratio = (p / s) ** 2

        def equation(x):
            return (2 - x) ** 2 - 4 * np.sqrt(1 - x) * np.sqrt(1 - x / ratio)

        left, right = 1e-8, 1 - 1e-8
        # Ignore the trivial x=0 root and isolate the physical positive branch.
        grid = np.linspace(left, right, 2048)
        values = np.asarray([equation(value) for value in grid])
        changes = np.flatnonzero(values[:-1] * values[1:] < 0)
        if not changes.size:
            raise ValueError("Rayleigh physical root could not be bracketed.")
        left, right = grid[changes[-1]], grid[changes[-1] + 1]
        left_value = equation(left)
        for _ in range(64):
            middle = 0.5 * (left + right)
            value = equation(middle)
            if left_value * value <= 0:
                right = middle
            else:
                left, left_value = middle, value
        self.p_velocity_m_s, self.s_velocity_m_s = p, s
        self.phase_velocity_m_s = float(s * np.sqrt(0.5 * (left + right)))


class AmbientNoiseCorrelationResult(StrictModule):
    lags_s: Array
    correlation: Array
    window_count: Array
    standard_error: Array
    successful: Array


class AmbientNoiseCorrelationPlan(StrictModule, NonTrainableState):
    window_samples: int = eqx.field(static=True)
    step_samples: int = eqx.field(static=True)
    sample_interval_s: float = eqx.field(static=True)
    normalization: Literal["none", "one-bit", "spectral-whitening"] = eqx.field(
        static=True
    )

    def __init__(
        self,
        window_samples: int,
        step_samples: int,
        sample_interval_s: float,
        /,
        *,
        normalization: Literal["none", "one-bit", "spectral-whitening"] = "none",
    ):
        window, step, interval = (
            int(window_samples),
            int(step_samples),
            float(sample_interval_s),
        )
        if (
            window < 4
            or step <= 0
            or step > window
            or not np.isfinite(interval)
            or interval <= 0
        ):
            raise ValueError("Ambient-noise window/step/sample interval are invalid.")
        if normalization not in ("none", "one-bit", "spectral-whitening"):
            raise ValueError("Unknown ambient-noise normalization.")
        self.window_samples, self.step_samples = window, step
        self.sample_interval_s, self.normalization = interval, normalization

    def evaluate(
        self, first: SampledSeries, second: SampledSeries, /
    ) -> AmbientNoiseCorrelationResult:
        if not isinstance(first, SampledSeries) or not isinstance(second, SampledSeries):
            raise TypeError("Ambient noise requires SampledSeries traces.")
        if first.values.ndim != 1 or second.values.shape != first.values.shape:
            raise ValueError("Ambient-noise inputs must be matching scalar traces.")
        count = first.values.size
        starts = range(0, count - self.window_samples + 1, self.step_samples)
        taper = jnp.hanning(self.window_samples)
        rows = []
        for start in starts:
            valid = jnp.all(
                first.sample_valid[start : start + self.window_samples]
            ) & jnp.all(second.sample_valid[start : start + self.window_samples])
            left = first.values[start : start + self.window_samples]
            right = second.values[start : start + self.window_samples]
            left = left - jnp.mean(left)
            right = right - jnp.mean(right)
            if self.normalization == "one-bit":
                left, right = jnp.sign(left), jnp.sign(right)
            left_spectrum = jnp.fft.rfft(taper * left)
            right_spectrum = jnp.fft.rfft(taper * right)
            cross = left_spectrum * jnp.conj(right_spectrum)
            if self.normalization == "spectral-whitening":
                cross = cross / jnp.maximum(jnp.abs(cross), jnp.finfo(left.dtype).tiny)
            correlation = jnp.fft.irfft(cross, n=self.window_samples)
            correlation = jnp.roll(correlation, self.window_samples // 2)
            rows.append(
                jnp.where(valid, correlation, jnp.full_like(correlation, jnp.nan))
            )
        if not rows:
            raise ValueError("Ambient-noise traces are shorter than one window.")
        windows = jnp.stack(rows)
        valid_windows = jnp.all(jnp.isfinite(windows), axis=1)
        number = jnp.sum(valid_windows)
        safe = jnp.where(valid_windows[:, None], windows, 0.0)
        mean = jnp.sum(safe, axis=0) / jnp.maximum(number, 1)
        variance = jnp.sum(
            jnp.where(valid_windows[:, None], (windows - mean) ** 2, 0.0), axis=0
        ) / jnp.maximum(number - 1, 1)
        error = jnp.sqrt(variance / jnp.maximum(number, 1))
        lags = (
            jnp.arange(self.window_samples) - self.window_samples // 2
        ) * self.sample_interval_s
        successful = number >= 2
        return AmbientNoiseCorrelationResult(lags, mean, number, error, successful)


class HVSRPlan(StrictModule, NonTrainableState):
    sample_interval_s: float = eqx.field(static=True)

    def __init__(self, sample_interval_s: float, /):
        interval = float(sample_interval_s)
        if not np.isfinite(interval) or interval <= 0:
            raise ValueError("HVSR sample interval must be positive and finite.")
        self.sample_interval_s = interval

    def evaluate(
        self, horizontal_x: ArrayLike, horizontal_y: ArrayLike, vertical: ArrayLike, /
    ) -> tuple[Array, Array]:
        x, y, z = (
            jnp.asarray(horizontal_x),
            jnp.asarray(horizontal_y),
            jnp.asarray(vertical),
        )
        if x.ndim != 1 or y.shape != x.shape or z.shape != x.shape or x.size < 4:
            raise ValueError("HVSR components must be matching scalar traces.")
        spectra = [
            jnp.abs(jnp.fft.rfft(value - jnp.mean(value))) ** 2 for value in (x, y, z)
        ]
        horizontal = jnp.sqrt(0.5 * (spectra[0] + spectra[1]))
        ratio = horizontal / jnp.maximum(jnp.sqrt(spectra[2]), jnp.finfo(x.dtype).tiny)
        frequency = jnp.fft.rfftfreq(x.size, self.sample_interval_s)
        return frequency, ratio


__all__ = [
    "AmbientNoiseCorrelationPlan",
    "AmbientNoiseCorrelationResult",
    "HVSRPlan",
    "HomogeneousRayleighWavePlan",
    "LayeredLoveWavePlan",
    "SurfaceWaveModes",
]
