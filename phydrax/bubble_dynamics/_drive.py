#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Excess far-field pressure drives `p_d(t)` with `p_∞(t) = p0 + p_d(t)`."""

from __future__ import annotations

from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import canonical_fingerprint
from .._trainable import fixed_field, parameter_field
from ..series import SampledSeries, SampledSeriesReconstruction, SeriesSupport
from ..typing import parse
from ._contracts import (
    AbstractBubblePressureDrive,
    PressureDriveEvaluation,
    scalar_parameter,
)


SampledDriveInterpolation: TypeAlias = Literal["linear", "cubic_hermite"]


class ConstantPressureDrive(AbstractBubblePressureDrive):
    """Constant excess pressure applied from `t = 0` (a pressure step)."""

    pressure: Array = parameter_field()
    drive_id: str = eqx.field(static=True)

    def __init__(self, pressure: ArrayLike, /) -> None:
        self.pressure = scalar_parameter(pressure, "pressure")
        self.drive_id = canonical_fingerprint({"kind": "bubble-drive-constant"})

    def evaluate(self, time: Array, /) -> PressureDriveEvaluation:
        pressure = jnp.broadcast_to(self.pressure, jnp.shape(time))
        return PressureDriveEvaluation(
            pressure, jnp.zeros_like(pressure), jnp.ones(jnp.shape(time), dtype=jnp.bool_)
        )

    def characteristic_pressure(self) -> Array:
        return jnp.abs(self.pressure)


class HarmonicPressureDrive(AbstractBubblePressureDrive):
    """Continuous wave `p_d = A sin(ω t + φ)`."""

    amplitude: Array = parameter_field()
    angular_frequency: Array = parameter_field()
    phase: Array = parameter_field()
    drive_id: str = eqx.field(static=True)

    def __init__(
        self,
        amplitude: ArrayLike,
        angular_frequency: ArrayLike,
        /,
        *,
        phase: ArrayLike = 0.0,
    ) -> None:
        self.amplitude = scalar_parameter(amplitude, "amplitude", lower=0.0, inclusive=True)
        self.angular_frequency = scalar_parameter(angular_frequency, "angular_frequency", lower=0.0)
        self.phase = scalar_parameter(phase, "phase")
        self.drive_id = canonical_fingerprint({"kind": "bubble-drive-harmonic"})

    def evaluate(self, time: Array, /) -> PressureDriveEvaluation:
        argument = self.angular_frequency * time + self.phase
        return PressureDriveEvaluation(
            self.amplitude * jnp.sin(argument),
            self.amplitude * self.angular_frequency * jnp.cos(argument),
            jnp.ones(jnp.shape(time), dtype=jnp.bool_),
        )

    def characteristic_pressure(self) -> Array:
        return self.amplitude


class PulsedPressureDrive(AbstractBubblePressureDrive):
    """Hann-windowed tone burst.

    `p_d = A sin²(π τ/T) sin(ω τ + φ)` for `0 ≤ τ = t − t₀ ≤ T` with burst length
    `T = 2π n/ω` (`n = cycle_count`), and zero otherwise; the envelope makes the
    pressure and its rate continuous at both burst edges.
    """

    amplitude: Array = parameter_field()
    angular_frequency: Array = parameter_field()
    cycle_count: Array = parameter_field()
    start_time: Array = parameter_field()
    phase: Array = parameter_field()
    drive_id: str = eqx.field(static=True)

    def __init__(
        self,
        amplitude: ArrayLike,
        angular_frequency: ArrayLike,
        cycle_count: ArrayLike,
        /,
        *,
        start_time: ArrayLike = 0.0,
        phase: ArrayLike = 0.0,
    ) -> None:
        self.amplitude = scalar_parameter(amplitude, "amplitude", lower=0.0, inclusive=True)
        self.angular_frequency = scalar_parameter(angular_frequency, "angular_frequency", lower=0.0)
        self.cycle_count = scalar_parameter(cycle_count, "cycle_count", lower=0.0)
        self.start_time = scalar_parameter(start_time, "start_time")
        self.phase = scalar_parameter(phase, "phase")
        self.drive_id = canonical_fingerprint({"kind": "bubble-drive-pulsed-hann"})

    @property
    def duration(self) -> Array:
        """Burst length `2π n/ω`."""
        return 2.0 * jnp.pi * self.cycle_count / self.angular_frequency

    def evaluate(self, time: Array, /) -> PressureDriveEvaluation:
        duration = self.duration
        local = time - self.start_time
        inside = (local >= 0.0) & (local <= duration)
        envelope_argument = jnp.pi * local / duration
        carrier = self.angular_frequency * local + self.phase
        envelope = jnp.sin(envelope_argument) ** 2
        pressure = self.amplitude * envelope * jnp.sin(carrier)
        rate = self.amplitude * (
            jnp.pi / duration * jnp.sin(2.0 * envelope_argument) * jnp.sin(carrier)
            + envelope * self.angular_frequency * jnp.cos(carrier)
        )
        return PressureDriveEvaluation(
            jnp.where(inside, pressure, 0.0),
            jnp.where(inside, rate, 0.0),
            jnp.ones(jnp.shape(time), dtype=jnp.bool_),
        )

    def characteristic_pressure(self) -> Array:
        return self.amplitude


class SampledPressureDrive(AbstractBubblePressureDrive):
    """Measured or synthesized excess pressure reconstructed natively.

    The samples are reconstructed by `series.SampledSeriesReconstruction` with
    `bounds="fill"`: outside the sampled interval the pressure is held at the
    nearest sample and `in_support` is false, which terminates a bubble solve
    with `SUPPORT_EXIT`.
    """

    reconstruction: SampledSeriesReconstruction = fixed_field()
    peak_pressure: Array = fixed_field()
    interpolation: SampledDriveInterpolation = eqx.field(static=True)
    drive_id: str = eqx.field(static=True)

    def __init__(
        self,
        times: ArrayLike,
        pressures: ArrayLike,
        /,
        *,
        interpolation: SampledDriveInterpolation = "cubic_hermite",
    ) -> None:
        method = parse(interpolation, SampledDriveInterpolation, "interpolation")
        times_ = np.asarray(times, dtype=np.float64)
        pressures_ = np.asarray(pressures, dtype=np.float64)
        if times_.ndim != 1 or times_.shape[0] < 2 or pressures_.shape != times_.shape:
            raise ValueError("times and pressures must be matching rank-1 arrays of length >= 2.")
        if not np.all(np.isfinite(times_)) or not np.all(np.isfinite(pressures_)):
            raise ValueError("Sampled drive values must be finite.")
        if not np.all(np.diff(times_) > 0.0):
            raise ValueError("Sampled drive times must be strictly increasing.")
        support = SeriesSupport(times_, coordinate_name="time", coordinate_id="bubble-drive-time")
        series = SampledSeries(support, pressures_, series_id="bubble-drive-pressure")
        self.reconstruction = SampledSeriesReconstruction(
            series, interpolation=method, bounds="fill"
        )
        self.peak_pressure = jnp.asarray(np.max(np.abs(pressures_)), dtype=jnp.float64)
        self.interpolation = method
        self.drive_id = canonical_fingerprint(
            {
                "kind": "bubble-drive-sampled",
                "interpolation": method,
                "times": times_,
                "pressures": pressures_,
            }
        )

    def evaluate(self, time: Array, /) -> PressureDriveEvaluation:
        value = self.reconstruction.evaluate(time)
        rate = self.reconstruction.evaluate(time, derivative_order=1)
        return PressureDriveEvaluation(
            jnp.asarray(value.values),
            jnp.where(value.support, jnp.asarray(rate.values), 0.0),
            value.support,
        )

    def characteristic_pressure(self) -> Array:
        return self.peak_pressure


__all__ = [
    "ConstantPressureDrive",
    "HarmonicPressureDrive",
    "PulsedPressureDrive",
    "SampledDriveInterpolation",
    "SampledPressureDrive",
]
