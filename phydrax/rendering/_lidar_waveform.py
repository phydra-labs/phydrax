#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Time-resolved hard-surface, atmospheric, and multipath LiDAR returns."""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, ArrayLike, PRNGKeyArray

from phydrax._interpolation import linear_interpolate

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..measurement import PulseResponse, WaveformSupport
from ._lidar import PreparedLidarSurface


class LidarWaveformEvidence(StrictModule, NonTrainableState):
    emitted_energy: Array
    received_energy: Array
    dropped_energy: Array
    finite: Array
    capacity_sufficient: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class LidarWaveformResult(StrictModule):
    values: Array
    evidence: LidarWaveformEvidence


def _shifted_pulse(
    times: Array, pulse_times: Array, pulse_values: Array, delays: Array
) -> Array:
    return jax.vmap(
        lambda delay: (
            linear_interpolate(
                pulse_times,
                pulse_values,
                times - delay,
                bounds="fill",
                fill_value=0.0,
            ).values
        )
    )(delays)


class HardSurfaceLidarWaveformPlan(StrictModule, NonTrainableState):
    surface: PreparedLidarSurface
    support: WaveformSupport = eqx.field(static=True)
    pulse_times: Array
    pulse_values: Array
    receiver_values: Array
    receiver_gains: Array
    wave_speed: float = eqx.field(static=True)
    backscatter: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        surface: PreparedLidarSurface,
        support: WaveformSupport,
        pulse: PulseResponse,
        receiver: PulseResponse,
        /,
        *,
        wave_speed: float,
        backscatter: float = 1.0,
        receiver_gains: ArrayLike | None = None,
    ):
        if not isinstance(surface, PreparedLidarSurface) or not isinstance(
            support, WaveformSupport
        ):
            raise TypeError(
                "surface and support must be prepared LiDAR and waveform contracts."
            )
        if surface.support_id != support.rays.support_id:
            raise ValueError("Waveform rays must match the prepared LiDAR surface rays.")
        speed, scatter = float(wave_speed), float(backscatter)
        if (
            not np.isfinite(speed)
            or speed <= 0.0
            or not np.isfinite(scatter)
            or scatter < 0.0
        ):
            raise ValueError("wave_speed must be positive and backscatter nonnegative.")
        gains = (
            np.ones((len(support.receiver_ids),))
            if receiver_gains is None
            else np.asarray(receiver_gains, dtype=float)
        )
        if (
            gains.shape != (len(support.receiver_ids),)
            or not np.all(np.isfinite(gains))
            or np.any(gains < 0.0)
        ):
            raise ValueError(
                "receiver_gains must be finite and nonnegative for every receiver."
            )
        self.surface = surface
        self.support = support
        self.pulse_times = jnp.asarray(pulse.times)
        self.pulse_values = jnp.asarray(pulse.amplitudes)
        self.receiver_values = jnp.asarray(receiver.amplitudes)
        self.receiver_gains = jnp.asarray(gains)
        self.wave_speed = speed
        self.backscatter = scatter
        self.plan_id = canonical_fingerprint(
            {
                "kind": "hard-surface-lidar-waveform",
                "surface": surface.prepared_id,
                "support": support.support_id,
                "pulse": pulse.response_id,
                "receiver": receiver.response_id,
                "speed": speed,
                "backscatter": scatter,
                "gains": gains.tolist(),
            }
        )

    def evaluate(
        self, vertices: ArrayLike, /, *, geometry_id: str
    ) -> LidarWaveformResult:
        hit = self.surface.predict(vertices, geometry_id=geometry_id)
        ranges = hit.prediction.values
        valid = hit.prediction.valid_mask
        delays = 2.0 * ranges / self.wave_speed
        pulse = _shifted_pulse(
            jnp.asarray(self.support.delay_axis.sample_times),
            self.pulse_times,
            self.pulse_values,
            delays,
        )
        incidence = hit.incidence_cosine
        amplitude = jnp.where(
            valid,
            self.backscatter
            * incidence
            / jnp.maximum(ranges * ranges, jnp.finfo(ranges.dtype).tiny),
            0.0,
        )
        waveform = pulse * amplitude[:, None]
        receiver = self.receiver_values / jnp.sum(self.receiver_values)
        waveform = jax.vmap(lambda value: jnp.convolve(value, receiver, mode="same"))(
            waveform
        )
        waveform = waveform[..., None] * self.receiver_gains
        widths = jnp.asarray(self.support.bin_widths)
        received = jnp.sum(waveform * widths[None, :, None])
        emitted = jnp.sum(
            jnp.asarray(self.support.rays.active_mask, dtype=waveform.dtype)
        )
        finite = jnp.all(jnp.isfinite(waveform))
        successful = finite & hit.evidence.successful
        evidence = LidarWaveformEvidence(
            emitted,
            received,
            jnp.maximum(emitted - received, 0.0),
            finite,
            hit.evidence.capacity_sufficient,
            successful,
            self.plan_id,
        )
        return LidarWaveformResult(waveform, evidence)


class LidarReturnExtractionResult(StrictModule, NonTrainableState):
    delays: Array
    amplitudes: Array
    valid: Array
    peak_count: Array
    capacity_sufficient: Array
    successful: Array


@dataclass(frozen=True, slots=True)
class LidarReturnExtractionPlan:
    pulse: PulseResponse
    threshold: float
    return_capacity: int
    minimum_bin_separation: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.pulse, PulseResponse):
            raise TypeError("pulse must be PulseResponse.")
        if (
            not np.isfinite(self.threshold)
            or self.threshold < 0.0
            or self.return_capacity < 1
            or self.minimum_bin_separation < 1
        ):
            raise ValueError(
                "threshold must be nonnegative and return capacities positive."
            )

    def evaluate(
        self, waveform: LidarWaveformResult, support: WaveformSupport, /
    ) -> LidarReturnExtractionResult:
        values = jnp.max(waveform.values, axis=-1)
        template = jnp.asarray(self.pulse.amplitudes)
        matched = jax.vmap(
            lambda value: jnp.convolve(value, template[::-1], mode="same")
        )(values)
        left = jnp.pad(matched[:, :-1], ((0, 0), (1, 0)), constant_values=-jnp.inf)
        right = jnp.pad(matched[:, 1:], ((0, 0), (0, 1)), constant_values=-jnp.inf)
        peaks = (matched >= left) & (matched >= right) & (matched >= self.threshold)
        candidates = jnp.where(peaks, matched, -jnp.inf)
        bin_indices = jnp.arange(matched.shape[1])

        def select_one(scores):
            initial = (
                scores,
                jnp.full((self.return_capacity,), -1, dtype=jnp.int32),
                jnp.zeros((self.return_capacity,), dtype=scores.dtype),
                jnp.zeros((self.return_capacity,), dtype=bool),
            )

            def choose(index, state):
                remaining, selected, amplitudes, valid = state
                peak = jnp.argmax(remaining).astype(jnp.int32)
                amplitude = remaining[peak]
                accepted = jnp.isfinite(amplitude)
                selected = selected.at[index].set(jnp.where(accepted, peak, -1))
                amplitudes = amplitudes.at[index].set(jnp.where(accepted, amplitude, 0.0))
                valid = valid.at[index].set(accepted)
                suppress = jnp.abs(bin_indices - peak) < self.minimum_bin_separation
                remaining = jnp.where(accepted & suppress, -jnp.inf, remaining)
                return remaining, selected, amplitudes, valid

            remaining, selected, amplitudes, valid = jax.lax.fori_loop(
                0, self.return_capacity, choose, initial
            )
            return selected, amplitudes, valid, jnp.max(remaining)

        selected, amplitudes, valid, remaining = jax.vmap(select_one)(candidates)
        safe_selected = jnp.maximum(selected, 0)
        delays = jnp.asarray(support.delay_axis.sample_times)[safe_selected]
        count = jnp.sum(valid, axis=1) + jnp.isfinite(remaining).astype(jnp.int32)
        capacity = jnp.all(~jnp.isfinite(remaining))
        successful = (
            waveform.evidence.successful
            & capacity
            & jnp.all(jnp.isfinite(jnp.where(valid, delays, 0.0)))
        )
        return LidarReturnExtractionResult(
            delays, jnp.where(valid, amplitudes, 0.0), valid, count, capacity, successful
        )


class AtmosphericLidarPlan(StrictModule, NonTrainableState):
    support: WaveformSupport = eqx.field(static=True)
    pulse_times: Array
    pulse_values: Array
    range_samples: Array
    segment_lengths: Array
    wave_speed: float = eqx.field(static=True)
    overlap: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        support: WaveformSupport,
        pulse: PulseResponse,
        range_samples: ArrayLike,
        segment_lengths: ArrayLike,
        /,
        *,
        wave_speed: float,
        overlap: ArrayLike = 1.0,
    ):
        ranges = np.asarray(range_samples, dtype=float)
        lengths = np.asarray(segment_lengths, dtype=float)
        if (
            ranges.ndim != 2
            or ranges.shape != lengths.shape
            or ranges.shape[0] != support.rays.sample_shape[0]
            or np.any(ranges < 0.0)
            or np.any(lengths < 0.0)
        ):
            raise ValueError(
                "range_samples and segment_lengths must share shape (ray_count, segment_count)."
            )
        overlap_ = np.broadcast_to(np.asarray(overlap, dtype=float), ranges.shape).copy()
        if np.any(overlap_ < 0.0) or not np.all(np.isfinite(overlap_)):
            raise ValueError("overlap must be finite and nonnegative.")
        self.support = support
        self.pulse_times = jnp.asarray(pulse.times)
        self.pulse_values = jnp.asarray(pulse.amplitudes)
        self.range_samples = jnp.asarray(ranges)
        self.segment_lengths = jnp.asarray(lengths)
        self.wave_speed = float(wave_speed)
        self.overlap = jnp.asarray(overlap_)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "atmospheric-lidar",
                "support": support.support_id,
                "pulse": pulse.response_id,
                "ranges": ranges.tolist(),
                "lengths": lengths.tolist(),
                "speed": self.wave_speed,
            }
        )

    def evaluate(
        self, extinction: ArrayLike, backscatter: ArrayLike, /
    ) -> LidarWaveformResult:
        alpha = jnp.asarray(extinction)
        beta = jnp.asarray(backscatter)
        if alpha.shape != self.range_samples.shape or beta.shape != alpha.shape:
            raise ValueError(
                "extinction and backscatter must match the atmospheric route shape."
            )
        optical_depth = jnp.cumsum(alpha * self.segment_lengths, axis=1)
        transmittance = jnp.exp(-2.0 * optical_depth)
        source = (
            beta
            * transmittance
            * self.overlap
            * self.segment_lengths
            / jnp.maximum(self.range_samples**2, jnp.finfo(alpha.dtype).tiny)
        )
        delays = 2.0 * self.range_samples / self.wave_speed
        times = jnp.asarray(self.support.delay_axis.sample_times)
        shifted = jax.vmap(
            jax.vmap(
                lambda delay: (
                    linear_interpolate(
                        self.pulse_times,
                        self.pulse_values,
                        times - delay,
                        bounds="fill",
                        fill_value=0.0,
                    ).values
                )
            )
        )(delays)
        waveform = jnp.sum(source[..., None] * shifted, axis=1)[..., None]
        waveform = jnp.broadcast_to(
            waveform, waveform.shape[:-1] + (len(self.support.receiver_ids),)
        )
        finite = (
            jnp.all(jnp.isfinite(waveform)) & jnp.all(alpha >= 0.0) & jnp.all(beta >= 0.0)
        )
        received = jnp.sum(waveform * jnp.asarray(self.support.bin_widths)[None, :, None])
        evidence = LidarWaveformEvidence(
            jnp.sum(beta * self.segment_lengths),
            received,
            0.0,
            finite,
            jnp.asarray(True),
            finite,
            self.plan_id,
        )
        return LidarWaveformResult(waveform, evidence)


class SpecularLidarMultipathPlan(StrictModule, NonTrainableState):
    support: WaveformSupport = eqx.field(static=True)
    pulse_times: Array
    pulse_values: Array
    wave_speed: float = eqx.field(static=True)
    path_capacity: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        support: WaveformSupport,
        pulse: PulseResponse,
        /,
        *,
        wave_speed: float,
        path_capacity: int,
    ):
        self.support = support
        self.pulse_times = jnp.asarray(pulse.times)
        self.pulse_values = jnp.asarray(pulse.amplitudes)
        self.wave_speed = float(wave_speed)
        self.path_capacity = int(path_capacity)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "specular-lidar-multipath",
                "support": support.support_id,
                "pulse": pulse.response_id,
                "speed": self.wave_speed,
                "capacity": self.path_capacity,
            }
        )

    def evaluate(
        self, path_lengths: ArrayLike, path_powers: ArrayLike, path_valid: ArrayLike, /
    ) -> LidarWaveformResult:
        lengths, powers, valid = (
            jnp.asarray(path_lengths),
            jnp.asarray(path_powers),
            jnp.asarray(path_valid, dtype=bool),
        )
        expected = (self.support.rays.sample_shape[0], self.path_capacity)
        if (
            lengths.shape != expected
            or powers.shape != expected
            or valid.shape != expected
        ):
            raise ValueError(f"Multipath arrays must have shape {expected}.")
        delays = lengths / self.wave_speed
        times = jnp.asarray(self.support.delay_axis.sample_times)
        shifted = jax.vmap(
            jax.vmap(
                lambda delay: (
                    linear_interpolate(
                        self.pulse_times,
                        self.pulse_values,
                        times - delay,
                        bounds="fill",
                        fill_value=0.0,
                    ).values
                )
            )
        )(delays)
        waveform = jnp.sum(
            jnp.where(valid[..., None], powers[..., None] * shifted, 0.0), axis=1
        )[..., None]
        waveform = jnp.broadcast_to(
            waveform, waveform.shape[:-1] + (len(self.support.receiver_ids),)
        )
        finite = jnp.all(jnp.isfinite(waveform)) & jnp.all(
            jnp.where(valid, powers >= 0.0, True)
        )
        received = jnp.sum(waveform * jnp.asarray(self.support.bin_widths)[None, :, None])
        emitted = jnp.sum(jnp.where(valid, powers, 0.0))
        return LidarWaveformResult(
            waveform,
            LidarWaveformEvidence(
                emitted,
                received,
                jnp.maximum(emitted - received, 0.0),
                finite,
                jnp.asarray(True),
                finite,
                self.plan_id,
            ),
        )


class TimeResolvedMultipleScatteringPlan(StrictModule, NonTrainableState):
    support: WaveformSupport = eqx.field(static=True)
    packet_count: int = eqx.field(static=True)
    event_count: int = eqx.field(static=True)
    extinction: float = eqx.field(static=True)
    scattering_albedo: float = eqx.field(static=True)
    wave_speed: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        support: WaveformSupport,
        /,
        *,
        packet_count: int,
        event_count: int,
        extinction: float,
        scattering_albedo: float,
        wave_speed: float,
    ):
        self.support = support
        self.packet_count = int(packet_count)
        self.event_count = int(event_count)
        self.extinction = float(extinction)
        self.scattering_albedo = float(scattering_albedo)
        self.wave_speed = float(wave_speed)
        if (
            self.packet_count < 1
            or self.event_count < 1
            or self.extinction <= 0.0
            or not 0.0 <= self.scattering_albedo <= 1.0
            or self.wave_speed <= 0.0
        ):
            raise ValueError(
                "Multiple-scattering parameters are outside their physical domains."
            )
        self.plan_id = canonical_fingerprint(
            {
                "kind": "time-resolved-multiple-scattering",
                "support": support.support_id,
                "packets": self.packet_count,
                "events": self.event_count,
                "extinction": self.extinction,
                "albedo": self.scattering_albedo,
                "speed": self.wave_speed,
            }
        )

    def evaluate(self, key: PRNGKeyArray, /) -> LidarWaveformResult:
        ray_count = self.support.rays.sample_shape[0]
        random = (
            jr.exponential(key, (ray_count, self.packet_count, self.event_count))
            / self.extinction
        )
        path = jnp.cumsum(random, axis=-1)
        weight = self.scattering_albedo ** (jnp.arange(self.event_count) + 1) / (
            4.0 * jnp.pi
        )
        delays = 2.0 * path / self.wave_speed
        bins = jnp.asarray(self.support.delay_axis.sample_times)
        nearest = jnp.argmin(jnp.abs(delays[..., None] - bins), axis=-1)
        waveform = jnp.zeros((ray_count, bins.size))
        ray_indices = jnp.broadcast_to(
            jnp.arange(ray_count)[:, None, None], nearest.shape
        )
        waveform = waveform.at[ray_indices.reshape((-1,)), nearest.reshape((-1,))].add(
            jnp.broadcast_to(weight, nearest.shape).reshape((-1,)) / self.packet_count
        )
        waveform = waveform[..., None]
        waveform = jnp.broadcast_to(
            waveform, waveform.shape[:-1] + (len(self.support.receiver_ids),)
        )
        finite = jnp.all(jnp.isfinite(waveform))
        received = jnp.sum(waveform * jnp.asarray(self.support.bin_widths)[None, :, None])
        return LidarWaveformResult(
            waveform,
            LidarWaveformEvidence(
                jnp.asarray(ray_count, dtype=waveform.dtype),
                received,
                0.0,
                finite,
                jnp.asarray(True),
                finite,
                self.plan_id,
            ),
        )


__all__ = [
    "AtmosphericLidarPlan",
    "HardSurfaceLidarWaveformPlan",
    "LidarReturnExtractionPlan",
    "LidarReturnExtractionResult",
    "LidarWaveformEvidence",
    "LidarWaveformResult",
    "SpecularLidarMultipathPlan",
    "TimeResolvedMultipleScatteringPlan",
]
