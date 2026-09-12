#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....series import SampledSeries
from ....signal import hann_window, tukey_window, WelchSpectrumPlan
from .._photometry import ObservationDataProvenance


WindowKind = Literal["none", "hann", "tukey"]


def _identifier(value: str, name: str, /) -> str:
    identifier = str(value).strip()
    if not identifier:
        raise ValueError(f"{name} must be non-empty.")
    return identifier


def _canonical_frequency(sample_count: int, sample_interval: float) -> np.ndarray:
    return np.fft.rfftfreq(int(sample_count), d=float(sample_interval))


def _time_values(value: ArrayLike | SampledSeries, /) -> Array:
    if isinstance(value, SampledSeries):
        if value.support.num_series != 1 or value.alignment != "node":
            raise ValueError(
                "Gravitational-wave strain requires one node-aligned series."
            )
        if not bool(value.support.connected_prefix_valid()):
            raise ValueError(
                "Gravitational-wave strain must be one connected valid series."
            )
        leaves = jax.tree_util.tree_leaves(value.values_for(0))
        if len(leaves) != 1:
            raise ValueError(
                "Gravitational-wave strain series must contain one array leaf."
            )
        return jnp.asarray(leaves[0])
    return jnp.asarray(value)


class OneSidedPowerSpectralDensity(StrictModule, NonTrainableState):
    """Canonical one-sided PSD on an exact real-FFT frequency grid."""

    frequency: Array
    values: Array
    active: Array
    provenance: ObservationDataProvenance
    sample_count: int = eqx.field(static=True)
    sample_interval: float = eqx.field(static=True)
    duration: float = eqx.field(static=True)
    psd_id: str = eqx.field(static=True)

    def __init__(
        self,
        frequency: ArrayLike,
        values: ArrayLike,
        provenance: ObservationDataProvenance,
        /,
        *,
        sample_count: int,
        sample_interval: float,
        active: ArrayLike | None = None,
        psd_id: str = "one-sided-psd",
    ):
        count = int(sample_count)
        interval = float(sample_interval)
        if count < 4 or not np.isfinite(interval) or interval <= 0.0:
            raise ValueError("PSD sample count and interval must be finite and positive.")
        frequency_host = np.asarray(frequency, dtype=float)
        values_host = np.asarray(values, dtype=float)
        expected = _canonical_frequency(count, interval)
        if (
            frequency_host.shape != expected.shape
            or values_host.shape != expected.shape
            or np.any(~np.isfinite(frequency_host))
            or np.any(np.diff(frequency_host) <= 0.0)
            or not np.allclose(
                frequency_host,
                expected,
                rtol=0.0,
                atol=16.0 * np.finfo(float).eps / interval,
            )
        ):
            raise ValueError("PSD must use the exact canonical one-sided FFT grid.")
        default_active = np.ones(expected.shape, dtype=bool)
        default_active[0] = False
        if count % 2 == 0:
            default_active[-1] = False
        active_host = default_active if active is None else np.asarray(active, dtype=bool)
        if active_host.shape != expected.shape or not np.any(active_host):
            raise ValueError("PSD active mask must select at least one frequency bin.")
        if active_host[0] or (count % 2 == 0 and active_host[-1]):
            raise ValueError("Active DC and Nyquist bins are not supported.")
        if np.any(~np.isfinite(values_host[active_host])) or np.any(
            values_host[active_host] <= 0.0
        ):
            raise ValueError("Active PSD values must be finite and strictly positive.")
        if not isinstance(provenance, ObservationDataProvenance):
            raise TypeError("provenance must be ObservationDataProvenance.")
        identifier = _identifier(psd_id, "psd_id")
        safe_values = np.where(active_host, values_host, 1.0)
        self.frequency = jnp.asarray(frequency_host)
        self.values = jnp.asarray(safe_values)
        self.active = jnp.asarray(active_host)
        self.provenance = provenance
        self.sample_count = count
        self.sample_interval = interval
        self.duration = count * interval
        self.psd_id = canonical_fingerprint(
            {
                "kind": "gravitational-wave-one-sided-psd",
                "label": identifier,
                "sample_count": count,
                "sample_interval": interval,
                "content": array_tree_fingerprint(
                    {
                        "frequency": frequency_host,
                        "values": safe_values,
                        "active": active_host,
                    }
                )["sha256"],
                "provenance": provenance.provenance_id,
            }
        )


class DetectorStrainData(StrictModule, NonTrainableState):
    """Prepared detector strain in seconds under the canonical FFT convention."""

    frequency: Array
    strain: Array
    active: Array
    psd: OneSidedPowerSpectralDensity
    provenance: ObservationDataProvenance
    window_power: Array
    detector_id: str = eqx.field(static=True)
    start_time_gps: float = eqx.field(static=True)
    data_id: str = eqx.field(static=True)

    def __init__(
        self,
        detector_id: str,
        strain: ArrayLike,
        psd: OneSidedPowerSpectralDensity,
        provenance: ObservationDataProvenance,
        /,
        *,
        start_time_gps: float,
        active: ArrayLike | None = None,
        window_power: float = 1.0,
    ):
        if not isinstance(psd, OneSidedPowerSpectralDensity):
            raise TypeError("psd must be OneSidedPowerSpectralDensity.")
        if not isinstance(provenance, ObservationDataProvenance):
            raise TypeError("provenance must be ObservationDataProvenance.")
        identifier = _identifier(detector_id, "detector_id")
        start = float(start_time_gps)
        power = float(window_power)
        values = np.asarray(strain)
        active_host = np.asarray(psd.active if active is None else active, dtype=bool)
        if (
            values.shape != tuple(psd.frequency.shape)
            or not np.issubdtype(values.dtype, np.complexfloating)
            or active_host.shape != values.shape
            or not np.any(active_host)
            or np.any(active_host & ~np.asarray(psd.active))
            or np.any(~np.isfinite(values[active_host]))
            or not np.isfinite(start)
            or not np.isfinite(power)
            or power <= 0.0
        ):
            raise ValueError(
                "Detector strain arrays, support, time, or window power are invalid."
            )
        safe = np.where(active_host, values, 0.0j)
        self.frequency = psd.frequency
        self.strain = jnp.asarray(safe)
        self.active = jnp.asarray(active_host)
        self.psd = psd
        self.provenance = provenance
        self.window_power = jnp.asarray(power, dtype=psd.values.dtype)
        self.detector_id = identifier
        self.start_time_gps = start
        self.data_id = canonical_fingerprint(
            {
                "kind": "gravitational-wave-detector-strain",
                "detector": identifier,
                "psd": psd.psd_id,
                "start_time_gps": start,
                "window_power": power,
                "content": array_tree_fingerprint(
                    {"strain": safe, "active": active_host}
                )["sha256"],
                "provenance": provenance.provenance_id,
            }
        )


class DetectorNetworkData(StrictModule, NonTrainableState):
    """Packed independent-detector data sharing one canonical frequency grid."""

    frequency: Array
    strain: Array
    psd: Array
    active: Array
    window_power: Array
    variance: Array
    inverse_variance: Array
    data_norm_by_detector: Array
    noise_log_probability_by_detector: Array
    detector_ids: tuple[str, ...] = eqx.field(static=True)
    data_ids: tuple[str, ...] = eqx.field(static=True)
    sample_count: int = eqx.field(static=True)
    sample_interval: float = eqx.field(static=True)
    duration: float = eqx.field(static=True)
    start_time_gps: float = eqx.field(static=True)
    network_id: str = eqx.field(static=True)

    def __init__(self, detectors: Sequence[DetectorStrainData], /):
        items = tuple(detectors)
        if not items or any(not isinstance(item, DetectorStrainData) for item in items):
            raise TypeError("detectors must contain DetectorStrainData values.")
        identifiers = tuple(item.detector_id for item in items)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Detector IDs must be unique.")
        reference = items[0]
        for item in items[1:]:
            if (
                item.psd.sample_count != reference.psd.sample_count
                or item.psd.sample_interval != reference.psd.sample_interval
                or item.start_time_gps != reference.start_time_gps
                or not bool(jnp.array_equal(item.frequency, reference.frequency))
            ):
                raise ValueError(
                    "Network detectors must share time and frequency support."
                )
        strain = jnp.stack(tuple(item.strain for item in items))
        psd = jnp.stack(tuple(item.psd.values for item in items))
        active = jnp.stack(tuple(item.active for item in items))
        power = jnp.stack(tuple(item.window_power for item in items))
        variance = 0.5 * reference.psd.duration * psd * power[:, None]
        inverse_variance = jnp.where(active, 1.0 / variance, 0.0)
        squared_data = jnp.abs(strain) ** 2
        data_norm = 2.0 * jnp.sum(squared_data * inverse_variance, axis=-1)
        noise_terms = jnp.where(
            active,
            -squared_data * inverse_variance - jnp.log(jnp.pi * variance),
            0.0,
        )
        self.frequency = reference.frequency
        self.strain = strain
        self.psd = psd
        self.active = active
        self.window_power = power
        self.variance = variance
        self.inverse_variance = inverse_variance
        self.data_norm_by_detector = data_norm
        self.noise_log_probability_by_detector = jnp.sum(noise_terms, axis=-1)
        self.detector_ids = identifiers
        self.data_ids = tuple(item.data_id for item in items)
        self.sample_count = reference.psd.sample_count
        self.sample_interval = reference.psd.sample_interval
        self.duration = reference.psd.duration
        self.start_time_gps = reference.start_time_gps
        self.network_id = canonical_fingerprint(
            {
                "kind": "gravitational-wave-detector-network-data",
                "detectors": list(identifiers),
                "data": list(self.data_ids),
                "duration": self.duration,
            }
        )

    @property
    def noise_log_probability(self) -> Array:
        return jnp.sum(self.noise_log_probability_by_detector)


class GravitationalWaveDataPlan(StrictModule, NonTrainableState):
    """Time-to-frequency preparation with explicit window, band, and notches."""

    sample_count: int = eqx.field(static=True)
    sample_interval: float = eqx.field(static=True)
    start_time_gps: float = eqx.field(static=True)
    minimum_frequency: float = eqx.field(static=True)
    maximum_frequency: float = eqx.field(static=True)
    notches: tuple[tuple[float, float], ...] = eqx.field(static=True)
    window: WindowKind = eqx.field(static=True)
    tukey_alpha: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        sample_count: int,
        sample_interval: float,
        /,
        *,
        start_time_gps: float,
        minimum_frequency: float,
        maximum_frequency: float,
        notches: Sequence[tuple[float, float]] = (),
        window: WindowKind = "tukey",
        tukey_alpha: float = 0.2,
    ):
        count = int(sample_count)
        interval = float(sample_interval)
        start = float(start_time_gps)
        lower, upper = float(minimum_frequency), float(maximum_frequency)
        alpha = float(tukey_alpha)
        notch_values = tuple((float(left), float(right)) for left, right in notches)
        nyquist = 0.5 / interval if interval > 0.0 else 0.0
        if (
            count < 4
            or not all(
                np.isfinite(value) for value in (interval, start, lower, upper, alpha)
            )
            or interval <= 0.0
            or not 0.0 < lower < upper < nyquist
            or window not in ("none", "hann", "tukey")
            or not 0.0 <= alpha <= 1.0
            or any(
                not np.isfinite(left) or not np.isfinite(right) or left >= right
                for left, right in notch_values
            )
        ):
            raise ValueError("Gravitational-wave data preparation settings are invalid.")
        self.sample_count = count
        self.sample_interval = interval
        self.start_time_gps = start
        self.minimum_frequency = lower
        self.maximum_frequency = upper
        self.notches = notch_values
        self.window = window
        self.tukey_alpha = alpha
        self.plan_id = canonical_fingerprint(
            {
                "kind": "gravitational-wave-data-plan",
                "sample_count": count,
                "sample_interval": interval,
                "start_time_gps": start,
                "band": [lower, upper],
                "notches": [list(value) for value in notch_values],
                "window": window,
                "tukey_alpha": alpha if window == "tukey" else None,
            }
        )

    @property
    def duration(self) -> float:
        return self.sample_count * self.sample_interval

    @property
    def frequencies(self) -> Array:
        return jnp.fft.rfftfreq(self.sample_count, self.sample_interval)

    def active_mask(self, psd: OneSidedPowerSpectralDensity, /) -> Array:
        if (
            psd.sample_count != self.sample_count
            or psd.sample_interval != self.sample_interval
        ):
            raise ValueError("PSD support does not match the data plan.")
        active = (
            psd.active
            & (psd.frequency >= self.minimum_frequency)
            & (psd.frequency <= self.maximum_frequency)
        )
        for left, right in self.notches:
            active = active & ~((psd.frequency >= left) & (psd.frequency <= right))
        if not bool(jnp.any(active)):
            raise ValueError("The analysis band and notches leave no active frequencies.")
        return active

    def _window_values(self, dtype) -> Array:
        if self.window == "none":
            return jnp.ones((self.sample_count,), dtype=dtype)
        if self.window == "hann":
            return hann_window(self.sample_count, periodic=False, dtype=dtype)
        return tukey_window(
            self.sample_count,
            self.tukey_alpha,
            periodic=False,
            dtype=dtype,
        )

    def prepare(
        self,
        detector_id: str,
        time_strain: ArrayLike | SampledSeries,
        psd: OneSidedPowerSpectralDensity,
        provenance: ObservationDataProvenance,
        /,
    ) -> DetectorStrainData:
        values = _time_values(time_strain)
        if (
            values.shape != (self.sample_count,)
            or not jnp.issubdtype(values.dtype, jnp.floating)
            or bool(jnp.any(~jnp.isfinite(values)))
        ):
            raise ValueError("Time-domain strain must be one finite real vector.")
        window = self._window_values(values.dtype)
        sample_frequency = 1.0 / self.sample_interval
        spectrum = jnp.fft.rfft(values * window) / sample_frequency
        return DetectorStrainData(
            detector_id,
            spectrum,
            psd,
            provenance,
            start_time_gps=self.start_time_gps,
            active=self.active_mask(psd),
            window_power=float(jnp.mean(window * window)),
        )

    def estimate_psd(
        self,
        time_strain: ArrayLike,
        provenance: ObservationDataProvenance,
        /,
        *,
        average: Literal["mean", "median"] = "median",
        overlap: int | None = None,
        psd_id: str = "estimated-psd",
    ) -> OneSidedPowerSpectralDensity:
        estimator = WelchSpectrumPlan(
            self.sample_interval,
            self.sample_count,
            overlap=overlap,
            window="tukey" if self.window == "tukey" else "hann",
            average=average,
            tukey_alpha=self.tukey_alpha,
        )
        result = estimator.evaluate(time_strain)
        return OneSidedPowerSpectralDensity(
            result.frequencies,
            result.power_spectral_density,
            provenance,
            sample_count=self.sample_count,
            sample_interval=self.sample_interval,
            psd_id=f"{psd_id}:{estimator.plan_id}",
        )


__all__ = [
    "DetectorNetworkData",
    "DetectorStrainData",
    "GravitationalWaveDataPlan",
    "OneSidedPowerSpectralDensity",
    "WindowKind",
]
