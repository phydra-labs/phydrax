#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from abc import abstractmethod
from collections.abc import Callable, Mapping
from typing import Any, Literal

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, PyTree

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from .._photometry import ObservationDataProvenance
from ._status import GravitationalWaveStatus


DerivativeLevel = Literal["none", "first", "higher"]


class WaveformCapabilities(StrictModule, NonTrainableState):
    polarization_ids: tuple[str, ...] = eqx.field(static=True)
    arbitrary_frequencies: bool = eqx.field(static=True)
    jittable: bool = eqx.field(static=True)
    batched: bool = eqx.field(static=True)
    derivative_level: DerivativeLevel = eqx.field(static=True)
    phase_parameter: str | None = eqx.field(static=True)
    phase_harmonic: int | None = eqx.field(static=True)
    distance_parameter: str | None = eqx.field(static=True)
    reference_distance: float | None = eqx.field(static=True)
    parameterization_id: str = eqx.field(static=True)
    capability_id: str = eqx.field(static=True)

    def __init__(
        self,
        polarization_ids: tuple[str, ...],
        /,
        *,
        arbitrary_frequencies: bool,
        jittable: bool,
        batched: bool,
        derivative_level: DerivativeLevel,
        parameterization_id: str,
        phase_parameter: str | None = None,
        phase_harmonic: int | None = None,
        distance_parameter: str | None = None,
        reference_distance: float | None = None,
    ):
        polarizations = tuple(str(value).strip() for value in polarization_ids)
        parameterization = str(parameterization_id).strip()
        if not polarizations or any(not value for value in polarizations):
            raise ValueError("Waveform polarization IDs must be non-empty.")
        if len(set(polarizations)) != len(polarizations):
            raise ValueError("Waveform polarization IDs must be unique.")
        if derivative_level not in ("none", "first", "higher") or not parameterization:
            raise ValueError("Waveform derivative level or parameterization is invalid.")
        if (phase_parameter is None) != (phase_harmonic is None):
            raise ValueError("Phase parameter and harmonic must be declared together.")
        if phase_parameter is not None and not str(phase_parameter).strip():
            raise ValueError("Phase parameter must be non-empty when declared.")
        if phase_harmonic is not None and int(phase_harmonic) == 0:
            raise ValueError("Phase harmonic must be nonzero.")
        if (distance_parameter is None) != (reference_distance is None):
            raise ValueError(
                "Distance parameter and reference distance must be declared together."
            )
        if distance_parameter is not None and not str(distance_parameter).strip():
            raise ValueError("Distance parameter must be non-empty when declared.")
        if reference_distance is not None and (
            not math.isfinite(float(reference_distance))
            or float(reference_distance) <= 0.0
        ):
            raise ValueError("Reference distance must be finite and positive.")
        self.polarization_ids = polarizations
        self.arbitrary_frequencies = bool(arbitrary_frequencies)
        self.jittable = bool(jittable)
        self.batched = bool(batched)
        self.derivative_level = derivative_level
        self.phase_parameter = (
            None if phase_parameter is None else str(phase_parameter).strip()
        )
        self.phase_harmonic = None if phase_harmonic is None else int(phase_harmonic)
        self.distance_parameter = (
            None if distance_parameter is None else str(distance_parameter).strip()
        )
        self.reference_distance = (
            None if reference_distance is None else float(reference_distance)
        )
        self.parameterization_id = parameterization
        self.capability_id = canonical_fingerprint(
            {
                "kind": "gravitational-wave-waveform-capabilities",
                "polarizations": list(polarizations),
                "arbitrary_frequencies": bool(arbitrary_frequencies),
                "jittable": bool(jittable),
                "batched": bool(batched),
                "derivative_level": derivative_level,
                "phase_parameter": self.phase_parameter,
                "phase_harmonic": self.phase_harmonic,
                "distance_parameter": self.distance_parameter,
                "reference_distance": self.reference_distance,
                "parameterization": parameterization,
            }
        )


class FrequencyDomainPolarizations(StrictModule):
    """Frequency-domain strain polarizations with values measured in seconds."""

    frequency: Array
    values: Array
    valid: Array
    status: Array
    polarization_ids: tuple[str, ...] = eqx.field(static=True)
    waveform_id: str = eqx.field(static=True)

    def __init__(
        self,
        frequency: ArrayLike,
        values: ArrayLike,
        polarization_ids: tuple[str, ...],
        /,
        *,
        waveform_id: str,
        valid: ArrayLike | None = None,
        invalid_status: ArrayLike | GravitationalWaveStatus = (
            GravitationalWaveStatus.NONFINITE_WAVEFORM
        ),
    ):
        frequencies = jnp.asarray(frequency)
        waveforms = jnp.asarray(values)
        polarizations = tuple(polarization_ids)
        if frequencies.ndim != 1 or waveforms.shape != (
            len(polarizations),
            frequencies.size,
        ):
            raise ValueError("Waveform values must have polarization-by-frequency shape.")
        if not jnp.issubdtype(waveforms.dtype, jnp.complexfloating):
            waveforms = waveforms.astype(jnp.result_type(waveforms.dtype, 1j))
        finite = jnp.all(jnp.isfinite(frequencies)) & jnp.all(jnp.isfinite(waveforms))
        declared = (
            jnp.asarray(True)
            if valid is None
            else jnp.asarray(valid, dtype=bool).reshape(())
        )
        valid_ = declared & finite
        status = jnp.where(
            valid_,
            int(GravitationalWaveStatus.SUCCESS),
            jnp.where(
                finite,
                jnp.asarray(invalid_status, dtype=jnp.int32).reshape(()),
                int(GravitationalWaveStatus.NONFINITE_WAVEFORM),
            ),
        ).astype(jnp.int32)
        self.frequency = frequencies
        self.values = jnp.where(valid_, waveforms, jnp.zeros_like(waveforms))
        self.valid = valid_
        self.status = status
        self.polarization_ids = polarizations
        self.waveform_id = str(waveform_id)


class AbstractFrequencyDomainWaveform(StrictModule):
    capabilities: WaveformCapabilities
    waveform_id: str = eqx.field(static=True)

    @abstractmethod
    def evaluate(
        self, frequency: ArrayLike, parameters: PyTree[Any], /
    ) -> FrequencyDomainPolarizations:
        raise NotImplementedError


class CallableFrequencyDomainWaveform(AbstractFrequencyDomainWaveform):
    """Declared pure callable waveform without parameter introspection or caching."""

    function: Callable[[Array, PyTree[Any]], ArrayLike | Mapping[str, ArrayLike]] = (
        eqx.field(static=True)
    )
    parameter_map: Callable[[PyTree[Any]], PyTree[Any]] | None = eqx.field(static=True)
    provenance: ObservationDataProvenance

    def __init__(
        self,
        function: Callable[[Array, PyTree[Any]], ArrayLike | Mapping[str, ArrayLike]],
        capabilities: WaveformCapabilities,
        provenance: ObservationDataProvenance,
        /,
        *,
        waveform_id: str,
        parameter_map: Callable[[PyTree[Any]], PyTree[Any]] | None = None,
    ):
        if not callable(function) or (
            parameter_map is not None and not callable(parameter_map)
        ):
            raise TypeError("Waveform function and parameter map must be callable.")
        if not isinstance(capabilities, WaveformCapabilities):
            raise TypeError("capabilities must be WaveformCapabilities.")
        if not isinstance(provenance, ObservationDataProvenance):
            raise TypeError("provenance must be ObservationDataProvenance.")
        identifier = str(waveform_id).strip()
        if not identifier:
            raise ValueError("waveform_id must be non-empty.")
        self.function = function
        self.parameter_map = parameter_map
        self.capabilities = capabilities
        self.provenance = provenance
        self.waveform_id = canonical_fingerprint(
            {
                "kind": "callable-frequency-domain-waveform",
                "label": identifier,
                "capabilities": capabilities.capability_id,
                "provenance": provenance.provenance_id,
            }
        )

    def evaluate(
        self, frequency: ArrayLike, parameters: PyTree[Any], /
    ) -> FrequencyDomainPolarizations:
        frequencies = jnp.asarray(frequency)
        physical = (
            parameters if self.parameter_map is None else self.parameter_map(parameters)
        )
        output = self.function(frequencies, physical)
        if isinstance(output, Mapping):
            if set(output) != set(self.capabilities.polarization_ids):
                raise ValueError(
                    "Waveform mapping keys must match declared polarizations."
                )
            values = jnp.stack(
                tuple(
                    jnp.asarray(output[name])
                    for name in self.capabilities.polarization_ids
                )
            )
        else:
            values = jnp.asarray(output)
        return FrequencyDomainPolarizations(
            frequencies,
            values,
            self.capabilities.polarization_ids,
            waveform_id=self.waveform_id,
        )


class SineGaussianWaveformPlan(AbstractFrequencyDomainWaveform, NonTrainableState):
    """Self-contained Gaussian spectral burst with plus/cross ellipticity."""

    provenance: ObservationDataProvenance

    def __init__(
        self,
        provenance: ObservationDataProvenance,
        /,
        *,
        parameterization_id: str = "sine-gaussian",
        waveform_id: str = "native-sine-gaussian",
    ):
        if not isinstance(provenance, ObservationDataProvenance):
            raise TypeError("provenance must be ObservationDataProvenance.")
        self.capabilities = WaveformCapabilities(
            ("plus", "cross"),
            arbitrary_frequencies=True,
            jittable=True,
            batched=False,
            derivative_level="higher",
            parameterization_id=parameterization_id,
            phase_parameter="phase",
            phase_harmonic=1,
            distance_parameter="luminosity_distance",
            reference_distance=1.0,
        )
        self.provenance = provenance
        self.waveform_id = canonical_fingerprint(
            {
                "kind": "native-sine-gaussian-waveform",
                "label": str(waveform_id),
                "capabilities": self.capabilities.capability_id,
                "provenance": provenance.provenance_id,
            }
        )

    def evaluate(
        self, frequency: ArrayLike, parameters: PyTree[Any], /
    ) -> FrequencyDomainPolarizations:
        if not isinstance(parameters, Mapping):
            raise TypeError("Sine-Gaussian parameters must be a mapping.")
        required = {
            "amplitude",
            "center_frequency",
            "quality_factor",
            "phase",
            "ellipticity",
            "luminosity_distance",
        }
        if set(parameters) != required:
            raise ValueError(
                f"Sine-Gaussian parameters must be exactly {sorted(required)}."
            )
        frequencies = jnp.asarray(frequency)
        amplitude = jnp.asarray(parameters["amplitude"]).reshape(())
        center = jnp.asarray(parameters["center_frequency"]).reshape(())
        quality = jnp.asarray(parameters["quality_factor"]).reshape(())
        phase = jnp.asarray(parameters["phase"]).reshape(())
        ellipticity = jnp.asarray(parameters["ellipticity"]).reshape(())
        distance = jnp.asarray(parameters["luminosity_distance"]).reshape(())
        scalars = jnp.stack((amplitude, center, quality, phase, ellipticity, distance))
        finite_scalars = jnp.all(jnp.isfinite(scalars))
        supported = (
            finite_scalars
            & (amplitude >= 0.0)
            & (center > 0.0)
            & (quality > 0.0)
            & (jnp.abs(ellipticity) <= 1.0)
            & (distance > 0.0)
        )
        safe_center = jnp.where(center > 0.0, center, 1.0)
        safe_quality = jnp.where(quality > 0.0, quality, 1.0)
        safe_distance = jnp.where(distance > 0.0, distance, 1.0)
        sigma = safe_center / safe_quality
        envelope = (
            amplitude
            / safe_distance
            * jnp.exp(-0.5 * ((frequencies - safe_center) / sigma) ** 2)
        )
        plus = envelope * jnp.exp(1j * phase)
        cross = 1j * ellipticity * plus
        return FrequencyDomainPolarizations(
            frequencies,
            jnp.stack((plus, cross)),
            self.capabilities.polarization_ids,
            waveform_id=self.waveform_id,
            valid=supported,
            invalid_status=jnp.where(
                finite_scalars,
                int(GravitationalWaveStatus.OUTSIDE_WAVEFORM_SUPPORT),
                int(GravitationalWaveStatus.NONFINITE_WAVEFORM),
            ),
        )


__all__ = [
    "AbstractFrequencyDomainWaveform",
    "CallableFrequencyDomainWaveform",
    "DerivativeLevel",
    "FrequencyDomainPolarizations",
    "SineGaussianWaveformPlan",
    "WaveformCapabilities",
]
