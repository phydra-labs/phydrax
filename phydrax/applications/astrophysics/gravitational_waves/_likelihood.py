#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from ...._fingerprint import canonical_fingerprint
from ...._strict import AbstractAttribute, StrictModule
from ....uq import AbstractPosteriorTerm
from ._data import DetectorNetworkData
from ._detector import DetectorResponsePlan, DetectorResponseResult
from ._status import GravitationalWaveStatus
from ._waveform import AbstractFrequencyDomainWaveform, FrequencyDomainPolarizations


class GravitationalWaveLikelihoodEvaluation(StrictModule):
    detector_signal: Array
    complex_data_signal_inner_product: Array
    signal_norm: Array
    data_norm: Array
    log_probability_by_detector: Array
    noise_log_probability_by_detector: Array
    log_likelihood_ratio_by_detector: Array
    complex_matched_filter_snr: Array
    optimal_snr_squared: Array
    valid: Array
    status: Array
    likelihood_id: str = eqx.field(static=True)
    approximation_id: str = eqx.field(static=True)

    @property
    def data_signal_inner_product(self) -> Array:
        return jnp.real(self.complex_data_signal_inner_product)

    @property
    def log_probability(self) -> Array:
        return jnp.sum(self.log_probability_by_detector)

    @property
    def noise_log_probability(self) -> Array:
        return jnp.sum(self.noise_log_probability_by_detector)

    @property
    def log_likelihood_ratio(self) -> Array:
        return jnp.sum(self.log_likelihood_ratio_by_detector)


class AbstractGravitationalWaveLikelihood(StrictModule):
    network: AbstractAttribute[DetectorNetworkData]
    response: AbstractAttribute[DetectorResponsePlan]
    waveform: AbstractAttribute[AbstractFrequencyDomainWaveform]
    parameterization_id: AbstractAttribute[str]
    likelihood_id: AbstractAttribute[str]
    approximation_id: AbstractAttribute[str]

    @abstractmethod
    def detector_signal(
        self, parameters: PyTree[Any], /, *, frequency: Array | None = None
    ) -> tuple[Array, DetectorResponseResult, FrequencyDomainPolarizations]:
        raise NotImplementedError

    @abstractmethod
    def evaluate(
        self, parameters: PyTree[Any], /
    ) -> GravitationalWaveLikelihoodEvaluation:
        raise NotImplementedError

    def log_probability(self, parameters: PyTree[Any], /) -> Array:
        evaluation = self.evaluate(parameters)
        return jnp.where(evaluation.valid, evaluation.log_probability, -jnp.inf)

    def noise_log_probability(self) -> Array:
        return self.network.noise_log_probability

    def log_likelihood_ratio(self, parameters: PyTree[Any], /) -> Array:
        evaluation = self.evaluate(parameters)
        return jnp.where(evaluation.valid, evaluation.log_likelihood_ratio, -jnp.inf)


class GravitationalWaveLikelihoodPlan(AbstractGravitationalWaveLikelihood):
    """Normalized independent-detector spectral Gaussian likelihood."""

    network: DetectorNetworkData
    response: DetectorResponsePlan
    waveform: AbstractFrequencyDomainWaveform
    waveform_parameters_fn: Callable[[PyTree[Any]], PyTree[Any]] = eqx.field(static=True)
    extrinsic_parameters_fn: Callable[
        [PyTree[Any]], tuple[Array, Array, Array, Array]
    ] = eqx.field(static=True)
    calibration_fn: Callable[[PyTree[Any], Array], Array] | None = eqx.field(static=True)
    calibration_id: str | None = eqx.field(static=True)
    parameterization_id: str = eqx.field(static=True)
    likelihood_id: str = eqx.field(static=True)
    approximation_id: str = eqx.field(static=True)

    def __init__(
        self,
        network: DetectorNetworkData,
        response: DetectorResponsePlan,
        waveform: AbstractFrequencyDomainWaveform,
        waveform_parameters: Callable[[PyTree[Any]], PyTree[Any]],
        extrinsic_parameters: Callable[[PyTree[Any]], tuple[Array, Array, Array, Array]],
        /,
        *,
        parameterization_id: str,
        calibration_id: str | None = None,
        calibration: Callable[[PyTree[Any], Array], Array] | None = None,
        approximation_id: str = "full-frequency",
    ):
        if not isinstance(network, DetectorNetworkData):
            raise TypeError("network must be DetectorNetworkData.")
        if not isinstance(response, DetectorResponsePlan):
            raise TypeError("response must be DetectorResponsePlan.")
        if not isinstance(waveform, AbstractFrequencyDomainWaveform):
            raise TypeError("waveform must implement AbstractFrequencyDomainWaveform.")
        if response.detector_ids != network.detector_ids:
            raise ValueError("Detector response and data network IDs disagree.")
        if waveform.capabilities.polarization_ids != ("plus", "cross"):
            raise ValueError(
                "The initial detector likelihood requires plus/cross polarizations."
            )
        if not callable(waveform_parameters) or not callable(extrinsic_parameters):
            raise TypeError("Likelihood parameter maps must be callable.")
        if calibration is not None and not callable(calibration):
            raise TypeError("calibration must be callable or None.")
        parameterization = str(parameterization_id).strip()
        if parameterization != waveform.capabilities.parameterization_id:
            raise ValueError(
                "Likelihood and waveform parameterization identities must agree."
            )
        calibration_identifier = (
            None if calibration_id is None else str(calibration_id).strip()
        )
        if (calibration is None) != (calibration_identifier is None):
            raise ValueError(
                "A calibration callback and calibration_id must be declared together."
            )
        approximation = str(approximation_id).strip()
        if not approximation:
            raise ValueError("approximation_id must be non-empty.")
        self.network = network
        self.response = response
        self.waveform = waveform
        self.waveform_parameters_fn = waveform_parameters
        self.extrinsic_parameters_fn = extrinsic_parameters
        self.calibration_fn = calibration
        self.calibration_id = calibration_identifier
        self.parameterization_id = parameterization
        self.approximation_id = approximation
        self.likelihood_id = canonical_fingerprint(
            {
                "kind": "gravitational-wave-spectral-likelihood",
                "network": network.network_id,
                "response": response.plan_id,
                "waveform": waveform.waveform_id,
                "parameterization": parameterization,
                "calibration": calibration_identifier,
                "approximation": approximation,
                "normalization": "proper-complex-positive-frequency",
            }
        )

    def waveform_parameters(self, parameters: PyTree[Any], /) -> PyTree[Any]:
        return self.waveform_parameters_fn(parameters)

    def extrinsic_parameters(
        self, parameters: PyTree[Any], /
    ) -> tuple[Array, Array, Array, Array]:
        return self.extrinsic_parameters_fn(parameters)

    def detector_signal(
        self, parameters: PyTree[Any], /, *, frequency: Array | None = None
    ) -> tuple[Array, DetectorResponseResult, FrequencyDomainPolarizations]:
        frequencies = (
            self.network.frequency if frequency is None else jnp.asarray(frequency)
        )
        polarizations = self.waveform.evaluate(
            frequencies, self.waveform_parameters(parameters)
        )
        right_ascension, declination, polarization, geocent_time = (
            self.extrinsic_parameters(parameters)
        )
        response = self.response.evaluate(
            right_ascension, declination, polarization, geocent_time
        )
        antenna = response.antenna
        signal = (
            antenna[:, 0, None] * polarizations.values[0][None, :]
            + antenna[:, 1, None] * polarizations.values[1][None, :]
        )
        arrival_offset = (
            response.time_delay_seconds
            + jnp.asarray(geocent_time).reshape(())
            - self.network.start_time_gps
        )
        delay_phase = jnp.exp(
            -2.0j * jnp.pi * frequencies[None, :] * arrival_offset[:, None]
        )
        signal = signal * delay_phase
        if self.calibration_fn is not None:
            calibration = jnp.asarray(self.calibration_fn(parameters, frequencies))
            if calibration.shape != signal.shape:
                raise ValueError(
                    "Calibration response must have detector-by-frequency shape."
                )
            signal = signal * calibration
        return signal, response, polarizations

    def evaluate(
        self, parameters: PyTree[Any], /
    ) -> GravitationalWaveLikelihoodEvaluation:
        signal, response, polarizations = self.detector_signal(parameters)
        inverse_variance = self.network.inverse_variance
        complex_inner = 2.0 * jnp.sum(
            jnp.conj(self.network.strain) * signal * inverse_variance,
            axis=-1,
        )
        matched = jnp.real(complex_inner)
        signal_norm = 2.0 * jnp.sum(jnp.abs(signal) ** 2 * inverse_variance, axis=-1)
        data_norm = self.network.data_norm_by_detector
        ratio = matched - 0.5 * signal_norm
        noise = self.network.noise_log_probability_by_detector
        log_probability = noise + ratio
        valid = (
            polarizations.valid
            & response.valid
            & jnp.all(jnp.isfinite(log_probability))
            & jnp.all(jnp.isfinite(signal_norm))
        )
        status = jnp.where(
            ~polarizations.valid,
            polarizations.status,
            jnp.where(
                ~response.valid,
                response.status,
                jnp.where(
                    valid,
                    int(GravitationalWaveStatus.SUCCESS),
                    int(GravitationalWaveStatus.NONFINITE_WAVEFORM),
                ),
            ),
        ).astype(jnp.int32)
        safe_norm = jnp.where(signal_norm > 0.0, signal_norm, 1.0)
        matched_filter = complex_inner / jnp.sqrt(safe_norm)
        return GravitationalWaveLikelihoodEvaluation(
            signal,
            complex_inner,
            signal_norm,
            data_norm,
            log_probability,
            noise,
            ratio,
            matched_filter,
            signal_norm,
            valid,
            status,
            self.likelihood_id,
            self.approximation_id,
        )


class GravitationalWavePosteriorTerm(AbstractPosteriorTerm):
    """One normalized detector-network event for a posterior problem."""

    likelihood: AbstractGravitationalWaveLikelihood

    def __init__(
        self,
        likelihood: AbstractGravitationalWaveLikelihood,
        /,
        *,
        label: str = "gravitational_wave_network",
    ):
        if not isinstance(likelihood, AbstractGravitationalWaveLikelihood):
            raise TypeError(
                "likelihood must implement AbstractGravitationalWaveLikelihood."
            )
        identifier = str(label).strip()
        if not identifier:
            raise ValueError("label must be non-empty.")
        self.likelihood = likelihood
        self.label = identifier

    def per_case_log_prob(self, parameters: PyTree[Any], /) -> Array:
        return self.likelihood.log_probability(parameters).reshape((1,))


__all__ = [
    "AbstractGravitationalWaveLikelihood",
    "GravitationalWaveLikelihoodEvaluation",
    "GravitationalWaveLikelihoodPlan",
    "GravitationalWavePosteriorTerm",
]
