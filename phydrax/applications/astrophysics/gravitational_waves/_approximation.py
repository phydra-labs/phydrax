#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, PyTree

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ._data import DetectorNetworkData
from ._detector import DetectorResponsePlan
from ._likelihood import (
    AbstractGravitationalWaveLikelihood,
    GravitationalWaveLikelihoodEvaluation,
    GravitationalWaveLikelihoodPlan,
)
from ._status import GravitationalWaveStatus
from ._waveform import AbstractFrequencyDomainWaveform


class LikelihoodApproximationPolicy(StrictModule, NonTrainableState):
    maximum_absolute_log_probability_error: float = eqx.field(static=True)
    maximum_rms_log_probability_error: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        maximum_absolute_log_probability_error: float = 0.1,
        maximum_rms_log_probability_error: float = 0.05,
    ):
        maximum = float(maximum_absolute_log_probability_error)
        rms = float(maximum_rms_log_probability_error)
        if (
            not np.isfinite(maximum)
            or not np.isfinite(rms)
            or maximum <= 0.0
            or rms <= 0.0
        ):
            raise ValueError(
                "Likelihood approximation error gates must be finite and positive."
            )
        if rms > maximum:
            raise ValueError("RMS likelihood error gate cannot exceed the maximum gate.")
        self.maximum_absolute_log_probability_error = maximum
        self.maximum_rms_log_probability_error = rms
        self.policy_id = canonical_fingerprint(
            {
                "kind": "gravitational-wave-likelihood-approximation-policy",
                "maximum_absolute_error": maximum,
                "maximum_rms_error": rms,
            }
        )


class LikelihoodApproximationReport(StrictModule, NonTrainableState):
    exact_log_probability: Array
    approximate_log_probability: Array
    absolute_error: Array
    maximum_absolute_error: Array
    rms_error: Array
    valid: Array
    validation_ids: tuple[str, ...] = eqx.field(static=True)
    exact_likelihood_id: str = eqx.field(static=True)
    approximate_likelihood_id: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)
    report_id: str = eqx.field(static=True)

    @property
    def passed(self) -> bool:
        return bool(self.valid)


class LinearQuadraticCompressedLikelihood(AbstractGravitationalWaveLikelihood):
    """Prepared linear/quadratic sufficient statistics at reduced frequency nodes."""

    base: GravitationalWaveLikelihoodPlan
    network: DetectorNetworkData
    response: DetectorResponsePlan
    waveform: AbstractFrequencyDomainWaveform
    linear_frequency: Array
    quadratic_frequency: Array
    linear_weights: Array
    quadratic_weights: Array
    shared_nodes: bool = eqx.field(static=True)
    parameterization_id: str = eqx.field(static=True)
    likelihood_id: str = eqx.field(static=True)
    approximation_id: str = eqx.field(static=True)

    def __init__(
        self,
        base: GravitationalWaveLikelihoodPlan,
        linear_frequency: ArrayLike,
        quadratic_frequency: ArrayLike,
        linear_weights: ArrayLike,
        quadratic_weights: ArrayLike,
        /,
        *,
        approximation_id: str,
    ):
        if not isinstance(base, GravitationalWaveLikelihoodPlan):
            raise TypeError("base must be the exact GravitationalWaveLikelihoodPlan.")
        if base.calibration_fn is not None:
            raise ValueError(
                "Reduced linear/quadratic likelihood does not support calibration callbacks."
            )
        linear_f = jnp.asarray(linear_frequency)
        quadratic_f = jnp.asarray(quadratic_frequency)
        linear_w = jnp.asarray(linear_weights)
        quadratic_w = jnp.asarray(quadratic_weights)
        detector_count = len(base.network.detector_ids)
        if (
            linear_f.ndim != 1
            or quadratic_f.ndim != 1
            or linear_f.size == 0
            or quadratic_f.size == 0
            or linear_w.shape != (detector_count, linear_f.size)
            or quadratic_w.shape != (detector_count, quadratic_f.size)
            or not jnp.issubdtype(linear_w.dtype, jnp.complexfloating)
            or jnp.iscomplexobj(quadratic_w)
            or not bool(jnp.all(jnp.isfinite(linear_f)))
            or not bool(jnp.all(jnp.isfinite(quadratic_f)))
            or not bool(jnp.all(jnp.isfinite(linear_w)))
            or not bool(jnp.all(jnp.isfinite(quadratic_w)))
            or not bool(jnp.all(jnp.diff(linear_f) > 0.0))
            or not bool(jnp.all(jnp.diff(quadratic_f) > 0.0))
        ):
            raise ValueError("Compressed likelihood nodes or weights are invalid.")
        approximation = str(approximation_id).strip()
        if not approximation or approximation == "full-frequency":
            raise ValueError(
                "Compressed likelihood requires a distinct approximation ID."
            )
        self.base = base
        self.network = base.network
        self.response = base.response
        self.waveform = base.waveform
        self.parameterization_id = base.parameterization_id
        self.linear_frequency = linear_f
        self.quadratic_frequency = quadratic_f
        self.linear_weights = linear_w
        self.quadratic_weights = quadratic_w
        self.shared_nodes = bool(
            np.array_equal(np.asarray(linear_f), np.asarray(quadratic_f))
        )
        self.approximation_id = approximation
        self.likelihood_id = canonical_fingerprint(
            {
                "kind": "gravitational-wave-linear-quadratic-likelihood",
                "base": base.likelihood_id,
                "approximation": approximation,
                "content": array_tree_fingerprint(
                    {
                        "linear_frequency": linear_f,
                        "quadratic_frequency": quadratic_f,
                        "linear_weights": linear_w,
                        "quadratic_weights": quadratic_w,
                    }
                )["sha256"],
            }
        )

    def detector_signal(
        self, parameters: PyTree[Any], /, *, frequency: Array | None = None
    ):
        return self.base.detector_signal(parameters, frequency=frequency)

    def evaluate(
        self, parameters: PyTree[Any], /
    ) -> GravitationalWaveLikelihoodEvaluation:
        linear_signal, linear_response, linear_polarizations = self.base.detector_signal(
            parameters, frequency=self.linear_frequency
        )
        if self.shared_nodes:
            quadratic_signal = linear_signal
            quadratic_valid = linear_polarizations.valid
        else:
            quadratic_signal, quadratic_response, quadratic_polarizations = (
                self.base.detector_signal(parameters, frequency=self.quadratic_frequency)
            )
            quadratic_valid = quadratic_response.valid & quadratic_polarizations.valid
        complex_inner = jnp.sum(self.linear_weights * linear_signal, axis=-1)
        signal_norm = jnp.sum(
            self.quadratic_weights * jnp.abs(quadratic_signal) ** 2, axis=-1
        )
        ratio = jnp.real(complex_inner) - 0.5 * signal_norm
        noise = self.network.noise_log_probability_by_detector
        log_probability = noise + ratio
        data_norm = self.network.data_norm_by_detector
        valid = (
            linear_response.valid
            & linear_polarizations.valid
            & quadratic_valid
            & jnp.all(jnp.isfinite(log_probability))
            & jnp.all(signal_norm >= 0.0)
        )
        status = jnp.where(
            valid,
            int(GravitationalWaveStatus.SUCCESS),
            int(GravitationalWaveStatus.APPROXIMATION_FAILURE),
        ).astype(jnp.int32)
        matched_filter = complex_inner / jnp.sqrt(
            jnp.where(signal_norm > 0.0, signal_norm, 1.0)
        )
        return GravitationalWaveLikelihoodEvaluation(
            linear_signal,
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


class QualifiedGravitationalWaveLikelihood(AbstractGravitationalWaveLikelihood):
    """Approximate likelihood admitted only after exact held-out comparison."""

    candidate: AbstractGravitationalWaveLikelihood
    qualification: LikelihoodApproximationReport
    network: DetectorNetworkData
    response: DetectorResponsePlan
    waveform: AbstractFrequencyDomainWaveform
    parameterization_id: str = eqx.field(static=True)
    likelihood_id: str = eqx.field(static=True)
    approximation_id: str = eqx.field(static=True)

    def __init__(
        self,
        candidate: AbstractGravitationalWaveLikelihood,
        qualification: LikelihoodApproximationReport,
        /,
    ):
        if not isinstance(candidate, AbstractGravitationalWaveLikelihood):
            raise TypeError(
                "candidate must implement AbstractGravitationalWaveLikelihood."
            )
        if not isinstance(qualification, LikelihoodApproximationReport):
            raise TypeError("qualification must be LikelihoodApproximationReport.")
        if qualification.approximate_likelihood_id != candidate.likelihood_id:
            raise ValueError("Approximation report belongs to a different likelihood.")
        if not qualification.passed:
            raise ValueError("Approximate likelihood failed its exact comparison gates.")
        self.candidate = candidate
        self.qualification = qualification
        self.network = candidate.network
        self.response = candidate.response
        self.waveform = candidate.waveform
        self.parameterization_id = candidate.parameterization_id
        self.likelihood_id = candidate.likelihood_id
        self.approximation_id = candidate.approximation_id

    def detector_signal(
        self, parameters: PyTree[Any], /, *, frequency: Array | None = None
    ):
        return self.candidate.detector_signal(parameters, frequency=frequency)

    def evaluate(
        self, parameters: PyTree[Any], /
    ) -> GravitationalWaveLikelihoodEvaluation:
        return self.candidate.evaluate(parameters)


def qualify_likelihood_approximation(
    exact: GravitationalWaveLikelihoodPlan,
    approximate: AbstractGravitationalWaveLikelihood,
    validation_parameters: Sequence[PyTree[Any]],
    validation_ids: Sequence[str],
    /,
    *,
    policy: LikelihoodApproximationPolicy | None = None,
) -> LikelihoodApproximationReport:
    if not isinstance(exact, GravitationalWaveLikelihoodPlan):
        raise TypeError("exact must be GravitationalWaveLikelihoodPlan.")
    if not isinstance(approximate, AbstractGravitationalWaveLikelihood):
        raise TypeError("approximate must implement AbstractGravitationalWaveLikelihood.")
    parameters = tuple(validation_parameters)
    identifiers = tuple(str(value).strip() for value in validation_ids)
    if (
        not parameters
        or len(parameters) != len(identifiers)
        or any(not value for value in identifiers)
        or len(set(identifiers)) != len(identifiers)
    ):
        raise ValueError(
            "Validation parameters and IDs must be non-empty, distinct, and aligned."
        )
    policy_ = LikelihoodApproximationPolicy() if policy is None else policy
    if not isinstance(policy_, LikelihoodApproximationPolicy):
        raise TypeError("policy must be LikelihoodApproximationPolicy or None.")
    exact_values = jnp.stack(tuple(exact.log_probability(value) for value in parameters))
    approximate_values = jnp.stack(
        tuple(approximate.log_probability(value) for value in parameters)
    )
    absolute = jnp.abs(approximate_values - exact_values)
    maximum = jnp.max(absolute)
    rms = jnp.sqrt(jnp.mean(absolute**2))
    valid = (
        jnp.all(jnp.isfinite(exact_values))
        & jnp.all(jnp.isfinite(approximate_values))
        & (maximum <= policy_.maximum_absolute_log_probability_error)
        & (rms <= policy_.maximum_rms_log_probability_error)
    )
    report_id = canonical_fingerprint(
        {
            "kind": "gravitational-wave-likelihood-approximation-report",
            "exact": exact.likelihood_id,
            "approximate": approximate.likelihood_id,
            "validation_ids": list(identifiers),
            "policy": policy_.policy_id,
            "values": array_tree_fingerprint(
                {"exact": exact_values, "approximate": approximate_values}
            )["sha256"],
        }
    )
    return LikelihoodApproximationReport(
        exact_values,
        approximate_values,
        absolute,
        maximum,
        rms,
        valid,
        identifiers,
        exact.likelihood_id,
        approximate.likelihood_id,
        policy_.policy_id,
        report_id,
    )


def qualify_likelihood(
    exact: GravitationalWaveLikelihoodPlan,
    candidate: AbstractGravitationalWaveLikelihood,
    validation_parameters: Sequence[PyTree[Any]],
    validation_ids: Sequence[str],
    /,
    *,
    policy: LikelihoodApproximationPolicy | None = None,
) -> QualifiedGravitationalWaveLikelihood:
    report = qualify_likelihood_approximation(
        exact,
        candidate,
        validation_parameters,
        validation_ids,
        policy=policy,
    )
    return QualifiedGravitationalWaveLikelihood(candidate, report)


__all__ = [
    "LikelihoodApproximationPolicy",
    "LikelihoodApproximationReport",
    "LinearQuadraticCompressedLikelihood",
    "QualifiedGravitationalWaveLikelihood",
    "qualify_likelihood",
    "qualify_likelihood_approximation",
]
