#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Finite complex measures and overlap-aware phase-quenched reweighting."""

from __future__ import annotations

from enum import IntEnum

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .. import ein
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._numerics import log_normalize, weight_ess
from .._strict import StrictModule
from .._trainable import NonTrainableState


class ComplexWeightStatus(IntEnum):
    """JAX-compatible finite complex-measure status codes."""

    SUCCESS = 0
    INVALID_WEIGHTS = 1
    NO_ACTIVE_SUPPORT = 2
    INSUFFICIENT_OVERLAP = 3
    INSUFFICIENT_EFFECTIVE_SAMPLES = 4
    NONFINITE_OBSERVABLE = 5
    UNRESOLVED_RATIO = 6


class ComplexWeightMeasure(StrictModule, NonTrainableState):
    """A finite measure represented by log magnitudes and unit complex phases."""

    samples: Array
    log_magnitudes: Array
    phases: Array
    mask: Array
    valid: Array
    measure_id: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)

    def __init__(
        self,
        samples: ArrayLike,
        log_magnitudes: ArrayLike,
        phases: ArrayLike,
        /,
        *,
        mask: ArrayLike | None = None,
        source_id: str,
    ):
        samples_host = np.asarray(samples)
        logs_host = np.asarray(log_magnitudes)
        phases_host = np.asarray(phases)
        if samples_host.ndim < 1 or samples_host.shape[0] < 1:
            raise ValueError("Complex measures require a nonempty leading support axis.")
        count = samples_host.shape[0]
        if logs_host.shape != (count,) or phases_host.shape != (count,):
            raise ValueError(
                "log_magnitudes and phases must match the leading sample axis."
            )
        if logs_host.dtype.kind not in "f" or phases_host.dtype.kind != "c":
            raise TypeError(
                "Complex weights require real log magnitudes and complex phases."
            )
        active_host = (
            np.ones((count,), dtype=np.bool_)
            if mask is None
            else np.asarray(mask, dtype=np.bool_)
        )
        if active_host.shape != (count,):
            raise ValueError("mask must match the leading sample axis.")
        source = str(source_id)
        if not source:
            raise ValueError("source_id must be non-empty.")
        admissible_logs = np.isfinite(logs_host) | np.isneginf(logs_host)
        positive = np.isfinite(logs_host)
        magnitudes = np.abs(phases_host)
        tolerance = 256.0 * np.finfo(logs_host.dtype).eps
        unit_phase = (~positive) | (np.abs(magnitudes - 1.0) <= tolerance)
        phase_finite = np.isfinite(phases_host.real) & np.isfinite(phases_host.imag)
        valid_host = bool(
            np.all(~active_host | (admissible_logs & unit_phase & phase_finite))
        )
        logs = jnp.asarray(logs_host)
        phases_ = jnp.asarray(phases_host)
        active = jnp.asarray(active_host)
        self.samples = jnp.asarray(samples_host)
        self.log_magnitudes = logs
        self.phases = phases_
        self.mask = active
        self.valid = jnp.asarray(valid_host)
        self.source_id = source
        self.measure_id = canonical_fingerprint(
            {
                "kind": "finite-complex-weight-measure",
                "samples": array_tree_fingerprint(samples_host),
                "log_magnitudes": array_tree_fingerprint(logs_host),
                "phases": array_tree_fingerprint(phases_host),
                "mask": array_tree_fingerprint(active_host),
                "source": source,
            }
        )

    @property
    def count(self) -> int:
        return self.log_magnitudes.shape[0]


def complex_weight_measure(
    samples: ArrayLike,
    weights: ArrayLike,
    /,
    *,
    mask: ArrayLike | None = None,
    source_id: str,
) -> ComplexWeightMeasure:
    """Construct the explicit log-magnitude/phase representation of complex weights."""
    weights_host = np.asarray(weights)
    if weights_host.ndim != 1 or weights_host.dtype.kind != "c":
        raise TypeError("weights must be a one-dimensional complex array.")
    if np.any(~np.isfinite(weights_host.real)) or np.any(~np.isfinite(weights_host.imag)):
        raise ValueError("Complex weights must be finite.")
    magnitude = np.abs(weights_host)
    positive = magnitude > 0.0
    log_magnitude = np.where(
        positive, np.log(np.where(positive, magnitude, 1.0)), -np.inf
    )
    phase = np.where(
        positive, weights_host / np.where(positive, magnitude, 1.0), 1.0 + 0.0j
    )
    return ComplexWeightMeasure(
        samples,
        log_magnitude,
        phase,
        mask=mask,
        source_id=source_id,
    )


class PhaseQuenchedReweightingPlan(StrictModule, NonTrainableState):
    """Finite overlap, effective-sample, and allocation contract."""

    minimum_average_phase: float = eqx.field(static=True)
    minimum_effective_sample_size: float = eqx.field(static=True)
    maximum_samples: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        minimum_average_phase: float = 1e-3,
        minimum_effective_sample_size: float = 2.0,
        maximum_samples: int = 1_000_000,
    ):
        overlap = float(minimum_average_phase)
        effective = float(minimum_effective_sample_size)
        maximum = int(maximum_samples)
        if not np.isfinite(overlap) or not 0.0 < overlap <= 1.0:
            raise ValueError("minimum_average_phase must lie in (0, 1].")
        if not np.isfinite(effective) or effective <= 1.0:
            raise ValueError("minimum_effective_sample_size must exceed one.")
        if maximum <= 1:
            raise ValueError("maximum_samples must exceed one.")
        self.minimum_average_phase = overlap
        self.minimum_effective_sample_size = effective
        self.maximum_samples = maximum
        self.plan_id = canonical_fingerprint(
            {
                "kind": "phase-quenched-reweighting-plan",
                "minimum_average_phase": overlap,
                "minimum_effective_sample_size": effective,
                "maximum_samples": maximum,
            }
        )


class PhaseQuenchedOverlapDiagnostics(StrictModule):
    """Denominator and positive-measure evidence available before observables."""

    average_phase: Array
    average_phase_magnitude: Array
    denominator_standard_error: Array
    denominator_signal_to_noise: Array
    phase_quenched_effective_sample_size: Array
    sign_effective_sample_size: Array
    active_samples: Array
    status: Array


class PreparedPhaseQuenchedMeasure(StrictModule, NonTrainableState):
    """Normalized positive reference measure with a frozen overlap decision."""

    measure: ComplexWeightMeasure
    normalized_magnitudes: Array
    log_phase_quenched_mass: Array
    diagnostics: PhaseQuenchedOverlapDiagnostics
    plan: PhaseQuenchedReweightingPlan
    prepared_id: str = eqx.field(static=True)


class RatioUncertainty(StrictModule):
    """Delta-method real/imaginary covariance for a complex reweighted ratio."""

    real_variance: Array
    imaginary_variance: Array
    real_imaginary_covariance: Array
    standard_error: Array
    denominator_standard_error: Array
    method: str = eqx.field(static=True)


class PhaseQuenchedReweightingResult(StrictModule):
    """Complex target expectation and complete overlap/ratio evidence."""

    value: Array
    numerator: Array
    denominator: Array
    uncertainty: RatioUncertainty
    overlap: PhaseQuenchedOverlapDiagnostics
    status: Array
    successful: Array
    measure_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    claim: str = eqx.field(static=True)


def _weighted_complex_mean(weights: Array, values: Array, /) -> Array:
    return ein.contract("i,i...->...", weights, values)


def _weighted_mean_variance(weights: Array, centered: Array, /) -> Array:
    """Unbiased fixed-weight variance estimate of the corresponding weighted mean."""
    output_ndim = centered.ndim - 1
    squared_weights = weights * weights
    denominator = 1.0 - jnp.sum(squared_weights)
    reshaped = squared_weights.reshape((squared_weights.shape[0],) + (1,) * output_ndim)
    numerator = jnp.sum(reshaped * centered, axis=0)
    return jnp.where(denominator > 0.0, numerator / denominator, jnp.inf)


def prepare_phase_quenched_reweighting(
    measure: ComplexWeightMeasure,
    plan: PhaseQuenchedReweightingPlan,
    /,
) -> PreparedPhaseQuenchedMeasure:
    """Normalize the positive measure and make overlap failure explicit."""
    if not isinstance(measure, ComplexWeightMeasure):
        raise TypeError("measure must be ComplexWeightMeasure.")
    if not isinstance(plan, PhaseQuenchedReweightingPlan):
        raise TypeError("plan must be PhaseQuenchedReweightingPlan.")
    if measure.count > plan.maximum_samples:
        raise ValueError(
            "Complex measure exceeds maximum_samples; no work array was allocated."
        )
    normalized, log_mass, normalization_valid = log_normalize(
        measure.log_magnitudes,
        axes=0,
        mask=measure.mask,
    )
    average_phase = _weighted_complex_mean(normalized, measure.phases)
    overlap = jnp.abs(average_phase)
    effective = weight_ess(normalized, axis=0)
    centered_phase = jnp.real(
        (measure.phases - average_phase) * jnp.conj(measure.phases - average_phase)
    )
    denominator_variance = _weighted_mean_variance(normalized, centered_phase)
    denominator_standard_error = jnp.sqrt(jnp.maximum(denominator_variance, 0.0))
    signal_to_noise = overlap / jnp.maximum(
        denominator_standard_error, jnp.finfo(normalized.dtype).tiny
    )
    sign_effective = effective * overlap * overlap
    active = jnp.sum(measure.mask & jnp.isfinite(measure.log_magnitudes), dtype=jnp.int32)
    inputs_valid = (
        measure.valid
        & normalization_valid
        & jnp.isfinite(overlap)
        & jnp.isfinite(effective)
    )
    status = jnp.where(
        active == 0,
        int(ComplexWeightStatus.NO_ACTIVE_SUPPORT),
        jnp.where(
            ~inputs_valid,
            int(ComplexWeightStatus.INVALID_WEIGHTS),
            jnp.where(
                overlap < plan.minimum_average_phase,
                int(ComplexWeightStatus.INSUFFICIENT_OVERLAP),
                jnp.where(
                    effective < plan.minimum_effective_sample_size,
                    int(ComplexWeightStatus.INSUFFICIENT_EFFECTIVE_SAMPLES),
                    int(ComplexWeightStatus.SUCCESS),
                ),
            ),
        ),
    ).astype(jnp.int32)
    diagnostics = PhaseQuenchedOverlapDiagnostics(
        average_phase=average_phase,
        average_phase_magnitude=overlap,
        denominator_standard_error=denominator_standard_error,
        denominator_signal_to_noise=signal_to_noise,
        phase_quenched_effective_sample_size=effective,
        sign_effective_sample_size=sign_effective,
        active_samples=active,
        status=status,
    )
    prepared_id = canonical_fingerprint(
        {
            "kind": "prepared-phase-quenched-measure",
            "measure": measure.measure_id,
            "plan": plan.plan_id,
        }
    )
    return PreparedPhaseQuenchedMeasure(
        measure=measure,
        normalized_magnitudes=normalized,
        log_phase_quenched_mass=log_mass,
        diagnostics=diagnostics,
        plan=plan,
        prepared_id=prepared_id,
    )


def phase_quenched_reweight(
    prepared: PreparedPhaseQuenchedMeasure,
    observable_values: ArrayLike,
    /,
) -> PhaseQuenchedReweightingResult:
    """Evaluate a complex target ratio with overlap-aware delta uncertainty."""
    if not isinstance(prepared, PreparedPhaseQuenchedMeasure):
        raise TypeError("prepared must be PreparedPhaseQuenchedMeasure.")
    observable = jnp.asarray(observable_values)
    count = prepared.measure.count
    if observable.ndim < 1 or observable.shape[0] != count:
        raise ValueError("observable_values must begin with the measure support axis.")
    positive_support = prepared.measure.mask & (prepared.normalized_magnitudes > 0.0)
    active = positive_support.reshape((count,) + (1,) * (observable.ndim - 1))
    finite_observable = jnp.all(
        ~active
        | (jnp.isfinite(jnp.real(observable)) & jnp.isfinite(jnp.imag(observable)))
    )
    safe_observable = jnp.where(active, observable, 0)
    phase_shape = (count,) + (1,) * (observable.ndim - 1)
    phased = prepared.measure.phases.reshape(phase_shape) * safe_observable
    numerator = _weighted_complex_mean(prepared.normalized_magnitudes, phased)
    denominator = prepared.diagnostics.average_phase
    denominator_valid = jnp.isfinite(denominator) & (jnp.abs(denominator) > 0.0)
    safe_denominator = jnp.where(denominator_valid, denominator, 1.0 + 0.0j)
    ratio = numerator / safe_denominator
    influence = (
        prepared.measure.phases.reshape(phase_shape)
        * (safe_observable - ratio)
        / safe_denominator
    )
    influence_real = jnp.real(influence)
    influence_imag = jnp.imag(influence)
    real_variance = _weighted_mean_variance(
        prepared.normalized_magnitudes, influence_real * influence_real
    )
    imaginary_variance = _weighted_mean_variance(
        prepared.normalized_magnitudes, influence_imag * influence_imag
    )
    covariance = _weighted_mean_variance(
        prepared.normalized_magnitudes, influence_real * influence_imag
    )
    standard_error = jnp.sqrt(jnp.maximum(real_variance + imaginary_variance, 0.0))
    status = jnp.where(
        prepared.diagnostics.status != int(ComplexWeightStatus.SUCCESS),
        prepared.diagnostics.status,
        jnp.where(
            ~finite_observable,
            int(ComplexWeightStatus.NONFINITE_OBSERVABLE),
            jnp.where(
                denominator_valid
                & jnp.all(jnp.isfinite(jnp.real(ratio)))
                & jnp.all(jnp.isfinite(jnp.imag(ratio)))
                & jnp.all(jnp.isfinite(standard_error)),
                int(ComplexWeightStatus.SUCCESS),
                int(ComplexWeightStatus.UNRESOLVED_RATIO),
            ),
        ),
    ).astype(jnp.int32)
    successful = status == int(ComplexWeightStatus.SUCCESS)
    output_dtype = jnp.result_type(ratio, 1j)
    value = jnp.where(
        successful, ratio, jnp.asarray(jnp.nan + 1j * jnp.nan, dtype=output_dtype)
    )
    uncertainty = RatioUncertainty(
        real_variance=real_variance,
        imaginary_variance=imaginary_variance,
        real_imaginary_covariance=covariance,
        standard_error=standard_error,
        denominator_standard_error=prepared.diagnostics.denominator_standard_error,
        method="self-normalized-complex-delta-covariance",
    )
    return PhaseQuenchedReweightingResult(
        value=value,
        numerator=numerator,
        denominator=denominator,
        uncertainty=uncertainty,
        overlap=prepared.diagnostics,
        status=status,
        successful=successful,
        measure_id=prepared.measure.measure_id,
        prepared_id=prepared.prepared_id,
        claim="finite-phase-quenched-ratio-with-overlap-qualified-uncertainty",
    )


__all__ = [
    "ComplexWeightMeasure",
    "ComplexWeightStatus",
    "PhaseQuenchedOverlapDiagnostics",
    "PhaseQuenchedReweightingPlan",
    "PhaseQuenchedReweightingResult",
    "PreparedPhaseQuenchedMeasure",
    "RatioUncertainty",
    "complex_weight_measure",
    "phase_quenched_reweight",
    "prepare_phase_quenched_reweighting",
]
