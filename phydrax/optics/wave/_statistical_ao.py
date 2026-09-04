#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._atmosphere import PreparedLayeredAtmosphere
from ._imaging import NormalizedTransferFunction


class StatisticalAOStatus(IntEnum):
    SUCCESS = 0
    NONFINITE = 1
    NEGATIVE_PSD = 2
    BUDGET_CLOSURE_FAILURE = 3
    INCOMPATIBLE_SUPPORT = 4


class StatisticalResidualAOPlan(StrictModule, NonTrainableState):
    """Statistical correction transfer, without WFS/DM/controller semantics."""

    control_cutoff: Array
    correction_gain: Array
    loop_delay: Array
    measurement_phase_variance: Array
    aliasing_phase_variance: Array

    def __init__(
        self,
        control_cutoff: ArrayLike,
        /,
        *,
        correction_gain: ArrayLike = 1.0,
        loop_delay: ArrayLike = 0.0,
        measurement_phase_variance: ArrayLike = 0.0,
        aliasing_phase_variance: ArrayLike = 0.0,
    ):
        cutoff = jnp.asarray(control_cutoff, dtype=float)
        gain = jnp.asarray(correction_gain, dtype=float)
        delay = jnp.asarray(loop_delay, dtype=float)
        measurement = jnp.asarray(measurement_phase_variance, dtype=float)
        aliasing = jnp.asarray(aliasing_phase_variance, dtype=float)
        values = (cutoff, gain, delay, measurement, aliasing)
        if any(value.shape != () for value in values):
            raise ValueError("Statistical AO plan parameters must be scalar.")
        cutoff = eqx.error_if(
            cutoff,
            (~jnp.isfinite(cutoff)) | (cutoff <= 0.0),
            "control_cutoff must be finite and positive.",
        )
        gain = eqx.error_if(
            gain,
            (~jnp.isfinite(gain)) | (gain < 0.0) | (gain > 1.0),
            "correction_gain must lie in [0, 1].",
        )
        delay = eqx.error_if(
            delay,
            (~jnp.isfinite(delay)) | (delay < 0.0),
            "loop_delay must be finite and nonnegative.",
        )
        measurement = eqx.error_if(
            measurement,
            (~jnp.isfinite(measurement)) | (measurement < 0.0),
            "measurement_phase_variance must be finite and nonnegative.",
        )
        aliasing = eqx.error_if(
            aliasing,
            (~jnp.isfinite(aliasing)) | (aliasing < 0.0),
            "aliasing_phase_variance must be finite and nonnegative.",
        )
        self.control_cutoff = cutoff
        self.correction_gain = gain
        self.loop_delay = delay
        self.measurement_phase_variance = measurement
        self.aliasing_phase_variance = aliasing

    def prepare(
        self, atmosphere: PreparedLayeredAtmosphere, /
    ) -> "PreparedStatisticalResidualAO":
        return prepare_statistical_residual_ao(self, atmosphere)


class ResidualAOErrorBudget(StrictModule, NonTrainableState):
    atmospheric_variance: Array
    fitting_variance: Array
    servo_lag_variance: Array
    measurement_variance: Array
    aliasing_variance: Array
    total_residual_variance: Array
    total_residual_rms: Array
    marechal_strehl: Array
    closure_error: Array


class StatisticalAOSamplingEvidence(StrictModule, NonTrainableState):
    frequency_cell_area: Array
    control_cutoff: Array
    controlled_mode_fraction: Array
    nyquist_frequencies: Array
    finite: Array
    nonnegative: Array
    budget_closed: Array
    valid: Array
    status: Array


class PreparedStatisticalResidualAO(StrictModule, NonTrainableState):
    plan: StatisticalResidualAOPlan
    atmosphere: PreparedLayeredAtmosphere
    spatial_frequencies: Array
    atmospheric_psd: Array
    fitting_psd: Array
    servo_lag_psd: Array
    measurement_noise_psd: Array
    aliasing_psd: Array
    total_residual_psd: Array
    controlled_modes: Array
    error_budget: ResidualAOErrorBudget
    evidence: StatisticalAOSamplingEvidence

    @property
    def valid(self) -> Array:
        return self.evidence.valid

    def execute(
        self,
        diffraction_limited: NormalizedTransferFunction,
        pupil_separation_scale: ArrayLike,
        /,
    ) -> "LongExposureOTFResult":
        return long_exposure_otf(
            diffraction_limited,
            self,
            pupil_separation_scale,
        )


class LongExposureSamplingEvidence(StrictModule, NonTrainableState):
    pupil_separation_scale: Array
    requested_separation_axes: tuple[Array, Array]
    residual_separation_axes: tuple[Array, Array]
    relative_alignment_error: Array
    aligned: Array


class LongExposureOTFResult(StrictModule):
    optical_transfer_function: Array
    modulation_transfer_function: Array
    atmospheric_transfer_function: Array
    residual_structure_function: Array
    error_budget: ResidualAOErrorBudget
    sampling: LongExposureSamplingEvidence
    finite: Array
    valid: Array
    status: Array


def _integrate_psd(power_spectral_density: Array, domain_area: Array, /) -> Array:
    return jnp.sum(power_spectral_density) / domain_area


def _normalized_white_psd(
    requested_variance: Array,
    support: Array,
    domain_area: Array,
    /,
) -> Array:
    mode_count = jnp.sum(support.astype(jnp.int32))
    safe_count = jnp.maximum(mode_count, 1)
    level = requested_variance * domain_area / safe_count
    return jnp.where(support, level, 0.0)


def prepare_statistical_residual_ao(
    plan: StatisticalResidualAOPlan,
    atmosphere: PreparedLayeredAtmosphere,
    /,
) -> PreparedStatisticalResidualAO:
    """Prepare residual phase PSD components and their closed variance budget."""
    if not isinstance(plan, StatisticalResidualAOPlan) or not isinstance(
        atmosphere, PreparedLayeredAtmosphere
    ):
        raise TypeError("Expected a statistical AO plan and prepared atmosphere.")
    reference = atmosphere.layers[0].screen
    frequencies = reference.spatial_frequencies
    shape = reference.plan.space.shape
    if any(layer.screen.plan.space.shape != shape for layer in atmosphere.layers):
        raise ValueError("All AO atmosphere layers must have the same spectral shape.")
    radial_frequency = jnp.sqrt(jnp.sum(frequencies * frequencies, axis=-1))
    controlled = (radial_frequency <= plan.control_cutoff) & reference.supported_modes
    atmospheric_psd = jnp.zeros(shape, dtype=reference.power_spectral_density.dtype)
    fitting_psd = jnp.zeros_like(atmospheric_psd)
    servo_lag_psd = jnp.zeros_like(atmospheric_psd)
    for layer in atmosphere.layers:
        if layer.screen.plan.space.space_id != reference.plan.space.space_id:
            raise ValueError("AO layers must share an identical periodic support.")
        layer_psd = layer.layer.strength_fraction * layer.screen.power_spectral_density
        temporal_frequency = contract("...i,i->...", frequencies, layer.layer.velocity)
        delayed_correction = plan.correction_gain * jnp.exp(
            -2j * jnp.pi * temporal_frequency * plan.loop_delay
        )
        residual_transfer = jnp.abs(1.0 - delayed_correction) ** 2
        atmospheric_psd = atmospheric_psd + layer_psd
        fitting_psd = fitting_psd + jnp.where(controlled, 0.0, layer_psd)
        servo_lag_psd = servo_lag_psd + jnp.where(
            controlled, residual_transfer * layer_psd, 0.0
        )
    lengths = reference.lengths
    domain_area = lengths[0] * lengths[1]
    frequency_cell_area = 1.0 / domain_area
    measurement_psd = _normalized_white_psd(
        plan.measurement_phase_variance,
        controlled,
        domain_area,
    )
    aliasing_psd = _normalized_white_psd(
        plan.aliasing_phase_variance,
        controlled,
        domain_area,
    )
    total_residual_psd = fitting_psd + servo_lag_psd + measurement_psd + aliasing_psd
    atmospheric_variance = _integrate_psd(atmospheric_psd, domain_area)
    fitting_variance = _integrate_psd(fitting_psd, domain_area)
    servo_lag_variance = _integrate_psd(servo_lag_psd, domain_area)
    measurement_variance = _integrate_psd(measurement_psd, domain_area)
    aliasing_variance = _integrate_psd(aliasing_psd, domain_area)
    total_variance = _integrate_psd(total_residual_psd, domain_area)
    component_sum = (
        fitting_variance + servo_lag_variance + measurement_variance + aliasing_variance
    )
    closure_error = jnp.abs(total_variance - component_sum) / jnp.maximum(
        total_variance, 1.0
    )
    budget = ResidualAOErrorBudget(
        atmospheric_variance,
        fitting_variance,
        servo_lag_variance,
        measurement_variance,
        aliasing_variance,
        total_variance,
        jnp.sqrt(jnp.maximum(total_variance, 0.0)),
        jnp.exp(-jnp.maximum(total_variance, 0.0)),
        closure_error,
    )
    psds = jnp.stack(
        (
            atmospheric_psd,
            fitting_psd,
            servo_lag_psd,
            measurement_psd,
            aliasing_psd,
            total_residual_psd,
        )
    )
    finite = jnp.all(jnp.isfinite(psds))
    nonnegative = jnp.all(psds >= 0.0)
    tolerance = 256.0 * jnp.finfo(psds.dtype).eps
    budget_closed = closure_error <= tolerance
    valid = finite & nonnegative & budget_closed
    status = jnp.where(
        ~finite,
        int(StatisticalAOStatus.NONFINITE),
        jnp.where(
            ~nonnegative,
            int(StatisticalAOStatus.NEGATIVE_PSD),
            jnp.where(
                ~budget_closed,
                int(StatisticalAOStatus.BUDGET_CLOSURE_FAILURE),
                int(StatisticalAOStatus.SUCCESS),
            ),
        ),
    ).astype(jnp.int32)
    mode_fraction = jnp.mean(controlled.astype(psds.dtype))
    nyquist = 0.5 / reference.spacings
    evidence = StatisticalAOSamplingEvidence(
        frequency_cell_area,
        plan.control_cutoff,
        mode_fraction,
        nyquist,
        finite,
        nonnegative,
        budget_closed,
        valid,
        status,
    )
    return PreparedStatisticalResidualAO(
        plan,
        atmosphere,
        frequencies,
        atmospheric_psd,
        fitting_psd,
        servo_lag_psd,
        measurement_psd,
        aliasing_psd,
        total_residual_psd,
        controlled,
        budget,
        evidence,
    )


def long_exposure_otf(
    diffraction_limited: NormalizedTransferFunction,
    residual: PreparedStatisticalResidualAO,
    pupil_separation_scale: ArrayLike,
    /,
) -> LongExposureOTFResult:
    """Apply the stationary Gaussian residual-phase OTF on aligned sampling."""
    if not isinstance(diffraction_limited, NormalizedTransferFunction) or not isinstance(
        residual, PreparedStatisticalResidualAO
    ):
        raise TypeError("Expected a normalized transfer function and residual AO PSD.")
    supplied_scale = jnp.asarray(pupil_separation_scale)
    if supplied_scale.shape != ():
        raise ValueError("pupil_separation_scale must be scalar.")
    if jnp.iscomplexobj(supplied_scale) or not jnp.issubdtype(
        supplied_scale.dtype, jnp.number
    ):
        raise TypeError("pupil_separation_scale must be real numeric data.")
    scale = supplied_scale.astype(residual.total_residual_psd.dtype)
    scale = eqx.error_if(
        scale,
        (~jnp.isfinite(scale)) | (scale <= 0.0),
        "pupil_separation_scale must be finite and positive.",
    )
    optical = diffraction_limited.optical_transfer_function
    if optical.shape != residual.total_residual_psd.shape:
        raise ValueError("Optical OTF and residual PSD supports must have equal shapes.")
    reference = residual.atmosphere.layers[0].screen
    requested_axes = tuple(axis * scale for axis in diffraction_limited.frequency_axes)
    residual_axes = tuple(
        jnp.fft.fftshift(
            jnp.fft.fftfreq(
                count,
                d=1.0 / (count * spacing),
            )
        )
        for count, spacing in zip(
            optical.shape,
            reference.spacings,
            strict=True,
        )
    )
    alignment_errors = jnp.stack(
        tuple(
            jnp.max(jnp.abs(requested - expected))
            / jnp.maximum(jnp.max(jnp.abs(expected)), spacing)
            for requested, expected, spacing in zip(
                requested_axes,
                residual_axes,
                reference.spacings,
                strict=True,
            )
        )
    )
    relative_alignment_error = jnp.max(alignment_errors)
    aligned = (
        relative_alignment_error <= 256.0 * jnp.finfo(relative_alignment_error.dtype).eps
    )
    sampling = LongExposureSamplingEvidence(
        scale,
        requested_axes,
        residual_axes,
        relative_alignment_error,
        aligned,
    )
    domain_area = reference.lengths[0] * reference.lengths[1]
    sample_count = optical.shape[0] * optical.shape[1]
    covariance = (
        jnp.fft.ifft2(residual.total_residual_psd).real * sample_count / domain_area
    )
    structure = 2.0 * jnp.maximum(covariance[0, 0] - covariance, 0.0)
    centered_structure = jnp.fft.fftshift(structure)
    atmospheric_candidate = jnp.exp(-0.5 * centered_structure)
    center = (optical.shape[0] // 2, optical.shape[1] // 2)
    nonzero_dc = jnp.abs(optical[center]) > 0.0
    safe_dc = jnp.where(nonzero_dc, optical[center], 1.0)
    combined_candidate = optical * atmospheric_candidate / safe_dc
    atmospheric_transfer = jnp.where(aligned, atmospheric_candidate, 0.0)
    combined = jnp.where(aligned, combined_candidate, 0.0)
    mtf = jnp.abs(combined)
    finite = (
        jnp.all(jnp.isfinite(combined_candidate))
        & jnp.all(jnp.isfinite(atmospheric_candidate))
        & jnp.all(jnp.isfinite(centered_structure))
    )
    compatible = (
        jnp.asarray(diffraction_limited.evidence.valid) & residual.valid & aligned
    )
    valid = finite & compatible & nonzero_dc
    status = jnp.where(
        ~finite,
        int(StatisticalAOStatus.NONFINITE),
        jnp.where(
            ~(compatible & nonzero_dc),
            int(StatisticalAOStatus.INCOMPATIBLE_SUPPORT),
            int(StatisticalAOStatus.SUCCESS),
        ),
    ).astype(jnp.int32)
    return LongExposureOTFResult(
        combined,
        mtf,
        atmospheric_transfer,
        centered_structure,
        residual.error_budget,
        sampling,
        finite,
        valid,
        status,
    )


__all__ = [
    "LongExposureOTFResult",
    "LongExposureSamplingEvidence",
    "PreparedStatisticalResidualAO",
    "ResidualAOErrorBudget",
    "StatisticalAOStatus",
    "StatisticalAOSamplingEvidence",
    "StatisticalResidualAOPlan",
    "long_exposure_otf",
    "prepare_statistical_residual_ao",
]
