#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._fields import IntensityPlane, PlaneFieldSpace, ScalarPlaneField


class ImagingStatus(IntEnum):
    SUCCESS = 0
    NONFINITE = 1
    ZERO_POWER = 2
    INSUFFICIENT_SAMPLING = 3
    INCOMPATIBLE_SUPPORT = 4


class FraunhoferImagingPlan(StrictModule, NonTrainableState):
    """Fixed input/output support for a scalar Fraunhofer transform."""

    input_space: PlaneFieldSpace
    output_space: PlaneFieldSpace
    focal_length: Array
    medium_wavenumber: Array
    pupil_diameter: Array

    def __init__(
        self,
        input_space: PlaneFieldSpace,
        output_space: PlaneFieldSpace,
        focal_length: ArrayLike,
        medium_wavenumber: ArrayLike,
        pupil_diameter: ArrayLike,
        /,
    ):
        if not isinstance(input_space, PlaneFieldSpace) or not isinstance(
            output_space, PlaneFieldSpace
        ):
            raise TypeError("Fraunhofer supports must be PlaneFieldSpace values.")
        if (
            input_space.topology != "finite-window"
            or output_space.topology != "finite-window"
        ):
            raise ValueError("Fraunhofer imaging requires two finite-window supports.")
        input_basis = np.asarray(input_space.transverse_basis)
        output_basis = np.asarray(output_space.transverse_basis)
        if not np.allclose(input_basis, output_basis, rtol=1e-10, atol=1e-12):
            raise ValueError(
                "Fraunhofer input and output transverse frames must be aligned."
            )
        focal = jnp.asarray(focal_length, dtype=float)
        wavenumber = jnp.asarray(medium_wavenumber)
        diameter = jnp.asarray(pupil_diameter, dtype=float)
        if focal.shape != () or wavenumber.shape != () or diameter.shape != ():
            raise ValueError(
                "focal_length, medium_wavenumber, and pupil_diameter are scalars."
            )
        if not jnp.issubdtype(wavenumber.dtype, jnp.complexfloating):
            wavenumber = wavenumber.astype(jnp.result_type(wavenumber.dtype, 1j))
        focal = eqx.error_if(
            focal,
            (~jnp.isfinite(focal)) | (focal <= 0.0),
            "focal_length must be finite and positive.",
        )
        diameter = eqx.error_if(
            diameter,
            (~jnp.isfinite(diameter)) | (diameter <= 0.0),
            "pupil_diameter must be finite and positive.",
        )
        wavenumber = eqx.error_if(
            wavenumber,
            (~jnp.isfinite(wavenumber)) | (jnp.real(wavenumber) <= 0.0),
            "medium_wavenumber must be finite with positive real part.",
        )
        self.input_space = input_space
        self.output_space = output_space
        self.focal_length = focal
        self.medium_wavenumber = wavenumber
        self.pupil_diameter = diameter

    def prepare(self, /) -> "PreparedFraunhoferImaging":
        input_axes = self.input_space.coordinate_axes
        output_axes = self.output_space.coordinate_axes
        input_measures = tuple(
            axis.measure(axis.primary_entity)
            for axis in self.input_space.grid.structured_axes
        )
        real_wavenumber = jnp.real(self.medium_wavenumber)
        scale = real_wavenumber / self.focal_length
        kernels = tuple(
            jnp.exp(-1j * scale * output_axis[:, None] * input_axis[None, :])
            * measure[None, :]
            for input_axis, output_axis, measure in zip(
                input_axes, output_axes, input_measures, strict=True
            )
        )
        output_spacings = jnp.stack(
            tuple(jnp.mean(jnp.diff(axis)) for axis in output_axes)
        )
        input_spacings = jnp.stack(tuple(jnp.mean(jnp.diff(axis)) for axis in input_axes))
        wavelength = 2.0 * jnp.pi / real_wavenumber
        airy_radius = 1.22 * wavelength * self.focal_length / self.pupil_diameter
        samples_per_airy_radius = airy_radius / jnp.max(output_spacings)
        frequency_steps = 1.0 / (
            jnp.asarray(self.input_space.shape, dtype=input_spacings.dtype)
            * input_spacings
        )
        output_half_widths = 0.5 * jnp.stack(
            tuple(axis[-1] - axis[0] for axis in output_axes)
        )
        airy_radii_covered = jnp.min(output_half_widths) / airy_radius
        maximum_unaliased_output = (
            jnp.pi * self.focal_length / (real_wavenumber * input_spacings)
        )
        maximum_requested_output = jnp.stack(
            tuple(jnp.max(jnp.abs(axis)) for axis in output_axes)
        )
        unaliased = jnp.all(maximum_requested_output <= maximum_unaliased_output)
        adequate = (
            (samples_per_airy_radius >= 2.0) & (airy_radii_covered >= 1.0) & unaliased
        )
        evidence = FraunhoferSamplingEvidence(
            input_spacings,
            output_spacings,
            frequency_steps,
            airy_radius,
            samples_per_airy_radius,
            output_half_widths,
            airy_radii_covered,
            maximum_unaliased_output,
            unaliased,
            adequate,
        )
        return PreparedFraunhoferImaging(self, kernels[0], kernels[1], evidence)


class FraunhoferSamplingEvidence(StrictModule, NonTrainableState):
    input_spacings: Array
    output_spacings: Array
    pupil_frequency_steps: Array
    airy_radius: Array
    samples_per_airy_radius: Array
    output_half_widths: Array
    airy_radii_covered: Array
    maximum_unaliased_output_coordinates: Array
    unaliased: Array
    adequate: Array


class PreparedFraunhoferImaging(StrictModule, NonTrainableState):
    plan: FraunhoferImagingPlan
    axis0_kernel: Array
    axis1_kernel: Array
    sampling: FraunhoferSamplingEvidence

    def execute(self, pupil_field: ScalarPlaneField, /) -> "FraunhoferPSFResult":
        return fraunhofer_psf(self, pupil_field)


class FraunhoferPSFResult(StrictModule):
    plane: IntensityPlane
    raw_intensity: Array
    input_power: Array
    captured_output_power: Array
    sampling: FraunhoferSamplingEvidence
    finite: Array
    valid: Array
    status: Array

    @property
    def successful(self) -> Array:
        return self.valid


class TransferFunctionEvidence(StrictModule, NonTrainableState):
    dc_value: Array
    hermitian_error: Array
    uniform_spacing_error: Array
    nyquist_frequencies: Array
    finite: Array
    valid: Array
    status: Array


class NormalizedTransferFunction(StrictModule):
    optical_transfer_function: Array
    modulation_transfer_function: Array
    frequency_axes: tuple[Array, Array]
    evidence: TransferFunctionEvidence


class StrehlSamplingEvidence(StrictModule, NonTrainableState):
    sample_spacings: Array
    uniform_spacing_error: Array
    same_support: Array
    same_angular_frequency: Array
    adequate: Array
    support_shape: tuple[int, int] = eqx.field(static=True)


class StrehlResult(StrictModule, NonTrainableState):
    ratio: Array
    aberrated_peak: Array
    reference_peak: Array
    aberrated_power: Array
    reference_power: Array
    sampling: StrehlSamplingEvidence
    finite: Array
    valid: Array
    status: Array


def fraunhofer_psf(
    prepared: PreparedFraunhoferImaging,
    pupil_field: ScalarPlaneField,
    /,
) -> FraunhoferPSFResult:
    """Evaluate the prepared physical DFT and return a unit-power PSF density."""
    if not isinstance(prepared, PreparedFraunhoferImaging) or not isinstance(
        pupil_field, ScalarPlaneField
    ):
        raise TypeError("Expected prepared Fraunhofer imaging and a scalar field.")
    plan = prepared.plan
    if pupil_field.space.space_id != plan.input_space.space_id:
        raise ValueError("Pupil field does not belong to the prepared input support.")
    transformed = contract(
        "ai,ij,bj->ab",
        prepared.axis0_kernel,
        pupil_field.values,
        prepared.axis1_kernel,
    )
    prefactor = (
        plan.medium_wavenumber
        * jnp.exp(1j * plan.medium_wavenumber * plan.focal_length)
        / (2.0 * jnp.pi * plan.focal_length)
    )
    raw_intensity = jnp.abs(prefactor * transformed) ** 2
    input_power = jnp.sum(
        jnp.abs(pupil_field.values) ** 2 * plan.input_space.area_weights
    )
    captured_power = jnp.sum(raw_intensity * plan.output_space.area_weights)
    safe_power = jnp.where(captured_power > 0.0, captured_power, 1.0)
    finite = (
        jnp.all(jnp.isfinite(raw_intensity))
        & jnp.isfinite(input_power)
        & jnp.isfinite(captured_power)
    )
    normalized = jnp.where(
        finite & (captured_power > 0.0) & jnp.isfinite(raw_intensity),
        raw_intensity / safe_power,
        0.0,
    )
    nonzero = (input_power > 0.0) & (captured_power > 0.0)
    adequate = prepared.sampling.adequate
    valid = finite & nonzero & adequate
    status = jnp.where(
        ~finite,
        int(ImagingStatus.NONFINITE),
        jnp.where(
            ~nonzero,
            int(ImagingStatus.ZERO_POWER),
            jnp.where(
                ~adequate,
                int(ImagingStatus.INSUFFICIENT_SAMPLING),
                int(ImagingStatus.SUCCESS),
            ),
        ),
    ).astype(jnp.int32)
    plane = IntensityPlane(
        plan.output_space,
        normalized,
        pupil_field.angular_frequency,
        pupil_field.longitudinal_coordinate + plan.focal_length,
    )
    return FraunhoferPSFResult(
        plane,
        raw_intensity,
        input_power,
        captured_power,
        prepared.sampling,
        finite,
        valid,
        status,
    )


def _axis_spacing_and_error(coordinates: Array, /) -> tuple[Array, Array]:
    differences = jnp.diff(coordinates)
    spacing = jnp.mean(differences)
    error = jnp.max(jnp.abs(differences - spacing)) / jnp.maximum(
        jnp.abs(spacing), jnp.finfo(coordinates.dtype).tiny
    )
    return spacing, error


def normalized_otf_mtf(intensity: IntensityPlane, /) -> NormalizedTransferFunction:
    """Return a centered normalized OTF and its nonnegative MTF magnitude."""
    if not isinstance(intensity, IntensityPlane):
        raise TypeError("intensity must be an IntensityPlane.")
    spacing_records = tuple(
        _axis_spacing_and_error(axis) for axis in intensity.space.coordinate_axes
    )
    spacings = jnp.stack(tuple(record[0] for record in spacing_records))
    spacing_error = jnp.max(jnp.stack(tuple(record[1] for record in spacing_records)))
    weighted = intensity.values * intensity.space.area_weights
    unshifted = jnp.fft.fft2(weighted)
    dc = unshifted[0, 0]
    safe_dc = jnp.where(jnp.abs(dc) > 0.0, dc, 1.0)
    normalized = unshifted / safe_dc
    first_indices = (-jnp.arange(normalized.shape[0])) % normalized.shape[0]
    second_indices = (-jnp.arange(normalized.shape[1])) % normalized.shape[1]
    conjugate_partner = jnp.conj(
        normalized[first_indices[:, None], second_indices[None, :]]
    )
    hermitian_error = jnp.max(jnp.abs(normalized - conjugate_partner))
    centered = jnp.fft.fftshift(normalized)
    mtf = jnp.abs(centered)
    frequency_axes = tuple(
        jnp.fft.fftshift(jnp.fft.fftfreq(count, d=spacing))
        for count, spacing in zip(intensity.space.shape, spacings, strict=True)
    )
    nyquist = 0.5 / spacings
    finite = (
        jnp.all(jnp.isfinite(centered))
        & jnp.all(jnp.isfinite(mtf))
        & jnp.isfinite(hermitian_error)
    )
    uniform = spacing_error <= 64.0 * jnp.finfo(spacings.dtype).eps
    valid = finite & (jnp.abs(dc) > 0.0) & uniform
    status = jnp.where(
        ~finite,
        int(ImagingStatus.NONFINITE),
        jnp.where(
            jnp.abs(dc) == 0.0,
            int(ImagingStatus.ZERO_POWER),
            jnp.where(
                ~uniform,
                int(ImagingStatus.INCOMPATIBLE_SUPPORT),
                int(ImagingStatus.SUCCESS),
            ),
        ),
    ).astype(jnp.int32)
    evidence = TransferFunctionEvidence(
        dc,
        hermitian_error,
        spacing_error,
        nyquist,
        finite,
        valid,
        status,
    )
    return NormalizedTransferFunction(centered, mtf, frequency_axes, evidence)


def strehl_ratio(
    aberrated: IntensityPlane,
    reference: IntensityPlane,
    /,
) -> StrehlResult:
    """Compare equal-power PSF peaks on the same fixed image support."""
    if not isinstance(aberrated, IntensityPlane) or not isinstance(
        reference, IntensityPlane
    ):
        raise TypeError("Strehl comparison requires two intensity planes.")
    compatible = aberrated.space.space_id == reference.space.space_id
    if not compatible:
        raise ValueError("Strehl planes must use the same fixed support.")
    spacing_records = tuple(
        _axis_spacing_and_error(axis) for axis in reference.space.coordinate_axes
    )
    spacings = jnp.stack(tuple(record[0] for record in spacing_records))
    spacing_error = jnp.max(jnp.stack(tuple(record[1] for record in spacing_records)))
    frequency_consistent = aberrated.angular_frequency == reference.angular_frequency
    uniform = spacing_error <= 64.0 * jnp.finfo(spacings.dtype).eps
    sampling_adequate = frequency_consistent & uniform
    sampling = StrehlSamplingEvidence(
        spacings,
        spacing_error,
        jnp.asarray(compatible),
        frequency_consistent,
        sampling_adequate,
        support_shape=reference.space.shape,
    )
    weights = reference.space.area_weights
    aberrated_power = jnp.sum(aberrated.values * weights)
    reference_power = jnp.sum(reference.values * weights)
    safe_aberrated_power = jnp.where(aberrated_power > 0.0, aberrated_power, 1.0)
    safe_reference_power = jnp.where(reference_power > 0.0, reference_power, 1.0)
    aberrated_peak = jnp.max(aberrated.values / safe_aberrated_power)
    reference_peak = jnp.max(reference.values / safe_reference_power)
    ratio = aberrated_peak / jnp.where(reference_peak > 0.0, reference_peak, 1.0)
    finite = (
        jnp.isfinite(ratio)
        & jnp.isfinite(aberrated_power)
        & jnp.isfinite(reference_power)
    )
    nonzero = (aberrated_power > 0.0) & (reference_power > 0.0) & (reference_peak > 0.0)
    valid = finite & nonzero & sampling_adequate
    status = jnp.where(
        ~finite,
        int(ImagingStatus.NONFINITE),
        jnp.where(
            ~nonzero,
            int(ImagingStatus.ZERO_POWER),
            jnp.where(
                ~sampling_adequate,
                int(ImagingStatus.INCOMPATIBLE_SUPPORT),
                int(ImagingStatus.SUCCESS),
            ),
        ),
    ).astype(jnp.int32)
    return StrehlResult(
        ratio,
        aberrated_peak,
        reference_peak,
        aberrated_power,
        reference_power,
        sampling,
        finite,
        valid,
        status,
    )


__all__ = [
    "FraunhoferImagingPlan",
    "FraunhoferPSFResult",
    "FraunhoferSamplingEvidence",
    "ImagingStatus",
    "NormalizedTransferFunction",
    "PreparedFraunhoferImaging",
    "StrehlResult",
    "StrehlSamplingEvidence",
    "TransferFunctionEvidence",
    "fraunhofer_psf",
    "normalized_otf_mtf",
    "strehl_ratio",
]
