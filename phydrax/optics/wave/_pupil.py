#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Sequence
from enum import IntEnum

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._fields import PlaneFieldSpace, ScalarPlaneField


class PupilSamplingStatus(IntEnum):
    SUCCESS = 0
    EMPTY_PUPIL = 1
    INSUFFICIENT_SAMPLING = 2
    NONFINITE = 3


class NollZernikeOPD(StrictModule):
    """Physical OPD coefficients on continuous unit-RMS Noll modes."""

    coefficients: Array
    pupil_radius: Array
    pupil_center: Array
    noll_indices: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        noll_indices: Sequence[int],
        coefficients: ArrayLike,
        pupil_radius: ArrayLike,
        /,
        *,
        pupil_center: ArrayLike = (0.0, 0.0),
    ):
        indices = tuple(int(index) for index in noll_indices)
        if not indices or any(index <= 0 for index in indices):
            raise ValueError("noll_indices must contain positive one-based indices.")
        if len(set(indices)) != len(indices):
            raise ValueError("noll_indices must be unique.")
        coefficients_ = jnp.asarray(coefficients)
        if not jnp.issubdtype(coefficients_.dtype, jnp.floating):
            coefficients_ = coefficients_.astype(float)
        radius = jnp.asarray(pupil_radius, dtype=coefficients_.dtype)
        center = jnp.asarray(pupil_center, dtype=coefficients_.dtype)
        if coefficients_.shape != (len(indices),):
            raise ValueError("coefficients must have one value per Noll index.")
        if radius.shape != () or center.shape != (2,):
            raise ValueError("pupil_radius must be scalar and pupil_center shape (2,).")
        coefficients_ = eqx.error_if(
            coefficients_,
            jnp.any(~jnp.isfinite(coefficients_)),
            "Zernike OPD coefficients must be finite.",
        )
        radius = eqx.error_if(
            radius,
            (~jnp.isfinite(radius)) | (radius <= 0.0),
            "pupil_radius must be finite and positive.",
        )
        center = eqx.error_if(
            center,
            jnp.any(~jnp.isfinite(center)),
            "pupil_center must be finite.",
        )
        self.coefficients = coefficients_
        self.pupil_radius = radius
        self.pupil_center = center
        self.noll_indices = indices


class PupilSamplingEvidence(StrictModule, NonTrainableState):
    pupil_sample_count: Array
    sampled_area: Array
    analytic_area: Array
    relative_area_error: Array
    samples_across_diameter: Array
    discrete_mode_rms: Array
    discrete_mode_means: Array
    finite: Array
    adequate: Array
    status: Array


class ZernikeOPDResult(StrictModule):
    space: PlaneFieldSpace
    opd: Array
    aperture: Array
    basis: Array
    evidence: PupilSamplingEvidence

    @property
    def valid(self) -> Array:
        return self.evidence.finite & (self.evidence.pupil_sample_count > 0)


def noll_to_radial_azimuthal(noll_index: int, /) -> tuple[int, int]:
    """Map a one-based Noll index to radial order and signed azimuthal order."""
    index = int(noll_index)
    if index <= 0:
        raise ValueError("Noll indices are one-based and must be positive.")
    radial = 0
    while index > (radial + 1) * (radial + 2) // 2:
        radial += 1
    row_start = radial * (radial + 1) // 2 + 1
    offset = index - row_start
    magnitudes: list[int] = []
    if radial % 2 == 0:
        magnitudes.append(0)
        start = 2
    else:
        start = 1
    for magnitude in range(start, radial + 1, 2):
        magnitudes.extend((magnitude, magnitude))
    magnitude = magnitudes[offset]
    azimuthal = 0 if magnitude == 0 else (magnitude if index % 2 == 0 else -magnitude)
    return radial, azimuthal


def _radial_zernike(radial: int, magnitude: int, radius: Array, /) -> Array:
    result = jnp.zeros_like(radius)
    half_sum = (radial + magnitude) // 2
    half_difference = (radial - magnitude) // 2
    for summation in range(half_difference + 1):
        coefficient = (
            (-1) ** summation
            * math.factorial(radial - summation)
            / (
                math.factorial(summation)
                * math.factorial(half_sum - summation)
                * math.factorial(half_difference - summation)
            )
        )
        result = result + coefficient * radius ** (radial - 2 * summation)
    return result


def noll_zernike(
    noll_index: int,
    transverse_coordinates: ArrayLike,
    /,
    *,
    pupil_radius: ArrayLike = 1.0,
    pupil_center: ArrayLike = (0.0, 0.0),
) -> Array:
    """Evaluate one continuous unit-RMS Noll mode, zero outside the pupil."""
    coordinates = jnp.asarray(transverse_coordinates)
    if coordinates.shape[-1:] != (2,):
        raise ValueError("transverse_coordinates must have shape (..., 2).")
    radius_scale = jnp.asarray(pupil_radius, dtype=coordinates.dtype)
    center = jnp.asarray(pupil_center, dtype=coordinates.dtype)
    if radius_scale.shape != () or center.shape != (2,):
        raise ValueError("pupil_radius must be scalar and pupil_center shape (2,).")
    normalized = (coordinates - center) / radius_scale
    radial_coordinate = jnp.sqrt(jnp.sum(normalized * normalized, axis=-1))
    radial_order, azimuthal_order = noll_to_radial_azimuthal(noll_index)
    magnitude = abs(azimuthal_order)
    radial = _radial_zernike(radial_order, magnitude, radial_coordinate)
    if azimuthal_order == 0:
        mode = math.sqrt(radial_order + 1.0) * radial
    else:
        safe_radius = jnp.where(radial_coordinate > 0.0, radial_coordinate, 1.0)
        unit_coordinate = (normalized[..., 0] + 1j * normalized[..., 1]) / safe_radius
        angular_factor = (
            jnp.real(unit_coordinate**magnitude)
            if azimuthal_order > 0
            else jnp.imag(unit_coordinate**magnitude)
        )
        mode = math.sqrt(2.0 * (radial_order + 1.0)) * radial * angular_factor
    return jnp.where(radial_coordinate <= 1.0, mode, 0.0)


def _minimum_axis_spacing(space: PlaneFieldSpace, /) -> Array:
    spacings = []
    for coordinates in space.coordinate_axes:
        differences = jnp.abs(jnp.diff(coordinates))
        spacings.append(
            jnp.min(differences) if differences.size else jnp.asarray(jnp.inf)
        )
    return jnp.max(jnp.stack(spacings))


def evaluate_noll_zernike_opd(
    space: PlaneFieldSpace,
    specification: NollZernikeOPD,
    /,
) -> ZernikeOPDResult:
    """Evaluate an immutable Noll OPD specification with sampling evidence."""
    if not isinstance(space, PlaneFieldSpace) or not isinstance(
        specification, NollZernikeOPD
    ):
        raise TypeError("Expected a PlaneFieldSpace and NollZernikeOPD.")
    coordinates = space.transverse_coordinates
    normalized = (coordinates - specification.pupil_center) / specification.pupil_radius
    aperture = jnp.sum(normalized * normalized, axis=-1) <= 1.0
    modes = jnp.stack(
        tuple(
            noll_zernike(
                index,
                coordinates,
                pupil_radius=specification.pupil_radius,
                pupil_center=specification.pupil_center,
            )
            for index in specification.noll_indices
        ),
        axis=0,
    )
    opd = contract("m,mij->ij", specification.coefficients, modes)
    opd = jnp.where(aperture, opd, 0.0)
    weights = jnp.where(aperture, space.area_weights, 0.0)
    sampled_area = jnp.sum(weights)
    safe_area = jnp.where(sampled_area > 0.0, sampled_area, 1.0)
    mode_means = jnp.sum(modes * weights[None, ...], axis=(-2, -1)) / safe_area
    mode_rms = jnp.sqrt(
        jnp.sum(modes * modes * weights[None, ...], axis=(-2, -1)) / safe_area
    )
    analytic_area = jnp.pi * specification.pupil_radius**2
    relative_area_error = jnp.abs(sampled_area - analytic_area) / analytic_area
    spacing = _minimum_axis_spacing(space)
    samples_across = 2.0 * specification.pupil_radius / spacing
    maximum_order = max(
        noll_to_radial_azimuthal(index)[0] for index in specification.noll_indices
    )
    sample_count = jnp.sum(aperture.astype(jnp.int32))
    finite = (
        jnp.all(jnp.isfinite(opd))
        & jnp.all(jnp.isfinite(mode_rms))
        & jnp.isfinite(sampled_area)
    )
    adequate = (
        finite
        & (sample_count >= len(specification.noll_indices))
        & (samples_across >= 2.0 * (maximum_order + 1))
    )
    status = jnp.where(
        ~finite,
        int(PupilSamplingStatus.NONFINITE),
        jnp.where(
            sample_count == 0,
            int(PupilSamplingStatus.EMPTY_PUPIL),
            jnp.where(
                ~adequate,
                int(PupilSamplingStatus.INSUFFICIENT_SAMPLING),
                int(PupilSamplingStatus.SUCCESS),
            ),
        ),
    ).astype(jnp.int32)
    evidence = PupilSamplingEvidence(
        sample_count,
        sampled_area,
        analytic_area,
        relative_area_error,
        samples_across,
        mode_rms,
        mode_means,
        finite,
        adequate,
        status,
    )
    return ZernikeOPDResult(space, opd, aperture, modes, evidence)


def apply_pupil_opd(
    field: ScalarPlaneField,
    opd: ZernikeOPDResult,
    medium_wavenumber: ArrayLike,
    /,
) -> ScalarPlaneField:
    """Apply a physical OPD and its circular aperture to a scalar field."""
    if not isinstance(field, ScalarPlaneField) or not isinstance(opd, ZernikeOPDResult):
        raise TypeError("Expected a ScalarPlaneField and ZernikeOPDResult.")
    if field.space.space_id != opd.space.space_id:
        raise ValueError("Field and OPD supports do not match.")
    wavenumber = jnp.asarray(medium_wavenumber)
    if wavenumber.shape != ():
        raise ValueError("medium_wavenumber must be scalar.")
    if not jnp.issubdtype(wavenumber.dtype, jnp.complexfloating):
        wavenumber = wavenumber.astype(jnp.result_type(wavenumber.dtype, 1j))
    wavenumber = eqx.error_if(
        wavenumber,
        (~jnp.isfinite(wavenumber)) | (jnp.abs(wavenumber) == 0.0),
        "medium_wavenumber must be finite and nonzero.",
    )
    values = jnp.where(
        opd.aperture,
        field.values * jnp.exp(1j * wavenumber * opd.opd),
        0.0,
    )
    return ScalarPlaneField(
        field.space,
        values,
        field.angular_frequency,
        field.longitudinal_coordinate,
    )


__all__ = [
    "NollZernikeOPD",
    "PupilSamplingEvidence",
    "PupilSamplingStatus",
    "ZernikeOPDResult",
    "apply_pupil_opd",
    "evaluate_noll_zernike_opd",
    "noll_to_radial_azimuthal",
    "noll_zernike",
]
