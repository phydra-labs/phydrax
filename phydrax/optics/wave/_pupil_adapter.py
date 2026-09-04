#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ..geometric._sequential import SequentialOpticsResult
from ._fields import PlaneFieldSpace, ScalarPlaneField


class PupilFieldAdapterStatus(IntEnum):
    """Terminal qualification status for sequential pupil-to-field conversion."""

    SUCCESS = 0
    SEQUENTIAL_TRACE_REJECTED = 1
    UNSUPPORTED_TOPOLOGY = 2
    SAMPLE_COORDINATE_MISMATCH = 3
    CAUSTIC_OR_FOLD = 4
    NONFINITE_INPUT = 5


class PupilFieldAdapterEvidence(StrictModule):
    """One-to-one sample and noncaustic evidence for a pupil conversion."""

    maximum_pupil_coordinate_residual: Array
    maximum_output_coordinate_residual: Array
    minimum_signed_jacobian_ratio: Array
    maximum_signed_jacobian_ratio: Array
    coordinate_tolerance: Array
    jacobian_tolerance: Array
    sequential_successful: Array
    topology_supported: Array
    samples_one_to_one: Array
    noncaustic: Array
    finite: Array
    accepted: Array
    status: Array
    ray_count: int = eqx.field(static=True)
    pupil_space_id: str = eqx.field(static=True)
    output_space_id: str = eqx.field(static=True)
    sequential_producer_id: str = eqx.field(static=True)


class PupilToScalarFieldResult(StrictModule):
    """Scalar plane field and typed rejection/evidence from ray lowering."""

    field: ScalarPlaneField
    evidence: PupilFieldAdapterEvidence


def _nonnegative_scalar(value: ArrayLike, name: str, /, *, positive: bool) -> Array:
    scalar = jnp.asarray(value)
    if scalar.ndim != 0 or not jnp.issubdtype(scalar.dtype, jnp.floating):
        raise TypeError(f"{name} must be a real scalar.")
    lower_failure = scalar <= 0 if positive else scalar < 0
    return eqx.error_if(
        scalar,
        (~jnp.isfinite(scalar)) | lower_failure,
        f"{name} must be finite and {'positive' if positive else 'nonnegative'}.",
    )


def _cell_determinants(coordinates: Array, /) -> Array:
    lower_left = coordinates[:-1, :-1]
    upper_left = coordinates[1:, :-1]
    lower_right = coordinates[:-1, 1:]
    upper_right = coordinates[1:, 1:]
    along_axis_0 = 0.5 * (upper_left - lower_left + upper_right - lower_right)
    along_axis_1 = 0.5 * (lower_right - lower_left + upper_right - upper_left)
    return (
        along_axis_0[..., 0] * along_axis_1[..., 1]
        - along_axis_0[..., 1] * along_axis_1[..., 0]
    )


def _cell_to_node_average(cell_values: Array, /) -> Array:
    padding = (((0, 1), (0, 1)), ((1, 0), (0, 1)), ((0, 1), (1, 0)), ((1, 0), (1, 0)))
    total = sum(jnp.pad(cell_values, pad) for pad in padding)
    ones = jnp.ones_like(cell_values)
    count = sum(jnp.pad(ones, pad) for pad in padding)
    return total / count


def sequential_pupil_to_scalar_field(
    result: SequentialOpticsResult,
    pupil_space: PlaneFieldSpace,
    output_space: PlaneFieldSpace,
    pupil_ray_origins: ArrayLike,
    pupil_values: ArrayLike,
    angular_frequency: ArrayLike,
    longitudinal_coordinate: ArrayLike,
    reference_wave_speed: ArrayLike,
    /,
    *,
    coordinate_tolerance: ArrayLike = 1.0e-7,
    jacobian_tolerance: ArrayLike = 1.0e-8,
) -> PupilToScalarFieldResult:
    """Lower an ordered, one-to-one, noncaustic ray pupil to a scalar field.

    The input origins explicitly bind the sequential ray order to pupil samples.
    The final ray origins must already coincide, in that order, with the output
    field samples; this adapter performs no interpolation or caustic synthesis.
    Amplitude is transported by the square root of the pupil/output area
    Jacobian, and phase uses exp(i omega optical_path/reference_wave_speed).
    """
    if not isinstance(result, SequentialOpticsResult):
        raise TypeError("result must be a SequentialOpticsResult.")
    if not isinstance(pupil_space, PlaneFieldSpace) or not isinstance(
        output_space, PlaneFieldSpace
    ):
        raise TypeError("pupil_space and output_space must be PlaneFieldSpace values.")
    if pupil_space.shape != output_space.shape:
        raise ValueError("Pupil and output spaces must have equal sample shape.")
    if any(size < 2 for size in pupil_space.shape):
        raise ValueError("Pupil conversion requires at least two samples per axis.")
    shape = pupil_space.shape
    ray_count = pupil_space.size
    origins = jnp.asarray(pupil_ray_origins)
    if origins.shape == (ray_count, 3):
        origins = origins.reshape(shape + (3,))
    if origins.shape != shape + (3,):
        raise ValueError(
            f"pupil_ray_origins must have shape {(ray_count, 3)} or {shape + (3,)}."
        )
    amplitudes = jnp.asarray(pupil_values)
    if amplitudes.shape != shape or not jnp.issubdtype(amplitudes.dtype, jnp.number):
        raise TypeError(f"pupil_values must be numeric with shape {shape}.")
    final_origins = result.rays.origins
    if final_origins.shape == (ray_count, 3):
        final_origins = final_origins.reshape(shape + (3,))
    if final_origins.shape != shape + (3,):
        raise ValueError("Sequential final ray origins do not match the pupil ray count.")
    optical_paths = result.rays.optical_path_lengths
    if optical_paths.shape == (ray_count,):
        optical_paths = optical_paths.reshape(shape)
    if optical_paths.shape != shape:
        raise ValueError("Sequential optical path lengths do not match the pupil shape.")
    valid = result.valid
    if valid.shape == (ray_count,):
        valid = valid.reshape(shape)
    if valid.shape != shape:
        raise ValueError("Sequential validity lanes do not match the pupil shape.")

    omega = _nonnegative_scalar(angular_frequency, "angular_frequency", positive=True)
    speed = _nonnegative_scalar(
        reference_wave_speed, "reference_wave_speed", positive=True
    )
    coordinate_tol = _nonnegative_scalar(
        coordinate_tolerance, "coordinate_tolerance", positive=False
    )
    jacobian_tol = _nonnegative_scalar(
        jacobian_tolerance, "jacobian_tolerance", positive=False
    )
    pupil_local = pupil_space.frame.inverse_apply(origins)
    output_local = output_space.frame.inverse_apply(final_origins)
    pupil_residual = jnp.max(jnp.abs(pupil_local - pupil_space.local_points))
    output_residual = jnp.max(jnp.abs(output_local - output_space.local_points))
    input_determinants = _cell_determinants(pupil_local[..., :2])
    output_determinants = _cell_determinants(output_local[..., :2])
    safe_input = jnp.where(input_determinants != 0, input_determinants, jnp.nan)
    signed_ratio = output_determinants / safe_input
    minimum_ratio = jnp.min(signed_ratio)
    maximum_ratio = jnp.max(signed_ratio)

    topology_supported = jnp.asarray(
        pupil_space.topology == "finite-window"
        and output_space.topology == "finite-window"
    )
    samples_one_to_one = (
        jnp.isfinite(pupil_residual)
        & jnp.isfinite(output_residual)
        & (pupil_residual <= coordinate_tol)
        & (output_residual <= coordinate_tol)
    )
    noncaustic = jnp.all(jnp.isfinite(signed_ratio)) & jnp.all(
        signed_ratio > jacobian_tol
    )
    sequential_successful = jnp.asarray(result.successful) & jnp.all(valid)
    finite = (
        jnp.all(jnp.isfinite(origins))
        & jnp.all(jnp.isfinite(final_origins))
        & jnp.all(jnp.isfinite(jnp.real(amplitudes)))
        & jnp.all(jnp.isfinite(jnp.imag(amplitudes)))
        & jnp.all(jnp.isfinite(optical_paths))
    )
    accepted = (
        sequential_successful
        & topology_supported
        & samples_one_to_one
        & noncaustic
        & finite
    )
    status = jnp.where(
        ~finite,
        int(PupilFieldAdapterStatus.NONFINITE_INPUT),
        jnp.where(
            ~sequential_successful,
            int(PupilFieldAdapterStatus.SEQUENTIAL_TRACE_REJECTED),
            jnp.where(
                ~topology_supported,
                int(PupilFieldAdapterStatus.UNSUPPORTED_TOPOLOGY),
                jnp.where(
                    ~samples_one_to_one,
                    int(PupilFieldAdapterStatus.SAMPLE_COORDINATE_MISMATCH),
                    jnp.where(
                        ~noncaustic,
                        int(PupilFieldAdapterStatus.CAUSTIC_OR_FOLD),
                        int(PupilFieldAdapterStatus.SUCCESS),
                    ),
                ),
            ),
        ),
    ).astype(jnp.int32)

    safe_ratio = jnp.where(signed_ratio > jacobian_tol, signed_ratio, 1.0)
    node_ratio = _cell_to_node_average(safe_ratio)
    phase = jnp.exp(1j * omega * optical_paths / speed)
    transported = amplitudes * phase / jnp.sqrt(node_ratio)
    complex_dtype = jnp.result_type(transported.dtype, jnp.complex64)
    rejected = jnp.asarray(jnp.nan + 1j * jnp.nan, dtype=complex_dtype)
    values = jnp.where(accepted, transported.astype(complex_dtype), rejected)
    field = ScalarPlaneField(
        output_space,
        values,
        omega,
        longitudinal_coordinate,
    )
    evidence = PupilFieldAdapterEvidence(
        maximum_pupil_coordinate_residual=pupil_residual,
        maximum_output_coordinate_residual=output_residual,
        minimum_signed_jacobian_ratio=minimum_ratio,
        maximum_signed_jacobian_ratio=maximum_ratio,
        coordinate_tolerance=coordinate_tol,
        jacobian_tolerance=jacobian_tol,
        sequential_successful=sequential_successful,
        topology_supported=topology_supported,
        samples_one_to_one=samples_one_to_one,
        noncaustic=noncaustic,
        finite=finite,
        accepted=accepted,
        status=status,
        ray_count=ray_count,
        pupil_space_id=pupil_space.space_id,
        output_space_id=output_space.space_id,
        sequential_producer_id=result.producer_id,
    )
    return PupilToScalarFieldResult(field=field, evidence=evidence)


__all__ = [
    "PupilFieldAdapterEvidence",
    "PupilFieldAdapterStatus",
    "PupilToScalarFieldResult",
    "sequential_pupil_to_scalar_field",
]
