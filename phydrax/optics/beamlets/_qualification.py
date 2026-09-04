#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..geometric._paraxial import _COORDINATE_CONVENTION, DifferentialRayMap


class NineRayQualificationStatus(IntEnum):
    SUCCESS = 0
    NONFINITE = 1
    INVALID_STENCIL = 2
    DIFFERENTIAL_MAP_INVALID = 3


class NineRayTraceSamples(StrictModule, NonTrainableState):
    """Chief and centered ± perturbations in the four canonical ray coordinates.

    Ray zero is the chief. Rays ``1 + 2*i`` and ``2 + 2*i`` are respectively
    the positive and negative perturbations of canonical coordinate ``i``.
    """

    input_phase_space: Array
    output_phase_space: Array
    perturbation_steps: Array
    valid: Array

    def __init__(
        self,
        input_phase_space: ArrayLike,
        output_phase_space: ArrayLike,
        perturbation_steps: ArrayLike,
        valid: ArrayLike = True,
        /,
    ):
        inputs = jnp.asarray(input_phase_space)
        outputs = jnp.asarray(output_phase_space)
        steps = jnp.asarray(perturbation_steps)
        if inputs.shape != outputs.shape or inputs.shape[-2:] != (9, 4):
            raise ValueError("Nine-ray phase-space arrays must have shape (..., 9, 4).")
        if steps.shape != inputs.shape[:-2] + (4,):
            raise ValueError("perturbation_steps must have shape (..., 4).")
        valid_ = jnp.broadcast_to(jnp.asarray(valid, dtype=bool), inputs.shape[:-2])
        self.input_phase_space = inputs
        self.output_phase_space = outputs
        self.perturbation_steps = steps
        self.valid = valid_


class NineRayQualification(StrictModule, NonTrainableState):
    estimated_jacobian: Array
    input_stencil_error: Array
    central_output_error: Array
    absolute_jacobian_error: Array
    relative_jacobian_error: Array
    observed_order: Array
    finite: Array
    valid: Array
    status: Array


def _estimated_jacobian(samples: NineRayTraceSamples, /) -> Array:
    columns = tuple(
        (
            samples.output_phase_space[..., 1 + 2 * axis, :]
            - samples.output_phase_space[..., 2 + 2 * axis, :]
        )
        / (
            2.0
            * jnp.where(
                samples.perturbation_steps[..., axis, None] > 0.0,
                samples.perturbation_steps[..., axis, None],
                1.0,
            )
        )
        for axis in range(4)
    )
    return jnp.stack(columns, axis=-1)


def _input_stencil_error(samples: NineRayTraceSamples, reference: Array, /) -> Array:
    expected = jnp.broadcast_to(
        reference[..., None, :],
        samples.input_phase_space.shape,
    )
    for axis in range(4):
        step = samples.perturbation_steps[..., axis]
        expected = expected.at[..., 1 + 2 * axis, axis].add(step)
        expected = expected.at[..., 2 + 2 * axis, axis].add(-step)
    difference = samples.input_phase_space - expected
    return jnp.sqrt(jnp.sum(difference * difference, axis=(-2, -1)))


def qualify_nine_ray_differential_map(
    differential_map: DifferentialRayMap,
    samples: NineRayTraceSamples,
    /,
    *,
    refined_samples: NineRayTraceSamples | None = None,
    stencil_tolerance: float = 1e-10,
) -> NineRayQualification:
    """Qualify one differential map using exactly nine traced rays.

    An optional second nine-ray stencil reports the centered-difference
    convergence order. It does not participate in beamlet execution.
    """
    if not isinstance(differential_map, DifferentialRayMap):
        raise TypeError("differential_map must be a DifferentialRayMap.")
    if differential_map.coordinate_convention != _COORDINATE_CONVENTION:
        raise ValueError("Differential map uses an incompatible coordinate convention.")
    if not isinstance(samples, NineRayTraceSamples):
        raise TypeError("samples must be NineRayTraceSamples.")
    jacobian = jnp.asarray(differential_map.jacobian)
    batch_shape = samples.input_phase_space.shape[:-2]
    if jacobian.shape != batch_shape + (4, 4):
        raise ValueError("Differential map and nine-ray batch shapes do not match.")
    estimated = _estimated_jacobian(samples)
    difference = estimated - jacobian
    absolute_error = jnp.sqrt(jnp.sum(difference * difference, axis=(-2, -1)))
    reference_norm = jnp.sqrt(jnp.sum(jacobian * jacobian, axis=(-2, -1)))
    relative_error = absolute_error / jnp.maximum(reference_norm, 1.0)
    input_error = _input_stencil_error(
        samples,
        jnp.asarray(differential_map.input_reference),
    )
    central_difference = samples.output_phase_space[..., 0, :] - jnp.asarray(
        differential_map.output_reference
    )
    central_error = jnp.sqrt(jnp.sum(central_difference * central_difference, axis=-1))
    observed_order = jnp.full(absolute_error.shape, jnp.nan, dtype=absolute_error.dtype)
    refined_valid = jnp.ones(absolute_error.shape, dtype=bool)
    refined_finite = jnp.ones(absolute_error.shape, dtype=bool)
    if refined_samples is not None:
        if not isinstance(refined_samples, NineRayTraceSamples):
            raise TypeError("refined_samples must be NineRayTraceSamples or None.")
        if refined_samples.input_phase_space.shape[:-2] != batch_shape:
            raise ValueError("Refined nine-ray samples have an incompatible batch shape.")
        refined = _estimated_jacobian(refined_samples)
        refined_error = jnp.sqrt(jnp.sum((refined - jacobian) ** 2, axis=(-2, -1)))
        refined_input_error = _input_stencil_error(
            refined_samples,
            jnp.asarray(differential_map.input_reference),
        )
        refined_central_difference = refined_samples.output_phase_space[
            ..., 0, :
        ] - jnp.asarray(differential_map.output_reference)
        refined_central_error = jnp.sqrt(jnp.sum(refined_central_difference**2, axis=-1))
        coarse_scale = jnp.max(jnp.abs(samples.perturbation_steps), axis=-1)
        refined_scale = jnp.max(jnp.abs(refined_samples.perturbation_steps), axis=-1)
        scale_ratio = coarse_scale / jnp.where(refined_scale > 0.0, refined_scale, 1.0)
        safe_refined_error = jnp.maximum(
            refined_error,
            jnp.finfo(refined_error.dtype).tiny,
        )
        observed_order = jnp.log(
            jnp.maximum(absolute_error, safe_refined_error) / safe_refined_error
        ) / jnp.log(jnp.where(scale_ratio > 1.0, scale_ratio, 2.0))
        refined_finite = (
            jnp.all(jnp.isfinite(refined_samples.input_phase_space), axis=(-2, -1))
            & jnp.all(jnp.isfinite(refined_samples.output_phase_space), axis=(-2, -1))
            & jnp.all(jnp.isfinite(refined_samples.perturbation_steps), axis=-1)
            & jnp.isfinite(refined_error)
            & jnp.isfinite(observed_order)
            & jnp.isfinite(refined_input_error)
            & jnp.isfinite(refined_central_error)
        )
        refined_valid = (
            refined_samples.valid
            & refined_finite
            & jnp.all(refined_samples.perturbation_steps > 0.0, axis=-1)
            & (scale_ratio > 1.0)
            & (refined_input_error <= stencil_tolerance)
            & (refined_central_error <= stencil_tolerance)
        )
    positive_steps = jnp.all(samples.perturbation_steps > 0.0, axis=-1)
    finite = (
        jnp.all(jnp.isfinite(samples.input_phase_space), axis=(-2, -1))
        & jnp.all(jnp.isfinite(samples.output_phase_space), axis=(-2, -1))
        & jnp.all(jnp.isfinite(samples.perturbation_steps), axis=-1)
        & jnp.all(jnp.isfinite(estimated), axis=(-2, -1))
        & jnp.isfinite(input_error)
        & jnp.isfinite(central_error)
        & jnp.isfinite(absolute_error)
        & jnp.isfinite(relative_error)
        & refined_finite
    )
    stencil_valid = (
        samples.valid
        & refined_valid
        & positive_steps
        & (input_error <= stencil_tolerance)
        & (central_error <= stencil_tolerance)
    )
    map_valid = jnp.asarray(differential_map.valid, dtype=bool)
    valid = finite & stencil_valid & map_valid
    status = jnp.where(
        ~finite,
        int(NineRayQualificationStatus.NONFINITE),
        jnp.where(
            ~stencil_valid,
            int(NineRayQualificationStatus.INVALID_STENCIL),
            jnp.where(
                ~map_valid,
                int(NineRayQualificationStatus.DIFFERENTIAL_MAP_INVALID),
                int(NineRayQualificationStatus.SUCCESS),
            ),
        ),
    ).astype(jnp.int32)
    return NineRayQualification(
        estimated,
        input_error,
        central_error,
        absolute_error,
        relative_error,
        observed_order,
        finite,
        valid,
        status,
    )


__all__ = [
    "NineRayQualification",
    "NineRayQualificationStatus",
    "NineRayTraceSamples",
    "qualify_nine_ray_differential_map",
]
