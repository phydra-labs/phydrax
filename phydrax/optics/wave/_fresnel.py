#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntFlag
from numbers import Integral

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._fields import PlaneFieldSpace, ScalarPlaneField, TangentialPlaneField


PlaneField = ScalarPlaneField | TangentialPlaneField


class FresnelPropagationStatus(IntFlag):
    """Fail-closed disposition of direct finite-plane Fresnel propagation."""

    SUCCESS = 0
    INVALID_DISTANCE = 1
    INVALID_WAVENUMBER = 2
    ZERO_DISTANCE_SPACE_MISMATCH = 4
    SAMPLING_LIMIT = 8
    PARAXIAL_LIMIT = 16
    POWER_LIMIT = 32
    NONFINITE = 64


class DirectFresnelPlan(StrictModule, NonTrainableState):
    """Static direct-quadrature policy between two explicit finite plane spaces.

    This is the separable direct Fresnel integral. It performs no FFT scaling,
    automatic output-grid selection, interpolation, or fallback to another
    propagator.
    """

    input_space: PlaneFieldSpace
    output_space: PlaneFieldSpace
    maximum_kernel_elements: int = eqx.field(static=True)
    maximum_workspace_bytes: int = eqx.field(static=True)
    maximum_sampling_phase_step: float = eqx.field(static=True)
    maximum_paraxial_angle: float = eqx.field(static=True)
    maximum_power_error: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        input_space: PlaneFieldSpace,
        output_space: PlaneFieldSpace,
        /,
        *,
        maximum_kernel_elements: int = 10_000_000,
        maximum_workspace_bytes: int = 512 * 1024 * 1024,
        maximum_sampling_phase_step: float = np.pi,
        maximum_paraxial_angle: float = 0.3,
        maximum_power_error: float = 5.0e-2,
    ):
        if not isinstance(input_space, PlaneFieldSpace) or not isinstance(
            output_space, PlaneFieldSpace
        ):
            raise TypeError(
                "input_space and output_space must be PlaneFieldSpace values."
            )
        if (
            input_space.topology != "finite-window"
            or output_space.topology != "finite-window"
        ):
            raise ValueError(
                "Direct Fresnel propagation requires two finite-window spaces."
            )
        if isinstance(maximum_kernel_elements, bool) or not isinstance(
            maximum_kernel_elements, Integral
        ):
            raise TypeError("maximum_kernel_elements must be an integer.")
        if isinstance(maximum_workspace_bytes, bool) or not isinstance(
            maximum_workspace_bytes, Integral
        ):
            raise TypeError("maximum_workspace_bytes must be an integer.")
        kernel_limit = int(maximum_kernel_elements)
        byte_limit = int(maximum_workspace_bytes)
        if kernel_limit <= 0 or byte_limit <= 0:
            raise ValueError("Fresnel resource limits must be strictly positive.")

        def positive(name: str, value: float) -> float:
            result = float(value)
            if not np.isfinite(result) or result <= 0.0:
                raise ValueError(f"{name} must be finite and strictly positive.")
            return result

        sampling = positive("maximum_sampling_phase_step", maximum_sampling_phase_step)
        paraxial = positive("maximum_paraxial_angle", maximum_paraxial_angle)
        power = float(maximum_power_error)
        if not np.isfinite(power) or not 0.0 <= power <= 1.0:
            raise ValueError("maximum_power_error must lie in [0, 1].")
        self.input_space = input_space
        self.output_space = output_space
        self.maximum_kernel_elements = kernel_limit
        self.maximum_workspace_bytes = byte_limit
        self.maximum_sampling_phase_step = sampling
        self.maximum_paraxial_angle = paraxial
        self.maximum_power_error = power
        self.plan_id = canonical_fingerprint(
            {
                "kind": "direct-fresnel-plan",
                "input_space": input_space.space_id,
                "output_space": output_space.space_id,
                "maximum_kernel_elements": kernel_limit,
                "maximum_workspace_bytes": byte_limit,
                "maximum_sampling_phase_step": sampling.hex(),
                "maximum_paraxial_angle": paraxial.hex(),
                "maximum_power_error": power.hex(),
            }
        )


class PreparedDirectFresnel(StrictModule, NonTrainableState):
    """Prepared coordinate geometry and resource accounting for direct Fresnel."""

    plan: DirectFresnelPlan
    input_axes: tuple[Array, Array]
    output_axes: tuple[Array, Array]
    input_weights: tuple[Array, Array]
    maximum_transverse_separation: Array
    identical_space: bool = eqx.field(static=True)
    kernel_elements: int = eqx.field(static=True)
    workspace_complex_elements_per_component: int = eqx.field(static=True)
    workspace_bytes_per_component: int = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def execute(
        self,
        field: PlaneField,
        distance: ArrayLike,
        medium_wavenumber: ArrayLike,
        /,
    ) -> FresnelPropagationResult:
        return propagate_direct_fresnel(self, field, distance, medium_wavenumber)


class FresnelPropagationEvidence(StrictModule):
    """Sampling, paraxiality, power, and resource evidence."""

    input_power: Array
    output_power: Array
    relative_power_error: Array
    maximum_sampling_phase_step: Array
    paraxial_angle_estimate: Array
    kernel_elements: Array
    workspace_bytes_per_component: Array
    finite: Array
    accepted: Array
    status: Array


class FresnelPropagationResult(StrictModule):
    """Field on the requested output space and direct-quadrature evidence."""

    field: PlaneField
    evidence: FresnelPropagationEvidence
    prepared_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.evidence.accepted

    @property
    def status(self) -> Array:
        return self.evidence.status


def _uniform_finite_axis(space: PlaneFieldSpace, index: int, /) -> tuple[Array, Array]:
    axis = space.grid.axes[index]
    if (
        axis.basis != "uniform"
        or axis.periodic
        or axis.primary_entity != "point"
        or not axis.lower_endpoint_included
        or not axis.upper_endpoint_included
    ):
        raise ValueError(
            "Direct Fresnel grids require endpoint-including uniform point axes."
        )
    nodes = np.asarray(axis.nodes, dtype=np.float64)
    if nodes.size < 2:
        raise ValueError("Direct Fresnel axes require at least two points.")
    differences = np.diff(nodes)
    spacing = float(differences[0])
    tolerance = 64.0 * np.finfo(nodes.dtype).eps * max(1.0, abs(spacing))
    if (
        not np.isfinite(spacing)
        or spacing <= 0.0
        or not np.all(np.isfinite(differences))
        or not np.allclose(differences, spacing, rtol=1.0e-10, atol=tolerance)
    ):
        raise ValueError("Direct Fresnel axes must be finite and uniformly increasing.")
    measures = axis.quad_weights
    return axis.nodes, measures


def prepare_direct_fresnel(plan: DirectFresnelPlan, /) -> PreparedDirectFresnel:
    """Validate both finite planes and preflight all fixed-shape workspaces."""
    if not isinstance(plan, DirectFresnelPlan):
        raise TypeError("plan must be a DirectFresnelPlan.")
    input_rotation = np.asarray(plan.input_space.frame.rotation, dtype=np.float64)
    output_rotation = np.asarray(plan.output_space.frame.rotation, dtype=np.float64)
    input_translation = np.asarray(plan.input_space.frame.translation, dtype=np.float64)
    output_translation = np.asarray(plan.output_space.frame.translation, dtype=np.float64)
    if not np.array_equal(input_rotation, output_rotation) or not np.array_equal(
        input_translation, output_translation
    ):
        raise ValueError(
            "Direct Fresnel propagation requires identical input/output plane frames."
        )
    input_data = tuple(_uniform_finite_axis(plan.input_space, axis) for axis in range(2))
    output_data = tuple(
        _uniform_finite_axis(plan.output_space, axis) for axis in range(2)
    )
    input_axes = (input_data[0][0], input_data[1][0])
    input_weights = (input_data[0][1], input_data[1][1])
    output_axes = (output_data[0][0], output_data[1][0])
    kernel_elements = sum(
        source.size * target.size
        for source, target in zip(input_axes, output_axes, strict=True)
    )
    if kernel_elements > plan.maximum_kernel_elements:
        raise ValueError(
            "Direct Fresnel separable kernels exceed maximum_kernel_elements."
        )
    input_elements = plan.input_space.size
    output_elements = plan.output_space.size
    workspace_elements = (
        kernel_elements
        + max(
            output_axes[0].size * input_axes[1].size,
            input_axes[0].size * output_axes[1].size,
        )
        + input_elements
        + output_elements
    )
    # Complex128 is the conservative fixed-width accounting. The plan preflights
    # the worst supported case (two tangential components) before any kernel exists.
    workspace_bytes = 16 * workspace_elements
    if 2 * workspace_bytes > plan.maximum_workspace_bytes:
        raise ValueError("Direct Fresnel workspace exceeds maximum_workspace_bytes.")
    separations = []
    for source, target in zip(input_axes, output_axes, strict=True):
        source_host = np.asarray(source, dtype=np.float64)
        target_host = np.asarray(target, dtype=np.float64)
        separations.append(
            max(
                abs(float(target_host[0] - source_host[-1])),
                abs(float(target_host[-1] - source_host[0])),
            )
        )
    maximum_separation = float(np.hypot(*separations))
    identical = plan.input_space.space_id == plan.output_space.space_id
    prepared_id = canonical_fingerprint(
        {
            "kind": "prepared-direct-fresnel",
            "plan": plan.plan_id,
            "input_shape": list(plan.input_space.shape),
            "output_shape": list(plan.output_space.shape),
            "kernel_elements": kernel_elements,
            "workspace_bytes_per_component": workspace_bytes,
            "identical_space": identical,
        }
    )
    return PreparedDirectFresnel(
        plan,
        input_axes,
        output_axes,
        input_weights,
        jnp.asarray(maximum_separation, dtype=input_axes[0].dtype),
        identical,
        kernel_elements,
        workspace_elements,
        workspace_bytes,
        prepared_id,
    )


def _intensity(values: Array, tangential: bool, /) -> Array:
    density = jnp.real(values * jnp.conj(values))
    return jnp.sum(density, axis=-1) if tangential else density


def _maximum_phase_step(phase: Array, /) -> Array:
    source_steps = jnp.max(jnp.abs(jnp.diff(phase, axis=1)))
    output_steps = jnp.max(jnp.abs(jnp.diff(phase, axis=0)))
    return jnp.maximum(source_steps, output_steps)


def propagate_direct_fresnel(
    prepared: PreparedDirectFresnel,
    field: PlaneField,
    distance: ArrayLike,
    medium_wavenumber: ArrayLike,
    /,
) -> FresnelPropagationResult:
    """Execute the separable direct Fresnel integral on the requested output grid."""
    if not isinstance(prepared, PreparedDirectFresnel):
        raise TypeError("prepared must be a PreparedDirectFresnel.")
    if not isinstance(field, (ScalarPlaneField, TangentialPlaneField)):
        raise TypeError("field must be a ScalarPlaneField or TangentialPlaneField.")
    if field.space.space_id != prepared.plan.input_space.space_id:
        raise ValueError("field does not belong to the prepared input space.")
    supplied_distance = jnp.asarray(distance)
    if supplied_distance.shape != ():
        raise ValueError("distance must be a scalar.")
    if jnp.iscomplexobj(supplied_distance) or not jnp.issubdtype(
        supplied_distance.dtype, jnp.number
    ):
        raise TypeError("distance must be real numeric data.")
    distance_ = supplied_distance.astype(
        jnp.result_type(supplied_distance.dtype, jnp.float32)
    )
    supplied_wavenumber = jnp.asarray(medium_wavenumber)
    if supplied_wavenumber.shape != ():
        raise ValueError("medium_wavenumber must be a scalar.")
    if jnp.iscomplexobj(supplied_wavenumber) or not jnp.issubdtype(
        supplied_wavenumber.dtype, jnp.number
    ):
        raise TypeError("medium_wavenumber must be real numeric data.")
    wavenumber = supplied_wavenumber.astype(
        jnp.result_type(supplied_wavenumber.dtype, jnp.float32)
    )
    distance_valid = jnp.isfinite(distance_) & (distance_ >= 0.0)
    wavenumber_valid = jnp.isfinite(wavenumber) & (wavenumber > 0.0)
    zero_distance = distance_ == 0.0
    safe_distance = jnp.where(distance_valid & ~zero_distance, distance_, 1.0)
    safe_wavenumber = jnp.where(wavenumber_valid, wavenumber, 1.0)
    complex_dtype = jnp.result_type(field.values.dtype, wavenumber.dtype, jnp.complex64)

    phase_matrices = []
    kernels = []
    for source, target, weights in zip(
        prepared.input_axes,
        prepared.output_axes,
        prepared.input_weights,
        strict=True,
    ):
        source_ = source.astype(distance_.dtype)
        target_ = target.astype(distance_.dtype)
        phase = (
            safe_wavenumber
            * (source_[None, :] - target_[:, None]) ** 2
            / (2.0 * safe_distance)
        )
        phase_matrices.append(phase)
        kernels.append(
            jnp.exp(1j * phase).astype(complex_dtype)
            * weights.astype(complex_dtype)[None, :]
        )
    intermediate = contract("ai,ij...->aj...", kernels[0], field.values)
    integral = contract("bj,aj...->ab...", kernels[1], intermediate)
    prefactor = (
        safe_wavenumber
        * jnp.exp(1j * safe_wavenumber * safe_distance)
        / (2.0 * jnp.pi * 1j * safe_distance)
    )
    propagated = prefactor.astype(complex_dtype) * integral
    if prepared.identical_space:
        output_values = jnp.where(zero_distance, field.values, propagated)
    else:
        output_values = propagated

    tangential = isinstance(field, TangentialPlaneField)
    input_weights = prepared.plan.input_space.area_weights.astype(distance_.dtype)
    output_weights = prepared.plan.output_space.area_weights.astype(distance_.dtype)
    input_power = jnp.sum(input_weights * _intensity(field.values, tangential))
    output_power = jnp.sum(output_weights * _intensity(output_values, tangential))
    power_error = jnp.where(
        input_power > 0.0,
        jnp.abs(output_power - input_power) / input_power,
        jnp.abs(output_power),
    )
    sampling_step = jnp.maximum(
        _maximum_phase_step(phase_matrices[0]),
        _maximum_phase_step(phase_matrices[1]),
    )
    paraxial_angle = jnp.arctan(
        prepared.maximum_transverse_separation.astype(distance_.dtype) / safe_distance
    )
    sampling_step = jnp.where(
        zero_distance & prepared.identical_space, 0.0, sampling_step
    )
    paraxial_angle = jnp.where(
        zero_distance & prepared.identical_space, 0.0, paraxial_angle
    )
    power_error = jnp.where(zero_distance & prepared.identical_space, 0.0, power_error)
    output_power = jnp.where(
        zero_distance & prepared.identical_space, input_power, output_power
    )
    finite = (
        jnp.all(jnp.isfinite(jnp.real(output_values)))
        & jnp.all(jnp.isfinite(jnp.imag(output_values)))
        & jnp.all(
            jnp.isfinite(
                jnp.stack(
                    (
                        input_power,
                        output_power,
                        power_error,
                        sampling_step,
                        paraxial_angle,
                    )
                )
            )
        )
    )
    status = jnp.where(
        distance_valid,
        int(FresnelPropagationStatus.SUCCESS),
        int(FresnelPropagationStatus.INVALID_DISTANCE),
    ).astype(jnp.int32)
    status = status | jnp.where(
        wavenumber_valid,
        int(FresnelPropagationStatus.SUCCESS),
        int(FresnelPropagationStatus.INVALID_WAVENUMBER),
    ).astype(jnp.int32)
    status = status | jnp.where(
        ~zero_distance | prepared.identical_space,
        int(FresnelPropagationStatus.SUCCESS),
        int(FresnelPropagationStatus.ZERO_DISTANCE_SPACE_MISMATCH),
    ).astype(jnp.int32)
    status = status | jnp.where(
        sampling_step <= prepared.plan.maximum_sampling_phase_step,
        int(FresnelPropagationStatus.SUCCESS),
        int(FresnelPropagationStatus.SAMPLING_LIMIT),
    ).astype(jnp.int32)
    status = status | jnp.where(
        paraxial_angle <= prepared.plan.maximum_paraxial_angle,
        int(FresnelPropagationStatus.SUCCESS),
        int(FresnelPropagationStatus.PARAXIAL_LIMIT),
    ).astype(jnp.int32)
    status = status | jnp.where(
        power_error <= prepared.plan.maximum_power_error,
        int(FresnelPropagationStatus.SUCCESS),
        int(FresnelPropagationStatus.POWER_LIMIT),
    ).astype(jnp.int32)
    status = status | jnp.where(
        finite,
        int(FresnelPropagationStatus.SUCCESS),
        int(FresnelPropagationStatus.NONFINITE),
    ).astype(jnp.int32)
    accepted = status == int(FresnelPropagationStatus.SUCCESS)
    output_coordinate = field.longitudinal_coordinate + jnp.where(
        distance_valid, distance_, 0.0
    )
    if tangential:
        output: PlaneField = TangentialPlaneField(
            prepared.plan.output_space,
            output_values,
            field.angular_frequency,
            output_coordinate,
        )
    else:
        output = ScalarPlaneField(
            prepared.plan.output_space,
            output_values,
            field.angular_frequency,
            output_coordinate,
        )
    evidence = FresnelPropagationEvidence(
        input_power,
        output_power,
        power_error,
        sampling_step,
        paraxial_angle,
        jnp.asarray(prepared.kernel_elements, dtype=jnp.int64),
        jnp.asarray(prepared.workspace_bytes_per_component, dtype=jnp.int64),
        finite,
        accepted,
        status,
    )
    return FresnelPropagationResult(output, evidence, prepared.prepared_id)


__all__ = [
    "DirectFresnelPlan",
    "FresnelPropagationEvidence",
    "FresnelPropagationResult",
    "FresnelPropagationStatus",
    "PreparedDirectFresnel",
    "prepare_direct_fresnel",
    "propagate_direct_fresnel",
]
