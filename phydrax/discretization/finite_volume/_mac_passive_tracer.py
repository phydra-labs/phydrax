#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bounded, nonconservative MacCormack transport for periodic MAC tracers."""

from __future__ import annotations

from enum import IntEnum
from math import prod
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._interpolation import (
    apply_gather_stencil,
    gather_patches,
    rectilinear_stencil,
)
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import ArraySpace
from .._spaces import DiscreteFieldSpace, TensorDofLayout
from ._incompressible import FaceVelocity, PreparedMACOperators


MACPassiveTracerCharacteristicIntegrator: TypeAlias = Literal["midpoint"]
MACPassiveTracerInterpolation: TypeAlias = Literal["multilinear"]
MACPassiveTracerDifferentiation: TypeAlias = Literal["almost_everywhere"]
MACPassiveTracerConservation: TypeAlias = Literal["diagnostic_only"]


class MACPassiveTracerStatus(IntEnum):
    """Portable status for one bounded passive-tracer advance."""

    SUCCESS = 0
    NONFINITE = 1
    DONOR_BOUND_FAILED = 2


class MACPassiveTracerMacCormackResult(StrictModule):
    """Candidate values and evidence from one two-pass MacCormack advance.

    The fixed-shape donor bounds come from the predictor departure stencil. The
    weighted integral defect is diagnostic: it never changes ``success``.
    """

    values: Array
    raw_values: Array
    donor_lower_bound: Array
    donor_upper_bound: Array
    lower_bound_defect: Array
    upper_bound_defect: Array
    limiter_active: Array
    limiter_active_count: Array
    maximum_maccormack_correction: Array
    maximum_limiter_correction: Array
    donor_bound_defect: Array
    source_minimum: Array
    source_maximum: Array
    result_minimum: Array
    result_maximum: Array
    integral_before: Array
    integral_after: Array
    integral_defect: Array
    maximum_displacement_cell_widths: Array
    finite: Array
    donor_bounded: Array
    success: Array
    status: Array
    differentiation: MACPassiveTracerDifferentiation = eqx.field(static=True)
    conservation: MACPassiveTracerConservation = eqx.field(static=True)
    field_space_id: str = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)
    support_id: str = eqx.field(static=True)
    grid_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)


class MACPassiveTracerMacCormackPlan(StrictModule, NonTrainableState):
    """Plan bounded MacCormack transport for one centered point-value tracer.

    This deliberately excludes conservative MAC scalar content. Only periodic
    Cartesian MAC operators and one real scalar field at the exact cell-center
    location are accepted. Route selection and clipping make differentiation
    almost everywhere rather than everywhere smooth.
    """

    operators: PreparedMACOperators
    tracer_space: DiscreteFieldSpace
    correction_strength: float = eqx.field(static=True)
    characteristic_integrator: MACPassiveTracerCharacteristicIntegrator = eqx.field(
        static=True
    )
    interpolation: MACPassiveTracerInterpolation = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        operators: PreparedMACOperators,
        tracer_space: DiscreteFieldSpace,
        /,
        *,
        correction_strength: float = 1.0,
        characteristic_integrator: MACPassiveTracerCharacteristicIntegrator = "midpoint",
        interpolation: MACPassiveTracerInterpolation = "multilinear",
    ):
        if not isinstance(operators, PreparedMACOperators):
            raise TypeError("operators must be PreparedMACOperators.")
        if not isinstance(tracer_space, DiscreteFieldSpace):
            raise TypeError("tracer_space must be a DiscreteFieldSpace.")
        if characteristic_integrator != "midpoint":
            raise ValueError("Passive-tracer characteristics require midpoint tracing.")
        if interpolation != "multilinear":
            raise ValueError("Passive-tracer interpolation must be multilinear.")
        correction = float(correction_strength)
        if not np.isfinite(correction) or not 0.0 <= correction <= 1.0:
            raise ValueError("correction_strength must be finite and lie in [0, 1].")

        discretization = operators.discretization
        grid = discretization.grid
        if any(not axis.periodic for axis in grid.structured_axes):
            raise ValueError(
                "Passive-tracer MacCormack transport requires periodic axes."
            )
        if any(int(size) < 2 for size in discretization.cell_shape):
            raise ValueError(
                "Multilinear passive-tracer transport needs two cells per axis."
            )
        if tracer_space.representation != "point_value":
            raise ValueError("Passive tracers require point_value representation.")
        if not isinstance(tracer_space.layout, TensorDofLayout):
            raise TypeError("Passive tracers require a TensorDofLayout.")
        if tracer_space.layout.component_shape:
            raise ValueError("Passive tracers must be scalar, not component-valued.")

        expected = grid.field_space(
            tracer_space.name,
            entity_layout=discretization.cell_layout,
            dtype=operators.pressure_space.dtype,
            representation="point_value",
            conformity=tracer_space.conformity,
        )
        if tracer_space.support_id != expected.support_id:
            raise ValueError("Passive-tracer support does not match the MAC grid.")
        if tracer_space.layout.location_id != discretization.cell_layout.location_id:
            raise ValueError("Passive tracers must use the exact centered cell location.")
        if tracer_space.layout.layout_id != expected.layout.layout_id:
            raise ValueError(
                "Passive-tracer layout does not match the centered grid layout."
            )
        tracer_vector_space = tracer_space.vector_space
        expected_vector_space = expected.vector_space
        if not isinstance(tracer_vector_space, ArraySpace):
            raise TypeError("Passive-tracer coordinates require an ArraySpace.")
        if not isinstance(expected_vector_space, ArraySpace):
            raise TypeError("The centered grid must provide an ArraySpace.")
        if tracer_vector_space.shape != discretization.cell_shape:
            raise ValueError("Passive-tracer shape must equal the MAC cell shape.")
        if np.dtype(tracer_vector_space.dtype) != np.dtype(
            operators.pressure_space.dtype
        ):
            raise TypeError("Passive-tracer dtype must match the MAC operator dtype.")
        if jnp.issubdtype(tracer_vector_space.dtype, jnp.complexfloating):
            raise TypeError("Passive-tracer values must be real-valued.")
        if tracer_vector_space.space_id != expected_vector_space.space_id:
            raise ValueError(
                "Passive-tracer vector-space identity does not match the grid."
            )
        if (
            tracer_vector_space.pairing.pairing_id
            != expected_vector_space.pairing.pairing_id
        ):
            raise ValueError("Passive-tracer pairing does not match the cell measure.")
        if tracer_space.reconstruction_id != expected.reconstruction_id:
            raise ValueError("Passive-tracer reconstruction does not match the MAC grid.")
        if tracer_space.projection_id != expected.projection_id:
            raise ValueError(
                "Passive-tracer projection identity does not match the grid."
            )

        self.operators = operators
        self.tracer_space = tracer_space
        self.correction_strength = correction
        self.characteristic_integrator = characteristic_integrator
        self.interpolation = interpolation
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mac-passive-tracer-maccormack-plan",
                "operators": operators.prepared_id,
                "field_space": tracer_space.field_space_id,
                "correction_strength": correction,
                "characteristic_integrator": characteristic_integrator,
                "interpolation": interpolation,
            }
        )

    def prepare(self, /) -> "PreparedMACPassiveTracerMacCormack":
        return PreparedMACPassiveTracerMacCormack(self)


class PreparedMACPassiveTracerMacCormack(StrictModule, NonTrainableState):
    """Prepared fixed-shape periodic multilinear MacCormack operator."""

    operators: PreparedMACOperators
    tracer_space: DiscreteFieldSpace
    cell_axis_coordinates: tuple[Array, ...]
    face_axis_coordinates: tuple[tuple[Array, ...], ...]
    cell_centers: Array
    cell_widths: Array
    domain_lower: Array
    periods_array: Array
    axis_bounds: tuple[tuple[float, float], ...] = eqx.field(static=True)
    periods: tuple[float, ...] = eqx.field(static=True)
    correction_strength: float = eqx.field(static=True)
    characteristic_integrator: MACPassiveTracerCharacteristicIntegrator = eqx.field(
        static=True
    )
    interpolation: MACPassiveTracerInterpolation = eqx.field(static=True)
    differentiation: MACPassiveTracerDifferentiation = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    cell_shape: tuple[int, ...] = eqx.field(static=True)
    stencil_evaluations: int = eqx.field(static=True)
    route_count: int = eqx.field(static=True)
    work_count: int = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(self, plan: MACPassiveTracerMacCormackPlan, /):
        if not isinstance(plan, MACPassiveTracerMacCormackPlan):
            raise TypeError("plan must be MACPassiveTracerMacCormackPlan.")
        tracer_vector_space = plan.tracer_space.vector_space
        if not isinstance(tracer_vector_space, ArraySpace):
            raise TypeError("The prepared tracer plan must own an ArraySpace.")
        discretization = plan.operators.discretization
        grid = discretization.grid
        dtype = tracer_vector_space.dtype
        dimension = len(discretization.cell_shape)
        cell_shape = discretization.cell_shape
        bounds = tuple(
            (float(axis.bounds[0]), float(axis.bounds[1]))
            for axis in grid.structured_axes
        )
        periods = tuple(upper - lower for lower, upper in bounds)
        cell_axis_coordinates = tuple(
            jnp.asarray(values, dtype=dtype)
            for values in discretization.cell_layout.coordinates_by_axis
        )
        face_axis_coordinates = tuple(
            tuple(
                jnp.asarray(values, dtype=dtype) for values in layout.coordinates_by_axis
            )
            for layout in discretization.face_layouts
        )
        width_components = []
        for axis, structured_axis in enumerate(grid.structured_axes):
            shape = [1] * dimension
            shape[axis] = int(structured_axis.interval_widths.size)
            width_components.append(
                jnp.broadcast_to(
                    jnp.asarray(structured_axis.interval_widths, dtype=dtype).reshape(
                        tuple(shape)
                    ),
                    cell_shape,
                )
            )
        cell_widths = jnp.stack(width_components, axis=-1)
        stencil_evaluations = 3 * dimension + 3
        corner_count = 1 << dimension
        route_count = stencil_evaluations * prod(cell_shape) * corner_count
        prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-mac-passive-tracer-maccormack",
                "plan": plan.plan_id,
                "cell_layout": discretization.cell_layout.layout_id,
                "face_layouts": [
                    layout.layout_id for layout in discretization.face_layouts
                ],
                "stencil_evaluations": stencil_evaluations,
                "route_count": route_count,
                "differentiation": "almost_everywhere",
                "conservation": "diagnostic_only",
            }
        )
        self.operators = plan.operators
        self.tracer_space = plan.tracer_space
        self.cell_axis_coordinates = cell_axis_coordinates
        self.face_axis_coordinates = face_axis_coordinates
        self.cell_centers = jnp.asarray(discretization.cell_centers, dtype=dtype)
        self.cell_widths = cell_widths
        self.domain_lower = jnp.asarray(tuple(lower for lower, _ in bounds), dtype=dtype)
        self.periods_array = jnp.asarray(periods, dtype=dtype)
        self.axis_bounds = bounds
        self.periods = periods
        self.correction_strength = plan.correction_strength
        self.characteristic_integrator = plan.characteristic_integrator
        self.interpolation = plan.interpolation
        self.differentiation = "almost_everywhere"
        self.dimension = dimension
        self.cell_shape = cell_shape
        self.stencil_evaluations = stencil_evaluations
        self.route_count = route_count
        self.work_count = stencil_evaluations
        self.prepared_id = prepared_id
        self.result_id = canonical_fingerprint(
            {
                "kind": "mac-passive-tracer-maccormack-result",
                "prepared": prepared_id,
                "field_space": plan.tracer_space.field_space_id,
                "grid": grid.prepared_id,
            }
        )

    def _stencil(self, axis_nodes: tuple[Array, ...], coordinates: Array, /):
        return rectilinear_stencil(
            axis_nodes,
            coordinates,
            boundary=("periodic",) * self.dimension,
            periods=self.periods,
            axis_bounds=self.axis_bounds,
        )

    def _wrap(self, coordinates: Array, /) -> Array:
        return (
            jnp.mod(coordinates - self.domain_lower, self.periods_array)
            + self.domain_lower
        )

    def _sample_velocity(
        self,
        velocity: FaceVelocity,
        coordinates: Array,
        /,
    ) -> Array:
        components = []
        for values, nodes in zip(velocity, self.face_axis_coordinates, strict=True):
            stencil = self._stencil(nodes, coordinates)
            interpolation = apply_gather_stencil(values.reshape((-1,)), stencil)
            components.append(interpolation.values)
        return jnp.stack(components, axis=-1)

    def advance(
        self,
        tracer: ArrayLike,
        velocity: FaceVelocity,
        step_size: ArrayLike,
        /,
    ) -> MACPassiveTracerMacCormackResult:
        """Advance one runtime scalar step with frozen face velocity.

        Output shapes depend only on the prepared grid. Search routes and limiter
        switches are piecewise constant, so derivatives are almost everywhere.
        """

        old = self.tracer_space.vector_space.validate(jnp.asarray(tracer))
        face_velocity = self.operators.validate_velocity(velocity)
        step = jnp.asarray(step_size, dtype=old.dtype)
        if step.shape != ():
            raise ValueError("step_size must be scalar.")
        step_finite = jnp.isfinite(step)
        safe_step = jnp.where(step_finite, step, 0.0)
        velocity_finite = jnp.asarray(True)
        safe_velocity = []
        for component in face_velocity:
            component_finite = jnp.all(jnp.isfinite(component))
            velocity_finite = velocity_finite & component_finite
            safe_velocity.append(jnp.where(jnp.isfinite(component), component, 0.0))
        safe_velocity_ = tuple(safe_velocity)

        center_velocity = self._sample_velocity(safe_velocity_, self.cell_centers)
        backward_midpoint_raw = self.cell_centers - 0.5 * safe_step * center_velocity
        characteristic_finite = jnp.all(jnp.isfinite(backward_midpoint_raw))
        backward_midpoint = self._wrap(
            jnp.where(
                jnp.isfinite(backward_midpoint_raw),
                backward_midpoint_raw,
                self.cell_centers,
            )
        )
        backward_midpoint_velocity = self._sample_velocity(
            safe_velocity_, backward_midpoint
        )
        departure_raw = self.cell_centers - safe_step * backward_midpoint_velocity
        characteristic_finite = characteristic_finite & jnp.all(
            jnp.isfinite(departure_raw)
        )
        departure = self._wrap(
            jnp.where(jnp.isfinite(departure_raw), departure_raw, self.cell_centers)
        )
        forward_midpoint_raw = self.cell_centers + 0.5 * safe_step * center_velocity
        characteristic_finite = characteristic_finite & jnp.all(
            jnp.isfinite(forward_midpoint_raw)
        )
        forward_midpoint = self._wrap(
            jnp.where(
                jnp.isfinite(forward_midpoint_raw),
                forward_midpoint_raw,
                self.cell_centers,
            )
        )
        forward_midpoint_velocity = self._sample_velocity(
            safe_velocity_, forward_midpoint
        )
        return_point_raw = self.cell_centers + safe_step * forward_midpoint_velocity
        characteristic_finite = characteristic_finite & jnp.all(
            jnp.isfinite(return_point_raw)
        )
        return_point = self._wrap(
            jnp.where(jnp.isfinite(return_point_raw), return_point_raw, self.cell_centers)
        )

        predictor_stencil = self._stencil(self.cell_axis_coordinates, departure)
        flattened_old = old.reshape((-1,))
        predictor = apply_gather_stencil(flattened_old, predictor_stencil).values.reshape(
            self.cell_shape
        )
        donor_values, donor_valid = gather_patches(flattened_old, predictor_stencil)
        donor_values = jnp.where(donor_valid, donor_values, old[..., None])
        donor_lower = jnp.min(donor_values, axis=-1)
        donor_upper = jnp.max(donor_values, axis=-1)

        reverse_stencil = self._stencil(self.cell_axis_coordinates, return_point)
        reverse = apply_gather_stencil(
            predictor.reshape((-1,)), reverse_stencil
        ).values.reshape(self.cell_shape)
        maccormack_correction = 0.5 * self.correction_strength * (old - reverse)
        raw = predictor + maccormack_correction
        values = jnp.clip(raw, donor_lower, donor_upper)
        lower_defect = jnp.maximum(donor_lower - raw, 0.0)
        upper_defect = jnp.maximum(raw - donor_upper, 0.0)
        limiter_active = (lower_defect > 0.0) | (upper_defect > 0.0)
        limiter_correction = values - raw

        final_lower_defect = jnp.max(jnp.maximum(donor_lower - values, 0.0))
        final_upper_defect = jnp.max(jnp.maximum(values - donor_upper, 0.0))
        donor_bound_defect = jnp.maximum(final_lower_defect, final_upper_defect)
        scale = jnp.maximum(
            1.0,
            jnp.maximum(jnp.max(jnp.abs(donor_lower)), jnp.max(jnp.abs(donor_upper))),
        )
        bound_tolerance = 8.0 * jnp.finfo(old.dtype).eps * scale
        donor_bounded = donor_bound_defect <= bound_tolerance

        integral_before = self.tracer_space.vector_space.inner(jnp.ones_like(old), old)
        integral_after = self.tracer_space.vector_space.inner(
            jnp.ones_like(values), values
        )
        backward_displacement = safe_step * backward_midpoint_velocity / self.cell_widths
        forward_displacement = safe_step * forward_midpoint_velocity / self.cell_widths
        maximum_displacement = jnp.maximum(
            jnp.max(jnp.sqrt(jnp.sum(backward_displacement**2, axis=-1))),
            jnp.max(jnp.sqrt(jnp.sum(forward_displacement**2, axis=-1))),
        )

        finite = (
            jnp.all(jnp.isfinite(old))
            & velocity_finite
            & step_finite
            & characteristic_finite
            & jnp.all(jnp.isfinite(values))
            & jnp.all(jnp.isfinite(raw))
            & jnp.isfinite(integral_before)
            & jnp.isfinite(integral_after)
            & jnp.isfinite(maximum_displacement)
        )
        success = finite & donor_bounded
        status = jnp.where(
            ~finite,
            jnp.asarray(MACPassiveTracerStatus.NONFINITE, dtype=jnp.int32),
            jnp.where(
                ~donor_bounded,
                jnp.asarray(MACPassiveTracerStatus.DONOR_BOUND_FAILED, dtype=jnp.int32),
                jnp.asarray(MACPassiveTracerStatus.SUCCESS, dtype=jnp.int32),
            ),
        )
        discretization = self.operators.discretization
        return MACPassiveTracerMacCormackResult(
            values=values,
            raw_values=raw,
            donor_lower_bound=donor_lower,
            donor_upper_bound=donor_upper,
            lower_bound_defect=lower_defect,
            upper_bound_defect=upper_defect,
            limiter_active=limiter_active,
            limiter_active_count=jnp.sum(limiter_active, dtype=jnp.int32),
            maximum_maccormack_correction=jnp.max(jnp.abs(maccormack_correction)),
            maximum_limiter_correction=jnp.max(jnp.abs(limiter_correction)),
            donor_bound_defect=donor_bound_defect,
            source_minimum=jnp.min(old),
            source_maximum=jnp.max(old),
            result_minimum=jnp.min(values),
            result_maximum=jnp.max(values),
            integral_before=integral_before,
            integral_after=integral_after,
            integral_defect=integral_after - integral_before,
            maximum_displacement_cell_widths=maximum_displacement,
            finite=finite,
            donor_bounded=donor_bounded,
            success=success,
            status=status,
            differentiation=self.differentiation,
            conservation="diagnostic_only",
            field_space_id=self.tracer_space.field_space_id,
            layout_id=self.tracer_space.layout.layout_id,
            support_id=self.tracer_space.support_id,
            grid_id=discretization.grid.prepared_id,
            method_id=self.prepared_id,
            result_id=self.result_id,
        )


__all__ = [
    "MACPassiveTracerCharacteristicIntegrator",
    "MACPassiveTracerConservation",
    "MACPassiveTracerDifferentiation",
    "MACPassiveTracerInterpolation",
    "MACPassiveTracerMacCormackPlan",
    "MACPassiveTracerMacCormackResult",
    "MACPassiveTracerStatus",
    "PreparedMACPassiveTracerMacCormack",
]
