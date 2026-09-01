#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from enum import IntFlag
from typing import Any, Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.finite_volume import (
    FaceVelocity,
    MACBoundaryStageData,
    PreparedMACBoundaryPlan,
    PreparedMACOperators,
)
from ..linalg import (
    DifferentiationPolicy,
    FunctionLinearOperator,
    GMRES,
    LinearSolvePolicy,
    LinearSolveResult,
    LinearSystem,
    OperatorProperties,
    solve,
    TolerancePolicy,
)


class MACSharpInterfaceStatus(IntFlag):
    SUCCESS = 0
    LINEAR_SOLVE_FAILED = 1
    DIVERGENCE_FAILED = 2
    GEOMETRY_FAILED = 4
    NONFINITE = 8


class MACSharpInterfaceGeometry(StrictModule, NonTrainableState):
    cell_fluid_fraction: Array
    face_fluid_aperture: FaceVelocity
    interface_area: Array
    interface_centroid: Array
    interface_normal: Array
    body_id: Array
    geometry_id: str = eqx.field(static=True)


class MACSharpInterfaceForce(StrictModule):
    force: Array
    torque: Array
    pressure_force: Array
    viscous_force: Array
    finite: Array


class MACSharpInterfaceProjectionResult(StrictModule):
    velocity: FaceVelocity
    pressure: Array
    divergence_before: Array
    divergence_after: Array
    divergence_norm: Array
    linear: LinearSolveResult
    force: MACSharpInterfaceForce
    stabilization_defect: Array
    status: Array
    accepted: Array
    plan_id: str = eqx.field(static=True)


class MACSharpInterfaceProjectionPlan(StrictModule, NonTrainableState):
    """Conservative cut-cell pressure projection and sharp traction integration."""

    operators: PreparedMACOperators
    boundaries: PreparedMACBoundaryPlan
    geometry: MACSharpInterfaceGeometry
    minimum_fluid_fraction: float = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    linear_policy: LinearSolvePolicy
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        operators: PreparedMACOperators,
        boundaries: PreparedMACBoundaryPlan,
        cell_fluid_fraction: ArrayLike,
        face_fluid_aperture: FaceVelocity,
        interface_area: ArrayLike,
        interface_centroid: ArrayLike,
        interface_normal: ArrayLike,
        body_id: ArrayLike,
        /,
        *,
        minimum_fluid_fraction: float = 1.0e-3,
        tolerance: float = 1.0e-9,
        linear_policy: LinearSolvePolicy | None = None,
    ):
        if boundaries.operators.prepared_id != operators.prepared_id:
            raise ValueError("Sharp-interface boundaries and operators differ.")
        fraction = np.asarray(cell_fluid_fraction)
        apertures = tuple(np.asarray(value) for value in face_fluid_aperture)
        area = np.asarray(interface_area)
        centroid = np.asarray(interface_centroid)
        normal = np.asarray(interface_normal)
        bodies = np.asarray(body_id)
        dimension = len(operators.discretization.cell_shape)
        if fraction.shape != operators.discretization.cell_shape:
            raise ValueError("cell_fluid_fraction must have the MAC cell shape.")
        if len(apertures) != dimension or any(
            value.shape != operators.discretization.face_layouts[axis].shape
            for axis, value in enumerate(apertures)
        ):
            raise ValueError("face_fluid_aperture does not match MAC face layouts.")
        if (
            area.shape != fraction.shape
            or bodies.shape != fraction.shape
            or centroid.shape != fraction.shape + (dimension,)
            or normal.shape != centroid.shape
        ):
            raise ValueError("Sharp-interface cell geometry shapes are incompatible.")
        if (
            np.any(~np.isfinite(fraction))
            or np.any((fraction < 0.0) | (fraction > 1.0))
            or any(
                np.any(~np.isfinite(value)) or np.any((value < 0.0) | (value > 1.0))
                for value in apertures
            )
            or np.any(~np.isfinite(area))
            or np.any(area < 0.0)
            or np.any(~np.isfinite(centroid))
            or np.any(~np.isfinite(normal))
        ):
            raise ValueError("Sharp-interface geometry is not finite/admissible.")
        cut = (fraction > 0.0) & (fraction < 1.0) & (area > 0.0)
        norm = np.linalg.norm(normal, axis=-1)
        if np.any(np.abs(norm[cut] - 1.0) > 1.0e-8):
            raise ValueError("Cut-cell interface normals must be unit length.")
        minimum = float(minimum_fluid_fraction)
        tolerance_ = float(tolerance)
        if not 0.0 < minimum <= 1.0 or tolerance_ <= 0.0:
            raise ValueError("Sharp-interface stabilization/tolerance is invalid.")
        geometry_id = canonical_fingerprint(
            {
                "kind": "mac-sharp-interface-geometry",
                "arrays": array_tree_fingerprint(
                    (fraction, apertures, area, centroid, normal, bodies)
                ),
            }
        )
        self.operators = operators
        self.boundaries = boundaries
        self.geometry = MACSharpInterfaceGeometry(
            jnp.asarray(fraction),
            tuple(jnp.asarray(value) for value in apertures),
            jnp.asarray(area),
            jnp.asarray(centroid),
            jnp.asarray(normal),
            jnp.asarray(bodies, dtype=jnp.int32),
            geometry_id,
        )
        self.minimum_fluid_fraction = minimum
        self.tolerance = tolerance_
        self.linear_policy = (
            LinearSolvePolicy(
                GMRES(restart=50),
                tolerance=TolerancePolicy(
                    relative=tolerance_,
                    absolute=tolerance_,
                    max_steps=1000,
                ),
                differentiation=DifferentiationPolicy("mathematical"),
            )
            if linear_policy is None
            else linear_policy
        )
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mac-sharp-interface-projection",
                "operators": operators.prepared_id,
                "boundaries": boundaries.prepared_id,
                "geometry": geometry_id,
                "minimum_fluid_fraction": minimum,
                "tolerance": tolerance_,
            }
        )

    def _effective_fraction(self) -> Array:
        fraction = self.geometry.cell_fluid_fraction
        return jnp.where(
            fraction > 0.0,
            jnp.maximum(fraction, self.minimum_fluid_fraction),
            1.0,
        )

    def divergence(self, velocity: FaceVelocity, /) -> Array:
        values = self.operators.validate_velocity(velocity)
        flux = tuple(
            aperture * value
            for aperture, value in zip(
                self.geometry.face_fluid_aperture, values, strict=True
            )
        )
        raw = self.operators.divergence(flux) / self._effective_fraction()
        return jnp.where(self.geometry.cell_fluid_fraction > 0.0, raw, 0.0)

    def gradient(
        self,
        pressure: ArrayLike,
        stage: MACBoundaryStageData,
        /,
    ) -> FaceVelocity:
        derivative = self.boundaries.pressure_gradient(
            pressure,
            stage,
            homogeneous=self.boundaries.closure_kind == "neumann",
        )
        return tuple(
            aperture * value
            for aperture, value in zip(
                self.geometry.face_fluid_aperture, derivative, strict=True
            )
        )

    def force(
        self,
        pressure: ArrayLike,
        /,
        *,
        viscous_traction: ArrayLike | None = None,
        reference_point: ArrayLike | None = None,
    ) -> MACSharpInterfaceForce:
        pressure_ = self.operators.validate_pressure(pressure)
        dimension = len(self.operators.discretization.cell_shape)
        viscous = (
            jnp.zeros(self.geometry.interface_normal.shape, dtype=pressure_.dtype)
            if viscous_traction is None
            else jnp.asarray(viscous_traction, dtype=pressure_.dtype)
        )
        if viscous.shape != self.geometry.interface_normal.shape:
            raise ValueError("viscous_traction must have one vector per cell.")
        pressure_density = -pressure_[..., None] * self.geometry.interface_normal
        pressure_force = jnp.sum(
            self.geometry.interface_area[..., None] * pressure_density,
            axis=tuple(range(dimension)),
        )
        viscous_force = jnp.sum(
            self.geometry.interface_area[..., None] * viscous,
            axis=tuple(range(dimension)),
        )
        traction = pressure_density + viscous
        point = (
            jnp.zeros((dimension,), dtype=pressure_.dtype)
            if reference_point is None
            else jnp.asarray(reference_point, dtype=pressure_.dtype)
        )
        arm = self.geometry.interface_centroid - point
        weighted = self.geometry.interface_area[..., None] * traction
        torque_density = (
            (arm[..., 0] * weighted[..., 1] - arm[..., 1] * weighted[..., 0])[..., None]
            if dimension == 2
            else jnp.cross(arm, weighted)
        )
        torque = jnp.sum(torque_density, axis=tuple(range(dimension)))
        total = pressure_force + viscous_force
        finite = jnp.all(jnp.isfinite(total)) & jnp.all(jnp.isfinite(torque))
        return MACSharpInterfaceForce(
            total, torque, pressure_force, viscous_force, finite
        )

    def project(
        self,
        velocity: FaceVelocity,
        inverse_momentum: FaceVelocity,
        boundary_stage: MACBoundaryStageData,
        /,
        *,
        pressure: ArrayLike | None = None,
        jump_source: ArrayLike | None = None,
        wall_velocity: FaceVelocity | None = None,
        viscous_traction: ArrayLike | None = None,
        reference_point: ArrayLike | None = None,
    ) -> MACSharpInterfaceProjectionResult:
        stage = self.boundaries.validate_stage(boundary_stage)
        boundary = self.boundaries.correction_descriptor(stage)
        values = self.operators.validate_velocity(velocity)
        inverse = self.operators.validate_velocity(inverse_momentum)
        wall = (
            tuple(jnp.zeros_like(value) for value in values)
            if wall_velocity is None
            else self.operators.validate_velocity(wall_velocity)
        )
        effective = tuple(
            aperture * value + (1.0 - aperture) * prescribed
            for aperture, value, prescribed in zip(
                self.geometry.face_fluid_aperture, values, wall, strict=True
            )
        )
        divergence_before = self.divergence(effective)
        source = (
            jnp.zeros_like(divergence_before)
            if jump_source is None
            else self.operators.validate_pressure(jump_source)
        )
        fraction = self.geometry.cell_fluid_fraction
        volumes = self.operators.discretization.cell_volumes.astype(
            divergence_before.dtype
        )

        def gauge(value):
            weight = volumes * fraction
            mean = jnp.sum(weight * value) / jnp.maximum(jnp.sum(weight), 1.0)
            return jnp.where(fraction > 0.0, value - mean, 0.0)

        def action(value):
            derivative = self.gradient(gauge(value), stage)
            correction = tuple(
                coefficient * gradient
                for coefficient, gradient in zip(inverse, derivative, strict=True)
            )
            fluid = gauge(-self.divergence(correction))
            return jnp.where(fraction > 0.0, fluid, value)

        operator = FunctionLinearOperator(
            action,
            source=self.operators.pressure_space,
            target=self.operators.pressure_space,
            transpose_action=action,
            properties=OperatorProperties(
                self_adjoint=True,
                positive_definite=True,
                evidence={
                    "self_adjoint": "construction",
                    "positive_definite": "construction",
                },
            ),
            operator_id=f"mac-sharp-pressure/{self.plan_id}",
        )
        right_hand_side = gauge(-divergence_before + source)
        initial = (
            jnp.zeros_like(divergence_before)
            if pressure is None
            else self.operators.validate_pressure(pressure)
        )
        linear = solve(
            LinearSystem(operator, problem_id=f"mac-sharp-pressure/{self.plan_id}"),
            right_hand_side,
            policy=self.linear_policy,
            initial_guess=initial,
        )
        pressure_value = gauge(linear.value)
        derivative = self.gradient(pressure_value, stage)
        corrected_open = tuple(
            value - coefficient * gradient
            for value, coefficient, gradient in zip(
                values, inverse, derivative, strict=True
            )
        )
        corrected = tuple(
            aperture * value + (1.0 - aperture) * prescribed
            for aperture, value, prescribed in zip(
                self.geometry.face_fluid_aperture,
                corrected_open,
                wall,
                strict=True,
            )
        )
        corrected = boundary.affine_velocity(corrected)
        divergence_after = self.divergence(corrected) - source
        divergence_norm = jnp.sqrt(jnp.sum(volumes * fraction * divergence_after**2))
        stabilization_defect = jnp.sum(
            volumes * (self._effective_fraction() - fraction) * divergence_after
        )
        integrated_force = self.force(
            pressure_value,
            viscous_traction=viscous_traction,
            reference_point=reference_point,
        )
        scale = jnp.sqrt(jnp.sum(volumes * fraction * divergence_before**2))
        finite = (
            jnp.isfinite(divergence_norm)
            & jnp.isfinite(stabilization_defect)
            & integrated_force.finite
        )
        divergence_valid = divergence_norm <= self.tolerance * jnp.maximum(scale, 1.0)
        status = jnp.asarray(int(MACSharpInterfaceStatus.SUCCESS), dtype=jnp.int32)
        status = status | jnp.where(
            linear.successful,
            0,
            int(MACSharpInterfaceStatus.LINEAR_SOLVE_FAILED),
        ).astype(jnp.int32)
        status = status | jnp.where(
            divergence_valid,
            0,
            int(MACSharpInterfaceStatus.DIVERGENCE_FAILED),
        ).astype(jnp.int32)
        status = status | jnp.where(
            finite, 0, int(MACSharpInterfaceStatus.NONFINITE)
        ).astype(jnp.int32)
        accepted = linear.successful & divergence_valid & finite
        return MACSharpInterfaceProjectionResult(
            corrected,
            pressure_value,
            divergence_before,
            divergence_after,
            divergence_norm,
            linear,
            integrated_force,
            stabilization_defect,
            status,
            accepted,
            self.plan_id,
        )


MACSharpGeometryProvider = Callable[[Array, Any], "MACSharpInterfaceGeometryData"]
MACInterfaceJumpSource = Callable[[Array, MACSharpInterfaceGeometry, Any], Array]


class MACSharpInterfaceGeometryData(StrictModule):
    cell_fluid_fraction: Array
    face_fluid_aperture: FaceVelocity
    interface_area: Array
    interface_centroid: Array
    interface_normal: Array
    body_id: Array
    swept_cell_volume_rate: Array


class MACMovingSharpInterfaceEpochResult(StrictModule):
    geometry: MACSharpInterfaceGeometryData
    projection: MACSharpInterfaceProjectionPlan
    time: Array
    step_size: Array
    volume_rate: Array
    swept_volume_rate: Array
    gcl_residual: Array
    maximum_gcl_residual: Array
    finite: Array
    accepted: Array
    epoch_id: str = eqx.field(static=True)


class MACMovingSharpInterfaceEpochPlan(StrictModule, NonTrainableState):
    """Fixed-capacity moving cut-cell geometry with swept-volume evidence."""

    operators: PreparedMACOperators
    boundaries: PreparedMACBoundaryPlan
    provider: MACSharpGeometryProvider = eqx.field(static=True)
    geometry_family_id: str = eqx.field(static=True)
    minimum_fluid_fraction: float = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        operators: PreparedMACOperators,
        boundaries: PreparedMACBoundaryPlan,
        provider: MACSharpGeometryProvider,
        /,
        *,
        geometry_family_id: str,
        minimum_fluid_fraction: float = 1.0e-3,
        tolerance: float = 1.0e-9,
    ):
        if not callable(provider):
            raise TypeError("provider must be callable.")
        identifier = str(geometry_family_id)
        minimum = float(minimum_fluid_fraction)
        tolerance_ = float(tolerance)
        if not identifier or not 0.0 < minimum <= 1.0 or tolerance_ <= 0.0:
            raise ValueError("Moving sharp-interface policy is invalid.")
        self.operators = operators
        self.boundaries = boundaries
        self.provider = provider
        self.geometry_family_id = identifier
        self.minimum_fluid_fraction = minimum
        self.tolerance = tolerance_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mac-moving-sharp-interface-epochs",
                "operators": operators.prepared_id,
                "boundaries": boundaries.prepared_id,
                "geometry_family": identifier,
                "minimum_fluid_fraction": minimum,
                "tolerance": tolerance_,
            }
        )

    def projection(
        self, time: ArrayLike, args: Any = None, /
    ) -> tuple[MACSharpInterfaceGeometryData, MACSharpInterfaceProjectionPlan]:
        data = self.provider(jnp.asarray(time), args)
        if not isinstance(data, MACSharpInterfaceGeometryData):
            raise TypeError(
                "Moving sharp-interface provider returned an invalid geometry."
            )
        projection = MACSharpInterfaceProjectionPlan(
            self.operators,
            self.boundaries,
            data.cell_fluid_fraction,
            data.face_fluid_aperture,
            data.interface_area,
            data.interface_centroid,
            data.interface_normal,
            data.body_id,
            minimum_fluid_fraction=self.minimum_fluid_fraction,
            tolerance=self.tolerance,
        )
        return data, projection

    def transition(
        self,
        previous_time: ArrayLike,
        previous: MACSharpInterfaceGeometryData,
        time: ArrayLike,
        args: Any = None,
        /,
    ) -> MACMovingSharpInterfaceEpochResult:
        if not isinstance(previous, MACSharpInterfaceGeometryData):
            raise TypeError("previous must be MACSharpInterfaceGeometryData.")
        previous_time_ = jnp.asarray(previous_time)
        time_ = jnp.asarray(time)
        step = time_ - previous_time_
        step = eqx.error_if(
            step,
            ~jnp.isfinite(step) | (step <= 0.0),
            "Moving sharp-interface epoch requires a positive time step.",
        )
        current, projection = self.projection(time_, args)
        volumes = self.operators.discretization.cell_volumes.astype(
            current.cell_fluid_fraction.dtype
        )
        volume_rate = (
            volumes * (current.cell_fluid_fraction - previous.cell_fluid_fraction) / step
        )
        residual = volume_rate - current.swept_cell_volume_rate
        maximum = jnp.max(jnp.abs(residual))
        scale = jnp.maximum(
            1.0,
            jnp.max(jnp.abs(volume_rate))
            + jnp.max(jnp.abs(current.swept_cell_volume_rate)),
        )
        finite = jnp.all(jnp.isfinite(residual)) & jnp.all(
            jnp.isfinite(current.swept_cell_volume_rate)
        )
        accepted = finite & (maximum <= self.tolerance * scale)
        return MACMovingSharpInterfaceEpochResult(
            current,
            projection,
            time_,
            step,
            volume_rate,
            current.swept_cell_volume_rate,
            residual,
            maximum,
            finite,
            accepted,
            canonical_fingerprint(
                {
                    "kind": "mac-moving-sharp-interface-epoch",
                    "plan": self.plan_id,
                    "geometry": projection.geometry.geometry_id,
                }
            ),
        )


class MACImmersedInterfaceProjectionPlan(StrictModule, NonTrainableState):
    """Sharp projection with an explicit pressure/stress-jump cell source."""

    sharp: MACSharpInterfaceProjectionPlan
    jump_source: MACInterfaceJumpSource = eqx.field(static=True)
    jump_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        sharp: MACSharpInterfaceProjectionPlan,
        jump_source: MACInterfaceJumpSource,
        /,
        *,
        jump_id: str,
    ):
        if not isinstance(sharp, MACSharpInterfaceProjectionPlan):
            raise TypeError("sharp must be MACSharpInterfaceProjectionPlan.")
        if not callable(jump_source):
            raise TypeError("jump_source must be callable.")
        identifier = str(jump_id)
        if not identifier:
            raise ValueError("jump_id must be nonempty.")
        self.sharp = sharp
        self.jump_source = jump_source
        self.jump_id = identifier
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mac-immersed-interface-projection",
                "sharp": sharp.plan_id,
                "jump": identifier,
            }
        )

    def project(
        self,
        time: ArrayLike,
        velocity: FaceVelocity,
        inverse_momentum: FaceVelocity,
        boundary_stage: MACBoundaryStageData,
        /,
        *,
        args: Any = None,
        pressure: ArrayLike | None = None,
        wall_velocity: FaceVelocity | None = None,
        viscous_traction: ArrayLike | None = None,
        reference_point: ArrayLike | None = None,
    ) -> MACSharpInterfaceProjectionResult:
        source = jnp.asarray(
            self.jump_source(jnp.asarray(time), self.sharp.geometry, args),
            dtype=self.sharp.operators.pressure_space.dtype,
        )
        source = self.sharp.operators.validate_pressure(source)
        return self.sharp.project(
            velocity,
            inverse_momentum,
            boundary_stage,
            pressure=pressure,
            jump_source=source,
            wall_velocity=wall_velocity,
            viscous_traction=viscous_traction,
            reference_point=reference_point,
        )


MACInterfaceEnforcement = Literal[
    "regularized-delta", "divergence-free", "sharp", "immersed-interface"
]


class MACInterfaceMethodSelector(StrictModule, NonTrainableState):
    """One explicit enforcement-family selection for configuration and replay."""

    method: MACInterfaceEnforcement = eqx.field(static=True)
    plan: object
    selector_id: str = eqx.field(static=True)

    def __init__(
        self,
        method: MACInterfaceEnforcement,
        plan: object,
        /,
    ):
        if method not in (
            "regularized-delta",
            "divergence-free",
            "sharp",
            "immersed-interface",
        ):
            raise ValueError("Unknown MAC interface enforcement family.")
        from ._mac_dfib import MACDFIBProjectionPlan
        from ._mac_immersed_boundary import MACImmersedBoundaryProjectionPlan

        expected = {
            "regularized-delta": MACImmersedBoundaryProjectionPlan,
            "divergence-free": MACDFIBProjectionPlan,
            "sharp": MACSharpInterfaceProjectionPlan,
            "immersed-interface": MACImmersedInterfaceProjectionPlan,
        }[method]
        if not isinstance(plan, expected):
            raise TypeError("Selected MAC interface family and projection plan differ.")
        self.method = method
        self.plan = plan
        self.selector_id = canonical_fingerprint(
            {
                "kind": "mac-interface-method-selector",
                "method": method,
                "plan": plan.plan_id,
            }
        )


__all__ = [
    "MACImmersedInterfaceProjectionPlan",
    "MACInterfaceEnforcement",
    "MACInterfaceJumpSource",
    "MACInterfaceMethodSelector",
    "MACMovingSharpInterfaceEpochPlan",
    "MACMovingSharpInterfaceEpochResult",
    "MACSharpGeometryProvider",
    "MACSharpInterfaceForce",
    "MACSharpInterfaceGeometry",
    "MACSharpInterfaceGeometryData",
    "MACSharpInterfaceProjectionPlan",
    "MACSharpInterfaceProjectionResult",
    "MACSharpInterfaceStatus",
]
