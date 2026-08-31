#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from collections.abc import Callable
from enum import StrEnum
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import AbstractAttribute, StrictModule
from ..._trainable import NonTrainableState
from ._dem_contact import (
    DEMContactBatch,
    DEMContactHistory,
    DEMContactResponse,
    HertzNormalContactPlan,
    PreparedDEMContactModel,
)
from ._rigid_sphere import (
    PreparedRigidSphereSet,
    RigidSphereKinematics,
    sphere_lever_torque,
    sphere_spin_velocity,
)


class DEMBarrierMotion(StrictModule):
    geometry: Any
    linear_velocity: Array
    angular_velocity: Array
    reference_point: Array
    valid: Array


class AbstractDEMBarrierMotionPlan(StrictModule, NonTrainableState):
    motion_id: AbstractAttribute[str]

    @abc.abstractmethod
    def evaluate(
        self,
        geometry: Any,
        time: Array,
        points: Array,
        args: Any,
        /,
    ) -> DEMBarrierMotion:
        raise NotImplementedError


class StaticDEMBarrierMotionPlan(AbstractDEMBarrierMotionPlan):
    motion_id: str = "dem-barrier-motion:static"

    def evaluate(self, geometry, time, points, args, /):
        del time, args
        dimension = points.shape[-1]
        angular_dimension = 1 if dimension == 2 else 3
        return DEMBarrierMotion(
            geometry,
            jnp.zeros((dimension,), dtype=points.dtype),
            jnp.zeros((angular_dimension,), dtype=points.dtype),
            jnp.zeros((dimension,), dtype=points.dtype),
            jnp.asarray(True),
        )


class PrescribedDEMBarrierMotionPlan(AbstractDEMBarrierMotionPlan):
    motion_function: Callable[[Any, Array, Array, Any], DEMBarrierMotion]
    motion_id: str = eqx.field(static=True)

    def __init__(
        self,
        motion_function: Callable[[Any, Array, Array, Any], DEMBarrierMotion],
        /,
        *,
        motion_id: str,
    ):
        if not callable(motion_function):
            raise TypeError("motion_function must be callable.")
        identifier = str(motion_id)
        if not identifier:
            raise ValueError("motion_id must be nonempty.")
        self.motion_function = motion_function
        self.motion_id = identifier

    def evaluate(self, geometry, time, points, args, /):
        result = self.motion_function(geometry, time, points, args)
        if not isinstance(result, DEMBarrierMotion):
            raise TypeError("motion_function must return DEMBarrierMotion.")
        return result


class ServoDEMBarrierState(StrictModule):
    displacement: Array
    velocity: Array
    integral_error: Array
    previous_error: Array
    saturated: Array


class ServoDEMBarrierMotionPlan(AbstractDEMBarrierMotionPlan):
    axis: Array
    target_force: float = eqx.field(static=True)
    proportional_gain: float = eqx.field(static=True)
    integral_gain: float = eqx.field(static=True)
    derivative_gain: float = eqx.field(static=True)
    velocity_limit: float = eqx.field(static=True)
    geometry_function: Callable[[Any, Array], Any]
    motion_id: str = eqx.field(static=True)

    def __init__(
        self,
        axis: Array,
        target_force: float,
        /,
        *,
        proportional_gain: float,
        integral_gain: float = 0.0,
        derivative_gain: float = 0.0,
        velocity_limit: float,
        geometry_function: Callable[[Any, Array], Any],
        motion_id: str,
    ):
        axis_host = np.asarray(axis)
        if axis_host.ndim != 1 or axis_host.size not in (2, 3):
            raise ValueError("Servo axis must be a 2-D or 3-D vector.")
        norm = np.linalg.norm(axis_host)
        values = tuple(
            float(value)
            for value in (
                target_force,
                proportional_gain,
                integral_gain,
                derivative_gain,
                velocity_limit,
            )
        )
        if (
            not np.isfinite(norm)
            or norm <= 0.0
            or any(not np.isfinite(value) for value in values)
            or values[1] < 0.0
            or values[2] < 0.0
            or values[3] < 0.0
            or values[4] <= 0.0
        ):
            raise ValueError("Servo axis, gains, target, and velocity limit are invalid.")
        if not callable(geometry_function):
            raise TypeError("geometry_function must be callable.")
        identifier = str(motion_id)
        if not identifier:
            raise ValueError("motion_id must be nonempty.")
        self.axis = jnp.asarray(axis_host / norm)
        self.target_force = values[0]
        self.proportional_gain = values[1]
        self.integral_gain = values[2]
        self.derivative_gain = values[3]
        self.velocity_limit = values[4]
        self.geometry_function = geometry_function
        self.motion_id = identifier

    def initialize(self, dtype: Any = jnp.float64, /) -> ServoDEMBarrierState:
        zero = jnp.zeros((), dtype=dtype)
        return ServoDEMBarrierState(zero, zero, zero, zero, jnp.asarray(False))

    def update(
        self,
        state: ServoDEMBarrierState,
        reaction_force: Array,
        step_size: Array,
        /,
    ) -> ServoDEMBarrierState:
        if not isinstance(state, ServoDEMBarrierState):
            raise TypeError("state must be a ServoDEMBarrierState.")
        measured = jnp.sum(jnp.asarray(reaction_force) * self.axis)
        error = jnp.asarray(self.target_force, dtype=measured.dtype) - measured
        integral = state.integral_error + step_size * error
        derivative = (error - state.previous_error) / jnp.asarray(step_size)
        command = (
            self.proportional_gain * error
            + self.integral_gain * integral
            + self.derivative_gain * derivative
        )
        velocity = jnp.clip(command, -self.velocity_limit, self.velocity_limit)
        saturated = jnp.abs(command) > self.velocity_limit
        integral = jnp.where(saturated, state.integral_error, integral)
        return ServoDEMBarrierState(
            state.displacement + step_size * velocity,
            velocity,
            integral,
            error,
            saturated,
        )

    def evaluate(self, geometry, time, points, args, /):
        del time
        if not isinstance(args, ServoDEMBarrierState):
            raise TypeError("Servo barrier motion requires ServoDEMBarrierState args.")
        moved_geometry = self.geometry_function(geometry, args.displacement)
        dimension = points.shape[-1]
        angular_dimension = 1 if dimension == 2 else 3
        return DEMBarrierMotion(
            moved_geometry,
            args.velocity * self.axis,
            jnp.zeros((angular_dimension,), dtype=points.dtype),
            jnp.zeros((dimension,), dtype=points.dtype),
            jnp.all(
                jnp.isfinite(
                    jnp.stack(
                        (
                            args.displacement,
                            args.velocity,
                            args.integral_error,
                            args.previous_error,
                        )
                    )
                )
            ),
        )


class DEMBarrierSide(StrEnum):
    INTERIOR = "interior"
    EXTERIOR = "exterior"


class ImplicitDEMBarrier(StrictModule):
    """Exact-SDF barrier with explicit prescribed kinematics."""

    geometry: Any
    side: DEMBarrierSide = eqx.field(static=True)
    motion: AbstractDEMBarrierMotionPlan
    material_id: int = eqx.field(static=True)
    barrier_id: str = eqx.field(static=True)

    def __init__(
        self,
        geometry: Any,
        side: DEMBarrierSide,
        material_id: int,
        /,
        *,
        barrier_id: str,
        motion: AbstractDEMBarrierMotionPlan | None = None,
    ):
        if not isinstance(side, DEMBarrierSide):
            raise TypeError("side must be a DEMBarrierSide.")
        capabilities = {value.value for value in geometry.capabilities}
        if "signed_distance" not in capabilities:
            raise ValueError("DEM barrier geometry must provide signed distance.")
        if "boundary_normal" not in capabilities:
            raise ValueError("DEM barrier geometry must provide boundary normals.")
        material = int(material_id)
        if material < 0:
            raise ValueError("Barrier material_id must be nonnegative.")
        identifier = str(barrier_id)
        if not identifier:
            raise ValueError("barrier_id must be nonempty.")
        certificate = geometry.field_certificate
        if certificate.sign_reliability.value != "reliable":
            raise ValueError("DEM barrier requires a reliable signed-distance sign.")
        if certificate.distance_semantics.value != "exact_signed_distance":
            raise ValueError("DEM barrier requires exact signed-distance semantics.")
        motion_ = StaticDEMBarrierMotionPlan() if motion is None else motion
        if not isinstance(motion_, AbstractDEMBarrierMotionPlan):
            raise TypeError("motion must be an AbstractDEMBarrierMotionPlan or None.")
        self.geometry = geometry
        self.side = side
        self.motion = motion_
        self.material_id = material
        self.barrier_id = canonical_fingerprint(
            {
                "kind": "implicit-dem-barrier",
                "user_id": identifier,
                "side": side.value,
                "material_id": material,
                "ambient_dimension": geometry.ambient_dimension,
                "motion": motion_.motion_id,
                "field_certificate": {
                    "distance": certificate.distance_semantics.value,
                    "sign": certificate.sign_reliability.value,
                    "validity_region": certificate.validity_region,
                },
                "schema": repr(geometry.schema),
            }
        )


class DEMBoundaryResponse(StrictModule):
    particle_force: Array
    particle_torque: Array
    reaction_force: Array
    reaction_torque: Array
    contact: DEMContactResponse
    wall_contact_velocity: Array
    wall_angular_velocity: Array
    wall_power: Array
    curvature_margin: Array
    successful: Array
    material_id: int = eqx.field(static=True)
    barrier_id: str = eqx.field(static=True)


def evaluate_dem_barrier(
    barrier: ImplicitDEMBarrier,
    bodies: PreparedRigidSphereSet,
    kinematics: RigidSphereKinematics,
    contact_model: PreparedDEMContactModel,
    previous_history: DEMContactHistory,
    step_size: Array,
    /,
    *,
    time: Array = jnp.asarray(0.0),
    args: Any = None,
    normal_tolerance: float = 1.0e-12,
    frame_tolerance: float = 1.0e-10,
) -> DEMBoundaryResponse:
    """Evaluate one exact-SDF barrier with explicit contact-point motion."""

    if not isinstance(barrier, ImplicitDEMBarrier):
        raise TypeError("barrier must be an ImplicitDEMBarrier.")
    if not isinstance(bodies, PreparedRigidSphereSet):
        raise TypeError("bodies must be a PreparedRigidSphereSet.")
    if not isinstance(kinematics, RigidSphereKinematics):
        raise TypeError("kinematics must be RigidSphereKinematics.")
    if not isinstance(contact_model, PreparedDEMContactModel):
        raise TypeError("contact_model must be a PreparedDEMContactModel.")
    tolerance = float(normal_tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("normal_tolerance must be finite and positive.")
    position = kinematics.position
    motion = barrier.motion.evaluate(barrier.geometry, jnp.asarray(time), position, args)
    if not isinstance(motion, DEMBarrierMotion):
        raise TypeError("Barrier motion must return DEMBarrierMotion.")
    geometry = motion.geometry
    if geometry.ambient_dimension != bodies.ambient_dimension:
        raise ValueError("DEM barrier dimension does not match rigid spheres.")

    def broadcast(name, value, width):
        array = jnp.asarray(value, dtype=position.dtype)
        if array.shape == (width,):
            return jnp.broadcast_to(array, (bodies.capacity, width))
        if array.shape != (bodies.capacity, width):
            raise ValueError(
                f"Barrier {name} must have shape {(width,)} or "
                f"{(bodies.capacity, width)}."
            )
        return array

    linear_velocity = broadcast(
        "linear_velocity", motion.linear_velocity, bodies.ambient_dimension
    )
    angular_velocity = broadcast(
        "angular_velocity", motion.angular_velocity, bodies.angular_dimension
    )
    reference_point = broadcast(
        "reference_point", motion.reference_point, bodies.ambient_dimension
    )
    signed_distance = geometry.signed_distance(position)
    outward = geometry.boundary_normal(position)
    if signed_distance.shape != (bodies.capacity,) or outward.shape != position.shape:
        raise ValueError("DEM barrier geometry returned incompatible field shapes.")
    if barrier.side is DEMBarrierSide.INTERIOR:
        clearance = -signed_distance
        allowed_normal = -outward
    else:
        clearance = signed_distance
        allowed_normal = outward
    normal_norm = jnp.linalg.norm(allowed_normal, axis=-1)
    safe_norm = jnp.where(normal_norm > tolerance, normal_norm, 1.0)
    normal = allowed_normal / safe_norm[:, None]
    overlap = jnp.maximum(bodies.radii - clearance, 0.0)
    active_particles = bodies.particles.active_mask
    degenerate = (
        active_particles
        & (overlap > 0.0)
        & (
            (normal_norm <= tolerance)
            | ~jnp.isfinite(clearance)
            | ~jnp.all(jnp.isfinite(normal), axis=-1)
        )
    )
    valid = active_particles & ~degenerate
    arm = -clearance[:, None] * normal
    contact_point = position + arm
    if isinstance(contact_model.plan.normal, HertzNormalContactPlan):
        capabilities = {value.value for value in geometry.capabilities}
        if "contact_curvature" not in capabilities:
            raise ValueError(
                "Hertz barrier contact requires certified contact curvature."
            )
        curvature = geometry.contact_curvature(contact_point)
        principal = curvature.principal_curvatures
        if principal.shape[1] == 2:
            isotropy_defect = jnp.abs(principal[:, 0] - principal[:, 1])
            isotropic = isotropy_defect <= tolerance
            wall_curvature = 0.5 * (principal[:, 0] + principal[:, 1])
        else:
            isotropy_defect = jnp.zeros_like(clearance)
            isotropic = jnp.ones_like(active_particles)
            wall_curvature = principal[:, 0]
        signed_curvature = (
            -wall_curvature if barrier.side is DEMBarrierSide.INTERIOR else wall_curvature
        )
        curvature_denominator = 1.0 / bodies.radii + signed_curvature
        curvature_valid = (
            curvature.valid
            & isotropic
            & jnp.isfinite(curvature_denominator)
            & (curvature_denominator > tolerance)
        )
        effective_radius = jnp.where(curvature_valid, 1.0 / curvature_denominator, 0.0)
        curvature_margin = jnp.min(
            jnp.where(
                active_particles,
                jnp.minimum(
                    curvature.regularity_margin,
                    jnp.minimum(
                        curvature_denominator,
                        tolerance - isotropy_defect,
                    ),
                ),
                jnp.inf,
            )
        )
        degenerate = degenerate | (active_particles & (overlap > 0.0) & ~curvature_valid)
        valid = active_particles & ~degenerate
    else:
        effective_radius = bodies.radii
        curvature_margin = jnp.asarray(jnp.inf, dtype=position.dtype)
    particle_contact_velocity = kinematics.velocity + sphere_spin_velocity(
        kinematics.angular_velocity, arm, bodies.ambient_dimension
    )
    wall_contact_velocity = linear_velocity + sphere_spin_velocity(
        angular_velocity,
        contact_point - reference_point,
        bodies.ambient_dimension,
    )
    relative_velocity = particle_contact_velocity - wall_contact_velocity
    normal_velocity = jnp.sum(relative_velocity * normal, axis=-1)
    tangential_velocity = relative_velocity - normal_velocity[:, None] * normal
    batch = DEMContactBatch(
        jnp.where(valid[:, None], normal, 0.0),
        jnp.where(valid, clearance - bodies.radii, 0.0),
        jnp.where(valid, overlap, 0.0),
        jnp.where(valid, effective_radius, 0.0),
        jnp.where(valid[:, None], arm, 0.0),
        jnp.zeros_like(arm),
        jnp.where(valid, normal_velocity, 0.0),
        jnp.where(valid[:, None], tangential_velocity, 0.0),
        kinematics.angular_velocity,
        angular_velocity,
        valid,
    )
    keys = jnp.arange(bodies.capacity, dtype=jnp.int64)
    continued = (
        previous_history.valid
        & (previous_history.pair_keys == keys)
        & previous_history.active
    )
    barrier_material = jnp.full((bodies.capacity,), barrier.material_id, dtype=jnp.int32)
    response = contact_model.evaluate(
        batch,
        previous_history,
        keys,
        active_particles,
        continued,
        bodies.inverse_masses,
        jnp.zeros_like(bodies.inverse_masses),
        bodies.radii,
        bodies.radii,
        bodies.material_ids,
        barrier_material,
        step_size,
        frame_tolerance=frame_tolerance,
    )
    particle_force = response.pair_force
    particle_torque = response.left_torque
    reaction_force = -jnp.sum(particle_force, axis=0)
    reaction_torque = jnp.sum(
        sphere_lever_torque(contact_point, -particle_force, bodies.ambient_dimension)
        + response.right_torque,
        axis=0,
    )
    wall_power = jnp.sum(particle_force * wall_contact_velocity) - jnp.sum(
        response.right_torque * angular_velocity
    )
    successful = (
        response.successful
        & ~jnp.any(degenerate)
        & jnp.asarray(motion.valid, dtype=bool)
        & jnp.all(jnp.isfinite(wall_contact_velocity))
        & jnp.isfinite(wall_power)
        & (curvature_margin > 0.0)
    )
    return DEMBoundaryResponse(
        particle_force,
        particle_torque,
        reaction_force,
        reaction_torque,
        response,
        wall_contact_velocity,
        angular_velocity,
        wall_power,
        curvature_margin,
        successful,
        barrier.material_id,
        barrier.barrier_id,
    )


__all__ = [
    "DEMBarrierSide",
    "DEMBoundaryResponse",
    "ImplicitDEMBarrier",
    "evaluate_dem_barrier",
]
