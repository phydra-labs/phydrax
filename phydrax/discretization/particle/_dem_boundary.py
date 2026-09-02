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
from ._dem_contact_state import DEMContactEvaluationContext
from ._dem_liquid import DEMBarrierCapillaryPlan
from ._pair_state import (
    IMPLICIT_BARRIER_INTERACTION,
    particle_wall_interaction_keys,
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


class DEMServoControlMode(StrEnum):
    FORCE = "force"
    TORQUE = "torque"


class ServoDEMBarrierState(StrictModule):
    displacement: Array
    velocity: Array
    integral_error: Array
    previous_error: Array
    command: Array
    effective_velocity_limit: Array
    saturation_margin: Array
    skin_margin: Array
    saturated: Array


class ServoDEMBarrierMotionPlan(AbstractDEMBarrierMotionPlan):
    axis: Array
    reference_point: Array
    target_value: float = eqx.field(static=True)
    proportional_gain: float = eqx.field(static=True)
    integral_gain: float = eqx.field(static=True)
    derivative_gain: float = eqx.field(static=True)
    velocity_limit: float = eqx.field(static=True)
    gain_schedule_ratio: float = eqx.field(static=True)
    minimum_gain_fraction: float = eqx.field(static=True)
    neighbor_skin: float | None = eqx.field(static=True)
    control_mode: DEMServoControlMode = eqx.field(static=True)
    geometry_function: Callable[[Any, Array], Any]
    motion_id: str = eqx.field(static=True)

    def __init__(
        self,
        axis: Array,
        target_value: float,
        /,
        *,
        proportional_gain: float,
        integral_gain: float = 0.0,
        derivative_gain: float = 0.0,
        velocity_limit: float,
        geometry_function: Callable[[Any, Array], Any],
        motion_id: str,
        control_mode: DEMServoControlMode = DEMServoControlMode.FORCE,
        reference_point: Array | None = None,
        gain_schedule_ratio: float = 1.0,
        minimum_gain_fraction: float = 0.1,
        neighbor_skin: float | None = None,
    ):
        axis_host = np.asarray(axis)
        if axis_host.ndim != 1 or axis_host.size not in (2, 3):
            raise ValueError("Servo axis must be a 2-D or 3-D vector.")
        if not isinstance(control_mode, DEMServoControlMode):
            raise TypeError("control_mode must be a DEMServoControlMode.")
        if control_mode is DEMServoControlMode.TORQUE and axis_host.size != 3:
            raise ValueError("Torque-controlled servo barriers require three dimensions.")
        norm = np.linalg.norm(axis_host)
        point = (
            np.zeros_like(axis_host)
            if reference_point is None
            else np.asarray(reference_point)
        )
        if point.shape != axis_host.shape or np.any(~np.isfinite(point)):
            raise ValueError("reference_point must match the finite servo axis.")
        values = tuple(
            float(value)
            for value in (
                target_value,
                proportional_gain,
                integral_gain,
                derivative_gain,
                velocity_limit,
                gain_schedule_ratio,
                minimum_gain_fraction,
            )
        )
        skin = None if neighbor_skin is None else float(neighbor_skin)
        if (
            not np.isfinite(norm)
            or norm <= 0.0
            or any(not np.isfinite(value) for value in values)
            or values[1] < 0.0
            or values[2] < 0.0
            or values[3] < 0.0
            or values[4] <= 0.0
            or values[5] <= 0.0
            or not 0.0 < values[6] <= 1.0
            or (skin is not None and (not np.isfinite(skin) or skin <= 0.0))
        ):
            raise ValueError("Servo axis, gains, target, and limits are invalid.")
        if not callable(geometry_function):
            raise TypeError("geometry_function must be callable.")
        identifier = str(motion_id)
        if not identifier:
            raise ValueError("motion_id must be nonempty.")
        self.axis = jnp.asarray(axis_host / norm)
        self.reference_point = jnp.asarray(point)
        self.target_value = values[0]
        self.proportional_gain = values[1]
        self.integral_gain = values[2]
        self.derivative_gain = values[3]
        self.velocity_limit = values[4]
        self.gain_schedule_ratio = values[5]
        self.minimum_gain_fraction = values[6]
        self.neighbor_skin = skin
        self.control_mode = control_mode
        self.geometry_function = geometry_function
        self.motion_id = identifier

    def initialize(self, dtype: Any = jnp.float64, /) -> ServoDEMBarrierState:
        zero = jnp.zeros((), dtype=dtype)
        return ServoDEMBarrierState(
            zero,
            zero,
            zero,
            zero,
            zero,
            jnp.asarray(self.velocity_limit, dtype=dtype),
            jnp.asarray(self.velocity_limit, dtype=dtype),
            jnp.asarray(jnp.inf, dtype=dtype),
            jnp.asarray(False),
        )

    def update(
        self,
        state: ServoDEMBarrierState,
        reaction_force: Array,
        step_size: Array,
        /,
        *,
        reaction_torque: Array | None = None,
        minimum_radius: Array | None = None,
    ) -> ServoDEMBarrierState:
        if not isinstance(state, ServoDEMBarrierState):
            raise TypeError("state must be a ServoDEMBarrierState.")
        dt = jnp.asarray(step_size)
        dt = eqx.error_if(
            dt,
            ~jnp.isfinite(dt) | (dt <= 0.0),
            "Servo step size must be finite and positive.",
        )
        if self.control_mode is DEMServoControlMode.FORCE:
            measured = jnp.sum(jnp.asarray(reaction_force) * self.axis)
        else:
            if reaction_torque is None:
                raise ValueError(
                    "Torque-controlled servo update requires reaction_torque."
                )
            measured = jnp.sum(jnp.asarray(reaction_torque) * self.axis)
        error = jnp.asarray(self.target_value, dtype=measured.dtype) - measured
        derivative = (error - state.previous_error) / dt
        proposed_integral = state.integral_error + dt * error
        proposed_command = (
            self.proportional_gain * error
            + self.integral_gain * proposed_integral
            + self.derivative_gain * derivative
        )
        target_scale = jnp.maximum(jnp.abs(self.target_value), 1.0)
        scheduled_fraction = jnp.clip(
            self.minimum_gain_fraction
            + self.gain_schedule_ratio * jnp.abs(error) / target_scale,
            self.minimum_gain_fraction,
            1.0,
        )
        effective_limit = self.velocity_limit * scheduled_fraction
        if minimum_radius is not None:
            radius_limit = self.gain_schedule_ratio * jnp.asarray(minimum_radius) / dt
            effective_limit = jnp.minimum(effective_limit, radius_limit)
        skin_margin = jnp.asarray(jnp.inf, dtype=measured.dtype)
        if self.neighbor_skin is not None:
            skin_limit = self.neighbor_skin / (2.0 * dt)
            effective_limit = jnp.minimum(effective_limit, skin_limit)
            skin_margin = skin_limit - jnp.abs(proposed_command)
        velocity = jnp.clip(proposed_command, -effective_limit, effective_limit)
        saturated = jnp.abs(proposed_command) > effective_limit
        drives_inward = jnp.sign(error) != jnp.sign(proposed_command)
        integral = jnp.where(
            ~saturated | drives_inward,
            proposed_integral,
            state.integral_error,
        )
        command = (
            self.proportional_gain * error
            + self.integral_gain * integral
            + self.derivative_gain * derivative
        )
        velocity = jnp.clip(command, -effective_limit, effective_limit)
        return ServoDEMBarrierState(
            state.displacement + dt * velocity,
            velocity,
            integral,
            error,
            command,
            effective_limit,
            effective_limit - jnp.abs(command),
            skin_margin,
            saturated,
        )

    def evaluate(self, geometry, time, points, args, /):
        del time
        if not isinstance(args, ServoDEMBarrierState):
            raise TypeError("Servo barrier motion requires ServoDEMBarrierState args.")
        moved_geometry = self.geometry_function(geometry, args.displacement)
        dimension = points.shape[-1]
        angular_dimension = 1 if dimension == 2 else 3
        if self.control_mode is DEMServoControlMode.FORCE:
            linear_velocity = args.velocity * self.axis
            angular_velocity = jnp.zeros((angular_dimension,), dtype=points.dtype)
        else:
            linear_velocity = jnp.zeros((dimension,), dtype=points.dtype)
            angular_velocity = args.velocity * self.axis
        valid = jnp.all(
            jnp.isfinite(
                jnp.stack(
                    (
                        args.displacement,
                        args.velocity,
                        args.integral_error,
                        args.previous_error,
                        args.command,
                        args.effective_velocity_limit,
                        args.saturation_margin,
                        args.skin_margin,
                    )
                )
            )
        )
        return DEMBarrierMotion(
            moved_geometry,
            linear_velocity,
            angular_velocity,
            self.reference_point,
            valid,
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

    @property
    def capillary_births(self) -> Array:
        return self.contact.cohesion_births

    @property
    def capillary_ruptures(self) -> Array:
        return self.contact.cohesion_ruptures

    @property
    def capillary_bridge_volume(self) -> Array:
        components = self.contact.next_history.cohesion.components
        if not components:
            return jnp.zeros_like(
                self.contact.cohesion_births, dtype=self.particle_force.dtype
            )
        return jnp.sum(
            jnp.stack(tuple(value.bridge_volume for value in components)), axis=0
        )

    @property
    def capillary_surface_area(self) -> Array:
        return self.contact.bridge_surface_area

    @property
    def capillary_fit_margin(self) -> Array:
        return jnp.minimum(
            self.contact.cohesion_model_validity_margin,
            self.contact.cohesion_fit_extrapolation_margin,
        )


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
    body_properties: Any = None,
    normal_tolerance: float = 1.0e-12,
    capillary_plan: DEMBarrierCapillaryPlan | None = None,
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
    if capillary_plan is not None:
        if not isinstance(capillary_plan, DEMBarrierCapillaryPlan):
            raise TypeError("capillary_plan must be DEMBarrierCapillaryPlan or None.")
        if capillary_plan.barrier_id != barrier.barrier_id:
            raise ValueError("Barrier capillary binding does not match the barrier.")
    tolerance = float(normal_tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("normal_tolerance must be finite and positive.")
    position = kinematics.position
    radii = bodies.radii if body_properties is None else body_properties.radii
    inverse_masses = (
        bodies.inverse_masses
        if body_properties is None
        else body_properties.inverse_masses
    )
    active_body_mask = (
        bodies.particles.active_mask
        if body_properties is None
        else body_properties.active
    )
    if (
        radii.shape != bodies.radii.shape
        or inverse_masses.shape != bodies.inverse_masses.shape
        or active_body_mask.shape != bodies.particles.active_mask.shape
    ):
        raise ValueError("Dynamic barrier body properties have invalid shapes.")
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
    overlap = jnp.maximum(radii - clearance, 0.0)
    active_particles = active_body_mask
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
    requires_curvature = isinstance(
        contact_model.plan.normal, HertzNormalContactPlan
    ) or (
        capillary_plan is not None
        and capillary_plan.geometry_policy == "isotropic_curvature"
    )
    if requires_curvature:
        capabilities = {value.value for value in geometry.capabilities}
        if "contact_curvature" not in capabilities:
            raise ValueError(
                "Hertz/isotropic-capillary barrier contact requires certified curvature."
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
        if (
            capillary_plan is not None
            and capillary_plan.geometry_policy == "isotropic_curvature"
        ):
            effective_radius, local_margin, curvature_valid = (
                capillary_plan.effective_radius(
                    radii,
                    signed_curvature,
                    isotropy_defect,
                    curvature.valid & isotropic,
                    tolerance=tolerance,
                )
            )
            local_margin = jnp.minimum(curvature.regularity_margin, local_margin)
        else:
            curvature_denominator = 1.0 / radii + signed_curvature
            curvature_valid = (
                curvature.valid
                & isotropic
                & jnp.isfinite(curvature_denominator)
                & (curvature_denominator > tolerance)
            )
            effective_radius = jnp.where(
                curvature_valid, 1.0 / curvature_denominator, 0.0
            )
            local_margin = jnp.minimum(
                curvature.regularity_margin,
                jnp.minimum(
                    curvature_denominator,
                    tolerance - isotropy_defect,
                ),
            )
        curvature_margin = jnp.min(jnp.where(active_particles, local_margin, jnp.inf))
        degenerate = degenerate | (active_particles & (overlap > 0.0) & ~curvature_valid)
        valid = active_particles & ~degenerate
    else:
        effective_radius = radii
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
        jnp.where(valid, clearance - radii, 0.0),
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
    route_ids = jnp.zeros((bodies.capacity,), dtype=jnp.int64)
    keys = particle_wall_interaction_keys(
        bodies.particles.particle_ids,
        route_ids,
        route_ids,
        route_ids,
        active_particles,
        interaction_kind=IMPLICIT_BARRIER_INTERACTION,
    )
    continued = (
        previous_history.valid
        & jnp.all(previous_history.pair_keys == keys, axis=-1)
        & previous_history.active
    )
    barrier_material = jnp.full((bodies.capacity,), barrier.material_id, dtype=jnp.int32)
    response = contact_model.evaluate(
        batch,
        previous_history,
        DEMContactEvaluationContext(
            keys,
            active_particles,
            continued,
            inverse_masses,
            jnp.zeros_like(inverse_masses),
            radii,
            radii,
            bodies.material_ids,
            barrier_material,
            step_size,
            -jnp.ones((), dtype=jnp.int32),
        ),
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
    "AbstractDEMBarrierMotionPlan",
    "DEMBarrierMotion",
    "DEMBarrierSide",
    "DEMBoundaryResponse",
    "DEMServoControlMode",
    "ImplicitDEMBarrier",
    "PrescribedDEMBarrierMotionPlan",
    "ServoDEMBarrierMotionPlan",
    "ServoDEMBarrierState",
    "StaticDEMBarrierMotionPlan",
    "evaluate_dem_barrier",
]
