#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._tree_math import tree_allfinite
from ...metrix._state_geometry import AbstractStateGeometry
from .._core import DiscretizationKey, DiscretizationRole, PreparationReport
from ._core import ParticleDiscretization


class RigidBodySetPlan(StrictModule, NonTrainableState):
    material_ids: Array
    inertia_body: Array
    fixed_mask: Array
    key: DiscretizationKey
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        material_ids: ArrayLike,
        inertia_body: ArrayLike,
        /,
        *,
        fixed_mask: ArrayLike | None = None,
        name: str = "rigid-bodies",
        plan_id: str | None = None,
    ):
        material = np.asarray(material_ids)
        inertia = np.asarray(inertia_body)
        if (
            material.ndim != 1
            or material.size == 0
            or not np.issubdtype(material.dtype, np.integer)
        ):
            raise TypeError("material_ids must be a nonempty rank-1 integer array.")
        count = material.size
        if inertia.shape not in ((count,), (count, 3, 3)):
            raise ValueError("inertia_body must have shape (N,) or (N,3,3).")
        if inertia.ndim == 1:
            valid_inertia = np.isfinite(inertia) & (inertia > 0.0)
        else:
            symmetric = np.allclose(inertia, np.swapaxes(inertia, -1, -2))
            eigenvalues = np.linalg.eigvalsh(inertia)
            valid_inertia = np.isfinite(inertia).all(axis=(-2, -1)) & (
                eigenvalues > 0.0
            ).all(axis=-1)
            if not symmetric:
                raise ValueError("Three-dimensional inertia tensors must be symmetric.")
        if not np.all(valid_inertia) or np.any(material < 0):
            raise ValueError("Rigid-body inertia and material IDs are invalid.")
        fixed = (
            np.zeros((count,), dtype=bool)
            if fixed_mask is None
            else np.asarray(fixed_mask, dtype=bool)
        )
        if fixed.shape != (count,):
            raise ValueError("fixed_mask must have body-capacity shape.")
        key = DiscretizationKey(
            name,
            DiscretizationRole.AUXILIARY,
            domain_labels=("material_point", "rigid_body"),
        )
        generated = canonical_fingerprint(
            {
                "kind": "rigid-body-set-plan",
                "values": array_tree_fingerprint(
                    {
                        "material_ids": material,
                        "inertia_body": inertia,
                        "fixed_mask": fixed,
                    }
                ),
                "key": key.key_id,
            }
        )
        self.material_ids = jnp.asarray(material, dtype=jnp.int32)
        self.inertia_body = jnp.asarray(inertia)
        self.fixed_mask = jnp.asarray(fixed)
        self.key = key
        self.plan_id = generated if plan_id is None else str(plan_id)
        if not self.plan_id:
            raise ValueError("plan_id must be nonempty.")

    def prepare(self, particles: ParticleDiscretization, /) -> PreparedRigidBodySet:
        return PreparedRigidBodySet(self, particles)


class PreparedRigidBodySet(StrictModule, NonTrainableState):
    plan: RigidBodySetPlan
    particles: ParticleDiscretization
    inertia_body: Array
    inverse_inertia_body: Array
    inverse_masses: Array
    material_ids: Array
    fixed_mask: Array
    key: DiscretizationKey
    preparation: PreparationReport
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: RigidBodySetPlan, particles: ParticleDiscretization, /):
        if not isinstance(plan, RigidBodySetPlan):
            raise TypeError("plan must be a RigidBodySetPlan.")
        if not isinstance(particles, ParticleDiscretization):
            raise TypeError("particles must be a ParticleDiscretization.")
        dimension = particles.ambient_dimension
        expected = (particles.capacity,) if dimension == 2 else (particles.capacity, 3, 3)
        if dimension not in (2, 3) or plan.inertia_body.shape != expected:
            raise ValueError(
                "Rigid-body inertia schema does not match dimension/capacity."
            )
        inertia_host = np.asarray(plan.inertia_body)
        inverse_host = (
            1.0 / inertia_host if dimension == 2 else np.linalg.inv(inertia_host)
        )
        active = particles.active_mask
        fixed = plan.fixed_mask & active
        mobile = active & ~fixed
        dtype = particles.safe_masses.dtype
        inertia = jnp.asarray(inertia_host, dtype=dtype)
        inverse_inertia = jnp.asarray(inverse_host, dtype=dtype)
        if dimension == 2:
            inverse_inertia = jnp.where(mobile, inverse_inertia, 0.0)
            inertia = jnp.where(active, inertia, 1.0)
        else:
            inverse_inertia = jnp.where(mobile[:, None, None], inverse_inertia, 0.0)
            identity = jnp.eye(3, dtype=dtype)
            inertia = jnp.where(active[:, None, None], inertia, identity)
        inverse_mass = jnp.where(mobile, 1.0 / particles.safe_masses, 0.0)
        preparation = PreparationReport(
            diagnostics=(
                "SO(2)/SO(3) rigid-body pose",
                "body-frame SPD inertia",
                "world-frame angular velocity",
            ),
            resource_counts={
                "body_capacity": particles.capacity,
                "active_bodies": particles.active_count,
                "ambient_dimension": dimension,
            },
        )
        self.plan = plan
        self.particles = particles
        self.inertia_body = inertia
        self.inverse_inertia_body = inverse_inertia
        self.inverse_masses = inverse_mass
        self.material_ids = jnp.where(active, plan.material_ids, 0)
        self.fixed_mask = fixed
        self.key = plan.key
        self.preparation = preparation
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-rigid-body-set",
                "plan": plan.plan_id,
                "particles": particles.prepared_id,
                "preparation": preparation.report_id,
            }
        )

    @property
    def capacity(self) -> int:
        return self.particles.capacity

    @property
    def ambient_dimension(self) -> int:
        return self.particles.ambient_dimension

    @property
    def angular_dimension(self) -> int:
        return 1 if self.ambient_dimension == 2 else 3

    @property
    def orientation_dimension(self) -> int:
        return 1 if self.ambient_dimension == 2 else 4

    def kinematics(
        self,
        position: ArrayLike,
        velocity: ArrayLike,
        orientation: ArrayLike,
        angular_velocity: ArrayLike,
        /,
    ) -> RigidBodyKinematics:
        position_ = jnp.asarray(position, dtype=self.particles.safe_masses.dtype)
        velocity_ = jnp.asarray(velocity, dtype=position_.dtype)
        orientation_ = jnp.asarray(orientation, dtype=position_.dtype)
        angular_ = jnp.asarray(angular_velocity, dtype=position_.dtype)
        if (
            position_.shape != (self.capacity, self.ambient_dimension)
            or velocity_.shape != position_.shape
        ):
            raise ValueError("Rigid-body position/velocity shape is invalid.")
        if orientation_.shape != (self.capacity, self.orientation_dimension):
            raise ValueError("Rigid-body orientation shape is invalid.")
        if angular_.shape != (self.capacity, self.angular_dimension):
            raise ValueError("Rigid-body angular-velocity shape is invalid.")
        if self.ambient_dimension == 3:
            orientation_ = _normalize_quaternion(orientation_)
        mobile = (self.particles.active_mask & ~self.fixed_mask)[:, None]
        velocity_ = jnp.where(mobile, velocity_, 0.0)
        angular_ = jnp.where(mobile, angular_, 0.0)
        return RigidBodyKinematics(position_, velocity_, orientation_, angular_)


class RigidBodyKinematics(StrictModule):
    position: Array
    velocity: Array
    orientation: Array
    angular_velocity: Array


class RigidBodyLoad(StrictModule):
    force: Array
    torque: Array


class RigidBodyStepResult(StrictModule):
    kinematics: RigidBodyKinematics
    load: RigidBodyLoad
    successful: Array


def _quaternion_multiply(left: Array, right: Array, /) -> Array:
    lw, lv = left[..., :1], left[..., 1:]
    rw, rv = right[..., :1], right[..., 1:]
    return jnp.concatenate(
        (
            lw * rw - jnp.sum(lv * rv, axis=-1, keepdims=True),
            lw * rv + rw * lv + jnp.cross(lv, rv),
        ),
        axis=-1,
    )


def _normalize_quaternion(value: Array, /) -> Array:
    norm = jnp.sqrt(jnp.sum(value * value, axis=-1, keepdims=True))
    normalized = value / jnp.maximum(norm, jnp.finfo(value.dtype).eps)
    sign = jnp.where(normalized[..., :1] < 0.0, -1.0, 1.0)
    return normalized * sign


def _quaternion_increment(rotation_vector: Array, /) -> Array:
    squared_angle = jnp.sum(rotation_vector * rotation_vector, axis=-1, keepdims=True)
    threshold = jnp.finfo(rotation_vector.dtype).eps
    safe_angle = jnp.sqrt(jnp.maximum(squared_angle, threshold))
    exact_scale = jnp.sin(0.5 * safe_angle) / safe_angle
    series_scale = 0.5 - squared_angle / 48.0 + squared_angle * squared_angle / 3840.0
    scale = jnp.where(squared_angle > threshold, exact_scale, series_scale)
    exact_scalar = jnp.cos(0.5 * safe_angle)
    series_scalar = 1.0 - squared_angle / 8.0 + squared_angle * squared_angle / 384.0
    scalar = jnp.where(squared_angle > threshold, exact_scalar, series_scalar)
    return jnp.concatenate((scalar, scale * rotation_vector), axis=-1)


def _quaternion_conjugate(value: Array, /) -> Array:
    return jnp.concatenate((value[..., :1], -value[..., 1:]), axis=-1)


def _quaternion_retract(value: Array, rotation_vector: Array, /) -> Array:
    return _normalize_quaternion(
        _quaternion_multiply(_quaternion_increment(rotation_vector), value)
    )


def _quaternion_relative_rotation_vector(reference: Array, point: Array, /) -> Array:
    relative = _normalize_quaternion(
        _quaternion_multiply(point, _quaternion_conjugate(reference))
    )
    vector = relative[..., 1:]
    squared_norm = jnp.sum(vector * vector, axis=-1, keepdims=True)
    threshold = jnp.finfo(relative.dtype).eps
    safe_norm = jnp.sqrt(jnp.maximum(squared_norm, threshold))
    exact_scale = 2.0 * jnp.arctan2(safe_norm, relative[..., :1]) / safe_norm
    series_scale = 2.0 + squared_norm / 3.0 + 3.0 * squared_norm * squared_norm / 20.0
    scale = jnp.where(squared_norm > threshold, exact_scale, series_scale)
    return scale * vector


def rigid_body_world_inertia(
    bodies: PreparedRigidBodySet,
    orientation: Array,
    /,
) -> tuple[Array, Array]:
    rotation = quaternion_rotation_matrix(orientation)
    inertia = contract(
        "...ij,...jk,...lk->...il",
        rotation,
        bodies.inertia_body,
        rotation,
    )
    inverse = contract(
        "...ij,...jk,...lk->...il",
        rotation,
        bodies.inverse_inertia_body,
        rotation,
    )
    return inertia, inverse


def _rigid_body_half_kick(
    bodies: PreparedRigidBodySet,
    kinematics: RigidBodyKinematics,
    load: RigidBodyLoad,
    step_size: Array,
    /,
) -> RigidBodyKinematics:
    mobile = (bodies.particles.active_mask & ~bodies.fixed_mask)[:, None]
    velocity = kinematics.velocity + 0.5 * step_size * (
        bodies.inverse_masses[:, None] * load.force
    )
    angular = kinematics.angular_velocity + 0.5 * step_size * (
        rigid_body_angular_acceleration(bodies, kinematics, load.torque)
    )
    return RigidBodyKinematics(
        kinematics.position,
        jnp.where(mobile, velocity, 0.0),
        kinematics.orientation,
        jnp.where(mobile, angular, 0.0),
    )


def rigid_body_drift(
    bodies: PreparedRigidBodySet,
    kinematics: RigidBodyKinematics,
    step_size: Array,
    /,
) -> RigidBodyKinematics:
    mobile = (bodies.particles.active_mask & ~bodies.fixed_mask)[:, None]
    position = jnp.where(
        mobile,
        kinematics.position + step_size * kinematics.velocity,
        kinematics.position,
    )
    if bodies.ambient_dimension == 2:
        orientation = kinematics.orientation + step_size * kinematics.angular_velocity
        orientation = (orientation + jnp.pi) % (2.0 * jnp.pi) - jnp.pi
    else:
        orientation = _quaternion_retract(
            kinematics.orientation,
            step_size * kinematics.angular_velocity,
        )
    return RigidBodyKinematics(
        position,
        kinematics.velocity,
        orientation,
        kinematics.angular_velocity,
    )


def _rigid_body_close_kick(
    bodies: PreparedRigidBodySet,
    kinematics: RigidBodyKinematics,
    load: RigidBodyLoad,
    step_size: Array,
    /,
) -> RigidBodyKinematics:
    mobile = (bodies.particles.active_mask & ~bodies.fixed_mask)[:, None]
    velocity = kinematics.velocity + 0.5 * step_size * (
        bodies.inverse_masses[:, None] * load.force
    )
    angular = kinematics.angular_velocity + 0.5 * step_size * (
        rigid_body_angular_acceleration(bodies, kinematics, load.torque)
    )
    return RigidBodyKinematics(
        kinematics.position,
        jnp.where(mobile, velocity, 0.0),
        kinematics.orientation,
        jnp.where(mobile, angular, 0.0),
    )


def _rigid_body_retract_pose(
    bodies: PreparedRigidBodySet,
    kinematics: RigidBodyKinematics,
    translation: Array,
    rotation: Array,
    /,
) -> RigidBodyKinematics:
    mobile = (bodies.particles.active_mask & ~bodies.fixed_mask)[:, None]
    return RigidBodyKinematics(
        jnp.where(mobile, kinematics.position + translation, kinematics.position),
        kinematics.velocity,
        jnp.where(
            mobile,
            _quaternion_retract(kinematics.orientation, rotation),
            kinematics.orientation,
        ),
        kinematics.angular_velocity,
    )


def quaternion_rotation_matrix(quaternion: Array, /) -> Array:
    q = _normalize_quaternion(quaternion)
    w, x, y, z = (q[..., index] for index in range(4))
    return jnp.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def rigid_body_angular_acceleration(
    bodies: PreparedRigidBodySet,
    kinematics: RigidBodyKinematics,
    torque: Array,
    /,
) -> Array:
    if bodies.ambient_dimension == 2:
        return bodies.inverse_inertia_body[:, None] * torque
    inertia_world, inverse_world = rigid_body_world_inertia(
        bodies, kinematics.orientation
    )
    angular_momentum = contract(
        "...ij,...j->...i", inertia_world, kinematics.angular_velocity
    )
    rhs = torque - jnp.cross(kinematics.angular_velocity, angular_momentum)
    return contract("...ij,...j->...i", inverse_world, rhs)


def rigid_body_kick_drift_kick(
    bodies: PreparedRigidBodySet,
    kinematics: RigidBodyKinematics,
    load: RigidBodyLoad,
    time: Array,
    step_size: Array,
    load_function: Callable[[Array, RigidBodyKinematics, Any], RigidBodyLoad],
    args: Any = None,
    /,
) -> RigidBodyStepResult:
    if not callable(load_function):
        raise TypeError("load_function must be callable.")
    half = _rigid_body_half_kick(bodies, kinematics, load, step_size)
    staged = rigid_body_drift(bodies, half, step_size)
    next_load = load_function(time + step_size, staged, args)
    if not isinstance(next_load, RigidBodyLoad):
        raise TypeError("load_function must return RigidBodyLoad.")
    result = _rigid_body_close_kick(bodies, staged, next_load, step_size)
    successful = (
        tree_allfinite(result)
        & jnp.all(jnp.isfinite(next_load.force))
        & jnp.all(jnp.isfinite(next_load.torque))
    )
    return RigidBodyStepResult(result, next_load, successful)


class RigidBodyStateGeometry(AbstractStateGeometry):
    bodies: PreparedRigidBodySet
    geometry_id: str = eqx.field(static=True)
    retraction_method: str = "rigid-body-lie-retraction"
    trivial: bool = False
    supports_exact_pullback: bool = False
    supports_commutator_free: bool = True

    def __init__(self, bodies: PreparedRigidBodySet, /):
        self.bodies = bodies
        self.geometry_id = f"state-geometry:rigid-body:{bodies.prepared_id}"

    def contains(self, state, /):
        finite = tree_allfinite(state)
        if self.bodies.ambient_dimension == 3:
            norm = jnp.sqrt(jnp.sum(state.orientation**2, axis=-1))
            finite = finite & jnp.all(jnp.abs(norm - 1.0) <= 1.0e-8)
        return finite

    def project_tangent(self, state, vector, /):
        del state
        return vector

    def to_local(self, state, tangent, /):
        del state
        return tangent

    def from_local(self, state, local_tangent, /):
        del state
        return local_tangent

    def retract(self, state, local_tangent, /):
        position = state.position + local_tangent.position
        velocity = state.velocity + local_tangent.velocity
        angular = state.angular_velocity + local_tangent.angular_velocity
        if self.bodies.ambient_dimension == 2:
            orientation = state.orientation + local_tangent.orientation
        else:
            orientation = _quaternion_retract(
                state.orientation,
                local_tangent.orientation[..., 1:],
            )
        return RigidBodyKinematics(position, velocity, orientation, angular)

    def inverse_retract(self, state, point, /):
        if self.bodies.ambient_dimension == 2:
            orientation = point.orientation - state.orientation
        else:
            rotation = _quaternion_relative_rotation_vector(
                state.orientation, point.orientation
            )
            orientation = jnp.concatenate(
                (jnp.zeros_like(rotation[..., :1]), rotation), axis=-1
            )
        return RigidBodyKinematics(
            point.position - state.position,
            point.velocity - state.velocity,
            orientation,
            point.angular_velocity - state.angular_velocity,
        )

    def pullback(self, state, local_tangent, tangent, /):
        del state, local_tangent
        return tangent


__all__ = [
    "PreparedRigidBodySet",
    "RigidBodyKinematics",
    "RigidBodyLoad",
    "RigidBodySetPlan",
    "RigidBodyStateGeometry",
    "RigidBodyStepResult",
    "quaternion_rotation_matrix",
    "rigid_body_angular_acceleration",
    "rigid_body_drift",
    "rigid_body_kick_drift_kick",
    "rigid_body_world_inertia",
]
