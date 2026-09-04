#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._tree_math import tree_allfinite
from ...linalg import DualSpace, PyTreeSpace
from ...metrix._quaternion_state_geometry import ScalarFirstQuaternionStateGeometry
from ...metrix._state_geometry import AbstractStateGeometry
from .._core import DiscretizationKey, DiscretizationRole, PreparationReport
from ._core import ParticleDiscretization, ParticleSetPlan


class RigidBodySetPlan(StrictModule, NonTrainableState):
    """Rigid-body material and COM-inertia data for COM-centred kinematics."""

    material_ids: Array
    inertia_com: Array
    fixed_mask: Array
    key: DiscretizationKey
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        material_ids: ArrayLike,
        inertia_com: ArrayLike,
        /,
        *,
        fixed_mask: ArrayLike | None = None,
        name: str = "rigid-bodies",
        plan_id: str | None = None,
    ):
        material = np.asarray(material_ids)
        inertia = np.asarray(inertia_com)
        if (
            material.ndim != 1
            or material.size == 0
            or not np.issubdtype(material.dtype, np.integer)
        ):
            raise TypeError("material_ids must be a nonempty rank-1 integer array.")
        count = material.size
        if inertia.shape not in ((count,), (count, 3, 3)):
            raise ValueError("inertia_com must have shape (N,) or (N,3,3).")
        if inertia.ndim == 1:
            valid_inertia = np.isfinite(inertia) & (inertia > 0.0)
        else:
            symmetric = np.allclose(inertia, np.swapaxes(inertia, -1, -2))
            eigenvalues = np.linalg.eigvalsh(inertia)
            valid_inertia = np.isfinite(inertia).all(axis=(-2, -1)) & (
                eigenvalues > 0.0
            ).all(axis=-1)
            if not symmetric:
                raise ValueError(
                    "Three-dimensional COM inertia tensors must be symmetric."
                )
        if not np.all(valid_inertia) or np.any(material < 0):
            raise ValueError("Rigid-body COM inertia and material IDs are invalid.")
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
                        "inertia_com": inertia,
                        "fixed_mask": fixed,
                    }
                ),
                "key": key.key_id,
            }
        )
        self.material_ids = jnp.asarray(material, dtype=jnp.int32)
        self.inertia_com = jnp.asarray(inertia)
        self.fixed_mask = jnp.asarray(fixed)
        self.key = key
        self.plan_id = generated if plan_id is None else str(plan_id)
        if not self.plan_id:
            raise ValueError("plan_id must be nonempty.")

    @property
    def inertia_body(self) -> Array:
        """Body-coordinate inertia about the centre of mass."""

        return self.inertia_com

    def prepare(self, particles: ParticleDiscretization, /) -> PreparedRigidBodySet:
        return PreparedRigidBodySet(self, particles)


class RigidBodyMassProperties(StrictModule, NonTrainableState):
    """Prepared COM-centred mass properties shared by maximal and reduced paths.

    Linear velocity is the centre-of-mass velocity, so the body-frame first
    moment is identically zero and ``inertia_com`` is the rotational block of
    the spatial inertia.
    """

    masses: Array
    inverse_masses: Array
    first_moments: Array
    inertia_com: Array
    inverse_inertia_com: Array
    properties_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: RigidBodySetPlan,
        particles: ParticleDiscretization,
        /,
    ):
        dimension = particles.ambient_dimension
        expected = (particles.capacity,) if dimension == 2 else (particles.capacity, 3, 3)
        if dimension not in (2, 3) or plan.inertia_com.shape != expected:
            raise ValueError(
                "Rigid-body COM inertia schema does not match dimension/capacity."
            )
        inertia_host = np.asarray(plan.inertia_com)
        inverse_host = (
            1.0 / inertia_host
            if dimension == 2
            else np.linalg.solve(
                inertia_host,
                np.eye(3, dtype=inertia_host.dtype),
            )
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
        self.masses = particles.safe_masses
        self.inverse_masses = jnp.where(mobile, 1.0 / particles.safe_masses, 0.0)
        self.first_moments = jnp.zeros((particles.capacity, dimension), dtype=dtype)
        self.inertia_com = inertia
        self.inverse_inertia_com = inverse_inertia
        self.properties_id = canonical_fingerprint(
            {
                "kind": "rigid-body-com-mass-properties",
                "plan": plan.plan_id,
                "particles": particles.prepared_id,
            }
        )


class PreparedRigidBodySet(StrictModule, NonTrainableState):
    plan: RigidBodySetPlan
    particles: ParticleDiscretization
    mass_properties: RigidBodyMassProperties
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
        mass_properties = RigidBodyMassProperties(plan, particles)
        active = particles.active_mask
        fixed = plan.fixed_mask & active
        preparation = PreparationReport(
            diagnostics=(
                "SO(2)/SO(3) COM-centred rigid-body pose",
                "body-frame SPD COM inertia",
                "world-frame angular velocity",
                "zero body-frame first moment",
            ),
            resource_counts={
                "body_capacity": particles.capacity,
                "active_bodies": particles.active_count,
                "ambient_dimension": particles.ambient_dimension,
            },
        )
        self.plan = plan
        self.particles = particles
        self.mass_properties = mass_properties
        self.material_ids = jnp.where(active, plan.material_ids, 0)
        self.fixed_mask = fixed
        self.key = plan.key
        self.preparation = preparation
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-rigid-body-set",
                "plan": plan.plan_id,
                "particles": particles.prepared_id,
                "mass_properties": mass_properties.properties_id,
                "preparation": preparation.report_id,
            }
        )

    @property
    def inertia_com(self) -> Array:
        return self.mass_properties.inertia_com

    @property
    def inverse_inertia_com(self) -> Array:
        return self.mass_properties.inverse_inertia_com

    @property
    def inverse_masses(self) -> Array:
        return self.mass_properties.inverse_masses

    @property
    def inertia_body(self) -> Array:
        """Body-coordinate inertia about the centre of mass."""

        return self.inertia_com

    @property
    def inverse_inertia_body(self) -> Array:
        """Inverse body-coordinate inertia about the centre of mass."""

        return self.inverse_inertia_com

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
    """Rigid-body pose and twist expressed at each centre of mass."""

    position: Array
    velocity: Array
    orientation: Array
    angular_velocity: Array


class RigidBodyReferenceFrameRebase(StrictModule, NonTrainableState):
    """Explicit old-body-origin to COM-centred reference-frame transfer."""

    center_of_mass_offsets: Array
    body_ids: Array
    source_prepared_id: str = eqx.field(static=True)
    source_particle_plan_id: str = eqx.field(static=True)
    source_body_plan_id: str = eqx.field(static=True)
    target_particle_plan_id: str = eqx.field(static=True)
    target_body_plan_id: str = eqx.field(static=True)
    rebase_id: str = eqx.field(static=True)

    def __init__(
        self,
        source: PreparedRigidBodySet,
        target_particles: ParticleSetPlan,
        target_bodies: RigidBodySetPlan,
        center_of_mass_offsets: ArrayLike,
        /,
    ):
        if not isinstance(source, PreparedRigidBodySet):
            raise TypeError("source must be a PreparedRigidBodySet.")
        if not isinstance(target_particles, ParticleSetPlan):
            raise TypeError("target_particles must be a ParticleSetPlan.")
        if not isinstance(target_bodies, RigidBodySetPlan):
            raise TypeError("target_bodies must be a RigidBodySetPlan.")
        if source.ambient_dimension != 3 or target_particles.ambient_dimension != 3:
            raise ValueError("Rigid-body reference-frame rebasing requires 3-D bodies.")
        offsets = np.asarray(center_of_mass_offsets)
        expected = (source.capacity, 3)
        if offsets.shape != expected or not np.all(np.isfinite(offsets)):
            raise ValueError(
                "center_of_mass_offsets must be finite with source body-capacity shape."
            )
        source_ids = np.asarray(source.particles.particle_ids)
        target_ids = np.asarray(target_particles.particle_ids)
        if (
            target_particles.particle_ids.shape != (source.capacity,)
            or target_bodies.material_ids.shape != (source.capacity,)
            or target_bodies.inertia_com.shape != (source.capacity, 3, 3)
            or not np.array_equal(target_ids, source_ids)
            or not np.array_equal(
                np.asarray(target_particles.active_mask),
                np.asarray(source.particles.active_mask),
            )
        ):
            raise ValueError(
                "Target particle/body plans do not preserve source body identity."
            )
        self.center_of_mass_offsets = jnp.asarray(
            offsets, dtype=source.particles.safe_masses.dtype
        )
        self.body_ids = jnp.asarray(source_ids)
        self.source_prepared_id = source.prepared_id
        self.source_particle_plan_id = source.particles.plan.plan_id
        self.source_body_plan_id = source.plan.plan_id
        self.target_particle_plan_id = target_particles.plan_id
        self.target_body_plan_id = target_bodies.plan_id
        self.rebase_id = canonical_fingerprint(
            {
                "kind": "rigid-body-reference-frame-rebase",
                "source_prepared": source.prepared_id,
                "source_particle_plan": source.particles.plan.plan_id,
                "source_body_plan": source.plan.plan_id,
                "target_particle_plan": target_particles.plan_id,
                "target_body_plan": target_bodies.plan_id,
                "body_ids": array_tree_fingerprint(source_ids),
                "center_of_mass_offsets": array_tree_fingerprint(offsets),
            }
        )

    def _require_owners(
        self,
        source: PreparedRigidBodySet,
        target: PreparedRigidBodySet,
        /,
    ) -> None:
        if not isinstance(source, PreparedRigidBodySet):
            raise TypeError("source must be a PreparedRigidBodySet.")
        if not isinstance(target, PreparedRigidBodySet):
            raise TypeError("target must be a PreparedRigidBodySet.")
        if (
            source.prepared_id != self.source_prepared_id
            or source.particles.plan.plan_id != self.source_particle_plan_id
            or source.plan.plan_id != self.source_body_plan_id
        ):
            raise ValueError("Source body identity does not match this rebase.")
        if (
            target.particles.plan.plan_id != self.target_particle_plan_id
            or target.plan.plan_id != self.target_body_plan_id
        ):
            raise ValueError("Target body identity does not match this rebase.")
        expected_ids = np.asarray(self.body_ids)
        if (
            source.ambient_dimension != 3
            or target.ambient_dimension != 3
            or source.capacity != target.capacity
            or not np.array_equal(np.asarray(source.particles.particle_ids), expected_ids)
            or not np.array_equal(np.asarray(target.particles.particle_ids), expected_ids)
        ):
            raise ValueError("Rebase owners have incompatible body support.")

    def rebase_kinematics(
        self,
        reference: RigidBodyKinematics,
        source: PreparedRigidBodySet,
        target: PreparedRigidBodySet,
        /,
    ) -> RigidBodyKinematics:
        """Shift an old-origin pose/twist to the target centre of mass."""

        self._require_owners(source, target)
        if not isinstance(reference, RigidBodyKinematics):
            raise TypeError("reference must be RigidBodyKinematics.")
        expected_vector = (source.capacity, 3)
        if (
            reference.position.shape != expected_vector
            or reference.velocity.shape != expected_vector
            or reference.orientation.shape != (source.capacity, 4)
            or reference.angular_velocity.shape != expected_vector
        ):
            raise ValueError("Reference kinematics do not match rebase body support.")
        leaves = (
            reference.position,
            reference.velocity,
            reference.orientation,
            reference.angular_velocity,
        )
        if not all(np.all(np.isfinite(np.asarray(leaf))) for leaf in leaves):
            raise ValueError("Reference kinematics must be finite.")
        orientation_norm = np.linalg.norm(np.asarray(reference.orientation), axis=-1)
        if not np.allclose(orientation_norm, 1.0, rtol=0.0, atol=1.0e-8):
            raise ValueError("Reference orientations must have unit norm.")
        rotation = quaternion_rotation_matrix(reference.orientation)
        world_offset = contract("bij,bj->bi", rotation, self.center_of_mass_offsets)
        return RigidBodyKinematics(
            reference.position + world_offset,
            reference.velocity + jnp.cross(reference.angular_velocity, world_offset),
            reference.orientation,
            reference.angular_velocity,
        )

    def rebase_local_points(
        self,
        points: ArrayLike,
        source: PreparedRigidBodySet,
        target: PreparedRigidBodySet,
        /,
    ) -> Array:
        """Map full-capacity old-local attachment points into COM coordinates."""

        self._require_owners(source, target)
        points_host = np.asarray(points)
        if (
            points_host.ndim < 2
            or points_host.shape[0] != source.capacity
            or points_host.shape[-1] != 3
            or not np.all(np.isfinite(points_host))
        ):
            raise ValueError(
                "Local points must be finite with leading body-capacity and "
                "trailing spatial axes."
            )
        offsets = self.center_of_mass_offsets.reshape(
            (source.capacity,) + (1,) * (points_host.ndim - 2) + (3,)
        )
        return jnp.asarray(points_host, dtype=offsets.dtype) - offsets


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


def _principal_angle(value: Array, /) -> Array:
    return jnp.arctan2(jnp.sin(value), jnp.cos(value))


def _planar_rotation_matrix(angle: Array, /) -> Array:
    cosine = jnp.cos(angle[..., 0])
    sine = jnp.sin(angle[..., 0])
    return jnp.stack(
        (cosine, -sine, sine, cosine),
        axis=-1,
    ).reshape(angle.shape[:-1] + (2, 2))


def _rigid_body_rotation_matrix(
    bodies: PreparedRigidBodySet,
    orientation: Array,
    /,
) -> Array:
    if bodies.ambient_dimension == 2:
        return _planar_rotation_matrix(orientation)
    return quaternion_rotation_matrix(orientation)


def _rigid_body_relative_rotation(
    bodies: PreparedRigidBodySet,
    reference: Array,
    point: Array,
    /,
) -> Array:
    if bodies.ambient_dimension == 2:
        return _principal_angle(point - reference)
    return _quaternion_relative_rotation_vector(reference, point)


def rigid_body_world_inertia(
    bodies: PreparedRigidBodySet,
    orientation: Array,
    /,
) -> tuple[Array, Array]:
    rotation = quaternion_rotation_matrix(orientation)
    inertia = contract(
        "...ij,...jk,...lk->...il",
        rotation,
        bodies.mass_properties.inertia_com,
        rotation,
    )
    inverse = contract(
        "...ij,...jk,...lk->...il",
        rotation,
        bodies.mass_properties.inverse_inertia_com,
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
        bodies.mass_properties.inverse_masses[:, None] * load.force
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
        bodies.mass_properties.inverse_masses[:, None] * load.force
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
    if bodies.ambient_dimension == 2:
        orientation = _principal_angle(kinematics.orientation + rotation)
    else:
        orientation = _quaternion_retract(kinematics.orientation, rotation)
    return RigidBodyKinematics(
        jnp.where(mobile, kinematics.position + translation, kinematics.position),
        kinematics.velocity,
        jnp.where(mobile, orientation, kinematics.orientation),
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
        return bodies.mass_properties.inverse_inertia_com[:, None] * torque
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
    """Four-space geometry for rigid-body state storage and physical twists."""

    bodies: PreparedRigidBodySet
    orientation_geometry: ScalarFirstQuaternionStateGeometry | None
    local_space: PyTreeSpace
    tangent_space: PyTreeSpace
    local_cotangent_space: DualSpace
    cotangent_space: DualSpace
    geometry_id: str = eqx.field(static=True)
    retraction_method: str = eqx.field(static=True)
    trivial: bool = eqx.field(static=True)
    supports_exact_inverse: bool = eqx.field(static=True)
    supports_exact_differential: bool = eqx.field(static=True)
    supports_transport: bool = eqx.field(static=True)
    supports_isometric_transport: bool = eqx.field(static=True)
    supports_commutator_free: bool = eqx.field(static=True)

    def __init__(self, bodies: PreparedRigidBodySet, /):
        if not isinstance(bodies, PreparedRigidBodySet):
            raise TypeError("bodies must be a PreparedRigidBodySet.")
        identifier = f"state-geometry:rigid-body:{bodies.prepared_id}"
        dtype = bodies.particles.safe_masses.dtype
        linear = jnp.zeros(
            (bodies.capacity, bodies.ambient_dimension),
            dtype=dtype,
        )
        angular = jnp.zeros(
            (bodies.capacity, bodies.angular_dimension),
            dtype=dtype,
        )
        role_template = RigidBodyKinematics(linear, linear, angular, angular)
        local_space = PyTreeSpace(
            role_template,
            space_id=f"{identifier}:local-space",
        )
        tangent_space = PyTreeSpace(
            role_template,
            space_id=f"{identifier}:tangent-space",
        )
        self.bodies = bodies
        self.orientation_geometry = (
            None
            if bodies.ambient_dimension == 2
            else ScalarFirstQuaternionStateGeometry(
                convention="spatial",
                tolerance=1.0e-8,
            )
        )
        self.local_space = local_space
        self.tangent_space = tangent_space
        self.local_cotangent_space = DualSpace(
            local_space,
            space_id=f"{identifier}:local-cotangent-space",
        )
        self.cotangent_space = DualSpace(
            tangent_space,
            space_id=f"{identifier}:cotangent-space",
        )
        self.geometry_id = identifier
        self.retraction_method = "rigid-body-canonical-lie-retraction:spatial"
        self.trivial = False
        self.supports_exact_inverse = True
        self.supports_exact_differential = True
        self.supports_transport = True
        self.supports_isometric_transport = True
        self.supports_commutator_free = True

    def _point(self, value, name, /):
        if not isinstance(value, RigidBodyKinematics):
            raise TypeError(f"{name} must be RigidBodyKinematics.")
        position = jnp.asarray(value.position)
        velocity = jnp.asarray(value.velocity)
        orientation = jnp.asarray(value.orientation)
        angular_velocity = jnp.asarray(value.angular_velocity)
        linear_shape = (self.bodies.capacity, self.bodies.ambient_dimension)
        orientation_shape = (
            self.bodies.capacity,
            self.bodies.orientation_dimension,
        )
        angular_shape = (
            self.bodies.capacity,
            self.bodies.angular_dimension,
        )
        if position.shape != linear_shape or velocity.shape != linear_shape:
            raise ValueError(f"{name} position and velocity shapes are invalid.")
        if orientation.shape != orientation_shape:
            raise ValueError(f"{name} orientation shape is invalid.")
        if angular_velocity.shape != angular_shape:
            raise ValueError(f"{name} angular-velocity shape is invalid.")
        return RigidBodyKinematics(
            position,
            velocity,
            orientation,
            angular_velocity,
        )

    def contains(self, state, /):
        if not isinstance(state, RigidBodyKinematics):
            return jnp.asarray(False)
        linear_shape = (self.bodies.capacity, self.bodies.ambient_dimension)
        orientation_shape = (
            self.bodies.capacity,
            self.bodies.orientation_dimension,
        )
        angular_shape = (
            self.bodies.capacity,
            self.bodies.angular_dimension,
        )
        if (
            jnp.shape(state.position) != linear_shape
            or jnp.shape(state.velocity) != linear_shape
            or jnp.shape(state.orientation) != orientation_shape
            or jnp.shape(state.angular_velocity) != angular_shape
        ):
            return jnp.asarray(False)
        finite = tree_allfinite(state)
        if self.bodies.ambient_dimension == 3:
            norm = jnp.linalg.norm(state.orientation, axis=-1)
            finite = finite & jnp.all(jnp.abs(norm - 1.0) <= 1.0e-8)
        return finite

    def project_tangent(self, state, vector, /):
        point = self._point(state, "Rigid-body state")
        ambient = self._point(vector, "Ambient rigid-body tangent")
        if self.orientation_geometry is None:
            orientation = ambient.orientation
        else:
            orientation = jax.vmap(self.orientation_geometry.project_tangent)(
                point.orientation,
                ambient.orientation,
            )
        return self.tangent_space.validate(
            RigidBodyKinematics(
                ambient.position,
                ambient.velocity,
                orientation,
                ambient.angular_velocity,
            )
        )

    def retract(self, state, local_tangent, /):
        point = self._point(state, "Rigid-body state")
        local = self.local_space.validate(local_tangent)
        if self.orientation_geometry is None:
            orientation = _principal_angle(point.orientation + local.orientation)
        else:
            orientation = jax.vmap(self.orientation_geometry.retract)(
                point.orientation,
                local.orientation,
            )
        return RigidBodyKinematics(
            point.position + local.position,
            point.velocity + local.velocity,
            orientation,
            point.angular_velocity + local.angular_velocity,
        )

    def inverse_retract(self, state, point, /):
        anchor = self._point(state, "Rigid-body chart anchor")
        target = self._point(point, "Rigid-body chart point")
        if self.orientation_geometry is None:
            orientation = _principal_angle(target.orientation - anchor.orientation)
        else:
            orientation = jax.vmap(self.orientation_geometry.inverse_retract)(
                anchor.orientation,
                target.orientation,
            )
        return self.local_space.validate(
            RigidBodyKinematics(
                target.position - anchor.position,
                target.velocity - anchor.velocity,
                orientation,
                target.angular_velocity - anchor.angular_velocity,
            )
        )

    def retraction_jvp(
        self,
        state,
        local_tangent,
        local_velocity,
        /,
    ):
        point = self._point(state, "Rigid-body state")
        local = self.local_space.validate(local_tangent)
        direction = self.local_space.validate(local_velocity)
        if self.orientation_geometry is None:
            orientation = direction.orientation
        else:
            orientation = jax.vmap(self.orientation_geometry.retraction_jvp)(
                point.orientation,
                local.orientation,
                direction.orientation,
            )
        return self.tangent_space.validate(
            RigidBodyKinematics(
                direction.position,
                direction.velocity,
                orientation,
                direction.angular_velocity,
            )
        )

    def retraction_inverse_jvp(
        self,
        state,
        point,
        tangent,
        /,
    ):
        anchor = self._point(state, "Rigid-body chart anchor")
        target = self._point(point, "Rigid-body chart point")
        physical = self.tangent_space.validate(tangent)
        if self.orientation_geometry is None:
            orientation = physical.orientation
        else:
            orientation = jax.vmap(self.orientation_geometry.retraction_inverse_jvp)(
                anchor.orientation,
                target.orientation,
                physical.orientation,
            )
        return self.local_space.validate(
            RigidBodyKinematics(
                physical.position,
                physical.velocity,
                orientation,
                physical.angular_velocity,
            )
        )

    def retraction_vjp(
        self,
        state,
        local_tangent,
        cotangent,
        /,
    ):
        point = self._point(state, "Rigid-body state")
        local = self.local_space.validate(local_tangent)
        physical = self.cotangent_space.validate(cotangent)
        if self.orientation_geometry is None:
            orientation = physical.orientation
        else:
            orientation = jax.vmap(self.orientation_geometry.retraction_vjp)(
                point.orientation,
                local.orientation,
                physical.orientation,
            )
        return self.local_cotangent_space.validate(
            RigidBodyKinematics(
                physical.position,
                physical.velocity,
                orientation,
                physical.angular_velocity,
            )
        )

    def transport_tangent(self, state, point, tangent, /):
        anchor = self._point(state, "Rigid-body transport source")
        target = self._point(point, "Rigid-body transport target")
        physical = self.tangent_space.validate(tangent)
        if self.orientation_geometry is None:
            orientation = physical.orientation
        else:
            orientation = jax.vmap(self.orientation_geometry.transport_tangent)(
                anchor.orientation,
                target.orientation,
                physical.orientation,
            )
        return self.tangent_space.validate(
            RigidBodyKinematics(
                physical.position,
                physical.velocity,
                orientation,
                physical.angular_velocity,
            )
        )

    def transport_cotangent_pullback(
        self,
        state,
        point,
        cotangent,
        /,
    ):
        anchor = self._point(state, "Rigid-body transport source")
        target = self._point(point, "Rigid-body transport target")
        physical = self.cotangent_space.validate(cotangent)
        if self.orientation_geometry is None:
            orientation = physical.orientation
        else:
            orientation = jax.vmap(
                self.orientation_geometry.transport_cotangent_pullback
            )(
                anchor.orientation,
                target.orientation,
                physical.orientation,
            )
        return self.cotangent_space.validate(
            RigidBodyKinematics(
                physical.position,
                physical.velocity,
                orientation,
                physical.angular_velocity,
            )
        )

    def cut_locus_margin(self, state, point, /):
        anchor = self._point(state, "Rigid-body chart anchor")
        target = self._point(point, "Rigid-body chart point")
        if self.orientation_geometry is None:
            difference = _principal_angle(target.orientation - anchor.orientation)
            return jnp.min(jnp.pi - jnp.abs(difference))
        margins = jax.vmap(self.orientation_geometry.cut_locus_margin)(
            anchor.orientation,
            target.orientation,
        )
        return jnp.min(margins)


__all__ = [
    "PreparedRigidBodySet",
    "RigidBodyKinematics",
    "RigidBodyMassProperties",
    "RigidBodyLoad",
    "RigidBodyReferenceFrameRebase",
    "RigidBodySetPlan",
    "RigidBodyStateGeometry",
    "RigidBodyStepResult",
    "quaternion_rotation_matrix",
    "rigid_body_angular_acceleration",
    "rigid_body_drift",
    "rigid_body_kick_drift_kick",
    "rigid_body_world_inertia",
]
