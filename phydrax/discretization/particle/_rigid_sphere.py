#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from .._core import DiscretizationKey, DiscretizationRole, PreparationReport
from ._core import ParticleDiscretization
from ._pairwise import ParticlePairGeometry, ParticlePairRelation


class RigidSphereSetPlan(StrictModule, NonTrainableState):
    """Rigid spherical properties attached to one material-particle support."""

    radii: Array
    material_ids: Array
    inertias: Array | None
    fixed_mask: Array
    key: DiscretizationKey
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        radii: ArrayLike,
        material_ids: ArrayLike,
        /,
        *,
        inertias: ArrayLike | None = None,
        fixed_mask: ArrayLike | None = None,
        name: str = "rigid-spheres",
        plan_id: str | None = None,
    ):
        radii_host = np.asarray(radii)
        material_host = np.asarray(material_ids)
        if radii_host.ndim != 1 or radii_host.size == 0:
            raise ValueError("radii must be a nonempty rank-1 array.")
        if material_host.shape != radii_host.shape or not np.issubdtype(
            material_host.dtype, np.integer
        ):
            raise TypeError("material_ids must be an integer array with the radii shape.")
        if np.any(~np.isfinite(radii_host)) or np.any(radii_host <= 0.0):
            raise ValueError("Sphere radii must be finite and positive.")
        if np.any(material_host < 0):
            raise ValueError("Sphere material IDs must be nonnegative.")
        inertia_host = None if inertias is None else np.asarray(inertias)
        if inertia_host is not None:
            if inertia_host.shape != radii_host.shape:
                raise ValueError("inertias must have the radii shape.")
            if np.any(~np.isfinite(inertia_host)) or np.any(inertia_host <= 0.0):
                raise ValueError("Sphere inertias must be finite and positive.")
        fixed_host = (
            np.zeros(radii_host.shape, dtype=bool)
            if fixed_mask is None
            else np.asarray(fixed_mask, dtype=bool)
        )
        if fixed_host.shape != radii_host.shape:
            raise ValueError("fixed_mask must have the radii shape.")
        key = DiscretizationKey(
            name,
            DiscretizationRole.AUXILIARY,
            domain_labels=("material_point", "rigid_sphere"),
        )
        generated = canonical_fingerprint(
            {
                "kind": "rigid-sphere-set-plan",
                "properties": array_tree_fingerprint(
                    {
                        "radii": radii_host,
                        "material_ids": material_host.astype(np.int64),
                        "inertias": inertia_host,
                        "fixed_mask": fixed_host,
                    }
                ),
                "inertia_model": "homogeneous-solid"
                if inertia_host is None
                else "explicit",
                "key": key.key_id,
            }
        )
        identifier = generated if plan_id is None else str(plan_id)
        if not identifier:
            raise ValueError("plan_id must be nonempty.")
        self.radii = jnp.asarray(radii_host)
        self.material_ids = jnp.asarray(material_host, dtype=jnp.int32)
        self.inertias = None if inertia_host is None else jnp.asarray(inertia_host)
        self.fixed_mask = jnp.asarray(fixed_host, dtype=bool)
        self.key = key
        self.plan_id = identifier

    def prepare(self, particles: ParticleDiscretization, /) -> PreparedRigidSphereSet:
        return PreparedRigidSphereSet(self, particles)


class PreparedRigidSphereSet(StrictModule, NonTrainableState):
    """Prepared isotropic rigid-sphere mass and rotational data."""

    plan: RigidSphereSetPlan
    particles: ParticleDiscretization
    radii: Array
    inertias: Array
    inverse_masses: Array
    inverse_inertias: Array
    material_ids: Array
    fixed_mask: Array
    key: DiscretizationKey
    preparation: PreparationReport
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: RigidSphereSetPlan, particles: ParticleDiscretization, /):
        if not isinstance(plan, RigidSphereSetPlan):
            raise TypeError("plan must be a RigidSphereSetPlan.")
        if not isinstance(particles, ParticleDiscretization):
            raise TypeError("particles must be a ParticleDiscretization.")
        if particles.ambient_dimension not in (2, 3):
            raise ValueError("Rigid spheres currently require ambient dimension 2 or 3.")
        if plan.radii.shape != (particles.capacity,):
            raise ValueError("Rigid-sphere properties must match particle capacity.")
        dtype = particles.safe_masses.dtype
        active = particles.active_mask
        radii = jnp.asarray(plan.radii, dtype=dtype)
        masses = particles.safe_masses
        if plan.inertias is None:
            factor = 0.5 if particles.ambient_dimension == 2 else 0.4
            inertias = factor * masses * radii * radii
        else:
            inertias = jnp.asarray(plan.inertias, dtype=dtype)
        fixed = plan.fixed_mask & active
        mobile = active & ~fixed
        inverse_masses = jnp.where(mobile, 1.0 / masses, 0.0)
        inverse_inertias = jnp.where(mobile, 1.0 / inertias, 0.0)
        radii = jnp.where(active, radii, 0.0)
        inertias = jnp.where(active, inertias, 1.0)
        material_ids = jnp.where(active, plan.material_ids, 0)
        preparation = PreparationReport(
            diagnostics=(
                "isotropic rigid spheres",
                "sphere orientation is physically unobservable and omitted",
                "fixed bodies retain reaction loads with zero inverse inertia and mass",
            ),
            resource_counts={
                "particle_capacity": particles.capacity,
                "active_spheres": particles.active_count,
                "fixed_spheres": int(np.count_nonzero(np.asarray(fixed))),
                "ambient_dimension": particles.ambient_dimension,
            },
        )
        self.plan = plan
        self.particles = particles
        self.radii = radii
        self.inertias = inertias
        self.inverse_masses = inverse_masses
        self.inverse_inertias = inverse_inertias
        self.material_ids = material_ids.astype(jnp.int32)
        self.fixed_mask = fixed
        self.key = plan.key
        self.preparation = preparation
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-rigid-sphere-set",
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
    def resource_evidence_id(self) -> str:
        return self.preparation.report_id

    def kinematics(
        self,
        position: ArrayLike,
        velocity: ArrayLike,
        angular_velocity: ArrayLike | None = None,
        /,
    ) -> RigidSphereKinematics:
        position_ = self._vectors("position", position)
        velocity_ = self._vectors("velocity", velocity)
        angular_shape = (self.capacity, self.angular_dimension)
        angular_ = (
            jnp.zeros(angular_shape, dtype=position_.dtype)
            if angular_velocity is None
            else jnp.asarray(angular_velocity, dtype=position_.dtype)
        )
        if angular_.shape != angular_shape:
            raise ValueError(f"angular_velocity must have shape {angular_shape}.")
        angular_ = eqx.error_if(
            angular_,
            jnp.any(
                jnp.where(
                    self.particles.active_mask[:, None],
                    ~jnp.isfinite(angular_),
                    False,
                )
            ),
            "Active sphere angular velocities must be finite.",
        )
        mobile = (self.particles.active_mask & ~self.fixed_mask)[:, None]
        velocity_ = jnp.where(mobile, velocity_, 0.0)
        angular_ = jnp.where(mobile, angular_, 0.0)
        return RigidSphereKinematics(position_, velocity_, angular_)

    def load(self, force: ArrayLike, torque: ArrayLike, /) -> RigidSphereLoad:
        force_ = self._vectors("force", force)
        torque_ = jnp.asarray(torque, dtype=force_.dtype)
        expected = (self.capacity, self.angular_dimension)
        if torque_.shape != expected:
            raise ValueError(f"torque must have shape {expected}.")
        torque_ = eqx.error_if(
            torque_,
            jnp.any(
                jnp.where(
                    self.particles.active_mask[:, None],
                    ~jnp.isfinite(torque_),
                    False,
                )
            ),
            "Active sphere torques must be finite.",
        )
        return RigidSphereLoad(
            jnp.where(self.particles.active_mask[:, None], force_, 0.0),
            jnp.where(self.particles.active_mask[:, None], torque_, 0.0),
        )

    def _vectors(self, name: str, value: ArrayLike, /) -> Array:
        array = jnp.asarray(value, dtype=self.radii.dtype)
        expected = (self.capacity, self.ambient_dimension)
        if array.shape != expected:
            raise ValueError(f"{name} must have shape {expected}.")
        array = eqx.error_if(
            array,
            jnp.any(
                jnp.where(
                    self.particles.active_mask[:, None], ~jnp.isfinite(array), False
                )
            ),
            f"Active sphere {name} values must be finite.",
        )
        return jnp.where(self.particles.active_mask[:, None], array, 0.0)


class RigidSphereKinematics(StrictModule):
    position: Array
    velocity: Array
    angular_velocity: Array


class RigidSphereLoad(StrictModule):
    force: Array
    torque: Array


class SpherePairContactGeometry(StrictModule):
    normal: Array
    gap: Array
    overlap: Array
    contact_point: Array
    left_arm: Array
    right_arm: Array
    relative_velocity: Array
    normal_velocity: Array
    tangential_velocity: Array
    valid: Array
    degenerate: Array
    successful: Array
    relation_schema_id: str = eqx.field(static=True)


def sphere_spin_velocity(
    angular_velocity: Array, arm: Array, ambient_dimension: int, /
) -> Array:
    if ambient_dimension == 2:
        omega = angular_velocity[..., 0]
        return jnp.stack((-omega * arm[..., 1], omega * arm[..., 0]), axis=-1)
    if ambient_dimension == 3:
        return jnp.cross(angular_velocity, arm)
    raise ValueError("Sphere spin velocity requires ambient dimension 2 or 3.")


def sphere_lever_torque(arm: Array, force: Array, ambient_dimension: int, /) -> Array:
    if ambient_dimension == 2:
        return (arm[..., 0] * force[..., 1] - arm[..., 1] * force[..., 0])[..., None]
    if ambient_dimension == 3:
        return jnp.cross(arm, force)
    raise ValueError("Sphere torque requires ambient dimension 2 or 3.")


def sphere_pair_contact_geometry(
    bodies: PreparedRigidSphereSet,
    kinematics: RigidSphereKinematics,
    pairs: ParticlePairRelation,
    geometry: ParticlePairGeometry,
    /,
    *,
    distance_tolerance: float = 1.0e-12,
) -> SpherePairContactGeometry:
    """Lift center geometry to one common-point rigid-sphere contact geometry."""

    if not isinstance(bodies, PreparedRigidSphereSet):
        raise TypeError("bodies must be a PreparedRigidSphereSet.")
    if not isinstance(kinematics, RigidSphereKinematics):
        raise TypeError("kinematics must be RigidSphereKinematics.")
    if not isinstance(pairs, ParticlePairRelation):
        raise TypeError("pairs must be a ParticlePairRelation.")
    if not isinstance(geometry, ParticlePairGeometry):
        raise TypeError("geometry must be ParticlePairGeometry.")
    if pairs.relation_schema_id != geometry.relation_schema_id:
        raise ValueError("Pair relation and geometry schemas do not match.")
    if (
        pairs.source_support_id != bodies.particles.support.support_id
        or not pairs.same_set
    ):
        raise ValueError("Pair relation does not belong to the rigid-sphere support.")
    tolerance = float(distance_tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("distance_tolerance must be finite and positive.")
    left = pairs.left_indices
    right = pairs.right_indices
    left_radius = bodies.radii[left]
    right_radius = bodies.radii[right]
    normal = geometry.direction
    gap = geometry.distance - left_radius - right_radius
    overlap = jnp.where(geometry.valid, jnp.maximum(-gap, 0.0), 0.0)
    left_length = 0.5 * (geometry.distance + left_radius - right_radius)
    right_length = 0.5 * (geometry.distance - left_radius + right_radius)
    left_arm = -left_length[:, None] * normal
    right_arm = right_length[:, None] * normal
    contact_point = kinematics.position[left] + left_arm
    left_contact_velocity = kinematics.velocity[left] + sphere_spin_velocity(
        kinematics.angular_velocity[left], left_arm, bodies.ambient_dimension
    )
    right_contact_velocity = kinematics.velocity[right] + sphere_spin_velocity(
        kinematics.angular_velocity[right], right_arm, bodies.ambient_dimension
    )
    relative = left_contact_velocity - right_contact_velocity
    normal_velocity = jnp.sum(relative * normal, axis=-1)
    tangential = relative - normal_velocity[:, None] * normal
    degenerate = geometry.valid & (overlap > 0.0) & (geometry.distance <= tolerance)
    valid = geometry.valid & ~degenerate
    mask = valid[:, None]
    return SpherePairContactGeometry(
        jnp.where(mask, normal, 0.0),
        jnp.where(valid, gap, 0.0),
        jnp.where(valid, overlap, 0.0),
        jnp.where(mask, contact_point, 0.0),
        jnp.where(mask, left_arm, 0.0),
        jnp.where(mask, right_arm, 0.0),
        jnp.where(mask, relative, 0.0),
        jnp.where(valid, normal_velocity, 0.0),
        jnp.where(mask, tangential, 0.0),
        valid,
        degenerate,
        ~jnp.any(degenerate),
        pairs.relation_schema_id,
    )


__all__ = [
    "PreparedRigidSphereSet",
    "RigidSphereKinematics",
    "RigidSphereLoad",
    "RigidSphereSetPlan",
    "SpherePairContactGeometry",
    "sphere_lever_torque",
    "sphere_pair_contact_geometry",
    "sphere_spin_velocity",
]
