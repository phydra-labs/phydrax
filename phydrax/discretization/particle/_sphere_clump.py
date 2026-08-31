#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._core import ParticleDiscretization
from ._pair_state import CLUMP_COMPONENT_INTERACTION, INTERACTION_KEY_WIDTH
from ._pairwise import ParticlePairRelation
from ._rigid_body import (
    PreparedRigidBodySet,
    quaternion_rotation_matrix,
    RigidBodyKinematics,
    RigidBodySetPlan,
)


class SphereClumpTemplatePlan(StrictModule, NonTrainableState):
    component_offset: Array
    component_radius: Array
    component_mass: Array
    component_material: Array
    inertia_body: Array
    total_mass: float = eqx.field(static=True)
    bounding_radius: float = eqx.field(static=True)
    component_count: int = eqx.field(static=True)
    template_id: str = eqx.field(static=True)

    def __init__(
        self,
        component_offset: ArrayLike,
        component_radius: ArrayLike,
        component_mass: ArrayLike,
        component_material: ArrayLike,
        /,
        *,
        center_tolerance: float = 1.0e-12,
        template_id: str | None = None,
    ):
        offset = np.asarray(component_offset)
        radius = np.asarray(component_radius)
        mass = np.asarray(component_mass)
        material = np.asarray(component_material)
        if offset.ndim != 2 or offset.shape[1] not in (2, 3) or offset.shape[0] == 0:
            raise ValueError("component_offset must have shape (components,2|3).")
        count, dimension = offset.shape
        if (
            radius.shape != (count,)
            or mass.shape != (count,)
            or material.shape != (count,)
        ):
            raise ValueError("Clump component arrays must share component count.")
        if not np.issubdtype(material.dtype, np.integer):
            raise TypeError("component_material must contain integers.")
        if (
            np.any(~np.isfinite(offset))
            or np.any(~np.isfinite(radius))
            or np.any(~np.isfinite(mass))
            or np.any(radius <= 0.0)
            or np.any(mass <= 0.0)
            or np.any(material < 0)
        ):
            raise ValueError("Clump component geometry/material properties are invalid.")
        total = float(np.sum(mass))
        center = np.sum(mass[:, None] * offset, axis=0) / total
        if np.linalg.norm(center) > float(center_tolerance):
            raise ValueError("Clump component offsets must be centered at mass COM.")
        if dimension == 2:
            inertia = np.sum(0.5 * mass * radius**2 + mass * np.sum(offset**2, axis=-1))
        else:
            identity = np.eye(3)
            inertia = np.zeros((3, 3))
            for component_offset_, component_radius_, component_mass_ in zip(
                offset, radius, mass, strict=True
            ):
                local = 0.4 * component_mass_ * component_radius_**2 * identity
                norm_squared = float(component_offset_ @ component_offset_)
                parallel = component_mass_ * (
                    norm_squared * identity
                    - np.outer(component_offset_, component_offset_)
                )
                inertia = inertia + local + parallel
            if np.any(np.linalg.eigvalsh(inertia) <= 0.0):
                raise ValueError("Clump inertia must be positive definite.")
        bound = float(np.max(np.linalg.norm(offset, axis=-1) + radius))
        generated = canonical_fingerprint(
            {
                "kind": "sphere-clump-template",
                "values": array_tree_fingerprint(
                    {
                        "offset": offset,
                        "radius": radius,
                        "mass": mass,
                        "material": material,
                    }
                ),
            }
        )
        self.component_offset = jnp.asarray(offset)
        self.component_radius = jnp.asarray(radius)
        self.component_mass = jnp.asarray(mass)
        self.component_material = jnp.asarray(material, dtype=jnp.int32)
        self.inertia_body = jnp.asarray(inertia)
        self.total_mass = total
        self.bounding_radius = bound
        self.component_count = count
        self.template_id = generated if template_id is None else str(template_id)
        if not self.template_id:
            raise ValueError("template_id must be nonempty.")


class RigidSphereClumpSetPlan(StrictModule, NonTrainableState):
    templates: tuple[SphereClumpTemplatePlan, ...]
    owner_template_ids: Array
    owner_material_ids: Array
    fixed_mask: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        templates: Sequence[SphereClumpTemplatePlan],
        owner_template_ids: ArrayLike,
        owner_material_ids: ArrayLike,
        /,
        *,
        fixed_mask: ArrayLike | None = None,
        plan_id: str | None = None,
    ):
        templates_ = tuple(templates)
        if not templates_ or any(
            not isinstance(value, SphereClumpTemplatePlan) for value in templates_
        ):
            raise TypeError("templates must contain SphereClumpTemplatePlan values.")
        dimensions = {value.component_offset.shape[1] for value in templates_}
        if len(dimensions) != 1:
            raise ValueError("All clump templates must share ambient dimension.")
        template_ids = np.asarray(owner_template_ids)
        material_ids = np.asarray(owner_material_ids)
        if template_ids.ndim != 1 or material_ids.shape != template_ids.shape:
            raise ValueError("Owner template/material IDs must be matching vectors.")
        if not np.issubdtype(template_ids.dtype, np.integer) or not np.issubdtype(
            material_ids.dtype, np.integer
        ):
            raise TypeError("Owner template/material IDs must be integers.")
        if (
            np.any(template_ids < 0)
            or np.any(template_ids >= len(templates_))
            or np.any(material_ids < 0)
        ):
            raise ValueError("Owner template/material IDs are out of range.")
        fixed = (
            np.zeros(template_ids.shape, dtype=bool)
            if fixed_mask is None
            else np.asarray(fixed_mask, dtype=bool)
        )
        if fixed.shape != template_ids.shape:
            raise ValueError("fixed_mask must have owner-capacity shape.")
        generated = canonical_fingerprint(
            {
                "kind": "rigid-sphere-clump-set-plan",
                "templates": [value.template_id for value in templates_],
                "owners": array_tree_fingerprint(
                    {
                        "template_ids": template_ids,
                        "material_ids": material_ids,
                        "fixed": fixed,
                    }
                ),
            }
        )
        self.templates = templates_
        self.owner_template_ids = jnp.asarray(template_ids, dtype=jnp.int32)
        self.owner_material_ids = jnp.asarray(material_ids, dtype=jnp.int32)
        self.fixed_mask = jnp.asarray(fixed)
        self.plan_id = generated if plan_id is None else str(plan_id)
        if not self.plan_id:
            raise ValueError("plan_id must be nonempty.")

    def prepare(
        self, particles: ParticleDiscretization, /
    ) -> PreparedRigidSphereClumpSet:
        return PreparedRigidSphereClumpSet(self, particles)


class PreparedRigidSphereClumpSet(StrictModule, NonTrainableState):
    plan: RigidSphereClumpSetPlan
    bodies: PreparedRigidBodySet
    component_offset: Array
    component_radius: Array
    component_material: Array
    component_valid: Array
    bounding_radius: Array
    max_components: int = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self, plan: RigidSphereClumpSetPlan, particles: ParticleDiscretization, /
    ):
        if plan.owner_template_ids.shape != (particles.capacity,):
            raise ValueError("Clump owners must match particle capacity.")
        dimension = particles.ambient_dimension
        if any(value.component_offset.shape[1] != dimension for value in plan.templates):
            raise ValueError("Clump template dimension does not match particles.")
        masses = np.asarray(particles.safe_masses)
        template_ids = np.asarray(plan.owner_template_ids)
        expected_mass = np.asarray(
            [plan.templates[index].total_mass for index in template_ids]
        )
        if not np.allclose(masses, expected_mass, rtol=1.0e-12, atol=1.0e-12):
            raise ValueError("Owner masses must equal selected clump template masses.")
        inertia = np.stack(
            [np.asarray(plan.templates[index].inertia_body) for index in template_ids]
        )
        body_plan = RigidBodySetPlan(
            plan.owner_material_ids,
            inertia,
            fixed_mask=plan.fixed_mask,
            name="rigid-sphere-clump-bodies",
        )
        bodies = body_plan.prepare(particles)
        maximum = max(value.component_count for value in plan.templates)
        template_count = len(plan.templates)
        offset = np.zeros((template_count, maximum, dimension))
        radius = np.ones((template_count, maximum))
        material = np.zeros((template_count, maximum), dtype=np.int32)
        valid = np.zeros((template_count, maximum), dtype=bool)
        bounds = np.zeros((template_count,))
        for index, template in enumerate(plan.templates):
            count = template.component_count
            offset[index, :count] = np.asarray(template.component_offset)
            radius[index, :count] = np.asarray(template.component_radius)
            material[index, :count] = np.asarray(template.component_material)
            valid[index, :count] = True
            bounds[index] = template.bounding_radius
        selected = np.asarray(plan.owner_template_ids)
        self.plan = plan
        self.bodies = bodies
        self.component_offset = jnp.asarray(
            offset[selected], dtype=particles.safe_masses.dtype
        )
        self.component_radius = jnp.asarray(
            radius[selected], dtype=particles.safe_masses.dtype
        )
        self.component_material = jnp.asarray(material[selected], dtype=jnp.int32)
        self.component_valid = jnp.asarray(valid[selected])
        self.bounding_radius = jnp.asarray(
            bounds[selected], dtype=particles.safe_masses.dtype
        )
        self.max_components = maximum
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-rigid-sphere-clump-set",
                "plan": plan.plan_id,
                "bodies": bodies.prepared_id,
                "max_components": maximum,
            }
        )

    def component_kinematics(
        self, kinematics: RigidBodyKinematics, /
    ) -> ClumpComponentKinematics:
        if self.bodies.ambient_dimension == 2:
            angle = kinematics.orientation[:, 0]
            cosine = jnp.cos(angle)[:, None]
            sine = jnp.sin(angle)[:, None]
            offset = self.component_offset
            world_offset = jnp.stack(
                (
                    cosine * offset[..., 0] - sine * offset[..., 1],
                    sine * offset[..., 0] + cosine * offset[..., 1],
                ),
                axis=-1,
            )
            omega = kinematics.angular_velocity[:, :1]
            spin = jnp.stack(
                (-omega * world_offset[..., 1], omega * world_offset[..., 0]),
                axis=-1,
            )
        else:
            rotation = quaternion_rotation_matrix(kinematics.orientation)
            world_offset = contract("...ij,...kj->...ki", rotation, self.component_offset)
            spin = jnp.cross(kinematics.angular_velocity[:, None, :], world_offset)
        position = kinematics.position[:, None, :] + world_offset
        velocity = kinematics.velocity[:, None, :] + spin
        return ClumpComponentKinematics(
            position,
            velocity,
            world_offset,
            self.component_radius,
            self.component_material,
            self.component_valid,
        )


class ClumpComponentKinematics(StrictModule):
    position: Array
    velocity: Array
    owner_offset: Array
    radius: Array
    material: Array
    valid: Array


class ClumpComponentPairBatch(StrictModule):
    left_owner: Array
    right_owner: Array
    left_component: Array
    right_component: Array
    left_position: Array
    right_position: Array
    left_velocity: Array
    right_velocity: Array
    left_owner_offset: Array
    right_owner_offset: Array
    left_radius: Array
    right_radius: Array
    left_material: Array
    right_material: Array
    component_pair_keys: Array
    valid: Array


def expand_clump_owner_pairs(
    clumps: PreparedRigidSphereClumpSet,
    kinematics: RigidBodyKinematics,
    owner_pairs: ParticlePairRelation,
    owner_pair_keys: Array,
    /,
) -> ClumpComponentPairBatch:
    if owner_pair_keys.shape != (owner_pairs.capacity, INTERACTION_KEY_WIDTH):
        raise ValueError("owner_pair_keys must use structured interaction identities.")
    components = clumps.component_kinematics(kinematics)
    capacity = owner_pairs.capacity
    maximum = clumps.max_components
    left_owner = jnp.repeat(owner_pairs.left_indices, maximum * maximum)
    right_owner = jnp.repeat(owner_pairs.right_indices, maximum * maximum)
    left_component = jnp.tile(
        jnp.repeat(jnp.arange(maximum, dtype=jnp.int32), maximum), capacity
    )
    right_component = jnp.tile(
        jnp.tile(jnp.arange(maximum, dtype=jnp.int32), maximum), capacity
    )
    owner_valid = jnp.repeat(owner_pairs.valid, maximum * maximum)
    valid = (
        owner_valid
        & components.valid[left_owner, left_component]
        & components.valid[right_owner, right_component]
    )
    owner_identity = jnp.repeat(
        owner_pair_keys.astype(jnp.int64), maximum * maximum, axis=0
    )
    pair_keys = jnp.stack(
        (
            jnp.full_like(left_component, CLUMP_COMPONENT_INTERACTION, dtype=jnp.int64),
            owner_identity[:, 1],
            owner_identity[:, 2],
            left_component.astype(jnp.int64),
            right_component.astype(jnp.int64),
        ),
        axis=-1,
    )
    return ClumpComponentPairBatch(
        left_owner,
        right_owner,
        left_component,
        right_component,
        components.position[left_owner, left_component],
        components.position[right_owner, right_component],
        components.velocity[left_owner, left_component],
        components.velocity[right_owner, right_component],
        components.owner_offset[left_owner, left_component],
        components.owner_offset[right_owner, right_component],
        components.radius[left_owner, left_component],
        components.radius[right_owner, right_component],
        components.material[left_owner, left_component],
        components.material[right_owner, right_component],
        jnp.where(valid[:, None], pair_keys, -jnp.ones_like(pair_keys)),
        valid,
    )


def reduce_clump_component_loads(
    clumps: PreparedRigidSphereClumpSet,
    component_pairs: ClumpComponentPairBatch,
    pair_force_on_left: Array,
    contact_point: Array,
    /,
) -> tuple[Array, Array]:
    force = jnp.where(component_pairs.valid[:, None], pair_force_on_left, 0.0)
    owner_force = jnp.zeros(
        (clumps.bodies.capacity, clumps.bodies.ambient_dimension), dtype=force.dtype
    )
    owner_force = owner_force.at[component_pairs.left_owner].add(force)
    owner_force = owner_force.at[component_pairs.right_owner].add(-force)
    left_arm = (
        contact_point - component_pairs.left_position + component_pairs.left_owner_offset
    )
    right_arm = (
        contact_point
        - component_pairs.right_position
        + component_pairs.right_owner_offset
    )
    if clumps.bodies.ambient_dimension == 2:
        left_torque = (left_arm[:, 0] * force[:, 1] - left_arm[:, 1] * force[:, 0])[
            :, None
        ]
        right_torque = (
            right_arm[:, 0] * (-force[:, 1]) - right_arm[:, 1] * (-force[:, 0])
        )[:, None]
    else:
        left_torque = jnp.cross(left_arm, force)
        right_torque = jnp.cross(right_arm, -force)
    owner_torque = jnp.zeros(
        (clumps.bodies.capacity, clumps.bodies.angular_dimension), dtype=force.dtype
    )
    owner_torque = owner_torque.at[component_pairs.left_owner].add(left_torque)
    owner_torque = owner_torque.at[component_pairs.right_owner].add(right_torque)
    return owner_force, owner_torque


__all__ = [
    "ClumpComponentKinematics",
    "ClumpComponentPairBatch",
    "PreparedRigidSphereClumpSet",
    "RigidSphereClumpSetPlan",
    "SphereClumpTemplatePlan",
    "expand_clump_owner_pairs",
    "reduce_clump_component_loads",
]
