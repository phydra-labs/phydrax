#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from ..._strict import StrictModule
from ._dem_contact import DEMContactBatch
from ._rigid_sphere import sphere_spin_velocity, SpherePairContactGeometry
from ._sphere_clump import ClumpComponentPairBatch


class RigidContactGeometry(StrictModule):
    normal: Array
    gap: Array
    overlap: Array
    effective_radius: Array
    contact_point: Array
    left_owner_arm: Array
    right_owner_arm: Array
    left_contact_arm: Array
    right_contact_arm: Array
    relative_velocity: Array
    normal_velocity: Array
    tangential_velocity: Array
    left_angular_velocity: Array
    right_angular_velocity: Array
    contact_keys: Array
    left_feature: Array
    right_feature: Array
    valid: Array
    degeneracy_code: Array
    feature_margin: Array
    successful: Array
    geometry_id: str = eqx.field(static=True)

    def as_contact_batch(self, /) -> DEMContactBatch:
        return DEMContactBatch(
            self.normal,
            self.gap,
            self.overlap,
            self.effective_radius,
            self.left_contact_arm,
            self.right_contact_arm,
            self.normal_velocity,
            self.tangential_velocity,
            self.left_angular_velocity,
            self.right_angular_velocity,
            self.valid,
        )


def sphere_contact_adapter(
    geometry: SpherePairContactGeometry,
    left_radius: Array,
    right_radius: Array,
    left_angular_velocity: Array,
    right_angular_velocity: Array,
    contact_keys: Array,
    /,
) -> RigidContactGeometry:
    radius_sum = left_radius + right_radius
    effective_radius = jnp.where(
        radius_sum > 0.0, left_radius * right_radius / radius_sum, 0.0
    )
    zero_feature = jnp.zeros((contact_keys.shape[0],), dtype=jnp.int32)
    degeneracy = geometry.degenerate.astype(jnp.int32)
    return RigidContactGeometry(
        geometry.normal,
        geometry.gap,
        geometry.overlap,
        effective_radius,
        geometry.contact_point,
        geometry.left_arm,
        geometry.right_arm,
        geometry.left_arm,
        geometry.right_arm,
        geometry.relative_velocity,
        geometry.normal_velocity,
        geometry.tangential_velocity,
        left_angular_velocity,
        right_angular_velocity,
        contact_keys,
        zero_feature,
        zero_feature,
        geometry.valid,
        degeneracy,
        jnp.where(geometry.valid, jnp.abs(geometry.gap), jnp.inf),
        geometry.successful,
        "rigid-contact:sphere-pair",
    )


def clump_component_contact_geometry(
    pairs: ClumpComponentPairBatch,
    left_angular_velocity: Array,
    right_angular_velocity: Array,
    /,
    *,
    distance_tolerance: float = 1.0e-12,
) -> RigidContactGeometry:
    displacement = pairs.left_position - pairs.right_position
    distance_squared = jnp.sum(displacement * displacement, axis=-1)
    distance = jnp.sqrt(distance_squared)
    positive = pairs.valid & (distance > distance_tolerance)
    safe_distance = jnp.where(positive, distance, 1.0)
    normal = jnp.where(positive[:, None], displacement / safe_distance[:, None], 0.0)
    gap = distance - pairs.left_radius - pairs.right_radius
    overlap = jnp.where(pairs.valid, jnp.maximum(-gap, 0.0), 0.0)
    degenerate = pairs.valid & (overlap > 0.0) & ~positive
    valid = pairs.valid & ~degenerate
    left_length = 0.5 * (distance + pairs.left_radius - pairs.right_radius)
    right_length = 0.5 * (distance - pairs.left_radius + pairs.right_radius)
    left_contact_arm = -left_length[:, None] * normal
    right_contact_arm = right_length[:, None] * normal
    contact_point = pairs.left_position + left_contact_arm
    left_owner_position = pairs.left_position - pairs.left_owner_offset
    right_owner_position = pairs.right_position - pairs.right_owner_offset
    left_owner_arm = contact_point - left_owner_position
    right_owner_arm = contact_point - right_owner_position
    dimension = displacement.shape[-1]
    left_velocity = pairs.left_velocity + sphere_spin_velocity(
        left_angular_velocity, left_contact_arm, dimension
    )
    right_velocity = pairs.right_velocity + sphere_spin_velocity(
        right_angular_velocity, right_contact_arm, dimension
    )
    relative = left_velocity - right_velocity
    normal_velocity = jnp.sum(relative * normal, axis=-1)
    tangential_velocity = relative - normal_velocity[:, None] * normal
    radius_sum = pairs.left_radius + pairs.right_radius
    effective_radius = jnp.where(
        radius_sum > 0.0,
        pairs.left_radius * pairs.right_radius / radius_sum,
        0.0,
    )
    feature = jnp.zeros(pairs.component_pair_keys.shape, dtype=jnp.int32)
    return RigidContactGeometry(
        jnp.where(valid[:, None], normal, 0.0),
        jnp.where(valid, gap, 0.0),
        jnp.where(valid, overlap, 0.0),
        jnp.where(valid, effective_radius, 0.0),
        jnp.where(valid[:, None], contact_point, 0.0),
        jnp.where(valid[:, None], left_owner_arm, 0.0),
        jnp.where(valid[:, None], right_owner_arm, 0.0),
        jnp.where(valid[:, None], left_contact_arm, 0.0),
        jnp.where(valid[:, None], right_contact_arm, 0.0),
        jnp.where(valid[:, None], relative, 0.0),
        jnp.where(valid, normal_velocity, 0.0),
        jnp.where(valid[:, None], tangential_velocity, 0.0),
        left_angular_velocity,
        right_angular_velocity,
        pairs.component_pair_keys,
        feature,
        feature,
        valid,
        degenerate.astype(jnp.int32),
        jnp.where(valid, jnp.abs(gap), jnp.inf),
        ~jnp.any(degenerate),
        "rigid-contact:clump-component-spheres",
    )


__all__ = [
    "RigidContactGeometry",
    "clump_component_contact_geometry",
    "sphere_contact_adapter",
]
