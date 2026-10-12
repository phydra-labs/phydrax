#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from ...sparse import EdgeRelation, RelationExecutionPlan
from ._dem_contact import DEMContactResponse
from ._precision import (
    ParticleAccumulation,
    ParticleExecutionPolicy,
    ParticlePrecisionPolicy,
)
from ._rigid_body import RigidBodyLoad
from ._rigid_contact import RigidContactGeometry
from ._rigid_sphere import sphere_lever_torque


def _reduce_owner_payload(
    geometry: RigidContactGeometry,
    left: Array,
    right: Array,
    active: Array,
    size: int,
    accumulation: ParticleAccumulation,
    /,
) -> Array:
    valid = geometry.valid & active
    left = jnp.where(valid[:, None], left, 0.0)
    right = jnp.where(valid[:, None], right, 0.0)
    count = left.shape[0]
    if count == 0:
        return jnp.zeros((size, left.shape[-1]), dtype=left.dtype)
    indices = jnp.stack(
        (geometry.left_owner_index, geometry.right_owner_index), axis=1
    ).reshape((-1,))
    values = jnp.stack((left, right), axis=1).reshape((2 * count, left.shape[-1]))
    endpoint_valid = jnp.broadcast_to(valid[:, None], (count, 2)).reshape((-1,))
    relation = EdgeRelation(
        jnp.arange(2 * count, dtype=jnp.int32),
        indices,
        source_size=2 * count,
        target_size=size,
        valid=endpoint_valid,
    )
    execution = RelationExecutionPlan().prepare(relation)
    result, _ = execution.reduce(values, accumulation=accumulation, output="dense")
    return result


def reduce_rigid_contact(
    geometry: RigidContactGeometry,
    response: DEMContactResponse,
    /,
    *,
    particle_capacity: int,
    ambient_dimension: int,
    angular_dimension: int,
    execution: ParticleExecutionPolicy,
    precision: ParticlePrecisionPolicy,
) -> RigidBodyLoad:
    """Reduce common-point forces and component moments onto COM owners.

    Contact laws report lever moments about the contact shape origin. The
    arm difference supplies the exact parallel-axis force moment for clumps;
    intrinsic rolling/twisting moments are retained unchanged.
    """
    pair_force = precision.accumulation(response.pair_force)
    if geometry.geometry_id == "rigid-contact:clump-component-spheres":
        left_torque = precision.accumulation(
            response.left_torque
            + sphere_lever_torque(
                geometry.left_owner_arm - geometry.left_contact_arm,
                response.pair_force,
                ambient_dimension,
            )
        )
        right_torque = precision.accumulation(
            response.right_torque
            + sphere_lever_torque(
                geometry.right_owner_arm - geometry.right_contact_arm,
                -response.pair_force,
                ambient_dimension,
            )
        )
    else:
        left_torque = precision.accumulation(response.left_torque)
        right_torque = precision.accumulation(response.right_torque)
    if execution.kernel_backend == "reference":
        force = _reduce_owner_payload(
            geometry,
            pair_force,
            -pair_force,
            response.active,
            particle_capacity,
            execution.accumulation,
        )
        torque = _reduce_owner_payload(
            geometry,
            left_torque,
            right_torque,
            response.active,
            particle_capacity,
            execution.accumulation,
        )
        return RigidBodyLoad(force, torque)
    left = jnp.concatenate((pair_force, left_torque), axis=-1)
    right = jnp.concatenate((-pair_force, right_torque), axis=-1)
    reduced = _reduce_owner_payload(
        geometry, left, right, response.active, particle_capacity, execution.accumulation
    )
    if reduced.shape != (particle_capacity, ambient_dimension + angular_dimension):
        raise ValueError("Fused DEM reduction produced an invalid payload shape.")
    return RigidBodyLoad(reduced[:, :ambient_dimension], reduced[:, ambient_dimension:])


__all__ = ["reduce_rigid_contact"]
