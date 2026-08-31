#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp

from ._dem_contact import DEMContactResponse
from ._pairwise import ParticlePairRelation, scatter_pair_exchange, scatter_pair_sum
from ._precision import ParticleExecutionPolicy, ParticlePrecisionPolicy
from ._rigid_sphere import RigidSphereLoad


def reduce_dem_contact(
    pairs: ParticlePairRelation,
    response: DEMContactResponse,
    /,
    *,
    particle_capacity: int,
    ambient_dimension: int,
    angular_dimension: int,
    execution: ParticleExecutionPolicy,
    precision: ParticlePrecisionPolicy,
) -> RigidSphereLoad:
    """Reduce contact force/torque through reference or one-pass fused payloads."""

    if execution.kernel_backend == "reference":
        force = scatter_pair_exchange(
            pairs,
            precision.accumulation(response.pair_force),
            size=particle_capacity,
            accumulation=execution.accumulation,
            valid=response.active,
        )
        torque = scatter_pair_sum(
            pairs,
            precision.accumulation(response.left_torque),
            precision.accumulation(response.right_torque),
            size=particle_capacity,
            accumulation=execution.accumulation,
            valid=response.active,
        )
        return RigidSphereLoad(force, torque)
    left = jnp.concatenate(
        (
            precision.accumulation(response.pair_force),
            precision.accumulation(response.left_torque),
        ),
        axis=-1,
    )
    right = jnp.concatenate(
        (
            -precision.accumulation(response.pair_force),
            precision.accumulation(response.right_torque),
        ),
        axis=-1,
    )
    reduced = scatter_pair_sum(
        pairs,
        left,
        right,
        size=particle_capacity,
        accumulation=execution.accumulation,
        valid=response.active,
    )
    expected_width = ambient_dimension + angular_dimension
    if reduced.shape != (particle_capacity, expected_width):
        raise ValueError("Fused DEM reduction produced an invalid payload shape.")
    return RigidSphereLoad(
        reduced[:, :ambient_dimension],
        reduced[:, ambient_dimension:],
    )


__all__ = ["reduce_dem_contact"]
