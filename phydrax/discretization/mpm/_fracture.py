#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..splatting import ParticleGridSplatState


class MPMFractureTopologyState(StrictModule):
    velocity_field_slots: Array
    crack_side: Array
    topology_generation: Array
    successful: Array


class MPMFieldPartitionFracturePlan(StrictModule, NonTrainableState):
    damage_threshold: float = eqx.field(static=True)
    maximum_fields: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        maximum_fields: int,
        /,
        *,
        damage_threshold: float = 0.95,
    ):
        maximum = int(maximum_fields)
        threshold = float(damage_threshold)
        if maximum < 2 or not 0.0 < threshold <= 1.0:
            raise ValueError("Field-partition fracture configuration is invalid.")
        self.damage_threshold = threshold
        self.maximum_fields = maximum
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mpm-field-partition-fracture",
                "maximum_fields": maximum,
                "damage_threshold": threshold,
            }
        )

    def update(
        self,
        damage: ArrayLike,
        crack_indicator: ArrayLike,
        committed_slots: ArrayLike,
        topology_generation: ArrayLike,
        /,
    ) -> MPMFractureTopologyState:
        damage_ = jnp.asarray(damage)
        indicator = jnp.asarray(crack_indicator)
        slots = jnp.asarray(committed_slots, dtype=jnp.int32)
        generation = jnp.asarray(topology_generation, dtype=jnp.int32)
        if damage_.shape != indicator.shape or slots.shape != damage_.shape:
            raise ValueError("Fracture partition arrays must share particle shape.")
        split = damage_ >= self.damage_threshold
        side = jnp.where(indicator >= 0.0, 1, -1).astype(jnp.int32)
        proposed = jnp.where(split, jnp.where(side > 0, 1, 0), slots)
        overflow = jnp.any(proposed >= self.maximum_fields)
        changed = jnp.any(proposed != slots)
        accepted = jnp.where(overflow, slots, proposed)
        return MPMFractureTopologyState(
            accepted,
            side,
            generation + (changed & ~overflow).astype(jnp.int32),
            ~overflow,
        )


class CPICCompatibilityState(StrictModule):
    compatible: Array
    particle_tags: Array
    node_tags: Array
    topology_generation: Array
    successful: Array


class CPICFracturePlan(StrictModule, NonTrainableState):
    maximum_tags: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, maximum_tags: int, /):
        maximum = int(maximum_tags)
        if maximum < 2:
            raise ValueError("CPIC requires at least two compatibility tags.")
        self.maximum_tags = maximum
        self.plan_id = canonical_fingerprint(
            {"kind": "cpic-fracture", "maximum_tags": maximum}
        )

    def build(
        self,
        routes: ParticleGridSplatState,
        particle_tags: ArrayLike,
        node_tags: ArrayLike,
        topology_generation: ArrayLike,
        /,
    ) -> CPICCompatibilityState:
        particles = jnp.asarray(particle_tags, dtype=jnp.int32)
        nodes = jnp.asarray(node_tags, dtype=jnp.int32).reshape((-1,))
        if particles.shape != routes.supported_mask.shape:
            raise ValueError("CPIC particle tags must match particle capacity.")
        if nodes.size != routes.stencil.source_size:
            raise ValueError("CPIC node tags must match target size.")
        routed_tags = nodes[routes.stencil.indices]
        compatible = routes.stencil.valid & (
            (routed_tags < 0) | (routed_tags == particles[:, None])
        )
        valid_tags = jnp.all(
            (particles >= 0) & (particles < self.maximum_tags)
        ) & jnp.all((nodes >= -1) & (nodes < self.maximum_tags))
        return CPICCompatibilityState(
            compatible,
            particles,
            nodes,
            jnp.asarray(topology_generation, dtype=jnp.int32),
            valid_tags & jnp.all(jnp.any(compatible, axis=1)),
        )

    def route_velocities(
        self,
        compatibility: CPICCompatibilityState,
        routes: ParticleGridSplatState,
        grid_velocity: ArrayLike,
        particle_velocity: ArrayLike,
        affine_velocity: ArrayLike,
        /,
    ) -> Array:
        grid = jnp.asarray(grid_velocity).reshape((-1, routes.route_offsets.shape[-1]))
        gathered = grid[routes.stencil.indices]
        particle = jnp.asarray(particle_velocity)
        affine = jnp.asarray(affine_velocity)
        ghost = particle[:, None, :] + ein.contract(
            "pij,prj->pri", affine, routes.route_offsets
        )
        return jnp.where(compatibility.compatible[..., None], gathered, ghost)


__all__ = [
    "CPICCompatibilityState",
    "CPICFracturePlan",
    "MPMFieldPartitionFracturePlan",
    "MPMFractureTopologyState",
]
