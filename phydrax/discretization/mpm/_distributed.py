#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._numerics._compensated import compensated_sum
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class MPMDistributedPlan(StrictModule, NonTrainableState):
    logical_grid_shape: tuple[int, ...] = eqx.field(static=True)
    block_shape: tuple[int, ...] = eqx.field(static=True)
    block_owner: Array
    device_count: int = eqx.field(static=True)
    particle_capacity_per_device: int = eqx.field(static=True)
    halo_blocks: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        logical_grid_shape,
        block_shape,
        block_owner: ArrayLike,
        /,
        *,
        device_count: int,
        particle_capacity_per_device: int,
        halo_blocks: int = 1,
    ):
        grid = tuple(int(value) for value in logical_grid_shape)
        block = tuple(int(value) for value in block_shape)
        owners = np.asarray(block_owner, dtype=np.int32)
        devices = int(device_count)
        capacity = int(particle_capacity_per_device)
        halo = int(halo_blocks)
        block_grid = tuple(g // b for g, b in zip(grid, block, strict=True))
        if (
            len(grid) != len(block)
            or any(
                g <= 0 or b <= 0 or g % b != 0 for g, b in zip(grid, block, strict=True)
            )
            or owners.shape != block_grid
            or devices <= 0
            or capacity <= 0
            or halo < 0
            or np.any(owners < 0)
            or np.any(owners >= devices)
        ):
            raise ValueError("Distributed MPM ownership plan is invalid.")
        self.logical_grid_shape = grid
        self.block_shape = block
        self.block_owner = jnp.asarray(owners)
        self.device_count = devices
        self.particle_capacity_per_device = capacity
        self.halo_blocks = halo
        self.plan_id = canonical_fingerprint(
            {
                "kind": "distributed-mpm-plan",
                "logical_grid_shape": grid,
                "block_shape": block,
                "block_owner": array_tree_fingerprint(owners),
                "device_count": devices,
                "particle_capacity_per_device": capacity,
                "halo_blocks": halo,
            }
        )


class MPMParticleMigration(StrictModule):
    owner: Array
    previous_owner: Array
    per_device_count: Array
    migrated: Array
    overflow: Array
    successful: Array


class MPMDistributedTransaction(StrictModule):
    local_success: Array
    global_success: Array
    failure_shards: Array
    commit_generation: Array


class MPMDistributedEvidence(StrictModule):
    migration: MPMParticleMigration
    halo_checksum: Array
    reduction_defect: Array
    transaction: MPMDistributedTransaction
    successful: Array


class MPMShardCheckpointManifest(StrictModule, NonTrainableState):
    generation: int = eqx.field(static=True)
    shard_payload_ids: tuple[str, ...] = eqx.field(static=True)
    ownership_plan_id: str = eqx.field(static=True)
    global_manifest_id: str = eqx.field(static=True)

    def __init__(
        self,
        generation: int,
        shard_payload_ids: Sequence[str],
        ownership_plan_id: str,
        /,
    ):
        generation_ = int(generation)
        shards = tuple(str(value) for value in shard_payload_ids)
        owner = str(ownership_plan_id)
        if (
            generation_ < 0
            or not shards
            or any(not value for value in shards)
            or not owner
        ):
            raise ValueError("Distributed checkpoint manifest is incomplete.")
        self.generation = generation_
        self.shard_payload_ids = shards
        self.ownership_plan_id = owner
        self.global_manifest_id = canonical_fingerprint(
            {
                "kind": "mpm-shard-checkpoint-manifest",
                "generation": generation_,
                "shards": shards,
                "ownership": owner,
            }
        )


def particle_owners(
    plan: MPMDistributedPlan,
    position: ArrayLike,
    bounds: ArrayLike,
    /,
) -> Array:
    value = jnp.asarray(position)
    bounds_ = jnp.asarray(bounds, dtype=value.dtype)
    if value.shape[-1] != len(plan.logical_grid_shape) or bounds_.shape != (
        2,
        value.shape[-1],
    ):
        raise ValueError("Distributed particle positions/bounds changed dimension.")
    normalized = (value - bounds_[0]) / (bounds_[1] - bounds_[0])
    logical = jnp.floor(normalized * jnp.asarray(plan.logical_grid_shape)).astype(
        jnp.int32
    )
    logical = jnp.clip(logical, 0, jnp.asarray(plan.logical_grid_shape) - 1)
    block = logical // jnp.asarray(plan.block_shape)
    return plan.block_owner[tuple(block[..., axis] for axis in range(block.shape[-1]))]


def migrate_particles(
    plan: MPMDistributedPlan,
    position: ArrayLike,
    bounds: ArrayLike,
    previous_owner: ArrayLike,
    active: ArrayLike,
    /,
) -> MPMParticleMigration:
    owner = particle_owners(plan, position, bounds)
    previous = jnp.asarray(previous_owner, dtype=jnp.int32)
    active_ = jnp.asarray(active, dtype=bool)
    if owner.shape != previous.shape or owner.shape != active_.shape:
        raise ValueError("Distributed particle ownership arrays changed shape.")
    counts = jnp.bincount(
        jnp.where(active_, owner, 0),
        weights=active_.astype(jnp.int32),
        length=plan.device_count,
    ).astype(jnp.int32)
    overflow = counts > plan.particle_capacity_per_device
    migrated = active_ & (owner != previous)
    return MPMParticleMigration(
        owner,
        previous,
        counts,
        migrated,
        overflow,
        ~jnp.any(overflow),
    )


def exchange_block_halo(
    plan: MPMDistributedPlan,
    block_values: ArrayLike,
    /,
) -> tuple[Array, Array]:
    values = jnp.asarray(block_values)
    if values.shape[: len(plan.block_owner.shape)] != plan.block_owner.shape:
        raise ValueError("Distributed halo values must begin with block-grid shape.")
    accumulated = values
    for axis in range(len(plan.logical_grid_shape)):
        for shift in range(1, plan.halo_blocks + 1):
            accumulated = accumulated + jnp.roll(values, shift, axis=axis)
            accumulated = accumulated + jnp.roll(values, -shift, axis=axis)
    checksum = jnp.asarray(
        int(hashlib.sha256(np.asarray(plan.block_owner).tobytes()).hexdigest()[:16], 16),
        dtype=jnp.uint64,
    )
    return accumulated, checksum


def distributed_p2g_reduce(shard_values: ArrayLike, /) -> tuple[Array, Array]:
    values = jnp.asarray(shard_values)
    if values.ndim < 1:
        raise ValueError("Distributed P2G values need leading shard axis.")
    reduced = compensated_sum(values, axis=0)
    reference = jnp.sum(values, axis=0)
    defect = jnp.linalg.norm(reduced - reference) / jnp.maximum(
        1.0, jnp.linalg.norm(reduced)
    )
    return reduced, defect


def distributed_global_transaction(
    local_success: ArrayLike,
    accepted_generation: ArrayLike,
    /,
) -> MPMDistributedTransaction:
    success = jnp.asarray(local_success, dtype=bool)
    generation = jnp.asarray(accepted_generation, dtype=jnp.int32)
    global_success = jnp.all(success)
    failures = jnp.nonzero(~success, size=success.size, fill_value=-1)[0]
    return MPMDistributedTransaction(
        success,
        global_success,
        failures,
        generation + global_success.astype(jnp.int32),
    )


__all__ = [
    "MPMDistributedEvidence",
    "MPMDistributedPlan",
    "MPMDistributedTransaction",
    "MPMParticleMigration",
    "MPMShardCheckpointManifest",
    "distributed_global_transaction",
    "distributed_p2g_reduce",
    "exchange_block_halo",
    "migrate_particles",
    "particle_owners",
]
