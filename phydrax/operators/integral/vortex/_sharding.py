#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jaxtyping import Array, ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....discretization.vortex._source import VortexSourceState, VortexTargetState


class VortexShardingEvidence(StrictModule):
    device_count: Array
    source_bytes: Array
    target_bytes: Array
    estimated_workspace_bytes: Array
    memory_budget_bytes: Array
    collective_count: Array
    supported: Array
    policy_id: str = eqx.field(static=True)


class VortexShardingPolicy(StrictModule, NonTrainableState):
    mesh: Mesh = eqx.field(static=True)
    strategy: str = eqx.field(static=True)
    accumulation: str = eqx.field(static=True)
    memory_budget_bytes: int = eqx.field(static=True)
    axis_name: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        mesh: Mesh,
        /,
        *,
        strategy: str = "target-sharded",
        accumulation: str = "deterministic",
        memory_budget_bytes: int = 2**30,
    ):
        if (
            not isinstance(mesh, Mesh)
            or strategy
            not in (
                "target-sharded",
                "source-sharded",
                "grid-sharded",
                "tree-leaf-sharded",
            )
            or accumulation not in ("fast", "deterministic", "compensated")
            or int(memory_budget_bytes) <= 0
        ):
            raise ValueError("Vortex sharding policy controls are invalid.")
        if len(mesh.axis_names) != 1:
            raise ValueError("Vortex sharding currently requires one explicit mesh axis.")
        self.mesh, self.strategy, self.accumulation = mesh, strategy, accumulation
        self.memory_budget_bytes, self.axis_name = (
            int(memory_budget_bytes),
            str(mesh.axis_names[0]),
        )
        self.policy_id = canonical_fingerprint(
            {
                "kind": "vortex-sharding-policy",
                "mesh_shape": dict(mesh.shape),
                "strategy": strategy,
                "accumulation": accumulation,
                "memory_budget_bytes": self.memory_budget_bytes,
            }
        )

    def preflight(
        self,
        source: VortexSourceState,
        target: VortexTargetState,
        /,
        *,
        payload_components: int,
    ) -> VortexShardingEvidence:
        devices = int(np.prod(tuple(self.mesh.shape.values())))
        source_bytes = (
            source.positions.nbytes
            + source.strength.nbytes
            + source.active_mask.nbytes
            + (0 if source.core_radius is None else source.core_radius.nbytes)
            + (0 if source.volume is None else source.volume.nbytes)
        )
        target_bytes = target.positions.nbytes + (
            0 if target.source_indices is None else target.source_indices.nbytes
        )
        interactions = source.capacity * target.capacity
        workspace = (
            interactions
            * int(payload_components)
            * np.dtype(source.positions.dtype).itemsize
        )
        if self.strategy == "target-sharded":
            workspace = math.ceil(workspace / devices)
            collective_count = 0
        elif self.strategy == "source-sharded":
            workspace = math.ceil(workspace / devices)
            collective_count = 1
        else:
            workspace = math.ceil(workspace / devices)
            collective_count = 2
        supported = (workspace + source_bytes + target_bytes) <= self.memory_budget_bytes
        return VortexShardingEvidence(
            jnp.asarray(devices, dtype=jnp.int32),
            jnp.asarray(source_bytes),
            jnp.asarray(target_bytes),
            jnp.asarray(workspace),
            jnp.asarray(self.memory_budget_bytes),
            jnp.asarray(collective_count, dtype=jnp.int32),
            jnp.asarray(supported),
            self.policy_id,
        )

    def source_sharding(self, /) -> NamedSharding:
        partition = self.axis_name if self.strategy == "source-sharded" else None
        return NamedSharding(self.mesh, PartitionSpec(partition, None))

    def target_sharding(self, /) -> NamedSharding:
        partition = self.axis_name if self.strategy == "target-sharded" else None
        return NamedSharding(self.mesh, PartitionSpec(partition, None))

    def grid_sharding(self, dimension: int, /) -> NamedSharding:
        if self.strategy != "grid-sharded":
            raise ValueError("grid_sharding requires grid-sharded strategy.")
        return NamedSharding(
            self.mesh, PartitionSpec(self.axis_name, *([None] * (int(dimension) - 1)))
        )


def vortex_collective_sum(value: ArrayLike, policy: VortexShardingPolicy, /) -> Array:
    array = jnp.asarray(value)
    if policy.strategy != "source-sharded":
        return array
    if policy.accumulation == "fast":
        return jax.lax.psum(array, policy.axis_name)
    if policy.accumulation == "deterministic":
        gathered = jax.lax.all_gather(array, policy.axis_name, tiled=False)
        return jnp.sum(gathered, axis=0)
    gathered = jax.lax.all_gather(array, policy.axis_name, tiled=False)
    total = jnp.zeros_like(array)
    correction = jnp.zeros_like(array)
    for index in range(int(np.prod(tuple(policy.mesh.shape.values())))):
        adjusted = gathered[index] - correction
        next_total = total + adjusted
        correction = (next_total - total) - adjusted
        total = next_total
    return total


__all__ = ["VortexShardingEvidence", "VortexShardingPolicy", "vortex_collective_sum"]
