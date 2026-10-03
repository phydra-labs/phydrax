#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from phydrax._execution_runtime import ExecutionGroup
from phydrax._fingerprint import canonical_fingerprint
from phydrax._strict import StrictModule
from phydrax._trainable import NonTrainableState

from ._distributed_relations import (
    DistributedNeighborQueryPlan,
    DistributedOwnershipPlan,
    DistributedPointLayout,
)
from ._morton import MortonAddressPlan


class DistributedMortonNeighborEvidence(NonTrainableState, StrictModule):
    """Global exactness and identity evidence for a distributed query."""

    successful: jax.Array
    complete: jax.Array
    finite: jax.Array
    stable_ids_unique: jax.Array
    shard_count: jax.Array
    local_source_capacity: jax.Array
    local_target_capacity: jax.Array
    maximum_local_candidates: jax.Array
    communicated_targets: jax.Array


class DistributedMortonNeighborResult(NonTrainableState, StrictModule):
    """Global-sharded exact neighbors using global logical source indices."""

    source_indices: jax.Array
    source_stable_ids: jax.Array
    distance_squared: jax.Array
    valid: jax.Array
    counts: jax.Array
    status: jax.Array
    evidence: DistributedMortonNeighborEvidence


class DistributedMortonNeighborQueryPlan(StrictModule):
    """Exact k-nearest neighbors over contiguous logical-row shards.

    Source and target rows are split into equal contiguous logical blocks, one
    per owner of ``execution_group``. Targets are never replicated: each target
    contacts only the owners its certified search shell reaches (see
    :class:`DistributedNeighborQueryPlan`). Rows are returned in logical target
    order with global logical source indices.
    """

    query_plan: DistributedNeighborQueryPlan
    source_capacity: int = eqx.field(static=True)
    target_capacity: int = eqx.field(static=True)
    maximum_neighbors: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        address_plan: MortonAddressPlan,
        source_capacity: int,
        target_capacity: int,
        maximum_neighbors: int,
        execution_group: ExecutionGroup,
        *,
        maximum_candidates: int | None = None,
        maximum_remote_owners: int | None = None,
        halo_capacity: int | None = None,
    ) -> None:
        if not isinstance(execution_group, ExecutionGroup):
            raise TypeError("execution_group must be a live ExecutionGroup.")
        sources = int(source_capacity)
        targets = int(target_capacity)
        neighbors = int(maximum_neighbors)
        shards = len(execution_group.devices)
        if sources < 1 or targets < 1:
            raise ValueError("Distributed Morton capacities must be positive.")
        if sources % shards or targets % shards:
            raise ValueError(
                "source_capacity and target_capacity must be divisible by the "
                "execution-group device count."
            )
        if neighbors < 1 or neighbors > sources:
            raise ValueError("maximum_neighbors must lie in [1, source_capacity].")
        query_plan = DistributedNeighborQueryPlan(
            DistributedOwnershipPlan(address_plan, execution_group, sources // shards),
            DistributedOwnershipPlan(address_plan, execution_group, targets // shards),
            neighbors,
            maximum_remote_owners=maximum_remote_owners,
            halo_capacity=halo_capacity,
            maximum_candidates=maximum_candidates,
        )
        self.query_plan = query_plan
        self.source_capacity = sources
        self.target_capacity = targets
        self.maximum_neighbors = neighbors
        self.plan_id = canonical_fingerprint(
            {
                "kind": "distributed-morton-neighbor-query-plan",
                "query_plan_id": query_plan.plan_id,
                "source_capacity": sources,
                "target_capacity": targets,
            }
        )

    def query(
        self,
        source_points: jax.Array,
        target_points: jax.Array,
        *,
        source_mask: jax.Array | None = None,
        target_mask: jax.Array | None = None,
        source_stable_ids: jax.Array | None = None,
        target_stable_ids: jax.Array | None = None,
        exclude_self: bool = False,
        radius: float | None = None,
    ) -> DistributedMortonNeighborResult:
        core = self.query_plan.core
        dimension = core.source_ownership.address_plan.dimension
        sources = jnp.asarray(source_points)
        targets = jnp.asarray(target_points)
        if sources.shape != (self.source_capacity, dimension):
            raise ValueError(
                f"source_points must have shape {(self.source_capacity, dimension)}."
            )
        if targets.shape != (self.target_capacity, dimension):
            raise ValueError(
                f"target_points must have shape {(self.target_capacity, dimension)}."
            )
        if (
            exclude_self
            and target_stable_ids is None
            and (self.source_capacity != self.target_capacity)
        ):
            raise ValueError(
                "exclude_self with unequal capacities requires target_stable_ids."
            )
        source_layout = DistributedPointLayout.from_blocks(
            core.source_ownership,
            sources,
            active=source_mask,
            stable_ids=source_stable_ids,
        )
        target_layout = DistributedPointLayout.from_blocks(
            core.target_ownership,
            targets,
            active=target_mask,
            stable_ids=target_stable_ids,
        )
        result = self.query_plan.query(
            source_layout, target_layout, exclude_self=exclude_self, radius=radius
        )
        evidence = result.evidence
        return DistributedMortonNeighborResult(
            source_indices=result.source_logical,
            source_stable_ids=result.source_stable_ids,
            distance_squared=result.distance_squared,
            valid=result.valid,
            counts=result.counts,
            status=result.status,
            evidence=DistributedMortonNeighborEvidence(
                successful=evidence.successful & evidence.finite,
                complete=evidence.successful,
                finite=evidence.finite,
                stable_ids_unique=evidence.stable_ids_unique,
                shard_count=evidence.owner_count,
                local_source_capacity=jnp.asarray(
                    core.source_ownership.local_capacity, dtype=jnp.int32
                ),
                local_target_capacity=jnp.asarray(
                    core.target_ownership.local_capacity, dtype=jnp.int32
                ),
                maximum_local_candidates=evidence.required_candidates,
                communicated_targets=evidence.communicated_targets,
            ),
        )


__all__ = [
    "DistributedMortonNeighborEvidence",
    "DistributedMortonNeighborQueryPlan",
    "DistributedMortonNeighborResult",
]
