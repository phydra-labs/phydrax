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
from ._search import ContactCandidateEpoch


class DistributedContactPartitionPlan(StrictModule, NonTrainableState):
    vertex_owner: Array
    rank_count: int = eqx.field(static=True)
    halo_capacity: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        vertex_owner: ArrayLike,
        /,
        *,
        rank_count: int,
        halo_capacity: int,
    ):
        owner = np.asarray(vertex_owner)
        ranks = int(rank_count)
        halo = int(halo_capacity)
        if owner.ndim != 1 or not np.issubdtype(owner.dtype, np.integer):
            raise TypeError("vertex_owner must be one integer vector.")
        if ranks <= 0 or halo < 0 or np.any(owner < 0) or np.any(owner >= ranks):
            raise ValueError("Distributed contact partition is invalid.")
        self.vertex_owner = jnp.asarray(owner, dtype=jnp.int32)
        self.rank_count = ranks
        self.halo_capacity = halo
        self.plan_id = canonical_fingerprint(
            {
                "kind": "distributed-contact-partition-plan",
                "vertex_owner": array_tree_fingerprint(owner),
                "rank_count": ranks,
                "halo_capacity": halo,
            }
        )


class DistributedContactEpoch(StrictModule, NonTrainableState):
    route_owner: Array
    participant_ranks: Array
    local_route_count: Array
    halo_route_count: Array
    halo_overflow: Array
    complete: Array
    partition_id: str = eqx.field(static=True)
    epoch_id: str = eqx.field(static=True)


def partition_contact_epoch(
    plan: DistributedContactPartitionPlan,
    epoch: ContactCandidateEpoch,
    /,
) -> DistributedContactEpoch:
    if not isinstance(plan, DistributedContactPartitionPlan):
        raise TypeError("plan must be DistributedContactPartitionPlan.")
    if not isinstance(epoch, ContactCandidateEpoch):
        raise TypeError("epoch must be ContactCandidateEpoch.")
    owners = []
    participant_ranks = []
    valid_values = []
    for batch in epoch.active_batches:
        safe = jnp.clip(batch.vertex_indices, 0, plan.vertex_owner.size - 1)
        endpoint_owner = plan.vertex_owner[safe]
        endpoint_valid = batch.vertex_indices >= 0
        maximum_owner = jnp.max(jnp.where(endpoint_valid, endpoint_owner, 0), axis=1)
        minimum_owner = jnp.min(
            jnp.where(endpoint_valid, endpoint_owner, plan.rank_count),
            axis=1,
        )
        owner = jnp.minimum(minimum_owner, maximum_owner)
        owners.append(owner)
        participant_ranks.append(jnp.stack((minimum_owner, maximum_owner), axis=-1))
        valid_values.append(batch.valid)
    if owners:
        route_owner = jnp.concatenate(tuple(owners))
        ranks = jnp.concatenate(tuple(participant_ranks))
        valid = jnp.concatenate(tuple(valid_values))
    else:
        route_owner = jnp.empty((0,), dtype=jnp.int32)
        ranks = jnp.empty((0, 2), dtype=jnp.int32)
        valid = jnp.empty((0,), dtype=bool)
    local_count = jnp.stack(
        tuple(
            jnp.sum(valid & (route_owner == rank), dtype=jnp.int32)
            for rank in range(plan.rank_count)
        )
    )
    cross_rank = valid & (ranks[:, 0] != ranks[:, 1])
    halo_count = jnp.stack(
        tuple(
            jnp.sum(
                cross_rank & ((ranks[:, 0] == rank) | (ranks[:, 1] == rank)),
                dtype=jnp.int32,
            )
            for rank in range(plan.rank_count)
        )
    )
    overflow = halo_count > plan.halo_capacity
    complete = epoch.successful & ~jnp.any(overflow)
    return DistributedContactEpoch(
        route_owner,
        ranks,
        local_count,
        halo_count,
        overflow,
        complete,
        plan.plan_id,
        epoch.epoch_id,
    )


__all__ = [
    "DistributedContactEpoch",
    "DistributedContactPartitionPlan",
    "partition_contact_epoch",
]
