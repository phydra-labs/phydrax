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
from ._distributed import DistributedContactEpoch


class ContactHaloExchangePlan(StrictModule, NonTrainableState):
    send_route_indices: Array
    send_target_ranks: Array
    send_valid: Array
    rank_count: int = eqx.field(static=True)
    halo_capacity: int = eqx.field(static=True)
    route_count: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @classmethod
    def from_distributed_epoch(
        cls,
        epoch: DistributedContactEpoch,
        /,
        *,
        rank_count: int,
        halo_capacity: int,
    ) -> ContactHaloExchangePlan:
        if not isinstance(epoch, DistributedContactEpoch):
            raise TypeError("epoch must be DistributedContactEpoch.")
        ranks = int(rank_count)
        capacity = int(halo_capacity)
        if ranks <= 0 or capacity < 0:
            raise ValueError("Halo rank count/capacity is invalid.")
        participant_ranks = np.asarray(epoch.participant_ranks)
        route_owner = np.asarray(epoch.route_owner)
        send_indices = np.zeros((ranks, capacity), dtype=np.int32)
        send_targets = np.zeros((ranks, capacity), dtype=np.int32)
        send_valid = np.zeros((ranks, capacity), dtype=bool)
        counts = np.zeros((ranks,), dtype=np.int32)
        for route, pair in enumerate(participant_ranks):
            first, second = int(pair[0]), int(pair[1])
            if first == second:
                continue
            owner = int(route_owner[route])
            target = second if owner == first else first
            slot = int(counts[owner])
            if slot < capacity:
                send_indices[owner, slot] = route
                send_targets[owner, slot] = target
                send_valid[owner, slot] = True
            counts[owner] += 1
        if np.any(counts > capacity):
            send_valid[:] = False
        return cls(
            jnp.asarray(send_indices),
            jnp.asarray(send_targets),
            jnp.asarray(send_valid),
            ranks,
            capacity,
            int(route_owner.size),
            canonical_fingerprint(
                {
                    "kind": "contact-halo-exchange-plan",
                    "partition": epoch.partition_id,
                    "epoch": epoch.epoch_id,
                    "send_indices": array_tree_fingerprint(send_indices),
                    "send_targets": array_tree_fingerprint(send_targets),
                    "send_valid": array_tree_fingerprint(send_valid),
                    "counts": array_tree_fingerprint(counts),
                }
            ),
        )


class ContactHaloPayload(StrictModule):
    values: Array
    route_indices: Array
    target_ranks: Array
    valid: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class ContactHaloExchangeEvidence(StrictModule):
    packed_count: Array
    received_count: Array
    duplicate_receives: Array
    conservation_defect: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class ContactHaloReduction(StrictModule):
    value: Array
    evidence: ContactHaloExchangeEvidence


def pack_contact_halo(
    plan: ContactHaloExchangePlan,
    route_values: ArrayLike,
    /,
) -> ContactHaloPayload:
    if not isinstance(plan, ContactHaloExchangePlan):
        raise TypeError("plan must be ContactHaloExchangePlan.")
    values = jnp.asarray(route_values)
    if values.shape[0] != plan.route_count:
        raise ValueError("Halo route values have invalid leading dimension.")
    gathered = values[plan.send_route_indices]
    mask = plan.send_valid
    condition = mask
    while condition.ndim < gathered.ndim:
        condition = condition[..., None]
    gathered = jnp.where(condition, gathered, 0.0)
    finite = jnp.all(jnp.isfinite(gathered))
    return ContactHaloPayload(
        gathered,
        plan.send_route_indices,
        plan.send_target_ranks,
        mask,
        finite,
        finite,
        plan.plan_id,
    )


def reduce_contact_halo(
    plan: ContactHaloExchangePlan,
    local_route_values: ArrayLike,
    received_values: ArrayLike,
    received_route_indices: ArrayLike,
    received_valid: ArrayLike,
    /,
) -> ContactHaloReduction:
    if not isinstance(plan, ContactHaloExchangePlan):
        raise TypeError("plan must be ContactHaloExchangePlan.")
    local = jnp.asarray(local_route_values)
    received = jnp.asarray(received_values, dtype=local.dtype)
    indices = jnp.asarray(received_route_indices, dtype=jnp.int32)
    valid = jnp.asarray(received_valid, dtype=bool)
    if (
        local.shape[0] != plan.route_count
        or received.shape[0] != indices.size
        or valid.shape != indices.shape
    ):
        raise ValueError("Received contact halo shapes are invalid.")
    if received.shape[1:] != local.shape[1:]:
        raise ValueError("Received contact halo value shape is incompatible.")
    safe = jnp.clip(indices, 0, plan.route_count - 1)
    condition = valid
    while condition.ndim < received.ndim:
        condition = condition[..., None]
    contribution = jnp.where(condition, received, 0.0)
    reduced = local.at[safe].add(contribution)
    equality = (
        (indices[:, None] == indices[None, :])
        & valid[:, None]
        & valid[None, :]
        & ~jnp.eye(indices.size, dtype=bool)
    )
    duplicates = jnp.sum(jnp.any(equality, axis=1), dtype=jnp.int32)
    expected_change = jnp.sum(contribution, axis=0)
    actual_change = jnp.sum(reduced - local, axis=0)
    conservation = jnp.max(jnp.abs(actual_change - expected_change), initial=0.0)
    finite = jnp.all(jnp.isfinite(reduced)) & jnp.isfinite(conservation)
    tolerance = (
        128.0
        * jnp.finfo(local.dtype).eps
        * jnp.maximum(1.0, jnp.max(jnp.abs(expected_change), initial=0.0))
    )
    evidence = ContactHaloExchangeEvidence(
        jnp.sum(plan.send_valid, dtype=jnp.int32),
        jnp.sum(valid, dtype=jnp.int32),
        duplicates,
        conservation,
        finite,
        finite & (duplicates == 0) & (conservation <= tolerance),
        plan.plan_id,
    )
    return ContactHaloReduction(reduced, evidence)


__all__ = [
    "ContactHaloExchangeEvidence",
    "ContactHaloExchangePlan",
    "ContactHaloPayload",
    "ContactHaloReduction",
    "pack_contact_halo",
    "reduce_contact_halo",
]
