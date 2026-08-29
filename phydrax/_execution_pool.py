#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ._fingerprint import canonical_fingerprint
from ._strict import StrictModule


class PoolExecutionSignature(StrictModule):
    """Static topology, method, precision, backend, and shard bucket identity."""

    topology_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)
    precision_id: str = eqx.field(static=True)
    backend_id: str = eqx.field(static=True)
    shard_count: int = eqx.field(static=True)
    signature_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        topology_id: str,
        method_id: str,
        precision_id: str,
        backend_id: str,
        shard_count: int = 1,
    ):
        values = tuple(
            str(value) for value in (topology_id, method_id, precision_id, backend_id)
        )
        shards = int(shard_count)
        if any(not value for value in values) or shards < 1:
            raise ValueError(
                "Pool execution signature values must be non-empty and valid."
            )
        (
            self.topology_id,
            self.method_id,
            self.precision_id,
            self.backend_id,
        ) = values
        self.shard_count = shards
        self.signature_id = canonical_fingerprint(
            {
                "kind": "pool-execution-signature",
                "topology": values[0],
                "method": values[1],
                "precision": values[2],
                "backend": values[3],
                "shards": shards,
            }
        )


class PoolRefill(StrictModule):
    task_ids: Array
    refill_mask: Array
    terminal_rank: Array
    next_task: Array
    completed: Array


def refill_completed_tasks(
    task_ids: ArrayLike,
    terminal: ArrayLike,
    next_task: Any,
    completed: Any,
    task_count: int,
    /,
) -> PoolRefill:
    """Route completed lanes to pending task IDs with deterministic lane priority."""
    ids = jnp.asarray(task_ids, dtype=jnp.int32)
    terminal_ = jnp.asarray(terminal, dtype=bool)
    if ids.ndim != 1 or terminal_.shape != ids.shape:
        raise ValueError("task_ids and terminal must be matching rank-one arrays.")
    count = int(task_count)
    if count < 1:
        raise ValueError("task_count must be positive.")
    next_ = jnp.asarray(next_task, dtype=jnp.int32)
    completed_ = jnp.asarray(completed, dtype=jnp.int32)
    rank = jnp.cumsum(terminal_.astype(jnp.int32)) - 1
    remaining = count - next_
    refill = terminal_ & (rank < remaining)
    next_ids = jnp.where(
        refill,
        next_ + rank,
        jnp.where(terminal_, count, ids),
    ).astype(jnp.int32)
    terminal_count = jnp.sum(terminal_, dtype=jnp.int32)
    refill_count = jnp.sum(refill, dtype=jnp.int32)
    return PoolRefill(
        next_ids,
        refill,
        rank,
        next_ + refill_count,
        completed_ + terminal_count,
    )


class FrontierAllocation(StrictModule):
    positions: Array
    accepted: Array
    next_count: Array
    overflow: Array


def allocate_frontier_slots(
    current_count: Any,
    spawn_counts: ArrayLike,
    capacity: int,
    /,
) -> FrontierAllocation:
    """Allocate deterministic fixed-capacity child slots for adaptive frontiers."""
    count = jnp.asarray(current_count, dtype=jnp.int32)
    spawns = jnp.asarray(spawn_counts, dtype=jnp.int32)
    limit = int(capacity)
    if spawns.ndim != 1 or limit < 1:
        raise ValueError("spawn_counts must be rank one and capacity positive.")
    spawns = eqx.error_if(
        spawns,
        jnp.any(spawns < 0),
        "spawn_counts must be non-negative.",
    )
    offsets = jnp.cumsum(spawns) - spawns
    positions = count + offsets
    accepted = positions + spawns <= limit
    accepted_count = jnp.sum(jnp.where(accepted, spawns, 0), dtype=jnp.int32)
    return FrontierAllocation(
        positions,
        accepted,
        count + accepted_count,
        jnp.any(~accepted & (spawns > 0)),
    )


def semantic_task_keys(root_key: Array, task_ids: ArrayLike, /) -> Array:
    """Derive task keys independently of lane placement and completion order."""
    ids = jnp.asarray(task_ids, dtype=jnp.uint32)
    if ids.ndim != 1:
        raise ValueError("task_ids must be rank one.")
    return jax.vmap(lambda task_id: jax.random.fold_in(root_key, task_id))(ids)


__all__ = [
    "FrontierAllocation",
    "PoolExecutionSignature",
    "PoolRefill",
    "allocate_frontier_slots",
    "refill_completed_tasks",
    "semantic_task_keys",
]
