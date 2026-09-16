#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class TriggerBufferPlan(StrictModule, NonTrainableState):
    capacity: int = eqx.field(static=True)
    service_per_tick: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, capacity: int, service_per_tick: int, /):
        capacity_ = int(capacity)
        service = int(service_per_tick)
        if capacity_ < 1 or service < 0:
            raise ValueError("Buffer capacity must be positive and service nonnegative.")
        self.capacity = capacity_
        self.service_per_tick = service
        self.plan_id = canonical_fingerprint(
            {
                "kind": "trigger-buffer-plan",
                "capacity": capacity_,
                "service_per_tick": service,
            }
        )


class TriggerBufferResult(StrictModule, NonTrainableState):
    occupancy: Array
    accepted: Array
    dropped: Array
    dead_time: Array
    maximum_occupancy: Array
    total_dropped: Array
    valid: Array
    plan_id: str = eqx.field(static=True)


def replay_trigger_buffer(
    plan: TriggerBufferPlan,
    arrivals: ArrayLike,
    /,
) -> TriggerBufferResult:
    if not isinstance(plan, TriggerBufferPlan):
        raise TypeError("plan must be TriggerBufferPlan.")
    arrivals_ = jnp.asarray(arrivals, dtype=jnp.int32)
    if arrivals_.ndim != 1:
        raise ValueError("arrivals must be a one-dimensional tick series.")

    def step(occupancy, arrival):
        after_service = jnp.maximum(occupancy - plan.service_per_tick, 0)
        available = plan.capacity - after_service
        accepted = jnp.minimum(jnp.maximum(arrival, 0), available)
        dropped = jnp.maximum(arrival - accepted, 0)
        next_occupancy = after_service + accepted
        return next_occupancy, (next_occupancy, accepted, dropped, dropped > 0)

    _, history = jax.lax.scan(step, jnp.asarray(0, dtype=jnp.int32), arrivals_)
    occupancy, accepted, dropped, dead_time = history
    valid = jnp.all(arrivals_ >= 0) & jnp.all(occupancy <= plan.capacity)
    return TriggerBufferResult(
        occupancy,
        accepted,
        dropped,
        dead_time,
        jnp.max(occupancy, initial=0),
        jnp.sum(dropped),
        valid,
        plan.plan_id,
    )


__all__ = ["TriggerBufferPlan", "TriggerBufferResult", "replay_trigger_buffer"]
