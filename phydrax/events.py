#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ._fingerprint import canonical_fingerprint
from ._strict import StrictModule
from ._trainable import NonTrainableState


EVENT_PROPOSED = 0
EVENT_COMMITTED = 1
EVENT_DEFERRED = 2
EVENT_REJECTED = 3


class DeterministicEventAddress(StrictModule, NonTrainableState):
    realization_id: str = eqx.field(static=True)
    macroepoch: int = eqx.field(static=True)
    process_id: int = eqx.field(static=True)
    source_id: int = eqx.field(static=True)
    generation: int = eqx.field(static=True)
    channel: int = eqx.field(static=True)
    ordinal: int = eqx.field(static=True)
    address_id: str = eqx.field(static=True)

    def __init__(
        self,
        realization_id: str,
        macroepoch: int,
        process_id: int,
        source_id: int,
        generation: int,
        channel: int,
        ordinal: int,
        /,
    ):
        realization = str(realization_id).strip()
        integers = tuple(
            int(value)
            for value in (
                macroepoch,
                process_id,
                source_id,
                generation,
                channel,
                ordinal,
            )
        )
        if not realization or any(value < 0 for value in integers):
            raise ValueError("Deterministic event address is invalid.")
        self.realization_id = realization
        (
            self.macroepoch,
            self.process_id,
            self.source_id,
            self.generation,
            self.channel,
            self.ordinal,
        ) = integers
        self.address_id = canonical_fingerprint(
            {
                "kind": "deterministic-event-address",
                "realization_id": realization,
                "values": list(integers),
            }
        )


class FixedCapacityEventState(StrictModule):
    source_ids: Array
    recipient_ids: Array
    channels: Array
    statuses: Array
    active_mask: Array
    overflow: Array

    def __init__(
        self,
        source_ids: ArrayLike,
        recipient_ids: ArrayLike,
        channels: ArrayLike,
        statuses: ArrayLike,
        active_mask: ArrayLike,
        overflow: ArrayLike,
        /,
    ):
        source = jnp.asarray(source_ids)
        recipient = jnp.asarray(recipient_ids, dtype=source.dtype)
        channel = jnp.asarray(channels, dtype=jnp.int32)
        status = jnp.asarray(statuses, dtype=jnp.int8)
        active = jnp.asarray(active_mask, dtype=bool)
        overflow_ = jnp.asarray(overflow, dtype=bool).reshape(())
        if (
            recipient.shape != source.shape
            or channel.shape != source.shape
            or status.shape != source.shape
            or active.shape != source.shape
        ):
            raise ValueError("Fixed-capacity event arrays must share a shape.")
        status = eqx.error_if(
            status,
            jnp.any(active & ((status < EVENT_PROPOSED) | (status > EVENT_REJECTED))),
            "Active event status is invalid.",
        )
        self.source_ids = source
        self.recipient_ids = recipient
        self.channels = channel
        self.statuses = status
        self.active_mask = active
        self.overflow = overflow_

    @property
    def committed(self) -> Array:
        return self.active_mask & (self.statuses == EVENT_COMMITTED)

    @property
    def deferred(self) -> Array:
        return self.active_mask & (self.statuses == EVENT_DEFERRED)

    @property
    def rejected(self) -> Array:
        return self.active_mask & (self.statuses == EVENT_REJECTED)


__all__ = [
    "DeterministicEventAddress",
    "EVENT_COMMITTED",
    "EVENT_DEFERRED",
    "EVENT_PROPOSED",
    "EVENT_REJECTED",
    "FixedCapacityEventState",
]
