#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Replayable distributed communication schedules.

Communication is represented as pure canonical-ID scatter/gather operators.  Runtime
transport is intentionally outside this module: a single rank and a distributed
runtime execute the identical recorded schedule, with no ``COMM_SELF`` branch.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


def _routes(value: ArrayLike, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{name} must be a one-dimensional integer array.")
    return array.astype(np.int64, copy=False)


class DistributedReplayEvent(StrictModule, NonTrainableState):
    """One recorded canonical-ID communication operation and its transpose."""

    canonical_ids: Array
    send_indices: Array
    receive_indices: Array
    sender_ranks: Array
    receiver_ranks: Array
    source_size: int = eqx.field(static=True)
    target_size: int = eqx.field(static=True)
    event_index: int = eqx.field(static=True)
    event_id: str = eqx.field(static=True)

    def __init__(
        self,
        canonical_ids: ArrayLike,
        send_indices: ArrayLike,
        receive_indices: ArrayLike,
        /,
        *,
        source_size: int,
        target_size: int,
        event_index: int,
        sender_ranks: ArrayLike | None = None,
        receiver_ranks: ArrayLike | None = None,
    ):
        ids = _routes(canonical_ids, "canonical_ids")
        send = _routes(send_indices, "send_indices")
        receive = _routes(receive_indices, "receive_indices")
        source_n, target_n, index = int(source_size), int(target_size), int(event_index)
        if (
            ids.size == 0
            or send.shape != ids.shape
            or receive.shape != ids.shape
            or source_n <= 0
            or target_n <= 0
            or index < 0
            or np.any(ids < 0)
            or np.unique(ids).size != ids.size
            or np.any(send < 0)
            or np.any(send >= source_n)
            or np.any(receive < 0)
            or np.any(receive >= target_n)
            or np.unique(receive).size != receive.size
        ):
            raise ValueError("Distributed replay event routes are invalid.")
        senders = (
            np.zeros(ids.shape, dtype=np.int32)
            if sender_ranks is None
            else _routes(sender_ranks, "sender_ranks")
        )
        receivers = (
            np.zeros(ids.shape, dtype=np.int32)
            if receiver_ranks is None
            else _routes(receiver_ranks, "receiver_ranks")
        )
        if (
            senders.shape != ids.shape
            or receivers.shape != ids.shape
            or np.any(senders < 0)
            or np.any(receivers < 0)
        ):
            raise ValueError("Distributed replay event ranks are invalid.")
        order = np.argsort(ids, kind="stable")
        ids, send, receive, senders, receivers = (
            ids[order],
            send[order],
            receive[order],
            senders[order],
            receivers[order],
        )
        self.canonical_ids = jnp.asarray(ids)
        self.send_indices = jnp.asarray(send, dtype=jnp.int32)
        self.receive_indices = jnp.asarray(receive, dtype=jnp.int32)
        self.sender_ranks = jnp.asarray(senders, dtype=jnp.int32)
        self.receiver_ranks = jnp.asarray(receivers, dtype=jnp.int32)
        self.source_size, self.target_size, self.event_index = source_n, target_n, index
        self.event_id = canonical_fingerprint(
            {
                "kind": "distributed-replay-event",
                "canonical_ids": array_tree_fingerprint(ids),
                "send_indices": array_tree_fingerprint(send),
                "receive_indices": array_tree_fingerprint(receive),
                "sender_ranks": array_tree_fingerprint(senders),
                "receiver_ranks": array_tree_fingerprint(receivers),
                "source_size": source_n,
                "target_size": target_n,
                "event_index": index,
            }
        )

    def communicate(self, source: ArrayLike, target: ArrayLike | None = None, /) -> Array:
        """Apply the recorded forward copy to a target partition buffer."""
        source_ = jnp.asarray(source)
        if source_.ndim == 0 or source_.shape[0] != self.source_size:
            raise ValueError("Replay source has the wrong leading dimension.")
        if target is None:
            target_ = jnp.zeros((self.target_size,) + source_.shape[1:], source_.dtype)
        else:
            target_ = jnp.asarray(target)
            if target_.shape != (self.target_size,) + source_.shape[1:]:
                raise ValueError("Replay target has incompatible shape.")
        return target_.at[self.receive_indices].set(source_[self.send_indices])

    def transpose_communicate(self, target_cotangent: ArrayLike, /) -> Array:
        """Apply the transpose communication, accumulating in canonical-ID order."""
        target = jnp.asarray(target_cotangent)
        if target.ndim == 0 or target.shape[0] != self.target_size:
            raise ValueError("Replay target cotangent has the wrong leading dimension.")
        result = jnp.zeros((self.source_size,) + target.shape[1:], target.dtype)
        return result.at[self.send_indices].add(target[self.receive_indices])


class DistributedReplaySchedule(StrictModule, NonTrainableState):
    """An ordered pure-data communication trace with reverse replay support."""

    events: tuple[DistributedReplayEvent, ...]
    schedule_id: str = eqx.field(static=True)

    def __init__(self, events: tuple[DistributedReplayEvent, ...], /):
        if not events or any(
            not isinstance(event, DistributedReplayEvent) for event in events
        ):
            raise ValueError("A distributed replay schedule requires replay events.")
        indices = tuple(event.event_index for event in events)
        if indices != tuple(range(len(events))):
            raise ValueError(
                "Distributed replay event indices must be consecutive from zero."
            )
        if any(
            events[index].target_size != events[index + 1].source_size
            for index in range(len(events) - 1)
        ):
            raise ValueError(
                "Adjacent distributed replay events have incompatible sizes."
            )
        self.events = events
        self.schedule_id = canonical_fingerprint(
            {
                "kind": "distributed-replay-schedule",
                "events": [event.event_id for event in events],
            }
        )

    def replay(self, source: ArrayLike, /) -> Array:
        """Replay all communication events in forward schedule order."""
        value = jnp.asarray(source)
        if value.ndim == 0 or value.shape[0] != self.events[0].source_size:
            raise ValueError("Replay source has the wrong leading dimension.")
        for event in self.events:
            value = event.communicate(value)
        return value

    def transpose_replay(self, target_cotangent: ArrayLike, /) -> Array:
        """Replay adjoint communication in exact reverse schedule order."""
        value = jnp.asarray(target_cotangent)
        if value.ndim == 0 or value.shape[0] != self.events[-1].target_size:
            raise ValueError("Replay target cotangent has the wrong leading dimension.")
        for event in reversed(self.events):
            value = event.transpose_communicate(value)
        return value


__all__ = ["DistributedReplayEvent", "DistributedReplaySchedule"]
