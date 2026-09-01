#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class VortexReplayEpoch(StrictModule):
    times: Array
    states: Array
    accepted_mask: Array
    epoch_index: int = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    backend_ids: tuple[str, ...] = eqx.field(static=True)
    epoch_id: str = eqx.field(static=True)

    def __init__(
        self,
        times: ArrayLike,
        states: ArrayLike,
        accepted_mask: ArrayLike,
        /,
        *,
        epoch_index: int,
        topology_id: str,
        backend_ids: tuple[str, ...],
    ):
        times_, states_, accepted = (
            jnp.asarray(times),
            jnp.asarray(states),
            jnp.asarray(accepted_mask, dtype=bool),
        )
        if (
            times_.ndim != 1
            or states_.shape[0] != times_.size
            or accepted.shape != times_.shape
            or not str(topology_id)
            or any(not str(value) for value in backend_ids)
        ):
            raise ValueError("Replay epoch arrays or identities are invalid.")
        self.times, self.states, self.accepted_mask = times_, states_, accepted
        self.epoch_index, self.topology_id, self.backend_ids = (
            int(epoch_index),
            str(topology_id),
            tuple(str(value) for value in backend_ids),
        )
        self.epoch_id = canonical_fingerprint(
            {
                "kind": "vortex-replay-epoch",
                "epoch_index": int(epoch_index),
                "topology_id": str(topology_id),
                "backend_ids": self.backend_ids,
                "state_shape": tuple(int(value) for value in states_.shape),
                "time_count": int(times_.size),
            }
        )


class VortexTransitionPullback(StrictModule, NonTrainableState):
    source_indices: Array
    target_indices: Array
    weights: Array
    source_size: int = eqx.field(static=True)
    target_size: int = eqx.field(static=True)
    pullback_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_indices: ArrayLike,
        target_indices: ArrayLike,
        weights: ArrayLike,
        /,
        *,
        source_size: int,
        target_size: int,
    ):
        source, target, weight = (
            jnp.asarray(source_indices),
            jnp.asarray(target_indices),
            jnp.asarray(weights),
        )
        if (
            source.ndim != 1
            or target.shape != source.shape
            or weight.shape != source.shape
            or int(source_size) <= 0
            or int(target_size) <= 0
        ):
            raise ValueError("Transition pullback arrays are invalid.")
        self.source_indices, self.target_indices, self.weights = (
            source.astype(jnp.int32),
            target.astype(jnp.int32),
            weight,
        )
        self.source_size, self.target_size = int(source_size), int(target_size)
        self.pullback_id = canonical_fingerprint(
            {
                "kind": "vortex-transition-pullback",
                "source_size": self.source_size,
                "target_size": self.target_size,
                "route_count": int(source.size),
            }
        )

    def apply(self, target_cotangent: ArrayLike, /) -> Array:
        cotangent = jnp.asarray(target_cotangent)
        if cotangent.shape[0] != self.target_size:
            raise ValueError("Target cotangent size does not match transition pullback.")
        payload = cotangent[self.target_indices] * self.weights.reshape(
            self.weights.shape + (1,) * (cotangent.ndim - 1)
        )
        return (
            jnp.zeros((self.source_size,) + cotangent.shape[1:], dtype=cotangent.dtype)
            .at[self.source_indices]
            .add(payload)
        )


class VortexReplayResult(StrictModule):
    terminal_state: Any
    epoch_states: tuple[Any, ...]
    deterministic: Array
    replay_id: str = eqx.field(static=True)


class VortexReplayPlan(StrictModule, NonTrainableState):
    epochs: tuple[VortexReplayEpoch, ...]
    transition_pullbacks: tuple[VortexTransitionPullback, ...]
    replay_id: str = eqx.field(static=True)

    def __init__(
        self,
        epochs: tuple[VortexReplayEpoch, ...],
        transition_pullbacks: tuple[VortexTransitionPullback, ...] = (),
        /,
    ):
        if (
            not epochs
            or any(not isinstance(epoch, VortexReplayEpoch) for epoch in epochs)
            or len(transition_pullbacks) not in (0, len(epochs) - 1)
        ):
            raise ValueError("Replay epochs/pullbacks are invalid.")
        if any(
            epochs[index + 1].epoch_index != epochs[index].epoch_index + 1
            for index in range(len(epochs) - 1)
        ):
            raise ValueError("Replay epoch indices must be consecutive.")
        self.epochs, self.transition_pullbacks = epochs, transition_pullbacks
        self.replay_id = canonical_fingerprint(
            {
                "kind": "vortex-replay-plan",
                "epochs": [epoch.epoch_id for epoch in epochs],
                "pullbacks": [pullback.pullback_id for pullback in transition_pullbacks],
            }
        )

    def replay(
        self,
        transition: Callable[[Any, VortexReplayEpoch, VortexReplayEpoch], Any]
        | None = None,
        /,
    ) -> VortexReplayResult:
        state = self.epochs[0].states[-1]
        states: list[Any] = [state]
        deterministic = jnp.asarray(True)
        for index in range(1, len(self.epochs)):
            next_epoch = self.epochs[index]
            if transition is None:
                if state.shape != next_epoch.states[0].shape:
                    raise ValueError(
                        "Topology-changing replay requires a transition callback."
                    )
                state = next_epoch.states[0]
            else:
                state = transition(state, self.epochs[index - 1], next_epoch)
            deterministic = deterministic & jnp.all(state == next_epoch.states[0])
            state = next_epoch.states[-1]
            states.append(state)
        return VortexReplayResult(state, tuple(states), deterministic, self.replay_id)

    def reverse_transition(self, epoch_index: int, cotangent: ArrayLike, /) -> Array:
        if not self.transition_pullbacks:
            raise ValueError("Replay plan has no transition pullbacks.")
        index = int(epoch_index)
        if index < 0 or index >= len(self.transition_pullbacks):
            raise ValueError("Transition pullback index is out of bounds.")
        return self.transition_pullbacks[index].apply(cotangent)


__all__ = [
    "VortexReplayEpoch",
    "VortexReplayPlan",
    "VortexReplayResult",
    "VortexTransitionPullback",
]
