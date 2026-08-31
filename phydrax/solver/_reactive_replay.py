#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._reactive_cfd_dem import (
    ReactiveCFDDEMCouplingState,
    ReactiveCFDDEMMacroStepResult,
)


class ReactiveCheckpointPolicy(StrictModule, NonTrainableState):
    interval: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(self, interval: int, /):
        value = int(interval)
        if value <= 0:
            raise ValueError("Reactive checkpoint interval must be positive.")
        self.interval = value
        self.policy_id = canonical_fingerprint(
            {"kind": "reactive-checkpoint-policy", "interval": value}
        )


class ReactiveReplayRecord(StrictModule, NonTrainableState):
    successful: Array
    coupling_residual: Array
    momentum_residual: Array
    energy_residual: Array
    species_residual: Array
    route_digest: Array
    replay_id: str = eqx.field(static=True)


class ReactiveReplayResult(StrictModule):
    final_state: ReactiveCFDDEMCouplingState
    successful: Array
    replay: ReactiveReplayRecord


class ReactiveCheckpointVJPResult(StrictModule):
    primal: Array
    initial_state_cotangent: ReactiveCFDDEMCouplingState
    replay: ReactiveReplayRecord
    replay_matched: Array


class ReactiveParameterEnsembleResult(StrictModule):
    outputs: Any
    successful: Array


def checkpointed_reactive_rollout(
    step_function: Callable[
        [ReactiveCFDDEMCouplingState, Array], ReactiveCFDDEMMacroStepResult
    ],
    initial_state: ReactiveCFDDEMCouplingState,
    step_count: int,
    checkpoint: ReactiveCheckpointPolicy,
    /,
) -> ReactiveReplayResult:
    if not callable(step_function):
        raise TypeError("step_function must be callable.")
    if not isinstance(initial_state, ReactiveCFDDEMCouplingState):
        raise TypeError("initial_state must be ReactiveCFDDEMCouplingState.")
    if not isinstance(checkpoint, ReactiveCheckpointPolicy):
        raise TypeError("checkpoint must be ReactiveCheckpointPolicy.")
    count = int(step_count)
    if count <= 0:
        raise ValueError("step_count must be positive.")
    padded = (
        (count + checkpoint.interval - 1) // checkpoint.interval
    ) * checkpoint.interval
    indices = jnp.arange(padded, dtype=jnp.int32).reshape((-1, checkpoint.interval))

    def one_step(carry, index):
        state, cumulative_success = carry

        def execute(_):
            result = step_function(state, index)
            if not isinstance(result, ReactiveCFDDEMMacroStepResult):
                raise TypeError(
                    "step_function must return ReactiveCFDDEMMacroStepResult."
                )
            evaluation = result.evaluation
            payload = (
                result.successful,
                evaluation.coupling_residual,
                jnp.linalg.norm(evaluation.momentum_residual),
                jnp.abs(evaluation.energy_residual),
                jnp.linalg.norm(evaluation.species_residual),
                _route_digest(result.accepted_state),
            )
            return result.accepted_state, cumulative_success & result.successful, payload

        def skip(_):
            dtype = state.dem_state.kinematics.position.dtype
            payload = (
                jnp.asarray(True),
                jnp.zeros((), dtype=dtype),
                jnp.zeros((), dtype=dtype),
                jnp.zeros((), dtype=dtype),
                jnp.zeros((), dtype=dtype),
                _route_digest(state),
            )
            return state, cumulative_success, payload

        next_state, success, payload = jax.lax.cond(
            index < count, execute, skip, operand=None
        )
        return (next_state, success), payload

    def block(carry, block_indices):
        return jax.lax.scan(one_step, carry, block_indices)

    checkpointed_block = jax.checkpoint(block)
    (final_state, successful), payload = jax.lax.scan(
        checkpointed_block,
        (initial_state, jnp.asarray(True)),
        indices,
    )
    flattened = jax.tree.map(lambda value: value.reshape((padded,))[:count], payload)
    replay = ReactiveReplayRecord(
        *flattened,
        canonical_fingerprint(
            {
                "kind": "reactive-replay-record",
                "checkpoint": checkpoint.policy_id,
                "step_count": count,
            }
        ),
    )
    return ReactiveReplayResult(final_state, successful, replay)


def checkpointed_reactive_vjp(
    loss: Callable[[ReactiveCFDDEMCouplingState], Array],
    step_function,
    initial_state,
    cotangent: Array,
    /,
    *,
    step_count: int,
    checkpoint: ReactiveCheckpointPolicy,
) -> ReactiveCheckpointVJPResult:
    if not callable(loss):
        raise TypeError("loss must be callable.")
    forward = checkpointed_reactive_rollout(
        step_function, initial_state, step_count, checkpoint
    )

    def terminal(state):
        result = checkpointed_reactive_rollout(
            step_function, state, step_count, checkpoint
        )
        return loss(result.final_state)

    primal, pullback = jax.vjp(terminal, initial_state)
    state_cotangent = pullback(jnp.asarray(cotangent, dtype=primal.dtype))[0]
    replayed = checkpointed_reactive_rollout(
        step_function, initial_state, step_count, checkpoint
    )
    replay_matched = reactive_replay_matches(forward.replay, replayed.replay)
    return ReactiveCheckpointVJPResult(
        primal,
        state_cotangent,
        forward.replay,
        replay_matched,
    )


def reactive_replay_matches(left: ReactiveReplayRecord, right: ReactiveReplayRecord, /):
    if not isinstance(left, ReactiveReplayRecord) or not isinstance(
        right, ReactiveReplayRecord
    ):
        raise TypeError("Reactive replay comparison requires replay records.")
    if left.replay_id != right.replay_id:
        return jnp.asarray(False)
    return (
        jnp.array_equal(left.successful, right.successful)
        & jnp.allclose(
            left.coupling_residual,
            right.coupling_residual,
            rtol=1.0e-12,
            atol=1.0e-14,
        )
        & jnp.allclose(
            left.momentum_residual,
            right.momentum_residual,
            rtol=1.0e-12,
            atol=1.0e-14,
        )
        & jnp.allclose(
            left.energy_residual,
            right.energy_residual,
            rtol=1.0e-12,
            atol=1.0e-14,
        )
        & jnp.allclose(
            left.species_residual,
            right.species_residual,
            rtol=1.0e-12,
            atol=1.0e-14,
        )
        & jnp.array_equal(left.route_digest, right.route_digest)
    )


def evaluate_reactive_parameter_ensemble(
    function: Callable[[Any], tuple[Any, Array]],
    parameters: Any,
    /,
) -> ReactiveParameterEnsembleResult:
    if not callable(function):
        raise TypeError("function must be callable.")
    outputs, successful = jax.vmap(function)(parameters)
    return ReactiveParameterEnsembleResult(outputs, successful)


def _route_digest(state):
    history = state.dem_state.particle_history
    slots = jnp.arange(history.pair_keys.shape[0], dtype=jnp.int64)
    keys = jnp.where(history.valid, history.pair_keys + 1, 0)
    active = history.active.astype(jnp.int64)
    conversion_active = jnp.sum(
        jnp.stack(
            tuple(
                jnp.sum(value.active.astype(jnp.int64))
                for value in state.conversion_state.batches
            )
        )
    )
    return jnp.sum(keys * (slots + 17) + 31 * active) + 47 * conversion_active


__all__ = [
    "ReactiveCheckpointPolicy",
    "ReactiveCheckpointVJPResult",
    "ReactiveParameterEnsembleResult",
    "ReactiveReplayRecord",
    "ReactiveReplayResult",
    "checkpointed_reactive_rollout",
    "checkpointed_reactive_vjp",
    "evaluate_reactive_parameter_ensemble",
    "reactive_replay_matches",
]
