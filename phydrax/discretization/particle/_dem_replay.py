#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._tree_math import tree_where
from ._dem import DEMRuntimeState, PreparedSoftSphereDEMDynamics


class DEMCheckpointPolicy(StrictModule, NonTrainableState):
    interval: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(self, interval: int, /):
        value = int(interval)
        if value <= 0:
            raise ValueError("checkpoint interval must be positive.")
        self.interval = value
        self.policy_id = canonical_fingerprint(
            {"kind": "dem-checkpoint-policy", "interval": value}
        )


class DEMReplayRecord(StrictModule, NonTrainableState):
    successful: Array
    rejection_reasons: Array
    residual: Array
    work: Array
    route_digest: Array
    active_contacts: Array
    sliding_contacts: Array
    cache_epoch: Array
    replay_id: str = eqx.field(static=True)


class DEMReplayResult(StrictModule):
    final_state: DEMRuntimeState
    successful: Array
    replay: DEMReplayRecord


class DEMCheckpointVJPResult(StrictModule):
    primal: Array
    initial_state_cotangent: DEMRuntimeState
    replay: DEMReplayRecord
    replay_matched: Array


def _route_digest(state: DEMRuntimeState, /) -> Array:
    history = state.particle_history
    slots = jnp.arange(history.pair_keys.shape[0], dtype=jnp.int64)
    keys = jnp.where(history.valid, history.pair_keys + 1, 0)
    active = history.active.astype(jnp.int64)
    sliding = history.sliding.astype(jnp.int64)
    return jnp.sum(keys * (slots + 17) + 31 * active + 47 * sliding)


def checkpointed_dem_rollout(
    dynamics: PreparedSoftSphereDEMDynamics,
    initial_state: DEMRuntimeState,
    /,
    *,
    t0: float,
    step_size: float,
    step_count: int,
    checkpoint: DEMCheckpointPolicy,
    args: Any = None,
) -> DEMReplayResult:
    """Run a deterministic segmented scan whose blocks rematerialize in reverse AD."""

    if not isinstance(dynamics, PreparedSoftSphereDEMDynamics):
        raise TypeError("dynamics must be PreparedSoftSphereDEMDynamics.")
    if not isinstance(initial_state, DEMRuntimeState):
        raise TypeError("initial_state must be a DEMRuntimeState.")
    if not isinstance(checkpoint, DEMCheckpointPolicy):
        raise TypeError("checkpoint must be a DEMCheckpointPolicy.")
    count = int(step_count)
    if count <= 0 or count % checkpoint.interval != 0:
        raise ValueError(
            "step_count must be positive and divisible by checkpoint interval."
        )
    start = float(t0)
    dt_value = float(step_size)
    if not np.isfinite(start) or not np.isfinite(dt_value) or dt_value <= 0.0:
        raise ValueError(
            "rollout time and step_size must be finite with positive step_size."
        )
    dtype = initial_state.kinematics.position.dtype
    dt = jnp.asarray(dt_value, dtype=dtype)
    indices = jnp.arange(count, dtype=jnp.int32).reshape(
        count // checkpoint.interval, checkpoint.interval
    )

    def one_step(carry, index):
        state, prior_success = carry
        time = jnp.asarray(start, dtype=dtype) + index * dt
        detail = dynamics.step_detailed(index, time, state, dt, args)
        accepted = tree_where(prior_success, detail.accepted_state, state)
        successful = prior_success & detail.successful
        cache_epoch = (
            accepted.neighborhood_cache.epoch
            if accepted.neighborhood_cache is not None
            else jnp.zeros((), dtype=jnp.int32)
        )
        payload = (
            successful,
            detail.rejection_reasons,
            detail.residual,
            detail.work,
            _route_digest(accepted),
            detail.evaluation.diagnostics.active_contacts,
            detail.evaluation.diagnostics.sliding_contacts,
            cache_epoch,
        )
        return (accepted, successful), payload

    def block(carry, block_indices):
        return jax.lax.scan(one_step, carry, block_indices)

    checkpointed_block = jax.checkpoint(block)
    (final_state, successful), payload = jax.lax.scan(
        checkpointed_block,
        (initial_state, jnp.asarray(True)),
        indices,
    )
    flattened = jax.tree.map(lambda value: value.reshape((count,)), payload)
    replay = DEMReplayRecord(
        *flattened,
        canonical_fingerprint(
            {
                "kind": "dem-replay-record",
                "dynamics": dynamics.prepared_id,
                "checkpoint": checkpoint.policy_id,
                "t0": start,
                "step_size": dt_value,
                "step_count": count,
            }
        ),
    )
    return DEMReplayResult(final_state, successful, replay)


def checkpointed_dem_vjp(
    loss: Callable[[DEMRuntimeState], Array],
    dynamics: PreparedSoftSphereDEMDynamics,
    initial_state: DEMRuntimeState,
    cotangent: Array,
    /,
    *,
    t0: float,
    step_size: float,
    step_count: int,
    checkpoint: DEMCheckpointPolicy,
    args: Any = None,
) -> DEMCheckpointVJPResult:
    if not callable(loss):
        raise TypeError("loss must be callable.")
    forward = checkpointed_dem_rollout(
        dynamics,
        initial_state,
        t0=t0,
        step_size=step_size,
        step_count=step_count,
        checkpoint=checkpoint,
        args=args,
    )

    def terminal(state):
        result = checkpointed_dem_rollout(
            dynamics,
            state,
            t0=t0,
            step_size=step_size,
            step_count=step_count,
            checkpoint=checkpoint,
            args=args,
        )
        return loss(result.final_state)

    primal, pullback = jax.vjp(terminal, initial_state)
    initial_cotangent = pullback(jnp.asarray(cotangent, dtype=primal.dtype))[0]
    replayed = checkpointed_dem_rollout(
        dynamics,
        initial_state,
        t0=t0,
        step_size=step_size,
        step_count=step_count,
        checkpoint=checkpoint,
        args=args,
    )
    return DEMCheckpointVJPResult(
        primal,
        initial_cotangent,
        forward.replay,
        dem_replay_matches(forward.replay, replayed.replay),
    )


def dem_replay_matches(left: DEMReplayRecord, right: DEMReplayRecord, /) -> Array:
    if not isinstance(left, DEMReplayRecord) or not isinstance(right, DEMReplayRecord):
        raise TypeError("Replay comparison requires DEMReplayRecord values.")
    if left.replay_id != right.replay_id:
        return jnp.asarray(False)
    return (
        jnp.array_equal(left.successful, right.successful)
        & jnp.array_equal(left.rejection_reasons, right.rejection_reasons)
        & jnp.array_equal(left.residual, right.residual)
        & jnp.array_equal(left.work, right.work)
        & jnp.array_equal(left.route_digest, right.route_digest)
        & jnp.array_equal(left.active_contacts, right.active_contacts)
        & jnp.array_equal(left.sliding_contacts, right.sliding_contacts)
        & jnp.array_equal(left.cache_epoch, right.cache_epoch)
    )


__all__ = [
    "DEMCheckpointPolicy",
    "DEMCheckpointVJPResult",
    "DEMReplayRecord",
    "DEMReplayResult",
    "checkpointed_dem_vjp",
    "checkpointed_dem_rollout",
    "dem_replay_matches",
]
