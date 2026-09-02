#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


CheckpointedScanMode: TypeAlias = Literal["full", "step", "block", "scheduled"]
ReplayCostModel: TypeAlias = Literal["uniform", "declared"]


class AdaptiveReplayPreparationPolicy(StrictModule, NonTrainableState):
    """Host budgets for a static cost-aware recomputation schedule."""

    maximum_checkpoint_bytes: int = eqx.field(static=True)
    maximum_schedule_operations: int = eqx.field(static=True)
    cost_model: ReplayCostModel = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        maximum_checkpoint_bytes: int,
        maximum_schedule_operations: int,
        /,
        *,
        cost_model: ReplayCostModel = "uniform",
    ):
        checkpoint_bytes = int(maximum_checkpoint_bytes)
        operations = int(maximum_schedule_operations)
        if checkpoint_bytes < 1 or operations < 1:
            raise ValueError("Replay byte and operation budgets must be positive.")
        if cost_model not in ("uniform", "declared"):
            raise ValueError("Unknown adaptive replay cost model.")
        self.maximum_checkpoint_bytes = checkpoint_bytes
        self.maximum_schedule_operations = operations
        self.cost_model = cost_model
        self.policy_id = canonical_fingerprint(
            {
                "kind": "adaptive-replay-preparation-policy",
                "maximum_checkpoint_bytes": checkpoint_bytes,
                "maximum_schedule_operations": operations,
                "cost_model": cost_model,
            }
        )


class PreparedReplaySchedule(StrictModule, NonTrainableState):
    """Immutable contiguous checkpoint/recomputation split tree."""

    block_lengths: tuple[int, ...] = eqx.field(static=True)
    checkpoint_slots: int = eqx.field(static=True)
    checkpoint_bytes: int = eqx.field(static=True)
    schedule_operations: int = eqx.field(static=True)
    predicted_work: float = eqx.field(static=True)
    step_count: int = eqx.field(static=True)
    state_bytes: int = eqx.field(static=True)
    schedule_id: str = eqx.field(static=True)


def _optimal_replay_blocks(costs: np.ndarray, block_count: int, /) -> tuple[int, ...]:
    count = int(costs.size)
    blocks = min(int(block_count), count)
    prefix = np.concatenate((np.zeros((1,), dtype=float), np.cumsum(costs)))
    infinity = float("inf")
    objective = np.full((blocks + 1, count + 1), infinity)
    predecessor = np.full((blocks + 1, count + 1), -1, dtype=np.int32)
    objective[0, 0] = 0.0
    for used in range(1, blocks + 1):
        for end in range(used, count + 1):
            for start in range(used - 1, end):
                length = end - start
                recomputation = (length - 1) * (prefix[end] - prefix[start])
                candidate = objective[used - 1, start] + recomputation
                if candidate < objective[used, end]:
                    objective[used, end] = candidate
                    predecessor[used, end] = start
    lengths: list[int] = []
    end = count
    for used in range(blocks, 0, -1):
        start = int(predecessor[used, end])
        if start < 0:
            raise ValueError("Replay schedule dynamic program found no feasible split.")
        lengths.append(end - start)
        end = start
    return tuple(reversed(lengths))


def prepare_replay_schedule(
    step_count: int,
    state_bytes: int,
    policy: AdaptiveReplayPreparationPolicy,
    /,
    *,
    step_costs: Array | None = None,
) -> PreparedReplaySchedule:
    """Prepare a bounded static split tree; runtime never observes or adapts cost."""

    if not isinstance(policy, AdaptiveReplayPreparationPolicy):
        raise TypeError("policy must be AdaptiveReplayPreparationPolicy.")
    count = int(step_count)
    bytes_per_state = int(state_bytes)
    if count < 1 or bytes_per_state < 1:
        raise ValueError("Replay step_count and state_bytes must be positive.")
    slots = min(count, policy.maximum_checkpoint_bytes // bytes_per_state)
    if slots < 1:
        raise ValueError("Replay byte budget cannot hold one checkpoint state.")
    if policy.cost_model == "uniform":
        if step_costs is not None:
            raise ValueError("Uniform replay preparation does not accept step_costs.")
        costs = np.ones((count,), dtype=float)
    else:
        if step_costs is None:
            raise ValueError("Declared replay preparation requires step_costs.")
        costs = np.asarray(step_costs, dtype=float)
        if costs.shape != (count,) or np.any(~np.isfinite(costs)) or np.any(costs < 0.0):
            raise ValueError("Declared replay costs must be finite nonnegative steps.")
    lengths = _optimal_replay_blocks(costs, slots)
    predicted = float(
        np.sum(costs)
        + sum(
            (length - 1) * np.sum(costs[start : start + length])
            for start, length in zip(np.cumsum((0, *lengths[:-1])), lengths, strict=True)
        )
    )
    operations = count + sum(length - 1 for length in lengths)
    if operations > policy.maximum_schedule_operations:
        raise ValueError("Prepared replay schedule exceeds the operation budget.")
    checkpoint_bytes = slots * bytes_per_state
    return PreparedReplaySchedule(
        lengths,
        slots,
        checkpoint_bytes,
        operations,
        predicted,
        count,
        bytes_per_state,
        schedule_id=canonical_fingerprint(
            {
                "kind": "prepared-replay-schedule",
                "policy": policy.policy_id,
                "block_lengths": lengths,
                "checkpoint_slots": slots,
                "checkpoint_bytes": checkpoint_bytes,
                "schedule_operations": operations,
                "predicted_work": predicted,
                "step_count": count,
                "state_bytes": bytes_per_state,
            }
        ),
    )


def _reshape_blocks(value: Array, block_count: int, block_size: int, /) -> Array:
    return value.reshape((block_count, block_size) + value.shape[1:])


def _flatten_blocks(value: Array, /) -> Array:
    return value.reshape((value.shape[0] * value.shape[1],) + value.shape[2:])


def checkpointed_scan(
    body: Callable[[Any, Any], tuple[Any, Any]],
    initial: Any,
    xs: Any,
    /,
    *,
    length: int,
    mode: CheckpointedScanMode,
    block_size: int | None = None,
    schedule: PreparedReplaySchedule | None = None,
) -> tuple[Any, Any]:
    """Run a fixed scan with declared rematerialization.

    Block and scheduled modes checkpoint each complete inner scan and retain only
    immutable boundary carries. ``xs`` must have the declared leading ``length``.
    """

    count = int(length)
    if count <= 0:
        raise ValueError("checkpointed_scan length must be positive.")
    if mode not in ("full", "step", "block", "scheduled"):
        raise ValueError("Unknown checkpointed scan mode.")
    if mode in ("full", "step"):
        if block_size is not None or schedule is not None:
            raise ValueError("Full/step replay accepts no block or schedule.")
        selected = jax.checkpoint(body) if mode == "step" else body
        return jax.lax.scan(selected, initial, xs, length=count)
    if mode == "scheduled":
        if block_size is not None:
            raise ValueError("Scheduled replay does not accept block_size.")
        if not isinstance(schedule, PreparedReplaySchedule):
            raise TypeError("Scheduled replay requires PreparedReplaySchedule.")
        if schedule.step_count != count or sum(schedule.block_lengths) != count:
            raise ValueError("Prepared replay schedule does not match scan length.")
        leaves = jax.tree.leaves(xs)
        if not leaves or any(value.shape[0] != count for value in leaves):
            raise ValueError("Every scheduled scan input must match length.")
        carry = initial
        outputs: list[Any] = []
        offset = 0
        for scheduled_size in schedule.block_lengths:
            scheduled_inputs = jax.tree.map(
                lambda value: value[offset : offset + scheduled_size], xs
            )

            def run_scheduled_block(block_carry, block_inputs):
                return jax.lax.scan(
                    body, block_carry, block_inputs, length=scheduled_size
                )

            carry, block_outputs = jax.checkpoint(run_scheduled_block)(
                carry, scheduled_inputs
            )
            outputs.append(block_outputs)
            offset += scheduled_size
        if outputs[0] is None:
            return carry, None
        return carry, jax.tree.map(
            lambda *values: jnp.concatenate(values, axis=0), *outputs
        )
    if schedule is not None:
        raise ValueError("Prepared schedule is valid only for scheduled replay.")

    size = None if block_size is None else int(block_size)
    if size is None or size <= 0:
        raise ValueError("Block checkpointing requires a positive block_size.")
    if size > count:
        size = count

    leaves = jax.tree.leaves(xs)
    if not leaves:
        raise ValueError("Block checkpointing requires explicit scan inputs.")
    if any(value.shape[0] != count for value in leaves):
        raise ValueError("Every checkpointed scan input must match length.")

    complete_count = count // size
    complete_length = complete_count * size
    remainder = count - complete_length
    carry = initial
    complete_outputs = None

    if complete_count:
        complete_inputs = jax.tree.map(
            lambda value: _reshape_blocks(value[:complete_length], complete_count, size),
            xs,
        )

        def run_block(block_carry, block_inputs):
            return jax.lax.scan(body, block_carry, block_inputs, length=size)

        carry, block_outputs = jax.lax.scan(
            jax.checkpoint(run_block),
            carry,
            complete_inputs,
            length=complete_count,
        )
        complete_outputs = (
            None
            if block_outputs is None
            else jax.tree.map(_flatten_blocks, block_outputs)
        )

    if not remainder:
        return carry, complete_outputs

    remainder_inputs = jax.tree.map(lambda value: value[complete_length:], xs)
    carry, remainder_outputs = jax.lax.scan(
        jax.checkpoint(body),
        carry,
        remainder_inputs,
        length=remainder,
    )
    if complete_outputs is None:
        return carry, remainder_outputs
    if remainder_outputs is None:
        return carry, complete_outputs
    return carry, jax.tree.map(
        lambda complete, remaining: jnp.concatenate((complete, remaining), axis=0),
        complete_outputs,
        remainder_outputs,
    )


__all__ = [
    "AdaptiveReplayPreparationPolicy",
    "CheckpointedScanMode",
    "PreparedReplaySchedule",
    "ReplayCostModel",
    "checkpointed_scan",
    "prepare_replay_schedule",
]
