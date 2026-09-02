#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fixed-capacity semantic Markov chunks and replay evidence."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Key, PyTree

from .._strict import StrictModule
from ._markov import MarkovSampleResult, MarkovState, MetropolisHastings, sample_markov


class MarkovChunkPlan(StrictModule):
    total_draws: int = eqx.field(static=True)
    chunk_size: int = eqx.field(static=True)
    steps_per_draw: int = eqx.field(static=True)
    chunk_count: int = eqx.field(static=True)
    capacity: int = eqx.field(static=True)

    def __init__(self, total_draws: int, chunk_size: int, /, *, steps_per_draw: int = 1):
        total, chunk, steps = int(total_draws), int(chunk_size), int(steps_per_draw)
        if total <= 0 or chunk <= 0 or steps <= 0:
            raise ValueError(
                "total_draws, chunk_size, and steps_per_draw must be positive."
            )
        count = (total + chunk - 1) // chunk
        self.total_draws = total
        self.chunk_size = chunk
        self.steps_per_draw = steps
        self.chunk_count = count
        self.capacity = count * chunk


class MarkovChunkResult(StrictModule):
    samples: PyTree[Array]
    log_target: Array
    active: Array
    accepted: Array
    final_state: MarkovState
    target_valid: Array
    root_key: Array
    chunk_offsets: Array
    replay_exact: Array
    plan: MarkovChunkPlan
    kernel_id: str = eqx.field(static=True)
    proposal_id: str = eqx.field(static=True)
    target_id: str = eqx.field(static=True)


def _pad_draw_axis(value: Array, padding: int, /, *, axis: int = 1) -> Array:
    widths = [(0, 0)] * value.ndim
    widths[axis] = (0, padding)
    return jnp.pad(value, tuple(widths))


def sample_markov_chunked(
    target: Any,
    kernel: MetropolisHastings,
    state: MarkovState,
    /,
    *,
    key: Key[Array, ""],
    plan: MarkovChunkPlan,
) -> MarkovChunkResult:
    """Execute exact continuation chunks without advancing inactive tail draws."""
    if not isinstance(plan, MarkovChunkPlan):
        raise TypeError("plan must be MarkovChunkPlan.")
    current = state
    results: list[MarkovSampleResult] = []
    consumed = 0
    for _ in range(plan.chunk_count):
        count = min(plan.chunk_size, plan.total_draws - consumed)
        result = sample_markov(
            target,
            kernel,
            current,
            key=key,
            num_draws=count,
            steps_per_draw=plan.steps_per_draw,
        )
        results.append(result)
        current = result.final_state
        consumed += count
    samples = jax.tree_util.tree_map(
        lambda *values: jnp.concatenate(values, axis=1),
        *(result.samples for result in results),
    )
    log_target_values = jnp.concatenate([result.log_target for result in results], axis=1)
    accepted = jnp.concatenate([result.accepted for result in results], axis=1)
    target_valid = jnp.concatenate([result.target_valid for result in results], axis=1)
    padding = plan.capacity - plan.total_draws
    samples = jax.tree_util.tree_map(
        lambda value: _pad_draw_axis(value, padding), samples
    )
    log_target_values = _pad_draw_axis(log_target_values, padding)
    accepted = _pad_draw_axis(accepted, padding)
    target_valid = _pad_draw_axis(target_valid, padding)
    active = jnp.arange(plan.capacity) < plan.total_draws
    return MarkovChunkResult(
        samples=samples,
        log_target=log_target_values,
        active=active,
        accepted=accepted,
        target_valid=target_valid,
        final_state=current,
        root_key=jnp.asarray(key),
        chunk_offsets=jnp.arange(plan.chunk_count, dtype=jnp.int32) * plan.chunk_size,
        replay_exact=jnp.asarray(True),
        plan=plan,
        kernel_id=kernel.kernel_id,
        proposal_id=kernel.proposal.proposal_id,
        target_id=results[0].target_id,
    )


__all__ = ["MarkovChunkPlan", "MarkovChunkResult", "sample_markov_chunked"]
