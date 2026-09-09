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

from .._iteration import (
    bind_iteration_scope,
    IterationCapabilities,
    IterationCoordinates,
    IterationEvidence,
    IterationPhase,
    IterationPlan,
    IterationRecord,
    IterationSession,
    IterationSessionState,
)
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
    iteration_evidence: tuple[IterationEvidence, ...]
    iteration_session_state: IterationSessionState | None = eqx.field(static=True)


class MarkovChunkIterationMetrics(StrictModule):
    completed_draws: Array
    acceptance_rate: Array
    target_valid: Array


def _chunk_iteration_record(
    phase,
    completed_draws,
    chunk_index,
    metrics,
    /,
    *,
    terminal=False,
) -> IterationRecord:
    return IterationRecord(
        IterationCoordinates(
            phase,
            completed_draws,
            invocation=chunk_index,
            attempt=completed_draws,
            accepted=completed_draws,
            active=metrics.target_valid,
            committed=jnp.any(metrics.target_valid),
            terminal=terminal,
        ),
        jnp.where(metrics.target_valid, 0, 1).astype(jnp.int32),
        metrics,
    )


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
    iteration: IterationPlan | None = None,
    session: IterationSession | None = None,
) -> MarkovChunkResult:
    """Execute exact continuation chunks without advancing inactive tail draws."""
    if not isinstance(plan, MarkovChunkPlan):
        raise TypeError("plan must be MarkovChunkPlan.")
    if iteration is not None and not isinstance(iteration, IterationPlan):
        raise TypeError("iteration must be IterationPlan or None.")
    if session is not None and not isinstance(session, IterationSession):
        raise TypeError("session must be IterationSession or None.")
    current = state
    results: list[MarkovSampleResult] = []
    evidences: list[IterationEvidence] = []
    consumed = 0
    stopped = False
    scope = None
    if session is not None:
        capabilities = IterationCapabilities(
            ("terminal", "segment"),
            host_stop=True,
            host_streaming=True,
            mapped_records=True,
        )
        scope = bind_iteration_scope(
            IterationPlan(granularity="segment"),
            capabilities,
            f"markov-chunks:{kernel.kernel_id}",
        )
        stopped = session.emit(
            scope,
            _chunk_iteration_record(
                IterationPhase.START,
                0,
                0,
                MarkovChunkIterationMetrics(
                    jnp.asarray(0, dtype=jnp.int32),
                    jnp.zeros_like(state.log_target),
                    state.valid,
                ),
            ),
        )
    for chunk_index in range(plan.chunk_count):
        if stopped:
            break
        count = min(plan.chunk_size, plan.total_draws - consumed)
        result = sample_markov(
            target,
            kernel,
            current,
            key=key,
            num_draws=count,
            steps_per_draw=plan.steps_per_draw,
            iteration=iteration,
        )
        results.append(result)
        if result.iteration_evidence is not None:
            evidences.append(result.iteration_evidence)
        current = result.final_state
        consumed += count
        if session is not None:
            assert scope is not None
            stopped = session.emit(
                scope,
                _chunk_iteration_record(
                    IterationPhase.COMMIT,
                    consumed,
                    chunk_index,
                    MarkovChunkIterationMetrics(
                        jnp.asarray(consumed, dtype=jnp.int32),
                        result.acceptance_rate,
                        result.final_state.valid,
                    ),
                ),
            )
    if results:
        samples = jax.tree_util.tree_map(
            lambda *values: jnp.concatenate(values, axis=1),
            *(result.samples for result in results),
        )
        log_target_values = jnp.concatenate(
            [result.log_target for result in results], axis=1
        )
        accepted = jnp.concatenate([result.accepted for result in results], axis=1)
        target_valid = jnp.concatenate(
            [result.target_valid for result in results], axis=1
        )
    else:
        samples = jax.tree.map(
            lambda value: jnp.zeros(
                (state.num_chains, 0, *value.shape[1:]), dtype=value.dtype
            ),
            state.position,
        )
        log_target_values = jnp.zeros((state.num_chains, 0), dtype=state.log_target.dtype)
        accepted = jnp.zeros((state.num_chains, 0, plan.steps_per_draw), dtype=bool)
        target_valid = jnp.zeros_like(accepted)
    padding = plan.capacity - consumed
    samples = jax.tree_util.tree_map(
        lambda value: _pad_draw_axis(value, padding), samples
    )
    log_target_values = _pad_draw_axis(log_target_values, padding)
    accepted = _pad_draw_axis(accepted, padding)
    target_valid = _pad_draw_axis(target_valid, padding)
    active = jnp.arange(plan.capacity) < consumed
    if session is not None:
        assert scope is not None
        session.emit(
            scope,
            _chunk_iteration_record(
                IterationPhase.TERMINAL,
                consumed,
                len(results),
                MarkovChunkIterationMetrics(
                    jnp.asarray(consumed, dtype=jnp.int32),
                    (
                        jnp.mean(accepted[:, :consumed].astype(float), axis=(1, 2))
                        if consumed
                        else jnp.zeros_like(state.log_target)
                    ),
                    current.valid,
                ),
                terminal=True,
            ),
        )
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
        target_id=current.target_id,
        iteration_evidence=tuple(evidences),
        iteration_session_state=None if session is None else session.snapshot(),
    )


__all__ = [
    "MarkovChunkIterationMetrics",
    "MarkovChunkPlan",
    "MarkovChunkResult",
    "sample_markov_chunked",
]
