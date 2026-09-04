#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact-count stochastic channel-state transitions with explicit PRNG lineage."""

from __future__ import annotations

from enum import IntFlag
from math import isfinite

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import numpy as np
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._units import ELECTROPHYSIOLOGY_UNITS


class StochasticChannelStatus(IntFlag):
    """Fail-closed stochastic transition status."""

    SUCCESS = 0
    NONFINITE_PROBABILITY = 1
    INVALID_PROBABILITY = 2
    COUNT_CONSERVATION_FAILURE = 4


class MarkovChannelPlan(StrictModule, NonTrainableState):
    """Continuous-time finite-state channel generator and fixed compartment count."""

    generator_per_ms: Array
    compartment_count: int = eqx.field(static=True)
    state_count: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, generator_per_ms: Array, compartment_count: int, /):
        generator_host = np.asarray(generator_per_ms, dtype=float)
        if (
            generator_host.ndim != 2
            or generator_host.shape[0] != generator_host.shape[1]
            or generator_host.shape[0] < 2
        ):
            raise ValueError("generator_per_ms must be square with at least two states.")
        if not np.all(np.isfinite(generator_host)):
            raise ValueError("generator_per_ms must be finite.")
        diagonal = np.diag(generator_host)
        off_diagonal = generator_host - np.diag(diagonal)
        if np.any(off_diagonal < 0.0) or np.any(diagonal > 0.0):
            raise ValueError(
                "A channel generator requires nonnegative off-diagonals and nonpositive diagonal."
            )
        if not np.allclose(np.sum(generator_host, axis=1), 0.0, rtol=0.0, atol=1.0e-12):
            raise ValueError("Every channel generator row must sum to zero.")
        if isinstance(compartment_count, bool) or not isinstance(compartment_count, int):
            raise TypeError("compartment_count must be an integer.")
        if compartment_count <= 0:
            raise ValueError("compartment_count must be positive.")
        self.generator_per_ms = jnp.asarray(generator_host, dtype=jnp.asarray(0.0).dtype)
        self.compartment_count = compartment_count
        self.state_count = generator_host.shape[0]
        self.plan_id = canonical_fingerprint(
            {
                "kind": "electrophysiology-markov-channel-v1",
                "generator_per_ms": generator_host.tolist(),
                "compartment_count": compartment_count,
                "units_id": ELECTROPHYSIOLOGY_UNITS.units_id,
            }
        )

    def prepare(self, dt_ms: float, /) -> PreparedMarkovChannel:
        return prepare_markov_channel(self, dt_ms)


class PreparedMarkovChannel(StrictModule, NonTrainableState):
    """Prepared row-stochastic transition matrix for one exact time step."""

    plan: MarkovChannelPlan
    transition_probability: Array
    dt_ms: float = eqx.field(static=True)
    runtime_id: str = eqx.field(static=True)

    def __init__(
        self, plan: MarkovChannelPlan, transition_probability: Array, dt_ms: float, /
    ):
        self.plan = plan
        self.transition_probability = transition_probability
        self.dt_ms = dt_ms
        self.runtime_id = canonical_fingerprint(
            {
                "kind": "prepared-electrophysiology-markov-channel-v1",
                "plan": plan.plan_id,
                "dt_ms": dt_ms,
            }
        )


class StochasticChannelState(StrictModule):
    """Integer state populations and next-use PRNG key."""

    counts: Array
    key: Array
    step_index: Array
    draw_count: Array


class PRNGLineage(StrictModule):
    """Parent, draw, and next keys proving single-use random lineage."""

    parent_key: Array
    draw_key: Array
    next_key: Array
    draw_index: Array


class StochasticChannelEvidence(StrictModule):
    """Population conservation and transition-probability evidence."""

    counts_before: Array
    counts_after: Array
    row_sum_residual: Array
    minimum_probability: Array
    finite: Array
    status: Array
    successful: Array


class StochasticChannelCandidate(StrictModule):
    """Uncommitted channel-state draw, evidence, and PRNG lineage."""

    proposed: StochasticChannelState
    lineage: PRNGLineage
    evidence: StochasticChannelEvidence


def prepare_markov_channel(
    plan: MarkovChannelPlan, dt_ms: float, /
) -> PreparedMarkovChannel:
    """Exponentiate a validated generator into a reusable transition matrix."""
    if not isinstance(plan, MarkovChannelPlan):
        raise TypeError("plan must be a MarkovChannelPlan.")
    if isinstance(dt_ms, bool):
        raise TypeError("dt_ms must be a real scalar, not bool.")
    dt = float(dt_ms)
    if not isfinite(dt) or dt <= 0.0:
        raise ValueError("dt_ms must be finite and positive.")
    probability = jsp.linalg.expm(plan.generator_per_ms * dt)
    probability = jnp.clip(probability, 0.0, 1.0)
    probability = probability / jnp.sum(probability, axis=1, keepdims=True)
    return PreparedMarkovChannel(plan, probability, dt)


def initialize_stochastic_channels(
    runtime: PreparedMarkovChannel, counts: Array, key: Array, /
) -> StochasticChannelState:
    """Validate exact nonnegative integer channel populations."""
    count_host = np.asarray(counts)
    expected = (runtime.plan.compartment_count, runtime.plan.state_count)
    if count_host.shape != expected:
        raise ValueError(f"counts must have shape {expected}.")
    if not np.issubdtype(count_host.dtype, np.integer):
        raise TypeError("counts must have an integer dtype.")
    if np.any(count_host < 0):
        raise ValueError("Channel counts must be nonnegative.")
    maximum_count = np.iinfo(np.int32).max
    if any(int(value) > maximum_count for value in count_host.flat):
        raise ValueError("Every channel-state count must fit signed int32 storage.")
    if any(sum(int(value) for value in row) > maximum_count for row in count_host):
        raise ValueError(
            "Every compartment's total channel population must fit signed int32 storage."
        )
    count_array = jnp.asarray(count_host, dtype=jnp.int32)
    key_array = jnp.asarray(key)
    if key_array.shape != jr.key(0).shape or key_array.dtype != jr.key(0).dtype:
        raise ValueError("key must be a JAX typed PRNG key.")
    return StochasticChannelState(
        count_array,
        key_array,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
    )


def _multinomial_exact(key: Array, count: Array, probabilities: Array, /) -> Array:
    state_count = probabilities.shape[0]
    draws = jnp.zeros((state_count,), dtype=jnp.int32)

    def sample(destination, carry):
        output, remaining_count, remaining_probability = carry
        probability = jnp.where(
            remaining_probability > 0.0,
            probabilities[destination] / remaining_probability,
            0.0,
        )
        probability = jnp.clip(probability, 0.0, 1.0)
        draw = jr.binomial(
            jr.fold_in(key, destination), n=remaining_count, p=probability
        ).astype(jnp.int32)
        output = output.at[destination].set(draw)
        return (
            output,
            remaining_count - draw,
            remaining_probability - probabilities[destination],
        )

    draws, remaining_count, _ = jax.lax.fori_loop(
        0,
        state_count - 1,
        sample,
        (draws, count, jnp.asarray(1.0, dtype=probabilities.dtype)),
    )
    return draws.at[state_count - 1].set(remaining_count)


def evaluate_stochastic_channel_transition(
    runtime: PreparedMarkovChannel, state: StochasticChannelState, /
) -> StochasticChannelCandidate:
    """Draw every source-state transition once under an explicit split lineage."""
    next_key, draw_key = jr.split(state.key)
    source_count = runtime.plan.compartment_count * runtime.plan.state_count
    source_keys = jr.split(draw_key, source_count)
    counts = state.counts.reshape((source_count,))
    probabilities = jnp.tile(
        runtime.transition_probability, (runtime.plan.compartment_count, 1)
    )
    transitions = jax.vmap(_multinomial_exact)(source_keys, counts, probabilities)
    transitions = transitions.reshape(
        (
            runtime.plan.compartment_count,
            runtime.plan.state_count,
            runtime.plan.state_count,
        )
    )
    proposed_counts = jnp.sum(transitions, axis=1, dtype=jnp.int32)
    before = jnp.sum(state.counts, axis=1)
    after = jnp.sum(proposed_counts, axis=1)
    row_residual = jnp.sum(runtime.transition_probability, axis=1) - 1.0
    minimum = jnp.min(runtime.transition_probability)
    finite = jnp.all(jnp.isfinite(runtime.transition_probability))
    probability_valid = (
        (minimum >= 0.0)
        & (jnp.max(runtime.transition_probability) <= 1.0)
        & (jnp.max(jnp.abs(row_residual)) <= 1.0e-5)
    )
    conserved = jnp.all(before == after)
    status = jnp.asarray(int(StochasticChannelStatus.SUCCESS), dtype=jnp.int32)
    status = jnp.where(
        finite,
        status,
        jnp.bitwise_or(status, int(StochasticChannelStatus.NONFINITE_PROBABILITY)),
    )
    status = jnp.where(
        probability_valid,
        status,
        jnp.bitwise_or(status, int(StochasticChannelStatus.INVALID_PROBABILITY)),
    )
    status = jnp.where(
        conserved,
        status,
        jnp.bitwise_or(status, int(StochasticChannelStatus.COUNT_CONSERVATION_FAILURE)),
    )
    successful = finite & probability_valid & conserved
    proposed = StochasticChannelState(
        proposed_counts, next_key, state.step_index + 1, state.draw_count + source_count
    )
    lineage = PRNGLineage(state.key, draw_key, next_key, state.draw_count)
    evidence = StochasticChannelEvidence(
        before, after, row_residual, minimum, finite, status, successful
    )
    return StochasticChannelCandidate(proposed, lineage, evidence)


def commit_stochastic_channel_transition(
    candidate: StochasticChannelCandidate, current: StochasticChannelState, /
) -> StochasticChannelState:
    """Commit a valid draw and consume its key, or preserve the parent key and counts."""
    return jax.lax.cond(
        candidate.evidence.successful,
        lambda _: candidate.proposed,
        lambda _: current,
        operand=None,
    )


__all__ = [
    "MarkovChannelPlan",
    "PRNGLineage",
    "PreparedMarkovChannel",
    "StochasticChannelCandidate",
    "StochasticChannelEvidence",
    "StochasticChannelState",
    "StochasticChannelStatus",
    "commit_stochastic_channel_transition",
    "evaluate_stochastic_channel_transition",
    "initialize_stochastic_channels",
    "prepare_markov_channel",
]
