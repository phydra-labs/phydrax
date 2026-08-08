#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import prod
from typing import Any, Literal, TypeAlias

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import numpy as np
import optimistix as optx
from jaxtyping import Array, ArrayLike

from .._frozendict import frozendict
from .._strict import StrictModule
from ..stochastic import (
    AbstractJumpProcess,
    CompositeStochasticRealization,
    JUMP_INVALID_INTENSITY,
    JUMP_MAX_EVENTS,
    JUMP_SOLVER_FAILURE,
    JUMP_SUCCESS,
    JumpEventBatch,
    PoissonClockRealization,
    StochasticRealization,
    StochasticTrajectory,
    WienerRealization,
)
from ._differential import DifferentialProblem


JumpAlgorithm: TypeAlias = Literal["next_reaction", "direct_ssa"]
GeneratorBoundaryPolicy: TypeAlias = Literal["error", "suppress", "leak"]


def _time_interval(
    t0: ArrayLike,
    t1: ArrayLike,
    /,
    *,
    support: tuple[float, float],
) -> tuple[Array, Array]:
    start = jnp.asarray(t0, dtype=float)
    end = jnp.asarray(t1, dtype=float)
    if start.shape != () or end.shape != ():
        raise ValueError("Jump solve time bounds must be scalar.")
    if bool(~(jnp.isfinite(start) & jnp.isfinite(end) & (end > start))):
        raise ValueError("Jump solve requires finite bounds with t1 > t0.")
    if bool((start < support[0]) | (end > support[1])):
        raise ValueError("Jump solve interval must lie within realization support.")
    return start, end


def _query_times(values: ArrayLike, /, *, t0: Array, t1: Array) -> Array:
    times = jnp.asarray(values, dtype=float)
    if times.ndim != 1 or times.shape[0] <= 0:
        raise ValueError("save_times must be a non-empty vector.")
    if bool(jnp.any(~jnp.isfinite(times))) or bool(jnp.any(jnp.diff(times) <= 0.0)):
        raise ValueError("save_times must be finite and strictly increasing.")
    if bool((times[0] < t0) | (times[-1] > t1)):
        raise ValueError("save_times must lie within the solve interval.")
    return times


def _initial_state(process: AbstractJumpProcess, value: ArrayLike, /) -> Array:
    state = jnp.asarray(value)
    if tuple(state.shape) != process.state_shape:
        raise ValueError(
            f"initial_state must have shape {process.state_shape}; got {state.shape}."
        )
    if bool(jnp.any(~jnp.isfinite(state))):
        raise ValueError("initial_state must be finite.")
    return state


def _event_capacity(
    realization: PoissonClockRealization,
    value: int | None,
    /,
) -> int:
    available = realization.num_channels * realization.max_events_per_channel
    capacity = available if value is None else int(value)
    if not 0 < capacity <= available:
        raise ValueError(f"max_events must lie in [1, {available}].")
    return capacity


def _empty_event_arrays(
    capacity: int,
    state_shape: tuple[int, ...],
    mark_shape: tuple[int, ...],
    state_dtype: jnp.dtype,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    return (
        jnp.full((capacity,), jnp.nan, dtype=float),
        jnp.full((capacity,), -1, dtype=jnp.int32),
        jnp.zeros((capacity,) + mark_shape, dtype=state_dtype),
        jnp.zeros((capacity,), dtype=bool),
        jnp.zeros((capacity,) + state_shape, dtype=state_dtype),
        jnp.zeros((capacity,) + state_shape, dtype=state_dtype),
    )


def _next_reaction_one(
    process: AbstractJumpProcess,
    initial_state: Array,
    start: Array,
    end: Array,
    thresholds: Array,
    mark_keys: Array,
    args: Any,
    max_events: int,
) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array]:
    channels = process.num_channels
    per_channel = int(thresholds.shape[-1])
    times, event_channels, marks, valid, pre_states, post_states = _empty_event_arrays(
        max_events,
        process.state_shape,
        process.mark_shape,
        initial_state.dtype,
    )
    initial = (
        start,
        initial_state,
        jnp.zeros((channels,), dtype=float),
        jnp.zeros((channels,), dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(JUMP_SUCCESS, dtype=jnp.int32),
        jnp.asarray(True),
        times,
        event_channels,
        marks,
        valid,
        pre_states,
        post_states,
    )

    def condition(carry):
        return carry[6] & (carry[4] < max_events)

    def body(carry):
        (
            time,
            state,
            integrated,
            counts,
            event_index,
            status,
            active,
            event_times,
            selected_channels,
            event_marks,
            event_valid,
            before_states,
            after_states,
        ) = carry
        rates = process.intensities(time, state, args)
        valid_rates = jnp.all(jnp.isfinite(rates) & (rates >= 0.0))
        exhausted = jnp.any((counts >= per_channel) & (rates > 0.0))
        safe_counts = jnp.minimum(counts, per_channel - 1)
        next_thresholds = thresholds[jnp.arange(channels), safe_counts]
        remaining = jnp.maximum(next_thresholds - integrated, 0.0)
        waiting = jnp.where(rates > 0.0, remaining / rates, jnp.inf)
        channel = jnp.argmin(waiting).astype(jnp.int32)
        elapsed = waiting[channel]
        event_time = time + elapsed
        has_event = valid_rates & ~exhausted & jnp.isfinite(elapsed) & (event_time <= end)

        def apply_event(_):
            mark_index = counts[channel]
            mark_key = mark_keys[channel, mark_index]
            mark = process.sample_mark(mark_key, event_time, state, channel, args)
            next_state = process.jump(state, channel, mark, args)
            next_integrated = integrated + rates * elapsed
            next_counts = counts.at[channel].add(1)
            return (
                event_time,
                next_state,
                next_integrated,
                next_counts,
                event_index + 1,
                status,
                jnp.asarray(True),
                event_times.at[event_index].set(event_time),
                selected_channels.at[event_index].set(channel),
                event_marks.at[event_index].set(mark),
                event_valid.at[event_index].set(True),
                before_states.at[event_index].set(state),
                after_states.at[event_index].set(next_state),
            )

        def finish(_):
            resolved_status = jnp.where(
                ~valid_rates,
                JUMP_INVALID_INTENSITY,
                jnp.where(exhausted, JUMP_MAX_EVENTS, JUMP_SUCCESS),
            ).astype(jnp.int32)
            return (
                end,
                state,
                integrated,
                counts,
                event_index,
                resolved_status,
                jnp.asarray(False),
                event_times,
                selected_channels,
                event_marks,
                event_valid,
                before_states,
                after_states,
            )

        return jax.lax.cond(has_event, apply_event, finish, operand=None)

    result = jax.lax.while_loop(condition, body, initial)
    status = jnp.where(
        result[6] & (result[4] >= max_events),
        JUMP_MAX_EVENTS,
        result[5],
    ).astype(jnp.int32)
    return (
        result[1],
        status,
        result[7],
        result[8],
        result[9],
        result[10],
        result[11],
        result[12],
    )


def _direct_ssa_one(
    process: AbstractJumpProcess,
    initial_state: Array,
    start: Array,
    end: Array,
    proposal_keys: Array,
    mark_keys: Array,
    args: Any,
    max_events: int,
) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array]:
    per_channel = int(mark_keys.shape[1])
    times, event_channels, marks, valid, pre_states, post_states = _empty_event_arrays(
        max_events,
        process.state_shape,
        process.mark_shape,
        initial_state.dtype,
    )
    initial = (
        start,
        initial_state,
        jnp.zeros((process.num_channels,), dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(JUMP_SUCCESS, dtype=jnp.int32),
        jnp.asarray(True),
        times,
        event_channels,
        marks,
        valid,
        pre_states,
        post_states,
    )

    def condition(carry):
        return carry[5] & (carry[3] < max_events)

    def body(carry):
        (
            time,
            state,
            counts,
            event_index,
            status,
            active,
            event_times,
            selected_channels,
            event_marks,
            event_valid,
            before_states,
            after_states,
        ) = carry
        rates = process.intensities(time, state, args)
        valid_rates = jnp.all(jnp.isfinite(rates) & (rates >= 0.0))
        total = jnp.sum(rates)
        wait_key = jr.fold_in(proposal_keys[event_index], 0)
        channel_key = jr.fold_in(proposal_keys[event_index], 1)
        elapsed = jr.exponential(wait_key, dtype=rates.dtype) / total
        event_time = time + elapsed
        channel = jr.categorical(
            channel_key,
            jnp.where(rates > 0.0, jnp.log(rates), -jnp.inf),
        ).astype(jnp.int32)
        exhausted = counts[channel] >= per_channel
        has_event = (
            valid_rates
            & (total > 0.0)
            & ~exhausted
            & jnp.isfinite(event_time)
            & (event_time <= end)
        )

        def apply_event(_):
            mark_index = counts[channel]
            mark_key = mark_keys[channel, mark_index]
            mark = process.sample_mark(mark_key, event_time, state, channel, args)
            next_state = process.jump(state, channel, mark, args)
            return (
                event_time,
                next_state,
                counts.at[channel].add(1),
                event_index + 1,
                status,
                jnp.asarray(True),
                event_times.at[event_index].set(event_time),
                selected_channels.at[event_index].set(channel),
                event_marks.at[event_index].set(mark),
                event_valid.at[event_index].set(True),
                before_states.at[event_index].set(state),
                after_states.at[event_index].set(next_state),
            )

        def finish(_):
            resolved_status = jnp.where(
                ~valid_rates,
                JUMP_INVALID_INTENSITY,
                jnp.where(exhausted, JUMP_MAX_EVENTS, JUMP_SUCCESS),
            ).astype(jnp.int32)
            return (
                end,
                state,
                counts,
                event_index,
                resolved_status,
                jnp.asarray(False),
                event_times,
                selected_channels,
                event_marks,
                event_valid,
                before_states,
                after_states,
            )

        return jax.lax.cond(has_event, apply_event, finish, operand=None)

    result = jax.lax.while_loop(condition, body, initial)
    status = jnp.where(
        result[5] & (result[3] >= max_events),
        JUMP_MAX_EVENTS,
        result[4],
    ).astype(jnp.int32)
    return (
        result[1],
        status,
        result[6],
        result[7],
        result[8],
        result[9],
        result[10],
        result[11],
    )


@eqx.filter_jit
def _next_reaction_paths(
    process: AbstractJumpProcess,
    initial_state: Array,
    start: Array,
    end: Array,
    thresholds: Array,
    mark_keys: Array,
    args: Any,
    max_events: int,
):
    return jax.vmap(
        lambda path_thresholds, path_marks: _next_reaction_one(
            process,
            initial_state,
            start,
            end,
            path_thresholds,
            path_marks,
            args,
            max_events,
        )
    )(thresholds, mark_keys)


@eqx.filter_jit
def _direct_ssa_paths(
    process: AbstractJumpProcess,
    initial_state: Array,
    start: Array,
    end: Array,
    proposal_keys: Array,
    mark_keys: Array,
    args: Any,
    max_events: int,
):
    return jax.vmap(
        lambda path_proposals, path_marks: _direct_ssa_one(
            process,
            initial_state,
            start,
            end,
            path_proposals,
            path_marks,
            args,
            max_events,
        )
    )(proposal_keys, mark_keys)


class JumpSolution(StrictModule):
    """Saved pure-jump states and their complete masked event stream."""

    times: Array
    states: Array
    valid: Array
    events: JumpEventBatch
    realization: PoissonClockRealization
    metadata: frozendict[str, Any]
    state_shape: tuple[int, ...] = eqx.field(static=True)
    algorithm: JumpAlgorithm = eqx.field(static=True)

    def __init__(
        self,
        times: ArrayLike,
        states: ArrayLike,
        valid: ArrayLike,
        events: JumpEventBatch,
        realization: PoissonClockRealization,
        /,
        *,
        state_shape: Sequence[int],
        algorithm: JumpAlgorithm,
        metadata: Mapping[str, Any] | None = None,
    ):
        if not isinstance(events, JumpEventBatch):
            raise TypeError("events must be a JumpEventBatch.")
        if not isinstance(realization, PoissonClockRealization):
            raise TypeError("realization must be a PoissonClockRealization.")
        if algorithm not in ("next_reaction", "direct_ssa"):
            raise ValueError("Unknown jump algorithm.")
        shape = tuple(int(size) for size in state_shape)
        time_values = jnp.asarray(times, dtype=float)
        state_values = jnp.asarray(states)
        expected = realization.sample_shape + (time_values.shape[0],) + shape
        if time_values.ndim != 1 or state_values.shape != expected:
            raise ValueError("JumpSolution states do not align with times and axes.")
        valid_values = jnp.asarray(valid, dtype=bool)
        if valid_values.shape != realization.sample_shape + (time_values.shape[0],):
            raise ValueError("JumpSolution valid has incompatible shape.")
        if events.batch_shape != realization.sample_shape:
            raise ValueError("Event and realization batch shapes must match.")
        self.times = time_values
        self.states = state_values
        self.valid = valid_values
        self.events = events
        self.realization = realization
        self.metadata = frozendict({} if metadata is None else metadata)
        self.state_shape = shape
        self.algorithm = algorithm

    @property
    def successful(self) -> Array:
        return self.events.successful

    def to_stochastic_trajectory(
        self,
        /,
        *,
        realization_axes: Sequence[str] | None = None,
        state_axes: Sequence[str] | None = None,
    ) -> StochasticTrajectory:
        resolved_realization_axes = (
            tuple(
                f"process_{index}" for index in range(len(self.realization.sample_shape))
            )
            if realization_axes is None
            else tuple(realization_axes)
        )
        resolved_state_axes = (
            tuple(f"state_{index}" for index in range(len(self.state_shape)))
            if state_axes is None
            else tuple(state_axes)
        )
        return StochasticTrajectory(
            self.times,
            self.states,
            valid=self.valid,
            realization_axes=resolved_realization_axes,
            realization_shape=self.realization.sample_shape,
            state_axes=resolved_state_axes,
            realizations=(self.realization,),
            metadata={
                **dict(self.metadata),
                "process_id": self.realization.process_id,
                "jump_algorithm": self.algorithm,
                "uncertainty_source": "process",
            },
        )


def _solution_from_paths(
    process: AbstractJumpProcess,
    realization: PoissonClockRealization,
    initial_state: Array,
    times: Array,
    arrays: tuple[Array, ...],
    /,
    *,
    algorithm: JumpAlgorithm,
) -> JumpSolution:
    sample_shape = realization.sample_shape
    path_count = prod(sample_shape) if sample_shape else 1
    state_shape = process.state_shape
    capacity = int(arrays[2].shape[-1])
    terminal, status, event_times, channels, marks, valid, before, after = arrays
    del terminal
    event_times = event_times.reshape(sample_shape + (capacity,))
    channels = channels.reshape(sample_shape + (capacity,))
    marks = marks.reshape(sample_shape + (capacity,) + process.mark_shape)
    valid = valid.reshape(sample_shape + (capacity,))
    status = status.reshape(sample_shape)
    before = before.reshape(sample_shape + (capacity,) + state_shape)
    after = after.reshape(sample_shape + (capacity,) + state_shape)
    events = JumpEventBatch(
        event_times,
        channels,
        marks,
        valid,
        status,
        mark_shape=process.mark_shape,
        state_shape=state_shape,
        pre_states=before,
        post_states=after,
    )
    states = events.states_at(times, initial_state)
    successful = events.successful
    solution_valid = jnp.broadcast_to(successful[..., None], sample_shape + times.shape)
    if path_count == 1 and not sample_shape:
        states = states.reshape(times.shape + state_shape)
    return JumpSolution(
        times,
        states,
        solution_valid,
        events,
        realization,
        state_shape=state_shape,
        algorithm=algorithm,
        metadata={"max_events": capacity},
    )


def solve_next_reaction(
    process: AbstractJumpProcess,
    realization: PoissonClockRealization,
    initial_state: ArrayLike,
    /,
    *,
    t0: ArrayLike,
    t1: ArrayLike,
    save_times: ArrayLike,
    args: Any = None,
    max_events: int | None = None,
) -> JumpSolution:
    """Solve a pure-jump process using reusable per-channel Poisson clocks."""
    if not isinstance(process, AbstractJumpProcess):
        raise TypeError("process must implement AbstractJumpProcess.")
    if not isinstance(realization, PoissonClockRealization):
        raise TypeError("realization must be a PoissonClockRealization.")
    if process.process_id != realization.process_id:
        raise ValueError("Process and Poisson realization process_id values must match.")
    if process.num_channels != realization.num_channels:
        raise ValueError("Process and realization channel counts must match.")
    start, end = _time_interval(t0, t1, support=realization.support)
    times = _query_times(save_times, t0=start, t1=end)
    state = _initial_state(process, initial_state)
    capacity = _event_capacity(realization, max_events)
    path_count = realization.num_paths
    thresholds = realization.thresholds.reshape(
        (path_count, process.num_channels, realization.max_events_per_channel)
    )
    mark_keys = realization.mark_keys.reshape(
        (path_count, process.num_channels, realization.max_events_per_channel)
        + tuple(realization.root_key.shape)
    )
    arrays = _next_reaction_paths(
        process,
        state,
        start,
        end,
        thresholds,
        mark_keys,
        args,
        capacity,
    )
    return _solution_from_paths(
        process,
        realization,
        state,
        times,
        arrays,
        algorithm="next_reaction",
    )


def solve_direct_ssa(
    process: AbstractJumpProcess,
    realization: PoissonClockRealization,
    initial_state: ArrayLike,
    /,
    *,
    t0: ArrayLike,
    t1: ArrayLike,
    save_times: ArrayLike,
    args: Any = None,
    max_events: int | None = None,
) -> JumpSolution:
    """Solve a pure-jump process with Gillespie's total-rate direct method."""
    if not isinstance(process, AbstractJumpProcess):
        raise TypeError("process must implement AbstractJumpProcess.")
    if not isinstance(realization, PoissonClockRealization):
        raise TypeError("realization must be a PoissonClockRealization.")
    if process.process_id != realization.process_id:
        raise ValueError("Process and Poisson realization process_id values must match.")
    if process.num_channels != realization.num_channels:
        raise ValueError("Process and realization channel counts must match.")
    start, end = _time_interval(t0, t1, support=realization.support)
    times = _query_times(save_times, t0=start, t1=end)
    state = _initial_state(process, initial_state)
    capacity = _event_capacity(realization, max_events)
    path_count = realization.num_paths
    proposal_keys = realization.direct_event_keys.reshape(
        (path_count, process.num_channels * realization.max_events_per_channel)
        + tuple(realization.root_key.shape)
    )
    mark_keys = realization.mark_keys.reshape(
        (path_count, process.num_channels, realization.max_events_per_channel)
        + tuple(realization.root_key.shape)
    )
    arrays = _direct_ssa_paths(
        process,
        state,
        start,
        end,
        proposal_keys,
        mark_keys,
        args,
        capacity,
    )
    return _solution_from_paths(
        process,
        realization,
        state,
        times,
        arrays,
        algorithm="direct_ssa",
    )


class FiniteStateGenerator(StrictModule):
    """Finite-state generator with explicit escaped-rate diagnostics."""

    states: Array
    matrix: Array
    escaped_rates: Array
    process_id: str = eqx.field(static=True)
    boundary_policy: GeneratorBoundaryPolicy = eqx.field(static=True)

    def transition_matrix(self, duration: ArrayLike, /) -> Array:
        time = jnp.asarray(duration, dtype=self.matrix.dtype)
        if time.shape != ():
            raise ValueError("duration must be scalar.")
        return jsp.linalg.expm(time * self.matrix)

    def stationary_distribution(self) -> Array:
        values, vectors = jnp.linalg.eig(self.matrix.T)
        index = jnp.argmin(jnp.abs(values))
        vector = jnp.real(vectors[:, index])
        vector = jnp.where(jnp.sum(vector) < 0.0, -vector, vector)
        vector = jnp.maximum(vector, 0.0)
        return vector / jnp.sum(vector)


def finite_state_generator(
    process: AbstractJumpProcess,
    states: ArrayLike,
    /,
    *,
    t: ArrayLike = 0.0,
    args: Any = None,
    boundary_policy: GeneratorBoundaryPolicy = "error",
) -> FiniteStateGenerator:
    """Construct a finite generator for an unmarked discrete jump process."""
    if not isinstance(process, AbstractJumpProcess):
        raise TypeError("process must implement AbstractJumpProcess.")
    if process.mark_shape:
        raise ValueError("Finite generators currently require unmarked jump processes.")
    if boundary_policy not in ("error", "suppress", "leak"):
        raise ValueError("Unknown boundary_policy.")
    state_values = jnp.asarray(states)
    if state_values.ndim != len(process.state_shape) + 1:
        raise ValueError("states must have one leading enumeration axis.")
    if tuple(state_values.shape[1:]) != process.state_shape:
        raise ValueError("Enumerated states have incompatible state shape.")
    count = int(state_values.shape[0])
    if count <= 0:
        raise ValueError("states must be non-empty.")
    host = np.asarray(jax.device_get(state_values)).reshape((count, -1))
    if np.unique(host, axis=0).shape[0] != count:
        raise ValueError("Enumerated finite states must be unique.")

    def one_state(state):
        rates = process.intensities(t, state, args)
        channels = jnp.arange(process.num_channels, dtype=jnp.int32)
        next_states = jax.vmap(
            lambda channel: process.jump(
                state,
                channel,
                jnp.asarray(0, dtype=state.dtype),
                args,
            )
        )(channels)
        return rates, next_states

    rates, next_states = jax.vmap(one_state)(state_values)
    matches = jnp.all(
        next_states[:, :, None, ...] == state_values[None, None, ...],
        axis=tuple(range(3, 3 + len(process.state_shape))),
    )
    matched = jnp.any(matches, axis=-1)
    escaped = jnp.sum(jnp.where(matched, 0.0, rates), axis=-1)
    if boundary_policy == "error" and bool(jnp.any(escaped > 0.0)):
        raise ValueError(
            "Finite state set omits reachable states; choose an explicit boundary policy."
        )
    off_diagonal = jnp.einsum("ik,ikj->ij", rates, matches.astype(rates.dtype))
    included_rates = jnp.sum(jnp.where(matched, rates, 0.0), axis=-1)
    diagonal_rates = (
        jnp.sum(rates, axis=-1) if boundary_policy == "leak" else included_rates
    )
    matrix = off_diagonal.at[jnp.arange(count), jnp.arange(count)].add(-diagonal_rates)
    return FiniteStateGenerator(
        states=state_values,
        matrix=matrix,
        escaped_rates=escaped,
        process_id=process.process_id,
        boundary_policy=boundary_policy,
    )


class JumpDifferentialProblem(StrictModule):
    """Continuous differential dynamics composed with finite-activity jumps."""

    differential: DifferentialProblem
    jumps: AbstractJumpProcess
    process_id: str = eqx.field(static=True)

    def __init__(
        self,
        differential: DifferentialProblem,
        jumps: AbstractJumpProcess,
        /,
        *,
        process_id: str | None = None,
    ):
        if not isinstance(differential, DifferentialProblem):
            raise TypeError("differential must be a DifferentialProblem.")
        if not isinstance(jumps, AbstractJumpProcess):
            raise TypeError("jumps must implement AbstractJumpProcess.")
        if tuple(differential.initial_state.shape) != jumps.state_shape:
            raise ValueError("Differential and jump state shapes must match.")
        resolved_id = jumps.process_id if process_id is None else str(process_id)
        if not resolved_id:
            raise ValueError("process_id must be non-empty.")
        self.differential = differential
        self.jumps = jumps
        self.process_id = resolved_id


class JumpDifferentialSolution(StrictModule):
    """Saved hybrid states with continuous and jump-driver provenance."""

    times: Array
    states: Array
    valid: Array
    events: JumpEventBatch
    realization: StochasticRealization
    metadata: frozendict[str, Any]
    state_shape: tuple[int, ...] = eqx.field(static=True)
    solver_name: str = eqx.field(static=True)

    def __init__(
        self,
        times: ArrayLike,
        states: ArrayLike,
        valid: ArrayLike,
        events: JumpEventBatch,
        realization: StochasticRealization,
        /,
        *,
        state_shape: Sequence[int],
        solver_name: str,
        metadata: Mapping[str, Any] | None = None,
    ):
        shape = tuple(int(size) for size in state_shape)
        time_values = jnp.asarray(times, dtype=float)
        state_values = jnp.asarray(states)
        expected = realization.sample_shape + (time_values.shape[0],) + shape
        if time_values.ndim != 1 or state_values.shape != expected:
            raise ValueError("Hybrid solution states do not align with declared axes.")
        valid_values = jnp.asarray(valid, dtype=bool)
        if valid_values.shape != realization.sample_shape + time_values.shape:
            raise ValueError("Hybrid solution validity has incompatible shape.")
        if events.batch_shape != realization.sample_shape:
            raise ValueError("Hybrid event and realization batch shapes must match.")
        if not isinstance(solver_name, str) or not solver_name:
            raise ValueError("solver_name must be non-empty.")
        self.times = time_values
        self.states = state_values
        self.valid = valid_values
        self.events = events
        self.realization = realization
        self.metadata = frozendict({} if metadata is None else metadata)
        self.state_shape = shape
        self.solver_name = solver_name

    @property
    def successful(self) -> Array:
        return self.events.successful

    def to_stochastic_trajectory(
        self,
        /,
        *,
        realization_axes: Sequence[str] | None = None,
        state_axes: Sequence[str] | None = None,
    ) -> StochasticTrajectory:
        resolved_realization_axes = (
            tuple(
                f"process_{index}" for index in range(len(self.realization.sample_shape))
            )
            if realization_axes is None
            else tuple(realization_axes)
        )
        resolved_state_axes = (
            tuple(f"state_{index}" for index in range(len(self.state_shape)))
            if state_axes is None
            else tuple(state_axes)
        )
        return StochasticTrajectory(
            self.times,
            self.states,
            valid=self.valid,
            realization_axes=resolved_realization_axes,
            realization_shape=self.realization.sample_shape,
            state_axes=resolved_state_axes,
            realizations=(self.realization,),
            metadata={
                **dict(self.metadata),
                "solver_name": self.solver_name,
                "uncertainty_source": "process",
            },
        )


def _levy_area(kind: str, /) -> type:
    if kind == "brownian":
        return dfx.BrownianIncrement
    if kind == "space_time":
        return dfx.SpaceTimeLevyArea
    if kind == "space_time_time":
        return dfx.SpaceTimeTimeLevyArea
    raise AssertionError(f"Unhandled Levy-area kind {kind!r}.")


def _hybrid_diffusion(
    differential: DifferentialProblem,
    path_sign: Array,
    num_channels: int,
):
    state_shape = tuple(differential.initial_state.shape)
    state_size = prod(state_shape) if state_shape else 1

    def evaluate(time, augmented_state, args):
        state = augmented_state[:state_size].reshape(state_shape)
        columns = []
        for term in differential.wiener_terms:
            value = jnp.asarray(term.coefficient(time, state, args))
            columns.append(value.reshape((state_size, term.noise_size)))
        state_diffusion = path_sign * jnp.concatenate(columns, axis=-1)
        hazard_diffusion = jnp.zeros(
            (num_channels, differential.noise_shape[0]),
            dtype=state_diffusion.dtype,
        )
        return jnp.concatenate((state_diffusion, hazard_diffusion), axis=0)

    return evaluate


class _ClippedBrownianPath(dfx.AbstractBrownianPath):
    """Clamp solver roundoff at a global Brownian support boundary."""

    path: dfx.AbstractBrownianPath
    _t0: Array
    _t1: Array
    levy_area: type = eqx.field(static=True)

    def __init__(
        self,
        path: dfx.AbstractBrownianPath,
        support: tuple[float, float],
        /,
    ):
        self.path = path
        self._t0 = jnp.asarray(support[0])
        self._t1 = jnp.asarray(support[1])
        self.levy_area = path.levy_area

    @property
    def t0(self) -> Array:  # ty: ignore[invalid-attribute-override]
        return self._t0

    @property
    def t1(self) -> Array:  # ty: ignore[invalid-attribute-override]
        return self._t1

    def evaluate(self, t0, t1=None, left=True, use_levy=False):
        start = jnp.clip(t0, self.t0, self.t1)
        end = None if t1 is None else jnp.clip(t1, self.t0, self.t1)
        return self.path.evaluate(start, end, left=left, use_levy=use_levy)


def _hybrid_terms(
    problem: JumpDifferentialProblem,
    poisson: PoissonClockRealization,
    wiener: WienerRealization | None,
    path_key: Array | None,
    path_sign: Array | None,
):
    differential = problem.differential
    jumps = problem.jumps
    state_shape = tuple(differential.initial_state.shape)
    state_size = prod(state_shape) if state_shape else 1

    def augmented_drift(time, augmented_state, args):
        state = augmented_state[:state_size].reshape(state_shape)
        rates = jumps.intensities(time, state, args)
        rates_valid = jnp.all(jnp.isfinite(rates) & (rates >= 0.0))
        rates = jnp.where(rates_valid, rates, jnp.nan)
        drift = jnp.asarray(differential.drift(time, state, args)).reshape((state_size,))
        return jnp.concatenate((drift, rates))

    drift_term = dfx.ODETerm(augmented_drift)
    if not differential.stochastic:
        return drift_term
    if wiener is None or path_key is None or path_sign is None:
        raise ValueError("Stochastic hybrid dynamics require a Wiener realization.")
    real_dtype = differential.initial_state.real.dtype
    brownian = dfx.VirtualBrownianTree(
        t0=wiener.support[0],
        t1=wiener.support[1],
        tol=wiener.tolerance,
        shape=jax.ShapeDtypeStruct(wiener.noise_shape, real_dtype),
        key=path_key,
        levy_area=_levy_area(wiener.levy_area),
    )
    control = _ClippedBrownianPath(brownian, wiener.support)
    return dfx.MultiTerm(
        drift_term,
        dfx.ControlTerm(
            _hybrid_diffusion(
                differential,
                jnp.asarray(path_sign, dtype=real_dtype),
                jumps.num_channels,
            ),
            control,
        ),
    )


def _hybrid_one(
    problem: JumpDifferentialProblem,
    poisson: PoissonClockRealization,
    wiener: WienerRealization | None,
    save_times: Array,
    thresholds: Array,
    mark_keys: Array,
    path_key: Array | None,
    path_sign: Array | None,
    solver: Any,
    stepsize_controller: Any,
    dt0: Array | None,
    root_finder: Any,
    max_steps: int,
    max_events: int,
):
    differential = problem.differential
    jumps = problem.jumps
    per_channel = int(thresholds.shape[-1])
    (
        event_times,
        event_channels,
        event_marks,
        event_valid,
        pre_states,
        post_states,
    ) = _empty_event_arrays(
        max_events,
        jumps.state_shape,
        jumps.mark_shape,
        differential.initial_state.dtype,
    )
    terms = _hybrid_terms(problem, poisson, wiener, path_key, path_sign)
    initial_carry = (
        save_times[0],
        differential.initial_state,
        jnp.zeros((jumps.num_channels,), dtype=float),
        jnp.zeros((jumps.num_channels,), dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(JUMP_SUCCESS, dtype=jnp.int32),
        event_times,
        event_channels,
        event_marks,
        event_valid,
        pre_states,
        post_states,
    )

    def advance_interval(carry, interval_end):
        def condition(inner):
            return (
                (inner[0] < interval_end)
                & (inner[5] == JUMP_SUCCESS)
                & (inner[4] < max_events)
            )

        def body(inner):
            (
                time,
                state,
                hazards,
                counts,
                event_index,
                status,
                times_buffer,
                channels_buffer,
                marks_buffer,
                valid_buffer,
                before_buffer,
                after_buffer,
            ) = inner
            rates = jumps.intensities(time, state, differential.args)
            valid_rates = jnp.all(jnp.isfinite(rates) & (rates >= 0.0))
            exhausted = jnp.any((counts >= per_channel) & (rates > 0.0))
            safe_counts = jnp.minimum(counts, per_channel - 1)
            next_thresholds = thresholds[jnp.arange(jumps.num_channels), safe_counts]

            def integrate(_):
                state_size = prod(jumps.state_shape) if jumps.state_shape else 1

                def event_condition(t, y, args, **kwargs):
                    del t, args, kwargs
                    return jnp.min(next_thresholds - y[state_size:])

                event = dfx.Event(
                    event_condition,
                    root_finder=root_finder,
                    direction=False,
                )
                initial_augmented = jnp.concatenate(
                    (state.reshape((state_size,)), hazards)
                )
                native = dfx.diffeqsolve(
                    terms,
                    solver,
                    t0=time,
                    t1=interval_end,
                    dt0=dt0,
                    y0=initial_augmented,
                    args=differential.args,
                    saveat=dfx.SaveAt(t1=True),
                    stepsize_controller=stepsize_controller,
                    adjoint=dfx.DirectAdjoint(),
                    event=event,
                    max_steps=max_steps,
                    throw=False,
                )
                next_time = native.ts[0]
                native_augmented = native.ys[0]
                event_occurred = jnp.asarray(native.event_mask, dtype=bool)
                native_ok = (native.result == dfx.RESULTS.successful) | (
                    native.result == dfx.RESULTS.event_occurred
                )
                if differential.stochastic:

                    def replay_event(_):
                        replay = dfx.diffeqsolve(
                            terms,
                            solver,
                            t0=time,
                            t1=next_time,
                            dt0=dt0,
                            y0=initial_augmented,
                            args=differential.args,
                            saveat=dfx.SaveAt(t1=True),
                            stepsize_controller=stepsize_controller,
                            adjoint=dfx.DirectAdjoint(),
                            max_steps=max_steps,
                            throw=False,
                        )
                        return replay.ys[0], replay.result == dfx.RESULTS.successful

                    next_augmented, replay_ok = jax.lax.cond(
                        event_occurred & native_ok & (next_time > time),
                        replay_event,
                        lambda _: (native_augmented, jnp.asarray(True)),
                        operand=None,
                    )
                else:
                    next_augmented = native_augmented
                    replay_ok = jnp.asarray(True)
                next_state = next_augmented[:state_size].reshape(jumps.state_shape)
                next_hazards = next_augmented[state_size:]
                solver_ok = native_ok & replay_ok
                channel = jnp.argmin(jnp.abs(next_thresholds - next_hazards)).astype(
                    jnp.int32
                )

                def apply_event(_):
                    mark_index = counts[channel]
                    mark = jumps.sample_mark(
                        mark_keys[channel, mark_index],
                        next_time,
                        next_state,
                        channel,
                        differential.args,
                    )
                    post_state = jumps.jump(
                        next_state,
                        channel,
                        mark,
                        differential.args,
                    )
                    return (
                        next_time,
                        post_state,
                        next_hazards,
                        counts.at[channel].add(1),
                        event_index + 1,
                        status,
                        times_buffer.at[event_index].set(next_time),
                        channels_buffer.at[event_index].set(channel),
                        marks_buffer.at[event_index].set(mark),
                        valid_buffer.at[event_index].set(True),
                        before_buffer.at[event_index].set(next_state),
                        after_buffer.at[event_index].set(post_state),
                    )

                def no_event(_):
                    next_status = jnp.where(
                        solver_ok,
                        status,
                        JUMP_SOLVER_FAILURE,
                    ).astype(jnp.int32)
                    return (
                        interval_end,
                        next_state,
                        next_hazards,
                        counts,
                        event_index,
                        next_status,
                        times_buffer,
                        channels_buffer,
                        marks_buffer,
                        valid_buffer,
                        before_buffer,
                        after_buffer,
                    )

                return jax.lax.cond(
                    event_occurred & solver_ok,
                    apply_event,
                    no_event,
                    operand=None,
                )

            def reject(_):
                rejected_status = jnp.where(
                    valid_rates,
                    JUMP_MAX_EVENTS,
                    JUMP_INVALID_INTENSITY,
                ).astype(jnp.int32)
                return (
                    interval_end,
                    state,
                    hazards,
                    counts,
                    event_index,
                    rejected_status,
                    times_buffer,
                    channels_buffer,
                    marks_buffer,
                    valid_buffer,
                    before_buffer,
                    after_buffer,
                )

            return jax.lax.cond(
                valid_rates & ~exhausted,
                integrate,
                reject,
                operand=None,
            )

        advanced = jax.lax.while_loop(condition, body, carry)
        exhausted_global = (
            (advanced[4] >= max_events)
            & (advanced[0] < interval_end)
            & (advanced[5] == JUMP_SUCCESS)
        )
        advanced = (
            advanced[:5]
            + (
                jnp.where(
                    exhausted_global,
                    JUMP_MAX_EVENTS,
                    advanced[5],
                ).astype(jnp.int32),
            )
            + advanced[6:]
        )
        return advanced, advanced[1]

    final_carry, saved_tail = jax.lax.scan(
        advance_interval,
        initial_carry,
        save_times[1:],
    )
    saved_states = jnp.concatenate(
        (differential.initial_state[None, ...], saved_tail),
        axis=0,
    )
    return (
        saved_states,
        final_carry[5],
        final_carry[6],
        final_carry[7],
        final_carry[8],
        final_carry[9],
        final_carry[10],
        final_carry[11],
    )


@eqx.filter_jit
def _hybrid_deterministic_paths(
    problem: JumpDifferentialProblem,
    poisson: PoissonClockRealization,
    save_times: Array,
    thresholds: Array,
    mark_keys: Array,
    solver: Any,
    stepsize_controller: Any,
    dt0: Array | None,
    root_finder: Any,
    max_steps: int,
    max_events: int,
):
    return jax.vmap(
        lambda path_thresholds, path_marks: _hybrid_one(
            problem,
            poisson,
            None,
            save_times,
            path_thresholds,
            path_marks,
            None,
            None,
            solver,
            stepsize_controller,
            dt0,
            root_finder,
            max_steps,
            max_events,
        )
    )(thresholds, mark_keys)


@eqx.filter_jit
def _hybrid_stochastic_paths(
    problem: JumpDifferentialProblem,
    poisson: PoissonClockRealization,
    wiener: WienerRealization,
    save_times: Array,
    thresholds: Array,
    mark_keys: Array,
    path_keys: Array,
    path_signs: Array,
    solver: Any,
    stepsize_controller: Any,
    dt0: Array | None,
    root_finder: Any,
    max_steps: int,
    max_events: int,
):
    return jax.vmap(
        lambda path_thresholds, path_marks, path_key, path_sign: _hybrid_one(
            problem,
            poisson,
            wiener,
            save_times,
            path_thresholds,
            path_marks,
            path_key,
            path_sign,
            solver,
            stepsize_controller,
            dt0,
            root_finder,
            max_steps,
            max_events,
        )
    )(thresholds, mark_keys, path_keys, path_signs)


def solve_jump_differential(
    problem: JumpDifferentialProblem,
    poisson_realization: PoissonClockRealization,
    /,
    *,
    save_times: ArrayLike,
    wiener_realization: WienerRealization | None = None,
    solver: Any | None = None,
    stepsize_controller: Any | None = None,
    dt0: ArrayLike | None = None,
    rtol: float = 1e-6,
    atol: float = 1e-8,
    event_rtol: float = 1e-7,
    event_atol: float = 1e-9,
    max_steps: int = 4096,
    max_events: int | None = None,
) -> JumpDifferentialSolution:
    """Integrate ODE/SDE dynamics with state-dependent random-time-change jumps."""
    if not isinstance(problem, JumpDifferentialProblem):
        raise TypeError("problem must be a JumpDifferentialProblem.")
    if not isinstance(poisson_realization, PoissonClockRealization):
        raise TypeError("poisson_realization must be a PoissonClockRealization.")
    differential = problem.differential
    jumps = problem.jumps
    if (
        differential.state_geometry is not None
        and not differential.state_geometry.trivial
    ):
        raise ValueError(
            "Hybrid jump integration does not support nontrivial state_geometry."
        )
    if poisson_realization.process_id != jumps.process_id:
        raise ValueError("Jump process and Poisson realization process_id values differ.")
    if poisson_realization.num_channels != jumps.num_channels:
        raise ValueError("Jump process and realization channel counts differ.")
    times = _query_times(
        save_times,
        t0=differential.t0,
        t1=differential.t1,
    )
    if not bool(
        jnp.isclose(times[0], differential.t0) & jnp.isclose(times[-1], differential.t1)
    ):
        raise ValueError("Hybrid save_times must include both problem endpoints.")
    capacity = _event_capacity(poisson_realization, max_events)
    path_count = poisson_realization.num_paths
    thresholds = poisson_realization.thresholds.reshape(
        (
            path_count,
            jumps.num_channels,
            poisson_realization.max_events_per_channel,
        )
    )
    mark_keys = poisson_realization.mark_keys.reshape(
        (
            path_count,
            jumps.num_channels,
            poisson_realization.max_events_per_channel,
        )
        + tuple(poisson_realization.root_key.shape)
    )
    root_finder = optx.Newton(
        rtol=float(event_rtol),
        atol=float(event_atol),
        norm=optx.rms_norm,
    )
    if differential.stochastic:
        if not isinstance(wiener_realization, WienerRealization):
            raise ValueError("Stochastic hybrid problems require a Wiener realization.")
        if wiener_realization.sample_shape != poisson_realization.sample_shape:
            raise ValueError("Wiener and Poisson sample shapes must match.")
        if wiener_realization.support != poisson_realization.support:
            raise ValueError("Wiener and Poisson supports must match.")
        if wiener_realization.noise_shape != differential.noise_shape:
            raise ValueError("Wiener and differential noise shapes must match.")
        selected_controller = (
            dfx.ConstantStepSize() if stepsize_controller is None else stepsize_controller
        )
        if dt0 is None and not isinstance(selected_controller, dfx.StepTo):
            raise ValueError(
                "Stochastic hybrid integration requires dt0 unless every step "
                "is declared with diffrax.StepTo."
            )
        resolved_dt0 = None if dt0 is None else jnp.asarray(dt0, dtype=float)
        if resolved_dt0 is not None and bool(
            jnp.abs(resolved_dt0) <= wiener_realization.tolerance
        ):
            raise ValueError("Wiener tolerance must be smaller than dt0.")
        selected_solver = (
            dfx.Euler()
            if solver is None and differential.interpretation == "ito"
            else dfx.EulerHeun()
            if solver is None
            else solver
        )
        path_keys = wiener_realization.path_keys.reshape(
            (path_count,) + tuple(wiener_realization.root_key.shape)
        )
        path_signs = wiener_realization.path_signs.reshape((path_count,))
        arrays = _hybrid_stochastic_paths(
            problem,
            poisson_realization,
            wiener_realization,
            times,
            thresholds,
            mark_keys,
            path_keys,
            path_signs,
            selected_solver,
            selected_controller,
            resolved_dt0,
            root_finder,
            int(max_steps),
            capacity,
        )
        realization: StochasticRealization = CompositeStochasticRealization(
            {"wiener": wiener_realization, "jump": poisson_realization}
        )
    else:
        if wiener_realization is not None:
            raise ValueError("Deterministic hybrid problems do not accept Wiener noise.")
        selected_solver = dfx.Tsit5() if solver is None else solver
        selected_controller = (
            dfx.PIDController(rtol=float(rtol), atol=float(atol))
            if stepsize_controller is None
            else stepsize_controller
        )
        resolved_dt0 = None if dt0 is None else jnp.asarray(dt0, dtype=float)
        arrays = _hybrid_deterministic_paths(
            problem,
            poisson_realization,
            times,
            thresholds,
            mark_keys,
            selected_solver,
            selected_controller,
            resolved_dt0,
            root_finder,
            int(max_steps),
            capacity,
        )
        realization = poisson_realization

    sample_shape = poisson_realization.sample_shape
    state_shape = jumps.state_shape
    states, status, event_times, channels, marks, valid, before, after = arrays
    states = states.reshape(sample_shape + times.shape + state_shape)
    status = status.reshape(sample_shape)
    event_times = event_times.reshape(sample_shape + (capacity,))
    channels = channels.reshape(sample_shape + (capacity,))
    marks = marks.reshape(sample_shape + (capacity,) + jumps.mark_shape)
    valid = valid.reshape(sample_shape + (capacity,))
    before = before.reshape(sample_shape + (capacity,) + state_shape)
    after = after.reshape(sample_shape + (capacity,) + state_shape)
    events = JumpEventBatch(
        event_times,
        channels,
        marks,
        valid,
        status,
        mark_shape=jumps.mark_shape,
        state_shape=state_shape,
        pre_states=before,
        post_states=after,
    )
    solution_valid = jnp.broadcast_to(
        events.successful[..., None],
        sample_shape + times.shape,
    )
    return JumpDifferentialSolution(
        times,
        states,
        solution_valid,
        events,
        realization,
        state_shape=state_shape,
        solver_name=type(selected_solver).__name__,
        metadata={
            "process_id": problem.process_id,
            "jump_process_id": jumps.process_id,
            "event_rtol": float(event_rtol),
            "event_atol": float(event_atol),
            "max_events": capacity,
        },
    )


__all__ = [
    "finite_state_generator",
    "FiniteStateGenerator",
    "GeneratorBoundaryPolicy",
    "JumpAlgorithm",
    "JumpDifferentialProblem",
    "JumpDifferentialSolution",
    "JumpSolution",
    "solve_direct_ssa",
    "solve_jump_differential",
    "solve_next_reaction",
]
