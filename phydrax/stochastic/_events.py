#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from math import isfinite, prod
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ._trajectory import StochasticTrajectory


CrossingDirection: TypeAlias = Literal["up", "down", "either"]
EventLocalization: TypeAlias = Literal["linear", "discrete"]
PathObservable: TypeAlias = Callable[[Array, Array], ArrayLike]
PathPredicate: TypeAlias = Callable[[Array, Array], ArrayLike]


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string.")
    return value


def _direction(value: str, /) -> CrossingDirection:
    if value not in ("up", "down", "either"):
        raise ValueError("direction must be 'up', 'down', or 'either'.")
    return value


def _localization(value: str, /) -> EventLocalization:
    if value not in ("linear", "discrete"):
        raise ValueError("localization must be 'linear' or 'discrete'.")
    return value


def _call_path(function: Callable, times: Array, states: Array, /) -> Array:
    values = jax.vmap(function)(times, states)
    result = jnp.asarray(values)
    if result.shape != times.shape:
        raise ValueError("Path event callables must return one scalar per time node.")
    return result


def _prefix_valid(valid: Array, /) -> Array:
    return jnp.cumprod(valid.astype(jnp.int32), axis=-1).astype(bool)


def _crossing_hits(values: Array, direction: CrossingDirection, /) -> Array:
    previous = values[..., :-1]
    current = values[..., 1:]
    if direction == "up":
        transitions = (previous < 0.0) & (current >= 0.0)
        initial = values[..., 0] >= 0.0
    elif direction == "down":
        transitions = (previous > 0.0) & (current <= 0.0)
        initial = values[..., 0] <= 0.0
    else:
        transitions = (
            ((previous < 0.0) & (current >= 0.0))
            | ((previous > 0.0) & (current <= 0.0))
            | (current == 0.0)
        )
        initial = values[..., 0] == 0.0
    return jnp.concatenate((initial[..., None], transitions), axis=-1)


def _localized_event_time(
    times: Array,
    values: Array,
    indices: Array,
    localization: EventLocalization,
    /,
) -> Array:
    node_time = jnp.take_along_axis(times, indices[..., None], axis=-1)[..., 0]
    if localization == "discrete":
        return node_time
    previous_index = jnp.maximum(indices - 1, 0)
    previous_time = jnp.take_along_axis(times, previous_index[..., None], axis=-1)[..., 0]
    previous_value = jnp.take_along_axis(values, previous_index[..., None], axis=-1)[
        ..., 0
    ]
    current_value = jnp.take_along_axis(values, indices[..., None], axis=-1)[..., 0]
    denominator = current_value - previous_value
    fraction = jnp.where(
        (indices > 0) & (denominator != 0.0),
        jnp.clip(-previous_value / denominator, 0.0, 1.0),
        0.0,
    )
    return previous_time + fraction * (node_time - previous_time)


class TerminalSetEvent(StrictModule):
    """Event that occurs when a complete path terminates inside a declared set."""

    predicate: PathPredicate
    score: PathObservable | None
    event_id: str = eqx.field(static=True)

    def __init__(
        self,
        predicate: PathPredicate,
        /,
        *,
        event_id: str,
        score: PathObservable | None = None,
    ):
        if not callable(predicate):
            raise TypeError("predicate must be callable.")
        if score is not None and not callable(score):
            raise TypeError("score must be callable or None.")
        self.predicate = predicate
        self.score = score
        self.event_id = _identifier(event_id, "event_id")


class ThresholdCrossingEvent(StrictModule):
    """First threshold crossing of a scalar path observable."""

    observable: PathObservable
    threshold: float = eqx.field(static=True)
    direction: CrossingDirection = eqx.field(static=True)
    localization: EventLocalization = eqx.field(static=True)
    event_id: str = eqx.field(static=True)

    def __init__(
        self,
        observable: PathObservable,
        threshold: float,
        /,
        *,
        direction: CrossingDirection = "up",
        localization: EventLocalization = "linear",
        event_id: str,
    ):
        if not callable(observable):
            raise TypeError("observable must be callable.")
        value = float(threshold)
        if not isfinite(value):
            raise ValueError("threshold must be finite.")
        self.observable = observable
        self.threshold = value
        self.direction = _direction(direction)
        self.localization = _localization(localization)
        self.event_id = _identifier(event_id, "event_id")


class AccumulatedPathEvent(StrictModule):
    """Threshold event for a trapezoidal accumulated path functional."""

    rate: PathObservable
    threshold: float = eqx.field(static=True)
    direction: CrossingDirection = eqx.field(static=True)
    localization: EventLocalization = eqx.field(static=True)
    event_id: str = eqx.field(static=True)

    def __init__(
        self,
        rate: PathObservable,
        threshold: float,
        /,
        *,
        direction: CrossingDirection = "up",
        localization: EventLocalization = "linear",
        event_id: str,
    ):
        if not callable(rate):
            raise TypeError("rate must be callable.")
        value = float(threshold)
        if not isfinite(value):
            raise ValueError("threshold must be finite.")
        self.rate = rate
        self.threshold = value
        self.direction = _direction(direction)
        self.localization = _localization(localization)
        self.event_id = _identifier(event_id, "event_id")


AtomicPathEvent: TypeAlias = (
    TerminalSetEvent | ThresholdCrossingEvent | AccumulatedPathEvent
)


class CompetingPathEvents(StrictModule):
    """Earliest event among ordered competing path-event definitions."""

    events: tuple[AtomicPathEvent, ...]
    event_id: str = eqx.field(static=True)

    def __init__(
        self,
        events: Sequence[AtomicPathEvent],
        /,
        *,
        event_id: str = "competing-events",
    ):
        resolved = tuple(events)
        if not resolved or any(
            not isinstance(
                event,
                (TerminalSetEvent, ThresholdCrossingEvent, AccumulatedPathEvent),
            )
            for event in resolved
        ):
            raise ValueError("CompetingPathEvents requires non-empty atomic events.")
        identifiers = tuple(event.event_id for event in resolved)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Competing path event IDs must be unique.")
        self.events = resolved
        self.event_id = _identifier(event_id, "event_id")


PathEvent: TypeAlias = AtomicPathEvent | CompetingPathEvents


class PathEventResult(StrictModule):
    """Path-aligned event times with explicit occurrence, censoring, and failure."""

    occurred: Array
    censored: Array
    failed: Array
    event_times: Array
    event_indices: Array
    event_codes: Array
    terminal_scores: Array
    event_ids: tuple[str, ...] = eqx.field(static=True)
    trajectory_ids: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        occurred: ArrayLike,
        censored: ArrayLike,
        failed: ArrayLike,
        event_times: ArrayLike,
        event_indices: ArrayLike,
        event_codes: ArrayLike,
        terminal_scores: ArrayLike,
        /,
        *,
        event_ids: Sequence[str],
        trajectory_ids: Sequence[str],
    ):
        occurrence = jnp.asarray(occurred, dtype=bool)
        shape = occurrence.shape
        censoring = jnp.asarray(censored, dtype=bool)
        failures = jnp.asarray(failed, dtype=bool)
        times = jnp.asarray(event_times, dtype=float)
        indices = jnp.asarray(event_indices, dtype=jnp.int32)
        codes = jnp.asarray(event_codes, dtype=jnp.int32)
        scores = jnp.asarray(terminal_scores, dtype=float)
        if any(
            value.shape != shape
            for value in (censoring, failures, times, indices, codes, scores)
        ):
            raise ValueError("Every PathEventResult array must share one path shape.")
        identifiers = tuple(_identifier(value, "event_id") for value in event_ids)
        count = prod(shape) if shape else 1
        path_ids = tuple(str(value) for value in trajectory_ids)
        if len(path_ids) != count:
            raise ValueError("trajectory_ids must contain one identifier per path.")
        self.occurred = occurrence
        self.censored = censoring
        self.failed = failures
        self.event_times = times
        self.event_indices = indices
        self.event_codes = codes
        self.terminal_scores = scores
        self.event_ids = identifiers
        self.trajectory_ids = path_ids

    @property
    def valid(self) -> Array:
        return ~self.failed

    @property
    def event_probability(self) -> Array:
        valid_count = jnp.sum(self.valid)
        return jnp.where(
            valid_count > 0,
            jnp.sum(self.occurred) / valid_count,
            jnp.nan,
        )


def path_event_scores(
    trajectory: StochasticTrajectory,
    event: PathEvent,
    /,
) -> Array:
    """Return a progress score at every saved node, with event at score zero or above."""
    if not isinstance(trajectory, StochasticTrajectory):
        raise TypeError("trajectory must be a StochasticTrajectory.")
    leading = trajectory.leading_shape
    count = prod(leading) if leading else 1
    times = trajectory.times.reshape((count, trajectory.num_times))
    states = trajectory.states.reshape(
        (count, trajectory.num_times) + trajectory.state_shape
    )

    def atomic_scores(atomic: AtomicPathEvent) -> Array:
        if isinstance(atomic, TerminalSetEvent):
            function = atomic.predicate if atomic.score is None else atomic.score
            values = jax.vmap(lambda t, x: _call_path(function, t, x))(times, states)
            return values.astype(float)
        if isinstance(atomic, ThresholdCrossingEvent):
            values = (
                jax.vmap(lambda t, x: _call_path(atomic.observable, t, x))(times, states)
                - atomic.threshold
            )
            if atomic.direction == "down":
                return -values
            if atomic.direction == "either":
                return -jnp.abs(values)
            return values
        rates = jax.vmap(lambda t, x: _call_path(atomic.rate, t, x))(times, states)
        intervals = times[:, 1:] - times[:, :-1]
        increments = 0.5 * intervals * (rates[:, :-1] + rates[:, 1:])
        cumulative = jnp.concatenate(
            (jnp.zeros((count, 1), dtype=rates.dtype), jnp.cumsum(increments, axis=-1)),
            axis=-1,
        )
        values = cumulative - atomic.threshold
        if atomic.direction == "down":
            return -values
        if atomic.direction == "either":
            return -jnp.abs(values)
        return values

    if isinstance(event, CompetingPathEvents):
        score_values = jnp.max(
            jnp.stack(tuple(atomic_scores(value) for value in event.events), axis=0),
            axis=0,
        )
    elif isinstance(
        event,
        (TerminalSetEvent, ThresholdCrossingEvent, AccumulatedPathEvent),
    ):
        score_values = atomic_scores(event)
    else:
        raise TypeError("event must be a supported PathEvent.")
    valid = _prefix_valid(trajectory.valid.reshape((count, trajectory.num_times)))
    score_values = jnp.where(valid, score_values, -jnp.inf)
    return score_values.reshape(leading + (trajectory.num_times,))


def _evaluate_atomic(
    trajectory: StochasticTrajectory,
    event: AtomicPathEvent,
    /,
) -> PathEventResult:
    leading = trajectory.leading_shape
    count = prod(leading) if leading else 1
    times = trajectory.times.reshape((count, trajectory.num_times))
    states = trajectory.states.reshape(
        (count, trajectory.num_times) + trajectory.state_shape
    )
    valid = trajectory.valid.reshape((count, trajectory.num_times))
    prefix = _prefix_valid(valid)
    complete = jnp.all(valid, axis=-1)
    scores = path_event_scores(trajectory, event).reshape((count, trajectory.num_times))
    if isinstance(event, TerminalSetEvent):
        terminal_hit = jnp.asarray(
            jax.vmap(event.predicate)(times[:, -1], states[:, -1]), dtype=bool
        )
        occurred = complete & terminal_hit
        indices = jnp.where(occurred, trajectory.num_times - 1, -1).astype(jnp.int32)
        event_times = jnp.where(occurred, times[:, -1], jnp.nan)
    else:
        if isinstance(event, ThresholdCrossingEvent):
            raw_values = (
                jax.vmap(lambda t, x: _call_path(event.observable, t, x))(times, states)
                - event.threshold
            )
            direction = event.direction
            localization = event.localization
        else:
            rates = jax.vmap(lambda t, x: _call_path(event.rate, t, x))(times, states)
            intervals = times[:, 1:] - times[:, :-1]
            increments = 0.5 * intervals * (rates[:, :-1] + rates[:, 1:])
            cumulative = jnp.concatenate(
                (
                    jnp.zeros((count, 1), dtype=rates.dtype),
                    jnp.cumsum(increments, axis=-1),
                ),
                axis=-1,
            )
            raw_values = cumulative - event.threshold
            direction = event.direction
            localization = event.localization
        hits = _crossing_hits(raw_values, direction) & prefix
        occurred = jnp.any(hits, axis=-1)
        candidate = jnp.argmax(hits, axis=-1).astype(jnp.int32)
        indices = jnp.where(occurred, candidate, -1)
        safe_indices = jnp.maximum(indices, 0)
        localized = _localized_event_time(
            times,
            raw_values,
            safe_indices,
            localization,
        )
        event_times = jnp.where(occurred, localized, jnp.nan)
    failed = ~occurred & ~complete
    censored = ~occurred & complete
    terminal_index = jnp.maximum(jnp.sum(prefix, axis=-1) - 1, 0).astype(jnp.int32)
    terminal_scores = jnp.take_along_axis(scores, terminal_index[:, None], axis=-1)[:, 0]
    return PathEventResult(
        occurred.reshape(leading),
        censored.reshape(leading),
        failed.reshape(leading),
        event_times.reshape(leading),
        indices.reshape(leading),
        jnp.where(occurred, 0, -1).astype(jnp.int32).reshape(leading),
        terminal_scores.reshape(leading),
        event_ids=(event.event_id,),
        trajectory_ids=trajectory.trajectory_ids,
    )


def evaluate_path_event(
    trajectory: StochasticTrajectory,
    event: PathEvent,
    /,
) -> PathEventResult:
    """Evaluate one atomic or competing event against saved stochastic trajectories."""
    if not isinstance(trajectory, StochasticTrajectory):
        raise TypeError("trajectory must be a StochasticTrajectory.")
    if isinstance(
        event,
        (TerminalSetEvent, ThresholdCrossingEvent, AccumulatedPathEvent),
    ):
        return _evaluate_atomic(trajectory, event)
    if not isinstance(event, CompetingPathEvents):
        raise TypeError("event must be a supported PathEvent.")
    results = tuple(_evaluate_atomic(trajectory, atomic) for atomic in event.events)
    event_times = jnp.stack(
        tuple(
            jnp.where(result.occurred, result.event_times, jnp.inf) for result in results
        ),
        axis=0,
    )
    code = jnp.argmin(event_times, axis=0).astype(jnp.int32)
    earliest = jnp.min(event_times, axis=0)
    occurred = jnp.isfinite(earliest)
    failed = ~occurred & results[0].failed
    censored = ~occurred & ~failed
    indices = jnp.take_along_axis(
        jnp.stack(tuple(result.event_indices for result in results), axis=0),
        code[None, ...],
        axis=0,
    )[0]
    scores = jnp.max(
        jnp.stack(tuple(result.terminal_scores for result in results), axis=0), axis=0
    )
    return PathEventResult(
        occurred,
        censored,
        failed,
        jnp.where(occurred, earliest, jnp.nan),
        jnp.where(occurred, indices, -1),
        jnp.where(occurred, code, -1),
        scores,
        event_ids=tuple(atomic.event_id for atomic in event.events),
        trajectory_ids=trajectory.trajectory_ids,
    )


__all__ = [
    "AccumulatedPathEvent",
    "AtomicPathEvent",
    "CompetingPathEvents",
    "CrossingDirection",
    "EventLocalization",
    "PathEvent",
    "PathEventResult",
    "TerminalSetEvent",
    "ThresholdCrossingEvent",
    "evaluate_path_event",
    "path_event_scores",
]
