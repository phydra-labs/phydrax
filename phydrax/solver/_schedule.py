#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from itertools import pairwise

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


class TimeLaw(StrictModule, NonTrainableState):
    value_function: Callable
    first_derivative_function: Callable
    second_derivative_function: Callable
    law_id: str = eqx.field(static=True)

    def __init__(
        self,
        value_function: Callable,
        first_derivative_function: Callable,
        second_derivative_function: Callable,
        /,
        *,
        law_id: str,
    ):
        if not all(
            callable(function)
            for function in (
                value_function,
                first_derivative_function,
                second_derivative_function,
            )
        ):
            raise TypeError("Time-law value and derivative functions must be callable.")
        identifier = str(law_id)
        if not identifier:
            raise ValueError("law_id must be non-empty.")
        self.value_function = value_function
        self.first_derivative_function = first_derivative_function
        self.second_derivative_function = second_derivative_function
        self.law_id = identifier

    def value(self, time: ArrayLike, args: object = None, /) -> Array:
        return jnp.asarray(self.value_function(jnp.asarray(time), args))

    def d1(self, time: ArrayLike, args: object = None, /) -> Array:
        return jnp.asarray(self.first_derivative_function(jnp.asarray(time), args))

    def d2(self, time: ArrayLike, args: object = None, /) -> Array:
        return jnp.asarray(self.second_derivative_function(jnp.asarray(time), args))

    @classmethod
    def constant(cls, value: ArrayLike, /, *, law_id: str = "constant") -> TimeLaw:
        value_ = jnp.asarray(value)
        return cls(
            lambda time, args: value_,
            lambda time, args: jnp.zeros_like(value_),
            lambda time, args: jnp.zeros_like(value_),
            law_id=law_id,
        )

    @classmethod
    def ramp(
        cls,
        start_time: float,
        end_time: float,
        start_value: ArrayLike,
        end_value: ArrayLike,
        /,
        *,
        law_id: str = "ramp",
    ) -> TimeLaw:
        t0 = float(start_time)
        t1 = float(end_time)
        start = jnp.asarray(start_value)
        end = jnp.asarray(end_value)
        if t1 <= t0 or start.shape != end.shape:
            raise ValueError("Ramp interval or value shapes are invalid.")
        slope = (end - start) / (t1 - t0)

        def value(time, args):
            fraction = jnp.clip((time - t0) / (t1 - t0), 0.0, 1.0)
            return start + fraction * (end - start)

        def first(time, args):
            active = (time > t0) & (time < t1)
            return jnp.where(active, slope, jnp.zeros_like(slope))

        return cls(
            value,
            first,
            lambda time, args: jnp.zeros_like(slope),
            law_id=law_id,
        )

    def scale(self, factor: ArrayLike, /, *, law_id: str | None = None) -> TimeLaw:
        factor_ = jnp.asarray(factor)
        identifier = (
            canonical_fingerprint({"kind": "scaled-time-law", "source": self.law_id})
            if law_id is None
            else str(law_id)
        )
        return TimeLaw(
            lambda time, args: factor_ * self.value(time, args),
            lambda time, args: factor_ * self.d1(time, args),
            lambda time, args: factor_ * self.d2(time, args),
            law_id=identifier,
        )


class ScheduleStepResult(StrictModule):
    state: object
    accepted: Array
    diagnostics: object


class SolveStage(StrictModule, NonTrainableState):
    stage_id: str = eqx.field(static=True)
    start_time: float = eqx.field(static=True)
    end_time: float = eqx.field(static=True)
    time_law: TimeLaw
    solve: Callable
    commit: Callable
    rollback: Callable

    def __init__(
        self,
        stage_id: str,
        start_time: float,
        end_time: float,
        time_law: TimeLaw,
        solve: Callable,
        /,
        *,
        commit: Callable = lambda state, diagnostics: state,
        rollback: Callable = lambda committed, candidate, diagnostics: committed,
    ):
        identifier = str(stage_id)
        start = float(start_time)
        end = float(end_time)
        if not identifier or end <= start or not isinstance(time_law, TimeLaw):
            raise ValueError("Solve stage identity, interval, or time law is invalid.")
        if not callable(solve) or not callable(commit) or not callable(rollback):
            raise TypeError("Solve stage callbacks must be callable.")
        self.stage_id = identifier
        self.start_time = start
        self.end_time = end
        self.time_law = time_law
        self.solve = solve
        self.commit = commit
        self.rollback = rollback

    def execute(self, state: object, args: object = None, /) -> ScheduleStepResult:
        result = self.solve(
            state,
            self.start_time,
            self.end_time,
            self.time_law,
            args,
        )
        if not isinstance(result, ScheduleStepResult):
            raise TypeError("Stage solve callback must return ScheduleStepResult.")
        accepted_state = (
            self.commit(result.state, result.diagnostics)
            if bool(jnp.asarray(result.accepted))
            else self.rollback(state, result.state, result.diagnostics)
        )
        return ScheduleStepResult(
            state=accepted_state,
            accepted=result.accepted,
            diagnostics=result.diagnostics,
        )


class SolveSchedule(StrictModule, NonTrainableState):
    stages: tuple[SolveStage, ...]
    schedule_id: str = eqx.field(static=True)

    def __init__(self, stages: Sequence[SolveStage], /):
        stages_ = tuple(stages)
        if not stages_ or not all(isinstance(stage, SolveStage) for stage in stages_):
            raise ValueError("SolveSchedule requires one or more SolveStage values.")
        identifiers = tuple(stage.stage_id for stage in stages_)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Solve stage IDs must be unique.")
        for first, second in pairwise(stages_):
            if first.end_time > second.start_time:
                raise ValueError("Solve schedule stages overlap in time.")
        self.stages = stages_
        self.schedule_id = canonical_fingerprint(
            {
                "kind": "solve-schedule",
                "stages": [
                    {
                        "stage_id": stage.stage_id,
                        "start": stage.start_time,
                        "end": stage.end_time,
                        "time_law": stage.time_law.law_id,
                    }
                    for stage in stages_
                ],
            }
        )

    def run(self, initial_state: object, args: object = None, /):
        state = initial_state
        results = []
        for stage in self.stages:
            result = stage.execute(state, args)
            results.append(result)
            state = result.state
            if not bool(jnp.asarray(result.accepted)):
                break
        return state, tuple(results)


__all__ = [
    "ScheduleStepResult",
    "SolveSchedule",
    "SolveStage",
    "TimeLaw",
]
