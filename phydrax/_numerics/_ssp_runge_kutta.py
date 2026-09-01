#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import AbstractAttribute, StrictModule
from .._temporal_precision import TemporalPrecisionPolicy
from .._trainable import NonTrainableState


class AbstractSSPRKStageTransform(StrictModule, NonTrainableState):
    transform_id: AbstractAttribute[str]

    @abc.abstractmethod
    def apply(
        self,
        stage_index: int,
        time: Array,
        candidate_state: Array,
        args: Any,
        /,
    ) -> StageTransformResult:
        raise NotImplementedError

    def __call__(
        self,
        stage_index: int,
        time: Array,
        candidate_state: Array,
        args: Any,
        /,
    ) -> StageTransformResult:
        return self.apply(stage_index, time, candidate_state, args)


class StageTransformResult(StrictModule):
    state: Array
    applied: Array
    successful: Array
    correction_norm: Array


class SSPRKStepResult(StrictModule):
    state: Array
    applied: Array
    successful: Array
    correction_norm: Array


StageTransform = Callable[[int, Array, Array, Any], StageTransformResult]


def _stage(
    index: int,
    time: Array,
    value: Array,
    args: Any,
    transform: StageTransform | None,
    /,
) -> StageTransformResult:
    candidate = jnp.asarray(value)
    if transform is None:
        return StageTransformResult(
            candidate,
            jnp.asarray(False),
            jnp.asarray(True),
            jnp.zeros((), dtype=candidate.real.dtype),
        )
    result = transform(index, jnp.asarray(time), candidate, args)
    if not isinstance(result, StageTransformResult):
        raise TypeError("SSP stage transforms must return StageTransformResult.")
    state = jnp.asarray(result.state)
    if state.shape != candidate.shape or state.dtype != candidate.dtype:
        raise ValueError("SSP stage transforms must preserve state shape and dtype.")
    if (
        result.applied.shape != ()
        or result.applied.dtype != jnp.dtype(bool)
        or result.successful.shape != ()
        or result.successful.dtype != jnp.dtype(bool)
        or result.correction_norm.shape != ()
    ):
        raise TypeError("SSP stage transform evidence must contain scalar arrays.")
    return StageTransformResult(
        state,
        result.applied,
        result.successful,
        jnp.asarray(result.correction_norm, dtype=candidate.real.dtype),
    )


def _step_result(
    state: Array,
    stages: tuple[StageTransformResult, ...],
    /,
) -> SSPRKStepResult:
    applied = jnp.asarray(False)
    successful = jnp.asarray(True)
    correction = jnp.zeros((), dtype=state.real.dtype)
    for stage in stages:
        applied = applied | stage.applied
        successful = successful & stage.successful
        correction = jnp.maximum(correction, stage.correction_norm)
    return SSPRKStepResult(state, applied, successful, correction)


def ssprk33_step_with_evidence(
    vector_field: Callable[[Array, Array, Any], ArrayLike],
    time: Array,
    state: Array,
    step_size: Array,
    args: Any = None,
    /,
    *,
    stage_transform: StageTransform | None = None,
    precision: TemporalPrecisionPolicy | None = None,
) -> SSPRKStepResult:
    """Advance SSPRK(3,3) and return aggregate stage-transform evidence."""
    if precision is not None and not isinstance(precision, TemporalPrecisionPolicy):
        raise TypeError("precision must be a TemporalPrecisionPolicy or None.")
    t = jnp.asarray(time)
    h = jnp.asarray(step_size)
    y0 = jnp.asarray(state)
    if precision is None:
        h_ = h
        y0_ = y0

        def stage_value(value):
            return jnp.asarray(value)

        def accumulation(value):
            return jnp.asarray(value)

    else:
        precision.validate_state(y0)
        h_ = precision.coefficient(jnp.asarray(h, dtype=y0.real.dtype))
        y0_ = precision.stage(y0)
        stage_value = precision.stage
        accumulation = precision.accumulation

    first_candidate = accumulation(y0_) + accumulation(
        h_ * stage_value(vector_field(t, y0_, args))
    )
    first = _stage(1, t + h_, stage_value(first_candidate), args, stage_transform)
    y1 = stage_value(first.state)
    second_increment = accumulation(y1 + h_ * stage_value(vector_field(t + h_, y1, args)))
    second_candidate = accumulation(0.75 * y0_) + accumulation(0.25 * second_increment)
    second = _stage(
        2,
        t + 0.5 * h_,
        stage_value(second_candidate),
        args,
        stage_transform,
    )
    y2 = stage_value(second.state)
    third_increment = accumulation(
        y2 + h_ * stage_value(vector_field(t + 0.5 * h_, y2, args))
    )
    third_candidate = accumulation((1.0 / 3.0) * y0_) + accumulation(
        (2.0 / 3.0) * third_increment
    )
    third = _stage(
        3,
        t + h_,
        stage_value(third_candidate),
        args,
        stage_transform,
    )
    result = jnp.asarray(third.state, dtype=y0.dtype)
    return _step_result(result, (first, second, third))


def ssprk33_step(
    vector_field: Callable[[Array, Array, Any], ArrayLike],
    time: Array,
    state: Array,
    step_size: Array,
    args: Any = None,
    /,
    *,
    stage_transform: StageTransform | None = None,
    precision: TemporalPrecisionPolicy | None = None,
) -> Array:
    """Advance one Shu--Osher SSPRK(3,3) step."""
    return ssprk33_step_with_evidence(
        vector_field,
        time,
        state,
        step_size,
        args,
        stage_transform=stage_transform,
        precision=precision,
    ).state


def ssprk54_step_with_evidence(
    vector_field: Callable[[Array, Array, Any], ArrayLike],
    time: Array,
    state: Array,
    step_size: Array,
    args: Any = None,
    /,
    *,
    stage_transform: StageTransform | None = None,
    precision: TemporalPrecisionPolicy | None = None,
) -> SSPRKStepResult:
    """Advance SSPRK(5,4) and return aggregate stage-transform evidence."""
    if precision is not None and not isinstance(precision, TemporalPrecisionPolicy):
        raise TypeError("precision must be a TemporalPrecisionPolicy or None.")
    t = jnp.asarray(time)
    h = jnp.asarray(step_size)
    y0 = jnp.asarray(state)
    if precision is None:
        h_ = h
        y0_ = y0

        def stage_value(value):
            return jnp.asarray(value)

        def accumulation(value):
            return jnp.asarray(value)

    else:
        precision.validate_state(y0)
        h_ = precision.coefficient(jnp.asarray(h, dtype=y0.real.dtype))
        y0_ = precision.stage(y0)
        stage_value = precision.stage
        accumulation = precision.accumulation

    first_candidate = accumulation(y0_) + accumulation(
        0.391752226571890 * h_ * stage_value(vector_field(t, y0_, args))
    )
    first = _stage(
        1,
        t + 0.391752226571890 * h_,
        stage_value(first_candidate),
        args,
        stage_transform,
    )
    y1 = stage_value(first.state)
    second_candidate = (
        accumulation(0.444370493651235 * y0_)
        + accumulation(0.555629506348765 * y1)
        + accumulation(
            0.368410593050371
            * h_
            * stage_value(vector_field(t + 0.391752226571890 * h_, y1, args))
        )
    )
    second = _stage(
        2,
        t + 0.586079689311540 * h_,
        stage_value(second_candidate),
        args,
        stage_transform,
    )
    y2 = stage_value(second.state)
    third_candidate = (
        accumulation(0.620101851488403 * y0_)
        + accumulation(0.379898148511597 * y2)
        + accumulation(
            0.251891774271694
            * h_
            * stage_value(vector_field(t + 0.586079689311540 * h_, y2, args))
        )
    )
    third = _stage(
        3,
        t + 0.474542363026870 * h_,
        stage_value(third_candidate),
        args,
        stage_transform,
    )
    y3 = stage_value(third.state)
    fourth_candidate = (
        accumulation(0.178079954393132 * y0_)
        + accumulation(0.821920045606868 * y3)
        + accumulation(
            0.544974750228521
            * h_
            * stage_value(vector_field(t + 0.474542363026870 * h_, y3, args))
        )
    )
    fourth = _stage(
        4,
        t + 0.935010631009240 * h_,
        stage_value(fourth_candidate),
        args,
        stage_transform,
    )
    y4 = stage_value(fourth.state)
    fifth_candidate = (
        accumulation(0.517231671970585 * y2)
        + accumulation(0.096059710526147 * y3)
        + accumulation(
            0.063692468666290
            * h_
            * stage_value(vector_field(t + 0.474542363026870 * h_, y3, args))
        )
        + accumulation(0.386708617503269 * y4)
        + accumulation(
            0.226007483236906
            * h_
            * stage_value(vector_field(t + 0.935010631009240 * h_, y4, args))
        )
    )
    fifth = _stage(
        5,
        t + h_,
        stage_value(fifth_candidate),
        args,
        stage_transform,
    )
    result = jnp.asarray(fifth.state, dtype=y0.dtype)
    return _step_result(result, (first, second, third, fourth, fifth))


def ssprk54_step(
    vector_field: Callable[[Array, Array, Any], ArrayLike],
    time: Array,
    state: Array,
    step_size: Array,
    args: Any = None,
    /,
    *,
    stage_transform: StageTransform | None = None,
    precision: TemporalPrecisionPolicy | None = None,
) -> Array:
    """Advance one five-stage, fourth-order optimal SSP Runge--Kutta step."""
    return ssprk54_step_with_evidence(
        vector_field,
        time,
        state,
        step_size,
        args,
        stage_transform=stage_transform,
        precision=precision,
    ).state


__all__ = [
    "AbstractSSPRKStageTransform",
    "SSPRKStepResult",
    "StageTransform",
    "StageTransformResult",
    "ssprk33_step",
    "ssprk33_step_with_evidence",
    "ssprk54_step",
    "ssprk54_step_with_evidence",
]
