#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key

from ...._doc import DOC_KEY0
from ..data import OperatorBatch


@dataclass(frozen=True)
class TeacherForcingSchedule:
    """Bounded linear or cosine teacher-forcing probability schedule."""

    start: float = 1.0
    end: float = 0.0
    transition_steps: int = 1
    kind: Literal["linear", "cosine"] = "linear"

    def __post_init__(self):
        if not 0.0 <= float(self.start) <= 1.0 or not 0.0 <= float(self.end) <= 1.0:
            raise ValueError("Teacher-forcing probabilities must lie in [0, 1].")
        if int(self.transition_steps) <= 0:
            raise ValueError("transition_steps must be positive.")
        if self.kind not in ("linear", "cosine"):
            raise ValueError("kind must be 'linear' or 'cosine'.")

    def __call__(self, step: Any, /) -> Array:
        fraction = jnp.clip(jnp.asarray(step) / float(self.transition_steps), 0.0, 1.0)
        if self.kind == "cosine":
            fraction = 0.5 - 0.5 * jnp.cos(jnp.pi * fraction)
        return self.start + fraction * (self.end - self.start)


@dataclass(frozen=True)
class RolloutHorizonSchedule:
    """Integer curriculum from short to long rollout horizons."""

    start: int
    end: int
    transition_steps: int

    def __post_init__(self):
        if int(self.start) <= 0 or int(self.end) < int(self.start):
            raise ValueError("Rollout horizons must satisfy 0 < start <= end.")
        if int(self.transition_steps) <= 0:
            raise ValueError("transition_steps must be positive.")

    def __call__(self, step: int, /) -> int:
        fraction = min(max(int(step), 0), self.transition_steps) / self.transition_steps
        return min(
            self.end,
            self.start + int(jnp.floor(fraction * (self.end - self.start))),
        )


@dataclass(frozen=True)
class OperatorRollout:
    """Predictions and case-level teacher-forcing choices for one rollout."""

    predictions: Array
    teacher_forcing_mask: Array


def _probability(
    value: float | TeacherForcingSchedule,
    training_step: int,
    /,
) -> Array:
    probability = (
        value(training_step)
        if isinstance(value, TeacherForcingSchedule)
        else jnp.asarray(value)
    )
    if bool(jnp.any((probability < 0.0) | (probability > 1.0))):
        raise ValueError("teacher_forcing must lie in [0, 1].")
    return probability


def autoregressive_operator_rollout(
    model: Callable,
    initial_batch: OperatorBatch,
    steps: int,
    advance: Callable[[OperatorBatch, Array, int], OperatorBatch],
    /,
    *,
    teacher_targets: Array | None = None,
    teacher_forcing: float | TeacherForcingSchedule = 0.0,
    training_step: int = 0,
    detach_feedback: bool = False,
    key: Key[Array, ""] = DOC_KEY0,
) -> OperatorRollout:
    """Roll an operator forward with scheduled, case-level teacher forcing."""
    count = int(steps)
    if count <= 0:
        raise ValueError("steps must be positive.")
    if teacher_targets is not None and int(teacher_targets.shape[0]) < count:
        raise ValueError("teacher_targets is shorter than the requested rollout.")
    probability = _probability(teacher_forcing, int(training_step))
    keys = jr.split(key, count * 2)
    batch = initial_batch
    predictions = []
    masks = []
    for index in range(count):
        prediction = jnp.asarray(model(batch, key=keys[2 * index]))
        predictions.append(prediction)
        if teacher_targets is None:
            use_teacher = jnp.zeros(batch.case_shape, dtype=bool)
            feedback = prediction
        else:
            target = jnp.asarray(teacher_targets[index])
            if target.shape != prediction.shape:
                raise ValueError("Teacher target and prediction shapes must match.")
            use_teacher = jr.bernoulli(
                keys[2 * index + 1],
                probability,
                shape=batch.case_shape,
            )
            broadcast = use_teacher.reshape(
                batch.case_shape + (1,) * (prediction.ndim - len(batch.case_shape))
            )
            feedback = jnp.where(broadcast, target, prediction)
        masks.append(use_teacher)
        if detach_feedback:
            feedback = jax.lax.stop_gradient(feedback)
        if index + 1 < count:
            batch = advance(batch, feedback, index)
            if not isinstance(batch, OperatorBatch):
                raise TypeError("advance must return an OperatorBatch.")
    return OperatorRollout(
        predictions=jnp.stack(predictions, axis=0),
        teacher_forcing_mask=jnp.stack(masks, axis=0),
    )


def autoregressive_operator_loss(
    model: Callable,
    initial_batch: OperatorBatch,
    targets: Array,
    advance: Callable[[OperatorBatch, Array, int], OperatorBatch],
    /,
    *,
    training_step: int,
    horizon: int | RolloutHorizonSchedule,
    teacher_forcing: float | TeacherForcingSchedule = 0.0,
    loss_fn: Callable[[Array, Array], Array] | None = None,
    detach_feedback: bool = False,
    key: Key[Array, ""] = DOC_KEY0,
) -> Array:
    """Evaluate a scheduled multi-step rollout objective."""
    count = (
        horizon(training_step)
        if isinstance(horizon, RolloutHorizonSchedule)
        else int(horizon)
    )
    target_values = jnp.asarray(targets)
    if int(target_values.shape[0]) < count:
        raise ValueError("targets is shorter than the scheduled rollout horizon.")
    rollout = autoregressive_operator_rollout(
        model,
        initial_batch,
        count,
        advance,
        teacher_targets=target_values,
        teacher_forcing=teacher_forcing,
        training_step=training_step,
        detach_feedback=detach_feedback,
        key=key,
    )
    if loss_fn is None:
        return jnp.mean(jnp.abs(rollout.predictions - target_values[:count]) ** 2)
    losses = jax.vmap(loss_fn)(rollout.predictions, target_values[:count])
    return jnp.mean(losses)


__all__ = [
    "OperatorRollout",
    "RolloutHorizonSchedule",
    "TeacherForcingSchedule",
    "autoregressive_operator_loss",
    "autoregressive_operator_rollout",
]
