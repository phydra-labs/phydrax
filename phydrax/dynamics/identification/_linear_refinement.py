#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass, replace

import equinox as eqx
import numpy as np

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import LinearSolveControl


@dataclass(frozen=True, slots=True)
class ProgressiveLinearRefinementRecord:
    validation_step: int
    metric: float
    smoothed_metric: float
    previous_steps: int
    current_steps: int
    plateau: bool
    refined: bool
    stopped: bool


@dataclass(frozen=True, slots=True)
class ProgressiveLinearRefinementState:
    current_steps: int
    validation_count: int = 0
    smoothed_metric: float | None = None
    history: tuple[float, ...] = ()
    plateau_checkpoint: float | None = None
    refinement_count: int = 0
    last_refinement_step: int | None = None
    stopped: bool = False


class ProgressiveLinearRefinementPolicy(StrictModule, NonTrainableState):
    """Validation-plateau controller for a training-only Krylov budget."""

    initial_steps: int = eqx.field(static=True)
    step_increment: int = eqx.field(static=True)
    maximum_steps: int = eqx.field(static=True)
    evaluation_steps: int = eqx.field(static=True)
    smoothing: float = eqx.field(static=True)
    grace_validations: int = eqx.field(static=True)
    plateau_relative_improvement: float = eqx.field(static=True)
    stop_relative_improvement: float = eqx.field(static=True)
    minimum_validations: int = eqx.field(static=True)
    coarse_relative_residual: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        initial_steps: int,
        step_increment: int,
        maximum_steps: int,
        evaluation_steps: int | None = None,
        smoothing: float = 0.9,
        grace_validations: int = 5,
        plateau_relative_improvement: float = 1e-3,
        stop_relative_improvement: float = 1e-4,
        minimum_validations: int = 6,
        coarse_relative_residual: float = 1.0,
    ):
        initial = int(initial_steps)
        increment = int(step_increment)
        maximum = int(maximum_steps)
        evaluation = maximum if evaluation_steps is None else int(evaluation_steps)
        grace = int(grace_validations)
        minimum = int(minimum_validations)
        scalars = tuple(
            float(value)
            for value in (
                smoothing,
                plateau_relative_improvement,
                stop_relative_improvement,
                coarse_relative_residual,
            )
        )
        if initial < 1 or increment < 1 or maximum < initial or evaluation < maximum:
            raise ValueError("Linear refinement iteration capacities are invalid.")
        if grace < 1 or minimum <= grace:
            raise ValueError("minimum_validations must exceed the positive grace window.")
        if any(not np.isfinite(value) for value in scalars):
            raise ValueError("Linear refinement scalars must be finite.")
        if not 0.0 <= scalars[0] < 1.0:
            raise ValueError("smoothing must lie in [0, 1).")
        if scalars[1] < 0.0 or scalars[2] < 0.0 or scalars[3] <= 0.0:
            raise ValueError("Refinement thresholds or residual guard are invalid.")
        self.initial_steps = initial
        self.step_increment = increment
        self.maximum_steps = maximum
        self.evaluation_steps = evaluation
        self.smoothing = scalars[0]
        self.grace_validations = grace
        self.plateau_relative_improvement = scalars[1]
        self.stop_relative_improvement = scalars[2]
        self.minimum_validations = minimum
        self.coarse_relative_residual = scalars[3]
        self.policy_id = canonical_fingerprint(
            {
                "kind": "progressive-linear-refinement",
                "initial_steps": initial,
                "step_increment": increment,
                "maximum_steps": maximum,
                "evaluation_steps": evaluation,
                "smoothing": scalars[0],
                "grace_validations": grace,
                "plateau_relative_improvement": scalars[1],
                "stop_relative_improvement": scalars[2],
                "minimum_validations": minimum,
                "coarse_relative_residual": scalars[3],
            }
        )

    def initialize(self) -> ProgressiveLinearRefinementState:
        return ProgressiveLinearRefinementState(self.initial_steps)

    def training_control(
        self,
        state: ProgressiveLinearRefinementState,
        /,
    ) -> LinearSolveControl:
        if not isinstance(state, ProgressiveLinearRefinementState):
            raise TypeError("state must be ProgressiveLinearRefinementState.")
        return LinearSolveControl(maximum_steps=state.current_steps)

    def evaluation_control(self) -> LinearSolveControl:
        return LinearSolveControl(maximum_steps=self.evaluation_steps)

    def observe(
        self,
        state: ProgressiveLinearRefinementState,
        metric: float,
        /,
        *,
        validation_step: int,
    ) -> tuple[ProgressiveLinearRefinementState, ProgressiveLinearRefinementRecord]:
        if not isinstance(state, ProgressiveLinearRefinementState):
            raise TypeError("state must be ProgressiveLinearRefinementState.")
        value = float(metric)
        step = int(validation_step)
        if not np.isfinite(value) or value < 0.0 or step < 0:
            raise ValueError("Validation metric and step must be finite and nonnegative.")
        smoothed = (
            value
            if state.smoothed_metric is None
            else self.smoothing * state.smoothed_metric + (1.0 - self.smoothing) * value
        )
        history = (*state.history, smoothed)
        count = state.validation_count + 1
        plateau = False
        refined = False
        stopped = state.stopped
        current = state.current_steps
        checkpoint = state.plateau_checkpoint
        refinement_count = state.refinement_count
        last_refinement = state.last_refinement_step
        if count >= self.minimum_validations and len(history) > self.grace_validations:
            prior = history[-self.grace_validations - 1]
            scale = max(abs(prior), np.finfo(float).tiny)
            relative_improvement = (prior - smoothed) / scale
            plateau = relative_improvement <= self.plateau_relative_improvement
        if plateau and not stopped and current < self.maximum_steps:
            if checkpoint is not None:
                scale = max(abs(checkpoint), np.finfo(float).tiny)
                level_improvement = (checkpoint - smoothed) / scale
                if level_improvement <= self.stop_relative_improvement:
                    stopped = True
            if not stopped:
                previous = current
                current = min(self.maximum_steps, current + self.step_increment)
                refined = current != previous
                refinement_count += int(refined)
                last_refinement = step if refined else last_refinement
                checkpoint = smoothed
                history = (smoothed,)
        next_state = replace(
            state,
            current_steps=current,
            validation_count=count,
            smoothed_metric=smoothed,
            history=history,
            plateau_checkpoint=checkpoint,
            refinement_count=refinement_count,
            last_refinement_step=last_refinement,
            stopped=stopped,
        )
        return next_state, ProgressiveLinearRefinementRecord(
            validation_step=step,
            metric=value,
            smoothed_metric=smoothed,
            previous_steps=state.current_steps,
            current_steps=current,
            plateau=plateau,
            refined=refined,
            stopped=stopped,
        )


__all__ = [
    "ProgressiveLinearRefinementPolicy",
    "ProgressiveLinearRefinementRecord",
    "ProgressiveLinearRefinementState",
]
