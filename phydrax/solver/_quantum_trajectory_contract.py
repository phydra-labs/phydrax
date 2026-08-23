#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..nonlinear import NonlinearWorkBudget


class QuantumTrajectoryStatus(IntEnum):
    SUCCESS = 0
    BACKEND_FAILED = 1
    ROOT_BRACKET_INVALID = 2
    ROOT_NOT_CONVERGED = 3
    EVENT_CAPACITY_EXHAUSTED = 4
    INVALID_SURVIVAL = 5
    INVALID_RATES = 6
    ZERO_JUMP_STATE = 7
    TRUNCATION_BUDGET_EXCEEDED = 8
    PROJECTION_FAILED = 9
    CHECKPOINT_INCOMPATIBLE = 10


class QuantumTrajectoryPlan(StrictModule):
    maximum_events: int = eqx.field(static=True)
    maximum_segments: int = eqx.field(static=True)
    root_iterations: int = eqx.field(static=True)
    root_tolerance: float = eqx.field(static=True)
    root_method: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    work_budget: NonlinearWorkBudget

    def __init__(
        self,
        *,
        maximum_events: int = 128,
        maximum_segments: int = 256,
        root_iterations: int = 64,
        root_tolerance: float = 1e-9,
        root_method: str = "toms748",
        plan_id: str = "quantum-trajectory-v2",
        work_budget: NonlinearWorkBudget | None = None,
    ):
        if min(maximum_events, maximum_segments, root_iterations) < 1:
            raise ValueError("Trajectory capacities must be positive.")
        if root_tolerance <= 0.0 or root_method not in (
            "toms748",
            "brent",
            "bisection",
        ):
            raise ValueError("Trajectory root policy is invalid.")
        self.maximum_events = int(maximum_events)
        self.maximum_segments = int(maximum_segments)
        self.root_iterations = int(root_iterations)
        self.root_tolerance = float(root_tolerance)
        self.root_method = root_method
        self.plan_id = str(plan_id)
        self.work_budget = (
            NonlinearWorkBudget.unlimited() if work_budget is None else work_budget
        )


class QuantumTrajectoryEventTable(StrictModule):
    times: Array
    channels: Array
    thresholds: Array
    root_residuals: Array
    bracket_widths: Array
    active: Array

    def __init__(
        self,
        times: ArrayLike,
        channels: ArrayLike,
        thresholds: ArrayLike,
        root_residuals: ArrayLike,
        bracket_widths: ArrayLike,
        active: ArrayLike,
        /,
    ):
        self.times = jnp.asarray(times)
        self.channels = jnp.asarray(channels, dtype=jnp.int32)
        self.thresholds = jnp.asarray(thresholds)
        self.root_residuals = jnp.asarray(root_residuals)
        self.bracket_widths = jnp.asarray(bracket_widths)
        self.active = jnp.asarray(active, dtype=bool)


class QuantumTrajectoryCheckpoint(StrictModule):
    state: Array
    time: Array
    threshold: Array
    event_ordinal: int = eqx.field(static=True)
    output_cursor: int = eqx.field(static=True)
    events: QuantumTrajectoryEventTable
    problem_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    status: Array

    def __init__(
        self,
        state: ArrayLike,
        time: ArrayLike,
        threshold: ArrayLike,
        events: QuantumTrajectoryEventTable,
        /,
        *,
        event_ordinal: int,
        output_cursor: int,
        problem_id: str,
        plan_id: str,
        status: int = int(QuantumTrajectoryStatus.SUCCESS),
    ):
        self.state = jnp.asarray(state)
        self.time = jnp.asarray(time)
        self.threshold = jnp.asarray(threshold)
        self.events = events
        self.event_ordinal = int(event_ordinal)
        self.output_cursor = int(output_cursor)
        self.problem_id = str(problem_id)
        self.plan_id = str(plan_id)
        self.status = jnp.asarray(status, dtype=jnp.int32)


__all__ = [
    "QuantumTrajectoryCheckpoint",
    "QuantumTrajectoryEventTable",
    "QuantumTrajectoryPlan",
    "QuantumTrajectoryStatus",
]
