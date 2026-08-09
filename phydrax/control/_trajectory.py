#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..dynamics import TimeGrid
from ._constraints import SampledControlFeasibility
from ._cost import SampledControlLoss
from ._problem import _identifier


CONTROL_SUCCESS = 0
CONTROL_DYNAMICS_FAILED = 1
CONTROL_COST_INVALID = 2
CONTROL_INFEASIBLE = 3


class ControlTrajectory(StrictModule):
    """State and applied-control history with explicit case and physical axes."""

    time_grid: TimeGrid
    states: Array
    controls: Array
    valid: Array
    status: Array
    backend_status: Any
    case_shape: tuple[int, ...] = eqx.field(static=True)
    state_shape: tuple[int, ...] = eqx.field(static=True)
    control_shape: tuple[int, ...] = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    dynamics_id: str = eqx.field(static=True)
    control_id: str = eqx.field(static=True)
    backend_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)
    discretization_id: str = eqx.field(static=True)
    approximation_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        time_grid: TimeGrid,
        states: ArrayLike,
        controls: ArrayLike,
        valid: ArrayLike,
        status: ArrayLike,
        backend_status: Any,
        case_shape: Sequence[int],
        state_shape: Sequence[int],
        control_shape: Sequence[int],
        problem_id: str,
        dynamics_id: str,
        control_id: str,
        backend_id: str,
        method_id: str,
        discretization_id: str,
        approximation_id: str,
    ):
        if not isinstance(time_grid, TimeGrid):
            raise TypeError("ControlTrajectory time_grid must be a TimeGrid.")
        cases = tuple(int(size) for size in case_shape)
        states_shape = tuple(int(size) for size in state_shape)
        controls_shape = tuple(int(size) for size in control_shape)
        if any(size <= 0 for size in cases + states_shape + controls_shape):
            raise ValueError("ControlTrajectory shape dimensions must be positive.")
        states_ = jnp.asarray(states)
        controls_ = jnp.asarray(controls)
        valid_ = jnp.asarray(valid, dtype=bool)
        status_ = jnp.asarray(status, dtype=jnp.int32)
        expected_states = cases + (time_grid.num_times,) + states_shape
        expected_controls = cases + (time_grid.num_steps,) + controls_shape
        expected_valid = cases + (time_grid.num_times,)
        if tuple(states_.shape) != expected_states:
            raise ValueError(
                f"ControlTrajectory states must have shape {expected_states}; "
                f"got {states_.shape}."
            )
        if tuple(controls_.shape) != expected_controls:
            raise ValueError(
                f"ControlTrajectory controls must have shape {expected_controls}; "
                f"got {controls_.shape}."
            )
        if tuple(valid_.shape) != expected_valid:
            raise ValueError(
                f"ControlTrajectory valid must have shape {expected_valid}; "
                f"got {valid_.shape}."
            )
        if tuple(status_.shape) != cases:
            raise ValueError(
                f"ControlTrajectory status must have case shape {cases}; "
                f"got {status_.shape}."
            )
        self.time_grid = time_grid
        self.states = states_
        self.controls = controls_
        self.valid = valid_
        self.status = status_
        self.backend_status = backend_status
        self.case_shape = cases
        self.state_shape = states_shape
        self.control_shape = controls_shape
        self.problem_id = _identifier(problem_id, "ControlTrajectory problem_id")
        self.dynamics_id = _identifier(dynamics_id, "ControlTrajectory dynamics_id")
        self.control_id = _identifier(control_id, "ControlTrajectory control_id")
        self.backend_id = _identifier(backend_id, "ControlTrajectory backend_id")
        self.method_id = _identifier(method_id, "ControlTrajectory method_id")
        self.discretization_id = _identifier(
            discretization_id, "ControlTrajectory discretization_id"
        )
        self.approximation_id = _identifier(
            approximation_id, "ControlTrajectory approximation_id"
        )

    @property
    def successful(self) -> Array:
        return (self.status == CONTROL_SUCCESS) & jnp.all(self.valid, axis=-1)

    @property
    def final_state(self) -> Array:
        axis = len(self.case_shape)
        return jnp.take(self.states, self.time_grid.num_times - 1, axis=axis)


class ControlResult(StrictModule):
    """Rollout, sampled objective, and sampled feasibility without conflation."""

    trajectory: ControlTrajectory
    parameters: Array
    sampled_loss: SampledControlLoss
    feasibility: SampledControlFeasibility
    valid: Array
    status: Array
    result_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        trajectory: ControlTrajectory,
        parameters: ArrayLike,
        sampled_loss: SampledControlLoss,
        feasibility: SampledControlFeasibility,
        result_id: str,
        method_id: str,
    ):
        if not isinstance(trajectory, ControlTrajectory):
            raise TypeError("ControlResult trajectory must be a ControlTrajectory.")
        if not isinstance(sampled_loss, SampledControlLoss):
            raise TypeError("sampled_loss must be a SampledControlLoss.")
        if not isinstance(feasibility, SampledControlFeasibility):
            raise TypeError("feasibility must be a SampledControlFeasibility.")
        if sampled_loss.case_shape != trajectory.case_shape:
            raise ValueError("sampled_loss case_shape must match the trajectory.")
        if feasibility.case_shape != trajectory.case_shape:
            raise ValueError("feasibility case_shape must match the trajectory.")
        valid = trajectory.successful & sampled_loss.valid
        status = jnp.where(
            trajectory.status != CONTROL_SUCCESS,
            trajectory.status,
            jnp.where(
                ~sampled_loss.valid,
                CONTROL_COST_INVALID,
                jnp.where(
                    ~feasibility.feasible,
                    CONTROL_INFEASIBLE,
                    CONTROL_SUCCESS,
                ),
            ),
        ).astype(jnp.int32)
        self.trajectory = trajectory
        self.parameters = jnp.asarray(parameters)
        self.sampled_loss = sampled_loss
        self.feasibility = feasibility
        self.valid = valid
        self.status = status
        self.result_id = _identifier(result_id, "ControlResult result_id")
        self.method_id = _identifier(method_id, "ControlResult method_id")

    @property
    def successful(self) -> Array:
        return self.status == CONTROL_SUCCESS


__all__ = [
    "CONTROL_COST_INVALID",
    "CONTROL_DYNAMICS_FAILED",
    "CONTROL_INFEASIBLE",
    "CONTROL_SUCCESS",
    "ControlResult",
    "ControlTrajectory",
]
