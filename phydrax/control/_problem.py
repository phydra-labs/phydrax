#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

import phydrax.control as control_api

from .._strict import StrictModule
from ..dynamics import TimeGrid
from ..dynamics._plant import PlantRuntimeState
from ._constraints import PathConstraint, TerminalConstraint
from ._cost import RunningCost, TerminalCost


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string.")
    return value


def _shape(value: Sequence[int], name: str, /) -> tuple[int, ...]:
    shape = tuple(value)
    if any(size <= 0 for size in shape):
        raise ValueError(f"{name} dimensions must be positive.")
    return shape


def _require_array_dynamics(
    problem: ControlProblem,
) -> control_api.DiscreteControlDynamics | control_api.DifferentialControlDynamics:
    """Admit only dynamics whose full Markov state is the optimization array."""
    dynamics = problem.dynamics
    if not isinstance(
        dynamics,
        (control_api.DiscreteControlDynamics, control_api.DifferentialControlDynamics),
    ):
        raise ValueError(
            "This solver requires array-system dynamics; a complete plant requires "
            "a state-aware solver and cannot be reconstructed from its vector view."
        )
    return dynamics


def _validate_trajectory_problem(
    problem: ControlProblem, trajectory: control_api.ControlTrajectory, /
) -> control_api.ControlTrajectory:
    if trajectory.problem_id != problem.problem_id:
        raise ValueError("trajectory problem_id does not match the ControlProblem.")
    if trajectory.dynamics_id != problem.dynamics.dynamics_id:
        raise ValueError("trajectory dynamics_id does not match the ControlProblem.")
    if trajectory.case_shape != problem.case_shape:
        raise ValueError("trajectory case_shape does not match the ControlProblem.")
    if trajectory.state_shape != problem.state_shape:
        raise ValueError("trajectory state_shape does not match the ControlProblem.")
    if trajectory.control_shape != problem.control_shape:
        raise ValueError("trajectory control_shape does not match the ControlProblem.")
    if (
        trajectory.time_grid.time_id != problem.time_grid.time_id
        or trajectory.time_grid.times.shape != problem.time_grid.times.shape
    ):
        raise ValueError("trajectory time grid does not match the ControlProblem.")
    checked_times = eqx.error_if(
        trajectory.time_grid.times,
        jnp.any(trajectory.time_grid.times != problem.time_grid.times),
        "trajectory time grid does not match the ControlProblem.",
    )
    return eqx.tree_at(
        lambda value: value.time_grid.times,
        trajectory,
        checked_times,
    )


class ControlProblem(StrictModule):
    """Finite-horizon control problem with explicit physical and case axes."""

    dynamics: (
        control_api.DiscreteControlDynamics
        | control_api.DifferentialControlDynamics
        | control_api.PlantControlDynamics
    )
    time_grid: TimeGrid
    _initial_state: Array | PlantRuntimeState
    running_cost: RunningCost | None = eqx.field(static=True)
    terminal_cost: TerminalCost | None = eqx.field(static=True)
    path_constraints: tuple[PathConstraint, ...]
    terminal_constraints: tuple[TerminalConstraint, ...]
    args: Any
    case_shape: tuple[int, ...] = eqx.field(static=True)
    state_shape: tuple[int, ...] = eqx.field(static=True)
    control_shape: tuple[int, ...] = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        dynamics: control_api.DiscreteControlDynamics
        | control_api.DifferentialControlDynamics
        | control_api.PlantControlDynamics,
        time_grid: TimeGrid,
        initial_state: ArrayLike | PlantRuntimeState,
        /,
        *,
        running_cost: RunningCost | None = None,
        terminal_cost: TerminalCost | None = None,
        path_constraints: Sequence[PathConstraint] = (),
        terminal_constraints: Sequence[TerminalConstraint] = (),
        args: Any = None,
        problem_id: str,
    ) -> None:
        from ._dynamics import DifferentialControlDynamics, DiscreteControlDynamics
        from ._plant_dynamics import PlantControlDynamics

        if not isinstance(
            dynamics,
            (DiscreteControlDynamics, DifferentialControlDynamics, PlantControlDynamics),
        ):
            raise TypeError(
                "ControlProblem dynamics must be a discrete, differential, or complete-plant dynamics contract."
            )
        if not isinstance(time_grid, TimeGrid):
            raise TypeError("ControlProblem time_grid must be a TimeGrid.")
        state_shape = dynamics.state_shape
        if isinstance(dynamics, PlantControlDynamics):
            if not isinstance(initial_state, PlantRuntimeState):
                raise TypeError(
                    "Plant control requires a complete initial PlantRuntimeState."
                )
            state, case_shape = dynamics.validate_initial_state(initial_state)
        else:
            if isinstance(initial_state, PlantRuntimeState):
                raise TypeError(
                    "Array control dynamics cannot consume a complete plant state."
                )
            state = jnp.asarray(initial_state)
            if state.ndim < len(state_shape) or (
                state_shape and tuple(state.shape[-len(state_shape) :]) != state_shape
            ):
                raise ValueError(
                    f"ControlProblem initial_state must end with dynamics state_shape {state_shape}; got {state.shape}."
                )
            case_shape = tuple(state.shape[: state.ndim - len(state_shape)])
            if any(size <= 0 for size in case_shape):
                raise ValueError("ControlProblem case dimensions must be positive.")
            if not jnp.issubdtype(state.dtype, jnp.inexact):
                state = state.astype(jnp.float64)
            state = eqx.error_if(
                state,
                jnp.any(~jnp.isfinite(state)),
                "ControlProblem initial_state must be finite.",
            )
        if running_cost is not None and not callable(running_cost):
            raise TypeError("ControlProblem running_cost must be callable or None.")
        if terminal_cost is not None and not callable(terminal_cost):
            raise TypeError("ControlProblem terminal_cost must be callable or None.")
        path = tuple(path_constraints)
        terminal = tuple(terminal_constraints)
        if any(not callable(constraint) for constraint in path):
            raise TypeError("Every ControlProblem path constraint must be callable.")
        if any(not callable(constraint) for constraint in terminal):
            raise TypeError("Every ControlProblem terminal constraint must be callable.")

        self.dynamics = dynamics
        self.time_grid = time_grid
        self._initial_state = state
        self.running_cost = running_cost
        self.terminal_cost = terminal_cost
        self.path_constraints = path
        self.terminal_constraints = terminal
        self.args = args
        self.case_shape = case_shape
        self.state_shape = state_shape
        self.control_shape = dynamics.control_shape
        self.problem_id = _identifier(problem_id, "ControlProblem problem_id")

    @property
    def initial_state(self) -> Array:
        """Continuous controller view; complete plant state remains authoritative."""
        from ._plant_dynamics import PlantControlDynamics

        if isinstance(self._initial_state, PlantRuntimeState):
            if not isinstance(self.dynamics, PlantControlDynamics):
                raise RuntimeError("Complete initial state has incompatible dynamics.")
            return self.dynamics.state_vector(self._initial_state)
        return self._initial_state

    def rollout(
        self,
        parameterization: control_api.AbstractControlParameterization,
        coefficients: ArrayLike,
        /,
        **solver_options: Any,
    ) -> control_api.ControlTrajectory:
        """Roll out one parameterized control without evaluating objective terms."""
        from ._parameterization import (
            _validate_parameterization_grid,
            AbstractControlParameterization,
        )

        if not isinstance(parameterization, AbstractControlParameterization):
            raise TypeError(
                "parameterization must implement AbstractControlParameterization."
            )
        if parameterization.control_shape != self.control_shape:
            raise ValueError(
                "Control parameterization control_shape does not match the problem."
            )
        parameterization = _validate_parameterization_grid(
            parameterization, self.time_grid
        )
        from ._plant_dynamics import PlantControlDynamics

        if isinstance(self.dynamics, PlantControlDynamics):
            if not isinstance(self._initial_state, PlantRuntimeState):
                raise RuntimeError("Plant control lost its complete initial state.")
            return self.dynamics.rollout(
                self.time_grid,
                self._initial_state,
                parameterization,
                coefficients,
                args=self.args,
                problem_id=self.problem_id,
                **solver_options,
            )
        return self.dynamics.rollout(
            self.time_grid,
            self.initial_state,
            parameterization,
            coefficients,
            args=self.args,
            problem_id=self.problem_id,
            **solver_options,
        )

    def evaluate(
        self,
        parameterization: control_api.AbstractControlParameterization,
        coefficients: ArrayLike,
        /,
        **solver_options: Any,
    ) -> control_api.ControlResult:
        """Roll out and separately evaluate sampled loss and sampled feasibility."""
        from ._constraints import evaluate_sampled_feasibility
        from ._cost import evaluate_sampled_cost
        from ._trajectory import ControlResult

        trajectory = self.rollout(parameterization, coefficients, **solver_options)
        sampled_loss = evaluate_sampled_cost(self, trajectory)
        feasibility = evaluate_sampled_feasibility(self, trajectory)
        return ControlResult(
            trajectory=trajectory,
            parameters=coefficients,
            sampled_loss=sampled_loss,
            feasibility=feasibility,
            result_namespace=f"control-result:{self.problem_id}",
            method_id=trajectory.method_id,
        )


__all__ = ["ControlProblem"]
