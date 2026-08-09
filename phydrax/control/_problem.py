#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..dynamics import TimeGrid
from ._constraints import PathConstraint, TerminalConstraint
from ._cost import RunningCost, TerminalCost


if TYPE_CHECKING:
    from ._dynamics import DifferentialControlDynamics, DiscreteControlDynamics
    from ._parameterization import AbstractControlParameterization
    from ._trajectory import ControlResult, ControlTrajectory


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string.")
    return value


def _shape(value: Sequence[int], name: str, /) -> tuple[int, ...]:
    shape = tuple(int(size) for size in value)
    if any(size <= 0 for size in shape):
        raise ValueError(f"{name} dimensions must be positive.")
    return shape


class ControlProblem(StrictModule):
    """Finite-horizon control problem with explicit physical and case axes."""

    dynamics: DiscreteControlDynamics | DifferentialControlDynamics
    time_grid: TimeGrid
    initial_state: Array
    running_cost: RunningCost | None
    terminal_cost: TerminalCost | None
    path_constraints: tuple[PathConstraint, ...]
    terminal_constraints: tuple[TerminalConstraint, ...]
    args: Any
    case_shape: tuple[int, ...] = eqx.field(static=True)
    state_shape: tuple[int, ...] = eqx.field(static=True)
    control_shape: tuple[int, ...] = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        dynamics: DiscreteControlDynamics | DifferentialControlDynamics,
        time_grid: TimeGrid,
        initial_state: ArrayLike,
        /,
        *,
        running_cost: RunningCost | None = None,
        terminal_cost: TerminalCost | None = None,
        path_constraints: Sequence[PathConstraint] = (),
        terminal_constraints: Sequence[TerminalConstraint] = (),
        args: Any = None,
        problem_id: str,
    ):
        from ._dynamics import DifferentialControlDynamics, DiscreteControlDynamics

        if not isinstance(
            dynamics, (DiscreteControlDynamics, DifferentialControlDynamics)
        ):
            raise TypeError(
                "ControlProblem dynamics must be DiscreteControlDynamics or "
                "DifferentialControlDynamics."
            )
        if not isinstance(time_grid, TimeGrid):
            raise TypeError("ControlProblem time_grid must be a TimeGrid.")
        state = jnp.asarray(initial_state)
        state_shape = dynamics.state_shape
        if state.ndim < len(state_shape) or (
            state_shape and tuple(state.shape[-len(state_shape) :]) != state_shape
        ):
            raise ValueError(
                "ControlProblem initial_state must end with dynamics state_shape "
                f"{state_shape}; got {state.shape}."
            )
        case_shape = tuple(
            int(size) for size in state.shape[: state.ndim - len(state_shape)]
        )
        if any(size <= 0 for size in case_shape):
            raise ValueError("ControlProblem case dimensions must be positive.")
        if not jnp.issubdtype(state.dtype, jnp.inexact):
            state = state.astype(float)
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
        self.initial_state = state
        self.running_cost = running_cost
        self.terminal_cost = terminal_cost
        self.path_constraints = path
        self.terminal_constraints = terminal
        self.args = args
        self.case_shape = case_shape
        self.state_shape = state_shape
        self.control_shape = dynamics.control_shape
        self.problem_id = _identifier(problem_id, "ControlProblem problem_id")

    def rollout(
        self,
        parameterization: AbstractControlParameterization,
        coefficients: ArrayLike,
        /,
        **solver_options: Any,
    ) -> ControlTrajectory:
        """Roll out one parameterized control without evaluating objective terms."""
        from ._parameterization import AbstractControlParameterization

        if not isinstance(parameterization, AbstractControlParameterization):
            raise TypeError(
                "parameterization must implement AbstractControlParameterization."
            )
        if parameterization.control_shape != self.control_shape:
            raise ValueError(
                "Control parameterization control_shape does not match the problem."
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
        parameterization: AbstractControlParameterization,
        coefficients: ArrayLike,
        /,
        **solver_options: Any,
    ) -> ControlResult:
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
            result_id=f"control-result:{self.problem_id}",
            method_id=trajectory.method_id,
        )


__all__ = ["ControlProblem"]
