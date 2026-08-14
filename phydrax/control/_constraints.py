#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from math import isfinite, prod
from typing import Any, Protocol, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule


if TYPE_CHECKING:
    from ._problem import ControlProblem
    from ._trajectory import ControlTrajectory


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string.")
    return value


class PathConstraint(Protocol):
    """Scalar path residual; values less than or equal to zero are feasible."""

    def __call__(
        self, time: Array, state: Array, control: Array, args: Any, /
    ) -> ArrayLike: ...


class TerminalConstraint(Protocol):
    """Scalar terminal residual; values less than or equal to zero are feasible."""

    def __call__(self, time: Array, state: Array, args: Any, /) -> ArrayLike: ...


class SampledControlFeasibility(StrictModule):
    """Sampled nonlinear residuals that make no continuous-domain certificate."""

    path_residuals: Array
    terminal_residuals: Array
    maximum_violation: Array
    feasible: Array
    case_shape: tuple[int, ...] = eqx.field(static=True)
    num_steps: int = eqx.field(static=True)
    num_path_constraints: int = eqx.field(static=True)
    num_terminal_constraints: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    certified: bool = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        path_residuals: ArrayLike,
        terminal_residuals: ArrayLike,
        maximum_violation: ArrayLike,
        feasible: ArrayLike,
        case_shape: tuple[int, ...],
        num_steps: int,
        num_path_constraints: int,
        num_terminal_constraints: int,
        tolerance: float,
        method_id: str,
    ):
        path = jnp.asarray(path_residuals)
        terminal = jnp.asarray(terminal_residuals)
        maximum = jnp.asarray(maximum_violation)
        feasible_ = jnp.asarray(feasible, dtype=bool)
        expected_path = case_shape + (num_steps, num_path_constraints)
        expected_terminal = case_shape + (num_terminal_constraints,)
        if path.shape != expected_path:
            raise ValueError(
                f"path_residuals must have shape {expected_path}; got {path.shape}."
            )
        if terminal.shape != expected_terminal:
            raise ValueError(
                "terminal_residuals must have shape "
                f"{expected_terminal}; got {terminal.shape}."
            )
        if maximum.shape != case_shape or feasible_.shape != case_shape:
            raise ValueError("maximum_violation and feasible must both have case_shape.")
        self.path_residuals = path
        self.terminal_residuals = terminal
        self.maximum_violation = maximum
        self.feasible = feasible_
        self.case_shape = case_shape
        self.num_steps = int(num_steps)
        self.num_path_constraints = int(num_path_constraints)
        self.num_terminal_constraints = int(num_terminal_constraints)
        self.tolerance = float(tolerance)
        self.certified = False
        self.method_id = _identifier(method_id, "SampledControlFeasibility method_id")


def _scalar_path(
    callback: Callable[..., ArrayLike],
    time: Array,
    state: Array,
    control: Array,
    args: Any,
    /,
) -> Array:
    value = jnp.asarray(callback(time, state, control, args))
    if value.shape != ():
        raise ValueError("PathConstraint must return one scalar per case and time.")
    return value


def _scalar_terminal(
    callback: TerminalConstraint,
    time: Array,
    state: Array,
    args: Any,
    /,
) -> Array:
    value = jnp.asarray(callback(time, state, args))
    if value.shape != ():
        raise ValueError("TerminalConstraint must return one scalar per case.")
    return value


def evaluate_sampled_feasibility(
    problem: ControlProblem,
    trajectory: ControlTrajectory,
    /,
    *,
    tolerance: float = 0.0,
) -> SampledControlFeasibility:
    """Check declared nonlinear constraints only at trajectory sample sites."""
    from ._problem import ControlProblem
    from ._trajectory import ControlTrajectory

    if not isinstance(problem, ControlProblem):
        raise TypeError("problem must be a ControlProblem.")
    if not isinstance(trajectory, ControlTrajectory):
        raise TypeError("trajectory must be a ControlTrajectory.")
    if trajectory.problem_id != problem.problem_id:
        raise ValueError("trajectory problem_id does not match the ControlProblem.")
    tolerance_ = float(tolerance)
    if not isfinite(tolerance_) or tolerance_ < 0.0:
        raise ValueError("feasibility tolerance must be finite and nonnegative.")

    cases = problem.case_shape
    count = prod(cases) if cases else 1
    states = trajectory.states.reshape(
        (count, problem.time_grid.num_times) + problem.state_shape
    )
    controls = trajectory.controls.reshape(
        (count, problem.time_grid.num_steps) + problem.control_shape
    )
    times = problem.time_grid.times
    path_columns: list[Array] = []
    for constraint in problem.path_constraints:

        def evaluate_case(case_states: Array, case_controls: Array) -> Array:
            return jax.vmap(
                lambda time, state, control: _scalar_path(
                    constraint,
                    time,
                    state,
                    control,
                    problem.args,
                )
            )(times[:-1], case_states[:-1], case_controls)

        path_columns.append(jax.vmap(evaluate_case)(states, controls))
    path_flat = (
        jnp.stack(path_columns, axis=-1)
        if path_columns
        else jnp.zeros(
            (count, problem.time_grid.num_steps, 0),
            dtype=trajectory.states.real.dtype,
        )
    )

    terminal_columns: list[Array] = []
    for constraint in problem.terminal_constraints:
        terminal_columns.append(
            jax.vmap(
                lambda state: _scalar_terminal(
                    constraint,
                    times[-1],
                    state,
                    problem.args,
                )
            )(states[:, -1])
        )
    terminal_flat = (
        jnp.stack(terminal_columns, axis=-1)
        if terminal_columns
        else jnp.zeros((count, 0), dtype=trajectory.states.real.dtype)
    )

    path = path_flat.reshape(
        cases + (problem.time_grid.num_steps, len(problem.path_constraints))
    )
    terminal = terminal_flat.reshape(cases + (len(problem.terminal_constraints),))
    path_finite = jnp.all(jnp.isfinite(path), axis=(-2, -1))
    terminal_finite = jnp.all(jnp.isfinite(terminal), axis=-1)
    path_feasible = jnp.all(path <= tolerance_, axis=(-2, -1))
    terminal_feasible = jnp.all(terminal <= tolerance_, axis=-1)
    feasible = (
        trajectory.successful
        & path_finite
        & terminal_finite
        & path_feasible
        & terminal_feasible
    )

    maximum = jnp.zeros(cases, dtype=trajectory.states.real.dtype)
    if problem.path_constraints:
        maximum = jnp.maximum(maximum, jnp.max(path, axis=(-2, -1)))
    if problem.terminal_constraints:
        maximum = jnp.maximum(maximum, jnp.max(terminal, axis=-1))
    maximum = jnp.maximum(maximum, 0.0)
    return SampledControlFeasibility(
        path_residuals=path,
        terminal_residuals=terminal,
        maximum_violation=maximum,
        feasible=feasible,
        case_shape=cases,
        num_steps=problem.time_grid.num_steps,
        num_path_constraints=len(problem.path_constraints),
        num_terminal_constraints=len(problem.terminal_constraints),
        tolerance=tolerance_,
        method_id="control-constraint:sampled-grid-noncertifying",
    )


__all__ = [
    "PathConstraint",
    "SampledControlFeasibility",
    "TerminalConstraint",
    "evaluate_sampled_feasibility",
]
