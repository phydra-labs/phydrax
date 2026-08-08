#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import prod
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


class RunningCost(Protocol):
    """Scalar running cost rate evaluated at one time, state, and control."""

    def __call__(
        self, time: Array, state: Array, control: Array, args: Any, /
    ) -> ArrayLike: ...


class TerminalCost(Protocol):
    """Scalar terminal cost evaluated at one terminal state."""

    def __call__(self, time: Array, state: Array, args: Any, /) -> ArrayLike: ...


class SampledControlLoss(StrictModule):
    """Left-rectangle running loss and terminal loss on a declared grid."""

    running_samples: Array
    running_integral: Array
    terminal: Array
    total: Array
    valid: Array
    case_shape: tuple[int, ...] = eqx.field(static=True)
    num_steps: int = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        running_samples: ArrayLike,
        running_integral: ArrayLike,
        terminal: ArrayLike,
        total: ArrayLike,
        valid: ArrayLike,
        case_shape: tuple[int, ...],
        num_steps: int,
        method_id: str,
    ):
        samples = jnp.asarray(running_samples)
        integral = jnp.asarray(running_integral)
        terminal_ = jnp.asarray(terminal)
        total_ = jnp.asarray(total)
        valid_ = jnp.asarray(valid, dtype=bool)
        if samples.shape != case_shape + (num_steps,):
            raise ValueError(
                "SampledControlLoss running_samples must have shape "
                "case_shape + (num_steps,)."
            )
        for name, value in (
            ("running_integral", integral),
            ("terminal", terminal_),
            ("total", total_),
            ("valid", valid_),
        ):
            if value.shape != case_shape:
                raise ValueError(
                    f"SampledControlLoss {name} must have case_shape {case_shape}."
                )
        self.running_samples = samples
        self.running_integral = integral
        self.terminal = terminal_
        self.total = total_
        self.valid = valid_
        self.case_shape = case_shape
        self.num_steps = int(num_steps)
        self.method_id = _identifier(method_id, "SampledControlLoss method_id")


def _scalar_running(
    callback: RunningCost,
    time: Array,
    state: Array,
    control: Array,
    args: Any,
    /,
) -> Array:
    value = jnp.asarray(callback(time, state, control, args))
    if value.shape != ():
        raise ValueError("RunningCost must return one scalar per case and time.")
    return value


def _scalar_terminal(
    callback: TerminalCost,
    time: Array,
    state: Array,
    args: Any,
    /,
) -> Array:
    value = jnp.asarray(callback(time, state, args))
    if value.shape != ():
        raise ValueError("TerminalCost must return one scalar per case.")
    return value


def evaluate_sampled_cost(
    problem: ControlProblem,
    trajectory: ControlTrajectory,
    /,
) -> SampledControlLoss:
    """Evaluate a grid-sampled objective without making a feasibility claim."""
    from ._problem import ControlProblem
    from ._trajectory import ControlTrajectory

    if not isinstance(problem, ControlProblem):
        raise TypeError("problem must be a ControlProblem.")
    if not isinstance(trajectory, ControlTrajectory):
        raise TypeError("trajectory must be a ControlTrajectory.")
    if trajectory.problem_id != problem.problem_id:
        raise ValueError("trajectory problem_id does not match the ControlProblem.")
    cases = problem.case_shape
    count = prod(cases) if cases else 1
    states = trajectory.states.reshape(
        (count, problem.time_grid.num_times) + problem.state_shape
    )
    controls = trajectory.controls.reshape(
        (count, problem.time_grid.num_steps) + problem.control_shape
    )
    times = problem.time_grid.times

    if problem.running_cost is None:
        running_flat = jnp.zeros(
            (count, problem.time_grid.num_steps), dtype=trajectory.states.real.dtype
        )
    else:
        running_callback = problem.running_cost

        def evaluate_case(case_states: Array, case_controls: Array) -> Array:
            return jax.vmap(
                lambda time, state, control: _scalar_running(
                    running_callback,
                    time,
                    state,
                    control,
                    problem.args,
                )
            )(times[:-1], case_states[:-1], case_controls)

        running_flat = jax.vmap(evaluate_case)(states, controls)

    if problem.terminal_cost is None:
        terminal_flat = jnp.zeros((count,), dtype=trajectory.states.real.dtype)
    else:
        terminal_callback = problem.terminal_cost
        terminal_flat = jax.vmap(
            lambda state: _scalar_terminal(
                terminal_callback,
                times[-1],
                state,
                problem.args,
            )
        )(states[:, -1])

    running = running_flat.reshape(cases + (problem.time_grid.num_steps,))
    terminal = terminal_flat.reshape(cases)
    running_integral = jnp.sum(running * problem.time_grid.durations, axis=-1)
    total = running_integral + terminal
    valid = (
        trajectory.successful
        & jnp.all(jnp.isfinite(running), axis=-1)
        & jnp.isfinite(terminal)
        & jnp.isfinite(total)
    )
    return SampledControlLoss(
        running_samples=running,
        running_integral=running_integral,
        terminal=terminal,
        total=total,
        valid=valid,
        case_shape=cases,
        num_steps=problem.time_grid.num_steps,
        method_id="control-cost:sampled-left-rectangle+terminal",
    )


__all__ = [
    "RunningCost",
    "SampledControlLoss",
    "TerminalCost",
    "evaluate_sampled_cost",
]
