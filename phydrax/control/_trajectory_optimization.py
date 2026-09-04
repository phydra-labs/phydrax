#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from math import prod
from typing import Any, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..dynamics import ContinuousSystem, DifferentialAlgebraicSystem, StateLayout
from ..linalg import AbstractVectorSpace
from ..metrix import AbstractStateGeometry, EuclideanStateGeometry
from ._cost import RunningCost, TerminalCost
from ._problem import _identifier


TrajectoryCost: TypeAlias = Callable[["TrajectoryOptimizationView", Any], ArrayLike]
PathConstraintFunction: TypeAlias = Callable[[Array, Array, Array, Any], ArrayLike]
TrajectoryConstraintFunction: TypeAlias = Callable[
    ["TrajectoryOptimizationView", Any], ArrayLike
]
TrajectoryDynamics: TypeAlias = ContinuousSystem | DifferentialAlgebraicSystem


def _inexact(value: ArrayLike, /) -> Array:
    array = jnp.asarray(value)
    if jnp.issubdtype(array.dtype, jnp.complexfloating):
        raise TypeError("Trajectory optimization arrays must be real-valued.")
    return array if jnp.issubdtype(array.dtype, jnp.inexact) else array.astype(float)


def _case_shape(value: Sequence[int], /) -> tuple[int, ...]:
    shape = tuple(int(size) for size in value)
    if any(size <= 0 for size in shape):
        raise ValueError("Trajectory optimization case dimensions must be positive.")
    return shape


def _positive_scale(value: ArrayLike, owner: str, /) -> Array:
    scale = _inexact(value)
    return eqx.error_if(
        scale,
        jnp.any(~jnp.isfinite(scale)) | jnp.any(scale <= 0.0),
        f"{owner} must be finite and positive.",
    )


class TrajectoryOptimizationContext(StrictModule):
    """Fixed arguments plus optimized shared parameters and physical duration."""

    args: Any
    parameters: Any
    duration: Array | None

    def __init__(self, args: Any, parameters: Any, duration: Array | None, /):
        self.args = args
        self.parameters = parameters
        self.duration = duration


class TrajectoryOptimizationView(StrictModule):
    """Physical node states and held interval controls exposed to global callbacks."""

    times: Array
    states: Array
    controls: Array
    state_geometry: AbstractStateGeometry
    case_shape: tuple[int, ...] = eqx.field(static=True)
    state_shape: tuple[int, ...] = eqx.field(static=True)
    control_shape: tuple[int, ...] = eqx.field(static=True)
    approximation_id: str = eqx.field(static=True)

    def __init__(
        self,
        times: ArrayLike,
        states: ArrayLike,
        controls: ArrayLike,
        /,
        *,
        case_shape: Sequence[int],
        state_shape: Sequence[int],
        control_shape: Sequence[int],
        state_geometry: AbstractStateGeometry | None = None,
        approximation_id: str = "control:direct-collocation:retracted-state-held-control",
    ):
        times_ = _inexact(times)
        cases = _case_shape(case_shape)
        state_event = tuple(int(size) for size in state_shape)
        control_event = tuple(int(size) for size in control_shape)
        geometry = EuclideanStateGeometry() if state_geometry is None else state_geometry
        if not isinstance(geometry, AbstractStateGeometry):
            raise TypeError("state_geometry must be an AbstractStateGeometry or None.")
        if not geometry.supports_exact_inverse:
            raise ValueError(
                "TrajectoryOptimizationView requires exact inverse-retraction geometry."
            )
        if times_.ndim != 1 or int(times_.size) < 2:
            raise ValueError("Trajectory times must be rank one with at least two nodes.")
        states_ = _inexact(states)
        controls_ = _inexact(controls)
        expected_states = cases + (int(times_.size),) + state_event
        expected_controls = cases + (int(times_.size) - 1,) + control_event
        if states_.shape != expected_states:
            raise ValueError(
                f"states must have shape {expected_states}; got {states_.shape}."
            )
        if controls_.shape != expected_controls:
            raise ValueError(
                f"controls must have shape {expected_controls}; got {controls_.shape}."
            )
        times_ = eqx.error_if(
            times_,
            jnp.any(~jnp.isfinite(times_)) | jnp.any(jnp.diff(times_) <= 0.0),
            "Trajectory times must be finite and strictly increasing.",
        )
        self.times = times_
        self.states = states_
        self.controls = controls_
        self.state_geometry = geometry
        self.case_shape = cases
        self.state_shape = state_event
        self.control_shape = control_event
        self.approximation_id = _identifier(approximation_id, "approximation_id")

    @property
    def num_nodes(self) -> int:
        return int(self.times.size)

    @property
    def num_intervals(self) -> int:
        return self.num_nodes - 1

    @property
    def initial_state(self) -> Array:
        return jnp.take(self.states, 0, axis=len(self.case_shape))

    @property
    def final_state(self) -> Array:
        return jnp.take(self.states, self.num_nodes - 1, axis=len(self.case_shape))

    def _query(self, query_times: ArrayLike, /) -> Array:
        query = _inexact(query_times).astype(self.times.dtype)
        return eqx.error_if(
            query,
            jnp.any(~jnp.isfinite(query))
            | jnp.any(query < self.times[0])
            | jnp.any(query > self.times[-1]),
            "Trajectory query times must lie inside the physical horizon.",
        )

    def evaluate_state(self, query_times: ArrayLike, /, *, left: bool = True) -> Array:
        """Retraction interpolation with output ``case + query + state``."""
        if not isinstance(left, bool):
            raise TypeError("left must be a bool.")
        query = self._query(query_times)
        side = "left" if left else "right"
        indices = jnp.searchsorted(self.times, query.reshape((-1,)), side=side) - 1
        indices = jnp.clip(indices, 0, self.num_intervals - 1)
        lower_time = self.times[indices]
        upper_time = self.times[indices + 1]
        weight = (query.reshape((-1,)) - lower_time) / (upper_time - lower_time)
        axis = len(self.case_shape)
        lower = jnp.take(self.states, indices, axis=axis)
        upper = jnp.take(self.states, indices + 1, axis=axis)
        sample_count = (prod(self.case_shape) if self.case_shape else 1) * int(query.size)
        flat_lower = lower.reshape((sample_count,) + self.state_shape)
        flat_upper = upper.reshape((sample_count,) + self.state_shape)
        flat_weight = jnp.broadcast_to(
            weight,
            self.case_shape + (int(query.size),),
        ).reshape((sample_count,))

        def interpolate(base, point, fraction):
            local = jnp.asarray(self.state_geometry.inverse_retract(base, point))
            return jnp.asarray(self.state_geometry.retract(base, fraction * local))

        values = jax.vmap(interpolate)(flat_lower, flat_upper, flat_weight)
        return values.reshape(self.case_shape + query.shape + self.state_shape)

    def evaluate_control(self, query_times: ArrayLike, /, *, left: bool = True) -> Array:
        """Held interval controls with output ``case + query + control``."""
        if not isinstance(left, bool):
            raise TypeError("left must be a bool.")
        query = self._query(query_times)
        side = "left" if left else "right"
        indices = jnp.searchsorted(self.times, query.reshape((-1,)), side=side) - 1
        indices = jnp.clip(indices, 0, self.num_intervals - 1)
        values = jnp.take(self.controls, indices, axis=len(self.case_shape))
        return values.reshape(self.case_shape + query.shape + self.control_shape)


class BoundedPathConstraint(StrictModule):
    """Bound-form fixed-shape residual evaluated at every collocation stage."""

    function: PathConstraintFunction
    lower: Any
    upper: Any
    scale: Array
    constraint_id: str = eqx.field(static=True)

    def __init__(
        self,
        function: PathConstraintFunction,
        /,
        *,
        lower: Any = -jnp.inf,
        upper: Any = jnp.inf,
        scale: ArrayLike = 1.0,
        constraint_id: str,
    ):
        if not callable(function):
            raise TypeError("BoundedPathConstraint function must be callable.")
        self.function = function
        self.lower = lower
        self.upper = upper
        self.scale = _positive_scale(scale, "BoundedPathConstraint scale")
        self.constraint_id = _identifier(constraint_id, "constraint_id")

    def __call__(
        self,
        time: Array,
        state: Array,
        control: Array,
        args: Any,
        /,
    ) -> Array:
        return _inexact(self.function(time, state, control, args))


class BoundedTrajectoryConstraint(StrictModule):
    """Bound-form residual over a complete collocation trajectory."""

    function: TrajectoryConstraintFunction
    lower: Any
    upper: Any
    scale: Array
    constraint_id: str = eqx.field(static=True)

    def __init__(
        self,
        function: TrajectoryConstraintFunction,
        /,
        *,
        lower: Any = -jnp.inf,
        upper: Any = jnp.inf,
        scale: ArrayLike = 1.0,
        constraint_id: str,
    ):
        if not callable(function):
            raise TypeError("BoundedTrajectoryConstraint function must be callable.")
        self.function = function
        self.lower = lower
        self.upper = upper
        self.scale = _positive_scale(scale, "BoundedTrajectoryConstraint scale")
        self.constraint_id = _identifier(constraint_id, "constraint_id")

    def __call__(self, trajectory: TrajectoryOptimizationView, args: Any, /) -> Array:
        return _inexact(self.function(trajectory, args))


class TrajectoryOptimizationProblem(StrictModule):
    """Continuous trajectory objective independent of its transcription and NLP solver."""

    dynamics: TrajectoryDynamics
    initial_state: Array | None
    running_cost: RunningCost | None
    terminal_cost: TerminalCost | None
    trajectory_cost: TrajectoryCost | None
    path_constraints: tuple[BoundedPathConstraint, ...]
    trajectory_constraints: tuple[BoundedTrajectoryConstraint, ...]
    parameter_space: AbstractVectorSpace | None
    state_layout: StateLayout
    args: Any
    case_shape: tuple[int, ...] = eqx.field(static=True)
    state_shape: tuple[int, ...] = eqx.field(static=True)
    control_shape: tuple[int, ...] = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    dynamics_id: str = eqx.field(static=True)

    def __init__(
        self,
        dynamics: TrajectoryDynamics,
        /,
        *,
        initial_state: ArrayLike | None = None,
        case_shape: Sequence[int] = (),
        running_cost: RunningCost | None = None,
        terminal_cost: TerminalCost | None = None,
        trajectory_cost: TrajectoryCost | None = None,
        path_constraints: Sequence[BoundedPathConstraint] = (),
        trajectory_constraints: Sequence[BoundedTrajectoryConstraint] = (),
        parameter_space: AbstractVectorSpace | None = None,
        args: Any = None,
        problem_id: str,
    ):
        if not isinstance(dynamics, (ContinuousSystem, DifferentialAlgebraicSystem)):
            raise TypeError(
                "TrajectoryOptimizationProblem dynamics must be ContinuousSystem or "
                "DifferentialAlgebraicSystem."
            )
        if isinstance(dynamics, ContinuousSystem):
            state_layout = dynamics.state_layout
            state_shape = state_layout.shape
            input_layout = dynamics.input_layout
        else:
            state_shape = dynamics.state_shape
            state_layout = StateLayout(
                state_shape,
                geometry=dynamics.state_geometry,
                layout_id=f"{dynamics.system_id}:state-layout",
            )
            input_layout = dynamics.input_layout
        if input_layout is None:
            raise ValueError("Trajectory optimization dynamics require explicit inputs.")
        requested_cases = _case_shape(case_shape)
        if initial_state is None:
            state = None
            cases = requested_cases
        else:
            state = _inexact(initial_state)
            if state.ndim < len(state_shape) or (
                state_shape and tuple(state.shape[-len(state_shape) :]) != state_shape
            ):
                raise ValueError(
                    "initial_state must end with dynamics state shape "
                    f"{state_shape}; got {state.shape}."
                )
            cases = tuple(state.shape[: state.ndim - len(state_shape)])
            if requested_cases and requested_cases != cases:
                raise ValueError(
                    f"case_shape {requested_cases} does not match initial_state cases "
                    f"{cases}."
                )
            state = eqx.error_if(
                state,
                jnp.any(~jnp.isfinite(state)),
                "Trajectory optimization initial_state must be finite.",
            )
        for callback, name in (
            (running_cost, "running_cost"),
            (terminal_cost, "terminal_cost"),
            (trajectory_cost, "trajectory_cost"),
        ):
            if callback is not None and not callable(callback):
                raise TypeError(f"{name} must be callable or None.")
        path = tuple(path_constraints)
        trajectory = tuple(trajectory_constraints)
        if any(not isinstance(item, BoundedPathConstraint) for item in path):
            raise TypeError("path_constraints must contain BoundedPathConstraint values.")
        if any(not isinstance(item, BoundedTrajectoryConstraint) for item in trajectory):
            raise TypeError(
                "trajectory_constraints must contain BoundedTrajectoryConstraint values."
            )
        if parameter_space is not None and not isinstance(
            parameter_space, AbstractVectorSpace
        ):
            raise TypeError("parameter_space must be an AbstractVectorSpace or None.")
        self.dynamics = dynamics
        self.state_layout = state_layout
        self.initial_state = state
        self.running_cost = running_cost
        self.terminal_cost = terminal_cost
        self.trajectory_cost = trajectory_cost
        self.path_constraints = path
        self.trajectory_constraints = trajectory
        self.parameter_space = parameter_space
        self.args = args
        self.case_shape = cases
        self.state_shape = tuple(state_shape)
        self.control_shape = tuple(input_layout.shape)
        self.problem_id = _identifier(problem_id, "problem_id")
        self.dynamics_id = dynamics.system_id


__all__ = [
    "BoundedPathConstraint",
    "BoundedTrajectoryConstraint",
    "PathConstraintFunction",
    "TrajectoryConstraintFunction",
    "TrajectoryCost",
    "TrajectoryDynamics",
    "TrajectoryOptimizationContext",
    "TrajectoryOptimizationProblem",
    "TrajectoryOptimizationView",
]
