#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, TypeAlias

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..solver import DifferentialProblem, solve_diffrax
from ._parameterization import AbstractControlParameterization
from ._problem import _identifier, _shape, ControlTimeGrid
from ._trajectory import CONTROL_DYNAMICS_FAILED, CONTROL_SUCCESS, ControlTrajectory


DiscreteTransition: TypeAlias = Callable[[Array, Array, Array, Any], ArrayLike]
DifferentialControlVectorField: TypeAlias = Callable[
    [Array, Array, Array, Any], ArrayLike
]


def _case_and_state(
    initial_state: ArrayLike,
    state_shape: tuple[int, ...],
    /,
) -> tuple[Array, tuple[int, ...]]:
    state = jnp.asarray(initial_state)
    if state.ndim < len(state_shape) or (
        state_shape and tuple(state.shape[-len(state_shape) :]) != state_shape
    ):
        raise ValueError(
            f"initial_state must end with state_shape {state_shape}; got {state.shape}."
        )
    cases = tuple(int(size) for size in state.shape[: state.ndim - len(state_shape)])
    if any(size <= 0 for size in cases):
        raise ValueError("Control rollout case dimensions must be positive.")
    if not jnp.issubdtype(state.dtype, jnp.inexact):
        state = state.astype(float)
    return state, cases


def _event_finite(values: Array, event_shape: tuple[int, ...], /) -> Array:
    if not event_shape:
        return jnp.isfinite(values)
    axes = tuple(range(values.ndim - len(event_shape), values.ndim))
    return jnp.all(jnp.isfinite(values), axis=axes)


def _event_where(
    valid: Array,
    values: Array,
    replacement: Array,
    event_shape: tuple[int, ...],
    /,
) -> Array:
    selector = valid.reshape(valid.shape + (1,) * len(event_shape))
    return jnp.where(selector, values, replacement)


def _batched_transition(
    transition: DiscreteTransition,
    time: Array,
    states: Array,
    controls: Array,
    args: Any,
    case_shape: tuple[int, ...],
    state_shape: tuple[int, ...],
    control_shape: tuple[int, ...],
    /,
) -> Array:
    count = 1
    for size in case_shape:
        count *= size
    flat_states = states.reshape((count,) + state_shape)
    flat_controls = controls.reshape((count,) + control_shape)

    def apply(state: Array, control: Array) -> Array:
        value = jnp.asarray(transition(time, state, control, args))
        if tuple(value.shape) != state_shape:
            raise ValueError(
                "Discrete control transition must return one state_shape per case."
            )
        return value

    return jax.vmap(apply)(flat_states, flat_controls).reshape(case_shape + state_shape)


class DiscreteControlDynamics(StrictModule):
    """Explicit discrete transition applied independently over declared cases."""

    transition: DiscreteTransition
    state_shape: tuple[int, ...] = eqx.field(static=True)
    control_shape: tuple[int, ...] = eqx.field(static=True)
    dynamics_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        transition: DiscreteTransition,
        /,
        *,
        state_shape: Sequence[int],
        control_shape: Sequence[int],
        dynamics_id: str,
        method_id: str = "explicit-discrete-transition",
    ):
        if not callable(transition):
            raise TypeError("DiscreteControlDynamics transition must be callable.")
        self.transition = transition
        self.state_shape = _shape(state_shape, "state_shape")
        self.control_shape = _shape(control_shape, "control_shape")
        self.dynamics_id = _identifier(dynamics_id, "dynamics_id")
        self.method_id = _identifier(method_id, "method_id")

    def rollout(
        self,
        time_grid: ControlTimeGrid,
        initial_state: ArrayLike,
        parameterization: AbstractControlParameterization,
        coefficients: ArrayLike,
        /,
        *,
        args: Any = None,
        problem_id: str,
    ) -> ControlTrajectory:
        if not isinstance(time_grid, ControlTimeGrid):
            raise TypeError("time_grid must be a ControlTimeGrid.")
        if not isinstance(parameterization, AbstractControlParameterization):
            raise TypeError(
                "parameterization must implement AbstractControlParameterization."
            )
        if parameterization.control_shape != self.control_shape:
            raise ValueError("parameterization control_shape does not match dynamics.")
        state, cases = _case_and_state(initial_state, self.state_shape)
        initial_valid = _event_finite(state, self.state_shape)
        safe_initial = _event_where(
            initial_valid,
            state,
            jnp.zeros_like(state),
            self.state_shape,
        )

        def step(
            carry: tuple[Array, Array], time: Array
        ) -> tuple[tuple[Array, Array], tuple[Array, Array, Array]]:
            current, current_valid = carry
            safe_current = _event_where(
                current_valid,
                current,
                safe_initial,
                self.state_shape,
            )
            evaluated_control = parameterization.evaluate(
                coefficients,
                time,
                case_shape=cases,
                state=safe_current,
            )
            if tuple(evaluated_control.shape) != cases + self.control_shape:
                raise ValueError(
                    "Control parameterization returned the wrong case/control shape."
                )
            control_finite = _event_finite(evaluated_control, self.control_shape)
            transition_valid = current_valid & control_finite
            safe_control = _event_where(
                control_finite,
                evaluated_control,
                jnp.zeros_like(evaluated_control),
                self.control_shape,
            )
            candidate_state = _batched_transition(
                self.transition,
                time,
                safe_current,
                safe_control,
                args,
                cases,
                self.state_shape,
                self.control_shape,
            )
            next_state = _event_where(
                transition_valid,
                candidate_state,
                jnp.full_like(candidate_state, jnp.nan),
                self.state_shape,
            )
            control = _event_where(
                current_valid,
                evaluated_control,
                jnp.full_like(evaluated_control, jnp.nan),
                self.control_shape,
            )
            next_valid = transition_valid & _event_finite(next_state, self.state_shape)
            return (next_state, next_valid), (next_state, control, next_valid)

        (_, _), (next_states, applied_controls, next_valid) = jax.lax.scan(
            step,
            (state, initial_valid),
            time_grid.times[:-1],
        )
        state_time_axis = len(cases)
        states = jnp.concatenate(
            (
                jnp.expand_dims(state, axis=state_time_axis),
                jnp.moveaxis(next_states, 0, state_time_axis),
            ),
            axis=state_time_axis,
        )
        controls = jnp.moveaxis(applied_controls, 0, state_time_axis)
        valid = jnp.concatenate(
            (
                jnp.expand_dims(initial_valid, axis=-1),
                jnp.moveaxis(next_valid, 0, -1),
            ),
            axis=-1,
        )
        status = jnp.where(
            jnp.all(valid, axis=-1),
            CONTROL_SUCCESS,
            CONTROL_DYNAMICS_FAILED,
        ).astype(jnp.int32)
        return ControlTrajectory(
            time_grid=time_grid,
            states=states,
            controls=controls,
            valid=valid,
            status=status,
            backend_status=status,
            case_shape=cases,
            state_shape=self.state_shape,
            control_shape=self.control_shape,
            problem_id=problem_id,
            dynamics_id=self.dynamics_id,
            control_id=parameterization.parameterization_id,
            backend_id="backend:jax:lax-scan",
            method_id=self.method_id,
            discretization_id=time_grid.time_id,
            approximation_id=parameterization.approximation_id,
        )


class DifferentialControlDynamics(StrictModule):
    """Controlled vector field lowered to the canonical differential solver path."""

    vector_field: DifferentialControlVectorField
    state_shape: tuple[int, ...] = eqx.field(static=True)
    control_shape: tuple[int, ...] = eqx.field(static=True)
    dynamics_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        vector_field: DifferentialControlVectorField,
        /,
        *,
        state_shape: Sequence[int],
        control_shape: Sequence[int],
        dynamics_id: str,
        method_id: str = "canonical-differential-problem",
    ):
        if not callable(vector_field):
            raise TypeError("DifferentialControlDynamics vector_field must be callable.")
        self.vector_field = vector_field
        self.state_shape = _shape(state_shape, "state_shape")
        self.control_shape = _shape(control_shape, "control_shape")
        self.dynamics_id = _identifier(dynamics_id, "dynamics_id")
        self.method_id = _identifier(method_id, "method_id")

    def rollout(
        self,
        time_grid: ControlTimeGrid,
        initial_state: ArrayLike,
        parameterization: AbstractControlParameterization,
        coefficients: ArrayLike,
        /,
        *,
        args: Any = None,
        problem_id: str,
        solver: Any | None = None,
        stepsize_controller: Any | None = None,
        adjoint: Any | None = None,
        dt0: ArrayLike | None = None,
        event: Any | None = None,
        rtol: float = 1.0e-6,
        atol: float = 1.0e-8,
        max_steps: int | None = 4096,
        throw: bool = False,
    ) -> ControlTrajectory:
        if not isinstance(time_grid, ControlTimeGrid):
            raise TypeError("time_grid must be a ControlTimeGrid.")
        if not isinstance(parameterization, AbstractControlParameterization):
            raise TypeError(
                "parameterization must implement AbstractControlParameterization."
            )
        if parameterization.control_shape != self.control_shape:
            raise ValueError("parameterization control_shape does not match dynamics.")
        state, cases = _case_and_state(initial_state, self.state_shape)
        case_count = 1
        for size in cases:
            case_count *= size
        initial_finite = _event_finite(state, self.state_shape)
        parameter_state_template = _event_where(
            initial_finite,
            state,
            jnp.zeros_like(state),
            self.state_shape,
        ).reshape((case_count,) + self.state_shape)

        def solve_case(case_index: Array, case_state: Array):
            def controlled_field(time: Array, current: Array, field_args: Any) -> Array:
                current_finite = _event_finite(current, self.state_shape)
                safe_current = _event_where(
                    current_finite,
                    current,
                    parameter_state_template[case_index],
                    self.state_shape,
                )
                if cases:
                    parameter_states = (
                        parameter_state_template.at[case_index]
                        .set(safe_current)
                        .reshape(cases + self.state_shape)
                    )
                    all_controls = parameterization.evaluate(
                        coefficients,
                        time,
                        case_shape=cases,
                        state=parameter_states,
                    )
                    if tuple(all_controls.shape) != cases + self.control_shape:
                        raise ValueError(
                            "Control parameterization returned the wrong case/control "
                            "shape."
                        )
                    control = all_controls.reshape((case_count,) + self.control_shape)[
                        case_index
                    ]
                else:
                    control = parameterization.evaluate(
                        coefficients,
                        time,
                        case_shape=(),
                        state=safe_current,
                    )
                    if tuple(control.shape) != self.control_shape:
                        raise ValueError(
                            "Control parameterization returned the wrong case/control "
                            "shape."
                        )
                control_finite = _event_finite(control, self.control_shape)
                safe_control = _event_where(
                    control_finite,
                    control,
                    jnp.zeros_like(control),
                    self.control_shape,
                )
                candidate = jnp.asarray(
                    self.vector_field(
                        time,
                        safe_current,
                        safe_control,
                        field_args,
                    )
                )
                if tuple(candidate.shape) != self.state_shape:
                    raise ValueError(
                        "Differential control vector_field must return one "
                        "state_shape per case."
                    )
                field_valid = current_finite & control_finite
                return _event_where(
                    field_valid,
                    candidate,
                    jnp.full_like(candidate, jnp.nan),
                    self.state_shape,
                )

            differential = DifferentialProblem(
                controlled_field,
                case_state,
                t0=time_grid.t0,
                t1=time_grid.t1,
                args=args,
            )
            return solve_diffrax(
                differential,
                save_times=time_grid.times,
                solver=solver,
                stepsize_controller=stepsize_controller,
                adjoint=adjoint,
                dt0=dt0,
                event=event,
                rtol=rtol,
                atol=atol,
                dense=False,
                max_steps=max_steps,
                throw=throw,
            )

        if cases:
            flat_state = state.reshape((case_count,) + self.state_shape)
            solution = jax.vmap(solve_case)(
                jnp.arange(case_count, dtype=jnp.int32),
                flat_state,
            )
            states = solution.states.reshape(
                cases + (time_grid.num_times,) + self.state_shape
            )

            def restore_case_shape(value: Any) -> Any:
                if eqx.is_array(value) and value.shape[:1] == (case_count,):
                    return value.reshape(cases + value.shape[1:])
                return value

            backend_status = jax.tree.map(
                restore_case_shape,
                solution.backend_result,
                is_leaf=eqx.is_array,
            )
        else:
            solution = solve_case(jnp.asarray(0, dtype=jnp.int32), state)
            states = solution.states
            backend_status = solution.backend_result

        state_time_axis = len(cases)
        solution_valid = jnp.asarray(solution.valid, dtype=bool).reshape(
            cases + (time_grid.num_times,)
        )
        state_finite = _event_finite(states, self.state_shape)
        saved_control_valid = solution_valid[..., :-1] & state_finite[..., :-1]
        saved_states = jnp.take(
            states,
            jnp.arange(time_grid.num_steps),
            axis=state_time_axis,
        )
        safe_saved_states = _event_where(
            saved_control_valid,
            saved_states,
            jnp.expand_dims(
                parameter_state_template.reshape(cases + self.state_shape),
                axis=state_time_axis,
            ),
            self.state_shape,
        )
        time_first_states = jnp.moveaxis(safe_saved_states, state_time_axis, 0)
        evaluated_controls_time_first = jax.vmap(
            lambda time, current: parameterization.evaluate(
                coefficients,
                time,
                case_shape=cases,
                state=current,
            )
        )(time_grid.times[:-1], time_first_states)
        evaluated_controls = jnp.moveaxis(
            evaluated_controls_time_first, 0, state_time_axis
        )
        if (
            tuple(evaluated_controls.shape)
            != cases + (time_grid.num_steps,) + self.control_shape
        ):
            raise ValueError(
                "Control parameterization returned the wrong trajectory control shape."
            )
        controls = _event_where(
            saved_control_valid,
            evaluated_controls,
            jnp.full_like(evaluated_controls, jnp.nan),
            self.control_shape,
        )

        control_finite = _event_finite(controls, self.control_shape)
        causal_control_valid = jnp.concatenate(
            (
                jnp.ones(cases + (1,), dtype=bool),
                jnp.cumprod(control_finite.astype(jnp.int32), axis=-1).astype(bool),
            ),
            axis=-1,
        )
        valid = state_finite & causal_control_valid & solution_valid
        backend_success = jnp.asarray(
            backend_status == dfx.RESULTS.successful, dtype=bool
        )
        status = jnp.where(
            backend_success & jnp.all(valid, axis=-1),
            CONTROL_SUCCESS,
            CONTROL_DYNAMICS_FAILED,
        ).astype(jnp.int32)
        return ControlTrajectory(
            time_grid=time_grid,
            states=states,
            controls=controls,
            valid=valid,
            status=status,
            backend_status=backend_status,
            case_shape=cases,
            state_shape=self.state_shape,
            control_shape=self.control_shape,
            problem_id=problem_id,
            dynamics_id=self.dynamics_id,
            control_id=parameterization.parameterization_id,
            backend_id="backend:diffrax",
            method_id=solution.solver_id,
            discretization_id=time_grid.time_id,
            approximation_id=parameterization.approximation_id,
        )


__all__ = [
    "DifferentialControlDynamics",
    "DifferentialControlVectorField",
    "DiscreteControlDynamics",
    "DiscreteTransition",
]
