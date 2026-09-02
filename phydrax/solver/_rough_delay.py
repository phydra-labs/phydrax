#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from .._strict import StrictModule
from ..metrix import AbstractStateGeometry, EuclideanStateGeometry
from ..stochastic import AbstractRoughControl
from ._delay import (
    _distributed_delay_value,
    _initial_term_value,
    _invalid_geometry_tangent,
    _validated_geometry_point,
    _validated_geometry_tangent,
    ConstantDelay,
    DelayHistory,
    DelayHistoryWindow,
    DelayTerm,
    DelayValues,
    DistributedDelay,
    FunctionalDelay,
    StateDependentDelay,
)
from ._rough import (
    _fractional_hurst,
    _save_indices,
    AbstractRoughSolver,
    Davie,
    RoughDifferentialSolution,
    RoughEuler,
)


RoughDelayVectorFields: TypeAlias = Callable[[Array, Array, DelayValues, Any], ArrayLike]
RoughDelayDrift: TypeAlias = Callable[[Array, Array, DelayValues, Any], ArrayLike]


class _ZeroRoughDelayDrift(eqx.Module):
    def __call__(self, time, state, memory, args):
        del time, memory, args
        return jnp.zeros_like(state)


class RoughDelayDifferentialProblem(StrictModule):
    """Retarded rough equation ``dY = V₀(t,Y,Yₜ)dt + V(t,Y,Yₜ)dX``."""

    vector_fields: RoughDelayVectorFields
    drift: RoughDelayDrift
    history: DelayHistory
    delay_terms: tuple[DelayTerm, ...]
    initial_state: Array
    t0: Array
    args: Any
    geometry: AbstractStateGeometry
    state_shape: tuple[int, ...] = eqx.field(static=True)
    driver_dimension: int = eqx.field(static=True)
    has_drift: bool = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        vector_fields: RoughDelayVectorFields,
        history: DelayHistory,
        delay_terms: Sequence[DelayTerm],
        /,
        *,
        t0: ArrayLike,
        driver_dimension: int,
        drift: RoughDelayDrift | None = None,
        args: Any = None,
        geometry: AbstractStateGeometry | None = None,
        problem_id: str = "rough-delay-differential-problem",
    ):
        if not callable(vector_fields):
            raise TypeError("vector_fields must be callable.")
        if not callable(history):
            raise TypeError("history must be callable.")
        terms = tuple(delay_terms)
        allowed = (ConstantDelay, StateDependentDelay, FunctionalDelay, DistributedDelay)
        if not terms or any(not isinstance(term, allowed) for term in terms):
            raise TypeError(
                "Rough delay terms must be constant, state-dependent, functional, "
                "or distributed retarded delays."
            )
        names = tuple(term.name for term in terms)
        if len(set(names)) != len(names):
            raise ValueError("Rough delay term names must be unique.")
        dimension = int(driver_dimension)
        if dimension <= 0:
            raise ValueError("driver_dimension must be positive.")
        if drift is None:
            resolved_drift: RoughDelayDrift = _ZeroRoughDelayDrift()
        else:
            if not callable(drift):
                raise TypeError("drift must be callable or None.")
            resolved_drift = drift
        start = jnp.asarray(t0, dtype=float)
        if start.shape != ():
            raise ValueError("t0 must be scalar.")
        start = eqx.error_if(start, ~jnp.isfinite(start), "t0 must be finite.")
        state = jnp.asarray(history(start, args))
        state_shape = tuple(int(size) for size in state.shape)
        if not state_shape or any(size <= 0 for size in state_shape):
            raise ValueError("history(t0, args) must have a non-empty positive shape.")
        resolved_geometry = EuclideanStateGeometry() if geometry is None else geometry
        if not isinstance(resolved_geometry, AbstractStateGeometry):
            raise TypeError("geometry must be an AbstractStateGeometry or None.")
        state = _validated_geometry_point(
            resolved_geometry,
            state,
            "Rough delay initial state lies outside the state geometry.",
        )
        if not resolved_geometry.trivial and any(
            isinstance(term, DistributedDelay) and term.reducer is None for term in terms
        ):
            raise ValueError(
                "Non-Euclidean rough DistributedDelay terms require an explicit reducer."
            )
        initial_values = tuple(
            _initial_term_value(
                term,
                start,
                state,
                history,
                None,
                args,
                state_shape,
                resolved_geometry,
            )
            for term in terms
        )
        memory = DelayValues(names, initial_values)
        fields = jnp.asarray(vector_fields(start, state, memory, args))
        expected_fields = state_shape + (dimension,)
        if fields.shape != expected_fields:
            raise ValueError(
                f"vector_fields must return shape {expected_fields}; got {fields.shape}."
            )
        field_columns = jnp.moveaxis(fields, -1, 0)
        invalid_fields = jax.vmap(
            lambda column: _invalid_geometry_tangent(
                resolved_geometry,
                state,
                column,
                state_shape,
            )
        )(field_columns)
        fields = eqx.error_if(
            fields,
            jnp.any(invalid_fields),
            "Rough delay vector fields must be tangent-compatible with geometry.",
        )
        drift_value = jnp.asarray(resolved_drift(start, state, memory, args))
        if drift_value.shape != state_shape:
            raise ValueError("drift must preserve the rough delay state shape.")
        drift_value = _validated_geometry_tangent(
            resolved_geometry,
            state,
            drift_value,
            state_shape,
            "Rough delay drift must be tangent-compatible with geometry.",
        )
        identifier = str(problem_id)
        if not identifier:
            raise ValueError("problem_id must be non-empty.")
        self.vector_fields = vector_fields
        self.drift = resolved_drift
        self.history = history
        self.delay_terms = terms
        self.initial_state = state
        self.t0 = start
        self.args = args
        self.geometry = resolved_geometry
        self.state_shape = state_shape
        self.driver_dimension = dimension
        self.has_drift = drift is not None
        self.problem_id = identifier

    @property
    def delay_names(self) -> tuple[str, ...]:
        return tuple(term.name for term in self.delay_terms)

    @property
    def minimum_delay(self) -> Array:
        return jnp.min(
            jnp.stack(tuple(jnp.asarray(term.minimum_delay) for term in self.delay_terms))
        )

    @property
    def maximum_delay(self) -> Array | None:
        values = tuple(term.maximum_delay for term in self.delay_terms)
        if any(value is None for value in values):
            return None
        return jnp.max(jnp.stack(tuple(jnp.asarray(value) for value in values)))


class _RoughDelayHistoryView(eqx.Module):
    times: Array
    states: Array
    current_index: Array
    initial_history: DelayHistory
    args: Any
    geometry: AbstractStateGeometry
    initial_time: Array
    state_shape: tuple[int, ...] = eqx.field(static=True)

    def value(self, time: Array, /, *, left: bool = False) -> Array:
        del left
        query = jnp.asarray(time, dtype=self.times.dtype)
        if query.shape != ():
            raise ValueError("Rough delay history value queries must be scalar.")
        query = eqx.error_if(
            query,
            ~jnp.isfinite(query) | (query > self.times[self.current_index]),
            "Rough delay history queried beyond the available causal interval.",
        )

        def initial_value(item):
            value = jnp.asarray(self.initial_history(item, self.args))
            if value.shape != self.state_shape:
                raise ValueError("Rough delay prehistory changed its state shape.")
            return value

        def computed_value(item):
            left_index = jnp.searchsorted(self.times, item, side="right") - 1
            left_index = jnp.maximum(left_index, 0)
            right_index = jnp.minimum(left_index + 1, self.current_index)
            left_time = self.times[left_index]
            right_time = self.times[right_index]
            denominator = right_time - left_time
            weight = jnp.where(denominator > 0.0, (item - left_time) / denominator, 0.0)
            return self.geometry.interpolate(
                self.states[left_index],
                self.states[right_index],
                weight,
            )

        value = jax.lax.cond(
            query <= self.initial_time,
            initial_value,
            computed_value,
            query,
        )
        return _validated_geometry_point(
            self.geometry,
            value,
            "A rough delayed history value lies outside the state geometry.",
        )

    def values(self, times: ArrayLike, /, *, left: bool = False) -> Array:
        query = jnp.asarray(times)
        values = jax.lax.map(
            lambda item: self.value(item, left=left),
            query.reshape((-1,)),
        )
        return values.reshape(query.shape + self.state_shape)

    def derivative(self, time: Array, /, *, left: bool = False) -> Array:
        del time, left
        raise ValueError("Rough delay history has no classical derivative channel.")

    def derivatives(self, times: ArrayLike, /, *, left: bool = False) -> Array:
        del times, left
        raise ValueError("Rough delay history has no classical derivative channel.")


def _rough_delay_memory(
    problem: RoughDelayDifferentialProblem,
    time: Array,
    state: Array,
    history: _RoughDelayHistoryView,
    /,
) -> DelayValues:
    values = []
    for term in problem.delay_terms:
        if isinstance(term, FunctionalDelay):
            window = DelayHistoryWindow(
                history,
                time,
                term.minimum_delay,
                term.maximum_delay,
            )
            value = jnp.asarray(term.functional(time, state, window, problem.args))
        elif isinstance(term, DistributedDelay):
            delayed = history.values(time - term.nodes)
            value = _distributed_delay_value(
                term,
                time,
                state,
                delayed,
                problem.args,
                problem.state_shape,
            )
        elif isinstance(term, ConstantDelay):
            value = history.value(time - term.delay)
        elif isinstance(term, StateDependentDelay):
            value = history.value(time - term.value(time, state, problem.args))
        else:
            raise TypeError(f"Unsupported rough delay term {type(term).__name__}.")
        if value.shape != problem.state_shape:
            raise ValueError(
                f"Rough delay term {term.name!r} changed its declared state shape."
            )
        if isinstance(term, DistributedDelay) or (
            isinstance(term, FunctionalDelay) and term.output_kind == "point"
        ):
            value = _validated_geometry_point(
                problem.geometry,
                value,
                f"Rough delay term {term.name!r} returned a point outside geometry.",
            )
        if (
            isinstance(term, FunctionalDelay)
            and term.output_kind == "tangent"
            and not problem.geometry.trivial
        ):
            value = _validated_geometry_tangent(
                problem.geometry,
                state,
                value,
                problem.state_shape,
                f"Rough FunctionalDelay {term.name!r} returned a non-tangent value.",
            )
        values.append(value)
    return DelayValues(problem.delay_names, tuple(values))


def _nearest_time_index(times: Array, value: Array, /) -> Array:
    upper = jnp.searchsorted(times, value, side="left")
    lower = jnp.maximum(upper - 1, 0)
    safe_upper = jnp.minimum(upper, int(times.size) - 1)
    return jnp.where(
        jnp.abs(times[lower] - value) <= jnp.abs(times[safe_upper] - value),
        lower,
        safe_upper,
    )


def _validate_rough_delay_control(
    problem: RoughDelayDifferentialProblem,
    control: AbstractRoughControl,
    solver: AbstractRoughSolver,
    /,
) -> bool:
    if control.dimension != problem.driver_dimension:
        raise ValueError("Problem and rough control driver dimensions must match.")
    if isinstance(solver, RoughEuler):
        required_depth = 1
        minimum_hurst = 0.5
        davie = False
    elif isinstance(solver, Davie):
        required_depth = 2
        minimum_hurst = 1.0 / 3.0
        davie = True
        if any(not isinstance(term, ConstantDelay) for term in problem.delay_terms):
            raise ValueError(
                "Davie rough delay execution requires constant point delays and "
                "their delayed cross iterated integrals."
            )
    else:
        raise TypeError("Rough delay execution currently supports RoughEuler or Davie.")
    if control.depth < required_depth:
        raise ValueError(
            f"{solver.solver_name} requires control depth at least {required_depth}."
        )
    hurst = _fractional_hurst(control)
    if hurst is not None and hurst <= minimum_hurst:
        threshold = "1/2" if minimum_hurst == 0.5 else "1/3"
        raise ValueError(
            f"{solver.solver_name} requires fractional Gaussian Hurst > {threshold}."
        )
    return davie


def _rough_delay_integrate(
    problem: RoughDelayDifferentialProblem,
    control: AbstractRoughControl,
    solver: AbstractRoughSolver,
    /,
) -> tuple[Array, Array, Mapping[str, Array]]:
    davie = _validate_rough_delay_control(problem, control, solver)
    first_level = control.levels[0]
    causal_violation = (
        jnp.abs(control.times[0] - problem.t0)
        > 100.0 * jnp.finfo(control.times.dtype).eps
    ) | jnp.any(jnp.diff(control.times) > problem.minimum_delay)
    if davie:
        tolerance = (
            100.0
            * jnp.finfo(control.times.dtype).eps
            * jnp.maximum(
                1.0,
                jnp.max(jnp.abs(control.times)),
            )
        )
        for term in problem.delay_terms:
            assert isinstance(term, ConstantDelay)
            shifted_start = control.times[:-1] - term.delay
            shifted_end = control.times[1:] - term.delay
            active = shifted_start >= problem.t0 - tolerance
            straddles = (shifted_start < problem.t0 - tolerance) & (
                shifted_end > problem.t0 + tolerance
            )
            delayed_index = _nearest_time_index(control.times, shifted_start)
            safe_index = jnp.minimum(delayed_index, control.num_steps - 1)
            aligned_start = (
                jnp.abs(control.times[safe_index] - shifted_start) <= tolerance
            )
            aligned_end = (
                jnp.abs(control.times[safe_index + 1] - shifted_end) <= tolerance
            )
            causal_violation = (
                causal_violation
                | jnp.any(straddles)
                | jnp.any(active & (~aligned_start | ~aligned_end))
            )
    first_level = eqx.error_if(
        first_level,
        causal_violation,
        "Rough delay control must start at t0, each partition step must not "
        "exceed the minimum delay, and Davie delay translations must align with "
        "partition nodes.",
    )
    second_level = control.levels[1] if davie else None
    steps = jnp.diff(control.times)
    indices = jnp.arange(control.num_steps, dtype=jnp.int32)

    def one_path(first, second):
        initial_buffer = (
            jnp.zeros(
                (control.num_steps + 1,) + problem.state_shape,
                dtype=problem.initial_state.dtype,
            )
            .at[0]
            .set(problem.initial_state)
        )

        def advance(carry, item):
            state, buffer = carry
            index, time, step, first_increment, second_increment = item
            history = _RoughDelayHistoryView(
                times=control.times,
                states=buffer,
                current_index=index,
                initial_history=problem.history,
                args=problem.args,
                geometry=problem.geometry,
                initial_time=problem.t0,
                state_shape=problem.state_shape,
            )

            def fields_at(candidate):
                memory = _rough_delay_memory(problem, time, candidate, history)
                return jnp.asarray(
                    problem.vector_fields(time, candidate, memory, problem.args)
                )

            memory = _rough_delay_memory(problem, time, state, history)
            drift = jnp.asarray(problem.drift(time, state, memory, problem.args))
            fields = fields_at(state)
            first_update = jnp.tensordot(
                fields,
                first_increment,
                axes=((-1,), (0,)),
            )
            if davie:
                directions = jnp.moveaxis(fields, -1, 0)
                derivatives = jax.vmap(
                    lambda direction: jax.jvp(
                        fields_at,
                        (state,),
                        (direction,),
                    )[1]
                )(directions)
                flattened = derivatives.reshape(
                    (
                        problem.driver_dimension,
                        int(state.size),
                        problem.driver_dimension,
                    )
                )
                current_second_update = ein.contract(
                    "isj,ij->s",
                    flattened,
                    second_increment,
                ).reshape(problem.state_shape)
                delayed_second_update = jnp.zeros_like(state)
                for memory_index, term in enumerate(problem.delay_terms):
                    assert isinstance(term, ConstantDelay)
                    delayed_time = time - term.delay
                    delayed_index = _nearest_time_index(control.times, delayed_time)
                    safe_delayed_index = jnp.clip(
                        delayed_index,
                        0,
                        control.num_steps - 1,
                    )

                    def delayed_correction(delayed_step_index):
                        delayed_state = buffer[delayed_step_index]
                        delayed_history = _RoughDelayHistoryView(
                            times=control.times,
                            states=buffer,
                            current_index=delayed_step_index,
                            initial_history=problem.history,
                            args=problem.args,
                            geometry=problem.geometry,
                            initial_time=problem.t0,
                            state_shape=problem.state_shape,
                        )
                        delayed_memory = _rough_delay_memory(
                            problem,
                            control.times[delayed_step_index],
                            delayed_state,
                            delayed_history,
                        )
                        delayed_fields = jnp.asarray(
                            problem.vector_fields(
                                control.times[delayed_step_index],
                                delayed_state,
                                delayed_memory,
                                problem.args,
                            )
                        )

                        def differentiate_delayed(direction):
                            def fields_with_delayed(delayed_value):
                                replaced = tuple(
                                    delayed_value
                                    if value_index == memory_index
                                    else memory.values[value_index]
                                    for value_index in range(len(memory))
                                )
                                return jnp.asarray(
                                    problem.vector_fields(
                                        time,
                                        state,
                                        DelayValues(problem.delay_names, replaced),
                                        problem.args,
                                    )
                                )

                            return jax.jvp(
                                fields_with_delayed,
                                (memory[memory_index],),
                                (direction,),
                            )[1]

                        delayed_derivatives = jax.vmap(differentiate_delayed)(
                            jnp.moveaxis(delayed_fields, -1, 0)
                        ).reshape(
                            (
                                problem.driver_dimension,
                                int(state.size),
                                problem.driver_dimension,
                            )
                        )
                        delayed_cross_level = 0.5 * ein.contract(
                            "i,j->ij",
                            first[delayed_step_index],
                            first_increment,
                        )
                        return ein.contract(
                            "isj,ij->s",
                            delayed_derivatives,
                            delayed_cross_level,
                        ).reshape(problem.state_shape)

                    delayed_second_update = delayed_second_update + jax.lax.cond(
                        delayed_time >= problem.t0 - tolerance,
                        delayed_correction,
                        lambda delayed_step_index: jnp.zeros_like(state),
                        safe_delayed_index,
                    )
                second_update = current_second_update + delayed_second_update
            else:
                second_update = jnp.zeros_like(state)
            ambient_update = step * drift + first_update + second_update
            tangent = problem.geometry.project_tangent(state, ambient_update)
            local_update = problem.geometry.to_local(state, tangent)
            next_state = problem.geometry.retract(state, local_update)
            next_buffer = buffer.at[index + 1].set(next_state)
            return (next_state, next_buffer), next_state

        if second is None:
            second = jnp.zeros((control.num_steps, 0), dtype=first.dtype)
        (_, _), stepped = jax.lax.scan(
            advance,
            (problem.initial_state, initial_buffer),
            (indices, control.times[:-1], steps, first, second),
        )
        return jnp.concatenate((problem.initial_state[None, ...], stepped), axis=0)

    if control.sample_shape:
        path_count = int(np.prod(control.sample_shape))
        first = first_level.reshape((path_count, control.num_steps, control.dimension))
        if second_level is None:
            second = jnp.zeros((path_count, control.num_steps, 0), dtype=first.dtype)
        else:
            second = second_level.reshape(
                (
                    path_count,
                    control.num_steps,
                    control.dimension,
                    control.dimension,
                )
            )
        states = jax.vmap(one_path)(first, second).reshape(
            control.sample_shape + (control.num_steps + 1,) + problem.state_shape
        )
    else:
        states = one_path(first_level, second_level)
    interval_shape = control.sample_shape + (control.num_steps,)
    statuses = jnp.zeros(interval_shape, dtype=jnp.int32)
    statistics = {
        "num_steps": jnp.ones(interval_shape, dtype=jnp.int32),
        "num_accepted_steps": jnp.ones(interval_shape, dtype=jnp.int32),
        "num_rejected_steps": jnp.zeros(interval_shape, dtype=jnp.int32),
    }
    return states, statuses, statistics


def solve_rough_delay(
    problem: RoughDelayDifferentialProblem,
    control: AbstractRoughControl,
    /,
    *,
    save_times: ArrayLike | None = None,
    solver: AbstractRoughSolver = Davie(),
) -> RoughDifferentialSolution:
    """Integrate a retarded RDE on a causal rough-control partition."""

    if not isinstance(problem, RoughDelayDifferentialProblem):
        raise TypeError("problem must be a RoughDelayDifferentialProblem.")
    if not isinstance(control, AbstractRoughControl):
        raise TypeError("control must be an AbstractRoughControl.")
    if not isinstance(solver, AbstractRoughSolver):
        raise TypeError("solver must be an AbstractRoughSolver.")
    selected_times = control.times if save_times is None else save_times
    save_indices, saved = _save_indices(control, selected_times)
    all_states, statuses, statistics = _rough_delay_integrate(
        problem,
        control,
        solver,
    )
    states = jnp.take(all_states, save_indices, axis=len(control.sample_shape))
    state_axes = tuple(range(len(control.sample_shape) + 1, states.ndim))
    valid = jnp.all(jnp.isfinite(states), axis=state_axes)
    return RoughDifferentialSolution(
        times=saved,
        states=states,
        valid=valid,
        statuses=statuses,
        control=control,
        solver=solver,
        state_shape=problem.state_shape,
        state_geometry_id=problem.geometry.geometry_id,
        statistics=statistics,
        metadata={
            "problem_id": problem.problem_id,
            "equation_kind": "rough-retarded",
            "num_intervals": control.num_steps,
            "driver_family": "geometric-rough-path",
            "driver_dimension": control.dimension,
            "control_depth": control.depth,
            "state_geometry_id": problem.geometry.geometry_id,
            "delay_names": problem.delay_names,
            "delay_term_types": tuple(
                type(term).__name__ for term in problem.delay_terms
            ),
            "minimum_delay": problem.minimum_delay,
            "infinite_memory": problem.maximum_delay is None,
            "maximum_delay": problem.maximum_delay,
            "history_interpolation": "retraction-linear",
            "delayed_second_level": (
                "grid-aligned-piecewise-linear-cross-integrals"
                if isinstance(solver, Davie)
                else "not-required-young"
            ),
            "prehistory_enhancement": "deterministic-zero-gubinelli-derivative",
        },
    )


__all__ = [
    "RoughDelayDifferentialProblem",
    "RoughDelayDrift",
    "RoughDelayVectorFields",
    "solve_rough_delay",
]
