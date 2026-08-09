#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any, cast

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import core as jax_core
from jaxtyping import Array, ArrayLike

from .._frozendict import frozendict
from ..stochastic import WienerRealization
from ._delay import (
    ConstantDelay,
    DelayDifferentialProblem,
    DerivativeDelay,
    DistributedDelay,
    FunctionalDelay,
    NeutralDelayProblem,
)
from ._delay_adjoint import CheckpointedDelayAdjoint, SegmentedDelayAdjoint
from ._delay_discontinuity import (
    constant_discontinuity_schedule,
    DynamicControllerState,
    StateDependentAdaptiveController,
    StateDependentDiscontinuityTracker,
    StateDependentFixedController,
)
from ._delay_history import EmptyDelayHistory, RollingDelayHistory
from ._delay_plan import (
    compile_delay_execution_plan,
    fixed_delay_history_capacity,
    resolve_delay_solver,
)
from ._diffrax_backend import (
    _realized_wiener_path,
    _validated_realization_interval,
)
from ._diffrax_delay_backend import (
    _bind_delay_history,
    _CausalAdaptiveStepSizeController,
    _CausalFixedStepSizeController,
    _DelayVectorField,
    _deterministic_delay_terms,
    _neutral_discontinuity_times,
    _neutral_vector_field,
    _NeutralInnerState,
    _resolved_discontinuity_depth,
    _underlying_neutral_vector_field,
    solve_diffrax_delay,
)
from ._diffrax_delay_stochastic import (
    _brownian_dense_info,
    _DelayDiffusionVectorField,
    _PathConsistentInterpolationFactory,
    _underlying_control_term,
    _validated_solver as _validated_stochastic_delay_solver,
    _validation_contract,
)
from ._geometric import AbstractGeometricSolver, SRKMK
from ._memory import MemoryEquationSolution
from ._save_schedule import validate_save_times


class SegmentedDelayResult(str, Enum):
    """Host-level terminal status for a segmented delay solve."""

    successful = "successful"
    event_occurred = "event_occurred"
    segment_limit_reached = "segment_limit_reached"
    history_capacity_exhausted = "history_capacity_exhausted"
    solver_failure = "solver_failure"


class _RollingRetardedSolverState(eqx.Module):
    inner_state: Any
    history: RollingDelayHistory
    error_template: Any
    dense_info_template: Any


class _RollingRetardedSolver(dfx.AbstractWrappedSolver):
    """Diffrax wrapper retaining only the declared live lag window."""

    solver: dfx.AbstractSolver  # ty: ignore[invalid-attribute-override]
    history_capacity: int = eqx.field(static=True)
    maximum_lag: Array

    @property
    def term_structure(self):  # ty: ignore[invalid-attribute-override]
        return self.solver.term_structure

    @property
    def interpolation_cls(self):  # ty: ignore[invalid-attribute-override]
        return self.solver.interpolation_cls

    @property
    def term_compatible_contr_kwargs(self):  # ty: ignore[invalid-attribute-override]
        return self.solver.term_compatible_contr_kwargs

    @property
    def root_finder(self):
        return cast(Any, self.solver).root_finder

    @property
    def root_find_max_steps(self):
        return cast(Any, self.solver).root_find_max_steps

    @property
    def scan_kind(self):
        return cast(Any, self.solver).scan_kind

    def order(self, terms):
        return self.solver.order(terms)

    def strong_order(self, terms):
        return self.solver.strong_order(terms)

    def error_order(self, terms):
        return self.solver.error_order(terms)

    def func(self, terms, t0, y0, args):
        return self.solver.func(terms, t0, y0, args)

    def init(self, terms, t0, t1, y0, args):
        provisional_state = self.solver.init(terms, t0, t1, y0, args)
        _, error_structure, dense_info_structure, _, _ = eqx.filter_eval_shape(
            self.solver.step,
            terms,
            t0,
            t1,
            y0,
            args,
            provisional_state,
            False,
        )

        def zero(value):
            if isinstance(value, jax.ShapeDtypeStruct):
                return jnp.zeros(value.shape, value.dtype)
            return value

        history = RollingDelayHistory.allocate(
            time=t0,
            dense_info_structure=dense_info_structure,
            capacity=self.history_capacity,
            interpolation_cls=self.solver.interpolation_cls,
            maximum_lag=self.maximum_lag,
        )
        bound_terms = _bind_delay_history(terms, history)
        inner_state = self.solver.init(bound_terms, t0, t1, y0, args)
        return _RollingRetardedSolverState(
            inner_state=inner_state,
            history=history,
            error_template=jax.tree.map(zero, error_structure),
            dense_info_template=jax.tree.map(zero, dense_info_structure),
        )

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        def exhausted(_):
            return (
                y0,
                solver_state.error_template,
                solver_state.dense_info_template,
                solver_state,
                dfx.RESULTS.internal_error,
            )

        def advance(_):
            bound_terms = _bind_delay_history(terms, solver_state.history)
            y1, y_error, dense_info, inner_state, result = self.solver.step(
                bound_terms,
                t0,
                t1,
                y0,
                args,
                solver_state.inner_state,
                made_jump,
            )
            history = solver_state.history.append(t0, t1, dense_info)
            state = _RollingRetardedSolverState(
                inner_state=inner_state,
                history=history,
                error_template=solver_state.error_template,
                dense_info_template=solver_state.dense_info_template,
            )
            return y1, y_error, dense_info, state, result

        return jax.lax.cond(solver_state.history.overflowed, exhausted, advance, None)


class _RollingNeutralRetardedSolver(_RollingRetardedSolver):
    """Rolling transformed neutral solver returning physical states."""

    initial_transformed_state: Array

    @property
    def interpolation_cls(self):  # ty: ignore[invalid-attribute-override]
        return dfx.LocalLinearInterpolation

    def init(self, terms, t0, t1, y0, args):
        dense_info = {"y0": y0, "y1": y0}
        history = RollingDelayHistory.allocate(
            time=t0,
            dense_info_structure=dense_info,
            capacity=self.history_capacity,
            interpolation_cls=self.interpolation_cls,
            maximum_lag=self.maximum_lag,
        )
        bound_terms = _bind_delay_history(terms, history)
        solver_state = self.solver.init(
            bound_terms,
            t0,
            t1,
            self.initial_transformed_state,
            args,
        )
        inner = _NeutralInnerState(
            solver_state=solver_state,
            transformed_state=self.initial_transformed_state,
        )
        return _RollingRetardedSolverState(
            inner_state=inner,
            history=history,
            error_template=None,
            dense_info_template=dense_info,
        )

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        def exhausted(_):
            return (
                y0,
                solver_state.error_template,
                solver_state.dense_info_template,
                solver_state,
                dfx.RESULTS.internal_error,
            )

        def advance(_):
            bound_terms = _bind_delay_history(terms, solver_state.history)
            inner = solver_state.inner_state
            transformed, y_error, _, next_solver_state, result = self.solver.step(
                bound_terms,
                t0,
                t1,
                inner.transformed_state,
                args,
                inner.solver_state,
                made_jump,
            )
            vector_field = _underlying_neutral_vector_field(bound_terms)
            y1 = vector_field.recovery.recover(t1, transformed, args)
            dense_info = {"y0": y0, "y1": y1}
            history = solver_state.history.append(t0, t1, dense_info)
            state = _RollingRetardedSolverState(
                inner_state=_NeutralInnerState(
                    solver_state=next_solver_state,
                    transformed_state=transformed,
                ),
                history=history,
                error_template=solver_state.error_template,
                dense_info_template=solver_state.dense_info_template,
            )
            return y1, y_error, dense_info, state, result

        return jax.lax.cond(solver_state.history.overflowed, exhausted, advance, None)


class _RollingStochasticRetardedSolver(_RollingRetardedSolver):
    """Rolling accepted history with the certified Wiener-path local extension."""

    heun: bool = eqx.field(static=True)
    path_key: Array
    path_shape: Any = eqx.field(static=True)
    path_levy_area: type = eqx.field(static=True)
    path_key_impl: Any = eqx.field(static=True)
    geometry: Any

    @property
    def interpolation_cls(self):  # ty: ignore[invalid-attribute-override]
        return _PathConsistentInterpolationFactory(
            path_shape=self.path_shape,
            path_levy_area=self.path_levy_area,
            path_key_impl=self.path_key_impl,
            heun=self.heun,
            geometry=self.geometry,
        )

    def _step_with_extension(self, terms, t0, t1, y0, args, inner_state, made_jump):
        y1, y_error, _, next_state, result = self.solver.step(
            terms,
            t0,
            t1,
            y0,
            args,
            inner_state,
            made_jump,
        )
        drift_term, wrapped_diffusion_term = terms.terms
        diffusion_term = _underlying_control_term(wrapped_diffusion_term)
        vector_field = diffusion_term.vector_field
        if not isinstance(vector_field, _DelayDiffusionVectorField):
            raise TypeError("Certified stochastic delay terms require delayed diffusion.")
        dense_info = {
            "y0": y0,
            "drift": jnp.asarray(drift_term.vf(t0, y0, args)),
            "diffusion": vector_field.freeze(t0, y0, args),
            **_brownian_dense_info(diffusion_term.control, self.path_key),
        }
        if self.geometry is not None:
            dense_info["y1"] = y1
        return y1, y_error, dense_info, next_state, result

    def init(self, terms, t0, t1, y0, args):
        provisional_state = self.solver.init(terms, t0, t1, y0, args)
        _, error_structure, dense_info_structure, _, _ = eqx.filter_eval_shape(
            self._step_with_extension,
            terms,
            t0,
            t1,
            y0,
            args,
            provisional_state,
            False,
        )

        def zero(value):
            if isinstance(value, jax.ShapeDtypeStruct):
                return jnp.zeros(value.shape, value.dtype)
            return value

        history = RollingDelayHistory.allocate(
            time=t0,
            dense_info_structure=dense_info_structure,
            capacity=self.history_capacity,
            interpolation_cls=self.interpolation_cls,
            maximum_lag=self.maximum_lag,
        )
        bound_terms = _bind_delay_history(terms, history)
        inner_state = self.solver.init(bound_terms, t0, t1, y0, args)
        return _RollingRetardedSolverState(
            inner_state=inner_state,
            history=history,
            error_template=jax.tree.map(zero, error_structure),
            dense_info_template=jax.tree.map(zero, dense_info_structure),
        )

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        def exhausted(_):
            return (
                y0,
                solver_state.error_template,
                solver_state.dense_info_template,
                solver_state,
                dfx.RESULTS.internal_error,
            )

        def advance(_):
            bound_terms = _bind_delay_history(terms, solver_state.history)
            y1, y_error, dense_info, inner_state, result = self._step_with_extension(
                bound_terms,
                t0,
                t1,
                y0,
                args,
                solver_state.inner_state,
                made_jump,
            )
            history = solver_state.history.append(t0, t1, dense_info)
            state = _RollingRetardedSolverState(
                inner_state=inner_state,
                history=history,
                error_template=solver_state.error_template,
                dense_info_template=solver_state.dense_info_template,
            )
            return y1, y_error, dense_info, state, result

        return jax.lax.cond(solver_state.history.overflowed, exhausted, advance, None)


class _ItoRollingStochasticRetardedSolver(
    _RollingStochasticRetardedSolver,
    dfx.AbstractItoSolver,
):
    pass


class _StratonovichRollingStochasticRetardedSolver(
    _RollingStochasticRetardedSolver,
    dfx.AbstractStratonovichSolver,
):
    pass


class _RestartableAdaptiveController(dfx.AbstractAdaptiveStepSizeController):
    """Adaptive wrapper retaining the exact next proposal across host windows."""

    controller: dfx.AbstractAdaptiveStepSizeController

    def __init__(self, controller: dfx.AbstractAdaptiveStepSizeController):
        self.controller = controller

    @property
    def rtol(self):
        return self.controller.rtol

    @property
    def atol(self):
        return self.controller.atol

    @property
    def norm(self):
        return self.controller.norm

    def wrap(self, direction):
        return _RestartableAdaptiveController(
            cast(dfx.AbstractAdaptiveStepSizeController, self.controller.wrap(direction))
        )

    def init(self, terms, t0, t1, y0, dt0, args, func, error_order):
        next_t1, inner_state = self.controller.init(
            terms, t0, t1, y0, dt0, args, func, error_order
        )
        return next_t1, (inner_state, next_t1 - t0)

    def adapt_step_size(
        self,
        t0,
        t1,
        y0,
        y1_candidate,
        args,
        y_error,
        error_order,
        controller_state,
    ):
        inner_state, _ = controller_state
        keep, next_t0, next_t1, made_jump, next_inner, result = (
            self.controller.adapt_step_size(
                t0,
                t1,
                y0,
                y1_candidate,
                args,
                y_error,
                error_order,
                inner_state,
            )
        )
        return (
            keep,
            next_t0,
            next_t1,
            made_jump,
            (next_inner, next_t1 - next_t0),
            result,
        )


class _RestartableFixedController(dfx.AbstractStepSizeController):
    """Fixed wrapper retaining the exact jump-clipped proposal across windows."""

    controller: dfx.AbstractStepSizeController

    def __init__(self, controller: dfx.AbstractStepSizeController):
        self.controller = controller

    def wrap(self, direction):
        return _RestartableFixedController(self.controller.wrap(direction))

    def init(self, terms, t0, t1, y0, dt0, args, func, error_order):
        next_t1, inner_state = self.controller.init(
            terms, t0, t1, y0, dt0, args, func, error_order
        )
        return next_t1, (inner_state, next_t1 - t0)

    def adapt_step_size(
        self,
        t0,
        t1,
        y0,
        y1_candidate,
        args,
        y_error,
        error_order,
        controller_state,
    ):
        inner_state, _ = controller_state
        keep, next_t0, next_t1, made_jump, next_inner, result = (
            self.controller.adapt_step_size(
                t0,
                t1,
                y0,
                y1_candidate,
                args,
                y_error,
                error_order,
                inner_state,
            )
        )
        return (
            keep,
            next_t0,
            next_t1,
            made_jump,
            (next_inner, next_t1 - next_t0),
            result,
        )


class DelaySegmentContinuation(eqx.Module):
    """Complete restart state at one accepted segmented-solve boundary."""

    time: Array
    state: Array
    solver_state: _RollingRetardedSolverState
    controller_state: Any
    made_jump: Array
    realization: WienerRealization | None
    stats: frozendict[str, Any]
    event_state: Any
    discontinuity_tracker: Any
    problem_id: str = eqx.field(static=True)
    solver_name: str = eqx.field(static=True)
    controller_mode: str = eqx.field(static=True)
    resumable: bool = eqx.field(static=True)

    @property
    def active_history(self) -> RollingDelayHistory:
        return self.solver_state.history


class DelaySegmentArchive(eqx.Module):
    """Host archive of finite compiled-segment interpolations."""

    starts: np.ndarray
    ends: np.ndarray
    interpolations: tuple[Any, ...]

    def _scalar(self, time: Array, left: bool, derivative: bool) -> Array:
        side = "left" if left else "right"
        index = jnp.searchsorted(self.ends, time, side=side)
        index = jnp.clip(index, 0, len(self.interpolations) - 1)
        first_query = jnp.clip(time, self.starts[0], self.ends[0])
        value = (
            self.interpolations[0].derivative(first_query, left=left)
            if derivative
            else self.interpolations[0].evaluate(first_query, left=left)
        )
        for position, interpolation in enumerate(self.interpolations[1:], start=1):
            query = jnp.clip(time, self.starts[position], self.ends[position])
            candidate = (
                interpolation.derivative(query, left=left)
                if derivative
                else interpolation.evaluate(query, left=left)
            )
            predicate = index == position
            predicate = jnp.reshape(
                predicate, predicate.shape + (1,) * (candidate.ndim - predicate.ndim)
            )
            value = jnp.where(predicate, candidate, value)
        return value

    def _validated_query(self, query_times: ArrayLike, /) -> Array:
        query = jnp.asarray(query_times)
        if jnp.iscomplexobj(query):
            raise TypeError("Segmented dense query times must be real-valued.")
        if query.size == 0:
            raise ValueError("Segmented dense query times must be non-empty.")
        query = query.astype(float)
        query = eqx.error_if(
            query,
            ~jnp.all(jnp.isfinite(query)),
            "Segmented dense query times must be finite.",
        )
        return eqx.error_if(
            query,
            jnp.any((query < self.starts[0]) | (query > self.ends[-1])),
            "Segmented dense query lies outside the archived solve interval.",
        )

    @eqx.filter_jit
    def evaluate(self, query_times: ArrayLike, /, *, left: bool = True) -> Array:
        if not isinstance(left, bool):
            raise TypeError("left must be a bool.")
        query = self._validated_query(query_times)
        values = jax.vmap(lambda time: self._scalar(time, left, False))(
            query.reshape((-1,))
        )
        return values.reshape(query.shape + values.shape[1:])

    @eqx.filter_jit
    def derivative(self, query_times: ArrayLike, /, *, left: bool = True) -> Array:
        if not isinstance(left, bool):
            raise TypeError("left must be a bool.")
        query = self._validated_query(query_times)
        values = jax.vmap(lambda time: self._scalar(time, left, True))(
            query.reshape((-1,))
        )
        return values.reshape(query.shape + values.shape[1:])




def _host_bool(value: Any, /) -> bool:
    return bool(np.asarray(jax.device_get(value)))


def _host_scalar(value: Any, /) -> float:
    return float(np.asarray(jax.device_get(value)))


def _accumulated_stats(
    previous: Mapping[str, Any], current: Mapping[str, Any], /
) -> frozendict[str, Any]:
    return frozendict(
        {
            "num_steps": previous.get("num_steps", 0) + current["num_steps"],
            "num_accepted_steps": previous.get("num_accepted_steps", 0)
            + current["num_accepted_steps"],
            "num_rejected_steps": previous.get("num_rejected_steps", 0)
            + current["num_rejected_steps"],
            "num_segments": previous.get("num_segments", 0) + 1,
        }
    )


def _discontinuity_schedule(
    problem: DelayDifferentialProblem | NeutralDelayProblem,
    solver: dfx.AbstractSolver,
    terms: Any,
    initial_discontinuities: ArrayLike | Sequence[float] | None,
    discontinuity_depth: int | None,
    max_discontinuities: int,
    /,
) -> tuple[Array, Array, Array, int]:
    if initial_discontinuities is None:
        declared_sources = (
            jnp.empty((0,), dtype=problem.t0.dtype)
            if problem.neutral
            else problem.t0.reshape((1,))
        )
    else:
        declared_sources = jnp.asarray(
            initial_discontinuities,
            dtype=problem.t0.dtype,
        )
        if declared_sources.ndim != 1:
            raise ValueError("initial_discontinuities must be a rank-1 array or None.")
        if not _host_bool(jnp.all(jnp.isfinite(declared_sources))):
            raise ValueError("initial_discontinuities must be finite.")
    lags = []
    for term in problem.delay_terms:
        if isinstance(term, ConstantDelay):
            lags.append(term.delay)
        elif isinstance(term, DistributedDelay):
            lags.extend(tuple(term.nodes))
        elif isinstance(term, FunctionalDelay):
            lags.extend(tuple(term.discontinuity_lags))
        elif isinstance(term, DerivativeDelay) and isinstance(term.delay, ConstantDelay):
            lags.append(term.delay.delay)
    schedule_lags = (
        jnp.stack(tuple(lags)) if lags else jnp.empty((0,), dtype=problem.t0.dtype)
    )
    if problem.neutral:
        compatibility_source = jnp.where(
            problem.initial_derivative_compatible,
            jnp.asarray(jnp.inf, dtype=problem.t0.dtype),
            problem.t0,
        ).reshape((1,))
        sources = jnp.concatenate((declared_sources, compatibility_source))
        schedule = _neutral_discontinuity_times(
            schedule_lags,
            sources,
            problem.t1,
            max_discontinuities=max_discontinuities,
        )
        tracker_times = sources
        tracker_generations = jnp.zeros(sources.shape, dtype=jnp.int32)
        depth = max_discontinuities
    else:
        depth = _resolved_discontinuity_depth(
            solver,
            terms,
            discontinuity_depth,
        )
        schedule, generations = constant_discontinuity_schedule(
            schedule_lags,
            declared_sources,
            depth=depth,
            max_discontinuities=max_discontinuities,
        )
        tracker_times = schedule
        tracker_generations = generations
    controller_schedule = jnp.sort(
        jnp.where(
            (schedule > problem.t0) & (schedule < problem.t1),
            schedule,
            jnp.asarray(jnp.inf, dtype=problem.t0.dtype),
        )
    )
    return controller_schedule, tracker_times, tracker_generations, depth


def solve_diffrax_delay_segmented(
    problem: DelayDifferentialProblem | NeutralDelayProblem,
    /,
    *,
    save_times: ArrayLike,
    solver: Any | None = None,
    stepsize_controller: Any | None = None,
    adjoint: Any | None = None,
    dt0: ArrayLike | None = None,
    event: Any | None = None,
    rtol: float = 1e-6,
    atol: float = 1e-8,
    dense: bool = False,
    history_capacity: int | None = None,
    history_margin: int = 2,
    max_steps_per_segment: int = 256,
    max_segments: int | None = None,
    continuation: DelaySegmentContinuation | None = None,
    realization: WienerRealization | None = None,
    initial_discontinuities: ArrayLike | Sequence[float] | None = None,
    discontinuity_depth: int | None = None,
    max_discontinuities: int = 8192,
    root_rtol: float = 1e-10,
    root_atol: float = 1e-12,
    max_root_iterations: int = 64,
    throw: bool = False,
) -> MemoryEquationSolution:
    """Run bounded compiled windows while preserving exact accepted solver state.

    The number of windows depends on runtime acceptance and event decisions, so this
    host driver deliberately rejects whole-solve JIT. Each finite Diffrax window is
    still compiled. Dense output archives windows on the host; without ``dense=True``
    no queries outside the returned saved values are supported.
    """
    if not isinstance(problem, (DelayDifferentialProblem, NeutralDelayProblem)):
        raise TypeError(
            "solve_diffrax_delay_segmented requires a delay differential problem."
        )
    if isinstance(adjoint, SegmentedDelayAdjoint):
        if max_segments is None:
            max_segments = adjoint.max_segments
        elif max_segments != adjoint.max_segments:
            raise ValueError(
                "max_segments must match SegmentedDelayAdjoint.max_segments."
            )
    host_inputs = (problem, save_times, realization, dt0)
    traced = any(
        isinstance(leaf, jax_core.Tracer) for leaf in jax.tree.leaves(host_inputs)
    )
    if traced:
        if not isinstance(adjoint, SegmentedDelayAdjoint):
            raise TypeError(
                "The segmented host driver cannot be transformed directly; "
                "differentiating solve_diffrax_delay_segmented requires "
                "SegmentedDelayAdjoint."
            )
        if continuation is not None:
            raise ValueError(
                "Segmented adjoint replay does not accept a continuation state."
            )
        if dense:
            raise ValueError("Segmented adjoint replay does not support dense output.")
        replay = solve_diffrax_delay(
            problem,
            save_times=save_times,
            solver=solver,
            stepsize_controller=stepsize_controller,
            adjoint=adjoint,
            dt0=dt0,
            event=event,
            rtol=rtol,
            atol=atol,
            dense=False,
            history_mode="rolling",
            history_capacity=(
                history_capacity
                if history_capacity is not None
                else max_steps_per_segment * adjoint.max_segments
                + history_margin
                + 2
            ),
            history_margin=history_margin,
            max_steps=max_steps_per_segment * adjoint.max_segments,
            realization=realization,
            initial_discontinuities=initial_discontinuities,
            discontinuity_depth=discontinuity_depth,
            max_discontinuities=max_discontinuities,
            root_rtol=root_rtol,
            root_atol=root_atol,
            max_root_iterations=max_root_iterations,
            throw=throw,
        )
        return MemoryEquationSolution(
            times=replay.times,
            states=replay.states,
            valid=replay.valid,
            interpolation=None,
            backend_result=replay.backend_result,
            stats={
                **dict(replay.stats),
                "execution_mode": "segmented-adjoint-replay",
                "max_steps_per_segment": max_steps_per_segment,
                "max_segments": adjoint.max_segments,
            },
            event_mask=replay.event_mask,
            realization=replay.realization,
            state_shape=replay.state_shape,
            solver_name=replay.solver_name,
            solver_id=f"{replay.solver_id}:segmented-adjoint-replay",
            resolved_method=(
                f"{replay.resolved_method}:segmented-adjoint-replay"
            ),
            metadata={
                **dict(replay.metadata),
                "execution_mode": "segmented-adjoint-replay",
                "adjoint": "SegmentedDelayAdjoint",
            },
        )
    if not isinstance(dense, bool) or not isinstance(throw, bool):
        raise TypeError("dense and throw must be bool values.")
    if (
        not isinstance(max_steps_per_segment, int)
        or isinstance(max_steps_per_segment, bool)
        or max_steps_per_segment <= 0
    ):
        raise ValueError("max_steps_per_segment must be a positive integer.")
    if max_segments is not None and (
        not isinstance(max_segments, int)
        or isinstance(max_segments, bool)
        or max_segments <= 0
    ):
        raise ValueError("max_segments must be a positive integer or None.")
    if (
        not isinstance(max_discontinuities, int)
        or isinstance(max_discontinuities, bool)
        or max_discontinuities <= 0
    ):
        raise ValueError("max_discontinuities must be a positive integer.")
    if root_rtol < 0.0 or root_atol <= 0.0:
        raise ValueError("root_rtol must be nonnegative and root_atol positive.")
    if (
        not isinstance(max_root_iterations, int)
        or isinstance(max_root_iterations, bool)
        or max_root_iterations <= 0
    ):
        raise ValueError("max_root_iterations must be a positive integer.")
    if (
        not isinstance(history_margin, int)
        or isinstance(history_margin, bool)
        or history_margin < 1
    ):
        raise ValueError("history_margin must be a positive integer.")
    maximum_lag = problem.maximum_delay
    if maximum_lag is None:
        raise ValueError(
            "Segmented delay solves require every delay term to declare a finite maximum lag."
        )
    maximum_lag_value = _host_scalar(maximum_lag)
    if not np.isfinite(maximum_lag_value) or maximum_lag_value <= 0.0:
        raise ValueError("The declared maximum delay must be finite and positive.")
    if problem.neutral and discontinuity_depth is not None:
        raise ValueError(
            "Neutral delay solves propagate discontinuities through the full horizon; "
            "discontinuity_depth must be None."
        )
    stochastic = problem.stochastic
    validated_realization: WienerRealization | None = None
    solve_start = problem.t0
    solve_end = problem.t1
    if stochastic:
        assert isinstance(problem, DelayDifferentialProblem)
        if realization is None:
            raise ValueError("Stochastic segmented delay requires a WienerRealization.")
        if not isinstance(realization, WienerRealization):
            raise TypeError("realization must be a WienerRealization or None.")
        validated_realization = realization
        if realization.sample_shape:
            raise ValueError(
                "Segmented stochastic delay currently requires a scalar realization."
            )
        if dt0 is None:
            raise ValueError("Stochastic segmented delay requires an explicit dt0.")
        if problem.neutral:
            raise ValueError("Stochastic neutral delay terms are not supported.")
        if discontinuity_depth not in (None, 1):
            raise ValueError(
                "Fixed-step stochastic segmented delay requires "
                "discontinuity_depth=1 or None."
            )
        selected_solver = resolve_delay_solver(problem, solver)
        heun = problem.interpretation == "stratonovich"
        solve_start, solve_end = _validated_realization_interval(
            _validation_contract(problem), validated_realization
        )
        if _host_scalar(jnp.abs(jnp.asarray(dt0))) <= _host_scalar(
            validated_realization.tolerance
        ):
            raise ValueError(
                "WienerRealization tolerance must be strictly smaller than dt0."
            )
    else:
        if realization is not None:
            raise ValueError("realization is only valid for stochastic delay problems.")
        if isinstance(problem, NeutralDelayProblem):
            selected_solver = dfx.Euler() if solver is None else solver
            if type(selected_solver) is not dfx.Euler:
                raise ValueError(
                    "NeutralDelayProblem execution currently requires diffrax.Euler."
                )
        else:
            selected_solver = resolve_delay_solver(problem, solver)
        heun = False
    execution_plan = compile_delay_execution_plan(
        problem,
        selected_solver,
        execution="segmented",
        history_mode="rolling",
    )
    if stochastic:
        assert isinstance(problem, DelayDifferentialProblem)
        assert validated_realization is not None
        selected_solver, heun = _validated_stochastic_delay_solver(
            problem, selected_solver, validated_realization
        )
    selected_adjoint = CheckpointedDelayAdjoint() if adjoint is None else adjoint
    if isinstance(selected_adjoint, dfx.BacksolveAdjoint):
        raise ValueError("BacksolveAdjoint is not supported for delay equations.")

    real_dtype = jnp.asarray(problem.initial_state).real.dtype
    history_context = _DelayVectorField(
        function=problem.drift,
        initial_history=problem.history,
        initial_derivative=problem.history_derivative,
        delay_terms=problem.delay_terms,
        initial_time=solve_start,
        state_shape=problem.state_shape,
        geometry=problem.state_geometry,
        computed_history=EmptyDelayHistory(
            problem.initial_state,
            (
                jnp.zeros_like(problem.initial_state)
                if stochastic
                else problem.initial_right_derivative
                if problem.neutral
                else None
            ),
        ),
    )
    if stochastic:
        assert validated_realization is not None
        brownian, path_sign = _realized_wiener_path(
            validated_realization,
            validated_realization.path_keys,
            validated_realization.path_signs,
            real_dtype,
        )
        diffusion_field = _DelayDiffusionVectorField(
            context=history_context,
            terms=problem.wiener_terms,
            path_sign=path_sign,
            state_shape=problem.state_shape,
        )
        terms = dfx.MultiTerm(
            dfx.ODETerm(history_context),
            dfx.ControlTerm(diffusion_field, brownian),
        )
    elif isinstance(problem, NeutralDelayProblem):
        vector_field = _neutral_vector_field(problem, history_context)
        terms = dfx.ODETerm(vector_field)
    else:
        vector_field = history_context
        terms = _deterministic_delay_terms(selected_solver, vector_field)
    (
        discontinuities,
        tracker_initial_times,
        tracker_initial_generations,
        depth,
    ) = _discontinuity_schedule(
        problem,
        selected_solver,
        terms,
        initial_discontinuities,
        discontinuity_depth,
        max_discontinuities,
    )
    stage_time_extent = execution_plan.stage_time_extent
    maximum_step = execution_plan.minimum_delay / stage_time_extent

    if stepsize_controller is None:
        base_controller = (
            dfx.PIDController(rtol=float(rtol), atol=float(atol))
            if isinstance(selected_solver, dfx.AbstractAdaptiveSolver)
            else dfx.ConstantStepSize()
        )
    else:
        base_controller = stepsize_controller
    if not isinstance(base_controller, dfx.AbstractStepSizeController):
        raise TypeError("stepsize_controller must be a Diffrax controller.")
    if isinstance(selected_solver, AbstractGeometricSolver):
        if dt0 is None:
            raise ValueError("Geometric segmented delay solvers require an explicit dt0.")
        if not isinstance(base_controller, dfx.ConstantStepSize):
            raise ValueError(
                "Geometric segmented delay solvers require diffrax.ConstantStepSize."
            )
    if isinstance(base_controller, dfx.AbstractAdaptiveStepSizeController):
        if not isinstance(selected_solver, dfx.AbstractAdaptiveSolver):
            raise ValueError("Adaptive controllers require an adaptive Diffrax solver.")
        if history_capacity is None:
            raise ValueError(
                "Adaptive segmented delay solves require an explicit history_capacity per window."
            )
        clipped = (
            base_controller
            if int(discontinuities.size) == 0
            else dfx.ClipStepSizeController(
                base_controller, jump_ts=jax.lax.stop_gradient(discontinuities)
            )
        )
        controller = _CausalAdaptiveStepSizeController(clipped, maximum_step)
        controller_mode = "adaptive"
        resolved_capacity = history_capacity
    else:
        if not isinstance(base_controller, dfx.ConstantStepSize):
            raise ValueError("Fixed segmented delay solves require ConstantStepSize.")
        if dt0 is None:
            raise ValueError("Fixed segmented delay solves require dt0.")
        required_capacity = fixed_delay_history_capacity(
            maximum_lag,
            dt0,
            margin=history_margin,
            breakpoints=discontinuities,
            initial_time=problem.t0,
        )
        if history_capacity is not None and history_capacity < required_capacity:
            raise ValueError(
                "history_capacity is below the exact fixed-step lag-window bound "
                f"({required_capacity})."
            )
        resolved_capacity = (
            required_capacity if history_capacity is None else history_capacity
        )
        controller = _CausalFixedStepSizeController(
            maximum_step=maximum_step,
            jump_ts=jax.lax.stop_gradient(discontinuities),
        )
        controller_mode = "fixed"
    if (
        not isinstance(resolved_capacity, int)
        or isinstance(resolved_capacity, bool)
        or resolved_capacity <= 0
    ):
        raise ValueError("history_capacity must be a positive integer.")
    dynamic_tracker = None
    tracking_mode = "constant-additive-fast-path"
    dynamic_delays = execution_plan.state_dependent_delays
    if dynamic_delays and stochastic:
        tracking_mode = "first-order-pathwise-untracked"
    elif dynamic_delays:
        if depth > 0:
            constant_lags = (
                jnp.stack(execution_plan.constant_lags)
                if execution_plan.constant_lags
                else jnp.empty((0,), dtype=problem.t0.dtype)
            )
            dynamic_tracker = StateDependentDiscontinuityTracker(
                dynamic_delays,
                constant_lags,
                tracker_initial_times,
                tracker_initial_generations,
                problem.t1,
                depth=depth,
                capacity=max_discontinuities,
                root_rtol=root_rtol,
                root_atol=root_atol,
                max_root_iterations=max_root_iterations,
            )
            if isinstance(controller, dfx.AbstractAdaptiveStepSizeController):
                controller = StateDependentAdaptiveController(controller, dynamic_tracker)
            else:
                controller = StateDependentFixedController(controller, dynamic_tracker)
            tracking_mode = (
                "sign-isolated-nonmonotone-roots"
                if any(
                    not delay.monotone_argument for delay in dynamic_delays
                )
                else "high-order-dynamic-roots"
            )
        else:
            tracking_mode = "tracking-disabled-depth-zero"
    if controller_mode == "adaptive":
        assert isinstance(controller, dfx.AbstractAdaptiveStepSizeController)
        controller = _RestartableAdaptiveController(controller)
    else:
        controller = _RestartableFixedController(controller)

    if stochastic:
        assert validated_realization is not None
        stochastic_wrapper = (
            _ItoRollingStochasticRetardedSolver
            if problem.interpretation == "ito"
            else _StratonovichRollingStochasticRetardedSolver
        )
        wrapped_solver = stochastic_wrapper(
            solver=selected_solver,
            history_capacity=resolved_capacity,
            maximum_lag=jnp.asarray(maximum_lag),
            heun=heun,
            path_key=validated_realization.path_keys,
            path_shape=brownian.shape,
            path_levy_area=brownian.levy_area,
            path_key_impl=jr.key_impl(validated_realization.path_keys),
            geometry=(
                selected_solver.geometry
                if isinstance(selected_solver, SRKMK)
                else None
            ),
        )
    elif isinstance(problem, NeutralDelayProblem):
        wrapped_solver = _RollingNeutralRetardedSolver(
            solver=selected_solver,
            history_capacity=resolved_capacity,
            maximum_lag=jnp.asarray(maximum_lag),
            initial_transformed_state=problem.transformed_initial_state,
        )
    else:
        wrapped_solver = _RollingRetardedSolver(
            solver=selected_solver,
            history_capacity=resolved_capacity,
            maximum_lag=jnp.asarray(maximum_lag),
        )
    if continuation is not None:
        if not isinstance(continuation, DelaySegmentContinuation):
            raise TypeError("continuation must be a DelaySegmentContinuation or None.")
        if not continuation.resumable:
            raise ValueError(
                "This continuation is terminal and cannot be resumed; event-terminated "
                "history is visibility-capped at its root."
            )
    start_time = solve_start if continuation is None else continuation.time
    times = validate_save_times(start_time, solve_end, save_times)
    states = jnp.zeros(
        (int(times.size),) + problem.state_shape, dtype=problem.initial_state.dtype
    )
    valid = jnp.zeros((int(times.size),), dtype=bool)

    if continuation is None:
        current_time = solve_start
        current_state = problem.initial_state
        solver_state = None
        controller_state = None
        made_jump = None
        accumulated = frozendict(
            {
                "num_steps": 0,
                "num_accepted_steps": 0,
                "num_rejected_steps": 0,
                "num_segments": 0,
            }
        )
    else:
        if continuation.problem_id != problem.problem_id:
            raise ValueError("Continuation problem identity does not match problem.")
        if continuation.solver_name != type(selected_solver).__name__:
            raise ValueError("Continuation solver identity does not match solver.")
        if continuation.controller_mode != controller_mode:
            raise ValueError("Continuation controller mode does not match controller.")
        if continuation.active_history.capacity != resolved_capacity:
            raise ValueError("Continuation active-history capacity does not match.")
        if continuation.realization is not validated_realization:
            raise ValueError("Continuation Wiener realization identity does not match.")
        current_time = continuation.time
        current_state = continuation.state
        solver_state = continuation.solver_state
        controller_state = continuation.controller_state
        made_jump = continuation.made_jump
        accumulated = continuation.stats

    archive_starts: list[np.ndarray] = []
    archive_ends: list[np.ndarray] = []
    archive_interpolations: list[Any] = []
    event_state = None if continuation is None else continuation.event_state
    final_result = SegmentedDelayResult.solver_failure
    segments_this_call = 0
    native_result: Any = dfx.RESULTS.successful

    while _host_scalar(current_time) < _host_scalar(solve_end):
        if max_segments is not None and segments_this_call >= max_segments:
            final_result = SegmentedDelayResult.segment_limit_reached
            break
        host_times = np.asarray(jax.device_get(times))
        pending = np.asarray(jax.device_get(~valid), dtype=bool)
        pending_requested_indices = np.flatnonzero(
            pending & (host_times >= _host_scalar(current_time))
        )
        if solver_state is None:
            segment_dt0 = dt0
        else:
            if controller_state is None:
                raise RuntimeError("Continuation controller state is missing.")
            segment_dt0 = controller_state[1]
        native = dfx.diffeqsolve(
            terms,
            wrapped_solver,
            t0=current_time,
            t1=solve_end,
            dt0=segment_dt0,
            y0=current_state,
            args=problem.args,
            saveat=dfx.SaveAt(
                subs={
                    "steps": dfx.SubSaveAt(t0=True, steps=True),
                    "final": dfx.SubSaveAt(t1=True),
                    "requested": dfx.SubSaveAt(
                        ts=times[jnp.asarray(pending_requested_indices)]
                    ),
                },
                dense=True,
                solver_state=True,
                controller_state=True,
                made_jump=True,
            ),
            stepsize_controller=controller,
            adjoint=selected_adjoint,
            event=event,
            max_steps=max_steps_per_segment,
            throw=False,
            solver_state=solver_state,
            controller_state=controller_state,
            made_jump=made_jump,
        )
        step_times = np.asarray(jax.device_get(native.ts["steps"]))
        finite_indices = np.flatnonzero(np.isfinite(step_times))
        final_time = jnp.asarray(native.ts["final"])[0]
        if _host_bool(jnp.isfinite(final_time)):
            next_time = final_time
            next_state = jnp.asarray(native.ys["final"])[0]
        elif finite_indices.size:
            last_index = int(finite_indices[-1])
            next_time = jnp.asarray(native.ts["steps"][last_index])
            next_state = jnp.asarray(native.ys["steps"][last_index])
        else:
            final_result = SegmentedDelayResult.solver_failure
            native_result = native.result
            break
        solver_state = native.solver_state
        if not isinstance(solver_state, _RollingRetardedSolverState):
            raise RuntimeError("Diffrax did not return rolling retarded solver state.")
        overflowed = _host_bool(solver_state.history.overflowed)
        if overflowed:
            next_time = solver_state.history.overflow_time
        accumulated = _accumulated_stats(accumulated, native.stats)
        if overflowed:
            accumulated = frozendict(
                {
                    **accumulated,
                    "num_steps": accumulated["num_steps"] - 1,
                    "num_accepted_steps": accumulated["num_accepted_steps"] - 1,
                }
            )
        segments_this_call += 1

        requested_times = np.asarray(jax.device_get(native.ts["requested"]))
        saved_requested_offsets = np.flatnonzero(
            np.isfinite(requested_times) & (requested_times <= _host_scalar(next_time))
        )
        if saved_requested_offsets.size:
            requested_indices = pending_requested_indices[saved_requested_offsets]
            indices = jnp.asarray(requested_indices)
            requested_states = jnp.asarray(native.ys["requested"])[
                jnp.asarray(saved_requested_offsets)
            ]
            states = states.at[indices].set(requested_states)
            valid = valid.at[indices].set(True)

        if dense and _host_scalar(next_time) > _host_scalar(current_time):
            archive_starts.append(np.asarray(jax.device_get(current_time)))
            archive_ends.append(np.asarray(jax.device_get(next_time)))
            archive_interpolations.append(jax.device_get(native.interpolation))

        controller_state = native.controller_state
        made_jump = native.made_jump
        event_state = native.event_mask
        native_result = native.result
        current_time = next_time
        current_state = next_state

        if overflowed:
            final_result = SegmentedDelayResult.history_capacity_exhausted
            break
        if _host_bool(native.result == dfx.RESULTS.event_occurred):
            final_result = SegmentedDelayResult.event_occurred
            break
        if _host_bool(native.result == dfx.RESULTS.successful):
            final_result = SegmentedDelayResult.successful
            break
        if _host_bool(native.result == dfx.RESULTS.max_steps_reached):
            continue
        final_result = SegmentedDelayResult.solver_failure
        break

    if solver_state is None:
        raise RuntimeError("Segmented delay solve produced no continuation state.")
    if final_result is SegmentedDelayResult.event_occurred:
        capped_history = solver_state.history.with_visible_end(current_time)
        solver_state = eqx.tree_at(
            lambda state: state.history,
            solver_state,
            capped_history,
        )
    if controller_state is None:
        raise RuntimeError("Segmented delay solve produced no controller state.")
    inner_controller_state = controller_state[0]
    if isinstance(inner_controller_state, DynamicControllerState):
        dynamic_state = inner_controller_state.discontinuities
        dynamic_root_count = dynamic_state.num_roots
        dynamic_root_times = dynamic_state.root_times
        internal_restarts = dynamic_state.num_restarts
        tracked_discontinuity_count = dynamic_state.count
    else:
        dynamic_state = None
        dynamic_root_count = jnp.asarray(0, dtype=jnp.int32)
        dynamic_root_times = jnp.empty((0,), dtype=problem.t0.dtype)
        internal_restarts = jnp.asarray(0, dtype=jnp.int32)
        tracked_discontinuity_count = jnp.sum(jnp.isfinite(discontinuities))
    continuation_out = DelaySegmentContinuation(
        time=current_time,
        state=current_state,
        solver_state=solver_state,
        controller_state=controller_state,
        made_jump=jnp.asarray(False) if made_jump is None else jnp.asarray(made_jump),
        realization=validated_realization,
        stats=accumulated,
        event_state=event_state,
        discontinuity_tracker=dynamic_state,
        problem_id=problem.problem_id,
        solver_name=type(selected_solver).__name__,
        controller_mode=controller_mode,
        resumable=final_result is SegmentedDelayResult.segment_limit_reached,
    )
    if throw and final_result not in (
        SegmentedDelayResult.successful,
        SegmentedDelayResult.event_occurred,
        SegmentedDelayResult.segment_limit_reached,
    ):
        if final_result is SegmentedDelayResult.history_capacity_exhausted:
            raise RuntimeError(
                "Segmented delay active history exhausted history_capacity before its lag window could be pruned."
            )
        raise RuntimeError(f"Segmented delay solve failed: {native_result}.")

    interpolation = None
    if dense and archive_interpolations:
        interpolation = DelaySegmentArchive(
            starts=np.stack(tuple(archive_starts)),
            ends=np.stack(tuple(archive_ends)),
            interpolations=tuple(archive_interpolations),
        )
    solver_name = type(selected_solver).__name__
    extension = (
        "srkmk-wiener-path"
        if stochastic and isinstance(selected_solver, SRKMK)
        else "euler-heun-wiener-path"
        if stochastic and heun
        else "euler-maruyama-wiener-path"
        if stochastic
        else None
    )
    if isinstance(selected_solver, AbstractGeometricSolver):
        solver_id = selected_solver.solver_id
        resolved_method = selected_solver.resolved_method
    elif stochastic:
        solver_id = (
            f"solver:diffrax-delay-stochastic:{solver_name}:segmented-retarded-v1"
        )
        resolved_method = f"{solver_name}:segmented-causal-wiener-path"
    elif isinstance(problem, NeutralDelayProblem):
        solver_id = "solver:diffrax-delay:Euler:segmented-transformed-neutral-v1"
        resolved_method = "Euler:segmented-transformed-neutral-method-of-steps"
    else:
        solver_id = f"solver:diffrax-delay:{solver_name}:segmented-retarded-v1"
        resolved_method = f"{solver_name}:segmented-causal-method-of-steps"
    stats = {
        **accumulated,
        "history_capacity": resolved_capacity,
        "num_delays": problem.num_delays,
        "active_history_bytes": continuation_out.active_history.allocated_bytes,
        "history_mode": "rolling",
        "history_max_occupancy": continuation_out.active_history.max_size,
        "num_history_evictions": continuation_out.active_history.num_evictions,
        "history_capacity_exhausted": continuation_out.active_history.overflowed,
        "retained_history_interval": jnp.stack(
            continuation_out.active_history.retained_interval
        ),
        "maximum_delay": maximum_lag,
        "minimum_delay": problem.minimum_delay,
        "stage_time_extent": stage_time_extent,
        "maximum_causal_step": maximum_step,
        "controller_mode": controller_mode,
        "discontinuity_depth": depth,
        "num_tracked_discontinuities": tracked_discontinuity_count,
        "num_dynamic_discontinuity_roots": dynamic_root_count,
        "dynamic_discontinuity_root_times": dynamic_root_times,
        "num_internal_discontinuity_restarts": internal_restarts,
        "state_dependent_tracking": tracking_mode,
        "functional_tracking": (
            "declared-lag-translations"
            if execution_plan.has_functional_delays
            else "not-applicable"
        ),
        "max_steps_per_segment": max_steps_per_segment,
        "continuous_extension": extension,
        "neutral_recovery_mode": (
            "implicit-root"
            if isinstance(problem, NeutralDelayProblem) and problem.implicit_recovery
            else "explicit"
            if isinstance(problem, NeutralDelayProblem)
            else None
        ),
        "neutral_recovery_solver": (
            type(problem.recovery_solver).__name__
            if isinstance(problem, NeutralDelayProblem) and problem.implicit_recovery
            else None
        ),
    }
    return MemoryEquationSolution(
        times=times,
        states=states,
        valid=valid,
        interpolation=interpolation,
        backend_result=final_result,
        stats=stats,
        event_mask=event_state,
        realization=realization,
        state_shape=problem.state_shape,
        solver_name=solver_name,
        solver_id=solver_id,
        resolved_method=resolved_method,
        metadata={
            "problem_id": problem.problem_id,
            "state_dependent_tracking": tracking_mode,
            "dynamic_discontinuity_root_times": dynamic_root_times,
            "backend": "diffrax",
            "delay_mode": (
                "segmented-transformed-neutral"
                if isinstance(problem, NeutralDelayProblem)
                else "segmented-functional-retarded"
                if execution_plan.has_functional_delays
                else "segmented-rolling-retarded"
            ),
            "native_result": native_result,
            "delay_term_types": tuple(
                type(term).__name__ for term in problem.delay_terms
            ),
            "functional_delay_contracts": tuple(
                {
                    "name": term.name,
                    "lag_interval": (term.minimum_delay, term.maximum_delay),
                    "output_kind": term.output_kind,
                    "discontinuity_lags": term.discontinuity_lags,
                }
                for term in problem.delay_terms
                if isinstance(term, FunctionalDelay)
            ),
            "continuation_resumable": continuation_out.resumable,
            "interpretation": problem.interpretation if stochastic else None,
            "driver_family": "wiener" if stochastic else "deterministic",
            "noise_id": problem.noise_id if stochastic else None,
            "realization_id": (
                validated_realization.realization_id
                if validated_realization is not None
                else None
            ),
            "coupling_id": (
                validated_realization.coupling_id
                if validated_realization is not None
                else None
            ),
        },
        continuation=continuation_out,
    )


__all__ = [
    "DelaySegmentArchive",
    "DelaySegmentContinuation",
    "SegmentedDelayResult",
    "fixed_delay_history_capacity",
    "solve_diffrax_delay_segmented",
]
