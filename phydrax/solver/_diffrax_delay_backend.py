#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx
from jax import core as jax_core
from jaxtyping import Array, ArrayLike

from ..linalg import AbstractRealCoordinateMap
from ..stochastic._wiener import WienerRealization
from ._delay import (
    _distributed_delay_value,
    _invalid_geometry_tangent,
    ConstantDelay,
    DelayDifferentialProblem,
    DelayHistoryWindow,
    DelayValues,
    DerivativeDelay,
    DistributedDelay,
    FunctionalDelay,
    NeutralDelayProblem,
)
from ._delay_adjoint import CheckpointedDelayAdjoint
from ._delay_discontinuity import (
    constant_discontinuity_schedule,
    DynamicControllerState,
    StateDependentAdaptiveController,
    StateDependentDiscontinuityTracker,
    StateDependentFixedController,
)
from ._delay_history import (
    _ComputedDelayHistory,
    DelayDenseInterpolation,
    DelayHistoryView,
    DenseDelayHistory,
    EmptyDelayHistory,
    RollingDelayHistory,
)
from ._delay_plan import (
    compile_delay_execution_plan,
    DelayHistoryMode,
    fixed_delay_history_capacity,
    resolve_delay_solver,
    stage_time_extent,
)
from ._diffrax_backend import _valid_values
from ._diffrax_state_packing import (
    _prepare_diffrax_state_adapter,
    _PreparedDiffraxStateAdapter,
    DiffraxComplexStatePolicy,
)
from ._geometric import AbstractGeometricSolver, SRKMK
from ._memory import MemoryEquationSolution
from ._save_schedule import validate_save_times


class _DelayValidation(eqx.Module):
    """Runtime delay checks attached to a consumed vector-field output."""

    predicates: Array
    messages: tuple[str, ...] = eqx.field(static=True)

    def apply(self, value: Array, /) -> Array:
        checked = value
        for index, message in enumerate(self.messages):
            checked = eqx.error_if(checked, self.predicates[index], message)
        return checked


class _CoordinateDelayHistory(eqx.Module):
    function: Any
    adapter: _PreparedDiffraxStateAdapter

    def __call__(self, time, args):
        public_args = self.adapter.unpack_args(args)
        value = self.function(time, public_args)
        return self.adapter.pack_state(value, owner="Delay history")


class _CoordinateDelayDerivative(eqx.Module):
    function: Any
    adapter: _PreparedDiffraxStateAdapter

    def __call__(self, time, args):
        public_args = self.adapter.unpack_args(args)
        value = self.function(time, public_args)
        return self.adapter.pack_state(value, owner="Delay history derivative")


class _PublicDelayWindow(eqx.Module):
    window: DelayHistoryWindow
    adapter: _PreparedDiffraxStateAdapter

    def value(self, time, /, *, left=False):
        return self.adapter.unpack_state(self.window.value(time, left=left))

    def values(self, times, /, *, left=False):
        return self.adapter.unpack_values(
            self.window.values(times, left=left),
            jnp.asarray(times).ndim,
        )

    def derivative(self, time, /, *, left=False):
        return self.adapter.unpack_state(self.window.derivative(time, left=left))

    def derivatives(self, times, /, *, left=False):
        return self.adapter.unpack_values(
            self.window.derivatives(times, left=left),
            jnp.asarray(times).ndim,
        )


class _CoordinateDelayInterpolation(eqx.Module):
    interpolation: DelayDenseInterpolation
    adapter: _PreparedDiffraxStateAdapter

    def evaluate(self, query_times, /, *, left=True):
        query = jnp.asarray(query_times)
        return self.adapter.unpack_values(
            self.interpolation.evaluate(query, left=left),
            query.ndim,
        )

    def derivative(self, query_times, /, *, left=True):
        query = jnp.asarray(query_times)
        return self.adapter.unpack_values(
            self.interpolation.derivative(query, left=left),
            query.ndim,
        )


class _DelayVectorField(eqx.Module):
    function: Any
    initial_history: Any
    initial_derivative: Any
    delay_terms: tuple[Any, ...]
    initial_time: Array
    state_shape: tuple[int, ...] = eqx.field(static=True)
    geometry: Any
    state_adapter: _PreparedDiffraxStateAdapter
    backend_shape: tuple[int, ...] = eqx.field(static=True)
    computed_history: _ComputedDelayHistory

    def _memory(
        self, time: Array, state: Array, args: Any, /
    ) -> tuple[DelayValues, _DelayValidation]:
        public_state = self.state_adapter.unpack_state(state)
        public_args = self.state_adapter.unpack_args(args)
        history = DelayHistoryView(
            initial_history=self.initial_history,
            initial_derivative=self.initial_derivative,
            args=args,
            initial_time=self.initial_time,
            computed_history=self.computed_history,
            state_shape=self.backend_shape,
            geometry=None,
        )
        values = []
        predicates = []
        messages = []
        for term in self.delay_terms:
            delayed_states = None
            if isinstance(term, FunctionalDelay):
                window = _PublicDelayWindow(
                    DelayHistoryWindow(
                        history,
                        time,
                        term.minimum_delay,
                        term.maximum_delay,
                    ),
                    self.state_adapter,
                )
                value = jnp.asarray(
                    term.functional(time, public_state, window, public_args)
                )
            elif isinstance(term, DistributedDelay):
                delayed_states = self.state_adapter.unpack_values(
                    history.values(time - term.nodes, left=False),
                    1,
                )
                value = _distributed_delay_value(
                    term,
                    time,
                    public_state,
                    delayed_states,
                    public_args,
                    self.state_shape,
                )
            elif isinstance(term, DerivativeDelay):
                point = term.delay
                lag = (
                    point.delay
                    if isinstance(point, ConstantDelay)
                    else point.value(time, public_state, public_args)
                )
                delayed_state = self.state_adapter.unpack_state(
                    history.value(time - lag, left=False)
                )
                delayed_states = delayed_state[None, ...]
                value = self.state_adapter.unpack_state(
                    history.derivative(time - lag, left=False)
                )
                if self.geometry is not None and not self.geometry.trivial:
                    predicates.append(
                        _invalid_geometry_tangent(
                            self.geometry,
                            delayed_state,
                            value,
                            self.state_shape,
                        )
                    )
                    messages.append(
                        f"DerivativeDelay {term.name!r} queried a history derivative "
                        "that is not tangent at the delayed state."
                    )
                if term.transport is not None:
                    value = jnp.asarray(
                        term.transport(
                            delayed_state,
                            public_state,
                            value,
                            public_args,
                        )
                    )
            else:
                lag = (
                    term.delay
                    if isinstance(term, ConstantDelay)
                    else term.value(time, public_state, public_args)
                )
                value = self.state_adapter.unpack_state(
                    history.value(time - lag, left=False)
                )
                delayed_states = value[None, ...]
            if value.shape != self.state_shape:
                raise ValueError(
                    f"Delay term {term.name!r} changed its declared state shape."
                )
            if self.geometry is not None and delayed_states is not None:
                historical_membership = jax.vmap(
                    lambda candidate: jnp.asarray(
                        self.geometry.contains(candidate), dtype=bool
                    )
                )(delayed_states)
                if historical_membership.shape != (int(delayed_states.shape[0]),):
                    raise ValueError(
                        "State geometry contains() must return a scalar boolean."
                    )
                predicates.append(~jnp.all(historical_membership))
                messages.append("A delayed history value lies outside state_geometry.")
            if (
                self.geometry is not None
                and not self.geometry.trivial
                and isinstance(term, DerivativeDelay)
            ):
                predicates.append(
                    _invalid_geometry_tangent(
                        self.geometry,
                        public_state,
                        value,
                        self.state_shape,
                    )
                )
                messages.append(
                    f"DerivativeDelay {term.name!r} transport must return a tangent "
                    "at the current state."
                )
            if self.geometry is not None and isinstance(term, DistributedDelay):
                membership = jnp.asarray(self.geometry.contains(value), dtype=bool)
                if membership.shape != ():
                    raise ValueError(
                        "State geometry contains() must return a scalar boolean."
                    )
                predicates.append(~membership)
                messages.append(
                    f"DistributedDelay {term.name!r} reducer returned a point "
                    "outside state_geometry."
                )
            if (
                self.geometry is not None
                and isinstance(term, FunctionalDelay)
                and term.output_kind == "point"
            ):
                membership = jnp.asarray(self.geometry.contains(value), dtype=bool)
                if membership.shape != ():
                    raise ValueError(
                        "State geometry contains() must return a scalar boolean."
                    )
                predicates.append(~membership)
                messages.append(
                    f"FunctionalDelay {term.name!r} returned a point outside "
                    "state_geometry."
                )
            if (
                self.geometry is not None
                and not self.geometry.trivial
                and isinstance(term, FunctionalDelay)
                and term.output_kind == "tangent"
            ):
                predicates.append(
                    _invalid_geometry_tangent(
                        self.geometry,
                        public_state,
                        value,
                        self.state_shape,
                    )
                )
                messages.append(
                    f"FunctionalDelay {term.name!r} returned a non-tangent value."
                )
            values.append(value)
        return (
            DelayValues(
                tuple(term.name for term in self.delay_terms),
                tuple(values),
            ),
            _DelayValidation(
                (
                    jnp.stack(tuple(predicates))
                    if predicates
                    else jnp.empty((0,), dtype=bool)
                ),
                tuple(messages),
            ),
        )

    def __call__(self, time: ArrayLike, state: Array, args: Any) -> Array:
        query = jnp.asarray(time)
        backend = jnp.asarray(state)
        if backend.shape != self.backend_shape:
            raise ValueError("Delay solver state changed its backend coordinate shape.")
        current = self.state_adapter.unpack_state(backend)
        if self.geometry is not None:
            membership = jnp.asarray(self.geometry.contains(current), dtype=bool)
            if membership.shape != ():
                raise ValueError(
                    "State geometry contains() must return a scalar boolean."
                )
            current = eqx.error_if(
                current,
                ~membership,
                "A geometric delay solver stage lies outside state_geometry.",
            )
        memory, validation = self._memory(query, backend, args)
        public_args = self.state_adapter.unpack_args(args)
        value = jnp.asarray(
            self.function(
                query,
                current,
                memory,
                public_args,
            )
        )
        if value.shape != self.state_shape:
            raise ValueError("Delay drift changed its declared state shape.")
        value = validation.apply(value)
        if self.geometry is not None:
            projected = jnp.asarray(self.geometry.project_tangent(current, value))
            if projected.shape != self.state_shape:
                raise ValueError(
                    "State geometry tangent projection changed the state shape."
                )
            if jnp.issubdtype(value.dtype, jnp.inexact):
                scale = jnp.maximum(
                    1.0,
                    jnp.maximum(jnp.max(jnp.abs(value)), jnp.max(jnp.abs(projected))),
                )
                tolerance = 256.0 * jnp.finfo(value.dtype).eps * scale
            else:
                tolerance = jnp.asarray(0, dtype=value.dtype)
            value = eqx.error_if(
                value,
                ~jnp.all(jnp.isfinite(value))
                | ~jnp.all(jnp.isfinite(projected))
                | jnp.any(jnp.abs(value - projected) > tolerance),
                "Geometric delay drift must be tangent-compatible with state_geometry.",
            )
        return self.state_adapter.pack_state(value, owner="Delay drift")


class _ZeroVectorField(eqx.Module):
    """Identically zero deterministic second term for drift-only SRKMK."""

    def __call__(self, time: ArrayLike, state: Array, args: Any) -> Array:
        del time, args
        return jnp.zeros_like(state)


def _bind_delay_history(
    terms: Any, history: DenseDelayHistory | RollingDelayHistory, /
) -> Any:
    def bind(value):
        if isinstance(value, _DelayVectorField):
            return eqx.tree_at(
                lambda vector_field: vector_field.computed_history,
                value,
                history,
            )
        return value

    return jax.tree.map(
        bind,
        terms,
        is_leaf=lambda value: isinstance(value, _DelayVectorField),
    )


class _RetardedSolverState(eqx.Module):
    inner_state: Any
    history: DenseDelayHistory | RollingDelayHistory


class _RetardedSolver(dfx.AbstractWrappedSolver):
    """Diffrax solver wrapper that threads accepted dense history through its state."""

    solver: dfx.AbstractSolver  # ty: ignore[invalid-attribute-override]
    history_capacity: int = eqx.field(static=True)
    history_mode: DelayHistoryMode = eqx.field(static=True)
    maximum_lag: Array | None

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
        _, _, dense_info_structure, _, _ = eqx.filter_eval_shape(
            self.solver.step,
            terms,
            t0,
            t1,
            y0,
            args,
            provisional_state,
            False,
        )
        if self.history_mode == "full":
            history: DenseDelayHistory | RollingDelayHistory = DenseDelayHistory.allocate(
                time=t0,
                dense_info_structure=dense_info_structure,
                capacity=self.history_capacity,
                interpolation_cls=self.solver.interpolation_cls,
            )
        else:
            if self.maximum_lag is None:
                raise ValueError("Rolling history requires a finite maximum delay.")
            history = RollingDelayHistory.allocate(
                time=t0,
                dense_info_structure=dense_info_structure,
                capacity=self.history_capacity,
                interpolation_cls=self.solver.interpolation_cls,
                maximum_lag=self.maximum_lag,
            )
        bound_terms = _bind_delay_history(terms, history)
        inner_state = self.solver.init(bound_terms, t0, t1, y0, args)
        return _RetardedSolverState(inner_state=inner_state, history=history)

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
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
        if isinstance(history, RollingDelayHistory):
            result = dfx.RESULTS.where(
                history.overflowed,
                dfx.RESULTS.max_steps_reached,
                result,
            )
        return (
            y1,
            y_error,
            dense_info,
            _RetardedSolverState(inner_state=inner_state, history=history),
            result,
        )


class _NeutralRecovery(eqx.Module):
    """Recover the physical state from one transformed neutral state."""

    context: _DelayVectorField
    neutral_functional: Any
    endpoint_neutral: Any
    initial_guess: Any
    root_finder: optx.AbstractRootFinder
    max_steps: int = eqx.field(static=True)
    state_shape: tuple[int, ...] = eqx.field(static=True)
    state_adapter: _PreparedDiffraxStateAdapter

    def _retarded(self, time: Array, memory: DelayValues, args: Any, /) -> Array:
        public_args = self.state_adapter.unpack_args(args)
        value = jnp.asarray(self.neutral_functional(time, memory, public_args))
        if value.shape != self.state_shape:
            raise ValueError("neutral_functional changed its declared state shape.")
        return value

    def transform(self, time: Array, state: Array, args: Any, /) -> Array:
        public_state = self.state_adapter.unpack_state(state)
        memory, validation = self.context._memory(time, state, args)
        neutral = self._retarded(time, memory, args)
        if self.endpoint_neutral is not None:
            endpoint = jnp.asarray(
                self.endpoint_neutral(
                    time,
                    public_state,
                    memory,
                    self.state_adapter.unpack_args(args),
                )
            )
            if endpoint.shape != self.state_shape:
                raise ValueError("endpoint_neutral changed its declared state shape.")
            neutral = neutral + endpoint
        transformed = validation.apply(public_state - neutral)
        return self.state_adapter.pack_state(
            transformed, owner="Neutral transformed state"
        )

    def recover(self, time: Array, transformed: Array, args: Any, /) -> Array:
        public_transformed = self.state_adapter.unpack_state(transformed)
        memory, validation = self.context._memory(time, transformed, args)
        retarded = self._retarded(time, memory, args)
        explicit = public_transformed + retarded
        if self.endpoint_neutral is None:
            return self.state_adapter.pack_state(
                validation.apply(explicit), owner="Recovered neutral state"
            )
        endpoint_neutral = self.endpoint_neutral

        def residual(candidate, packed):
            query_time, delayed_memory, packed_args, target, retarded_value = packed
            public_args = self.state_adapter.unpack_args(packed_args)
            endpoint = jnp.asarray(
                endpoint_neutral(
                    query_time,
                    candidate,
                    delayed_memory,
                    public_args,
                )
            )
            if endpoint.shape != self.state_shape:
                raise ValueError("endpoint_neutral changed its declared state shape.")
            return candidate - retarded_value - endpoint - target

        initial = (
            explicit
            if self.initial_guess is None
            else jnp.asarray(
                self.initial_guess(
                    time,
                    public_transformed,
                    memory,
                    self.state_adapter.unpack_args(args),
                )
            )
        )
        if initial.shape != self.state_shape:
            raise ValueError("recovery_initial_guess must preserve the state shape.")
        recovered = optx.root_find(
            residual,
            self.root_finder,
            initial,
            args=(time, memory, args, public_transformed, retarded),
            max_steps=self.max_steps,
            throw=True,
        ).value
        return self.state_adapter.pack_state(
            validation.apply(recovered), owner="Recovered neutral state"
        )


class _NeutralVectorField(eqx.Module):
    """Transformed-state differential with physical-state recovery."""

    differential: Any
    recovery: _NeutralRecovery
    state_shape: tuple[int, ...] = eqx.field(static=True)
    state_adapter: _PreparedDiffraxStateAdapter

    def __call__(self, time: ArrayLike, transformed: Array, args: Any) -> Array:
        query = jnp.asarray(time)
        physical_backend = self.recovery.recover(query, transformed, args)
        physical = self.state_adapter.unpack_state(physical_backend)
        memory, validation = self.recovery.context._memory(query, physical_backend, args)
        public_args = self.state_adapter.unpack_args(args)
        value = jnp.asarray(self.differential(query, physical, memory, public_args))
        if value.shape != self.state_shape:
            raise ValueError("Neutral differential changed its declared state shape.")
        value = validation.apply(value)
        return self.state_adapter.pack_state(value, owner="Neutral differential")


def _neutral_vector_field(
    problem: NeutralDelayProblem,
    context: _DelayVectorField,
    /,
) -> _NeutralVectorField:
    recovery = _NeutralRecovery(
        context=context,
        neutral_functional=problem.neutral_functional,
        endpoint_neutral=problem.endpoint_neutral,
        initial_guess=problem.recovery_initial_guess,
        root_finder=problem.recovery_solver,
        max_steps=problem.recovery_max_steps,
        state_shape=problem.state_shape,
        state_adapter=context.state_adapter,
    )
    return _NeutralVectorField(
        differential=problem.differential,
        recovery=recovery,
        state_shape=problem.state_shape,
        state_adapter=context.state_adapter,
    )


def _underlying_neutral_vector_field(terms: Any, /) -> _NeutralVectorField:
    fields = tuple(
        value
        for value in jax.tree.leaves(
            terms,
            is_leaf=lambda value: isinstance(value, _NeutralVectorField),
        )
        if isinstance(value, _NeutralVectorField)
    )
    if len(fields) != 1:
        raise TypeError("Neutral delay execution requires one neutral vector field.")
    return fields[0]


class _NeutralInnerState(eqx.Module):
    solver_state: Any
    transformed_state: Array


class _NeutralRetardedSolver(_RetardedSolver):
    """Fixed-step transformed neutral solver returning physical states."""

    initial_transformed_state: Array

    @property
    def interpolation_cls(self):  # ty: ignore[invalid-attribute-override]
        return dfx.LocalLinearInterpolation

    def init(self, terms, t0, t1, y0, args):
        dense_info_structure = {"y0": y0, "y1": y0}
        if self.history_mode == "full":
            history: DenseDelayHistory | RollingDelayHistory = DenseDelayHistory.allocate(
                time=t0,
                dense_info_structure=dense_info_structure,
                capacity=self.history_capacity,
                interpolation_cls=self.interpolation_cls,
            )
        else:
            if self.maximum_lag is None:
                raise ValueError("Rolling history requires a finite maximum delay.")
            history = RollingDelayHistory.allocate(
                time=t0,
                dense_info_structure=dense_info_structure,
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
        return _RetardedSolverState(inner_state=inner, history=history)

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
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
        if isinstance(history, RollingDelayHistory):
            result = dfx.RESULTS.where(
                history.overflowed,
                dfx.RESULTS.max_steps_reached,
                result,
            )
        next_inner = _NeutralInnerState(
            solver_state=next_solver_state,
            transformed_state=transformed,
        )
        return (
            y1,
            y_error,
            dense_info,
            _RetardedSolverState(inner_state=next_inner, history=history),
            result,
        )


def _stage_time_extent(solver: dfx.AbstractSolver, /) -> Array:
    return stage_time_extent(solver)


class _CausalAdaptiveStepSizeController(dfx.AbstractAdaptiveStepSizeController):
    """Adaptive controller wrapper keeping every stage within accepted history."""

    controller: dfx.AbstractAdaptiveStepSizeController
    maximum_step: Array

    def __init__(
        self,
        controller: dfx.AbstractAdaptiveStepSizeController,
        maximum_step: Array,
    ):
        self.controller = controller
        self.maximum_step = maximum_step

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
        return eqx.tree_at(
            lambda controller: controller.controller,
            self,
            self.controller.wrap(direction),
        )

    def _cap(self, start, proposed_end):
        boundary = jax.lax.stop_gradient(start + self.maximum_step)
        causal_end = jnp.nextafter(boundary, jax.lax.stop_gradient(start))
        return jax.lax.stop_gradient(jnp.minimum(proposed_end, causal_end))

    def init(self, terms, t0, t1, y0, dt0, args, func, error_order):
        next_t1, state = self.controller.init(
            terms,
            t0,
            t1,
            y0,
            dt0,
            args,
            func,
            error_order,
        )
        return self._cap(t0, next_t1), state

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
        keep, next_t0, next_t1, made_jump, state, result = (
            self.controller.adapt_step_size(
                jax.lax.stop_gradient(t0),
                jax.lax.stop_gradient(t1),
                y0,
                y1_candidate,
                args,
                y_error,
                error_order,
                controller_state,
            )
        )
        return (
            keep,
            next_t0,
            self._cap(next_t0, next_t1),
            made_jump,
            state,
            result,
        )


class _CausalFixedStepSizeController(dfx.AbstractStepSizeController):
    """Fixed controller that shortens steps at known delay discontinuities."""

    maximum_step: Array
    jump_ts: Array

    def wrap(self, direction):
        return type(self)(
            maximum_step=self.maximum_step,
            jump_ts=jnp.sort(self.jump_ts * direction),
        )

    def _next_jump(self, index, dtype):
        if int(self.jump_ts.size) == 0:
            return jnp.asarray(jnp.inf, dtype=dtype)
        safe = jnp.minimum(index, int(self.jump_ts.size) - 1)
        return jnp.where(
            index < int(self.jump_ts.size),
            self.jump_ts[safe],
            jnp.asarray(jnp.inf, dtype=dtype),
        )

    def _end_before_jump(self, proposed_end, next_jump):
        previous = jnp.nextafter(next_jump, jnp.asarray(-jnp.inf))
        return jnp.minimum(proposed_end, previous)

    def init(self, terms, t0, t1, y0, dt0, args, func, error_order):
        del terms, t1, y0, args, func, error_order
        if dt0 is None:
            raise ValueError("Fixed-step delay solves require dt0.")
        step = jnp.asarray(dt0)
        step = eqx.error_if(
            step,
            ~jnp.isfinite(step) | (step <= 0.0) | (step > self.maximum_step),
            "dt0 must be positive and no larger than the causal delay step bound.",
        )
        index = jnp.searchsorted(self.jump_ts, t0, side="right")
        next_jump = self._next_jump(index, jnp.result_type(t0))
        return self._end_before_jump(t0 + step, next_jump), (step, index)

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
        del t0, y0, y1_candidate, args, y_error, error_order
        step, old_index = controller_state
        old_jump = self._next_jump(old_index, jnp.result_type(t1))
        reached_jump = jnp.nextafter(t1, jnp.asarray(jnp.inf)) == old_jump
        next_t0 = jnp.where(
            reached_jump,
            jnp.nextafter(old_jump, jnp.asarray(jnp.inf)),
            t1,
        )
        index = jnp.searchsorted(self.jump_ts, next_t0, side="right")
        next_jump = self._next_jump(index, jnp.result_type(t1))
        next_end = self._end_before_jump(next_t0 + step, next_jump)
        return (
            True,
            next_t0,
            next_end,
            reached_jump,
            (step, index),
            dfx.RESULTS.successful,
        )


def _delay_discontinuity_times(
    delays: Array,
    initial_discontinuities: Array,
    /,
    *,
    depth: int,
    max_discontinuities: int,
) -> Array:
    """Generate additive constant-delay descendants through one finite depth."""
    times, _ = constant_discontinuity_schedule(
        delays,
        initial_discontinuities,
        depth=depth,
        max_discontinuities=max_discontinuities,
    )
    return times


def _neutral_discontinuity_times(
    delays: Array,
    initial_discontinuities: Array,
    horizon: Array,
    /,
    *,
    max_discontinuities: int,
) -> Array:
    """Generate the ordered additive closure before a finite solve horizon.

    The output has static capacity and is padded with ``inf``. This keeps the
    schedule transformable when delay values or the solve horizon are traced.
    """
    dtype = jnp.result_type(delays, initial_discontinuities, horizon)
    lag_values = jnp.asarray(delays, dtype=dtype)
    sources = jnp.asarray(initial_discontinuities, dtype=dtype)
    limit = jnp.asarray(horizon, dtype=dtype)
    schedule = jnp.full((max_discontinuities,), jnp.inf, dtype=dtype)
    epsilon = 64.0 * jnp.finfo(dtype).eps

    def next_after(previous, known):
        tolerance = epsilon * jnp.maximum(1.0, jnp.abs(previous))
        threshold = jnp.where(jnp.isneginf(previous), previous, previous + tolerance)
        if int(sources.size) == 0:
            source_candidate = jnp.asarray(jnp.inf, dtype=dtype)
        else:
            source_candidate = jnp.min(jnp.where(sources > threshold, sources, jnp.inf))
        if int(lag_values.size) == 0:
            descendant_candidate = jnp.asarray(jnp.inf, dtype=dtype)
        else:
            target = threshold - lag_values
            indices = jax.vmap(
                lambda value: jnp.searchsorted(known, value, side="right")
            )(target)
            safe = jnp.minimum(indices, max_discontinuities - 1)
            descendants = known[safe] + lag_values
            descendants = jnp.where(
                indices < max_discontinuities,
                descendants,
                jnp.inf,
            )
            descendant_candidate = jnp.min(descendants)
        return jnp.minimum(source_candidate, descendant_candidate)

    def append(index, known):
        previous = jnp.where(index == 0, -jnp.inf, known[index - 1])
        candidate = next_after(previous, known)
        candidate = jnp.where(candidate < limit, candidate, jnp.inf)
        return known.at[index].set(candidate)

    schedule = jax.lax.fori_loop(0, max_discontinuities, append, schedule)
    next_candidate = next_after(schedule[-1], schedule)
    return eqx.error_if(
        schedule,
        next_candidate < limit,
        "Neutral discontinuity schedule exceeds max_discontinuities before t1.",
    )


def _resolved_discontinuity_depth(
    solver: dfx.AbstractSolver, terms: Any, depth: int | None
) -> int:
    if depth is not None:
        if not isinstance(depth, int) or isinstance(depth, bool) or depth < 0:
            raise ValueError("discontinuity_depth must be a nonnegative integer or None.")
        return depth
    order = solver.order(terms)
    if order is None:
        raise ValueError(
            "discontinuity_depth is required when the solver does not declare an order."
        )
    return int(order)


def _native_delay_solution(
    problem: DelayDifferentialProblem | NeutralDelayProblem,
    save_times: Array,
    *,
    terms: Any,
    solver: dfx.AbstractSolver,
    stepsize_controller: dfx.AbstractStepSizeController,
    adjoint: Any,
    dt0: ArrayLike | None,
    event: Any | None,
    dense: bool,
    history_mode: DelayHistoryMode,
    history_capacity: int,
    maximum_lag: Array | None,
    max_steps: int | None,
    throw: bool,
    state_adapter: _PreparedDiffraxStateAdapter,
):
    if isinstance(problem, NeutralDelayProblem):
        wrapped_solver = _NeutralRetardedSolver(
            solver=solver,
            history_capacity=history_capacity,
            history_mode=history_mode,
            maximum_lag=maximum_lag,
            initial_transformed_state=state_adapter.pack_state(
                problem.transformed_initial_state,
                owner="Initial neutral transformed state",
            ),
        )
    else:
        wrapped_solver = _RetardedSolver(
            solver=solver,
            history_capacity=history_capacity,
            history_mode=history_mode,
            maximum_lag=maximum_lag,
        )
    saveat = dfx.SaveAt(
        subs={
            "requested": dfx.SubSaveAt(ts=save_times),
            "final": dfx.SubSaveAt(t1=True),
        },
        solver_state=dense or history_mode == "rolling",
        controller_state=True,
    )
    return dfx.diffeqsolve(
        terms,
        wrapped_solver,
        t0=problem.t0,
        t1=problem.t1,
        dt0=dt0,
        y0=state_adapter.pack_state(problem.initial_state, owner="Initial delay state"),
        args=state_adapter.pack_args(problem.args),
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        adjoint=adjoint,
        event=state_adapter.wrap_event(event),
        max_steps=max_steps,
        throw=bool(throw and history_mode == "full"),
    )


def _resolved_delay_solver(
    problem: DelayDifferentialProblem,
    solver: Any | None,
    /,
) -> dfx.AbstractSolver:
    return resolve_delay_solver(problem, solver)


def _validate_delay_geometry_solver(
    problem: DelayDifferentialProblem,
    solver: dfx.AbstractSolver,
    /,
) -> None:
    compile_delay_execution_plan(
        problem,
        solver,
        execution="whole",
        history_mode="full",
    )


def _delay_solver_provenance(
    solver: dfx.AbstractSolver,
    /,
    *,
    neutral: bool,
) -> tuple[str, str]:
    if isinstance(solver, AbstractGeometricSolver):
        return solver.solver_id, solver.resolved_method
    name = type(solver).__name__
    equation_kind = "neutral" if neutral else "retarded"
    return (
        f"solver:diffrax-delay:{name}:{equation_kind}-v1",
        f"{name}:causal-{equation_kind}-method-of-steps",
    )


def _deterministic_delay_terms(
    solver: dfx.AbstractSolver,
    vector_field: _DelayVectorField,
    /,
) -> Any:
    drift = dfx.ODETerm(vector_field)
    if isinstance(solver, SRKMK):
        return dfx.MultiTerm(drift, dfx.ODETerm(_ZeroVectorField()))
    return drift


def solve_diffrax_delay(
    problem: DelayDifferentialProblem | NeutralDelayProblem,
    /,
    *,
    save_times: ArrayLike,
    realization: WienerRealization | None = None,
    solver: Any | None = None,
    stepsize_controller: Any | None = None,
    adjoint: Any | None = None,
    dt0: ArrayLike | None = None,
    event: Any | None = None,
    rtol: float = 1e-6,
    atol: float = 1e-8,
    dense: bool = False,
    history_mode: DelayHistoryMode = "full",
    history_capacity: int | None = None,
    history_margin: int = 2,
    max_steps: int | None = 4096,
    initial_discontinuities: ArrayLike | Sequence[float] | None = None,
    discontinuity_depth: int | None = None,
    max_discontinuities: int = 8192,
    root_rtol: float = 1e-10,
    root_atol: float = 1e-12,
    max_root_iterations: int = 64,
    throw: bool = False,
    complex_state_policy: DiffraxComplexStatePolicy | None = None,
    state_coordinates: AbstractRealCoordinateMap | None = None,
) -> MemoryEquationSolution:
    """Solve a declared delay differential equation through Diffrax."""
    if not isinstance(problem, (DelayDifferentialProblem, NeutralDelayProblem)):
        raise TypeError(
            "solve_diffrax_delay requires a DelayDifferentialProblem or "
            "NeutralDelayProblem."
        )
    state_adapter = _prepare_diffrax_state_adapter(
        problem.initial_state,
        complex_state_policy,
        state_coordinates,
        problem.state_geometry,
    )
    if not isinstance(dense, bool):
        raise TypeError("dense must be a bool.")
    if history_mode not in ("full", "rolling"):
        raise ValueError("history_mode must be 'full' or 'rolling'.")
    if max_steps is not None and (
        not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps <= 0
    ):
        raise ValueError("max_steps must be a positive integer or None.")
    if history_mode == "full" and max_steps is None:
        raise ValueError("Full delay history requires finite max_steps.")
    if history_mode == "full" and history_capacity is not None:
        raise ValueError("history_capacity is only valid with history_mode='rolling'.")
    if history_capacity is not None and (
        not isinstance(history_capacity, int)
        or isinstance(history_capacity, bool)
        or history_capacity <= 0
    ):
        raise ValueError("history_capacity must be a positive integer or None.")
    if (
        not isinstance(history_margin, int)
        or isinstance(history_margin, bool)
        or history_margin < 1
    ):
        raise ValueError("history_margin must be a positive integer.")
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
    if problem.stochastic:
        assert isinstance(problem, DelayDifferentialProblem)
        from ._diffrax_delay_stochastic import _solve_diffrax_delay_stochastic

        return _solve_diffrax_delay_stochastic(
            problem,
            save_times=save_times,
            realization=realization,
            solver=solver,
            stepsize_controller=stepsize_controller,
            adjoint=adjoint,
            dt0=dt0,
            event=event,
            rtol=rtol,
            atol=atol,
            dense=dense,
            max_steps=max_steps,
            initial_discontinuities=initial_discontinuities,
            discontinuity_depth=discontinuity_depth,
            max_discontinuities=max_discontinuities,
            history_mode=history_mode,
            history_capacity=history_capacity,
            history_margin=history_margin,
            throw=throw,
            state_adapter=state_adapter,
        )
    if realization is not None:
        raise ValueError(
            "Deterministic delay problems do not accept a WienerRealization."
        )

    if problem.neutral and discontinuity_depth is not None:
        raise ValueError(
            "Neutral delay solves propagate discontinuities through the full horizon; "
            "discontinuity_depth must be None."
        )

    if isinstance(problem, NeutralDelayProblem):
        selected_solver = dfx.Euler() if solver is None else solver
        if type(selected_solver) is not dfx.Euler:
            raise ValueError(
                "NeutralDelayProblem execution currently requires diffrax.Euler."
            )
    else:
        selected_solver = _resolved_delay_solver(problem, solver)
    execution_plan = compile_delay_execution_plan(
        problem,
        selected_solver,
        execution="whole",
        history_mode=history_mode,
    )
    stage_time_extent = execution_plan.stage_time_extent

    times = validate_save_times(problem.t0, problem.t1, save_times)
    initial_computed_derivative = (
        problem.initial_right_derivative
        if problem.neutral
        else jnp.zeros_like(problem.initial_state)
    )
    packed_initial = state_adapter.pack_state(
        problem.initial_state, owner="Initial delay state"
    )
    packed_derivative = state_adapter.pack_state(
        initial_computed_derivative,
        owner="Initial delay derivative",
    )
    empty_history = EmptyDelayHistory(
        packed_initial,
        packed_derivative,
    )
    history_context = _DelayVectorField(
        function=problem.drift,
        initial_history=_CoordinateDelayHistory(problem.history, state_adapter),
        initial_derivative=(
            None
            if problem.history_derivative is None
            else _CoordinateDelayDerivative(problem.history_derivative, state_adapter)
        ),
        delay_terms=problem.delay_terms,
        initial_time=problem.t0,
        state_shape=problem.state_shape,
        geometry=problem.state_geometry,
        state_adapter=state_adapter,
        backend_shape=state_adapter.backend_shape,
        computed_history=empty_history,
    )
    if isinstance(problem, NeutralDelayProblem):
        vector_field = _neutral_vector_field(problem, history_context)
        terms = dfx.ODETerm(vector_field)
    else:
        vector_field = history_context
        terms = _deterministic_delay_terms(selected_solver, vector_field)
    if problem.neutral:
        depth = None
    else:
        depth = _resolved_discontinuity_depth(
            selected_solver,
            terms,
            discontinuity_depth,
        )
    if initial_discontinuities is None:
        declared_sources = (
            jnp.empty((0,), dtype=problem.t0.dtype)
            if problem.neutral
            else problem.t0.reshape((1,))
        )
    else:
        declared_sources = jnp.asarray(initial_discontinuities, dtype=problem.t0.dtype)
        if declared_sources.ndim != 1:
            raise ValueError("initial_discontinuities must be a rank-1 array or None.")
        declared_sources = eqx.error_if(
            declared_sources,
            ~jnp.all(jnp.isfinite(declared_sources)),
            "initial_discontinuities must be finite.",
        )
    if problem.neutral:
        compatibility_source = jnp.where(
            problem.initial_derivative_compatible,
            jnp.asarray(jnp.inf, dtype=problem.t0.dtype),
            problem.t0,
        ).reshape((1,))
        sources = jnp.concatenate((declared_sources, compatibility_source))
    else:
        sources = declared_sources
    constant_lags = execution_plan.constant_lags
    dynamic_delays = execution_plan.state_dependent_delays
    schedule_lags = (
        jnp.stack(constant_lags)
        if constant_lags
        else jnp.empty((0,), dtype=problem.t0.dtype)
    )
    tracker_depth: int
    if problem.neutral:
        discontinuities = _neutral_discontinuity_times(
            schedule_lags,
            sources,
            problem.t1,
            max_discontinuities=max_discontinuities,
        )
        tracker_initial_times = sources
        tracker_initial_generations = jnp.zeros(
            sources.shape,
            dtype=jnp.int32,
        )
        tracker_depth = max_discontinuities
    else:
        assert depth is not None
        discontinuities, discontinuity_generations = constant_discontinuity_schedule(
            schedule_lags,
            sources,
            depth=depth,
            max_discontinuities=max_discontinuities,
        )
        tracker_initial_times = discontinuities
        tracker_initial_generations = discontinuity_generations
        tracker_depth = depth
    controller_discontinuities = jnp.sort(
        jnp.where(
            (discontinuities > problem.t0) & (discontinuities < problem.t1),
            discontinuities,
            jnp.asarray(jnp.inf, dtype=problem.t0.dtype),
        )
    )
    if stepsize_controller is None:
        base_controller = (
            dfx.PIDController(rtol=float(rtol), atol=float(atol))
            if isinstance(selected_solver, dfx.AbstractAdaptiveSolver)
            else dfx.ConstantStepSize()
        )
    else:
        base_controller = stepsize_controller
    if not isinstance(base_controller, dfx.AbstractStepSizeController):
        raise TypeError(
            "stepsize_controller must be a Diffrax AbstractStepSizeController."
        )
    if isinstance(selected_solver, AbstractGeometricSolver):
        if dt0 is None:
            raise ValueError("Geometric delay solvers require an explicit fixed dt0.")
        if not isinstance(base_controller, dfx.ConstantStepSize):
            raise ValueError("Geometric delay solvers require diffrax.ConstantStepSize.")
    if isinstance(problem, NeutralDelayProblem):
        if dt0 is None:
            raise ValueError("NeutralDelayProblem execution requires an explicit dt0.")
        if not isinstance(base_controller, dfx.ConstantStepSize):
            raise ValueError(
                "NeutralDelayProblem execution requires diffrax.ConstantStepSize."
            )

    dynamic_tracker = None
    tracking_mode = "constant-additive-fast-path"
    if dynamic_delays:
        if tracker_depth > 0:
            dynamic_tracker = StateDependentDiscontinuityTracker(
                dynamic_delays,
                schedule_lags,
                tracker_initial_times,
                tracker_initial_generations,
                problem.t1,
                depth=tracker_depth,
                capacity=max_discontinuities,
                root_rtol=root_rtol,
                root_atol=root_atol,
                max_root_iterations=max_root_iterations,
            )
            tracking_mode = (
                "sign-isolated-nonmonotone-roots"
                if any(not delay.monotone_argument for delay in dynamic_delays)
                else "high-order-dynamic-roots"
            )
        else:
            tracking_mode = "tracking-disabled-depth-zero"
    maximum_step = problem.minimum_delay / stage_time_extent
    if isinstance(base_controller, dfx.AbstractAdaptiveStepSizeController):
        if not isinstance(selected_solver, dfx.AbstractAdaptiveSolver):
            raise ValueError(
                "Adaptive step-size controllers require an adaptive Diffrax solver."
            )
        if int(controller_discontinuities.size) == 0:
            clipped_controller = base_controller
        else:
            clipped_controller = dfx.ClipStepSizeController(
                base_controller,
                jump_ts=jax.lax.stop_gradient(controller_discontinuities),
            )
        controller = _CausalAdaptiveStepSizeController(
            controller=clipped_controller,
            maximum_step=maximum_step,
        )
        controller_mode = "adaptive"
    else:
        if not isinstance(base_controller, dfx.ConstantStepSize):
            raise ValueError(
                "Fixed-step delay solves currently require ConstantStepSize."
            )
        controller = _CausalFixedStepSizeController(
            maximum_step=maximum_step,
            jump_ts=jax.lax.stop_gradient(controller_discontinuities),
        )
        controller_mode = "fixed"
    if dynamic_tracker is not None:
        if isinstance(controller, dfx.AbstractAdaptiveStepSizeController):
            controller = StateDependentAdaptiveController(controller, dynamic_tracker)
        else:
            controller = StateDependentFixedController(controller, dynamic_tracker)
    if history_mode == "full":
        assert max_steps is not None
        resolved_history_capacity = max_steps
        retained_maximum_lag = None
    else:
        retained_maximum_lag = execution_plan.maximum_delay
        if retained_maximum_lag is None:
            raise ValueError("Rolling history requires a finite maximum delay.")
        if history_capacity is not None:
            resolved_history_capacity = history_capacity
        elif controller_mode == "adaptive":
            raise ValueError(
                "Adaptive rolling history requires an explicit history_capacity."
            )
        else:
            if dt0 is None:
                raise ValueError(
                    "Fixed-step rolling history requires dt0 when history_capacity "
                    "is inferred."
                )
            if any(
                isinstance(leaf, jax_core.Tracer)
                for leaf in jax.tree.leaves((retained_maximum_lag, dt0))
            ):
                raise ValueError(
                    "Traced rolling solves require an explicit history_capacity."
                )
            resolved_history_capacity = fixed_delay_history_capacity(
                retained_maximum_lag,
                dt0,
                margin=history_margin,
                breakpoints=controller_discontinuities,
                initial_time=problem.t0,
            )
    selected_adjoint = CheckpointedDelayAdjoint() if adjoint is None else adjoint
    if isinstance(selected_adjoint, dfx.BacksolveAdjoint):
        raise ValueError("BacksolveAdjoint is not supported for delay equations.")
    native = _native_delay_solution(
        problem,
        times,
        terms=terms,
        solver=selected_solver,
        stepsize_controller=controller,
        adjoint=selected_adjoint,
        dt0=dt0,
        event=event,
        dense=dense,
        history_mode=history_mode,
        history_capacity=resolved_history_capacity,
        maximum_lag=retained_maximum_lag,
        max_steps=max_steps,
        throw=throw,
        state_adapter=state_adapter,
    )
    if isinstance(native.controller_state, DynamicControllerState):
        dynamic_state = native.controller_state.discontinuities
        tracked_discontinuities = dynamic_state.count
        dynamic_root_count = dynamic_state.num_roots
        internal_restarts = dynamic_state.num_restarts
        tracked_discontinuity_times = dynamic_state.times
        dynamic_root_times = dynamic_state.root_times
    else:
        tracked_discontinuities = jnp.sum(jnp.isfinite(discontinuities))
        dynamic_root_count = jnp.asarray(0, dtype=jnp.int32)
        internal_restarts = jnp.asarray(0, dtype=jnp.int32)
        tracked_discontinuity_times = discontinuities
        dynamic_root_times = jnp.empty((0,), dtype=problem.t0.dtype)
    native_times = jnp.asarray(native.ts["requested"])
    native_states = state_adapter.unpack_values(native.ys["requested"], 1)
    final_time = jnp.asarray(native.ts["final"])[0]
    solver_state = native.solver_state
    rolling_history = None
    if history_mode == "rolling":
        if not isinstance(solver_state, _RetardedSolverState):
            raise RuntimeError("Diffrax did not return rolling retarded solver state.")
        if not isinstance(solver_state.history, RollingDelayHistory):
            raise RuntimeError("Diffrax did not return rolling delay history.")
        rolling_history = solver_state.history
        if throw:
            native_states = eqx.error_if(
                native_states,
                rolling_history.overflowed,
                "Rolling delay history exhausted history_capacity before its lag "
                "window could be pruned.",
            )
    interpolation = None
    if dense:
        if not isinstance(solver_state, _RetardedSolverState):
            raise RuntimeError("Diffrax did not return retarded solver state.")
        history = DelayHistoryView(
            initial_history=_CoordinateDelayHistory(problem.history, state_adapter),
            initial_derivative=(
                None
                if problem.history_derivative is None
                else _CoordinateDelayDerivative(problem.history_derivative, state_adapter)
            ),
            args=state_adapter.pack_args(problem.args),
            initial_time=problem.t0,
            computed_history=solver_state.history,
            state_shape=state_adapter.backend_shape,
            geometry=None,
        )
        interpolation = DelayDenseInterpolation(
            history=history,
            final_time=final_time,
            lower_time=(
                rolling_history.retained_interval[0]
                if rolling_history is not None
                else None
            ),
        )
        if state_adapter.active:
            interpolation = _CoordinateDelayInterpolation(interpolation, state_adapter)
    solver_name = type(selected_solver).__name__
    solver_id, resolved_method = _delay_solver_provenance(
        selected_solver,
        neutral=problem.neutral,
    )
    if isinstance(problem, NeutralDelayProblem):
        solver_id = "solver:diffrax-delay:Euler:transformed-neutral-v1"
        resolved_method = "Euler:transformed-neutral-method-of-steps"
    if dynamic_tracker is not None:
        solver_id = f"{solver_id}:state-dependent"
        resolved_method = f"{resolved_method}:dynamic-discontinuity-tracking"
    elif dynamic_delays:
        resolved_method = f"{resolved_method}:low-order-state-dependent"
    distributed_quadrature = tuple(
        {
            "name": term.name,
            "family": term.quadrature_family,
            "order": term.quadrature_order,
            "node_count": term.node_count,
            "effective_lag_range": term.effective_lag_range,
        }
        for term in problem.delay_terms
        if isinstance(term, DistributedDelay)
    )
    functional_contracts = tuple(
        {
            "name": term.name,
            "lag_interval": (term.minimum_delay, term.maximum_delay),
            "output_kind": term.output_kind,
            "discontinuity_lags": term.discontinuity_lags,
            "infinite_memory": term.infinite_memory,
        }
        for term in problem.delay_terms
        if isinstance(term, FunctionalDelay)
    )
    stats = {
        **native.stats,
        "num_delays": problem.num_delays,
        "history_mode": history_mode,
        "history_capacity": resolved_history_capacity,
        "history_max_occupancy": (
            rolling_history.max_size if rolling_history is not None else None
        ),
        "num_history_evictions": (
            rolling_history.num_evictions if rolling_history is not None else 0
        ),
        "history_capacity_exhausted": (
            rolling_history.overflowed if rolling_history is not None else False
        ),
        "active_history_bytes": (
            rolling_history.allocated_bytes if rolling_history is not None else None
        ),
        "retained_history_interval": (
            jnp.stack(rolling_history.retained_interval)
            if rolling_history is not None
            else None
        ),
        "discontinuity_depth": depth,
        "num_tracked_discontinuities": tracked_discontinuities,
        "num_dynamic_discontinuity_roots": dynamic_root_count,
        "dynamic_discontinuity_root_times": dynamic_root_times,
        "num_internal_discontinuity_restarts": internal_restarts,
        "state_dependent_tracking": tracking_mode,
        "infinite_memory": execution_plan.has_infinite_memory,
        "functional_tracking": (
            "declared-lag-translations"
            if execution_plan.has_functional_delays
            else "not-applicable"
        ),
        "minimum_delay": problem.minimum_delay,
        "stage_time_extent": stage_time_extent,
        "maximum_causal_step": problem.minimum_delay / stage_time_extent,
        "controller_mode": controller_mode,
        "initial_derivative_compatible": problem.initial_derivative_compatible,
        "neutral_discontinuity_horizon": problem.t1 if problem.neutral else None,
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
        "stage_abscissae": (
            selected_solver.stage_abscissae
            if isinstance(selected_solver, AbstractGeometricSolver)
            else None
        ),
    }
    return MemoryEquationSolution(
        times=native_times,
        states=native_states,
        valid=_valid_values(native_times, native_states, sample_ndim=0),
        interpolation=interpolation,
        backend_result=native.result,
        stats=stats,
        event_mask=native.event_mask,
        realization=None,
        state_shape=problem.state_shape,
        solver_name=solver_name,
        solver_id=solver_id,
        resolved_method=resolved_method,
        metadata={
            "problem_id": problem.problem_id,
            "backend": "diffrax",
            "state_coordinate_evidence_id": (
                None
                if state_adapter.evidence is None
                else state_adapter.evidence.evidence_id
            ),
            "delay_mode": (
                "transformed-neutral"
                if isinstance(problem, NeutralDelayProblem)
                else "declared-neutral"
                if problem.neutral
                else "declared-functional-retarded"
                if execution_plan.has_functional_delays
                else "declared-retarded"
            ),
            "state_geometry_id": problem.state_geometry_id,
            "distributed_delay_quadrature": distributed_quadrature,
            "state_dependent_tracking": tracking_mode,
            "declared_discontinuity_sources": declared_sources,
            "tracked_discontinuity_times": tracked_discontinuity_times,
            "dynamic_discontinuity_root_times": dynamic_root_times,
            "functional_delay_contracts": functional_contracts,
            "infinite_memory": execution_plan.has_infinite_memory,
            "history_mode": history_mode,
            "retained_history_interval": (
                jnp.stack(rolling_history.retained_interval)
                if rolling_history is not None
                else None
            ),
            "initial_derivative_source_time": (
                jnp.where(
                    problem.initial_derivative_compatible,
                    jnp.asarray(jnp.inf, dtype=problem.t0.dtype),
                    problem.t0,
                )
                if problem.neutral
                else None
            ),
            "initial_derivative_source_active": (
                ~problem.initial_derivative_compatible if problem.neutral else None
            ),
            "initial_left_derivative": problem.initial_left_derivative,
            "initial_right_derivative": problem.initial_right_derivative,
            "initial_derivative_jump": problem.initial_derivative_jump,
            "initial_derivative_compatible": problem.initial_derivative_compatible,
        },
    )


__all__ = ["solve_diffrax_delay"]
