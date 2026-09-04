#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from math import prod
from typing import Any, ClassVar

import diffrax as dfx
import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import core as jax_core
from jaxtyping import Array, ArrayLike

from ..stochastic._wiener import WienerRealization
from ._delay import (
    _invalid_geometry_tangent,
    ConstantDelay,
    DelayDifferentialProblem,
    DelayValues,
    DelayWienerTerm,
    DistributedDelay,
    FunctionalDelay,
)
from ._delay_adjoint import CheckpointedDelayAdjoint
from ._delay_history import (
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
)
from ._diffrax_backend import (
    _realized_wiener_path,
    _reshape_native_sample_shape,
    _valid_values,
    _validated_realization_interval,
    _validated_stochastic_solver,
)
from ._diffrax_delay_backend import (
    _bind_delay_history,
    _CausalFixedStepSizeController,
    _CoordinateDelayDerivative,
    _CoordinateDelayHistory,
    _CoordinateDelayInterpolation,
    _delay_discontinuity_times,
    _DelayValidation,
    _DelayVectorField,
    _RetardedSolver,
    _RetardedSolverState,
)
from ._diffrax_state_packing import _PreparedDiffraxStateAdapter
from ._geometric import (
    _local_velocity,
    _physical_tangent,
    _term_vector_field,
    AbstractGeometricSolver,
    GeometricODETerm,
    SRKMK,
)
from ._memory import MemoryEquationSolution
from ._save_schedule import validate_save_times


class _OrdinaryStochasticContract(eqx.Module):
    """The ordinary-backend attributes used by its stochastic validators."""

    t0: Array
    t1: Array
    noise_shape: tuple[int, ...] = eqx.field(static=True)
    noise_id: str | None = eqx.field(static=True)
    interpretation: str = eqx.field(static=True)
    additive_noise: bool = eqx.field(static=True)


class _FrozenDelayDiffusion(eqx.Module):
    """Diffusion with one step's delayed observations frozen at its left endpoint."""

    dynamic_terms: Any
    static_terms: eqxi.Static
    memory: DelayValues
    validation: _DelayValidation
    time: Array
    dynamic_args: Any
    static_args: eqxi.Static
    path_sign: Array
    geometry: Any
    state_shape: tuple[int, ...] = eqx.field(static=True)
    tangent_shape: tuple[int, ...] = eqx.field(static=True)
    state_adapter: _PreparedDiffraxStateAdapter

    def __call__(self, state: Array, /) -> Array:
        public_state = self.state_adapter.unpack_state(state)
        terms = eqx.combine(self.dynamic_terms, self.static_terms.value)
        packed_args = eqx.combine(self.dynamic_args, self.static_args.value)
        args = self.state_adapter.unpack_args(packed_args)
        columns = []
        for term in terms:
            value = jnp.asarray(
                term.coefficient(self.time, public_state, self.memory, args)
            )
            expected = self.tangent_shape + term.noise_shape
            if tuple(value.shape) != expected:
                raise ValueError(
                    f"DelayWienerTerm {term.name!r} changed its declared coefficient "
                    f"shape; expected tangent-plus-noise {expected}, got {value.shape}."
                )
            columns.append(value.reshape(self.tangent_shape + (term.noise_size,)))
        coefficient = self.validation.apply(
            self.path_sign * jnp.concatenate(columns, axis=-1)
        )
        if self.geometry is not None:
            invalid = jax.vmap(
                lambda column: _invalid_geometry_tangent(
                    self.geometry,
                    public_state,
                    column,
                    self.tangent_shape,
                ),
                in_axes=-1,
            )(coefficient)
            coefficient = eqx.error_if(
                coefficient,
                jnp.any(invalid),
                "Geometric delay diffusion must be tangent-compatible with "
                "state_geometry.",
            )
        return self.state_adapter.pack_diffusion(
            coefficient,
            (coefficient.shape[-1],),
            output_shape=self.tangent_shape,
        )


class _DelayDiffusionVectorField(eqx.Module):
    """Combined delayed diffusion driven by the realization's single Wiener path."""

    context: _DelayVectorField
    terms: tuple[DelayWienerTerm, ...]
    path_sign: Array
    state_shape: tuple[int, ...] = eqx.field(static=True)
    state_adapter: _PreparedDiffraxStateAdapter

    def freeze(self, time: Array, state: Array, args: Any, /) -> _FrozenDelayDiffusion:
        memory, validation = self.context._memory(time, state, args)
        dynamic_terms, static_terms = eqx.partition(self.terms, eqx.is_array)
        dynamic_args, static_args = eqx.partition(args, eqx.is_array)
        return _FrozenDelayDiffusion(
            dynamic_terms=dynamic_terms,
            static_terms=eqxi.Static(static_terms),
            memory=memory,
            validation=validation,
            time=time,
            dynamic_args=dynamic_args,
            static_args=eqxi.Static(static_args),
            path_sign=self.path_sign,
            state_adapter=self.state_adapter,
            geometry=self.context.geometry,
            tangent_shape=self.context.tangent_shape,
            state_shape=self.state_shape,
        )

    def __call__(self, time: ArrayLike, state: Array, args: Any) -> Array:
        query = jnp.asarray(time)
        return self.freeze(query, state, args)(state)


class _PathConsistentDelayInterpolation(dfx.AbstractLocalInterpolation):
    """Euler local extension evaluated on the integration Wiener path."""

    t0: Array  # ty: ignore[invalid-attribute-override]
    t1: Array  # ty: ignore[invalid-attribute-override]
    y0: Array
    drift: Array
    diffusion: _FrozenDelayDiffusion
    brownian: dfx.AbstractBrownianPath
    heun: ClassVar[bool] = False

    @staticmethod
    def _contract(coefficient: Array, increment: Array, /) -> Array:
        return jnp.tensordot(coefficient, increment, axes=((-1,), (0,)))

    def _value(self, time: Array, left: bool, /) -> Array:
        increment = jnp.asarray(
            self.brownian.evaluate(self.t0, time, left=left, use_levy=False)
        )
        coefficient = self.diffusion(self.y0)
        first = self._contract(coefficient, increment)
        if self.heun:
            predictor = self.y0 + first
            corrected = self._contract(self.diffusion(predictor), increment)
            stochastic = 0.5 * (first + corrected)
        else:
            stochastic = first
        return self.y0 + (time - self.t0) * self.drift + stochastic

    def evaluate(
        self,
        t0: ArrayLike,
        t1: ArrayLike | None = None,
        left: bool = True,
    ) -> Array:
        start = jnp.asarray(t0)
        if t1 is None:
            return self._value(start, left)
        end = jnp.asarray(t1)
        return self._value(end, left) - self._value(start, left)


class _EulerHeunPathConsistentDelayInterpolation(_PathConsistentDelayInterpolation):
    """Stratonovich Euler--Heun local extension on the realized path."""

    heun: ClassVar[bool] = True


class _SRKMKPathConsistentDelayInterpolation(_PathConsistentDelayInterpolation):
    """Intrinsic Stratonovich extension on the realized Wiener path."""

    y1: Array
    geometry: Any
    heun: ClassVar[bool] = True

    def _value(self, time: Array, left: bool, /) -> Array:
        increment = jnp.asarray(
            self.brownian.evaluate(self.t0, time, left=left, use_levy=False)
        )
        drift_tangent = _physical_tangent(
            self.geometry,
            self.y0,
            (time - self.t0) * self.drift,
            "SRKMK delay interpolation drift",
        )
        drift_local = _local_velocity(
            self.geometry,
            self.y0,
            self.y0,
            drift_tangent,
        )
        first_tangent = self._contract(self.diffusion(self.y0), increment)
        first_tangent = _physical_tangent(
            self.geometry,
            self.y0,
            first_tangent,
            "SRKMK delay interpolation diffusion",
        )
        diffusion_local = _local_velocity(
            self.geometry,
            self.y0,
            self.y0,
            first_tangent,
        )
        predictor = self.geometry.retract(self.y0, diffusion_local)
        corrected_tangent = self._contract(self.diffusion(predictor), increment)
        corrected_tangent = _physical_tangent(
            self.geometry,
            predictor,
            corrected_tangent,
            "SRKMK delay interpolation corrected diffusion",
        )
        corrected_local = _local_velocity(
            self.geometry,
            self.y0,
            predictor,
            corrected_tangent,
        )
        interior = self.geometry.retract(
            self.y0,
            drift_local + 0.5 * (diffusion_local + corrected_local),
        )
        return jnp.where(
            time <= self.t0,
            self.y0,
            jnp.where(time >= self.t1, self.y1, interior),
        )


class _PathConsistentInterpolationFactory(eqx.Module):
    """Rebuild a Wiener path from float-encoded key bits for dense evaluation."""

    path_shape: Any = eqx.field(static=True)
    path_levy_area: type = eqx.field(static=True)
    path_key_impl: Any = eqx.field(static=True)
    heun: bool = eqx.field(static=True)
    geometry: Any

    def __call__(
        self,
        *,
        t0,
        t1,
        y0,
        drift,
        diffusion,
        path_t0,
        path_t1,
        path_tolerance,
        path_key_bits,
        y1=None,
    ):
        key_data = jax.lax.bitcast_convert_type(path_key_bits, jnp.uint32)
        path_key = jr.wrap_key_data(key_data, impl=self.path_key_impl)
        brownian = dfx.VirtualBrownianTree(
            t0=path_t0,
            t1=path_t1,
            tol=path_tolerance,
            shape=self.path_shape,
            key=path_key,
            levy_area=self.path_levy_area,
        )
        if self.geometry is not None:
            if y1 is None:
                raise RuntimeError("SRKMK delay interpolation requires its endpoint.")
            return _SRKMKPathConsistentDelayInterpolation(
                t0=t0,
                t1=t1,
                y0=y0,
                y1=y1,
                drift=drift,
                diffusion=diffusion,
                brownian=brownian,
                geometry=self.geometry,
            )
        interpolation_type = (
            _EulerHeunPathConsistentDelayInterpolation
            if self.heun
            else _PathConsistentDelayInterpolation
        )
        return interpolation_type(
            t0=t0,
            t1=t1,
            y0=y0,
            drift=drift,
            diffusion=diffusion,
            brownian=brownian,
        )


class _VectorizedDelayDenseInterpolation(eqx.Module):
    """One accepted-history interpolation for every realization sample."""

    solver_states: _RetardedSolverState
    final_times: Array
    lower_times: Array
    initial_history: Any
    initial_derivative: Any
    args: Any
    initial_time: Array
    geometry: Any
    sample_shape: tuple[int, ...] = eqx.field(static=True)
    state_shape: tuple[int, ...] = eqx.field(static=True)
    state_adapter: _PreparedDiffraxStateAdapter

    backend_tangent_shape: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        solver_states: _RetardedSolverState,
        final_times: Array,
        problem: DelayDifferentialProblem,
        sample_shape: tuple[int, ...],
        state_adapter: _PreparedDiffraxStateAdapter,
        /,
    ):
        samples = tuple(int(size) for size in sample_shape)
        sample_ndim = len(samples)
        sample_count = prod(samples)

        def flatten(value):
            if eqx.is_array(value):
                return value.reshape((sample_count,) + value.shape[sample_ndim:])
            return value

        self.solver_states = jax.tree.map(flatten, solver_states)
        self.final_times = jnp.asarray(final_times).reshape((-1,))
        history = solver_states.history
        if isinstance(history, RollingDelayHistory):
            self.lower_times = jnp.asarray(history.retained_interval[0]).reshape((-1,))
        else:
            self.lower_times = jnp.broadcast_to(problem.t0, (sample_count,))
        self.initial_history = _CoordinateDelayHistory(problem.history, state_adapter)
        self.initial_derivative = (
            None
            if problem.history_derivative is None
            else _CoordinateDelayDerivative(
                problem.history_derivative,
                state_adapter,
                problem.tangent_shape,
            )
        )
        self.args = state_adapter.pack_args(problem.args)
        self.initial_time = problem.t0
        self.geometry = None
        self.sample_shape = samples
        self.state_shape = state_adapter.backend_shape
        packed_zero = state_adapter.pack_tangent(
            jnp.zeros(problem.tangent_shape, dtype=problem.initial_state.dtype),
            problem.tangent_shape,
        )
        self.backend_tangent_shape = tuple(int(size) for size in packed_zero.shape)
        self.state_adapter = state_adapter

    @eqx.filter_jit
    def evaluate(
        self,
        query_times: ArrayLike,
        /,
        *,
        left: bool = True,
    ) -> Array:
        if not isinstance(left, bool):
            raise TypeError("left must be a bool.")
        query = jnp.asarray(query_times)
        if jnp.iscomplexobj(query):
            raise TypeError("Dense delay query times must be real-valued.")
        if query.size == 0:
            raise ValueError("Dense delay query times must be non-empty.")
        query = query.astype(float)
        query = eqx.error_if(
            query,
            ~jnp.all(jnp.isfinite(query)),
            "Dense delay query times must be finite.",
        )
        query = eqx.error_if(
            query,
            jnp.any(
                (query < jnp.max(self.lower_times)) | (query > jnp.min(self.final_times))
            ),
            "Dense delay query times must lie within every solution interval.",
        )

        def evaluate_one(solver_state):
            history = DelayHistoryView(
                initial_history=self.initial_history,
                initial_derivative=self.initial_derivative,
                args=self.args,
                initial_time=self.initial_time,
                computed_history=solver_state.history,
                state_shape=self.state_shape,
                geometry=self.geometry,
                derivative_shape=self.backend_tangent_shape,
            )
            return history.values(query, left=left)

        values = eqx.filter_vmap(evaluate_one)(self.solver_states)
        public = self.state_adapter.unpack_values(values, 1 + query.ndim)
        return public.reshape(
            self.sample_shape + query.shape + self.state_adapter.state_shape
        )


def _underlying_control_term(term: Any, /) -> dfx.ControlTerm:
    """Extract the unique ControlTerm below Diffrax direction wrappers."""
    leaves = jax.tree.leaves(
        term,
        is_leaf=lambda value: isinstance(value, dfx.ControlTerm),
    )
    controls = tuple(value for value in leaves if isinstance(value, dfx.ControlTerm))
    if len(controls) != 1:
        raise TypeError(
            "Certified stochastic delay diffusion requires exactly one ControlTerm."
        )
    return controls[0]


def _brownian_dense_info(
    control: dfx.AbstractPath,
    path_key: Array,
    /,
) -> dict[str, Array]:
    if not isinstance(control, dfx.VirtualBrownianTree):
        raise TypeError(
            "Certified stochastic delay history requires VirtualBrownianTree."
        )
    key_data = jr.key_data(path_key)
    return {
        "path_t0": jnp.asarray(control.t0),
        "path_t1": jnp.asarray(control.t1),
        "path_tolerance": jnp.asarray(control.tol)
        * (jnp.asarray(control.t1) - jnp.asarray(control.t0)),
        "path_key_bits": jax.lax.bitcast_convert_type(key_data, jnp.float32),
    }


class _StochasticRetardedSolver(_RetardedSolver):
    """Retarded stochastic solver with a path-consistent accepted history."""

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

    def _step_with_extension(
        self,
        terms,
        t0,
        t1,
        y0,
        args,
        inner_state,
        made_jump,
    ):
        y1, y_error, _, next_state, result = self.solver.step(
            terms,
            t0,
            t1,
            y0,
            args,
            inner_state,
            made_jump,
        )
        drift_term, diffusion_term = terms.terms
        control_term = _underlying_control_term(diffusion_term)
        vector_field = control_term.vector_field
        if not isinstance(vector_field, _DelayDiffusionVectorField):
            raise TypeError("Certified stochastic delay terms require delayed diffusion.")
        dense_info = {
            "y0": y0,
            "drift": jnp.asarray(_term_vector_field(drift_term, t0, y0, args)),
            "diffusion": vector_field.freeze(t0, y0, args),
            **_brownian_dense_info(control_term.control, self.path_key),
        }
        if self.geometry is not None:
            dense_info["y1"] = y1
        return y1, y_error, dense_info, next_state, result

    def init(self, terms, t0, t1, y0, args):
        provisional_state = self.solver.init(terms, t0, t1, y0, args)
        _, _, dense_info_structure, _, _ = eqx.filter_eval_shape(
            self._step_with_extension,
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
        inner_state = self.solver.init(bound_terms, t0, t1, y0, args)
        return _RetardedSolverState(inner_state=inner_state, history=history)

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
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


class _ItoStochasticRetardedSolver(
    _StochasticRetardedSolver,
    dfx.AbstractItoSolver,
):
    """Itô marker for the certified Euler retarded wrapper."""


class _StratonovichStochasticRetardedSolver(
    _StochasticRetardedSolver,
    dfx.AbstractStratonovichSolver,
):
    """Stratonovich marker for the certified Euler--Heun retarded wrapper."""


def _validation_contract(
    problem: DelayDifferentialProblem, /
) -> _OrdinaryStochasticContract:
    return _OrdinaryStochasticContract(
        t0=problem.t0,
        t1=problem.t1,
        noise_shape=problem.noise_shape,
        noise_id=problem.noise_id,
        interpretation=problem.interpretation,
        additive_noise=all(term.structure == "additive" for term in problem.wiener_terms),
    )


def _validated_solver(
    problem: DelayDifferentialProblem,
    solver: Any,
    realization: WienerRealization,
    /,
) -> tuple[dfx.AbstractSolver, bool]:
    if not isinstance(solver, dfx.AbstractSolver):
        raise TypeError("solver must be a Diffrax AbstractSolver or None.")
    if (
        not isinstance(solver, SRKMK)
        and solver.interpolation_cls is not dfx.LocalLinearInterpolation
    ):
        raise ValueError(
            "The stochastic delay solver interpolation does not satisfy the "
            "certified Euler local-history contract."
        )
    geometric = isinstance(solver, SRKMK)
    if problem.interpretation == "ito":
        if type(solver) is not dfx.Euler:
            raise ValueError(
                "Certified Itô stochastic delay execution requires diffrax.Euler."
            )
        heun = False
    else:
        if type(solver) is not dfx.EulerHeun and not geometric:
            raise ValueError(
                "Certified Stratonovich stochastic delay execution requires "
                "diffrax.EulerHeun or phydrax.solver.SRKMK."
            )
        heun = True
    contract = _validation_contract(problem)
    _validated_stochastic_solver(contract, solver, realization)
    _validated_realization_interval(contract, realization)
    return solver, heun


def _discontinuities(
    problem: DelayDifferentialProblem,
    initial_discontinuities: ArrayLike | Sequence[float] | None,
    /,
    *,
    max_discontinuities: int,
) -> Array:
    if initial_discontinuities is None:
        sources = problem.t0.reshape((1,))
    else:
        sources = jnp.asarray(initial_discontinuities, dtype=problem.t0.dtype)
        if sources.ndim != 1:
            raise ValueError("initial_discontinuities must be a rank-1 array or None.")
        sources = eqx.error_if(
            sources,
            ~jnp.all(jnp.isfinite(sources)),
            "initial_discontinuities must be finite.",
        )
    lag_values = []
    for term in problem.delay_terms:
        if isinstance(term, ConstantDelay):
            lag_values.append(term.delay)
        elif isinstance(term, DistributedDelay):
            lag_values.extend(tuple(term.nodes))
        elif isinstance(term, FunctionalDelay):
            lag_values.extend(tuple(term.discontinuity_lags))
    lags = (
        jnp.stack(tuple(lag_values))
        if lag_values
        else jnp.empty((0,), dtype=problem.t0.dtype)
    )
    return _delay_discontinuity_times(
        lags,
        sources,
        depth=1,
        max_discontinuities=max_discontinuities,
    )


def _native_stochastic_delay_solution(
    problem: DelayDifferentialProblem,
    save_times: Array,
    *,
    path_key: Array,
    path_sign: Array,
    realization: WienerRealization,
    integration_start: Array,
    integration_end: Array,
    solver: dfx.AbstractSolver,
    heun: bool,
    controller: dfx.AbstractStepSizeController,
    adjoint: Any,
    dt0: ArrayLike,
    event: Any | None,
    dense: bool,
    history_mode: DelayHistoryMode,
    history_capacity: int,
    maximum_lag: Array | None,
    max_steps: int | None,
    throw: bool,
    state_adapter: _PreparedDiffraxStateAdapter,
):
    real_dtype = jnp.asarray(problem.initial_state).real.dtype
    brownian, signed_path = _realized_wiener_path(
        realization,
        path_key,
        path_sign,
        real_dtype,
    )
    packed_initial = state_adapter.pack_state(
        problem.initial_state, owner="Initial stochastic delay state"
    )
    packed_derivative = state_adapter.pack_tangent(
        jnp.zeros(problem.tangent_shape, dtype=problem.initial_state.dtype),
        problem.tangent_shape,
        owner="Initial stochastic delay derivative",
    )
    empty_history = EmptyDelayHistory(
        packed_initial,
        packed_derivative,
    )
    delay_context = _DelayVectorField(
        function=problem.drift,
        initial_history=_CoordinateDelayHistory(problem.history, state_adapter),
        initial_derivative=(
            None
            if problem.history_derivative is None
            else _CoordinateDelayDerivative(
                problem.history_derivative,
                state_adapter,
                problem.tangent_shape,
            )
        ),
        delay_terms=problem.delay_terms,
        initial_time=integration_start,
        state_shape=problem.state_shape,
        tangent_shape=problem.tangent_shape,
        geometry=problem.state_geometry,
        state_adapter=state_adapter,
        backend_shape=state_adapter.backend_shape,
        backend_tangent_shape=tuple(int(size) for size in packed_derivative.shape),
        computed_history=empty_history,
    )
    diffusion_field = _DelayDiffusionVectorField(
        context=delay_context,
        terms=problem.wiener_terms,
        path_sign=signed_path,
        state_shape=problem.state_shape,
        state_adapter=state_adapter,
    )
    drift_term = (
        GeometricODETerm(delay_context)
        if isinstance(solver, AbstractGeometricSolver)
        else dfx.ODETerm(delay_context)
    )
    terms = dfx.MultiTerm(
        drift_term,
        dfx.ControlTerm(diffusion_field, brownian),
    )
    wrapper_type = (
        _StratonovichStochasticRetardedSolver if heun else _ItoStochasticRetardedSolver
    )
    wrapped_solver = wrapper_type(
        solver=solver,
        history_capacity=history_capacity,
        history_mode=history_mode,
        maximum_lag=maximum_lag,
        heun=heun,
        path_key=path_key,
        path_shape=brownian.shape,
        path_levy_area=brownian.levy_area,
        path_key_impl=jr.key_impl(path_key),
        geometry=solver.geometry if isinstance(solver, SRKMK) else None,
    )
    saveat = dfx.SaveAt(
        subs={
            "requested": dfx.SubSaveAt(ts=save_times),
            "final": dfx.SubSaveAt(t1=True),
        },
        solver_state=dense or history_mode == "rolling",
    )
    return dfx.diffeqsolve(
        terms,
        wrapped_solver,
        t0=integration_start,
        t1=integration_end,
        dt0=dt0,
        y0=packed_initial,
        args=state_adapter.pack_args(problem.args),
        saveat=saveat,
        stepsize_controller=controller,
        adjoint=adjoint,
        event=state_adapter.wrap_event(event),
        max_steps=max_steps,
        throw=bool(throw and history_mode == "full"),
    )


def _solve_diffrax_delay_stochastic(
    problem: DelayDifferentialProblem,
    /,
    *,
    save_times: ArrayLike,
    realization: WienerRealization | None,
    solver: Any | None,
    stepsize_controller: Any | None,
    adjoint: Any | None,
    dt0: ArrayLike | None,
    event: Any | None,
    rtol: float,
    atol: float,
    dense: bool,
    max_steps: int | None,
    initial_discontinuities: ArrayLike | Sequence[float] | None,
    discontinuity_depth: int | None,
    max_discontinuities: int,
    history_mode: DelayHistoryMode,
    history_capacity: int | None,
    history_margin: int,
    throw: bool,
    state_adapter: _PreparedDiffraxStateAdapter,
) -> MemoryEquationSolution:
    """Execute one certified fixed-step stochastic retarded problem."""
    del rtol, atol
    if realization is None:
        raise ValueError("Stochastic delay problems require a WienerRealization.")
    if not isinstance(realization, WienerRealization):
        raise TypeError("realization must be a WienerRealization or None.")
    if dt0 is None:
        raise ValueError("Stochastic Diffrax delay solves require an explicit dt0.")
    if problem.neutral:
        raise ValueError("Stochastic neutral delay terms are not supported.")
    if discontinuity_depth not in (None, 1):
        raise ValueError(
            "Fixed-step stochastic delay execution requires discontinuity_depth=1 "
            "or None."
        )

    selected_solver = resolve_delay_solver(problem, solver)
    execution_plan = compile_delay_execution_plan(
        problem,
        selected_solver,
        execution="whole",
        history_mode=history_mode,
    )
    selected_solver, heun = _validated_solver(problem, selected_solver, realization)
    integration_start, integration_end = _validated_realization_interval(
        _validation_contract(problem),
        realization,
    )
    selected_adjoint = CheckpointedDelayAdjoint() if adjoint is None else adjoint
    if isinstance(selected_adjoint, dfx.BacksolveAdjoint):
        raise ValueError("BacksolveAdjoint is not supported for delay equations.")
    base_controller = (
        dfx.ConstantStepSize() if stepsize_controller is None else stepsize_controller
    )
    if not isinstance(base_controller, dfx.ConstantStepSize):
        raise ValueError(
            "Certified stochastic delay solvers require diffrax.ConstantStepSize."
        )

    times = validate_save_times(problem.t0, problem.t1, save_times)
    step = jnp.asarray(dt0)
    if step.shape != () or jnp.iscomplexobj(step):
        raise ValueError("dt0 must be a real scalar.")
    step = eqx.error_if(
        step,
        ~jnp.isfinite(step) | (step <= 0.0) | (step > problem.minimum_delay),
        "dt0 must be positive and no larger than the causal delay step bound.",
    )
    step = eqx.error_if(
        step,
        jnp.abs(step) <= realization.tolerance,
        "WienerRealization tolerance must be strictly smaller than the fixed "
        "integration step.",
    )
    discontinuities = _discontinuities(
        problem,
        initial_discontinuities,
        max_discontinuities=max_discontinuities,
    )
    controller = _CausalFixedStepSizeController(
        maximum_step=problem.minimum_delay,
        jump_ts=jax.lax.stop_gradient(discontinuities),
    )

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
        else:
            if any(
                isinstance(leaf, jax_core.Tracer)
                for leaf in jax.tree.leaves((retained_maximum_lag, step))
            ):
                raise ValueError(
                    "Traced rolling solves require an explicit history_capacity."
                )
            resolved_history_capacity = fixed_delay_history_capacity(
                retained_maximum_lag,
                step,
                margin=history_margin,
                breakpoints=discontinuities,
                initial_time=problem.t0,
            )

    def one(path_key: Array, path_sign: Array):
        return _native_stochastic_delay_solution(
            problem,
            times,
            path_key=path_key,
            path_sign=path_sign,
            realization=realization,
            integration_start=integration_start,
            integration_end=integration_end,
            solver=selected_solver,
            heun=heun,
            controller=controller,
            adjoint=selected_adjoint,
            dt0=step,
            event=event,
            dense=dense,
            max_steps=max_steps,
            throw=throw,
            history_mode=history_mode,
            history_capacity=resolved_history_capacity,
            maximum_lag=retained_maximum_lag,
            state_adapter=state_adapter,
        )

    if realization.sample_shape:
        count = realization.num_paths
        key_shape = tuple(realization.root_key.shape)
        keys = realization.path_keys.reshape((count,) + key_shape)
        signs = realization.path_signs.reshape((count,))
        native = _reshape_native_sample_shape(
            eqx.filter_vmap(one)(keys, signs),
            realization.sample_shape,
        )
        sample_ndim = len(realization.sample_shape)
    else:
        native = one(realization.path_keys, realization.path_signs)
        sample_ndim = 0

    solver_state = native.solver_state
    rolling_history = None
    if history_mode == "rolling":
        if not isinstance(solver_state, _RetardedSolverState):
            raise RuntimeError("Diffrax did not return rolling retarded solver state.")
        if not isinstance(solver_state.history, RollingDelayHistory):
            raise RuntimeError("Diffrax did not return rolling delay history.")
        rolling_history = solver_state.history
        if realization.sample_shape:
            sample_ndim = len(realization.sample_shape)

            def first_sample(value):
                if eqx.is_array(value):
                    return value[(0,) * sample_ndim]
                return value

            history_bytes = jax.tree.map(first_sample, rolling_history).allocated_bytes
        else:
            history_bytes = rolling_history.allocated_bytes
    else:
        history_bytes = None
    native_states = state_adapter.unpack_values(
        native.ys["requested"],
        len(realization.sample_shape) + 1,
    )
    if rolling_history is not None and throw:
        native_states = eqx.error_if(
            native_states,
            jnp.any(rolling_history.overflowed),
            "Rolling delay history exhausted history_capacity before its lag "
            "window could be pruned.",
        )
    interpolation = None
    if dense:
        if not isinstance(solver_state, _RetardedSolverState):
            raise RuntimeError("Diffrax did not return stochastic retarded solver state.")
        if realization.sample_shape:
            interpolation = _VectorizedDelayDenseInterpolation(
                solver_state,
                jnp.asarray(native.ts["final"])[..., 0],
                problem,
                realization.sample_shape,
                state_adapter,
            )
        else:
            packed_derivative = state_adapter.pack_tangent(
                jnp.zeros(problem.tangent_shape, dtype=problem.initial_state.dtype),
                problem.tangent_shape,
                owner="Initial stochastic delay derivative",
            )
            final_time = jnp.asarray(native.ts["final"])[0]
            history = DelayHistoryView(
                initial_history=_CoordinateDelayHistory(problem.history, state_adapter),
                initial_derivative=(
                    None
                    if problem.history_derivative is None
                    else _CoordinateDelayDerivative(
                        problem.history_derivative,
                        state_adapter,
                        problem.tangent_shape,
                    )
                ),
                args=state_adapter.pack_args(problem.args),
                initial_time=problem.t0,
                computed_history=solver_state.history,
                state_shape=state_adapter.backend_shape,
                derivative_shape=tuple(int(size) for size in packed_derivative.shape),
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
                interpolation = _CoordinateDelayInterpolation(
                    interpolation,
                    state_adapter,
                    problem.tangent_shape,
                )

    solver_name = type(selected_solver).__name__
    if isinstance(selected_solver, AbstractGeometricSolver):
        extension = "srkmk-wiener-path"
        solver_id = selected_solver.solver_id
        resolved_method = selected_solver.resolved_method
    else:
        extension = "euler-heun-wiener-path" if heun else "euler-maruyama-wiener-path"
        solver_id = f"solver:diffrax-delay-stochastic:{solver_name}:retarded-v1"
        resolved_method = f"{solver_name}:causal-{extension}"
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
        "active_history_bytes": history_bytes,
        "retained_history_interval": (
            jnp.stack(rolling_history.retained_interval, axis=-1)
            if rolling_history is not None
            else None
        ),
        "discontinuity_depth": 1,
        "num_tracked_discontinuities": int(discontinuities.size),
        "state_dependent_tracking": (
            "first-order-pathwise-untracked"
            if problem.has_state_dependent_delays
            else "not-applicable"
        ),
        "functional_tracking": (
            "declared-lag-translations"
            if execution_plan.has_functional_delays
            else "not-applicable"
        ),
        "minimum_delay": problem.minimum_delay,
        "stage_time_extent": execution_plan.stage_time_extent,
        "maximum_causal_step": problem.minimum_delay,
        "controller_mode": "fixed",
        "continuous_extension": extension,
    }
    return MemoryEquationSolution(
        times=times,
        states=native_states,
        valid=_valid_values(times, native_states, sample_ndim=sample_ndim),
        interpolation=interpolation,
        backend_result=native.result,
        stats=stats,
        event_mask=native.event_mask,
        realization=realization,
        state_shape=problem.state_shape,
        solver_name=solver_name,
        solver_id=solver_id,
        resolved_method=resolved_method,
        metadata={
            "problem_id": problem.problem_id,
            "backend": "diffrax",
            "delay_mode": (
                "declared-functional-retarded-stochastic"
                if execution_plan.has_functional_delays
                else "declared-retarded-stochastic"
            ),
            "state_dependent_tracking": (
                "first-order-pathwise-untracked"
                if problem.has_state_dependent_delays
                else "not-applicable"
            ),
            "infinite_memory": execution_plan.has_infinite_memory,
            "distributed_delay_quadrature": tuple(
                {
                    "name": term.name,
                    "family": term.quadrature_family,
                    "order": term.quadrature_order,
                    "node_count": term.node_count,
                    "effective_lag_range": term.effective_lag_range,
                }
                for term in problem.delay_terms
                if isinstance(term, DistributedDelay)
            ),
            "functional_delay_contracts": tuple(
                {
                    "name": term.name,
                    "lag_interval": (term.minimum_delay, term.maximum_delay),
                    "output_kind": term.output_kind,
                    "discontinuity_lags": term.discontinuity_lags,
                    "infinite_memory": term.infinite_memory,
                }
                for term in problem.delay_terms
                if isinstance(term, FunctionalDelay)
            ),
            "state_geometry_id": problem.state_geometry_id,
            "driver_family": "wiener",
            "interpretation": problem.interpretation,
            "noise_id": problem.noise_id,
            "wiener_term_names": tuple(term.name for term in problem.wiener_terms),
            "wiener_term_slices": problem.wiener_term_slices,
            "noise_structures": tuple(term.structure for term in problem.wiener_terms),
            "basis_ids": tuple(term.basis_id for term in problem.wiener_terms),
            "realization_id": realization.realization_id,
            "coupling_id": realization.coupling_id,
            "levy_area": realization.levy_area,
            "continuous_extension": extension,
            "history_mode": history_mode,
            "retained_history_interval": (
                jnp.stack(rolling_history.retained_interval, axis=-1)
                if rolling_history is not None
                else None
            ),
        },
    )


__all__: list[str] = []
