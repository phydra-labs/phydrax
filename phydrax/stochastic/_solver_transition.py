#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, ArrayLike

from ._process import AbstractPathwiseTransition
from ._state_space import (
    AbstractTransitionKernel,
    StateSpaceStepContext,
    TransitionSample,
)
from ._state_space_input import SampledStateSpaceInput


JumpTransitionAlgorithm: TypeAlias = Literal["next_reaction", "direct_ssa"]


def _shape(value: Sequence[int], /, *, owner: str) -> tuple[int, ...]:
    shape = tuple(int(size) for size in value)
    if any(size <= 0 for size in shape):
        raise ValueError(f"{owner} dimensions must be positive.")
    return shape


def _name(value: str, /, *, owner: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{owner} must be a non-empty string.")
    return value


def _interval(
    t0: ArrayLike, t1: ArrayLike, /
) -> tuple[Array, Array, tuple[float, float]]:
    start = jnp.asarray(t0, dtype=float)
    end = jnp.asarray(t1, dtype=float)
    if start.shape != () or end.shape != ():
        raise ValueError("Transition solver times must be scalar.")
    support = (float(start), float(end))
    if not np.isfinite(support).all() or not support[1] > support[0]:
        raise ValueError("Transition solver times require finite t1 > t0.")
    return start, end, support


def _array_interval(t0: ArrayLike, t1: ArrayLike, /) -> tuple[Array, Array]:
    start = jnp.asarray(t0, dtype=float)
    end = jnp.asarray(t1, dtype=float)
    if start.shape != () or end.shape != ():
        raise ValueError("Transition solver times must be scalar.")
    end = eqx.error_if(
        end,
        ~(jnp.isfinite(start) & jnp.isfinite(end) & (end > start)),
        "Transition solver times require finite t1 > t0.",
    )
    return start, end


def _transition_sample(
    values: Array,
    valid: Array,
    status: Array,
    /,
    *,
    process_id: str,
    approximation_id: str,
) -> TransitionSample:
    return TransitionSample(
        values=jnp.asarray(values),
        valid=jnp.asarray(valid, dtype=bool),
        status=jnp.asarray(status, dtype=jnp.int32),
        process_id=process_id,
        approximation_id=approximation_id,
    )


def _input_controller(
    context: StateSpaceStepContext,
    controller: Any,
    dt0: Array | None,
    start: Array,
    end: Array,
    /,
    *,
    stochastic: bool,
    rtol: float,
    atol: float,
) -> tuple[Any, Array | None]:
    """Force solver steps across every declared exogenous-input breakpoint."""
    import diffrax as dfx

    breakpoint_mask = np.asarray(context.input_breakpoint_valid, dtype=bool)
    if not np.any(breakpoint_mask):
        return controller, dt0
    breakpoints = np.asarray(context.input_breakpoints, dtype=float)[breakpoint_mask]
    signal = context.input_signal
    discontinuous = (
        isinstance(signal, SampledStateSpaceInput)
        and signal.interpolation == "zero-order-hold"
    )

    resolved = controller
    if resolved is None and not stochastic:
        resolved = dfx.PIDController(rtol=rtol, atol=atol)
    if isinstance(resolved, dfx.AbstractAdaptiveStepSizeController):
        return (
            dfx.ClipStepSizeController(
                resolved,
                jump_ts=breakpoints if discontinuous else None,
                step_ts=None if discontinuous else breakpoints,
            ),
            dt0,
        )

    if resolved is None:
        resolved = dfx.ConstantStepSize()
    if isinstance(resolved, dfx.StepTo):
        base_times = np.asarray(resolved.ts, dtype=float)
    elif isinstance(resolved, dfx.ConstantStepSize):
        if dt0 is None:
            raise ValueError(
                "Fixed-step input breakpoints require an explicit transition dt0."
            )
        start_value = float(start)
        end_value = float(end)
        step = abs(float(dt0))
        step_count = int(np.ceil((end_value - start_value) / step))
        base_times = np.linspace(start_value, end_value, step_count + 1)
    else:
        raise TypeError(
            "Input breakpoints require an adaptive, ConstantStepSize, or StepTo "
            "transition controller."
        )

    if discontinuous:
        breakpoint_steps = np.concatenate(
            (
                np.nextafter(breakpoints, -np.inf),
                np.nextafter(breakpoints, np.inf),
            )
        )
    else:
        breakpoint_steps = breakpoints
    schedule = np.unique(
        np.concatenate(
            (
                base_times,
                breakpoint_steps,
                np.asarray([float(start), float(end)]),
            )
        )
    )
    schedule = schedule[(schedule >= float(start)) & (schedule <= float(end))]
    return dfx.StepTo(ts=jnp.asarray(schedule, dtype=start.dtype)), None


class DifferentialTransitionKernel(AbstractTransitionKernel):
    """One-interval ODE/SDE transition evaluated by the canonical Diffrax backend."""

    drift: Callable
    wiener_terms: tuple[Any, ...]
    solver: Any
    stepsize_controller: Any
    adjoint: Any
    dt0: Array | None
    state_shape: tuple[int, ...] = eqx.field(static=True)
    process_id: str = eqx.field(static=True)
    approximation_id: str = eqx.field(static=True)
    has_log_density: bool = eqx.field(static=True)
    interpretation: str = eqx.field(static=True)
    wiener_tolerance: float = eqx.field(static=True)
    rtol: float = eqx.field(static=True)
    atol: float = eqx.field(static=True)
    max_steps: int = eqx.field(static=True)

    def __init__(
        self,
        drift: Callable,
        /,
        *,
        state_shape: Sequence[int],
        process_id: str,
        wiener_terms: Sequence[Any] = (),
        interpretation: str = "ito",
        solver: Any = None,
        stepsize_controller: Any = None,
        adjoint: Any = None,
        dt0: ArrayLike | None = None,
        wiener_tolerance: float = 1e-3,
        rtol: float = 1e-6,
        atol: float = 1e-8,
        max_steps: int = 4096,
        approximation_id: str = "diffrax-transition",
    ):
        from ..solver import WienerTerm

        if not callable(drift):
            raise TypeError("drift must be callable.")
        terms = tuple(wiener_terms)
        if any(not isinstance(term, WienerTerm) for term in terms):
            raise TypeError("wiener_terms must contain only WienerTerm objects.")
        if interpretation not in ("ito", "stratonovich"):
            raise ValueError("interpretation must be 'ito' or 'stratonovich'.")
        tolerance = float(wiener_tolerance)
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("wiener_tolerance must be finite and positive.")
        if terms and dt0 is None:
            raise ValueError("Stochastic differential transitions require explicit dt0.")
        resolved_dt0 = None if dt0 is None else jnp.asarray(dt0, dtype=float)
        if resolved_dt0 is not None and (
            resolved_dt0.shape != () or not bool(jnp.isfinite(resolved_dt0))
        ):
            raise ValueError("dt0 must be a finite scalar or None.")
        if terms and resolved_dt0 is not None and abs(float(resolved_dt0)) <= tolerance:
            raise ValueError("wiener_tolerance must be smaller than abs(dt0).")
        if int(max_steps) < 1:
            raise ValueError("max_steps must be positive.")
        self.drift = drift
        self.wiener_terms = terms
        self.solver = solver
        self.stepsize_controller = stepsize_controller
        self.adjoint = adjoint
        self.dt0 = resolved_dt0
        self.state_shape = _shape(state_shape, owner="state_shape")
        self.process_id = _name(process_id, owner="process_id")
        self.approximation_id = _name(approximation_id, owner="approximation_id")
        self.has_log_density = False
        self.interpretation = interpretation
        self.wiener_tolerance = tolerance
        self.rtol = float(rtol)
        self.atol = float(atol)
        self.max_steps = int(max_steps)

    def sample(self, key, state, t0, t1, context, /) -> TransitionSample:
        from ..solver import DifferentialProblem, solve_diffrax
        from ._wiener import WienerRealization

        state_array = jnp.asarray(state)
        if state_array.shape != self.state_shape:
            raise ValueError(
                f"state must have shape {self.state_shape}; got {state_array.shape}."
            )
        start, end, support = _interval(t0, t1)
        controller, dt0 = _input_controller(
            context,
            self.stepsize_controller,
            self.dt0,
            start,
            end,
            stochastic=bool(self.wiener_terms),
            rtol=self.rtol,
            atol=self.atol,
        )
        problem = DifferentialProblem(
            self.drift,
            state_array,
            t0=start,
            t1=end,
            args=context,
            wiener_terms=self.wiener_terms,
            interpretation=self.interpretation,
        )
        realization = (
            WienerRealization(
                key,
                problem.noise_shape,
                support=support,
                tolerance=self.wiener_tolerance,
                noise_id=problem.noise_id,
                label=f"{self.process_id}:transition",
            )
            if problem.stochastic
            else None
        )
        solution = solve_diffrax(
            problem,
            save_times=jnp.asarray([end]),
            realization=realization,
            solver=self.solver,
            stepsize_controller=controller,
            adjoint=self.adjoint,
            dt0=dt0,
            rtol=self.rtol,
            atol=self.atol,
            max_steps=self.max_steps,
            throw=False,
        )
        values = solution.states[-1]
        valid = context.input_valid & solution.valid[-1] & jnp.all(jnp.isfinite(values))
        return _transition_sample(
            values,
            valid,
            jnp.where(valid, 0, 1),
            process_id=self.process_id,
            approximation_id=self.approximation_id,
        )

    def log_prob(self, next_state, state, t0, t1, context, /) -> Array:
        del next_state, state, t0, t1, context
        raise ValueError("Differential solver transitions do not provide a density.")


class JumpTransitionKernel(AbstractTransitionKernel):
    """One-interval finite-activity pure-jump transition."""

    process: Any
    state_shape: tuple[int, ...] = eqx.field(static=True)
    process_id: str = eqx.field(static=True)
    approximation_id: str = eqx.field(static=True)
    has_log_density: bool = eqx.field(static=True)
    algorithm: JumpTransitionAlgorithm = eqx.field(static=True)
    max_events_per_channel: int = eqx.field(static=True)
    max_events: int | None = eqx.field(static=True)

    def __init__(
        self,
        process: Any,
        /,
        *,
        max_events_per_channel: int,
        algorithm: JumpTransitionAlgorithm = "next_reaction",
        max_events: int | None = None,
        approximation_id: str | None = None,
    ):
        from ._jump import AbstractJumpProcess

        if not isinstance(process, AbstractJumpProcess):
            raise TypeError("process must implement AbstractJumpProcess.")
        capacity = int(max_events_per_channel)
        if capacity < 1:
            raise ValueError("max_events_per_channel must be positive.")
        if algorithm not in ("next_reaction", "direct_ssa"):
            raise ValueError("algorithm must be 'next_reaction' or 'direct_ssa'.")
        total_capacity = None if max_events is None else int(max_events)
        if total_capacity is not None and total_capacity < 1:
            raise ValueError("max_events must be positive or None.")
        self.process = process
        self.state_shape = process.state_shape
        self.process_id = process.process_id
        self.approximation_id = _name(
            approximation_id or f"{algorithm}-transition",
            owner="approximation_id",
        )
        self.has_log_density = False
        self.algorithm = algorithm
        self.max_events_per_channel = capacity
        self.max_events = total_capacity

    def sample(self, key, state, t0, t1, context, /) -> TransitionSample:
        from ..solver import solve_direct_ssa, solve_next_reaction
        from ._jump import PoissonClockRealization

        state_array = jnp.asarray(state)
        if state_array.shape != self.state_shape:
            raise ValueError(
                f"state must have shape {self.state_shape}; got {state_array.shape}."
            )
        start, end, support = _interval(t0, t1)
        realization = PoissonClockRealization(
            key,
            self.process.num_channels,
            support=support,
            max_events_per_channel=self.max_events_per_channel,
            process_id=self.process_id,
            label=f"{self.process_id}:transition",
        )
        solve = (
            solve_next_reaction if self.algorithm == "next_reaction" else solve_direct_ssa
        )
        solution = solve(
            self.process,
            realization,
            state_array,
            t0=start,
            t1=end,
            save_times=jnp.asarray([end]),
            args=context,
            max_events=self.max_events,
        )
        values = solution.states[-1]
        valid = solution.valid[-1] & solution.successful & jnp.all(jnp.isfinite(values))
        return _transition_sample(
            values,
            valid,
            solution.events.status,
            process_id=self.process_id,
            approximation_id=self.approximation_id,
        )

    def log_prob(self, next_state, state, t0, t1, context, /) -> Array:
        del next_state, state, t0, t1, context
        raise ValueError("Jump solver transitions do not provide a density.")


class JumpDifferentialTransitionKernel(AbstractTransitionKernel):
    """One-interval ODE/SDE plus finite-activity jump transition."""

    drift: Callable
    jump_process: Any
    wiener_terms: tuple[Any, ...]
    solver: Any
    stepsize_controller: Any
    dt0: Array | None
    state_shape: tuple[int, ...] = eqx.field(static=True)
    process_id: str = eqx.field(static=True)
    approximation_id: str = eqx.field(static=True)
    has_log_density: bool = eqx.field(static=True)
    interpretation: str = eqx.field(static=True)
    max_events_per_channel: int = eqx.field(static=True)
    max_events: int | None = eqx.field(static=True)
    wiener_tolerance: float = eqx.field(static=True)
    rtol: float = eqx.field(static=True)
    atol: float = eqx.field(static=True)
    event_rtol: float = eqx.field(static=True)
    event_atol: float = eqx.field(static=True)
    max_steps: int = eqx.field(static=True)

    def __init__(
        self,
        drift: Callable,
        jump_process: Any,
        /,
        *,
        state_shape: Sequence[int],
        max_events_per_channel: int,
        process_id: str | None = None,
        wiener_terms: Sequence[Any] = (),
        interpretation: str = "ito",
        solver: Any = None,
        stepsize_controller: Any = None,
        dt0: ArrayLike | None = None,
        max_events: int | None = None,
        wiener_tolerance: float = 1e-3,
        rtol: float = 1e-6,
        atol: float = 1e-8,
        event_rtol: float = 1e-7,
        event_atol: float = 1e-9,
        max_steps: int = 4096,
        approximation_id: str = "jump-differential-transition",
    ):
        from ..solver import WienerTerm
        from ._jump import AbstractJumpProcess

        if not callable(drift):
            raise TypeError("drift must be callable.")
        if not isinstance(jump_process, AbstractJumpProcess):
            raise TypeError("jump_process must implement AbstractJumpProcess.")
        shape = _shape(state_shape, owner="state_shape")
        if shape != jump_process.state_shape:
            raise ValueError("state_shape must match the jump process state_shape.")
        terms = tuple(wiener_terms)
        if any(not isinstance(term, WienerTerm) for term in terms):
            raise TypeError("wiener_terms must contain only WienerTerm objects.")
        if interpretation not in ("ito", "stratonovich"):
            raise ValueError("interpretation must be 'ito' or 'stratonovich'.")
        capacity = int(max_events_per_channel)
        if capacity < 1:
            raise ValueError("max_events_per_channel must be positive.")
        total_capacity = None if max_events is None else int(max_events)
        if total_capacity is not None and total_capacity < 1:
            raise ValueError("max_events must be positive or None.")
        tolerance = float(wiener_tolerance)
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("wiener_tolerance must be finite and positive.")
        resolved_dt0 = None if dt0 is None else jnp.asarray(dt0, dtype=float)
        if resolved_dt0 is not None and (
            resolved_dt0.shape != () or not bool(jnp.isfinite(resolved_dt0))
        ):
            raise ValueError("dt0 must be a finite scalar or None.")
        if terms and resolved_dt0 is None:
            raise ValueError("Stochastic hybrid transitions require explicit dt0.")
        if terms and resolved_dt0 is not None and abs(float(resolved_dt0)) <= tolerance:
            raise ValueError("wiener_tolerance must be smaller than abs(dt0).")
        self.drift = drift
        self.jump_process = jump_process
        self.wiener_terms = terms
        self.solver = solver
        self.stepsize_controller = stepsize_controller
        self.dt0 = resolved_dt0
        self.state_shape = shape
        self.process_id = _name(
            jump_process.process_id if process_id is None else process_id,
            owner="process_id",
        )
        self.approximation_id = _name(approximation_id, owner="approximation_id")
        self.has_log_density = False
        self.interpretation = interpretation
        self.max_events_per_channel = capacity
        self.max_events = total_capacity
        self.wiener_tolerance = tolerance
        self.rtol = float(rtol)
        self.atol = float(atol)
        self.event_rtol = float(event_rtol)
        self.event_atol = float(event_atol)
        self.max_steps = int(max_steps)

    def sample(self, key, state, t0, t1, context, /) -> TransitionSample:
        from ..solver import (
            DifferentialProblem,
            JumpDifferentialProblem,
            solve_jump_differential,
        )
        from ._jump import PoissonClockRealization
        from ._wiener import WienerRealization

        state_array = jnp.asarray(state)
        if state_array.shape != self.state_shape:
            raise ValueError(
                f"state must have shape {self.state_shape}; got {state_array.shape}."
            )
        start, end, support = _interval(t0, t1)
        controller, dt0 = _input_controller(
            context,
            self.stepsize_controller,
            self.dt0,
            start,
            end,
            stochastic=bool(self.wiener_terms),
            rtol=self.rtol,
            atol=self.atol,
        )
        differential = DifferentialProblem(
            self.drift,
            state_array,
            t0=start,
            t1=end,
            args=context,
            wiener_terms=self.wiener_terms,
            interpretation=self.interpretation,
        )
        problem = JumpDifferentialProblem(
            differential, self.jump_process, process_id=self.process_id
        )
        poisson_key, wiener_key = jr.split(key)
        poisson = PoissonClockRealization(
            poisson_key,
            self.jump_process.num_channels,
            support=support,
            max_events_per_channel=self.max_events_per_channel,
            process_id=self.jump_process.process_id,
            label=f"{self.process_id}:jump-transition",
        )
        wiener = (
            WienerRealization(
                wiener_key,
                differential.noise_shape,
                support=support,
                tolerance=self.wiener_tolerance,
                noise_id=differential.noise_id,
                label=f"{self.process_id}:wiener-transition",
            )
            if differential.stochastic
            else None
        )
        solution = solve_jump_differential(
            problem,
            poisson,
            save_times=jnp.asarray([start, end]),
            wiener_realization=wiener,
            solver=self.solver,
            stepsize_controller=controller,
            dt0=dt0,
            rtol=self.rtol,
            atol=self.atol,
            event_rtol=self.event_rtol,
            event_atol=self.event_atol,
            max_steps=self.max_steps,
            max_events=self.max_events,
        )
        values = solution.states[-1]
        valid = (
            context.input_valid
            & solution.valid[-1]
            & solution.events.successful
            & jnp.all(jnp.isfinite(values))
        )
        return _transition_sample(
            values,
            valid,
            solution.events.status,
            process_id=self.process_id,
            approximation_id=self.approximation_id,
        )

    def log_prob(self, next_state, state, t0, t1, context, /) -> Array:
        del next_state, state, t0, t1, context
        raise ValueError("Hybrid solver transitions do not provide a density.")


class FiniteStateTransitionKernel(AbstractTransitionKernel):
    """Exact matrix-exponential transition law for a closed finite-state generator."""

    generator: Any
    state_shape: tuple[int, ...] = eqx.field(static=True)
    process_id: str = eqx.field(static=True)
    approximation_id: str = eqx.field(static=True)
    has_log_density: bool = eqx.field(static=True)

    def __init__(
        self,
        generator: Any,
        /,
        *,
        approximation_id: str = "exact-finite-state-transition",
    ):
        from ..solver import FiniteStateGenerator

        if not isinstance(generator, FiniteStateGenerator):
            raise TypeError("generator must be a FiniteStateGenerator.")
        if generator.boundary_policy == "leak":
            raise ValueError(
                "FiniteStateTransitionKernel requires a closed or suppressed generator."
            )
        self.generator = generator
        self.state_shape = tuple(generator.states.shape[1:])
        self.process_id = generator.process_id
        self.approximation_id = _name(approximation_id, owner="approximation_id")
        self.has_log_density = True

    def _indices(self, state: ArrayLike, /) -> tuple[Array, Array, tuple[int, ...]]:
        values = jnp.asarray(state)
        if (
            values.ndim < len(self.state_shape)
            or tuple(values.shape[-len(self.state_shape) :]) != self.state_shape
        ):
            raise ValueError("state has an incompatible trailing state shape.")
        batch_shape = values.shape[: -len(self.state_shape)]
        flat = values.reshape((-1,) + self.state_shape)
        state_axes = tuple(range(2, 2 + len(self.state_shape)))
        matches = jnp.all(
            flat[:, None, ...] == self.generator.states[None, ...],
            axis=state_axes,
        )
        valid = jnp.any(matches, axis=-1)
        indices = jnp.argmax(matches, axis=-1)
        return indices, valid, batch_shape

    def sample(self, key, state, t0, t1, context, /) -> TransitionSample:
        del context
        start, end = _array_interval(t0, t1)
        values = jnp.asarray(state)
        indices, valid, batch_shape = self._indices(values)
        matrix = self.generator.transition_matrix(end - start)
        probabilities = matrix[indices]
        matrix_valid = (
            jnp.all(jnp.isfinite(matrix))
            & jnp.all(matrix >= 0.0)
            & jnp.all(jnp.abs(jnp.sum(matrix, axis=-1) - 1.0) <= 1e-5)
        )
        logits = jnp.where(
            probabilities > 0.0,
            jnp.log(probabilities),
            -jnp.inf,
        )
        draws = jr.categorical(key, logits, axis=-1)
        next_values = self.generator.states[draws].reshape(batch_shape + self.state_shape)
        valid_shape = batch_shape
        valid_values = valid.reshape(valid_shape) & matrix_valid
        output = jnp.where(
            valid_values[..., *([None] * len(self.state_shape))],
            next_values,
            values,
        )
        return _transition_sample(
            output,
            valid_values,
            jnp.where(valid_values, 0, 1),
            process_id=self.process_id,
            approximation_id=self.approximation_id,
        )

    def log_prob(self, next_state, state, t0, t1, context, /) -> Array:
        del context
        start, end = _array_interval(t0, t1)
        current, current_valid, batch_shape = self._indices(state)
        following, following_valid, next_batch_shape = self._indices(next_state)
        if batch_shape != next_batch_shape:
            raise ValueError("state and next_state batch shapes must match.")
        matrix = self.generator.transition_matrix(end - start)
        probabilities = matrix[current, following]
        valid = current_valid & following_valid & (probabilities > 0.0)
        return jnp.where(
            valid.reshape(batch_shape),
            jnp.log(probabilities.reshape(batch_shape)),
            -jnp.inf,
        )


class PathwiseTransitionKernel(AbstractTransitionKernel):
    """Adapter from an explicit-driver pathwise law to a sampled Markov kernel."""

    law: AbstractPathwiseTransition
    driver_sampler: Callable[[Array, Array, Array, StateSpaceStepContext], Array]
    state_shape: tuple[int, ...] = eqx.field(static=True)
    process_id: str = eqx.field(static=True)
    approximation_id: str = eqx.field(static=True)
    has_log_density: bool = eqx.field(static=True)

    def __init__(
        self,
        law: AbstractPathwiseTransition,
        driver_sampler: Callable[[Array, Array, Array, StateSpaceStepContext], Array],
        /,
        *,
        approximation_id: str = "sampled-pathwise-transition",
    ):
        if not isinstance(law, AbstractPathwiseTransition):
            raise TypeError("law must implement AbstractPathwiseTransition.")
        if not callable(driver_sampler):
            raise TypeError("driver_sampler must be callable.")
        self.law = law
        self.driver_sampler = driver_sampler
        self.state_shape = law.state_shape
        self.process_id = law.process_id
        self.approximation_id = _name(approximation_id, owner="approximation_id")
        self.has_log_density = False

    def sample(self, key, state, t0, t1, context, /) -> TransitionSample:
        state_array = jnp.asarray(state)
        if tuple(state_array.shape[-len(self.state_shape) :]) != self.state_shape:
            raise ValueError("state has an incompatible trailing state shape.")
        driver = jnp.asarray(
            self.driver_sampler(key, jnp.asarray(t0), jnp.asarray(t1), context)
        )
        if tuple(driver.shape[-len(self.law.driver_shape) :]) != self.law.driver_shape:
            raise ValueError("driver_sampler returned an incompatible driver shape.")
        values = jnp.asarray(
            self.law.pathwise_transition(
                state_array,
                t0=t0,
                t1=t1,
                driver_increment=driver,
            )
        )
        valid = jnp.all(jnp.isfinite(values))
        return _transition_sample(
            values,
            valid,
            jnp.where(valid, 0, 1),
            process_id=self.process_id,
            approximation_id=self.approximation_id,
        )

    def log_prob(self, next_state, state, t0, t1, context, /) -> Array:
        del next_state, state, t0, t1, context
        raise ValueError("Sampled pathwise transitions do not provide a density.")


__all__ = [
    "DifferentialTransitionKernel",
    "FiniteStateTransitionKernel",
    "JumpDifferentialTransitionKernel",
    "JumpTransitionAlgorithm",
    "JumpTransitionKernel",
    "PathwiseTransitionKernel",
]
