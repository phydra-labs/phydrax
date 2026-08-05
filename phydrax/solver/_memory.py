#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from math import prod
from typing import Any, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._frozendict import frozendict
from .._interpolation import apply_gather_stencil, linear_stencil_from_indices
from .._strict import StrictModule
from ..stochastic import StochasticTrajectory, WienerRealization


VolterraVectorField: TypeAlias = Callable[[Array, Array, Any], ArrayLike]
VolterraKernel: TypeAlias = Callable[[Array, Array, Any], ArrayLike]
VolterraFreeTerm: TypeAlias = Callable[[Array, Any], ArrayLike]
DelayVectorField: TypeAlias = Callable[[Array, Array, Array, Any], ArrayLike]
DelayHistory: TypeAlias = Callable[[Array, Any], ArrayLike]


class _ConstantFreeTerm(eqx.Module):
    value: Array

    def __call__(self, time, args):
        del time, args
        return self.value


class _UnitVolterraKernel(eqx.Module):
    def __call__(self, target, source, args):
        del target, source, args
        return jnp.asarray(1.0)


class StochasticVolterraProblem(StrictModule):
    """Explicit stochastic Volterra integral equation.

    The solver approximates
    ``X(t)=g(t)+∫K_b(t,s)b(s,X(s))ds+∫K_σ(t,s)G(s,X(s))dW(s)``.
    Kernels may be scalar or have exact ``state_shape`` and act componentwise.
    """

    drift: VolterraVectorField
    diffusion: VolterraVectorField | None
    drift_kernel: VolterraKernel
    diffusion_kernel: VolterraKernel
    free_term: VolterraFreeTerm
    initial_state: Array
    t0: Array
    t1: Array
    args: Any
    state_shape: tuple[int, ...] = eqx.field(static=True)
    noise_shape: tuple[int, ...] = eqx.field(static=True)
    noise_id: str | None = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        drift: VolterraVectorField,
        initial_state: ArrayLike,
        /,
        *,
        t0: ArrayLike,
        t1: ArrayLike,
        diffusion: VolterraVectorField | None = None,
        noise_shape: Sequence[int] | None = None,
        noise_id: str | None = None,
        drift_kernel: VolterraKernel | None = None,
        diffusion_kernel: VolterraKernel | None = None,
        free_term: VolterraFreeTerm | None = None,
        args: Any = None,
        problem_id: str = "stochastic-volterra-problem",
    ):
        if not callable(drift):
            raise TypeError("drift must be callable.")
        if diffusion is not None and not callable(diffusion):
            raise TypeError("diffusion must be callable or None.")
        for name, value in (
            ("drift_kernel", drift_kernel),
            ("diffusion_kernel", diffusion_kernel),
            ("free_term", free_term),
        ):
            if value is not None and not callable(value):
                raise TypeError(f"{name} must be callable or None.")
        state = jnp.asarray(initial_state)
        state_shape = tuple(int(size) for size in state.shape)
        if not state_shape or any(size <= 0 for size in state_shape):
            raise ValueError("initial_state must have a non-empty positive shape.")
        start = jnp.asarray(t0, dtype=float)
        end = jnp.asarray(t1, dtype=float)
        if (
            start.shape != ()
            or end.shape != ()
            or not bool(jnp.isfinite(start) & jnp.isfinite(end) & (end > start))
        ):
            raise ValueError("StochasticVolterraProblem requires finite scalar t1 > t0.")
        if diffusion is None:
            if noise_shape is not None or noise_id is not None:
                raise ValueError(
                    "noise_shape and noise_id are only valid with stochastic diffusion."
                )
            resolved_noise_shape: tuple[int, ...] = ()
        else:
            if noise_shape is None:
                raise ValueError("noise_shape is required with diffusion.")
            resolved_noise_shape = tuple(int(size) for size in noise_shape)
            if not resolved_noise_shape or any(
                size <= 0 for size in resolved_noise_shape
            ):
                raise ValueError("noise_shape must contain positive dimensions.")
            if noise_id is not None and (not isinstance(noise_id, str) or not noise_id):
                raise ValueError("noise_id must be non-empty or None.")
        resolved_drift_kernel = (
            _UnitVolterraKernel() if drift_kernel is None else drift_kernel
        )
        resolved_diffusion_kernel = (
            _UnitVolterraKernel() if diffusion_kernel is None else diffusion_kernel
        )
        resolved_free: VolterraFreeTerm = (
            _ConstantFreeTerm(state) if free_term is None else free_term
        )
        drift_value = jnp.asarray(drift(start, state, args))
        if drift_value.shape != state_shape:
            raise ValueError("drift must preserve initial_state shape.")
        free_value = jnp.asarray(resolved_free(start, args))
        if free_value.shape != state_shape or not bool(jnp.allclose(free_value, state)):
            raise ValueError("free_term(t0, args) must equal initial_state.")
        _validate_kernel(
            resolved_drift_kernel(end, start, args),
            state_shape,
            owner="drift_kernel",
        )
        if diffusion is not None:
            diffusion_value = jnp.asarray(diffusion(start, state, args))
            expected = state_shape + resolved_noise_shape
            if diffusion_value.shape != expected:
                raise ValueError(
                    f"diffusion must return shape {expected}; got {diffusion_value.shape}."
                )
            _validate_kernel(
                resolved_diffusion_kernel(end, start, args),
                state_shape,
                owner="diffusion_kernel",
            )
        identifier = str(problem_id)
        if not identifier:
            raise ValueError("problem_id must be non-empty.")
        self.drift = drift
        self.diffusion = diffusion
        self.drift_kernel = resolved_drift_kernel
        self.diffusion_kernel = resolved_diffusion_kernel
        self.free_term = resolved_free
        self.initial_state = state
        self.t0 = start
        self.t1 = end
        self.args = args
        self.state_shape = state_shape
        self.noise_shape = resolved_noise_shape
        self.noise_id = noise_id
        self.problem_id = identifier

    @property
    def stochastic(self) -> bool:
        return self.diffusion is not None


class StochasticDelayProblem(StrictModule):
    """Stochastic differential equation with one or more constant discrete delays."""

    drift: DelayVectorField
    diffusion: DelayVectorField | None
    history: DelayHistory
    delays: Array
    initial_state: Array
    t0: Array
    t1: Array
    args: Any
    state_shape: tuple[int, ...] = eqx.field(static=True)
    noise_shape: tuple[int, ...] = eqx.field(static=True)
    noise_id: str | None = eqx.field(static=True)
    num_delays: int = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        drift: DelayVectorField,
        history: DelayHistory,
        delays: ArrayLike,
        /,
        *,
        t0: ArrayLike,
        t1: ArrayLike,
        diffusion: DelayVectorField | None = None,
        noise_shape: Sequence[int] | None = None,
        noise_id: str | None = None,
        args: Any = None,
        problem_id: str = "stochastic-delay-problem",
    ):
        if not callable(drift) or not callable(history):
            raise TypeError("drift and history must be callable.")
        if diffusion is not None and not callable(diffusion):
            raise TypeError("diffusion must be callable or None.")
        start = jnp.asarray(t0, dtype=float)
        end = jnp.asarray(t1, dtype=float)
        if (
            start.shape != ()
            or end.shape != ()
            or not bool(jnp.isfinite(start) & jnp.isfinite(end) & (end > start))
        ):
            raise ValueError("StochasticDelayProblem requires finite scalar t1 > t0.")
        delay_values = jnp.asarray(delays, dtype=float).reshape((-1,))
        if int(delay_values.size) <= 0 or bool(
            jnp.any(~jnp.isfinite(delay_values)) | jnp.any(delay_values <= 0.0)
        ):
            raise ValueError("delays must be a non-empty vector of positive values.")
        state = jnp.asarray(history(start, args))
        state_shape = tuple(int(size) for size in state.shape)
        if not state_shape or any(size <= 0 for size in state_shape):
            raise ValueError("history must return a non-empty state shape.")
        delayed = jax.vmap(lambda delay: jnp.asarray(history(start - delay, args)))(
            delay_values
        )
        expected_delayed = (int(delay_values.size),) + state_shape
        if delayed.shape != expected_delayed:
            raise ValueError(
                f"history values must stack to shape {expected_delayed}; got {delayed.shape}."
            )
        drift_value = jnp.asarray(drift(start, state, delayed, args))
        if drift_value.shape != state_shape:
            raise ValueError("drift must preserve the history state shape.")
        if diffusion is None:
            if noise_shape is not None or noise_id is not None:
                raise ValueError(
                    "noise_shape and noise_id are only valid with stochastic diffusion."
                )
            resolved_noise_shape: tuple[int, ...] = ()
        else:
            if noise_shape is None:
                raise ValueError("noise_shape is required with diffusion.")
            resolved_noise_shape = tuple(int(size) for size in noise_shape)
            if not resolved_noise_shape or any(
                size <= 0 for size in resolved_noise_shape
            ):
                raise ValueError("noise_shape must contain positive dimensions.")
            diffusion_value = jnp.asarray(diffusion(start, state, delayed, args))
            expected = state_shape + resolved_noise_shape
            if diffusion_value.shape != expected:
                raise ValueError(
                    f"diffusion must return shape {expected}; got {diffusion_value.shape}."
                )
            if noise_id is not None and (not isinstance(noise_id, str) or not noise_id):
                raise ValueError("noise_id must be non-empty or None.")
        identifier = str(problem_id)
        if not identifier:
            raise ValueError("problem_id must be non-empty.")
        self.drift = drift
        self.diffusion = diffusion
        self.history = history
        self.delays = delay_values
        self.initial_state = state
        self.t0 = start
        self.t1 = end
        self.args = args
        self.state_shape = state_shape
        self.noise_shape = resolved_noise_shape
        self.noise_id = noise_id
        self.num_delays = int(delay_values.size)
        self.problem_id = identifier

    @property
    def stochastic(self) -> bool:
        return self.diffusion is not None


class MemoryEquationSolution(StrictModule):
    """Saved Volterra or delay trajectory with global driver provenance."""

    times: Array
    states: Array
    valid: Array
    realization: WienerRealization | None
    metadata: frozendict[str, Any]
    sample_shape: tuple[int, ...] = eqx.field(static=True)
    state_shape: tuple[int, ...] = eqx.field(static=True)
    solver_name: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        times: ArrayLike,
        states: ArrayLike,
        valid: ArrayLike,
        realization: WienerRealization | None,
        state_shape: Sequence[int],
        solver_name: str,
        metadata: Mapping[str, Any] | None = None,
    ):
        if realization is not None and not isinstance(realization, WienerRealization):
            raise TypeError("realization must be a WienerRealization or None.")
        time_values = jnp.asarray(times, dtype=float)
        state_values = jnp.asarray(states)
        valid_values = jnp.asarray(valid, dtype=bool)
        sample_shape = () if realization is None else realization.sample_shape
        shape = tuple(int(size) for size in state_shape)
        expected_states = sample_shape + (int(time_values.size),) + shape
        expected_valid = sample_shape + (int(time_values.size),)
        if time_values.ndim != 1 or int(time_values.size) < 2:
            raise ValueError("times must be a rank-1 grid with at least two nodes.")
        if state_values.shape != expected_states or valid_values.shape != expected_valid:
            raise ValueError("Memory-equation output does not align with declared axes.")
        if not isinstance(solver_name, str) or not solver_name:
            raise ValueError("solver_name must be non-empty.")
        self.times = time_values
        self.states = state_values
        self.valid = valid_values
        self.realization = realization
        self.metadata = frozendict({} if metadata is None else metadata)
        self.sample_shape = sample_shape
        self.state_shape = shape
        self.solver_name = solver_name

    @property
    def successful(self) -> Array:
        return jnp.all(self.valid, axis=-1)

    def to_stochastic_trajectory(
        self,
        /,
        *,
        realization_axes: Sequence[str] | None = None,
        state_axes: Sequence[str] | None = None,
    ) -> StochasticTrajectory:
        axes = (
            tuple(f"path_{index}" for index in range(len(self.sample_shape)))
            if realization_axes is None
            else tuple(realization_axes)
        )
        resolved_state_axes = (
            tuple(f"state_{index}" for index in range(len(self.state_shape)))
            if state_axes is None
            else tuple(state_axes)
        )
        return StochasticTrajectory(
            self.times,
            self.states,
            valid=self.valid,
            realization_axes=axes,
            realization_shape=self.sample_shape,
            state_axes=resolved_state_axes,
            realizations=(self.realization,),
            metadata={
                **dict(self.metadata),
                "solver_name": self.solver_name,
                "uncertainty_source": "process",
            },
        )


def _validate_kernel(
    value: ArrayLike, state_shape: tuple[int, ...], /, *, owner: str
) -> Array:
    kernel = jnp.asarray(value)
    if kernel.shape not in ((), state_shape):
        raise ValueError(f"{owner} must return a scalar or exact state shape.")
    return kernel


def _weighted_kernel_value(
    kernel: Array,
    value: Array,
    state_shape: tuple[int, ...],
    trailing_rank: int,
    /,
) -> Array:
    if kernel.shape == ():
        return kernel * value
    return kernel.reshape(state_shape + (1,) * trailing_rank) * value


def _time_grid(t0: Array, t1: Array, times: ArrayLike, /) -> Array:
    grid = jnp.asarray(times, dtype=float)
    if grid.ndim != 1 or int(grid.size) < 2:
        raise ValueError("times must be a rank-1 grid with at least two nodes.")
    host = np.asarray(jax.device_get(grid))
    if np.any(~np.isfinite(host)) or np.any(np.diff(host) <= 0.0):
        raise ValueError("times must be finite and strictly increasing.")
    tolerance = 100.0 * np.finfo(float).eps * max(1.0, abs(float(t0)), abs(float(t1)))
    if (
        abs(float(host[0]) - float(t0)) > tolerance
        or abs(float(host[-1]) - float(t1)) > tolerance
    ):
        raise ValueError("times must span exactly from problem t0 through t1.")
    return grid


def _wiener_increments(
    *,
    stochastic: bool,
    noise_shape: tuple[int, ...],
    noise_id: str | None,
    t0: Array,
    t1: Array,
    times: Array,
    realization: WienerRealization | None,
    dtype,
) -> tuple[Array, tuple[int, ...]]:
    if not stochastic:
        if realization is not None:
            raise ValueError(
                "Deterministic memory equations do not accept a realization."
            )
        return jnp.zeros((int(times.size) - 1, 0), dtype=dtype), ()
    if not isinstance(realization, WienerRealization):
        raise TypeError("Stochastic memory equations require a WienerRealization.")
    if realization.noise_shape != noise_shape:
        raise ValueError("Wiener realization noise_shape does not match the problem.")
    if realization.noise_id != noise_id:
        raise ValueError("Wiener realization noise_id does not match the problem.")
    if realization.levy_area != "brownian":
        raise ValueError("Memory solvers require a Brownian-increment realization.")
    if realization.support[0] > float(t0) or realization.support[1] < float(t1):
        raise ValueError("Wiener realization support must cover the problem interval.")
    return (
        realization.increments(times[:-1], times[1:], dtype=dtype),
        realization.sample_shape,
    )


def _solution_valid(states: Array, sample_shape: tuple[int, ...]) -> Array:
    state_axes = tuple(range(len(sample_shape) + 1, states.ndim))
    return jnp.all(jnp.isfinite(states), axis=state_axes)


def solve_stochastic_volterra(
    problem: StochasticVolterraProblem,
    /,
    *,
    times: ArrayLike,
    realization: WienerRealization | None = None,
) -> MemoryEquationSolution:
    """Solve a stochastic Volterra equation by explicit left-point convolution."""
    if not isinstance(problem, StochasticVolterraProblem):
        raise TypeError("problem must be a StochasticVolterraProblem.")
    grid = _time_grid(problem.t0, problem.t1, times)
    increments, sample_shape = _wiener_increments(
        stochastic=problem.stochastic,
        noise_shape=problem.noise_shape,
        noise_id=problem.noise_id,
        t0=problem.t0,
        t1=problem.t1,
        times=grid,
        realization=realization,
        dtype=problem.initial_state.real.dtype,
    )
    num_times = int(grid.size)
    num_steps = num_times - 1
    step_sizes = jnp.diff(grid)

    def one_path(path_increments):
        states = jnp.zeros(
            (num_times,) + problem.state_shape, dtype=problem.initial_state.dtype
        )
        states = states.at[0].set(problem.initial_state)

        def outer(index, state_buffer):
            target = grid[index]
            drift_initial = jnp.zeros(
                problem.state_shape, dtype=problem.initial_state.dtype
            )
            noise_initial = jnp.zeros_like(drift_initial)

            def inner(source_index, accumulators):
                drift_sum, noise_sum = accumulators
                source = grid[source_index]
                state = state_buffer[source_index]
                drift = jnp.asarray(problem.drift(source, state, problem.args))
                if drift.shape != problem.state_shape:
                    raise ValueError("drift must preserve the declared state shape.")
                drift_kernel = _validate_kernel(
                    problem.drift_kernel(target, source, problem.args),
                    problem.state_shape,
                    owner="drift_kernel",
                )
                drift_sum = drift_sum + step_sizes[source_index] * _weighted_kernel_value(
                    drift_kernel,
                    drift,
                    problem.state_shape,
                    0,
                )
                if problem.stochastic:
                    assert problem.diffusion is not None
                    diffusion = jnp.asarray(
                        problem.diffusion(source, state, problem.args)
                    )
                    expected = problem.state_shape + problem.noise_shape
                    if diffusion.shape != expected:
                        raise ValueError(
                            f"diffusion must return shape {expected}; got {diffusion.shape}."
                        )
                    diffusion_kernel = _validate_kernel(
                        problem.diffusion_kernel(target, source, problem.args),
                        problem.state_shape,
                        owner="diffusion_kernel",
                    )
                    weighted = _weighted_kernel_value(
                        diffusion_kernel,
                        diffusion,
                        problem.state_shape,
                        len(problem.noise_shape),
                    )
                    state_axes = tuple(range(len(problem.state_shape), weighted.ndim))
                    increment_axes = tuple(range(path_increments[source_index].ndim))
                    noise_sum = noise_sum + jnp.tensordot(
                        weighted,
                        path_increments[source_index],
                        axes=(state_axes, increment_axes),
                    )
                return drift_sum, noise_sum

            drift_sum, noise_sum = jax.lax.fori_loop(
                0,
                index,
                inner,
                (drift_initial, noise_initial),
            )
            free = jnp.asarray(problem.free_term(target, problem.args))
            if free.shape != problem.state_shape:
                raise ValueError("free_term must preserve the declared state shape.")
            return state_buffer.at[index].set(free + drift_sum + noise_sum)

        return jax.lax.fori_loop(1, num_times, outer, states)

    if sample_shape:
        flat = increments.reshape((prod(sample_shape), num_steps) + problem.noise_shape)
        states = jax.vmap(one_path)(flat).reshape(
            sample_shape + (num_times,) + problem.state_shape
        )
    else:
        states = one_path(increments)
    return MemoryEquationSolution(
        times=grid,
        states=states,
        valid=_solution_valid(states, sample_shape),
        realization=realization,
        state_shape=problem.state_shape,
        solver_name="StochasticVolterraEuler",
        metadata={
            "problem_id": problem.problem_id,
            "num_steps": num_steps,
            "quadrature": "left",
        },
    )


def solve_stochastic_delay(
    problem: StochasticDelayProblem,
    /,
    *,
    times: ArrayLike,
    realization: WienerRealization | None = None,
) -> MemoryEquationSolution:
    """Solve a multi-delay SDE by Euler--Maruyama and causal linear history lookup."""
    if not isinstance(problem, StochasticDelayProblem):
        raise TypeError("problem must be a StochasticDelayProblem.")
    grid = _time_grid(problem.t0, problem.t1, times)
    increments, sample_shape = _wiener_increments(
        stochastic=problem.stochastic,
        noise_shape=problem.noise_shape,
        noise_id=problem.noise_id,
        t0=problem.t0,
        t1=problem.t1,
        times=grid,
        realization=realization,
        dtype=problem.initial_state.real.dtype,
    )
    num_times = int(grid.size)
    num_steps = num_times - 1
    step_sizes = jnp.diff(grid)

    def one_path(path_increments):
        states = jnp.zeros(
            (num_times,) + problem.state_shape, dtype=problem.initial_state.dtype
        )
        states = states.at[0].set(problem.initial_state)

        def outer(index, state_buffer):
            time = grid[index]
            current = state_buffer[index]

            def delayed_state(delay):
                query = time - delay

                def from_history(value):
                    history_value = jnp.asarray(problem.history(value, problem.args))
                    if history_value.shape != problem.state_shape:
                        raise ValueError(
                            "history must preserve the declared state shape."
                        )
                    return history_value

                def from_solution(value):
                    left = jnp.searchsorted(grid, value, side="right") - 1
                    left = jnp.clip(left, 0, index)
                    right = jnp.minimum(left + 1, index)
                    denominator = grid[right] - grid[left]
                    fraction = jnp.where(
                        denominator > 0.0,
                        (value - grid[left]) / denominator,
                        0.0,
                    )
                    stencil = linear_stencil_from_indices(
                        left,
                        right,
                        fraction,
                        source_size=num_times,
                    )
                    return apply_gather_stencil(state_buffer, stencil).values

                return jax.lax.cond(
                    query <= problem.t0,
                    from_history,
                    from_solution,
                    query,
                )

            delayed = jax.vmap(delayed_state)(problem.delays)
            drift = jnp.asarray(problem.drift(time, current, delayed, problem.args))
            if drift.shape != problem.state_shape:
                raise ValueError("drift must preserve the declared state shape.")
            update = step_sizes[index] * drift
            if problem.stochastic:
                assert problem.diffusion is not None
                diffusion = jnp.asarray(
                    problem.diffusion(time, current, delayed, problem.args)
                )
                expected = problem.state_shape + problem.noise_shape
                if diffusion.shape != expected:
                    raise ValueError(
                        f"diffusion must return shape {expected}; got {diffusion.shape}."
                    )
                state_axes = tuple(range(len(problem.state_shape), diffusion.ndim))
                increment_axes = tuple(range(path_increments[index].ndim))
                update = update + jnp.tensordot(
                    diffusion,
                    path_increments[index],
                    axes=(state_axes, increment_axes),
                )
            return state_buffer.at[index + 1].set(current + update)

        return jax.lax.fori_loop(0, num_steps, outer, states)

    if sample_shape:
        flat = increments.reshape((prod(sample_shape), num_steps) + problem.noise_shape)
        states = jax.vmap(one_path)(flat).reshape(
            sample_shape + (num_times,) + problem.state_shape
        )
    else:
        states = one_path(increments)
    return MemoryEquationSolution(
        times=grid,
        states=states,
        valid=_solution_valid(states, sample_shape),
        realization=realization,
        state_shape=problem.state_shape,
        solver_name="StochasticDelayEulerMaruyama",
        metadata={
            "problem_id": problem.problem_id,
            "num_steps": num_steps,
            "num_delays": problem.num_delays,
            "history_interpolation": "linear",
        },
    )


__all__ = [
    "DelayHistory",
    "DelayVectorField",
    "MemoryEquationSolution",
    "StochasticDelayProblem",
    "StochasticVolterraProblem",
    "VolterraFreeTerm",
    "VolterraKernel",
    "VolterraVectorField",
    "solve_stochastic_delay",
    "solve_stochastic_volterra",
]
