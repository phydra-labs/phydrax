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
from jaxtyping import Array, ArrayLike

from .._frozendict import frozendict
from .._strict import StrictModule
from ..stochastic._trajectory import _TrajectoryRecord, StochasticTrajectory
from ..stochastic._wiener import WienerRealization
from ._solution_validation import validate_solution_arrays


VolterraVectorField: TypeAlias = Callable[[Array, Array, Any], ArrayLike]
VolterraKernel: TypeAlias = Callable[[Array, Array, Any], ArrayLike]
VolterraFreeTerm: TypeAlias = Callable[[Array, Any], ArrayLike]
ConvolutionKernel: TypeAlias = Callable[[Array, Any], ArrayLike]


class _ConstantFreeTerm(eqx.Module):
    value: Array

    def __call__(self, time, args):
        del time, args
        return self.value


class _UnitVolterraKernel(eqx.Module):
    def __call__(self, target, source, args):
        del target, source, args
        return jnp.asarray(1.0)


class _ConvolutionKernelAdapter(eqx.Module):
    kernel: ConvolutionKernel

    def __call__(self, target, source, args):
        return self.kernel(target - source, args)


class _UnitConvolutionKernel(eqx.Module):
    def __call__(self, lag, args):
        del lag, args
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
        if start.shape != () or end.shape != ():
            raise ValueError("StochasticVolterraProblem requires scalar t0 and t1.")
        start = eqx.error_if(
            start,
            ~jnp.isfinite(start) | ~jnp.isfinite(end) | (end <= start),
            "StochasticVolterraProblem requires finite scalar t1 > t0.",
        )
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
        if free_value.shape != state_shape:
            raise ValueError("free_term must preserve initial_state shape.")
        free_value = eqx.error_if(
            free_value,
            ~jnp.allclose(free_value, state),
            "free_term(t0, args) must equal initial_state.",
        )
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


class ConvolutionVolterraProblem(StrictModule):
    """Volterra equation with translation-invariant causal kernels.

    Kernels receive ``(lag, args)`` and are evaluated only for positive causal lags.
    The wrapped generic Volterra contract supports deterministic and Wiener-driven
    equations without duplicating either path or shape validation.
    """

    volterra: StochasticVolterraProblem
    drift_kernel: ConvolutionKernel
    diffusion_kernel: ConvolutionKernel

    def __init__(
        self,
        drift: VolterraVectorField,
        initial_state: ArrayLike,
        /,
        *,
        t0: ArrayLike,
        t1: ArrayLike,
        kernel: ConvolutionKernel | None = None,
        diffusion: VolterraVectorField | None = None,
        diffusion_kernel: ConvolutionKernel | None = None,
        noise_shape: Sequence[int] | None = None,
        noise_id: str | None = None,
        free_term: VolterraFreeTerm | None = None,
        args: Any = None,
        problem_id: str = "convolution-volterra-problem",
    ):
        if kernel is not None and not callable(kernel):
            raise TypeError("kernel must be callable or None.")
        if diffusion_kernel is not None and not callable(diffusion_kernel):
            raise TypeError("diffusion_kernel must be callable or None.")
        drift_lag_kernel = _UnitConvolutionKernel() if kernel is None else kernel
        noise_lag_kernel = (
            _UnitConvolutionKernel() if diffusion_kernel is None else diffusion_kernel
        )
        self.volterra = StochasticVolterraProblem(
            drift,
            initial_state,
            t0=t0,
            t1=t1,
            diffusion=diffusion,
            noise_shape=noise_shape,
            noise_id=noise_id,
            drift_kernel=_ConvolutionKernelAdapter(drift_lag_kernel),
            diffusion_kernel=_ConvolutionKernelAdapter(noise_lag_kernel),
            free_term=free_term,
            args=args,
            problem_id=problem_id,
        )
        self.drift_kernel = drift_lag_kernel
        self.diffusion_kernel = noise_lag_kernel

    @property
    def stochastic(self) -> bool:
        return self.volterra.stochastic

    @property
    def initial_state(self) -> Array:
        return self.volterra.initial_state

    @property
    def state_shape(self) -> tuple[int, ...]:
        return self.volterra.state_shape

    @property
    def noise_shape(self) -> tuple[int, ...]:
        return self.volterra.noise_shape

    @property
    def noise_id(self) -> str | None:
        return self.volterra.noise_id

    @property
    def t0(self) -> Array:
        return self.volterra.t0

    @property
    def t1(self) -> Array:
        return self.volterra.t1

    @property
    def args(self) -> Any:
        return self.volterra.args

    @property
    def drift(self) -> VolterraVectorField:
        return self.volterra.drift

    @property
    def diffusion(self) -> VolterraVectorField | None:
        return self.volterra.diffusion

    @property
    def free_term(self) -> VolterraFreeTerm:
        return self.volterra.free_term

    @property
    def problem_id(self) -> str:
        return self.volterra.problem_id


class MemoryEquationSolution(StrictModule):
    """Saved Volterra or delay trajectory with global driver provenance."""

    times: Array
    states: Array
    valid: Array
    interpolation: Any | None
    backend_result: Any
    stats: frozendict[str, Any]
    event_mask: Any
    realization: WienerRealization | None
    metadata: frozendict[str, Any]
    continuation: Any
    sample_shape: tuple[int, ...] = eqx.field(static=True)
    state_shape: tuple[int, ...] = eqx.field(static=True)
    solver_name: str = eqx.field(static=True)
    solver_id: str = eqx.field(static=True)
    resolved_method: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        times: ArrayLike,
        states: ArrayLike,
        valid: ArrayLike,
        realization: WienerRealization | None,
        state_shape: Sequence[int],
        solver_name: str,
        interpolation: Any | None = None,
        backend_result: Any = None,
        stats: Mapping[str, Any] | None = None,
        event_mask: Any = None,
        solver_id: str | None = None,
        resolved_method: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        continuation: Any = None,
    ):
        if realization is not None and not isinstance(realization, WienerRealization):
            raise TypeError("realization must be a WienerRealization or None.")
        sample_shape = () if realization is None else realization.sample_shape
        arrays = validate_solution_arrays(
            times,
            states,
            valid,
            sample_shape=sample_shape,
            state_shape=state_shape,
            time_layout="shared",
            owner="MemoryEquationSolution",
        )
        time_values = arrays.times
        state_values = arrays.states
        valid_values = arrays.valid
        shape = arrays.state_shape
        if not isinstance(solver_name, str) or not solver_name:
            raise ValueError("solver_name must be non-empty.")
        resolved_solver_id = f"solver:{solver_name}" if solver_id is None else solver_id
        resolved_solver_method = (
            solver_name if resolved_method is None else resolved_method
        )
        if not isinstance(resolved_solver_id, str) or not resolved_solver_id:
            raise ValueError("solver_id must be non-empty.")
        if not isinstance(resolved_solver_method, str) or not resolved_solver_method:
            raise ValueError("resolved_method must be non-empty.")
        if interpolation is not None and not callable(
            getattr(interpolation, "evaluate", None)
        ):
            raise TypeError("interpolation must define evaluate().")
        self.times = time_values
        self.states = state_values
        self.valid = valid_values
        self.interpolation = interpolation
        self.backend_result = backend_result
        self.stats = frozendict({} if stats is None else dict(stats))
        self.event_mask = event_mask
        self.realization = realization
        self.metadata = frozendict({} if metadata is None else metadata)
        self.continuation = continuation
        self.sample_shape = sample_shape
        self.state_shape = shape
        self.solver_name = solver_name
        self.solver_id = resolved_solver_id
        self.resolved_method = resolved_solver_method

    @property
    def num_times(self) -> int:
        return int(self.times.size)

    @property
    def has_dense_interpolation(self) -> bool:
        """Whether dense evaluation is available between saved times."""
        return self.interpolation is not None

    def evaluate(
        self,
        query_times: ArrayLike,
        /,
        *,
        left: bool = True,
    ) -> Array:
        """Evaluate dense output with shape query_shape + state_shape."""
        if self.interpolation is None:
            raise ValueError(
                "MemoryEquationSolution has no dense interpolation; "
                "use a solver with dense output enabled."
            )
        return self.interpolation.evaluate(query_times, left=left)

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
        record = _TrajectoryRecord(
            self.times,
            self.states,
            state_shape=self.state_shape,
            realization_shape=self.sample_shape,
            valid=self.valid,
            realizations=(self.realization,),
            solver_name=self.solver_name,
            solver_id=self.solver_id,
            resolved_method=self.resolved_method,
            uncertainty_source="process",
            metadata=self.metadata,
        )
        return record.to_stochastic_trajectory(
            realization_axes=axes,
            state_axes=resolved_state_axes,
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
    grid = eqx.error_if(
        grid,
        ~jnp.all(jnp.isfinite(grid)) | jnp.any(jnp.diff(grid) <= 0.0),
        "times must be finite and strictly increasing.",
    )
    tolerance = (
        100.0
        * jnp.finfo(grid.dtype).eps
        * jnp.maximum(1.0, jnp.maximum(jnp.abs(t0), jnp.abs(t1)))
    )
    return eqx.error_if(
        grid,
        (jnp.abs(grid[0] - t0) > tolerance) | (jnp.abs(grid[-1] - t1) > tolerance),
        "times must span exactly from problem t0 through t1.",
    )


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
                def contribute(values):
                    drift_sum, noise_sum = values
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
                    drift_sum = drift_sum + step_sizes[
                        source_index
                    ] * _weighted_kernel_value(
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
                                f"diffusion must return shape {expected}; "
                                f"got {diffusion.shape}."
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

                return jax.lax.cond(
                    source_index < index,
                    contribute,
                    lambda values: values,
                    accumulators,
                )

            drift_sum, noise_sum = jax.lax.fori_loop(
                0,
                num_steps,
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
        solver_id="solver:volterra:left-convolution-euler:v1",
        resolved_method="explicit-left-convolution",
        stats={
            "num_steps": num_steps,
            "num_accepted_steps": num_steps,
            "num_rejected_steps": 0,
        },
        metadata={
            "problem_id": problem.problem_id,
            "num_steps": num_steps,
            "quadrature": "left",
        },
    )


def solve_convolution_volterra(
    problem: ConvolutionVolterraProblem,
    /,
    *,
    times: ArrayLike,
    realization: WienerRealization | None = None,
) -> MemoryEquationSolution:
    """Solve a causal translation-invariant convolution by direct accumulation."""
    if not isinstance(problem, ConvolutionVolterraProblem):
        raise TypeError("problem must be a ConvolutionVolterraProblem.")
    native = solve_stochastic_volterra(
        problem.volterra,
        times=times,
        realization=realization,
    )
    return MemoryEquationSolution(
        times=native.times,
        states=native.states,
        valid=native.valid,
        interpolation=native.interpolation,
        backend_result=native.backend_result,
        stats=native.stats,
        event_mask=native.event_mask,
        realization=native.realization,
        state_shape=native.state_shape,
        solver_name="CausalConvolutionEuler",
        solver_id="solver:volterra:causal-convolution-euler:v1",
        resolved_method="explicit-left-causal-convolution",
        metadata={
            **dict(native.metadata),
            "kernel_structure": "translation-invariant",
            "convolution_backend": "direct-causal",
        },
    )


__all__ = [
    "ConvolutionKernel",
    "ConvolutionVolterraProblem",
    "MemoryEquationSolution",
    "StochasticVolterraProblem",
    "VolterraFreeTerm",
    "VolterraKernel",
    "VolterraVectorField",
    "solve_convolution_volterra",
    "solve_stochastic_volterra",
]
