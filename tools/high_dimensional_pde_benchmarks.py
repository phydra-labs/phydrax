from __future__ import annotations

import argparse
import json
import platform
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from jaxtyping import Array, Key

import phydrax as phx
from phydrax.operators import (
    coordinate_second_derivative_samples,
    DimensionSamplingPolicy,
    stochastic_trace_samples,
    StochasticTracePolicy,
)


BenchmarkProblem = Literal[
    "quadratic-heat",
    "linear-hjb",
    "ornstein-uhlenbeck-score",
    "quartic-laplacian",
]


@dataclass(frozen=True)
class HighDimensionalMethodSpec:
    problem_id: str
    method: str
    target_type: str
    supported_dimensions: tuple[int, ...]
    stochastic_error: bool


HIGH_DIMENSIONAL_METHOD_MATRIX = (
    HighDimensionalMethodSpec(
        "quadratic-heat",
        "query-feynman-kac",
        "point-value",
        (10, 50, 100, 500, 1000),
        True,
    ),
    HighDimensionalMethodSpec(
        "linear-hjb",
        "deep-picard",
        "global-function",
        (10, 50, 100),
        True,
    ),
    HighDimensionalMethodSpec(
        "linear-hjb",
        "deep-bsde",
        "point-value",
        (10, 50, 100),
        True,
    ),
    HighDimensionalMethodSpec(
        "linear-hjb",
        "deep-splitting",
        "time-slice-field",
        (10, 50, 100),
        True,
    ),
    HighDimensionalMethodSpec(
        "quartic-laplacian",
        "hutchinson-trace",
        "differential-operator",
        (10, 50, 100, 500, 1000),
        True,
    ),
    HighDimensionalMethodSpec(
        "quartic-laplacian",
        "dimension-sampling",
        "differential-operator",
        (10, 50, 100, 500, 1000),
        True,
    ),
    HighDimensionalMethodSpec(
        "ornstein-uhlenbeck-score",
        "implicit-score-matching",
        "score-field",
        (10, 50, 100, 500, 1000),
        True,
    ),
)


@dataclass(frozen=True)
class HighDimensionalBenchmarkRecord:
    problem_id: str
    method: str
    dimension: int
    seed: int
    value: float
    reference: float
    absolute_error: float
    reported_standard_error: float | None
    num_samples: int
    compile_ms: float
    mean_wall_ms: float
    finite: bool

    target_type: str = "point-value"
    gradient_error: float | None = None
    residual_error: float | None = None
    empirical_standard_error: float | None = None
    valid_fraction: float = 1.0
    working_set_bytes: int | None = None
    control_error: float | None = None
    control_standard_error: float | None = None
    terminal_error: float | None = None
    acceptance_tolerance: float | None = None
    total_wall_ms: float | None = None
    status: str = "completed"
    success: bool = True

    @property
    def passed(self) -> bool:
        if not self.finite or not self.success or self.status != "completed":
            return False
        if self.acceptance_tolerance is not None:
            return self.absolute_error <= self.acceptance_tolerance
        if self.reported_standard_error is None:
            return self.absolute_error < 1e-10
        return self.absolute_error <= 5.0 * self.reported_standard_error + 1e-10


@dataclass(frozen=True)
class SemidiscretePDEBenchmarkRecord:
    grid_size: int
    compilation_id: str
    operator_id: str
    resolved_method: str
    compiler_wall_ms: float
    compiled_jit_ms: float
    compiled_mean_wall_ms: float
    handwritten_jit_ms: float
    handwritten_mean_wall_ms: float
    maximum_drift_error: float
    parameter_gradient_error: float
    finite: bool

    @property
    def passed(self) -> bool:
        return (
            self.finite
            and self.maximum_drift_error <= 1e-10
            and self.parameter_gradient_error <= 1e-9
        )


class _LinearHJBValue(eqx.Module):
    time_coefficient: Array

    def __call__(self, time: Array, state: Array, *, key=None) -> Array:
        del key
        return jnp.asarray(
            [jnp.mean(state) + self.time_coefficient * (1.0 - time)]
        )


class _LinearHJBControl(eqx.Module):
    coefficient: Array
    dimension: int = eqx.field(static=True)

    def __call__(self, time: Array, state: Array, *, key=None) -> Array:
        del time, state, key
        return jnp.full((1, self.dimension), self.coefficient)

class _OrnsteinUhlenbeckScore(eqx.Module):
    variance: Array

    def __call__(self, state: Array, time: Array, *, key=None) -> Array:
        del time, key
        return -state / self.variance



def quadratic_heat_terminal(state: Array, /) -> Array:
    """Terminal value whose Brownian conditional expectation is analytic."""
    value = jnp.asarray(state)
    return jnp.mean(value**2, axis=-1)


def quadratic_heat_value(
    time: Array,
    state: Array,
    /,
    *,
    terminal_time: float = 1.0,
    diffusion_scale: float = 1.0,
) -> Array:
    value = jnp.asarray(state)
    return jnp.mean(value**2, axis=-1) + float(diffusion_scale) ** 2 * (
        float(terminal_time) - jnp.asarray(time)
    )


def quadratic_heat_gradient(state: Array, /) -> Array:
    value = jnp.asarray(state)
    return 2.0 * value / float(value.shape[-1])


def linear_hjb_terminal(state: Array, /) -> Array:
    value = jnp.asarray(state)
    return jnp.mean(value, axis=-1)


def linear_hjb_value(
    time: Array,
    state: Array,
    /,
    *,
    terminal_time: float = 1.0,
) -> Array:
    value = jnp.asarray(state)
    dimension = int(value.shape[-1])
    gradient_norm_squared = 1.0 / float(dimension)
    return jnp.mean(value, axis=-1) + 0.5 * gradient_norm_squared * (
        float(terminal_time) - jnp.asarray(time)
    )


def linear_hjb_gradient(state: Array, /) -> Array:
    value = jnp.asarray(state)
    return jnp.full_like(value, 1.0 / float(value.shape[-1]))


def ornstein_uhlenbeck_variance(
    time: Array,
    /,
    *,
    rate: float = 1.0,
    diffusion_scale: float = 1.0,
    initial_variance: float = 1.0,
) -> Array:
    t = jnp.asarray(time)
    decay = jnp.exp(-2.0 * float(rate) * t)
    stationary = float(diffusion_scale) ** 2 / (2.0 * float(rate))
    return float(initial_variance) * decay + stationary * (1.0 - decay)


def ornstein_uhlenbeck_score(
    time: Array,
    state: Array,
    /,
    *,
    rate: float = 1.0,
    diffusion_scale: float = 1.0,
    initial_variance: float = 1.0,
) -> Array:
    variance = ornstein_uhlenbeck_variance(
        time,
        rate=rate,
        diffusion_scale=diffusion_scale,
        initial_variance=initial_variance,
    )
    return -jnp.asarray(state) / jnp.expand_dims(variance, axis=-1)


def quartic_field(state: Array, /) -> Array:
    value = jnp.asarray(state)
    return jnp.mean(value**4, axis=-1)


def quartic_laplacian(state: Array, /) -> Array:
    value = jnp.asarray(state)
    return 12.0 * jnp.mean(value**2, axis=-1)


def _block(value: Any) -> None:
    jax.block_until_ready(value)


def _measure(
    function,
    argument: Array,
    /,
    *,
    repeats: int,
) -> tuple[Array, float, float]:
    compiled = jax.jit(function)
    started = time.perf_counter()
    value = compiled(argument)
    _block(value)
    compile_ms = 1e3 * (time.perf_counter() - started)
    started = time.perf_counter()
    for _ in range(int(repeats)):
        value = compiled(argument)
        _block(value)
    wall_ms = 1e3 * (time.perf_counter() - started) / float(repeats)
    return value, compile_ms, wall_ms


def _measure_thunk(
    function,
    /,
    *,
    repeats: int,
):
    started = time.perf_counter()
    value = function()
    _block(value)
    first_ms = 1e3 * (time.perf_counter() - started)
    started = time.perf_counter()
    for _ in range(int(repeats)):
        value = function()
        _block(value)
    wall_ms = 1e3 * (time.perf_counter() - started) / float(repeats)
    return value, max(first_ms - wall_ms, 0.0), wall_ms, first_ms


def run_semidiscrete_pde_compiler_benchmark(
    grid_size: int = 128,
    /,
    *,
    repeats: int = 5,
) -> SemidiscretePDEBenchmarkRecord:
    """Compare compiled and handwritten periodic reaction-diffusion drifts."""
    size = int(grid_size)
    repeat_count = int(repeats)
    if size < 4 or repeat_count <= 0:
        raise ValueError("grid_size must be at least four and repeats must be positive.")

    x = phx.equations.PDECoordinate(
        "x",
        "space",
        bounds=(0.0, 1.0),
        periodic=True,
    )
    t = phx.equations.PDECoordinate("t", "time", bounds=(0.0, 1.0))
    field = phx.equations.PDEField("u", coordinates=("x", "t"))
    parameter = phx.equations.PDEParameter("kappa", value=0.05)
    u = phx.equations.PDEExpression.field("u")
    problem = phx.equations.PDEProblemIR(
        coordinates=(x, t),
        fields=(field,),
        parameters=(parameter,),
        equations=(
            phx.equations.PDEEquation(
                "reaction-diffusion",
                u.derivative("t"),
                phx.equations.PDEExpression.parameter("kappa")
                * u.laplacian("x")
                + u * (1.0 - u),
            ),
        ),
    )
    axis = phx.domain.FourierAxisSpec(size).materialize(0.0, 1.0)
    spatial = phx.solver.TensorGridDiscretization((axis,))
    started = time.perf_counter()
    compiled = phx.equations.compile_semidiscrete_pde(problem, spatial)
    compiler_wall_ms = 1e3 * (time.perf_counter() - started)

    state = 0.2 + 0.1 * jnp.sin(2.0 * jnp.pi * axis.nodes)
    coefficient = jnp.asarray(0.07)

    def compiled_drift(arguments):
        value, diffusivity = arguments
        return compiled(0.0, value, {"kappa": diffusivity})

    def handwritten_drift(arguments):
        value, diffusivity = arguments
        return diffusivity * spatial.laplacian(value) + value * (1.0 - value)

    benchmark_arguments = (state, coefficient)
    compiled_value, compiled_jit_ms, compiled_wall_ms = _measure(
        compiled_drift,
        benchmark_arguments,
        repeats=repeat_count,
    )
    handwritten_value, handwritten_jit_ms, handwritten_wall_ms = _measure(
        handwritten_drift,
        benchmark_arguments,
        repeats=repeat_count,
    )
    compiled_gradient = jax.grad(
        lambda diffusivity: jnp.sum(
            compiled_drift((state, diffusivity)) ** 2
        )
    )(coefficient)
    handwritten_gradient = jax.grad(
        lambda diffusivity: jnp.sum(
            handwritten_drift((state, diffusivity)) ** 2
        )
    )(coefficient)
    maximum_error = jnp.max(jnp.abs(compiled_value - handwritten_value))
    gradient_error = jnp.abs(compiled_gradient - handwritten_gradient)
    finite = bool(
        jnp.all(jnp.isfinite(compiled_value))
        & jnp.isfinite(compiled_gradient)
        & jnp.isfinite(maximum_error)
        & jnp.isfinite(gradient_error)
    )
    return SemidiscretePDEBenchmarkRecord(
        grid_size=size,
        compilation_id=compiled.compilation_id,
        operator_id=compiled.semilinear_drift.operator_id,
        resolved_method=compiled.resolved_method,
        compiler_wall_ms=float(compiler_wall_ms),
        compiled_jit_ms=float(compiled_jit_ms),
        compiled_mean_wall_ms=float(compiled_wall_ms),
        handwritten_jit_ms=float(handwritten_jit_ms),
        handwritten_mean_wall_ms=float(handwritten_wall_ms),
        maximum_drift_error=float(maximum_error),
        parameter_gradient_error=float(gradient_error),
        finite=finite,
    )


def _quadratic_heat_problem(dimension: int, /) -> phx.stochastic.BSDEProblem:
    times = jnp.asarray([0.0, 1.0])
    process_id = f"rank-one-brownian-{dimension}"

    def forward_sampler(key):
        increments = jr.normal(key, (1, 1, 1))
        initial = jnp.zeros((1, 1, dimension), dtype=increments.dtype)
        terminal = initial + jnp.broadcast_to(increments, initial.shape)
        return phx.stochastic.BSDEPathBatch(
            times,
            jnp.concatenate((initial, terminal), axis=1),
            increments,
            sample_shape=(1,),
            state_shape=(dimension,),
            noise_shape=(1,),
            path_id=f"quadratic-heat-forward-{dimension}",
            process_id=process_id,
        )

    return phx.stochastic.BSDEProblem(
        forward_sampler,
        lambda current_time, state, args: jnp.zeros_like(state),
        lambda current_time, state, args: jnp.ones(
            (dimension, 1),
            dtype=state.dtype,
        ),
        lambda current_time, state, value, control, args: jnp.zeros_like(value),
        lambda state, args: jnp.asarray([quadratic_heat_terminal(state)]),
        state_shape=(dimension,),
        noise_shape=(1,),
        output_shape=(1,),
        problem_id=f"quadratic-heat-{dimension}",
        process_id=process_id,
    )


def _query_feynman_kac_record(
    key: Key[Array, ""],
    dimension: int,
    /,
    *,
    num_samples: int,
    repeats: int,
    seed: int,
) -> HighDimensionalBenchmarkRecord:
    state_key, label_key = jr.split(key)
    state = jr.normal(state_key, (dimension,))
    problem = _quadratic_heat_problem(dimension)
    plan = phx.stochastic.FeynmanKacSamplingPlan(
        terminal_time=1.0,
        sampling_mode="queries",
        num_paths_per_query=num_samples,
        num_time_steps=1,
        control_target_mode="martingale",
    )

    def operation():
        labels, paths = phx.stochastic.query_feynman_kac_labels(
            problem,
            plan,
            query_times=jnp.asarray([0.0]),
            query_states=state[None, :],
            key=label_key,
            return_paths=True,
        )
        if labels.control_targets is None or labels.control_standard_errors is None:
            raise RuntimeError("Martingale benchmark labels require control targets.")
        return (
            labels.value_targets,
            labels.value_standard_errors,
            labels.control_targets,
            labels.control_standard_errors,
            labels.valid,
            paths.states,
            paths.wiener_increments,
        )

    measured, compile_ms, wall_ms, total_ms = _measure_thunk(
        operation,
        repeats=repeats,
    )
    (
        value_targets,
        value_errors,
        control_targets,
        control_errors,
        valid,
        path_states,
        path_increments,
    ) = measured
    value = float(value_targets[0, 0])
    reference = float(quadratic_heat_value(jnp.asarray(0.0), state))
    control = float(control_targets[0, 0, 0])
    control_reference = float(2.0 * jnp.mean(state))
    standard_error = float(value_errors[0, 0])
    control_standard_error = float(control_errors[0, 0, 0])
    control_error = abs(control - control_reference)
    return HighDimensionalBenchmarkRecord(
        problem_id="quadratic-heat",
        method="query-feynman-kac",
        dimension=dimension,
        seed=seed,
        value=value,
        reference=reference,
        absolute_error=abs(value - reference),
        reported_standard_error=standard_error,
        num_samples=num_samples,
        compile_ms=compile_ms,
        mean_wall_ms=wall_ms,
        finite=all(
            isfinite(item)
            for item in (value, standard_error, control, control_standard_error)
        ),
        target_type="point-value",
        control_error=control_error,
        control_standard_error=control_standard_error,
        valid_fraction=float(jnp.mean(valid)),
        working_set_bytes=int(path_states.nbytes + path_increments.nbytes),
        total_wall_ms=total_ms,
        success=control_error <= 5.0 * control_standard_error + 1e-10,
    )


def _direct_quadratic_heat_record(
    key: Key[Array, ""],
    dimension: int,
    /,
    *,
    num_samples: int,
    repeats: int,
    seed: int,
) -> HighDimensionalBenchmarkRecord:
    state_key, noise_key = jr.split(key)
    state = jr.normal(state_key, (dimension,))
    noise = jr.normal(noise_key, (num_samples, dimension))
    terminal_states = state + noise
    samples = quadratic_heat_terminal(terminal_states)
    reference = float(quadratic_heat_value(jnp.asarray(0.0), state))
    operation = lambda values: jnp.mean(quadratic_heat_terminal(values))
    estimate, compile_ms, wall_ms = _measure(
        operation,
        terminal_states,
        repeats=repeats,
    )
    standard_error = jnp.std(samples, ddof=1) / jnp.sqrt(float(num_samples))
    value = float(estimate)
    error = abs(value - reference)
    return HighDimensionalBenchmarkRecord(
        problem_id="quadratic-heat",
        method="direct-monte-carlo",
        dimension=dimension,
        seed=seed,
        value=value,
        reference=reference,
        absolute_error=error,
        reported_standard_error=float(standard_error),
        num_samples=num_samples,
        compile_ms=compile_ms,
        mean_wall_ms=wall_ms,
        finite=isfinite(value) and isfinite(float(standard_error)),
    )


def _analytic_record(
    problem: BenchmarkProblem,
    key: Key[Array, ""],
    dimension: int,
    /,
    *,
    repeats: int,
    seed: int,
) -> HighDimensionalBenchmarkRecord:
    state = jr.normal(key, (dimension,))
    if problem == "linear-hjb":
        operation = lambda value: linear_hjb_value(jnp.asarray(0.0), value)
    elif problem == "ornstein-uhlenbeck-score":
        operation = lambda value: jnp.linalg.norm(
            ornstein_uhlenbeck_score(jnp.asarray(0.5), value)
        )
    elif problem == "quartic-laplacian":
        operation = lambda value: quartic_laplacian(value)
    else:
        raise ValueError(f"Unsupported analytic benchmark {problem!r}.")
    estimate, compile_ms, wall_ms = _measure(operation, state, repeats=repeats)
    value = float(estimate)
    return HighDimensionalBenchmarkRecord(
        problem_id=problem,
        method="analytic-reference",
        dimension=dimension,
        seed=seed,
        value=value,
        reference=value,
        absolute_error=0.0,
        reported_standard_error=None,
        num_samples=1,
        compile_ms=compile_ms,
        mean_wall_ms=wall_ms,
        finite=isfinite(value),
    )


def _linear_hjb_problem(
    dimension: int,
    /,
    *,
    num_paths: int = 1,
    num_time_steps: int = 1,
    antithetic: bool = False,
) -> phx.stochastic.BSDEProblem:
    process_id = f"linear-hjb-{dimension}"
    path_count = int(num_paths)
    step_count = int(num_time_steps)
    if path_count < 1 or step_count < 1:
        raise ValueError("Linear HJB path and step counts must be positive.")
    if antithetic and path_count % 2:
        raise ValueError("Antithetic Linear HJB paths require an even path count.")
    times = jnp.linspace(0.0, 1.0, step_count + 1)

    def forward_sampler(key):
        independent_count = path_count // 2 if antithetic else path_count
        independent = jr.normal(key, (independent_count, step_count, dimension))
        increments = (
            jnp.concatenate((independent, -independent), axis=0)
            if antithetic
            else independent
        ) / jnp.sqrt(float(step_count))
        initial = jnp.zeros((path_count, 1, dimension), dtype=increments.dtype)
        states = jnp.concatenate((initial, jnp.cumsum(increments, axis=1)), axis=1)
        return phx.stochastic.BSDEPathBatch(
            times,
            states,
            increments,
            sample_shape=(path_count,),
            state_shape=(dimension,),
            noise_shape=(dimension,),
            path_id=f"{process_id}-forward",
            process_id=process_id,
        )

    return phx.stochastic.BSDEProblem(
        forward_sampler,
        lambda current_time, state, args: jnp.zeros_like(state),
        lambda current_time, state, args: jnp.eye(dimension, dtype=state.dtype),
        lambda current_time, state, value, control, args: jnp.asarray(
            [0.5 * jnp.sum(control**2)]
        ),
        lambda state, args: jnp.asarray([linear_hjb_terminal(state)]),
        state_shape=(dimension,),
        noise_shape=(dimension,),
        output_shape=(1,),
        problem_id=process_id,
        process_id=process_id,
    )


def _linear_hjb_solver(dimension: int, /) -> phx.solver.FunctionalSolver:
    domain = phx.domain.HyperRectangle(
        jnp.full((dimension,), -4.0),
        jnp.full((dimension,), 4.0),
        label="x",
    ) @ phx.domain.TimeInterval(0.0, 1.0)
    value = phx.domain.DomainFunction(
        domain=domain,
        deps=("t", "x"),
        func=_LinearHJBValue(jnp.asarray(0.0)),
    )
    return phx.solver.FunctionalSolver(
        functions={"value": value},
        terms=(),
    )

def _linear_hjb_shooting_solver(dimension: int, /) -> phx.solver.FunctionalSolver:
    domain = phx.domain.HyperRectangle(
        jnp.full((dimension,), -4.0),
        jnp.full((dimension,), 4.0),
        label="x",
    ) @ phx.domain.TimeInterval(0.0, 1.0)
    control = phx.domain.DomainFunction(
        domain=domain,
        deps=("t", "x"),
        func=_LinearHJBControl(jnp.asarray(0.0), dimension),
    )
    return phx.solver.FunctionalSolver(
        functions={
            "initial": domain.Parameter(jnp.asarray([0.0])),
            "control": control,
        },
        terms=(),
    )


def _deep_picard_record(
    key: Key[Array, ""],
    dimension: int,
    /,
    *,
    num_paths: int,
    num_queries: int,
    inner_num_iter: int,
    seed: int,
) -> HighDimensionalBenchmarkRecord:
    problem = _linear_hjb_problem(dimension)
    query_key, _ = jr.split(key)
    query_times = jnp.linspace(0.0, 0.9, num_queries)
    query_states = jr.normal(query_key, (num_queries, dimension))
    plan = phx.stochastic.FeynmanKacSamplingPlan(
        terminal_time=1.0,
        sampling_mode="queries",
        num_paths_per_query=num_paths,
        num_time_steps=4,
        antithetic=True,
        refresh_mode="fixed",
    )

    def solve_once():
        return phx.solver.solve_deep_picard(
            _linear_hjb_solver(dimension),
            problem,
            value_name="value",
            sampling_plan=plan,
            num_picard_steps=1,
            inner_num_iter=inner_num_iter,
            optim=optax.adam(0.01),
            query_times=query_times,
            query_states=query_states,
            initial_source="current",
            seed=seed,
            jit=True,
            keep_best=False,
        )

    started = time.perf_counter()
    first = solve_once()
    _block(first.diagnostics.target_rmse)
    first_ms = 1e3 * (time.perf_counter() - started)
    started = time.perf_counter()
    result = solve_once()
    _block(result.diagnostics.target_rmse)
    steady_ms = 1e3 * (time.perf_counter() - started)

    predictor = result.solver["value"]
    prediction_key = jr.key(seed)
    predictions = jax.vmap(
        lambda current_time, state: predictor.func(
            current_time,
            state,
            key=prediction_key,
        )[0]
    )(query_times, query_states)
    references = linear_hjb_value(query_times, query_states)
    global_rmse = float(jnp.sqrt(jnp.mean((predictions - references) ** 2)))
    gradient = jax.grad(
        lambda state: predictor.func(
            query_times[0],
            state,
            key=prediction_key,
        )[0]
    )(query_states[0])
    gradient_error = float(
        jnp.sqrt(jnp.mean((gradient - linear_hjb_gradient(query_states[0])) ** 2))
    )
    target_error = float(result.diagnostics.target_rmse[-1])
    terminal_error = float(result.diagnostics.terminal_rmse[-1])
    finite = all(
        isfinite(value)
        for value in (
            global_rmse,
            gradient_error,
            target_error,
            terminal_error,
        )
    )
    return HighDimensionalBenchmarkRecord(
        problem_id="linear-hjb",
        method="deep-picard",
        dimension=dimension,
        seed=seed,
        value=global_rmse,
        reference=0.0,
        absolute_error=global_rmse,
        reported_standard_error=None,
        num_samples=num_paths * num_queries,
        compile_ms=max(first_ms - steady_ms, 0.0),
        mean_wall_ms=steady_ms / float(inner_num_iter),
        finite=finite,
        target_type="global-function",
        gradient_error=gradient_error,
        residual_error=target_error,
        terminal_error=terminal_error,
        valid_fraction=float(result.diagnostics.valid_fraction[-1]),
        working_set_bytes=int(
            query_states.nbytes
            + num_queries * num_paths * 5 * dimension * 8
        ),
        acceptance_tolerance=1e-3,
        total_wall_ms=steady_ms,
        success=bool(result.diagnostics.finite[-1])
        and max(gradient_error, target_error, terminal_error) <= 1e-3,
    )

def _deep_bsde_record(
    key: Key[Array, ""],
    dimension: int,
    /,
    *,
    num_paths: int,
    num_time_steps: int,
    num_iter: int,
    seed: int,
) -> HighDimensionalBenchmarkRecord:
    problem = _linear_hjb_problem(
        dimension,
        num_paths=num_paths,
        num_time_steps=num_time_steps,
        antithetic=True,
    )
    validation_paths = problem.sample(jr.fold_in(key, 1))

    def solve_once():
        return phx.solver.solve_deep_bsde(
            _linear_hjb_shooting_solver(dimension),
            problem,
            initial_value_name="initial",
            control_name="control",
            num_iter=num_iter,
            optim=optax.adam(0.05),
            sampling_mode="resample",
            validation_paths=validation_paths,
            seed=seed,
            jit=True,
            keep_best=False,
        )

    started = time.perf_counter()
    first = solve_once()
    _block(first.diagnostics.terminal_rmse)
    first_ms = 1e3 * (time.perf_counter() - started)
    started = time.perf_counter()
    result = solve_once()
    _block(result.diagnostics.terminal_rmse)
    steady_ms = 1e3 * (time.perf_counter() - started)

    zero_state = jnp.zeros((dimension,))
    initial_value = float(result.solver["initial"].func(key=key)[0])
    reference = float(linear_hjb_value(jnp.asarray(0.0), zero_state))
    control = result.solver["control"].func(
        jnp.asarray(0.0),
        zero_state,
        key=key,
    )[0]
    control_error = float(
        jnp.sqrt(jnp.mean((control - linear_hjb_gradient(zero_state)) ** 2))
    )
    terminal_error = float(result.diagnostics.terminal_rmse)
    error = abs(initial_value - reference)
    finite = all(
        isfinite(value)
        for value in (initial_value, reference, error, control_error, terminal_error)
    )
    tolerance = 5e-3
    return HighDimensionalBenchmarkRecord(
        problem_id="linear-hjb",
        method="deep-bsde",
        dimension=dimension,
        seed=seed,
        value=initial_value,
        reference=reference,
        absolute_error=error,
        reported_standard_error=None,
        num_samples=num_paths * num_time_steps,
        compile_ms=max(first_ms - steady_ms, 0.0),
        mean_wall_ms=steady_ms / float(num_iter),
        finite=finite,
        target_type="point-value",
        control_error=control_error,
        residual_error=terminal_error,
        terminal_error=terminal_error,
        valid_fraction=float(result.diagnostics.valid_fraction),
        working_set_bytes=int(
            result.rollout.paths.states.nbytes
            + result.rollout.paths.wiener_increments.nbytes
        ),
        acceptance_tolerance=tolerance,
        total_wall_ms=steady_ms,
        success=bool(result.diagnostics.finite)
        and max(error, control_error, terminal_error) <= tolerance,
    )


def _deep_splitting_record(
    key: Key[Array, ""],
    dimension: int,
    /,
    *,
    num_paths: int,
    num_time_steps: int,
    inner_num_iter: int,
    seed: int,
) -> HighDimensionalBenchmarkRecord:
    problem = _linear_hjb_problem(
        dimension,
        num_paths=num_paths,
        num_time_steps=num_time_steps,
        antithetic=True,
    )
    training_paths = problem.sample(jr.fold_in(key, 1))
    validation_paths = problem.sample(jr.fold_in(key, 2))

    def solve_once():
        return phx.solver.solve_deep_splitting(
            _linear_hjb_solver(dimension),
            problem,
            value_name="value",
            inner_num_iter=inner_num_iter,
            optim=optax.adam(0.05),
            sampling_mode="fixed",
            fixed_paths=training_paths,
            validation_paths=validation_paths,
            seed=seed,
            jit=True,
            keep_best=False,
        )

    started = time.perf_counter()
    first = solve_once()
    _block(first.diagnostics.one_step_rmse)
    first_ms = 1e3 * (time.perf_counter() - started)
    started = time.perf_counter()
    result = solve_once()
    _block(result.diagnostics.one_step_rmse)
    steady_ms = 1e3 * (time.perf_counter() - started)

    query_times = jnp.linspace(0.0, 1.0, 2 * num_time_steps + 1)
    query_states = jr.normal(
        jr.fold_in(key, 3),
        (query_times.shape[0], dimension),
    )
    predictions = jax.vmap(
        lambda current_time, state: result.solution(current_time, state)[0]
    )(query_times, query_states)
    references = linear_hjb_value(query_times, query_states)
    global_rmse = float(jnp.sqrt(jnp.mean((predictions - references) ** 2)))
    zero_state = jnp.zeros((dimension,))
    control = result.solution.control(jnp.asarray(0.0), zero_state)[0]
    gradient_error = float(
        jnp.sqrt(jnp.mean((control - linear_hjb_gradient(zero_state)) ** 2))
    )
    terminal_error = abs(
        float(result.solution.at_node(num_time_steps, zero_state)[0])
        - float(linear_hjb_terminal(zero_state))
    )
    one_step_error = float(jnp.max(result.diagnostics.one_step_rmse))
    finite = all(
        isfinite(value)
        for value in (
            global_rmse,
            gradient_error,
            terminal_error,
            one_step_error,
        )
    )
    tolerance = 5e-3
    return HighDimensionalBenchmarkRecord(
        problem_id="linear-hjb",
        method="deep-splitting",
        dimension=dimension,
        seed=seed,
        value=global_rmse,
        reference=0.0,
        absolute_error=global_rmse,
        reported_standard_error=None,
        num_samples=num_paths * num_time_steps,
        compile_ms=max(first_ms - steady_ms, 0.0),
        mean_wall_ms=steady_ms / float(num_time_steps * inner_num_iter),
        finite=finite,
        target_type="time-slice-field",
        gradient_error=gradient_error,
        residual_error=one_step_error,
        terminal_error=terminal_error,
        valid_fraction=float(jnp.min(result.diagnostics.valid_fraction)),
        working_set_bytes=int(
            training_paths.states.nbytes
            + training_paths.wiener_increments.nbytes
            + validation_paths.states.nbytes
            + validation_paths.wiener_increments.nbytes
        ),
        acceptance_tolerance=tolerance,
        total_wall_ms=steady_ms,
        success=bool(jnp.all(result.diagnostics.finite))
        and max(global_rmse, gradient_error, terminal_error) <= tolerance,
    )


def _hutchinson_laplacian_record(
    key: Key[Array, ""],
    dimension: int,
    /,
    *,
    num_probes: int,
    repeats: int,
    seed: int,
) -> HighDimensionalBenchmarkRecord:
    state_key, probe_key = jr.split(key)
    state = jr.normal(state_key, (dimension,))
    policy = StochasticTracePolicy(num_probes, distribution="normal")

    def operation(value):
        return stochastic_trace_samples(
            quartic_field,
            value,
            lambda current, direction: direction,
            probe_key,
            policy=policy,
        ).mean

    estimate, compile_ms, wall_ms = _measure(operation, state, repeats=repeats)
    samples = stochastic_trace_samples(
        quartic_field,
        state,
        lambda current, direction: direction,
        probe_key,
        policy=policy,
    )
    value = float(estimate)
    reference = float(quartic_laplacian(state))
    standard_error = float(samples.standard_error)
    return HighDimensionalBenchmarkRecord(
        problem_id="quartic-laplacian",
        method="hutchinson-trace",
        dimension=dimension,
        seed=seed,
        value=value,
        reference=reference,
        absolute_error=abs(value - reference),
        reported_standard_error=standard_error,
        num_samples=num_probes,
        compile_ms=compile_ms,
        mean_wall_ms=wall_ms,
        finite=isfinite(value) and isfinite(standard_error),
        target_type="differential-operator",
        residual_error=abs(value - reference),
        working_set_bytes=int((num_probes + 1) * state.nbytes),
    )


def _dimension_laplacian_record(
    key: Key[Array, ""],
    dimension: int,
    /,
    *,
    num_probes: int,
    repeats: int,
    seed: int,
) -> HighDimensionalBenchmarkRecord:
    state_key, sample_key = jr.split(key)
    state = jr.normal(state_key, (dimension,))
    count = min(int(num_probes), dimension)
    policy = DimensionSamplingPolicy(dimension, count)

    def operation(value):
        return coordinate_second_derivative_samples(
            quartic_field,
            value,
            sample_key,
            policy,
        ).mean

    estimate, compile_ms, wall_ms = _measure(operation, state, repeats=repeats)
    samples = coordinate_second_derivative_samples(
        quartic_field,
        state,
        sample_key,
        policy,
    )
    value = float(estimate)
    reference = float(quartic_laplacian(state))
    standard_error = float(samples.standard_error)
    return HighDimensionalBenchmarkRecord(
        problem_id="quartic-laplacian",
        method="dimension-sampling",
        dimension=dimension,
        seed=seed,
        value=value,
        reference=reference,
        absolute_error=abs(value - reference),
        reported_standard_error=standard_error,
        num_samples=count,
        compile_ms=compile_ms,
        mean_wall_ms=wall_ms,
        finite=isfinite(value) and isfinite(standard_error),
        target_type="differential-operator",
        residual_error=abs(value - reference),
        working_set_bytes=int((count + 1) * state.nbytes),
    )


def _implicit_score_record(
    key: Key[Array, ""],
    dimension: int,
    /,
    *,
    num_samples: int,
    num_probes: int,
    repeats: int,
    seed: int,
) -> HighDimensionalBenchmarkRecord:
    sample_key, objective_key = jr.split(key)
    time_value = jnp.asarray(0.5)
    variance = ornstein_uhlenbeck_variance(time_value)
    states = jnp.sqrt(variance) * jr.normal(
        sample_key,
        (num_samples, 1, dimension),
    )
    trajectory = phx.stochastic.StochasticTrajectory(
        jnp.asarray([time_value]),
        states,
        realization_axes=("path",),
        realization_shape=(num_samples,),
        time_axis="saved_time",
        state_axes=("state",),
    )
    samples = phx.stochastic.trajectory_state_time_samples(
        trajectory,
        time_label="t",
    )
    space = phx.domain.HyperRectangle(
        jnp.full((dimension,), -10.0),
        jnp.full((dimension,), 10.0),
        label="x",
    )
    domain = space @ phx.domain.TimeInterval(0.0, 1.0)
    score = phx.domain.DomainFunction(
        domain=domain,
        deps=("x", "t"),
        func=_OrnsteinUhlenbeckScore(variance),
    )
    term = phx.terms.ScoreMatchingTerm(
        "score",
        samples,
        policy=phx.terms.ScoreMatchingPolicy(
            "implicit",
            num_probes=num_probes,
        ),
    )
    functions = {"score": score}

    def operation(value):
        del value
        return term.loss(functions, key=objective_key)

    estimate, compile_ms, wall_ms = _measure(
        operation,
        jnp.asarray(0.0),
        repeats=repeats,
    )
    diagnostics = term.diagnostics(functions, key=objective_key)
    value = float(estimate)
    reference = -0.5 * float(dimension) / float(variance)
    standard_error = float(diagnostics.path_standard_error)
    score_values = jax.vmap(
        lambda state: score.func(state, time_value, key=objective_key)
    )(states[:, 0, :])
    score_reference = ornstein_uhlenbeck_score(time_value, states[:, 0, :])
    score_error = float(
        jnp.sqrt(jnp.mean((score_values - score_reference) ** 2))
    )
    divergence_reference = -float(dimension) / float(variance)
    divergence_error = abs(
        float(diagnostics.mean_divergence) - divergence_reference
    )
    return HighDimensionalBenchmarkRecord(
        problem_id="ornstein-uhlenbeck-score",
        method="implicit-score-matching",
        dimension=dimension,
        seed=seed,
        value=value,
        reference=reference,
        absolute_error=abs(value - reference),
        reported_standard_error=standard_error,
        num_samples=num_samples,
        compile_ms=compile_ms,
        mean_wall_ms=wall_ms,
        finite=bool(diagnostics.finite) and isfinite(standard_error),
        target_type="score-field",
        gradient_error=score_error,
        residual_error=divergence_error,
        valid_fraction=float(diagnostics.valid_fraction),
        working_set_bytes=int((num_probes + 1) * states.nbytes),
        success=score_error <= 1e-10 and divergence_error <= 1e-10,
    )


def _benchmark_environment() -> dict[str, Any]:
    device = jax.devices()[0]
    return {
        "jax_version": jax.__version__,
        "python_version": platform.python_version(),
        "system": platform.system(),
        "system_release": platform.release(),
        "machine": platform.machine(),
        "backend": jax.default_backend(),
        "device_kind": device.device_kind,
        "x64_enabled": bool(jax.config.x64_enabled),
    }


def run_high_dimensional_method_benchmarks(
    dimensions: Sequence[int] = (10, 50, 100, 500, 1000),
    /,
    *,
    num_samples: int = 4096,
    num_probes: int = 64,
    repeats: int = 3,
    seed: int = 0,
    include_training: bool = False,
    score_samples: int = 128,
    deep_picard_paths: int = 32,
    deep_picard_queries: int = 16,
    deep_picard_iterations: int = 120,
    deep_bsde_paths: int = 32,
    deep_bsde_time_steps: int = 4,
    deep_bsde_iterations: int = 120,
    deep_splitting_paths: int = 32,
    deep_splitting_time_steps: int = 4,
    deep_splitting_iterations: int = 80,
) -> dict[str, Any]:
    """Exercise probabilistic, randomized, density, and optional training methods."""
    dims = tuple(int(value) for value in dimensions)
    if not dims or any(value < 2 for value in dims):
        raise ValueError("method benchmark dimensions must be at least two.")
    if (
        int(num_samples) < 2
        or int(num_probes) < 2
        or int(score_samples) < 2
    ):
        raise ValueError(
            "num_samples, num_probes, and score_samples must be at least two."
        )
    if int(repeats) < 1:
        raise ValueError("repeats must be positive.")
    picard_paths = int(deep_picard_paths)
    picard_queries = int(deep_picard_queries)
    picard_iterations = int(deep_picard_iterations)
    bsde_paths = int(deep_bsde_paths)
    bsde_time_steps = int(deep_bsde_time_steps)
    bsde_iterations = int(deep_bsde_iterations)
    splitting_paths = int(deep_splitting_paths)
    splitting_time_steps = int(deep_splitting_time_steps)
    splitting_iterations = int(deep_splitting_iterations)
    if picard_paths < 2 or picard_paths % 2:
        raise ValueError("deep_picard_paths must be an even integer of at least two.")
    if picard_queries < 1 or picard_iterations < 1:
        raise ValueError(
            "deep_picard_queries and deep_picard_iterations must be positive."
        )
    if bsde_paths < 2 or bsde_paths % 2:
        raise ValueError("deep_bsde_paths must be an even integer of at least two.")
    if bsde_time_steps < 1 or bsde_iterations < 1:
        raise ValueError(
            "deep_bsde_time_steps and deep_bsde_iterations must be positive."
        )
    if splitting_paths < 2 or splitting_paths % 2:
        raise ValueError(
            "deep_splitting_paths must be an even integer of at least two."
        )
    if splitting_time_steps < 1 or splitting_iterations < 1:
        raise ValueError(
            "deep_splitting_time_steps and deep_splitting_iterations must be positive."
        )
    deep_dimensions = {
        spec.method: spec.supported_dimensions
        for spec in HIGH_DIMENSIONAL_METHOD_MATRIX
        if spec.method in ("deep-picard", "deep-bsde", "deep-splitting")
    }
    root = jr.key(int(seed))
    records: list[HighDimensionalBenchmarkRecord] = []
    for dimension_index, dimension in enumerate(dims):
        dimension_key = jr.fold_in(root, dimension_index)
        records.append(
            _query_feynman_kac_record(
                jr.fold_in(dimension_key, 0),
                dimension,
                num_samples=int(num_samples),
                repeats=int(repeats),
                seed=int(seed),
            )
        )
        records.append(
            _hutchinson_laplacian_record(
                jr.fold_in(dimension_key, 1),
                dimension,
                num_probes=int(num_probes),
                repeats=int(repeats),
                seed=int(seed),
            )
        )
        records.append(
            _dimension_laplacian_record(
                jr.fold_in(dimension_key, 2),
                dimension,
                num_probes=int(num_probes),
                repeats=int(repeats),
                seed=int(seed),
            )
        )
        records.append(
            _implicit_score_record(
                jr.fold_in(dimension_key, 3),
                dimension,
                num_samples=int(score_samples),
                num_probes=int(num_probes),
                repeats=int(repeats),
                seed=int(seed),
            )
        )
        if include_training and dimension in deep_dimensions["deep-picard"]:
            records.append(
                _deep_picard_record(
                    jr.fold_in(dimension_key, 4),
                    dimension,
                    num_paths=picard_paths,
                    num_queries=picard_queries,
                    inner_num_iter=picard_iterations,
                    seed=int(seed),
                )
            )
        if include_training and dimension in deep_dimensions["deep-bsde"]:
            records.append(
                _deep_bsde_record(
                    jr.fold_in(dimension_key, 5),
                    dimension,
                    num_paths=bsde_paths,
                    num_time_steps=bsde_time_steps,
                    num_iter=bsde_iterations,
                    seed=int(seed),
                )
            )
        if include_training and dimension in deep_dimensions["deep-splitting"]:
            records.append(
                _deep_splitting_record(
                    jr.fold_in(dimension_key, 6),
                    dimension,
                    num_paths=splitting_paths,
                    num_time_steps=splitting_time_steps,
                    inner_num_iter=splitting_iterations,
                    seed=int(seed),
                )
            )
    return {
        "schema_version": 2,
        "seed": int(seed),
        "dimensions": list(dims),
        "configuration": {
            "num_samples": int(num_samples),
            "score_samples": int(score_samples),
            "num_probes": int(num_probes),
            "repeats": int(repeats),
            "deep_picard_paths": picard_paths,
            "deep_picard_queries": picard_queries,
            "deep_picard_iterations": picard_iterations,
            "deep_bsde_paths": bsde_paths,
            "deep_bsde_time_steps": bsde_time_steps,
            "deep_bsde_iterations": bsde_iterations,
            "deep_splitting_paths": splitting_paths,
            "deep_splitting_time_steps": splitting_time_steps,
            "deep_splitting_iterations": splitting_iterations,
        },
        "environment": _benchmark_environment(),
        "method_matrix": [asdict(spec) for spec in HIGH_DIMENSIONAL_METHOD_MATRIX],
        "training_methods_included": bool(include_training),
        "records": [asdict(record) | {"passed": record.passed} for record in records],
        "passed": all(record.passed for record in records),
    }


def run_high_dimensional_reference_benchmarks(
    dimensions: Sequence[int] = (10, 100),
    /,
    *,
    num_samples: int = 4096,
    repeats: int = 3,
    seed: int = 0,
) -> dict[str, Any]:
    """Run analytic and direct-Monte-Carlo high-dimensional reference cases."""
    dims = tuple(int(value) for value in dimensions)
    if not dims or any(value < 1 for value in dims):
        raise ValueError("dimensions must contain positive integers.")
    if int(num_samples) < 2:
        raise ValueError("num_samples must be at least two.")
    if int(repeats) < 1:
        raise ValueError("repeats must be positive.")
    root = jr.key(int(seed))
    records: list[HighDimensionalBenchmarkRecord] = []
    for dimension_index, dimension in enumerate(dims):
        dimension_key = jr.fold_in(root, dimension_index)
        records.append(
            _direct_quadratic_heat_record(
                jr.fold_in(dimension_key, 0),
                dimension,
                num_samples=int(num_samples),
                repeats=int(repeats),
                seed=int(seed),
            )
        )
        for problem_index, problem in enumerate(
            (
                "linear-hjb",
                "ornstein-uhlenbeck-score",
                "quartic-laplacian",
            ),
            start=1,
        ):
            records.append(
                _analytic_record(
                    problem,
                    jr.fold_in(dimension_key, problem_index),
                    dimension,
                    repeats=int(repeats),
                    seed=int(seed),
                )
            )
    return {
        "schema_version": 1,
        "seed": int(seed),
        "dimensions": list(dims),
        "configuration": {
            "num_samples": int(num_samples),
            "repeats": int(repeats),
        },
        "environment": _benchmark_environment(),
        "records": [asdict(record) | {"passed": record.passed} for record in records],
        "passed": all(record.passed for record in records),
    }


def _parse_dimensions(value: str, /) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=("reference", "methods"), default="reference")
    parser.add_argument("--dimensions", default="10,100")
    parser.add_argument("--num-samples", type=int, default=4096)
    parser.add_argument("--num-probes", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-training", action="store_true")
    parser.add_argument("--deep-picard-paths", type=int, default=32)
    parser.add_argument("--deep-picard-queries", type=int, default=16)
    parser.add_argument("--deep-picard-iterations", type=int, default=120)
    parser.add_argument("--deep-bsde-paths", type=int, default=32)
    parser.add_argument("--deep-bsde-time-steps", type=int, default=4)
    parser.add_argument("--deep-bsde-iterations", type=int, default=120)
    parser.add_argument("--deep-splitting-paths", type=int, default=32)
    parser.add_argument("--deep-splitting-time-steps", type=int, default=4)
    parser.add_argument("--deep-splitting-iterations", type=int, default=80)
    parser.add_argument("--score-samples", type=int, default=128)
    args = parser.parse_args()
    dimensions = _parse_dimensions(args.dimensions)
    if args.suite == "reference":
        result = run_high_dimensional_reference_benchmarks(
            dimensions,
            num_samples=args.num_samples,
            repeats=args.repeats,
            seed=args.seed,
        )
    else:
        result = run_high_dimensional_method_benchmarks(
            dimensions,
            num_samples=args.num_samples,
            num_probes=args.num_probes,
            repeats=args.repeats,
            seed=args.seed,
            include_training=args.include_training,
            deep_picard_paths=args.deep_picard_paths,
            deep_picard_queries=args.deep_picard_queries,
            deep_picard_iterations=args.deep_picard_iterations,
            deep_bsde_paths=args.deep_bsde_paths,
            deep_bsde_time_steps=args.deep_bsde_time_steps,
            deep_bsde_iterations=args.deep_bsde_iterations,
            deep_splitting_paths=args.deep_splitting_paths,
            deep_splitting_time_steps=args.deep_splitting_time_steps,
            deep_splitting_iterations=args.deep_splitting_iterations,
            score_samples=args.score_samples,
        )
    print(json.dumps(result, sort_keys=True))

if __name__ == "__main__":
    main()
