#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np

import phydrax as phx


jax.config.update("jax_enable_x64", True)


_ROBERTSON_REFERENCE = jnp.asarray(
    (
        0.99960068268829472,
        3.6450478878442331e-05,
        0.00036286683282682475,
    )
)


@dataclass(frozen=True)
class _Case:
    name: str
    prepared: phx.solver.PreparedDAESolve
    parameter: jax.Array
    args_from_parameter: Callable[[jax.Array], Any]
    observable: Callable[[phx.solver.DifferentialAlgebraicSolution], jax.Array]
    trajectory_error: Callable[
        [phx.solver.DifferentialAlgebraicSolution, jax.Array], jax.Array
    ]
    trajectory_reference: str
    trajectory_tolerance: float
    constraint_residual: (
        Callable[[phx.solver.DifferentialAlgebraicSolution, jax.Array], jax.Array] | None
    )
    constraint_reference: str | None
    constraint_tolerance: float
    reference_gradient: Callable[[jax.Array], jax.Array] | None
    gradient_reference: str
    finite_difference_step: float
    gradient_relative_tolerance: float
    gradient_absolute_tolerance: float
    model_setup_ms: float
    solver_preparation_ms: float
    metadata: dict[str, Any]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark native differentiable fixed-grid Phydrax DAE solves."
    )
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--spatial-points", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def _block(tree: Any, /) -> None:
    for leaf in jax.tree.leaves(tree):
        if isinstance(leaf, jax.Array):
            leaf.block_until_ready()


def _logical_array_bytes(tree: Any, /) -> int:
    return sum(
        int(leaf.nbytes)
        for leaf in jax.tree.leaves(tree)
        if isinstance(leaf, (jax.Array, np.ndarray))
    )


def _elapsed_ms(started: float, /) -> float:
    return 1e3 * (time.perf_counter() - started)


def _finite_float(value: Any, /) -> float | None:
    number = float(np.asarray(value))
    return number if math.isfinite(number) else None


def _summary(samples: list[float], /) -> dict[str, float]:
    values = np.asarray(samples, dtype=float)
    return {
        "mean_ms": float(np.mean(values)),
        "standard_deviation_ms": float(np.std(values)),
        "median_ms": float(np.median(values)),
        "minimum_ms": float(np.min(values)),
        "maximum_ms": float(np.max(values)),
    }


def _policy(*, residual_tolerance: float = 1e-10) -> phx.solver.DAESolvePolicy:
    stage_termination = phx.nonlinear.NonlinearTermination(
        absolute_residual=residual_tolerance,
        relative_residual=0.0,
        absolute_step=0.0,
        relative_step=0.0,
        maximum_steps=20,
    )
    initialization_termination = phx.nonlinear.NonlinearTermination(
        absolute_residual=residual_tolerance,
        relative_residual=0.0,
        absolute_step=0.0,
        relative_step=0.0,
        maximum_steps=32,
    )
    return phx.solver.DAESolvePolicy(
        integration_method="bdf2",
        nonlinear_method=phx.nonlinear.NewtonKrylov(),
        nonlinear_termination=stage_termination,
        initialization_method=phx.nonlinear.NewtonTrustRegion(),
        initialization_termination=initialization_termination,
        max_step_ratio=2.0,
        failure="status",
    )


def _prepare(
    problem: phx.solver.DifferentialAlgebraicProblem,
    grid: phx.dynamics.TimeGrid,
    policy: phx.solver.DAESolvePolicy,
    /,
) -> tuple[phx.solver.PreparedDAESolve, float]:
    started = time.perf_counter()
    prepared = phx.solver.prepare_dae(problem, grid, policy=policy)
    _block(prepared)
    return prepared, _elapsed_ms(started)


def _lower_compile(
    function: Any,
    argument: jax.Array,
    /,
) -> tuple[Any, float, float]:
    started = time.perf_counter()
    lowered = function.lower(argument)
    lowering_ms = _elapsed_ms(started)
    started = time.perf_counter()
    compiled = lowered.compile()
    compilation_ms = _elapsed_ms(started)
    return compiled, lowering_ms, compilation_ms


def _execute(function: Any, argument: jax.Array, /) -> tuple[Any, float]:
    started = time.perf_counter()
    result = function(argument)
    _block(result)
    return result, _elapsed_ms(started)


def _status_histogram(status: jax.Array, /) -> dict[str, int]:
    values, counts = np.unique(np.asarray(status), return_counts=True)
    return {
        phx.solver.DAEStatus(int(value)).name.lower(): int(count)
        for value, count in zip(values, counts, strict=True)
    }


def _independent_residual_norm(
    case: _Case,
    solution: phx.solver.DifferentialAlgebraicSolution,
    /,
) -> jax.Array:
    system = case.prepared.problem.system
    runtime_args = case.args_from_parameter(case.parameter)
    residuals = jax.vmap(
        lambda time, state, state_rate: system.scaled_residual(
            time,
            state,
            state_rate,
            runtime_args,
        )
    )(solution.times, solution.states, solution.state_rates)
    flattened = residuals.reshape((residuals.shape[0], -1))
    return jnp.max(jnp.sqrt(jnp.mean(jnp.square(jnp.abs(flattened)), axis=1)))


def _benchmark_case(case: _Case, repeats: int, /) -> dict[str, Any]:
    solve = jax.jit(
        lambda value: phx.solver.solve_dae(
            case.prepared,
            args=case.args_from_parameter(value),
        )
    )
    compiled_solve, forward_lowering_ms, forward_compilation_ms = _lower_compile(
        solve,
        case.parameter,
    )
    solution, forward_first_ms = _execute(compiled_solve, case.parameter)
    solution, forward_warmup_ms = _execute(compiled_solve, case.parameter)
    forward_samples: list[float] = []
    for _ in range(repeats):
        solution, elapsed = _execute(compiled_solve, case.parameter)
        forward_samples.append(elapsed)

    value_and_grad = jax.jit(
        jax.value_and_grad(
            lambda value: case.observable(
                phx.solver.solve_dae(
                    case.prepared,
                    args=case.args_from_parameter(value),
                )
            )
        )
    )
    (
        compiled_value_and_grad,
        gradient_lowering_ms,
        gradient_compilation_ms,
    ) = _lower_compile(value_and_grad, case.parameter)
    (objective, gradient), gradient_first_ms = _execute(
        compiled_value_and_grad,
        case.parameter,
    )
    (objective, gradient), gradient_warmup_ms = _execute(
        compiled_value_and_grad,
        case.parameter,
    )
    gradient_samples: list[float] = []
    for _ in range(repeats):
        (objective, gradient), elapsed = _execute(
            compiled_value_and_grad,
            case.parameter,
        )
        gradient_samples.append(elapsed)

    finite_difference_step: float | None = None
    if case.reference_gradient is None:
        epsilon = jnp.asarray(case.finite_difference_step, dtype=case.parameter.dtype)
        upper_solution = compiled_solve(case.parameter + epsilon)
        lower_solution = compiled_solve(case.parameter - epsilon)
        upper = case.observable(upper_solution)
        lower = case.observable(lower_solution)
        gradient_reference = (upper - lower) / (2.0 * epsilon)
        finite_difference_step = case.finite_difference_step
    else:
        gradient_reference = case.reference_gradient(case.parameter)
    _block(gradient_reference)

    trajectory_error = case.trajectory_error(solution, case.parameter)
    constraint_residual = (
        jnp.asarray(0.0)
        if case.constraint_residual is None
        else case.constraint_residual(solution, case.parameter)
    )
    independent_residual = _independent_residual_norm(case, solution)
    reported_residual = jnp.max(solution.residual_norm)
    reported_constraint = jnp.max(solution.constraint_norm)
    residual_threshold = jnp.max(solution.residual_threshold)
    gradient_absolute_error = jnp.abs(gradient - gradient_reference)
    gradient_relative_error = gradient_absolute_error / jnp.maximum(
        jnp.abs(gradient_reference),
        jnp.asarray(1e-30, dtype=gradient.dtype),
    )
    _block(
        (
            trajectory_error,
            constraint_residual,
            independent_residual,
            reported_residual,
            reported_constraint,
            residual_threshold,
            gradient_absolute_error,
            gradient_relative_error,
        )
    )

    finite_outputs = bool(
        jnp.all(jnp.isfinite(solution.states))
        & jnp.all(jnp.isfinite(solution.state_rates))
        & jnp.isfinite(objective)
        & jnp.isfinite(gradient)
        & jnp.isfinite(gradient_reference)
        & jnp.isfinite(trajectory_error)
        & jnp.isfinite(constraint_residual)
        & jnp.isfinite(independent_residual)
    )
    gradient_within_tolerance = bool(
        gradient_absolute_error
        <= case.gradient_absolute_tolerance
        + case.gradient_relative_tolerance * jnp.abs(gradient_reference)
    )
    residual_within_tolerance = bool(independent_residual <= residual_threshold)
    trajectory_within_tolerance = bool(trajectory_error <= case.trajectory_tolerance)
    constraint_within_tolerance = bool(constraint_residual <= case.constraint_tolerance)
    passed = (
        bool(solution.successful)
        and finite_outputs
        and gradient_within_tolerance
        and residual_within_tolerance
        and trajectory_within_tolerance
        and constraint_within_tolerance
    )

    initialization_iterations = int(solution.initialization.nonlinear_iterations)
    initialization_linear_iterations = int(solution.initialization.linear_iterations)
    return {
        "name": case.name,
        "metadata": case.metadata,
        "problem_id": solution.problem_id,
        "system_id": solution.system_id,
        "time_id": solution.time_id,
        "plan_id": solution.plan_id,
        "prepared_id": solution.prepared_id,
        "grid_points": int(solution.times.size),
        "state_shape": list(solution.state_shape),
        "successful": bool(solution.successful),
        "finite_outputs": finite_outputs,
        "status_histogram": _status_histogram(solution.status),
        "initialization_status": phx.solver.DAEInitializationStatus(
            int(solution.initialization.status)
        ).name.lower(),
        "certificates": {
            "trajectory": {
                "reference": case.trajectory_reference,
                "relative_error": _finite_float(trajectory_error),
                "tolerance": case.trajectory_tolerance,
                "passed": trajectory_within_tolerance,
            },
            "constraint": {
                "applicable": case.constraint_residual is not None,
                "reference": case.constraint_reference,
                "maximum_absolute_residual": _finite_float(constraint_residual),
                "tolerance": case.constraint_tolerance,
                "passed": constraint_within_tolerance,
            },
            "residual": {
                "independent_maximum_scaled_rms": _finite_float(independent_residual),
                "reported_maximum_scaled_rms": _finite_float(reported_residual),
                "reported_maximum_constraint_rms": _finite_float(reported_constraint),
                "threshold": _finite_float(residual_threshold),
                "passed": residual_within_tolerance,
            },
            "gradient": {
                "objective": _finite_float(objective),
                "value": _finite_float(gradient),
                "reference_value": _finite_float(gradient_reference),
                "reference": case.gradient_reference,
                "symmetric_finite_difference_step": finite_difference_step,
                "absolute_error": _finite_float(gradient_absolute_error),
                "relative_error": _finite_float(gradient_relative_error),
                "absolute_tolerance": case.gradient_absolute_tolerance,
                "relative_tolerance": case.gradient_relative_tolerance,
                "passed": gradient_within_tolerance,
            },
        },
        "solver_evidence": {
            "integration_method": solution.integration_method,
            "nonlinear_method_id": solution.nonlinear_method_id,
            "differentiation_mode": solution.differentiation_mode,
            "grid_origin": solution.grid_origin,
            "approximation_id": solution.approximation_id,
            "stage_linear_plan_id": solution.stage_linear_plan_id,
            "initialization_linear_plan_id": solution.initialization_linear_plan_id,
            "stage_linear_template_reused": (
                solution.stage_linear_plan_id == case.prepared.stage_solve.linear_plan_id
            ),
            "initialization_nonlinear_iterations": initialization_iterations,
            "initialization_linear_iterations": initialization_linear_iterations,
            "maximum_stage_nonlinear_iterations": int(
                jnp.max(solution.nonlinear_iterations)
            ),
            "total_stage_nonlinear_iterations": int(
                jnp.sum(solution.nonlinear_iterations)
            ),
            "total_residual_evaluations": int(jnp.sum(solution.residual_evaluations)),
            "total_jacobian_preparations": int(jnp.sum(solution.jacobian_preparations)),
            "total_linear_solves": int(jnp.sum(solution.linear_solves)),
            "total_linear_iterations": int(jnp.sum(solution.linear_iterations)),
            "total_globalization_rejections": int(
                jnp.sum(solution.globalization_rejections)
            ),
            "total_setup_refreshes": int(jnp.sum(solution.setup_refreshes)),
            "total_numeric_refreshes": int(jnp.sum(solution.numeric_refreshes)),
        },
        "timing": {
            "unit": "milliseconds",
            "model_setup_ms": case.model_setup_ms,
            "solver_preparation_ms": case.solver_preparation_ms,
            "forward": {
                "lowering_ms": forward_lowering_ms,
                "compilation_ms": forward_compilation_ms,
                "first_execution_ms": forward_first_ms,
                "warmup_execution_ms": forward_warmup_ms,
                "steady_samples_ms": forward_samples,
                "steady_summary": _summary(forward_samples),
            },
            "value_and_grad": {
                "lowering_ms": gradient_lowering_ms,
                "compilation_ms": gradient_compilation_ms,
                "first_execution_ms": gradient_first_ms,
                "warmup_execution_ms": gradient_warmup_ms,
                "steady_samples_ms": gradient_samples,
                "steady_summary": _summary(gradient_samples),
            },
        },
        "storage": {
            "scope": (
                "logical array payloads; excludes allocator pools and compiled "
                "executables"
            ),
            "prepared_array_bytes": _logical_array_bytes(case.prepared),
            "solution_array_bytes": _logical_array_bytes(solution),
            "state_trajectory_bytes": int(solution.states.nbytes),
            "state_rate_trajectory_bytes": int(solution.state_rates.nbytes),
        },
        "passed": passed,
    }


def _scalar_linear(steps: int) -> _Case:
    started = time.perf_counter()
    system = phx.dynamics.DifferentialAlgebraicSystem(
        lambda time, state, state_rate, rate: state_rate + rate * state,
        state_shape=(1,),
        structure=phx.dynamics.DAEStructure(("differential",)),
        system_id="benchmark:dae:scalar-linear",
    )
    problem = phx.solver.DifferentialAlgebraicProblem(
        system,
        jnp.ones(1),
        args=jnp.asarray(1.0),
        problem_id="benchmark:dae:scalar-linear",
    )
    grid = phx.dynamics.TimeGrid(
        jnp.linspace(0.0, 1.0, steps + 1),
        time_id=f"benchmark:dae:scalar-linear:{steps}",
    )
    _block((problem, grid))
    model_setup_ms = _elapsed_ms(started)
    prepared, preparation_ms = _prepare(problem, grid, _policy())
    return _Case(
        name="scalar-linear",
        prepared=prepared,
        parameter=jnp.asarray(1.0),
        args_from_parameter=lambda value: value,
        observable=lambda solution: solution.states[-1, 0],
        trajectory_error=lambda solution, rate: (
            jnp.linalg.norm(solution.states[:, 0] - jnp.exp(-rate * solution.times))
            / jnp.linalg.norm(jnp.exp(-rate * solution.times))
        ),
        trajectory_reference="closed-form exponential trajectory",
        trajectory_tolerance=max(5.0 / steps**2, 2e-4),
        constraint_residual=None,
        constraint_reference=None,
        constraint_tolerance=0.0,
        reference_gradient=lambda rate: -jnp.exp(-rate),
        gradient_reference="closed-form terminal derivative",
        finite_difference_step=1e-4,
        gradient_relative_tolerance=max(10.0 / steps**2, 5e-3),
        gradient_absolute_tolerance=1e-8,
        model_setup_ms=model_setup_ms,
        solver_preparation_ms=preparation_ms,
        metadata={"class": "linear ODE in implicit residual form"},
    )


def _vector_linear(steps: int) -> _Case:
    started = time.perf_counter()
    mass = jnp.asarray((1.0, 2.0, 3.0, 4.0))
    stiffness = jnp.asarray((1.0, 4.0, 9.0, 16.0))
    initial = jnp.asarray((1.0, -0.5, 0.25, 0.75))
    rates = stiffness / mass
    system = phx.dynamics.DifferentialAlgebraicSystem.from_mass_matrix(
        jnp.diag(mass),
        lambda time, state, scale: -scale * stiffness * state,
        state_shape=(4,),
        structure=phx.dynamics.DAEStructure(("differential",) * 4),
        state_scale=jnp.abs(initial),
        system_id="benchmark:dae:vector-linear",
    )
    problem = phx.solver.DifferentialAlgebraicProblem(
        system,
        initial,
        args=jnp.asarray(0.7),
        problem_id="benchmark:dae:vector-linear",
    )
    grid = phx.dynamics.TimeGrid(
        jnp.linspace(0.0, 0.5, steps + 1),
        time_id=f"benchmark:dae:vector-linear:{steps}",
    )
    _block((problem, grid))
    model_setup_ms = _elapsed_ms(started)
    prepared, preparation_ms = _prepare(problem, grid, _policy())

    def exact(scale: jax.Array, times: jax.Array) -> jax.Array:
        return initial * jnp.exp(-scale * times[..., None] * rates)

    return _Case(
        name="vector-linear-mass",
        prepared=prepared,
        parameter=jnp.asarray(0.7),
        args_from_parameter=lambda value: value,
        observable=lambda solution: jnp.sum(solution.states[-1]),
        trajectory_error=lambda solution, scale: (
            jnp.linalg.norm(solution.states - exact(scale, solution.times))
            / jnp.linalg.norm(exact(scale, solution.times))
        ),
        trajectory_reference="closed-form diagonal mass-matrix trajectory",
        trajectory_tolerance=max(12.0 / steps**2, 5e-4),
        constraint_residual=None,
        constraint_reference=None,
        constraint_tolerance=0.0,
        reference_gradient=lambda scale: jnp.sum(
            -0.5 * rates * exact(scale, jnp.asarray(0.5))
        ),
        gradient_reference="closed-form terminal derivative",
        finite_difference_step=1e-4,
        gradient_relative_tolerance=max(20.0 / steps**2, 8e-3),
        gradient_absolute_tolerance=1e-8,
        model_setup_ms=model_setup_ms,
        solver_preparation_ms=preparation_ms,
        metadata={"class": "diagonal nonsingular mass matrix", "state_size": 4},
    )


def _robertson(steps: int) -> _Case:
    started = time.perf_counter()

    def residual(time, state, state_rate, scale):
        first, second, third = state
        forward = 0.04 * first - 1e4 * second * third
        return jnp.asarray(
            (
                state_rate[0] + scale * forward,
                state_rate[1] - scale * (forward - 3e7 * second**2),
                first + second + third - 1.0,
            )
        )

    system = phx.dynamics.DifferentialAlgebraicSystem(
        residual,
        state_shape=(3,),
        structure=phx.dynamics.DAEStructure(
            ("differential", "differential", "algebraic")
        ),
        state_scale=jnp.asarray((1.0, 1e-4, 1e-3)),
        state_rate_scale=jnp.asarray((0.04, 0.04, 1.0)),
        residual_scale=jnp.asarray((0.04, 0.04, 1.0)),
        system_id="benchmark:dae:robertson",
    )
    problem = phx.solver.DifferentialAlgebraicProblem(
        system,
        jnp.asarray((1.0, 0.0, 0.0)),
        args=jnp.asarray(1.0),
        problem_id="benchmark:dae:robertson",
    )
    grid = phx.dynamics.TimeGrid(
        jnp.linspace(0.0, 1e-2, steps + 1),
        time_id=f"benchmark:dae:robertson:{steps}",
    )
    _block((problem, grid))
    model_setup_ms = _elapsed_ms(started)
    prepared, preparation_ms = _prepare(
        problem,
        grid,
        _policy(residual_tolerance=1e-9),
    )
    return _Case(
        name="robertson-semi-explicit",
        prepared=prepared,
        parameter=jnp.asarray(1.0),
        args_from_parameter=lambda value: value,
        observable=lambda solution: solution.states[-1, 2],
        trajectory_error=lambda solution, scale: (
            jnp.linalg.norm(solution.states[-1] - _ROBERTSON_REFERENCE)
            / jnp.linalg.norm(_ROBERTSON_REFERENCE)
        ),
        trajectory_reference="high-accuracy Robertson terminal state at t=0.01",
        trajectory_tolerance=max(2e-3 / steps**2, 5e-7),
        constraint_residual=lambda solution, scale: jnp.max(
            jnp.abs(jnp.sum(solution.states, axis=1) - 1.0)
        ),
        constraint_reference="species conservation y1 + y2 + y3 = 1",
        constraint_tolerance=2e-9,
        reference_gradient=None,
        gradient_reference="symmetric finite difference of the discrete BDF solve",
        finite_difference_step=1e-4,
        gradient_relative_tolerance=5e-4,
        gradient_absolute_tolerance=1e-8,
        model_setup_ms=model_setup_ms,
        solver_preparation_ms=preparation_ms,
        metadata={
            "class": "stiff semi-explicit chemical kinetics DAE",
            "reference_time": 0.01,
        },
    )


def _circuit(steps: int) -> _Case:
    started = time.perf_counter()
    inductance = 0.3
    resistance = 0.4
    load_resistance = 1.2

    def residual(time, state, state_rate, capacitance):
        voltage, current, branch_current = state
        return jnp.asarray(
            (
                capacitance * state_rate[0] + branch_current - 1.0,
                inductance * state_rate[1] - voltage + resistance * current,
                branch_current - current - voltage / load_resistance,
            )
        )

    system = phx.dynamics.DifferentialAlgebraicSystem(
        residual,
        state_shape=(3,),
        structure=phx.dynamics.DAEStructure(
            ("differential", "differential", "algebraic")
        ),
        system_id="benchmark:dae:rlc-circuit",
    )
    problem = phx.solver.DifferentialAlgebraicProblem(
        system,
        jnp.zeros(3),
        args=jnp.asarray(0.8),
        problem_id="benchmark:dae:rlc-circuit",
    )
    grid = phx.dynamics.TimeGrid(
        jnp.linspace(0.0, 1.0, steps + 1),
        time_id=f"benchmark:dae:circuit:{steps}",
    )
    _block((problem, grid))
    model_setup_ms = _elapsed_ms(started)
    prepared, preparation_ms = _prepare(problem, grid, _policy())

    def exact(capacitance: jax.Array, times: jax.Array) -> jax.Array:
        matrix = jnp.asarray(
            (
                (-1.0 / (capacitance * load_resistance), -1.0 / capacitance),
                (1.0 / inductance, -resistance / inductance),
            )
        )
        forcing = jnp.asarray((1.0 / capacitance, 0.0))

        def state_at(time):
            dynamic = jnp.linalg.solve(
                matrix,
                (jsp.linalg.expm(matrix * time) - jnp.eye(2)) @ forcing,
            )
            voltage, current = dynamic
            branch_current = current + voltage / load_resistance
            return jnp.asarray((voltage, current, branch_current))

        return jax.vmap(state_at)(jnp.atleast_1d(times))

    terminal_voltage_gradient = jax.grad(
        lambda capacitance: exact(capacitance, jnp.asarray((1.0,)))[0, 0]
    )
    return _Case(
        name="singular-mass-circuit",
        prepared=prepared,
        parameter=jnp.asarray(0.8),
        args_from_parameter=lambda value: value,
        observable=lambda solution: solution.states[-1, 0],
        trajectory_error=lambda solution, capacitance: (
            jnp.linalg.norm(solution.states - exact(capacitance, solution.times))
            / jnp.linalg.norm(exact(capacitance, solution.times))
        ),
        trajectory_reference="closed-form reduced linear circuit trajectory",
        trajectory_tolerance=max(15.0 / steps**2, 8e-4),
        constraint_residual=lambda solution, capacitance: jnp.max(
            jnp.abs(
                solution.states[:, 2]
                - solution.states[:, 1]
                - solution.states[:, 0] / load_resistance
            )
        ),
        constraint_reference="Kirchhoff branch-current constraint",
        constraint_tolerance=1e-10,
        reference_gradient=terminal_voltage_gradient,
        gradient_reference="closed-form matrix-exponential terminal derivative",
        finite_difference_step=1e-4,
        gradient_relative_tolerance=max(70.0 / steps**2, 1.2e-2),
        gradient_absolute_tolerance=1e-8,
        model_setup_ms=model_setup_ms,
        solver_preparation_ms=preparation_ms,
        metadata={"class": "linear circuit with singular mass matrix"},
    )


def _reaction_diffusion(steps: int, spatial_points: int) -> _Case:
    started = time.perf_counter()
    x = phx.equations.PDECoordinate(
        "x",
        "space",
        bounds=(0.0, 1.0),
        periodic=True,
    )
    time_coordinate = phx.equations.PDECoordinate(
        "t",
        "time",
        bounds=(0.0, 0.02),
    )
    fields = (
        phx.equations.PDEField("u", coordinates=("x", "t")),
        phx.equations.PDEField("equilibrium", coordinates=("x", "t")),
    )
    diffusivity = phx.equations.PDEParameter("diffusivity", value=0.05)
    u = phx.equations.PDEExpression.field("u")
    equilibrium = phx.equations.PDEExpression.field("equilibrium")
    problem_ir = phx.equations.PDEProblemIR(
        coordinates=(x, time_coordinate),
        fields=fields,
        parameters=(diffusivity,),
        equations=(
            phx.equations.PDEEquation(
                "diffusion",
                u.derivative("t"),
                phx.equations.PDEExpression.parameter("diffusivity") * u.laplacian("x"),
            ),
            phx.equations.PDEEquation("local-equilibrium", equilibrium, u**2),
        ),
    )
    axis = phx.domain.FourierAxisSpec(spatial_points).materialize(0.0, 1.0)
    spatial = phx.solver.TensorGridDiscretization((axis,))
    compiled = phx.equations.compile_semidiscrete_dae(
        problem_ir,
        spatial,
        equation_targets={"diffusion": "u", "local-equilibrium": "equilibrium"},
        state_scale=jnp.asarray((1.0, 1.0)),
        state_rate_scale=jnp.asarray((1.0, 1.0)),
        residual_scale=jnp.asarray((1.0, 1.0)),
    )
    initial_u = jnp.sin(2.0 * jnp.pi * axis.nodes)
    initial = compiled.layout.pack(
        {"u": initial_u, "equilibrium": jnp.zeros_like(initial_u)}
    )
    problem = phx.solver.DifferentialAlgebraicProblem(
        compiled.system,
        initial,
        args={"diffusivity": jnp.asarray(0.05)},
        problem_id="benchmark:dae:reaction-diffusion",
    )
    grid = phx.dynamics.TimeGrid(
        jnp.linspace(0.0, 0.02, steps + 1),
        time_id=f"benchmark:dae:reaction-diffusion:{steps}:{spatial_points}",
    )
    _block((problem, grid, compiled))
    model_setup_ms = _elapsed_ms(started)
    prepared, preparation_ms = _prepare(problem, grid, _policy())
    eigenvalue = jnp.vdot(initial_u, spatial.laplacian(initial_u)) / jnp.vdot(
        initial_u, initial_u
    )

    def exact(diffusivity: jax.Array, times: jax.Array) -> jax.Array:
        physical_u = (
            jnp.exp(diffusivity * eigenvalue * times[:, None]) * initial_u[None, :]
        )
        return jnp.stack((physical_u, physical_u**2), axis=-1)

    def amplitude(solution: phx.solver.DifferentialAlgebraicSolution) -> jax.Array:
        terminal = compiled.layout.field(solution.states[-1], "u")
        return jnp.vdot(initial_u, terminal) / jnp.vdot(initial_u, initial_u)

    def equilibrium_residual(
        solution: phx.solver.DifferentialAlgebraicSolution,
        parameter: jax.Array,
    ) -> jax.Array:
        del parameter
        physical_u = jax.vmap(lambda state: compiled.layout.field(state, "u"))(
            solution.states
        )
        physical_equilibrium = jax.vmap(
            lambda state: compiled.layout.field(state, "equilibrium")
        )(solution.states)
        return jnp.max(jnp.abs(physical_equilibrium - physical_u**2))

    return _Case(
        name="constrained-reaction-diffusion",
        prepared=prepared,
        parameter=jnp.asarray(0.05),
        args_from_parameter=lambda value: {"diffusivity": value},
        observable=amplitude,
        trajectory_error=lambda solution, parameter: (
            jnp.linalg.norm(solution.states - exact(parameter, solution.times))
            / jnp.linalg.norm(exact(parameter, solution.times))
        ),
        trajectory_reference="exact semidiscrete Fourier-mode trajectory",
        trajectory_tolerance=max(15.0 / steps**2, 5e-4),
        constraint_residual=equilibrium_residual,
        constraint_reference="pointwise equilibrium = u**2",
        constraint_tolerance=1e-9,
        reference_gradient=lambda parameter: (
            0.02 * eigenvalue * jnp.exp(0.02 * parameter * eigenvalue)
        ),
        gradient_reference="exact semidiscrete Fourier-mode terminal derivative",
        finite_difference_step=1e-4,
        gradient_relative_tolerance=max(20.0 / steps**2, 8e-3),
        gradient_absolute_tolerance=1e-8,
        model_setup_ms=model_setup_ms,
        solver_preparation_ms=preparation_ms,
        metadata={
            "class": "semidiscrete PDE DAE",
            "spatial_points": spatial_points,
            "structural_assumption": compiled.structural_report.index_assumption,
            "regularity_verified": compiled.structural_report.regularity_verified,
        },
    )


def _environment() -> dict[str, Any]:
    device = jax.devices()[0]
    return {
        "python_version": platform.python_version(),
        "jax_version": jax.__version__,
        "backend": jax.default_backend(),
        "device_kind": device.device_kind,
        "machine": platform.machine(),
        "system": platform.system(),
        "system_release": platform.release(),
        "x64_enabled": bool(jax.config.read("jax_enable_x64")),
    }


def main() -> None:
    arguments = _parser().parse_args()
    requested_steps = int(arguments.steps)
    requested_spatial_points = int(arguments.spatial_points)
    requested_repeats = int(arguments.repeats)
    steps = min(requested_steps, 8) if arguments.smoke else requested_steps
    spatial_points = (
        min(requested_spatial_points, 8) if arguments.smoke else requested_spatial_points
    )
    repeats = 1 if arguments.smoke else requested_repeats
    if steps < 4 or spatial_points < 4 or repeats < 1:
        raise ValueError("steps, spatial-points, and repeats are below benchmark minima.")

    cases = (
        _scalar_linear(steps),
        _vector_linear(steps),
        _robertson(steps),
        _circuit(steps),
        _reaction_diffusion(steps, spatial_points),
    )
    records = tuple(_benchmark_case(case, repeats) for case in cases)
    report = {
        "schema_version": "phydrax-dae-benchmark-v2",
        "protocol": {
            "timing": (
                "model setup, solver preparation, lowering, compilation, first "
                "execution, warmup, and synchronized steady executions are separate"
            ),
            "synchronization": "every measured JAX execution blocks all array leaves",
            "gradient": (
                "discrete implicit value-and-grad compared with an independent "
                "closed-form or symmetric finite-difference reference"
            ),
            "memory": "logical result payload only; no cumulative allocator counters",
        },
        "configuration": {
            "requested": {
                "steps": requested_steps,
                "spatial_points": requested_spatial_points,
                "repeats": requested_repeats,
            },
            "effective": {
                "steps": steps,
                "spatial_points": spatial_points,
                "repeats": repeats,
                "smoke": bool(arguments.smoke),
            },
        },
        "environment": _environment(),
        "cases": records,
        "passed": all(record["passed"] for record in records),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
