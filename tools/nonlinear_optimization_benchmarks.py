#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


def _block(tree: Any) -> Any:
    return jax.tree.map(jax.block_until_ready, tree)


def _timed(operation: Callable[[], Any], /) -> tuple[Any, float]:
    started = time.perf_counter()
    result = operation()
    _block(result)
    return result, time.perf_counter() - started

def _compiled(operation: Callable[[], Any], /) -> tuple[Any, float]:
    started = time.perf_counter()
    executable = operation()
    return executable, time.perf_counter() - started




def _nonlinear_diagonal_objective(parameters: jax.Array, weights: jax.Array) -> jax.Array:
    return jnp.sum(0.5 * weights * parameters**2 + 0.025 * parameters**4)


def _matrix_free_newton(size: int) -> dict[str, Any]:
    weights = jnp.linspace(0.5, 2.0, size)
    initial = jnp.full((size,), 2.0)
    problem = phx.optim.MinimizationProblem(
        lambda parameters, current_weights: _nonlinear_diagonal_objective(
            parameters,
            current_weights,
        ),
        problem_id=f"nonlinear-diagonal-{size}",
    )
    termination = phx.optim.OptimizationTermination(
        absolute_optimality=1e-9,
        relative_optimality=0.0,
        maximum_steps=30,
    )

    def solve(current_weights):
        return phx.optim.minimize(
            problem,
            initial,
            method=phx.optim.NewtonKrylov(),
            termination=termination,
            args=current_weights,
        )

    result, eager_seconds = _timed(lambda: solve(weights))
    executable, compile_seconds = _compiled(
        lambda: jax.jit(solve).lower(weights).compile()
    )
    compiled_result, first_execution_seconds = _timed(
        lambda: executable(weights)
    )
    _, steady_execution_seconds = _timed(lambda: executable(weights))
    return {
        "case": "matrix-free-newton-krylov",
        "size": size,
        "eager_wall_seconds": eager_seconds,
        "jit_compile_seconds": compile_seconds,
        "jit_first_execution_seconds": first_execution_seconds,
        "jit_steady_execution_seconds": steady_execution_seconds,
        "initial_objective": float(problem.value(initial, weights)[0]),
        "final_objective": float(result.objective),
        "compiled_final_objective": float(compiled_result.objective),
        "status": int(result.status),
        "iterations": int(result.diagnostics.iterations),
        "objective_evaluations": int(result.diagnostics.objective_evaluations),
        "hvp_evaluations": int(result.diagnostics.hvp_evaluations),
        "linear_iterations": int(result.diagnostics.linear_iterations),
        "setup_refreshes": int(result.diagnostics.setup_refreshes),
        "numeric_refreshes": int(result.diagnostics.numeric_refreshes),
        "parameter_bytes": int(initial.nbytes),
        "dense_hessian_bytes": int(size * size * initial.dtype.itemsize),
    }


def _newton_iteration_lifecycle(size: int) -> dict[str, Any]:
    weights = jnp.linspace(0.5, 2.0, size)
    parameters = jnp.full((size,), 2.0)
    value_function = lambda candidate: _nonlinear_diagonal_objective(
        candidate,
        weights,
    )
    method = phx.optim.NewtonKrylov()
    termination = phx.optim.OptimizationTermination(
        absolute_optimality=1e-9,
        relative_optimality=0.0,
        maximum_steps=30,
    )
    state, setup_seconds = _timed(
        lambda: method.prepare_state(value_function, parameters)
    )
    def step(current_parameters, current_state):
        return method.step(
            value_function,
            current_parameters,
            current_state,
            termination=termination,
        )

    executable, compile_seconds = _compiled(
        lambda: jax.jit(step).lower(parameters, state).compile()
    )
    first, first_step_seconds = _timed(lambda: executable(parameters, state))
    next_parameters, next_state, _ = first
    second, steady_step_seconds = _timed(
        lambda: executable(next_parameters, next_state)
    )
    _, steady_state, _ = second
    return {
        "case": "newton-prepared-refresh-lifecycle",
        "size": size,
        "setup_wall_seconds": setup_seconds,
        "step_compile_seconds": compile_seconds,
        "first_step_wall_seconds": first_step_seconds,
        "steady_step_wall_seconds": steady_step_seconds,
        "setup_refreshes_after_first_step": int(next_state.setup_refreshes),
        "numeric_refreshes_after_first_step": int(next_state.numeric_refreshes),
        "setup_refreshes_after_steady_step": int(steady_state.setup_refreshes),
        "numeric_refreshes_after_steady_step": int(
            steady_state.numeric_refreshes
        ),
    }


def _dense_newton_reference(size: int) -> dict[str, Any]:
    weights = jnp.linspace(0.5, 2.0, size)
    initial = jnp.full((size,), 2.0)

    def objective(parameters):
        return _nonlinear_diagonal_objective(parameters, weights)

    gradient = jax.grad(objective)
    hessian = jax.hessian(objective)

    def solve():
        parameters = initial
        for _ in range(12):
            parameters = parameters - jnp.linalg.solve(
                hessian(parameters),
                gradient(parameters),
            )
        return parameters

    parameters, wall_seconds = _timed(solve)
    return {
        "case": "dense-newton-reference",
        "size": size,
        "wall_seconds": wall_seconds,
        "initial_objective": float(objective(initial)),
        "final_objective": float(objective(parameters)),
        "iterations": 12,
        "parameter_bytes": int(initial.nbytes),
        "dense_hessian_bytes": int(size * size * initial.dtype.itemsize),
    }


def _line_search_accounting() -> dict[str, Any]:
    def rosenbrock(parameters, _):
        x, y = parameters
        return (1.0 - x) ** 2 + 100.0 * (y - x**2) ** 2

    initial = jnp.asarray([-1.2, 1.0])
    result, wall_seconds = _timed(
        lambda: phx.optim.minimize(
            rosenbrock,
            initial,
            method=phx.optim.NewtonKrylov(),
            termination=phx.optim.OptimizationTermination(
                absolute_optimality=1e-8,
                relative_optimality=0.0,
                maximum_steps=100,
            ),
        )
    )
    return {
        "case": "frozen-objective-line-search",
        "wall_seconds": wall_seconds,
        "initial_objective": float(rosenbrock(initial, None)),
        "final_objective": float(result.objective),
        "status": int(result.status),
        "accepted_steps": int(result.diagnostics.accepted_steps),
        "objective_evaluations": int(result.diagnostics.objective_evaluations),
        "globalization_evaluations": int(result.diagnostics.globalization_evaluations),
        "direction_fallbacks": int(result.diagnostics.direction_fallbacks),
    }


def _common_random_numbers(sample_size: int, repeats: int) -> dict[str, Any]:
    def sampler(key, size):
        return jr.normal(key, (size,))

    fixed = phx.optim.MonteCarloSampling(sampler, sample_size, refresh="fixed")
    refreshing = phx.optim.MonteCarloSampling(
        sampler,
        sample_size,
        refresh="per_iteration",
    )
    risk = phx.optim.ExpectationRisk()
    key = jr.key(42)

    def estimates(policy):
        values = []
        for iteration in range(repeats):
            batch = policy.sample(key, iteration)
            values.append(risk.evaluate(batch.scenarios**2, batch.weights))
        return jnp.asarray(values)

    fixed_values, fixed_seconds = _timed(lambda: estimates(fixed))
    refreshing_values, refreshing_seconds = _timed(lambda: estimates(refreshing))
    return {
        "case": "common-random-numbers",
        "sample_size": sample_size,
        "repeats": repeats,
        "fixed_wall_seconds": fixed_seconds,
        "refreshing_wall_seconds": refreshing_seconds,
        "fixed_estimator_variance": float(jnp.var(fixed_values)),
        "refreshing_estimator_variance": float(jnp.var(refreshing_values)),
        "fixed_estimate": float(fixed_values[0]),
        "refreshing_estimate_mean": float(jnp.mean(refreshing_values)),
    }


def _poorly_scaled_bounds() -> dict[str, Any]:
    bounds = phx.optim.Bounds(-1.0, 1.0)

    def objective(parameters, _):
        return 1e6 * (parameters[0] - 0.2) ** 2 + 1e-3 * (parameters[1] + 0.4) ** 2

    result, wall_seconds = _timed(
        lambda: phx.optim.minimize(
            phx.optim.MinimizationProblem(
                objective,
                bounds=bounds,
                problem_id="poorly-scaled-box",
            ),
            jnp.asarray([0.9, 0.9]),
            method=phx.optim.ActiveSetNewton(),
            termination=phx.optim.OptimizationTermination(
                absolute_optimality=1e-8,
                relative_optimality=0.0,
                maximum_steps=30,
            ),
        )
    )
    return {
        "case": "poorly-scaled-bound-constrained",
        "wall_seconds": wall_seconds,
        "status": int(result.status),
        "final_objective": float(result.objective),
        "optimality_norm": float(result.diagnostics.final_optimality_norm),
        "primal_feasibility": float(result.diagnostics.primal_feasibility),
        "active_constraints": int(result.diagnostics.active_constraints),
        "direction_fallbacks": int(result.diagnostics.direction_fallbacks),
    }


def _matrix_free_constraints(size: int) -> dict[str, Any]:
    initial = jnp.zeros((size,))
    problem = phx.optim.MinimizationProblem(
        lambda parameters, _: 0.5 * jnp.sum((parameters - 2.0) ** 2),
        bounds=phx.optim.Bounds(-jnp.inf, 1.0),
        problem_id=f"matrix-free-bounds-{size}",
    )
    result, wall_seconds = _timed(
        lambda: phx.optim.minimize(
            problem,
            initial,
            method=phx.optim.PrimalDualNewtonKrylov(),
            termination=phx.optim.OptimizationTermination(
                absolute_optimality=1e-7,
                relative_optimality=0.0,
                maximum_steps=30,
            ),
        )
    )
    return {
        "case": "matrix-free-primal-dual-kkt",
        "size": size,
        "wall_seconds": wall_seconds,
        "status": int(result.status),
        "final_objective": float(result.objective),
        "primal_feasibility": float(result.diagnostics.primal_feasibility),
        "dual_feasibility": float(result.diagnostics.dual_feasibility),
        "complementarity": float(result.diagnostics.complementarity),
        "hvp_evaluations": int(result.diagnostics.hvp_evaluations),
        "jvp_evaluations": int(result.diagnostics.jvp_evaluations),
        "vjp_evaluations": int(result.diagnostics.vjp_evaluations),
        "setup_refreshes": int(result.diagnostics.setup_refreshes),
        "numeric_refreshes": int(result.diagnostics.numeric_refreshes),
        "parameter_bytes": int(initial.nbytes),
        "estimated_dense_kkt_bytes": int(
            (2 * size) * (2 * size) * initial.dtype.itemsize
        ),
    }


def _reduced_state_design(size: int) -> dict[str, Any]:
    initial_state = jnp.zeros((size,))
    initial_design = jnp.zeros((size,))
    problem = phx.optim.StateDesignProblem(
        lambda state, design, _: state - design,
        lambda state, design, _: (
            jnp.sum((state - 2.0) ** 2) + 0.1 * jnp.sum(design**2)
        ),
        problem_id=f"reduced-state-design-{size}",
    )
    result, wall_seconds = _timed(
        lambda: phx.optim.solve_state_design(
            problem,
            initial_state,
            initial_design,
            method=phx.optim.ReducedNewtonKrylov(),
            termination=phx.optim.OptimizationTermination(
                absolute_optimality=1e-6,
                relative_optimality=0.0,
                maximum_steps=10,
            ),
        )
    )
    return {
        "case": "matrix-free-reduced-newton-krylov",
        "size": size,
        "wall_seconds": wall_seconds,
        "status": int(result.status),
        "final_objective": float(result.objective),
        "primal_feasibility": float(result.diagnostics.primal_feasibility),
        "dual_feasibility": float(result.diagnostics.dual_feasibility),
        "linear_solves": int(result.diagnostics.linear_solves),
        "setup_refreshes": int(result.diagnostics.setup_refreshes),
        "numeric_refreshes": int(result.diagnostics.numeric_refreshes),
        "state_design_bytes": int(initial_state.nbytes + initial_design.nbytes),
        "dense_reduced_hessian_bytes": int(
            size * size * initial_design.dtype.itemsize
        ),
    }


def _fold_continuation() -> dict[str, Any]:
    problem = phx.continuation.ParameterContinuationProblem(
        lambda state, parameter, _: jnp.asarray([state[0] ** 2 + parameter - 1.0]),
        problem_id="quadratic-fold",
    )
    branch, wall_seconds = _timed(
        lambda: phx.continuation.continue_branch(
            problem,
            jnp.asarray([1.0]),
            jnp.asarray(0.0),
            num_steps=12,
            method=phx.continuation.PseudoArclengthContinuation(
                initial_step=0.2,
                maximum_step=0.25,
            ),
        )
    )
    return {
        "case": "pseudo-arclength-fold",
        "wall_seconds": wall_seconds,
        "status": int(branch.status),
        "accepted_points": len(branch.points),
        "fold_events": len(branch.fold_brackets),
        "maximum_residual_norm": max(
            float(point.residual_norm) for point in branch.points
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Microbenchmarks for Phydrax nonlinear optimization contracts."
    )
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--sample-size", type=int, default=256)
    parser.add_argument("--sampling-repeats", type=int, default=32)
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    size = 16 if arguments.smoke else int(arguments.size)
    sample_size = 32 if arguments.smoke else int(arguments.sample_size)
    repeats = 4 if arguments.smoke else int(arguments.sampling_repeats)
    if size < 1 or sample_size < 1 or repeats < 2:
        raise ValueError(
            "Benchmark sizes must be positive and repeats must be at least two."
        )

    scaling_sizes = (
        tuple(sorted({max(1, size // 4), max(1, size // 2), size}))
        if arguments.smoke
        else tuple(sorted({size, max(size, 1024), max(size, 4096)}))
    )
    records = (
        *(_matrix_free_newton(current_size) for current_size in scaling_sizes),
        _newton_iteration_lifecycle(size),
        _dense_newton_reference(size),
        _line_search_accounting(),
        _common_random_numbers(sample_size, repeats),
        _poorly_scaled_bounds(),
        _matrix_free_constraints(size),
        _reduced_state_design(size),
        _fold_continuation(),
    )
    for record in records:
        print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()
