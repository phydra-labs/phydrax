#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from .._linear_refresh import LinearRefreshState
from .._strict import StrictModule
from ._iterative._base import AbstractScalarIterativeMethod
from ._iterative._types import (
    _tree_allfinite,
    _tree_norm,
    _validate_real_inexact_tree,
    IterativeStepMetrics,
    MinimizationProblem,
    MinimizationResult,
    OptimizationDiagnostics,
    OptimizationProvenance,
    OptimizationStatus,
    OptimizationTermination,
)


class ScalarIterativeState(StrictModule):
    """Persistent accepted-point state and counters for native scalar methods."""

    iteration: Array
    initial_optimality_norm: Array
    accepted_steps: Array
    rejected_steps: Array
    objective_evaluations: Array
    gradient_evaluations: Array
    hvp_evaluations: Array
    linear_solves: Array
    linear_iterations: Array
    setup_refreshes: Array
    numeric_refreshes: Array
    linear_refresh_state: LinearRefreshState | None
    direction_fallbacks: Array
    metrics: IterativeStepMetrics

    def __init__(
        self,
        *,
        iteration: Any = 0,
        initial_optimality_norm: Any = jnp.nan,
        accepted_steps: Any = 0,
        rejected_steps: Any = 0,
        objective_evaluations: Any = 0,
        gradient_evaluations: Any = 0,
        hvp_evaluations: Any = 0,
        linear_solves: Any = 0,
        linear_iterations: Any = 0,
        setup_refreshes: Any = 0,
        numeric_refreshes: Any = 0,
        linear_refresh_state: LinearRefreshState | None = None,
        direction_fallbacks: Any = 0,
        metrics: IterativeStepMetrics | None = None,
    ):
        self.iteration = jnp.asarray(iteration, dtype=jnp.int32)
        self.initial_optimality_norm = jnp.asarray(initial_optimality_norm)
        self.accepted_steps = jnp.asarray(accepted_steps, dtype=jnp.int32)
        self.rejected_steps = jnp.asarray(rejected_steps, dtype=jnp.int32)
        self.objective_evaluations = jnp.asarray(objective_evaluations, dtype=jnp.int32)
        self.gradient_evaluations = jnp.asarray(gradient_evaluations, dtype=jnp.int32)
        self.hvp_evaluations = jnp.asarray(hvp_evaluations, dtype=jnp.int32)
        self.linear_solves = jnp.asarray(linear_solves, dtype=jnp.int32)
        self.linear_iterations = jnp.asarray(linear_iterations, dtype=jnp.int32)
        self.setup_refreshes = jnp.asarray(setup_refreshes, dtype=jnp.int32)
        self.numeric_refreshes = jnp.asarray(numeric_refreshes, dtype=jnp.int32)
        if linear_refresh_state is not None and not isinstance(
            linear_refresh_state, LinearRefreshState
        ):
            raise TypeError("linear_refresh_state must be a LinearRefreshState or None.")
        self.linear_refresh_state = linear_refresh_state
        self.direction_fallbacks = jnp.asarray(direction_fallbacks, dtype=jnp.int32)
        self.metrics = IterativeStepMetrics() if metrics is None else metrics


class _ScalarRun(StrictModule):
    parameters: PyTree[Array]
    state: ScalarIterativeState
    status: Array

    def __init__(
        self,
        parameters: PyTree[Any],
        state: ScalarIterativeState,
        status: Any,
        /,
    ):
        self.parameters = parameters
        self.state = state
        self.status = jnp.asarray(status, dtype=jnp.int32)


def _run_scalar_iterations(
    method: AbstractScalarIterativeMethod,
    value_function,
    initial_parameters: PyTree[Any],
    termination: OptimizationTermination,
    /,
) -> _ScalarRun:
    state = method.prepare_state(value_function, initial_parameters)
    state, static_state = eqx.partition(state, eqx.is_array)
    initial_status = jnp.where(
        _tree_allfinite(initial_parameters),
        int(OptimizationStatus.ITERATING),
        int(OptimizationStatus.NONFINITE_INPUT),
    ).astype(jnp.int32)

    def condition(carry):
        _, current_state, status = carry
        within_evaluations = (
            jnp.asarray(True)
            if termination.maximum_evaluations is None
            else current_state.objective_evaluations < termination.maximum_evaluations
        )
        return (
            (status == int(OptimizationStatus.ITERATING))
            & (current_state.iteration < termination.maximum_steps)
            & within_evaluations
        )

    def body(carry):
        current_parameters, dynamic_state, _ = carry
        current_state = eqx.combine(dynamic_state, static_state)
        next_parameters, next_state, _ = method.step(
            value_function,
            current_parameters,
            current_state,
            termination=termination,
        )
        next_status = method.step_metrics(next_state).status
        if termination.maximum_evaluations is not None:
            exhausted = (next_status == int(OptimizationStatus.ITERATING)) & (
                next_state.objective_evaluations >= termination.maximum_evaluations
            )
            next_status = jnp.where(
                exhausted,
                int(OptimizationStatus.MAXIMUM_EVALUATIONS_REACHED),
                next_status,
            )
        dynamic_next_state, _ = eqx.partition(next_state, eqx.is_array)
        return next_parameters, dynamic_next_state, next_status

    parameters, state, status = jax.lax.while_loop(
        condition,
        body,
        (initial_parameters, state, initial_status),
    )
    state = eqx.combine(state, static_state)
    status = jnp.where(
        status == int(OptimizationStatus.ITERATING),
        int(OptimizationStatus.MAXIMUM_STEPS_REACHED),
        status,
    )
    return _ScalarRun(parameters, state, status)


def solve_scalar_iterative(
    method: AbstractScalarIterativeMethod,
    problem: MinimizationProblem,
    initial_parameters: PyTree[Any],
    /,
    *,
    termination: OptimizationTermination,
    args: Any,
) -> MinimizationResult:
    """Execute a native accepted-point scalar method to terminal status."""

    if not isinstance(problem, MinimizationProblem):
        raise TypeError("problem must be a MinimizationProblem.")
    if not isinstance(termination, OptimizationTermination):
        raise TypeError("termination must be an OptimizationTermination.")
    if problem.bounds is not None or problem.constraints:
        raise ValueError(
            "This scalar iterative method is unconstrained; use a bound or "
            "nonlinear constrained method."
        )
    parameters = _validate_real_inexact_tree(
        initial_parameters,
        name="initial_parameters",
    )
    provenance = OptimizationProvenance(
        problem_id=problem.problem_id,
        method=method.method_id,
        backend="phydrax-native",
        globalization=method.globalization_id,
        matrix_free=method.capabilities.matrix_free,
        implicit_differentiation=method.capabilities.implicit_differentiation,
    )

    def value_function(candidate):
        value, _ = problem.value(candidate, args)
        return value

    run = _run_scalar_iterations(
        method,
        value_function,
        parameters,
        termination,
    )
    parameters, state, status = run.parameters, run.state, run.status
    (final_value, final_auxiliary), final_gradient = problem.value_and_gradient(
        parameters,
        args,
    )
    finite_final = (
        jnp.isfinite(final_value)
        & _tree_allfinite(parameters)
        & _tree_allfinite(final_gradient)
    )
    final_optimality = _tree_norm(final_gradient)
    initial_nonfinite = status == int(OptimizationStatus.NONFINITE_INPUT)
    status = jnp.where(
        initial_nonfinite,
        status,
        jnp.where(
            ~finite_final,
            int(OptimizationStatus.NONFINITE_EVALUATION),
            jnp.where(
                final_optimality
                <= termination.optimality_threshold(state.initial_optimality_norm),
                int(OptimizationStatus.SUCCESS),
                status,
            ),
        ),
    )
    final_metrics = method.step_metrics(state)
    if method.globalization_id == "strong-wolfe":
        globalization_evaluations = jnp.maximum(
            state.objective_evaluations - 1,
            0,
        )
    elif method.globalization_id == "trust-region-ratio":
        globalization_evaluations = state.accepted_steps + state.rejected_steps
    else:
        globalization_evaluations = (
            state.objective_evaluations - state.gradient_evaluations
        )
    diagnostics = OptimizationDiagnostics(
        iterations=state.iteration,
        accepted_steps=state.accepted_steps,
        rejected_steps=state.rejected_steps,
        objective_evaluations=state.objective_evaluations + 1,
        gradient_evaluations=state.gradient_evaluations + 1,
        hvp_evaluations=state.hvp_evaluations,
        linear_solves=state.linear_solves,
        setup_refreshes=state.setup_refreshes,
        numeric_refreshes=state.numeric_refreshes,
        linear_iterations=state.linear_iterations,
        globalization_evaluations=globalization_evaluations,
        initial_optimality_norm=state.initial_optimality_norm,
        final_optimality_norm=final_optimality,
        final_step_norm=final_metrics.step_norm,
        accepted_step_size=final_metrics.accepted_step_size,
        damping=final_metrics.damping,
        reduction_ratio=final_metrics.reduction_ratio,
        direction_fallbacks=state.direction_fallbacks,
    )
    return MinimizationResult(
        parameters,
        final_value,
        final_auxiliary,
        status,
        diagnostics,
        provenance,
    )


__all__ = ["ScalarIterativeState", "solve_scalar_iterative"]
