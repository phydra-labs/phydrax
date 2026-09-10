#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from .._iteration import (
    bind_iteration_scope,
    finalize_iteration,
    initialize_iteration,
    IterationCapabilities,
    IterationCoordinates,
    IterationPhase,
    IterationPlan,
    IterationRecord,
    IterationRuntimeState,
    IterationScope,
    update_iteration,
)
from .._linear_refresh import LinearRefreshState
from .._strict import StrictModule
from ._iterative._base import AbstractScalarIterativeMethod
from ._iterative._types import (
    _PreparedMinimizationValue,
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


def _scalar_iteration_record(
    state: ScalarIterativeState,
    status,
    phase,
    /,
    *,
    active=True,
    committed=False,
    terminal=False,
) -> IterationRecord:
    return IterationRecord(
        IterationCoordinates(
            phase,
            state.iteration,
            attempt=state.iteration,
            accepted=state.accepted_steps,
            rejected=state.rejected_steps,
            active=active,
            committed=committed,
            terminal=terminal,
        ),
        status,
        state.metrics,
    )


def _run_scalar_iterations(
    method: AbstractScalarIterativeMethod,
    value_function,
    initial_parameters: PyTree[Any],
    termination: OptimizationTermination,
    iteration: IterationPlan | None = None,
    /,
) -> tuple[
    _ScalarRun,
    IterationScope | None,
    IterationCapabilities | None,
    IterationRuntimeState | None,
]:
    state = method.prepare_state(value_function, initial_parameters)
    state, static_state = eqx.partition(state, eqx.is_array)
    initial_status = jnp.where(
        _tree_allfinite(initial_parameters),
        int(OptimizationStatus.ITERATING),
        int(OptimizationStatus.NONFINITE_INPUT),
    ).astype(jnp.int32)
    iteration_scope = None
    iteration_capabilities = None
    iteration_state = None
    if iteration is not None:
        iteration_capabilities = IterationCapabilities(
            ("terminal", "step", "attempt"),
            device_stop=True,
            mapped_records=True,
        )
        if iteration.stop_rule is not None and iteration.granularity == "terminal":
            raise ValueError("Terminal-only optimization observation cannot stop.")
        iteration_scope = bind_iteration_scope(
            iteration,
            iteration_capabilities,
            method.method_id,
        )
        iteration_state = initialize_iteration(
            iteration,
            _scalar_iteration_record(
                eqx.combine(state, static_state),
                initial_status,
                IterationPhase.START,
            ),
        )

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

    if iteration is None:
        parameters, state, status = jax.lax.while_loop(
            condition,
            body,
            (initial_parameters, state, initial_status),
        )
    else:
        assert iteration_state is not None

        def observed_condition(carry):
            return condition(carry[:3]) & ~carry[3].stop_requested

        def observed_body(carry):
            previous_state = eqx.combine(carry[1], static_state)
            next_carry = body(carry[:3])
            next_state = eqx.combine(next_carry[1], static_state)
            advanced = next_state.iteration > previous_state.iteration
            committed = next_state.accepted_steps > previous_state.accepted_steps
            selected = advanced & (committed if iteration.granularity == "step" else True)
            phase = jnp.where(
                committed,
                int(IterationPhase.COMMIT),
                int(IterationPhase.ATTEMPT),
            )
            record = _scalar_iteration_record(
                next_state,
                next_carry[2],
                phase,
                active=selected,
                committed=committed,
            )
            observed = update_iteration(
                iteration,
                carry[3],
                record,
                allow_stop=committed,
            )
            return (*next_carry, observed)

        parameters, state, status, iteration_state = jax.lax.while_loop(
            observed_condition,
            observed_body,
            (initial_parameters, state, initial_status, iteration_state),
        )
    state = eqx.combine(state, static_state)
    status = jnp.where(
        status == int(OptimizationStatus.ITERATING),
        int(OptimizationStatus.MAXIMUM_STEPS_REACHED),
        status,
    )
    if iteration_state is not None:
        status = jnp.where(
            iteration_state.stop_requested,
            int(OptimizationStatus.USER_STOPPED),
            status,
        )
    return (
        _ScalarRun(parameters, state, status),
        iteration_scope,
        iteration_capabilities,
        iteration_state,
    )


def solve_scalar_iterative(
    method: AbstractScalarIterativeMethod,
    problem: MinimizationProblem,
    initial_parameters: PyTree[Any],
    /,
    *,
    termination: OptimizationTermination,
    args: Any,
    iteration: IterationPlan | None = None,
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

    value_function = _PreparedMinimizationValue(problem, args)

    (
        run,
        iteration_scope,
        iteration_capabilities,
        iteration_state,
    ) = _run_scalar_iterations(
        method,
        value_function,
        parameters,
        termination,
        iteration,
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
    result = MinimizationResult(
        parameters,
        final_value,
        final_auxiliary,
        status,
        diagnostics,
        provenance,
        method_evidence=problem.hessian_action_kind,
    )
    if iteration is None:
        return result
    assert iteration_scope is not None
    assert iteration_capabilities is not None
    assert iteration_state is not None
    terminal = _scalar_iteration_record(
        state,
        status,
        IterationPhase.TERMINAL,
        active=True,
        committed=result.successful,
        terminal=True,
    )
    evidence = finalize_iteration(
        iteration,
        iteration_scope,
        iteration_capabilities,
        iteration_state,
        terminal,
    )
    return eqx.tree_at(
        lambda value: value.iteration_evidence,
        result,
        evidence,
        is_leaf=lambda value: value is None,
    )


__all__ = ["ScalarIterativeState", "solve_scalar_iterative"]
