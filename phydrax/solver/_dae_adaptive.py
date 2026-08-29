#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..nonlinear import (
    implicit_root_result,
    NonlinearStatus,
    refresh_nonlinear,
)
from ..nonlinear._prepared import (
    _seed_nonlinear_continuation,
    _solve_prepared_nonlinear_stateful,
)
from ._bdf_method import (
    bdf_predict as _general_bdf_predict,
    bdf_rate as _general_bdf_rate,
    bdf_shift_offset as _general_bdf_shift_offset,
    BDFMethod,
)
from ._dae_initialization import (
    _initialize_dae,
    _masked_rms,
    DAEInitializationResult,
    DAEInitializationStatus,
)
from ._differential_algebraic import (
    _dense_initial_regularity,
    _dense_stage_regularity,
    _history_stage_arguments,
    _initial_regularity,
    _linear_failure,
    _regularity_status,
    DAEAttemptHistory,
    DAEAttemptStatus,
    DAEContinuation,
    DAERegularityEvidence,
    DAERegularityStatus,
    DAEReplayEvidence,
    DAEStatus,
    DAEStepHistory,
    DAETerminationStatus,
    DifferentialAlgebraicSolution,
    PreparedDAESolve,
)


_RUNNING = -1


class _NodeArchive(StrictModule):
    states: Array
    rates: Array
    valid: Array
    rate_valid: Array
    status: Array
    residual_norm: Array
    residual_threshold: Array
    differential_norm: Array
    constraint_norm: Array


class _StepArchive(StrictModule):
    times: Array
    step_sizes: Array
    orders: Array
    error_ratios: Array
    source_attempts: Array
    valid: Array
    save_indices: Array


class _AttemptArchive(StrictModule):
    times: Array
    step_sizes: Array
    orders: Array
    status: Array
    error_ratios: Array
    nonlinear_status: Array
    nonlinear_iterations: Array
    residual_evaluations: Array
    jacobian_preparations: Array
    linear_solves: Array
    linear_iterations: Array
    globalization_rejections: Array
    setup_refreshes: Array
    numeric_refreshes: Array
    stale_retries: Array
    linear_rejections: Array
    residual_certifications: Array
    valid: Array


class _RegularityArchive(StrictModule):
    status: Array
    rank: Array
    condition: Array
    valid: Array


class _AdaptiveCarry(StrictModule):
    time: Array
    states: Array
    rates: Array
    times: Array
    step_sizes: Array
    history_depth: Array
    accepted_order: Array
    previous_error_ratio: Array
    proposed_step_size: Array
    save_index: Array
    accepted_count: Array
    attempt_count: Array
    consecutive_rejections: Array
    jacobian_age: Array
    last_alpha: Array
    last_nonlinear_iterations: Array
    force_refresh: Array
    terminal_status: Array
    retained_nonlinear: Any
    nodes: _NodeArchive
    steps: _StepArchive
    attempts: _AttemptArchive
    regularity: _RegularityArchive


def _error_coefficient(
    step_size: Array,
    previous_step_size: Array,
    previous_previous_step_size: Array,
    order: Array,
    /,
) -> Array:
    alpha_1 = step_size / (step_size + previous_step_size)
    alpha_2 = step_size / (step_size + previous_step_size + previous_previous_step_size)
    second = jnp.maximum(jnp.abs(alpha_1 + alpha_2 - 0.5), alpha_2)
    higher = jnp.reciprocal(order.astype(step_size.dtype) + 1.0)
    return jnp.where(order == 1, alpha_1, jnp.where(order == 2, second, higher))


def _initial_step_size(
    state: Array,
    state_rate: Array,
    interval: Array,
    differential_variables: Array,
    relative_tolerance: Array,
    absolute_tolerance: Array,
    policy,
    /,
) -> Array:
    if policy.initial_step is not None:
        estimate = jnp.asarray(policy.initial_step, dtype=state.real.dtype)
    else:
        weights = absolute_tolerance + relative_tolerance * jnp.abs(state)
        safe_weights = jnp.maximum(weights, jnp.finfo(state.real.dtype).tiny)
        state_scale = _masked_rms(state / safe_weights, differential_variables)
        rate_scale = _masked_rms(state_rate / safe_weights, differential_variables)
        estimate = jnp.where(
            (state_scale < 1e-5) | (rate_scale < 1e-5),
            1e-3 * interval,
            1e-2 * state_scale / rate_scale,
        )
    maximum = (
        interval
        if policy.maximum_step is None
        else jnp.minimum(interval, policy.maximum_step)
    )
    minimum = (
        jnp.asarray(0.0, dtype=estimate.dtype)
        if policy.minimum_step is None
        else jnp.asarray(policy.minimum_step, dtype=estimate.dtype)
    )
    return jnp.clip(estimate, minimum, maximum)


def _minimum_step(time: Array, policy, /) -> Array:
    if policy.minimum_step is not None:
        return jnp.asarray(policy.minimum_step, dtype=time.dtype)
    return (
        16.0
        * jnp.finfo(time.dtype).eps
        * jnp.maximum(jnp.abs(time), jnp.asarray(1.0, dtype=time.dtype))
    )


def _accepted_factor(
    error_ratio: Array,
    previous_error_ratio: Array,
    order: Array,
    policy,
    /,
) -> Array:
    exponent = 1.0 / (order.astype(error_ratio.dtype) + 1.0)
    proportional = jnp.maximum(error_ratio, 1e-12) ** (-0.7 * exponent)
    integral = jnp.maximum(previous_error_ratio, 1e-12) ** (0.4 * exponent)
    return jnp.clip(
        policy.safety * proportional * integral,
        policy.accepted_growth_minimum,
        policy.accepted_growth_maximum,
    )


def _rejected_factor(error_ratio: Array, order: Array, policy, /) -> Array:
    exponent = 1.0 / (order.astype(error_ratio.dtype) + 1.0)
    proposed = policy.safety * jnp.maximum(error_ratio, 1e-12) ** (-exponent)
    return jnp.clip(
        proposed,
        policy.rejected_shrink_minimum,
        policy.rejected_shrink_maximum,
    )


def _scaled_problem_residual(problem, time, state, state_rate, args, /):
    inputs = (
        None
        if problem.input_policy is None
        else problem.input_policy.evaluate(time, state, args)
    )
    return problem.system.scaled_residual(
        time,
        state,
        state_rate,
        args,
        inputs=inputs,
    )


def _continuation_initialization(
    prepared: PreparedDAESolve,
    continuation: DAEContinuation,
    args: Any,
    /,
) -> DAEInitializationResult:
    problem = prepared.problem
    system = problem.system
    policy = prepared.plan.policy
    adaptive = policy.adaptive
    if adaptive is None:
        raise ValueError("Continuation initialization requires adaptive policy.")
    scaled = _scaled_problem_residual(
        problem,
        continuation.time,
        continuation.state,
        continuation.state_rate,
        args,
    )
    differential_equations = system.structure.differential_equation_mask(
        system.state_shape
    )
    algebraic_equations = system.structure.algebraic_equation_mask(system.state_shape)
    residual_norm = _masked_rms(scaled, jnp.ones(system.state_shape, dtype=bool))
    differential_norm = _masked_rms(scaled, differential_equations)
    constraint_norm = _masked_rms(scaled, algebraic_equations)
    finite = (
        jnp.all(jnp.isfinite(continuation.state))
        & jnp.all(jnp.isfinite(continuation.state_rate))
        & jnp.isfinite(residual_norm)
    )
    residual_accepted = residual_norm <= adaptive.residual_tolerance
    constraint_accepted = constraint_norm <= adaptive.constraint_tolerance
    valid = finite & residual_accepted & constraint_accepted
    status = jnp.where(
        ~finite,
        int(DAEInitializationStatus.NONFINITE),
        jnp.where(
            residual_accepted & constraint_accepted,
            int(DAEInitializationStatus.SUCCESS),
            int(DAEInitializationStatus.RESIDUAL_TOO_LARGE),
        ),
    ).astype(jnp.int32)
    zero = jnp.zeros_like(continuation.state)
    return DAEInitializationResult(
        state=continuation.state,
        state_rate=continuation.state_rate,
        state_correction=zero,
        rate_correction=zero,
        fixed_state_mask=prepared.initialization.fixed_state_mask,
        fixed_rate_mask=prepared.initialization.fixed_rate_mask,
        rate_valid=jnp.ones(system.state_shape, dtype=bool) & valid,
        residual_norm=residual_norm,
        residual_threshold=jnp.asarray(
            adaptive.residual_tolerance, dtype=residual_norm.dtype
        ),
        differential_residual_norm=differential_norm,
        constraint_norm=constraint_norm,
        valid=valid,
        status=status,
        nonlinear_result=None,
        initialization_id=prepared.problem.initialization.initialization_id,
    )


def _stage_regularity(
    prepared,
    state,
    arguments,
    nonlinear_result,
    accepted_count,
    candidate_solved,
    /,
):
    policy = prepared.plan.policy.regularity
    dimension = int(prepared.problem.initial_state.size)
    if policy.mode == "periodic":
        requested = candidate_solved & ((accepted_count % policy.interval) == 0)

        def probe(_):
            rank, condition, finite = _dense_stage_regularity(prepared, state, arguments)
            status = _regularity_status(
                rank,
                condition,
                finite,
                dimension,
                policy.condition_limit,
            )
            return status, rank, condition, jnp.asarray(True)

        def skip(_):
            return (
                jnp.asarray(int(DAERegularityStatus.NOT_RUN), dtype=jnp.int32),
                jnp.asarray(-1, dtype=jnp.int32),
                jnp.asarray(jnp.nan, dtype=state.real.dtype),
                jnp.asarray(False),
            )

        return jax.lax.cond(requested, probe, skip, operand=None)
    diagnostics = nonlinear_result.diagnostics
    status = _regularity_status(
        diagnostics.final_linear_rank,
        diagnostics.final_linear_condition_estimate,
        diagnostics.final_linear_converged,
        dimension,
        policy.condition_limit,
    )
    return (
        status,
        diagnostics.final_linear_rank,
        diagnostics.final_linear_condition_estimate,
        candidate_solved,
    )


def _validate_continuation(
    prepared: PreparedDAESolve,
    continuation: DAEContinuation,
    /,
) -> None:
    problem = prepared.problem
    policy = prepared.plan.policy
    if continuation.problem_id != problem.problem_id:
        raise ValueError("DAE continuation problem identity does not match.")
    if continuation.system_id != problem.system.system_id:
        raise ValueError("DAE continuation system identity does not match.")
    expected_policy_id = (
        None if problem.input_policy is None else problem.input_policy.policy_id
    )
    if continuation.input_policy_id != expected_policy_id:
        raise ValueError("DAE continuation input policy identity does not match.")
    if continuation.state_shape != problem.system.state_shape:
        raise ValueError("DAE continuation state shape does not match.")
    if continuation.state_dtype != str(problem.initial_state.dtype):
        expected = str(jnp.dtype(problem.initial_state.dtype))
        observed = str(jnp.dtype(continuation.state.dtype))
        if observed != expected:
            raise ValueError("DAE continuation state dtype does not match.")
    if continuation.method_id != policy.method.method_id:
        raise ValueError("DAE continuation temporal method does not match.")
    if continuation.initialization_id != problem.initialization.initialization_id:
        raise ValueError("DAE continuation initialization contract does not match.")
    if continuation.nonlinear_method_id != policy.nonlinear_method.method_id:
        raise ValueError("DAE continuation nonlinear method does not match.")
    if continuation.stage_linear_plan_id != prepared.stage_linear_plan_id:
        raise ValueError("DAE continuation linear plan does not match.")
    if continuation.nonlinear_solve is None:
        raise ValueError("Adaptive continuation is missing retained nonlinear state.")


def _initialize_archives(prepared, initialization, /):
    problem = prepared.problem
    adaptive = prepared.plan.policy.adaptive
    dtype = problem.initial_state.real.dtype
    node_count = prepared.time_grid.num_points
    accepted_capacity = adaptive.maximum_accepted_steps
    attempt_capacity = adaptive.maximum_attempts
    nan_state = jnp.full(
        (node_count,) + problem.system.state_shape,
        jnp.nan,
        dtype=problem.initial_state.dtype,
    )
    nodes = _NodeArchive(
        states=nan_state.at[0].set(initialization.state),
        rates=nan_state.at[0].set(initialization.state_rate),
        valid=jnp.zeros((node_count,), dtype=bool).at[0].set(initialization.valid),
        rate_valid=jnp.zeros((node_count,) + problem.system.state_shape, dtype=bool)
        .at[0]
        .set(initialization.rate_valid),
        status=jnp.full((node_count,), int(DAEStatus.NOT_RUN), dtype=jnp.int32)
        .at[0]
        .set(
            jnp.where(
                initialization.valid,
                int(DAEStatus.SUCCESS),
                int(DAEStatus.INITIALIZATION_FAILED),
            )
        ),
        residual_norm=jnp.full((node_count,), jnp.inf, dtype=dtype)
        .at[0]
        .set(initialization.residual_norm),
        residual_threshold=jnp.full((node_count,), jnp.inf, dtype=dtype)
        .at[0]
        .set(initialization.residual_threshold),
        differential_norm=jnp.full((node_count,), jnp.inf, dtype=dtype)
        .at[0]
        .set(initialization.differential_residual_norm),
        constraint_norm=jnp.full((node_count,), jnp.inf, dtype=dtype)
        .at[0]
        .set(initialization.constraint_norm),
    )
    steps = _StepArchive(
        times=jnp.full((accepted_capacity,), jnp.nan, dtype=dtype),
        step_sizes=jnp.full((accepted_capacity,), jnp.nan, dtype=dtype),
        orders=jnp.zeros((accepted_capacity,), dtype=jnp.int32),
        error_ratios=jnp.full((accepted_capacity,), jnp.inf, dtype=dtype),
        source_attempts=jnp.full((accepted_capacity,), -1, dtype=jnp.int32),
        valid=jnp.zeros((accepted_capacity,), dtype=bool),
        save_indices=jnp.full((node_count,), -2, dtype=jnp.int32).at[0].set(-1),
    )
    attempts = _AttemptArchive(
        times=jnp.full((attempt_capacity,), jnp.nan, dtype=dtype),
        step_sizes=jnp.full((attempt_capacity,), jnp.nan, dtype=dtype),
        orders=jnp.zeros((attempt_capacity,), dtype=jnp.int32),
        status=jnp.full(
            (attempt_capacity,), int(DAEAttemptStatus.NOT_RUN), dtype=jnp.int32
        ),
        error_ratios=jnp.full((attempt_capacity,), jnp.inf, dtype=dtype),
        nonlinear_status=jnp.zeros((attempt_capacity,), dtype=jnp.int32),
        nonlinear_iterations=jnp.zeros((attempt_capacity,), dtype=jnp.int32),
        residual_evaluations=jnp.zeros((attempt_capacity,), dtype=jnp.int32),
        jacobian_preparations=jnp.zeros((attempt_capacity,), dtype=jnp.int32),
        linear_solves=jnp.zeros((attempt_capacity,), dtype=jnp.int32),
        linear_iterations=jnp.zeros((attempt_capacity,), dtype=jnp.int32),
        globalization_rejections=jnp.zeros((attempt_capacity,), dtype=jnp.int32),
        setup_refreshes=jnp.zeros((attempt_capacity,), dtype=jnp.int32),
        numeric_refreshes=jnp.zeros((attempt_capacity,), dtype=jnp.int32),
        stale_retries=jnp.zeros((attempt_capacity,), dtype=jnp.int32),
        linear_rejections=jnp.zeros((attempt_capacity,), dtype=jnp.int32),
        residual_certifications=jnp.zeros((attempt_capacity,), dtype=jnp.int32),
        valid=jnp.zeros((attempt_capacity,), dtype=bool),
    )
    regularity = _RegularityArchive(
        status=jnp.full(
            (accepted_capacity,), int(DAERegularityStatus.NOT_RUN), dtype=jnp.int32
        ),
        rank=jnp.full((accepted_capacity,), -1, dtype=jnp.int32),
        condition=jnp.full((accepted_capacity,), jnp.nan, dtype=dtype),
        valid=jnp.zeros((accepted_capacity,), dtype=bool),
    )
    return nodes, steps, attempts, regularity


def _set_attempt(
    archive: _AttemptArchive,
    index: Array,
    *,
    time: Array,
    step_size: Array,
    order: Array,
    status: Array,
    error_ratio: Array,
    nonlinear_result,
    stale_retry: Array,
    residual_certified: Array,
) -> _AttemptArchive:
    diagnostics = nonlinear_result.diagnostics
    return _AttemptArchive(
        times=archive.times.at[index].set(time),
        step_sizes=archive.step_sizes.at[index].set(step_size),
        orders=archive.orders.at[index].set(order),
        status=archive.status.at[index].set(status),
        error_ratios=archive.error_ratios.at[index].set(error_ratio),
        nonlinear_status=archive.nonlinear_status.at[index].set(nonlinear_result.status),
        nonlinear_iterations=archive.nonlinear_iterations.at[index].set(
            diagnostics.iterations
        ),
        residual_evaluations=archive.residual_evaluations.at[index].set(
            diagnostics.residual_evaluations
        ),
        jacobian_preparations=archive.jacobian_preparations.at[index].set(
            diagnostics.jacobian_preparations
        ),
        linear_solves=archive.linear_solves.at[index].set(diagnostics.linear_solves),
        linear_iterations=archive.linear_iterations.at[index].set(
            diagnostics.linear_iterations
        ),
        globalization_rejections=archive.globalization_rejections.at[index].set(
            diagnostics.rejected_steps
        ),
        setup_refreshes=archive.setup_refreshes.at[index].set(
            diagnostics.setup_refreshes
        ),
        numeric_refreshes=archive.numeric_refreshes.at[index].set(
            diagnostics.numeric_refreshes
        ),
        stale_retries=archive.stale_retries.at[index].set(stale_retry.astype(jnp.int32)),
        linear_rejections=archive.linear_rejections.at[index].set(
            _linear_failure(nonlinear_result.status).astype(jnp.int32)
        ),
        residual_certifications=archive.residual_certifications.at[index].set(
            residual_certified.astype(jnp.int32)
        ),
        valid=archive.valid.at[index].set(True),
    )


def _adaptive_primal(
    prepared: PreparedDAESolve,
    args: Any,
    initial_state: ArrayLike | None,
    initial_state_rate: ArrayLike | None,
    continuation: DAEContinuation | None,
    /,
) -> DifferentialAlgebraicSolution:
    problem = prepared.problem
    system = problem.system
    policy = prepared.plan.policy
    adaptive = policy.adaptive
    if adaptive is None:
        raise ValueError("Adaptive execution requires a DAEAdaptivePolicy.")
    temporal_method = policy.method
    if not isinstance(temporal_method, BDFMethod):
        raise ValueError("Adaptive DAE execution requires BDFMethod.")
    save_times = jax.lax.stop_gradient(prepared.time_grid.times)
    differential_variables = system.structure.differential_variable_mask(
        system.state_shape
    )
    differential_equations = system.structure.differential_equation_mask(
        system.state_shape
    )
    algebraic_equations = system.structure.algebraic_equation_mask(system.state_shape)
    relative_tolerance = jnp.broadcast_to(
        adaptive.relative_tolerance, system.state_shape
    ).astype(problem.initial_state.real.dtype)
    absolute_tolerance = jnp.broadcast_to(
        adaptive.absolute_tolerance, system.state_shape
    ).astype(problem.initial_state.real.dtype)

    if continuation is None:
        state_guess = problem.initial_state if initial_state is None else initial_state
        rate_guess = (
            problem.initial_state_rate
            if initial_state_rate is None
            else initial_state_rate
        )
        initialization = _initialize_dae(
            prepared.initialization,
            state_guess,
            rate_guess,
            save_times[0],
            args=args,
            termination=policy.initialization_termination,
        )
        initial_step = _initial_step_size(
            initialization.state,
            initialization.state_rate,
            save_times[1] - save_times[0],
            differential_variables,
            relative_tolerance,
            absolute_tolerance,
            adaptive,
        )
        history_states = jnp.broadcast_to(
            initialization.state,
            (6,) + system.state_shape,
        )
        history_rates = jnp.broadcast_to(
            initialization.state_rate,
            (6,) + system.state_shape,
        )
        history_times = jnp.full((6,), save_times[0], dtype=save_times.dtype)
        history_steps = jnp.full((5,), initial_step, dtype=save_times.dtype)
        history_depth = jnp.asarray(1, dtype=jnp.int32)
        accepted_order = jnp.asarray(1, dtype=jnp.int32)
        previous_error = jnp.asarray(1.0, dtype=save_times.dtype)
        proposed_step = initial_step
        jacobian_age = jnp.asarray(0, dtype=jnp.int32)
        last_alpha = 1.0 / initial_step
        retained_nonlinear = prepared.stage_solve
        initial_terminal = jnp.where(
            initialization.valid,
            _RUNNING,
            int(DAETerminationStatus.INITIALIZATION_FAILED),
        ).astype(jnp.int32)
    else:
        _validate_continuation(prepared, continuation)
        boundary_time = eqx.error_if(
            continuation.time,
            continuation.time != save_times[0],
            "The TimeGrid must begin exactly at the DAE continuation boundary.",
        )
        continuation = eqx.tree_at(
            lambda value: value.time,
            continuation,
            boundary_time,
        )
        initialization = _continuation_initialization(prepared, continuation, args)
        history_states = continuation.states
        history_rates = continuation.state_rates
        history_times = continuation.times
        history_steps = continuation.step_sizes
        history_depth = continuation.history_depth
        accepted_order = continuation.accepted_order
        previous_error = continuation.previous_error_ratio
        proposed_step = continuation.proposed_step_size
        jacobian_age = continuation.jacobian_age
        last_alpha = continuation.last_alpha
        _, stage_static = eqx.partition(prepared.stage_solve, eqx.is_array)
        retained_nonlinear = eqx.combine(continuation.nonlinear_solve, stage_static)
        initial_terminal = jnp.where(
            initialization.valid,
            _RUNNING,
            int(DAETerminationStatus.CONTINUATION_INCONSISTENT),
        ).astype(jnp.int32)
    if policy.regularity.mode == "periodic":
        (
            consistency_status,
            consistency_rank,
            consistency_condition,
        ) = _dense_initial_regularity(
            prepared,
            initialization,
            save_times[0],
            args,
        )
    else:
        consistency_status, consistency_rank, consistency_condition = _initial_regularity(
            initialization,
            int(problem.initial_state.size),
            policy.regularity.condition_limit,
        )
    consistency_failed = (policy.regularity.failure == "status") & (
        consistency_status == int(DAERegularityStatus.NUMERICALLY_SINGULAR)
    )
    initial_terminal = jnp.where(
        (initial_terminal == _RUNNING) & consistency_failed,
        int(DAETerminationStatus.REGULARITY_FAILED),
        initial_terminal,
    ).astype(jnp.int32)

    nodes, steps, attempts, regularity = _initialize_archives(prepared, initialization)
    retained_dynamic, retained_static = eqx.partition(retained_nonlinear, eqx.is_array)
    carry = _AdaptiveCarry(
        time=save_times[0],
        states=history_states,
        rates=history_rates,
        times=history_times,
        step_sizes=history_steps,
        history_depth=history_depth,
        accepted_order=accepted_order,
        previous_error_ratio=previous_error,
        proposed_step_size=proposed_step,
        save_index=jnp.asarray(1, dtype=jnp.int32),
        accepted_count=jnp.asarray(0, dtype=jnp.int32),
        attempt_count=jnp.asarray(0, dtype=jnp.int32),
        consecutive_rejections=jnp.asarray(0, dtype=jnp.int32),
        jacobian_age=jacobian_age,
        last_alpha=last_alpha,
        last_nonlinear_iterations=jnp.asarray(0, dtype=jnp.int32),
        force_refresh=jnp.asarray(False),
        terminal_status=initial_terminal,
        retained_nonlinear=retained_dynamic,
        nodes=nodes,
        steps=steps,
        attempts=attempts,
        regularity=regularity,
    )

    def condition(current):
        return (
            (current.terminal_status == _RUNNING)
            & (current.save_index < save_times.size)
            & (current.accepted_count < adaptive.maximum_accepted_steps)
            & (current.attempt_count < adaptive.maximum_attempts)
            & (current.consecutive_rejections <= adaptive.maximum_consecutive_rejections)
        )

    def body(current):
        target_time = save_times[current.save_index]
        remaining = target_time - current.time
        maximum_step = (
            remaining
            if adaptive.maximum_step is None
            else jnp.minimum(remaining, adaptive.maximum_step)
        )
        proposed = jnp.minimum(current.proposed_step_size, maximum_step)
        ratio_limited = jnp.minimum(
            proposed,
            policy.max_step_ratio * current.step_sizes[0],
        )
        candidate_step = jnp.minimum(ratio_limited, remaining)
        boundary_tolerance = (
            8.0
            * jnp.finfo(save_times.dtype).eps
            * jnp.maximum(
                1.0,
                jnp.maximum(jnp.abs(target_time), jnp.abs(current.time)),
            )
        )
        step_size = jnp.where(
            remaining - candidate_step <= boundary_tolerance,
            remaining,
            candidate_step,
        )
        ratio = step_size / current.step_sizes[0]
        ratio_valid = (
            (ratio >= 1.0 / policy.max_step_ratio)
            & (ratio <= policy.max_step_ratio)
            & (current.consecutive_rejections == 0)
        )
        available_order = jnp.minimum(
            jnp.asarray(temporal_method.maximum_order, dtype=jnp.int32),
            jnp.maximum(current.history_depth - 1, 1),
        )
        order = jnp.where(ratio_valid, available_order, 1).astype(jnp.int32)
        stage_time = current.time + step_size
        predictor = _general_bdf_predict(
            current.states[:5],
            current.rates[:5],
            current.times[:5],
            stage_time,
            order,
            current.history_depth,
        )
        alpha, _ = _general_bdf_shift_offset(
            current.states[:5],
            current.times[:5],
            stage_time,
            order,
        )
        alpha_ratio = jnp.maximum(
            alpha / current.last_alpha,
            current.last_alpha / alpha,
        )
        iteration_refresh = (
            jnp.asarray(False)
            if policy.temporal_reuse.refresh_after_iterations is None
            else (
                current.last_nonlinear_iterations
                >= policy.temporal_reuse.refresh_after_iterations
            )
        )
        reuse = (
            policy.temporal_reuse.enabled
            & ~current.force_refresh
            & (current.jacobian_age > 0)
            & (current.jacobian_age < policy.temporal_reuse.maximum_jacobian_age)
            & (alpha_ratio <= policy.temporal_reuse.maximum_alpha_ratio)
            & ~iteration_refresh
        )
        arguments = _history_stage_arguments(
            target_time=stage_time,
            state_history=current.states[:5],
            history_times=current.times[:5],
            order=order,
            model_args=args,
        )
        retained_current = eqx.combine(current.retained_nonlinear, retained_static)

        reused = _seed_nonlinear_continuation(
            retained_current,
            prepared.stage_problem,
            predictor,
            args=arguments,
            defer_refresh_steps=policy.nonlinear_termination.maximum_steps,
        )
        refreshed = refresh_nonlinear(
            retained_current,
            prepared.stage_problem,
            predictor,
            args=arguments,
        )
        reused_dynamic, _ = eqx.partition(reused, eqx.is_array)
        refreshed_dynamic, refreshed_static = eqx.partition(refreshed, eqx.is_array)
        seeded_dynamic = jax.lax.cond(
            reuse,
            lambda _: reused_dynamic,
            lambda _: refreshed_dynamic,
            operand=None,
        )
        seeded = eqx.combine(seeded_dynamic, refreshed_static)
        nonlinear_result, retained = _solve_prepared_nonlinear_stateful(seeded)
        retained_dynamic_, _ = eqx.partition(retained, eqx.is_array)
        state = jnp.asarray(nonlinear_result.state)
        state_rate = _general_bdf_rate(
            state,
            current.states[:5],
            current.times[:5],
            stage_time,
            order,
        )
        scaled = _scaled_problem_residual(
            prepared.problem,
            current.time + step_size,
            state,
            state_rate,
            args,
        )
        residual_norm = _masked_rms(scaled, jnp.ones(system.state_shape, dtype=bool))
        differential_norm = _masked_rms(scaled, differential_equations)
        constraint_norm = _masked_rms(scaled, algebraic_equations)
        correction = state - predictor
        coefficient = _error_coefficient(
            step_size,
            current.step_sizes[0],
            current.step_sizes[1],
            order,
        )
        weights = absolute_tolerance + relative_tolerance * jnp.maximum(
            jnp.abs(state), jnp.abs(current.states[0])
        )
        scaled_error = (
            coefficient
            * correction
            / jnp.maximum(weights, jnp.finfo(state.real.dtype).tiny)
        )
        error_ratio = _masked_rms(scaled_error, differential_variables)
        nonlinear_success = nonlinear_result.status == int(NonlinearStatus.SUCCESS)
        finite = (
            jnp.all(jnp.isfinite(state))
            & jnp.all(jnp.isfinite(state_rate))
            & jnp.isfinite(residual_norm)
            & jnp.isfinite(error_ratio)
        )
        residual_certified = residual_norm <= adaptive.residual_tolerance
        constraint_certified = constraint_norm <= adaptive.constraint_tolerance
        local_error_accepted = error_ratio <= 1.0
        candidate_solved = nonlinear_success & finite & residual_certified
        (
            regularity_status,
            regularity_rank,
            regularity_condition,
            regularity_valid,
        ) = _stage_regularity(
            prepared,
            state,
            arguments,
            nonlinear_result,
            current.accepted_count,
            candidate_solved,
        )
        regularity_failed = (
            (policy.regularity.failure == "status")
            & (regularity_status == int(DAERegularityStatus.NUMERICALLY_SINGULAR))
            & regularity_valid
        )
        accepted = (
            candidate_solved
            & constraint_certified
            & local_error_accepted
            & ~regularity_failed
        )
        stale_retry = reuse & ~candidate_solved
        attempt_status = jnp.where(
            accepted,
            int(DAEAttemptStatus.ACCEPTED),
            jnp.where(
                stale_retry,
                int(DAEAttemptStatus.STALE_JACOBIAN_RETRY),
                jnp.where(
                    regularity_failed,
                    int(DAEAttemptStatus.REGULARITY_REJECTED),
                    jnp.where(
                        ~finite,
                        int(DAEAttemptStatus.NONFINITE_REJECTED),
                        jnp.where(
                            _linear_failure(nonlinear_result.status),
                            int(DAEAttemptStatus.LINEAR_REJECTED),
                            jnp.where(
                                ~nonlinear_success,
                                int(DAEAttemptStatus.NONLINEAR_REJECTED),
                                jnp.where(
                                    ~residual_certified,
                                    int(DAEAttemptStatus.RESIDUAL_REJECTED),
                                    jnp.where(
                                        ~constraint_certified,
                                        int(DAEAttemptStatus.CONSTRAINT_REJECTED),
                                        int(DAEAttemptStatus.LOCAL_ERROR_REJECTED),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ).astype(jnp.int32)
        attempts_ = _set_attempt(
            current.attempts,
            current.attempt_count,
            time=current.time,
            step_size=step_size,
            order=order,
            status=attempt_status,
            error_ratio=error_ratio,
            nonlinear_result=nonlinear_result,
            stale_retry=stale_retry,
            residual_certified=residual_certified,
        )
        attempt_count = current.attempt_count + 1
        accepted_time = jnp.where(
            step_size >= remaining,
            target_time,
            current.time + step_size,
        )
        lands_on_save = accepted_time == target_time

        def accept(_):
            accepted_index = current.accepted_count
            steps_ = _StepArchive(
                times=current.steps.times.at[accepted_index].set(accepted_time),
                step_sizes=current.steps.step_sizes.at[accepted_index].set(step_size),
                orders=current.steps.orders.at[accepted_index].set(order),
                error_ratios=current.steps.error_ratios.at[accepted_index].set(
                    error_ratio
                ),
                source_attempts=current.steps.source_attempts.at[accepted_index].set(
                    current.attempt_count
                ),
                valid=current.steps.valid.at[accepted_index].set(True),
                save_indices=jax.lax.cond(
                    lands_on_save,
                    lambda values: values.at[current.save_index].set(accepted_index),
                    lambda values: values,
                    current.steps.save_indices,
                ),
            )
            nodes_ = jax.lax.cond(
                lands_on_save,
                lambda archive: _NodeArchive(
                    states=archive.states.at[current.save_index].set(state),
                    rates=archive.rates.at[current.save_index].set(state_rate),
                    valid=archive.valid.at[current.save_index].set(True),
                    rate_valid=archive.rate_valid.at[current.save_index].set(
                        jnp.ones(system.state_shape, dtype=bool)
                    ),
                    status=archive.status.at[current.save_index].set(
                        int(DAEStatus.SUCCESS)
                    ),
                    residual_norm=archive.residual_norm.at[current.save_index].set(
                        residual_norm
                    ),
                    residual_threshold=archive.residual_threshold.at[
                        current.save_index
                    ].set(adaptive.residual_tolerance),
                    differential_norm=archive.differential_norm.at[
                        current.save_index
                    ].set(differential_norm),
                    constraint_norm=archive.constraint_norm.at[current.save_index].set(
                        constraint_norm
                    ),
                ),
                lambda archive: archive,
                current.nodes,
            )
            regularity_ = _RegularityArchive(
                status=current.regularity.status.at[accepted_index].set(
                    regularity_status
                ),
                rank=current.regularity.rank.at[accepted_index].set(regularity_rank),
                condition=current.regularity.condition.at[accepted_index].set(
                    regularity_condition
                ),
                valid=current.regularity.valid.at[accepted_index].set(regularity_valid),
            )
            diagnostics = nonlinear_result.diagnostics
            refreshed_during_solve = diagnostics.jacobian_preparations > 0
            next_age = jnp.where(
                ~reuse | refreshed_during_solve,
                1,
                current.jacobian_age + 1,
            ).astype(jnp.int32)
            factor = _accepted_factor(
                error_ratio,
                current.previous_error_ratio,
                order,
                adaptive,
            )
            next_step = step_size * factor
            if adaptive.maximum_step is not None:
                next_step = jnp.minimum(next_step, adaptive.maximum_step)
            return _AdaptiveCarry(
                time=accepted_time,
                states=jnp.concatenate((state[None, ...], current.states[:-1]), axis=0),
                rates=jnp.concatenate(
                    (state_rate[None, ...], current.rates[:-1]), axis=0
                ),
                times=jnp.concatenate((accepted_time[None], current.times[:-1]), axis=0),
                step_sizes=jnp.concatenate(
                    (step_size[None], current.step_sizes[:-1]), axis=0
                ),
                history_depth=jnp.minimum(current.history_depth + 1, 6),
                accepted_order=order,
                previous_error_ratio=jnp.maximum(error_ratio, 1e-12),
                proposed_step_size=next_step,
                save_index=current.save_index + lands_on_save.astype(jnp.int32),
                accepted_count=accepted_index + 1,
                attempt_count=attempt_count,
                consecutive_rejections=jnp.asarray(0, dtype=jnp.int32),
                jacobian_age=next_age,
                last_alpha=alpha,
                last_nonlinear_iterations=diagnostics.iterations,
                force_refresh=jnp.asarray(False),
                terminal_status=jnp.asarray(_RUNNING, dtype=jnp.int32),
                retained_nonlinear=retained_dynamic_,
                nodes=nodes_,
                steps=steps_,
                attempts=attempts_,
                regularity=regularity_,
            )

        def reject(_):
            nonlinear_failure = ~nonlinear_success | ~finite
            factor = jnp.where(
                stale_retry,
                1.0,
                jnp.where(
                    nonlinear_failure,
                    adaptive.nonlinear_failure_shrink,
                    _rejected_factor(error_ratio, order, adaptive),
                ),
            )
            next_step = step_size * factor
            minimum = _minimum_step(current.time, adaptive)
            exhausted = (~stale_retry & (next_step < minimum)) | (
                (current.time + next_step) == current.time
            )
            next_rejections = current.consecutive_rejections + 1
            terminal = jnp.where(
                regularity_failed,
                int(DAETerminationStatus.REGULARITY_FAILED),
                jnp.where(
                    exhausted,
                    int(DAETerminationStatus.MINIMUM_STEP_REACHED),
                    jnp.where(
                        next_rejections > adaptive.maximum_consecutive_rejections,
                        int(DAETerminationStatus.REPEATED_REJECTIONS),
                        _RUNNING,
                    ),
                ),
            ).astype(jnp.int32)
            return _AdaptiveCarry(
                time=current.time,
                states=current.states,
                rates=current.rates,
                times=current.times,
                step_sizes=current.step_sizes,
                history_depth=current.history_depth,
                accepted_order=current.accepted_order,
                previous_error_ratio=current.previous_error_ratio,
                proposed_step_size=jnp.where(stale_retry, step_size, next_step),
                save_index=current.save_index,
                accepted_count=current.accepted_count,
                attempt_count=attempt_count,
                consecutive_rejections=next_rejections,
                jacobian_age=current.jacobian_age,
                last_alpha=current.last_alpha,
                last_nonlinear_iterations=nonlinear_result.diagnostics.iterations,
                force_refresh=jnp.asarray(True),
                terminal_status=terminal,
                retained_nonlinear=retained_dynamic_,
                nodes=current.nodes,
                steps=current.steps,
                attempts=attempts_,
                regularity=current.regularity,
            )

        return jax.lax.cond(accepted, accept, reject, operand=None)

    carry = jax.lax.while_loop(condition, body, carry)
    terminal_status = jnp.where(
        carry.terminal_status != _RUNNING,
        carry.terminal_status,
        jnp.where(
            carry.save_index >= save_times.size,
            int(DAETerminationStatus.SUCCESS),
            jnp.where(
                carry.accepted_count >= adaptive.maximum_accepted_steps,
                int(DAETerminationStatus.MAXIMUM_ACCEPTED_STEPS_REACHED),
                jnp.where(
                    carry.attempt_count >= adaptive.maximum_attempts,
                    int(DAETerminationStatus.MAXIMUM_ATTEMPTS_REACHED),
                    int(DAETerminationStatus.REPEATED_REJECTIONS),
                ),
            ),
        ),
    ).astype(jnp.int32)
    continuation_out = DAEContinuation(
        time=carry.time,
        states=carry.states,
        state_rates=carry.rates,
        times=carry.times,
        step_sizes=carry.step_sizes,
        history_depth=carry.history_depth,
        accepted_order=carry.accepted_order,
        previous_error_ratio=carry.previous_error_ratio,
        proposed_step_size=carry.proposed_step_size,
        jacobian_age=carry.jacobian_age,
        last_alpha=carry.last_alpha,
        nonlinear_solve=carry.retained_nonlinear,
        problem_id=problem.problem_id,
        system_id=system.system_id,
        input_policy_id=(
            None if problem.input_policy is None else problem.input_policy.policy_id
        ),
        method_id=policy.method.method_id,
        initialization_id=problem.initialization.initialization_id,
        nonlinear_method_id=policy.nonlinear_method.method_id,
        stage_linear_plan_id=prepared.stage_linear_plan_id,
    )
    regularity_out = DAERegularityEvidence(
        consistency_status=consistency_status,
        consistency_rank=consistency_rank,
        consistency_condition_estimate=consistency_condition,
        stage_status=carry.regularity.status,
        stage_rank=carry.regularity.rank,
        stage_condition_estimate=carry.regularity.condition,
        stage_valid=carry.regularity.valid,
        consistency_operator="configured-consistency-coordinate-jacobian",
        stage_operator="implicit-stage:F_y+shift*F_ydot",
    )
    if policy.failure == "error":
        node_states = eqx.error_if(
            carry.nodes.states,
            terminal_status != int(DAETerminationStatus.SUCCESS),
            "Adaptive DAE solve failed.",
        )
    else:
        node_states = carry.nodes.states
    return DifferentialAlgebraicSolution(
        times=save_times,
        states=node_states,
        state_rates=carry.nodes.rates,
        valid=carry.nodes.valid,
        rate_valid=carry.nodes.rate_valid,
        status=carry.nodes.status,
        residual_norm=carry.nodes.residual_norm,
        residual_threshold=carry.nodes.residual_threshold,
        differential_residual_norm=carry.nodes.differential_norm,
        constraint_norm=carry.nodes.constraint_norm,
        step_history=DAEStepHistory(
            accepted_times=carry.steps.times,
            step_sizes=carry.steps.step_sizes,
            orders=carry.steps.orders,
            error_ratios=carry.steps.error_ratios,
            source_attempt_indices=carry.steps.source_attempts,
            valid=carry.steps.valid,
            count=carry.accepted_count,
            save_step_indices=carry.steps.save_indices,
        ),
        attempt_history=DAEAttemptHistory(
            times=carry.attempts.times,
            proposed_step_sizes=carry.attempts.step_sizes,
            orders=carry.attempts.orders,
            status=carry.attempts.status,
            error_ratios=carry.attempts.error_ratios,
            nonlinear_status=carry.attempts.nonlinear_status,
            nonlinear_iterations=carry.attempts.nonlinear_iterations,
            residual_evaluations=carry.attempts.residual_evaluations,
            jacobian_preparations=carry.attempts.jacobian_preparations,
            linear_solves=carry.attempts.linear_solves,
            linear_iterations=carry.attempts.linear_iterations,
            globalization_rejections=carry.attempts.globalization_rejections,
            setup_refreshes=carry.attempts.setup_refreshes,
            numeric_refreshes=carry.attempts.numeric_refreshes,
            stale_jacobian_retries=carry.attempts.stale_retries,
            linear_rejections=carry.attempts.linear_rejections,
            residual_certifications=carry.attempts.residual_certifications,
            valid=carry.attempts.valid,
            count=carry.attempt_count,
        ),
        initialization=initialization,
        continuation=continuation_out,
        regularity=regularity_out,
        replay=DAEReplayEvidence(
            accepted_steps=carry.accepted_count,
            selected_chunk_size=prepared.plan.replay_chunk_size,
            estimated_memory_bytes=prepared.plan.replay_memory_bytes,
            checkpointing=policy.replay.checkpointing,
        ),
        termination_status=terminal_status,
        problem_id=problem.problem_id,
        system_id=system.system_id,
        input_policy_id=(
            None if problem.input_policy is None else problem.input_policy.policy_id
        ),
        time_id=prepared.time_grid.time_id,
        plan_id=prepared.plan.plan_id,
        prepared_id=prepared.prepared_id,
        source_discretization_bundle=problem.discretization_bundle,
        nonlinear_method_id=policy.nonlinear_method.method_id,
        stage_linear_plan_id=prepared.stage_linear_plan_id,
        initialization_linear_plan_id=prepared.initialization_linear_plan_id,
        method_id=policy.method.method_id,
        adaptive=True,
    )


def _replay_initial(
    prepared,
    args,
    initial_state,
    initial_state_rate,
    continuation,
    /,
):
    policy = prepared.plan.policy
    times = jax.lax.stop_gradient(prepared.time_grid.times)
    if continuation is None:
        state = prepared.problem.initial_state if initial_state is None else initial_state
        rate = (
            prepared.problem.initial_state_rate
            if initial_state_rate is None
            else initial_state_rate
        )
        initialization = _initialize_dae(
            prepared.initialization,
            state,
            rate,
            times[0],
            args=args,
            termination=policy.initialization_termination,
        )
        states = jnp.broadcast_to(
            initialization.state,
            (6,) + prepared.problem.system.state_shape,
        )
        rates = jnp.broadcast_to(
            initialization.state_rate,
            (6,) + prepared.problem.system.state_shape,
        )
        step_sizes = jnp.full(
            (5,),
            jax.lax.stop_gradient(
                _initial_step_size(
                    initialization.state,
                    initialization.state_rate,
                    times[1] - times[0],
                    prepared.problem.system.structure.differential_variable_mask(
                        prepared.problem.system.state_shape
                    ),
                    jnp.broadcast_to(
                        policy.adaptive.relative_tolerance,
                        prepared.problem.system.state_shape,
                    ),
                    jnp.broadcast_to(
                        policy.adaptive.absolute_tolerance,
                        prepared.problem.system.state_shape,
                    ),
                    policy.adaptive,
                )
            ),
            dtype=times.dtype,
        )
        history_times = jnp.full((6,), times[0], dtype=times.dtype)
    else:
        initialization = _continuation_initialization(prepared, continuation, args)
        states = continuation.states
        rates = continuation.state_rates
        history_times = jax.lax.stop_gradient(continuation.times)
        step_sizes = jax.lax.stop_gradient(continuation.step_sizes)
    return initialization, states, rates, history_times, step_sizes


class _ReplayCarry(StrictModule):
    states: Array
    rates: Array
    times: Array
    step_sizes: Array
    saved_states: Array
    saved_rates: Array


def _replay_solution(
    prepared,
    args,
    initial_state,
    initial_state_rate,
    continuation,
    frozen,
    /,
):
    initialization, states, rates, history_times, previous_steps = _replay_initial(
        prepared,
        args,
        initial_state,
        initial_state_rate,
        continuation,
    )
    saved_states = jnp.full_like(frozen.states, jnp.nan).at[0].set(initialization.state)
    saved_rates = (
        jnp.full_like(frozen.state_rates, jnp.nan).at[0].set(initialization.state_rate)
    )
    initial = _ReplayCarry(
        states=states,
        rates=rates,
        times=history_times,
        step_sizes=previous_steps,
        saved_states=saved_states,
        saved_rates=saved_rates,
    )
    schedule_times = jax.lax.stop_gradient(frozen.step_history.accepted_times)
    schedule_steps = jax.lax.stop_gradient(frozen.step_history.step_sizes)
    schedule_orders = jax.lax.stop_gradient(frozen.step_history.orders)
    schedule_valid = jax.lax.stop_gradient(frozen.step_history.valid)
    save_indices = jax.lax.stop_gradient(frozen.step_history.save_step_indices)
    indices = jnp.arange(schedule_steps.size, dtype=jnp.int32)

    def replay_step(current, values):
        index, time, step_size, order, valid = values
        safe_step_size = jnp.where(valid, step_size, 1.0)
        safe_order = jnp.where(valid, order, 1).astype(jnp.int32)
        safe_time = jnp.where(
            valid,
            time,
            current.times[0] + safe_step_size,
        )

        def execute(carry):
            predictor = _general_bdf_predict(
                carry.states[:5],
                carry.rates[:5],
                carry.times[:5],
                safe_time,
                safe_order,
            )
            arguments = _history_stage_arguments(
                target_time=safe_time,
                state_history=carry.states[:5],
                history_times=carry.times[:5],
                order=safe_order,
                model_args=args,
                active=valid,
            )
            seeded = refresh_nonlinear(
                prepared.stage_solve,
                prepared.stage_problem,
                predictor,
                args=arguments,
            )
            result = implicit_root_result(seeded)
            state = jnp.asarray(result.state)
            rate = _general_bdf_rate(
                state,
                carry.states[:5],
                carry.times[:5],
                safe_time,
                safe_order,
            )
            matches = save_indices == index
            should_save = jnp.any(matches)
            save_index = jnp.argmax(matches).astype(jnp.int32)
            saved_states_ = jax.lax.cond(
                should_save,
                lambda output: output.at[save_index].set(state),
                lambda output: output,
                carry.saved_states,
            )
            saved_rates_ = jax.lax.cond(
                should_save,
                lambda output: output.at[save_index].set(rate),
                lambda output: output,
                carry.saved_rates,
            )
            return _ReplayCarry(
                states=jnp.concatenate((state[None, ...], carry.states[:-1]), axis=0),
                rates=jnp.concatenate((rate[None, ...], carry.rates[:-1]), axis=0),
                times=jnp.concatenate((safe_time[None], carry.times[:-1]), axis=0),
                step_sizes=jnp.concatenate(
                    (safe_step_size[None], carry.step_sizes[:-1]), axis=0
                ),
                saved_states=saved_states_,
                saved_rates=saved_rates_,
            )

        return jax.lax.cond(valid, execute, lambda carry: carry, current), None

    xs = (
        indices,
        schedule_times,
        schedule_steps,
        schedule_orders,
        schedule_valid,
    )
    if prepared.plan.policy.replay.checkpointing == "full":
        replayed, _ = jax.lax.scan(replay_step, initial, xs)
    else:
        chunk_size = prepared.plan.replay_chunk_size
        capacity = schedule_steps.size
        padded_capacity = ((capacity + chunk_size - 1) // chunk_size) * chunk_size
        padding = padded_capacity - capacity

        def pad(value, fill):
            return jnp.pad(value, ((0, padding),), constant_values=fill)

        padded = (
            pad(indices, 0).reshape((-1, chunk_size)),
            pad(schedule_times, 0.0).reshape((-1, chunk_size)),
            pad(schedule_steps, 1.0).reshape((-1, chunk_size)),
            pad(schedule_orders, 1).reshape((-1, chunk_size)),
            pad(schedule_valid, False).reshape((-1, chunk_size)),
        )

        @jax.checkpoint
        def replay_chunk(current, chunk):
            return jax.lax.scan(replay_step, current, chunk)[0]

        def chunk_step(current, chunk):
            return replay_chunk(current, chunk), None

        replayed, _ = jax.lax.scan(chunk_step, initial, padded)

    system = prepared.problem.system
    differential_equations = system.structure.differential_equation_mask(
        system.state_shape
    )
    algebraic_equations = system.structure.algebraic_equation_mask(system.state_shape)

    def certify(time, state, rate):
        scaled = _scaled_problem_residual(
            prepared.problem,
            time,
            state,
            rate,
            args,
        )
        return (
            _masked_rms(scaled, jnp.ones(system.state_shape, dtype=bool)),
            _masked_rms(scaled, differential_equations),
            _masked_rms(scaled, algebraic_equations),
        )

    residual, differential, constraint = jax.vmap(certify)(
        jax.lax.stop_gradient(frozen.times),
        replayed.saved_states,
        replayed.saved_rates,
    )
    continuation_out = eqx.tree_at(
        lambda value: (
            value.time,
            value.states,
            value.state_rates,
            value.times,
            value.step_sizes,
        ),
        frozen.continuation,
        (
            replayed.times[0],
            replayed.states,
            replayed.rates,
            replayed.times,
            replayed.step_sizes,
        ),
    )
    stopped = jax.tree.map(
        lambda value: jax.lax.stop_gradient(value) if eqx.is_array(value) else value,
        frozen,
    )
    return eqx.tree_at(
        lambda value: (
            value.states,
            value.state_rates,
            value.residual_norm,
            value.differential_residual_norm,
            value.constraint_norm,
            value.initialization,
            value.continuation,
        ),
        stopped,
        (
            replayed.saved_states,
            replayed.saved_rates,
            residual,
            differential,
            constraint,
            initialization,
            continuation_out,
        ),
    )


@eqx.filter_custom_jvp
def solve_adaptive_dae(
    prepared: PreparedDAESolve,
    args: Any,
    initial_state: ArrayLike | None,
    initial_state_rate: ArrayLike | None,
    continuation: DAEContinuation | None,
    /,
) -> DifferentialAlgebraicSolution:
    return _adaptive_primal(
        prepared,
        args,
        initial_state,
        initial_state_rate,
        continuation,
    )


@solve_adaptive_dae.def_jvp
def _solve_adaptive_dae_jvp(primals, tangents):
    prepared, args, initial_state, initial_state_rate, continuation = primals
    primal = _adaptive_primal(
        prepared,
        args,
        initial_state,
        initial_state_rate,
        continuation,
    )

    def replay(prepared_, args_, initial_state_, initial_state_rate_, continuation_):
        return _replay_solution(
            prepared_,
            args_,
            initial_state_,
            initial_state_rate_,
            continuation_,
            primal,
        )

    _, tangent = eqx.filter_jvp(replay, primals, tangents)
    successful = primal.termination_status == int(DAETerminationStatus.SUCCESS)
    tangent = eqx.tree_at(
        lambda value: (value.states, value.state_rates),
        tangent,
        (
            jnp.where(successful, tangent.states, jnp.nan),
            jnp.where(successful, tangent.state_rates, jnp.nan),
        ),
    )
    return primal, tangent


__all__ = ["solve_adaptive_dae"]
