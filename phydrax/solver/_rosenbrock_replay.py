#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jaxtyping import Array

from .._array_tree import ArrayPyTreeSchema
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._numerics._checkpointed_scan import checkpointed_scan
from .._strict import StrictModule
from ..discretization import RealizedTemporalMesh, TemporalMesh
from ..dynamics import TimeGrid
from ..linalg import (
    ArraySpace,
    FunctionLinearOperator,
    LinearSolvePlan,
    LinearSolvePolicy,
    LinearSystem,
    plan,
)
from ._differential import DifferentialProblem, DifferentialSolution
from ._fixed_step import FixedStepReplayPolicy
from ._rosenbrock import (
    _DEFAULT_ARGS,
    _default_linear_policy,
    _error_ratio,
    _JacobianAction,
    _rosenbrock_step,
    _ShiftedJacobianAction,
    _solve_rosenbrock_fixed,
    RosenbrockAdaptivePolicy,
    RosenbrockWMethod,
)
from ._temporal_method import (
    native_differentiation_evidence,
    TemporalSolveEvidence,
)
from ._temporal_precision import TemporalPrecisionPolicy


class RosenbrockReplayStatus(IntEnum):
    SUCCESS = 0
    INVALID_INPUT = 1
    NONFINITE = 2
    LINEAR_SOLVE_FAILED = 3
    ERROR_RATIO_EXCEEDED = 4
    INCOMPLETE = 5


class RosenbrockReplayAdequacy(StrictModule):
    """Stepwise evidence that one prescribed Rosenbrock schedule remains usable."""

    step_error_ratio: Array
    step_adequate: Array
    stage_status: Array
    stage_residual_norm: Array
    stage_relative_residual: Array
    stage_finite: Array
    stage_converged: Array
    stage_iterations: Array
    maximum_error_ratio: Array
    first_failed_step: Array
    completed: Array
    refresh_required: Array
    status: Array
    schedule_id: str = eqx.field(static=True)


class PreparedRosenbrockSolve(StrictModule):
    """Static Rosenbrock configuration with reusable matrix-free solve selection."""

    problem: DifferentialProblem
    time_grid: TimeGrid
    method: RosenbrockWMethod
    adaptive: RosenbrockAdaptivePolicy | None
    linear_plan: LinearSolvePlan
    replay: FixedStepReplayPolicy
    precision: TemporalPrecisionPolicy
    state_schema: ArrayPyTreeSchema
    argument_schema: ArrayPyTreeSchema
    space: ArraySpace
    replay_state_bytes: int = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)


class ScheduledRosenbrockSolve(StrictModule):
    """One immutable accepted schedule prepared for repeated pure-JAX replay."""

    prepared: PreparedRosenbrockSolve
    temporal_mesh: TemporalMesh
    step_starts: Array
    step_sizes: Array
    save_steps: Array
    reference_initial_state: Array
    reference_args: Any
    replay: FixedStepReplayPolicy
    source_realization_id: str = eqx.field(static=True)
    record_point_id: str = eqx.field(static=True)
    schedule_id: str = eqx.field(static=True)
    numeric_version: int = eqx.field(static=True)


class _AdaptiveCarry(StrictModule):
    time: Array
    state: Array
    step_size: Array
    accepted_count: Array
    attempt_count: Array
    save_index: Array
    step_sizes: Array
    accepted_times: Array
    step_valid: Array
    save_steps: Array
    successful: Array


class _ReplayCarry(StrictModule):
    state: Array
    successful: Array
    output_index: Array
    output_states: Array
    output_valid: Array
    first_failed_step: Array
    status: Array


class _ReplayStepMetrics(StrictModule):
    error_ratio: Array
    adequate: Array
    stage_status: Array
    residual_norm: Array
    relative_residual: Array
    stage_finite: Array
    stage_converged: Array
    iterations: Array


def _checkpointing_name(replay: FixedStepReplayPolicy, /) -> str:
    if replay.mode == "full":
        return "full-replay"
    if replay.mode == "step":
        return "bounded-rematerialization"
    return "chunked-replay"


def _checkpoint_count(
    replay: FixedStepReplayPolicy,
    step_count: int,
    /,
) -> int | None:
    if replay.mode == "block":
        assert replay.block_size is not None
        return (step_count + replay.block_size - 1) // replay.block_size
    if replay.mode == "scheduled":
        assert replay.schedule is not None
        return replay.schedule.checkpoint_slots
    return None


def _runtime_inputs(
    prepared: PreparedRosenbrockSolve,
    args: Any,
    initial_state: Any | None,
    /,
) -> tuple[Array, Any, Array]:
    state = prepared.problem.initial_state if initial_state is None else initial_state
    prepared.state_schema.validate(state)
    if not eqx.is_array(state):
        raise TypeError("Rosenbrock state must be one array.")
    state_ = jnp.asarray(state)
    runtime_args = prepared.problem.args if args is _DEFAULT_ARGS else args
    prepared.argument_schema.validate(runtime_args)
    finite = jnp.all(jnp.isfinite(state_)) & prepared.argument_schema.finite_mask(
        runtime_args
    )
    return state_, runtime_args, finite


def _linear_plan(
    problem: DifferentialProblem,
    method: RosenbrockWMethod,
    policy: LinearSolvePolicy,
    space: ArraySpace,
    step_size: Array,
    /,
) -> LinearSolvePlan:
    if policy.failure.mode != "status":
        raise ValueError("Rosenbrock replay requires linear failure mode 'status'.")
    if policy.differentiation.mode != "mathematical":
        raise ValueError(
            "Rosenbrock replay requires mathematical linear differentiation."
        )
    if policy.preconditioning is not None or policy.recycling is not None:
        raise ValueError(
            "The first qualified Rosenbrock replay route excludes preconditioning "
            "and recycling."
        )
    time = jnp.asarray(problem.t0, dtype=problem.initial_state.real.dtype)
    jacobian = _JacobianAction(problem.drift, time, problem.initial_state, problem.args)
    operator = FunctionLinearOperator(
        _ShiftedJacobianAction(jacobian, step_size * method.stage[0][0]),
        source=space,
        target=space,
        operator_id=f"{method.method_id}:shifted-jacobian",
    )
    selected = plan(LinearSystem(operator), policy)
    if selected.backend != "native-krylov":
        raise ValueError("Rosenbrock replay currently requires native-krylov.")
    return selected


def prepare_rosenbrock(
    problem: DifferentialProblem,
    time_grid: TimeGrid,
    /,
    *,
    method: RosenbrockWMethod | None = None,
    adaptive: RosenbrockAdaptivePolicy | None = None,
    linear_policy: LinearSolvePolicy | None = None,
    replay: FixedStepReplayPolicy | None = None,
    precision: TemporalPrecisionPolicy | None = None,
) -> PreparedRosenbrockSolve:
    """Prepare one deterministic Euclidean RA34PW2 solve."""
    if not isinstance(problem, DifferentialProblem) or problem.stochastic:
        raise TypeError("prepare_rosenbrock requires a deterministic problem.")
    if not isinstance(time_grid, TimeGrid):
        raise TypeError("time_grid must be a TimeGrid.")
    geometry = problem.state_geometry
    if geometry is not None and not geometry.trivial:
        raise ValueError("Rosenbrock-W currently requires Euclidean state geometry.")
    if not eqx.is_array(problem.initial_state):
        raise TypeError("Rosenbrock-W currently requires one array state.")
    state = jnp.asarray(problem.initial_state)
    if state.size == 0 or not jnp.issubdtype(state.dtype, jnp.inexact):
        raise TypeError("Rosenbrock-W state must be a nonempty inexact array.")
    times = np.asarray(jax.device_get(time_grid.times))
    if not np.isclose(times[0], problem.t0) or not np.isclose(times[-1], problem.t1):
        raise ValueError("TimeGrid endpoints must match the differential problem.")
    selected = RosenbrockWMethod() if method is None else method
    if not isinstance(selected, RosenbrockWMethod):
        raise TypeError("method must be RosenbrockWMethod or None.")
    if adaptive is not None and not isinstance(adaptive, RosenbrockAdaptivePolicy):
        raise TypeError("adaptive must be RosenbrockAdaptivePolicy or None.")
    if adaptive is not None and adaptive.maximum_accepted_steps < time_grid.num_steps:
        raise ValueError(
            "maximum_accepted_steps must cover every requested output interval."
        )
    precision_ = TemporalPrecisionPolicy() if precision is None else precision
    if not isinstance(precision_, TemporalPrecisionPolicy):
        raise TypeError("precision must be a TemporalPrecisionPolicy or None.")
    precision_.validate_implicit_state(state)
    replay_ = FixedStepReplayPolicy() if replay is None else replay
    if not isinstance(replay_, FixedStepReplayPolicy):
        raise TypeError("replay must be a FixedStepReplayPolicy or None.")
    state_schema = ArrayPyTreeSchema.from_tree(state, case_ndim=0)
    argument_schema = ArrayPyTreeSchema.from_tree(problem.args, case_ndim=0)
    if not bool(np.asarray(jax.device_get(state_schema.finite_mask(state)))):
        raise ValueError("Rosenbrock initial state must be finite.")
    if not bool(np.asarray(jax.device_get(argument_schema.finite_mask(problem.args)))):
        raise ValueError("Rosenbrock array arguments must be finite.")
    policy = _default_linear_policy(state) if linear_policy is None else linear_policy
    if not isinstance(policy, LinearSolvePolicy):
        raise TypeError("linear_policy must be a LinearSolvePolicy or None.")
    space = ArraySpace(state.shape, dtype=state.dtype)
    first_step = (
        times[1] - times[0]
        if adaptive is None
        else min(adaptive.initial_step, times[1] - times[0])
    )
    linear_plan = _linear_plan(
        problem,
        selected,
        policy,
        space,
        jnp.asarray(first_step, dtype=state.real.dtype),
    )
    output_bytes = time_grid.num_steps * state_schema.intrinsic_storage_bytes
    replay_state_bytes = (
        state_schema.intrinsic_storage_bytes
        + output_bytes
        + time_grid.num_steps
        + 4 * np.dtype(np.int32).itemsize
    )
    prepared_id = canonical_fingerprint(
        {
            "kind": "prepared-rosenbrock-solve",
            "problem_id": problem.problem_id,
            "time_grid_id": time_grid.time_id,
            "times": array_tree_fingerprint(times),
            "method_id": selected.method_id,
            "adaptive_policy_id": None if adaptive is None else adaptive.policy_id,
            "linear_plan_id": linear_plan.plan_id,
            "replay_policy_id": replay_.policy_id,
            "precision_policy_id": precision_.policy_id,
            "state_schema_id": state_schema.content_id,
            "argument_schema_id": argument_schema.content_id,
        }
    )
    return PreparedRosenbrockSolve(
        problem=problem,
        time_grid=time_grid,
        method=selected,
        adaptive=adaptive,
        linear_plan=linear_plan,
        replay=replay_,
        precision=precision_,
        state_schema=state_schema,
        argument_schema=argument_schema,
        space=space,
        replay_state_bytes=replay_state_bytes,
        prepared_id=prepared_id,
    )


def _empty_stage_metrics(state: Array, /) -> tuple[Array, ...]:
    real_dtype = state.real.dtype
    return (
        jnp.full((4,), -1, dtype=jnp.int32),
        jnp.zeros((4,), dtype=real_dtype),
        jnp.zeros((4,), dtype=real_dtype),
        jnp.ones((4,), dtype=bool),
        jnp.ones((4,), dtype=bool),
        jnp.zeros((4,), dtype=jnp.int32),
    )


def _run_replay(
    prepared: PreparedRosenbrockSolve,
    state: Array,
    args: Any,
    input_finite: Array,
    step_starts: Array,
    step_sizes: Array,
    active: Array,
    save_steps: Array,
    replay: FixedStepReplayPolicy,
    schedule_id: str,
    record_completed: Array,
    /,
) -> tuple[Array, Array, Array, RosenbrockReplayAdequacy]:
    controller = prepared.adaptive
    if controller is None:
        raise ValueError("Adaptive policy is required for accepted-schedule replay.")
    length = int(step_sizes.shape[0])
    output_count = prepared.time_grid.num_steps
    if replay.mode == "scheduled":
        assert replay.schedule is not None
        if replay.schedule.step_count != length:
            raise ValueError(
                "Prepared replay schedule does not match accepted step count."
            )
        if replay.schedule.state_bytes != prepared.replay_state_bytes:
            raise ValueError(
                "Prepared replay schedule does not match replay boundary bytes."
            )
    output_states = jnp.full(
        (output_count,) + state.shape,
        jnp.asarray(jnp.nan, dtype=state.dtype),
        dtype=state.dtype,
    )
    initial_status = jnp.where(
        input_finite,
        int(RosenbrockReplayStatus.SUCCESS),
        int(RosenbrockReplayStatus.INVALID_INPUT),
    ).astype(jnp.int32)
    initial = _ReplayCarry(
        state=state,
        successful=input_finite,
        output_index=jnp.asarray(0, dtype=jnp.int32),
        output_states=output_states,
        output_valid=jnp.zeros((output_count,), dtype=bool),
        first_failed_step=jnp.asarray(-1, dtype=jnp.int32),
        status=initial_status,
    )
    indices = jnp.arange(length, dtype=jnp.int32)

    def body(carry: _ReplayCarry, values):
        step_index, start, step_size, is_active = values
        should_execute = carry.successful & is_active

        def execute(_):
            result = _rosenbrock_step(
                prepared.problem,
                prepared.method,
                prepared.linear_plan,
                prepared.space,
                start,
                carry.state,
                step_size,
                args,
                prepared.precision,
            )
            ratio = _error_ratio(
                carry.state,
                result.next_state,
                result.defect,
                controller,
                prepared.precision,
            )
            finite = (
                jnp.all(jnp.isfinite(result.next_state))
                & jnp.all(jnp.isfinite(result.defect))
                & jnp.isfinite(ratio)
            )
            linear_ok = (
                jnp.all(result.stage_status == 0)
                & jnp.all(result.stage_finite)
                & jnp.all(result.stage_converged)
            )
            adequate = linear_ok & finite & (ratio <= 1.0)
            status = jnp.where(
                ~linear_ok,
                int(RosenbrockReplayStatus.LINEAR_SOLVE_FAILED),
                jnp.where(
                    ~finite,
                    int(RosenbrockReplayStatus.NONFINITE),
                    int(RosenbrockReplayStatus.ERROR_RATIO_EXCEEDED),
                ),
            ).astype(jnp.int32)
            return (
                result.next_state,
                adequate,
                ratio,
                status,
                result.stage_status,
                result.residual_norm,
                result.relative_residual,
                result.stage_finite,
                result.stage_converged,
                result.iterations,
            )

        def skip(_):
            stage = _empty_stage_metrics(carry.state)
            return (
                carry.state,
                jnp.asarray(True),
                jnp.asarray(0.0, dtype=carry.state.real.dtype),
                carry.status,
                *stage,
            )

        (
            candidate,
            step_adequate,
            error_ratio,
            failure_status,
            stage_status,
            residual_norm,
            relative_residual,
            stage_finite,
            stage_converged,
            iterations,
        ) = lax.cond(should_execute, execute, skip, operand=None)
        failed_now = should_execute & ~step_adequate
        committed_state = jnp.where(
            should_execute & step_adequate,
            candidate,
            carry.state,
        )
        safe_output_index = jnp.clip(carry.output_index, 0, output_count - 1)
        expected_step = save_steps[safe_output_index]
        saves_output = (
            should_execute
            & step_adequate
            & (carry.output_index < output_count)
            & (step_index == expected_step)
        )
        next_output_states = lax.cond(
            saves_output,
            lambda values: values.at[safe_output_index].set(candidate),
            lambda values: values,
            carry.output_states,
        )
        next_output_valid = lax.cond(
            saves_output,
            lambda values: values.at[safe_output_index].set(True),
            lambda values: values,
            carry.output_valid,
        )
        next_successful = carry.successful & (~is_active | step_adequate)
        next_first_failed = jnp.where(
            failed_now & (carry.first_failed_step < 0),
            step_index,
            carry.first_failed_step,
        )
        next_status = jnp.where(failed_now, failure_status, carry.status)
        next_carry = _ReplayCarry(
            state=committed_state,
            successful=next_successful,
            output_index=carry.output_index + saves_output.astype(jnp.int32),
            output_states=next_output_states,
            output_valid=next_output_valid,
            first_failed_step=next_first_failed,
            status=next_status,
        )
        metrics = _ReplayStepMetrics(
            error_ratio=error_ratio,
            adequate=is_active & carry.successful & step_adequate,
            stage_status=stage_status,
            residual_norm=residual_norm,
            relative_residual=relative_residual,
            stage_finite=stage_finite,
            stage_converged=stage_converged,
            iterations=iterations,
        )
        return next_carry, metrics

    final, metrics = checkpointed_scan(
        body,
        initial,
        (indices, step_starts, step_sizes, active),
        length=length,
        mode=replay.mode,
        block_size=replay.block_size,
        schedule=replay.schedule,
    )
    completed = (
        record_completed
        & final.successful
        & (final.output_index == output_count)
        & jnp.all(final.output_valid)
    )
    status = jnp.where(
        completed,
        int(RosenbrockReplayStatus.SUCCESS),
        jnp.where(
            final.status == int(RosenbrockReplayStatus.SUCCESS),
            int(RosenbrockReplayStatus.INCOMPLETE),
            final.status,
        ),
    ).astype(jnp.int32)
    valid_ratios = jnp.where(
        active & jnp.isfinite(metrics.error_ratio), metrics.error_ratio, 0
    )
    maximum_error_ratio = jnp.where(
        completed,
        jnp.max(valid_ratios),
        jnp.asarray(jnp.inf, dtype=state.real.dtype),
    )
    adequacy = RosenbrockReplayAdequacy(
        step_error_ratio=metrics.error_ratio,
        step_adequate=metrics.adequate,
        stage_status=metrics.stage_status,
        stage_residual_norm=metrics.residual_norm,
        stage_relative_residual=metrics.relative_residual,
        stage_finite=metrics.stage_finite,
        stage_converged=metrics.stage_converged,
        stage_iterations=metrics.iterations,
        maximum_error_ratio=maximum_error_ratio,
        first_failed_step=final.first_failed_step,
        completed=completed,
        refresh_required=~completed,
        status=status,
        schedule_id=schedule_id,
    )
    states = jnp.concatenate((state[None, ...], final.output_states), axis=0)
    valid = jnp.concatenate((jnp.asarray([True]), final.output_valid))
    return states, valid, status, adequacy


def _solve_adaptive(
    prepared: PreparedRosenbrockSolve,
    state: Array,
    args: Any,
    input_finite: Array,
    /,
) -> DifferentialSolution:
    controller = prepared.adaptive
    if controller is None:
        raise ValueError("Prepared solve is not adaptive.")
    times = lax.stop_gradient(prepared.time_grid.times)
    initial_step = jnp.minimum(
        jnp.asarray(controller.initial_step, dtype=times.dtype),
        times[1] - times[0],
    )
    if controller.maximum_step is not None:
        initial_step = jnp.minimum(initial_step, controller.maximum_step)
    initial = _AdaptiveCarry(
        time=times[0],
        state=state,
        step_size=initial_step,
        accepted_count=jnp.asarray(0, dtype=jnp.int32),
        attempt_count=jnp.asarray(0, dtype=jnp.int32),
        save_index=jnp.asarray(0, dtype=jnp.int32),
        step_sizes=jnp.zeros((controller.maximum_accepted_steps,), dtype=times.dtype),
        accepted_times=jnp.zeros((controller.maximum_accepted_steps,), dtype=times.dtype),
        step_valid=jnp.zeros((controller.maximum_accepted_steps,), dtype=bool),
        save_steps=jnp.full((prepared.time_grid.num_steps,), -1, dtype=jnp.int32),
        successful=input_finite,
    )

    def condition(current: _AdaptiveCarry):
        return (
            current.successful
            & (current.save_index < prepared.time_grid.num_steps)
            & (current.accepted_count < controller.maximum_accepted_steps)
            & (current.attempt_count < controller.maximum_attempts)
        )

    def body(current: _AdaptiveCarry):
        target = times[current.save_index + 1]
        remaining = target - current.time
        step_size = jnp.minimum(current.step_size, remaining)
        if controller.maximum_step is not None:
            step_size = jnp.minimum(step_size, controller.maximum_step)
        result = _rosenbrock_step(
            prepared.problem,
            prepared.method,
            prepared.linear_plan,
            prepared.space,
            current.time,
            current.state,
            step_size,
            args,
            prepared.precision,
        )
        ratio = _error_ratio(
            current.state,
            result.next_state,
            result.defect,
            controller,
            prepared.precision,
        )
        accepted = result.successful & jnp.isfinite(ratio) & (ratio <= 1.0)
        accepted_index = current.accepted_count
        lands_on_save = step_size == remaining
        endpoint = current.time + step_size
        step_sizes = lax.cond(
            accepted,
            lambda values: values.at[accepted_index].set(step_size),
            lambda values: values,
            current.step_sizes,
        )
        accepted_times = lax.cond(
            accepted,
            lambda values: values.at[accepted_index].set(endpoint),
            lambda values: values,
            current.accepted_times,
        )
        step_valid = lax.cond(
            accepted,
            lambda values: values.at[accepted_index].set(True),
            lambda values: values,
            current.step_valid,
        )
        save_steps = lax.cond(
            accepted & lands_on_save,
            lambda values: values.at[current.save_index].set(accepted_index),
            lambda values: values,
            current.save_steps,
        )
        safe_ratio = jnp.where(jnp.isfinite(ratio), ratio, jnp.inf)
        raw_factor = controller.safety * jnp.maximum(
            safe_ratio, jnp.finfo(step_size.dtype).tiny
        ) ** (-1.0 / 3.0)
        accepted_factor = jnp.clip(
            raw_factor, controller.minimum_factor, controller.maximum_factor
        )
        rejected_factor = jnp.clip(raw_factor, controller.minimum_factor, 1.0)
        factor = jnp.where(accepted, accepted_factor, rejected_factor)
        next_step = step_size * factor
        if controller.maximum_step is not None:
            next_step = jnp.minimum(next_step, controller.maximum_step)
        next_successful = current.successful & (
            accepted | (next_step >= controller.minimum_step)
        )
        return _AdaptiveCarry(
            time=jnp.where(accepted, endpoint, current.time),
            state=jnp.where(accepted, result.next_state, current.state),
            step_size=next_step,
            accepted_count=current.accepted_count + accepted.astype(jnp.int32),
            attempt_count=current.attempt_count + 1,
            save_index=current.save_index + (accepted & lands_on_save).astype(jnp.int32),
            step_sizes=step_sizes,
            accepted_times=accepted_times,
            step_valid=step_valid,
            save_steps=save_steps,
            successful=next_successful,
        )

    record = lax.while_loop(condition, body, initial)
    record_completed = record.successful & (
        record.save_index == prepared.time_grid.num_steps
    )
    step_sizes = lax.stop_gradient(record.step_sizes)
    accepted_times = lax.stop_gradient(record.accepted_times)
    active = lax.stop_gradient(record.step_valid)
    save_steps = lax.stop_gradient(record.save_steps)
    safe_endpoints = jnp.where(active, accepted_times, times[0])
    step_starts = jnp.concatenate((times[:1], safe_endpoints[:-1]))
    states, valid, status, adequacy = _run_replay(
        prepared,
        state,
        args,
        input_finite,
        step_starts,
        step_sizes,
        active,
        save_steps,
        prepared.replay,
        prepared.prepared_id,
        record_completed,
    )
    successful = adequacy.completed
    mesh = RealizedTemporalMesh(
        times[0],
        accepted_times,
        active,
        record.accepted_count,
        adaptive=True,
        source_plan_id=prepared.prepared_id,
        requested_time_id=prepared.time_grid.time_id,
    )
    evidence = TemporalSolveEvidence(
        prepared.method.capabilities,
        native_differentiation_evidence(
            "adjoint:frozen-accepted-grid-linear-solves",
            adaptive=True,
            checkpointing=_checkpointing_name(prepared.replay),
            checkpoint_count=_checkpoint_count(
                prepared.replay,
                controller.maximum_accepted_steps,
            ),
        ),
        equation_form="explicit-ode",
        backend_id="backend:phydrax:rosenbrock-w",
        configuration_id=prepared.prepared_id,
        controller_id=controller.policy_id,
        event_id=None,
        adaptive=True,
        dense=False,
        maximum_steps=controller.maximum_attempts,
        precision_evidence=prepared.precision.evidence_for(state, times),
    )
    return DifferentialSolution(
        times=times,
        states=jax.vmap(prepared.precision.output)(states),
        valid=valid,
        backend_result=status,
        stats={
            "accepted_steps": record.accepted_count,
            "attempts": record.attempt_count,
            "rejected_steps": record.attempt_count - record.accepted_count,
            "accepted_step_sizes": step_sizes,
            "accepted_step_mask": active,
            "save_steps": save_steps,
            "replay_adequacy": adequacy,
            "linear_iterations": jnp.sum(adequacy.stage_iterations, dtype=jnp.int32),
        },
        solver_name="RA34PW2",
        interpretation=prepared.problem.interpretation,
        state_geometry_id=prepared.problem.state_geometry_id,
        solver_id=prepared.method.method_id,
        resolved_method="RA34PW2:adaptive-frozen-grid",
        discretization_bundle=prepared.problem.discretization_bundle,
        backend_successful=successful,
        temporal_evidence=evidence,
        temporal_mesh=mesh,
        problem_id=prepared.problem.problem_id,
    )


def solve_rosenbrock(
    problem_or_prepared: DifferentialProblem | PreparedRosenbrockSolve,
    time_grid: TimeGrid | None = None,
    /,
    *,
    method: RosenbrockWMethod | None = None,
    adaptive: RosenbrockAdaptivePolicy | None = None,
    linear_policy: LinearSolvePolicy | None = None,
    replay: FixedStepReplayPolicy | None = None,
    args: Any = _DEFAULT_ARGS,
    initial_state: Any | None = None,
    precision: TemporalPrecisionPolicy | None = None,
) -> DifferentialSolution:
    """Solve fixed or adaptive RA34PW2 through one prepared contract."""
    if isinstance(problem_or_prepared, PreparedRosenbrockSolve):
        if time_grid is not None or any(
            value is not None
            for value in (method, adaptive, linear_policy, replay, precision)
        ):
            raise ValueError(
                "Prepared Rosenbrock solves do not accept configuration overrides."
            )
        prepared = problem_or_prepared
    else:
        if not isinstance(problem_or_prepared, DifferentialProblem):
            raise TypeError("problem_or_prepared must be a differential problem or plan.")
        if time_grid is None:
            raise TypeError("Raw Rosenbrock solves require a TimeGrid.")
        prepared = prepare_rosenbrock(
            problem_or_prepared,
            time_grid,
            method=method,
            adaptive=adaptive,
            linear_policy=linear_policy,
            replay=replay,
            precision=precision,
        )
    state, runtime_args, finite = _runtime_inputs(prepared, args, initial_state)
    runtime_problem = eqx.tree_at(
        lambda value: value.initial_state,
        prepared.problem,
        state,
    )
    if prepared.adaptive is None:
        return _solve_rosenbrock_fixed(
            runtime_problem,
            prepared.time_grid,
            method=prepared.method,
            linear_policy=prepared.linear_plan,
            args=runtime_args,
            precision=prepared.precision,
        )
    return _solve_adaptive(prepared, state, runtime_args, finite)


def _scheduled_from_source(
    prepared: PreparedRosenbrockSolve,
    source: DifferentialSolution,
    reference_args: Any,
    reference_initial_state: Any | None,
    replay: FixedStepReplayPolicy | None,
    numeric_version: int,
    /,
) -> ScheduledRosenbrockSolve:
    if prepared.adaptive is None:
        raise ValueError("Only adaptive Rosenbrock sources can be scheduled.")
    if not isinstance(source, DifferentialSolution):
        raise TypeError("source must be a DifferentialSolution.")
    if source.temporal_mesh is None or not source.temporal_mesh.adaptive:
        raise ValueError("source must retain an adaptive realized temporal mesh.")
    if source.temporal_evidence is None or (
        source.temporal_evidence.configuration_id != prepared.prepared_id
    ):
        raise ValueError("source and prepared Rosenbrock configuration do not match.")
    if not bool(np.asarray(jax.device_get(source.successful))):
        raise ValueError("Only a completed successful source can be scheduled.")
    state, args, finite = _runtime_inputs(
        prepared,
        reference_args,
        reference_initial_state,
    )
    if not bool(np.asarray(jax.device_get(finite))):
        raise ValueError("Scheduled replay reference inputs must be finite.")
    real_dtype = np.asarray(state).real.dtype
    epsilon = np.finfo(real_dtype).eps
    if not np.allclose(
        np.asarray(jax.device_get(source.states[0])),
        np.asarray(jax.device_get(state)),
        rtol=64 * epsilon,
        atol=64 * epsilon,
    ):
        raise ValueError("Reference initial state does not match the adaptive source.")
    count = int(np.asarray(jax.device_get(source.temporal_mesh.count)))
    if count <= 0:
        raise ValueError("Scheduled replay requires at least one accepted step.")
    endpoints = np.asarray(jax.device_get(source.temporal_mesh.accepted_times[:count]))
    step_sizes = np.asarray(jax.device_get(source.stats["accepted_step_sizes"][:count]))
    save_steps = np.asarray(jax.device_get(source.stats["save_steps"]), dtype=np.int32)
    starts = np.concatenate((np.asarray([prepared.problem.t0]), endpoints[:-1]))
    if (
        np.any(~np.isfinite(starts))
        or np.any(~np.isfinite(endpoints))
        or np.any(~np.isfinite(step_sizes))
        or np.any(step_sizes <= 0)
        or np.any(np.diff(endpoints) <= 0)
        or not np.allclose(
            starts + step_sizes, endpoints, rtol=32 * epsilon, atol=32 * epsilon
        )
    ):
        raise ValueError("Accepted starts, step sizes, and endpoints are inconsistent.")
    expected_outputs = np.arange(prepared.time_grid.num_steps, dtype=np.int32)
    if (
        save_steps.shape != expected_outputs.shape
        or np.any(save_steps < 0)
        or np.any(save_steps >= count)
        or not np.allclose(
            endpoints[save_steps],
            np.asarray(jax.device_get(prepared.time_grid.times[1:])),
            rtol=32 * epsilon,
            atol=32 * epsilon,
        )
    ):
        raise ValueError("Accepted schedule does not cover every requested output.")
    replay_ = FixedStepReplayPolicy() if replay is None else replay
    if not isinstance(replay_, FixedStepReplayPolicy):
        raise TypeError("replay must be a FixedStepReplayPolicy or None.")
    if replay_.mode == "scheduled" and replay_.schedule.step_count != count:
        raise ValueError("Prepared replay schedule does not match accepted step count.")
    record_point_id = canonical_fingerprint(
        {
            "kind": "rosenbrock-record-point",
            "initial_state": array_tree_fingerprint(np.asarray(state)),
            "args": array_tree_fingerprint(args),
        }
    )
    internal_mesh = TemporalMesh(
        np.concatenate((np.asarray([prepared.problem.t0]), endpoints)),
        role="internal",
        realized=True,
        source_plan_id=prepared.prepared_id,
    )
    schedule_id = canonical_fingerprint(
        {
            "kind": "scheduled-rosenbrock-solve",
            "prepared_id": prepared.prepared_id,
            "mesh_id": internal_mesh.mesh_id,
            "step_sizes": array_tree_fingerprint(step_sizes),
            "save_steps": array_tree_fingerprint(save_steps),
            "record_point_id": record_point_id,
            "replay_policy_id": replay_.policy_id,
            "numeric_version": int(numeric_version),
        }
    )
    scheduled = ScheduledRosenbrockSolve(
        prepared=prepared,
        temporal_mesh=internal_mesh,
        step_starts=lax.stop_gradient(jnp.asarray(starts)),
        step_sizes=lax.stop_gradient(jnp.asarray(step_sizes)),
        save_steps=lax.stop_gradient(jnp.asarray(save_steps, dtype=jnp.int32)),
        reference_initial_state=state,
        reference_args=args,
        replay=replay_,
        source_realization_id=source.temporal_mesh.mesh_id,
        record_point_id=record_point_id,
        schedule_id=schedule_id,
        numeric_version=int(numeric_version),
    )
    return scheduled


def schedule_rosenbrock(
    prepared: PreparedRosenbrockSolve,
    source: DifferentialSolution,
    /,
    *,
    reference_args: Any = _DEFAULT_ARGS,
    reference_initial_state: Any | None = None,
    replay: FixedStepReplayPolicy | None = None,
) -> ScheduledRosenbrockSolve:
    """Prepare a completed adaptive source for repeated fixed-schedule replay."""
    if not isinstance(prepared, PreparedRosenbrockSolve):
        raise TypeError("prepared must be PreparedRosenbrockSolve.")
    return _scheduled_from_source(
        prepared,
        source,
        reference_args,
        reference_initial_state,
        replay,
        0,
    )


def solve_scheduled_rosenbrock(
    scheduled: ScheduledRosenbrockSolve,
    /,
    *,
    args: Any = _DEFAULT_ARGS,
    initial_state: Any | None = None,
) -> DifferentialSolution:
    """Replay one immutable accepted schedule at compatible runtime inputs."""
    if not isinstance(scheduled, ScheduledRosenbrockSolve):
        raise TypeError("scheduled must be ScheduledRosenbrockSolve.")
    prepared = scheduled.prepared
    state = scheduled.reference_initial_state if initial_state is None else initial_state
    runtime_args = scheduled.reference_args if args is _DEFAULT_ARGS else args
    state, runtime_args, finite = _runtime_inputs(prepared, runtime_args, state)
    count = scheduled.temporal_mesh.interval_count
    active = jnp.ones((count,), dtype=bool)
    states, valid, status, adequacy = _run_replay(
        prepared,
        state,
        runtime_args,
        finite,
        lax.stop_gradient(scheduled.step_starts),
        lax.stop_gradient(scheduled.step_sizes),
        active,
        lax.stop_gradient(scheduled.save_steps),
        scheduled.replay,
        scheduled.schedule_id,
        jnp.asarray(True),
    )
    accepted_times = lax.stop_gradient(scheduled.temporal_mesh.nodes[1:])
    mesh = RealizedTemporalMesh(
        scheduled.temporal_mesh.nodes[0],
        accepted_times,
        active,
        count,
        adaptive=False,
        source_plan_id=scheduled.schedule_id,
        requested_time_id=prepared.time_grid.time_id,
    )
    evidence = TemporalSolveEvidence(
        prepared.method.capabilities,
        native_differentiation_evidence(
            "adjoint:prescribed-grid-linear-solves",
            adaptive=False,
            checkpointing=_checkpointing_name(scheduled.replay),
            checkpoint_count=_checkpoint_count(scheduled.replay, count),
        ),
        equation_form="explicit-ode",
        backend_id="backend:phydrax:rosenbrock-w",
        configuration_id=scheduled.schedule_id,
        controller_id=f"controller:scheduled:{scheduled.source_realization_id}",
        event_id=None,
        adaptive=False,
        dense=False,
        maximum_steps=count,
        precision_evidence=prepared.precision.evidence_for(
            state, prepared.time_grid.times
        ),
    )
    return DifferentialSolution(
        times=lax.stop_gradient(prepared.time_grid.times),
        states=jax.vmap(prepared.precision.output)(states),
        valid=valid,
        backend_result=status,
        stats={
            "accepted_steps": jnp.asarray(count, dtype=jnp.int32),
            "accepted_step_sizes": scheduled.step_sizes,
            "replay_adequacy": adequacy,
            "linear_iterations": jnp.sum(adequacy.stage_iterations, dtype=jnp.int32),
            "numeric_version": jnp.asarray(scheduled.numeric_version, dtype=jnp.int32),
        },
        solver_name="RA34PW2",
        interpretation=prepared.problem.interpretation,
        state_geometry_id=prepared.problem.state_geometry_id,
        solver_id=prepared.method.method_id,
        resolved_method="RA34PW2:scheduled-frozen-grid",
        discretization_bundle=prepared.problem.discretization_bundle,
        backend_successful=adequacy.completed,
        temporal_evidence=evidence,
        temporal_mesh=mesh,
        problem_id=prepared.problem.problem_id,
    )


def refresh_rosenbrock_schedule(
    scheduled: ScheduledRosenbrockSolve,
    source: DifferentialSolution,
    /,
    *,
    reference_args: Any = _DEFAULT_ARGS,
    reference_initial_state: Any | None = None,
    replay: FixedStepReplayPolicy | None = None,
) -> ScheduledRosenbrockSolve:
    """Replace an inadequate schedule only after an explicit fresh source solve."""
    if not isinstance(scheduled, ScheduledRosenbrockSolve):
        raise TypeError("scheduled must be ScheduledRosenbrockSolve.")
    replay_ = scheduled.replay if replay is None else replay
    return _scheduled_from_source(
        scheduled.prepared,
        source,
        reference_args,
        reference_initial_state,
        replay_,
        scheduled.numeric_version + 1,
    )


__all__ = [
    "PreparedRosenbrockSolve",
    "RosenbrockReplayAdequacy",
    "RosenbrockReplayStatus",
    "ScheduledRosenbrockSolve",
    "prepare_rosenbrock",
    "refresh_rosenbrock_schedule",
    "schedule_rosenbrock",
    "solve_rosenbrock",
    "solve_scheduled_rosenbrock",
]
