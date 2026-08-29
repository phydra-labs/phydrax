#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from .._execution_pool import refill_completed_tasks
from .._precision import PrecisionEvidenceEnvelope
from .._strict import StrictModule
from ..linalg import (
    DenseLinearOperator,
    DenseLU,
    LinearSolvePolicy,
    LinearSystem,
    solve as solve_linear,
)
from ._precision import NonlinearPrecisionPolicy
from ._types import NonlinearStatus


def _axis_norm(
    value: Array,
    axis: int,
    precision: NonlinearPrecisionPolicy,
    /,
) -> Array:
    accumulated = precision.accumulation(value)
    return precision.decision(
        jnp.sqrt(
            jnp.sum(
                jnp.real(jnp.conj(accumulated) * accumulated),
                axis=axis,
            )
        )
    )


class BatchedRootResult(StrictModule):
    state: Array
    residual: Array
    status: Array
    iterations: Array
    residual_evaluations: Array
    jacobian_evaluations: Array
    accepted_steps: Array
    residual_norm: Array
    precision_evidence: PrecisionEvidenceEnvelope = eqx.field(static=True)
    precision_policy_id: str = eqx.field(static=True)

    @property
    def successful(self):
        return self.status == int(NonlinearStatus.SUCCESS)


class RootPoolEvidence(StrictModule):
    """Stable task routing and lane-utilization evidence for one root pool."""

    task_count: int = eqx.field(static=True)
    lane_count: int = eqx.field(static=True)
    rounds: Array
    refills: Array
    active_lane_iterations: Array
    available_lane_iterations: Array
    completion_order: Array
    completion_lane: Array
    completion_round: Array

    @property
    def utilization(self) -> Array:
        return self.active_lane_iterations / jnp.maximum(
            self.available_lane_iterations,
            1,
        )


class PooledRootResult(StrictModule):
    """Input-ordered root results plus completion-pool evidence."""

    result: BatchedRootResult
    evidence: RootPoolEvidence

    @property
    def successful(self) -> Array:
        return self.result.successful


class SmallRootKernel(StrictModule):
    """Fused masked dense Newton kernel for batches of small systems."""

    residual: Callable[[Array, Any], Array]
    maximum_dimension: int = eqx.field(static=True)
    maximum_steps: int = eqx.field(static=True)
    absolute_tolerance: float = eqx.field(static=True)
    relative_tolerance: float = eqx.field(static=True)
    minimum_damping: float = eqx.field(static=True)
    precision: NonlinearPrecisionPolicy

    def __init__(
        self,
        residual: Callable[[Array, Any], Array],
        /,
        *,
        maximum_dimension: int = 16,
        maximum_steps: int = 16,
        absolute_tolerance: float = 1e-8,
        relative_tolerance: float = 1e-8,
        minimum_damping: float = 1e-4,
        precision: NonlinearPrecisionPolicy | None = None,
    ):
        if not callable(residual):
            raise TypeError("residual must be callable.")
        dimension = int(maximum_dimension)
        steps = int(maximum_steps)
        if dimension < 1 or steps < 1:
            raise ValueError("Small-root dimensions and steps must be positive.")
        precision_ = NonlinearPrecisionPolicy() if precision is None else precision
        if not isinstance(precision_, NonlinearPrecisionPolicy):
            raise TypeError("precision must be NonlinearPrecisionPolicy or None.")
        precision_.validate_tolerance(absolute_tolerance)
        self.residual = residual
        self.maximum_dimension = dimension
        self.maximum_steps = steps
        self.absolute_tolerance = float(absolute_tolerance)
        self.relative_tolerance = float(relative_tolerance)
        self.minimum_damping = float(minimum_damping)
        self.precision = precision_

    def solve(self, initial_states: Any, args: Any, /) -> BatchedRootResult:
        states = self.precision.state(jnp.asarray(initial_states))
        if states.ndim != 2:
            raise ValueError("initial_states must have shape (batch, dimension).")
        if states.shape[1] > self.maximum_dimension:
            raise ValueError("Small-root dimension exceeds maximum_dimension.")
        residual_function = self.residual
        evaluate = jax.vmap(residual_function)
        jacobian = jax.vmap(jax.jacfwd(residual_function))
        residuals = self.precision.residual(evaluate(states, args))
        if residuals.shape != states.shape:
            raise ValueError("Small-root residual shape must match state shape.")
        initial_norms = _axis_norm(residuals, 1, self.precision)
        thresholds = self.precision.decision(
            self.absolute_tolerance + self.relative_tolerance * initial_norms
        )
        active = jnp.isfinite(initial_norms) & (initial_norms > thresholds)
        iterations = jnp.zeros((states.shape[0],), dtype=jnp.int32)
        evaluations = jnp.ones_like(iterations)
        jacobian_evaluations = jnp.zeros_like(iterations)
        accepted_steps = jnp.zeros_like(iterations)

        class _Run(StrictModule):
            states: Array
            residuals: Array
            active: Array
            iterations: Array
            evaluations: Array
            jacobian_evaluations: Array
            accepted_steps: Array

        run = _Run(
            states,
            residuals,
            active,
            iterations,
            evaluations,
            jacobian_evaluations,
            accepted_steps,
        )

        def body(_, current):
            matrices = jacobian(current.states, args)
            regularized = (
                matrices
                + 1e-12 * jnp.eye(states.shape[1], dtype=states.dtype)[None, :, :]
            )
            directions = self.precision.direction(
                solve_linear(
                    LinearSystem(DenseLinearOperator(regularized)),
                    -current.residuals,
                    policy=self.precision.bind_linear(LinearSolvePolicy(DenseLU())),
                ).value
            )
            rates = jnp.asarray(
                [1.0, 0.5, 0.25, 0.125, 0.0625],
                dtype=states.dtype,
            )
            trial_states = jnp.asarray(
                current.states[None, :, :]
                + rates[:, None, None] * directions[None, :, :],
                dtype=states.dtype,
            )
            trial_residuals = self.precision.residual(
                jax.vmap(lambda candidates: evaluate(candidates, args))(trial_states)
            )
            trial_norms = _axis_norm(trial_residuals, 2, self.precision)
            trial_norms = jnp.where(
                jnp.isfinite(trial_norms),
                trial_norms,
                jnp.inf,
            )
            current_norms = _axis_norm(current.residuals, 1, self.precision)
            best_index = jnp.argmin(trial_norms, axis=0)
            selected_states = jnp.take_along_axis(
                trial_states,
                best_index[None, :, None],
                axis=0,
            )[0]
            selected_residuals = jnp.take_along_axis(
                trial_residuals,
                best_index[None, :, None],
                axis=0,
            )[0]
            selected_norms = jnp.take_along_axis(
                trial_norms,
                best_index[None, :],
                axis=0,
            )[0]
            accepted = current.active & (selected_norms < current_norms)
            next_states = jnp.where(
                current.active[:, None], selected_states, current.states
            )
            next_residuals = jnp.where(
                current.active[:, None], selected_residuals, current.residuals
            )
            norms = _axis_norm(next_residuals, 1, self.precision)
            next_active = current.active & accepted & (norms > thresholds)
            return _Run(
                next_states,
                next_residuals,
                next_active,
                current.iterations + current.active.astype(jnp.int32),
                current.evaluations + 5 * current.active.astype(jnp.int32),
                current.jacobian_evaluations + current.active.astype(jnp.int32),
                current.accepted_steps + accepted.astype(jnp.int32),
            )

        run = jax.lax.fori_loop(0, self.maximum_steps, body, run)
        norms = _axis_norm(run.residuals, 1, self.precision)
        finite = jnp.all(jnp.isfinite(run.states), axis=1) & jnp.all(
            jnp.isfinite(run.residuals), axis=1
        )
        status = jnp.where(
            finite & (norms <= thresholds),
            int(NonlinearStatus.SUCCESS),
            jnp.where(
                ~finite,
                int(NonlinearStatus.NONFINITE_EVALUATION),
                jnp.where(
                    run.active,
                    int(NonlinearStatus.MAXIMUM_STEPS_REACHED),
                    int(NonlinearStatus.RESIDUAL_STAGNATION),
                ),
            ),
        ).astype(jnp.int32)
        output_states = self.precision.output(run.states)
        return BatchedRootResult(
            state=output_states,
            residual=run.residuals,
            status=status,
            iterations=run.iterations,
            residual_evaluations=run.evaluations,
            jacobian_evaluations=run.jacobian_evaluations,
            accepted_steps=run.accepted_steps,
            residual_norm=norms,
            precision_evidence=self.precision.evidence_for(
                run.states,
                run.residuals,
                output_value=output_states,
            ),
            precision_policy_id=self.precision.policy_id,
        )

    def solve_pooled(
        self,
        initial_states: Any,
        args: Any,
        /,
        *,
        lane_count: int,
    ) -> PooledRootResult:
        """Solve an input-ordered task family through a fixed-width lane pool."""
        tasks = self.precision.state(jnp.asarray(initial_states))
        if tasks.ndim != 2:
            raise ValueError("initial_states must have shape (tasks, dimension).")
        task_count, dimension = (int(size) for size in tasks.shape)
        lanes = int(lane_count)
        if task_count < 1:
            raise ValueError("A root pool requires at least one task.")
        if lanes < 1:
            raise ValueError("lane_count must be positive.")
        if dimension > self.maximum_dimension:
            raise ValueError("Small-root dimension exceeds maximum_dimension.")
        lanes = min(lanes, task_count)

        def argument_axis(leaf):
            if not eqx.is_array(leaf):
                return None
            if leaf.ndim == 0:
                return None
            if leaf.shape[0] != task_count:
                raise ValueError(
                    "Every non-scalar root argument array must have one leading "
                    "task axis."
                )
            return 0

        argument_axes = jax.tree_util.tree_map(argument_axis, args)

        def take_arguments(task_ids):
            safe_ids = jnp.minimum(task_ids, task_count - 1)

            def take(leaf, axis):
                return leaf[safe_ids] if axis == 0 else leaf

            return jax.tree_util.tree_map(take, args, argument_axes)

        residual_function = self.residual
        evaluate = jax.vmap(residual_function, in_axes=(0, argument_axes))
        jacobian = jax.vmap(
            jax.jacfwd(residual_function),
            in_axes=(0, argument_axes),
        )
        task_ids = jnp.arange(task_count, dtype=jnp.int32)
        all_residuals = self.precision.residual(evaluate(tasks, take_arguments(task_ids)))
        if all_residuals.shape != tasks.shape:
            raise ValueError("Small-root residual shape must match state shape.")
        initial_norms = _axis_norm(all_residuals, 1, self.precision)
        all_thresholds = self.precision.decision(
            self.absolute_tolerance + self.relative_tolerance * initial_norms
        )

        initial_lane_ids = jnp.arange(lanes, dtype=jnp.int32)
        lane_states = tasks[initial_lane_ids]
        lane_residuals = all_residuals[initial_lane_ids]
        lane_thresholds = all_thresholds[initial_lane_ids]
        lane_norms = initial_norms[initial_lane_ids]
        lane_finite = jnp.all(jnp.isfinite(lane_states), axis=1) & jnp.all(
            jnp.isfinite(lane_residuals),
            axis=1,
        )
        lane_status = jnp.where(
            ~lane_finite,
            int(NonlinearStatus.NONFINITE_EVALUATION),
            jnp.where(
                lane_norms <= lane_thresholds,
                int(NonlinearStatus.SUCCESS),
                int(NonlinearStatus.ITERATING),
            ),
        ).astype(jnp.int32)

        class _PoolRun(StrictModule):
            task_ids: Array
            states: Array
            residuals: Array
            thresholds: Array
            status: Array
            iterations: Array
            evaluations: Array
            jacobian_evaluations: Array
            accepted_steps: Array
            output_states: Array
            output_residuals: Array
            output_status: Array
            output_iterations: Array
            output_evaluations: Array
            output_jacobians: Array
            output_accepted: Array
            output_norms: Array
            completion_order: Array
            completion_lane: Array
            completion_round: Array
            next_task: Array
            completed: Array
            round: Array
            refills: Array
            active_lane_iterations: Array
            available_lane_iterations: Array

        integer_lanes = jnp.zeros((lanes,), dtype=jnp.int32)
        integer_tasks = jnp.zeros((task_count,), dtype=jnp.int32)
        run = _PoolRun(
            task_ids=initial_lane_ids,
            states=lane_states,
            residuals=lane_residuals,
            thresholds=lane_thresholds,
            status=lane_status,
            iterations=integer_lanes,
            evaluations=jnp.ones_like(integer_lanes),
            jacobian_evaluations=integer_lanes,
            accepted_steps=integer_lanes,
            output_states=jnp.zeros_like(tasks),
            output_residuals=jnp.zeros_like(all_residuals),
            output_status=jnp.full(
                (task_count,),
                int(NonlinearStatus.MAXIMUM_STEPS_REACHED),
                dtype=jnp.int32,
            ),
            output_iterations=integer_tasks,
            output_evaluations=integer_tasks,
            output_jacobians=integer_tasks,
            output_accepted=integer_tasks,
            output_norms=jnp.full(
                (task_count,),
                jnp.inf,
                dtype=initial_norms.dtype,
            ),
            completion_order=jnp.full(
                (task_count,),
                task_count,
                dtype=jnp.int32,
            ),
            completion_lane=jnp.full(
                (task_count,),
                -1,
                dtype=jnp.int32,
            ),
            completion_round=jnp.full(
                (task_count,),
                -1,
                dtype=jnp.int32,
            ),
            next_task=jnp.asarray(lanes, dtype=jnp.int32),
            completed=jnp.asarray(0, dtype=jnp.int32),
            round=jnp.asarray(0, dtype=jnp.int32),
            refills=jnp.asarray(0, dtype=jnp.int32),
            active_lane_iterations=jnp.asarray(0, dtype=jnp.int32),
            available_lane_iterations=jnp.asarray(0, dtype=jnp.int32),
        )
        maximum_rounds = task_count * (self.maximum_steps + 2) + 1

        def condition(current):
            return (current.completed < task_count) & (current.round < maximum_rounds)

        def body(current):
            lane_active = current.task_ids < task_count
            terminal = lane_active & (current.status != int(NonlinearStatus.ITERATING))
            commit_ids = jnp.where(terminal, current.task_ids, task_count)
            output_norms = _axis_norm(current.residuals, 1, self.precision)
            output_states = current.output_states.at[commit_ids].set(
                current.states,
                mode="drop",
            )
            output_residuals = current.output_residuals.at[commit_ids].set(
                current.residuals,
                mode="drop",
            )
            output_status = current.output_status.at[commit_ids].set(
                current.status,
                mode="drop",
            )
            output_iterations = current.output_iterations.at[commit_ids].set(
                current.iterations,
                mode="drop",
            )
            output_evaluations = current.output_evaluations.at[commit_ids].set(
                current.evaluations,
                mode="drop",
            )
            output_jacobians = current.output_jacobians.at[commit_ids].set(
                current.jacobian_evaluations,
                mode="drop",
            )
            output_accepted = current.output_accepted.at[commit_ids].set(
                current.accepted_steps,
                mode="drop",
            )
            output_norm_values = current.output_norms.at[commit_ids].set(
                output_norms,
                mode="drop",
            )

            routing = refill_completed_tasks(
                current.task_ids,
                terminal,
                current.next_task,
                current.completed,
                task_count,
            )
            terminal_rank = routing.terminal_rank
            completion_positions = jnp.where(
                terminal,
                current.completed + terminal_rank,
                task_count,
            )
            completion_order = current.completion_order.at[completion_positions].set(
                current.task_ids,
                mode="drop",
            )
            completion_lane = current.completion_lane.at[completion_positions].set(
                jnp.arange(lanes, dtype=jnp.int32),
                mode="drop",
            )
            completion_round = current.completion_round.at[completion_positions].set(
                jnp.full((lanes,), current.round, dtype=jnp.int32),
                mode="drop",
            )

            refill = routing.refill_mask
            next_ids = routing.task_ids
            safe_ids = jnp.minimum(next_ids, task_count - 1)
            loaded_states = tasks[safe_ids]
            loaded_residuals = all_residuals[safe_ids]
            loaded_thresholds = all_thresholds[safe_ids]
            states = jnp.where(
                terminal[:, None],
                loaded_states,
                current.states,
            )
            residuals = jnp.where(
                terminal[:, None],
                loaded_residuals,
                current.residuals,
            )
            thresholds = jnp.where(
                terminal,
                loaded_thresholds,
                current.thresholds,
            )
            iterations = jnp.where(terminal, 0, current.iterations)
            evaluations = jnp.where(terminal, 1, current.evaluations)
            jacobian_evaluations = jnp.where(
                terminal,
                0,
                current.jacobian_evaluations,
            )
            accepted_steps = jnp.where(terminal, 0, current.accepted_steps)
            active = next_ids < task_count
            norms = _axis_norm(residuals, 1, self.precision)
            finite = jnp.all(jnp.isfinite(states), axis=1) & jnp.all(
                jnp.isfinite(residuals),
                axis=1,
            )
            status = jnp.where(
                ~active,
                int(NonlinearStatus.ITERATING),
                jnp.where(
                    ~finite,
                    int(NonlinearStatus.NONFINITE_EVALUATION),
                    jnp.where(
                        norms <= thresholds,
                        int(NonlinearStatus.SUCCESS),
                        int(NonlinearStatus.ITERATING),
                    ),
                ),
            ).astype(jnp.int32)
            step_mask = active & (status == int(NonlinearStatus.ITERATING))
            lane_args = take_arguments(next_ids)
            matrices = jacobian(states, lane_args)
            regularized = (
                matrices
                + 1e-12
                * jnp.eye(
                    dimension,
                    dtype=tasks.dtype,
                )[None, :, :]
            )
            directions = self.precision.direction(
                solve_linear(
                    LinearSystem(DenseLinearOperator(regularized)),
                    -residuals,
                    policy=self.precision.bind_linear(LinearSolvePolicy(DenseLU())),
                ).value
            )
            rates = jnp.asarray(
                [1.0, 0.5, 0.25, 0.125, 0.0625],
                dtype=tasks.dtype,
            )
            trial_states = jnp.asarray(
                states[None, :, :] + rates[:, None, None] * directions[None, :, :],
                dtype=tasks.dtype,
            )
            trial_residuals = self.precision.residual(
                jax.vmap(lambda candidates: evaluate(candidates, lane_args))(trial_states)
            )
            trial_norms = _axis_norm(trial_residuals, 2, self.precision)
            trial_norms = jnp.where(
                jnp.isfinite(trial_norms),
                trial_norms,
                jnp.inf,
            )
            current_norms = _axis_norm(residuals, 1, self.precision)
            best_index = jnp.argmin(trial_norms, axis=0)
            selected_states = jnp.take_along_axis(
                trial_states,
                best_index[None, :, None],
                axis=0,
            )[0]
            selected_residuals = jnp.take_along_axis(
                trial_residuals,
                best_index[None, :, None],
                axis=0,
            )[0]
            selected_norms = jnp.take_along_axis(
                trial_norms,
                best_index[None, :],
                axis=0,
            )[0]
            accepted = step_mask & (selected_norms < current_norms)
            next_states = jnp.where(
                accepted[:, None],
                selected_states,
                states,
            )
            next_residuals = jnp.where(
                accepted[:, None],
                selected_residuals,
                residuals,
            )
            next_iterations = iterations + step_mask.astype(jnp.int32)
            next_evaluations = evaluations + 5 * step_mask.astype(jnp.int32)
            next_jacobians = jacobian_evaluations + step_mask.astype(jnp.int32)
            next_accepted = accepted_steps + accepted.astype(jnp.int32)
            next_norms = _axis_norm(next_residuals, 1, self.precision)
            next_finite = jnp.all(jnp.isfinite(next_states), axis=1) & jnp.all(
                jnp.isfinite(next_residuals),
                axis=1,
            )
            next_status = jnp.where(
                ~active,
                int(NonlinearStatus.ITERATING),
                jnp.where(
                    ~next_finite,
                    int(NonlinearStatus.NONFINITE_EVALUATION),
                    jnp.where(
                        next_norms <= thresholds,
                        int(NonlinearStatus.SUCCESS),
                        jnp.where(
                            step_mask & ~accepted,
                            int(NonlinearStatus.RESIDUAL_STAGNATION),
                            jnp.where(
                                next_iterations >= self.maximum_steps,
                                int(NonlinearStatus.MAXIMUM_STEPS_REACHED),
                                int(NonlinearStatus.ITERATING),
                            ),
                        ),
                    ),
                ),
            ).astype(jnp.int32)
            refill_count = jnp.sum(refill, dtype=jnp.int32)
            return _PoolRun(
                task_ids=next_ids,
                states=next_states,
                residuals=next_residuals,
                thresholds=thresholds,
                status=next_status,
                iterations=next_iterations,
                evaluations=next_evaluations,
                jacobian_evaluations=next_jacobians,
                accepted_steps=next_accepted,
                output_states=output_states,
                output_residuals=output_residuals,
                output_status=output_status,
                output_iterations=output_iterations,
                output_evaluations=output_evaluations,
                output_jacobians=output_jacobians,
                output_accepted=output_accepted,
                output_norms=output_norm_values,
                completion_order=completion_order,
                completion_lane=completion_lane,
                completion_round=completion_round,
                next_task=routing.next_task,
                completed=routing.completed,
                round=current.round + 1,
                refills=current.refills + refill_count,
                active_lane_iterations=(
                    current.active_lane_iterations + jnp.sum(step_mask, dtype=jnp.int32)
                ),
                available_lane_iterations=(
                    current.available_lane_iterations + jnp.sum(active, dtype=jnp.int32)
                ),
            )

        run = jax.lax.while_loop(condition, body, run)
        checked_completed = eqx.error_if(
            run.completed,
            run.completed != task_count,
            "Root pool exhausted its finite execution bound.",
        )
        del checked_completed
        output_states = self.precision.output(run.output_states)
        result = BatchedRootResult(
            state=output_states,
            residual=run.output_residuals,
            status=run.output_status,
            iterations=run.output_iterations,
            residual_evaluations=run.output_evaluations,
            jacobian_evaluations=run.output_jacobians,
            accepted_steps=run.output_accepted,
            residual_norm=run.output_norms,
            precision_evidence=self.precision.evidence_for(
                run.output_states,
                run.output_residuals,
                output_value=output_states,
            ),
            precision_policy_id=self.precision.policy_id,
        )
        evidence = RootPoolEvidence(
            task_count=task_count,
            lane_count=lanes,
            rounds=run.round,
            refills=run.refills,
            active_lane_iterations=run.active_lane_iterations,
            available_lane_iterations=run.available_lane_iterations,
            completion_order=run.completion_order,
            completion_lane=run.completion_lane,
            completion_round=run.completion_round,
        )
        return PooledRootResult(result, evidence)


def batched_small_root(
    residual: Callable[[Array, Any], Array],
    initial_states: Any,
    args: Any,
    /,
    **kernel_options: Any,
) -> BatchedRootResult:
    return SmallRootKernel(residual, **kernel_options).solve(initial_states, args)


def pooled_small_root(
    residual: Callable[[Array, Any], Array],
    initial_states: Any,
    args: Any,
    /,
    *,
    lane_count: int,
    **kernel_options: Any,
) -> PooledRootResult:
    return SmallRootKernel(residual, **kernel_options).solve_pooled(
        initial_states,
        args,
        lane_count=lane_count,
    )


__all__ = [
    "BatchedRootResult",
    "PooledRootResult",
    "RootPoolEvidence",
    "SmallRootKernel",
    "batched_small_root",
    "pooled_small_root",
]
