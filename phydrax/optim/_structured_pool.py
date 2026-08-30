#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Sequence

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._execution_pool import PoolExecutionSignature
from .._strict import StrictModule
from ._iterative import OptimizationStatus, OptimizationTermination
from ._sparse_kkt import plan_sparse_augmented_kkt
from ._structured_ipm import (
    _residuals,
    advance_sparse_structured_ipm,
    finalize_sparse_structured_ipm,
    initialize_sparse_structured_ipm,
)
from ._structured_nonlinear import (
    PreparedStructuredNonlinearProgram,
    StructuredNonlinearResult,
    StructuredNonlinearWarmStart,
)


class StructuredPoolEvidence(StrictModule):
    """Stable task placement and utilization for one structured solve pool."""

    signature: PoolExecutionSignature

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


class PooledStructuredNonlinearResult(StrictModule):
    """Input-ordered structured results plus lane-completion evidence."""

    results: tuple[StructuredNonlinearResult, ...]
    evidence: StructuredPoolEvidence

    @property
    def successful(self) -> Array:
        return jnp.stack(tuple(result.successful for result in self.results))


def solve_pooled_structured_nonlinear(
    prepared: PreparedStructuredNonlinearProgram,
    initial_coordinates: ArrayLike,
    /,
    *,
    method: Any,
    termination: OptimizationTermination | None = None,
    lane_count: int,
    warm_starts: Sequence[StructuredNonlinearWarmStart | None] | None = None,
) -> PooledStructuredNonlinearResult:
    """Advance independent structured NLPs through a deterministic hot-refill pool."""
    from ._interior_point import PrimalDualInteriorPoint

    if not isinstance(prepared, PreparedStructuredNonlinearProgram):
        raise TypeError("prepared must be a PreparedStructuredNonlinearProgram.")
    if (
        not isinstance(method, PrimalDualInteriorPoint)
        or method.mode != "sparse-augmented"
    ):
        raise TypeError(
            "Pooled structured execution requires "
            "PrimalDualInteriorPoint(mode='sparse-augmented')."
        )
    termination_ = OptimizationTermination() if termination is None else termination
    if not isinstance(termination_, OptimizationTermination):
        raise TypeError("termination must be OptimizationTermination or None.")
    initial = jnp.asarray(initial_coordinates)
    if initial.ndim != 2 or initial.shape[1] != prepared.program.num_variables:
        raise ValueError(
            "initial_coordinates must have shape (tasks, program.num_variables)."
        )
    task_count = int(initial.shape[0])
    lanes = min(int(lane_count), task_count)
    if task_count < 1 or lanes < 1:
        raise ValueError("Structured pool task and lane counts must be positive.")
    starts = (None,) * task_count if warm_starts is None else tuple(warm_starts)
    if len(starts) != task_count:
        raise ValueError("warm_starts must identify every structured task.")
    if any(
        start is not None and start.structure_id != prepared.structure_id
        for start in starts
    ):
        raise ValueError("A pooled warm start has the wrong structure ID.")

    plan = plan_sparse_augmented_kkt(prepared.template)
    states = [
        initialize_sparse_structured_ipm(prepared, plan, initial[index], starts[index])
        for index in range(lanes)
    ]
    initial_norms = [_residuals(prepared, plan, state)[-1] for state in states]
    task_ids = list(range(lanes))
    results: list[StructuredNonlinearResult | None] = [None] * task_count
    completion_order = []
    completion_lane = []
    completion_round = []
    next_task = lanes
    completed = 0
    rounds = 0
    refills = 0
    active_lane_iterations = 0
    available_lane_iterations = 0
    maximum_rounds = task_count * (termination_.maximum_steps + 2) + 1

    while completed < task_count and rounds < maximum_rounds:
        for lane in range(lanes):
            task_id = task_ids[lane]
            if task_id >= task_count:
                continue
            state = states[lane]
            available_lane_iterations += 1
            if (
                int(state.status) == int(OptimizationStatus.ITERATING)
                and int(state.iteration) < termination_.maximum_steps
            ):
                state = advance_sparse_structured_ipm(
                    prepared,
                    plan,
                    state,
                    termination_,
                    method.structured_linear_policy,
                    fraction_to_boundary=method.fraction_to_boundary,
                    sufficient_decrease=method.sufficient_decrease,
                    maximum_line_search_steps=method.maximum_line_search_steps,
                    regularization=method.kkt_regularization,
                )
                active_lane_iterations += 1
            if (
                int(state.status) == int(OptimizationStatus.ITERATING)
                and int(state.iteration) >= termination_.maximum_steps
            ):
                state = eqx.tree_at(
                    lambda value: value.status,
                    state,
                    jnp.asarray(
                        int(OptimizationStatus.MAXIMUM_STEPS_REACHED),
                        dtype=jnp.int32,
                    ),
                )
            states[lane] = state
            if int(state.status) == int(OptimizationStatus.ITERATING):
                continue
            results[task_id] = finalize_sparse_structured_ipm(
                prepared,
                plan,
                state,
                termination=termination_,
                linear_policy=method.structured_linear_policy,
                method_id=method.method_id,
                initial_norm=initial_norms[lane],
            )
            completion_order.append(task_id)
            completion_lane.append(lane)
            completion_round.append(rounds)
            completed += 1
            if next_task < task_count:
                task_ids[lane] = next_task
                states[lane] = initialize_sparse_structured_ipm(
                    prepared,
                    plan,
                    initial[next_task],
                    starts[next_task],
                )
                initial_norms[lane] = _residuals(
                    prepared,
                    plan,
                    states[lane],
                )[-1]
                next_task += 1
                refills += 1
            else:
                task_ids[lane] = task_count
        rounds += 1
    if completed != task_count or any(result is None for result in results):
        raise RuntimeError("Structured pool exhausted its finite execution bound.")
    completed_results = tuple(
        result for result in results if isinstance(result, StructuredNonlinearResult)
    )
    evidence = StructuredPoolEvidence(
        task_count=task_count,
        lane_count=lanes,
        rounds=jnp.asarray(rounds, dtype=jnp.int32),
        refills=jnp.asarray(refills, dtype=jnp.int32),
        active_lane_iterations=jnp.asarray(active_lane_iterations, dtype=jnp.int32),
        available_lane_iterations=jnp.asarray(
            available_lane_iterations,
            dtype=jnp.int32,
        ),
        completion_order=jnp.asarray(completion_order, dtype=jnp.int32),
        completion_lane=jnp.asarray(completion_lane, dtype=jnp.int32),
        completion_round=jnp.asarray(completion_round, dtype=jnp.int32),
        signature=PoolExecutionSignature(
            topology_id=prepared.structure_id,
            method_id=method.method_id,
            precision_id=str(prepared.variable_lower.dtype),
            backend_id=method.structured_linear_policy.method.name,
            shard_count=1,
        ),
    )
    return PooledStructuredNonlinearResult(completed_results, evidence)


__all__ = [
    "PooledStructuredNonlinearResult",
    "StructuredPoolEvidence",
    "solve_pooled_structured_nonlinear",
]
