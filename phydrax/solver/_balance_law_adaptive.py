#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum
from math import isfinite
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._temporal import RealizedTemporalMesh
from ..stochastic._realization import StochasticRealization
from ._balance_law import BalanceLawRuntimeState, PreparedBalanceLawRuntime


class BalanceLawAdaptiveStatus(IntEnum):
    SUCCESS = 0
    RETRY_LIMIT_REACHED = 1
    MINIMUM_STEP_REACHED = 2
    STEP_CAPACITY_REACHED = 3
    ATTEMPT_CAPACITY_REACHED = 4


class BalanceLawAdaptivePolicy(StrictModule, NonTrainableState):
    """Bounded adaptive controller for transactional balance-law intervals."""

    maximum_steps: int = eqx.field(static=True)
    maximum_retries: int = eqx.field(static=True)
    safety_factor: float = eqx.field(static=True)
    reduction_factor: float = eqx.field(static=True)
    growth_factor: float = eqx.field(static=True)
    minimum_step_size: float = eqx.field(static=True)
    maximum_step_size: float = eqx.field(static=True)
    attempt_capacity: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        maximum_steps: int,
        /,
        *,
        maximum_retries: int = 4,
        safety_factor: float = 0.95,
        reduction_factor: float = 0.5,
        growth_factor: float = 1.25,
        minimum_step_size: float = 1e-12,
        maximum_step_size: float = np.inf,
    ):
        steps = int(maximum_steps)
        retries = int(maximum_retries)
        safety = float(safety_factor)
        reduction = float(reduction_factor)
        growth = float(growth_factor)
        minimum = float(minimum_step_size)
        maximum = float(maximum_step_size)
        if (
            steps <= 0
            or retries < 0
            or not isfinite(safety)
            or not 0.0 < safety <= 1.0
            or not isfinite(reduction)
            or not 0.0 < reduction < 1.0
            or not isfinite(growth)
            or growth < 1.0
            or not isfinite(minimum)
            or minimum <= 0.0
            or np.isnan(maximum)
            or maximum <= 0.0
            or maximum < minimum
        ):
            raise ValueError("Balance-law adaptive policy is invalid.")
        self.maximum_steps = steps
        self.maximum_retries = retries
        self.safety_factor = safety
        self.reduction_factor = reduction
        self.growth_factor = growth
        self.minimum_step_size = minimum
        self.maximum_step_size = maximum
        self.attempt_capacity = steps * (retries + 1)
        self.policy_id = canonical_fingerprint(
            {
                "kind": "balance-law-adaptive-policy",
                "maximum_steps": steps,
                "maximum_retries": retries,
                "safety_factor": safety,
                "reduction_factor": reduction,
                "growth_factor": growth,
                "minimum_step_size": minimum,
                "maximum_step_size": None if np.isinf(maximum) else maximum,
            }
        )


class BalanceLawDecisionJournal(StrictModule):
    """Fixed-capacity evidence for every attempted adaptive interval.

    ``limiting_process_indices`` is ``-1`` for transport, nonnegative for a
    process in runtime order, and ``-2`` for unused journal slots.
    """

    attempted: Array
    accepted: Array
    start_times: Array
    end_times: Array
    requested_step_sizes: Array
    stable_step_limits: Array
    stability_margins: Array
    limiting_process_indices: Array
    retry_numbers: Array
    balance_statuses: Array
    attempt_count: Array
    accepted_count: Array
    reached_final_time: Array
    process_ids: tuple[str, ...] = eqx.field(static=True)
    source_plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        attempted: Array,
        accepted: Array,
        start_times: Array,
        end_times: Array,
        requested_step_sizes: Array,
        stable_step_limits: Array,
        stability_margins: Array,
        limiting_process_indices: Array,
        retry_numbers: Array,
        balance_statuses: Array,
        attempt_count: Array,
        accepted_count: Array,
        reached_final_time: Array,
        process_ids: tuple[str, ...],
        source_plan_id: str,
    ):
        attempted_ = jnp.asarray(attempted, dtype=bool)
        capacity = attempted_.size
        aligned = (
            accepted,
            start_times,
            end_times,
            requested_step_sizes,
            stable_step_limits,
            stability_margins,
            limiting_process_indices,
            retry_numbers,
            balance_statuses,
        )
        if attempted_.ndim != 1 or any(
            jnp.asarray(value).shape != (capacity,) for value in aligned
        ):
            raise ValueError(
                "Balance-law decision-journal arrays must be aligned rank one."
            )
        identifiers = tuple(str(identifier) for identifier in process_ids)
        plan_id = str(source_plan_id)
        if not plan_id or any(not identifier for identifier in identifiers):
            raise ValueError("Decision-journal provenance must be non-empty.")
        self.attempted = attempted_
        self.accepted = jnp.asarray(accepted, dtype=bool)
        self.start_times = jnp.asarray(start_times)
        self.end_times = jnp.asarray(end_times)
        self.requested_step_sizes = jnp.asarray(requested_step_sizes)
        self.stable_step_limits = jnp.asarray(stable_step_limits)
        self.stability_margins = jnp.asarray(stability_margins)
        self.limiting_process_indices = jnp.asarray(
            limiting_process_indices, dtype=jnp.int32
        )
        self.retry_numbers = jnp.asarray(retry_numbers, dtype=jnp.int32)
        self.balance_statuses = jnp.asarray(balance_statuses, dtype=jnp.int32)
        self.attempt_count = jnp.asarray(attempt_count, dtype=jnp.int32).reshape(())
        self.accepted_count = jnp.asarray(accepted_count, dtype=jnp.int32).reshape(())
        self.reached_final_time = jnp.asarray(reached_final_time, dtype=bool).reshape(())
        self.process_ids = identifiers
        self.source_plan_id = plan_id

    @property
    def capacity(self) -> int:
        return int(self.attempted.size)


class AdaptiveBalanceLawRolloutResult(StrictModule):
    final_state: BalanceLawRuntimeState
    realized_mesh: RealizedTemporalMesh
    journal: BalanceLawDecisionJournal
    completed: Array
    status: Array
    plan_id: str = eqx.field(static=True)


class AdaptiveBalanceLawRolloutPlan(StrictModule, NonTrainableState):
    """Realize a bounded adaptive balance-law schedule transactionally."""

    runtime: PreparedBalanceLawRuntime
    final_time: float = eqx.field(static=True)
    policy: BalanceLawAdaptivePolicy
    requested_time_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        runtime: PreparedBalanceLawRuntime,
        final_time: float,
        policy: BalanceLawAdaptivePolicy,
        /,
    ):
        if not isinstance(runtime, PreparedBalanceLawRuntime):
            raise TypeError("runtime must be PreparedBalanceLawRuntime.")
        if not isinstance(policy, BalanceLawAdaptivePolicy):
            raise TypeError("policy must be BalanceLawAdaptivePolicy.")
        target = float(final_time)
        if not isfinite(target):
            raise ValueError("Adaptive balance-law final_time must be finite.")
        requested_time_id = canonical_fingerprint(
            {"kind": "balance-law-requested-time", "final_time": target}
        )
        self.runtime = runtime
        self.final_time = target
        self.policy = policy
        self.requested_time_id = requested_time_id
        self.plan_id = canonical_fingerprint(
            {
                "kind": "adaptive-balance-law-rollout",
                "runtime": runtime.runtime_id,
                "final_time": target,
                "policy": policy.policy_id,
            }
        )

    def rollout(
        self,
        initial_state: BalanceLawRuntimeState,
        args: Any = None,
        realization: StochasticRealization | None = None,
        /,
    ) -> AdaptiveBalanceLawRolloutResult:
        if not isinstance(initial_state, BalanceLawRuntimeState):
            raise TypeError("initial_state must be BalanceLawRuntimeState.")
        if initial_state.process_ids != self.runtime.process_ids:
            raise ValueError("Balance-law runtime state process order changed.")

        initial_time = initial_state.time
        target = jnp.asarray(self.final_time, dtype=initial_time.dtype)
        initial_step = jnp.asarray(
            initial_state.transport_state.step_size, dtype=initial_time.dtype
        )
        initial_time = eqx.error_if(
            initial_time,
            ~jnp.isfinite(initial_time)
            | ~jnp.isfinite(initial_step)
            | (initial_step <= 0.0)
            | (target <= initial_time),
            "Adaptive balance-law rollout requires finite target > initial time and positive step.",
        )
        policy = self.policy
        maximum_step = jnp.asarray(policy.maximum_step_size, dtype=initial_time.dtype)
        minimum_step = jnp.asarray(policy.minimum_step_size, dtype=initial_time.dtype)
        first_step = jnp.minimum(initial_step, maximum_step)
        accepted_times = jnp.full(
            (policy.maximum_steps,), initial_time, dtype=initial_time.dtype
        )

        initial_carry = (
            initial_state,
            first_step,
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            accepted_times,
            jnp.asarray(False),
            jnp.asarray(False),
            jnp.asarray(
                BalanceLawAdaptiveStatus.ATTEMPT_CAPACITY_REACHED, dtype=jnp.int32
            ),
        )

        def step(carry, _):
            (
                state,
                next_step,
                retry_number,
                accepted_count,
                accepted_time_buffer,
                finished,
                failed,
                terminal_status,
            ) = carry
            active = ~finished & ~failed

            def execute(_):
                start = state.time
                remaining = target - start
                requested = jnp.minimum(jnp.minimum(next_step, maximum_step), remaining)
                end = start + requested
                result = self.runtime.advance_prescribed(
                    state, start, end, args, realization
                )
                all_limits = jnp.concatenate(
                    (result.transport.stable_step_size[None], result.process_step_limits)
                )
                limiting = jnp.argmin(all_limits).astype(jnp.int32) - 1
                stable_limit = jnp.min(all_limits)
                finite_limit = jnp.isfinite(stable_limit) & (stable_limit > 0.0)
                safe_limit = jnp.where(
                    finite_limit,
                    policy.safety_factor * stable_limit,
                    requested * policy.growth_factor,
                )
                margin = stable_limit / requested - 1.0
                accepted = result.accepted

                write_index = jnp.minimum(accepted_count, policy.maximum_steps - 1)
                next_buffer = jax.lax.cond(
                    accepted,
                    lambda value: value.at[write_index].set(result.runtime_state.time),
                    lambda value: value,
                    accepted_time_buffer,
                )
                next_count = accepted_count + accepted.astype(jnp.int32)
                tolerance = (
                    64.0 * jnp.finfo(start.dtype).eps * jnp.maximum(jnp.abs(target), 1.0)
                )
                next_finished = accepted & (
                    jnp.abs(result.runtime_state.time - target) <= tolerance
                )
                step_capacity_failed = (
                    accepted & ~next_finished & (next_count >= policy.maximum_steps)
                )

                reduced = jnp.minimum(requested * policy.reduction_factor, safe_limit)
                reduced = jnp.maximum(reduced, minimum_step)
                grown = jnp.minimum(requested * policy.growth_factor, safe_limit)
                grown = jnp.maximum(grown, minimum_step)
                selected_step = jnp.where(accepted, grown, reduced)
                next_retry = jnp.where(accepted, 0, retry_number + 1).astype(jnp.int32)
                minimum_failed = (
                    ~accepted & (requested <= minimum_step) & (remaining > minimum_step)
                )
                retry_failed = ~accepted & (next_retry > policy.maximum_retries)
                next_failed = step_capacity_failed | minimum_failed | retry_failed
                selected_status = jnp.where(
                    step_capacity_failed,
                    int(BalanceLawAdaptiveStatus.STEP_CAPACITY_REACHED),
                    jnp.where(
                        minimum_failed,
                        int(BalanceLawAdaptiveStatus.MINIMUM_STEP_REACHED),
                        jnp.where(
                            retry_failed,
                            int(BalanceLawAdaptiveStatus.RETRY_LIMIT_REACHED),
                            terminal_status,
                        ),
                    ),
                ).astype(jnp.int32)
                next_state = result.runtime_state
                next_carry = (
                    next_state,
                    selected_step,
                    next_retry,
                    next_count,
                    next_buffer,
                    finished | next_finished,
                    failed | next_failed,
                    selected_status,
                )
                output = (
                    jnp.asarray(True),
                    accepted,
                    start,
                    end,
                    requested,
                    stable_limit,
                    margin,
                    limiting,
                    retry_number,
                    result.status,
                )
                return next_carry, output

            def inactive(_):
                nan = jnp.asarray(jnp.nan, dtype=initial_time.dtype)
                output = (
                    jnp.asarray(False),
                    jnp.asarray(False),
                    nan,
                    nan,
                    nan,
                    nan,
                    nan,
                    jnp.asarray(-2, dtype=jnp.int32),
                    jnp.asarray(-1, dtype=jnp.int32),
                    jnp.asarray(-1, dtype=jnp.int32),
                )
                return carry, output

            return jax.lax.cond(active, execute, inactive, operand=None)

        final_carry, outputs = jax.lax.scan(
            step,
            initial_carry,
            jnp.arange(policy.attempt_capacity, dtype=jnp.int32),
        )
        (
            final_state,
            _,
            _,
            accepted_count,
            accepted_time_buffer,
            finished,
            failed,
            terminal_status,
        ) = final_carry
        (
            attempted,
            accepted,
            start_times,
            end_times,
            requested_steps,
            stable_limits,
            margins,
            limiting_processes,
            retry_numbers,
            balance_statuses,
        ) = outputs
        attempt_count = jnp.sum(attempted.astype(jnp.int32))
        status = jnp.where(
            finished,
            int(BalanceLawAdaptiveStatus.SUCCESS),
            jnp.where(
                failed,
                terminal_status,
                int(BalanceLawAdaptiveStatus.ATTEMPT_CAPACITY_REACHED),
            ),
        ).astype(jnp.int32)
        valid = jnp.arange(policy.maximum_steps, dtype=jnp.int32) < accepted_count
        realized = RealizedTemporalMesh(
            initial_time,
            accepted_time_buffer,
            valid,
            accepted_count,
            adaptive=True,
            source_plan_id=self.plan_id,
            requested_time_id=self.requested_time_id,
        )
        journal = BalanceLawDecisionJournal(
            attempted=attempted,
            accepted=accepted,
            start_times=start_times,
            end_times=end_times,
            requested_step_sizes=requested_steps,
            stable_step_limits=stable_limits,
            stability_margins=margins,
            limiting_process_indices=limiting_processes,
            retry_numbers=retry_numbers,
            balance_statuses=balance_statuses,
            attempt_count=attempt_count,
            accepted_count=accepted_count,
            reached_final_time=finished,
            process_ids=self.runtime.process_ids,
            source_plan_id=self.plan_id,
        )
        return AdaptiveBalanceLawRolloutResult(
            final_state=final_state,
            realized_mesh=realized,
            journal=journal,
            completed=finished,
            status=status,
            plan_id=self.plan_id,
        )


__all__ = [
    "AdaptiveBalanceLawRolloutPlan",
    "AdaptiveBalanceLawRolloutResult",
    "BalanceLawAdaptivePolicy",
    "BalanceLawAdaptiveStatus",
    "BalanceLawDecisionJournal",
]
