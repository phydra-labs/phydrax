#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum
from math import isfinite

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import RealizedTemporalMesh
from ..discretization.mpm import MPMRuntimeState, PreparedMPMDynamics
from ..equations import MaterialPointArguments


class MPMAdaptiveStatus(IntEnum):
    SUCCESS = 0
    RETRY_LIMIT_REACHED = 1
    MINIMUM_STEP_REACHED = 2
    STEP_CAPACITY_REACHED = 3
    ATTEMPT_CAPACITY_REACHED = 4
    INVALID_INITIAL_STATE = 5


class MPMAdaptivePolicy(StrictModule, NonTrainableState):
    maximum_steps: int = eqx.field(static=True)
    maximum_retries: int = eqx.field(static=True)
    attempt_capacity: int = eqx.field(static=True)
    reduction_factor: float = eqx.field(static=True)
    growth_factor: float = eqx.field(static=True)
    safety_factor: float = eqx.field(static=True)
    minimum_step_size: float = eqx.field(static=True)
    maximum_step_size: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        maximum_steps: int,
        /,
        *,
        maximum_retries: int = 4,
        reduction_factor: float = 0.5,
        growth_factor: float = 1.5,
        safety_factor: float = 0.9,
        minimum_step_size: float = 1.0e-12,
        maximum_step_size: float = np.inf,
    ):
        steps = int(maximum_steps)
        retries = int(maximum_retries)
        reduction = float(reduction_factor)
        growth = float(growth_factor)
        safety = float(safety_factor)
        minimum = float(minimum_step_size)
        maximum = float(maximum_step_size)
        if (
            steps <= 0
            or retries < 0
            or not isfinite(reduction)
            or not 0.0 < reduction < 1.0
            or not isfinite(growth)
            or growth < 1.0
            or not isfinite(safety)
            or not 0.0 < safety <= 1.0
            or not isfinite(minimum)
            or minimum <= 0.0
            or np.isnan(maximum)
            or maximum < minimum
        ):
            raise ValueError("MPM adaptive policy is invalid.")
        self.maximum_steps = steps
        self.maximum_retries = retries
        self.attempt_capacity = steps * (retries + 1)
        self.reduction_factor = reduction
        self.growth_factor = growth
        self.safety_factor = safety
        self.minimum_step_size = minimum
        self.maximum_step_size = maximum
        self.policy_id = canonical_fingerprint(
            {
                "kind": "mpm-adaptive-policy",
                "maximum_steps": steps,
                "maximum_retries": retries,
                "reduction_factor": reduction,
                "growth_factor": growth,
                "safety_factor": safety,
                "minimum_step_size": minimum,
                "maximum_step_size": None if np.isinf(maximum) else maximum,
            }
        )


class MPMAdaptiveAttemptJournal(StrictModule):
    attempted: Array
    accepted: Array
    start_times: Array
    requested_step_sizes: Array
    stable_step_limits: Array
    suggested_step_sizes: Array
    limiting_processes: Array
    retry_numbers: Array
    statuses: Array
    rejection_reasons: Array
    route_digests: Array
    schedule_codes: Array
    topology_generations: Array
    attempt_count: Array
    accepted_count: Array
    reached_final_time: Array
    source_plan_id: str = eqx.field(static=True)


class AdaptiveMPMRolloutResult(StrictModule):
    final_state: MPMRuntimeState
    realized_mesh: RealizedTemporalMesh
    journal: MPMAdaptiveAttemptJournal
    completed: Array
    finite: Array
    status: Array
    plan_id: str = eqx.field(static=True)


class AdaptiveMPMRolloutPlan(StrictModule, NonTrainableState):
    """Bounded transactional explicit MPM adaptation with stopped decisions."""

    dynamics: PreparedMPMDynamics
    policy: MPMAdaptivePolicy
    final_time: float = eqx.field(static=True)
    initial_step_size: float = eqx.field(static=True)
    requested_time_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        dynamics: PreparedMPMDynamics,
        policy: MPMAdaptivePolicy,
        /,
        *,
        final_time: float,
        initial_step_size: float,
    ):
        if not isinstance(dynamics, PreparedMPMDynamics):
            raise TypeError("dynamics must be PreparedMPMDynamics.")
        if not isinstance(policy, MPMAdaptivePolicy):
            raise TypeError("policy must be MPMAdaptivePolicy.")
        target = float(final_time)
        initial = float(initial_step_size)
        if not isfinite(target) or not isfinite(initial) or initial <= 0.0:
            raise ValueError("Adaptive MPM final time and initial step must be finite.")
        self.dynamics = dynamics
        self.policy = policy
        self.final_time = target
        self.initial_step_size = initial
        self.requested_time_id = canonical_fingerprint(
            {"kind": "mpm-requested-time", "final_time": target}
        )
        self.plan_id = canonical_fingerprint(
            {
                "kind": "adaptive-mpm-rollout",
                "dynamics": dynamics.prepared_id,
                "policy": policy.policy_id,
                "final_time": target,
                "initial_step_size": initial,
            }
        )

    def rollout(
        self,
        initial_state: MPMRuntimeState,
        arguments: MaterialPointArguments,
        /,
    ) -> AdaptiveMPMRolloutResult:
        if not isinstance(initial_state, MPMRuntimeState):
            raise TypeError("initial_state must be MPMRuntimeState.")
        if not isinstance(arguments, MaterialPointArguments):
            raise TypeError("arguments must be MaterialPointArguments.")
        dtype = initial_state.time.dtype
        target = jnp.asarray(self.final_time, dtype=dtype)
        policy = self.policy
        minimum = jnp.asarray(policy.minimum_step_size, dtype=dtype)
        maximum = jnp.asarray(policy.maximum_step_size, dtype=dtype)
        first_step = jnp.minimum(
            jnp.asarray(self.initial_step_size, dtype=dtype), maximum
        )
        initial_valid = (
            jnp.isfinite(initial_state.time)
            & (target > initial_state.time)
            & jnp.isfinite(first_step)
            & (first_step > 0.0)
        )
        accepted_times = jnp.full(
            (policy.maximum_steps,), initial_state.time, dtype=dtype
        )
        carry0 = (
            initial_state,
            first_step,
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            accepted_times,
            jnp.asarray(False),
            ~initial_valid,
            jnp.asarray(int(MPMAdaptiveStatus.INVALID_INITIAL_STATE), dtype=jnp.int32),
        )
        stopped_arguments = jax.tree.map(
            lambda leaf: jax.lax.stop_gradient(leaf) if eqx.is_array(leaf) else leaf,
            arguments,
            is_leaf=lambda value: value is None,
        )

        def attempt(carry, _):
            (
                state,
                next_step,
                retry,
                accepted_count,
                time_buffer,
                finished,
                failed,
                terminal_status,
            ) = carry
            active = ~finished & ~failed

            def execute(_):
                start = state.time
                remaining = target - start
                requested = jnp.minimum(jnp.minimum(next_step, maximum), remaining)
                detail = self.dynamics.step_detailed(
                    state, jax.lax.stop_gradient(requested), stopped_arguments
                )
                accepted = detail.successful
                stable_limit = detail.restriction.selected
                suggested = jnp.minimum(
                    detail.suggested_step,
                    policy.safety_factor * stable_limit,
                )
                suggested = jnp.where(
                    jnp.isfinite(suggested) & (suggested > 0.0),
                    suggested,
                    requested * policy.reduction_factor,
                )
                write_index = jnp.minimum(accepted_count, policy.maximum_steps - 1)
                next_buffer = jax.lax.cond(
                    accepted,
                    lambda value: value.at[write_index].set(detail.accepted_state.time),
                    lambda value: value,
                    time_buffer,
                )
                next_count = accepted_count + accepted.astype(jnp.int32)
                tolerance = (
                    64.0 * jnp.finfo(dtype).eps * jnp.maximum(jnp.abs(target), 1.0)
                )
                next_finished = accepted & (
                    jnp.abs(detail.accepted_state.time - target) <= tolerance
                )
                step_capacity_failed = (
                    accepted & ~next_finished & (next_count >= policy.maximum_steps)
                )
                next_retry = jnp.where(accepted, 0, retry + 1).astype(jnp.int32)
                minimum_failed = (
                    ~accepted & (requested <= minimum) & (remaining > minimum)
                )
                retry_failed = ~accepted & (next_retry > policy.maximum_retries)
                next_failed = step_capacity_failed | minimum_failed | retry_failed
                reduced = jnp.maximum(
                    minimum,
                    jnp.minimum(requested * policy.reduction_factor, suggested),
                )
                grown = jnp.maximum(
                    minimum,
                    jnp.minimum(requested * policy.growth_factor, suggested),
                )
                selected_next = jax.lax.stop_gradient(jnp.where(accepted, grown, reduced))
                selected_status = jnp.where(
                    step_capacity_failed,
                    int(MPMAdaptiveStatus.STEP_CAPACITY_REACHED),
                    jnp.where(
                        minimum_failed,
                        int(MPMAdaptiveStatus.MINIMUM_STEP_REACHED),
                        jnp.where(
                            retry_failed,
                            int(MPMAdaptiveStatus.RETRY_LIMIT_REACHED),
                            terminal_status,
                        ),
                    ),
                ).astype(jnp.int32)
                next_carry = (
                    detail.accepted_state,
                    selected_next,
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
                    requested,
                    stable_limit,
                    detail.suggested_step,
                    detail.restriction.limiting_process,
                    retry,
                    detail.accepted_state.last_status,
                    detail.rejection_reasons,
                    detail.diagnostics.transfer.route_digest,
                    detail.diagnostics.schedule.schedule_code,
                    detail.accepted_state.topology_generation,
                )
                return next_carry, output

            def inactive(_):
                nan = jnp.asarray(jnp.nan, dtype=dtype)
                output = (
                    jnp.asarray(False),
                    jnp.asarray(False),
                    nan,
                    nan,
                    nan,
                    nan,
                    jnp.asarray(-1, dtype=jnp.int32),
                    jnp.asarray(-1, dtype=jnp.int32),
                    jnp.asarray(-1, dtype=jnp.int32),
                    jnp.asarray(0, dtype=jnp.int32),
                    jnp.asarray(0, dtype=jnp.int64),
                    jnp.asarray(-1, dtype=jnp.int32),
                    state.topology_generation,
                )
                return carry, output

            return jax.lax.cond(active, execute, inactive, operand=None)

        final_carry, outputs = jax.lax.scan(
            attempt,
            carry0,
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
            requested_steps,
            stable_limits,
            suggested_steps,
            limiting_processes,
            retry_numbers,
            statuses,
            rejection_reasons,
            route_digests,
            schedule_codes,
            topology_generations,
        ) = outputs
        status = jnp.where(
            finished,
            int(MPMAdaptiveStatus.SUCCESS),
            jnp.where(
                failed,
                terminal_status,
                int(MPMAdaptiveStatus.ATTEMPT_CAPACITY_REACHED),
            ),
        ).astype(jnp.int32)
        valid = jnp.arange(policy.maximum_steps, dtype=jnp.int32) < accepted_count
        realized = RealizedTemporalMesh(
            initial_state.time,
            accepted_time_buffer,
            valid,
            accepted_count,
            adaptive=True,
            source_plan_id=self.plan_id,
            requested_time_id=self.requested_time_id,
        )
        journal = MPMAdaptiveAttemptJournal(
            attempted,
            accepted,
            start_times,
            requested_steps,
            stable_limits,
            suggested_steps,
            limiting_processes,
            retry_numbers,
            statuses,
            rejection_reasons,
            route_digests,
            schedule_codes,
            topology_generations,
            jnp.sum(attempted.astype(jnp.int32)),
            accepted_count,
            finished,
            self.plan_id,
        )
        return AdaptiveMPMRolloutResult(
            final_state,
            realized,
            journal,
            finished,
            ~failed & jnp.isfinite(final_state.time),
            status,
            self.plan_id,
        )


__all__ = [
    "AdaptiveMPMRolloutPlan",
    "AdaptiveMPMRolloutResult",
    "MPMAdaptiveAttemptJournal",
    "MPMAdaptivePolicy",
    "MPMAdaptiveStatus",
]
