#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from enum import IntEnum
from math import isfinite
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from .._numerics._checkpointed_scan import checkpointed_scan, CheckpointedScanMode
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..equations._mac_incompressible import CompiledMACIncompressibleDynamics
from ._fixed_step import AbstractFixedStepMethod, FixedStepResult


class MACAdaptiveStatus(IntEnum):
    SUCCESS = 0
    RETRY_LIMIT_REACHED = 1
    MINIMUM_STEP_REACHED = 2
    STEP_CAPACITY_REACHED = 3
    ATTEMPT_CAPACITY_REACHED = 4
    INVALID_RESTRICTION = 5
    STEP_FAILED = 6


def _stop_gradient_tree(value: Any, /) -> Any:
    return jax.tree.map(
        lambda leaf: jax.lax.stop_gradient(leaf) if eqx.is_array(leaf) else leaf,
        value,
    )


def _finite_state(state: Array, /) -> Array:
    return jnp.all(jnp.isfinite(state))


def _validate_step_result(result: FixedStepResult, reference: Array, /) -> None:
    if not isinstance(result, FixedStepResult):
        raise TypeError("MAC fixed-step method must return FixedStepResult.")
    if (
        not eqx.is_array(result.accepted_state)
        or result.accepted_state.shape != reference.shape
        or result.accepted_state.dtype != reference.dtype
    ):
        raise TypeError("MAC fixed-step accepted state changed shape or dtype.")
    if not eqx.is_array(result.successful) or result.successful.shape != ():
        raise TypeError("MAC fixed-step successful evidence must be scalar.")


class MACNamedRateLimit(StrictModule, NonTrainableState):
    """One named nonnegative inverse-time restriction."""

    evaluate: Callable[[Array, Array, Any], Array]
    name: str = eqx.field(static=True)
    scale: float = eqx.field(static=True)
    rate_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        evaluate: Callable[[Array, Array, Any], Array],
        /,
        *,
        scale: float = 1.0,
        rate_id: str,
    ):
        name_ = str(name)
        identifier = str(rate_id)
        scale_ = float(scale)
        if not name_ or not identifier:
            raise ValueError("MAC rate-limit name and rate_id must be non-empty.")
        if not callable(evaluate):
            raise TypeError("MAC rate-limit evaluate must be callable.")
        if not isfinite(scale_) or scale_ <= 0.0:
            raise ValueError("MAC rate-limit scale must be finite and positive.")
        self.evaluate = evaluate
        self.name = name_
        self.scale = scale_
        self.rate_id = canonical_fingerprint(
            {
                "kind": "mac-named-rate-limit",
                "name": name_,
                "scale": scale_,
                "source": identifier,
            }
        )


class MACCompositeStepRestriction(StrictModule):
    rates: Array
    step_limits: Array
    selected_rate: Array
    selected_step: Array
    limiting_index: Array
    valid: Array
    finite: Array
    names: tuple[str, ...] = eqx.field(static=True)
    controller_id: str = eqx.field(static=True)


class MACCompositeStepController(StrictModule, NonTrainableState):
    """Compose the built-in advective/diffusive limits with named rate limits."""

    dynamics: CompiledMACIncompressibleDynamics
    additional_limits: tuple[MACNamedRateLimit, ...]
    safety_factor: float = eqx.field(static=True)
    names: tuple[str, ...] = eqx.field(static=True)
    controller_id: str = eqx.field(static=True)

    def __init__(
        self,
        dynamics: CompiledMACIncompressibleDynamics,
        /,
        *,
        additional_limits: tuple[MACNamedRateLimit, ...] = (),
        safety_factor: float = 0.9,
    ):
        if not isinstance(dynamics, CompiledMACIncompressibleDynamics):
            raise TypeError("dynamics must be CompiledMACIncompressibleDynamics.")
        limits = tuple(additional_limits)
        if any(not isinstance(limit, MACNamedRateLimit) for limit in limits):
            raise TypeError("additional_limits must contain MACNamedRateLimit values.")
        safety = float(safety_factor)
        if not isfinite(safety) or not 0.0 < safety <= 1.0:
            raise ValueError("MAC controller safety_factor must lie in (0, 1].")
        names = ("advective", "diffusive") + tuple(limit.name for limit in limits)
        if len(set(names)) != len(names):
            raise ValueError("MAC composite rate-limit names must be unique.")
        self.dynamics = dynamics
        self.additional_limits = limits
        self.safety_factor = safety
        self.names = names
        self.controller_id = canonical_fingerprint(
            {
                "kind": "mac-composite-step-controller",
                "dynamics": dynamics.compilation_id,
                "safety_factor": safety,
                "rate_limits": [limit.rate_id for limit in limits],
            }
        )

    def restriction(
        self, time: Array, state: Array, args: Any = None, /
    ) -> MACCompositeStepRestriction:
        state_ = self.dynamics.validate_state(state)
        built_in = self.dynamics.step_restriction(state_)
        dtype = state_.dtype
        built_in_limits = jnp.stack(
            (
                jnp.asarray(built_in.advective, dtype=dtype),
                jnp.asarray(built_in.diffusive, dtype=dtype),
            )
        )
        built_in_rates = jnp.where(
            jnp.isinf(built_in_limits) & (built_in_limits > 0.0),
            0.0,
            1.0 / built_in_limits,
        )
        extra_rates = tuple(
            jnp.asarray(limit.evaluate(time, state_, args), dtype=dtype).reshape(())
            for limit in self.additional_limits
        )
        rates = (
            built_in_rates
            if not extra_rates
            else jnp.concatenate((built_in_rates, jnp.stack(extra_rates)))
        )
        scales = jnp.asarray(
            (1.0, 1.0) + tuple(limit.scale for limit in self.additional_limits),
            dtype=dtype,
        )
        valid_rates = jnp.isfinite(rates) & (rates >= 0.0)
        safe_rates = jnp.where(rates > 0.0, rates, 1.0)
        step_limits = jnp.where(rates > 0.0, scales / safe_rates, jnp.inf)
        limiting_index = jnp.argmin(step_limits).astype(jnp.int32)
        selected_step = jnp.min(step_limits)
        selected_rate = rates[limiting_index]
        valid = jnp.all(valid_rates) & (selected_step > 0.0)
        finite = jnp.all(jnp.isfinite(rates))
        return MACCompositeStepRestriction(
            rates,
            step_limits,
            selected_rate,
            selected_step,
            limiting_index,
            valid,
            finite,
            self.names,
            self.controller_id,
        )


class MACAdaptivePolicy(StrictModule, NonTrainableState):
    maximum_steps: int = eqx.field(static=True)
    maximum_retries: int = eqx.field(static=True)
    attempt_capacity: int = eqx.field(static=True)
    reduction_factor: float = eqx.field(static=True)
    growth_factor: float = eqx.field(static=True)
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
        minimum_step_size: float = 1e-12,
        maximum_step_size: float = np.inf,
    ):
        steps = int(maximum_steps)
        retries = int(maximum_retries)
        reduction = float(reduction_factor)
        growth = float(growth_factor)
        minimum = float(minimum_step_size)
        maximum = float(maximum_step_size)
        if (
            steps <= 0
            or retries < 0
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
            raise ValueError("MAC adaptive policy is invalid.")
        self.maximum_steps = steps
        self.maximum_retries = retries
        self.attempt_capacity = steps * (retries + 1)
        self.reduction_factor = reduction
        self.growth_factor = growth
        self.minimum_step_size = minimum
        self.maximum_step_size = maximum
        self.policy_id = canonical_fingerprint(
            {
                "kind": "mac-adaptive-policy",
                "maximum_steps": steps,
                "maximum_retries": retries,
                "reduction_factor": reduction,
                "growth_factor": growth,
                "minimum_step_size": minimum,
                "maximum_step_size": None if np.isinf(maximum) else maximum,
            }
        )


class MACAcceptedGridTrace(StrictModule):
    """Fixed-capacity accepted grid; valid steps are a contiguous prefix."""

    initial_time: Array
    times: Array
    step_sizes: Array
    valid_steps: Array
    accepted_step_count: Array
    reached_final_time: Array
    finite: Array
    source_plan_id: str = eqx.field(static=True)
    dynamics_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    @property
    def capacity(self) -> int:
        return int(self.step_sizes.size)


class MACAdaptiveAttemptJournal(StrictModule):
    attempted: Array
    accepted: Array
    start_times: Array
    requested_step_sizes: Array
    stable_step_limits: Array
    limiting_indices: Array
    retry_numbers: Array
    method_successful: Array
    restriction_valid: Array
    attempt_count: Array
    accepted_count: Array
    rate_names: tuple[str, ...] = eqx.field(static=True)
    source_plan_id: str = eqx.field(static=True)


class MACAdaptiveRuntimeState(StrictModule):
    """Complete immutable continuation state for adaptive MAC execution."""

    state: Array
    time: Array
    accepted_step_count: Array
    requested_next_step: Array
    status: Array
    retry_count: Array
    grid_times: Array
    grid_step_sizes: Array
    grid_valid_steps: Array
    output_cursor: Array
    forcing_state: Array
    dynamics_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)
    controller_id: str = eqx.field(static=True)


class MACAdaptiveAdvanceResult(StrictModule):
    rollout: "MACAdaptiveRolloutResult"
    runtime_state: MACAdaptiveRuntimeState


class MACAdaptiveRolloutResult(StrictModule):
    final_state: Array
    grid: MACAcceptedGridTrace
    journal: MACAdaptiveAttemptJournal
    successful: Array
    finite: Array
    status: Array
    plan_id: str = eqx.field(static=True)


class MACAdaptiveRolloutPlan(StrictModule, NonTrainableState):
    """Bounded transactional MAC adaptation with stopped controller decisions."""

    dynamics: CompiledMACIncompressibleDynamics
    method: AbstractFixedStepMethod
    controller: MACCompositeStepController
    policy: MACAdaptivePolicy
    final_time: float = eqx.field(static=True)
    initial_step_size: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        dynamics: CompiledMACIncompressibleDynamics,
        method: AbstractFixedStepMethod,
        controller: MACCompositeStepController,
        policy: MACAdaptivePolicy,
        /,
        *,
        final_time: float,
        initial_step_size: float,
    ):
        if not isinstance(dynamics, CompiledMACIncompressibleDynamics):
            raise TypeError("dynamics must be CompiledMACIncompressibleDynamics.")
        if not isinstance(method, AbstractFixedStepMethod):
            raise TypeError("method must be an AbstractFixedStepMethod.")
        if not isinstance(controller, MACCompositeStepController):
            raise TypeError("controller must be MACCompositeStepController.")
        if not isinstance(policy, MACAdaptivePolicy):
            raise TypeError("policy must be MACAdaptivePolicy.")
        if controller.dynamics.compilation_id != dynamics.compilation_id:
            raise ValueError("MAC controller and rollout dynamics differ.")
        target = float(final_time)
        initial_step = float(initial_step_size)
        if not isfinite(target) or not isfinite(initial_step) or initial_step <= 0.0:
            raise ValueError("MAC adaptive final time and initial step must be finite.")
        self.dynamics = dynamics
        self.method = method
        self.controller = controller
        self.policy = policy
        self.final_time = target
        self.initial_step_size = initial_step
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mac-adaptive-rollout",
                "dynamics": dynamics.compilation_id,
                "method": method.method_id,
                "controller": controller.controller_id,
                "policy": policy.policy_id,
                "final_time": target,
                "initial_step_size": initial_step,
            }
        )

    def initialize(
        self,
        initial_time: Array,
        initial_state: Array,
        /,
        *,
        forcing_state: Array | None = None,
    ) -> MACAdaptiveRuntimeState:
        state = self.dynamics.validate_state(initial_state)
        time = jnp.asarray(initial_time, dtype=state.dtype).reshape(())
        first_step = jnp.minimum(
            jnp.asarray(self.initial_step_size, dtype=state.dtype),
            jnp.asarray(self.policy.maximum_step_size, dtype=state.dtype),
        )
        forcing = (
            jnp.zeros((0,), dtype=state.dtype)
            if forcing_state is None
            else jnp.asarray(forcing_state)
        )
        times = jnp.full((self.policy.maximum_steps + 1,), time, dtype=state.dtype)
        return MACAdaptiveRuntimeState(
            state=state,
            time=time,
            accepted_step_count=jnp.asarray(0, dtype=jnp.int32),
            requested_next_step=first_step,
            status=jnp.asarray(
                int(MACAdaptiveStatus.ATTEMPT_CAPACITY_REACHED), dtype=jnp.int32
            ),
            retry_count=jnp.asarray(0, dtype=jnp.int32),
            grid_times=times,
            grid_step_sizes=jnp.zeros((self.policy.maximum_steps,), dtype=state.dtype),
            grid_valid_steps=jnp.zeros((self.policy.maximum_steps,), dtype=bool),
            output_cursor=jnp.asarray(0, dtype=jnp.int32),
            forcing_state=forcing,
            dynamics_id=self.dynamics.compilation_id,
            method_id=self.method.method_id,
            controller_id=self.controller.controller_id,
        )

    def advance(
        self,
        runtime_state: MACAdaptiveRuntimeState,
        final_time: Array,
        args: Any = None,
        /,
    ) -> MACAdaptiveAdvanceResult:
        """Continue from an explicit state without regenerating controller history."""
        if not isinstance(runtime_state, MACAdaptiveRuntimeState):
            raise TypeError("runtime_state must be MACAdaptiveRuntimeState.")
        if (
            runtime_state.dynamics_id != self.dynamics.compilation_id
            or runtime_state.method_id != self.method.method_id
            or runtime_state.controller_id != self.controller.controller_id
        ):
            raise ValueError("MAC runtime continuation identities do not match the plan.")
        target = float(np.asarray(final_time))
        requested = float(np.asarray(runtime_state.requested_next_step))
        segment_plan = MACAdaptiveRolloutPlan(
            self.dynamics,
            self.method,
            self.controller,
            self.policy,
            final_time=target,
            initial_step_size=requested,
        )
        result = segment_plan._rollout_segment(
            runtime_state.time, runtime_state.state, args
        )
        segment_count = result.grid.accepted_step_count
        capacity_left = self.policy.maximum_steps - runtime_state.accepted_step_count
        count = jnp.minimum(segment_count, capacity_left)

        def merge(index, buffers):
            times, steps, valid = buffers
            destination = runtime_state.accepted_step_count + index
            active = index < count
            times = jax.lax.cond(
                active,
                lambda value: value.at[destination + 1].set(result.grid.times[index + 1]),
                lambda value: value,
                times,
            )
            steps = jax.lax.cond(
                active,
                lambda value: value.at[destination].set(result.grid.step_sizes[index]),
                lambda value: value,
                steps,
            )
            valid = jax.lax.cond(
                active,
                lambda value: value.at[destination].set(True),
                lambda value: value,
                valid,
            )
            return times, steps, valid

        times, steps, valid = jax.lax.fori_loop(
            0,
            self.policy.maximum_steps,
            merge,
            (
                runtime_state.grid_times,
                runtime_state.grid_step_sizes,
                runtime_state.grid_valid_steps,
            ),
        )
        last_attempt = jnp.maximum(result.journal.attempt_count - 1, 0)
        last_requested = result.journal.requested_step_sizes[last_attempt]
        last_stable = result.journal.stable_step_limits[last_attempt]
        last_accepted = result.journal.accepted[last_attempt]
        minimum = jnp.asarray(self.policy.minimum_step_size, dtype=steps.dtype)
        safe_stable = jnp.where(jnp.isfinite(last_stable), last_stable, jnp.inf)
        grown = jnp.maximum(
            minimum,
            jnp.minimum(last_requested * self.policy.growth_factor, safe_stable),
        )
        reduced = jnp.maximum(
            minimum,
            jnp.minimum(last_requested * self.policy.reduction_factor, safe_stable),
        )
        next_step = jnp.where(last_accepted, grown, reduced)
        final_runtime = MACAdaptiveRuntimeState(
            state=result.final_state,
            time=result.grid.times[segment_count],
            accepted_step_count=runtime_state.accepted_step_count + count,
            requested_next_step=next_step,
            status=result.status,
            retry_count=jnp.where(
                last_accepted,
                0,
                result.journal.retry_numbers[last_attempt] + 1,
            ).astype(jnp.int32),
            grid_times=times,
            grid_step_sizes=steps,
            grid_valid_steps=valid,
            output_cursor=runtime_state.output_cursor + count,
            forcing_state=runtime_state.forcing_state,
            dynamics_id=runtime_state.dynamics_id,
            method_id=runtime_state.method_id,
            controller_id=runtime_state.controller_id,
        )
        return MACAdaptiveAdvanceResult(result, final_runtime)

    def rollout(
        self, initial_time: Array, initial_state: Array, args: Any = None, /
    ) -> MACAdaptiveRolloutResult:
        runtime = self.initialize(initial_time, initial_state)
        return self.advance(runtime, self.final_time, args).rollout

    def _rollout_segment(
        self, initial_time: Array, initial_state: Array, args: Any = None, /
    ) -> MACAdaptiveRolloutResult:
        state0 = self.dynamics.validate_state(initial_state)
        time0 = jnp.asarray(initial_time, dtype=state0.dtype).reshape(())
        target = jnp.asarray(self.final_time, dtype=state0.dtype)
        policy = self.policy
        minimum = jnp.asarray(policy.minimum_step_size, dtype=state0.dtype)
        maximum = jnp.asarray(policy.maximum_step_size, dtype=state0.dtype)
        first_step = jnp.minimum(
            jnp.asarray(self.initial_step_size, dtype=state0.dtype), maximum
        )
        initial_valid = (
            jnp.isfinite(time0)
            & (target > time0)
            & _finite_state(state0)
            & (first_step > 0.0)
        )
        times = jnp.full((policy.maximum_steps + 1,), time0, dtype=state0.dtype)
        steps = jnp.zeros((policy.maximum_steps,), dtype=state0.dtype)
        valid_steps = jnp.zeros((policy.maximum_steps,), dtype=bool)
        initial_status = jnp.where(
            initial_valid,
            int(MACAdaptiveStatus.ATTEMPT_CAPACITY_REACHED),
            int(MACAdaptiveStatus.STEP_FAILED),
        ).astype(jnp.int32)
        carry0 = (
            state0,
            time0,
            first_step,
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            times,
            steps,
            valid_steps,
            jnp.asarray(False),
            ~initial_valid,
            initial_status,
        )
        stopped_args = _stop_gradient_tree(args)

        def attempt(carry, attempt_index):
            (
                state,
                time,
                next_step,
                retry,
                accepted_count,
                time_buffer,
                step_buffer,
                valid_buffer,
                finished,
                failed,
                status,
            ) = carry
            active = ~finished & ~failed

            def execute(_):
                remaining = target - time
                requested = jnp.minimum(jnp.minimum(next_step, maximum), remaining)
                decision = self.controller.restriction(
                    jax.lax.stop_gradient(time),
                    jax.lax.stop_gradient(state),
                    stopped_args,
                )
                stable = self.controller.safety_factor * decision.selected_step
                tolerance = (
                    64.0 * jnp.finfo(state.dtype).eps * jnp.maximum(jnp.abs(stable), 1.0)
                )
                restriction_accepts = decision.valid & (
                    (requested <= stable + tolerance) | jnp.isinf(stable)
                )
                result = self.method.step(accepted_count, time, state, requested, args)
                _validate_step_result(result, state)
                method_ok = result.successful & _finite_state(result.accepted_state)
                accepted = jax.lax.stop_gradient(restriction_accepts & method_ok)
                write_index = jnp.minimum(accepted_count, policy.maximum_steps - 1)
                accepted_time = time + requested
                next_times = jax.lax.cond(
                    accepted,
                    lambda value: value.at[write_index + 1].set(accepted_time),
                    lambda value: value,
                    time_buffer,
                )
                next_steps = jax.lax.cond(
                    accepted,
                    lambda value: value.at[write_index].set(requested),
                    lambda value: value,
                    step_buffer,
                )
                next_valid = jax.lax.cond(
                    accepted,
                    lambda value: value.at[write_index].set(True),
                    lambda value: value,
                    valid_buffer,
                )
                count = accepted_count + accepted.astype(jnp.int32)
                next_time = jnp.where(accepted, accepted_time, time)
                next_state = jnp.where(accepted, result.accepted_state, state)
                time_tolerance = (
                    64.0 * jnp.finfo(state.dtype).eps * jnp.maximum(jnp.abs(target), 1.0)
                )
                reached = accepted & (jnp.abs(next_time - target) <= time_tolerance)
                capacity_failed = accepted & ~reached & (count >= policy.maximum_steps)
                invalid_failed = ~decision.valid
                next_retry = jnp.where(accepted, 0, retry + 1).astype(jnp.int32)
                retry_failed = (
                    ~accepted & method_ok & (next_retry > policy.maximum_retries)
                )
                method_failed = (
                    ~accepted & ~method_ok & (next_retry > policy.maximum_retries)
                )
                minimum_failed = (
                    ~accepted & (requested <= minimum) & (remaining > minimum)
                )
                next_failed = (
                    capacity_failed
                    | invalid_failed
                    | retry_failed
                    | method_failed
                    | minimum_failed
                )
                next_status = jnp.where(
                    invalid_failed,
                    int(MACAdaptiveStatus.INVALID_RESTRICTION),
                    jnp.where(
                        capacity_failed,
                        int(MACAdaptiveStatus.STEP_CAPACITY_REACHED),
                        jnp.where(
                            minimum_failed,
                            int(MACAdaptiveStatus.MINIMUM_STEP_REACHED),
                            jnp.where(
                                method_failed,
                                int(MACAdaptiveStatus.STEP_FAILED),
                                jnp.where(
                                    retry_failed,
                                    int(MACAdaptiveStatus.RETRY_LIMIT_REACHED),
                                    status,
                                ),
                            ),
                        ),
                    ),
                ).astype(jnp.int32)
                safe_stable = jnp.where(
                    decision.valid,
                    stable,
                    requested * policy.reduction_factor,
                )
                reduced = jnp.maximum(
                    minimum,
                    jnp.minimum(requested * policy.reduction_factor, safe_stable),
                )
                grown = jnp.maximum(
                    minimum,
                    jnp.minimum(requested * policy.growth_factor, safe_stable),
                )
                selected = jax.lax.stop_gradient(jnp.where(accepted, grown, reduced))
                next_carry = (
                    next_state,
                    next_time,
                    selected,
                    next_retry,
                    count,
                    next_times,
                    next_steps,
                    next_valid,
                    finished | reached,
                    failed | next_failed,
                    next_status,
                )
                output = (
                    jnp.asarray(True),
                    accepted,
                    time,
                    requested,
                    decision.selected_step,
                    decision.limiting_index,
                    retry,
                    method_ok,
                    decision.valid,
                )
                return next_carry, output

            def inactive(_):
                nan = jnp.asarray(jnp.nan, dtype=state0.dtype)
                output = (
                    jnp.asarray(False),
                    jnp.asarray(False),
                    nan,
                    nan,
                    nan,
                    jnp.asarray(-1, dtype=jnp.int32),
                    jnp.asarray(-1, dtype=jnp.int32),
                    jnp.asarray(False),
                    jnp.asarray(False),
                )
                return carry, output

            return jax.lax.cond(active, execute, inactive, operand=None)

        final_carry, journal_values = jax.lax.scan(
            attempt,
            carry0,
            jnp.arange(policy.attempt_capacity, dtype=jnp.int32),
        )
        (
            final_state,
            _,
            _,
            _,
            accepted_count,
            time_buffer,
            step_buffer,
            valid_buffer,
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
            limiting_indices,
            retries,
            method_successful,
            restriction_valid,
        ) = journal_values
        attempt_count = jnp.sum(attempted.astype(jnp.int32))
        status = jnp.where(
            finished,
            int(MACAdaptiveStatus.SUCCESS),
            jnp.where(
                failed,
                terminal_status,
                int(MACAdaptiveStatus.ATTEMPT_CAPACITY_REACHED),
            ),
        ).astype(jnp.int32)
        trace_finite = (
            jnp.all(jnp.isfinite(time_buffer))
            & jnp.all(jnp.where(valid_buffer, jnp.isfinite(step_buffer), True))
            & _finite_state(final_state)
        )
        successful = finished & trace_finite
        stopped_state = jax.lax.stop_gradient(final_state)
        grid = MACAcceptedGridTrace(
            jax.lax.stop_gradient(time0),
            jax.lax.stop_gradient(time_buffer),
            jax.lax.stop_gradient(step_buffer),
            jax.lax.stop_gradient(valid_buffer),
            jax.lax.stop_gradient(accepted_count),
            jax.lax.stop_gradient(finished),
            jax.lax.stop_gradient(trace_finite),
            self.plan_id,
            self.dynamics.compilation_id,
            self.method.method_id,
        )
        journal = MACAdaptiveAttemptJournal(
            jax.lax.stop_gradient(attempted),
            jax.lax.stop_gradient(accepted),
            jax.lax.stop_gradient(start_times),
            jax.lax.stop_gradient(requested_steps),
            jax.lax.stop_gradient(stable_limits),
            jax.lax.stop_gradient(limiting_indices),
            jax.lax.stop_gradient(retries),
            jax.lax.stop_gradient(method_successful),
            jax.lax.stop_gradient(restriction_valid),
            jax.lax.stop_gradient(attempt_count),
            jax.lax.stop_gradient(accepted_count),
            self.controller.names,
            self.plan_id,
        )
        return MACAdaptiveRolloutResult(
            stopped_state,
            grid,
            journal,
            jax.lax.stop_gradient(successful),
            jax.lax.stop_gradient(trace_finite),
            jax.lax.stop_gradient(status),
            self.plan_id,
        )


class MACFrozenGridReplayResult(StrictModule):
    final_state: Array
    states: Array
    step_successful: Array
    completed: Array
    finite: Array
    grid_valid: Array
    replay_id: str = eqx.field(static=True)
    source_plan_id: str = eqx.field(static=True)
    dynamics_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)


class MACFrozenGridReplayPlan(StrictModule, NonTrainableState):
    """Differentiable replay of a stopped, already accepted MAC grid."""

    dynamics: CompiledMACIncompressibleDynamics
    method: AbstractFixedStepMethod
    checkpointing: CheckpointedScanMode = eqx.field(static=True)
    block_size: int | None = eqx.field(static=True)
    replay_id: str = eqx.field(static=True)

    def __init__(
        self,
        dynamics: CompiledMACIncompressibleDynamics,
        method: AbstractFixedStepMethod,
        /,
        *,
        checkpointing: CheckpointedScanMode = "block",
        block_size: int | None = 16,
    ):
        if not isinstance(dynamics, CompiledMACIncompressibleDynamics):
            raise TypeError("dynamics must be CompiledMACIncompressibleDynamics.")
        if not isinstance(method, AbstractFixedStepMethod):
            raise TypeError("method must be an AbstractFixedStepMethod.")
        if checkpointing not in ("full", "step", "block"):
            raise ValueError("Unknown MAC replay checkpointing mode.")
        size = None if block_size is None else int(block_size)
        if checkpointing == "block":
            if size is None or size <= 0:
                raise ValueError("Block MAC replay requires positive block_size.")
        elif size is not None:
            raise ValueError("block_size is valid only for block MAC replay.")
        self.dynamics = dynamics
        self.method = method
        self.checkpointing = checkpointing
        self.block_size = size
        self.replay_id = canonical_fingerprint(
            {
                "kind": "mac-frozen-grid-replay",
                "dynamics": dynamics.compilation_id,
                "method": method.method_id,
                "checkpointing": checkpointing,
                "block_size": size,
            }
        )

    def replay(
        self,
        initial_state: Array,
        grid: MACAcceptedGridTrace,
        args: Any = None,
        /,
    ) -> MACFrozenGridReplayResult:
        if not isinstance(grid, MACAcceptedGridTrace):
            raise TypeError("grid must be MACAcceptedGridTrace.")
        if grid.dynamics_id != self.dynamics.compilation_id:
            raise ValueError("Frozen MAC grid dynamics identity changed.")
        if grid.method_id != self.method.method_id:
            raise ValueError("Frozen MAC grid method identity changed.")
        state0 = self.dynamics.validate_state(initial_state)
        count = grid.capacity
        steps = jax.lax.stop_gradient(grid.step_sizes)
        valid = jax.lax.stop_gradient(grid.valid_steps)
        times = jax.lax.stop_gradient(grid.times[:-1])
        indices = jnp.arange(count, dtype=jnp.int32)

        def advance(carry, inputs):
            state, prior_success = carry
            index, time, step_size, active = inputs

            def execute(_):
                result = self.method.step(index, time, state, step_size, args)
                _validate_step_result(result, state)
                successful = result.successful & _finite_state(result.accepted_state)
                next_state = jnp.where(successful, result.accepted_state, state)
                return (next_state, prior_success & successful), (
                    next_state,
                    successful,
                )

            def inactive(_):
                return (state, prior_success), (state, jnp.asarray(True))

            return jax.lax.cond(active & prior_success, execute, inactive, operand=None)

        (final_state, method_success), outputs = checkpointed_scan(
            advance,
            (state0, jnp.asarray(True)),
            (indices, times, steps, valid),
            length=count,
            mode=self.checkpointing,
            block_size=self.block_size,
        )
        states, step_success = outputs
        all_states = jnp.concatenate((state0[None, :], states), axis=0)
        expected_valid = indices < grid.accepted_step_count
        grid_valid = (
            grid.finite
            & grid.reached_final_time
            & (grid.accepted_step_count >= 0)
            & (grid.accepted_step_count <= count)
            & jnp.all(valid == expected_valid)
            & jnp.all(jnp.where(valid, steps > 0.0, steps == 0.0))
            & jnp.all(jnp.where(valid, grid.times[1:] > grid.times[:-1], True))
        )
        active_success = jnp.all(jnp.where(valid, step_success, True))
        finite = _finite_state(final_state) & jnp.all(
            jnp.where(valid[:, None], jnp.isfinite(states), True)
        )
        completed = grid_valid & method_success & active_success & finite
        return MACFrozenGridReplayResult(
            final_state,
            all_states,
            step_success,
            completed,
            finite,
            grid_valid,
            self.replay_id,
            grid.source_plan_id,
            self.dynamics.compilation_id,
            self.method.method_id,
        )


__all__ = [
    "MACAcceptedGridTrace",
    "MACAdaptiveAttemptJournal",
    "MACAdaptiveAdvanceResult",
    "MACAdaptivePolicy",
    "MACAdaptiveRolloutPlan",
    "MACAdaptiveRolloutResult",
    "MACAdaptiveRuntimeState",
    "MACAdaptiveStatus",
    "MACCompositeStepController",
    "MACCompositeStepRestriction",
    "MACFrozenGridReplayPlan",
    "MACFrozenGridReplayResult",
    "MACNamedRateLimit",
]
