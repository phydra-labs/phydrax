#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from collections.abc import Callable, Sequence
from math import isfinite
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._hybrid_event import HybridReplayPolicy
from ._partitioned_coupling_graph import PreparedCoupling
from ._partitioned_coupling_runtime import advance_coupling_window
from ._partitioned_coupling_types import CouplingState, CouplingStatus
from ._segmented_execution import (
    FixedCapacitySegmentEvidence,
    FixedCapacitySegmentPolicy,
    FixedCapacitySegmentStep,
    run_fixed_capacity_segments,
)


AdaptiveCouplingRetention: TypeAlias = Literal["final", "windows"]


class AdaptiveCouplingWindowPolicy(StrictModule, NonTrainableState):
    """Reliable local-error PI control with bounded retry semantics."""

    initial_size: float = eqx.field(static=True)
    minimum_size: float = eqx.field(static=True)
    maximum_size: float = eqx.field(static=True)
    absolute_tolerance: float = eqx.field(static=True)
    relative_tolerance: float = eqx.field(static=True)
    safety: float = eqx.field(static=True)
    minimum_factor: float = eqx.field(static=True)
    maximum_factor: float = eqx.field(static=True)
    maximum_attempts: int = eqx.field(static=True)
    retryable_statuses: tuple[int, ...] = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        initial_size: float,
        minimum_size: float,
        maximum_size: float,
        /,
        *,
        absolute_tolerance: float,
        relative_tolerance: float,
        safety: float = 0.9,
        minimum_factor: float = 0.2,
        maximum_factor: float = 5.0,
        maximum_attempts: int = 8,
        retryable_statuses: Sequence[int | CouplingStatus] = (),
    ):
        initial = float(initial_size)
        minimum = float(minimum_size)
        maximum = float(maximum_size)
        absolute = float(absolute_tolerance)
        relative = float(relative_tolerance)
        safety_ = float(safety)
        minimum_factor_ = float(minimum_factor)
        maximum_factor_ = float(maximum_factor)
        attempts = int(maximum_attempts)
        statuses = tuple(int(value) for value in retryable_statuses)
        if (
            not all(
                isfinite(value)
                for value in (
                    initial,
                    minimum,
                    maximum,
                    absolute,
                    relative,
                    safety_,
                    minimum_factor_,
                    maximum_factor_,
                )
            )
            or minimum <= 0.0
            or not minimum <= initial <= maximum
            or absolute < 0.0
            or relative < 0.0
            or absolute + relative <= 0.0
            or safety_ <= 0.0
            or not 0.0 < minimum_factor_ <= 1.0
            or maximum_factor_ < 1.0
            or attempts < 1
            or len(set(statuses)) != len(statuses)
        ):
            raise ValueError("Adaptive coupling window policy is invalid.")
        self.initial_size = initial
        self.minimum_size = minimum
        self.maximum_size = maximum
        self.absolute_tolerance = absolute
        self.relative_tolerance = relative
        self.safety = safety_
        self.minimum_factor = minimum_factor_
        self.maximum_factor = maximum_factor_
        self.maximum_attempts = attempts
        self.retryable_statuses = statuses
        self.policy_id = canonical_fingerprint(
            {
                "kind": "adaptive-coupling-window-policy",
                "initial_size": initial,
                "minimum_size": minimum,
                "maximum_size": maximum,
                "absolute_tolerance": absolute,
                "relative_tolerance": relative,
                "safety": safety_,
                "minimum_factor": minimum_factor_,
                "maximum_factor": maximum_factor_,
                "maximum_attempts": attempts,
                "retryable_statuses": statuses,
            }
        )


class AdaptiveCouplingRolloutPlan(StrictModule, NonTrainableState):
    """Fixed segment/window/event capacities for one adaptive rollout epoch."""

    maximum_windows: int = eqx.field(static=True)
    segment_policy: FixedCapacitySegmentPolicy
    window_policy: AdaptiveCouplingWindowPolicy
    replay_policy: HybridReplayPolicy
    retention: AdaptiveCouplingRetention = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        maximum_windows: int,
        segment_policy: FixedCapacitySegmentPolicy,
        window_policy: AdaptiveCouplingWindowPolicy,
        replay_policy: HybridReplayPolicy,
        /,
        *,
        retention: AdaptiveCouplingRetention = "final",
    ):
        windows = int(maximum_windows)
        if windows < 1 or windows > segment_policy.maximum_segments:
            raise ValueError("maximum_windows must fit the DCD segment capacity.")
        if segment_policy.maximum_steps_per_segment < window_policy.maximum_attempts:
            raise ValueError("DCD step capacity must cover every window attempt.")
        if not isinstance(replay_policy, HybridReplayPolicy):
            raise TypeError("replay_policy must be HybridReplayPolicy.")
        if retention not in ("final", "windows"):
            raise ValueError("Unknown adaptive coupling retention policy.")
        self.maximum_windows = windows
        self.segment_policy = segment_policy
        self.window_policy = window_policy
        self.replay_policy = replay_policy
        self.retention = retention
        self.plan_id = canonical_fingerprint(
            {
                "kind": "adaptive-coupling-rollout",
                "maximum_windows": windows,
                "segments": segment_policy.policy_id,
                "window": window_policy.policy_id,
                "replay": replay_policy.policy_id,
                "retention": retention,
            }
        )


class _AdaptiveCouplingCarry(StrictModule):
    state: CouplingState
    window_size: Array
    previous_error_ratio: Array
    final_time: Array
    terminal_status: Array
    accepted_windows: Array


class AdaptiveCouplingSolution(StrictModule):
    final_state: CouplingState
    segment_evidence: FixedCapacitySegmentEvidence
    accepted_windows: Array
    terminal_status: Array
    successful: Array
    exact_final_time: Array
    plan_id: str = eqx.field(static=True)


class _AttemptCarry(StrictModule):
    accepted_state: CouplingState
    trial_size: Array
    next_size: Array
    previous_ratio: Array
    attempts: Array
    done: Array
    accepted: Array
    terminal_status: Array


def _select_state(
    predicate: Array, candidate: CouplingState, old: CouplingState
) -> CouplingState:
    return eqx.tree_at(
        lambda value: (
            value.participant_states,
            value.exchange_values,
            value.time,
            value.window_index,
        ),
        old,
        (
            jax.tree.map(
                lambda new, prior: jnp.where(predicate, new, prior),
                candidate.participant_states,
                old.participant_states,
            ),
            jax.tree.map(
                lambda new, prior: jnp.where(predicate, new, prior),
                candidate.exchange_values,
                old.exchange_values,
            ),
            jnp.where(predicate, candidate.time, old.time),
            jnp.where(predicate, candidate.window_index, old.window_index),
        ),
    )


def rollout_adaptive_coupling(
    prepared: PreparedCoupling,
    initial_state: CouplingState,
    final_time: ArrayLike,
    plan: AdaptiveCouplingRolloutPlan,
    /,
    *,
    args: Any = None,
) -> AdaptiveCouplingSolution:
    """Run transactional PI-controlled windows on the canonical DCD segment runner."""

    if not isinstance(prepared, PreparedCoupling):
        raise TypeError("prepared must be PreparedCoupling.")
    if not isinstance(plan, AdaptiveCouplingRolloutPlan):
        raise TypeError("plan must be AdaptiveCouplingRolloutPlan.")
    target = jnp.asarray(final_time, dtype=initial_state.time.dtype).reshape(())
    target = eqx.error_if(
        target,
        ~jnp.isfinite(target) | (target <= initial_state.time),
        "Adaptive coupling final_time must exceed the initial time.",
    )
    policy = plan.window_policy
    initial_carry = _AdaptiveCouplingCarry(
        initial_state,
        jnp.asarray(policy.initial_size, dtype=target.dtype),
        jnp.asarray(1.0, dtype=target.dtype),
        target,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
    )
    retryable = jnp.asarray(policy.retryable_statuses, dtype=jnp.int32)

    def advance_segment(
        carry: _AdaptiveCouplingCarry, segment_index: Array
    ) -> FixedCapacitySegmentStep[_AdaptiveCouplingCarry]:
        remaining = carry.final_time - carry.state.time
        trial = jnp.minimum(carry.window_size, remaining)
        attempts = _AttemptCarry(
            carry.state,
            trial,
            carry.window_size,
            carry.previous_error_ratio,
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(False),
            jnp.asarray(False),
            carry.terminal_status,
        )

        def attempt_body(_: int, attempt: _AttemptCarry) -> _AttemptCarry:
            active = ~attempt.done

            def execute(_: None):
                result = advance_coupling_window(
                    prepared, carry.state, attempt.trial_size, args
                )
                diagnostics = result.diagnostics
                scale = policy.absolute_tolerance + policy.relative_tolerance * (
                    diagnostics.participant_error_reference_norms
                )
                ratios = diagnostics.participant_error_norms / jnp.maximum(
                    scale, jnp.finfo(scale.dtype).tiny
                )
                reliable = jnp.all(diagnostics.participant_error_reliable)
                ratio = jnp.max(ratios, initial=0.0)
                error_accept = reliable & jnp.isfinite(ratio) & (ratio <= 1.0)
                accepted = result.successful & error_accept
                status_retryable = jnp.any(retryable == result.status)
                retry = (~accepted) & (
                    (result.successful & reliable & jnp.isfinite(ratio))
                    | status_retryable
                )
                hard_failure = (~accepted) & ~retry
                order = jnp.maximum(
                    jnp.min(diagnostics.participant_error_orders), 1
                ).astype(ratio.dtype)
                safe_ratio = jnp.maximum(
                    jnp.where(jnp.isfinite(ratio), ratio, 2.0),
                    jnp.finfo(ratio.dtype).tiny,
                )
                proportional = safe_ratio ** (-0.7 / (order + 1.0))
                integral = jnp.maximum(
                    attempt.previous_ratio, jnp.finfo(ratio.dtype).tiny
                ) ** (0.3 / (order + 1.0))
                factor = jnp.clip(
                    policy.safety * proportional * integral,
                    policy.minimum_factor,
                    policy.maximum_factor,
                )
                candidate_size = jnp.clip(
                    attempt.trial_size * factor,
                    policy.minimum_size,
                    policy.maximum_size,
                )
                retry_size = jnp.maximum(
                    policy.minimum_size,
                    jnp.minimum(candidate_size, attempt.trial_size * 0.9),
                )
                exhausted_at_minimum = retry & (
                    (attempt.trial_size <= policy.minimum_size)
                    & (retry_size >= attempt.trial_size)
                )
                done = accepted | hard_failure | exhausted_at_minimum
                terminal_status = jnp.where(
                    accepted,
                    int(CouplingStatus.SUCCESS),
                    jnp.where(
                        hard_failure,
                        result.status,
                        jnp.where(
                            exhausted_at_minimum,
                            int(CouplingStatus.CERTIFICATION_FAILURE),
                            attempt.terminal_status,
                        ),
                    ),
                ).astype(jnp.int32)
                accepted_state = _select_state(
                    accepted, result.accepted_state, attempt.accepted_state
                )
                return _AttemptCarry(
                    accepted_state,
                    jnp.where(retry, retry_size, attempt.trial_size),
                    jnp.where(accepted, candidate_size, attempt.next_size),
                    jnp.where(reliable, safe_ratio, attempt.previous_ratio),
                    attempt.attempts + 1,
                    done,
                    accepted,
                    terminal_status,
                )

            return jax.lax.cond(active, execute, lambda _: attempt, operand=None)

        attempts = jax.lax.fori_loop(0, policy.maximum_attempts, attempt_body, attempts)
        exhausted = ~attempts.done
        terminal_status = jnp.where(
            exhausted,
            int(CouplingStatus.WORK_EXHAUSTED),
            attempts.terminal_status,
        ).astype(jnp.int32)
        accepted_windows = carry.accepted_windows + attempts.accepted.astype(jnp.int32)
        reached_final = attempts.accepted & (
            attempts.accepted_state.time >= carry.final_time
        )
        failed = ~attempts.accepted
        terminal = (
            reached_final
            | failed
            | exhausted
            | (accepted_windows >= plan.maximum_windows)
        )
        terminal_status = jnp.where(
            (accepted_windows >= plan.maximum_windows) & ~reached_final,
            int(CouplingStatus.WORK_EXHAUSTED),
            terminal_status,
        )
        next_carry = _AdaptiveCouplingCarry(
            attempts.accepted_state,
            attempts.next_size,
            attempts.previous_ratio,
            carry.final_time,
            terminal_status,
            accepted_windows,
        )
        return FixedCapacitySegmentStep(
            next_carry,
            carry.state.time,
            attempts.accepted_state.time,
            attempts.attempts,
            0,
            terminal,
            terminal_status,
        )

    carry, evidence = run_fixed_capacity_segments(
        plan.segment_policy, initial_carry, advance_segment
    )
    exact_final = carry.state.time == carry.final_time
    successful = (
        evidence.successful
        & exact_final
        & (carry.terminal_status == int(CouplingStatus.SUCCESS))
    )
    return AdaptiveCouplingSolution(
        carry.state,
        evidence,
        carry.accepted_windows,
        carry.terminal_status,
        successful,
        exact_final,
        plan.plan_id,
    )


class CouplingTopologyRequest(StrictModule):
    """Numeric fixed-shape boundary request; it never mutates a live graph."""

    requested: Array
    participant_epoch_codes: Array
    waveform_required_samples: Array
    topology_code: Array
    status: Array

    def __init__(
        self,
        requested: ArrayLike,
        participant_epoch_codes: ArrayLike,
        waveform_required_samples: ArrayLike,
        topology_code: ArrayLike,
        status: ArrayLike = 0,
        /,
    ):
        requested_ = jnp.asarray(requested, dtype=bool).reshape(())
        participant = jnp.asarray(participant_epoch_codes, dtype=jnp.int32)
        capacities = jnp.asarray(waveform_required_samples, dtype=jnp.int32)
        topology = jnp.asarray(topology_code, dtype=jnp.int32).reshape(())
        status_ = jnp.asarray(status, dtype=jnp.int32).reshape(())
        if participant.ndim != 1 or capacities.shape != participant.shape:
            raise ValueError("Topology request arrays must have equal participant shape.")
        self.requested = requested_
        self.participant_epoch_codes = participant
        self.waveform_required_samples = capacities
        self.topology_code = topology
        self.status = status_


class CouplingEpochTransferResult(StrictModule):
    value: Any
    successful: Array


class AbstractCouplingEpochTransfer(StrictModule, NonTrainableState):
    transfer_id: str = eqx.field(static=True)

    @abc.abstractmethod
    def apply(self, value: Any, args: Any = None, /) -> CouplingEpochTransferResult:
        raise NotImplementedError


class IdentityCouplingEpochTransfer(AbstractCouplingEpochTransfer):
    transfer_id: str = eqx.field(static=True, default="coupling-epoch:identity")

    def apply(self, value: Any, args: Any = None, /) -> CouplingEpochTransferResult:
        del args
        return CouplingEpochTransferResult(value, jnp.asarray(True))


class CallableCouplingEpochTransfer(AbstractCouplingEpochTransfer):
    function: Callable[[Any, Any], CouplingEpochTransferResult]
    transfer_id: str = eqx.field(static=True)

    def __init__(
        self,
        function: Callable[[Any, Any], CouplingEpochTransferResult],
        /,
        *,
        transfer_id: str,
    ):
        if not callable(function) or not transfer_id:
            raise ValueError("Callable epoch transfer requires a function and ID.")
        self.function = function
        self.transfer_id = str(transfer_id)

    def apply(self, value: Any, args: Any = None, /) -> CouplingEpochTransferResult:
        result = self.function(value, args)
        if not isinstance(result, CouplingEpochTransferResult):
            raise TypeError("Epoch transfer must return CouplingEpochTransferResult.")
        return result


class PreparedCouplingEpoch(StrictModule, NonTrainableState):
    prepared_coupling: PreparedCoupling
    participant_epoch_ids: tuple[str, ...] = eqx.field(static=True)
    waveform_capacity_ids: tuple[str, ...] = eqx.field(static=True)
    epoch_id: str = eqx.field(static=True)

    def __init__(
        self,
        prepared_coupling: PreparedCoupling,
        participant_epoch_ids: Sequence[str],
        waveform_capacity_ids: Sequence[str],
        /,
    ):
        participant = tuple(str(value) for value in participant_epoch_ids)
        waveform = tuple(str(value) for value in waveform_capacity_ids)
        if len(participant) != len(prepared_coupling.subsystems):
            raise ValueError("One participant epoch ID is required per subsystem.")
        if not all(participant) or not all(waveform):
            raise ValueError("Coupling epoch IDs must be non-empty.")
        self.prepared_coupling = prepared_coupling
        self.participant_epoch_ids = participant
        self.waveform_capacity_ids = waveform
        self.epoch_id = canonical_fingerprint(
            {
                "kind": "prepared-coupling-epoch",
                "prepared": prepared_coupling.plan_id,
                "participants": participant,
                "waveforms": waveform,
            }
        )


class CouplingEpochTransitionPlan(StrictModule, NonTrainableState):
    participant_state_transfers: tuple[AbstractCouplingEpochTransfer, ...]
    exchange_transfers: tuple[AbstractCouplingEpochTransfer, ...]
    added_initializers: tuple[AbstractCouplingEpochTransfer, ...]
    removed_finalizers: tuple[AbstractCouplingEpochTransfer, ...]
    source_subsystem_ids: tuple[str, ...] = eqx.field(static=True)
    target_subsystem_ids: tuple[str, ...] = eqx.field(static=True)
    source_exchange_ids: tuple[str, ...] = eqx.field(static=True)
    target_exchange_ids: tuple[str, ...] = eqx.field(static=True)
    transition_id: str = eqx.field(static=True)

    def __init__(
        self,
        participant_state_transfers: Sequence[AbstractCouplingEpochTransfer],
        exchange_transfers: Sequence[AbstractCouplingEpochTransfer],
        added_initializers: Sequence[AbstractCouplingEpochTransfer],
        removed_finalizers: Sequence[AbstractCouplingEpochTransfer],
        /,
        *,
        source_subsystem_ids: Sequence[str],
        target_subsystem_ids: Sequence[str],
        source_exchange_ids: Sequence[str],
        target_exchange_ids: Sequence[str],
        transition_id: str,
    ):
        routes = (
            tuple(participant_state_transfers),
            tuple(exchange_transfers),
            tuple(added_initializers),
            tuple(removed_finalizers),
        )
        if any(
            not isinstance(value, AbstractCouplingEpochTransfer)
            for group in routes
            for value in group
        ):
            raise TypeError("Coupling epoch routes must be explicit epoch transfers.")
        identifier = str(transition_id)
        if not identifier:
            raise ValueError("transition_id must be non-empty.")
        self.participant_state_transfers = routes[0]
        self.exchange_transfers = routes[1]
        self.added_initializers = routes[2]
        self.removed_finalizers = routes[3]
        self.source_subsystem_ids = tuple(str(value) for value in source_subsystem_ids)
        self.target_subsystem_ids = tuple(str(value) for value in target_subsystem_ids)
        self.source_exchange_ids = tuple(str(value) for value in source_exchange_ids)
        self.target_exchange_ids = tuple(str(value) for value in target_exchange_ids)
        self.transition_id = identifier


class CouplingEpochTransitionResult(StrictModule, NonTrainableState):
    epoch: PreparedCouplingEpoch
    state: CouplingState
    successful: Array
    request: CouplingTopologyRequest
    transition_id: str = eqx.field(static=True)


def transition_coupling_epoch(
    current_epoch: PreparedCouplingEpoch,
    current_state: CouplingState,
    target_epoch: PreparedCouplingEpoch,
    transition: CouplingEpochTransitionPlan,
    request: CouplingTopologyRequest,
    /,
    *,
    accepted_window: bool,
    args: Any = None,
) -> CouplingEpochTransitionResult:
    """Apply every declared source-owned transfer, then atomically accept the epoch."""

    if not bool(np.asarray(request.requested)):
        return CouplingEpochTransitionResult(
            current_epoch,
            current_state,
            jnp.asarray(True),
            request,
            transition.transition_id,
        )
    if not accepted_window:
        return CouplingEpochTransitionResult(
            current_epoch,
            current_state,
            jnp.asarray(False),
            request,
            transition.transition_id,
        )
    source_subsystems = current_state.subsystem_ids
    target_subsystems = target_epoch.prepared_coupling.reference_state.subsystem_ids
    source_exchanges = current_state.exchange_ids
    target_exchanges = target_epoch.prepared_coupling.reference_state.exchange_ids
    if (
        source_subsystems != transition.source_subsystem_ids
        or target_subsystems != transition.target_subsystem_ids
        or source_exchanges != transition.source_exchange_ids
        or target_exchanges != transition.target_exchange_ids
    ):
        raise ValueError("Coupling epoch transition IDs do not match prepared graphs.")
    retained_subsystems = tuple(
        value for value in target_subsystems if value in source_subsystems
    )
    retained_exchanges = tuple(
        value for value in target_exchanges if value in source_exchanges
    )
    added_subsystems = tuple(
        value for value in target_subsystems if value not in source_subsystems
    )
    removed_subsystems = tuple(
        value for value in source_subsystems if value not in target_subsystems
    )
    if (
        len(transition.participant_state_transfers) != len(retained_subsystems)
        or len(transition.exchange_transfers) != len(retained_exchanges)
        or len(transition.added_initializers) != len(added_subsystems)
        or len(transition.removed_finalizers) != len(removed_subsystems)
    ):
        raise ValueError("Coupling epoch transition lacks an explicit transfer route.")
    source_state = dict(
        zip(source_subsystems, current_state.participant_states, strict=True)
    )
    source_values = dict(
        zip(source_exchanges, current_state.exchange_values, strict=True)
    )
    retained_state_transfers = dict(
        zip(retained_subsystems, transition.participant_state_transfers, strict=True)
    )
    retained_exchange_transfers = dict(
        zip(retained_exchanges, transition.exchange_transfers, strict=True)
    )
    initializers = dict(zip(added_subsystems, transition.added_initializers, strict=True))
    candidate_states: list[Any] = []
    successful = jnp.asarray(True)
    for subsystem_id in target_subsystems:
        if subsystem_id in source_state:
            result = retained_state_transfers[subsystem_id].apply(
                source_state[subsystem_id], args
            )
        else:
            result = initializers[subsystem_id].apply(None, args)
        candidate_states.append(result.value)
        successful = successful & result.successful
    candidate_values: list[Any] = []
    for exchange_id in target_exchanges:
        if exchange_id in source_values:
            result = retained_exchange_transfers[exchange_id].apply(
                source_values[exchange_id], args
            )
        else:
            reference_index = target_exchanges.index(exchange_id)
            result = CouplingEpochTransferResult(
                target_epoch.prepared_coupling.reference_state.exchange_values[
                    reference_index
                ],
                jnp.asarray(True),
            )
        candidate_values.append(result.value)
        successful = successful & result.successful
    for subsystem_id, finalizer in zip(
        removed_subsystems, transition.removed_finalizers, strict=True
    ):
        result = finalizer.apply(source_state[subsystem_id], args)
        successful = successful & result.successful
    candidate = CouplingState(
        tuple(candidate_states),
        tuple(candidate_values),
        current_state.time,
        current_state.window_index,
        subsystem_ids=target_subsystems,
        exchange_ids=target_exchanges,
    )
    if bool(np.asarray(successful)):
        return CouplingEpochTransitionResult(
            target_epoch,
            candidate,
            successful,
            request,
            transition.transition_id,
        )
    return CouplingEpochTransitionResult(
        current_epoch,
        current_state,
        successful,
        request,
        transition.transition_id,
    )


__all__ = [
    "AbstractCouplingEpochTransfer",
    "AdaptiveCouplingRolloutPlan",
    "AdaptiveCouplingSolution",
    "AdaptiveCouplingWindowPolicy",
    "CallableCouplingEpochTransfer",
    "CouplingEpochTransferResult",
    "CouplingEpochTransitionPlan",
    "CouplingEpochTransitionResult",
    "CouplingTopologyRequest",
    "IdentityCouplingEpochTransfer",
    "PreparedCouplingEpoch",
    "rollout_adaptive_coupling",
    "transition_coupling_epoch",
]
