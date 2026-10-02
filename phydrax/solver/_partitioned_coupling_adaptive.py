#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from collections.abc import Callable, Mapping, Sequence
from math import isfinite
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import (
    array_tree_fingerprint,
    array_tree_signature,
    canonical_fingerprint,
)
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..lifecycle import (
    Composition,
    CompositionDependency,
    CompositionEntry,
    CompositionRole,
    CompositionTransport,
)
from ..typing import checked, parse
from ._hybrid_event import HybridReplayPolicy
from ._partitioned_coupling_graph import PreparedCoupling
from ._partitioned_coupling_runtime import (
    advance_coupling_window,
    coupling_counts_complete,
)
from ._partitioned_coupling_types import (
    CouplingPort,
    CouplingState,
    CouplingStatus,
    CouplingWindowResult,
)
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
    ) -> None:
        initial = float(initial_size)
        minimum = float(minimum_size)
        maximum = float(maximum_size)
        absolute = float(absolute_tolerance)
        relative = float(relative_tolerance)
        safety_ = float(safety)
        minimum_factor_ = float(minimum_factor)
        maximum_factor_ = float(maximum_factor)
        attempts = int(maximum_attempts)
        statuses = tuple(retryable_statuses)
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

    @checked
    def __init__(
        self,
        maximum_windows: int,
        segment_policy: FixedCapacitySegmentPolicy,
        window_policy: AdaptiveCouplingWindowPolicy,
        replay_policy: HybridReplayPolicy,
        /,
        *,
        retention: AdaptiveCouplingRetention = "final",
    ) -> None:
        windows = int(maximum_windows)
        if windows < 1 or windows > segment_policy.maximum_segments:
            raise ValueError("maximum_windows must fit the DCD segment capacity.")
        if segment_policy.maximum_steps_per_segment < window_policy.maximum_attempts:
            raise ValueError("DCD step capacity must cover every window attempt.")
        retention = parse(retention, AdaptiveCouplingRetention, "retention")
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


class _AttemptWork(StrictModule):
    """Work of every executed attempt, rejected or accepted."""

    attempts: Array
    rejected: Array
    participant_work: Array
    participant_evaluations: Array
    coupling_iterations: Array
    maximum_rejected_error_ratio: Array


class _AdaptiveCouplingCarry(StrictModule):
    state: CouplingState
    window_size: Array
    previous_error_ratio: Array
    final_time: Array
    terminal_status: Array
    accepted_windows: Array
    work: _AttemptWork


class AdaptiveCouplingSolution(StrictModule):
    """Adaptive rollout result with work evidence of every executed attempt.

    `participant_work`, `participant_evaluations`, and `coupling_iterations` sum
    all executed window attempts, including rejected ones that restarted the same
    checkpoint; `rejected_attempts` counts those replays and
    `maximum_rejected_error_ratio` retains the largest reliable rejected local
    error ratio. `counts_complete` is true when every window's participant work
    evidence is exact: explicit sweeps and fixed-point interface iterations whose
    participants report complete counts. General root interface methods report
    only the final evaluation's work, so their rollouts stay incomplete.
    """

    final_state: CouplingState
    segment_evidence: FixedCapacitySegmentEvidence
    accepted_windows: Array
    terminal_status: Array
    successful: Array
    exact_final_time: Array
    attempted_windows: Array
    rejected_attempts: Array
    participant_work: Array
    participant_evaluations: Array
    coupling_iterations: Array
    maximum_rejected_error_ratio: Array
    counts_complete: bool = eqx.field(static=True)
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
    work: _AttemptWork


def _select_state(
    predicate: Array, candidate: CouplingState, old: CouplingState
) -> CouplingState:
    return eqx.tree_at(
        lambda value: (
            value.participant_states,
            value.exchange_values,
            value.time,
            value.window_index,
            value.cumulative_exchange_budget,
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
            jnp.where(
                predicate,
                candidate.cumulative_exchange_budget,
                old.cumulative_exchange_budget,
            ),
        ),
    )


def _initial_work(prepared: PreparedCoupling, dtype: Any, /) -> _AttemptWork:
    count = len(prepared.subsystems)
    return _AttemptWork(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.zeros((count,), dtype=jnp.int32),
        jnp.zeros((count,), dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0, dtype=dtype),
    )


def _record_attempt(
    work: _AttemptWork,
    result: CouplingWindowResult,
    accepted: Array,
    reliable_ratio: Array,
    /,
) -> _AttemptWork:
    """Add one executed attempt; rejected replays keep their full work."""
    diagnostics = result.diagnostics
    rejected_ratio = jnp.where(
        ~accepted & jnp.isfinite(reliable_ratio), reliable_ratio, 0.0
    ).astype(work.maximum_rejected_error_ratio.dtype)
    return _AttemptWork(
        work.attempts + 1,
        work.rejected + (~accepted).astype(jnp.int32),
        work.participant_work + diagnostics.participant_work,
        work.participant_evaluations + diagnostics.participant_evaluations,
        work.coupling_iterations + diagnostics.coupling_iterations,
        jnp.maximum(work.maximum_rejected_error_ratio, rejected_ratio),
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
    """Run transactional PI-controlled windows on the canonical DCD segment runner.

    Every rejected attempt replays the same accepted checkpoint, so every
    participant must declare deterministic replay.
    """

    if not isinstance(prepared, PreparedCoupling):
        raise TypeError("prepared must be PreparedCoupling.")
    if not isinstance(plan, AdaptiveCouplingRolloutPlan):
        raise TypeError("plan must be AdaptiveCouplingRolloutPlan.")
    nondeterministic = sorted(
        subsystem.subsystem_id
        for subsystem in prepared.subsystems
        if not subsystem.capabilities.deterministic_replay
    )
    if nondeterministic:
        raise ValueError(
            "Adaptive window retries replay the accepted checkpoint and require "
            "deterministic replay; refused participants: " + ", ".join(nondeterministic)
        )
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
        _initial_work(prepared, target.dtype),
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
            carry.work,
        )

        def attempt_body(_: int, attempt: _AttemptCarry) -> _AttemptCarry:
            active = ~attempt.done

            def execute(_: None) -> _AttemptCarry:
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
                # A successful window fails hard only when its acceptance lacks a
                # reliable finite error estimate; its own SUCCESS status would lie.
                failure_status = jnp.where(
                    result.successful,
                    int(CouplingStatus.UNRELIABLE_ERROR_ESTIMATE),
                    result.status,
                )
                terminal_status = jnp.where(
                    accepted,
                    int(CouplingStatus.SUCCESS),
                    jnp.where(
                        hard_failure,
                        failure_status,
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
                    _record_attempt(
                        attempt.work,
                        result,
                        accepted,
                        jnp.where(reliable, ratio, jnp.nan),
                    ),
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
            attempts.work,
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
    work = carry.work
    return AdaptiveCouplingSolution(
        carry.state,
        evidence,
        carry.accepted_windows,
        carry.terminal_status,
        successful,
        exact_final,
        work.attempts,
        work.rejected,
        work.participant_work,
        work.participant_evaluations,
        work.coupling_iterations,
        work.maximum_rejected_error_ratio,
        coupling_counts_complete(prepared),
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
    ) -> None:
        requested_ = jnp.asarray(requested, dtype=jnp.bool_).reshape(())
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
    ) -> None:
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
    participant_epoch_codes: tuple[int, ...] = eqx.field(static=True)
    waveform_required_samples: tuple[int, ...] = eqx.field(static=True)
    topology_code: int = eqx.field(static=True)
    epoch_id: str = eqx.field(static=True)

    def __init__(
        self,
        prepared_coupling: PreparedCoupling,
        participant_epoch_ids: Sequence[str],
        waveform_capacity_ids: Sequence[str],
        /,
        *,
        participant_epoch_codes: Sequence[int],
        waveform_required_samples: Sequence[int],
        topology_code: int,
    ) -> None:
        participant = tuple(str(value) for value in participant_epoch_ids)
        waveform = tuple(str(value) for value in waveform_capacity_ids)
        participant_codes = tuple(participant_epoch_codes)
        waveform_samples = tuple(waveform_required_samples)
        if (
            len(participant_codes) != len(participant)
            or len(waveform_samples) != len(participant)
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in (*participant_codes, *waveform_samples)
            )
            or isinstance(topology_code, bool)
            or not isinstance(topology_code, int)
            or topology_code < 0
        ):
            raise ValueError("Coupling epoch numeric request contract is invalid.")
        if len(participant) != len(prepared_coupling.subsystems):
            raise ValueError("One participant epoch ID is required per subsystem.")
        if not all(participant) or not all(waveform):
            raise ValueError("Coupling epoch IDs must be non-empty.")
        self.prepared_coupling = prepared_coupling
        self.participant_epoch_ids = participant
        self.waveform_capacity_ids = waveform
        self.participant_epoch_codes = participant_codes
        self.waveform_required_samples = waveform_samples
        self.topology_code = topology_code
        self.epoch_id = canonical_fingerprint(
            {
                "kind": "prepared-coupling-epoch",
                "prepared": prepared_coupling.plan_id,
                "participants": participant,
                "waveforms": waveform,
                "participant_codes": participant_codes,
                "waveform_required_samples": waveform_samples,
                "topology_code": topology_code,
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
    ) -> None:
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


def _port_budget_contract(port: CouplingPort, /) -> tuple[Any, ...]:
    measurement = port.measurement
    return (
        None if port.quantity is None else port.quantity.compatibility_id,
        port.temporal_kind,
        port.frame,
        None
        if measurement is None
        else (
            measurement.unit.dimension.dimension_id,
            measurement.unit.reference_system_id,
            measurement.representation,
            measurement.component_ids,
        ),
    )


def _exchange_budget_contracts(
    epoch: PreparedCouplingEpoch, /
) -> dict[str, tuple[Any, ...]]:
    prepared = epoch.prepared_coupling
    ports = {
        port.port_id: port
        for subsystem in prepared.subsystems
        for port in (*subsystem.input_ports, *subsystem.output_ports)
    }
    clock = None if prepared.time_unit is None else prepared.time_unit.unit_id
    return {
        exchange.exchange_id: (
            None if exchange.temporal is None else exchange.temporal.kind,
            clock,
            _port_budget_contract(ports[exchange.source_port_id]),
            _port_budget_contract(ports[exchange.target_port_id]),
        )
        for exchange in prepared.exchanges
    }


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

    if current_state.graph_id != current_epoch.prepared_coupling.graph_id:
        raise ValueError(
            "Coupling state physical graph identity does not match its source epoch."
        )

    if not bool(np.asarray(request.requested)):
        return CouplingEpochTransitionResult(
            current_epoch,
            current_state,
            jnp.asarray(True),
            request,
            transition.transition_id,
        )
    request_matches = (
        int(np.asarray(request.status)) == int(CouplingStatus.SUCCESS)
        and int(np.asarray(request.topology_code)) == target_epoch.topology_code
        and np.array_equal(
            np.asarray(request.participant_epoch_codes),
            np.asarray(target_epoch.participant_epoch_codes, dtype=np.int32),
        )
        and np.array_equal(
            np.asarray(request.waveform_required_samples),
            np.asarray(target_epoch.waveform_required_samples, dtype=np.int32),
        )
    )
    if not request_matches:
        return CouplingEpochTransitionResult(
            current_epoch,
            current_state,
            jnp.asarray(False),
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
    # Epoch transfers may change coordinates and measurement weights, but cannot
    # silently relabel an accepted inventory as a different quantity, frame,
    # component inventory, storage representation, or temporal meaning.
    source_contracts = _exchange_budget_contracts(current_epoch)
    target_contracts = _exchange_budget_contracts(target_epoch)
    for exchange_id in retained_exchanges:
        if source_contracts[exchange_id] != target_contracts[exchange_id]:
            raise ValueError(
                "Epoch transfer cannot change a retained exchange's physical budget contract."
            )
    target_rows = target_epoch.prepared_coupling.reference_state.budget_row_ids
    source_budget = dict(
        zip(
            current_state.budget_row_ids,
            current_state.cumulative_exchange_budget,
            strict=True,
        )
    )
    for row_id, row in source_budget.items():
        if row_id not in target_rows and np.any(np.asarray(row) != 0):
            raise ValueError(
                "An epoch transition cannot discard accepted physical budgets."
            )
    candidate_budget = jnp.stack(
        tuple(
            source_budget[row_id]
            if row_id in source_budget
            else jnp.zeros((2,), dtype=current_state.time.dtype)
            for row_id in target_rows
        )
    )
    candidate = CouplingState(
        tuple(candidate_states),
        tuple(candidate_values),
        current_state.time,
        current_state.window_index,
        subsystem_ids=target_subsystems,
        exchange_ids=target_exchanges,
        cumulative_exchange_budget=candidate_budget,
        budget_row_ids=target_rows,
        graph_id=target_epoch.prepared_coupling.graph_id,
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


_EPOCH_ENTRY = "coupling/epoch"
_CLOCK_ENTRY = "coupling/clock"
_BUDGET_ENTRY = "coupling/budget"


def _participant_entry_ids(subsystem_id: str, /) -> dict[str, str]:
    return {
        "native": f"{subsystem_id}/native",
        "model-state": f"{subsystem_id}/model-state",
        "rng": f"{subsystem_id}/rng",
        "history": f"{subsystem_id}/windows",
    }


def _exchange_entry_id(exchange_id: str, /) -> str:
    return f"exchange/{exchange_id}"


def _layout(kind: str, owner: str, value: Any, /, **identity: Any) -> str:
    return canonical_fingerprint(
        {
            "kind": kind,
            "owner": owner,
            "signature": array_tree_signature(value),
            **identity,
        }
    )


def _boundary(epoch: PreparedCouplingEpoch, state: CouplingState, /) -> str:
    # One host read of the accepted boundary clock; compositions are staged there.
    return canonical_fingerprint(
        {
            "kind": "coupling-boundary",
            "epoch": epoch.epoch_id,
            "time": np.asarray(state.time).item(),
            "window": np.asarray(state.window_index).item(),
        }
    )


def _revision(boundary: str, entry_id: str, value: Any, /) -> str:
    # Distinct states staged at one boundary (a numeric refresh) are distinct
    # revisions; the value fingerprint is one host read at staging.
    return canonical_fingerprint(
        {
            "kind": "coupling-boundary-revision",
            "boundary": boundary,
            "entry": entry_id,
            "value": array_tree_fingerprint(value),
        }
    )


def _participant_entries(
    epoch: PreparedCouplingEpoch,
    state: CouplingState,
    boundary: str,
    index: int,
    dependencies: Sequence[CompositionDependency],
    independent: bool,
    /,
) -> tuple[CompositionEntry, ...]:
    """Native state, model state, carried key, and window history of one participant."""
    from .coupling._method_participants import MethodParticipantState

    subsystem = epoch.prepared_coupling.subsystems[index]
    sid = subsystem.subsystem_id
    bundle = subsystem.discretization_bundle_id
    if bundle is None:
        raise ValueError(
            f"Participant {sid!r} needs a declared discretization_bundle_id to take "
            "part in a composition; structure identity is never inferred."
        )
    ids = _participant_entry_ids(sid)
    checkpoint = state.participant_states[index]
    native = (
        checkpoint.native
        if isinstance(checkpoint, MethodParticipantState)
        else checkpoint
    )

    def entry(
        value: Any,
        name: str,
        role: CompositionRole,
        structure: str,
        bindings: Sequence[CompositionDependency],
    ) -> CompositionEntry:
        return CompositionEntry(
            value,
            entry_id=ids[name],
            role=role,
            owner_id=sid,
            structure_id=structure,
            revision_id=_revision(boundary, ids[name], value),
            semantics_id=f"{sid}:{name}",
            dependencies=bindings,
        )

    native_entry = entry(
        native,
        "native",
        "physical-state",
        bundle,
        dependencies,
    )
    entries = [native_entry]
    if not isinstance(checkpoint, MethodParticipantState):
        return tuple(entries)
    # Model state and carried keys bind the native structure unless their owner
    # declared them independent of the discretization.
    bound = () if independent else (native_entry.binding("structure"),)
    if checkpoint.model_state is not None:
        layout = _layout("participant-model-state", sid, checkpoint.model_state)
        entries.append(
            entry(checkpoint.model_state, "model-state", "model-state", layout, bound)
        )
    if checkpoint.key_data is not None:
        layout = _layout("participant-key", sid, checkpoint.key_data)
        entries.append(entry(checkpoint.key_data, "rng", "rng", layout, bound))
    history = (checkpoint.accepted_windows, checkpoint.native_steps)
    windows = _layout("participant-windows", sid, history)
    entries.append(entry(history, "history", "history", windows, ()))
    return tuple(entries)


def _budget_contract_id(epoch: PreparedCouplingEpoch, /) -> str:
    contracts = _exchange_budget_contracts(epoch)
    return canonical_fingerprint(
        {
            "kind": "coupling-budget-contract",
            "rows": list(epoch.prepared_coupling.reference_state.budget_row_ids),
            "exchanges": [[key, repr(contracts[key])] for key in sorted(contracts)],
        }
    )


def _exchange_entries(
    epoch: PreparedCouplingEpoch, state: CouplingState, boundary: str, /
) -> tuple[CompositionEntry, ...]:
    prepared = epoch.prepared_coupling
    ports = {
        port.port_id: port
        for subsystem in prepared.subsystems
        for port in subsystem.input_ports
    }
    contracts = _exchange_budget_contracts(epoch)
    entries = []
    for exchange, value in zip(prepared.exchanges, state.exchange_values, strict=True):
        port = ports[exchange.target_port_id]
        entry_id = _exchange_entry_id(exchange.exchange_id)
        entries.append(
            CompositionEntry(
                value,
                entry_id=entry_id,
                role="exchange-state",
                owner_id=exchange.exchange_id,
                structure_id=_layout(
                    "coupling-exchange-value",
                    exchange.exchange_id,
                    value,
                    port=port.port_id,
                    field=None
                    if port.field_space is None
                    else port.field_space.field_space_id,
                    waveform=None
                    if port.waveform_plan is None
                    else port.waveform_plan.plan_id,
                ),
                revision_id=_revision(boundary, entry_id, value),
                semantics_id=canonical_fingerprint(
                    {
                        "kind": "coupling-exchange-semantics",
                        "exchange": exchange.exchange_id,
                        "contract": repr(contracts[exchange.exchange_id]),
                    }
                ),
            )
        )
    return tuple(entries)


def _ledger_entries(
    epoch: PreparedCouplingEpoch, state: CouplingState, boundary: str, /
) -> tuple[CompositionEntry, CompositionEntry]:
    rows = state.budget_row_ids
    budget = CompositionEntry(
        state.cumulative_exchange_budget,
        entry_id=_BUDGET_ENTRY,
        role="budget",
        owner_id="coupling",
        structure_id=canonical_fingerprint(
            {"kind": "coupling-budget-rows", "rows": list(rows)}
        ),
        revision_id=_revision(boundary, _BUDGET_ENTRY, state.cumulative_exchange_budget),
        semantics_id=_budget_contract_id(epoch),
    )
    clock_value = (state.time, state.window_index)
    clock = CompositionEntry(
        clock_value,
        entry_id=_CLOCK_ENTRY,
        role="history",
        owner_id="coupling",
        structure_id=_layout("coupling-clock", "coupling", clock_value),
        revision_id=_revision(boundary, _CLOCK_ENTRY, clock_value),
        semantics_id="coupling:accepted-clock",
    )
    return budget, clock


def _epoch_entry(
    epoch: PreparedCouplingEpoch,
    entries: Sequence[CompositionEntry],
    independent: frozenset[str],
    artifacts: Sequence[CompositionDependency],
    /,
) -> CompositionEntry:
    """The prepared graph binds every state layout and contract it was prepared for."""
    bindings: list[CompositionDependency] = list(artifacts)
    for item in entries:
        match item.role:
            case "model-state" | "rng" if item.owner_id in independent:
                # Explicit same-semantics binding of discretization-independent state.
                bindings.append(item.binding("semantics"))
            case "physical-state" | "model-state" | "rng" | "exchange-state":
                bindings.append(item.binding("structure"))
            case "budget":
                bindings.extend((item.binding("structure"), item.binding("semantics")))
            case _:
                pass
    prepared = epoch.prepared_coupling
    return CompositionEntry(
        epoch,
        entry_id=_EPOCH_ENTRY,
        role="prepared-graph",
        owner_id="coupling",
        structure_id=epoch.epoch_id,
        revision_id=prepared.plan_id,
        semantics_id=canonical_fingerprint(
            {
                "kind": "coupling-composition",
                "subsystems": sorted(item.subsystem_id for item in prepared.subsystems),
                "exchanges": sorted(item.exchange_id for item in prepared.exchanges),
            }
        ),
        dependencies=bindings,
    )


def coupling_composition_entries(
    epoch: PreparedCouplingEpoch,
    state: CouplingState,
    /,
    *,
    native_dependencies: Mapping[str, Sequence[CompositionDependency]] | None = None,
    discretization_independent: Sequence[str] = (),
    epoch_dependencies: Sequence[CompositionDependency] = (),
) -> tuple[CompositionEntry, ...]:
    """Split one accepted coupling boundary into canonical composition entries.

    Entries: `<subsystem>/native` (physical state whose structure identity is the
    declared `discretization_bundle_id`, for example the owner's topology epoch
    ID), `<subsystem>/model-state`,
    `<subsystem>/rng`, `<subsystem>/windows` for native method participants,
    `exchange/<exchange>` boundary values, the `coupling/budget` ledger (its
    semantics is the physical budget contract of every exchange),
    `coupling/clock`, and the `coupling/epoch` prepared graph binding all of them.

    `native_dependencies` binds each participant's native state to the owner
    artifacts it lives on (for example its discretization entry). Model state and
    carried keys bind the native structure, so changing a discretization refuses
    them unless an explicit transport is staged or the participant is listed in
    `discretization_independent`, in which case the prepared graph binds them by
    semantics only. `epoch_dependencies` binds the prepared graph to the owner
    artifacts it was prepared from (participant methods, interface transfers), so
    replacing any of them forces the graph to be reprepared.
    """

    if not isinstance(epoch, PreparedCouplingEpoch) or not isinstance(
        state, CouplingState
    ):
        raise TypeError(
            "Composition entries need a PreparedCouplingEpoch and CouplingState."
        )
    prepared = epoch.prepared_coupling
    reference = prepared.reference_state
    if (
        state.graph_id != prepared.graph_id
        or state.subsystem_ids != reference.subsystem_ids
        or state.exchange_ids != reference.exchange_ids
        or state.budget_row_ids != reference.budget_row_ids
    ):
        raise ValueError("Coupling state does not belong to the prepared epoch.")
    extra = {} if native_dependencies is None else dict(native_dependencies)
    independent = frozenset(discretization_independent)
    unknown = sorted((set(extra) | independent) - set(reference.subsystem_ids))
    if unknown:
        raise ValueError("Unknown coupling participants: " + ", ".join(unknown))
    boundary = _boundary(epoch, state)
    entries: list[CompositionEntry] = []
    for index, subsystem_id in enumerate(reference.subsystem_ids):
        entries.extend(
            _participant_entries(
                epoch,
                state,
                boundary,
                index,
                tuple(extra.get(subsystem_id, ())),
                subsystem_id in independent,
            )
        )
    entries.extend(_exchange_entries(epoch, state, boundary))
    entries.extend(_ledger_entries(epoch, state, boundary))
    entries.append(_epoch_entry(epoch, entries, independent, tuple(epoch_dependencies)))
    return tuple(entries)


def coupling_state_from_composition(
    composition: Composition, /
) -> tuple[PreparedCouplingEpoch, CouplingState]:
    """Assemble the accepted coupling boundary of a published composition.

    The prepared epoch's bindings already guarantee that every participant,
    exchange, and budget entry has the structure and contract it was prepared
    for; the accepted clock and cumulative budgets continue unchanged.
    """
    from .coupling._method_participants import MethodParticipantState

    if not isinstance(composition, Composition):
        raise TypeError("composition must be a Composition.")
    epoch = composition.value(_EPOCH_ENTRY)
    if not isinstance(epoch, PreparedCouplingEpoch):
        raise TypeError(f"{_EPOCH_ENTRY!r} must hold a PreparedCouplingEpoch.")
    reference = epoch.prepared_coupling.reference_state
    states: list[Any] = []
    for subsystem_id, template in zip(
        reference.subsystem_ids, reference.participant_states, strict=True
    ):
        ids = _participant_entry_ids(subsystem_id)
        native = composition.value(ids["native"])
        if not isinstance(template, MethodParticipantState):
            states.append(native)
            continue
        accepted_windows, native_steps = composition.value(ids["history"])
        states.append(
            MethodParticipantState(
                native,
                None
                if template.model_state is None
                else composition.value(ids["model-state"]),
                None if template.key_data is None else composition.value(ids["rng"]),
                accepted_windows,
                native_steps,
            )
        )
    time, window_index = composition.value(_CLOCK_ENTRY)
    state = CouplingState(
        tuple(states),
        tuple(
            composition.value(_exchange_entry_id(exchange_id))
            for exchange_id in reference.exchange_ids
        ),
        time,
        window_index,
        subsystem_ids=reference.subsystem_ids,
        exchange_ids=reference.exchange_ids,
        cumulative_exchange_budget=composition.value(_BUDGET_ENTRY),
        budget_row_ids=reference.budget_row_ids,
        graph_id=epoch.prepared_coupling.graph_id,
    )
    return epoch, state


def coupling_exchange_transport(
    source: Composition,
    target_epoch: PreparedCouplingEpoch,
    target_entries: Sequence[CompositionEntry],
    exchange_id: str,
    /,
) -> CompositionTransport:
    """Re-derive one exchange boundary value through the target epoch's route.

    The target value is the target epoch's own lowering of the transported source
    checkpoint through the declared spatial transfer and temporal conversion
    (whole-window amounts restart at zero; their consumed content stays in the
    retained budget ledger). It carries no conservation claim of its own, so its
    provenance is checked instead: the target epoch must have been lowered at
    the source's accepted clock time from exactly the staged `<subsystem>/native`
    target entries, and a native kept on an unchanged structure must be the
    accepted source state itself.
    """
    from .coupling._method_participants import MethodParticipantState

    entry_id = _exchange_entry_id(exchange_id)
    reference = target_epoch.prepared_coupling.reference_state
    if exchange_id not in reference.exchange_ids:
        raise ValueError(f"Target epoch has no exchange {exchange_id!r}.")
    entries = tuple(target_entries)
    staged = {item.entry_id: item for item in entries}
    if len(staged) != len(entries):
        raise ValueError("Target entries must have unique entry IDs.")
    if entry_id not in staged:
        raise ValueError(f"Target entries must contain exactly one {entry_id!r}.")
    target = staged[entry_id]
    derived = reference.exchange_values[reference.exchange_ids.index(exchange_id)]
    if target.value is not derived:
        raise ValueError(
            f"{entry_id!r} is not the target epoch's derived boundary value."
        )
    # Host staging boundary: the accepted clock is read once.
    source_time = np.asarray(source.value(_CLOCK_ENTRY)[0])
    if not np.array_equal(np.asarray(reference.time), source_time):
        raise ValueError(
            f"{entry_id!r} was lowered at another time than the accepted source "
            "boundary; prepare the target epoch at the source coupling/clock time."
        )
    for subsystem_id, template in zip(
        reference.subsystem_ids, reference.participant_states, strict=True
    ):
        native_id = _participant_entry_ids(subsystem_id)["native"]
        native = (
            template.native if isinstance(template, MethodParticipantState) else template
        )
        if native_id not in staged or staged[native_id].value is not native:
            raise ValueError(
                f"{entry_id!r} was not lowered from the staged {native_id!r} target "
                "entry."
            )
        kept = source.entry(native_id)
        if staged[native_id].structure_id == kept.structure_id and not bool(
            eqx.tree_equal(native, kept.value)
        ):
            raise ValueError(
                f"{entry_id!r} was lowered from a {native_id!r} other than the "
                "accepted source state on its unchanged structure."
            )
    exchange = target_epoch.prepared_coupling.exchanges[
        reference.exchange_ids.index(exchange_id)
    ]
    route = canonical_fingerprint(
        {
            "kind": "coupling-exchange-rederivation",
            "epoch": target_epoch.epoch_id,
            "exchange": exchange_id,
            "transfer": None
            if exchange.transfer is None
            else exchange.transfer.transfer_id,
            "temporal": None if exchange.temporal is None else exchange.temporal.kind,
        }
    )
    return CompositionTransport(
        "physical-remap",
        (entry_id,),
        (target,),
        source_structure_ids=(source.entry(entry_id).structure_id,),
        route_id=route,
        successful=True,
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
    "coupling_composition_entries",
    "coupling_exchange_transport",
    "coupling_state_from_composition",
    "rollout_adaptive_coupling",
    "transition_coupling_epoch",
]
