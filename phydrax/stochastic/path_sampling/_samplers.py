#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Prepared TPS, TIS, and RETIS sampling lifecycles."""

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Key

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._core import (
    _fixed_step_time_grid_valid,
    FunctionalDynamicsKernel,
    path_trajectory_id,
    PathBuffer,
    PathLineageLog,
    select_path,
)
from ._moves import (
    _validated_modification,
    _validated_selection,
    AbstractShootingModifier,
    AbstractShootingSelector,
    IdentityShootingModifier,
    PathMoveResult,
    PathProposalEvaluation,
    propose_one_way_shooting,
    propose_path_reversal,
    propose_path_shift,
    propose_replica_exchange,
    propose_two_way_shooting,
    UniformShootingSelector,
)
from ._targets import (
    AbstractPathAction,
    AbstractPathEnsemble,
    DeterministicPathAction,
    FixedPathEnsemble,
    InterfaceNetworkPlan,
    NormalizedStochasticPathAction,
    path_log_target,
)


def _identity(value: str, name: str, /) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string.")
    return value


class TPSPlan(StrictModule, NonTrainableState):
    """Immutable transition-path-sampling move and evidence policy."""

    ensemble: AbstractPathEnsemble
    kernel: FunctionalDynamicsKernel
    action: AbstractPathAction
    selector: AbstractShootingSelector
    modifier: AbstractShootingModifier
    move_kind: str = eqx.field(static=True)
    maximum_shift: int = eqx.field(static=True)
    lineage_capacity: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        ensemble: AbstractPathEnsemble,
        kernel: FunctionalDynamicsKernel,
        action: AbstractPathAction,
        /,
        *,
        selector: AbstractShootingSelector | None = None,
        modifier: AbstractShootingModifier | None = None,
        move_kind: str = "two-way-shooting",
        maximum_shift: int = 1,
        lineage_capacity: int = 1024,
        plan_id: str | None = None,
    ):
        if not isinstance(ensemble, AbstractPathEnsemble):
            raise TypeError("ensemble must implement AbstractPathEnsemble.")
        if not isinstance(kernel, FunctionalDynamicsKernel):
            raise TypeError("kernel must be FunctionalDynamicsKernel.")
        if not isinstance(action, AbstractPathAction):
            raise TypeError("action must implement AbstractPathAction.")
        if not isinstance(action.action_id, str) or not action.action_id:
            raise ValueError("action_id must be non-empty.")
        if (
            isinstance(action, (DeterministicPathAction, NormalizedStochasticPathAction))
            and action.kernel.kernel_id != kernel.kernel_id
        ):
            raise ValueError("The path action and propagation kernel must agree.")
        selector_ = UniformShootingSelector() if selector is None else selector
        modifier_ = IdentityShootingModifier() if modifier is None else modifier
        if not isinstance(selector_, AbstractShootingSelector):
            raise TypeError("selector must implement AbstractShootingSelector.")
        if not isinstance(modifier_, AbstractShootingModifier):
            raise TypeError("modifier must implement AbstractShootingModifier.")
        if move_kind not in (
            "one-way-shooting",
            "two-way-shooting",
            "shifting",
            "path-reversal",
        ):
            raise ValueError("Unknown TPS move_kind.")
        if (
            move_kind in ("one-way-shooting", "two-way-shooting", "shifting")
            and not kernel.capabilities.supports_backward
        ):
            raise ValueError("The selected TPS move requires backward-capable dynamics.")
        if (
            move_kind in ("one-way-shooting", "two-way-shooting", "shifting")
            and not kernel.capabilities.fixed_step
        ):
            raise ValueError("TPS regrowth moves require fixed-step dynamics.")
        if move_kind == "path-reversal" and not kernel.capabilities.reversible:
            raise ValueError("Path reversal requires reversible dynamics.")
        shift, history = int(maximum_shift), int(lineage_capacity)
        if shift <= 0 or history <= 0:
            raise ValueError("maximum_shift and lineage_capacity must be positive.")
        if move_kind == "shifting" and (
            not isinstance(ensemble, FixedPathEnsemble) or shift >= ensemble.path_length
        ):
            raise ValueError(
                "TPS shifting requires a FixedPathEnsemble and maximum_shift below its length."
            )
        identity = plan_id or canonical_fingerprint(
            {
                "kind": "transition-path-sampling-plan-v1",
                "ensemble": ensemble.ensemble_id,
                "kernel": kernel.kernel_id,
                "action": action.action_id,
                "selector": selector_.selector_id,
                "modifier": modifier_.modifier_id,
                "move": move_kind,
                "maximum_shift": shift,
                "lineage_capacity": history,
            }
        )
        self.ensemble = ensemble
        self.kernel = kernel
        self.action = action
        self.selector = selector_
        self.modifier = modifier_
        self.move_kind = move_kind
        self.maximum_shift = shift
        self.lineage_capacity = history
        self.plan_id = _identity(identity, "plan_id")


class PreparedTPS(StrictModule, NonTrainableState):
    """Validated initial TPS trajectory and immutable execution identity."""

    plan: TPSPlan
    initial_path: PathBuffer
    prepared_id: str = eqx.field(static=True)
    initial_trajectory_id: str = eqx.field(static=True)


class TPSState(StrictModule, NonTrainableState):
    """JIT-compatible TPS state with explicit accepted and rejected lineage."""

    path: PathBuffer
    log_target: Array
    step_index: Array
    accepted_count: Array
    rejected_count: Array
    trajectory_serial: Array
    proposal_serial: Array
    lineage: PathLineageLog
    last_evaluation: PathProposalEvaluation
    prepared_id: str = eqx.field(static=True)


class TPSStep(StrictModule):
    state: TPSState
    move: PathMoveResult
    prepared_id: str = eqx.field(static=True)


def _zero_evaluation(plan: TPSPlan, path: PathBuffer, /) -> PathProposalEvaluation:
    target = path_log_target(plan.ensemble, plan.action, path)
    if plan.move_kind == "path-reversal":
        dtype = jnp.result_type(target.dtype, jnp.float32)
    else:
        selector = (
            UniformShootingSelector(endpoint_margin=0)
            if plan.move_kind == "shifting"
            else plan.selector
        )
        modifier = (
            IdentityShootingModifier() if plan.move_kind == "shifting" else plan.modifier
        )
        selection_key, modifier_key = jax.random.split(jax.random.key(0))
        selection = _validated_selection(selector, selection_key, path)
        modification = _validated_modification(
            modifier, modifier_key, path.positions[selection.index]
        )
        density = plan.kernel.transition_log_density(
            path.positions[0], path.positions[1], path.direction
        )
        dtype = jnp.result_type(
            target.dtype,
            selection.log_probability.dtype,
            modification.forward_log_density.dtype,
            modification.reverse_log_density.dtype,
            density.dtype,
            jnp.float32,
        )
    zero, true = jnp.asarray(0.0, dtype=dtype), jnp.asarray(True)
    return PathProposalEvaluation(
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        jnp.asarray(0, jnp.int32),
    )


def _cast_evaluation(
    evaluation: PathProposalEvaluation, dtype, /
) -> PathProposalEvaluation:
    return PathProposalEvaluation(
        jnp.asarray(evaluation.target_log_ratio, dtype=dtype),
        jnp.asarray(evaluation.selector_log_ratio, dtype=dtype),
        jnp.asarray(evaluation.modifier_log_ratio, dtype=dtype),
        jnp.asarray(evaluation.propagation_log_ratio, dtype=dtype),
        jnp.asarray(evaluation.length_log_ratio, dtype=dtype),
        jnp.asarray(evaluation.exchange_log_ratio, dtype=dtype),
        jnp.asarray(evaluation.log_acceptance_ratio, dtype=dtype),
        evaluation.target_valid,
        evaluation.selector_valid,
        evaluation.modifier_valid,
        evaluation.propagation_valid,
        evaluation.length_valid,
        evaluation.exchange_valid,
        evaluation.proposal_valid,
        evaluation.propagation_status,
    )


def prepare_tps(plan: TPSPlan, initial_path: PathBuffer, /) -> PreparedTPS:
    """Validate one initial path and freeze plan/prepared/trajectory identities."""

    if not isinstance(plan, TPSPlan):
        raise TypeError("plan must be TPSPlan.")
    if not isinstance(initial_path, PathBuffer):
        raise TypeError("initial_path must be PathBuffer.")
    if plan.move_kind in (
        "one-way-shooting",
        "two-way-shooting",
        "shifting",
    ) and not bool(_fixed_step_time_grid_valid(initial_path, plan.kernel.time_step)):
        raise ValueError(
            "TPS regrowth requires active path times spaced by kernel.time_step."
        )
    if not bool(plan.ensemble.contains(initial_path)):
        raise ValueError("initial_path is outside the TPS ensemble.")
    target = path_log_target(plan.ensemble, plan.action, initial_path)
    if not bool(jnp.isfinite(target)):
        raise ValueError("initial_path has a nonfinite path target.")
    if plan.move_kind in ("one-way-shooting", "two-way-shooting"):
        selection = _validated_selection(plan.selector, jax.random.key(0), initial_path)
        if not bool(selection.valid):
            raise ValueError(
                "Initial path has no finite positive-mass shooting selection."
            )
    trajectory_id = path_trajectory_id(initial_path)
    prepared_id = canonical_fingerprint(
        {
            "kind": "prepared-transition-path-sampling-v1",
            "plan": plan.plan_id,
            "trajectory": trajectory_id,
            "capacity": initial_path.capacity,
            "event_shape": list(initial_path.event_shape),
        }
    )
    return PreparedTPS(plan, initial_path, prepared_id, trajectory_id)


def initialize_tps(prepared: PreparedTPS, /) -> TPSState:
    if not isinstance(prepared, PreparedTPS):
        raise TypeError("prepared must be PreparedTPS.")
    target = path_log_target(
        prepared.plan.ensemble, prepared.plan.action, prepared.initial_path
    )
    return TPSState(
        prepared.initial_path,
        target,
        jnp.asarray(0, jnp.uint32),
        jnp.asarray(0, jnp.uint32),
        jnp.asarray(0, jnp.uint32),
        jnp.asarray(0, jnp.uint32),
        jnp.asarray(0, jnp.uint32),
        PathLineageLog.empty(prepared.plan.lineage_capacity),
        _zero_evaluation(prepared.plan, prepared.initial_path),
        prepared.prepared_id,
    )


def _propose_tps(
    plan: TPSPlan,
    path: PathBuffer,
    key: Key[Array, ""],
    /,
) -> PathMoveResult:
    if plan.move_kind == "one-way-shooting":
        return propose_one_way_shooting(
            plan.ensemble,
            plan.action,
            plan.kernel,
            plan.selector,
            plan.modifier,
            path,
            key,
        )
    if plan.move_kind == "two-way-shooting":
        return propose_two_way_shooting(
            plan.ensemble,
            plan.action,
            plan.kernel,
            plan.selector,
            plan.modifier,
            path,
            key,
        )
    if plan.move_kind == "shifting":
        return propose_path_shift(
            plan.ensemble,
            plan.action,
            plan.kernel,
            path,
            key,
            maximum_shift=plan.maximum_shift,
        )
    return propose_path_reversal(plan.ensemble, plan.action, plan.kernel, path, key)


def tps_step(
    prepared: PreparedTPS,
    state: TPSState,
    key: Key[Array, ""],
    /,
) -> TPSStep:
    """Execute exactly one path proposal and commit or reject without retry."""

    if not isinstance(prepared, PreparedTPS) or not isinstance(state, TPSState):
        raise TypeError("tps_step requires PreparedTPS and TPSState.")
    if state.prepared_id != prepared.prepared_id:
        raise ValueError("TPS state belongs to a different prepared runtime.")
    move = _propose_tps(prepared.plan, state.path, key)
    evidence_dtype = state.last_evaluation.log_acceptance_ratio.dtype
    actual_dtype = move.evaluation.log_acceptance_ratio.dtype
    if jnp.result_type(evidence_dtype, actual_dtype) != evidence_dtype:
        raise ValueError(
            "Runtime proposal evidence exceeded the prepared evidence dtype."
        )
    evaluation = _cast_evaluation(move.evaluation, evidence_dtype)
    lineage_available = state.lineage.count < state.lineage.capacity
    accepted = move.accepted & lineage_available
    committed = select_path(state.path, move.proposed, accepted)
    candidate_serial = state.proposal_serial + jnp.asarray(1, jnp.uint32)
    lineage = state.lineage.append(state.trajectory_serial, candidate_serial, accepted)
    trajectory_serial = jnp.where(accepted, candidate_serial, state.trajectory_serial)
    log_target = path_log_target(prepared.plan.ensemble, prepared.plan.action, committed)
    next_state = TPSState(
        committed,
        log_target,
        state.step_index + jnp.asarray(1, jnp.uint32),
        state.accepted_count + accepted.astype(jnp.uint32),
        state.rejected_count + (~accepted).astype(jnp.uint32),
        trajectory_serial,
        candidate_serial,
        lineage,
        evaluation,
        state.prepared_id,
    )
    committed_move = PathMoveResult(
        move.current,
        move.proposed,
        committed,
        evaluation,
        accepted,
        move.shooting_index,
        move.candidate_shooting_index,
    )
    return TPSStep(next_state, committed_move, prepared.prepared_id)


class TISPlan(StrictModule, NonTrainableState):
    """Transition-interface sampling policy over an ordered interface network."""

    network: InterfaceNetworkPlan
    kernel: FunctionalDynamicsKernel
    action: AbstractPathAction
    selector: AbstractShootingSelector
    modifier: AbstractShootingModifier
    move_kind: str = eqx.field(static=True)
    lineage_capacity: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        network: InterfaceNetworkPlan,
        kernel: FunctionalDynamicsKernel,
        action: AbstractPathAction,
        /,
        *,
        selector: AbstractShootingSelector | None = None,
        modifier: AbstractShootingModifier | None = None,
        move_kind: str = "two-way-shooting",
        lineage_capacity: int = 1024,
        plan_id: str | None = None,
    ):
        if not isinstance(network, InterfaceNetworkPlan):
            raise TypeError("network must be InterfaceNetworkPlan.")
        if not isinstance(kernel, FunctionalDynamicsKernel) or not isinstance(
            action, AbstractPathAction
        ):
            raise TypeError("kernel and action must implement path-sampling contracts.")
        if (
            isinstance(action, (DeterministicPathAction, NormalizedStochasticPathAction))
            and action.kernel.kernel_id != kernel.kernel_id
        ):
            raise ValueError("The path action and propagation kernel must agree.")
        selector_ = UniformShootingSelector() if selector is None else selector
        modifier_ = IdentityShootingModifier() if modifier is None else modifier
        if not isinstance(selector_, AbstractShootingSelector) or not isinstance(
            modifier_, AbstractShootingModifier
        ):
            raise TypeError("selector and modifier must implement shooting contracts.")
        if move_kind not in ("one-way-shooting", "two-way-shooting", "path-reversal"):
            raise ValueError("TIS move_kind is unsupported.")
        if (
            move_kind in ("one-way-shooting", "two-way-shooting")
            and not kernel.capabilities.supports_backward
        ):
            raise ValueError("TIS shooting requires backward-capable dynamics.")
        if (
            move_kind in ("one-way-shooting", "two-way-shooting")
            and not kernel.capabilities.fixed_step
        ):
            raise ValueError("TIS shooting requires fixed-step dynamics.")
        if move_kind == "path-reversal" and not kernel.capabilities.reversible:
            raise ValueError("TIS path reversal requires reversible dynamics.")
        history = int(lineage_capacity)
        if history <= 0:
            raise ValueError("lineage_capacity must be positive.")
        identity = plan_id or canonical_fingerprint(
            {
                "kind": "transition-interface-sampling-plan-v1",
                "network": network.network_id,
                "kernel": kernel.kernel_id,
                "action": action.action_id,
                "selector": selector_.selector_id,
                "modifier": modifier_.modifier_id,
                "move": move_kind,
                "lineage_capacity": history,
            }
        )
        self.network = network
        self.kernel = kernel
        self.action = action
        self.selector = selector_
        self.modifier = modifier_
        self.move_kind = move_kind
        self.lineage_capacity = history
        self.plan_id = _identity(identity, "plan_id")


class PreparedTIS(StrictModule, NonTrainableState):
    plan: TISPlan
    replicas: tuple[PreparedTPS, ...]
    prepared_id: str = eqx.field(static=True)


class TISState(StrictModule, NonTrainableState):
    replicas: tuple[TPSState, ...]
    step_index: Array


class TISStep(StrictModule):
    state: TISState
    replica_index: int = eqx.field(static=True)
    move: TPSStep
    prepared_id: str = eqx.field(static=True)


def prepare_tis(plan: TISPlan, initial_paths: Sequence[PathBuffer], /) -> PreparedTIS:
    if not isinstance(plan, TISPlan):
        raise TypeError("plan must be TISPlan.")
    paths = tuple(initial_paths)
    if len(paths) != plan.network.interface_count:
        raise ValueError("TIS requires one initial path per interface.")
    replicas = tuple(
        prepare_tps(
            TPSPlan(
                plan.network.ensemble(index),
                plan.kernel,
                plan.action,
                selector=plan.selector,
                modifier=plan.modifier,
                move_kind=plan.move_kind,
                lineage_capacity=plan.lineage_capacity,
            ),
            path,
        )
        for index, path in enumerate(paths)
    )
    prepared_id = canonical_fingerprint(
        {
            "kind": "prepared-transition-interface-sampling-v1",
            "plan": plan.plan_id,
            "replicas": [replica.prepared_id for replica in replicas],
        }
    )
    return PreparedTIS(plan, replicas, prepared_id)


def initialize_tis(prepared: PreparedTIS, /) -> TISState:
    if not isinstance(prepared, PreparedTIS):
        raise TypeError("prepared must be PreparedTIS.")
    return TISState(
        tuple(initialize_tps(replica) for replica in prepared.replicas),
        jnp.asarray(0, jnp.uint32),
    )


def tis_step(
    prepared: PreparedTIS,
    state: TISState,
    key: Key[Array, ""],
    /,
    *,
    replica_index: int = 0,
) -> TISStep:
    if not isinstance(prepared, PreparedTIS) or not isinstance(state, TISState):
        raise TypeError("tis_step requires PreparedTIS and TISState.")
    if len(state.replicas) != len(prepared.replicas) or any(
        replica_state.prepared_id != replica_prepared.prepared_id
        for replica_state, replica_prepared in zip(
            state.replicas, prepared.replicas, strict=True
        )
    ):
        raise ValueError("TIS state belongs to a different prepared runtime.")
    index = int(replica_index)
    if index < 0 or index >= len(prepared.replicas):
        raise IndexError("replica_index is outside the TIS network.")
    move = tps_step(prepared.replicas[index], state.replicas[index], key)
    replicas = state.replicas[:index] + (move.state,) + state.replicas[index + 1 :]
    return TISStep(
        TISState(replicas, state.step_index + jnp.asarray(1, jnp.uint32)),
        index,
        move,
        prepared.prepared_id,
    )


class RETISPlan(StrictModule, NonTrainableState):
    """Replica-exchange TIS policy including the minus ensemble."""

    tis: TISPlan
    plan_id: str = eqx.field(static=True)

    def __init__(self, tis: TISPlan, /, *, plan_id: str | None = None):
        if not isinstance(tis, TISPlan):
            raise TypeError("tis must be TISPlan.")
        identity = plan_id or canonical_fingerprint(
            {"kind": "replica-exchange-tis-plan-v1", "tis": tis.plan_id, "minus": True}
        )
        self.tis = tis
        self.plan_id = _identity(identity, "plan_id")


class PreparedRETIS(StrictModule, NonTrainableState):
    plan: RETISPlan
    replicas: tuple[PreparedTPS, ...]
    prepared_id: str = eqx.field(static=True)


class RETISState(StrictModule, NonTrainableState):
    replicas: tuple[TPSState, ...]
    step_index: Array
    exchange_count: Array
    accepted_exchange_count: Array


class RETISStep(StrictModule):
    state: RETISState
    evaluation: PathProposalEvaluation
    accepted: Array
    move_kind: str = eqx.field(static=True)
    replica_index: int = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)


def prepare_retis(
    plan: RETISPlan, initial_paths: Sequence[PathBuffer], /
) -> PreparedRETIS:
    if not isinstance(plan, RETISPlan):
        raise TypeError("plan must be RETISPlan.")
    paths = tuple(initial_paths)
    if len(paths) != plan.tis.network.interface_count + 1:
        raise ValueError("RETIS requires one minus and one path per interface.")
    if any(not isinstance(path, PathBuffer) for path in paths):
        raise TypeError("RETIS initial paths must be PathBuffer values.")
    reference = paths[0]
    reference_signature = (
        reference.positions.shape,
        reference.positions.dtype,
        reference.times.shape,
        reference.times.dtype,
        reference.mask.shape,
        reference.mask.dtype,
        reference.lineage.shape,
        reference.lineage.dtype,
    )
    for path in paths[1:]:
        signature = (
            path.positions.shape,
            path.positions.dtype,
            path.times.shape,
            path.times.dtype,
            path.mask.shape,
            path.mask.dtype,
            path.lineage.shape,
            path.lineage.dtype,
        )
        if signature != reference_signature:
            raise ValueError(
                "All RETIS paths must share exchange-compatible shapes and dtypes."
            )
    ensembles = (plan.tis.network.minus_ensemble(),) + tuple(
        plan.tis.network.ensemble(index)
        for index in range(plan.tis.network.interface_count)
    )
    replicas = tuple(
        prepare_tps(
            TPSPlan(
                ensemble,
                plan.tis.kernel,
                plan.tis.action,
                selector=plan.tis.selector,
                modifier=plan.tis.modifier,
                move_kind=plan.tis.move_kind,
                lineage_capacity=plan.tis.lineage_capacity,
            ),
            path,
        )
        for ensemble, path in zip(ensembles, paths, strict=True)
    )
    prepared_id = canonical_fingerprint(
        {
            "kind": "prepared-replica-exchange-tis-v1",
            "plan": plan.plan_id,
            "replicas": [replica.prepared_id for replica in replicas],
        }
    )
    return PreparedRETIS(plan, replicas, prepared_id)


def initialize_retis(prepared: PreparedRETIS, /) -> RETISState:
    if not isinstance(prepared, PreparedRETIS):
        raise TypeError("prepared must be PreparedRETIS.")
    return RETISState(
        tuple(initialize_tps(replica) for replica in prepared.replicas),
        jnp.asarray(0, jnp.uint32),
        jnp.asarray(0, jnp.uint32),
        jnp.asarray(0, jnp.uint32),
    )


def retis_step(
    prepared: PreparedRETIS,
    state: RETISState,
    key: Key[Array, ""],
    /,
    *,
    move_kind: str = "shooting",
    replica_index: int = 0,
) -> RETISStep:
    """Execute one statically selected RETIS shooting or neighbor-exchange move."""

    if not isinstance(prepared, PreparedRETIS) or not isinstance(state, RETISState):
        raise TypeError("retis_step requires PreparedRETIS and RETISState.")
    if len(state.replicas) != len(prepared.replicas) or any(
        replica_state.prepared_id != replica_prepared.prepared_id
        for replica_state, replica_prepared in zip(
            state.replicas, prepared.replicas, strict=True
        )
    ):
        raise ValueError("RETIS state belongs to a different prepared runtime.")
    index = int(replica_index)
    if move_kind == "shooting":
        if index < 0 or index >= len(state.replicas):
            raise IndexError("replica_index is outside RETIS replicas.")
        move = tps_step(prepared.replicas[index], state.replicas[index], key)
        replicas = state.replicas[:index] + (move.state,) + state.replicas[index + 1 :]
        next_state = RETISState(
            replicas,
            state.step_index + jnp.asarray(1, jnp.uint32),
            state.exchange_count,
            state.accepted_exchange_count,
        )
        return RETISStep(
            next_state,
            move.move.evaluation,
            move.move.accepted,
            "shooting",
            index,
            prepared.prepared_id,
        )
    if move_kind != "exchange":
        raise ValueError("RETIS move_kind must be 'shooting' or 'exchange'.")
    if index < 0 or index >= len(state.replicas) - 1:
        raise IndexError("Exchange replica_index must identify a neighboring pair.")
    left_prepared, right_prepared = prepared.replicas[index], prepared.replicas[index + 1]
    left_state, right_state = state.replicas[index], state.replicas[index + 1]
    exchange = propose_replica_exchange(
        left_prepared.plan.ensemble,
        right_prepared.plan.ensemble,
        prepared.plan.tis.action,
        left_state.path,
        right_state.path,
        key,
    )
    lineage_available = (left_state.lineage.count < left_state.lineage.capacity) & (
        right_state.lineage.count < right_state.lineage.capacity
    )
    exchange_accepted = exchange.accepted & lineage_available
    left_path = select_path(left_state.path, exchange.left, exchange_accepted)
    right_path = select_path(right_state.path, exchange.right, exchange_accepted)
    serial = state.step_index + jnp.asarray(1, jnp.uint32)
    evidence_dtype = left_state.last_evaluation.log_acceptance_ratio.dtype
    if right_state.last_evaluation.log_acceptance_ratio.dtype != evidence_dtype:
        raise ValueError("RETIS replica evidence dtypes must agree.")
    if (
        jnp.result_type(evidence_dtype, exchange.evaluation.log_acceptance_ratio.dtype)
        != evidence_dtype
    ):
        raise ValueError("Exchange evidence exceeded the prepared RETIS evidence dtype.")
    exchange_evaluation = _cast_evaluation(exchange.evaluation, evidence_dtype)
    left_candidate_serial = jnp.maximum(
        left_state.proposal_serial, right_state.proposal_serial
    ) + jnp.asarray(1, jnp.uint32)
    right_candidate_serial = left_candidate_serial + jnp.asarray(1, jnp.uint32)

    def exchanged(
        old: TPSState,
        path: PathBuffer,
        ensemble: AbstractPathEnsemble,
        candidate_serial: Array,
    ) -> TPSState:
        lineage = old.lineage.append(
            old.trajectory_serial, candidate_serial, exchange_accepted
        )
        return TPSState(
            path,
            path_log_target(ensemble, prepared.plan.tis.action, path),
            old.step_index + jnp.asarray(1, jnp.uint32),
            old.accepted_count + exchange_accepted.astype(jnp.uint32),
            old.rejected_count + (~exchange_accepted).astype(jnp.uint32),
            jnp.where(exchange_accepted, candidate_serial, old.trajectory_serial),
            candidate_serial,
            lineage,
            exchange_evaluation,
            old.prepared_id,
        )

    left = exchanged(
        left_state,
        left_path,
        left_prepared.plan.ensemble,
        left_candidate_serial,
    )
    right = exchanged(
        right_state,
        right_path,
        right_prepared.plan.ensemble,
        right_candidate_serial,
    )
    replicas = state.replicas[:index] + (left, right) + state.replicas[index + 2 :]
    next_state = RETISState(
        replicas,
        serial,
        state.exchange_count + jnp.asarray(1, jnp.uint32),
        state.accepted_exchange_count + exchange_accepted.astype(jnp.uint32),
    )
    return RETISStep(
        next_state,
        exchange_evaluation,
        exchange_accepted,
        "exchange",
        index,
        prepared.prepared_id,
    )


__all__ = [
    "initialize_retis",
    "initialize_tis",
    "initialize_tps",
    "PreparedRETIS",
    "PreparedTIS",
    "PreparedTPS",
    "prepare_retis",
    "prepare_tis",
    "prepare_tps",
    "RETISPlan",
    "RETISState",
    "RETISStep",
    "retis_step",
    "TISPlan",
    "TISState",
    "TISStep",
    "tis_step",
    "TPSPlan",
    "TPSState",
    "TPSStep",
    "tps_step",
]
