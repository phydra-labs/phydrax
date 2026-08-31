#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum, IntFlag

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._rigid_body import PreparedRigidBodySet
from ._rigid_joints import PreparedRigidJointGraph, RigidJointRowLayout


class RigidTopologyEventKind(IntEnum):
    """Stable event kinds stored in a rigid-topology journal."""

    JOINT_BREAK = 1
    BODY_DEACTIVATION = 2
    BODY_ACTIVATION = 3
    JOINT_DEACTIVATION = 4
    JOINT_ACTIVATION = 5


class RigidTopologyFailure(IntFlag):
    """Bitwise rejection taxonomy for an atomic topology transition."""

    NONE = 0
    REPLAY_DIGEST_MISMATCH = 1 << 0
    INVALID_LOADING = 1 << 1
    INVALID_DERIVATIVE = 1 << 2
    EVENT_CAPACITY_OVERFLOW = 1 << 3
    PROPOSAL_CONFLICT = 1 << 4
    PRECONDITION_FAILED = 1 << 5
    INACTIVE_JOINT_ENDPOINT = 1 << 6
    INVALID_GUARD_MARGIN = 1 << 7
    INVALID_STATE = 1 << 8


def _capacity_parameter(value: ArrayLike, count: int, name: str, /) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 0:
        return np.full((count,), array.item(), dtype=array.dtype)
    if array.shape != (count,):
        raise ValueError(f"{name} must be scalar or have joint-capacity shape.")
    return array.copy()


class BreakableRigidJointLawPlan(StrictModule, NonTrainableState):
    """Fixed-capacity irreversible mixed-mode law for rigid joints.

    The lower arming surface and positive derivative margin make overload
    crossings unambiguous. A joint can break only after visiting the armed
    side, then crossing the failure surface in the loading direction.
    """

    joint_ids: Array
    initiation_loading: Array
    failure_loading: Array
    arming_loading: Array
    fracture_energy: Array
    minimum_loading_rate: Array
    initial_active_mask: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        joint_ids: ArrayLike,
        initiation_loading: ArrayLike,
        failure_loading: ArrayLike,
        fracture_energy: ArrayLike,
        /,
        *,
        arming_loading: ArrayLike | None = None,
        minimum_loading_rate: ArrayLike = 1.0e-12,
        initial_active_mask: ArrayLike | None = None,
        plan_id: str | None = None,
    ):
        identifiers = np.asarray(joint_ids)
        if identifiers.ndim != 1 or not np.issubdtype(identifiers.dtype, np.integer):
            raise TypeError("joint_ids must be a rank-1 integer array.")
        identifiers = identifiers.astype(np.int64, copy=False)
        count = identifiers.size
        if np.unique(identifiers).size != count:
            raise ValueError("Breakable rigid-joint IDs must be unique.")
        initiation = _capacity_parameter(initiation_loading, count, "initiation_loading")
        failure = _capacity_parameter(failure_loading, count, "failure_loading")
        fracture = _capacity_parameter(fracture_energy, count, "fracture_energy")
        arming = (
            0.5 * initiation
            if arming_loading is None
            else _capacity_parameter(arming_loading, count, "arming_loading")
        )
        derivative = _capacity_parameter(
            minimum_loading_rate, count, "minimum_loading_rate"
        )
        active = (
            np.ones((count,), dtype=bool)
            if initial_active_mask is None
            else np.asarray(initial_active_mask, dtype=bool)
        )
        if active.shape != (count,):
            raise ValueError("initial_active_mask must have joint-capacity shape.")
        numeric = (initiation, failure, arming, fracture, derivative)
        if (
            any(np.any(~np.isfinite(value)) for value in numeric)
            or np.any(arming < 0.0)
            or np.any(initiation <= arming)
            or np.any(failure <= initiation)
            or np.any(fracture <= 0.0)
            or np.any(derivative <= 0.0)
        ):
            raise ValueError(
                "Damage thresholds, fracture energy, and derivative margins "
                "must be finite and strictly admissible."
            )
        generated = canonical_fingerprint(
            {
                "kind": "breakable-rigid-joint-law-plan",
                "arrays": array_tree_fingerprint(
                    {
                        "joint_ids": identifiers,
                        "initiation": initiation,
                        "failure": failure,
                        "arming": arming,
                        "fracture": fracture,
                        "minimum_loading_rate": derivative,
                        "initial_active": active,
                    }
                ),
            }
        )
        self.joint_ids = jnp.asarray(identifiers, dtype=jnp.int64)
        self.initiation_loading = jnp.asarray(initiation)
        self.failure_loading = jnp.asarray(failure)
        self.arming_loading = jnp.asarray(arming)
        self.fracture_energy = jnp.asarray(fracture)
        self.minimum_loading_rate = jnp.asarray(derivative)
        self.initial_active_mask = jnp.asarray(active, dtype=bool)
        self.plan_id = generated if plan_id is None else str(plan_id)
        if not self.plan_id:
            raise ValueError("plan_id must be nonempty.")

    @property
    def capacity(self) -> int:
        return int(self.joint_ids.shape[0])

    def initialize_state(
        self,
        dtype: np.dtype | None = None,
        /,
        *,
        initial_loading: ArrayLike | None = None,
        first_event_id: int = 0,
    ) -> BreakableRigidJointState:
        dtype_ = self.initiation_loading.dtype if dtype is None else dtype
        loading = (
            jnp.zeros((self.capacity,), dtype=dtype_)
            if initial_loading is None
            else jnp.asarray(initial_loading, dtype=dtype_)
        )
        if loading.shape != (self.capacity,):
            raise ValueError("initial_loading must have joint-capacity shape.")
        active = self.initial_active_mask
        armed = active & jnp.isfinite(loading) & (loading <= self.arming_loading)
        return BreakableRigidJointState(
            active_mask=active,
            damage=jnp.zeros((self.capacity,), dtype=dtype_),
            maximum_loading=jnp.where(jnp.isfinite(loading), loading, 0.0),
            cumulative_fracture_dissipation=jnp.zeros((self.capacity,), dtype=dtype_),
            armed=armed,
            break_step=-jnp.ones((self.capacity,), dtype=jnp.int32),
            break_event_id=-jnp.ones((self.capacity,), dtype=jnp.int64),
            next_event_id=jnp.asarray(first_event_id, dtype=jnp.int64),
            law_id=self.plan_id,
        )


class BreakableRigidJointState(StrictModule):
    active_mask: Array
    damage: Array
    maximum_loading: Array
    cumulative_fracture_dissipation: Array
    armed: Array
    break_step: Array
    break_event_id: Array
    next_event_id: Array
    law_id: str = eqx.field(static=True)


class BreakableRigidJointTransition(StrictModule):
    candidate_state: BreakableRigidJointState
    accepted_state: BreakableRigidJointState
    newly_broken_mask: Array
    ordered_break_event_ids: Array
    arming_guard_margin: Array
    failure_guard_margin: Array
    derivative_guard_margin: Array
    break_guard_margin: Array
    finite_evidence: Array
    valid_evidence: Array
    failure_reasons: Array
    successful: Array


def update_breakable_rigid_joints(
    plan: BreakableRigidJointLawPlan,
    state: BreakableRigidJointState,
    equivalent_loading: ArrayLike,
    loading_derivative: ArrayLike,
    step_index: ArrayLike,
    /,
    *,
    first_event_id: ArrayLike | None = None,
) -> BreakableRigidJointTransition:
    """Propose and atomically accept one irreversible joint-damage update."""

    if not isinstance(plan, BreakableRigidJointLawPlan):
        raise TypeError("plan must be a BreakableRigidJointLawPlan.")
    if not isinstance(state, BreakableRigidJointState):
        raise TypeError("state must be a BreakableRigidJointState.")
    if state.law_id != plan.plan_id:
        raise ValueError("Breakable rigid-joint state belongs to another law.")
    count = plan.capacity
    loading = jnp.asarray(equivalent_loading, dtype=state.damage.dtype)
    derivative = jnp.asarray(loading_derivative, dtype=state.damage.dtype)
    if loading.shape != (count,) or derivative.shape != (count,):
        raise ValueError("Joint loading and derivative must have joint-capacity shape.")
    if (
        state.active_mask.shape != (count,)
        or state.damage.shape != (count,)
        or state.maximum_loading.shape != (count,)
        or state.cumulative_fracture_dissipation.shape != (count,)
        or state.armed.shape != (count,)
        or state.break_step.shape != (count,)
        or state.break_event_id.shape != (count,)
        or state.next_event_id.shape != ()
    ):
        raise ValueError("Breakable rigid-joint state has incompatible fixed shapes.")
    finite_loading = jnp.isfinite(loading)
    finite_derivative = jnp.isfinite(derivative)
    valid_loading_rows = finite_loading & (loading >= 0.0)
    valid_derivative_rows = finite_derivative
    active = state.active_mask
    finite_evidence = jnp.all(~active | (finite_loading & finite_derivative))
    valid_evidence = jnp.all(~active | (valid_loading_rows & valid_derivative_rows))
    safe_loading = jnp.where(valid_loading_rows, loading, state.maximum_loading)
    maximum = jnp.where(
        active, jnp.maximum(state.maximum_loading, safe_loading), state.maximum_loading
    )
    trial_damage = jnp.clip(
        (maximum - plan.initiation_loading)
        / (plan.failure_loading - plan.initiation_loading),
        0.0,
        1.0,
    )
    damage = jnp.where(active, jnp.maximum(state.damage, trial_damage), state.damage)
    increment = damage - state.damage
    dissipation = state.cumulative_fracture_dissipation + increment * plan.fracture_energy
    armed = active & (
        state.armed | (valid_loading_rows & (loading <= plan.arming_loading))
    )
    arming_guard = plan.arming_loading - loading
    failure_guard = loading - plan.failure_loading
    derivative_guard = derivative - plan.minimum_loading_rate
    break_guard = jnp.minimum(failure_guard, derivative_guard)
    newly_broken = (
        active
        & armed
        & valid_loading_rows
        & valid_derivative_rows
        & (damage >= 1.0)
        & (failure_guard >= 0.0)
        & (derivative_guard >= 0.0)
    )
    invalid_key = jnp.asarray(jnp.iinfo(jnp.int64).max, dtype=jnp.int64)
    order = jnp.argsort(jnp.where(newly_broken, plan.joint_ids, invalid_key), stable=True)
    rank_by_row = (
        jnp.zeros((count,), dtype=jnp.int64)
        .at[order]
        .set(jnp.arange(count, dtype=jnp.int64))
    )
    first = (
        state.next_event_id
        if first_event_id is None
        else jnp.asarray(first_event_id, dtype=jnp.int64)
    )
    event_ids = jnp.where(newly_broken, first + rank_by_row, state.break_event_id)
    break_count = jnp.sum(newly_broken, dtype=jnp.int64)
    candidate = BreakableRigidJointState(
        active_mask=active & ~newly_broken,
        damage=damage,
        maximum_loading=maximum,
        cumulative_fracture_dissipation=dissipation,
        armed=armed & ~newly_broken,
        break_step=jnp.where(
            newly_broken,
            jnp.asarray(step_index, dtype=jnp.int32),
            state.break_step,
        ),
        break_event_id=event_ids,
        next_event_id=first + break_count,
        law_id=plan.plan_id,
    )
    invalid_loading = jnp.any(active & ~valid_loading_rows)
    invalid_derivative = jnp.any(active & ~valid_derivative_rows)
    invalid_state = (
        jnp.any(~jnp.isfinite(state.damage))
        | jnp.any(~jnp.isfinite(state.maximum_loading))
        | jnp.any(~jnp.isfinite(state.cumulative_fracture_dissipation))
        | jnp.any(state.damage < 0.0)
        | jnp.any(state.damage > 1.0)
        | jnp.any(state.cumulative_fracture_dissipation < 0.0)
        | (state.next_event_id < 0)
    )
    invalid_guards = jnp.any(
        active
        & ~(
            jnp.isfinite(arming_guard)
            & jnp.isfinite(failure_guard)
            & jnp.isfinite(derivative_guard)
            & jnp.isfinite(break_guard)
        )
    )
    reasons = jnp.asarray(int(RigidTopologyFailure.NONE), dtype=jnp.int32)
    conditions = (
        (invalid_loading, RigidTopologyFailure.INVALID_LOADING),
        (invalid_derivative, RigidTopologyFailure.INVALID_DERIVATIVE),
        (invalid_guards, RigidTopologyFailure.INVALID_GUARD_MARGIN),
        (invalid_state, RigidTopologyFailure.INVALID_STATE),
    )
    for condition, code in conditions:
        reasons = reasons | jnp.where(condition, int(code), 0).astype(jnp.int32)
    successful = reasons == int(RigidTopologyFailure.NONE)
    accepted = _tree_where(successful, candidate, state)
    ordered_ids = jnp.where(
        newly_broken[order], first + jnp.arange(count, dtype=jnp.int64), -1
    )
    return BreakableRigidJointTransition(
        candidate_state=candidate,
        accepted_state=accepted,
        newly_broken_mask=newly_broken,
        ordered_break_event_ids=ordered_ids,
        arming_guard_margin=arming_guard,
        failure_guard_margin=failure_guard,
        derivative_guard_margin=derivative_guard,
        break_guard_margin=break_guard,
        finite_evidence=finite_evidence,
        valid_evidence=valid_evidence,
        failure_reasons=reasons,
        successful=successful,
    )


def _transaction_table(
    value: ArrayLike | None,
    valid: ArrayLike | None,
    transaction_count: int,
    name: str,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    if value is None:
        if valid is not None:
            raise ValueError(f"{name}_valid requires a corresponding ID table.")
        return (
            np.empty((transaction_count, 0), dtype=np.int64),
            np.empty((transaction_count, 0), dtype=bool),
        )
    array = np.asarray(value)
    if array.ndim == 1:
        array = array[:, None]
    if (
        array.ndim != 2
        or array.shape[0] != transaction_count
        or not np.issubdtype(array.dtype, np.integer)
    ):
        raise TypeError(f"{name} must be a rank-2 integer transaction table.")
    array = array.astype(np.int64, copy=False)
    mask = (
        np.ones(array.shape, dtype=bool)
        if valid is None
        else np.asarray(valid, dtype=bool)
    )
    if mask.shape != array.shape:
        raise ValueError(f"{name}_valid must match its ID table shape.")
    for row, row_mask in zip(array, mask, strict=True):
        selected = row[row_mask]
        if np.unique(selected).size != selected.size:
            raise ValueError(f"{name} contains duplicate valid IDs in a transaction.")
    return array, mask


class RigidTopologyPlan(StrictModule, NonTrainableState):
    """Predeclared fixed-capacity rigid topology transaction table."""

    breakable_joints: BreakableRigidJointLawPlan
    transaction_ids: Array
    predecessor_body_ids: Array
    predecessor_body_valid: Array
    successor_body_ids: Array
    successor_body_valid: Array
    deactivated_joint_ids: Array
    deactivated_joint_valid: Array
    activated_joint_ids: Array
    activated_joint_valid: Array
    event_capacity: int = eqx.field(static=True)
    initial_contact_cache_epoch: int = eqx.field(static=True)
    initial_replay_digest: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        breakable_joints: BreakableRigidJointLawPlan,
        transaction_ids: ArrayLike,
        /,
        *,
        predecessor_body_ids: ArrayLike | None = None,
        predecessor_body_valid: ArrayLike | None = None,
        successor_body_ids: ArrayLike | None = None,
        successor_body_valid: ArrayLike | None = None,
        deactivated_joint_ids: ArrayLike | None = None,
        deactivated_joint_valid: ArrayLike | None = None,
        activated_joint_ids: ArrayLike | None = None,
        activated_joint_valid: ArrayLike | None = None,
        event_capacity: int,
        initial_contact_cache_epoch: int = 0,
        initial_replay_digest: int = 0,
        plan_id: str | None = None,
    ):
        if not isinstance(breakable_joints, BreakableRigidJointLawPlan):
            raise TypeError("breakable_joints must be a BreakableRigidJointLawPlan.")
        transactions = np.asarray(transaction_ids)
        if transactions.ndim != 1 or not np.issubdtype(transactions.dtype, np.integer):
            raise TypeError("transaction_ids must be a rank-1 integer array.")
        transactions = transactions.astype(np.int64, copy=False)
        if np.unique(transactions).size != transactions.size:
            raise ValueError("Rigid topology transaction IDs must be unique.")
        count = transactions.size
        predecessors, predecessor_mask = _transaction_table(
            predecessor_body_ids,
            predecessor_body_valid,
            count,
            "predecessor_body_ids",
        )
        successors, successor_mask = _transaction_table(
            successor_body_ids,
            successor_body_valid,
            count,
            "successor_body_ids",
        )
        deactivated, deactivated_mask = _transaction_table(
            deactivated_joint_ids,
            deactivated_joint_valid,
            count,
            "deactivated_joint_ids",
        )
        activated, activated_mask = _transaction_table(
            activated_joint_ids,
            activated_joint_valid,
            count,
            "activated_joint_ids",
        )
        for index in range(count):
            predecessor_set = set(predecessors[index, predecessor_mask[index]].tolist())
            successor_set = set(successors[index, successor_mask[index]].tolist())
            if predecessor_set & successor_set:
                raise ValueError(
                    "A transaction cannot activate and deactivate the same body."
                )
            deactivated_set = set(deactivated[index, deactivated_mask[index]].tolist())
            activated_set = set(activated[index, activated_mask[index]].tolist())
            if deactivated_set & activated_set:
                raise ValueError(
                    "A transaction cannot activate and deactivate the same joint."
                )
        events = int(event_capacity)
        epoch = int(initial_contact_cache_epoch)
        digest = int(initial_replay_digest)
        if events <= 0 or epoch < 0 or digest < 0:
            raise ValueError(
                "Event capacity, contact-cache epoch, and replay digest are invalid."
            )
        generated = canonical_fingerprint(
            {
                "kind": "rigid-topology-plan",
                "breakable_joints": breakable_joints.plan_id,
                "event_capacity": events,
                "initial_contact_cache_epoch": epoch,
                "initial_replay_digest": digest,
                "arrays": array_tree_fingerprint(
                    {
                        "transaction_ids": transactions,
                        "predecessor_body_ids": predecessors,
                        "predecessor_body_valid": predecessor_mask,
                        "successor_body_ids": successors,
                        "successor_body_valid": successor_mask,
                        "deactivated_joint_ids": deactivated,
                        "deactivated_joint_valid": deactivated_mask,
                        "activated_joint_ids": activated,
                        "activated_joint_valid": activated_mask,
                    }
                ),
            }
        )
        self.breakable_joints = breakable_joints
        self.transaction_ids = jnp.asarray(transactions, dtype=jnp.int64)
        self.predecessor_body_ids = jnp.asarray(predecessors, dtype=jnp.int64)
        self.predecessor_body_valid = jnp.asarray(predecessor_mask, dtype=bool)
        self.successor_body_ids = jnp.asarray(successors, dtype=jnp.int64)
        self.successor_body_valid = jnp.asarray(successor_mask, dtype=bool)
        self.deactivated_joint_ids = jnp.asarray(deactivated, dtype=jnp.int64)
        self.deactivated_joint_valid = jnp.asarray(deactivated_mask, dtype=bool)
        self.activated_joint_ids = jnp.asarray(activated, dtype=jnp.int64)
        self.activated_joint_valid = jnp.asarray(activated_mask, dtype=bool)
        self.event_capacity = events
        self.initial_contact_cache_epoch = epoch
        self.initial_replay_digest = digest
        self.plan_id = generated if plan_id is None else str(plan_id)
        if not self.plan_id:
            raise ValueError("plan_id must be nonempty.")

    @property
    def transaction_capacity(self) -> int:
        return int(self.transaction_ids.shape[0])

    def prepare(
        self,
        bodies: PreparedRigidBodySet,
        joints: PreparedRigidJointGraph,
        /,
    ) -> PreparedRigidTopology:
        return PreparedRigidTopology(self, bodies, joints)


def _joint_endpoint_arrays(
    joints: PreparedRigidJointGraph, /
) -> tuple[np.ndarray, np.ndarray]:
    plans = (
        joints.plan.fixed,
        joints.plan.ball,
        joints.plan.hinge,
        joints.plan.prismatic,
        joints.plan.distance,
    )
    left = [
        np.asarray(value.left_body_ids, dtype=np.int64)
        for value in plans
        if value is not None
    ]
    right = [
        np.asarray(value.right_body_ids, dtype=np.int64)
        for value in plans
        if value is not None
    ]
    return (
        np.concatenate(left) if left else np.empty((0,), dtype=np.int64),
        np.concatenate(right) if right else np.empty((0,), dtype=np.int64),
    )


def _map_predeclared_ids(
    values: np.ndarray,
    valid: np.ndarray,
    support_ids: np.ndarray,
    name: str,
    /,
) -> np.ndarray:
    if values.shape != valid.shape:
        raise ValueError(f"{name} ID and validity tables do not match.")
    if values.size == 0:
        return np.zeros(values.shape, dtype=np.int32)
    if support_ids.size == 0:
        if np.any(valid):
            raise ValueError(f"{name} references an empty prepared support.")
        return np.zeros(values.shape, dtype=np.int32)
    sorted_order = np.argsort(support_ids, kind="stable")
    sorted_ids = support_ids[sorted_order]
    safe_values = np.where(valid, values, sorted_ids[0])
    ranks = np.searchsorted(sorted_ids, safe_values)
    clipped = np.clip(ranks, 0, sorted_ids.size - 1)
    matches = (ranks < sorted_ids.size) & (sorted_ids[clipped] == safe_values)
    if not np.all(matches | ~valid):
        raise ValueError(f"{name} contains an ID absent from prepared support.")
    return np.where(valid, sorted_order[clipped], 0).astype(np.int32)


class PreparedRigidTopology(StrictModule, NonTrainableState):
    plan: RigidTopologyPlan
    bodies: PreparedRigidBodySet
    joints: PreparedRigidJointGraph
    joint_ids: Array
    joint_left_body_indices: Array
    joint_right_body_indices: Array
    graph_to_law_indices: Array
    law_to_graph_indices: Array
    predecessor_body_ids: Array
    predecessor_body_indices: Array
    successor_body_ids: Array
    successor_body_indices: Array
    deactivated_joint_indices: Array
    activated_joint_indices: Array
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: RigidTopologyPlan,
        bodies: PreparedRigidBodySet,
        joints: PreparedRigidJointGraph,
        /,
    ):
        if not isinstance(plan, RigidTopologyPlan):
            raise TypeError("plan must be a RigidTopologyPlan.")
        if not isinstance(bodies, PreparedRigidBodySet):
            raise TypeError("bodies must be a PreparedRigidBodySet.")
        if not isinstance(joints, PreparedRigidJointGraph):
            raise TypeError("joints must be a PreparedRigidJointGraph.")
        if joints.bodies.prepared_id != bodies.prepared_id:
            raise ValueError("Rigid joint graph and bodies belong to different supports.")
        joint_ids = np.asarray(joints.row_layout.joint_ids, dtype=np.int64)
        left_ids, right_ids = _joint_endpoint_arrays(joints)
        if left_ids.shape != joint_ids.shape or right_ids.shape != joint_ids.shape:
            raise ValueError("Prepared rigid-joint layout and endpoints disagree.")
        law_ids = np.asarray(plan.breakable_joints.joint_ids, dtype=np.int64)
        if law_ids.size != joint_ids.size or set(law_ids.tolist()) != set(
            joint_ids.tolist()
        ):
            raise ValueError(
                "Breakable-joint IDs must exactly cover the prepared joint graph."
            )
        law_lookup = {int(identifier): index for index, identifier in enumerate(law_ids)}
        graph_to_law = np.asarray(
            [law_lookup[int(identifier)] for identifier in joint_ids], dtype=np.int32
        )
        law_to_graph = np.empty((law_ids.size,), dtype=np.int32)
        law_to_graph[graph_to_law] = np.arange(joint_ids.size, dtype=np.int32)
        body_ids = np.asarray(bodies.particles.particle_ids, dtype=np.int64)
        endpoint_valid = np.ones(left_ids.shape, dtype=bool)
        left_indices = _map_predeclared_ids(
            left_ids, endpoint_valid, body_ids, "joint left endpoints"
        )
        right_indices = _map_predeclared_ids(
            right_ids, endpoint_valid, body_ids, "joint right endpoints"
        )
        predecessor_indices = _map_predeclared_ids(
            np.asarray(plan.predecessor_body_ids),
            np.asarray(plan.predecessor_body_valid),
            body_ids,
            "predecessor bodies",
        )
        successor_indices = _map_predeclared_ids(
            np.asarray(plan.successor_body_ids),
            np.asarray(plan.successor_body_valid),
            body_ids,
            "successor bodies",
        )
        deactivated_indices = _map_predeclared_ids(
            np.asarray(plan.deactivated_joint_ids),
            np.asarray(plan.deactivated_joint_valid),
            joint_ids,
            "deactivated joints",
        )
        activated_indices = _map_predeclared_ids(
            np.asarray(plan.activated_joint_ids),
            np.asarray(plan.activated_joint_valid),
            joint_ids,
            "activated joints",
        )
        initial_graph_active = np.asarray(plan.breakable_joints.initial_active_mask)[
            graph_to_law
        ]
        initial_body_active = np.asarray(bodies.particles.active_mask, dtype=bool)
        if np.any(
            initial_graph_active
            & ~(initial_body_active[left_indices] & initial_body_active[right_indices])
        ):
            raise ValueError("Initially active rigid joints require active endpoints.")
        self.plan = plan
        self.bodies = bodies
        self.joints = joints
        self.joint_ids = joints.row_layout.joint_ids
        self.joint_left_body_indices = jnp.asarray(left_indices, dtype=jnp.int32)
        self.joint_right_body_indices = jnp.asarray(right_indices, dtype=jnp.int32)
        self.graph_to_law_indices = jnp.asarray(graph_to_law, dtype=jnp.int32)
        self.law_to_graph_indices = jnp.asarray(law_to_graph, dtype=jnp.int32)
        self.predecessor_body_ids = plan.predecessor_body_ids
        self.predecessor_body_indices = jnp.asarray(predecessor_indices, dtype=jnp.int32)
        self.successor_body_ids = plan.successor_body_ids
        self.successor_body_indices = jnp.asarray(successor_indices, dtype=jnp.int32)
        self.deactivated_joint_indices = jnp.asarray(deactivated_indices, dtype=jnp.int32)
        self.activated_joint_indices = jnp.asarray(activated_indices, dtype=jnp.int32)
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-rigid-topology",
                "plan": plan.plan_id,
                "bodies": bodies.prepared_id,
                "joints": joints.prepared_id,
                "row_layout": joints.row_layout.layout_id,
            }
        )

    @property
    def joint_capacity(self) -> int:
        return self.joints.row_layout.joint_count

    @property
    def constraint_row_capacity(self) -> int:
        return self.joints.row_layout.row_count

    @property
    def maximum_proposal_count(self) -> int:
        plan = self.plan
        return int(
            self.joint_capacity
            + plan.predecessor_body_ids.size
            + plan.successor_body_ids.size
            + plan.deactivated_joint_ids.size
            + plan.activated_joint_ids.size
        )

    def initialize_state(
        self,
        dtype: np.dtype | None = None,
        /,
        *,
        initial_loading: ArrayLike | None = None,
    ) -> RigidTopologyState:
        dtype_ = self.bodies.particles.safe_masses.dtype if dtype is None else dtype
        joint_state = self.plan.breakable_joints.initialize_state(
            dtype_, initial_loading=initial_loading, first_event_id=0
        )
        events = self.plan.event_capacity
        journal = RigidTopologyEventJournal(
            event_ids=-jnp.ones((events,), dtype=jnp.int64),
            event_kinds=jnp.zeros((events,), dtype=jnp.int8),
            entity_ids=-jnp.ones((events,), dtype=jnp.int64),
            transaction_ids=-jnp.ones((events,), dtype=jnp.int64),
            step_indices=-jnp.ones((events,), dtype=jnp.int32),
            guard_margins=jnp.zeros((events,), dtype=dtype_),
            predecessor_ids=-jnp.ones((events,), dtype=jnp.int64),
            successor_ids=-jnp.ones((events,), dtype=jnp.int64),
            valid=jnp.zeros((events,), dtype=bool),
            prepared_id=self.prepared_id,
        )
        return RigidTopologyState(
            body_active_mask=self.bodies.particles.active_mask,
            joint_state=joint_state,
            journal=journal,
            contact_cache_epoch=jnp.asarray(
                self.plan.initial_contact_cache_epoch, dtype=jnp.int64
            ),
            replay_digest=jnp.asarray(self.plan.initial_replay_digest, dtype=jnp.int64),
            accepted_transition_count=jnp.zeros((), dtype=jnp.int64),
            prepared_id=self.prepared_id,
        )

    def proposal(
        self,
        requested_transactions: ArrayLike,
        expected_replay_digest: ArrayLike,
        /,
    ) -> RigidTopologyProposal:
        requested = jnp.asarray(requested_transactions, dtype=bool)
        if requested.shape != (self.plan.transaction_capacity,):
            raise ValueError(
                "requested_transactions must have transaction-capacity shape."
            )
        digest = jnp.asarray(expected_replay_digest, dtype=jnp.int64)
        if digest.shape != ():
            raise ValueError("expected_replay_digest must be scalar.")
        return RigidTopologyProposal(requested, digest, self.prepared_id)

    def dual_gauge(
        self, joint_active_mask: ArrayLike, dtype: np.dtype | None = None, /
    ) -> InactiveRigidJointDualGauge:
        active = jnp.asarray(joint_active_mask, dtype=bool)
        if active.shape != self.joints.row_layout.joint_ids.shape:
            raise ValueError("joint_active_mask must have prepared joint capacity.")
        dtype_ = self.bodies.particles.safe_masses.dtype if dtype is None else dtype
        row_active = self.joints.row_layout.row_active(active)
        inactive = ~row_active
        return InactiveRigidJointDualGauge(
            row_layout=self.joints.row_layout,
            joint_active_mask=active,
            row_active_mask=row_active,
            inactive_row_mask=inactive,
            gauge_diagonal=inactive.astype(dtype_),
            gauge_rhs=jnp.zeros((self.constraint_row_capacity,), dtype=dtype_),
            finite_evidence=jnp.asarray(True),
            prepared_id=self.prepared_id,
        )


class RigidTopologyEventJournal(StrictModule):
    event_ids: Array
    event_kinds: Array
    entity_ids: Array
    transaction_ids: Array
    step_indices: Array
    guard_margins: Array
    predecessor_ids: Array
    successor_ids: Array
    valid: Array
    prepared_id: str = eqx.field(static=True)


class RigidTopologyState(StrictModule):
    body_active_mask: Array
    joint_state: BreakableRigidJointState
    journal: RigidTopologyEventJournal
    contact_cache_epoch: Array
    replay_digest: Array
    accepted_transition_count: Array
    prepared_id: str = eqx.field(static=True)


class RigidTopologyProposal(StrictModule):
    requested_transactions: Array
    expected_replay_digest: Array
    prepared_id: str = eqx.field(static=True)


class InactiveRigidJointDualGauge(StrictModule):
    """Explicit equations ``lambda[row] = 0`` for inactive joint rows."""

    row_layout: RigidJointRowLayout
    joint_active_mask: Array
    row_active_mask: Array
    inactive_row_mask: Array
    gauge_diagonal: Array
    gauge_rhs: Array
    finite_evidence: Array
    prepared_id: str = eqx.field(static=True)


class RigidTopologyEventBatch(StrictModule):
    event_kinds: Array
    entity_ids: Array
    transaction_ids: Array
    guard_margins: Array
    predecessor_ids: Array
    successor_ids: Array
    valid: Array


class RigidTopologyRejectionEvidence(StrictModule):
    failure_reasons: Array
    replay_digest_mismatch: Array
    invalid_loading: Array
    invalid_derivative: Array
    event_capacity_overflow: Array
    proposal_conflict: Array
    precondition_failed: Array
    inactive_joint_endpoint: Array
    invalid_guard_margin: Array
    invalid_state: Array


class RigidTopologyTransition(StrictModule):
    candidate_state: RigidTopologyState
    accepted_state: RigidTopologyState
    proposed_events: RigidTopologyEventBatch
    newly_broken_mask: Array
    body_activation_mask: Array
    body_deactivation_mask: Array
    joint_activation_mask: Array
    joint_deactivation_mask: Array
    multiplier_reset_joint_mask: Array
    multiplier_reset_row_mask: Array
    dual_gauge: InactiveRigidJointDualGauge
    transaction_guard_margins: Array
    rejection: RigidTopologyRejectionEvidence
    finite_evidence: Array
    valid_evidence: Array
    successful: Array
    prepared_id: str = eqx.field(static=True)


def _scatter_counts(
    indices: Array,
    valid: Array,
    requested: Array,
    capacity: int,
    /,
) -> Array:
    if indices.size == 0:
        return jnp.zeros((capacity,), dtype=jnp.int32)
    selected = valid & requested[:, None]
    safe = jnp.where(selected, indices, 0).reshape(-1)
    contribution = selected.astype(jnp.int32).reshape(-1)
    return jnp.zeros((capacity,), dtype=jnp.int32).at[safe].add(contribution)


def _table_precondition(
    current: Array,
    indices: Array,
    valid: Array,
    requested: Array,
    required_value: bool,
    /,
) -> tuple[Array, Array]:
    if indices.size == 0:
        return jnp.asarray(True), jnp.ones(requested.shape, dtype=jnp.float32)
    selected = valid & requested[:, None]
    observed = current[indices]
    satisfied = observed if required_value else ~observed
    row_ok = jnp.all(~selected | satisfied, axis=1)
    row_margin = jnp.min(
        jnp.where(selected, jnp.where(satisfied, 1.0, -1.0), jnp.inf), axis=1
    )
    row_margin = jnp.where(jnp.any(selected, axis=1), row_margin, 1.0)
    return jnp.all(row_ok | ~requested), row_margin


def _event_columns(
    prepared: PreparedRigidTopology,
    break_transition: BreakableRigidJointTransition,
    requested: Array,
    transaction_guards: Array,
    /,
) -> tuple[RigidTopologyEventBatch, Array]:
    plan = prepared.plan
    graph_break = break_transition.newly_broken_mask[prepared.graph_to_law_indices]
    joint_count = prepared.joint_capacity

    def flattened(
        identifiers: Array,
        valid: Array,
        kind: RigidTopologyEventKind,
        guards: Array,
        source: Array,
        activation: bool,
        /,
    ) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array]:
        shape = identifiers.shape
        transaction = jnp.broadcast_to(plan.transaction_ids[:, None], shape)
        selected = valid & jnp.broadcast_to(requested[:, None], shape)
        guard = jnp.broadcast_to(guards[:, None], shape)
        kinds = jnp.full(shape, int(kind), dtype=jnp.int8)
        predecessor = jnp.where(activation, -1, identifiers)
        successor = jnp.where(activation, identifiers, -1)
        return tuple(
            value.reshape(-1)
            for value in (
                kinds,
                identifiers,
                transaction,
                guard,
                predecessor,
                successor,
                selected,
                source,
            )
        )

    break_columns = (
        jnp.full((joint_count,), int(RigidTopologyEventKind.JOINT_BREAK), dtype=jnp.int8),
        prepared.joint_ids,
        -jnp.ones((joint_count,), dtype=jnp.int64),
        break_transition.break_guard_margin[prepared.graph_to_law_indices],
        prepared.joint_ids,
        -jnp.ones((joint_count,), dtype=jnp.int64),
        graph_break,
        jnp.arange(joint_count, dtype=jnp.int32),
    )
    columns = (
        break_columns,
        flattened(
            plan.predecessor_body_ids,
            plan.predecessor_body_valid,
            RigidTopologyEventKind.BODY_DEACTIVATION,
            transaction_guards,
            -jnp.ones(plan.predecessor_body_ids.shape, dtype=jnp.int32),
            False,
        ),
        flattened(
            plan.successor_body_ids,
            plan.successor_body_valid,
            RigidTopologyEventKind.BODY_ACTIVATION,
            transaction_guards,
            -jnp.ones(plan.successor_body_ids.shape, dtype=jnp.int32),
            True,
        ),
        flattened(
            plan.deactivated_joint_ids,
            plan.deactivated_joint_valid,
            RigidTopologyEventKind.JOINT_DEACTIVATION,
            transaction_guards,
            -jnp.ones(plan.deactivated_joint_ids.shape, dtype=jnp.int32),
            False,
        ),
        flattened(
            plan.activated_joint_ids,
            plan.activated_joint_valid,
            RigidTopologyEventKind.JOINT_ACTIVATION,
            transaction_guards,
            -jnp.ones(plan.activated_joint_ids.shape, dtype=jnp.int32),
            True,
        ),
    )
    concatenated = tuple(
        jnp.concatenate(tuple(group[index] for group in columns), axis=0)
        for index in range(8)
    )
    kinds, entity, transaction, guard, predecessor, successor, valid, source = (
        concatenated
    )
    invalid_id = jnp.asarray(jnp.iinfo(jnp.int64).max, dtype=jnp.int64)
    invalid_kind = jnp.asarray(jnp.iinfo(jnp.int8).max, dtype=jnp.int8)
    declaration = jnp.arange(entity.shape[0], dtype=jnp.int32)
    order = jnp.lexsort(
        (
            declaration,
            jnp.where(valid, transaction, invalid_id),
            jnp.where(valid, kinds, invalid_kind),
            jnp.where(valid, entity, invalid_id),
        )
    )
    return (
        RigidTopologyEventBatch(
            event_kinds=kinds[order],
            entity_ids=entity[order],
            transaction_ids=transaction[order],
            guard_margins=guard[order],
            predecessor_ids=predecessor[order],
            successor_ids=successor[order],
            valid=valid[order],
        ),
        source[order],
    )


def _append_event_journal(
    journal: RigidTopologyEventJournal,
    events: RigidTopologyEventBatch,
    first_event_id: Array,
    step_index: Array,
    /,
) -> RigidTopologyEventJournal:
    used = jnp.sum(journal.valid, dtype=jnp.int32)
    rank = jnp.cumsum(events.valid.astype(jnp.int32)) - 1
    capacity = journal.valid.shape[0]

    def append_one(index: int, current: RigidTopologyEventJournal):
        target = used + rank[index]
        safe_target = jnp.clip(target, 0, capacity - 1)
        event_id = first_event_id + rank[index].astype(jnp.int64)
        should_write = events.valid[index] & (target < capacity)

        def write_event(value: RigidTopologyEventJournal):
            return RigidTopologyEventJournal(
                event_ids=value.event_ids.at[safe_target].set(event_id),
                event_kinds=value.event_kinds.at[safe_target].set(
                    events.event_kinds[index]
                ),
                entity_ids=value.entity_ids.at[safe_target].set(events.entity_ids[index]),
                transaction_ids=value.transaction_ids.at[safe_target].set(
                    events.transaction_ids[index]
                ),
                step_indices=value.step_indices.at[safe_target].set(step_index),
                guard_margins=value.guard_margins.at[safe_target].set(
                    events.guard_margins[index]
                ),
                predecessor_ids=value.predecessor_ids.at[safe_target].set(
                    events.predecessor_ids[index]
                ),
                successor_ids=value.successor_ids.at[safe_target].set(
                    events.successor_ids[index]
                ),
                valid=value.valid.at[safe_target].set(True),
                prepared_id=value.prepared_id,
            )

        return jax.lax.cond(should_write, write_event, lambda value: value, current)

    return jax.lax.fori_loop(0, events.valid.shape[0], append_one, journal)


def _event_replay_digest(
    initial_digest: Array,
    first_event_id: Array,
    events: RigidTopologyEventBatch,
    step_index: Array,
    /,
) -> Array:
    rank = jnp.cumsum(events.valid.astype(jnp.int64)) - 1
    prime = jnp.asarray(1_099_511_628_211, dtype=jnp.int64)
    salt = jnp.asarray(1_461_466_560_413, dtype=jnp.int64)
    positive_mask = jnp.asarray(jnp.iinfo(jnp.int64).max, dtype=jnp.int64)

    def mix(index: int, digest: Array) -> Array:
        event_id = first_event_id + rank[index]
        token = (
            (event_id + 1) * jnp.asarray(1_000_003, dtype=jnp.int64)
            + events.entity_ids[index] * jnp.asarray(97, dtype=jnp.int64)
            + events.transaction_ids[index] * jnp.asarray(193, dtype=jnp.int64)
            + events.event_kinds[index].astype(jnp.int64)
            * jnp.asarray(389, dtype=jnp.int64)
            + jnp.asarray(step_index, dtype=jnp.int64) * jnp.asarray(769, dtype=jnp.int64)
        )
        updated = ((digest ^ (token + salt)) * prime) & positive_mask
        return jnp.where(events.valid[index], updated, digest)

    return jax.lax.fori_loop(0, events.valid.shape[0], mix, initial_digest)


def apply_rigid_topology_transactions(
    prepared: PreparedRigidTopology,
    state: RigidTopologyState,
    proposal: RigidTopologyProposal,
    equivalent_loading: ArrayLike,
    loading_derivative: ArrayLike,
    step_index: ArrayLike,
    /,
) -> RigidTopologyTransition:
    """Evaluate one deterministic all-or-nothing composite topology transition."""

    if not isinstance(prepared, PreparedRigidTopology):
        raise TypeError("prepared must be a PreparedRigidTopology.")
    if not isinstance(state, RigidTopologyState):
        raise TypeError("state must be a RigidTopologyState.")
    if not isinstance(proposal, RigidTopologyProposal):
        raise TypeError("proposal must be a RigidTopologyProposal.")
    if state.prepared_id != prepared.prepared_id:
        raise ValueError("Rigid topology state belongs to another prepared topology.")
    if proposal.prepared_id != prepared.prepared_id:
        raise ValueError("Rigid topology proposal belongs to another prepared topology.")
    if state.journal.prepared_id != prepared.prepared_id:
        raise ValueError("Rigid topology journal belongs to another prepared topology.")
    if state.joint_state.law_id != prepared.plan.breakable_joints.plan_id:
        raise ValueError("Rigid topology joint state belongs to another damage law.")
    if proposal.requested_transactions.shape != (prepared.plan.transaction_capacity,):
        raise ValueError("Rigid topology proposal has incompatible fixed shape.")
    if proposal.expected_replay_digest.shape != ():
        raise ValueError("Rigid topology replay digest must be scalar.")
    if state.body_active_mask.shape != (prepared.bodies.capacity,):
        raise ValueError("Rigid topology body mask has incompatible fixed shape.")
    if state.journal.valid.shape != (prepared.plan.event_capacity,):
        raise ValueError("Rigid topology journal has incompatible fixed capacity.")

    requested = proposal.requested_transactions
    current_body = state.body_active_mask
    current_joint = state.joint_state.active_mask[prepared.graph_to_law_indices]
    first_event_id = state.joint_state.next_event_id
    break_transition = update_breakable_rigid_joints(
        prepared.plan.breakable_joints,
        state.joint_state,
        equivalent_loading,
        loading_derivative,
        step_index,
        first_event_id=first_event_id,
    )
    broken_graph = break_transition.newly_broken_mask[prepared.graph_to_law_indices]
    predecessor_count = _scatter_counts(
        prepared.predecessor_body_indices,
        prepared.plan.predecessor_body_valid,
        requested,
        prepared.bodies.capacity,
    )
    successor_count = _scatter_counts(
        prepared.successor_body_indices,
        prepared.plan.successor_body_valid,
        requested,
        prepared.bodies.capacity,
    )
    deactivated_count = _scatter_counts(
        prepared.deactivated_joint_indices,
        prepared.plan.deactivated_joint_valid,
        requested,
        prepared.joint_capacity,
    )
    activated_count = _scatter_counts(
        prepared.activated_joint_indices,
        prepared.plan.activated_joint_valid,
        requested,
        prepared.joint_capacity,
    )
    body_deactivation = predecessor_count > 0
    body_activation = successor_count > 0
    declared_joint_deactivation = deactivated_count > 0
    joint_activation = activated_count > 0
    joint_deactivation = broken_graph | declared_joint_deactivation
    predecessor_ok, predecessor_margin = _table_precondition(
        current_body,
        prepared.predecessor_body_indices,
        prepared.plan.predecessor_body_valid,
        requested,
        True,
    )
    successor_ok, successor_margin = _table_precondition(
        current_body,
        prepared.successor_body_indices,
        prepared.plan.successor_body_valid,
        requested,
        False,
    )
    deactivated_ok, deactivated_margin = _table_precondition(
        current_joint,
        prepared.deactivated_joint_indices,
        prepared.plan.deactivated_joint_valid,
        requested,
        True,
    )
    activated_ok, activated_margin = _table_precondition(
        current_joint,
        prepared.activated_joint_indices,
        prepared.plan.activated_joint_valid,
        requested,
        False,
    )
    transaction_guards = jnp.minimum(
        jnp.minimum(predecessor_margin, successor_margin),
        jnp.minimum(deactivated_margin, activated_margin),
    ).astype(state.joint_state.damage.dtype)
    precondition_failed = ~(predecessor_ok & successor_ok & deactivated_ok & activated_ok)
    broken_law = state.joint_state.break_step >= 0
    activation_of_broken = jnp.any(
        joint_activation[prepared.law_to_graph_indices] & broken_law
    )
    proposal_conflict = (
        jnp.any(predecessor_count > 1)
        | jnp.any(successor_count > 1)
        | jnp.any(deactivated_count > 1)
        | jnp.any(activated_count > 1)
        | jnp.any(body_deactivation & body_activation)
        | jnp.any(joint_deactivation & joint_activation)
        | jnp.any(broken_graph & declared_joint_deactivation)
    )
    precondition_failed = precondition_failed | activation_of_broken
    candidate_body = (current_body & ~body_deactivation) | body_activation
    candidate_joint = (current_joint & ~joint_deactivation) | joint_activation
    endpoint_active = (
        candidate_body[prepared.joint_left_body_indices]
        & candidate_body[prepared.joint_right_body_indices]
    )
    inactive_joint_endpoint = jnp.any(candidate_joint & ~endpoint_active)
    proposed_events, event_source_joint = _event_columns(
        prepared, break_transition, requested, transaction_guards
    )
    proposed_count = jnp.sum(proposed_events.valid, dtype=jnp.int32)
    used_events = jnp.sum(state.journal.valid, dtype=jnp.int32)
    event_capacity_overflow = used_events + proposed_count > prepared.plan.event_capacity
    replay_digest_mismatch = proposal.expected_replay_digest != state.replay_digest
    guard_finite = jnp.all(
        ~proposed_events.valid | jnp.isfinite(proposed_events.guard_margins)
    ) & jnp.all(jnp.isfinite(transaction_guards))
    invalid_guard_margin = ~guard_finite
    journal_packed = ~jnp.any(state.journal.valid[1:] & ~state.journal.valid[:-1])
    current_endpoint_active = (
        current_body[prepared.joint_left_body_indices]
        & current_body[prepared.joint_right_body_indices]
    )
    invalid_state = (
        ~journal_packed
        | jnp.any(current_joint & ~current_endpoint_active)
        | (state.contact_cache_epoch < 0)
        | (state.replay_digest < 0)
        | (state.accepted_transition_count < 0)
        | (state.joint_state.next_event_id < 0)
    )
    invalid_loading = (
        break_transition.failure_reasons & int(RigidTopologyFailure.INVALID_LOADING)
    ) != 0
    invalid_derivative = (
        break_transition.failure_reasons & int(RigidTopologyFailure.INVALID_DERIVATIVE)
    ) != 0
    invalid_state = invalid_state | (
        (break_transition.failure_reasons & int(RigidTopologyFailure.INVALID_STATE)) != 0
    )
    reasons = jnp.asarray(int(RigidTopologyFailure.NONE), dtype=jnp.int32)
    conditions = (
        (replay_digest_mismatch, RigidTopologyFailure.REPLAY_DIGEST_MISMATCH),
        (invalid_loading, RigidTopologyFailure.INVALID_LOADING),
        (invalid_derivative, RigidTopologyFailure.INVALID_DERIVATIVE),
        (event_capacity_overflow, RigidTopologyFailure.EVENT_CAPACITY_OVERFLOW),
        (proposal_conflict, RigidTopologyFailure.PROPOSAL_CONFLICT),
        (precondition_failed, RigidTopologyFailure.PRECONDITION_FAILED),
        (inactive_joint_endpoint, RigidTopologyFailure.INACTIVE_JOINT_ENDPOINT),
        (invalid_guard_margin, RigidTopologyFailure.INVALID_GUARD_MARGIN),
        (invalid_state, RigidTopologyFailure.INVALID_STATE),
    )
    for condition, code in conditions:
        reasons = reasons | jnp.where(condition, int(code), 0).astype(jnp.int32)
    successful = reasons == int(RigidTopologyFailure.NONE)
    candidate_journal = _append_event_journal(
        state.journal,
        proposed_events,
        first_event_id,
        jnp.asarray(step_index, dtype=jnp.int32),
    )
    rank = jnp.cumsum(proposed_events.valid.astype(jnp.int64)) - 1
    proposed_event_ids = first_event_id + rank
    break_event_graph = state.joint_state.break_event_id[prepared.graph_to_law_indices]

    def assign_break_event(index: int, values: Array) -> Array:
        if prepared.joint_capacity == 0:
            return values
        source = event_source_joint[index]
        safe_source = jnp.clip(source, 0, prepared.joint_capacity - 1)
        is_break = (
            proposed_events.valid[index]
            & (
                proposed_events.event_kinds[index]
                == int(RigidTopologyEventKind.JOINT_BREAK)
            )
            & (source >= 0)
        )
        return jax.lax.cond(
            is_break,
            lambda array: array.at[safe_source].set(proposed_event_ids[index]),
            lambda array: array,
            values,
        )

    break_event_graph = jax.lax.fori_loop(
        0,
        proposed_events.valid.shape[0],
        assign_break_event,
        break_event_graph,
    )
    candidate_law_active = candidate_joint[prepared.law_to_graph_indices]
    candidate_joint_state = BreakableRigidJointState(
        active_mask=candidate_law_active,
        damage=break_transition.candidate_state.damage,
        maximum_loading=break_transition.candidate_state.maximum_loading,
        cumulative_fracture_dissipation=(
            break_transition.candidate_state.cumulative_fracture_dissipation
        ),
        armed=break_transition.candidate_state.armed & candidate_law_active,
        break_step=break_transition.candidate_state.break_step,
        break_event_id=break_event_graph[prepared.law_to_graph_indices],
        next_event_id=first_event_id + proposed_count.astype(jnp.int64),
        law_id=prepared.plan.breakable_joints.plan_id,
    )
    replay_digest = _event_replay_digest(
        state.replay_digest,
        first_event_id,
        proposed_events,
        jnp.asarray(step_index, dtype=jnp.int32),
    )
    topology_changed = jnp.any(body_activation | body_deactivation) | jnp.any(
        candidate_joint != current_joint
    )
    candidate_state = RigidTopologyState(
        body_active_mask=candidate_body,
        joint_state=candidate_joint_state,
        journal=candidate_journal,
        contact_cache_epoch=state.contact_cache_epoch
        + topology_changed.astype(jnp.int64),
        replay_digest=replay_digest,
        accepted_transition_count=state.accepted_transition_count
        + jnp.asarray(1, dtype=jnp.int64),
        prepared_id=prepared.prepared_id,
    )
    accepted_state = _tree_where(successful, candidate_state, state)
    accepted_joint = accepted_state.joint_state.active_mask[prepared.graph_to_law_indices]
    changed_body = body_activation | body_deactivation
    incident_changed_body = (
        changed_body[prepared.joint_left_body_indices]
        | changed_body[prepared.joint_right_body_indices]
    )
    reset_joint = (
        (candidate_joint != current_joint) | incident_changed_body
    ) & successful
    reset_rows = prepared.joints.row_layout.row_active(reset_joint)
    dual_gauge = prepared.dual_gauge(accepted_joint, state.joint_state.damage.dtype)
    rejection = RigidTopologyRejectionEvidence(
        failure_reasons=reasons,
        replay_digest_mismatch=replay_digest_mismatch,
        invalid_loading=invalid_loading,
        invalid_derivative=invalid_derivative,
        event_capacity_overflow=event_capacity_overflow,
        proposal_conflict=proposal_conflict,
        precondition_failed=precondition_failed,
        inactive_joint_endpoint=inactive_joint_endpoint,
        invalid_guard_margin=invalid_guard_margin,
        invalid_state=invalid_state,
    )
    finite_evidence = break_transition.finite_evidence & guard_finite
    valid_evidence = break_transition.valid_evidence & ~invalid_state
    return RigidTopologyTransition(
        candidate_state=candidate_state,
        accepted_state=accepted_state,
        proposed_events=proposed_events,
        newly_broken_mask=broken_graph,
        body_activation_mask=body_activation,
        body_deactivation_mask=body_deactivation,
        joint_activation_mask=joint_activation,
        joint_deactivation_mask=joint_deactivation,
        multiplier_reset_joint_mask=reset_joint,
        multiplier_reset_row_mask=reset_rows,
        dual_gauge=dual_gauge,
        transaction_guard_margins=transaction_guards,
        rejection=rejection,
        finite_evidence=finite_evidence,
        valid_evidence=valid_evidence,
        successful=successful,
        prepared_id=prepared.prepared_id,
    )


def _tree_where(condition: Array, proposed, current, /):
    return jax.tree.map(
        lambda new, old: jnp.where(condition, new, old), proposed, current
    )


__all__ = [
    "BreakableRigidJointLawPlan",
    "BreakableRigidJointState",
    "BreakableRigidJointTransition",
    "InactiveRigidJointDualGauge",
    "PreparedRigidTopology",
    "RigidTopologyEventBatch",
    "RigidTopologyEventJournal",
    "RigidTopologyEventKind",
    "RigidTopologyFailure",
    "RigidTopologyPlan",
    "RigidTopologyProposal",
    "RigidTopologyRejectionEvidence",
    "RigidTopologyState",
    "RigidTopologyTransition",
    "apply_rigid_topology_transactions",
    "update_breakable_rigid_joints",
]
