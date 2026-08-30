#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._rigid_body import quaternion_rotation_matrix


class TopologyEventPlan(StrictModule, NonTrainableState):
    owner_capacity: int = eqx.field(static=True)
    maximum_children: int = eqx.field(static=True)
    event_capacity: int = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        owner_capacity: int,
        maximum_children: int,
        event_capacity: int,
        dimension: int,
        /,
    ):
        owner = int(owner_capacity)
        children = int(maximum_children)
        events = int(event_capacity)
        dimension_ = int(dimension)
        if owner <= 0 or children < 2 or events <= 0 or dimension_ not in (2, 3):
            raise ValueError("Topology-event capacities/dimension are invalid.")
        self.owner_capacity = owner
        self.maximum_children = children
        self.event_capacity = events
        self.dimension = dimension_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "topology-event-plan",
                "owner_capacity": owner,
                "maximum_children": children,
                "event_capacity": events,
                "dimension": dimension_,
            }
        )


class TopologyPoolState(StrictModule):
    stable_ids: Array
    active: Array
    parent_ids: Array
    position: Array
    velocity: Array
    orientation: Array
    angular_velocity: Array
    mass: Array
    inertia_body: Array
    next_event_id: Array


class TopologyEventRecord(StrictModule):
    event_ids: Array
    valid: Array
    step_index: Array
    time: Array
    source_owner: Array
    child_owners: Array
    child_valid: Array
    cause_bond_id: Array
    mass_residual: Array
    linear_momentum_residual: Array
    angular_momentum_residual: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class TopologyEventResult(StrictModule):
    candidate_state: TopologyPoolState
    accepted_state: TopologyPoolState
    record: TopologyEventRecord
    successful: Array


def initialize_topology_event_record(
    plan: TopologyEventPlan, dtype: np.dtype = np.float64, /
) -> TopologyEventRecord:
    events = plan.event_capacity
    return TopologyEventRecord(
        -jnp.ones((events,), dtype=jnp.int64),
        jnp.zeros((events,), dtype=bool),
        -jnp.ones((events,), dtype=jnp.int32),
        jnp.zeros((events,), dtype=dtype),
        -jnp.ones((events,), dtype=jnp.int32),
        -jnp.ones((events, plan.maximum_children), dtype=jnp.int32),
        jnp.zeros((events, plan.maximum_children), dtype=bool),
        -jnp.ones((events,), dtype=jnp.int64),
        jnp.zeros((events,), dtype=dtype),
        jnp.zeros((events, plan.dimension), dtype=dtype),
        jnp.zeros((events, 1 if plan.dimension == 2 else 3), dtype=dtype),
        jnp.asarray(True),
        plan.plan_id,
    )


def split_preallocated_owner(
    plan: TopologyEventPlan,
    state: TopologyPoolState,
    record: TopologyEventRecord,
    source_owner: Array,
    child_owners: ArrayLike,
    child_valid: ArrayLike,
    child_mass: ArrayLike,
    child_offset: ArrayLike,
    child_inertia_body: ArrayLike,
    step_index: Array,
    time: Array,
    /,
    *,
    cause_bond_id: ArrayLike = -1,
    tolerance: float = 1.0e-10,
) -> TopologyEventResult:
    if record.plan_id != plan.plan_id:
        raise ValueError("Topology event record does not match plan.")
    source = jnp.asarray(source_owner, dtype=jnp.int32)
    children = jnp.asarray(child_owners, dtype=jnp.int32)
    valid_children = jnp.asarray(child_valid, dtype=bool)
    masses = jnp.asarray(child_mass, dtype=state.mass.dtype)
    offsets = jnp.asarray(child_offset, dtype=state.position.dtype)
    inertias = jnp.asarray(child_inertia_body, dtype=state.inertia_body.dtype)
    if (
        children.shape != (plan.maximum_children,)
        or valid_children.shape != children.shape
        or masses.shape != children.shape
        or offsets.shape != (plan.maximum_children, plan.dimension)
    ):
        raise ValueError("Topology split child arrays do not match plan capacities.")
    expected_inertia_shape = (
        (plan.maximum_children,) if plan.dimension == 2 else (plan.maximum_children, 3, 3)
    )
    if inertias.shape != expected_inertia_shape:
        raise ValueError("Child inertia shape does not match topology dimension.")
    safe_children = jnp.where(valid_children, children, 0)
    child_slots_valid = (
        (children >= 0) & (children < plan.owner_capacity) & valid_children
    )
    sorted_children = jnp.sort(jnp.where(valid_children, children, plan.owner_capacity))
    no_duplicate_children = ~jnp.any(
        (sorted_children[1:] == sorted_children[:-1])
        & (sorted_children[1:] < plan.owner_capacity)
    )
    event_slot = jnp.sum(record.valid, dtype=jnp.int32)
    capacity_ok = event_slot < plan.event_capacity
    source_active = state.active[source]
    children_inactive = jnp.all(~state.active[safe_children] | ~valid_children)
    mass_residual = jnp.sum(jnp.where(valid_children, masses, 0.0)) - state.mass[source]
    mass_ok = jnp.abs(mass_residual) <= tolerance * jnp.maximum(state.mass[source], 1.0)
    if plan.dimension == 2:
        angle = state.orientation[source, 0]
        cosine = jnp.cos(angle)
        sine = jnp.sin(angle)
        world_offset = jnp.stack(
            (
                cosine * offsets[:, 0] - sine * offsets[:, 1],
                sine * offsets[:, 0] + cosine * offsets[:, 1],
            ),
            axis=-1,
        )
        omega = state.angular_velocity[source, 0]
        child_velocity = state.velocity[source] + jnp.stack(
            (-omega * world_offset[:, 1], omega * world_offset[:, 0]), axis=-1
        )
        child_orientation = jnp.broadcast_to(
            state.orientation[source], (plan.maximum_children, 1)
        )
        child_angular = jnp.broadcast_to(
            state.angular_velocity[source], (plan.maximum_children, 1)
        )
    else:
        rotation = quaternion_rotation_matrix(state.orientation[source : source + 1])[0]
        world_offset = contract("ij,kj->ki", rotation, offsets)
        child_velocity = state.velocity[source] + jnp.cross(
            state.angular_velocity[source], world_offset
        )
        child_orientation = jnp.broadcast_to(
            state.orientation[source], (plan.maximum_children, 4)
        )
        child_angular = jnp.broadcast_to(
            state.angular_velocity[source], (plan.maximum_children, 3)
        )
    child_position = state.position[source] + world_offset
    momentum_before = state.mass[source] * state.velocity[source]
    momentum_after = jnp.sum(
        jnp.where(valid_children[:, None], masses[:, None] * child_velocity, 0.0),
        axis=0,
    )
    linear_residual = momentum_after - momentum_before
    linear_ok = jnp.linalg.norm(linear_residual) <= tolerance * jnp.maximum(
        jnp.linalg.norm(momentum_before), 1.0
    )
    if plan.dimension == 2:
        angular_before = state.inertia_body[source] * state.angular_velocity[source, 0]
        orbital = (
            world_offset[:, 0] * masses * child_velocity[:, 1]
            - world_offset[:, 1] * masses * child_velocity[:, 0]
        )
        angular_after = jnp.sum(
            jnp.where(
                valid_children,
                inertias * child_angular[:, 0] + orbital,
                0.0,
            )
        )
        angular_residual = (angular_after - angular_before)[None]
    else:
        rotation = quaternion_rotation_matrix(state.orientation[source : source + 1])[0]
        source_world_inertia = contract(
            "ij,jk,lk->il", rotation, state.inertia_body[source], rotation
        )
        angular_before = contract(
            "ij,j->i", source_world_inertia, state.angular_velocity[source]
        )
        child_world_inertia = contract("ij,kjl,ml->kim", rotation, inertias, rotation)
        spin = contract("kij,kj->ki", child_world_inertia, child_angular)
        orbital = jnp.cross(world_offset, masses[:, None] * child_velocity)
        angular_after = jnp.sum(
            jnp.where(valid_children[:, None], spin + orbital, 0.0), axis=0
        )
        angular_residual = angular_after - angular_before
    angular_ok = jnp.linalg.norm(angular_residual) <= tolerance * jnp.maximum(
        jnp.linalg.norm(angular_before), 1.0
    )
    successful = (
        capacity_ok
        & source_active
        & children_inactive
        & mass_ok
        & linear_ok
        & angular_ok
        & no_duplicate_children
        & jnp.all(child_slots_valid | ~valid_children)
        & jnp.all(jnp.isfinite(child_position))
        & jnp.all(jnp.isfinite(child_velocity))
    )
    active = state.active.at[source].set(False)
    active = active.at[safe_children].set(
        jnp.where(valid_children, True, active[safe_children])
    )
    parent = state.parent_ids.at[safe_children].set(
        jnp.where(
            valid_children, state.stable_ids[source], state.parent_ids[safe_children]
        )
    )
    position = state.position.at[safe_children].set(
        jnp.where(valid_children[:, None], child_position, state.position[safe_children])
    )
    velocity = state.velocity.at[safe_children].set(
        jnp.where(valid_children[:, None], child_velocity, state.velocity[safe_children])
    )
    orientation = state.orientation.at[safe_children].set(
        jnp.where(
            valid_children[:, None], child_orientation, state.orientation[safe_children]
        )
    )
    angular = state.angular_velocity.at[safe_children].set(
        jnp.where(
            valid_children[:, None], child_angular, state.angular_velocity[safe_children]
        )
    )
    mass = state.mass.at[safe_children].set(
        jnp.where(valid_children, masses, state.mass[safe_children])
    )
    inertia_mask = valid_children.reshape(
        (plan.maximum_children,) + (1,) * (inertias.ndim - 1)
    )
    inertia = state.inertia_body.at[safe_children].set(
        jnp.where(inertia_mask, inertias, state.inertia_body[safe_children])
    )
    candidate = TopologyPoolState(
        state.stable_ids,
        active,
        parent,
        position,
        velocity,
        orientation,
        angular,
        mass,
        inertia,
        state.next_event_id + jnp.asarray(1, dtype=jnp.int64),
    )
    accepted = jax_tree_where(successful, candidate, state)
    event_ids = record.event_ids.at[event_slot].set(state.next_event_id)
    valid_events = record.valid.at[event_slot].set(successful)
    steps = record.step_index.at[event_slot].set(step_index)
    times = record.time.at[event_slot].set(time)
    sources = record.source_owner.at[event_slot].set(source)
    child_records = record.child_owners.at[event_slot].set(children)
    child_masks = record.child_valid.at[event_slot].set(valid_children)
    cause = record.cause_bond_id.at[event_slot].set(
        jnp.asarray(cause_bond_id, dtype=jnp.int64)
    )
    mass_residuals = record.mass_residual.at[event_slot].set(mass_residual)
    linear_residuals = record.linear_momentum_residual.at[event_slot].set(linear_residual)
    angular_residuals = record.angular_momentum_residual.at[event_slot].set(
        angular_residual
    )
    candidate_record = TopologyEventRecord(
        event_ids,
        valid_events,
        steps,
        times,
        sources,
        child_records,
        child_masks,
        cause,
        mass_residuals,
        linear_residuals,
        angular_residuals,
        successful,
        plan.plan_id,
    )
    accepted_record = jax_tree_where(successful, candidate_record, record)
    return TopologyEventResult(candidate, accepted, accepted_record, successful)


def jax_tree_where(condition: Array, proposed, current, /):
    import jax

    return jax.tree.map(
        lambda new, old: jnp.where(condition, new, old), proposed, current
    )


__all__ = [
    "TopologyEventPlan",
    "TopologyEventRecord",
    "TopologyEventResult",
    "TopologyPoolState",
    "initialize_topology_event_record",
    "split_preallocated_owner",
]
