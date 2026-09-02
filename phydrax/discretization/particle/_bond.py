#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._rigid_body import (
    PreparedRigidBodySet,
    quaternion_rotation_matrix,
    RigidBodyKinematics,
)


class FixedBondGraphPlan(StrictModule, NonTrainableState):
    left_particle_ids: Array
    right_particle_ids: Array
    bond_ids: Array
    anchor_left: Array
    anchor_right: Array
    rest_vector: Array
    cross_section: Array
    normal_stiffness: Array
    shear_stiffness: Array
    bending_stiffness: Array
    twisting_stiffness: Array
    damping: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        left_particle_ids: ArrayLike,
        right_particle_ids: ArrayLike,
        bond_ids: ArrayLike,
        anchor_left: ArrayLike,
        anchor_right: ArrayLike,
        rest_vector: ArrayLike,
        /,
        *,
        cross_section: ArrayLike,
        normal_stiffness: ArrayLike,
        shear_stiffness: ArrayLike,
        bending_stiffness: ArrayLike,
        twisting_stiffness: ArrayLike,
        damping: ArrayLike = 0.0,
        plan_id: str | None = None,
    ):
        left = np.asarray(left_particle_ids)
        right = np.asarray(right_particle_ids)
        identifiers = np.asarray(bond_ids)
        if left.ndim != 1 or right.shape != left.shape or identifiers.shape != left.shape:
            raise ValueError("Bond endpoint and bond-ID arrays must be matching vectors.")
        if not all(
            np.issubdtype(value.dtype, np.integer) for value in (left, right, identifiers)
        ):
            raise TypeError("Bond endpoint and bond IDs must be integers.")
        count = left.size
        anchor_left_ = np.asarray(anchor_left)
        anchor_right_ = np.asarray(anchor_right)
        rest = np.asarray(rest_vector)
        if (
            anchor_left_.shape != anchor_right_.shape
            or anchor_left_.shape != rest.shape
            or anchor_left_.shape[0] != count
            or anchor_left_.ndim != 2
            or anchor_left_.shape[1] not in (2, 3)
        ):
            raise ValueError("Bond anchors/rest vectors must have shape (bonds,2|3).")
        if np.any(left == right) or np.unique(identifiers).size != count:
            raise ValueError("Bond IDs must be unique and endpoints distinct.")
        endpoint_pairs = np.sort(np.stack((left, right), axis=-1), axis=-1)
        if np.unique(endpoint_pairs, axis=0).shape[0] != count:
            raise ValueError("Duplicate unordered bond endpoints are not allowed.")
        parameters = tuple(
            np.broadcast_to(np.asarray(value), (count,)).copy()
            for value in (
                cross_section,
                normal_stiffness,
                shear_stiffness,
                bending_stiffness,
                twisting_stiffness,
                damping,
            )
        )
        if (
            np.any(~np.isfinite(anchor_left_))
            or np.any(~np.isfinite(anchor_right_))
            or np.any(~np.isfinite(rest))
            or np.any(np.linalg.norm(rest, axis=-1) <= 0.0)
            or any(np.any(~np.isfinite(value)) for value in parameters)
            or any(np.any(value <= 0.0) for value in parameters[:-1])
            or np.any(parameters[-1] < 0.0)
        ):
            raise ValueError(
                "Bond geometry and parameters must be finite and admissible."
            )
        generated = canonical_fingerprint(
            {
                "kind": "fixed-bond-graph-plan",
                "values": array_tree_fingerprint(
                    {
                        "left": left,
                        "right": right,
                        "bond_ids": identifiers,
                        "anchor_left": anchor_left_,
                        "anchor_right": anchor_right_,
                        "rest": rest,
                        "parameters": parameters,
                    }
                ),
            }
        )
        self.left_particle_ids = jnp.asarray(left, dtype=jnp.int64)
        self.right_particle_ids = jnp.asarray(right, dtype=jnp.int64)
        self.bond_ids = jnp.asarray(identifiers, dtype=jnp.int64)
        self.anchor_left = jnp.asarray(anchor_left_)
        self.anchor_right = jnp.asarray(anchor_right_)
        self.rest_vector = jnp.asarray(rest)
        (
            self.cross_section,
            self.normal_stiffness,
            self.shear_stiffness,
            self.bending_stiffness,
            self.twisting_stiffness,
            self.damping,
        ) = tuple(jnp.asarray(value) for value in parameters)
        self.plan_id = generated if plan_id is None else str(plan_id)
        if not self.plan_id:
            raise ValueError("plan_id must be nonempty.")

    def prepare(self, bodies: PreparedRigidBodySet, /) -> PreparedFixedBondGraph:
        return PreparedFixedBondGraph(self, bodies)


class PreparedFixedBondGraph(StrictModule, NonTrainableState):
    plan: FixedBondGraphPlan
    bodies: PreparedRigidBodySet
    left_indices: Array
    right_indices: Array
    rest_length: Array
    rest_direction: Array
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: FixedBondGraphPlan, bodies: PreparedRigidBodySet, /):
        if not isinstance(plan, FixedBondGraphPlan):
            raise TypeError("plan must be a FixedBondGraphPlan.")
        if not isinstance(bodies, PreparedRigidBodySet):
            raise TypeError("bodies must be a PreparedRigidBodySet.")
        if plan.anchor_left.shape[1] != bodies.ambient_dimension:
            raise ValueError("Bond dimension does not match rigid bodies.")
        sorted_ids = jnp.sort(bodies.particles.particle_ids)
        left_rank = jnp.searchsorted(sorted_ids, plan.left_particle_ids)
        right_rank = jnp.searchsorted(sorted_ids, plan.right_particle_ids)
        logical_order = jnp.argsort(bodies.particles.particle_ids)
        left_indices = logical_order[left_rank]
        right_indices = logical_order[right_rank]
        left_match = bodies.particles.particle_ids[left_indices] == plan.left_particle_ids
        right_match = (
            bodies.particles.particle_ids[right_indices] == plan.right_particle_ids
        )
        if not bool(np.all(np.asarray(left_match & right_match))):
            raise ValueError("Bond endpoint ID is absent from rigid-body support.")
        active = bodies.particles.active_mask
        if not bool(np.all(np.asarray(active[left_indices] & active[right_indices]))):
            raise ValueError("Bond endpoints must be active bodies.")
        rest_length = jnp.linalg.norm(plan.rest_vector, axis=-1)
        self.plan = plan
        self.bodies = bodies
        self.left_indices = left_indices.astype(jnp.int32)
        self.right_indices = right_indices.astype(jnp.int32)
        self.rest_length = rest_length
        self.rest_direction = plan.rest_vector / rest_length[:, None]
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-fixed-bond-graph",
                "plan": plan.plan_id,
                "bodies": bodies.prepared_id,
            }
        )

    @property
    def capacity(self) -> int:
        return int(self.left_indices.shape[0])

    def initialize_state(self, dtype: np.dtype | None = None) -> DEMBondState:
        dtype_ = self.bodies.particles.safe_masses.dtype if dtype is None else dtype
        count = self.capacity
        return DEMBondState(
            jnp.ones((count,), dtype=bool),
            jnp.zeros((count,), dtype=dtype_),
            jnp.zeros((count,), dtype=dtype_),
            -jnp.ones((count,), dtype=jnp.int32),
        )


class DEMBondState(StrictModule):
    intact: Array
    damage: Array
    cumulative_fracture_energy: Array
    break_step: Array


class DEMBondEvaluation(StrictModule):
    body_force: Array
    body_torque: Array
    bond_force: Array
    left_moment: Array
    right_moment: Array
    stored_energy: Array
    equivalent_loading: Array
    net_force: Array
    net_torque: Array
    successful: Array


def _rotate_local(orientation: Array, vector: Array, dimension: int, /) -> Array:
    if dimension == 2:
        angle = orientation[:, 0]
        cosine = jnp.cos(angle)
        sine = jnp.sin(angle)
        return jnp.stack(
            (
                cosine * vector[:, 0] - sine * vector[:, 1],
                sine * vector[:, 0] + cosine * vector[:, 1],
            ),
            axis=-1,
        )
    rotation = quaternion_rotation_matrix(orientation)
    return contract("...ij,...j->...i", rotation, vector)


def evaluate_bonds(
    bonds: PreparedFixedBondGraph,
    kinematics: RigidBodyKinematics,
    state: DEMBondState,
    /,
) -> DEMBondEvaluation:
    if not isinstance(bonds, PreparedFixedBondGraph):
        raise TypeError("bonds must be PreparedFixedBondGraph.")
    if not isinstance(state, DEMBondState):
        raise TypeError("state must be DEMBondState.")
    left = bonds.left_indices
    right = bonds.right_indices
    dimension = bonds.bodies.ambient_dimension
    left_anchor_offset = _rotate_local(
        kinematics.orientation[left], bonds.plan.anchor_left, dimension
    )
    right_anchor_offset = _rotate_local(
        kinematics.orientation[right], bonds.plan.anchor_right, dimension
    )
    left_anchor = kinematics.position[left] + left_anchor_offset
    right_anchor = kinematics.position[right] + right_anchor_offset
    current_vector = right_anchor - left_anchor
    displacement = current_vector - bonds.plan.rest_vector
    axial = jnp.sum(displacement * bonds.rest_direction, axis=-1)
    shear = displacement - axial[:, None] * bonds.rest_direction
    relative_velocity = kinematics.velocity[right] - kinematics.velocity[left]
    force = (
        bonds.plan.normal_stiffness[:, None] * axial[:, None] * bonds.rest_direction
        + bonds.plan.shear_stiffness[:, None] * shear
        + bonds.plan.damping[:, None] * relative_velocity
    )
    if dimension == 2:
        relative_rotation = (
            kinematics.orientation[right, 0] - kinematics.orientation[left, 0]
        )[:, None]
        twist = relative_rotation
        bend = jnp.zeros_like(relative_rotation)
    else:
        relative_rotation = (
            kinematics.angular_velocity[right] - kinematics.angular_velocity[left]
        )
        twist_scalar = jnp.sum(relative_rotation * bonds.rest_direction, axis=-1)
        twist = twist_scalar[:, None] * bonds.rest_direction
        bend = relative_rotation - twist
    moment = (
        bonds.plan.twisting_stiffness[:, None] * twist
        + bonds.plan.bending_stiffness[:, None] * bend
    )
    scale = state.intact.astype(force.dtype) * (1.0 - state.damage)
    force = scale[:, None] * force
    moment = scale[:, None] * moment
    body_force = jnp.zeros((bonds.bodies.capacity, dimension), dtype=force.dtype)
    body_force = body_force.at[left].add(force)
    body_force = body_force.at[right].add(-force)
    left_moment = (
        jnp.cross(left_anchor_offset, force)
        if dimension == 3
        else (
            left_anchor_offset[:, 0] * force[:, 1]
            - left_anchor_offset[:, 1] * force[:, 0]
        )[:, None]
    )
    right_moment = (
        jnp.cross(right_anchor_offset, -force)
        if dimension == 3
        else (
            right_anchor_offset[:, 0] * (-force[:, 1])
            - right_anchor_offset[:, 1] * (-force[:, 0])
        )[:, None]
    )
    left_moment = left_moment + moment
    right_moment = right_moment - moment
    body_torque = jnp.zeros(
        (bonds.bodies.capacity, bonds.bodies.angular_dimension), dtype=force.dtype
    )
    body_torque = body_torque.at[left].add(left_moment)
    body_torque = body_torque.at[right].add(right_moment)
    stored = (
        0.5
        * scale
        * (
            bonds.plan.normal_stiffness * axial**2
            + bonds.plan.shear_stiffness * jnp.sum(shear**2, axis=-1)
            + bonds.plan.bending_stiffness * jnp.sum(bend**2, axis=-1)
            + bonds.plan.twisting_stiffness * jnp.sum(twist**2, axis=-1)
        )
    )
    equivalent = jnp.sqrt(
        axial**2
        + jnp.sum(shear**2, axis=-1)
        + jnp.sum(bend**2, axis=-1)
        + jnp.sum(twist**2, axis=-1)
    )
    net_force = jnp.sum(body_force, axis=0)
    net_torque = jnp.sum(body_torque, axis=0)
    successful = (
        jnp.all(jnp.isfinite(body_force))
        & jnp.all(jnp.isfinite(body_torque))
        & jnp.all(stored >= 0.0)
    )
    return DEMBondEvaluation(
        body_force,
        body_torque,
        force,
        left_moment,
        right_moment,
        stored,
        equivalent,
        net_force,
        net_torque,
        successful,
    )


class MixedModeBondDamagePlan(StrictModule, NonTrainableState):
    initiation_loading: Array
    failure_loading: Array
    fracture_energy: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        initiation_loading: ArrayLike,
        failure_loading: ArrayLike,
        fracture_energy: ArrayLike,
        /,
        *,
        plan_id: str | None = None,
    ):
        initiation = np.asarray(initiation_loading)
        failure = np.asarray(failure_loading)
        energy = np.asarray(fracture_energy)
        if (
            initiation.shape != failure.shape
            or initiation.shape != energy.shape
            or initiation.ndim != 1
        ):
            raise ValueError("Damage parameters must be matching bond vectors.")
        if (
            np.any(~np.isfinite(initiation))
            or np.any(~np.isfinite(failure))
            or np.any(~np.isfinite(energy))
            or np.any(initiation <= 0.0)
            or np.any(failure <= initiation)
            or np.any(energy <= 0.0)
        ):
            raise ValueError("Damage initiation/failure/energy parameters are invalid.")
        generated = canonical_fingerprint(
            {
                "kind": "mixed-mode-bond-damage-plan",
                "values": array_tree_fingerprint(
                    {"initiation": initiation, "failure": failure, "energy": energy}
                ),
            }
        )
        self.initiation_loading = jnp.asarray(initiation)
        self.failure_loading = jnp.asarray(failure)
        self.fracture_energy = jnp.asarray(energy)
        self.plan_id = generated if plan_id is None else str(plan_id)
        if not self.plan_id:
            raise ValueError("plan_id must be nonempty.")

    def update(
        self,
        state: DEMBondState,
        evaluation: DEMBondEvaluation,
        step_index: Array,
        /,
    ) -> DEMBondState:
        if self.initiation_loading.shape != state.damage.shape:
            raise ValueError("Damage plan does not match bond state capacity.")
        trial = jnp.clip(
            (evaluation.equivalent_loading - self.initiation_loading)
            / (self.failure_loading - self.initiation_loading),
            0.0,
            1.0,
        )
        damage = jnp.maximum(state.damage, trial)
        increment = damage - state.damage
        intact = state.intact & (damage < 1.0)
        newly_broken = state.intact & ~intact
        break_step = jnp.where(newly_broken, step_index, state.break_step)
        fracture = state.cumulative_fracture_energy + increment * self.fracture_energy
        fracture = eqx.error_if(
            fracture,
            jnp.any(damage < state.damage)
            | jnp.any(fracture < state.cumulative_fracture_energy),
            "Bond damage and fracture energy must be irreversible.",
        )
        return DEMBondState(intact, damage, fracture, break_step)


__all__ = [
    "DEMBondEvaluation",
    "DEMBondState",
    "FixedBondGraphPlan",
    "MixedModeBondDamagePlan",
    "PreparedFixedBondGraph",
    "evaluate_bonds",
]
