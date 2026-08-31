#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._tree_math import tree_allfinite
from ._rigid_body import (
    _principal_angle,
    _quaternion_conjugate,
    _quaternion_multiply,
    _rigid_body_relative_rotation,
    _rigid_body_retract_pose,
    _rigid_body_rotation_matrix,
    PreparedRigidBodySet,
    RigidBodyKinematics,
)


def _joint_vectors(
    owner: str,
    joint_ids: ArrayLike,
    left_body_ids: ArrayLike,
    right_body_ids: ArrayLike,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    identifiers = np.asarray(joint_ids)
    left = np.asarray(left_body_ids)
    right = np.asarray(right_body_ids)
    if (
        identifiers.ndim != 1
        or left.shape != identifiers.shape
        or right.shape != identifiers.shape
    ):
        raise ValueError(f"{owner} joint IDs and endpoint IDs must be matching vectors.")
    if not all(
        np.issubdtype(value.dtype, np.integer) for value in (identifiers, left, right)
    ):
        raise TypeError(f"{owner} joint and endpoint IDs must be integers.")
    if np.unique(identifiers).size != identifiers.size:
        raise ValueError(f"{owner} joint IDs must be unique.")
    if np.any(left == right):
        raise ValueError(f"{owner} joint endpoints must be distinct.")
    return (
        identifiers.astype(np.int64, copy=False),
        left.astype(np.int64, copy=False),
        right.astype(np.int64, copy=False),
    )


def _reference_vectors(
    owner: str,
    value: ArrayLike,
    count: int,
    /,
) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[0] != count or array.shape[1] not in (2, 3):
        raise ValueError(f"{owner} must have shape ({count}, 2) or ({count}, 3).")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{owner} must be finite.")
    return array


class BallJointSetPlan(StrictModule, NonTrainableState):
    joint_ids: Array
    left_body_ids: Array
    right_body_ids: Array
    reference_anchors: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        joint_ids: ArrayLike,
        left_body_ids: ArrayLike,
        right_body_ids: ArrayLike,
        reference_anchors: ArrayLike,
        /,
        *,
        plan_id: str | None = None,
    ):
        identifiers, left, right = _joint_vectors(
            "Ball", joint_ids, left_body_ids, right_body_ids
        )
        anchors = _reference_vectors(
            "Ball reference_anchors", reference_anchors, identifiers.size
        )
        generated = canonical_fingerprint(
            {
                "kind": "ball-joint-set-plan",
                "values": array_tree_fingerprint(
                    {
                        "joint_ids": identifiers,
                        "left": left,
                        "right": right,
                        "anchors": anchors,
                    }
                ),
            }
        )
        self.joint_ids = jnp.asarray(identifiers)
        self.left_body_ids = jnp.asarray(left)
        self.right_body_ids = jnp.asarray(right)
        self.reference_anchors = jnp.asarray(anchors)
        self.plan_id = generated if plan_id is None else str(plan_id)
        if not self.plan_id:
            raise ValueError("plan_id must be nonempty.")

    @property
    def count(self) -> int:
        return int(self.joint_ids.shape[0])


class FixedJointSetPlan(StrictModule, NonTrainableState):
    joint_ids: Array
    left_body_ids: Array
    right_body_ids: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        joint_ids: ArrayLike,
        left_body_ids: ArrayLike,
        right_body_ids: ArrayLike,
        /,
        *,
        plan_id: str | None = None,
    ):
        identifiers, left, right = _joint_vectors(
            "Fixed", joint_ids, left_body_ids, right_body_ids
        )
        generated = canonical_fingerprint(
            {
                "kind": "fixed-joint-set-plan",
                "values": array_tree_fingerprint(
                    {"joint_ids": identifiers, "left": left, "right": right}
                ),
            }
        )
        self.joint_ids = jnp.asarray(identifiers)
        self.left_body_ids = jnp.asarray(left)
        self.right_body_ids = jnp.asarray(right)
        self.plan_id = generated if plan_id is None else str(plan_id)
        if not self.plan_id:
            raise ValueError("plan_id must be nonempty.")

    @property
    def count(self) -> int:
        return int(self.joint_ids.shape[0])


class HingeJointSetPlan(StrictModule, NonTrainableState):
    joint_ids: Array
    left_body_ids: Array
    right_body_ids: Array
    reference_anchors: Array
    reference_axes: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        joint_ids: ArrayLike,
        left_body_ids: ArrayLike,
        right_body_ids: ArrayLike,
        reference_anchors: ArrayLike,
        reference_axes: ArrayLike,
        /,
        *,
        plan_id: str | None = None,
    ):
        identifiers, left, right = _joint_vectors(
            "Hinge", joint_ids, left_body_ids, right_body_ids
        )
        anchors = _reference_vectors(
            "Hinge reference_anchors", reference_anchors, identifiers.size
        )
        axes = _reference_vectors(
            "Hinge reference_axes", reference_axes, identifiers.size
        )
        if anchors.shape[1] != 3 or axes.shape[1] != 3:
            raise ValueError("Hinge reference geometry must be three-dimensional.")
        norms = np.linalg.norm(axes, axis=-1)
        if np.any(norms <= np.finfo(float).eps):
            raise ValueError("Hinge reference axes must be nonzero.")
        axes = axes / norms[:, None]
        generated = canonical_fingerprint(
            {
                "kind": "hinge-joint-set-plan",
                "values": array_tree_fingerprint(
                    {
                        "joint_ids": identifiers,
                        "left": left,
                        "right": right,
                        "anchors": anchors,
                        "axes": axes,
                    }
                ),
            }
        )
        self.joint_ids = jnp.asarray(identifiers)
        self.left_body_ids = jnp.asarray(left)
        self.right_body_ids = jnp.asarray(right)
        self.reference_anchors = jnp.asarray(anchors)
        self.reference_axes = jnp.asarray(axes)
        self.plan_id = generated if plan_id is None else str(plan_id)
        if not self.plan_id:
            raise ValueError("plan_id must be nonempty.")

    @property
    def count(self) -> int:
        return int(self.joint_ids.shape[0])


class PrismaticJointSetPlan(StrictModule, NonTrainableState):
    joint_ids: Array
    left_body_ids: Array
    right_body_ids: Array
    reference_anchors: Array
    reference_axes: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        joint_ids: ArrayLike,
        left_body_ids: ArrayLike,
        right_body_ids: ArrayLike,
        reference_anchors: ArrayLike,
        reference_axes: ArrayLike,
        /,
        *,
        plan_id: str | None = None,
    ):
        identifiers, left, right = _joint_vectors(
            "Prismatic", joint_ids, left_body_ids, right_body_ids
        )
        anchors = _reference_vectors(
            "Prismatic reference_anchors", reference_anchors, identifiers.size
        )
        axes = _reference_vectors(
            "Prismatic reference_axes", reference_axes, identifiers.size
        )
        if anchors.shape != axes.shape:
            raise ValueError("Prismatic anchors and axes must have matching dimensions.")
        norms = np.linalg.norm(axes, axis=-1)
        if np.any(norms <= np.finfo(float).eps):
            raise ValueError("Prismatic reference axes must be nonzero.")
        axes = axes / norms[:, None]
        generated = canonical_fingerprint(
            {
                "kind": "prismatic-joint-set-plan",
                "values": array_tree_fingerprint(
                    {
                        "joint_ids": identifiers,
                        "left": left,
                        "right": right,
                        "anchors": anchors,
                        "axes": axes,
                    }
                ),
            }
        )
        self.joint_ids = jnp.asarray(identifiers)
        self.left_body_ids = jnp.asarray(left)
        self.right_body_ids = jnp.asarray(right)
        self.reference_anchors = jnp.asarray(anchors)
        self.reference_axes = jnp.asarray(axes)
        self.plan_id = generated if plan_id is None else str(plan_id)
        if not self.plan_id:
            raise ValueError("plan_id must be nonempty.")

    @property
    def count(self) -> int:
        return int(self.joint_ids.shape[0])


class DistanceJointSetPlan(StrictModule, NonTrainableState):
    joint_ids: Array
    left_body_ids: Array
    right_body_ids: Array
    reference_left_anchors: Array
    reference_right_anchors: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        joint_ids: ArrayLike,
        left_body_ids: ArrayLike,
        right_body_ids: ArrayLike,
        reference_left_anchors: ArrayLike,
        reference_right_anchors: ArrayLike,
        /,
        *,
        plan_id: str | None = None,
    ):
        identifiers, left, right = _joint_vectors(
            "Distance", joint_ids, left_body_ids, right_body_ids
        )
        left_anchors = _reference_vectors(
            "Distance reference_left_anchors",
            reference_left_anchors,
            identifiers.size,
        )
        right_anchors = _reference_vectors(
            "Distance reference_right_anchors",
            reference_right_anchors,
            identifiers.size,
        )
        if left_anchors.shape != right_anchors.shape:
            raise ValueError("Distance anchors must have matching dimensions.")
        rest = np.linalg.norm(right_anchors - left_anchors, axis=-1)
        if np.any(rest <= np.finfo(float).eps):
            raise ValueError("Distance-joint reference lengths must be positive.")
        generated = canonical_fingerprint(
            {
                "kind": "distance-joint-set-plan",
                "values": array_tree_fingerprint(
                    {
                        "joint_ids": identifiers,
                        "left": left,
                        "right": right,
                        "left_anchors": left_anchors,
                        "right_anchors": right_anchors,
                    }
                ),
            }
        )
        self.joint_ids = jnp.asarray(identifiers)
        self.left_body_ids = jnp.asarray(left)
        self.right_body_ids = jnp.asarray(right)
        self.reference_left_anchors = jnp.asarray(left_anchors)
        self.reference_right_anchors = jnp.asarray(right_anchors)
        self.plan_id = generated if plan_id is None else str(plan_id)
        if not self.plan_id:
            raise ValueError("plan_id must be nonempty.")

    @property
    def count(self) -> int:
        return int(self.joint_ids.shape[0])


class RigidJointGraphPlan(StrictModule, NonTrainableState):
    fixed: FixedJointSetPlan | None
    ball: BallJointSetPlan | None
    hinge: HingeJointSetPlan | None
    prismatic: PrismaticJointSetPlan | None
    distance: DistanceJointSetPlan | None
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        fixed: FixedJointSetPlan | None = None,
        ball: BallJointSetPlan | None = None,
        hinge: HingeJointSetPlan | None = None,
        prismatic: PrismaticJointSetPlan | None = None,
        distance: DistanceJointSetPlan | None = None,
        plan_id: str | None = None,
    ):
        expected = (
            ("fixed", fixed, FixedJointSetPlan),
            ("ball", ball, BallJointSetPlan),
            ("hinge", hinge, HingeJointSetPlan),
            ("prismatic", prismatic, PrismaticJointSetPlan),
            ("distance", distance, DistanceJointSetPlan),
        )
        for name, value, kind in expected:
            if value is not None and not isinstance(value, kind):
                raise TypeError(f"{name} must be a {kind.__name__} or None.")
        identifier_arrays = tuple(
            np.asarray(value.joint_ids) for _, value, _ in expected if value is not None
        )
        if identifier_arrays:
            identifiers = np.concatenate(identifier_arrays)
            if np.unique(identifiers).size != identifiers.size:
                raise ValueError("Joint IDs must be globally unique across joint kinds.")
        generated = canonical_fingerprint(
            {
                "kind": "rigid-joint-graph-plan",
                "plans": {
                    name: None if value is None else value.plan_id
                    for name, value, _ in expected
                },
            }
        )
        self.fixed = fixed
        self.ball = ball
        self.hinge = hinge
        self.prismatic = prismatic
        self.distance = distance
        self.plan_id = generated if plan_id is None else str(plan_id)
        if not self.plan_id:
            raise ValueError("plan_id must be nonempty.")

    def constraint_count_for_dimension(self, dimension: int, /) -> int:
        if dimension not in (2, 3):
            raise ValueError("Rigid joint dimension must be two or three.")
        angular = 1 if dimension == 2 else 3
        return (
            (dimension + angular) * (0 if self.fixed is None else self.fixed.count)
            + dimension * (0 if self.ball is None else self.ball.count)
            + (0 if self.hinge is None else 5 * self.hinge.count)
            + (dimension - 1 + angular)
            * (0 if self.prismatic is None else self.prismatic.count)
            + (0 if self.distance is None else self.distance.count)
        )

    @property
    def constraint_count(self) -> int:
        return self.constraint_count_for_dimension(3)

    def prepare(
        self,
        bodies: PreparedRigidBodySet,
        reference: RigidBodyKinematics,
        /,
    ) -> PreparedRigidJointGraph:
        return PreparedRigidJointGraph(self, bodies, reference)


class RigidJointResiduals(StrictModule):
    fixed_translation: Array
    fixed_rotation: Array
    ball_anchor: Array
    hinge_anchor: Array
    hinge_axis: Array
    prismatic_translation: Array
    prismatic_rotation: Array
    distance: Array


class RigidJointMultipliers(StrictModule):
    fixed_translation: Array
    fixed_rotation: Array
    ball_anchor: Array
    hinge_anchor: Array
    hinge_axis: Array
    prismatic_translation: Array
    prismatic_rotation: Array
    distance: Array


class _RigidMobileIncrement(StrictModule):
    translation: Array
    rotation: Array


class RigidJointKind(IntEnum):
    FIXED = 1
    BALL = 2
    HINGE = 3
    PRISMATIC = 4
    DISTANCE = 5


class RigidJointRowLayout(StrictModule, NonTrainableState):
    joint_ids: Array
    joint_kinds: Array
    row_joint_slots: Array
    row_kinds: Array
    row_local_indices: Array
    layout_id: str = eqx.field(static=True)

    @property
    def joint_count(self) -> int:
        return int(self.joint_ids.shape[0])

    @property
    def row_count(self) -> int:
        return int(self.row_joint_slots.shape[0])

    def row_active(self, joint_active: Array, /) -> Array:
        if joint_active.shape != self.joint_ids.shape:
            raise ValueError("joint_active shape does not match joint row layout.")
        return joint_active[self.row_joint_slots]


class PreparedRigidJointGraph(StrictModule, NonTrainableState):
    plan: RigidJointGraphPlan
    bodies: PreparedRigidBodySet
    mobile_indices: Array
    row_layout: RigidJointRowLayout
    fixed_left: Array
    fixed_right: Array
    fixed_rest_offset: Array
    fixed_rest_orientation: Array
    ball_left: Array
    ball_right: Array
    ball_anchor_left: Array
    ball_anchor_right: Array
    ball_rest_orientation: Array
    hinge_left: Array
    hinge_right: Array
    hinge_anchor_left: Array
    hinge_anchor_right: Array
    hinge_axis_left: Array
    hinge_axis_right: Array
    hinge_transverse_left_1: Array
    hinge_transverse_left_2: Array
    hinge_transverse_right_1: Array
    hinge_transverse_right_2: Array
    prismatic_left: Array
    prismatic_right: Array
    prismatic_anchor_left: Array
    prismatic_anchor_right: Array
    prismatic_axis_left: Array
    prismatic_transverse_left: Array
    prismatic_rest_orientation: Array
    distance_left: Array
    distance_right: Array
    distance_anchor_left: Array
    distance_anchor_right: Array
    distance_rest_length: Array
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: RigidJointGraphPlan,
        bodies: PreparedRigidBodySet,
        reference: RigidBodyKinematics,
        /,
    ):
        if not isinstance(plan, RigidJointGraphPlan):
            raise TypeError("plan must be a RigidJointGraphPlan.")
        if not isinstance(bodies, PreparedRigidBodySet):
            raise TypeError("bodies must be a PreparedRigidBodySet.")
        if not isinstance(reference, RigidBodyKinematics):
            raise TypeError("reference must be RigidBodyKinematics.")
        dimension = bodies.ambient_dimension
        angular_dimension = bodies.angular_dimension
        if dimension == 2 and plan.hinge is not None:
            raise ValueError("HingeJointSetPlan is three-dimensional only.")
        expected_position = (bodies.capacity, dimension)
        if (
            reference.position.shape != expected_position
            or reference.velocity.shape != expected_position
            or reference.orientation.shape
            != (bodies.capacity, bodies.orientation_dimension)
            or reference.angular_velocity.shape != (bodies.capacity, angular_dimension)
        ):
            raise ValueError("Reference rigid-body kinematics have incompatible shapes.")
        if not bool(np.asarray(tree_allfinite(reference))):
            raise ValueError("Reference rigid-body kinematics must be finite.")
        if dimension == 3:
            orientation_norm = np.linalg.norm(np.asarray(reference.orientation), axis=-1)
            if not np.allclose(orientation_norm, 1.0, rtol=0.0, atol=1.0e-8):
                raise ValueError("Reference rigid-body orientations must have unit norm.")
        for joint_plan in (plan.ball, plan.prismatic, plan.distance):
            if joint_plan is None:
                continue
            vectors = (
                (joint_plan.reference_anchors,)
                if not isinstance(joint_plan, DistanceJointSetPlan)
                else (
                    joint_plan.reference_left_anchors,
                    joint_plan.reference_right_anchors,
                )
            )
            if any(value.shape[-1] != dimension for value in vectors):
                raise ValueError(
                    "Joint reference geometry dimension does not match bodies."
                )

        active = np.asarray(bodies.particles.active_mask, dtype=bool)
        fixed_mask = np.asarray(bodies.fixed_mask, dtype=bool)
        mobile_mask = active & ~fixed_mask
        mobile_indices = np.flatnonzero(mobile_mask).astype(np.int32)

        fixed_left, fixed_right = self._endpoints(plan.fixed, bodies, active, fixed_mask)
        ball_left, ball_right = self._endpoints(plan.ball, bodies, active, fixed_mask)
        hinge_left, hinge_right = self._endpoints(plan.hinge, bodies, active, fixed_mask)
        prismatic_left, prismatic_right = self._endpoints(
            plan.prismatic, bodies, active, fixed_mask
        )
        distance_left, distance_right = self._endpoints(
            plan.distance, bodies, active, fixed_mask
        )
        if (
            plan.constraint_count_for_dimension(dimension)
            > (dimension + angular_dimension) * mobile_indices.size
        ):
            raise ValueError("Joint rows exceed available mobile rigid-body coordinates.")
        layout_parts = (
            (
                plan.fixed,
                RigidJointKind.FIXED,
                dimension + angular_dimension,
            ),
            (plan.ball, RigidJointKind.BALL, dimension),
            (plan.hinge, RigidJointKind.HINGE, 5),
            (
                plan.prismatic,
                RigidJointKind.PRISMATIC,
                dimension - 1 + angular_dimension,
            ),
            (plan.distance, RigidJointKind.DISTANCE, 1),
        )
        joint_ids_parts = []
        joint_kind_parts = []
        row_slot_parts = []
        row_kind_parts = []
        row_local_parts = []
        joint_offset = 0
        for joint_plan, kind, rows_per_joint in layout_parts:
            if joint_plan is None:
                continue
            count = joint_plan.count
            joint_ids_parts.append(np.asarray(joint_plan.joint_ids, dtype=np.int64))
            joint_kind_parts.append(np.full((count,), int(kind), dtype=np.int32))
            row_slot_parts.append(
                np.repeat(
                    np.arange(joint_offset, joint_offset + count, dtype=np.int32),
                    rows_per_joint,
                )
            )
            row_kind_parts.append(
                np.full((count * rows_per_joint,), int(kind), dtype=np.int32)
            )
            row_local_parts.append(
                np.tile(np.arange(rows_per_joint, dtype=np.int32), count)
            )
            joint_offset += count
        joint_ids_layout = (
            np.concatenate(joint_ids_parts)
            if joint_ids_parts
            else np.empty((0,), dtype=np.int64)
        )
        joint_kinds_layout = (
            np.concatenate(joint_kind_parts)
            if joint_kind_parts
            else np.empty((0,), dtype=np.int32)
        )
        row_slots_layout = (
            np.concatenate(row_slot_parts)
            if row_slot_parts
            else np.empty((0,), dtype=np.int32)
        )
        row_kinds_layout = (
            np.concatenate(row_kind_parts)
            if row_kind_parts
            else np.empty((0,), dtype=np.int32)
        )
        row_local_layout = (
            np.concatenate(row_local_parts)
            if row_local_parts
            else np.empty((0,), dtype=np.int32)
        )
        row_layout = RigidJointRowLayout(
            jnp.asarray(joint_ids_layout),
            jnp.asarray(joint_kinds_layout),
            jnp.asarray(row_slots_layout),
            jnp.asarray(row_kinds_layout),
            jnp.asarray(row_local_layout),
            canonical_fingerprint(
                {
                    "kind": "rigid-joint-row-layout",
                    "dimension": dimension,
                    "values": array_tree_fingerprint(
                        {
                            "joint_ids": joint_ids_layout,
                            "joint_kinds": joint_kinds_layout,
                            "row_slots": row_slots_layout,
                            "row_kinds": row_kinds_layout,
                            "row_local": row_local_layout,
                        }
                    ),
                }
            ),
        )

        position = np.asarray(reference.position)
        orientation = reference.orientation
        rotation = np.asarray(_rigid_body_rotation_matrix(bodies, orientation))

        if plan.fixed is None:
            fixed_rest_offset = np.empty((0, dimension), dtype=position.dtype)
            fixed_rest_orientation = jnp.empty(
                (0, bodies.orientation_dimension), dtype=orientation.dtype
            )
        else:
            relative = position[fixed_right] - position[fixed_left]
            fixed_rest_offset = contract("nji,nj->ni", rotation[fixed_left], relative)
            if dimension == 2:
                fixed_rest_orientation = _principal_angle(
                    orientation[fixed_right] - orientation[fixed_left]
                )
            else:
                fixed_rest_orientation = _quaternion_multiply(
                    _quaternion_conjugate(orientation[fixed_left]),
                    orientation[fixed_right],
                )

        ball_anchor_left, ball_anchor_right = self._local_anchors(
            None if plan.ball is None else np.asarray(plan.ball.reference_anchors),
            ball_left,
            ball_right,
            position,
            rotation,
        )
        if plan.ball is None:
            ball_rest_orientation = jnp.empty(
                (0, bodies.orientation_dimension), dtype=orientation.dtype
            )
        elif dimension == 2:
            ball_rest_orientation = _principal_angle(
                orientation[ball_right] - orientation[ball_left]
            )
        else:
            ball_rest_orientation = _quaternion_multiply(
                _quaternion_conjugate(orientation[ball_left]),
                orientation[ball_right],
            )
        hinge_anchor_left, hinge_anchor_right = self._local_anchors(
            None if plan.hinge is None else np.asarray(plan.hinge.reference_anchors),
            hinge_left,
            hinge_right,
            position,
            rotation,
        )

        if plan.hinge is None:
            hinge_axis_left = np.empty((0, 3), dtype=position.dtype)
            hinge_axis_right = np.empty((0, 3), dtype=position.dtype)
            left_transverse_1 = np.empty((0, 3), dtype=position.dtype)
            left_transverse_2 = np.empty((0, 3), dtype=position.dtype)
            transverse_1 = np.empty((0, 3), dtype=position.dtype)
            transverse_2 = np.empty((0, 3), dtype=position.dtype)
        else:
            axes = np.asarray(plan.hinge.reference_axes, dtype=position.dtype)
            hinge_axis_left = contract("nji,nj->ni", rotation[hinge_left], axes)
            hinge_axis_right = contract("nji,nj->ni", rotation[hinge_right], axes)
            seeds = np.eye(3, dtype=position.dtype)[np.argmin(np.abs(axes), axis=-1)]
            basis_1_world = np.cross(axes, seeds)
            basis_1_world /= np.linalg.norm(basis_1_world, axis=-1)[:, None]
            basis_2_world = np.cross(axes, basis_1_world)
            left_transverse_1 = contract(
                "nji,nj->ni", rotation[hinge_left], basis_1_world
            )
            left_transverse_2 = contract(
                "nji,nj->ni", rotation[hinge_left], basis_2_world
            )
            transverse_1 = contract("nji,nj->ni", rotation[hinge_right], basis_1_world)
            transverse_2 = contract("nji,nj->ni", rotation[hinge_right], basis_2_world)

        prismatic_anchor_left, prismatic_anchor_right = self._local_anchors(
            None
            if plan.prismatic is None
            else np.asarray(plan.prismatic.reference_anchors),
            prismatic_left,
            prismatic_right,
            position,
            rotation,
        )
        if plan.prismatic is None:
            prismatic_axis_left = np.empty((0, dimension), dtype=position.dtype)
            prismatic_transverse_left = np.empty(
                (0, dimension - 1, dimension), dtype=position.dtype
            )
            prismatic_rest_orientation = jnp.empty(
                (0, bodies.orientation_dimension), dtype=orientation.dtype
            )
        else:
            axes = np.asarray(plan.prismatic.reference_axes, dtype=position.dtype)
            prismatic_axis_left = contract("nji,nj->ni", rotation[prismatic_left], axes)
            if dimension == 2:
                normal_world = np.stack((-axes[:, 1], axes[:, 0]), axis=-1)
                transverse_world = normal_world[:, None, :]
                prismatic_rest_orientation = _principal_angle(
                    orientation[prismatic_right] - orientation[prismatic_left]
                )
            else:
                seeds = np.eye(3, dtype=position.dtype)[np.argmin(np.abs(axes), axis=-1)]
                first = np.cross(axes, seeds)
                first /= np.linalg.norm(first, axis=-1)[:, None]
                transverse_world = np.stack((first, np.cross(axes, first)), axis=1)
                prismatic_rest_orientation = _quaternion_multiply(
                    _quaternion_conjugate(orientation[prismatic_left]),
                    orientation[prismatic_right],
                )
            prismatic_transverse_left = contract(
                "nji,nkj->nki", rotation[prismatic_left], transverse_world
            )

        if plan.distance is None:
            distance_anchor_left = np.empty((0, dimension), dtype=position.dtype)
            distance_anchor_right = np.empty((0, dimension), dtype=position.dtype)
            distance_rest_length = np.empty((0,), dtype=position.dtype)
        else:
            distance_anchor_left = contract(
                "nji,nj->ni",
                rotation[distance_left],
                np.asarray(plan.distance.reference_left_anchors)
                - position[distance_left],
            )
            distance_anchor_right = contract(
                "nji,nj->ni",
                rotation[distance_right],
                np.asarray(plan.distance.reference_right_anchors)
                - position[distance_right],
            )
            distance_rest_length = np.linalg.norm(
                np.asarray(plan.distance.reference_right_anchors)
                - np.asarray(plan.distance.reference_left_anchors),
                axis=-1,
            )

        self.plan = plan
        self.bodies = bodies
        self.mobile_indices = jnp.asarray(mobile_indices)
        self.row_layout = row_layout
        self.fixed_left = jnp.asarray(fixed_left)
        self.fixed_right = jnp.asarray(fixed_right)
        self.fixed_rest_offset = jnp.asarray(fixed_rest_offset, dtype=position.dtype)
        self.fixed_rest_orientation = fixed_rest_orientation
        self.ball_left = jnp.asarray(ball_left)
        self.ball_right = jnp.asarray(ball_right)
        self.ball_anchor_left = jnp.asarray(ball_anchor_left, dtype=position.dtype)
        self.ball_anchor_right = jnp.asarray(ball_anchor_right, dtype=position.dtype)
        self.ball_rest_orientation = ball_rest_orientation
        self.hinge_left = jnp.asarray(hinge_left)
        self.hinge_right = jnp.asarray(hinge_right)
        self.hinge_anchor_left = jnp.asarray(hinge_anchor_left, dtype=position.dtype)
        self.hinge_anchor_right = jnp.asarray(hinge_anchor_right, dtype=position.dtype)
        self.hinge_axis_left = jnp.asarray(hinge_axis_left, dtype=position.dtype)
        self.hinge_axis_right = jnp.asarray(hinge_axis_right, dtype=position.dtype)
        self.hinge_transverse_left_1 = jnp.asarray(
            left_transverse_1, dtype=position.dtype
        )
        self.hinge_transverse_left_2 = jnp.asarray(
            left_transverse_2, dtype=position.dtype
        )
        self.hinge_transverse_right_1 = jnp.asarray(transverse_1, dtype=position.dtype)
        self.hinge_transverse_right_2 = jnp.asarray(transverse_2, dtype=position.dtype)
        self.prismatic_left = jnp.asarray(prismatic_left)
        self.prismatic_right = jnp.asarray(prismatic_right)
        self.prismatic_anchor_left = jnp.asarray(
            prismatic_anchor_left, dtype=position.dtype
        )
        self.prismatic_anchor_right = jnp.asarray(
            prismatic_anchor_right, dtype=position.dtype
        )
        self.prismatic_axis_left = jnp.asarray(prismatic_axis_left, dtype=position.dtype)
        self.prismatic_transverse_left = jnp.asarray(
            prismatic_transverse_left, dtype=position.dtype
        )
        self.prismatic_rest_orientation = prismatic_rest_orientation
        self.distance_left = jnp.asarray(distance_left)
        self.distance_right = jnp.asarray(distance_right)
        self.distance_anchor_left = jnp.asarray(
            distance_anchor_left, dtype=position.dtype
        )
        self.distance_anchor_right = jnp.asarray(
            distance_anchor_right, dtype=position.dtype
        )
        self.distance_rest_length = jnp.asarray(
            distance_rest_length, dtype=position.dtype
        )
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-rigid-joint-graph",
                "plan": plan.plan_id,
                "bodies": bodies.prepared_id,
                "reference": array_tree_fingerprint(
                    {
                        "position": position,
                        "orientation": np.asarray(reference.orientation),
                    }
                ),
            }
        )

    @staticmethod
    def _endpoints(plan, bodies, active, fixed_mask):
        if plan is None:
            empty = np.empty((0,), dtype=np.int32)
            return empty, empty
        body_ids = np.asarray(bodies.particles.particle_ids)
        order = np.argsort(body_ids)
        sorted_ids = body_ids[order]
        left_rank = np.searchsorted(sorted_ids, np.asarray(plan.left_body_ids))
        right_rank = np.searchsorted(sorted_ids, np.asarray(plan.right_body_ids))
        left_valid = (left_rank < sorted_ids.size) & (
            sorted_ids[np.minimum(left_rank, max(sorted_ids.size - 1, 0))]
            == np.asarray(plan.left_body_ids)
        )
        right_valid = (right_rank < sorted_ids.size) & (
            sorted_ids[np.minimum(right_rank, max(sorted_ids.size - 1, 0))]
            == np.asarray(plan.right_body_ids)
        )
        if not np.all(left_valid & right_valid):
            raise ValueError("Joint endpoint ID is absent from rigid-body support.")
        left = order[left_rank].astype(np.int32)
        right = order[right_rank].astype(np.int32)
        if not np.all(active[left] & active[right]):
            raise ValueError("Joint endpoints must be active bodies.")
        if np.any(fixed_mask[left] & fixed_mask[right]):
            raise ValueError("A joint must have at least one mobile endpoint.")
        return left, right

    @staticmethod
    def _local_anchors(anchors, left, right, position, rotation):
        if anchors is None:
            empty = np.empty((0, position.shape[-1]), dtype=position.dtype)
            return empty, empty
        left_offset = anchors - position[left]
        right_offset = anchors - position[right]
        return (
            contract("nji,nj->ni", rotation[left], left_offset),
            contract("nji,nj->ni", rotation[right], right_offset),
        )

    @property
    def mobile_count(self) -> int:
        return int(self.mobile_indices.shape[0])

    @property
    def constraint_count(self) -> int:
        return self.plan.constraint_count_for_dimension(self.bodies.ambient_dimension)

    def pack_residuals(self, value: RigidJointResiduals, /) -> Array:
        return jnp.concatenate(
            (
                jnp.concatenate(
                    (value.fixed_translation, value.fixed_rotation), axis=-1
                ).reshape(-1),
                value.ball_anchor.reshape(-1),
                jnp.concatenate((value.hinge_anchor, value.hinge_axis), axis=-1).reshape(
                    -1
                ),
                jnp.concatenate(
                    (
                        value.prismatic_translation,
                        value.prismatic_rotation,
                    ),
                    axis=-1,
                ).reshape(-1),
                value.distance.reshape(-1),
            ),
            axis=0,
        )

    def pack_multipliers(self, value: RigidJointMultipliers, /) -> Array:
        return self.pack_residuals(
            RigidJointResiduals(
                value.fixed_translation,
                value.fixed_rotation,
                value.ball_anchor,
                value.hinge_anchor,
                value.hinge_axis,
                value.prismatic_translation,
                value.prismatic_rotation,
                value.distance,
            )
        )

    def empty_multipliers(self, dtype=None, /) -> RigidJointMultipliers:
        dtype_ = self.bodies.particles.safe_masses.dtype if dtype is None else dtype
        dimension = self.bodies.ambient_dimension
        angular = self.bodies.angular_dimension
        return RigidJointMultipliers(
            jnp.zeros((self.fixed_left.shape[0], dimension), dtype=dtype_),
            jnp.zeros((self.fixed_left.shape[0], angular), dtype=dtype_),
            jnp.zeros((self.ball_left.shape[0], dimension), dtype=dtype_),
            jnp.zeros((self.hinge_left.shape[0], 3), dtype=dtype_),
            jnp.zeros((self.hinge_left.shape[0], 2), dtype=dtype_),
            jnp.zeros((self.prismatic_left.shape[0], dimension - 1), dtype=dtype_),
            jnp.zeros((self.prismatic_left.shape[0], angular), dtype=dtype_),
            jnp.zeros((self.distance_left.shape[0],), dtype=dtype_),
        )

    def empty_increment(self, dtype=None, /) -> _RigidMobileIncrement:
        dtype_ = self.bodies.particles.safe_masses.dtype if dtype is None else dtype
        return _RigidMobileIncrement(
            jnp.zeros((self.mobile_count, self.bodies.ambient_dimension), dtype=dtype_),
            jnp.zeros((self.mobile_count, self.bodies.angular_dimension), dtype=dtype_),
        )

    def retract(
        self,
        base: RigidBodyKinematics,
        increment: _RigidMobileIncrement,
        /,
    ) -> RigidBodyKinematics:
        translation = (
            jnp.zeros_like(base.position)
            .at[self.mobile_indices]
            .set(increment.translation)
        )
        rotation = (
            jnp.zeros_like(base.angular_velocity)
            .at[self.mobile_indices]
            .set(increment.rotation)
        )
        return _rigid_body_retract_pose(self.bodies, base, translation, rotation)

    def residuals(self, kinematics: RigidBodyKinematics, /) -> RigidJointResiduals:
        rotation = _rigid_body_rotation_matrix(self.bodies, kinematics.orientation)
        dimension = self.bodies.ambient_dimension

        fixed_relative = (
            kinematics.position[self.fixed_right] - kinematics.position[self.fixed_left]
        )
        fixed_translation = (
            contract("...ji,...j->...i", rotation[self.fixed_left], fixed_relative)
            - self.fixed_rest_offset
        )
        if dimension == 2:
            fixed_current_orientation = _principal_angle(
                kinematics.orientation[self.fixed_right]
                - kinematics.orientation[self.fixed_left]
            )
        else:
            fixed_current_orientation = _quaternion_multiply(
                _quaternion_conjugate(kinematics.orientation[self.fixed_left]),
                kinematics.orientation[self.fixed_right],
            )
        fixed_rotation = _rigid_body_relative_rotation(
            self.bodies,
            self.fixed_rest_orientation,
            fixed_current_orientation,
        )

        ball_left_offset = contract(
            "...ij,...j->...i", rotation[self.ball_left], self.ball_anchor_left
        )
        ball_right_offset = contract(
            "...ij,...j->...i", rotation[self.ball_right], self.ball_anchor_right
        )
        ball_anchor = (
            kinematics.position[self.ball_left]
            + ball_left_offset
            - kinematics.position[self.ball_right]
            - ball_right_offset
        )

        if dimension == 3:
            hinge_left_offset = contract(
                "...ij,...j->...i", rotation[self.hinge_left], self.hinge_anchor_left
            )
            hinge_right_offset = contract(
                "...ij,...j->...i",
                rotation[self.hinge_right],
                self.hinge_anchor_right,
            )
            hinge_anchor = (
                kinematics.position[self.hinge_left]
                + hinge_left_offset
                - kinematics.position[self.hinge_right]
                - hinge_right_offset
            )
            left_axis = contract(
                "...ij,...j->...i", rotation[self.hinge_left], self.hinge_axis_left
            )
            transverse_1 = contract(
                "...ij,...j->...i",
                rotation[self.hinge_right],
                self.hinge_transverse_right_1,
            )
            transverse_2 = contract(
                "...ij,...j->...i",
                rotation[self.hinge_right],
                self.hinge_transverse_right_2,
            )
            hinge_axis = jnp.stack(
                (
                    jnp.sum(left_axis * transverse_1, axis=-1),
                    jnp.sum(left_axis * transverse_2, axis=-1),
                ),
                axis=-1,
            )
        else:
            dtype = kinematics.position.dtype
            hinge_anchor = jnp.zeros((0, 3), dtype=dtype)
            hinge_axis = jnp.zeros((0, 2), dtype=dtype)

        prismatic_left_offset = contract(
            "...ij,...j->...i",
            rotation[self.prismatic_left],
            self.prismatic_anchor_left,
        )
        prismatic_right_offset = contract(
            "...ij,...j->...i",
            rotation[self.prismatic_right],
            self.prismatic_anchor_right,
        )
        prismatic_separation = (
            kinematics.position[self.prismatic_right]
            + prismatic_right_offset
            - kinematics.position[self.prismatic_left]
            - prismatic_left_offset
        )
        prismatic_transverse_world = contract(
            "...ij,...kj->...ki",
            rotation[self.prismatic_left],
            self.prismatic_transverse_left,
        )
        prismatic_translation = jnp.sum(
            prismatic_transverse_world * prismatic_separation[:, None, :],
            axis=-1,
        )
        if dimension == 2:
            prismatic_current_orientation = _principal_angle(
                kinematics.orientation[self.prismatic_right]
                - kinematics.orientation[self.prismatic_left]
            )
        else:
            prismatic_current_orientation = _quaternion_multiply(
                _quaternion_conjugate(kinematics.orientation[self.prismatic_left]),
                kinematics.orientation[self.prismatic_right],
            )
        prismatic_rotation = _rigid_body_relative_rotation(
            self.bodies,
            self.prismatic_rest_orientation,
            prismatic_current_orientation,
        )

        distance_left_offset = contract(
            "...ij,...j->...i",
            rotation[self.distance_left],
            self.distance_anchor_left,
        )
        distance_right_offset = contract(
            "...ij,...j->...i",
            rotation[self.distance_right],
            self.distance_anchor_right,
        )
        distance_vector = (
            kinematics.position[self.distance_right]
            + distance_right_offset
            - kinematics.position[self.distance_left]
            - distance_left_offset
        )
        distance = (
            jnp.sum(distance_vector * distance_vector, axis=-1)
            - self.distance_rest_length * self.distance_rest_length
        ) / (2.0 * self.distance_rest_length)
        return RigidJointResiduals(
            fixed_translation,
            fixed_rotation,
            ball_anchor,
            hinge_anchor,
            hinge_axis,
            prismatic_translation,
            prismatic_rotation,
            distance,
        )

    def hinge_alignment(self, kinematics: RigidBodyKinematics, /) -> Array:
        if self.bodies.ambient_dimension == 2:
            return jnp.ones((0,), dtype=kinematics.position.dtype)
        rotation = _rigid_body_rotation_matrix(self.bodies, kinematics.orientation)
        left_axis = contract(
            "...ij,...j->...i", rotation[self.hinge_left], self.hinge_axis_left
        )
        right_axis = contract(
            "...ij,...j->...i", rotation[self.hinge_right], self.hinge_axis_right
        )
        return jnp.sum(left_axis * right_axis, axis=-1)

    def velocity_residuals(
        self,
        kinematics: RigidBodyKinematics,
        linear_velocity: Array,
        angular_velocity: Array,
        /,
    ) -> RigidJointResiduals:
        zero = self.empty_increment(kinematics.position.dtype)
        tangent = _RigidMobileIncrement(linear_velocity, angular_velocity)
        function = lambda increment: self.residuals(self.retract(kinematics, increment))
        return jax.jvp(function, (zero,), (tangent,))[1]

    def current_velocity_residuals(
        self, kinematics: RigidBodyKinematics, /
    ) -> RigidJointResiduals:
        return self.velocity_residuals(
            kinematics,
            kinematics.velocity[self.mobile_indices],
            kinematics.angular_velocity[self.mobile_indices],
        )


def rigid_joint_pairing(
    multipliers: RigidJointMultipliers,
    residuals: RigidJointResiduals,
    /,
) -> Array:
    return (
        jnp.vdot(multipliers.fixed_translation, residuals.fixed_translation)
        + jnp.vdot(multipliers.fixed_rotation, residuals.fixed_rotation)
        + jnp.vdot(multipliers.ball_anchor, residuals.ball_anchor)
        + jnp.vdot(multipliers.hinge_anchor, residuals.hinge_anchor)
        + jnp.vdot(multipliers.hinge_axis, residuals.hinge_axis)
        + jnp.vdot(multipliers.prismatic_translation, residuals.prismatic_translation)
        + jnp.vdot(multipliers.prismatic_rotation, residuals.prismatic_rotation)
        + jnp.vdot(multipliers.distance, residuals.distance)
    ).real


def rigid_joint_maximum_residual(residuals: RigidJointResiduals, /) -> Array:
    leaves = jax.tree.leaves(residuals)
    maxima = tuple(
        jnp.max(jnp.abs(value), initial=jnp.asarray(0.0, dtype=value.dtype))
        for value in leaves
    )
    return jnp.max(jnp.stack(maxima), initial=0.0)


__all__ = [
    "DistanceJointSetPlan",
    "BallJointSetPlan",
    "FixedJointSetPlan",
    "HingeJointSetPlan",
    "PrismaticJointSetPlan",
    "PreparedRigidJointGraph",
    "RigidJointGraphPlan",
    "RigidJointKind",
    "RigidJointRowLayout",
    "RigidJointMultipliers",
    "RigidJointResiduals",
    "rigid_joint_maximum_residual",
]
