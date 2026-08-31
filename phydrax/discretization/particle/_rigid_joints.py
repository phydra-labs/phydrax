#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

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
    _quaternion_conjugate,
    _quaternion_multiply,
    _quaternion_relative_rotation_vector,
    _rigid_body_retract_pose,
    PreparedRigidBodySet,
    quaternion_rotation_matrix,
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
    if array.shape != (count, 3):
        raise ValueError(f"{owner} must have shape ({count}, 3).")
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


class RigidJointGraphPlan(StrictModule, NonTrainableState):
    fixed: FixedJointSetPlan | None
    ball: BallJointSetPlan | None
    hinge: HingeJointSetPlan | None
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        fixed: FixedJointSetPlan | None = None,
        ball: BallJointSetPlan | None = None,
        hinge: HingeJointSetPlan | None = None,
        plan_id: str | None = None,
    ):
        if fixed is not None and not isinstance(fixed, FixedJointSetPlan):
            raise TypeError("fixed must be a FixedJointSetPlan or None.")
        if ball is not None and not isinstance(ball, BallJointSetPlan):
            raise TypeError("ball must be a BallJointSetPlan or None.")
        if hinge is not None and not isinstance(hinge, HingeJointSetPlan):
            raise TypeError("hinge must be a HingeJointSetPlan or None.")
        identifier_arrays = tuple(
            np.asarray(value.joint_ids)
            for value in (fixed, ball, hinge)
            if value is not None
        )
        if identifier_arrays:
            identifiers = np.concatenate(identifier_arrays)
            if np.unique(identifiers).size != identifiers.size:
                raise ValueError("Joint IDs must be globally unique across joint kinds.")
        generated = canonical_fingerprint(
            {
                "kind": "rigid-joint-graph-plan",
                "fixed": None if fixed is None else fixed.plan_id,
                "ball": None if ball is None else ball.plan_id,
                "hinge": None if hinge is None else hinge.plan_id,
            }
        )
        self.fixed = fixed
        self.ball = ball
        self.hinge = hinge
        self.plan_id = generated if plan_id is None else str(plan_id)
        if not self.plan_id:
            raise ValueError("plan_id must be nonempty.")

    @property
    def constraint_count(self) -> int:
        return (
            6 * (0 if self.fixed is None else self.fixed.count)
            + 3 * (0 if self.ball is None else self.ball.count)
            + 5 * (0 if self.hinge is None else self.hinge.count)
        )

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


class RigidJointMultipliers(StrictModule):
    fixed_translation: Array
    fixed_rotation: Array
    ball_anchor: Array
    hinge_anchor: Array
    hinge_axis: Array


class _RigidMobileIncrement(StrictModule):
    translation: Array
    rotation: Array


class PreparedRigidJointGraph(StrictModule, NonTrainableState):
    plan: RigidJointGraphPlan
    bodies: PreparedRigidBodySet
    mobile_indices: Array
    fixed_left: Array
    fixed_right: Array
    fixed_rest_offset: Array
    fixed_rest_orientation: Array
    ball_left: Array
    ball_right: Array
    ball_anchor_left: Array
    ball_anchor_right: Array
    hinge_left: Array
    hinge_right: Array
    hinge_anchor_left: Array
    hinge_anchor_right: Array
    hinge_axis_left: Array
    hinge_axis_right: Array
    hinge_transverse_right_1: Array
    hinge_transverse_right_2: Array
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
        if bodies.ambient_dimension != 3:
            raise ValueError(
                "Rigid joint graphs currently require three-dimensional bodies."
            )
        expected_position = (bodies.capacity, 3)
        if (
            reference.position.shape != expected_position
            or reference.velocity.shape != expected_position
            or reference.orientation.shape != (bodies.capacity, 4)
            or reference.angular_velocity.shape != expected_position
        ):
            raise ValueError("Reference rigid-body kinematics have incompatible shapes.")
        if not bool(np.asarray(tree_allfinite(reference))):
            raise ValueError("Reference rigid-body kinematics must be finite.")
        orientation_norm = np.linalg.norm(np.asarray(reference.orientation), axis=-1)
        if not np.allclose(orientation_norm, 1.0, rtol=0.0, atol=1.0e-8):
            raise ValueError("Reference rigid-body orientations must have unit norm.")

        active = np.asarray(bodies.particles.active_mask, dtype=bool)
        fixed_mask = np.asarray(bodies.fixed_mask, dtype=bool)
        mobile_mask = active & ~fixed_mask
        mobile_indices = np.flatnonzero(mobile_mask).astype(np.int32)

        fixed_left, fixed_right = self._endpoints(plan.fixed, bodies, active, fixed_mask)
        ball_left, ball_right = self._endpoints(plan.ball, bodies, active, fixed_mask)
        hinge_left, hinge_right = self._endpoints(plan.hinge, bodies, active, fixed_mask)
        if plan.constraint_count > 6 * mobile_indices.size:
            raise ValueError("Joint rows exceed available mobile rigid-body coordinates.")

        position = np.asarray(reference.position)
        orientation = reference.orientation
        rotation = np.asarray(quaternion_rotation_matrix(orientation))

        if plan.fixed is None:
            fixed_rest_offset = np.empty((0, 3), dtype=position.dtype)
            fixed_rest_orientation = jnp.empty((0, 4), dtype=orientation.dtype)
        else:
            relative = position[fixed_right] - position[fixed_left]
            fixed_rest_offset = contract("nji,nj->ni", rotation[fixed_left], relative)
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
            transverse_1 = contract("nji,nj->ni", rotation[hinge_right], basis_1_world)
            transverse_2 = contract("nji,nj->ni", rotation[hinge_right], basis_2_world)

        self.plan = plan
        self.bodies = bodies
        self.mobile_indices = jnp.asarray(mobile_indices)
        self.fixed_left = jnp.asarray(fixed_left)
        self.fixed_right = jnp.asarray(fixed_right)
        self.fixed_rest_offset = jnp.asarray(fixed_rest_offset, dtype=position.dtype)
        self.fixed_rest_orientation = fixed_rest_orientation
        self.ball_left = jnp.asarray(ball_left)
        self.ball_right = jnp.asarray(ball_right)
        self.ball_anchor_left = jnp.asarray(ball_anchor_left, dtype=position.dtype)
        self.ball_anchor_right = jnp.asarray(ball_anchor_right, dtype=position.dtype)
        self.hinge_left = jnp.asarray(hinge_left)
        self.hinge_right = jnp.asarray(hinge_right)
        self.hinge_anchor_left = jnp.asarray(hinge_anchor_left, dtype=position.dtype)
        self.hinge_anchor_right = jnp.asarray(hinge_anchor_right, dtype=position.dtype)
        self.hinge_axis_left = jnp.asarray(hinge_axis_left, dtype=position.dtype)
        self.hinge_axis_right = jnp.asarray(hinge_axis_right, dtype=position.dtype)
        self.hinge_transverse_right_1 = jnp.asarray(transverse_1, dtype=position.dtype)
        self.hinge_transverse_right_2 = jnp.asarray(transverse_2, dtype=position.dtype)
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
            empty = np.empty((0, 3), dtype=position.dtype)
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
        return self.plan.constraint_count

    def empty_multipliers(self, dtype=None, /) -> RigidJointMultipliers:
        dtype_ = self.bodies.particles.safe_masses.dtype if dtype is None else dtype
        return RigidJointMultipliers(
            jnp.zeros((self.fixed_left.shape[0], 3), dtype=dtype_),
            jnp.zeros((self.fixed_left.shape[0], 3), dtype=dtype_),
            jnp.zeros((self.ball_left.shape[0], 3), dtype=dtype_),
            jnp.zeros((self.hinge_left.shape[0], 3), dtype=dtype_),
            jnp.zeros((self.hinge_left.shape[0], 2), dtype=dtype_),
        )

    def empty_increment(self, dtype=None, /) -> _RigidMobileIncrement:
        dtype_ = self.bodies.particles.safe_masses.dtype if dtype is None else dtype
        return _RigidMobileIncrement(
            jnp.zeros((self.mobile_count, 3), dtype=dtype_),
            jnp.zeros((self.mobile_count, 3), dtype=dtype_),
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
        rotation = quaternion_rotation_matrix(kinematics.orientation)

        fixed_relative = (
            kinematics.position[self.fixed_right] - kinematics.position[self.fixed_left]
        )
        fixed_translation = (
            contract("...ji,...j->...i", rotation[self.fixed_left], fixed_relative)
            - self.fixed_rest_offset
        )
        fixed_current_orientation = _quaternion_multiply(
            _quaternion_conjugate(kinematics.orientation[self.fixed_left]),
            kinematics.orientation[self.fixed_right],
        )
        fixed_rotation = _quaternion_relative_rotation_vector(
            self.fixed_rest_orientation, fixed_current_orientation
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

        hinge_left_offset = contract(
            "...ij,...j->...i", rotation[self.hinge_left], self.hinge_anchor_left
        )
        hinge_right_offset = contract(
            "...ij,...j->...i", rotation[self.hinge_right], self.hinge_anchor_right
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
        return RigidJointResiduals(
            fixed_translation,
            fixed_rotation,
            ball_anchor,
            hinge_anchor,
            hinge_axis,
        )

    def hinge_alignment(self, kinematics: RigidBodyKinematics, /) -> Array:
        rotation = quaternion_rotation_matrix(kinematics.orientation)
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
    ).real


def rigid_joint_maximum_residual(residuals: RigidJointResiduals, /) -> Array:
    leaves = jax.tree.leaves(residuals)
    maxima = tuple(
        jnp.max(jnp.abs(value), initial=jnp.asarray(0.0, dtype=value.dtype))
        for value in leaves
    )
    return jnp.max(jnp.stack(maxima), initial=0.0)


__all__ = [
    "BallJointSetPlan",
    "FixedJointSetPlan",
    "HingeJointSetPlan",
    "PreparedRigidJointGraph",
    "RigidJointGraphPlan",
    "RigidJointMultipliers",
    "RigidJointResiduals",
    "rigid_joint_maximum_residual",
]
