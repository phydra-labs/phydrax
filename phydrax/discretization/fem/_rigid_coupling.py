#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


if TYPE_CHECKING:
    from ..particle._rigid_body import (
        PreparedRigidBodySet,
        RigidBodyKinematics,
        RigidBodyLoad,
    )
from ...linalg import (
    ArraySpace,
    BlockSpace,
    FunctionLinearOperator,
)
from ._point_interpolation import PreparedFiniteElementPointInterpolation


class AttachmentRankEvidence(StrictModule, NonTrainableState):
    row_count: int = eqx.field(static=True)
    coordinate_count: int = eqx.field(static=True)
    numerical_rank: int = eqx.field(static=True)
    smallest_singular_value: float = eqx.field(static=True)
    rank_tolerance: float = eqx.field(static=True)
    full_row_rank: bool = eqx.field(static=True)


class RigidDeformableAttachmentEvaluation(StrictModule):
    deformable_points: Array
    rigid_points: Array
    body_levers: Array
    residual: Array
    maximum_residual: Array
    finite: Array


class RigidDeformableKKTLinearization(StrictModule, NonTrainableState):
    interpolation: PreparedFiniteElementPointInterpolation
    bodies: PreparedRigidBodySet
    body_indices: Array
    body_levers: Array
    operator_id: str = eqx.field(static=True)

    def constraint_action(
        self,
        deformable_increment: ArrayLike,
        body_translation_increment: ArrayLike,
        body_rotation_increment: ArrayLike,
        /,
    ) -> Array:
        translation = jnp.asarray(body_translation_increment)
        rotation = jnp.asarray(body_rotation_increment)
        expected = (self.bodies.capacity, 3)
        if translation.shape != expected or rotation.shape != expected:
            raise ValueError("Rigid KKT increments must have body-capacity 3-D shape.")
        deformable = self.interpolation.interpolate(deformable_increment)
        rotational = jnp.cross(self.body_levers, rotation[self.body_indices])
        return deformable - translation[self.body_indices] + rotational

    def transpose_action(
        self,
        multiplier: ArrayLike,
        /,
    ) -> tuple[Array, RigidBodyLoad]:
        from ..particle._rigid_body import RigidBodyLoad

        value = jnp.asarray(multiplier, dtype=self.body_levers.dtype)
        if value.shape != self.body_levers.shape:
            raise ValueError("Attachment multiplier shape is invalid.")
        deformable = self.interpolation.transpose_scatter(value)
        force = (
            jnp.zeros((self.bodies.capacity, 3), dtype=value.dtype)
            .at[self.body_indices]
            .add(-value)
        )
        torque = (
            jnp.zeros_like(force)
            .at[self.body_indices]
            .add(jnp.cross(self.body_levers, -value))
        )
        return deformable, RigidBodyLoad(force, torque)

    def as_linear_operator(self, /) -> FunctionLinearOperator:
        dtype = self.interpolation.field_space.dtype
        rigid_space = ArraySpace((self.bodies.capacity, 3), dtype=dtype)
        source = BlockSpace(
            (self.interpolation.field_space, rigid_space, rigid_space),
            names=("deformable", "rigid_translation", "rigid_rotation"),
        )
        target = ArraySpace(self.body_levers.shape, dtype=dtype)

        def transpose(multiplier):
            deformable, rigid = self.transpose_action(multiplier)
            return deformable, rigid.force, rigid.torque

        return FunctionLinearOperator(
            lambda increment: self.constraint_action(*increment),
            source=source,
            target=target,
            transpose_action=transpose,
            operator_id=self.operator_id,
        )

    def duality_residual(
        self,
        deformable_increment: ArrayLike,
        body_translation_increment: ArrayLike,
        body_rotation_increment: ArrayLike,
        multiplier: ArrayLike,
        /,
    ) -> Array:
        constraint = self.constraint_action(
            deformable_increment,
            body_translation_increment,
            body_rotation_increment,
        )
        deformable_load, rigid_load = self.transpose_action(multiplier)
        left = jnp.sum(constraint * jnp.asarray(multiplier))
        right = (
            jnp.sum(jnp.asarray(deformable_increment) * deformable_load)
            + jnp.sum(jnp.asarray(body_translation_increment) * rigid_load.force)
            + jnp.sum(jnp.asarray(body_rotation_increment) * rigid_load.torque)
        )
        return left - right


class AttachmentActionReactionCertificate(StrictModule):
    deformable_resultant: Array
    rigid_resultant: Array
    force_balance: Array
    deformable_moment: Array
    rigid_moment: Array
    moment_balance: Array
    interpolation_duality_residual: Array
    maximum_force_defect: Array
    maximum_moment_defect: Array
    finite: Array
    valid: Array


class RigidDeformableKKTPayload(StrictModule):
    constraint_residual: Array
    multiplier: Array
    deformable_load: Array
    rigid_load: RigidBodyLoad
    linearization: RigidDeformableKKTLinearization
    operator: FunctionLinearOperator
    certificate: AttachmentActionReactionCertificate
    finite: Array
    valid: Array


class RigidDeformableAttachmentPlan(StrictModule, NonTrainableState):
    """Fixed FE-to-rigid translational constraints with exact KKT actions."""

    interpolation: PreparedFiniteElementPointInterpolation
    bodies: PreparedRigidBodySet
    body_indices: Array
    local_anchors: Array
    rank: AttachmentRankEvidence
    tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        interpolation: PreparedFiniteElementPointInterpolation,
        bodies: PreparedRigidBodySet,
        body_ids: ArrayLike,
        local_anchors: ArrayLike,
        /,
        *,
        rank_tolerance: float = 1.0e-10,
        tolerance: float = 1.0e-9,
        plan_id: str | None = None,
    ):
        if not isinstance(interpolation, PreparedFiniteElementPointInterpolation):
            raise TypeError("interpolation must be prepared FE point interpolation.")
        from ..particle._rigid_body import PreparedRigidBodySet

        if not isinstance(bodies, PreparedRigidBodySet):
            raise TypeError("bodies must be PreparedRigidBodySet.")
        if interpolation.ambient_dimension != 3 or bodies.ambient_dimension != 3:
            raise ValueError(
                "Rigid-deformable attachments currently require 3-D supports."
            )
        if interpolation.derivative_axis is not None or interpolation.value_shape != (
            interpolation.ambient_dimension,
        ):
            raise ValueError(
                "Rigid coupling requires value interpolation of a nodal vector field "
                "matching mesh dimension."
            )
        identifiers = np.asarray(body_ids)
        anchors = np.asarray(local_anchors)
        count = interpolation.attachment_count
        if (
            identifiers.shape != (count,)
            or not np.issubdtype(identifiers.dtype, np.integer)
            or anchors.shape != (count, 3)
            or np.any(~np.isfinite(anchors))
        ):
            raise ValueError("Attachment body IDs/local anchors have invalid layout.")
        particle_ids = np.asarray(bodies.particles.particle_ids)
        order = np.argsort(particle_ids, kind="stable")
        sorted_ids = particle_ids[order]
        ranks = np.searchsorted(sorted_ids, identifiers)
        safe = np.minimum(ranks, max(sorted_ids.size - 1, 0))
        present = (ranks < sorted_ids.size) & (sorted_ids[safe] == identifiers)
        if not np.all(present):
            raise ValueError("An attachment body ID is absent from rigid-body support.")
        indices = order[ranks].astype(np.int32)
        active = np.asarray(bodies.particles.active_mask, dtype=np.bool_)
        if not np.all(active[indices]):
            raise ValueError("Attachments require active rigid bodies.")
        if np.any(np.asarray(bodies.fixed_mask, dtype=np.bool_)[indices]):
            raise ValueError("KKT attachments require mobile rigid bodies.")
        rank_limit = float(rank_tolerance)
        physical_limit = float(tolerance)
        if (
            not isfinite(rank_limit)
            or rank_limit <= 0.0
            or not isfinite(physical_limit)
            or physical_limit < 0.0
        ):
            raise ValueError("Attachment rank/physical tolerances are invalid.")
        rows = 3 * count
        field_coordinates = interpolation.field_space.size
        rigid_coordinates = 6 * bodies.capacity
        matrix = np.zeros((rows, field_coordinates + rigid_coordinates))
        routes = np.asarray(interpolation.dof_routes)
        weights = np.asarray(interpolation.weights)
        for attachment in range(count):
            row = 3 * attachment
            for local, dof in enumerate(routes[attachment]):
                for component in range(3):
                    matrix[row + component, 3 * int(dof) + component] += weights[
                        attachment, local
                    ]
            body = int(indices[attachment])
            rigid_start = field_coordinates + 6 * body
            matrix[row : row + 3, rigid_start : rigid_start + 3] -= np.eye(3)
            x, y, z = anchors[attachment]
            cross_matrix = np.asarray(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))
            matrix[row : row + 3, rigid_start + 3 : rigid_start + 6] += cross_matrix
        singular_values = np.linalg.svd(matrix, compute_uv=False)
        largest = max(float(singular_values[0]), 1.0)
        numerical_rank = int(np.count_nonzero(singular_values > rank_limit * largest))
        smallest = float(singular_values[-1])
        evidence = AttachmentRankEvidence(
            rows,
            matrix.shape[1],
            numerical_rank,
            smallest,
            rank_limit,
            numerical_rank == rows,
        )
        if not evidence.full_row_rank:
            raise ValueError(
                "Rigid-deformable attachment rows are duplicate or rank deficient."
            )
        generated = canonical_fingerprint(
            {
                "kind": "rigid-deformable-attachment-plan",
                "interpolation": interpolation.prepared_id,
                "bodies": bodies.prepared_id,
                "body_indices": array_tree_fingerprint(indices),
                "local_anchors": array_tree_fingerprint(anchors),
                "rank_tolerance": rank_limit.hex(),
                "tolerance": physical_limit.hex(),
            }
        )
        identifier = generated if plan_id is None else str(plan_id)
        if not identifier:
            raise ValueError("plan_id must be non-empty or None.")
        self.interpolation = interpolation
        self.bodies = bodies
        self.body_indices = jnp.asarray(indices)
        self.local_anchors = jnp.asarray(anchors, dtype=interpolation.weights.dtype)
        self.rank = evidence
        self.tolerance = physical_limit
        self.plan_id = identifier

    def evaluate(
        self,
        displacement: ArrayLike,
        kinematics: RigidBodyKinematics,
        /,
    ) -> RigidDeformableAttachmentEvaluation:
        from ..particle._rigid_body import (
            quaternion_rotation_matrix,
            RigidBodyKinematics,
        )

        if not isinstance(kinematics, RigidBodyKinematics):
            raise TypeError("kinematics must be RigidBodyKinematics.")
        expected = (self.bodies.capacity, 3)
        if (
            kinematics.position.shape != expected
            or kinematics.velocity.shape != expected
            or kinematics.orientation.shape != (self.bodies.capacity, 4)
            or kinematics.angular_velocity.shape != expected
        ):
            raise ValueError("Rigid kinematics do not match the attachment body set.")
        rotation = quaternion_rotation_matrix(kinematics.orientation)
        levers = contract("aij,aj->ai", rotation[self.body_indices], self.local_anchors)
        rigid_points = kinematics.position[self.body_indices] + levers
        deformable_points = (
            self.interpolation.reference_positions
            + self.interpolation.interpolate(displacement)
        )
        residual = deformable_points - rigid_points
        maximum = jnp.max(jnp.abs(residual), initial=0.0)
        finite = jnp.all(
            jnp.isfinite(
                jnp.concatenate(
                    (deformable_points, rigid_points, levers, residual), axis=0
                )
            )
        )
        return RigidDeformableAttachmentEvaluation(
            deformable_points,
            rigid_points,
            levers,
            residual,
            maximum,
            finite,
        )

    def linearization(
        self,
        evaluation: RigidDeformableAttachmentEvaluation,
        /,
    ) -> RigidDeformableKKTLinearization:
        if not isinstance(evaluation, RigidDeformableAttachmentEvaluation):
            raise TypeError("evaluation must be RigidDeformableAttachmentEvaluation.")
        return RigidDeformableKKTLinearization(
            self.interpolation,
            self.bodies,
            self.body_indices,
            evaluation.body_levers,
            canonical_fingerprint(
                {
                    "kind": "rigid-deformable-kkt-linearization",
                    "plan": self.plan_id,
                }
            ),
        )

    def kkt_payload(
        self,
        displacement: ArrayLike,
        kinematics: RigidBodyKinematics,
        multiplier: ArrayLike,
        /,
    ) -> RigidDeformableKKTPayload:
        evaluation = self.evaluate(displacement, kinematics)
        linearization = self.linearization(evaluation)
        operator = linearization.as_linear_operator()
        value = jnp.asarray(multiplier, dtype=evaluation.residual.dtype)
        if value.shape != evaluation.residual.shape:
            raise ValueError(
                "Attachment multiplier must match constraint residual shape."
            )
        deformable_load, rigid_load = linearization.transpose_action(value)
        deformable_resultant = jnp.sum(deformable_load, axis=0)
        rigid_resultant = jnp.sum(rigid_load.force, axis=0)
        force_balance = deformable_resultant + rigid_resultant
        deformed_dofs = self.interpolation.deformed_dof_positions(displacement)
        deformable_moment = jnp.sum(jnp.cross(deformed_dofs, deformable_load), axis=0)
        rigid_moment = jnp.sum(
            jnp.cross(kinematics.position, rigid_load.force) + rigid_load.torque,
            axis=0,
        )
        moment_balance = deformable_moment + rigid_moment
        duality = self.interpolation.duality_evidence(displacement, value)
        maximum_force = jnp.max(jnp.abs(force_balance), initial=0.0)
        maximum_moment = jnp.max(jnp.abs(moment_balance), initial=0.0)
        finite = evaluation.finite & jnp.all(
            jnp.isfinite(
                jnp.concatenate(
                    (
                        deformable_load,
                        rigid_load.force,
                        rigid_load.torque,
                        force_balance[None, :],
                        moment_balance[None, :],
                    ),
                    axis=0,
                )
            )
        )
        certificate_valid = (
            finite
            & duality.valid
            & (maximum_force <= self.tolerance)
            & (maximum_moment <= self.tolerance)
        )
        valid = certificate_valid & (evaluation.maximum_residual <= self.tolerance)
        certificate = AttachmentActionReactionCertificate(
            deformable_resultant,
            rigid_resultant,
            force_balance,
            deformable_moment,
            rigid_moment,
            moment_balance,
            duality.residual,
            maximum_force,
            maximum_moment,
            finite,
            certificate_valid,
        )
        return RigidDeformableKKTPayload(
            evaluation.residual,
            value,
            deformable_load,
            rigid_load,
            linearization,
            operator,
            certificate,
            finite,
            valid,
        )


__all__ = [
    "AttachmentActionReactionCertificate",
    "AttachmentRankEvidence",
    "RigidDeformableAttachmentEvaluation",
    "RigidDeformableAttachmentPlan",
    "RigidDeformableKKTLinearization",
    "RigidDeformableKKTPayload",
]
