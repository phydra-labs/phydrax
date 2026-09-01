#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, PyTree
from opt_einsum import contract

from ..._fingerprint import canonical_fingerprint
from ...discretization.contact._guarantee import (
    ContactCapability,
    ContactGuaranteeLevel,
)
from ...discretization.contact._participant import (
    AbstractContactParticipant,
    FunctionContactParticipant,
    LinearContactParticipant,
    ParticipantTrajectoryBounds,
)
from ...discretization.contact._precision import ContactPrecisionPolicy
from ...discretization.contact._surface import (
    CollisionSurfacePlan,
    ContactPairPolicy,
    PreparedCollisionSurface,
    selection_collision_operator,
)
from ...linalg import (
    AbstractLinearOperator,
    AbstractVectorSpace,
    ArraySpace,
    BlockSpace,
)


def _rigid_rotation(rotation_vector):
    dimension = rotation_vector.shape[-1]
    if dimension == 1:
        angle = rotation_vector[..., 0]
        cosine = jnp.cos(angle)
        sine = jnp.sin(angle)
        return jnp.stack((cosine, -sine, sine, cosine), axis=-1).reshape(
            rotation_vector.shape[:-1] + (2, 2)
        )
    angle_squared = jnp.sum(rotation_vector * rotation_vector, axis=-1)
    angle = jnp.sqrt(jnp.maximum(angle_squared, jnp.finfo(rotation_vector.dtype).eps))
    safe_angle = jnp.where(angle > 0.0, angle, 1.0)
    axis = rotation_vector / safe_angle[..., None]
    x, y, z = (axis[..., index] for index in range(3))
    zero = jnp.zeros_like(x)
    skew = jnp.stack((zero, -z, y, z, zero, -x, -y, x, zero), axis=-1).reshape(
        rotation_vector.shape[:-1] + (3, 3)
    )
    identity = jnp.broadcast_to(jnp.eye(3, dtype=rotation_vector.dtype), skew.shape)
    sine_scale = jnp.where(
        angle_squared > jnp.finfo(rotation_vector.dtype).eps,
        jnp.sin(angle),
        angle - angle**3 / 6.0,
    )
    cosine_scale = jnp.where(
        angle_squared > jnp.finfo(rotation_vector.dtype).eps,
        1.0 - jnp.cos(angle),
        0.5 * angle_squared - angle_squared**2 / 24.0,
    )
    return (
        identity
        + sine_scale[..., None, None] * skew
        + cosine_scale[..., None, None] * contract("...ij,...jk->...ik", skew, skew)
    )


class RigidContactParticipant(AbstractContactParticipant):
    plan: CollisionSurfacePlan
    local_vertices: Array
    vertex_owner: Array
    body_count: int = eqx.field(static=True)
    space: BlockSpace
    _participant_id: str = eqx.field(static=True)
    _capabilities: ContactCapability = eqx.field(static=True)

    def __init__(
        self,
        plan: CollisionSurfacePlan,
        local_vertices: ArrayLike,
        vertex_owner: ArrayLike,
        /,
        *,
        body_count: int,
    ):
        if not isinstance(plan, CollisionSurfacePlan):
            raise TypeError("plan must be CollisionSurfacePlan.")
        vertices = jnp.asarray(local_vertices)
        owner = np.asarray(vertex_owner)
        bodies = int(body_count)
        if vertices.shape != (
            plan.vertex_count,
            plan.ambient_dimension,
        ):
            raise ValueError("Rigid local vertices do not match collision topology.")
        if owner.shape != (plan.vertex_count,) or not np.issubdtype(
            owner.dtype, np.integer
        ):
            raise TypeError("Rigid vertex owners must be one integer vector.")
        if bodies <= 0 or np.any(owner < 0) or np.any(owner >= bodies):
            raise ValueError("Rigid vertex owner is invalid.")
        angular_dimension = 1 if plan.ambient_dimension == 2 else 3
        space = BlockSpace(
            (
                ArraySpace((bodies, plan.ambient_dimension), dtype=vertices.dtype),
                ArraySpace((bodies, angular_dimension), dtype=vertices.dtype),
            ),
            names=("translation", "rotation"),
        )
        self.plan = plan
        self.local_vertices = vertices
        self.vertex_owner = jnp.asarray(owner, dtype=jnp.int32)
        self.body_count = bodies
        self.space = space
        self._participant_id = canonical_fingerprint(
            {
                "kind": "rigid-contact-participant",
                "surface": plan.topology_id,
                "body_count": bodies,
                "owner": owner.tolist(),
            }
        )
        self._capabilities = (
            ContactCapability.STATIC_DISTANCE
            | ContactCapability.NONLINEAR_TRAJECTORY
            | ContactCapability.DIFFERENTIABLE_KINEMATICS
            | ContactCapability.FORCE_PULLBACK
        )

    @property
    def source_space(self) -> AbstractVectorSpace:
        return self.space

    @property
    def surface_plan(self) -> CollisionSurfacePlan:
        return self.plan

    @property
    def participant_id(self) -> str:
        return self._participant_id

    @property
    def capabilities(self) -> ContactCapability:
        return self._capabilities

    def positions(self, state: PyTree, /) -> Array:
        translation, rotation_vector = self.space.validate(state)
        rotation = _rigid_rotation(rotation_vector)
        world_offset = contract(
            "vij,vj->vi",
            rotation[self.vertex_owner],
            self.local_vertices,
        )
        return translation[self.vertex_owner] + world_offset

    def velocities(self, state: PyTree, rates: PyTree, /) -> Array:
        translation, rotation_vector = self.space.validate(state)
        linear_velocity, angular_velocity = self.space.validate(rates)
        rotation = _rigid_rotation(rotation_vector)
        world_offset = contract(
            "vij,vj->vi",
            rotation[self.vertex_owner],
            self.local_vertices,
        )
        if self.plan.ambient_dimension == 2:
            spin = angular_velocity[self.vertex_owner, 0]
            rotational = jnp.stack(
                (-spin * world_offset[:, 1], spin * world_offset[:, 0]),
                axis=-1,
            )
        else:
            rotational = jnp.cross(angular_velocity[self.vertex_owner], world_offset)
        return linear_velocity[self.vertex_owner] + rotational

    def force_pullback(self, state: PyTree, surface_force: ArrayLike, /):
        translation, rotation_vector = self.space.validate(state)
        del translation
        force = jnp.asarray(surface_force, dtype=self.local_vertices.dtype)
        if force.shape != self.local_vertices.shape:
            raise ValueError("Rigid surface force has invalid shape.")
        rotation = _rigid_rotation(rotation_vector)
        arm = contract(
            "vij,vj->vi",
            rotation[self.vertex_owner],
            self.local_vertices,
        )
        body_force = (
            jnp.zeros((self.body_count, self.plan.ambient_dimension), dtype=force.dtype)
            .at[self.vertex_owner]
            .add(force)
        )
        if self.plan.ambient_dimension == 2:
            torque_value = arm[:, 0] * force[:, 1] - arm[:, 1] * force[:, 0]
            body_torque = (
                jnp.zeros((self.body_count, 1), dtype=force.dtype)
                .at[self.vertex_owner, 0]
                .add(torque_value)
            )
        else:
            body_torque = (
                jnp.zeros((self.body_count, 3), dtype=force.dtype)
                .at[self.vertex_owner]
                .add(jnp.cross(arm, force))
            )
        return body_force, body_torque

    def trajectory_bounds(
        self, start_state: PyTree, end_state: PyTree, /
    ) -> ParticipantTrajectoryBounds:
        start_translation, _ = self.space.validate(start_state)
        end_translation, _ = self.space.validate(end_state)
        radius = (
            jnp.zeros((self.body_count,), dtype=self.local_vertices.dtype)
            .at[self.vertex_owner]
            .max(jnp.sqrt(jnp.sum(self.local_vertices * self.local_vertices, axis=-1)))
        )
        lower_center = jnp.minimum(start_translation, end_translation)
        upper_center = jnp.maximum(start_translation, end_translation)
        lower = lower_center[self.vertex_owner] - radius[self.vertex_owner, None]
        upper = upper_center[self.vertex_owner] + radius[self.vertex_owner, None]
        finite = jnp.all(jnp.isfinite(lower)) & jnp.all(jnp.isfinite(upper))
        return ParticipantTrajectoryBounds(
            lower,
            upper,
            jnp.asarray(
                int(ContactGuaranteeLevel.ANALYTIC_CONSERVATIVE),
                dtype=jnp.int32,
            ),
            finite,
            finite,
            self.participant_id,
        )


def make_articulated_contact_participant(
    plan: CollisionSurfacePlan,
    source_space: AbstractVectorSpace,
    forward_kinematics: Callable[[PyTree], Array],
    /,
    *,
    velocity_action: Callable[[PyTree, PyTree], Array] | None = None,
    pullback_action: Callable[[PyTree, Array], PyTree] | None = None,
    bounds_action: Callable[[PyTree, PyTree], tuple[Array, Array]] | None = None,
    participant_id: str | None = None,
) -> FunctionContactParticipant:
    return FunctionContactParticipant(
        plan,
        source_space,
        forward_kinematics,
        velocity_action=velocity_action,
        pullback_action=pullback_action,
        bounds_action=bounds_action,
        participant_id=participant_id,
    )


def prepare_point_contact_participant(
    source_space: ArraySpace,
    reference_positions: ArrayLike,
    /,
    *,
    vertex_ids: ArrayLike | None = None,
    body_ids: ArrayLike | None = None,
    material_ids: ArrayLike | None = None,
    minimum_separation: ArrayLike = 0.0,
    precision: ContactPrecisionPolicy | None = None,
) -> LinearContactParticipant:
    reference = jnp.asarray(reference_positions)
    if reference.ndim != 2 or reference.shape != source_space.shape:
        raise ValueError(
            "Point contact reference positions must match the source ArraySpace."
        )
    count, dimension = reference.shape
    identifiers = (
        jnp.arange(count, dtype=jnp.int64)
        if vertex_ids is None
        else jnp.asarray(vertex_ids)
    )
    policy = ContactPairPolicy(
        count,
        body_ids=body_ids,
        material_ids=material_ids,
    )
    plan = CollisionSurfacePlan(
        identifiers,
        ambient_dimension=dimension,
        edges=jnp.empty((0, 2), dtype=jnp.int32),
        codimensional_mask=jnp.ones((count,), dtype=bool),
        pair_policy=policy,
        minimum_separation=minimum_separation,
        allow_isolated_vertices=True,
    )
    surface = PreparedCollisionSurface(
        plan,
        reference,
        selection_collision_operator(source_space, jnp.arange(count, dtype=jnp.int32)),
        precision=precision,
    )
    return LinearContactParticipant(surface)


def prepare_mpm_contact_participant(
    plan: CollisionSurfacePlan,
    reference_positions: ArrayLike,
    grid_to_contact_operator: AbstractLinearOperator,
    /,
    *,
    precision: ContactPrecisionPolicy | None = None,
) -> LinearContactParticipant:
    surface = PreparedCollisionSurface(
        plan,
        reference_positions,
        grid_to_contact_operator,
        precision=precision,
    )
    return LinearContactParticipant(surface)


__all__ = [
    "RigidContactParticipant",
    "make_articulated_contact_participant",
    "prepare_mpm_contact_participant",
    "prepare_point_contact_participant",
]
