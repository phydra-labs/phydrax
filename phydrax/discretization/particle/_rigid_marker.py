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
from ...linalg import FunctionLinearOperator, OperatorProperties, PyTreeSpace
from .._lagrangian_marker import (
    LagrangianMarkerDiscretization,
    LagrangianMarkerKinematics,
)
from ._rigid_body import (
    PreparedRigidBodySet,
    quaternion_rotation_matrix,
    rigid_body_world_inertia,
    RigidBodyKinematics,
)


class RigidGeneralizedVelocity(StrictModule):
    translation: Array
    rotation: Array


class RigidMarkerMapPlan(StrictModule, NonTrainableState):
    markers: LagrangianMarkerDiscretization
    bodies: PreparedRigidBodySet
    marker_owner: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        markers: LagrangianMarkerDiscretization,
        bodies: PreparedRigidBodySet,
        marker_owner: ArrayLike,
        /,
    ):
        if not isinstance(markers, LagrangianMarkerDiscretization):
            raise TypeError("markers must be LagrangianMarkerDiscretization.")
        if not isinstance(bodies, PreparedRigidBodySet):
            raise TypeError("bodies must be PreparedRigidBodySet.")
        if markers.ambient_dimension != bodies.ambient_dimension:
            raise ValueError("Marker and rigid-body dimensions differ.")
        owner = np.asarray(marker_owner)
        if owner.shape != (markers.capacity,) or not np.issubdtype(
            owner.dtype, np.integer
        ):
            raise TypeError("marker_owner must be an integer marker-capacity vector.")
        active = np.asarray(markers.active_mask)
        if np.any(owner[active] < 0) or np.any(owner[active] >= bodies.capacity):
            raise ValueError("Active marker owners must name valid rigid bodies.")
        body_active = np.asarray(bodies.particles.active_mask)
        if np.any(~body_active[owner[active]]):
            raise ValueError("Active markers cannot belong to inactive rigid bodies.")
        self.markers = markers
        self.bodies = bodies
        self.marker_owner = jnp.asarray(owner, dtype=jnp.int32)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "rigid-marker-map-plan",
                "markers": markers.prepared_id,
                "bodies": bodies.prepared_id,
                "marker_owner": array_tree_fingerprint(owner),
            }
        )

    def prepare(self, /) -> PreparedRigidMarkerMap:
        return PreparedRigidMarkerMap(self)


class PreparedRigidMarkerMap(StrictModule, NonTrainableState):
    plan: RigidMarkerMapPlan
    markers: LagrangianMarkerDiscretization
    bodies: PreparedRigidBodySet
    marker_owner: Array
    active_owner: Array
    mobile_indices: Array
    owner_mobile_slot: Array
    generalized_velocity_space: PyTreeSpace
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: RigidMarkerMapPlan, /):
        if not isinstance(plan, RigidMarkerMapPlan):
            raise TypeError("plan must be RigidMarkerMapPlan.")
        active_body = np.asarray(plan.bodies.particles.active_mask)
        fixed = np.asarray(plan.bodies.fixed_mask)
        mobile = np.flatnonzero(active_body & ~fixed).astype(np.int32)
        owner_slot = np.full((plan.bodies.capacity,), -1, dtype=np.int32)
        owner_slot[mobile] = np.arange(mobile.size, dtype=np.int32)
        dtype = plan.markers.reference_position.dtype
        zero = RigidGeneralizedVelocity(
            jnp.zeros((mobile.size, plan.bodies.ambient_dimension), dtype=dtype),
            jnp.zeros((mobile.size, plan.bodies.angular_dimension), dtype=dtype),
        )
        self.plan = plan
        self.markers = plan.markers
        self.bodies = plan.bodies
        self.marker_owner = plan.marker_owner
        self.active_owner = plan.marker_owner[plan.markers.active_indices]
        self.mobile_indices = jnp.asarray(mobile, dtype=jnp.int32)
        self.owner_mobile_slot = jnp.asarray(owner_slot, dtype=jnp.int32)
        self.generalized_velocity_space = PyTreeSpace(zero)
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-rigid-marker-map",
                "plan": plan.plan_id,
                "mobile_bodies": mobile.tolist(),
            }
        )

    def _world_offset(self, kinematics: RigidBodyKinematics, /) -> Array:
        reference = self.markers.reference_position
        owner = self.marker_owner
        if self.bodies.ambient_dimension == 2:
            angle = kinematics.orientation[owner, 0]
            cosine = jnp.cos(angle)
            sine = jnp.sin(angle)
            return jnp.stack(
                (
                    cosine * reference[:, 0] - sine * reference[:, 1],
                    sine * reference[:, 0] + cosine * reference[:, 1],
                ),
                axis=-1,
            )
        rotation = quaternion_rotation_matrix(kinematics.orientation)
        return contract("...ij,...j->...i", rotation[owner], reference)

    def evaluate(self, kinematics: RigidBodyKinematics, /) -> LagrangianMarkerKinematics:
        if not isinstance(kinematics, RigidBodyKinematics):
            raise TypeError("kinematics must be RigidBodyKinematics.")
        offset = self._world_offset(kinematics)
        owner = self.marker_owner
        position = kinematics.position[owner] + offset
        if self.bodies.ambient_dimension == 2:
            omega = kinematics.angular_velocity[owner, 0]
            spin = jnp.stack((-omega * offset[:, 1], omega * offset[:, 0]), axis=-1)
        else:
            spin = jnp.cross(kinematics.angular_velocity[owner], offset)
        velocity = kinematics.velocity[owner] + spin
        return self.markers.kinematics(position, velocity)

    def velocity_operator(
        self, kinematics: RigidBodyKinematics, /
    ) -> FunctionLinearOperator:
        offset = self._world_offset(kinematics)[self.markers.active_indices]
        slots = self.owner_mobile_slot[self.active_owner]
        mobile_marker = slots >= 0
        safe_slots = jnp.maximum(slots, 0)

        def action(value: RigidGeneralizedVelocity):
            value = self.generalized_velocity_space.validate(value)
            translation = value.translation[safe_slots]
            angular = value.rotation[safe_slots]
            if self.bodies.ambient_dimension == 2:
                omega = angular[:, 0]
                spin = jnp.stack((-omega * offset[:, 1], omega * offset[:, 0]), axis=-1)
            else:
                spin = jnp.cross(angular, offset)
            return jnp.where(mobile_marker[:, None], translation + spin, 0.0)

        return FunctionLinearOperator(
            action,
            source=self.generalized_velocity_space,
            target=self.markers.active_velocity_space,
            operator_id=f"rigid-marker-velocity/{self.prepared_id}",
        )

    def generalized_mass_operator(
        self,
        kinematics: RigidBodyKinematics,
        inverse_momentum_coefficient: ArrayLike,
        /,
    ) -> FunctionLinearOperator:
        coefficient = jnp.asarray(
            inverse_momentum_coefficient,
            dtype=self.markers.reference_position.dtype,
        ).reshape(())
        inverse = 1.0 / coefficient
        masses = self.bodies.particles.safe_masses[self.mobile_indices]
        if self.bodies.ambient_dimension == 2:
            mobile_inertia = self.bodies.inertia_body[self.mobile_indices]

            def angular_action(rotation):
                return mobile_inertia[:, None] * rotation

        else:
            inertia, _ = rigid_body_world_inertia(self.bodies, kinematics.orientation)
            mobile_inertia = inertia[self.mobile_indices]

            def angular_action(rotation):
                return contract(
                    "...ij,...j->...i",
                    mobile_inertia,
                    rotation,
                )

        def action(value: RigidGeneralizedVelocity):
            value = self.generalized_velocity_space.validate(value)
            return RigidGeneralizedVelocity(
                inverse * masses[:, None] * value.translation,
                inverse * angular_action(value.rotation),
            )

        return FunctionLinearOperator(
            action,
            source=self.generalized_velocity_space,
            target=self.generalized_velocity_space,
            transpose_action=action,
            properties=OperatorProperties(
                self_adjoint=True,
                positive_definite=True,
                block_diagonal=True,
                evidence={
                    "self_adjoint": "construction",
                    "positive_definite": "construction",
                    "block_diagonal": "construction",
                },
            ),
            operator_id=f"rigid-marker-mass/{self.prepared_id}",
        )

    def generalized_velocity(
        self, kinematics: RigidBodyKinematics, /
    ) -> RigidGeneralizedVelocity:
        return RigidGeneralizedVelocity(
            kinematics.velocity[self.mobile_indices],
            kinematics.angular_velocity[self.mobile_indices],
        )

    def with_generalized_velocity(
        self,
        kinematics: RigidBodyKinematics,
        value: RigidGeneralizedVelocity,
        /,
    ) -> RigidBodyKinematics:
        value = self.generalized_velocity_space.validate(value)
        velocity = kinematics.velocity.at[self.mobile_indices].set(value.translation)
        angular = kinematics.angular_velocity.at[self.mobile_indices].set(value.rotation)
        return RigidBodyKinematics(
            kinematics.position,
            velocity,
            kinematics.orientation,
            angular,
        )


__all__ = [
    "PreparedRigidMarkerMap",
    "RigidGeneralizedVelocity",
    "RigidMarkerMapPlan",
]
