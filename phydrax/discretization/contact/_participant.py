#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from collections.abc import Callable, Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, PyTree

from ..._fingerprint import canonical_fingerprint
from ..._identity import callable_payload
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import AbstractVectorSpace, ArraySpace, DualSpace
from ._guarantee import ContactCapability, ContactGuaranteeLevel
from ._surface import (
    CollisionSurfacePlan,
    ContactPairPolicy,
    PreparedCollisionSurface,
)


class ParticipantTrajectoryBounds(StrictModule):
    lower: Array
    upper: Array
    guarantee_level: Array
    finite: Array
    successful: Array
    participant_id: str = eqx.field(static=True)


class ParticipantDualityEvidence(StrictModule):
    position_pairing: Array
    state_pairing: Array
    residual: Array
    scale: Array
    finite: Array
    valid: Array
    participant_id: str = eqx.field(static=True)


class AbstractContactParticipant(StrictModule, NonTrainableState):
    """Kinematic map with explicit configuration, tangent, and dual roles."""

    @property
    @abc.abstractmethod
    def source_space(self) -> AbstractVectorSpace:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def tangent_space(self) -> AbstractVectorSpace:
        raise NotImplementedError

    @property
    def effort_space(self) -> DualSpace:
        return DualSpace(self.tangent_space)

    @property
    @abc.abstractmethod
    def contact_velocity_space(self) -> ArraySpace:
        raise NotImplementedError

    @property
    def contact_effort_space(self) -> DualSpace:
        return DualSpace(self.contact_velocity_space)

    @property
    @abc.abstractmethod
    def surface_plan(self) -> CollisionSurfacePlan:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def participant_id(self) -> str:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def capabilities(self) -> ContactCapability:
        raise NotImplementedError

    @abc.abstractmethod
    def positions(self, state: PyTree[Any], /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def velocities(self, state: PyTree[Any], rates: PyTree[Any], /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def effort_pullback(
        self, state: PyTree[Any], surface_effort: ArrayLike, /
    ) -> PyTree[Array]:
        raise NotImplementedError

    @abc.abstractmethod
    def trajectory_bounds(
        self, start_state: PyTree[Any], end_state: PyTree[Any], /
    ) -> ParticipantTrajectoryBounds:
        raise NotImplementedError

    def duality_evidence(
        self,
        state: PyTree[Any],
        direction: PyTree[Any],
        surface_effort: ArrayLike,
        /,
    ) -> ParticipantDualityEvidence:
        state_ = self.source_space.validate(state)
        direction_ = self.tangent_space.validate(direction)
        position_direction = self.contact_velocity_space.validate(
            self.velocities(state_, direction_)
        )
        effort = self.contact_effort_space.validate(surface_effort)
        pulled = self.effort_space.validate(self.effort_pullback(state_, effort))
        position_pairing = self.contact_effort_space.pair(effort, position_direction)
        state_pairing = self.effort_space.pair(pulled, direction_)
        residual = position_pairing - state_pairing
        scale = jnp.maximum(
            1.0,
            jnp.maximum(jnp.abs(position_pairing), jnp.abs(state_pairing)),
        )
        tolerance = jnp.finfo(position_direction.dtype).eps * max(
            64,
            8 * self.surface_plan.vertex_count * self.surface_plan.ambient_dimension,
        )
        finite = jnp.all(
            jnp.isfinite(jnp.stack((position_pairing, state_pairing, residual, scale)))
        )
        return ParticipantDualityEvidence(
            position_pairing,
            state_pairing,
            residual,
            scale,
            finite,
            finite & (jnp.abs(residual) <= tolerance * scale),
            self.participant_id,
        )


class LinearContactParticipant(AbstractContactParticipant):
    surface: PreparedCollisionSurface
    _capabilities: ContactCapability = eqx.field(static=True)
    _participant_id: str = eqx.field(static=True)

    def __init__(self, surface: PreparedCollisionSurface, /):
        if not isinstance(surface, PreparedCollisionSurface):
            raise TypeError("surface must be PreparedCollisionSurface.")
        self.surface = surface
        self._capabilities = (
            ContactCapability.STATIC_DISTANCE
            | ContactCapability.LINEAR_TRAJECTORY
            | ContactCapability.DIFFERENTIABLE_KINEMATICS
            | ContactCapability.EFFORT_PULLBACK
        )
        self._participant_id = canonical_fingerprint(
            {
                "kind": "linear-contact-participant",
                "surface": surface.prepared_id,
            }
        )

    @property
    def source_space(self) -> AbstractVectorSpace:
        return self.surface.source_space

    @property
    def tangent_space(self) -> AbstractVectorSpace:
        return self.surface.tangent_space

    @property
    def contact_velocity_space(self) -> ArraySpace:
        return self.surface.contact_velocity_space

    @property
    def surface_plan(self) -> CollisionSurfacePlan:
        return self.surface.plan

    @property
    def participant_id(self) -> str:
        return self._participant_id

    @property
    def capabilities(self) -> ContactCapability:
        return self._capabilities

    def positions(self, state: PyTree[Any], /) -> Array:
        return self.surface.positions(state)

    def velocities(self, state: PyTree[Any], rates: PyTree[Any], /) -> Array:
        self.source_space.validate(state)
        return self.contact_velocity_space.validate(
            self.surface.map_values(self.tangent_space.validate(rates))
        )

    def effort_pullback(
        self, state: PyTree[Any], surface_effort: ArrayLike, /
    ) -> PyTree[Array]:
        self.source_space.validate(state)
        return self.effort_space.validate(self.surface.effort_pullback(surface_effort))

    def trajectory_bounds(
        self, start_state: PyTree[Any], end_state: PyTree[Any], /
    ) -> ParticipantTrajectoryBounds:
        start = self.positions(start_state)
        end = self.positions(end_state)
        lower = jnp.minimum(start, end)
        upper = jnp.maximum(start, end)
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


class FunctionContactParticipant(AbstractContactParticipant):
    """Explicit nonlinear participant with algebraic tangent/effort actions."""

    plan: CollisionSurfacePlan
    configuration: AbstractVectorSpace
    tangent: AbstractVectorSpace
    contact_space: ArraySpace
    position_action: Callable[[PyTree[Any]], Array] = eqx.field(static=True)
    velocity_action: Callable[[PyTree[Any], PyTree[Any]], Array] | None = eqx.field(
        static=True
    )
    effort_pullback_action: Callable[[PyTree[Any], Array], PyTree[Array]] | None = (
        eqx.field(static=True)
    )
    bounds_action: Callable[[PyTree[Any], PyTree[Any]], tuple[Array, Array]] | None = (
        eqx.field(static=True)
    )
    _capabilities: ContactCapability = eqx.field(static=True)
    _participant_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: CollisionSurfacePlan,
        source_space: AbstractVectorSpace,
        position_action: Callable[[PyTree[Any]], Array],
        /,
        *,
        tangent_space: AbstractVectorSpace,
        velocity_action: (Callable[[PyTree[Any], PyTree[Any]], Array] | None) = None,
        effort_pullback_action: (
            Callable[[PyTree[Any], Array], PyTree[Array]] | None
        ) = None,
        bounds_action: (
            Callable[[PyTree[Any], PyTree[Any]], tuple[Array, Array]] | None
        ) = None,
        participant_id: str | None = None,
    ):
        if not isinstance(plan, CollisionSurfacePlan):
            raise TypeError("plan must be CollisionSurfacePlan.")
        if not isinstance(source_space, AbstractVectorSpace):
            raise TypeError("source_space must be AbstractVectorSpace.")
        if not isinstance(tangent_space, AbstractVectorSpace):
            raise TypeError("tangent_space must be AbstractVectorSpace.")
        if not callable(position_action):
            raise TypeError("position_action must be callable.")
        if velocity_action is not None and not callable(velocity_action):
            raise TypeError("velocity_action must be callable or None.")
        if effort_pullback_action is not None and not callable(effort_pullback_action):
            raise TypeError("effort_pullback_action must be callable or None.")
        if bounds_action is not None and not callable(bounds_action):
            raise TypeError("bounds_action must be callable or None.")
        if not tangent_space.compatible(source_space) and (
            velocity_action is None or effort_pullback_action is None
        ):
            raise ValueError(
                "Distinct configuration and tangent spaces require explicit "
                "velocity_action and effort_pullback_action."
            )
        position_structure = jax.eval_shape(position_action, source_space.structure())
        expected_shape = (plan.vertex_count, plan.ambient_dimension)
        if (
            not isinstance(position_structure, jax.ShapeDtypeStruct)
            or position_structure.shape != expected_shape
        ):
            raise ValueError(
                f"position_action must produce one array of shape {expected_shape}."
            )
        contact_space = ArraySpace(expected_shape, dtype=position_structure.dtype)
        if participant_id is None:
            identifier = canonical_fingerprint(
                {
                    "kind": "function-contact-participant",
                    "surface": plan.topology_id,
                    "source_space": source_space.space_id,
                    "tangent_space": tangent_space.space_id,
                    "effort_space": DualSpace(tangent_space).space_id,
                    "contact_velocity_space": contact_space.space_id,
                    "contact_effort_space": DualSpace(contact_space).space_id,
                    "position_action": callable_payload(position_action),
                    "velocity_action": (
                        None
                        if velocity_action is None
                        else callable_payload(velocity_action)
                    ),
                    "effort_pullback_action": (
                        None
                        if effort_pullback_action is None
                        else callable_payload(effort_pullback_action)
                    ),
                    "bounds_action": (
                        None if bounds_action is None else callable_payload(bounds_action)
                    ),
                }
            )
        else:
            identifier = str(participant_id)
        if not identifier:
            raise ValueError("participant_id must be nonempty or None.")
        capabilities = (
            ContactCapability.STATIC_DISTANCE
            | ContactCapability.DIFFERENTIABLE_KINEMATICS
            | ContactCapability.EFFORT_PULLBACK
        )
        if bounds_action is not None:
            capabilities |= ContactCapability.NONLINEAR_TRAJECTORY
        self.plan = plan
        self.configuration = source_space
        self.tangent = tangent_space
        self.contact_space = contact_space
        self.position_action = position_action
        self.velocity_action = velocity_action
        self.effort_pullback_action = effort_pullback_action
        self.bounds_action = bounds_action
        self._capabilities = capabilities
        self._participant_id = identifier

    @property
    def source_space(self) -> AbstractVectorSpace:
        return self.configuration

    @property
    def tangent_space(self) -> AbstractVectorSpace:
        return self.tangent

    @property
    def contact_velocity_space(self) -> ArraySpace:
        return self.contact_space

    @property
    def surface_plan(self) -> CollisionSurfacePlan:
        return self.plan

    @property
    def participant_id(self) -> str:
        return self._participant_id

    @property
    def capabilities(self) -> ContactCapability:
        return self._capabilities

    def positions(self, state: PyTree[Any], /) -> Array:
        state_ = self.source_space.validate(state)
        return self.contact_velocity_space.validate(self.position_action(state_))

    def velocities(self, state: PyTree[Any], rates: PyTree[Any], /) -> Array:
        state_ = self.source_space.validate(state)
        rates_ = self.tangent_space.validate(rates)
        if self.velocity_action is not None:
            value = self.velocity_action(state_, rates_)
        else:
            value = jax.jvp(self.positions, (state_,), (rates_,))[1]
        return self.contact_velocity_space.validate(value)

    def effort_pullback(
        self, state: PyTree[Any], surface_effort: ArrayLike, /
    ) -> PyTree[Array]:
        state_ = self.source_space.validate(state)
        effort = self.contact_effort_space.validate(
            jnp.asarray(surface_effort, dtype=self.contact_velocity_space.dtype)
        )
        if self.effort_pullback_action is not None:
            pulled = self.effort_pullback_action(state_, effort)
        else:
            _, pullback = jax.vjp(self.positions, state_)
            pulled = pullback(effort)[0]
        return self.effort_space.validate(pulled)

    def trajectory_bounds(
        self, start_state: PyTree[Any], end_state: PyTree[Any], /
    ) -> ParticipantTrajectoryBounds:
        start_ = self.source_space.validate(start_state)
        end_ = self.source_space.validate(end_state)
        if self.bounds_action is None:
            start = self.positions(start_)
            end = self.positions(end_)
            lower = jnp.minimum(start, end)
            upper = jnp.maximum(start, end)
            level = ContactGuaranteeLevel.HEURISTIC
        else:
            lower, upper = self.bounds_action(start_, end_)
            lower = self.contact_velocity_space.validate(lower)
            upper = self.contact_velocity_space.validate(upper)
            level = ContactGuaranteeLevel.ANALYTIC_CONSERVATIVE
        finite = (
            jnp.all(jnp.isfinite(lower))
            & jnp.all(jnp.isfinite(upper))
            & jnp.all(lower <= upper)
        )
        return ParticipantTrajectoryBounds(
            lower,
            upper,
            jnp.asarray(int(level), dtype=jnp.int32),
            finite,
            finite,
            self.participant_id,
        )


class ContactParticipantScene(StrictModule, NonTrainableState):
    """Several independently parameterized contact participants."""

    participants: tuple[AbstractContactParticipant, ...]
    vertex_offsets: tuple[int, ...] = eqx.field(static=True)
    edge_offsets: tuple[int, ...] = eqx.field(static=True)
    face_offsets: tuple[int, ...] = eqx.field(static=True)
    scene_id: str = eqx.field(static=True)

    def __init__(self, participants: Sequence[AbstractContactParticipant], /):
        values = tuple(participants)
        if not values or not all(
            isinstance(value, AbstractContactParticipant) for value in values
        ):
            raise TypeError("participants must contain contact participant values.")
        dimension = values[0].surface_plan.ambient_dimension
        reference_pair_policy = values[0].surface_plan.pair_policy
        for value in values[1:]:
            if value.surface_plan.ambient_dimension != dimension:
                raise ValueError("Contact participants must share one ambient dimension.")
            candidate = value.surface_plan.pair_policy
            if (
                candidate.unrestricted != reference_pair_policy.unrestricted
                or not np.array_equal(
                    np.asarray(candidate.allowed_participant_pairs),
                    np.asarray(reference_pair_policy.allowed_participant_pairs),
                )
            ):
                raise ValueError(
                    "Contact participant-pair policies must agree reciprocally."
                )
        for feature_slice, kind_name in (
            ("vertex_slice", "vertex"),
            ("edge_slice", "edge"),
            ("face_slice", "face"),
        ):
            identifiers = np.concatenate(
                tuple(
                    np.asarray(participant.surface_plan.feature_policy.feature_ids)[
                        getattr(
                            participant.surface_plan.feature_policy,
                            feature_slice,
                        )
                    ]
                    for participant in values
                )
            )
            if np.unique(identifiers).size != identifiers.size:
                raise ValueError(
                    "Contact participant scene "
                    f"{kind_name} feature IDs must be globally unique."
                )
        vertex_offsets = [0]
        edge_offsets = [0]
        face_offsets = [0]
        for value in values:
            vertex_offsets.append(vertex_offsets[-1] + value.surface_plan.vertex_count)
            edge_offsets.append(edge_offsets[-1] + value.surface_plan.edge_count)
            face_offsets.append(face_offsets[-1] + value.surface_plan.face_count)
        self.participants = values
        self.vertex_offsets = tuple(vertex_offsets)
        self.edge_offsets = tuple(edge_offsets)
        self.face_offsets = tuple(face_offsets)
        self.scene_id = canonical_fingerprint(
            {
                "kind": "contact-participant-scene",
                "participants": [value.participant_id for value in values],
            }
        )

    @property
    def ambient_dimension(self) -> int:
        return self.participants[0].surface_plan.ambient_dimension

    @property
    def vertex_count(self) -> int:
        return self.vertex_offsets[-1]

    @property
    def edge_count(self) -> int:
        return self.edge_offsets[-1]

    @property
    def face_count(self) -> int:
        return self.face_offsets[-1]

    @property
    def edges(self) -> Array:
        return jnp.concatenate(
            tuple(
                participant.surface_plan.edges + self.vertex_offsets[index]
                for index, participant in enumerate(self.participants)
            ),
            axis=0,
        )

    @property
    def faces(self) -> Array:
        values = tuple(
            participant.surface_plan.faces + self.vertex_offsets[index]
            for index, participant in enumerate(self.participants)
            if participant.surface_plan.face_count
        )
        if not values:
            return jnp.empty((0, 3), dtype=jnp.int32)
        return jnp.concatenate(values, axis=0)

    def _feature_values(self, name: str, feature_slice: str, /) -> Array:
        return jnp.concatenate(
            tuple(
                getattr(participant.surface_plan.feature_policy, name)[
                    getattr(participant.surface_plan.feature_policy, feature_slice)
                ]
                for participant in self.participants
            )
        )

    def _all_feature_values(self, name: str, /) -> Array:
        return jnp.concatenate(
            (
                self._feature_values(name, "vertex_slice"),
                self._feature_values(name, "edge_slice"),
                self._feature_values(name, "face_slice"),
            )
        )

    @property
    def vertex_feature_ids(self) -> Array:
        return self._feature_values("feature_ids", "vertex_slice")

    @property
    def edge_feature_ids(self) -> Array:
        return self._feature_values("feature_ids", "edge_slice")

    @property
    def face_feature_ids(self) -> Array:
        return self._feature_values("feature_ids", "face_slice")

    @property
    def feature_ids(self) -> Array:
        return self._all_feature_values("feature_ids")

    @property
    def vertex_participant_ids(self) -> Array:
        return self._feature_values("participant_ids", "vertex_slice")

    @property
    def feature_participant_ids(self) -> Array:
        return self._all_feature_values("participant_ids")

    @property
    def vertex_body_ids(self) -> Array:
        return self._feature_values("body_ids", "vertex_slice")

    @property
    def feature_body_ids(self) -> Array:
        return self._all_feature_values("body_ids")

    @property
    def vertex_material_ids(self) -> Array:
        return self._feature_values("material_ids", "vertex_slice")

    @property
    def feature_material_ids(self) -> Array:
        return self._all_feature_values("material_ids")

    @property
    def vertex_patch_ids(self) -> Array:
        return self._feature_values("patch_ids", "vertex_slice")

    @property
    def feature_patch_ids(self) -> Array:
        return self._all_feature_values("patch_ids")

    @property
    def vertex_static_mask(self) -> Array:
        return self._feature_values("static_mask", "vertex_slice")

    @property
    def feature_static_mask(self) -> Array:
        return self._all_feature_values("static_mask")

    @property
    def feature_physical_radius(self) -> Array:
        return self._all_feature_values("physical_radius")

    @property
    def feature_solver_clearance(self) -> Array:
        return self._all_feature_values("solver_clearance")

    @property
    def feature_proxy_error(self) -> Array:
        return self._all_feature_values("proxy_error")

    @property
    def feature_contact_extent(self) -> Array:
        return (
            self.feature_physical_radius
            + self.feature_solver_clearance
            + self.feature_proxy_error
        )

    @property
    def pair_policy(self) -> ContactPairPolicy:
        return self.participants[0].surface_plan.pair_policy

    def positions(self, states: Sequence[PyTree[Any]], /) -> Array:
        values = tuple(states)
        if len(values) != len(self.participants):
            raise ValueError("Participant state count does not match scene.")
        return jnp.concatenate(
            tuple(
                participant.positions(state)
                for participant, state in zip(self.participants, values, strict=True)
            ),
            axis=0,
        )

    def velocities(
        self,
        states: Sequence[PyTree[Any]],
        rates: Sequence[PyTree[Any]],
        /,
    ) -> Array:
        states_ = tuple(states)
        rates_ = tuple(rates)
        if len(states_) != len(self.participants) or len(rates_) != len(
            self.participants
        ):
            raise ValueError("Participant state/rate count does not match scene.")
        return jnp.concatenate(
            tuple(
                participant.velocities(state, rate)
                for participant, state, rate in zip(
                    self.participants, states_, rates_, strict=True
                )
            ),
            axis=0,
        )

    def effort_pullback(
        self,
        states: Sequence[PyTree[Any]],
        scene_effort: ArrayLike,
        /,
    ) -> tuple[PyTree[Array], ...]:
        states_ = tuple(states)
        effort = jnp.asarray(scene_effort)
        expected = (self.vertex_count, self.ambient_dimension)
        if len(states_) != len(self.participants) or effort.shape != expected:
            raise ValueError("Participant scene effort/state shape is invalid.")
        return tuple(
            participant.effort_pullback(
                state,
                effort[self.vertex_offsets[index] : self.vertex_offsets[index + 1]],
            )
            for index, (participant, state) in enumerate(
                zip(self.participants, states_, strict=True)
            )
        )


__all__ = [
    "AbstractContactParticipant",
    "ContactParticipantScene",
    "FunctionContactParticipant",
    "LinearContactParticipant",
    "ParticipantDualityEvidence",
    "ParticipantTrajectoryBounds",
]
