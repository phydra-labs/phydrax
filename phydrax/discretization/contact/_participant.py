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
from jaxtyping import Array, ArrayLike, PyTree

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import AbstractVectorSpace
from ._guarantee import ContactCapability, ContactGuaranteeLevel
from ._surface import CollisionSurfacePlan, PreparedCollisionSurface


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
    """Kinematic map from one mechanics state to collision vertices."""

    @property
    @abc.abstractmethod
    def source_space(self) -> AbstractVectorSpace:
        raise NotImplementedError

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
    def force_pullback(
        self, state: PyTree[Any], surface_force: ArrayLike, /
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
        surface_force: ArrayLike,
        /,
    ) -> ParticipantDualityEvidence:
        state_ = self.source_space.validate(state)
        direction_ = self.source_space.validate(direction)
        force = jnp.asarray(surface_force)
        _, position_direction = jax.jvp(self.positions, (state_,), (direction_,))
        pulled = self.force_pullback(state_, force)
        position_pairing = jnp.sum(position_direction * force)
        state_pairing = self.source_space.inner(direction_, pulled)
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
            | ContactCapability.FORCE_PULLBACK
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
        return self.surface.map_values(self.source_space.validate(rates))

    def force_pullback(
        self, state: PyTree[Any], surface_force: ArrayLike, /
    ) -> PyTree[Array]:
        self.source_space.validate(state)
        return self.surface.pullback(surface_force)

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
    """Explicit nonlinear participant with JAX-compatible actions."""

    plan: CollisionSurfacePlan
    space: AbstractVectorSpace
    position_action: Callable[[PyTree[Any]], Array] = eqx.field(static=True)
    velocity_action: Callable[[PyTree[Any], PyTree[Any]], Array] | None = eqx.field(
        static=True
    )
    pullback_action: Callable[[PyTree[Any], Array], PyTree[Array]] | None = eqx.field(
        static=True
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
        velocity_action: (Callable[[PyTree[Any], PyTree[Any]], Array] | None) = None,
        pullback_action: (Callable[[PyTree[Any], Array], PyTree[Array]] | None) = None,
        bounds_action: (
            Callable[[PyTree[Any], PyTree[Any]], tuple[Array, Array]] | None
        ) = None,
        participant_id: str | None = None,
    ):
        if not isinstance(plan, CollisionSurfacePlan):
            raise TypeError("plan must be CollisionSurfacePlan.")
        if not isinstance(source_space, AbstractVectorSpace):
            raise TypeError("source_space must be AbstractVectorSpace.")
        if not callable(position_action):
            raise TypeError("position_action must be callable.")
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "function-contact-participant",
                    "surface": plan.topology_id,
                    "space": source_space.space_id,
                    "position_action": position_action,
                    "velocity_action": velocity_action,
                    "pullback_action": pullback_action,
                    "bounds_action": bounds_action,
                }
            )
            if participant_id is None
            else str(participant_id)
        )
        if not identifier:
            raise ValueError("participant_id must be nonempty or None.")
        capabilities = (
            ContactCapability.STATIC_DISTANCE
            | ContactCapability.DIFFERENTIABLE_KINEMATICS
            | ContactCapability.FORCE_PULLBACK
        )
        if bounds_action is not None:
            capabilities |= ContactCapability.NONLINEAR_TRAJECTORY
        self.plan = plan
        self.space = source_space
        self.position_action = position_action
        self.velocity_action = velocity_action
        self.pullback_action = pullback_action
        self.bounds_action = bounds_action
        self._capabilities = capabilities
        self._participant_id = identifier

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

    def positions(self, state: PyTree[Any], /) -> Array:
        state_ = self.source_space.validate(state)
        value = jnp.asarray(self.position_action(state_))
        expected = (
            self.surface_plan.vertex_count,
            self.surface_plan.ambient_dimension,
        )
        if value.shape != expected:
            raise ValueError(f"Nonlinear contact positions must have shape {expected}.")
        return value

    def velocities(self, state: PyTree[Any], rates: PyTree[Any], /) -> Array:
        state_ = self.source_space.validate(state)
        rates_ = self.source_space.validate(rates)
        if self.velocity_action is not None:
            return jnp.asarray(self.velocity_action(state_, rates_))
        return jax.jvp(self.positions, (state_,), (rates_,))[1]

    def force_pullback(
        self, state: PyTree[Any], surface_force: ArrayLike, /
    ) -> PyTree[Array]:
        state_ = self.source_space.validate(state)
        force = jnp.asarray(surface_force)
        if self.pullback_action is not None:
            return self.source_space.validate(self.pullback_action(state_, force))
        _, pullback = jax.vjp(self.positions, state_)
        return pullback(force)[0]

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
            lower = jnp.asarray(lower)
            upper = jnp.asarray(upper, dtype=lower.dtype)
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
        if any(value.surface_plan.ambient_dimension != dimension for value in values[1:]):
            raise ValueError("Contact participants must share one ambient dimension.")
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

    @property
    def vertex_body_ids(self) -> Array:
        return jnp.concatenate(
            tuple(
                participant.surface_plan.pair_policy.body_ids
                for participant in self.participants
            )
        )

    @property
    def vertex_material_ids(self) -> Array:
        return jnp.concatenate(
            tuple(
                participant.surface_plan.pair_policy.material_ids
                for participant in self.participants
            )
        )

    @property
    def vertex_patch_ids(self) -> Array:
        return jnp.concatenate(
            tuple(
                participant.surface_plan.pair_policy.patch_ids
                for participant in self.participants
            )
        )

    @property
    def vertex_static_mask(self) -> Array:
        return jnp.concatenate(
            tuple(
                participant.surface_plan.pair_policy.static_mask
                for participant in self.participants
            )
        )

    @property
    def minimum_separation(self) -> Array:
        return jnp.concatenate(
            tuple(
                participant.surface_plan.vertex_minimum_separation
                for participant in self.participants
            )
        )

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

    def force_pullback(
        self,
        states: Sequence[PyTree[Any]],
        scene_force: ArrayLike,
        /,
    ) -> tuple[PyTree[Array], ...]:
        states_ = tuple(states)
        force = jnp.asarray(scene_force)
        expected = (self.vertex_count, self.ambient_dimension)
        if len(states_) != len(self.participants) or force.shape != expected:
            raise ValueError("Participant scene force/state shape is invalid.")
        return tuple(
            participant.force_pullback(
                state,
                force[self.vertex_offsets[index] : self.vertex_offsets[index + 1]],
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
