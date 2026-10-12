#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

from collections.abc import Sequence
from enum import IntFlag
from math import isfinite
from typing import Any, assert_never, final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike
from jaxtyping import PyTree

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._tree_math import tree_add_scaled, tree_allfinite, tree_where
from ...discretization.contact._ccd import (
    CertifiedAABBCCDPlan,
    collision_free_step_limit,
    ContactSafetyEvidence,
    InclusionCCDPlan,
)
from ...discretization.contact._guarantee import ContactCapability, ContactGuaranteeLevel
from ...discretization.contact._participant import ContactParticipantScene
from ...discretization.contact._search import (
    ContactCandidateEpoch,
    DenseContactSearchPlan,
    SweepAndPruneContactSearchPlan,
)
from ...linalg import BlockSpace, DualSpace, GMRES
from ...nonlinear._newton import NewtonKrylov
from ...nonlinear._types import (
    NonlinearDiagnostics,
    NonlinearResult,
    NonlinearSystemProblem,
    NonlinearTermination,
)
from ...typing import checked, parse
from ..solid_mechanics._fem_dynamics import FiniteElementDynamicsState
from ._friction import (
    ContactFrictionEvaluation,
    ContactFrictionState,
    PreparedLaggedCoulombFriction,
)
from ._implicit_participant import (
    AbstractImplicitContactParticipant,
    ContactKinematicLinearization,
    ContactMeritMode,
    ContactParticipantEvaluation,
    ContactParticipantStepLimit,
    linearize_contact_kinematics,
    participant_residual,
)
from ._potential import ContactPotentialEvaluation, PreparedConvergentContactPotential


ContactCCDPlan = InclusionCCDPlan | CertifiedAABBCCDPlan
type ContactCoordinates = tuple[PyTree[Array], ...]


class ContactRejectionReason(IntFlag):
    NONE = 0
    SEARCH = 1 << 0
    CCD = 1 << 1
    PARTICIPANT = 1 << 2
    NONLINEAR = 1 << 3
    CONTACT = 1 << 4
    NONFINITE = 1 << 5
    FRICTION = 1 << 6
    ENVELOPE = 1 << 7
    POTENTIAL = 1 << 8


@final
class ContactSolvePolicy(StrictModule, NonTrainableState):
    __strict_contract__ = True
    merit: ContactMeritMode = eqx.field(static=True)
    method: NewtonKrylov
    termination: NonlinearTermination
    maximum_lag_iterations: int = eqx.field(static=True)
    lag_tolerance: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        merit: ContactMeritMode = "automatic",
        method: NewtonKrylov | None = None,
        termination: NonlinearTermination | None = None,
        maximum_lag_iterations: int = 4,
        lag_tolerance: float = 1e-8,
    ) -> None:
        merit_ = parse(merit, ContactMeritMode, "merit")
        method_ = NewtonKrylov() if method is None else method
        termination_ = NonlinearTermination() if termination is None else termination
        if not isinstance(method_, NewtonKrylov):
            raise TypeError("method must be NewtonKrylov.")
        if not isinstance(method_.linear_policy.method, GMRES):
            raise ValueError("Monolithic contact requires nonsymmetric GMRES.")
        if not isinstance(termination_, NonlinearTermination):
            raise TypeError("termination must be NonlinearTermination.")
        if maximum_lag_iterations < 1:
            raise ValueError("maximum_lag_iterations must be positive.")
        if not isfinite(lag_tolerance) or lag_tolerance <= 0:
            raise ValueError("lag_tolerance must be finite and positive.")
        self.merit = merit_
        self.method = method_
        self.termination = termination_
        self.maximum_lag_iterations = maximum_lag_iterations
        self.lag_tolerance = lag_tolerance


@final
class MonolithicContactPlan(StrictModule, NonTrainableState):
    participants: tuple[AbstractImplicitContactParticipant, ...]
    contact_scene: ContactParticipantScene
    contact: PreparedConvergentContactPotential
    search: DenseContactSearchPlan | SweepAndPruneContactSearchPlan
    ccd: ContactCCDPlan
    geometry_owner_indices: tuple[int | None, ...] = eqx.field(static=True)
    fixed_geometry_states: tuple[PyTree[Array] | None, ...]
    friction: PreparedLaggedCoulombFriction | None
    policy: ContactSolvePolicy
    dof_space: BlockSpace
    linear_trajectory: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        participants: Sequence[AbstractImplicitContactParticipant],
        contact_scene: ContactParticipantScene,
        contact: PreparedConvergentContactPotential,
        search: DenseContactSearchPlan | SweepAndPruneContactSearchPlan,
        ccd: ContactCCDPlan,
        /,
        *,
        geometry_owner_indices: Sequence[int | None],
        fixed_geometry_states: Sequence[PyTree[Array] | None],
        friction: PreparedLaggedCoulombFriction | None = None,
        policy: ContactSolvePolicy | None = None,
    ) -> None:
        members = tuple(participants)
        if not members or not all(
            isinstance(p, AbstractImplicitContactParticipant) for p in members
        ):
            raise TypeError("participants must contain mechanics participants.")
        ids = tuple(p.participant_id for p in members)
        if len(set(ids)) != len(ids):
            raise ValueError("Mechanics participant IDs must be unique.")
        owners = tuple(geometry_owner_indices)
        fixed = tuple(fixed_geometry_states)
        if len(owners) != len(contact_scene.participants) or len(fixed) != len(owners):
            raise ValueError(
                "Each geometry participant requires an explicit owner binding."
            )
        validated_fixed: list[PyTree[Array] | None] = []
        for geometry, owner, value in zip(
            contact_scene.participants, owners, fixed, strict=True
        ):
            if owner is None:
                if value is None:
                    raise ValueError("Fixed geometry requires an explicit configuration.")
                validated_fixed.append(geometry.source_space.validate(value))
            else:
                if owner < 0 or owner >= len(members) or value is not None:
                    raise ValueError("Geometry owner index/fixed state is invalid.")
                if not geometry.source_space.compatible(
                    members[owner].dof_space
                ) or not geometry.tangent_space.compatible(members[owner].dof_space):
                    raise ValueError("Geometry and mechanics DOF spaces must agree.")
                validated_fixed.append(None)
            if owner is not None and not geometry.capabilities & (
                ContactCapability.LINEAR_TRAJECTORY
                | ContactCapability.NONLINEAR_TRAJECTORY
            ):
                raise ValueError(
                    "Nonlinear monolithic geometry requires a conservative full-path provider."
                )
        linear_trajectory = all(
            owner is None
            or bool(geometry.capabilities & ContactCapability.LINEAR_TRAJECTORY)
            for geometry, owner in zip(contact_scene.participants, owners, strict=True)
        )
        if not linear_trajectory and isinstance(ccd, CertifiedAABBCCDPlan):
            raise ValueError(
                "Nonlinear enclosure paths cannot claim rounding-certified CCD without a rounding certificate."
            )
        if contact.scene.scene_id != contact_scene.scene_id:
            raise ValueError("Contact potential must belong to this participant scene.")
        if search.activation_distance != contact.plan.activation_distance:
            raise ValueError("Search and barrier activation distances must agree.")
        if friction is not None and friction.scene.scene_id != contact_scene.scene_id:
            raise ValueError("Friction must belong to this participant scene.")
        policy_ = ContactSolvePolicy() if policy is None else policy
        if policy_.merit == "potential" and not all(p.has_potential for p in members):
            raise ValueError(
                "Potential merit requires every mechanics contribution to have a verified potential."
            )
        space = BlockSpace(tuple(p.dof_space for p in members), names=ids)
        self.participants = members
        self.contact_scene = contact_scene
        self.contact = contact
        self.search = search
        self.ccd = ccd
        self.geometry_owner_indices = owners
        self.fixed_geometry_states = tuple(validated_fixed)
        self.friction = friction
        self.policy = policy_
        self.dof_space = space
        self.linear_trajectory = linear_trajectory
        self.plan_id = canonical_fingerprint(
            {
                "kind": "monolithic-contact",
                "participants": ids,
                "scene": contact_scene.scene_id,
                "contact": contact.prepared_id,
                "search": search.plan_id,
                "ccd": ccd.plan_id,
                "owners": owners,
                "friction": None if friction is None else friction.prepared_id,
                "merit": policy_.merit,
                "method": policy_.method.method_id,
                "termination": repr(policy_.termination),
                "lag": (policy_.maximum_lag_iterations, policy_.lag_tolerance),
                "numeric": array_tree_fingerprint(
                    (members, contact, tuple(validated_fixed))
                ),
                "method_settings": repr(policy_.method),
            }
        )

    def geometry_states(self, coordinates: ContactCoordinates, /) -> ContactCoordinates:
        values: list[PyTree[Array]] = []
        for owner, fixed in zip(
            self.geometry_owner_indices, self.fixed_geometry_states, strict=True
        ):
            if owner is None:
                if fixed is None:
                    raise ValueError("Missing fixed geometry state.")
                values.append(fixed)
            else:
                values.append(coordinates[owner])
        return tuple(values)


@final
class MonolithicContactState(StrictModule, NonTrainableState):
    coordinates: ContactCoordinates
    histories: tuple[PyTree[Array], ...]
    time: Array
    step: Array
    state_version: Array
    friction_state: ContactFrictionState | None
    plan_id: str = eqx.field(static=True)
    state_id: str = eqx.field(static=True)


@checked
def initialize_monolithic_contact_state(
    plan: MonolithicContactPlan,
    coordinates: Sequence[PyTree[Array]],
    histories: Sequence[PyTree[Array]],
    /,
    *,
    time: ArrayLike = 0.0,
) -> MonolithicContactState:
    q = plan.dof_space.validate(tuple(coordinates))
    history = tuple(histories)
    if len(history) != len(q):
        raise ValueError("Each mechanics participant requires a committed history.")
    dtype = jax.tree.leaves(q)[0].dtype
    time_ = jnp.asarray(time, dtype=dtype)
    if time_.shape != () or not np.isfinite(np.asarray(time_)).all():
        raise ValueError("time must be one finite scalar.")
    frame = None
    if plan.friction is not None:
        positions = plan.contact_scene.positions(plan.geometry_states(q))
        epoch = plan.search.build(plan.contact_scene, positions)
        frame = plan.friction.build_state(positions, epoch, state_version=0)
    return MonolithicContactState(
        q,
        history,
        time_,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        frame,
        plan.plan_id,
        canonical_fingerprint({"kind": "monolithic-contact-state", "plan": plan.plan_id}),
    )


def _kinematic_linearizations(
    plan: MonolithicContactPlan, coordinates: ContactCoordinates, /
) -> tuple[ContactKinematicLinearization, ...]:
    return tuple(
        linearize_contact_kinematics(geometry, q)
        for geometry, q in zip(
            plan.contact_scene.participants,
            plan.geometry_states(coordinates),
            strict=True,
        )
    )


@final
class _ContactEvaluation(StrictModule):
    participants: tuple[ContactParticipantEvaluation, ...]
    contact: ContactPotentialEvaluation
    friction: ContactFrictionEvaluation | None
    successful: Array


@final
class _ContactTrajectory(StrictModule):
    plan: MonolithicContactPlan
    coordinates: ContactCoordinates
    direction: ContactCoordinates

    def __call__(
        self, lower_time: Array, upper_time: Array, /
    ) -> tuple[Array, Array, Array]:
        first = jax.tree.map(
            lambda q, dq: q + lower_time.astype(q.dtype) * dq,
            self.coordinates,
            self.direction,
        )
        second = jax.tree.map(
            lambda q, dq: q + upper_time.astype(q.dtype) * dq,
            self.coordinates,
            self.direction,
        )
        first_states = self.plan.geometry_states(first)
        second_states = self.plan.geometry_states(second)
        lower: list[Array] = []
        upper: list[Array] = []
        successful = jnp.asarray(True, dtype=jnp.bool_)
        for geometry, owner, q0, q1 in zip(
            self.plan.contact_scene.participants,
            self.plan.geometry_owner_indices,
            first_states,
            second_states,
            strict=True,
        ):
            if owner is None:
                positions = geometry.positions(q0)
                lower.append(positions)
                upper.append(positions)
            else:
                evidence = geometry.trajectory_bounds(q0, q1)
                lower.append(evidence.lower)
                upper.append(evidence.upper)
                successful = (
                    successful
                    & evidence.finite
                    & evidence.successful
                    & (
                        evidence.guarantee_level
                        >= int(ContactGuaranteeLevel.ENCLOSURE_CONSERVATIVE)
                    )
                )
        return (
            jnp.concatenate(tuple(lower), axis=0),
            jnp.concatenate(tuple(upper), axis=0),
            successful,
        )


def _nonlinear_envelope_limit(
    epoch: ContactCandidateEpoch, trajectory: _ContactTrajectory, maximum: Array, /
) -> Array:
    zero = jnp.zeros_like(maximum)

    def enclosed(rate: Array) -> Array:
        lower, upper, successful = trajectory(zero, rate)
        extent = jnp.maximum(
            jnp.abs(lower - epoch.reference_positions),
            jnp.abs(upper - epoch.reference_positions),
        )
        squared_radius = jnp.sum(extent * extent, axis=-1)
        return (
            successful
            & jnp.all(jnp.isfinite(squared_radius))
            & jnp.all(squared_radius <= epoch.envelope_radius**2)
        )

    def bisect(_: None) -> Array:
        def body(index: int, bracket: tuple[Array, Array]) -> tuple[Array, Array]:
            del index
            lower, upper = bracket
            midpoint = (lower + upper) * 0.5
            valid = enclosed(midpoint)
            return jnp.where(valid, midpoint, lower), jnp.where(valid, upper, midpoint)

        lower, _ = jax.lax.fori_loop(0, 40, body, (zero, maximum))
        return lower

    limit = jax.lax.cond(enclosed(maximum), lambda _: maximum, bisect, None)
    return jnp.where(enclosed(zero), limit, zero)


@final
class _ContactRuntime(StrictModule):
    plan: MonolithicContactPlan
    accepted: MonolithicContactState
    epoch: ContactCandidateEpoch
    step_size: Array | None
    friction_state: ContactFrictionState | None
    potential_merit: bool = eqx.field(static=True)

    def __call__(
        self, coordinates: ContactCoordinates, args: Any, /
    ) -> tuple[ContactCoordinates, _ContactEvaluation]:
        return self.residual(coordinates, args)

    def evaluations(
        self, coordinates: ContactCoordinates, args: Any, /
    ) -> _ContactEvaluation:
        values = tuple(
            p.evaluate(q, h, qn, self.accepted.time, self.step_size, args)
            for p, q, h, qn in zip(
                self.plan.participants,
                coordinates,
                self.accepted.histories,
                self.accepted.coordinates,
                strict=True,
            )
        )
        geometry = self.plan.geometry_states(coordinates)
        positions = self.plan.contact_scene.positions(geometry)
        contact = self.plan.contact.evaluate(
            positions, self.epoch, participant_states=geometry
        )
        friction = None
        if self.plan.friction is not None:
            if self.friction_state is None or self.step_size is None:
                raise ValueError(
                    "Incremental friction requires a frozen frame and step size."
                )
            rates = tuple(
                p.contact_velocity(q, h, qn, self.accepted.time, self.step_size, args)
                for p, q, h, qn in zip(
                    self.plan.participants,
                    coordinates,
                    self.accepted.histories,
                    self.accepted.coordinates,
                    strict=True,
                )
            )
            geometry_rates = tuple(
                geometry_member.tangent_space.zeros() if owner is None else rates[owner]
                for geometry_member, owner in zip(
                    self.plan.contact_scene.participants,
                    self.plan.geometry_owner_indices,
                    strict=True,
                )
            )
            velocity = self.plan.contact_scene.velocities(geometry, geometry_rates)
            friction = self.plan.friction.evaluate(
                velocity, self.friction_state, participant_states=geometry
            )
        successful = contact.successful & self.epoch.successful
        for value in values:
            successful = (
                successful
                & value.successful
                & value.admissible
                & value.tangent_successful
            )
            if self.potential_merit:
                successful = (
                    successful
                    & value.potential_consistency_checked
                    & jnp.isfinite(value.potential_consistency_defect)
                    & (
                        value.potential_consistency_defect
                        <= self.plan.policy.lag_tolerance
                    )
                )
        if friction is not None:
            successful = successful & friction.successful
        return _ContactEvaluation(values, contact, friction, successful)

    def residual(
        self, coordinates: ContactCoordinates, args: Any, /
    ) -> tuple[ContactCoordinates, _ContactEvaluation]:
        evidence = self.evaluations(coordinates, args)
        residual = tuple(
            participant_residual(p, q, h, qn, self.accepted.time, self.step_size, args)
            for p, q, h, qn in zip(
                self.plan.participants,
                coordinates,
                self.accepted.histories,
                self.accepted.coordinates,
                strict=True,
            )
        )
        result = list(residual)
        linearizations = _kinematic_linearizations(self.plan, coordinates)
        for force in (
            evidence.contact.surface_force,
            None if evidence.friction is None else evidence.friction.surface_force,
        ):
            if force is not None:
                for index, (owner, linearization) in enumerate(
                    zip(self.plan.geometry_owner_indices, linearizations, strict=True)
                ):
                    if owner is not None:
                        begin, end = self.plan.contact_scene.vertex_offsets[
                            index : index + 2
                        ]
                        value = linearization.tangent.transpose_mv(force[begin:end])
                        result[owner] = tree_add_scaled(result[owner], value, -1.0)
        return tuple(result), evidence

    def potential(self, coordinates: ContactCoordinates, args: Any, /) -> Array:
        energy = self.plan.contact.energy(
            self.plan.contact_scene.positions(self.plan.geometry_states(coordinates)),
            self.epoch,
        )
        for p, q, h, qn in zip(
            self.plan.participants,
            coordinates,
            self.accepted.histories,
            self.accepted.coordinates,
            strict=True,
        ):
            contribution = p.potential(q, h, qn, self.accepted.time, self.step_size, args)
            if contribution is None:
                raise ValueError(
                    "Potential merit encountered a nonconservative contribution."
                )
            energy = energy + contribution
        if self.plan.friction is not None:
            evidence = self.evaluations(coordinates, args)
            if evidence.friction is None or self.step_size is None:
                raise ValueError("Missing friction merit contribution.")
            scale = self.plan.participants[0].contact_velocity_scale(self.step_size)
            if scale is None:
                raise ValueError("Friction has no conservative incremental potential.")
            energy = energy + evidence.friction.energy / scale
        return energy

    def limits(
        self, coordinates: ContactCoordinates, direction: ContactCoordinates, args: Any, /
    ) -> tuple[ContactSafetyEvidence, tuple[ContactParticipantStepLimit, ...], Array]:
        proposed = tree_add_scaled(coordinates, direction, 1.0)
        start = self.plan.contact_scene.positions(self.plan.geometry_states(coordinates))
        end = self.plan.contact_scene.positions(self.plan.geometry_states(proposed))
        trajectory = (
            None
            if self.plan.linear_trajectory
            else _ContactTrajectory(self.plan, coordinates, direction)
        )
        safety = collision_free_step_limit(
            self.plan.ccd,
            self.plan.contact_scene,
            self.epoch,
            start,
            end,
            trajectory_bounds=trajectory,
        )
        limits = tuple(
            p.step_limit(q, dq, qn, self.accepted.time, self.step_size, args)
            for p, q, dq, qn in zip(
                self.plan.participants,
                coordinates,
                direction,
                self.accepted.coordinates,
                strict=True,
            )
        )
        alpha = jnp.where(safety.successful & safety.conservative, safety.step_size, 0.0)
        for limit in limits:
            valid = (
                limit.successful
                & limit.conservative
                & limit.guarantee.successful
                & jnp.isfinite(limit.maximum_step)
                & (limit.maximum_step >= 0.0)
                & (limit.maximum_step <= 1.0)
            )
            alpha = jnp.minimum(alpha, jnp.where(valid, limit.maximum_step, 0.0))
        if trajectory is not None:
            return (
                safety,
                limits,
                _nonlinear_envelope_limit(self.epoch, trajectory, alpha),
            )
        # Exact exit time from the frozen search envelope for each linear sample.
        offset = start - self.epoch.reference_positions
        movement = end - start
        a = jnp.sum(movement * movement, axis=-1)
        b = jnp.sum(offset * movement, axis=-1)
        c = jnp.sum(offset * offset, axis=-1) - self.epoch.envelope_radius**2
        safe_a = jnp.where(a > 0.0, a, 1.0)
        discriminant = b * b - a * c
        exits = (-b + jnp.sqrt(jnp.maximum(discriminant, 0.0))) / safe_a
        exits = jnp.where(a > 0.0, exits, 1.0)
        envelope_valid = jnp.all(c <= 0.0) & jnp.all(jnp.isfinite(exits))
        envelope_alpha = jnp.where(
            envelope_valid, jnp.minimum(1.0, jnp.min(exits, initial=1.0)), 0.0
        )
        return safety, limits, jnp.minimum(alpha, envelope_alpha)

    def step_limit(
        self, coordinates: ContactCoordinates, direction: ContactCoordinates, args: Any, /
    ) -> Array:
        return self.limits(coordinates, direction, args)[2]

    def problem(self) -> NonlinearSystemProblem:
        identifier = f"{self.plan.plan_id}/{self.epoch.epoch_id}"
        return NonlinearSystemProblem(
            self,
            state_space=self.plan.dof_space,
            residual_space=DualSpace(self.plan.dof_space),
            has_aux=True,
            validity=_contact_validity,
            potential=_ContactPotentialAction(self) if self.potential_merit else None,
            potential_id=f"{identifier}/potential" if self.potential_merit else None,
            step_limit=_ContactStepLimitAction(self),
            step_limit_id=f"{identifier}/safe-step",
            problem_id=identifier,
        )


def _contact_validity(
    coordinates: ContactCoordinates,
    residual: ContactCoordinates,
    evidence: _ContactEvaluation,
    args: Any,
    /,
) -> Array:
    del coordinates, residual, args
    return evidence.successful


@final
class _ContactPotentialAction(StrictModule):
    runtime: _ContactRuntime

    def __call__(self, coordinates: ContactCoordinates, args: Any, /) -> Array:
        return self.runtime.potential(coordinates, args)


@final
class _ContactStepLimitAction(StrictModule):
    runtime: _ContactRuntime

    def __call__(
        self, coordinates: ContactCoordinates, direction: ContactCoordinates, args: Any, /
    ) -> Array:
        return self.runtime.step_limit(coordinates, direction, args)


def _potential_mode(plan: MonolithicContactPlan, step_size: Array | None, /) -> bool:
    compatible = all(p.has_potential for p in plan.participants)
    if plan.friction is not None:
        if step_size is None:
            raise ValueError("Lagged friction is incremental-only.")
        scales = tuple(p.contact_velocity_scale(step_size) for p in plan.participants)
        compatible = (
            compatible
            and plan.linear_trajectory
            and all(scale is not None for scale in scales)
        )
        if compatible:
            host = tuple(np.asarray(scale) for scale in scales)
            compatible = all(
                np.isfinite(value).all() and np.array_equal(value, host[0])
                for value in host
            )
    match plan.policy.merit:
        case "automatic":
            return compatible
        case "potential":
            if not compatible:
                raise ValueError(
                    "Potential merit requires compatible conservative contributions."
                )
            return True
        case "residual":
            return False
        case unreachable:
            assert_never(unreachable)


@final
class PreparedMonolithicContactStep(StrictModule, NonTrainableState):
    plan: MonolithicContactPlan
    accepted: MonolithicContactState
    step_size: Array
    args: Any
    candidate_epoch: ContactCandidateEpoch
    runtime: _ContactRuntime
    prepared_id: str = eqx.field(static=True)

    def kinematic_linearizations(
        self, coordinates: ContactCoordinates, /
    ) -> tuple[ContactKinematicLinearization, ...]:
        return _kinematic_linearizations(self.plan, coordinates)

    @property
    def problem(self) -> NonlinearSystemProblem:
        return self.runtime.problem()

    def residual_jvp(
        self,
        coordinates: ContactCoordinates,
        direction: ContactCoordinates,
        args: Any = None,
        /,
    ) -> ContactCoordinates:
        arguments = self.args if args is None else args
        return jax.jvp(
            lambda q: self.problem.residual(q, arguments), (coordinates,), (direction,)
        )[1]


@final
class PreparedMonolithicContactEquilibrium(StrictModule, NonTrainableState):
    plan: MonolithicContactPlan
    initial: MonolithicContactState
    args: Any
    candidate_epoch: ContactCandidateEpoch
    runtime: _ContactRuntime
    prepared_id: str = eqx.field(static=True)

    def kinematic_linearizations(
        self, coordinates: ContactCoordinates, /
    ) -> tuple[ContactKinematicLinearization, ...]:
        return _kinematic_linearizations(self.plan, coordinates)

    @property
    def problem(self) -> NonlinearSystemProblem:
        return self.runtime.problem()

    def residual_jvp(
        self,
        coordinates: ContactCoordinates,
        direction: ContactCoordinates,
        args: Any = None,
        /,
    ) -> ContactCoordinates:
        arguments = self.args if args is None else args
        return jax.jvp(
            lambda q: self.problem.residual(q, arguments), (coordinates,), (direction,)
        )[1]


def _prepare_runtime(
    plan: MonolithicContactPlan,
    accepted: MonolithicContactState,
    step_size: Array | None,
    args: Any,
    /,
) -> tuple[_ContactRuntime, str]:
    if accepted.plan_id != plan.plan_id:
        raise ValueError("Accepted state belongs to another monolithic plan.")
    plan.dof_space.validate(accepted.coordinates)
    potential = _potential_mode(plan, step_size)
    if potential:
        for participant, q, history in zip(
            plan.participants, accepted.coordinates, accepted.histories, strict=True
        ):
            evidence = participant.evaluate(q, history, q, accepted.time, step_size, args)
            consistent = (
                np.asarray(evidence.potential_consistency_checked)
                & np.isfinite(np.asarray(evidence.potential_consistency_defect))
                & (
                    np.asarray(evidence.potential_consistency_defect)
                    <= plan.policy.lag_tolerance
                )
            )
            if not consistent:
                if plan.policy.merit == "potential":
                    raise ValueError(
                        "Potential merit requires verified residual-energy consistency."
                    )
                potential = False
                break
    positions = plan.contact_scene.positions(plan.geometry_states(accepted.coordinates))
    epoch = plan.search.build(plan.contact_scene, positions)
    frame = None
    if plan.friction is not None:
        frame = plan.friction.build_state(
            positions, epoch, state_version=accepted.state_version
        )
    runtime = _ContactRuntime(plan, accepted, epoch, step_size, frame, potential)
    return runtime, canonical_fingerprint(
        {
            "kind": "prepared-monolithic-contact",
            "plan": plan.plan_id,
            "epoch": epoch.epoch_id,
            "mode": "equilibrium" if step_size is None else "incremental",
        }
    )


@checked
def prepare_monolithic_contact_step(
    plan: MonolithicContactPlan,
    accepted: MonolithicContactState,
    step_size: ArrayLike,
    /,
    *,
    args: Any = None,
) -> PreparedMonolithicContactStep:
    dt = jnp.asarray(step_size, dtype=accepted.time.dtype)
    if dt.shape != () or not np.isfinite(np.asarray(dt)).all() or np.asarray(dt) <= 0:
        raise ValueError("step_size must be one finite positive scalar.")
    runtime, identifier = _prepare_runtime(plan, accepted, dt, args)
    return PreparedMonolithicContactStep(
        plan, accepted, dt, args, runtime.epoch, runtime, identifier
    )


@checked
def prepare_monolithic_contact_equilibrium(
    plan: MonolithicContactPlan, initial: MonolithicContactState, /, *, args: Any = None
) -> PreparedMonolithicContactEquilibrium:
    runtime, identifier = _prepare_runtime(plan, initial, None, args)
    return PreparedMonolithicContactEquilibrium(
        plan, initial, args, runtime.epoch, runtime, identifier
    )


@final
class ContactSolveDiagnostics(StrictModule):
    converged: Array
    iterations: Array
    residual_evaluations: Array
    jvp_actions: Array
    linear_iterations: Array
    line_search_evaluations: Array
    initial_residual_norm: Array
    final_residual_norm: Array
    accepted_step_size: Array
    lag_iterations: Array
    lag_residual: Array
    candidate_count: Array
    overflow_count: Array
    minimum_gap: Array
    minimum_feature_margin: Array
    potential_merit_used: Array
    status_reason: Array


@final
class MonolithicContactResult(StrictModule):
    """Atomic candidate and device-selected state with a host commit boundary.

    ``accepted_state`` selects every numerical field but retains staged FEM
    material trials and their previous static revisions, as standalone FEM
    dynamics does. Use ``commit()`` before preparing the next material-bearing
    step; candidates and participant evidence remain uncommitted.
    """

    previous: MonolithicContactState
    candidate: MonolithicContactState
    accepted_state: MonolithicContactState
    accepted: Array
    rollback_applied: Array
    diagnostics: ContactSolveDiagnostics
    participant_evidence: tuple[ContactParticipantEvaluation, ...]
    contact: ContactPotentialEvaluation
    friction: ContactFrictionEvaluation | None
    safety: ContactSafetyEvidence
    participant_step_limits: tuple[ContactParticipantStepLimit, ...]
    rejection_reasons: Array
    candidate_epoch: ContactCandidateEpoch
    step_size: Array | None
    arguments: Any
    nonlinear: NonlinearResult
    plan_id: str = eqx.field(static=True)

    def commit(self, /) -> MonolithicContactState:
        """Commit accepted FEM material trials and their static host revisions."""
        if not any(
            isinstance(history, FiniteElementDynamicsState)
            and history.materials is not None
            for history in self.candidate.histories
        ):
            return self.accepted_state
        # Static material revisions cannot share a device conditional's PyTree
        # with the previous revision. Numerical solving remains transformable.
        if not bool(self.accepted):
            return self.previous
        histories: list[PyTree[Array]] = []
        for history in self.candidate.histories:
            if (
                isinstance(history, FiniteElementDynamicsState)
                and history.materials is not None
            ):
                history = eqx.tree_at(
                    lambda value: value.materials, history, history.materials.commit()
                )
            histories.append(history)
        return eqx.tree_at(
            lambda value: value.histories, self.candidate, tuple(histories)
        )

    def rollback(self, /) -> MonolithicContactState:
        return self.previous


def _rejection_bit(condition: Array, reason: ContactRejectionReason, /) -> Array:
    return jnp.where(
        condition,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(int(reason), dtype=jnp.int32),
    )


def _accumulate_work(
    previous: NonlinearDiagnostics, current: NonlinearDiagnostics, /
) -> NonlinearDiagnostics:
    return eqx.tree_at(
        lambda d: (
            d.iterations,
            d.residual_evaluations,
            d.jvp_evaluations,
            d.jacobian_preparations,
            d.linear_iterations,
            d.accepted_steps,
            d.rejected_steps,
        ),
        previous,
        (
            previous.iterations + current.iterations,
            previous.residual_evaluations + current.residual_evaluations,
            previous.jvp_evaluations + current.jvp_evaluations,
            previous.jacobian_preparations + current.jacobian_preparations,
            previous.linear_iterations + current.linear_iterations,
            previous.accepted_steps + current.accepted_steps,
            previous.rejected_steps + current.rejected_steps,
        ),
    )


def _solve(runtime: _ContactRuntime, args: Any, /) -> MonolithicContactResult:
    plan = runtime.plan
    previous = runtime.accepted
    q = previous.coordinates
    frame = runtime.friction_state
    zero = jnp.asarray(0, dtype=previous.time.dtype)
    root_result = plan.policy.method.solve(
        runtime.problem(), q, termination=plan.policy.termination, args=args
    )
    sums = root_result.diagnostics
    q = plan.dof_space.validate(root_result.state)
    lag_count = jnp.asarray(1, dtype=jnp.int32)
    lag_residual = zero
    converged_lag = jnp.asarray(True, dtype=jnp.bool_)
    if plan.friction is not None:
        if frame is None:
            raise ValueError("Missing frozen friction frame.")
        friction = plan.friction
        refreshed = friction.build_state(
            plan.contact_scene.positions(plan.geometry_states(q)),
            runtime.epoch,
            state_version=previous.state_version + 1,
        )
        lag_residual = friction.lag_residual(frame, refreshed)

        def condition(
            carry: tuple[
                ContactCoordinates,
                ContactFrictionState,
                NonlinearResult,
                NonlinearDiagnostics,
                Array,
                Array,
            ],
        ) -> Array:
            coordinates, current_frame, result, work, count, defect = carry
            del coordinates, current_frame, work
            return (
                result.successful
                & (count < plan.policy.maximum_lag_iterations)
                & (defect > plan.policy.lag_tolerance)
            )

        def body(
            carry: tuple[
                ContactCoordinates,
                ContactFrictionState,
                NonlinearResult,
                NonlinearDiagnostics,
                Array,
                Array,
            ],
        ) -> tuple[
            ContactCoordinates,
            ContactFrictionState,
            NonlinearResult,
            NonlinearDiagnostics,
            Array,
            Array,
        ]:
            coordinates, current_frame, _, work, count, _ = carry
            current_runtime = eqx.tree_at(
                lambda r: r.friction_state, runtime, current_frame
            )
            result = plan.policy.method.solve(
                current_runtime.problem(),
                coordinates,
                termination=plan.policy.termination,
                args=args,
            )
            candidate_q = plan.dof_space.validate(result.state)
            candidate_frame = friction.build_state(
                plan.contact_scene.positions(plan.geometry_states(candidate_q)),
                runtime.epoch,
                state_version=previous.state_version + 1,
            )
            defect = friction.lag_residual(current_frame, candidate_frame)
            updated_work = _accumulate_work(work, result.diagnostics)
            return candidate_q, candidate_frame, result, updated_work, count + 1, defect

        q, frame, root_result, sums, lag_count, lag_residual = jax.lax.while_loop(
            condition, body, (q, refreshed, root_result, sums, lag_count, lag_residual)
        )
        converged_lag = lag_residual <= plan.policy.lag_tolerance
    final_runtime = eqx.tree_at(
        lambda r: r.friction_state, runtime, frame, is_leaf=lambda item: item is None
    )
    residual, evidence = final_runtime.residual(q, args)
    final_norm = jnp.sqrt(jnp.real(DualSpace(plan.dof_space).inner(residual, residual)))
    physical_converged = final_norm <= plan.policy.termination.residual_threshold(
        sums.initial_residual_norm
    )
    direction = jax.tree.map(lambda value, old: value - old, q, previous.coordinates)
    safety, limits, alpha = runtime.limits(previous.coordinates, direction, args)
    mechanics_valid = jnp.asarray(True, dtype=jnp.bool_)
    potential_valid = jnp.asarray(True, dtype=jnp.bool_)
    for participant, limit in zip(evidence.participants, limits, strict=True):
        mechanics_valid = (
            mechanics_valid
            & participant.successful
            & participant.admissible
            & participant.tangent_successful
            & limit.successful
            & limit.conservative
            & limit.guarantee.successful
            & (limit.maximum_step == 1.0)
        )
        if runtime.potential_merit:
            potential_valid = (
                potential_valid
                & participant.potential_consistency_checked
                & jnp.isfinite(participant.potential_consistency_defect)
                & (
                    jnp.abs(participant.potential_consistency_defect)
                    <= plan.policy.lag_tolerance
                )
            )
    trajectory = _ContactTrajectory(plan, previous.coordinates, direction)
    lower, upper, bounds_valid = trajectory(jnp.zeros_like(alpha), jnp.ones_like(alpha))
    extent = jnp.maximum(
        jnp.abs(lower - runtime.epoch.reference_positions),
        jnp.abs(upper - runtime.epoch.reference_positions),
    )
    envelope_valid = bounds_valid & jnp.all(
        jnp.sum(extent * extent, axis=-1) <= runtime.epoch.envelope_radius**2
    )
    rejection = _rejection_bit(runtime.epoch.successful, ContactRejectionReason.SEARCH)
    rejection = rejection | _rejection_bit(
        root_result.successful & physical_converged, ContactRejectionReason.NONLINEAR
    )
    rejection = rejection | _rejection_bit(
        mechanics_valid, ContactRejectionReason.PARTICIPANT
    )
    rejection = rejection | _rejection_bit(
        evidence.contact.successful, ContactRejectionReason.CONTACT
    )
    rejection = rejection | _rejection_bit(
        safety.successful & safety.conservative & (safety.step_size >= 1.0),
        ContactRejectionReason.CCD,
    )
    rejection = rejection | _rejection_bit(
        envelope_valid, ContactRejectionReason.ENVELOPE
    )
    rejection = rejection | _rejection_bit(
        potential_valid, ContactRejectionReason.POTENTIAL
    )
    friction_valid = converged_lag & (
        jnp.asarray(True, dtype=jnp.bool_)
        if evidence.friction is None
        else evidence.friction.successful
    )
    rejection = rejection | _rejection_bit(
        friction_valid, ContactRejectionReason.FRICTION
    )
    rejection = rejection | _rejection_bit(
        tree_allfinite(q) & tree_allfinite(residual), ContactRejectionReason.NONFINITE
    )
    accepted = rejection == 0
    advance = zero if runtime.step_size is None else runtime.step_size
    candidate = MonolithicContactState(
        q,
        tuple(value.trial_history for value in evidence.participants),
        previous.time + advance,
        previous.step + jnp.asarray(runtime.step_size is not None, dtype=jnp.int32),
        previous.state_version + 1,
        frame,
        plan.plan_id,
        previous.state_id,
    )
    selected = tree_where(accepted, candidate, previous)
    overflow = sum(
        (batch.overflow.astype(jnp.int32) for batch in runtime.epoch.active_batches),
        jnp.asarray(0, dtype=jnp.int32),
    )
    diagnostics = ContactSolveDiagnostics(
        accepted,
        sums.iterations,
        sums.residual_evaluations,
        sums.jvp_evaluations,
        sums.linear_iterations,
        sums.residual_evaluations - sums.jacobian_preparations,
        sums.initial_residual_norm,
        final_norm,
        alpha,
        lag_count,
        lag_residual,
        runtime.epoch.candidate_count,
        overflow,
        evidence.contact.minimum_gap,
        evidence.contact.minimum_feature_margin,
        jnp.asarray(runtime.potential_merit, dtype=jnp.bool_),
        root_result.status,
    )
    return MonolithicContactResult(
        previous,
        candidate,
        selected,
        accepted,
        ~accepted,
        diagnostics,
        evidence.participants,
        evidence.contact,
        evidence.friction,
        safety,
        limits,
        rejection,
        runtime.epoch,
        runtime.step_size,
        args,
        root_result,
        plan.plan_id,
    )


@eqx.filter_jit
@checked
def solve_monolithic_contact_step(
    prepared: PreparedMonolithicContactStep, /
) -> MonolithicContactResult:
    """Solve one step; call ``result.commit()`` to promote FEM material revisions."""
    return _solve(prepared.runtime, prepared.args)


@eqx.filter_jit
@checked
def solve_monolithic_contact_equilibrium(
    prepared: PreparedMonolithicContactEquilibrium, /
) -> MonolithicContactResult:
    """Solve equilibrium; call ``result.commit()`` to promote FEM material revisions."""
    return _solve(prepared.runtime, prepared.args)


__all__ = [
    "ContactRejectionReason",
    "ContactSolvePolicy",
    "MonolithicContactPlan",
    "MonolithicContactState",
    "PreparedMonolithicContactStep",
    "PreparedMonolithicContactEquilibrium",
    "ContactSolveDiagnostics",
    "MonolithicContactResult",
    "initialize_monolithic_contact_state",
    "prepare_monolithic_contact_step",
    "prepare_monolithic_contact_equilibrium",
    "solve_monolithic_contact_step",
    "solve_monolithic_contact_equilibrium",
]
