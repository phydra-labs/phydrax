# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import abc
from collections.abc import Callable, Mapping
from typing import Any, final, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
from jaxtyping import PyTree

from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.contact._guarantee import ContactGuaranteeEvidence
from ...discretization.contact._participant import AbstractContactParticipant
from ...linalg import AbstractLinearOperator, AbstractVectorSpace, FunctionLinearOperator
from ...typing import checked


type ContactMeritMode = Literal["automatic", "potential", "residual"]


@final
class ContactParticipantEvaluation(StrictModule):
    residual: PyTree[Array]
    trial_history: PyTree[Array]
    potential_energy: Array | None
    successful: Array
    admissible: Array
    tangent_successful: Array
    potential_consistency_checked: Array
    potential_consistency_defect: Array
    suggested_step: Array
    status_code: Array
    diagnostics: Mapping[str, Array]

    @checked
    def __init__(
        self,
        residual: PyTree[Array],
        trial_history: PyTree[Array],
        potential_energy: ArrayLike | None,
        successful: ArrayLike,
        admissible: ArrayLike,
        tangent_successful: ArrayLike,
        potential_consistency_checked: ArrayLike,
        potential_consistency_defect: ArrayLike,
        suggested_step: ArrayLike,
        status_code: ArrayLike,
        diagnostics: Mapping[str, ArrayLike],
    ) -> None:
        leaves = jax.tree.leaves(residual)
        if not leaves:
            raise ValueError("Contact participant residual must contain array leaves.")
        dtype = leaves[0].real.dtype
        success = jnp.asarray(successful, dtype=jnp.bool_)
        domain = jnp.asarray(admissible, dtype=jnp.bool_)
        tangent = jnp.asarray(tangent_successful, dtype=jnp.bool_)
        consistency = jnp.asarray(potential_consistency_checked, dtype=jnp.bool_)
        defect = jnp.asarray(potential_consistency_defect, dtype=dtype)
        suggestion = jnp.asarray(suggested_step, dtype=dtype)
        status = jnp.asarray(status_code, dtype=jnp.int32)
        potential = (
            None
            if potential_energy is None
            else jnp.asarray(potential_energy, dtype=dtype)
        )
        if any(
            value.shape != ()
            for value in (
                success,
                domain,
                tangent,
                consistency,
                defect,
                suggestion,
                status,
            )
        ):
            raise ValueError("Contact participant decision evidence must be scalar.")
        if potential is not None and potential.shape != ():
            raise ValueError("Contact participant potential must be scalar or None.")
        self.residual = residual
        self.trial_history = trial_history
        self.potential_energy = potential
        self.successful = success
        self.admissible = domain
        self.tangent_successful = tangent
        self.potential_consistency_checked = consistency
        self.potential_consistency_defect = defect
        self.suggested_step = suggestion
        self.status_code = status
        self.diagnostics = {
            name: jnp.asarray(value) for name, value in diagnostics.items()
        }


@final
class ContactParticipantStepLimit(StrictModule):
    maximum_step: Array
    successful: Array
    conservative: Array
    guarantee: ContactGuaranteeEvidence
    reason_code: Array
    limiting_feature_key: Array
    participant_id: str = eqx.field(static=True)


@final
class ContactKinematicLinearization(StrictModule):
    positions: Array
    tangent: AbstractLinearOperator
    participant_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)


class AbstractImplicitContactParticipant(StrictModule, NonTrainableState):
    @property
    @abc.abstractmethod
    def dof_space(self) -> AbstractVectorSpace:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def participant_id(self) -> str:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def has_potential(self) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate(
        self,
        state: PyTree[Array],
        committed_history: PyTree[Array],
        accepted_state: PyTree[Array],
        time: Array,
        step_size: Array | None,
        args: Any,
        /,
    ) -> ContactParticipantEvaluation:
        raise NotImplementedError

    @abc.abstractmethod
    def residual_jvp(
        self,
        state: PyTree[Array],
        direction: PyTree[Array],
        committed_history: PyTree[Array],
        accepted_state: PyTree[Array],
        time: Array,
        step_size: Array | None,
        args: Any,
        /,
    ) -> PyTree[Array]:
        raise NotImplementedError

    @abc.abstractmethod
    def potential(
        self,
        state: PyTree[Array],
        committed_history: PyTree[Array],
        accepted_state: PyTree[Array],
        time: Array,
        step_size: Array | None,
        args: Any,
        /,
    ) -> Array | None:
        raise NotImplementedError

    @abc.abstractmethod
    def potential_hvp(
        self,
        state: PyTree[Array],
        direction: PyTree[Array],
        committed_history: PyTree[Array],
        accepted_state: PyTree[Array],
        time: Array,
        step_size: Array | None,
        args: Any,
        /,
    ) -> PyTree[Array] | None:
        raise NotImplementedError

    @abc.abstractmethod
    def step_limit(
        self,
        state: PyTree[Array],
        direction: PyTree[Array],
        accepted_state: PyTree[Array],
        time: Array,
        step_size: Array | None,
        args: Any,
        /,
    ) -> ContactParticipantStepLimit:
        raise NotImplementedError

    def contact_velocity(
        self,
        state: PyTree[Array],
        committed_history: PyTree[Array],
        accepted_state: PyTree[Array],
        time: Array,
        step_size: Array,
        args: Any,
        /,
    ) -> PyTree[Array]:
        del committed_history, time, args
        return jax.tree.map(
            lambda current, accepted: (current - accepted) / step_size,
            state,
            accepted_state,
        )

    def contact_velocity_scale(self, step_size: Array, /) -> Array | None:
        return jnp.ones_like(step_size) / step_size


@final
class FunctionalImplicitContactParticipant(AbstractImplicitContactParticipant):
    space: AbstractVectorSpace
    residual_action: Callable[..., ContactParticipantEvaluation]
    residual_jvp_action: Callable[..., PyTree[Array]]
    step_limit_action: Callable[..., ContactParticipantStepLimit]
    potential_action: Callable[..., Array] | None
    potential_hvp_action: Callable[..., PyTree[Array]] | None
    velocity_action: Callable[..., PyTree[Array]] | None
    velocity_scale_action: Callable[[Array], Array | None] | None
    _participant_id: str = eqx.field(static=True)

    @property
    def dof_space(self) -> AbstractVectorSpace:
        return self.space

    @property
    def participant_id(self) -> str:
        return self._participant_id

    @property
    def has_potential(self) -> bool:
        return self.potential_action is not None

    def evaluate(
        self,
        state: PyTree[Array],
        committed_history: PyTree[Array],
        accepted_state: PyTree[Array],
        time: Array,
        step_size: Array | None,
        args: Any,
        /,
    ) -> ContactParticipantEvaluation:
        return self.residual_action(
            self.space.validate(state),
            committed_history,
            self.space.validate(accepted_state),
            time,
            step_size,
            args,
        )

    def residual_jvp(
        self,
        state: PyTree[Array],
        direction: PyTree[Array],
        committed_history: PyTree[Array],
        accepted_state: PyTree[Array],
        time: Array,
        step_size: Array | None,
        args: Any,
        /,
    ) -> PyTree[Array]:
        return self.residual_jvp_action(
            self.space.validate(state),
            self.space.validate(direction),
            committed_history,
            self.space.validate(accepted_state),
            time,
            step_size,
            args,
        )

    def potential(
        self,
        state: PyTree[Array],
        committed_history: PyTree[Array],
        accepted_state: PyTree[Array],
        time: Array,
        step_size: Array | None,
        args: Any,
        /,
    ) -> Array | None:
        if self.potential_action is None:
            return None
        return self.potential_action(
            state, committed_history, accepted_state, time, step_size, args
        )

    def potential_hvp(
        self,
        state: PyTree[Array],
        direction: PyTree[Array],
        committed_history: PyTree[Array],
        accepted_state: PyTree[Array],
        time: Array,
        step_size: Array | None,
        args: Any,
        /,
    ) -> PyTree[Array] | None:
        if self.potential_hvp_action is None:
            return None
        return self.potential_hvp_action(
            state, direction, committed_history, accepted_state, time, step_size, args
        )

    def step_limit(
        self,
        state: PyTree[Array],
        direction: PyTree[Array],
        accepted_state: PyTree[Array],
        time: Array,
        step_size: Array | None,
        args: Any,
        /,
    ) -> ContactParticipantStepLimit:
        return self.step_limit_action(
            state, direction, accepted_state, time, step_size, args
        )

    def contact_velocity(
        self,
        state: PyTree[Array],
        committed_history: PyTree[Array],
        accepted_state: PyTree[Array],
        time: Array,
        step_size: Array,
        args: Any,
        /,
    ) -> PyTree[Array]:
        if self.velocity_action is None:
            return super().contact_velocity(
                state, committed_history, accepted_state, time, step_size, args
            )
        return self.velocity_action(
            state, committed_history, accepted_state, time, step_size, args
        )

    def contact_velocity_scale(self, step_size: Array, /) -> Array | None:
        if self.velocity_scale_action is None:
            return super().contact_velocity_scale(step_size)
        return self.velocity_scale_action(step_size)


@checked
def make_implicit_contact_participant(
    dof_space: AbstractVectorSpace,
    residual_action: Callable[..., ContactParticipantEvaluation],
    residual_jvp_action: Callable[..., PyTree[Array]],
    step_limit_action: Callable[..., ContactParticipantStepLimit],
    /,
    *,
    participant_id: str,
    potential_action: Callable[..., Array] | None = None,
    potential_hvp_action: Callable[..., PyTree[Array]] | None = None,
    velocity_action: Callable[..., PyTree[Array]] | None = None,
    velocity_scale_action: Callable[[Array], Array | None] | None = None,
) -> FunctionalImplicitContactParticipant:
    if not participant_id:
        raise ValueError("participant_id must be nonempty.")
    if (potential_action is None) != (potential_hvp_action is None):
        raise ValueError("Potential and exact HVP actions must be supplied together.")
    if (velocity_action is None) != (velocity_scale_action is None):
        raise ValueError(
            "Velocity and derivative scale actions must be supplied together."
        )
    return FunctionalImplicitContactParticipant(
        dof_space,
        residual_action,
        residual_jvp_action,
        step_limit_action,
        potential_action,
        potential_hvp_action,
        velocity_action,
        velocity_scale_action,
        participant_id,
    )


@final
class _KinematicAction(StrictModule, NonTrainableState):
    geometry: AbstractContactParticipant
    state: PyTree[Array]

    def __call__(self, direction: PyTree[Array], /) -> Array:
        return self.geometry.velocities(self.state, direction)


@final
class _KinematicTranspose(StrictModule, NonTrainableState):
    geometry: AbstractContactParticipant
    state: PyTree[Array]

    def __call__(self, effort: Array, /) -> PyTree[Array]:
        return self.geometry.effort_pullback(self.state, effort)


@checked
def linearize_contact_kinematics(
    geometry: AbstractContactParticipant, state: PyTree[Array], /
) -> ContactKinematicLinearization:
    base = geometry.source_space.validate(state)
    tangent = FunctionLinearOperator(
        _KinematicAction(geometry, base),
        source=geometry.tangent_space,
        target=geometry.contact_velocity_space,
        transpose_action=_KinematicTranspose(geometry, base),
        operator_id=f"{geometry.participant_id}:kinematic-tangent",
    )
    return ContactKinematicLinearization(
        geometry.positions(base),
        tangent,
        geometry.participant_id,
        geometry.surface_plan.topology_id,
    )


@eqx.filter_custom_jvp
def participant_residual(
    participant: AbstractImplicitContactParticipant,
    state: PyTree[Array],
    committed_history: PyTree[Array],
    accepted_state: PyTree[Array],
    time: Array,
    step_size: Array | None,
    args: Any,
    /,
) -> PyTree[Array]:
    """Physical residual with the participant's authoritative state tangent."""
    return participant.evaluate(
        state, committed_history, accepted_state, time, step_size, args
    ).residual


@participant_residual.def_jvp
def _participant_residual_jvp(
    primals: tuple[
        AbstractImplicitContactParticipant,
        PyTree[Array],
        PyTree[Array],
        PyTree[Array],
        Array,
        Array | None,
        Any,
    ],
    tangents: tuple[Any, ...],
) -> tuple[PyTree[Array], PyTree[Array]]:
    participant, state, history, accepted, time, step_size, args = primals
    owner_dot, state_dot, history_dot, accepted_dot, time_dot, step_dot, args_dot = (
        tangents
    )
    context = (participant, history, accepted, time, step_size, args)
    context_dot = (owner_dot, history_dot, accepted_dot, time_dot, step_dot, args_dot)

    def context_residual(
        values: tuple[
            AbstractImplicitContactParticipant,
            PyTree[Array],
            PyTree[Array],
            Array,
            Array | None,
            Any,
        ],
        /,
    ) -> PyTree[Array]:
        owner, old, base, clock, dt, arguments = values
        return owner.evaluate(state, old, base, clock, dt, arguments).residual

    residual, context_action = eqx.filter_jvp(
        context_residual, (context,), (context_dot,)
    )
    direction = jax.tree.map(
        lambda value, tangent: jnp.zeros_like(value) if tangent is None else tangent,
        state,
        state_dot,
        is_leaf=lambda value: value is None,
    )
    action = participant.residual_jvp(
        state, direction, history, accepted, time, step_size, args
    )
    derivative = jax.tree.map(
        lambda physical, extra: physical if extra is None else physical + extra,
        action,
        context_action,
        is_leaf=lambda value: value is None,
    )
    return residual, derivative
