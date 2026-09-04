#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import AbstractAttribute, StrictModule
from ._grid import EvolutionGrid, IterationGrid, TimeGrid
from ._layout import StateLayout
from ._system import (
    AbstractInputPolicy,
    ContinuousSystem,
    DiscreteStepContext,
    DiscreteSystem,
    DiscreteTransitionEvidence,
    DiscreteTransitionResult,
)


EVOLUTION_SUCCESS = 0
EVOLUTION_NONFINITE = 1
EVOLUTION_OUTSIDE_GEOMETRY = 2
EVOLUTION_BACKEND_FAILED = 3


System = ContinuousSystem | DiscreteSystem


def _identifier(value: str, owner: str, /) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{owner} must be a non-empty string.")
    return value


def _finite_state(state: Array, state_rank: int, /) -> Array:
    finite = jnp.isfinite(state)
    if state_rank:
        finite = jnp.all(finite, axis=tuple(range(finite.ndim - state_rank, finite.ndim)))
    return finite


class EvolutionStep(StrictModule):
    """One pathwise evolution segment with explicit validity and provenance."""

    source_coordinate: Array
    target_coordinate: Array
    final_state: Array
    valid: Array
    status: Array
    backend_status: Array
    system_id: str = eqx.field(static=True)
    evolution_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)
    backend_id: str = eqx.field(static=True)
    discretization_id: str = eqx.field(static=True)
    approximation_id: str = eqx.field(static=True)
    transition_evidence: DiscreteTransitionEvidence | None = None


class EvolutionTangentStep(StrictModule):
    """One primal evolution segment and one propagated local tangent."""

    primal: EvolutionStep
    tangent: Array
    valid: Array
    status: Array
    tangent_method_id: str = eqx.field(static=True)


class EvolutionTrajectory(StrictModule):
    """Unbatched pathwise states saved on one explicit evolution grid."""

    grid: EvolutionGrid
    states: Array
    valid: Array
    status: Array
    backend_status: Array
    state_layout: StateLayout
    system_id: str = eqx.field(static=True)
    evolution_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)
    backend_id: str = eqx.field(static=True)
    discretization_id: str = eqx.field(static=True)
    approximation_id: str = eqx.field(static=True)
    transition_evidence: DiscreteTransitionEvidence | None = None

    @property
    def successful(self) -> Array:
        return jnp.all(self.valid) & jnp.all(self.status == EVOLUTION_SUCCESS)

    @property
    def final_state(self) -> Array:
        return self.states[-1]


class AbstractEvolution(StrictModule):
    """Pathwise segment evolution independent of analysis algorithms."""

    system: AbstractAttribute[System]
    evolution_id: AbstractAttribute[str]
    method_id: AbstractAttribute[str]
    backend_id: AbstractAttribute[str]
    discretization_id: AbstractAttribute[str]
    approximation_id: AbstractAttribute[str]

    @property
    def state_layout(self) -> StateLayout:
        return self.system.state_layout

    @abc.abstractmethod
    def advance(
        self,
        state: ArrayLike,
        source_coordinate: ArrayLike,
        target_coordinate: ArrayLike,
        args: Any = None,
        /,
    ) -> EvolutionStep:
        raise NotImplementedError


class AbstractDifferentiableEvolution(AbstractEvolution):
    """Pathwise evolution with an explicit local tangent action."""

    tangent_method_id: AbstractAttribute[str]

    @abc.abstractmethod
    def tangent_action(
        self,
        state: ArrayLike,
        tangent: ArrayLike,
        source_coordinate: ArrayLike,
        target_coordinate: ArrayLike,
        args: Any = None,
        /,
    ) -> EvolutionTangentStep:
        raise NotImplementedError


class DiscreteEvolution(AbstractDifferentiableEvolution):
    """One-step evolution and JVP for a declared discrete system."""

    system: DiscreteSystem
    input_policy: AbstractInputPolicy | None
    evolution_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)
    backend_id: str = eqx.field(static=True)
    discretization_id: str = eqx.field(static=True)
    approximation_id: str = eqx.field(static=True)
    tangent_method_id: str = eqx.field(static=True)

    def __init__(
        self,
        system: DiscreteSystem,
        /,
        *,
        input_policy: AbstractInputPolicy | None = None,
        evolution_id: str | None = None,
    ):
        if not isinstance(system, DiscreteSystem):
            raise TypeError("DiscreteEvolution system must be a DiscreteSystem.")
        if input_policy is not None and not isinstance(input_policy, AbstractInputPolicy):
            raise TypeError("input_policy must be an AbstractInputPolicy or None.")
        system_input_layout = system.input_layout
        if system_input_layout is None:
            if input_policy is not None:
                raise ValueError(
                    "An autonomous DiscreteEvolution does not accept an input policy."
                )
        elif input_policy is None:
            raise ValueError(
                "An input-driven DiscreteEvolution requires exactly one input policy."
            )
        elif input_policy.input_layout.layout_id != system_input_layout.layout_id:
            raise ValueError("Input policy and system input layouts must match exactly.")
        resolved_id = (
            f"{system.system_id}:discrete-evolution"
            if evolution_id is None
            else _identifier(evolution_id, "DiscreteEvolution evolution_id")
        )
        self.system = system
        self.input_policy = input_policy
        self.evolution_id = resolved_id
        self.method_id = "explicit-discrete-transition"
        self.backend_id = "backend:jax"
        self.discretization_id = "one-map-iterate"
        self.approximation_id = "exact-declared-transition"
        self.tangent_method_id = "jax-jvp:declared-transition"

    def _map_result(
        self, context: DiscreteStepContext, state: Array, args: Any, /
    ) -> DiscreteTransitionResult:
        inputs = (
            None
            if self.input_policy is None
            else self.input_policy.evaluate_step(context, state, args)
        )
        return self.system.evaluate_result(context, state, args, inputs=inputs)

    def _map(self, context: DiscreteStepContext, state: Array, args: Any, /) -> Array:
        result = self._map_result(context, state, args)
        return jnp.where(
            result.successful,
            result.accepted_state,
            jnp.full_like(result.accepted_state, jnp.nan),
        )

    def advance(
        self,
        state: ArrayLike,
        source_coordinate: ArrayLike,
        target_coordinate: ArrayLike,
        args: Any = None,
        /,
    ) -> EvolutionStep:
        state_array = jnp.asarray(state)
        source = jnp.asarray(source_coordinate)
        target = jnp.asarray(target_coordinate)
        if source.shape != () or target.shape != ():
            raise ValueError("Evolution segment coordinates must be scalar.")
        context = DiscreteStepContext(source, target, jnp.asarray(0, dtype=jnp.int32))
        transition = self._map_result(context, state_array, args)
        final_state = transition.accepted_state
        finite = jnp.all(jnp.isfinite(final_state))
        membership = jnp.asarray(
            self.state_layout.geometry.contains(final_state), dtype=bool
        )
        if membership.shape != ():
            raise ValueError("State geometry contains() must return one scalar boolean.")
        valid = transition.successful & finite & membership
        status = jnp.where(
            ~transition.successful,
            EVOLUTION_BACKEND_FAILED,
            jnp.where(
                ~finite,
                EVOLUTION_NONFINITE,
                jnp.where(
                    ~membership,
                    EVOLUTION_OUTSIDE_GEOMETRY,
                    EVOLUTION_SUCCESS,
                ),
            ),
        ).astype(jnp.int32)
        evidence = DiscreteTransitionEvidence(
            transition.candidate_state[None, ...],
            transition.accepted_state[None, ...],
            jnp.asarray([True]),
            transition.successful[None],
            transition.status[None],
        )
        return EvolutionStep(
            source_coordinate=source,
            target_coordinate=target,
            final_state=final_state,
            valid=valid,
            status=status,
            backend_status=transition.status,
            system_id=self.system.system_id,
            evolution_id=self.evolution_id,
            method_id=self.method_id,
            backend_id=self.backend_id,
            discretization_id=self.discretization_id,
            approximation_id=self.approximation_id,
            transition_evidence=evidence,
        )

    def tangent_action(
        self,
        state: ArrayLike,
        tangent: ArrayLike,
        source_coordinate: ArrayLike,
        target_coordinate: ArrayLike,
        args: Any = None,
        /,
    ) -> EvolutionTangentStep:
        state_array = jnp.asarray(state)
        vector = jnp.asarray(tangent)
        if state_array.shape != self.state_layout.shape:
            raise ValueError(
                f"state must have shape {self.state_layout.shape}; got {state_array.shape}."
            )
        if vector.shape != self.state_layout.shape:
            raise ValueError(
                f"tangent must have shape {self.state_layout.shape}; got {vector.shape}."
            )
        source = jnp.asarray(source_coordinate)
        target = jnp.asarray(target_coordinate)
        primal = self.advance(state_array, source, target, args)
        geometry = self.state_layout.geometry
        if geometry.trivial:
            _, propagated = jax.jvp(
                lambda point: self._map(
                    DiscreteStepContext(source, target, jnp.asarray(0, dtype=jnp.int32)),
                    point,
                    args,
                ),
                (state_array,),
                (vector,),
            )
        else:
            zero = jnp.zeros_like(state_array)

            def local_map(local):
                perturbed = geometry.retract(state_array, local)
                endpoint = self._map(
                    DiscreteStepContext(source, target, jnp.asarray(0, dtype=jnp.int32)),
                    perturbed,
                    args,
                )
                return geometry.inverse_retract(primal.final_state, endpoint)

            _, propagated = jax.jvp(local_map, (zero,), (vector,))
        propagated = jnp.where(
            primal.valid,
            propagated,
            jnp.full_like(propagated, jnp.nan),
        )
        tangent_finite = jnp.all(jnp.isfinite(propagated))
        valid = primal.valid & tangent_finite
        status = jnp.where(
            ~primal.valid,
            primal.status,
            jnp.where(tangent_finite, EVOLUTION_SUCCESS, EVOLUTION_NONFINITE),
        ).astype(jnp.int32)
        return EvolutionTangentStep(
            primal=primal,
            tangent=propagated,
            valid=valid,
            status=status,
            tangent_method_id=self.tangent_method_id,
        )


def evolve(
    evolution: AbstractEvolution,
    initial_state: ArrayLike,
    grid: EvolutionGrid,
    /,
    *,
    args: Any = None,
) -> EvolutionTrajectory:
    """Apply one pathwise evolution to every adjacent pair in an explicit grid."""
    if not isinstance(evolution, AbstractEvolution):
        raise TypeError("evolution must be an AbstractEvolution.")
    if not isinstance(grid, (TimeGrid, IterationGrid)):
        raise TypeError("grid must be a TimeGrid or IterationGrid.")
    initial = jnp.asarray(initial_state)
    if initial.shape != evolution.state_layout.shape:
        raise ValueError(
            f"initial_state must have shape {evolution.state_layout.shape}; got {initial.shape}."
        )
    initial_finite = jnp.all(jnp.isfinite(initial))
    initial_member = jnp.asarray(
        evolution.state_layout.geometry.contains(initial), dtype=bool
    )
    if initial_member.shape != ():
        raise ValueError("State geometry contains() must return one scalar boolean.")
    initial_valid = initial_finite & initial_member
    sources = grid.coordinates[:-1]
    targets = grid.coordinates[1:]

    def step(carry, coordinates):
        state, prior_valid = carry
        source, target = coordinates
        result = evolution.advance(state, source, target, args)
        valid = prior_valid & result.valid
        next_state = jnp.where(prior_valid, result.final_state, state)
        evidence = result.transition_evidence
        if evidence is None:
            candidate = result.final_state
            accepted = result.final_state
            transition_attempted = jnp.asarray(True)
            transition_successful = result.valid
            transition_status = result.backend_status
        else:
            candidate = evidence.candidate_states[0]
            transition_attempted = evidence.attempted[0]
            accepted = evidence.accepted_states[0]
            transition_successful = evidence.successful[0]
            transition_status = evidence.status[0]
        recorded_attempted = prior_valid & transition_attempted
        recorded_candidate = jnp.where(
            recorded_attempted,
            candidate,
            jnp.full_like(candidate, jnp.nan),
        )
        recorded_accepted = jnp.where(
            recorded_attempted,
            accepted,
            jnp.full_like(accepted, jnp.nan),
        )
        recorded_successful = recorded_attempted & transition_successful
        recorded_status = jnp.where(
            recorded_attempted,
            transition_status,
            jnp.asarray(0, dtype=jnp.int32),
        ).astype(jnp.int32)
        evolution_status = jnp.where(
            prior_valid,
            result.status,
            jnp.asarray(EVOLUTION_SUCCESS, dtype=jnp.int32),
        ).astype(jnp.int32)
        backend_status = jnp.where(
            recorded_attempted,
            result.backend_status,
            jnp.asarray(0, dtype=jnp.int32),
        ).astype(jnp.int32)
        return (next_state, valid), (
            next_state,
            valid,
            evolution_status,
            backend_status,
            recorded_candidate,
            recorded_accepted,
            recorded_attempted,
            recorded_successful,
            recorded_status,
        )

    (
        (_, _),
        (
            final_states,
            valid_steps,
            statuses,
            backend_statuses,
            candidate_states,
            accepted_states,
            transition_attempted,
            transition_successful,
            transition_status,
        ),
    ) = jax.lax.scan(
        step,
        (initial, initial_valid),
        (sources, targets),
    )
    states = jnp.concatenate((initial[None, ...], final_states), axis=0)
    valid = jnp.concatenate((initial_valid[None], valid_steps), axis=0)
    transition_evidence = (
        DiscreteTransitionEvidence(
            candidate_states,
            accepted_states,
            transition_attempted,
            transition_successful,
            transition_status,
        )
        if isinstance(evolution.system, DiscreteSystem)
        else None
    )
    return EvolutionTrajectory(
        grid=grid,
        states=states,
        valid=valid,
        status=statuses,
        backend_status=backend_statuses,
        state_layout=evolution.state_layout,
        system_id=evolution.system.system_id,
        evolution_id=evolution.evolution_id,
        method_id=evolution.method_id,
        backend_id=evolution.backend_id,
        discretization_id=evolution.discretization_id,
        approximation_id=evolution.approximation_id,
        transition_evidence=transition_evidence,
    )


__all__ = [
    "AbstractDifferentiableEvolution",
    "AbstractEvolution",
    "DiscreteEvolution",
    "EVOLUTION_BACKEND_FAILED",
    "EVOLUTION_NONFINITE",
    "EVOLUTION_OUTSIDE_GEOMETRY",
    "EVOLUTION_SUCCESS",
    "EvolutionStep",
    "EvolutionTangentStep",
    "EvolutionTrajectory",
    "evolve",
]
