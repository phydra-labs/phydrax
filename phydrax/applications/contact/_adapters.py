#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, PyTree

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.contact import (
    ContactCandidateEpoch,
    ContactParticipantScene,
    DenseContactSearchPlan,
    SweepAndPruneContactSearchPlan,
)
from ...nn.parameters import ParameterSubspace
from ...solver import prepare_virtual_work_equilibrium, PreparedFieldEquilibrium
from ._closure import ContactClosurePlan
from ._coupling import (
    CrossDiscretizationContactResult,
    evaluate_cross_discretization_contact,
)
from ._route_state import ContactRouteState


NeuralContactStateTrace = Callable[[Mapping[str, Any], Any], Sequence[PyTree[Any]]]
NeuralContactRateTrace = Callable[[Mapping[str, Any], Any], Sequence[PyTree[Any]]]


class NeuralContactEvaluation(StrictModule):
    """Canonical fixed-manifold contact and participant virtual work."""

    contact: CrossDiscretizationContactResult
    virtual_work: tuple[PyTree[Array], ...]
    adapter_id: str = eqx.field(static=True)


class NeuralContactAdapter(StrictModule, NonTrainableState):
    """Neural traces evaluated on one canonical candidate manifold.

    Search is frozen in ``candidate_epoch`` while participant kinematics, closure
    response, and force pullbacks remain differentiable. Route history is never
    mutated by evaluation; callers explicitly commit ``contact.candidate_state``
    or retain ``contact.previous_state``.
    """

    scene: ContactParticipantScene
    search: DenseContactSearchPlan | SweepAndPruneContactSearchPlan
    closure_plan: ContactClosurePlan
    route_state: ContactRouteState
    candidate_epoch: ContactCandidateEpoch
    rest_positions: Array
    state_trace: NeuralContactStateTrace = eqx.field(static=True)
    rate_trace: NeuralContactRateTrace | None = eqx.field(static=True)
    step_size: float = eqx.field(static=True)
    activation_distance: float | None = eqx.field(static=True)
    adapter_id: str = eqx.field(static=True)

    def __init__(
        self,
        scene: ContactParticipantScene,
        search: DenseContactSearchPlan | SweepAndPruneContactSearchPlan,
        closure_plan: ContactClosurePlan,
        route_state: ContactRouteState,
        candidate_epoch: ContactCandidateEpoch,
        rest_positions: ArrayLike,
        state_trace: NeuralContactStateTrace,
        /,
        *,
        adapter_id: str,
        rate_trace: NeuralContactRateTrace | None = None,
        step_size: float = 1.0,
        activation_distance: float | None = None,
    ):
        if not isinstance(scene, ContactParticipantScene):
            raise TypeError("scene must be ContactParticipantScene.")
        if not isinstance(
            search, (DenseContactSearchPlan, SweepAndPruneContactSearchPlan)
        ):
            raise TypeError("search must be a canonical contact search plan.")
        if not isinstance(closure_plan, ContactClosurePlan):
            raise TypeError("closure_plan must be ContactClosurePlan.")
        if not isinstance(route_state, ContactRouteState):
            raise TypeError("route_state must be ContactRouteState.")
        if not isinstance(candidate_epoch, ContactCandidateEpoch):
            raise TypeError("candidate_epoch must be ContactCandidateEpoch.")
        if candidate_epoch.search_id != search.plan_id:
            raise ValueError("candidate_epoch belongs to another search plan.")
        if route_state.closure_id != closure_plan.closure_id:
            raise ValueError("route_state belongs to another contact closure.")
        if not callable(state_trace) or (
            rate_trace is not None and not callable(rate_trace)
        ):
            raise TypeError("Neural contact state and rate traces must be callable.")
        identifier = str(adapter_id)
        step = float(step_size)
        if not identifier or step <= 0.0:
            raise ValueError("adapter_id and step_size must be valid.")
        supplied_rest = jnp.asarray(rest_positions)
        expected_rest_shape = (scene.vertex_count, scene.ambient_dimension)
        if supplied_rest.shape != expected_rest_shape:
            raise ValueError(f"rest_positions must have shape {expected_rest_shape}.")
        self.scene = scene
        self.search = search
        self.closure_plan = closure_plan
        self.route_state = route_state
        self.candidate_epoch = candidate_epoch
        self.rest_positions = supplied_rest
        self.state_trace = state_trace
        self.rate_trace = rate_trace
        self.step_size = step
        self.activation_distance = activation_distance
        self.adapter_id = canonical_fingerprint(
            {
                "kind": "neural-contact-adapter",
                "declared_id": identifier,
                "scene": scene.scene_id,
                "search": search.plan_id,
                "closure": closure_plan.closure_id,
                "epoch": candidate_epoch.epoch_id,
            }
        )

    def field_jet(
        self,
        functions: Mapping[str, Any],
        args: Any = None,
        /,
    ) -> tuple[PyTree[Any], ...]:
        states = tuple(self.state_trace(functions, args))
        if len(states) != len(self.scene.participants):
            raise ValueError("Neural contact state trace changed participant count.")
        for participant, state in zip(self.scene.participants, states, strict=True):
            participant.source_space.validate(state)
        return states

    def _rates(
        self,
        functions: Mapping[str, Any],
        args: Any,
        /,
    ) -> tuple[PyTree[Any], ...]:
        if self.rate_trace is None:
            return tuple(p.tangent_space.zeros() for p in self.scene.participants)
        rates = tuple(self.rate_trace(functions, args))
        if len(rates) != len(self.scene.participants):
            raise ValueError("Neural contact rate trace changed participant count.")
        for participant, rate in zip(self.scene.participants, rates, strict=True):
            participant.tangent_space.validate(rate)
        return rates

    def _evaluate_states(
        self,
        functions: Mapping[str, Any],
        states: tuple[PyTree[Any], ...],
        args: Any,
        /,
    ) -> CrossDiscretizationContactResult:
        return evaluate_cross_discretization_contact(
            self.scene,
            states,
            self._rates(functions, args),
            self.search,
            self.closure_plan,
            self.route_state,
            self.step_size,
            self.rest_positions,
            activation_distance=self.activation_distance,
            candidate_epoch=self.candidate_epoch,
        )

    def evaluate(
        self,
        functions: Mapping[str, Any],
        args: Any = None,
        /,
    ) -> NeuralContactEvaluation:
        states = self.field_jet(functions, args)
        contact = self._evaluate_states(functions, states, args)
        virtual_work = tuple(
            jax.tree.map(lambda leaf: -leaf, value)
            for value in contact.generalized_efforts
        )
        return NeuralContactEvaluation(contact, virtual_work, self.adapter_id)

    def virtual_work(
        self,
        functions: Mapping[str, Any],
        states: tuple[PyTree[Any], ...],
        args: Any = None,
        /,
    ) -> tuple[PyTree[Array], ...]:
        contact = self._evaluate_states(functions, states, args)
        return tuple(
            jax.tree.map(lambda leaf: -leaf, value)
            for value in contact.generalized_efforts
        )

    def prepare_equilibrium(
        self,
        functions: Mapping[str, Any],
        parameter_subspace: ParameterSubspace,
        /,
        *,
        problem_id: str = "neural-contact",
    ) -> PreparedFieldEquilibrium:
        return prepare_virtual_work_equilibrium(
            functions,
            _neural_contact_field_jet,
            _neural_contact_virtual_work,
            parameter_subspace,
            self,
            realization_id=self.candidate_epoch.epoch_id,
            provenance_id=self.adapter_id,
            problem_id=problem_id,
        )


def _neural_contact_field_jet(
    functions: Mapping[str, Any],
    realization: NeuralContactAdapter,
    args: Any,
    /,
) -> PyTree[Array]:
    return realization.field_jet(functions, args)


def _neural_contact_virtual_work(
    functions: Mapping[str, Any],
    states: tuple[PyTree[Any], ...],
    realization: NeuralContactAdapter,
    args: Any,
    /,
) -> PyTree[Array]:
    return realization.virtual_work(functions, states, args)


__all__ = [
    "NeuralContactAdapter",
    "NeuralContactEvaluation",
    "NeuralContactRateTrace",
    "NeuralContactStateTrace",
]
