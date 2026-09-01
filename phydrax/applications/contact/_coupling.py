#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
from jaxtyping import Array, ArrayLike, PyTree

from ..._strict import StrictModule
from ...discretization.contact._kinematics import (
    ContactKinematicsEpoch,
    evaluate_contact_kinematics,
)
from ...discretization.contact._participant import ContactParticipantScene
from ...discretization.contact._search import (
    ContactCandidateEpoch,
    DenseContactSearchPlan,
    SweepAndPruneContactSearchPlan,
)
from ._closure import (
    ContactClosureEvaluation,
    ContactClosurePlan,
    evaluate_contact_closure,
)
from ._route_state import (
    ContactRouteState,
    ContactRouteStateTransition,
    remap_contact_route_state,
)
from ._smooth_formulation import (
    assemble_smooth_contact,
    SmoothContactAssembly,
)


class CrossDiscretizationContactResult(StrictModule):
    positions: Array
    velocities: Array
    candidate_epoch: ContactCandidateEpoch
    kinematics: ContactKinematicsEpoch
    route_transition: ContactRouteStateTransition
    closure: ContactClosureEvaluation
    assembly: SmoothContactAssembly
    generalized_forces: tuple[PyTree[Array], ...]
    successful: Array
    scene_id: str = eqx.field(static=True)


def evaluate_cross_discretization_contact(
    scene: ContactParticipantScene,
    states,
    rates,
    search: DenseContactSearchPlan | SweepAndPruneContactSearchPlan,
    closure_plan: ContactClosurePlan,
    route_state: ContactRouteState,
    step_size: ArrayLike,
    rest_positions: ArrayLike,
    /,
    *,
    activation_distance: float | None = None,
    driving_jump: ArrayLike | None = None,
) -> CrossDiscretizationContactResult:
    if not isinstance(scene, ContactParticipantScene):
        raise TypeError("scene must be ContactParticipantScene.")
    if not isinstance(search, (DenseContactSearchPlan, SweepAndPruneContactSearchPlan)):
        raise TypeError("search must be a concrete contact search plan.")
    positions = scene.positions(states)
    velocities = scene.velocities(states, rates)
    epoch = search.build(scene, positions)
    kinematics = evaluate_contact_kinematics(
        scene,
        epoch,
        positions,
        velocities,
        step_size,
        rest_positions=rest_positions,
        activation_distance=activation_distance,
    )
    route_transition = remap_contact_route_state(route_state, kinematics)
    closure = evaluate_contact_closure(
        closure_plan,
        kinematics,
        route_transition.candidate,
        driving_jump=driving_jump,
    )
    assembly = assemble_smooth_contact(kinematics, closure, positions)
    generalized = scene.force_pullback(states, assembly.surface_force)
    successful = (
        epoch.successful
        & kinematics.evidence.successful
        & closure.evidence.successful
        & route_transition.successful
        & assembly.successful
    )
    return CrossDiscretizationContactResult(
        positions,
        velocities,
        epoch,
        kinematics,
        route_transition,
        closure,
        assembly,
        generalized,
        successful,
        scene.scene_id,
    )


__all__ = [
    "CrossDiscretizationContactResult",
    "evaluate_cross_discretization_contact",
]
