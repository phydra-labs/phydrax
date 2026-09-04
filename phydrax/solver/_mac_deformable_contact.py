#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.contact import (
    ContactParticipantScene,
    DenseContactSearchPlan,
    SweepAndPruneContactSearchPlan,
)
from ..linalg import DualSpace


if TYPE_CHECKING:
    from ..applications.contact._closure import ContactClosurePlan
    from ..applications.contact._coupling import CrossDiscretizationContactResult
    from ..applications.contact._route_state import ContactRouteState


DeformableContactKinematics = Callable[[Array, Array, Any], tuple[Array, Array]]
DeformableContactAssembly = Callable[[Array, Array, Any], Array]


class DeformableContactResidualEvaluation(StrictModule):
    residual: Array
    contact: CrossDiscretizationContactResult
    normal_power: Array
    dissipation_rate: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class DeformableContactResidualPlan(StrictModule, NonTrainableState):
    """Canonical participant/closure contact exposed as a structural residual."""

    scene: ContactParticipantScene
    search: DenseContactSearchPlan | SweepAndPruneContactSearchPlan
    closure_plan: ContactClosurePlan
    route_state: ContactRouteState
    rest_positions: Array
    query_kinematics: DeformableContactKinematics = eqx.field(static=True)
    surface_kinematics: DeformableContactKinematics = eqx.field(static=True)
    assemble_residual: DeformableContactAssembly = eqx.field(static=True)
    step_size: float = eqx.field(static=True)
    activation_distance: float | None = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        scene: ContactParticipantScene,
        search: DenseContactSearchPlan | SweepAndPruneContactSearchPlan,
        closure_plan: ContactClosurePlan,
        route_state: ContactRouteState,
        rest_positions: ArrayLike,
        query_kinematics: DeformableContactKinematics,
        surface_kinematics: DeformableContactKinematics,
        assemble_residual: DeformableContactAssembly,
        /,
        *,
        kinematics_id: str,
        assembly_id: str,
        step_size: float = 1.0,
        activation_distance: float | None = None,
    ):
        from ..applications.contact._closure import ContactClosurePlan
        from ..applications.contact._route_state import ContactRouteState

        if not isinstance(scene, ContactParticipantScene) or len(scene.participants) != 2:
            raise TypeError(
                "Deformable contact requires exactly two canonical participants."
            )
        if not isinstance(
            search, (DenseContactSearchPlan, SweepAndPruneContactSearchPlan)
        ):
            raise TypeError("search must be a canonical contact search plan.")
        if not isinstance(closure_plan, ContactClosurePlan):
            raise TypeError("closure_plan must be ContactClosurePlan.")
        if not isinstance(route_state, ContactRouteState):
            raise TypeError("route_state must be ContactRouteState.")
        if route_state.closure_id != closure_plan.closure_id:
            raise ValueError("route_state belongs to another contact closure.")
        if not all(
            callable(value)
            for value in (
                query_kinematics,
                surface_kinematics,
                assemble_residual,
            )
        ):
            raise TypeError("Deformable contact adapters must be callable.")
        kinematics_identifier = str(kinematics_id)
        assembly_identifier = str(assembly_id)
        step = float(step_size)
        activation = None if activation_distance is None else float(activation_distance)
        if (
            not kinematics_identifier
            or not assembly_identifier
            or step <= 0.0
            or (activation is not None and activation <= 0.0)
        ):
            raise ValueError("Deformable contact residual identities/scales are invalid.")
        rest = jnp.asarray(rest_positions)
        expected = (scene.vertex_count, scene.ambient_dimension)
        if rest.shape != expected:
            raise ValueError(f"rest_positions must have shape {expected}.")
        self.scene = scene
        self.search = search
        self.closure_plan = closure_plan
        self.route_state = route_state
        self.rest_positions = rest
        self.query_kinematics = query_kinematics
        self.surface_kinematics = surface_kinematics
        self.assemble_residual = assemble_residual
        self.step_size = step
        self.activation_distance = activation
        self.plan_id = canonical_fingerprint(
            {
                "kind": "deformable-contact-structural-residual",
                "scene": scene.scene_id,
                "search": search.plan_id,
                "closure": closure_plan.closure_id,
                "state_version": array_tree_fingerprint(route_state.state_version),
                "kinematics": kinematics_identifier,
                "assembly": assembly_identifier,
            }
        )

    def evaluate(
        self,
        configuration: Array,
        velocity: Array,
        args: Any = None,
        /,
    ) -> DeformableContactResidualEvaluation:
        from ..applications.contact._coupling import (
            evaluate_cross_discretization_contact,
        )

        query_state, query_rate = self.query_kinematics(configuration, velocity, args)
        surface_state, surface_rate = self.surface_kinematics(
            configuration, velocity, args
        )
        states = (query_state, surface_state)
        rates = (query_rate, surface_rate)
        contact = evaluate_cross_discretization_contact(
            self.scene,
            states,
            rates,
            self.search,
            self.closure_plan,
            self.route_state,
            self.step_size,
            self.rest_positions,
            activation_distance=self.activation_distance,
        )
        query_force, surface_force = contact.generalized_efforts
        query_residual = jax.tree.map(lambda leaf: -leaf, query_force)
        surface_residual = jax.tree.map(lambda leaf: -leaf, surface_force)
        residual = self.assemble_residual(query_residual, surface_residual, args)
        normal_power = sum(
            (
                DualSpace(participant.tangent_space).pair(effort, rate)
                for participant, rate, effort in zip(
                    self.scene.participants,
                    rates,
                    contact.generalized_efforts,
                    strict=True,
                )
            ),
            start=jnp.asarray(0.0, dtype=contact.positions.dtype),
        )
        dissipation_rate = jnp.maximum(-normal_power, 0.0)
        finite = (
            contact.assembly.finite
            & jnp.all(jnp.isfinite(residual))
            & jnp.isfinite(normal_power)
            & jnp.isfinite(dissipation_rate)
        )
        successful = contact.successful & finite
        return DeformableContactResidualEvaluation(
            residual,
            contact,
            normal_power,
            dissipation_rate,
            finite,
            successful,
            self.plan_id,
        )

    def __call__(
        self,
        configuration: Array,
        velocity: Array,
        args: Any = None,
        /,
    ) -> Array:
        return self.evaluate(configuration, velocity, args).residual


__all__ = [
    "DeformableContactAssembly",
    "DeformableContactKinematics",
    "DeformableContactResidualEvaluation",
    "DeformableContactResidualPlan",
]
