#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, PyTree

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...nn.parameters import ParameterSubspace
from ...solver import prepare_virtual_work_equilibrium, PreparedFieldEquilibrium
from ._laws import (
    AbstractNormalContactLaw,
    PenaltyContactLaw,
    PenaltyConvergenceEvidence,
)
from ._mechanics import FixedEpochContactOperator
from ._state import (
    AcceptedContactState,
    ContactEpochTransaction,
    ContactEvaluation,
    ContactStateTransaction,
)


class DeformableMPMContactGeometry(StrictModule):
    """Fixed-capacity deformable-MPM plane-contact kinematics."""

    gap: Array
    normal: Array
    relative_velocity: Array
    valid: Array
    finite: Array
    successful: Array


class DeformableMPMContactTranspose(StrictModule):
    """Exact query/reaction scatter with action-reaction evidence."""

    query_force: Array
    surface_force: Array
    balance_residual: Array
    finite: Array
    successful: Array


class DeformableMPMContactPlan(StrictModule, NonTrainableState):
    """Prepared fixed-capacity particle-to-plane deformable contact routes."""

    query_indices: Array
    plane_points: Array
    plane_normals: Array
    activation_distance: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        query_indices: ArrayLike,
        plane_points: ArrayLike,
        plane_normals: ArrayLike,
        /,
        *,
        activation_distance: float,
        plan_id: str = "deformable-mpm-contact",
    ):
        indices = jnp.asarray(query_indices, dtype=jnp.int32)
        points = jnp.asarray(plane_points)
        normals = jnp.asarray(plane_normals)
        if indices.ndim != 1 or indices.size == 0:
            raise ValueError("query_indices must be one non-empty vector.")
        if points.ndim != 2 or points.shape != normals.shape:
            raise ValueError("Plane points and normals must share shape (route, dim).")
        if points.shape[0] != indices.size or points.shape[1] not in (2, 3):
            raise ValueError("Every query route requires one 2-D or 3-D plane.")
        normal_norm = jnp.linalg.norm(normals, axis=-1)
        if bool(jnp.any(~jnp.isfinite(points))) or bool(
            jnp.any(~jnp.isfinite(normals))
        ):
            raise ValueError("Plane contact geometry must be finite.")
        if bool(jnp.any(normal_norm <= 0.0)):
            raise ValueError("Plane normals must be nonzero.")
        distance = float(activation_distance)
        if not np.isfinite(distance) or distance <= 0.0:
            raise ValueError("activation_distance must be finite and positive.")
        identifier = str(plan_id)
        if not identifier:
            raise ValueError("plan_id must be non-empty.")
        self.query_indices = indices
        self.plane_points = points
        self.plane_normals = normals / normal_norm[:, None]
        self.activation_distance = jnp.asarray(distance, dtype=points.dtype)
        self.plan_id = identifier

    def prepare(self, particle_count: int, /) -> "PreparedDeformableMPMContact":
        count = int(particle_count)
        if count <= 0 or bool(jnp.any(self.query_indices >= count)):
            raise ValueError("query_indices must lie inside the particle layout.")
        return PreparedDeformableMPMContact(self, count)


class PreparedDeformableMPMContact(StrictModule, NonTrainableState):
    """One immutable particle layout with exact contact evaluation and transpose."""

    plan: DeformableMPMContactPlan
    particle_count: int = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: DeformableMPMContactPlan, particle_count: int, /):
        self.plan = plan
        self.particle_count = int(particle_count)
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-deformable-mpm-contact",
                "plan": plan.plan_id,
                "particles": self.particle_count,
            }
        )

    def evaluate(
        self,
        query_position: ArrayLike,
        query_velocity: ArrayLike,
        surface_position: ArrayLike,
        surface_velocity: ArrayLike,
        /,
    ) -> DeformableMPMContactGeometry:
        position = jnp.asarray(query_position)
        velocity = jnp.asarray(query_velocity)
        surface_position_ = jnp.asarray(surface_position)
        surface_velocity_ = jnp.asarray(surface_velocity)
        expected = (self.particle_count, self.plan.plane_points.shape[-1])
        if (
            position.shape != expected
            or velocity.shape != expected
            or surface_position_.shape != expected
            or surface_velocity_.shape != expected
        ):
            raise ValueError("MPM contact states must preserve particle shape.")
        selected_position = position[self.plan.query_indices]
        selected_velocity = velocity[self.plan.query_indices]
        selected_surface_velocity = surface_velocity_[self.plan.query_indices]
        offset = selected_position - self.plan.plane_points
        gap = jnp.sum(offset * self.plan.plane_normals, axis=-1)
        relative_velocity = selected_velocity - selected_surface_velocity
        finite = (
            jnp.all(jnp.isfinite(position))
            & jnp.all(jnp.isfinite(velocity))
            & jnp.all(jnp.isfinite(gap))
        )
        valid = gap <= self.plan.activation_distance
        return DeformableMPMContactGeometry(
            gap,
            self.plan.plane_normals,
            relative_velocity,
            valid,
            finite,
            finite & jnp.all(valid),
        )

    def transpose(
        self,
        geometry: DeformableMPMContactGeometry,
        route_force: ArrayLike,
        /,
    ) -> DeformableMPMContactTranspose:
        force = jnp.asarray(route_force)
        if force.shape != self.plan.plane_points.shape:
            raise ValueError("route_force must preserve route and spatial axes.")
        query_force = jnp.zeros(
            (self.particle_count, force.shape[-1]), dtype=force.dtype
        ).at[self.plan.query_indices].add(force)
        surface_force = -query_force
        balance = jnp.sum(query_force + surface_force, axis=0)
        finite = jnp.all(jnp.isfinite(query_force)) & jnp.all(
            jnp.isfinite(surface_force)
        )
        return DeformableMPMContactTranspose(
            query_force,
            surface_force,
            balance,
            finite,
            finite & jnp.all(geometry.valid),
        )


class FiniteElementContactAssembly(StrictModule):
    """Penalty-contact FE residual and convergence evidence on one search epoch."""

    contact: ContactEvaluation
    plus_residual: Array
    minus_residual: Array
    convergence: PenaltyConvergenceEvidence
    finite: Array
    boundary_id: str = eqx.field(static=True)


class FiniteElementContactBoundary(StrictModule, NonTrainableState):
    """Exterior-node/facet scatter adapter for frictionless normal penalty contact."""

    operator: FixedEpochContactOperator
    boundary_id: str = eqx.field(static=True)

    def __init__(self, operator: FixedEpochContactOperator, /):
        if not isinstance(operator, FixedEpochContactOperator):
            raise TypeError(
                "FiniteElementContactBoundary requires FixedEpochContactOperator."
            )
        if not isinstance(operator.normal_law, PenaltyContactLaw):
            raise TypeError("FiniteElementContactBoundary is the penalty-law FE adapter.")
        if operator.friction_law is not None:
            raise ValueError("The normal penalty FE boundary is frictionless.")
        self.operator = operator
        self.boundary_id = canonical_fingerprint(
            {
                "kind": "finite-element-contact-boundary",
                "operator": operator.operator_id,
            }
        )

    def assemble(
        self,
        accepted: AcceptedContactState,
        plus_coordinates: ArrayLike | None = None,
        minus_coordinates: ArrayLike | None = None,
        /,
    ) -> FiniteElementContactAssembly:
        evaluation = self.operator.evaluate(accepted, plus_coordinates, minus_coordinates)
        convergence = self.operator.normal_law.convergence_evidence(
            evaluation.normal_pressure,
            evaluation.primal_violation,
        )
        return FiniteElementContactAssembly(
            contact=evaluation,
            plus_residual=-evaluation.plus_nodal_forces,
            minus_residual=-evaluation.minus_nodal_forces,
            convergence=convergence,
            finite=evaluation.finite,
            boundary_id=self.boundary_id,
        )

    def tangent_action(
        self,
        accepted: AcceptedContactState,
        plus_coordinates: ArrayLike,
        minus_coordinates: ArrayLike,
        plus_direction: ArrayLike,
        minus_direction: ArrayLike,
        /,
    ) -> tuple[Array, Array]:
        """Differentiate the fixed-patch residual, including current-normal geometry."""
        plus = jnp.asarray(plus_coordinates)
        minus = jnp.asarray(minus_coordinates)
        plus_tangent = jnp.asarray(plus_direction)
        minus_tangent = jnp.asarray(minus_direction)
        if plus_tangent.shape != plus.shape or minus_tangent.shape != minus.shape:
            raise ValueError(
                "FE contact tangent directions must match coordinate layouts."
            )

        def residual(plus_state: Array, minus_state: Array, /) -> tuple[Array, Array]:
            evaluation = self.operator.evaluate(accepted, plus_state, minus_state)
            return -evaluation.plus_nodal_forces, -evaluation.minus_nodal_forces

        _, action = jax.jvp(
            residual,
            (plus, minus),
            (plus_tangent, minus_tangent),
        )
        return action

    def attempt(
        self,
        accepted: AcceptedContactState,
        plus_coordinates: ArrayLike | None = None,
        minus_coordinates: ArrayLike | None = None,
        /,
    ) -> ContactStateTransaction:
        return self.operator.attempt(accepted, plus_coordinates, minus_coordinates)

    def attempt_epoch(
        self,
        previous: AcceptedContactState,
        plus_coordinates: ArrayLike | None = None,
        minus_coordinates: ArrayLike | None = None,
        /,
    ) -> ContactEpochTransaction:
        return self.operator.attempt_epoch(previous, plus_coordinates, minus_coordinates)


CoordinateTrace = Callable[[Mapping[str, Any], Array, Any], ArrayLike]
PressureTrace = Callable[[Mapping[str, Any], Any, Any], ArrayLike]


class FixedEpochNeuralContactEvaluation(StrictModule):
    """Neural trace evaluation and its residual cotangent at frozen search topology."""

    contact: ContactEvaluation
    plus_current_coordinates: Array
    minus_current_coordinates: Array
    plus_virtual_work: Array
    minus_virtual_work: Array
    normal_pressure_virtual_work: Array | None
    adapter_id: str = eqx.field(static=True)


def _neural_contact_field_jet(
    functions: Mapping[str, Any],
    realization: FixedEpochNeuralContactAdapter,
    args: Any,
    /,
) -> PyTree[Array]:
    return realization.field_jet(functions, args)


def _neural_contact_virtual_work(
    functions: Mapping[str, Any],
    jets: PyTree[Array],
    realization: FixedEpochNeuralContactAdapter,
    args: Any,
    /,
) -> PyTree[Array]:
    return realization.virtual_work(functions, jets, args)


class FixedEpochNeuralContactAdapter(StrictModule, NonTrainableState):
    """Neural current-coordinate traces coupled through contact virtual work.

    Trace callables receive ``(functions, epoch_coordinates, args)`` and must
    return current coordinates with the same node layout. Search, closest-point
    coordinates, and active history are frozen outside differentiation.
    """

    operator: FixedEpochContactOperator
    accepted: AcceptedContactState
    plus_trace: CoordinateTrace
    minus_trace: CoordinateTrace
    normal_pressure_trace: PressureTrace | None
    adapter_id: str = eqx.field(static=True)

    def __init__(
        self,
        operator: FixedEpochContactOperator,
        accepted: AcceptedContactState,
        plus_trace: CoordinateTrace,
        minus_trace: CoordinateTrace,
        /,
        *,
        adapter_id: str,
        normal_pressure_trace: PressureTrace | None = None,
    ):
        if not isinstance(operator, FixedEpochContactOperator):
            raise TypeError("Neural contact requires FixedEpochContactOperator.")
        operator._require_state(accepted)
        if not callable(plus_trace) or not callable(minus_trace):
            raise TypeError("Neural contact coordinate traces must be callable.")
        if normal_pressure_trace is not None and not callable(normal_pressure_trace):
            raise TypeError("normal_pressure_trace must be callable or None.")
        if operator.requires_pressure_unknown != (normal_pressure_trace is not None):
            raise ValueError(
                "PDAS neural contact requires exactly one normal-pressure trace."
            )
        identifier = str(adapter_id)
        if not identifier:
            raise ValueError("adapter_id must be nonempty.")
        self.operator = operator
        self.accepted = accepted
        self.plus_trace = plus_trace
        self.minus_trace = minus_trace
        self.normal_pressure_trace = normal_pressure_trace
        self.adapter_id = identifier

    def field_jet(
        self,
        functions: Mapping[str, Any],
        args: Any = None,
        /,
    ) -> dict[str, Array]:
        configuration = self.operator.query.configuration
        plus = jnp.asarray(
            self.plus_trace(functions, configuration.plus.current_coordinates, args)
        )
        minus = jnp.asarray(
            self.minus_trace(functions, configuration.minus.current_coordinates, args)
        )
        if plus.shape != configuration.plus.current_coordinates.shape:
            raise ValueError("Neural plus trace changed the fixed-epoch node layout.")
        if minus.shape != configuration.minus.current_coordinates.shape:
            raise ValueError("Neural minus trace changed the fixed-epoch node layout.")
        jets = {"minus": minus, "plus": plus}
        if self.normal_pressure_trace is not None:
            pressure = jnp.asarray(
                self.normal_pressure_trace(functions, self.operator.query, args)
            )
            if pressure.shape != self.operator.query.patches.gaps.shape:
                raise ValueError(
                    "Neural normal-pressure trace must match the contact-patch layout."
                )
            jets["normal_pressure"] = pressure
        return jets

    def evaluate(
        self,
        functions: Mapping[str, Any],
        args: Any = None,
        /,
    ) -> FixedEpochNeuralContactEvaluation:
        jets = self.field_jet(functions, args)
        contact = self.operator.evaluate(
            self.accepted,
            jets["plus"],
            jets["minus"],
            normal_pressure=jets.get("normal_pressure"),
        )
        return FixedEpochNeuralContactEvaluation(
            contact=contact,
            plus_current_coordinates=jets["plus"],
            minus_current_coordinates=jets["minus"],
            plus_virtual_work=-contact.plus_nodal_forces,
            minus_virtual_work=-contact.minus_nodal_forces,
            normal_pressure_virtual_work=(
                contact.complementarity_residual if "normal_pressure" in jets else None
            ),
            adapter_id=self.adapter_id,
        )

    def virtual_work(
        self,
        functions: Mapping[str, Any],
        jets: PyTree[Array],
        args: Any = None,
        /,
    ) -> dict[str, Array]:
        expected = (
            {"minus", "normal_pressure", "plus"}
            if self.normal_pressure_trace is not None
            else {"minus", "plus"}
        )
        if not isinstance(jets, Mapping) or set(jets) != expected:
            raise ValueError("Neural contact jets do not match the configured traces.")
        contact = self.operator.evaluate(
            self.accepted,
            jets["plus"],
            jets["minus"],
            normal_pressure=jets.get("normal_pressure"),
        )
        cotangent = {
            "minus": -contact.minus_nodal_forces,
            "plus": -contact.plus_nodal_forces,
        }
        if "normal_pressure" in jets:
            cotangent["normal_pressure"] = contact.complementarity_residual
        return cotangent

    def prepare_equilibrium(
        self,
        functions: Mapping[str, Any],
        parameter_subspace: ParameterSubspace,
        /,
        *,
        problem_id: str = "fixed-epoch-neural-contact",
    ) -> PreparedFieldEquilibrium:
        """Construct the canonical parameter-space virtual-work root problem."""
        return prepare_virtual_work_equilibrium(
            functions,
            _neural_contact_field_jet,
            _neural_contact_virtual_work,
            parameter_subspace,
            self,
            realization_id=self.operator.query.query_id,
            provenance_id=self.adapter_id,
            problem_id=problem_id,
        )


class DeformableMPMContactEvaluation(StrictModule):
    """Typed deformable-MPM contact result, distinct from rigid-obstacle impulse."""

    geometry: DeformableMPMContactGeometry
    normal_pressure: Array
    route_force: Array
    transpose: DeformableMPMContactTranspose
    normal_power: Array
    finite: Array
    successful: Array
    adapter_id: str = eqx.field(static=True)


class DeformableMPMContactAdapter(StrictModule, NonTrainableState):
    """Shared normal-law adapter over deformable particle contact interpolation."""

    prepared_contact: PreparedDeformableMPMContact
    normal_law: AbstractNormalContactLaw
    adapter_id: str = eqx.field(static=True)

    def __init__(
        self,
        prepared_contact: PreparedDeformableMPMContact,
        normal_law: AbstractNormalContactLaw,
        /,
    ):
        if not isinstance(prepared_contact, PreparedDeformableMPMContact):
            raise TypeError("prepared_contact must be PreparedDeformableMPMContact.")
        if not isinstance(normal_law, AbstractNormalContactLaw):
            raise TypeError("normal_law must implement AbstractNormalContactLaw.")
        self.prepared_contact = prepared_contact
        self.normal_law = normal_law
        self.adapter_id = canonical_fingerprint(
            {
                "kind": "deformable-mpm-contact-adapter",
                "prepared": prepared_contact.prepared_id,
                "law": normal_law.law_id,
            }
        )

    def evaluate(
        self,
        query_position: ArrayLike,
        query_velocity: ArrayLike,
        surface_position: ArrayLike,
        surface_velocity: ArrayLike,
        /,
        *,
        accepted_pressure: ArrayLike | None = None,
        normal_pressure: ArrayLike | None = None,
    ) -> DeformableMPMContactEvaluation:
        geometry = self.prepared_contact.evaluate(
            query_position,
            query_velocity,
            surface_position,
            surface_velocity,
        )
        normal = self.normal_law.evaluate(
            geometry.gap,
            geometry.normal,
            accepted_pressure,
            normal_pressure=normal_pressure,
        )
        route_force = jnp.where(geometry.valid[:, None], normal.traction, 0.0)
        transpose = self.prepared_contact.transpose(geometry, route_force)
        normal_power = jnp.sum(route_force * geometry.relative_velocity)
        finite = (
            geometry.finite
            & jnp.all(jnp.isfinite(normal.pressure))
            & jnp.all(jnp.isfinite(route_force))
            & jnp.isfinite(normal_power)
            & transpose.finite
        )
        return DeformableMPMContactEvaluation(
            geometry=geometry,
            normal_pressure=normal.pressure,
            route_force=route_force,
            transpose=transpose,
            normal_power=normal_power,
            finite=finite,
            successful=geometry.successful & transpose.successful & finite,
            adapter_id=self.adapter_id,
        )


__all__ = [
    "DeformableMPMContactAdapter",
    "DeformableMPMContactEvaluation",
    "DeformableMPMContactGeometry",
    "DeformableMPMContactPlan",
    "DeformableMPMContactTranspose",
    "PreparedDeformableMPMContact",
    "FiniteElementContactAssembly",
    "FiniteElementContactBoundary",
    "FixedEpochNeuralContactAdapter",
    "FixedEpochNeuralContactEvaluation",
]
