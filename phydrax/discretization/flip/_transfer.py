#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._interpolation import GatherStencil
from ..._sharp_measures import QualifiedSharpGeometry
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..finite_volume import FaceVelocity, PreparedMACOperators
from ..particle import ParticleDiscretization, ParticlePrecisionPolicy
from ..splatting import (
    AbstractStructuredSplatAssignment,
    ParticleGridSplatBudget,
    ParticleGridSplatPlan,
    ParticleGridSplatState,
    PreparedParticleGridSplat,
    SplatExecutionPolicy,
    TensorBSplineSplatAssignment,
)
from ._types import (
    FLIPGridToParticleResult,
    FLIPParticleToGridResult,
    FLIPTransferState,
)


class FLIPParticleTransferPlan(StrictModule, NonTrainableState):
    """Matched cell/face splats for fixed-particle FLIP transfer."""

    operators: PreparedMACOperators
    assignment: AbstractStructuredSplatAssignment
    execution: SplatExecutionPolicy
    precision: ParticlePrecisionPolicy
    budget: ParticleGridSplatBudget
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        operators: PreparedMACOperators,
        /,
        *,
        assignment: AbstractStructuredSplatAssignment | None = None,
        execution: SplatExecutionPolicy | None = None,
        precision: ParticlePrecisionPolicy | None = None,
        budget: ParticleGridSplatBudget | None = None,
    ):
        if not isinstance(operators, PreparedMACOperators):
            raise TypeError("operators must be PreparedMACOperators.")
        assignment_ = (
            TensorBSplineSplatAssignment(1) if assignment is None else assignment
        )
        execution_ = SplatExecutionPolicy() if execution is None else execution
        precision_ = ParticlePrecisionPolicy() if precision is None else precision
        budget_ = ParticleGridSplatBudget() if budget is None else budget
        if not isinstance(assignment_, AbstractStructuredSplatAssignment):
            raise TypeError("assignment must be a structured splat assignment.")
        self.operators = operators
        self.assignment = assignment_
        self.execution = execution_
        self.precision = precision_
        self.budget = budget_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "flip-particle-transfer-plan",
                "operators": operators.prepared_id,
                "assignment": assignment_.assignment_id,
                "execution": execution_.policy_id,
                "precision": precision_.policy_id,
                "budget": budget_.budget_id,
            }
        )

    def prepare(
        self, particles: ParticleDiscretization, /
    ) -> PreparedFLIPParticleTransfer:
        return PreparedFLIPParticleTransfer(self, particles)


class PreparedFLIPParticleTransfer(StrictModule, NonTrainableState):
    plan: FLIPParticleTransferPlan
    particles: ParticleDiscretization
    cell: PreparedParticleGridSplat
    faces: tuple[PreparedParticleGridSplat, ...]
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self, plan: FLIPParticleTransferPlan, particles: ParticleDiscretization, /
    ):
        if not isinstance(plan, FLIPParticleTransferPlan):
            raise TypeError("plan must be FLIPParticleTransferPlan.")
        if not isinstance(particles, ParticleDiscretization):
            raise TypeError("particles must be ParticleDiscretization.")
        operators = plan.operators
        if particles.ambient_dimension != len(operators.discretization.cell_shape):
            raise ValueError("FLIP particles and MAC grid dimensions must match.")
        grid = operators.discretization.grid

        def prepared_for(layout):
            return ParticleGridSplatPlan(
                grid,
                location=grid.location(layout.offsets),
                assignment=plan.assignment,
                boundary="reject",
                execution=plan.execution,
                precision=plan.precision,
                budget=plan.budget,
            ).prepare(particles)

        cell = prepared_for(operators.discretization.cell_layout)
        faces = tuple(
            prepared_for(layout) for layout in operators.discretization.face_layouts
        )
        self.plan = plan
        self.particles = particles
        self.cell = cell
        self.faces = faces
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-flip-particle-transfer",
                "plan": plan.plan_id,
                "particles": particles.prepared_id,
                "cell": cell.prepared_id,
                "faces": [value.prepared_id for value in faces],
            }
        )

    @property
    def dimension(self) -> int:
        return self.particles.ambient_dimension

    def build(
        self,
        position: ArrayLike,
        /,
        *,
        active_mask: ArrayLike | None = None,
    ) -> FLIPTransferState:
        return FLIPTransferState(
            self.cell.build(position, active_mask=active_mask),
            tuple(value.build(position, active_mask=active_mask) for value in self.faces),
            self.prepared_id,
        )

    def _validate_state(self, state: FLIPTransferState, /) -> None:
        if (
            not isinstance(state, FLIPTransferState)
            or state.transfer_id != self.prepared_id
        ):
            raise ValueError("FLIP transfer state belongs to another prepared transfer.")

    def _validate_geometry(self, geometry: QualifiedSharpGeometry | None, /) -> str:
        if geometry is None:
            return ""
        if (
            not isinstance(geometry, QualifiedSharpGeometry)
            or geometry.operator_id != self.plan.operators.prepared_id
        ):
            raise ValueError("FLIP geometry binds another prepared MAC grid.")
        return geometry.realization_id

    def _solid_aware_state(
        self,
        state: ParticleGridSplatState,
        target_open_fraction,
        /,
    ) -> ParticleGridSplatState:
        fraction = jnp.asarray(target_open_fraction)
        if fraction.size != state.stencil.source_size:
            raise ValueError("Solid support does not match the FLIP transfer target.")
        flat = fraction.reshape(-1)
        valid = state.stencil.valid
        safe_indices = jnp.where(valid, state.stencil.indices, 0)
        route_open = flat[safe_indices]
        raw = jnp.where(valid, state.stencil.weights * route_open, 0.0)
        partition = jnp.sum(raw, axis=-1)
        scale = jnp.maximum(jnp.max(jnp.abs(raw), initial=0.0), 1.0)
        tolerance = jnp.finfo(raw.dtype).eps * max(16, raw.shape[-1]) * scale
        supported = state.source_active_mask & (partition > tolerance)
        weights = jnp.where(
            supported[..., None],
            raw / jnp.where(supported[..., None], partition[..., None], 1.0),
            0.0,
        )
        route_valid = valid & (route_open > 0.0) & supported[..., None]
        positive = route_valid & (weights > 0.0)
        minimum = jnp.min(jnp.where(positive, weights, jnp.inf), initial=jnp.inf)
        minimum = jnp.where(jnp.isfinite(minimum), minimum, 0.0)
        invalid = state.invalid_geometry_mask | (state.source_active_mask & ~supported)
        active_supported = jnp.all(~state.source_active_mask | supported)
        stencil = GatherStencil(
            indices=state.stencil.indices,
            weights=weights,
            source_size=state.stencil.source_size,
            valid=route_valid,
            support=supported,
            case_shape=state.stencil.case_shape,
        )
        return ParticleGridSplatState(
            stencil=stencil,
            assignment_state=state.assignment_state,
            source_active_mask=state.source_active_mask,
            supported_mask=supported,
            truncated_support_mask=state.truncated_support_mask,
            out_of_domain_mask=state.out_of_domain_mask,
            invalid_geometry_mask=invalid,
            partition_sums=jnp.where(supported, jnp.sum(weights, axis=-1), 0.0),
            minimum_route_weight=minimum,
            minimum_domain_margin=state.minimum_domain_margin,
            valid_route_count=jnp.sum(route_valid, dtype=jnp.int32),
            dropped_source_count=jnp.sum(
                state.source_active_mask & ~supported, dtype=jnp.int32
            ),
            invalid_geometry_count=jnp.sum(invalid, dtype=jnp.int32),
            successful=state.successful & active_supported,
            prepared_id=state.prepared_id,
        )

    def particle_to_grid(
        self,
        state: FLIPTransferState,
        velocity: ArrayLike,
        reference_density: ArrayLike,
        /,
        *,
        masses: ArrayLike | None = None,
        geometry: QualifiedSharpGeometry | None = None,
    ) -> FLIPParticleToGridResult:
        self._validate_state(state)
        geometry_id = self._validate_geometry(geometry)
        values = jnp.asarray(velocity, dtype=self.particles.safe_masses.dtype)
        expected = (self.particles.capacity, self.dimension)
        if values.shape != expected:
            raise ValueError(f"velocity must have shape {expected}.")
        density = jnp.asarray(reference_density, dtype=values.dtype).reshape(())
        density = eqx.error_if(
            density,
            ~jnp.isfinite(density) | (density <= 0.0),
            "reference_density must be positive and finite.",
        )
        masses = (
            self.particles.masses.astype(values.dtype)
            if masses is None
            else jnp.asarray(masses, dtype=values.dtype)
        )
        if masses.shape != (self.particles.capacity,):
            raise ValueError("masses must have particle-capacity shape.")
        volume = masses / density
        cell_state = (
            state.cell
            if geometry is None
            else self._solid_aware_state(state.cell, geometry.cell_fluid_fraction)
        )
        cell = self.cell.deposit_content(cell_state, volume)
        cell_volume = (
            self.plan.operators.discretization.cell_volumes.astype(values.dtype)
            if geometry is None
            else geometry.cell_fluid_measure.astype(values.dtype)
        )
        cell_active = (
            jnp.ones_like(cell_volume, dtype=bool)
            if geometry is None
            else geometry.cell_active
        )
        liquid_fraction = jnp.where(
            cell_active,
            cell.content / jnp.where(cell_active, cell_volume, 1.0),
            0.0,
        )
        face_mass = []
        face_momentum = []
        face_velocity = []
        face_support = []
        momentum_defect = jnp.asarray(0.0, dtype=values.dtype)
        successful = cell.successful
        for axis, (transfer, route) in enumerate(
            zip(self.faces, state.faces, strict=True)
        ):
            route = (
                route
                if geometry is None
                else self._solid_aware_state(route, geometry.face_open_fraction[axis])
            )
            payload = jnp.stack((masses, masses * values[:, axis]), axis=-1)
            result = transfer.deposit_content(route, payload)
            mass = result.content[..., 0]
            momentum = result.content[..., 1]
            scale = jnp.maximum(jnp.max(jnp.abs(mass), initial=0.0), 1.0)
            tolerance = jnp.finfo(mass.dtype).eps * max(16, transfer.route_count) * scale
            support = mass > tolerance
            if geometry is not None:
                support = support & geometry.face_active[axis]
            velocity_component = jnp.where(
                support, momentum / jnp.where(support, mass, 1.0), 0.0
            )
            face_mass.append(mass)
            face_momentum.append(momentum)
            face_velocity.append(velocity_component)
            face_support.append(support)
            momentum_defect = jnp.maximum(
                momentum_defect, result.balance.maximum_absolute_balance_defect
            )
            successful = (
                successful & result.successful & jnp.all(jnp.isfinite(velocity_component))
            )
        geometry_accepted = jnp.asarray(True) if geometry is None else geometry.accepted
        finite = (
            jnp.all(jnp.isfinite(liquid_fraction))
            & jnp.all(jnp.isfinite(values))
            & jnp.all(
                jnp.stack(tuple(jnp.all(jnp.isfinite(value)) for value in face_velocity))
            )
        )
        return FLIPParticleToGridResult(
            cell.content,
            liquid_fraction,
            tuple(face_mass),
            tuple(face_momentum),
            tuple(face_velocity),
            tuple(face_support),
            cell.balance.maximum_absolute_balance_defect,
            momentum_defect,
            finite,
            successful & finite & geometry_accepted,
            geometry_id,
            self.prepared_id,
        )

    def grid_to_particle(
        self,
        state: FLIPTransferState,
        old_velocity: FaceVelocity,
        new_velocity: FaceVelocity,
        /,
        geometry: QualifiedSharpGeometry | None = None,
    ) -> FLIPGridToParticleResult:
        self._validate_state(state)
        geometry_id = self._validate_geometry(geometry)
        old = self.plan.operators.validate_velocity(old_velocity)
        new = self.plan.operators.validate_velocity(new_velocity)
        pic = []
        increment = []
        supports = []
        for axis, (transfer, route, previous, current) in enumerate(
            zip(self.faces, state.faces, old, new, strict=True)
        ):
            route = (
                route
                if geometry is None
                else self._solid_aware_state(route, geometry.face_open_fraction[axis])
            )
            pic_result = transfer.gather(route, current)
            delta_result = transfer.gather(route, current - previous)
            pic.append(pic_result.values)
            increment.append(delta_result.values)
            supports.append(pic_result.support & delta_result.support)
        pic_values = jnp.stack(tuple(pic), axis=-1)
        increments = jnp.stack(tuple(increment), axis=-1)
        support = jnp.all(jnp.stack(tuple(supports)), axis=0)
        geometry_accepted = jnp.asarray(True) if geometry is None else geometry.accepted
        finite = jnp.all(jnp.isfinite(pic_values)) & jnp.all(jnp.isfinite(increments))
        return FLIPGridToParticleResult(
            pic_values,
            increments,
            support,
            finite,
            support.all() & finite & geometry_accepted,
            geometry_id,
            self.prepared_id,
        )


__all__ = ["FLIPParticleTransferPlan", "PreparedFLIPParticleTransfer"]
